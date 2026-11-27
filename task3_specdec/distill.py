"""
Policy distillation: off-policy and on-policy LoRA students (PDF §3 steps 3-4).

Fixed settings (PDF table, p.13):
  seeds 6304, 6305 x {off-policy, on-policy}, each from the ORIGINAL 0.5B checkpoint;
  quota 40,000 teacher-scored response tokens per run = 30,000 math + 10,000 short;
  caps 192 math / 32 short; temperature 1, top-p 1 (top-k 0, repetition penalty 1);
  microbatch 1; one optimizer update per 256 teacher-scored tokens (final update may be
  shorter); AdamW lr 3e-4 (constant), weight decay 0.01; FP16 frozen backbone, FP32 LoRA
  parameters and optimizer state, FP16 autocast + dynamic GradScaler, unscale before
  clipping at global norm 1.0; LoRA r=8, alpha=16, dropout 0 on q/k/v/o_proj.

Objective (both methods): forward token KL(p_T(.|h) || q_phi(.|h)) over the full common
vocabulary at every supervised response prefix h, SUMMED over the update's supervised
tokens and divided by their number (exactly 256, or fewer for the final update) — so
the loss is the mean per-token KL of the update.  The sampled token is never a target.

Exact update boundaries: a response that crosses a 256-token boundary is split; the
tokens after the boundary are scored by a fresh student forward AFTER the optimizer step
(teacher log-probs of the response are computed once).  A response that would exceed
its category quota is truncated to the permitted prefix and the truncation recorded.

Prompt schedule: within each category a fixed seeded permutation is cycled round-robin
(prompts are reused after exhausting a category).  The next presentation comes from the
category with the smaller filled fraction of its quota (ties -> math).
Off-policy: a FROZEN teacher pool (`build_teacher_pool`: 1 rollout per math prompt, 8 per
short prompt, made once before training and shared by both seeds); visit v of prompt p
presents rollout v mod R_p.  On-policy: a fresh rollout from the CURRENT student (no
gradient), rollout seed recorded.

Logged per presentation: source id, category, visit, rollout seed, response token ids,
termination reason, response length, teacher-scored tokens, quota truncation.  Costs are
timed separately: teacher generation (pool), teacher scoring, student rollout, student
forward/backward, optimizer steps (merging is timed in merge.py).
"""

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from common.runio import read_jsonl, write_json, write_jsonl
from .generation import MATH_CAP, SHORT_CAP, chat_ids, sample_batch, sample_one
from .speculative import QWEN_EOS_IDS

MATH_QUOTA = 30_000
SHORT_QUOTA = 10_000
QUOTA = {"math": MATH_QUOTA, "short": SHORT_QUOTA}
TOTAL_QUOTA = MATH_QUOTA + SHORT_QUOTA
TOKENS_PER_UPDATE = 256
CAP = {"math": MATH_CAP, "short": SHORT_CAP}
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEEDS = (6304, 6305)
POOL_ROLLOUTS = {"math": 1, "short": 8}
LORA = dict(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none")


def _h(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def _sync(dev):
    if str(dev).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class PromptSchedule:
    """Seeded round-robin permutation within each category; reuse after exhaustion."""

    def __init__(self, prompts: Dict[str, List[dict]], seed: int):
        self.order = {}
        for c in ("math", "short"):
            perm = list(prompts[c])
            random.Random(f"PA4-T3-schedule:{seed}:{c}").shuffle(perm)
            self.order[c] = perm
        self.pos = {"math": 0, "short": 0}
        self.visits: Dict[str, int] = {}

    def next(self, category: str):
        L = self.order[category]
        p = L[self.pos[category] % len(L)]
        self.pos[category] += 1
        v = self.visits.get(p["id"], 0)
        self.visits[p["id"]] = v + 1
        return p, v


def next_category(scored: Dict[str, int], quota: Dict[str, int] = QUOTA) -> Optional[str]:
    """Category with the smaller filled fraction of its quota (ties -> math); None when full."""
    open_ = [c for c in ("math", "short") if scored[c] < quota.get(c, 0)]
    if not open_:
        return None
    return min(open_, key=lambda c: (scored[c] / quota[c], c != "math"))


# ── teacher pool (off-policy) ─────────────────────────────────────────────────

def build_teacher_pool(teacher, tokenizer, prompts: Dict[str, List[dict]], out_path: str, device="cuda",
                       batch_size: int = 32, seed: int = 6304, log=print) -> dict:
    """Frozen teacher rollouts for every training prompt (R_math=1, R_short=8), batched.
    Rollout seed of a batch = hash(seed, category, batch index); torch.manual_seed."""
    rows = []
    t0 = time.perf_counter()
    for c in ("math", "short"):
        items = [(p, r) for p in prompts[c] for r in range(POOL_ROLLOUTS[c])]
        for b in range(0, len(items), batch_size):
            chunk = items[b:b + batch_size]
            bseed = _h("PA4-T3-pool", seed, c, b)
            torch.manual_seed(bseed)
            outs = sample_batch(teacher, [chat_ids(tokenizer, p["text"]) for p, _ in chunk], CAP[c],
                                device=device)
            for j, ((p, r), o) in enumerate(zip(chunk, outs)):
                rows.append({"source_id": p["id"], "category": c, "rollout": r, "rollout_seed": bseed,
                             "batch_row": j, "response_ids": o["tokens"], "termination": o["termination"],
                             "n_tokens": len(o["tokens"])})
            log(f"[pool] {c} {min(b + batch_size, len(items))}/{len(items)}")
    _sync(device)
    secs = time.perf_counter() - t0
    write_jsonl(out_path, rows)
    meta = {"n_rollouts": len(rows), "teacher_generation_seconds": secs, "rollouts_per_prompt": POOL_ROLLOUTS,
            "tokens": {c: sum(r["n_tokens"] for r in rows if r["category"] == c) for c in ("math", "short")},
            "eos_rate": {c: sum(r["termination"] == "eos" for r in rows if r["category"] == c) /
                         max(1, sum(r["category"] == c for r in rows)) for c in ("math", "short")}}
    return meta


def load_pool(path: str) -> Dict[tuple, dict]:
    return {(r["source_id"], r["rollout"]): r for r in read_jsonl(path)}


# ── student ───────────────────────────────────────────────────────────────────

def make_lora_student(base_model):
    """LoRA on the frozen FP16 backbone; adapter parameters in FP32."""
    from peft import LoraConfig, TaskType, get_peft_model
    model = get_peft_model(base_model, LoraConfig(task_type=TaskType.CAUSAL_LM, **LORA))
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    return model


def param_counts(model) -> dict:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": tot, "trainable_adapter_params": tr, "backbone_params": tot - tr}


def _response_logprobs(model, prompt_ids: List[int], resp: List[int], start: int, end: int, device):
    """log-probs (end-start, V) fp32 for response tokens [start, end) given prompt+resp[:i]."""
    ids = torch.tensor([prompt_ids + resp[:end]], device=device)
    out = model(input_ids=ids, use_cache=False, logits_to_keep=end - start + 1)
    return F.log_softmax(out.logits[0, :-1].float(), dim=-1)


def kl_sum(teacher_logp: torch.Tensor, student_logp: torch.Tensor) -> torch.Tensor:
    """sum over rows of KL(p_T || q) = sum p_T (log p_T - log q)."""
    return (teacher_logp.exp() * (teacher_logp - student_logp)).sum()


def distill(method: str, teacher, student_base, tokenizer, prompts: Dict[str, List[dict]], seed: int,
            out_dir: str, device="cuda", pool: Optional[Dict[tuple, dict]] = None,
            pool_meta: Optional[dict] = None, quota: Optional[Dict[str, int]] = None,
            tokens_per_update: int = TOKENS_PER_UPDATE, log=print) -> dict:
    """Run one distillation (method 'off' or 'on'). Writes curves.json,
    presentations.jsonl.gz, adapter/ and summary.json (last) to out_dir."""
    assert method in ("off", "on")
    if method == "off":
        assert pool is not None, "off-policy needs the frozen teacher pool"
    quota = dict(quota or QUOTA)
    total_quota = sum(quota.values())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    student = make_lora_student(student_base)
    counts = param_counts(student)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    sched = PromptSchedule(prompts, seed)
    teacher.eval()
    scored = {"math": 0, "short": 0}
    pres, curves = [], []
    cost = {"teacher_scoring_s": 0.0, "student_rollout_s": 0.0, "student_fwd_bwd_s": 0.0, "optimizer_s": 0.0}
    upd = {"n": 0, "target": min(tokens_per_update, total_quota), "loss_sum": 0.0, "by_cat": {"math": 0, "short": 0}}
    stats = {"updates": 0, "skipped_updates": 0, "skipped_at": [], "truncated_by_quota": 0}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_run = time.perf_counter()
    k = 0
    ac = (lambda: torch.autocast("cuda", dtype=torch.float16)) if str(device).startswith("cuda") else \
        (lambda: torch.autocast("cpu", enabled=False))
    while True:
        cat = next_category(scored, quota)
        if cat is None:
            break
        p, visit = sched.next(cat)
        pids = chat_ids(tokenizer, p["text"])
        if method == "off":
            r = pool[(p["id"], visit % POOL_ROLLOUTS[cat])]
            resp, term, rseed = list(r["response_ids"]), r["termination"], r["rollout_seed"]
        else:
            rseed = _h("PA4-T3-onpolicy", seed, k)
            torch.manual_seed(rseed)
            student.eval()
            _sync(device)
            t0 = time.perf_counter()
            with ac():
                o = sample_one(student, pids, CAP[cat], QWEN_EOS_IDS, device)
            _sync(device)
            cost["student_rollout_s"] += time.perf_counter() - t0
            resp, term = o["tokens"], o["termination"]
        n_allowed = min(len(resp), quota[cat] - scored[cat])
        truncated = n_allowed < len(resp)
        stats["truncated_by_quota"] += int(truncated)
        if n_allowed > 0:
            _sync(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                t_logp = _response_logprobs(teacher, pids, resp, 0, n_allowed, device)
            _sync(device)
            cost["teacher_scoring_s"] += time.perf_counter() - t0
            s = 0
            while s < n_allowed:
                e = min(n_allowed, s + upd["target"] - upd["n"])
                student.train()
                _sync(device)
                t0 = time.perf_counter()
                with ac():
                    s_logp = _response_logprobs(student, pids, resp, s, e, device)
                loss = kl_sum(t_logp[s:e], s_logp) / upd["target"]
                scaler.scale(loss).backward()
                _sync(device)
                cost["student_fwd_bwd_s"] += time.perf_counter() - t0
                upd["loss_sum"] += float(loss) * upd["target"]
                upd["n"] += e - s
                upd["by_cat"][cat] += e - s
                scored[cat] += e - s
                s = e
                if upd["n"] == upd["target"]:
                    t0 = time.perf_counter()
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_([q for q in student.parameters() if q.requires_grad],
                                                        GRAD_CLIP)
                    sc0 = scaler.get_scale() if scaler.is_enabled() else 1.0
                    scaler.step(opt)
                    scaler.update()
                    skipped = scaler.is_enabled() and scaler.get_scale() < sc0
                    opt.zero_grad(set_to_none=True)
                    _sync(device)
                    cost["optimizer_s"] += time.perf_counter() - t0
                    stats["updates"] += 1
                    if skipped:
                        stats["skipped_updates"] += 1
                        stats["skipped_at"].append(stats["updates"])
                    done = scored["math"] + scored["short"]
                    curves.append({"update": stats["updates"], "loss_mean_token_kl": upd["loss_sum"] / upd["n"],
                                   "n_supervised": upd["n"], "by_category": dict(upd["by_cat"]),
                                   "teacher_scored_tokens": done, "scored_math": scored["math"],
                                   "scored_short": scored["short"], "lr": opt.param_groups[0]["lr"],
                                   "grad_norm": float(gn), "skipped": bool(skipped),
                                   "grad_scale": scaler.get_scale() if scaler.is_enabled() else None,
                                   "elapsed_s": time.perf_counter() - t_run})
                    if stats["updates"] % 20 == 0:
                        log(f"[{method} s{seed}] update {stats['updates']} tokens {done}/{total_quota} "
                            f"KL {curves[-1]['loss_mean_token_kl']:.4f}")
                    upd = {"n": 0, "target": min(tokens_per_update, total_quota - done), "loss_sum": 0.0,
                           "by_cat": {"math": 0, "short": 0}}
            del t_logp
        pres.append({"k": k, "source_id": p["id"], "category": cat, "visit": visit, "rollout_seed": rseed,
                     "response_ids": resp, "termination": term, "n_response_tokens": len(resp),
                     "teacher_scored_tokens": n_allowed, "truncated_by_quota": truncated})
        k += 1
    assert upd["n"] == 0, "final partial update left unapplied"
    train_s = time.perf_counter() - t_run
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if str(device).startswith("cuda") and torch.cuda.is_available() else None
    student.save_pretrained(str(out / "adapter"))
    write_jsonl(out / "presentations.jsonl.gz", pres)
    write_json(out / "curves.json", curves, indent=None)
    lens = {c: [x["n_response_tokens"] for x in pres if x["category"] == c] for c in ("math", "short")}
    summary = {
        "method": {"off": "off-policy", "on": "on-policy"}[method], "seed": seed, "quota": quota,
        "scored_tokens": scored, "n_presentations": len(pres), **stats,
        "presentations_by_category": {c: len(v) for c, v in lens.items()},
        "mean_response_len": {c: (sum(v) / len(v) if v else None) for c, v in lens.items()},
        "eos_rate": {c: (sum(x["termination"] == "eos" for x in pres if x["category"] == c) / len(v) if v else None)
                     for c, v in lens.items()},
        "costs_s": {**cost, "teacher_generation_s_pool": (pool_meta or {}).get("teacher_generation_seconds")
                    if method == "off" else 0.0, "wall_train_s": train_s},
        "peak_alloc_mb": peak, "params": counts,
        "settings": {"lr": LR, "weight_decay": WEIGHT_DECAY, "tokens_per_update": tokens_per_update,
                     "grad_clip": GRAD_CLIP, "lora": LORA, "caps": CAP, "precision":
                     "FP16 backbone, FP32 LoRA + AdamW state, FP16 autocast, dynamic GradScaler"},
        "teacher_pool_note": ("shared frozen pool built once for both off-policy seeds; its generation time "
                              "is charged in full to each off-policy run's total") if method == "off" else None,
    }
    write_json(out / "summary.json", summary)
    return summary
