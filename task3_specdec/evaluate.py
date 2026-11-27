"""
Speculative-decoding evaluation (PDF §3 steps 2, 6, 7).

`run_spec_condition(draft, gamma)` — the EOS-aware primary benchmark on the 120 held-out
prompts x 2 paired repetitions with a 64-new-token cap, each speculative generation
ADJACENT to a target-only generation of the same prompt/rep; their order alternates
((prompt index + rep + gamma) even -> speculative first), i.e. counterbalanced.  Then the
fixed-64-token throughput diagnostic on 24 math prompts with EOS ignored (timing only).
Two untimed warm-up generations precede each condition.

`isolated_costs` — single-step draft and target decode latency and the target's
(γ+1)-token verification latency at several cache lengths: c = t_draft/t_target,
v_γ = t_target(γ+1 tokens)/t_target(1 token).

`cost_model` — Ŝ_γ(α) = (1 + Σ_{j=1..γ} α^j) / (γ c + v_γ) and the break-even α
(smallest α in [0,1] with 1 + Σ α^j > γ c + v_γ, or None if no such α).

`standalone_accuracy` — each draft as a model: 512-token math / 32-token short caps,
2 repetitions, batched sampling, fixed extraction and format checks.
"""

import math
import time
from typing import Dict, List, Optional, Sequence

import torch

from .generation import chat_ids, sample_batch, score_response
from .speculative import QWEN_EOS_IDS, forward_tokens, prefill, speculative_generate, target_only_generate

GAMMAS = (2, 4)
N_REPS = 2
SPEED_CAP = 64
N_FIXED64 = 24
QUALITY_CAP = {"math": 512, "short": 32}
COST_LENGTHS = (64, 256, 1024)


def _seed(*parts) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def _rec(st, prefix: str) -> dict:
    s = st.summary()
    s.pop("accepted_per_round", None)
    return {f"{prefix}{k}": v for k, v in s.items()}


def run_spec_condition(target, draft, tokenizer, prompts: Sequence[dict], gamma: int, draft_name: str,
                       device="cuda", n_reps: int = N_REPS, cap: int = SPEED_CAP, n_fixed64: int = N_FIXED64,
                       log=print) -> dict:
    """Returns {'primary': [...], 'fixed64': [...]} per-generation rows."""
    ids = [chat_ids(tokenizer, p["text"]) for p in prompts]
    for w in range(2):                                    # warm-up (not recorded)
        torch.manual_seed(_seed("warm", draft_name, gamma, w))
        speculative_generate(target, draft, ids[w], 8, gamma, QWEN_EOS_IDS, device)
        target_only_generate(target, ids[w], 8, QWEN_EOS_IDS, device)
    out = {"primary": [], "fixed64": []}
    plans = [("primary", list(range(len(prompts))), QWEN_EOS_IDS, cap),
             ("fixed64", [i for i, p in enumerate(prompts) if p["domain"] == "math"][:n_fixed64], frozenset(), 64)]
    for kind, idxs, eos, mx in plans:
        for i in idxs:
            p = prompts[i]
            for rep in range(n_reps):
                spec_first = (i + rep + gamma) % 2 == 0
                row = {"prompt_id": p["id"], "workload": p["workload"], "domain": p["domain"], "rep": rep,
                       "gamma": gamma, "draft": draft_name, "kind": kind, "spec_first": spec_first,
                       "prompt_len": len(ids[i])}
                s_seed, t_seed = _seed("spec", kind, p["id"], rep, gamma), _seed("target", kind, p["id"], rep, gamma)
                for which in (("spec", "target") if spec_first else ("target", "spec")):
                    if which == "spec":
                        torch.manual_seed(s_seed)
                        st = speculative_generate(target, draft, ids[i], mx, gamma, eos, device)
                        row.update(_rec(st, "spec_"))
                        row["spec_seed"] = s_seed
                        row["spec_tokens"] = st.tokens
                        row["spec_accepted_per_round"] = st.accepted_per_round
                    else:
                        torch.manual_seed(t_seed)
                        st = target_only_generate(target, ids[i], mx, eos, device)
                        row.update(_rec(st, "target_"))
                        row["target_seed"] = t_seed
                        row["target_tokens"] = st.tokens
                row["time_ratio_target_over_spec"] = row["target_t_total_ms"] / row["spec_t_total_ms"]
                row["tps_ratio_spec_over_target"] = (
                    row["spec_tokens_per_s"] / row["target_tokens_per_s"]
                    if row["spec_tokens_per_s"] and row["target_tokens_per_s"] else None)
                out[kind].append(row)
            if kind == "primary" and (i + 1) % 20 == 0:
                log(f"[spec {draft_name} g{gamma}] {i + 1}/{len(prompts)}")
    return out


def _time_ms(fn, device, n_warmup=3, n_rep=10) -> float:
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_rep):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t0))
    ts.sort()
    return ts[len(ts) // 2]


@torch.no_grad()
def isolated_costs(models: Dict[str, torch.nn.Module], device="cuda", lengths=COST_LENGTHS, gammas=GAMMAS,
                   n_rep: int = 10) -> List[dict]:
    """Median latency of one decode step (1 token) and of a (γ+1)-token verification at
    each cache length, per model.  Prefix = fixed pseudo-random token ids."""
    rows = []
    g = torch.Generator().manual_seed(6304)
    vocab = min(m.config.vocab_size for m in models.values())
    lo, hi = min(1000, vocab // 4), min(30000, vocab)
    for L in lengths:
        prefix = torch.randint(lo, hi, (L + 1,), generator=g).tolist()
        for name, m in models.items():
            cache = prefill(m, prefix, device)            # holds prefix[:-1] (L tokens)

            def step(n):
                forward_tokens(m, cache, prefix[-1:] + [11] * (n - 1), device, n_logits=n)
                cache.crop(L)

            row = {"model": name, "cache_len": L, "decode_step_ms": _time_ms(lambda: step(1), device, n_rep=n_rep)}
            for gm in gammas:
                row[f"verify_{gm + 1}_tokens_ms"] = _time_ms(lambda: step(gm + 1), device, n_rep=n_rep)
            rows.append(row)
            del cache
    return rows


def cost_model(c: float, v: float, gamma: int, alpha: Optional[float] = None) -> dict:
    cost = gamma * c + v

    def S(a):
        return (1 + sum(a ** j for j in range(1, gamma + 1))) / cost

    be = None
    if 1 + gamma > cost:                                  # numerator at α=1 exceeds the cost
        lo, hi = 0.0, 1.0
        if 1 > cost:
            be = 0.0
        else:
            for _ in range(60):
                mid = (lo + hi) / 2
                if 1 + sum(mid ** j for j in range(1, gamma + 1)) > cost:
                    hi = mid
                else:
                    lo = mid
            be = hi
    out = {"gamma": gamma, "c": c, "v_gamma": v, "denominator_gamma_c_plus_v": cost, "break_even_alpha": be,
           "break_even_exists": be is not None}
    if alpha is not None:
        out.update(alpha=alpha, predicted_speedup=S(alpha))
    return out


def alpha_from_rounds(accepted_per_round: Sequence[int], gamma: int) -> Optional[float]:
    """α̂ such that the iid-model expected accepted proposals per full round,
    Σ_{j=1..γ} α^j, equals the observed mean over full-length (γ-proposal) rounds."""
    if not accepted_per_round:
        return None
    m = sum(accepted_per_round) / len(accepted_per_round)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(mid ** j for j in range(1, gamma + 1)) < m:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def standalone_accuracy(model, tokenizer, prompts: Sequence[dict], name: str, device="cuda",
                        n_reps: int = N_REPS, batch_size: int = 30) -> List[dict]:
    rows = []
    for rep in range(n_reps):
        for dom in ("math", "short"):
            ps = [p for p in prompts if p["domain"] == dom]
            for b in range(0, len(ps), batch_size):
                chunk = ps[b:b + batch_size]
                seed = _seed("standalone", name, dom, rep, b)
                torch.manual_seed(seed)
                outs = sample_batch(model, [chat_ids(tokenizer, p["text"]) for p in chunk], QUALITY_CAP[dom],
                                    device=device)
                for p, o in zip(chunk, outs):
                    text = tokenizer.decode(o["tokens"], skip_special_tokens=True)
                    rows.append({"draft": name, "prompt_id": p["id"], "workload": p["workload"], "domain": dom,
                                 "rep": rep, "seed": seed, "n_tokens": len(o["tokens"]),
                                 "termination": o["termination"], "response": text,
                                 **score_response(p, text, o["termination"])})
    return rows
