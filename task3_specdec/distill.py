"""
Policy distillation: off-policy and on-policy (PDF §3 steps 3–4).

Shared settings:
  - Teacher: Qwen/Qwen2.5-1.5B-Instruct (frozen)
  - Student: Qwen/Qwen2.5-0.5B-Instruct + LoRA adapters on q/k/v/o_proj
  - LoRA: r=8, alpha=16, dropout=0, targets q_proj k_proj v_proj o_proj
  - Optimizer: AdamW LR 3e-4, weight_decay 0.01
  - Precision: FP16 frozen backbone, FP32 adapters + optimizer states
  - Gradient clipping: global norm 1.0 (normalize KL by supervised tokens per update)
  - Token quota: 40,000 teacher-scored tokens per seed (30k math + 10k short)
  - Response caps: 192 math, 32 short; temperature=1.0, top_p=1.0
  - Microbatch 1; one update per 256 teacher-scored tokens (short final update OK)
  - Sampling: temperature=1, top_p=1, repetition_penalty=1.0
  - EOS IDs: {151645, 151643} for Qwen2.5-Instruct

Distillation objective (both methods):
  L_KD(phi; H) = (1/|H|) * sum_{h in H} KL(p_T(·|h) || q_phi(·|h))
  where H is the set of supervised response PREFIXES and KL is over the full vocab.
  Forward KL from teacher to student — do NOT use the sampled token as a hard label.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType


MATH_QUOTA = 30_000
SHORT_QUOTA = 10_000
TOTAL_QUOTA = MATH_QUOTA + SHORT_QUOTA
TOKENS_PER_UPDATE = 256
MATH_CAP = 192
SHORT_CAP = 32
TEMPERATURE = 1.0
TOP_P = 1.0
REPET_PENALTY = 1.0
EOS_IDS = {151645, 151643}

LORA_CONFIG = dict(r=8, lora_alpha=16, lora_dropout=0.0,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                   bias="none", task_type=TaskType.CAUSAL_LM)


def _build_lora_student(student_base_model) -> any:
    cfg = LoraConfig(**LORA_CONFIG)
    return get_peft_model(student_base_model, cfg)


def _teacher_generate(
    teacher_model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new: int,
    device: str,
    seed: int,
) -> List[int]:
    """Generate a response from the teacher at temperature=1, top_p=1."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    with torch.no_grad():
        out = teacher_model.generate(
            prompt_ids.to(device),
            max_new_tokens=max_new,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPET_PENALTY,
            eos_token_id=list(EOS_IDS),
            use_cache=True,
        )
    return out[0, prompt_ids.shape[1]:].tolist()


def _kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    Forward KL(teacher || student) averaged over supervised token positions.
    Both tensors: (n_tokens, vocab_size), fp32.
    """
    p_t = F.softmax(teacher_logits, dim=-1)
    log_q = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(log_q, p_t, reduction="batchmean")


def _make_optimizer(student_model) -> AdamW:
    return AdamW(
        [p for p in student_model.parameters() if p.requires_grad],
        lr=3e-4,
        weight_decay=0.01,
    )


# ── Off-policy distillation ────────────────────────────────────────────────────

def offpolicy_distill(
    teacher_model,
    student_base_model,
    tokenizer,
    math_prompts: List[dict],     # {"id": str, "text": str, "category": "math"}
    short_prompts: List[dict],    # {"id": str, "text": str, "category": "short"}
    seed: int = 6304,
    device: str = "cuda",
    result_dir: str = "results/task3",
) -> dict:
    """
    Off-policy distillation.

    1. Prepare a FROZEN teacher-response pool from training prompts.
    2. Cycle through the pool deterministically up to the token quota.
    3. For each teacher response prefix, compute KL loss and update student.

    Records teacher rollout cost and training cost separately.
    """
    rng = random.Random(seed)
    result_path = Path(result_dir) / f"offpolicy_seed{seed}"
    result_path.mkdir(parents=True, exist_ok=True)

    student_model = _build_lora_student(student_base_model)
    student_model.to(device)
    # Frozen backbone in FP16; adapters in FP32.
    for name, param in student_model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()
        else:
            param.requires_grad_(False)

    optimizer = _make_optimizer(student_model)
    scaler = torch.cuda.amp.GradScaler()

    # Build 800-prompt pool: 500 math + 300 short, round-robin permuted.
    pool = _build_prompt_pool(math_prompts, short_prompts, seed)

    # Step 1: collect frozen teacher responses.
    teacher_pool = _collect_teacher_responses(teacher_model, tokenizer, pool, seed, device)
    teacher_pool_path = result_path / "teacher_pool.json"
    with open(teacher_pool_path, "w") as fh:
        json.dump([{k: v for k, v in r.items() if k != "token_ids"} for r in teacher_pool], fh, indent=2)

    # Step 2: train.
    curves = {"step": [], "loss": [], "lr": [], "scored_tokens": []}
    scored_tokens = 0
    update_buffer_tokens = 0
    update_buffer_logits = []  # (teacher_logits, student_logits) pairs
    optimizer_step = 0
    skipped_updates = 0

    pool_idx = 0
    while scored_tokens < TOTAL_QUOTA:
        rec = teacher_pool[pool_idx % len(teacher_pool)]
        pool_idx += 1
        prefix_ids = torch.tensor([rec["prefix_ids"]], dtype=torch.long, device=device)
        t_logits = torch.tensor(rec["teacher_logits"], dtype=torch.float32, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            s_out = student_model(prefix_ids)
        s_logits = s_out.logits[0, :-1, :].float()  # predict next tokens
        t_logits_ = t_logits[:s_logits.shape[0], :]  # align lengths

        n_tokens = s_logits.shape[0]
        update_buffer_logits.append((t_logits_.detach(), s_logits))
        update_buffer_tokens += n_tokens
        scored_tokens += n_tokens

        if update_buffer_tokens >= TOKENS_PER_UPDATE or scored_tokens >= TOTAL_QUOTA:
            cat_t = torch.cat([x[0] for x in update_buffer_logits], dim=0)
            cat_s = torch.cat([x[1] for x in update_buffer_logits], dim=0)
            loss = _kl_loss(cat_s, cat_t)
            # Normalize by supervised tokens in this update.
            loss = loss / (update_buffer_tokens + 1e-8) * TOKENS_PER_UPDATE

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in student_model.parameters() if p.requires_grad], 1.0
            )
            if torch.isfinite(torch.tensor(grad_norm.item())):
                scaler.step(optimizer)
                optimizer_step += 1
            else:
                skipped_updates += 1
            scaler.update()

            curves["step"].append(optimizer_step)
            curves["loss"].append(loss.item())
            curves["scored_tokens"].append(scored_tokens)
            update_buffer_logits = []
            update_buffer_tokens = 0

    with open(result_path / "curves.json", "w") as fh:
        json.dump({**curves, "skipped_updates": skipped_updates, "total_scored_tokens": scored_tokens}, fh, indent=2)

    return {"model": student_model, "curves": curves, "result_dir": str(result_path)}


def onpolicy_distill(
    teacher_model,
    student_base_model,
    tokenizer,
    math_prompts: List[dict],
    short_prompts: List[dict],
    seed: int = 6304,
    device: str = "cuda",
    result_dir: str = "results/task3",
) -> dict:
    """
    On-policy distillation.

    For each training step, generate a fresh response PREFIX from the CURRENT STUDENT,
    compute the TEACHER's distribution at those prefixes, and minimise KL(teacher||student).

    Generation cost is logged separately and not differentiated through.
    """
    rng = random.Random(seed)
    result_path = Path(result_dir) / f"onpolicy_seed{seed}"
    result_path.mkdir(parents=True, exist_ok=True)

    student_model = _build_lora_student(student_base_model)
    student_model.to(device)
    for name, param in student_model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()
        else:
            param.requires_grad_(False)

    optimizer = _make_optimizer(student_model)
    scaler = torch.cuda.amp.GradScaler()

    pool = _build_prompt_pool(math_prompts, short_prompts, seed)
    curves = {"step": [], "loss": [], "scored_tokens": [], "rollout_tokens": []}
    scored_tokens = 0
    update_buffer_tokens = 0
    update_buffer_logits = []
    optimizer_step = 0
    skipped_updates = 0
    total_rollout_tokens = 0

    pool_idx = 0
    while scored_tokens < TOTAL_QUOTA:
        rec = pool[pool_idx % len(pool)]
        pool_idx += 1
        is_math = rec["category"] == "math"
        cap = MATH_CAP if is_math else SHORT_CAP

        prompt_ids = tokenizer(rec["text"], return_tensors="pt")["input_ids"].to(device)
        gen_seed = seed + pool_idx

        # Generate from current student.
        student_model.eval()
        with torch.no_grad():
            gen_out = student_model.generate(
                prompt_ids,
                max_new_tokens=cap,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPET_PENALTY,
                eos_token_id=list(EOS_IDS),
                use_cache=True,
            )
        response_ids = gen_out[0, prompt_ids.shape[1]:]
        total_rollout_tokens += response_ids.shape[0]

        # Score with teacher and student.
        full_ids = gen_out  # (1, prompt + response)
        with torch.no_grad():
            t_out = teacher_model(full_ids)
        t_logits = t_out.logits[0, prompt_ids.shape[1] - 1: -1, :].float()

        student_model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            s_out = student_model(full_ids)
        s_logits = s_out.logits[0, prompt_ids.shape[1] - 1: -1, :].float()

        n_tokens = s_logits.shape[0]
        update_buffer_logits.append((t_logits.detach(), s_logits))
        update_buffer_tokens += n_tokens
        scored_tokens += n_tokens

        if update_buffer_tokens >= TOKENS_PER_UPDATE or scored_tokens >= TOTAL_QUOTA:
            cat_t = torch.cat([x[0] for x in update_buffer_logits], dim=0)
            cat_s = torch.cat([x[1] for x in update_buffer_logits], dim=0)
            loss = _kl_loss(cat_s, cat_t)
            loss = loss / (update_buffer_tokens + 1e-8) * TOKENS_PER_UPDATE

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in student_model.parameters() if p.requires_grad], 1.0
            )
            if torch.isfinite(torch.tensor(grad_norm.item())):
                scaler.step(optimizer)
                optimizer_step += 1
            else:
                skipped_updates += 1
            scaler.update()

            curves["step"].append(optimizer_step)
            curves["loss"].append(loss.item())
            curves["scored_tokens"].append(scored_tokens)
            curves["rollout_tokens"].append(total_rollout_tokens)
            update_buffer_logits = []
            update_buffer_tokens = 0

    with open(result_path / "curves.json", "w") as fh:
        json.dump({**curves, "skipped_updates": skipped_updates,
                   "total_scored_tokens": scored_tokens,
                   "total_rollout_tokens": total_rollout_tokens}, fh, indent=2)

    return {"model": student_model, "curves": curves, "result_dir": str(result_path)}


# ── Prompt pool construction ───────────────────────────────────────────────────

def _build_prompt_pool(
    math_prompts: List[dict],
    short_prompts: List[dict],
    seed: int,
) -> List[dict]:
    """
    Build an 800-prompt pool via fixed, seeded round-robin permutation within
    the two quota categories.  Reuse prompts after exhausting a category.
    PDF: "selecting source prompts through a fixed, seeded round-robin permutation
    within the two quota categories; reuse prompts after exhausting a category."
    """
    rng = random.Random(seed)
    math_perm = list(math_prompts)
    short_perm = list(short_prompts)
    rng.shuffle(math_perm)
    rng.shuffle(short_perm)

    pool = []
    mi, si = 0, 0
    while len(pool) < 800:
        if mi < len(math_perm):
            pool.append({**math_perm[mi % len(math_perm)], "category": "math"})
            mi += 1
        if si < len(short_perm) and len(pool) < 800:
            pool.append({**short_perm[si % len(short_perm)], "category": "short"})
            si += 1
    return pool[:800]


def _collect_teacher_responses(
    teacher_model,
    tokenizer,
    pool: List[dict],
    seed: int,
    device: str,
) -> List[dict]:
    """Generate and cache teacher responses for off-policy distillation."""
    teacher_pool = []
    for i, rec in enumerate(pool):
        is_math = rec["category"] == "math"
        cap = MATH_CAP if is_math else SHORT_CAP
        prompt_ids = tokenizer(rec["text"], return_tensors="pt")["input_ids"].to(device)
        response = _teacher_generate(teacher_model, tokenizer, prompt_ids, cap, device, seed + i)
        full_ids = torch.tensor(prompt_ids[0].tolist() + response, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            t_out = teacher_model(full_ids)
        t_logits = t_out.logits[0, prompt_ids.shape[1] - 1: -1, :].float().cpu().tolist()
        teacher_pool.append({
            "source_id": rec.get("id", str(i)),
            "category": rec["category"],
            "prefix_ids": full_ids[0, :prompt_ids.shape[1] + len(response)].tolist(),
            "teacher_logits": t_logits,
            "n_response_tokens": len(response),
            "termination": "eos" if response[-1] in EOS_IDS else "cap",
        })
    return teacher_pool
