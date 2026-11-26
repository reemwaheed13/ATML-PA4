"""
MQAR training and evaluation loop (Part B).

Fixed training protocol (PDF §2 Part B):
  Optimizer: AdamW, peak LR 8e-4, weight decay 0.01
  Batch: 32 sequences, 4,500 optimizer updates
  Schedule: 200 warmup steps → linear decay to 15% of peak LR
  Gradient clipping: global norm 1.0
  Seeds: 6304, 6305, 6306 (each run independently re-initialised)
  Checkpoint evaluation: steps 500, 1000, 1500, 2000, 3000, 4500

Loss: cross-entropy at supervised query positions only.
Label-space: 64 classes (value IDs 128..191, offset by -128).
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .mqar_model import MQARTransformer, make_mqar_model


CHECKPOINT_STEPS = [500, 1000, 1500, 2000, 3000, 4500]
VALUE_OFFSET = 128      # value ID 128 → class 0, 191 → class 63


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_warmup_decay(
    step: int, warmup_steps: int, total_steps: int, min_frac: float = 0.15
) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(min_frac, 1.0 - t * (1.0 - min_frac))


def mqar_collate(batch: List[dict], device: str):
    """
    batch: list of MQAR example dicts from generate_mqar / locked JSONL.
    Returns tensors ready for the model.
    """
    token_ids = torch.tensor([ex["tokens"] for ex in batch], dtype=torch.long, device=device)
    # query_positions: (B, K)
    query_positions = torch.tensor(
        [ex["query_positions"] for ex in batch], dtype=torch.long, device=device
    )
    # answer_ids: (B, K) in range 128..191 → labels 0..63
    answer_ids = torch.tensor(
        [ex["answer_ids"] for ex in batch], dtype=torch.long, device=device
    ) - VALUE_OFFSET
    return token_ids, query_positions, answer_ids


def train_mqar(
    attn_type: str,
    train_records: List[dict],
    val_records: Dict[str, List[dict]],   # {"near": [...], "far": [...], "mixed": [...]}
    seed: int = 6304,
    device: str = "cuda",
    window: int = 64,
    n_updates: int = 4500,
    batch_size: int = 32,
    peak_lr: float = 8e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,
    grad_clip: float = 1.0,
    min_lr_frac: float = 0.15,
    result_dir: str = "results/task2",
    checkpoint_steps: List[int] = CHECKPOINT_STEPS,
) -> dict:
    """
    Train one MQAR model from scratch.

    Returns dict with training curves and final checkpoint path.
    """
    _set_seed(seed)
    model = make_mqar_model(attn_type, window=window, device=device)
    optimizer = AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: linear_warmup_decay(step, warmup_steps, n_updates, min_lr_frac),
    )
    loss_fn = nn.CrossEntropyLoss()

    result_path = Path(result_dir) / f"{attn_type}_seed{seed}"
    result_path.mkdir(parents=True, exist_ok=True)

    curves = {"step": [], "loss": [], "val": {r: [] for r in val_records}}
    rng = random.Random(seed)

    model.train()
    step = 0
    epoch_data = list(train_records)

    while step < n_updates:
        rng.shuffle(epoch_data)
        for batch_start in range(0, len(epoch_data), batch_size):
            if step >= n_updates:
                break
            batch = epoch_data[batch_start: batch_start + batch_size]
            if len(batch) == 0:
                continue
            # Pad to full batch if needed (drop partial batch for simplicity).
            if len(batch) < batch_size:
                continue

            token_ids, query_pos, labels = mqar_collate(batch, device)

            logits = model(token_ids, query_pos)  # (B, K, 64)
            B, K, C = logits.shape
            loss = loss_fn(logits.reshape(B * K, C), labels.reshape(B * K))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 100 == 0 or step in checkpoint_steps:
                curves["step"].append(step)
                curves["loss"].append(loss.item())

            if step in checkpoint_steps:
                for regime, recs in val_records.items():
                    acc = evaluate_mqar(model, recs, device, batch_size)["accuracy"]
                    curves["val"][regime].append({"step": step, "accuracy": acc})

                ckpt_path = result_path / f"ckpt_step{step}.pt"
                torch.save(model.state_dict(), ckpt_path)

    # Save curves.
    with open(result_path / "curves.json", "w") as fh:
        json.dump(curves, fh, indent=2)

    return {"curves": curves, "model": model, "result_dir": str(result_path)}


@torch.no_grad()
def evaluate_mqar(
    model: MQARTransformer,
    records: List[dict],
    device: str,
    batch_size: int = 64,
) -> dict:
    """
    Evaluate MQAR accuracy on a list of example records.

    Returns dict with accuracy, per-example records (for distance analysis).
    """
    model.eval()
    correct = 0
    total = 0
    per_example = []

    for i in range(0, len(records), batch_size):
        batch = records[i: i + batch_size]
        token_ids, query_pos, labels = mqar_collate(batch, device)
        logits = model(token_ids, query_pos)  # (B, K, 64)
        preds = logits.argmax(dim=-1)          # (B, K)
        match = (preds == labels)              # (B, K)
        correct += match.sum().item()
        total += match.numel()

        for b, ex in enumerate(batch):
            for ki in range(labels.shape[1]):
                per_example.append({
                    "regime": ex.get("regime", "?"),
                    "K": ex.get("K", 0),
                    "distance": ex["distances"][ki] if "distances" in ex else -1,
                    "correct": match[b, ki].item(),
                })

    model.train()
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "per_example": per_example,
    }
