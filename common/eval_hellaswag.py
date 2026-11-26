"""
Shared HellaSwag evaluator.

Scoring convention (matches PDF §1 and lm-evaluation-harness):
  - For each of the 4 candidate endings, compute the conditional log-likelihood
    given the shared context: sum of log P(tok | context + prev_ending_toks).
  - raw_acc : argmax(log-likelihoods) == label
  - acc_norm: argmax(log-likelihood / len(ending)) == label
              where len is the NUMBER OF CHARACTERS in the ending string.
              The PDF explicitly says: "do not normalize by tokenizer token count."
  - Saves per-example candidate scores (all 4 raw and normalised) for qualitative
    analysis and bootstrap resampling.
"""

import gzip
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def _iter_records(path: str):
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_context(example: dict) -> str:
    """Reproduce lm-evaluation-harness HellaSwag context construction."""
    ctx = example.get("ctx", "")
    if not ctx:
        # Fallback: ctx_a + " " + ctx_b
        ctx = (example.get("ctx_a", "") + " " + example.get("ctx_b", "")).strip()
    activity = example.get("activity_label", "")
    if activity:
        return activity + ": " + ctx
    return ctx


@torch.no_grad()
def evaluate_hellaswag(
    model,
    tokenizer,
    locked_hellaswag_path: str,
    device: str,
    dtype=torch.float16,
    save_path: str = None,
    subset_ids: set = None,
) -> dict:
    """
    Args
    ----
    model                 HF CausalLM already on `device`, in eval mode.
    tokenizer             Matching tokenizer.
    locked_hellaswag_path Path to hellaswag_val_locked.jsonl(.gz).
    device                e.g. "cuda:0".
    save_path             If given, save per-example scores as JSON.
    subset_ids            If given, only evaluate these order IDs (for the
                          512-example diagnostic subset).

    Returns
    -------
    dict with keys:
      n_examples   int
      raw_acc      float
      acc_norm     float
      per_example  list of dicts (order, label, raw_scores, norm_scores, raw_pred, norm_pred)
    """
    model.eval()
    tokenizer.padding_side = "right"

    per_example = []
    n_correct_raw = 0
    n_correct_norm = 0
    n_total = 0

    for rec in _iter_records(locked_hellaswag_path):
        order = rec["order"]
        if subset_ids is not None and order not in subset_ids:
            continue

        example = rec["example"]
        label = int(example["label"])
        context_str = _build_context(example)
        endings = example["endings"]  # list of 4 strings

        raw_scores = []
        norm_scores = []

        for ending in endings:
            full_text = context_str + " " + ending
            ctx_enc = tokenizer(context_str + " ", return_tensors="pt", add_special_tokens=True)
            full_enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)

            ctx_len = ctx_enc["input_ids"].shape[1]
            full_ids = full_enc["input_ids"].to(device)

            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=True):
                logits = model(full_ids).logits  # (1, L, V)

            log_probs = F.log_softmax(logits[0].float(), dim=-1)  # (L, V)

            # Score only the ending tokens (positions ctx_len .. full_len-1).
            # logit at position t-1 predicts token at position t.
            ending_nll = 0.0
            ending_n_toks = 0
            full_ids_list = full_ids[0].tolist()
            for t in range(ctx_len, len(full_ids_list)):
                ending_nll += -log_probs[t - 1, full_ids_list[t]].item()
                ending_n_toks += 1

            raw_ll = -ending_nll  # higher = better
            char_len = len(ending)
            norm_ll = raw_ll / char_len if char_len > 0 else raw_ll

            raw_scores.append(raw_ll)
            norm_scores.append(norm_ll)

        raw_pred = int(raw_scores.index(max(raw_scores)))
        norm_pred = int(norm_scores.index(max(norm_scores)))

        n_correct_raw += int(raw_pred == label)
        n_correct_norm += int(norm_pred == label)
        n_total += 1

        per_example.append(
            {
                "order": order,
                "label": label,
                "raw_scores": raw_scores,
                "norm_scores": norm_scores,
                "raw_pred": raw_pred,
                "norm_pred": norm_pred,
            }
        )

    result = {
        "n_examples": n_total,
        "raw_acc": n_correct_raw / n_total if n_total else 0.0,
        "acc_norm": n_correct_norm / n_total if n_total else 0.0,
        "per_example": per_example,
    }

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            json.dump(result, fh, indent=2)

    return result
