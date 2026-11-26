"""
Shared perplexity evaluator.

Protocol (matches PDF §1 "Quality evaluation"):
  - Each record in the locked JSONL represents one non-overlapping 512-token window
    tokenized from a single document.  Windows from different documents are independent;
    no hidden state is carried across them.
  - The first token of each window is used as left context only and is NOT counted in
    the scored-token denominator.
  - score_target_offsets: list of within-window token positions (0-based) that are
    scored.  Normally [1, 2, ..., len(token_ids)-1].  The locked file may shorten the
    last window of a document, so we always use the value from the file when present.
  - PPL = exp( -1/N * sum_t log p(x_t | x_{<t}) ), where N = number of scored tokens.
  - Returns N, mean NLL, PPL, and per-window records for bootstrap resampling.
"""

import gzip
import json
import math
from pathlib import Path
from typing import Optional

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


@torch.no_grad()
def evaluate_perplexity(
    model,
    locked_windows_path: str,
    device: str,
    dtype=torch.float16,
    batch_size: int = 1,
) -> dict:
    """
    Args
    ----
    model               HF CausalLM already on `device`, in eval mode.
    locked_windows_path Path to *_windows_locked.jsonl(.gz).
    device              e.g. "cuda:0".
    dtype               Autocast dtype for forward pass.
    batch_size          Number of windows to process per forward pass.
                        Windows are all 512 tokens so batching is safe.

    Returns
    -------
    dict with keys:
      N              int   total scored tokens
      mean_nll       float mean negative log-likelihood per token
      ppl            float exp(mean_nll)
      per_window     list of {"nll": float, "n": int} – one entry per window,
                     used by bootstrap_paired_ppl.
    """
    model.eval()
    records = list(_iter_records(locked_windows_path))

    total_nll = 0.0
    total_n = 0
    per_window = []

    # Process in mini-batches of equal-length windows.
    # All windows in a locked file are the same length (512 tokens) except
    # possibly the last window of some documents; we handle variable lengths
    # by processing one at a time when lengths differ.
    i = 0
    while i < len(records):
        batch_recs = records[i : i + batch_size]
        i += batch_size

        # Check if all same length; fall back to loop if not.
        lengths = [len(r["token_ids"]) for r in batch_recs]
        if len(set(lengths)) == 1:
            _process_batch(batch_recs, model, device, dtype, per_window)
        else:
            for r in batch_recs:
                _process_batch([r], model, device, dtype, per_window)

    for entry in per_window:
        total_nll += entry["nll"]
        total_n += entry["n"]

    mean_nll = total_nll / total_n if total_n > 0 else float("inf")
    return {
        "N": total_n,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "per_window": per_window,
    }


def _process_batch(batch_recs, model, device, dtype, per_window):
    B = len(batch_recs)
    L = len(batch_recs[0]["token_ids"])
    ids = torch.tensor([r["token_ids"] for r in batch_recs], dtype=torch.long, device=device)

    with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=True):
        logits = model(ids).logits  # (B, L, V)

    log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, L, V) in fp32

    for b, rec in enumerate(batch_recs):
        token_ids = rec["token_ids"]
        L = len(token_ids)
        # Determine which positions are scored.
        if "score_target_offsets" in rec:
            offsets = rec["score_target_offsets"]
        else:
            # Default: exclude the first token (position 0).
            offsets = list(range(1, L))

        # Guard the offset convention: a scored target at window position t is
        # predicted from the logits at position t-1, so t==0 would silently read
        # log_probs[..., -1] (the last position) and t>=L would run off the end.
        # Both would corrupt the pooled NLL without raising, so reject them here.
        # Per PDF §1 the first token of each window is context-only and unscored.
        for t in offsets:
            if t < 1 or t >= L:
                raise ValueError(
                    f"score_target_offset {t} out of range for window of length {L}; "
                    "offsets must lie in [1, L-1] (position 0 is context-only)."
                )

        nll_sum = 0.0
        for t in offsets:
            # logit at position t-1 is used to predict position t.
            target_id = token_ids[t]
            nll_sum += -log_probs[b, t - 1, target_id].item()

        per_window.append({"nll": nll_sum, "n": len(offsets)})


def save_result(result: dict, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in result.items() if k != "per_window"}
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    # Save per-window for bootstrap separately.
    pw_path = p.with_name(p.stem + "_per_window.json")
    with open(pw_path, "w") as fh:
        json.dump(result["per_window"], fh)
