"""
Shared perplexity evaluator.

Protocol (matches PDF §1 "Quality evaluation"):
  - Each record in the locked JSONL represents one non-overlapping 512-token window
    tokenized from a single document.  Windows from different documents are independent;
    no hidden state is carried across them.
  - The first token of each window is used as left context only and is NOT counted in
    the scored-token denominator.
  - score_target_offsets: a HALF-OPEN RANGE [start, end) of within-window token
    positions that are scored, i.e. positions start, start+1, ..., end-1.  It is NOT
    a list of individual positions.  Normally [1, len(token_ids)) with
    scored_tokens == len(token_ids) - 1.  The locked file may shorten the last window
    of a document, so we always use the value from the file when present and we
    cross-check it against the record's `scored_tokens` count.
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
    expected_total_tokens: Optional[int] = 50000,
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
    expected_total_tokens
                        If not None, assert the pooled scored-token count N equals
                        this value.  The locked evaluation file is fixed at 50000
                        scored tokens, so a mismatch means a truncated/wrong file.
                        Pass None (or the true count) for unit tests on tiny files.

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

    if expected_total_tokens is not None:
        assert total_n == expected_total_tokens, (
            f"pooled scored-token count N={total_n} != expected "
            f"{expected_total_tokens}; the locked window file looks truncated or wrong."
        )

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
        L = len(rec["token_ids"])
        start, end = _scored_range(rec, L)

        # Positions t in [start, end) are scored.  Each target at window position t
        # is predicted from the logits at position t-1, so we gather log-probs from
        # rows [start-1, end-1) and compare against the token ids at [start, end).
        pos = torch.arange(start, end, device=ids.device)
        targets = ids[b, pos]                                     # (n,)
        lp = log_probs[b, pos - 1]                                # (n, V)
        nll = -lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # (n,)

        per_window.append({"nll": float(nll.sum().item()), "n": end - start})


def _scored_range(rec, L):
    """Resolve the half-open scored range [start, end) for one window record.

    `score_target_offsets` is stored as a half-open range [start, end) — e.g.
    [1, 512] means positions 1..511 are scored (511 tokens), NOT the two positions
    {1, 512}.  Position 0 is left context only and is never scored, so start >= 1;
    end may not exceed the window length L.  When the record carries a
    `scored_tokens` count we assert it matches the range length.
    """
    if "score_target_offsets" in rec:
        start, end = rec["score_target_offsets"]
    else:
        # Default: score every token except the context-only position 0.
        start, end = 1, L

    if not (1 <= start < end <= L):
        raise ValueError(
            f"score_target_offsets [{start}, {end}) invalid for window of length {L}; "
            "require 1 <= start < end <= L (position 0 is context-only)."
        )

    positions = range(start, end)
    if "scored_tokens" in rec:
        assert len(positions) == rec["scored_tokens"], (
            f"scored_tokens={rec['scored_tokens']} disagrees with range "
            f"[{start}, {end}) of length {len(positions)}."
        )
    return start, end


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
