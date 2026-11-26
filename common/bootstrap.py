"""
Paired bootstrap resampling for 95% confidence intervals.

PDF requirements:
  - P3–P2 and Q3–Q2 HellaSwag differences: resample HellaSwag examples.
  - P3–P2 and Q3–Q2 perplexity differences: resample whole fixed evaluation
    windows and recompute pooled token NLL before exponentiating.
  - Task 2C and Task 3: resample over prompts (≥2,000 resamples).

All CIs are two-sided 95%: (2.5th, 97.5th percentiles of the bootstrap
distribution of the difference).
"""

import math
import random
from typing import List, Dict, Tuple


# ── HellaSwag bootstrap ────────────────────────────────────────────────────────

def _hellaswag_acc(per_example: List[dict], indices: List[int], metric: str) -> float:
    """Compute accuracy over a subset of per-example records."""
    pred_key = "raw_pred" if metric == "raw_acc" else "norm_pred"
    correct = sum(per_example[i]["label"] == per_example[i][pred_key] for i in indices)
    return correct / len(indices) if indices else 0.0


def bootstrap_paired_hellaswag(
    per_example_a: List[dict],
    per_example_b: List[dict],
    metric: str = "acc_norm",
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 6304,
) -> Dict:
    """
    Paired bootstrap CI for the difference (method_b.metric - method_a.metric).

    Both lists must contain the same examples in the same order (matched by
    per_example[i]["order"] if order matters, but index-matched here for speed).

    Returns
    -------
    dict: observed_diff, ci_lo, ci_hi, n_resamples, metric
    """
    assert len(per_example_a) == len(per_example_b), "Must be matched example lists"
    n = len(per_example_a)
    rng = random.Random(seed)

    obs_a = _hellaswag_acc(per_example_a, list(range(n)), metric)
    obs_b = _hellaswag_acc(per_example_b, list(range(n)), metric)
    observed_diff = obs_b - obs_a

    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        diffs.append(
            _hellaswag_acc(per_example_b, idx, metric)
            - _hellaswag_acc(per_example_a, idx, metric)
        )

    diffs.sort()
    lo = diffs[int((alpha / 2) * n_resamples)]
    hi = diffs[int((1 - alpha / 2) * n_resamples)]

    return {
        "observed_diff": observed_diff,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_resamples": n_resamples,
        "metric": metric,
    }


# ── Perplexity bootstrap ───────────────────────────────────────────────────────

def _pooled_nll(per_window: List[dict], indices: List[int]) -> float:
    """Recompute pooled mean NLL from resampled window records."""
    total_nll = sum(per_window[i]["nll"] for i in indices)
    total_n = sum(per_window[i]["n"] for i in indices)
    return total_nll / total_n if total_n > 0 else float("inf")


def bootstrap_paired_ppl(
    per_window_a: List[dict],
    per_window_b: List[dict],
    metric: str = "ppl",
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 6304,
) -> Dict:
    """
    Paired bootstrap CI for the difference (method_b.ppl - method_a.ppl).

    Resamples whole fixed evaluation windows; recomputes pooled token NLL before
    exponentiating — matches the PDF requirement exactly.

    per_window_{a,b}: lists of {"nll": float, "n": int}, one entry per window.
    """
    assert len(per_window_a) == len(per_window_b), "Must be matched window lists"
    n = len(per_window_a)
    rng = random.Random(seed)

    obs_nll_a = _pooled_nll(per_window_a, list(range(n)))
    obs_nll_b = _pooled_nll(per_window_b, list(range(n)))

    if metric == "ppl":
        obs_a = math.exp(obs_nll_a)
        obs_b = math.exp(obs_nll_b)
    else:
        obs_a, obs_b = obs_nll_a, obs_nll_b

    observed_diff = obs_b - obs_a

    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        nll_a = _pooled_nll(per_window_a, idx)
        nll_b = _pooled_nll(per_window_b, idx)
        if metric == "ppl":
            diffs.append(math.exp(nll_b) - math.exp(nll_a))
        else:
            diffs.append(nll_b - nll_a)

    diffs.sort()
    lo = diffs[int((alpha / 2) * n_resamples)]
    hi = diffs[int((1 - alpha / 2) * n_resamples)]

    return {
        "observed_diff": observed_diff,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_resamples": n_resamples,
        "metric": metric,
    }


# ── Prompt-level bootstrap (Task 2C, Task 3) ──────────────────────────────────

def bootstrap_paired_accuracy(
    correct_a: List[int],
    correct_b: List[int],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 6304,
) -> Dict:
    """
    Paired bootstrap CI for accuracy difference (b - a).
    correct_{a,b}: list of 0/1 per prompt, in matched order.
    """
    assert len(correct_a) == len(correct_b)
    n = len(correct_a)
    rng = random.Random(seed)

    obs_diff = sum(correct_b) / n - sum(correct_a) / n

    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        acc_a = sum(correct_a[i] for i in idx) / n
        acc_b = sum(correct_b[i] for i in idx) / n
        diffs.append(acc_b - acc_a)

    diffs.sort()
    lo = diffs[int((alpha / 2) * n_resamples)]
    hi = diffs[int((1 - alpha / 2) * n_resamples)]

    return {"observed_diff": obs_diff, "ci_lo": lo, "ci_hi": hi, "n_resamples": n_resamples}
