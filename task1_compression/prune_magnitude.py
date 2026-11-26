"""
P1: Unstructured per-output-row magnitude pruning.

For each eligible projection W ∈ R^{d_out × d_in}:
  S_ij^mag = |W_ij|
  For each output row i, zero out the fraction `sparsity` of weights with
  smallest scores.

The original checkpoint is NOT modified in-place; callers must restore weights
between sparsity levels.  Use restore_weights() for this.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch

from .baseline import enumerate_eligible_projections, exact_row_prune_mask


def magnitude_prune(
    model,
    sparsity: float,
    inplace: bool = True,
) -> Dict[str, dict]:
    """
    Apply unstructured per-row magnitude pruning to all eligible projections.

    sparsity  fraction of weights to zero per output row (e.g. 0.50).
    inplace   if True, modifies model weights in-place and returns masks.

    Returns
    -------
    dict mapping projection name → {
        "mask": BoolTensor (d_out, d_in) True=kept,
        "achieved_sparsity": float,
        "scores": Tensor (d_out, d_in) – raw |W| scores, CPU fp32
    }
    """
    assert 0.0 <= sparsity <= 1.0
    result = {}

    for name, module in enumerate_eligible_projections(model):
        W = module.weight.data  # (d_out, d_in), fp16 on GPU
        d_out, d_in = W.shape
        n_zero = int(sparsity * d_in)  # weights to zero per output row

        scores = W.float().abs()  # (d_out, d_in)

        # Per-row: zero exactly the n_zero smallest-score columns.
        mask = exact_row_prune_mask(scores, n_zero)

        if inplace:
            module.weight.data *= mask.to(W.dtype)

        achieved = 1.0 - mask.float().mean().item()
        result[name] = {
            "mask": mask.cpu(),
            "achieved_sparsity": achieved,
            "scores": scores.cpu(),
        }

    return result


def save_prune_result(masks_dict: Dict, sparsity: float, method: str, result_dir: str):
    """Save achieved sparsity and per-projection stats (not the masks, which are large)."""
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    summary = {
        "method": method,
        "target_sparsity": sparsity,
        "projections": {
            name: {
                "achieved_sparsity": v["achieved_sparsity"],
            }
            for name, v in masks_dict.items()
        },
        "global_achieved_sparsity": sum(
            v["achieved_sparsity"] for v in masks_dict.values()
        ) / len(masks_dict),
    }
    fname = Path(result_dir) / f"{method}_s{int(sparsity*100):03d}_summary.json"
    with open(fname, "w") as fh:
        json.dump(summary, fh, indent=2)
    return str(fname)
