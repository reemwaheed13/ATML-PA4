"""
Structured sparsity diagnostic: 2:4 mask for 50% Wanda condition.

The 2:4 mask retains the two highest-ranked Wanda-scored weights in each
group of four CONTIGUOUS INPUT-CHANNEL weights.

This is a quality and representation study only.
PDF: "do not claim T4 sparse-tensor-core acceleration."
"""

import torch
from typing import Dict
from .baseline import enumerate_eligible_projections
from .prune_wanda import collect_wanda_scores


def apply_24_mask(
    model,
    calibration_inputs: Dict[str, torch.Tensor],
    inplace: bool = True,
) -> Dict[str, dict]:
    """
    Apply 2:4 structured sparsity to all eligible projections.

    For each output row i and each group of 4 contiguous input columns
    [j, j+1, j+2, j+3], retain the 2 highest-ranked weights by Wanda score.

    If d_in is not divisible by 4, the last group is padded (trailing weights
    are treated as having zero score and are zeroed).

    Returns dict matching prune_wanda output structure.
    """
    scores_dict = collect_wanda_scores(model, calibration_inputs)
    result = {}

    for name, module in enumerate_eligible_projections(model):
        scores = scores_dict[name]  # (d_out, d_in) fp32
        d_out, d_in = scores.shape

        # Pad to multiple of 4.
        pad = (4 - d_in % 4) % 4
        if pad:
            scores_padded = torch.cat([scores, scores.new_zeros(d_out, pad)], dim=1)
        else:
            scores_padded = scores
        d_in_pad = scores_padded.shape[1]
        n_groups = d_in_pad // 4

        # Reshape to (d_out, n_groups, 4), find top-2 per group.
        s_grouped = scores_padded.reshape(d_out, n_groups, 4)
        _, top2_idx = torch.topk(s_grouped, k=2, dim=2, largest=True, sorted=False)
        mask_grouped = torch.zeros(d_out, n_groups, 4, dtype=torch.bool)
        mask_grouped.scatter_(2, top2_idx, True)
        mask = mask_grouped.reshape(d_out, d_in_pad)[:, :d_in]

        if inplace:
            W = module.weight.data
            module.weight.data = (W.float().cpu() * mask.float()).to(W.dtype).to(W.device)

        achieved = 1.0 - mask.float().mean().item()
        result[name] = {
            "mask": mask.cpu(),
            "achieved_sparsity": achieved,
            "pattern": "2:4",
        }

    return result
