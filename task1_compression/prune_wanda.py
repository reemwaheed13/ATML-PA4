"""
P2: Activation-aware pruning (Wanda).

Reference: Sun et al. (2024) – A Simple and Effective Pruning Approach for LLMs.

Wanda score: S_ij^Wanda = |W_ij| * ||X_{j,:}||_2
where X ∈ R^{n_calib × d_in} is the matrix of calibration inputs to projection W.

For each output row i, zero out the fraction `sparsity` of weights with smallest
Wanda scores.  Per-row masking convention is identical to P1.

Calibration: register forward hooks to collect the input activation tensor X for
each projection, run all calibration sequences, then remove hooks and score.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .baseline import enumerate_eligible_projections


# ── Activation collection ──────────────────────────────────────────────────────

def collect_calibration_inputs(
    model,
    calibration_token_ids: List[List[int]],
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    Run calibration sequences through the model and collect the input matrix X
    for every eligible projection.

    Returns
    -------
    dict mapping projection name → X tensor of shape (n_tokens, d_in), fp32, CPU.
    (Rows are individual token positions across all calibration sequences,
    concatenated; this matches the original Wanda implementation.)
    """
    buffers: Dict[str, List[torch.Tensor]] = {}
    hooks = []

    for name, module in enumerate_eligible_projections(model):
        buffers[name] = []

        def make_hook(n):
            def hook(mod, inp, out):
                # inp[0] shape: (B, T, d_in) or (B*T, d_in)
                x = inp[0].detach().float().cpu()
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                buffers[n].append(x)
            return hook

        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for ids in calibration_token_ids:
            t = torch.tensor([ids], dtype=torch.long, device=device)
            model(t)

    for h in hooks:
        h.remove()

    return {name: torch.cat(xs, dim=0) for name, xs in buffers.items()}


def collect_wanda_scores(
    model,
    calibration_inputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute Wanda scores for all eligible projections given pre-collected
    calibration inputs.

    Returns dict mapping name → score tensor (d_out, d_in), fp32, CPU.
    """
    scores = {}
    for name, module in enumerate_eligible_projections(model):
        X = calibration_inputs[name]  # (n_tokens, d_in), fp32, CPU
        W = module.weight.data.float().cpu()  # (d_out, d_in)

        # ||X_{j,:}||_2 for each column j (input feature dimension).
        col_norms = X.norm(p=2, dim=0)  # (d_in,)

        # S_ij = |W_ij| * col_norm_j
        scores[name] = W.abs() * col_norms.unsqueeze(0)  # (d_out, d_in)
    return scores


# ── Pruning ────────────────────────────────────────────────────────────────────

def wanda_prune(
    model,
    sparsity: float,
    calibration_inputs: Optional[Dict[str, torch.Tensor]] = None,
    calibration_token_ids: Optional[List[List[int]]] = None,
    device: str = "cuda:0",
    inplace: bool = True,
) -> Dict[str, dict]:
    """
    Apply Wanda pruning at the given sparsity.

    Provide either `calibration_inputs` (pre-collected) or `calibration_token_ids`
    (will be collected here).  Pre-collected is preferred to share the same
    activations across sparsity levels.

    Returns dict matching the structure of magnitude_prune(), plus "wanda_scores".
    """
    assert 0.0 <= sparsity <= 1.0

    if calibration_inputs is None:
        assert calibration_token_ids is not None
        calibration_inputs = collect_calibration_inputs(model, calibration_token_ids, device)

    scores_dict = collect_wanda_scores(model, calibration_inputs)
    result = {}

    for name, module in enumerate_eligible_projections(model):
        scores = scores_dict[name]  # (d_out, d_in), fp32, CPU
        d_out, d_in = scores.shape
        n_zero = int(sparsity * d_in)

        if n_zero > 0:
            topk_vals, _ = torch.topk(scores, n_zero, dim=1, largest=False, sorted=False)
            threshold = topk_vals.max(dim=1, keepdim=True).values
            mask = scores > threshold
            at_threshold = scores == threshold
            needed_zeros = n_zero - (~mask).sum(dim=1, keepdim=True)
            for i in range(d_out):
                nz = needed_zeros[i].item()
                if nz > 0:
                    tie_cols = at_threshold[i].nonzero(as_tuple=False).squeeze(1)
                    mask[i, tie_cols[:nz]] = False
        else:
            mask = torch.ones(d_out, d_in, dtype=torch.bool)

        if inplace:
            W = module.weight.data
            module.weight.data = (W.float().cpu() * mask.float()).to(W.dtype).to(W.device)

        achieved = 1.0 - mask.float().mean().item()
        result[name] = {
            "mask": mask.cpu(),
            "achieved_sparsity": achieved,
            "scores": scores,
            "wanda_scores": scores,
        }

    return result
