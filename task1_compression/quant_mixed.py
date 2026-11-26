"""
Q3: Mixed-precision quantisation.

Uses the same AWQ-style quantiser as Q2; only bit widths change per projection.
Allowed bits: {2, 3, 4, 8} (PDF §1 step 7).

Two allocations must be constructed:
  - Q3-INT4: total packed bytes ≤ Q2 uniform INT4 budget.
  - Q3-INT3: total packed bytes ≤ Q2 uniform INT3 budget.

Both budgets INCLUDE scale/zero-point/group metadata and uncompressed parameters.

Sensitivity measure: mean per-channel activation magnitude variance across
calibration sequences (a proxy for how much each projection's output changes
under quantisation noise).  Higher variance → more sensitive → assign higher bits.

Allocation: sort by sensitivity, assign bits from {2,3,4,8} highest-to-lowest,
accumulate packed bytes, stop before exceeding budget.

CRITICAL: Allocations are chosen using ONLY calibration (128-seq) inputs,
not dev or test results.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .baseline import enumerate_eligible_projections
from .quant_awq import awq_scale_search, apply_awq_quantization
from .quant_uniform import group_quantize, group_dequantize, packed_bytes


ALLOWED_BITS = [2, 3, 4, 8]


def projection_sensitivity(
    module: torch.nn.Linear,
    X: torch.Tensor,
) -> float:
    """
    Per-projection sensitivity: variance of the per-channel activation norm
    across calibration tokens (higher = more sensitive to quantisation).

    Uses calibration inputs only.
    """
    col_norms = X.float().abs().mean(dim=0)  # (d_in,)
    return col_norms.var().item()


def measure_all_sensitivities(
    model,
    calibration_inputs: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Return dict name → sensitivity float for all eligible projections."""
    sens = {}
    for name, module in enumerate_eligible_projections(model):
        X = calibration_inputs[name]
        sens[name] = projection_sensitivity(module, X)
    return sens


def allocate_bits(
    sensitivities: Dict[str, float],
    model,
    budget_total_bytes: int,
    n_uncompressed: int,
    allowed_bits: List[int] = ALLOWED_BITS,
    group_size: int = 128,
) -> Dict[str, int]:
    """
    Assign bit widths to projections so total packed storage ≤ budget.

    Strategy:
      1. Sort projections by sensitivity (descending) — most sensitive first.
      2. Start by assigning minimum bits (2) to all.
      3. For each projection in sensitivity order, try assigning the next
         higher bit width while remaining within budget.

    Returns dict mapping projection name → assigned bits.
    """
    # Enumerate projections in model order.
    proj_names = [name for name, _ in enumerate_eligible_projections(model)]
    proj_sizes = {
        name: module.weight.numel()
        for name, module in enumerate_eligible_projections(model)
    }

    # Fixed overhead: uncompressed parameters (FP16 = 2 bytes each).
    fixed_bytes = n_uncompressed * 2

    def total_quant_bytes(assignment: Dict[str, int]) -> int:
        total = fixed_bytes
        for pname, b in assignment.items():
            n = proj_sizes[pname]
            total += math.ceil(n * b / 8) + math.ceil(n / group_size) * 4
        return total

    # Start from minimum bits.
    assignment = {name: min(allowed_bits) for name in proj_names}

    # Greedily upgrade sensitivity-ordered projections.
    sorted_names = sorted(proj_names, key=lambda n: sensitivities.get(n, 0.0), reverse=True)

    for pname in sorted_names:
        current_b = assignment[pname]
        for b in sorted(allowed_bits):
            if b <= current_b:
                continue
            assignment[pname] = b
            if total_quant_bytes(assignment) > budget_total_bytes:
                assignment[pname] = current_b  # revert
                break
            current_b = b

    return assignment


def mixed_precision_quantize(
    model,
    calibration_inputs: Dict[str, torch.Tensor],
    bit_assignment: Dict[str, int],
    group_size: int = 128,
) -> Dict[str, dict]:
    """
    Apply AWQ-style quantisation with per-projection bit widths from bit_assignment.

    Returns dict matching the structure of quant_awq.apply_awq_quantization().
    """
    result = {}

    for name, module in enumerate_eligible_projections(model):
        bits = bit_assignment[name]
        X = calibration_inputs[name].float()
        W = module.weight.data.float().cpu()
        d_out, d_in = W.shape

        from .quant_awq import awq_scale_search, _find_predecessor, SCALE_EXPONENTS
        best_scale, best_exp, best_err = awq_scale_search(W, X, bits, group_size, SCALE_EXPONENTS)

        W_scaled = W * best_scale.unsqueeze(0)
        q_int, scales, zeros = group_quantize(W_scaled, bits, group_size)
        W_scaled_hat = group_dequantize(q_int, scales, zeros, bits, group_size, d_in)
        W_q_hat = W_scaled_hat * (1.0 / best_scale).unsqueeze(0)

        module.weight.data = W_q_hat.to(module.weight.dtype).to(module.weight.device)

        payload = packed_bytes(q_int[:, :d_in], bits)
        n_groups = math.ceil(d_in / group_size)
        meta = n_groups * 4 * d_out

        result[name] = {
            "bits": bits,
            "packed_bytes": payload,
            "metadata_bytes": meta,
            "total_bytes": payload + meta,
            "calibration_error": best_err,
            "group_size": group_size,
        }

    return result
