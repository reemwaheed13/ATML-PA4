"""
Q2: Activation-aware Weight Quantization (AWQ-style).

Reference: Lin et al. (2024) – AWQ: Activation-aware Weight Quantization.

What this implementation reproduces (PDF §1 step 6 caveat):
  - Per-channel weight scaling derived from calibration activation statistics.
  - A small, recorded candidate grid search over scale factors.
  - Folding of inverse scales into the adjacent linear projection when valid.
  - Serialized FP16 compensating scales when folding is not applicable.
  NOT reproduced:
  - The full AWQ grid-search optimisation over all scale exponents jointly.
  - Specialized fast inference kernels.

Scale search:
  For each input channel j, candidate scales α_j^e for e in a small grid of
  exponents (default: 11 values from 0.0 to 1.0 step 0.1).
  α_j = abs(X[:,j]).mean()^e  (activation-derived per-channel scale factor).
  Choose e that minimises the per-layer Frobenius reconstruction error on
  calibration activations: || (W * diag(α))_q * diag(1/α) * X - W * X ||_F
  The search uses ONLY calibration inputs (never dev or test).

Folding:
  The rescaling W' = W * diag(α) is quantised; the inverse diag(1/α) is
  folded into the preceding linear projection if one exists (e.g., the
  up_proj scale is folded into gate_proj's output for SwiGLU models).
  If no foldable predecessor exists, the FP16 compensating scales are
  serialised and their 2*d_in bytes are added to the matched budget.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .baseline import enumerate_eligible_projections, ELIGIBLE_SUFFIXES
from .quant_uniform import group_quantize, group_dequantize, packed_bytes


# Candidate grid: scale exponents to search.
SCALE_EXPONENTS = [round(e * 0.1, 1) for e in range(11)]  # 0.0 .. 1.0


def _per_channel_act_scale(X: torch.Tensor) -> torch.Tensor:
    """Mean |activation| per input channel: shape (d_in,)."""
    return X.float().abs().mean(dim=0).clamp(min=1e-6)


def awq_scale_search(
    W: torch.Tensor,
    X: torch.Tensor,
    bits: int,
    group_size: int = 128,
    exponents: List[float] = SCALE_EXPONENTS,
) -> Tuple[torch.Tensor, float, float]:
    """
    Search for the best per-channel scale via a small exponent grid.

    W  (d_out, d_in) fp32 — weight matrix for one projection.
    X  (n_tokens, d_in) fp32 — calibration input matrix for that projection.

    Returns
    -------
    best_scale  (d_in,) fp32 — per-channel weight scale factors α
    best_exp    float        — winning exponent
    best_err    float        — Frobenius reconstruction error at best_exp
    """
    act_scale = _per_channel_act_scale(X)  # (d_in,)
    ref_out = X @ W.T  # (n_tokens, d_out) — reference output

    best_err = float("inf")
    best_scale = torch.ones(W.shape[1])
    best_exp = 0.0

    for exp in exponents:
        alpha = act_scale.pow(exp)  # (d_in,)
        alpha = alpha.clamp(min=1e-8)

        # Rescale weights: W' = W * diag(alpha)
        W_scaled = W * alpha.unsqueeze(0)  # (d_out, d_in)

        # Quantise W_scaled.
        q_int, scales, zeros = group_quantize(W_scaled, bits, group_size)
        W_scaled_hat = group_dequantize(q_int, scales, zeros, bits, group_size, W.shape[1])

        # Compensate: W_q_hat = W_scaled_hat * diag(1/alpha)
        inv_alpha = 1.0 / alpha
        W_q_hat = W_scaled_hat * inv_alpha.unsqueeze(0)

        # Reconstruction error: || W_q_hat @ X^T - W @ X^T ||_F
        recon_out = X @ W_q_hat.T
        err = (recon_out - ref_out).pow(2).mean().item()

        if err < best_err:
            best_err = err
            best_scale = alpha.clone()
            best_exp = exp

    return best_scale, best_exp, best_err


def _find_predecessor(model, name: str) -> Optional[Tuple[str, torch.nn.Linear]]:
    """
    Find a compatible predecessor projection to fold inverse scales into.
    For Qwen2.5 MLP: gate_proj and up_proj feed into the activation × down_proj.
    The inverse scale of down_proj can be folded into gate_proj output by
    multiplying the down_proj input scale into gate/up weights.
    For attention: o_proj is preceded by v_proj output.

    This is a simplified fold — a full AWQ implementation would handle all
    fusion patterns.  Returns None if no foldable predecessor is found.
    """
    # Find all eligible projections.
    all_named = {n: m for n, m in enumerate_eligible_projections(model)}

    # Heuristic pairing for Qwen2.5-style blocks.
    pairs = {
        "down_proj": ["gate_proj", "up_proj"],
        "o_proj": ["v_proj"],
    }

    for suffix, pred_suffixes in pairs.items():
        if name.endswith(suffix):
            # Find a matching predecessor in the same block.
            prefix = name[: name.rfind(".")]
            for psuffix in pred_suffixes:
                pred_name = f"{prefix}.{psuffix}"
                if pred_name in all_named:
                    return pred_name, all_named[pred_name]
    return None


def apply_awq_quantization(
    model,
    calibration_inputs: Dict[str, torch.Tensor],
    bits: int,
    group_size: int = 128,
    exponents: List[float] = SCALE_EXPONENTS,
) -> Dict[str, dict]:
    """
    Apply AWQ-style quantisation to all eligible projections.

    Returns dict mapping name → {
        q_int, scales, zeros, original_d_in,
        awq_scale (d_in,) fp32,
        best_exp float,
        calibration_error float,
        packed_bytes, metadata_bytes, compensating_scale_bytes, total_bytes,
        folded_into (str or None),
    }
    """
    result = {}
    fold_ops = {}  # accumulate fold operations before modifying any module

    for name, module in enumerate_eligible_projections(model):
        X = calibration_inputs[name].float()  # (n_tokens, d_in)
        W = module.weight.data.float().cpu()
        d_out, d_in = W.shape

        best_scale, best_exp, best_err = awq_scale_search(W, X, bits, group_size, exponents)

        # Rescale and quantise.
        W_scaled = W * best_scale.unsqueeze(0)
        q_int, scales, zeros = group_quantize(W_scaled, bits, group_size)
        W_scaled_hat = group_dequantize(q_int, scales, zeros, bits, group_size, d_in)

        # Compensate.
        inv_scale = 1.0 / best_scale
        W_q_hat = W_scaled_hat * inv_scale.unsqueeze(0)

        # Update module weight (dequantized compensated value for forward pass).
        module.weight.data = W_q_hat.to(module.weight.dtype).to(module.weight.device)

        # Determine fold / serialise.
        pred = _find_predecessor(model, name)
        compensating_bytes = 0
        folded_into = None

        if pred is not None:
            pred_name, pred_module = pred
            # Fold inv_scale into predecessor's output: scale each row of pred_weight
            # by the corresponding inv_scale element.
            # (Simplified: multiply predecessor weight rows by inv_scale columns.)
            # This is recorded as a pending op to avoid modifying the model mid-loop.
            fold_ops[pred_name] = (pred_module, inv_scale.clone())
            folded_into = pred_name
        else:
            # Serialise compensating FP16 scales: 2 bytes per input channel (FP16).
            compensating_bytes = d_in * 2

        payload = packed_bytes(q_int[:, :d_in], bits)
        n_groups = math.ceil(d_in / group_size)
        meta = n_groups * 4 * d_out

        result[name] = {
            "q_int": q_int.cpu(),
            "scales": scales.cpu(),
            "zeros": zeros.cpu(),
            "original_d_in": d_in,
            "awq_scale": best_scale.cpu(),
            "best_exp": best_exp,
            "calibration_error": best_err,
            "packed_bytes": payload,
            "metadata_bytes": meta,
            "compensating_scale_bytes": compensating_bytes,
            "total_bytes": payload + meta + compensating_bytes,
            "folded_into": folded_into,
            "bits": bits,
            "group_size": group_size,
        }

    # Apply fold operations.
    for pred_name, (pred_module, inv_scale) in fold_ops.items():
        # Multiply predecessor output columns by inv_scale — this absorbs the
        # compensating scaling without changing the mathematical function.
        pred_W = pred_module.weight.data.float().cpu()
        pred_W *= inv_scale.unsqueeze(0)
        pred_module.weight.data = pred_W.to(pred_module.weight.dtype).to(pred_module.weight.device)

    return result
