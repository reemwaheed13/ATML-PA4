"""
Q1: Uniform group-wise asymmetric weight-only PTQ.

Quantizer spec (PDF §1 step 5):
  - b ∈ {8, 4, 3} bits
  - Group size 128 along the INPUT COLUMN dimension.
  - Asymmetric: separate scale s and integer zero-point z per group.
  - Rounding: round-to-nearest-even (default torch.round).
  - Zero-point convention: z = clip(round(-wmin / s), 0, 2^b - 1).
  - Packing: weights are packed into uint8 tensors for storage accounting;
    for NUMERICAL EVALUATION we dequantize back to fp16.
  - Metadata: FP16 scale (2 B) + FP16 zero-point (2 B) = 4 B per group.

IMPORTANT: These conventions MUST remain identical across Q1/Q2/Q3 —
only the input weights change (Q2 rescales before quantising, Q3 changes
per-projection bit widths).

No actual low-bit kernels are available on most Colab setups; we therefore:
  1. Compute quantised integers and store them packed.
  2. Dequantize to fp16 for forward-pass evaluation.
  3. Report packed bytes separately.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import torch

from .baseline import enumerate_eligible_projections


def group_quantize(
    W: torch.Tensor,
    bits: int,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric group-wise quantisation of a weight matrix W (d_out, d_in).

    Groups are formed along the INPUT (column / d_in) dimension.
    The last group is zero-padded to reach group_size if d_in % group_size != 0.

    Returns
    -------
    q_int   (d_out, n_groups * group_size) int32 — integer codes in [0, 2^b-1]
    scales  (d_out, n_groups)              float32
    zeros   (d_out, n_groups)              int32  — zero-points
    """
    d_out, d_in = W.shape
    max_int = 2 ** bits - 1

    # Pad d_in to a multiple of group_size.
    pad = (group_size - d_in % group_size) % group_size
    if pad:
        W = torch.cat([W, W.new_zeros(d_out, pad)], dim=1)
    d_in_padded = W.shape[1]
    n_groups = d_in_padded // group_size

    W_grouped = W.float().reshape(d_out, n_groups, group_size)  # (d_out, G, gs)

    w_min = W_grouped.amin(dim=2)  # (d_out, G)
    w_max = W_grouped.amax(dim=2)

    scales = (w_max - w_min) / max_int  # (d_out, G)
    scales = scales.clamp(min=1e-8)     # avoid /0

    # zero-point: z = clip(round(-wmin / s), 0, max_int)
    zeros = torch.clamp(torch.round(-w_min / scales), 0, max_int).to(torch.int32)

    # Quantise: q = clip(round(w / s) + z, 0, max_int)
    s_exp = scales.unsqueeze(2)      # (d_out, G, 1)
    z_exp = zeros.unsqueeze(2).float()
    q_int = torch.clamp(torch.round(W_grouped / s_exp) + z_exp, 0, max_int).to(torch.int32)
    q_int = q_int.reshape(d_out, d_in_padded)

    return q_int, scales, zeros


def group_dequantize(
    q_int: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bits: int,
    group_size: int = 128,
    original_d_in: Optional[int] = None,
) -> torch.Tensor:
    """
    Dequantize: ŵ = s * (q - z)
    Returns fp32 tensor of shape (d_out, original_d_in).
    """
    d_out, d_in_padded = q_int.shape
    n_groups = scales.shape[1]

    q_grouped = q_int.float().reshape(d_out, n_groups, group_size)
    z_exp = zeros.float().unsqueeze(2)
    s_exp = scales.unsqueeze(2)

    W_hat = s_exp * (q_grouped - z_exp)
    W_hat = W_hat.reshape(d_out, d_in_padded)

    if original_d_in is not None:
        W_hat = W_hat[:, :original_d_in]
    return W_hat


def packed_bytes(q_int: torch.Tensor, bits: int) -> int:
    """Bytes for bit-packed payload (without metadata)."""
    n_weights = q_int.numel()
    return math.ceil(n_weights * bits / 8)


def quantize_model_uniform(
    model,
    bits: int,
    group_size: int = 128,
    inplace: bool = True,
) -> Dict[str, dict]:
    """
    Apply Q1 uniform PTQ to all eligible projections.

    Returns dict mapping projection name → {
        "q_int": Tensor (CPU, int32),
        "scales": Tensor (CPU, fp32),
        "zeros":  Tensor (CPU, int32),
        "original_d_in": int,
        "packed_bytes": int,
        "metadata_bytes": int,
        "total_bytes": int,
        "bits": int,
        "group_size": int,
    }
    If inplace=True, replaces model weights with dequantized fp16.
    """
    result = {}
    for name, module in enumerate_eligible_projections(model):
        W = module.weight.data.float().cpu()
        d_out, d_in = W.shape

        q_int, scales, zeros = group_quantize(W, bits, group_size)

        # Dequantize for numerical evaluation.
        W_hat = group_dequantize(q_int, scales, zeros, bits, group_size, d_in)

        if inplace:
            module.weight.data = W_hat.to(module.weight.dtype).to(module.weight.device)

        payload = packed_bytes(q_int[:, :d_in], bits)
        n_groups = math.ceil(d_in / group_size)
        meta = n_groups * 4 * d_out  # 4 B per group per row

        result[name] = {
            "q_int": q_int.cpu(),
            "scales": scales.cpu(),
            "zeros": zeros.cpu(),
            "original_d_in": d_in,
            "packed_bytes": payload,
            "metadata_bytes": meta,
            "total_bytes": payload + meta,
            "bits": bits,
            "group_size": group_size,
        }

    return result


def save_quant_summary(result: Dict, path: str):
    summary = {
        name: {k: v for k, v in info.items() if not isinstance(v, torch.Tensor)}
        for name, info in result.items()
    }
    total_packed = sum(v["packed_bytes"] for v in result.values())
    total_meta = sum(v["metadata_bytes"] for v in result.values())
    summary["_totals"] = {
        "packed_bytes": total_packed,
        "metadata_bytes": total_meta,
        "total_bytes": total_packed + total_meta,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
