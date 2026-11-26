"""
Q2: Activation-aware Weight Quantization (AWQ-style), serialized-compensation variant.

Reference: Lin et al. (2024) - AWQ: Activation-aware Weight Quantization.

PDF §1 step 6: "Fold inverse scales into the adjacent compatible linear projection
when algebraically valid; otherwise serialize the compensating FP16 channel scales
and include their bytes in the matched budget."  This implementation takes the
SERIALIZED option for every eligible projection:

  - For projection W (d_out, d_in) choose a per-input-channel scale alpha (d_in,).
  - Quantize W' = W * diag(alpha) with the unchanged Q1 group quantizer
    (asymmetric, group 128 along input columns).
  - Store alpha as FP16 compensating scales: compensating_scale_bytes = 2 * d_in,
    included in total_bytes.  folded_into is always None.
  - The effective weight used for evaluation is
        W_hat = dequant(quant(W * diag(alpha))) * diag(1 / alpha),
    i.e. the compensation is applied to the quantized weight itself, so no other
    module is modified (norms, embeddings, other projections stay untouched).

No folding is performed.  (Folding into a predecessor would require scaling that
predecessor's OUTPUT rows, is invalid through SiLU for gate_proj, must respect GQA
for v_proj -> o_proj, and must happen before the predecessor is quantized; the
serialized option avoids all of that at a cost of 2*d_in bytes per projection.)

What is reproduced from the published AWQ procedure:
  - Activation-derived per-channel scales alpha = mean|x_j|^e.
  - A small recorded grid over the exponent e (SCALE_EXPONENTS), chosen per
    projection by minimum calibration output distortion (computed exactly from the
    calibration Gram matrix).
NOT reproduced:
  - Weight clipping search, joint block-level scale search, folding into
    preceding norms/projections, and fast low-bit inference kernels.

The scale search runs on GPU when available (W and the Gram matrix are moved to
the device in fp32 per projection and freed afterwards); results are returned on CPU.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch

from .baseline import enumerate_eligible_projections, CalibStats
from .quant_uniform import group_quantize, group_dequantize, packed_bytes


# Candidate grid: scale exponents to search.
SCALE_EXPONENTS = [round(e * 0.1, 1) for e in range(11)]  # 0.0 .. 1.0


def _default_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _per_channel_act_scale(X: torch.Tensor) -> torch.Tensor:
    """Mean |activation| per input channel: shape (d_in,)."""
    return X.float().abs().mean(dim=0).clamp(min=1e-6)


def awq_compensated_weight(W: torch.Tensor, alpha: torch.Tensor, bits: int, group_size: int = 128):
    """
    W_hat = dequant(quant(W * diag(alpha))) * diag(1/alpha), all fp32 on W's device.
    Returns (W_hat, q_int, scales, zeros).
    """
    d_in = W.shape[1]
    W_scaled = W * alpha.unsqueeze(0)
    q_int, scales, zeros = group_quantize(W_scaled, bits, group_size)
    W_scaled_hat = group_dequantize(q_int, scales, zeros, bits, group_size, d_in)
    W_hat = W_scaled_hat * (1.0 / alpha).unsqueeze(0)
    return W_hat, q_int, scales, zeros


def _trace_dgd(D: torch.Tensor, gram: torch.Tensor) -> float:
    """tr(D G D^T) = ||X D^T||_F^2 for X with Gram G."""
    return ((D @ gram) * D).sum().item()


def awq_scale_search(
    W: torch.Tensor,
    calib,
    bits: int,
    group_size: int = 128,
    exponents: List[float] = SCALE_EXPONENTS,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, float, float]:
    """
    Search for the best per-channel scale via a small exponent grid.

    W      (d_out, d_in) weight matrix for one projection (any device/dtype).
    calib  :class:`CalibStats` for this projection (needs the Gram matrix)
           OR a raw X tensor of shape (n_tokens, d_in) (legacy fallback).
    device Compute device; defaults to CUDA when available.

    The reconstruction error is the mean squared calibration-output distortion
        (1 / (N * d_out)) * || (W_hat - W) X^T ||_F^2
    computed from the Gram matrix via ||D X^T||_F^2 = tr(D G D^T).

    Returns (best_scale (d_in,) fp32 CPU, best_exp, best_err).
    """
    dev = _default_device(device)
    Wd = W.detach().to(dev, torch.float32)
    d_out = Wd.shape[0]
    if isinstance(calib, CalibStats):
        if calib.gram is None:
            raise ValueError(
                "awq_scale_search needs the Gram matrix; collect stats with "
                "collect_calibration_stats(..., with_gram=True)."
            )
        act_scale = calib.col_absmean.to(dev, torch.float32).clamp(min=1e-6)
        gram = calib.gram.to(dev, torch.float32)
        n_tokens = calib.n_tokens
    else:
        X = calib.to(dev, torch.float32)
        act_scale = _per_channel_act_scale(X)
        gram = X.t() @ X
        n_tokens = X.shape[0]
        del X

    best_err = float("inf")
    best_scale = torch.ones(Wd.shape[1], device=dev)
    best_exp = 0.0
    for exp in exponents:
        alpha = act_scale.pow(exp).clamp(min=1e-8)
        W_hat, *_ = awq_compensated_weight(Wd, alpha, bits, group_size)
        err = _trace_dgd(W_hat - Wd, gram) / (n_tokens * d_out)
        if err < best_err:
            best_err = err
            best_scale = alpha.clone()
            best_exp = exp
        del W_hat

    out = best_scale.cpu()
    del Wd, gram, act_scale, best_scale
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return out, best_exp, best_err


def awq_search_scales(
    model,
    calibration_inputs: Dict[str, "CalibStats"],
    bits: int,
    group_size: int = 128,
    exponents: List[float] = SCALE_EXPONENTS,
    device: Optional[str] = None,
) -> Dict[str, dict]:
    """
    Run the scale search for every eligible projection WITHOUT modifying the model.

    Also performs the PDF §1 step-6 pre-rounding check per projection: compare
    W_r = (W * diag(alpha)) * diag(1/alpha) with W in fp32 (max abs diff, and the
    relative output error tr(D G D^T) / tr(W G W^T) with D = W_r - W).

    Returns name -> {alpha (CPU fp32), best_exp, calibration_error,
                     prerounding_max_abs_diff, prerounding_rel_output_err}.
    """
    dev = _default_device(device)
    out = {}
    for name, module in enumerate_eligible_projections(model):
        calib = calibration_inputs[name]
        alpha, best_exp, best_err = awq_scale_search(
            module.weight.data, calib, bits, group_size, exponents, dev
        )
        entry = {"alpha": alpha, "best_exp": best_exp, "calibration_error": best_err}
        if isinstance(calib, CalibStats) and calib.gram is not None:
            W = module.weight.data.to(dev, torch.float32)
            a = alpha.to(dev)
            W_r = (W * a.unsqueeze(0)) * (1.0 / a).unsqueeze(0)
            G = calib.gram.to(dev, torch.float32)
            D = W_r - W
            denom = _trace_dgd(W, G)
            entry["prerounding_max_abs_diff"] = D.abs().max().item()
            entry["prerounding_rel_output_err"] = _trace_dgd(D, G) / denom if denom > 0 else 0.0
            del W, a, W_r, G, D
        out[name] = entry
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def apply_awq_quantization(
    model,
    calibration_inputs: Dict[str, "CalibStats"],
    bits: int,
    group_size: int = 128,
    exponents: List[float] = SCALE_EXPONENTS,
    device: Optional[str] = None,
    searched: Optional[Dict[str, dict]] = None,
    keep_codes: bool = False,
) -> Dict[str, dict]:
    """
    Apply AWQ-style quantization (serialized FP16 compensating scales) to every
    eligible projection.  Only eligible projection weights are modified.

    searched    Optional output of :func:`awq_search_scales` to reuse (so the
                search is not run twice); otherwise the search runs here.
    keep_codes  If True, also return q_int/scales/zeros (large; off by default).

    Returns dict mapping name -> {
        original_d_in, awq_scale (d_in,) fp32 CPU, best_exp, calibration_error,
        packed_bytes, metadata_bytes, compensating_scale_bytes, total_bytes,
        folded_into (always None), bits, group_size, [q_int, scales, zeros]
    }
    """
    dev = _default_device(device)
    # Search everything first (on unmodified weights), then quantize.
    if searched is None:
        searched = awq_search_scales(model, calibration_inputs, bits, group_size, exponents, dev)

    result = {}
    for name, module in enumerate_eligible_projections(model):
        s = searched[name]
        alpha = s["alpha"]
        W = module.weight.data.to(dev, torch.float32)
        d_out, d_in = W.shape

        W_hat, q_int, scales, zeros = awq_compensated_weight(W, alpha.to(dev), bits, group_size)
        module.weight.data = W_hat.to(module.weight.dtype).to(module.weight.device)

        payload = packed_bytes(q_int[:, :d_in], bits)
        meta = math.ceil(d_in / group_size) * 4 * d_out
        comp = 2 * d_in  # serialized FP16 compensating scale per input channel

        entry = {
            "original_d_in": d_in,
            "awq_scale": alpha.cpu(),
            "best_exp": s["best_exp"],
            "calibration_error": s["calibration_error"],
            "packed_bytes": payload,
            "metadata_bytes": meta,
            "compensating_scale_bytes": comp,
            "total_bytes": payload + meta + comp,
            "folded_into": None,
            "bits": bits,
            "group_size": group_size,
        }
        for k in ("prerounding_max_abs_diff", "prerounding_rel_output_err"):
            if k in s:
                entry[k] = s[k]
        if keep_codes:
            entry["q_int"] = q_int.cpu()
            entry["scales"] = scales.cpu()
            entry["zeros"] = zeros.cpu()
        result[name] = entry
        del W, W_hat, q_int, scales, zeros

    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return result
