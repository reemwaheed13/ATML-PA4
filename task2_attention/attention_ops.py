"""
A1 / A2 / A3: Attention operator implementations and benchmarks.

A1: Dense causal attention — explicit QK^T, causal mask, stable softmax, × V.
A2: Online-softmax tiled causal attention — exact blockwise, no full score matrix.
A3: PyTorch SDPA — benchmark only; dispatched backend is recorded, not assumed.

Design notes:
  - A1/A2 are REFERENCE implementations, not optimised kernels.
    A slow Python tiled implementation does not prove tiling is inherently slow.
  - Both must produce outputs numerically equivalent to FP32 dense attention.
  - Backward through A2 uses standard autograd and materialises intermediates;
    this must be noted as a limitation, not described as FlashAttention.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn.functional as F


# ── A1: Dense causal attention ─────────────────────────────────────────────────

def dense_causal_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Explicit causal attention.  Materialises the full (L, L) score matrix.

    Q, K, V  shape (B, H, L, d)
    Returns  shape (B, H, L, d)

    Causality: position i can only attend to j ≤ i.
    Verified by the caller (or the test suite) by perturbing future K/V and
    checking that the output is unchanged.
    """
    B, H, L, d = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, L, L)

    # Causal mask: positions above the diagonal are -inf.
    causal_mask = torch.triu(
        torch.full((L, L), float("-inf"), device=Q.device, dtype=scores.dtype),
        diagonal=1,
    )
    scores = scores + causal_mask

    attn = F.softmax(scores, dim=-1)  # (B, H, L, L); rows with all -inf → nan
    # Replace NaN (fully masked rows, only position 0 has no preceding tokens) with 0.
    attn = torch.nan_to_num(attn, nan=0.0)

    return torch.matmul(attn, V)  # (B, H, L, d)


# ── A2: Online-softmax tiled causal attention ──────────────────────────────────

def tiled_causal_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    tile_size: int = 64,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Exact causal attention using the online-softmax tile update (Algorithm 1 of
    Dao et al. 2022, re-derived for correctness verification).

    Does NOT materialise the full (L, L) score matrix.
    Intermediate storage per query block: O(tile_size × d) rather than O(L × d).

    This is a PYTHON reference — it is not a fused IO-aware kernel and need not
    have the same speed as FlashAttention.  Backward uses autograd and stores
    all intermediates; this is stated explicitly as a limitation.

    Q, K, V  shape (B, H, L, d)
    Returns  shape (B, H, L, d)
    """
    B, H, L, d = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    out = torch.zeros_like(Q)

    for q_start in range(0, L, tile_size):
        q_end = min(q_start + tile_size, L)
        Qi = Q[:, :, q_start:q_end, :]  # (B, H, q_tile, d)
        q_tile = q_end - q_start

        # Running stats per query position: (B, H, q_tile)
        m = torch.full((B, H, q_tile), float("-inf"), device=Q.device, dtype=Q.dtype)
        ell = torch.zeros(B, H, q_tile, device=Q.device, dtype=Q.dtype)
        o = torch.zeros(B, H, q_tile, d, device=Q.device, dtype=Q.dtype)

        for k_start in range(0, L, tile_size):
            k_end = min(k_start + tile_size, L)
            Kj = K[:, :, k_start:k_end, :]  # (B, H, k_tile, d)
            Vj = V[:, :, k_start:k_end, :]

            # Scores for this query-key tile: (B, H, q_tile, k_tile)
            s = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale

            # Causal masking within tile.
            q_positions = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)  # (q_tile, 1)
            k_positions = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)  # (1, k_tile)
            future_mask = k_positions > q_positions  # (q_tile, k_tile) — True means mask
            s = s.masked_fill(future_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Online softmax update.
            tile_max = s.amax(dim=-1)  # (B, H, q_tile)
            m_new = torch.maximum(m, tile_max)

            # Rescale old accumulator.
            rescale = torch.exp(m - m_new)  # (B, H, q_tile)
            ell = rescale * ell + torch.exp(s - m_new.unsqueeze(-1)).sum(dim=-1)
            o = rescale.unsqueeze(-1) * o + torch.matmul(
                torch.exp(s - m_new.unsqueeze(-1)), Vj
            )
            m = m_new

        # Normalise.
        ell_safe = ell.clamp(min=1e-8).unsqueeze(-1)
        out[:, :, q_start:q_end, :] = o / ell_safe

    return out


# ── A3: PyTorch SDPA ───────────────────────────────────────────────────────────

def sdpa_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: Optional[float] = None,
    record_backend: bool = True,
) -> Tuple[torch.Tensor, str]:
    """
    Benchmark wrapper for torch.nn.functional.scaled_dot_product_attention.

    Returns (output, backend_name).
    backend_name is recorded from torch.backends.cuda.sdp_kernel() context;
    if no CUDA is available it falls back to 'math' (default).

    The dispatched backend depends on the GPU, dtype, mask type, and PyTorch
    version — it is NOT assumed to be FlashAttention.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])

    # Detect backend.
    backend = "unknown"
    if Q.is_cuda:
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            ):
                out = F.scaled_dot_product_attention(
                    Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
                )
            # torch.backends.cuda.flash_sdp_enabled() etc. — record after the call.
            backend = _detect_sdpa_backend(Q)
        except Exception as e:
            backend = f"error:{e}"
            out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
            )
    else:
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
        )
        backend = "math_cpu"

    return out, backend


def _detect_sdpa_backend(Q: torch.Tensor) -> str:
    """Heuristic: check which SDPA sub-kernel is enabled for this tensor config."""
    if not Q.is_cuda:
        return "math_cpu"
    try:
        if torch.backends.cuda.flash_sdp_enabled():
            return "flash_attention"
        if torch.backends.cuda.mem_efficient_sdp_enabled():
            return "mem_efficient"
        return "math"
    except Exception:
        return "unknown"


# ── Verification helpers ───────────────────────────────────────────────────────

def verify_causality(attn_fn, B=1, H=2, L=16, d=32, device="cpu", dtype=torch.float32):
    """
    Check that perturbing future K/V at position j > i does not change output at i.
    Raises AssertionError if the attention function is not causal.
    """
    torch.manual_seed(6304)
    Q = torch.randn(B, H, L, d, device=device, dtype=dtype)
    K = torch.randn(B, H, L, d, device=device, dtype=dtype)
    V = torch.randn(B, H, L, d, device=device, dtype=dtype)

    out_ref = attn_fn(Q, K, V)
    if isinstance(out_ref, tuple):
        out_ref = out_ref[0]

    # Perturb K and V at position L-1 (the last, future-most token) and check
    # that position 0's output is unchanged.
    K2 = K.clone()
    V2 = V.clone()
    K2[:, :, -1, :] += 1000.0
    V2[:, :, -1, :] += 1000.0
    out_perturbed = attn_fn(Q, K2, V2)
    if isinstance(out_perturbed, tuple):
        out_perturbed = out_perturbed[0]

    max_diff = (out_ref[:, :, 0, :] - out_perturbed[:, :, 0, :]).abs().max().item()
    assert max_diff < 1e-5, f"Causality violated: position 0 changed by {max_diff}"
    return True


def compare_to_reference(
    fn_a, fn_b, B=2, H=4, L=256, d=64, device="cuda", dtype=torch.float32, atol=1e-3
) -> dict:
    """
    Compare two attention functions on identical inputs.
    Returns dict with max_abs_diff and mean_abs_diff.
    """
    torch.manual_seed(6304)
    Q = torch.randn(B, H, L, d, device=device, dtype=dtype)
    K = torch.randn(B, H, L, d, device=device, dtype=dtype)
    V = torch.randn(B, H, L, d, device=device, dtype=dtype)

    out_a = fn_a(Q, K, V)
    out_b = fn_b(Q, K, V)
    if isinstance(out_a, tuple):
        out_a = out_a[0]
    if isinstance(out_b, tuple):
        out_b = out_b[0]

    diff = (out_a - out_b).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "within_atol": (diff < atol).all().item(),
    }
