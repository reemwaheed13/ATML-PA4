"""
A1 / A2 / A3: attention operators (PDF §2 Part A).

A1 dense_causal_attention  explicit Q K^T, causal mask, numerically stable softmax, x V.
                           Materialises the (L, L) score matrix.
A2 tiled_causal_attention  exact blockwise forward with the online-softmax update
                           (m' = max(m, max s); l' = e^{m-m'} l + sum e^{s-m'};
                           o' = e^{m-m'} o + sum e^{s-m'} V; A = o'/l').  Never materialises
                           the full score matrix: per (query tile, key tile) it holds one
                           tile x tile score block; running m, l, o are kept in FP32.
                           Key tiles entirely in the future of a query tile are skipped.
                           (FP64 inputs keep FP64 statistics, used only by the unit tests.)
A3 sdpa_attention          torch.nn.functional.scaled_dot_product_attention, optionally
                           forcing a backend; `dispatched_sdpa_backend` records the kernel
                           that actually ran (from CUDA profiler kernel names), instead of
                           assuming FlashAttention.

All three accept an optional key-validity mask `key_mask` (B, L) bool (True = valid);
a query row with no valid causal key ("all-masked") returns 0 in A1/A2 (no NaN, finite
gradients).  What SDPA returns for such rows is recorded, not assumed.

LIMITATIONS (stated in every Part A result): A1/A2 are Python reference
implementations.  The A2 backward pass is plain autograd through the Python loop and
stores every tile's intermediates, so its backward memory/time say nothing about a fused
IO-aware kernel; A2 is NOT FlashAttention.
"""

import math
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _stable_softmax(s: torch.Tensor) -> torch.Tensor:
    """softmax over the last dim that returns 0 for rows that are entirely -inf."""
    m = s.amax(dim=-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp(s - m)
    l = e.sum(dim=-1, keepdim=True)
    return e / torch.where(l > 0, l, torch.ones_like(l))


def _allowed(L: int, device, key_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """(B or 1, 1, L, L) bool: causal & valid key."""
    c = torch.ones(L, L, dtype=torch.bool, device=device).tril()
    if key_mask is None:
        return c.view(1, 1, L, L)
    return c.view(1, 1, L, L) & key_mask.view(key_mask.shape[0], 1, 1, L)


def dense_causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                           scale: Optional[float] = None,
                           key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """A1. Q, K, V (B, H, L, d) -> (B, H, L, d), computed in the input dtype."""
    L, d = Q.shape[-2], Q.shape[-1]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    s = torch.matmul(Q, K.transpose(-2, -1)) * scale
    s = s.masked_fill(~_allowed(L, Q.device, key_mask), float("-inf"))
    return torch.matmul(_stable_softmax(s), V)


def tiled_causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, tile_size: int = 64,
                           scale: Optional[float] = None,
                           key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """A2. Exact online-softmax tiled causal attention; FP32 running statistics,
    output cast back to the input dtype."""
    B, H, L, d = Q.shape
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    acc = torch.float64 if Q.dtype == torch.float64 else torch.float32
    out_tiles = []
    for qs in range(0, L, tile_size):
        qe = min(qs + tile_size, L)
        Qi = Q[:, :, qs:qe].to(acc)
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=Q.device, dtype=acc)
        l = torch.zeros(B, H, qe - qs, 1, device=Q.device, dtype=acc)
        o = torch.zeros(B, H, qe - qs, d, device=Q.device, dtype=acc)
        qpos = torch.arange(qs, qe, device=Q.device).view(-1, 1)
        for ks in range(0, qe, tile_size):          # tiles with ks >= qe are fully future: skipped
            ke = min(ks + tile_size, L)
            s = torch.matmul(Qi, K[:, :, ks:ke].to(acc).transpose(-2, -1)) * scale
            allowed = torch.arange(ks, ke, device=Q.device).view(1, -1) <= qpos
            allowed = allowed.view(1, 1, qe - qs, ke - ks)
            if key_mask is not None:
                allowed = allowed & key_mask[:, ks:ke].view(B, 1, 1, ke - ks)
            s = s.masked_fill(~allowed, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            alpha = torch.exp(m - m_safe)            # 0 when m = -inf
            p = torch.exp(s - m_safe)
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            o = alpha * o + torch.matmul(p, V[:, :, ks:ke].to(acc))
            m = m_new
        out_tiles.append(o / torch.where(l > 0, l, torch.ones_like(l)))
    return torch.cat(out_tiles, dim=2).to(Q.dtype)


# ── A3: SDPA ──────────────────────────────────────────────────────────────────

def _sdpa_backends():
    from torch.nn.attention import SDPBackend
    names = {"flash": "FLASH_ATTENTION", "efficient": "EFFICIENT_ATTENTION", "math": "MATH",
             "cudnn": "CUDNN_ATTENTION"}
    return {k: getattr(SDPBackend, v) for k, v in names.items() if hasattr(SDPBackend, v)}


def sdpa_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: Optional[float] = None,
                   backend: Optional[str] = None,
                   key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """A3. backend None = default dispatch; else 'flash' | 'efficient' | 'math' | 'cudnn'
    is forced via torch.nn.attention.sdpa_kernel (raises RuntimeError if unsupported)."""
    scale = 1.0 / math.sqrt(Q.shape[-1]) if scale is None else scale
    if backend is None:
        ctx = nullcontext()
    else:
        from torch.nn.attention import sdpa_kernel
        bk = _sdpa_backends()
        if backend not in bk:
            raise RuntimeError(f"SDPA backend {backend!r} not available in this torch build: not supported")
        ctx = sdpa_kernel([bk[backend]])
    with ctx:
        if key_mask is None:
            return F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=scale)
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=_allowed(Q.shape[-2], Q.device, key_mask),
                                              scale=scale)


KERNEL_PATTERNS = [("flash", ("flash",)), ("cudnn", ("cudnn",)),
                   ("efficient", ("efficient", "fmha", "mem_eff", "cutlass")),]


def classify_kernels(names: List[str]) -> str:
    low = [n.lower() for n in names]
    for label, pats in KERNEL_PATTERNS:
        if any(p in n for n in low for p in pats):
            return label
    return "math" if low else "unknown"


def dispatched_sdpa_backend(Q, K, V, backward: bool = False) -> dict:
    """Run default-dispatch SDPA once under the CUDA profiler and classify the kernels
    that actually executed.  Returns {'backend', 'kernels' (top names)}."""
    if not Q.is_cuda:
        return {"backend": "math_cpu", "kernels": []}
    from torch.profiler import ProfilerActivity, profile
    q, k, v = (t.detach().clone().requires_grad_(backward) for t in (Q, K, V))
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = sdpa_attention(q, k, v)
        if backward:
            out.sum().backward()
        torch.cuda.synchronize()
    names = sorted({e.key for e in prof.key_averages() if e.device_type is not None
                    and "cuda" in str(e.device_type).lower()})
    if not names:
        names = sorted({e.key for e in prof.key_averages()})
    attn_like = [n for n in names if any(p in n.lower() for p in
                                         ("attn", "attention", "fmha", "flash", "softmax", "sdp", "cudnn"))]
    return {"backend": classify_kernels(attn_like or names), "kernels": (attn_like or names)[:12]}


# ── Verification helpers ──────────────────────────────────────────────────────

def verify_causality(attn_fn, B=1, H=2, L=32, d=16, j=None, device="cpu", dtype=torch.float32) -> dict:
    """Perturb keys/values at every position >= j; outputs at every i < j must be
    unchanged and outputs at i >= j must change."""
    j = L // 2 if j is None else j
    g = torch.Generator(device="cpu").manual_seed(6304)
    Q, K, V = (torch.randn(B, H, L, d, generator=g).to(device, dtype) for _ in range(3))
    K2, V2 = K.clone(), V.clone()
    K2[:, :, j:] += 100.0 * torch.randn(B, H, L - j, d, generator=g).to(device, dtype)
    V2[:, :, j:] += 100.0
    a, b = attn_fn(Q, K, V), attn_fn(Q, K2, V2)
    past = (a[:, :, :j] - b[:, :, :j]).abs().max().item()
    fut = (a[:, :, j:] - b[:, :, j:]).abs().max().item()
    return {"max_change_before_j": past, "max_change_from_j": fut, "j": j, "L": L,
            "causal": past == 0.0 and fut > 0.0}


def compare(out: torch.Tensor, ref: torch.Tensor) -> dict:
    d = (out.float() - ref.float()).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rel_fro": float(d.norm() / ref.float().norm().clamp_min(1e-30)),
            "finite": bool(torch.isfinite(out).all())}
