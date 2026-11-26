"""
Part C: KV cache compression for Qwen/Qwen2.5-1.5B-Instruct.

Five methods compared at one matched INT4 reference-byte budget (the actual
serialized-byte size of the uniform channel-wise-K / token-wise-V INT4 cache):

  C2.i  sink_plus_recent_eviction   – 8 sink + 64 recent + fill to budget (FP16)
  C2.ii question_aware_eviction     – 8 sink + 64 recent + top question-scored (FP16)
  C3.i  quantize_kv_tokenwise       – INT4 token-wise for both K and V
  C3.ii quantize_kv_channelwise_k   – INT4 channel-wise-K / token-wise-V
  C4    hybrid_kv                   – FP16 selected + INT2 channel-wise for rest

IMPORTANT: "Cache editing must preserve the original absolute rotary position IDs
of retained tokens and use a consistent physical cache length when appending new
tokens." (PDF §2 C1)

The KV cache in transformers (DynamicCache) stores key/value tensors per layer.
Shape: each layer's K or V is (B, H_kv, L, d_h).

Quantisation convention (fixed across C3/C4):
  group_size=64, FP16 scale + FP16 zero-point (4 bytes metadata per group),
  asymmetric integer quantisation.
  Zero-point: z = clip(round(-wmin / s), 0, 2^b - 1).
  Trailing r = L mod 64 tokens remain FP16 for channel-wise-K layout.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from common.byte_accounting import kv_cache_bytes_int, kv_cache_bytes_fp16


GROUP_SIZE = 64
SINK_TOKENS = 8
RECENT_TOKENS = 64
META_BYTES_PER_GROUP = 4   # FP16 scale (2 B) + FP16 zero-point (2 B)


# ── Quantisation primitives ────────────────────────────────────────────────────

def _quantize_groups(
    x: torch.Tensor,
    bits: int,
    group_size: int = GROUP_SIZE,
    dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Asymmetric group quantisation along `dim`.

    Returns (q_int, scales, zeros, original_size_along_dim).
    Trailing incomplete group is zero-padded internally.
    """
    orig_size = x.shape[dim]
    max_int = 2 ** bits - 1

    # Move target dim to last position.
    x = x.movedim(dim, -1)
    shape_prefix = x.shape[:-1]
    x_flat = x.reshape(-1, orig_size).float()  # (N, L)

    pad = (group_size - orig_size % group_size) % group_size
    if pad:
        x_flat = torch.cat([x_flat, x_flat.new_zeros(x_flat.shape[0], pad)], dim=1)
    L_pad = x_flat.shape[1]
    n_groups = L_pad // group_size

    x_g = x_flat.reshape(-1, n_groups, group_size)   # (N, G, gs)
    w_min = x_g.amin(dim=2, keepdim=True)
    w_max = x_g.amax(dim=2, keepdim=True)
    scales = ((w_max - w_min) / max_int).clamp(min=1e-8)
    zeros = torch.clamp(torch.round(-w_min / scales), 0, max_int).to(torch.int32)
    q_int = torch.clamp(torch.round(x_g / scales) + zeros.float(), 0, max_int).to(torch.int32)

    # Remove padding from q_int.
    q_int = q_int.reshape(-1, L_pad)[:, :orig_size].reshape(*shape_prefix, orig_size)
    scales = scales.squeeze(-1).reshape(*shape_prefix, n_groups)
    zeros = zeros.squeeze(-1).reshape(*shape_prefix, n_groups)

    return q_int, scales, zeros, orig_size


def _dequantize_groups(
    q_int: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bits: int,
    group_size: int = GROUP_SIZE,
    dim: int = -1,
) -> torch.Tensor:
    """Inverse of _quantize_groups."""
    orig_size = q_int.shape[dim]

    q_int = q_int.movedim(dim, -1)
    shape_prefix = q_int.shape[:-1]
    q_flat = q_int.reshape(-1, orig_size).float()
    s_flat = scales.reshape(-1, scales.shape[-1])  # (N, G)
    z_flat = zeros.reshape(-1, zeros.shape[-1]).float()

    n_groups = s_flat.shape[1]
    # Expand s, z to (N, L).
    s_exp = s_flat.unsqueeze(-1).expand(-1, -1, group_size).reshape(-1, n_groups * group_size)
    z_exp = z_flat.unsqueeze(-1).expand(-1, -1, group_size).reshape(-1, n_groups * group_size)
    s_exp = s_exp[:, :orig_size]
    z_exp = z_exp[:, :orig_size]

    w_hat = s_exp * (q_flat - z_exp)
    w_hat = w_hat.reshape(*shape_prefix, orig_size).movedim(-1, dim)
    return w_hat


# ── Byte accounting for a single layer's K or V tensor ────────────────────────

def _channel_wise_k_bytes(L: int, n_channels: int, bits: int, group_size: int = GROUP_SIZE) -> int:
    """
    Channel-wise-K: groups along token axis per channel.
    n_channels = H_kv * d_h.
    """
    full_groups = L // group_size
    tail = L % group_size
    per_channel = full_groups * (group_size * bits // 8 + META_BYTES_PER_GROUP) + 2 * tail
    return n_channels * per_channel


def _token_wise_bytes(n_tokens: int, dim: int, bits: int, group_size: int = GROUP_SIZE) -> int:
    """Token-wise: groups along feature dim per token."""
    n_groups = math.ceil(dim / group_size)
    per_token = math.ceil(dim * bits / 8) + n_groups * META_BYTES_PER_GROUP
    return n_tokens * per_token


# ── C2: Token eviction ─────────────────────────────────────────────────────────

def _int4_reference_budget(cache: DynamicCache, n_layers: int, n_kv_heads: int, head_dim: int) -> int:
    """Compute the matched INT4 budget = channel-wise-K/token-wise-V INT4 bytes."""
    L = cache.get_seq_length()
    return kv_cache_bytes_int(L, n_layers, n_kv_heads, head_dim, bits=4, group_size=GROUP_SIZE)


def sink_plus_recent_eviction(
    cache: DynamicCache,
    budget_bytes: int,
    n_kv_heads: int,
    head_dim: int,
    sink: int = SINK_TOKENS,
    recent: int = RECENT_TOKENS,
) -> Tuple[DynamicCache, dict]:
    """
    C2.i: Retain sink tokens + most-recent tokens, fill remainder with more recent tokens.
    Budget is FP16 bytes of retained positions.

    Returns (new_cache, metadata).
    metadata contains: n_retained, retained_positions (list), bytes_used.
    """
    n_layers = len(cache.key_cache)
    L = cache.get_seq_length()
    bytes_per_pos = 4 * n_layers * n_kv_heads * head_dim  # FP16 K+V per position
    max_positions = budget_bytes // bytes_per_pos

    # Must retain at least sink + recent.
    safeguard = sink + recent
    n_retained = min(L, max(safeguard, max_positions))

    # Selection: sink (0..sink-1) + most recent (L-recent..L-1) + middle fill.
    sink_pos = list(range(min(sink, L)))
    recent_pos = list(range(max(sink, L - recent), L))
    already = set(sink_pos) | set(recent_pos)
    remaining_budget = n_retained - len(already)

    # Fill with most-recent positions not already selected.
    fill = [i for i in range(L - recent - 1, sink - 1, -1) if i not in already]
    fill = fill[:remaining_budget]
    retain_pos = sorted(set(sink_pos) | set(recent_pos) | set(fill))[:n_retained]

    new_cache = _evict_to_positions(cache, retain_pos)
    return new_cache, {
        "n_retained": len(retain_pos),
        "retained_positions": retain_pos,
        "bytes_used": len(retain_pos) * bytes_per_pos,
        "method": "sink_plus_recent",
    }


def question_aware_eviction(
    cache: DynamicCache,
    budget_bytes: int,
    model,
    question_input_ids: torch.Tensor,
    n_kv_heads: int,
    head_dim: int,
    device: str,
    sink: int = SINK_TOKENS,
    recent: int = RECENT_TOKENS,
) -> Tuple[DynamicCache, dict]:
    """
    C2.ii: Question-aware retention (SnapKV-inspired).
    Score past positions using attention from the question suffix to the context.
    Fix 8 sink + 64 recent as safeguards; fill with top-scored positions.

    NOTE: This is a simplified fixed policy, NOT a full reproduction of the
    published head-specific SnapKV algorithm.  Attributed accordingly.
    """
    n_layers = len(cache.key_cache)
    L = cache.get_seq_length()
    bytes_per_pos = 4 * n_layers * n_kv_heads * head_dim
    max_positions = budget_bytes // bytes_per_pos

    safeguard_pos = set(range(min(sink, L))) | set(range(max(sink, L - recent), L))
    n_fill = max(0, min(L, max_positions) - len(safeguard_pos))

    # Compute attention scores: run question tokens through model attending to cache.
    # Use the query projections to score cached keys.
    importance = _score_positions_by_attention(model, cache, question_input_ids, device)
    # importance: (L,) aggregated over layers and heads

    # Rank positions by importance, excluding safeguards.
    candidates = [(importance[i].item(), i) for i in range(L) if i not in safeguard_pos]
    candidates.sort(reverse=True)
    fill_pos = [pos for _, pos in candidates[:n_fill]]

    retain_pos = sorted(safeguard_pos | set(fill_pos))

    new_cache = _evict_to_positions(cache, retain_pos)
    return new_cache, {
        "n_retained": len(retain_pos),
        "retained_positions": retain_pos,
        "bytes_used": len(retain_pos) * bytes_per_pos,
        "method": "question_aware",
    }


def _score_positions_by_attention(model, cache, question_ids: torch.Tensor, device: str) -> torch.Tensor:
    """
    Run question tokens through the model (teacher-forced), extract attention weights
    over the cached context, and aggregate over layers and heads.
    Returns (L,) importance scores (sum of attention over question tokens).

    Uses the working SDPA model; we extract query projections and compute
    FP32 attention against cached keys.
    """
    L = cache.get_seq_length()
    importance = torch.zeros(L, device=device)

    with torch.no_grad():
        # Run a forward pass to get attention patterns.
        outputs = model(
            question_ids.to(device),
            past_key_values=cache,
            output_attentions=True,
            use_cache=False,  # don't extend cache
        )
        if outputs.attentions is not None:
            for layer_attn in outputs.attentions:
                # layer_attn: (B, H, Q_len, KV_len)
                # Sum over question tokens and heads to get (KV_len,) importance.
                if layer_attn.shape[-1] >= L:
                    importance += layer_attn[0, :, :, :L].sum(dim=(0, 1)).float()

    return importance


def _evict_to_positions(cache: DynamicCache, retain_pos: List[int]) -> DynamicCache:
    """
    Create a new DynamicCache retaining only the given position indices.
    Original absolute position IDs are preserved via position_ids in subsequent calls.
    """
    idx = torch.tensor(retain_pos)
    new_cache = DynamicCache()
    for layer_idx in range(len(cache.key_cache)):
        K = cache.key_cache[layer_idx][:, :, idx, :]   # (B, H_kv, n_retain, d_h)
        V = cache.value_cache[layer_idx][:, :, idx, :]
        new_cache.update(K, V, layer_idx)
    return new_cache


# ── C3: KV quantisation ────────────────────────────────────────────────────────

def quantize_kv_tokenwise(
    cache: DynamicCache,
    bits: int = 4,
    group_size: int = GROUP_SIZE,
) -> Tuple[DynamicCache, dict]:
    """
    C3.i: Token-wise quantisation of both K and V.
    Groups along the feature (d_h) dimension per token.
    Dequantizes before returning to maintain a standard DynamicCache.
    """
    new_cache = DynamicCache()
    total_bytes = 0
    for layer_idx in range(len(cache.key_cache)):
        K = cache.key_cache[layer_idx]  # (B, H_kv, L, d_h)
        V = cache.value_cache[layer_idx]
        # Quantize/dequantize along dim=-1 (feature) per token.
        K_q, Ks, Kz, _ = _quantize_groups(K, bits, group_size, dim=-1)
        K_hat = _dequantize_groups(K_q, Ks, Kz, bits, group_size, dim=-1).to(K.dtype)
        V_q, Vs, Vz, _ = _quantize_groups(V, bits, group_size, dim=-1)
        V_hat = _dequantize_groups(V_q, Vs, Vz, bits, group_size, dim=-1).to(V.dtype)
        new_cache.update(K_hat, V_hat, layer_idx)

        L = K.shape[2]
        d_h = K.shape[3]
        # Bytes: n_tokens × (packed_payload + metadata)
        n_groups = math.ceil(d_h / group_size)
        total_bytes += 2 * K.shape[0] * K.shape[1] * L * (
            math.ceil(d_h * bits / 8) + n_groups * META_BYTES_PER_GROUP
        )
    return new_cache, {"method": "tokenwise_int4", "total_bytes": total_bytes, "bits": bits}


def quantize_kv_channelwise_k(
    cache: DynamicCache,
    bits: int = 4,
    group_size: int = GROUP_SIZE,
) -> Tuple[DynamicCache, dict]:
    """
    C3.ii: Channel-wise-K / token-wise-V quantisation.
    Keys: groups along token axis (dim=2) per channel.
    Values: groups along feature axis (dim=-1) per token.
    Trailing r = L mod 64 tokens remain FP16 for K (and V).
    """
    new_cache = DynamicCache()
    total_bytes = 0

    for layer_idx in range(len(cache.key_cache)):
        K = cache.key_cache[layer_idx]  # (B, H_kv, L, d_h)
        V = cache.value_cache[layer_idx]
        B, H_kv, L, d_h = K.shape

        tail = L % group_size
        L_full = L - tail

        # Key: quantize full groups along token dim, keep tail FP16.
        if L_full > 0:
            K_full = K[:, :, :L_full, :]
            # Quantize along dim=2 (token axis): groups of group_size tokens per channel.
            K_q, Ks, Kz, _ = _quantize_groups(K_full, bits, group_size, dim=2)
            K_full_hat = _dequantize_groups(K_q, Ks, Kz, bits, group_size, dim=2).to(K.dtype)
            K_hat = torch.cat([K_full_hat, K[:, :, L_full:, :]], dim=2)
        else:
            K_hat = K

        # Value: token-wise (feature-grouped).
        V_q, Vs, Vz, _ = _quantize_groups(V, bits, group_size, dim=-1)
        V_hat = _dequantize_groups(V_q, Vs, Vz, bits, group_size, dim=-1).to(V.dtype)

        new_cache.update(K_hat, V_hat, layer_idx)

        # Count bytes: M_b(L) formula from PDF.
        n_channels = B * H_kv * d_h
        full_groups_k = L_full // group_size
        k_bytes = n_channels * (
            full_groups_k * (group_size * bits // 8 + META_BYTES_PER_GROUP) + 2 * tail
        )
        n_groups_v = math.ceil(d_h / group_size)
        v_bytes = B * H_kv * L * (math.ceil(d_h * bits / 8) + n_groups_v * META_BYTES_PER_GROUP)
        total_bytes += k_bytes + v_bytes

    return new_cache, {
        "method": "channelwise_k_tokenwise_v_int4",
        "total_bytes": total_bytes,
        "bits": bits,
    }


# ── C4: Hybrid ─────────────────────────────────────────────────────────────────

def hybrid_kv(
    cache: DynamicCache,
    budget_bytes: int,
    importance_scores: torch.Tensor,   # (L,) from question_aware scoring
    n_kv_heads: int,
    head_dim: int,
    bits_low: int = 2,
    group_size: int = GROUP_SIZE,
    sink: int = SINK_TOKENS,
    recent: int = RECENT_TOKENS,
) -> Tuple[DynamicCache, dict]:
    """
    C4: Hybrid precision — FP16 for selected + INT2 channel-wise-K/token-wise-V
    for the rest.  Stay within budget_bytes.

    1. Keep sink (8) + recent (64) in FP16 always.
    2. Fill additional FP16 budget from top question-aware positions.
    3. Store remainder as INT2 channel-wise-K/token-wise-V in complete 64-token groups.
    4. Any incomplete residual for INT2 → FP16.
    5. Accumulate bytes; stop adding FP16 positions if budget would be exceeded.
    """
    n_layers = len(cache.key_cache)
    L = cache.get_seq_length()
    bytes_per_fp16_pos = 4 * n_layers * n_kv_heads * head_dim

    # Fixed safeguard positions.
    safeguard = set(range(min(sink, L))) | set(range(max(sink, L - recent), L))

    # Rank remaining positions by importance.
    candidates = [(importance_scores[i].item(), i) for i in range(L) if i not in safeguard]
    candidates.sort(reverse=True)

    # Greedily add FP16 positions within budget.
    fp16_positions = set(safeguard)

    # Estimate INT2 storage for the remaining positions.
    def int2_bytes_estimate(n_low: int) -> int:
        # Complete groups only; residual goes FP16.
        full_groups = n_low // group_size
        tail = n_low % group_size
        n_channels = n_layers * n_kv_heads * head_dim  # per K or V
        # K: channel-wise, V: token-wise
        k_bytes = 2 * n_channels * (
            full_groups * (group_size * bits_low // 8 + META_BYTES_PER_GROUP) + 2 * tail
        )
        n_g_v = math.ceil(head_dim / group_size)
        v_bytes = 2 * n_layers * n_kv_heads * n_low * (
            math.ceil(head_dim * bits_low / 8) + n_g_v * META_BYTES_PER_GROUP
        )
        # Position indices for FP16 selection (2 bytes per index).
        idx_bytes = len(fp16_positions) * 2
        return k_bytes + v_bytes + idx_bytes

    for _, pos in candidates:
        n_low = L - len(fp16_positions) - 1  # after adding this pos
        fp16_bytes = len(fp16_positions) * bytes_per_fp16_pos + bytes_per_fp16_pos
        low_bytes = int2_bytes_estimate(n_low)
        if fp16_bytes + low_bytes <= budget_bytes:
            fp16_positions.add(pos)
        else:
            break  # stop adding FP16 positions

    fp16_pos = sorted(fp16_positions)
    low_pos = sorted(i for i in range(L) if i not in fp16_positions)

    # Build combined cache.
    new_cache = DynamicCache()
    for layer_idx in range(len(cache.key_cache)):
        K = cache.key_cache[layer_idx]
        V = cache.value_cache[layer_idx]

        fp16_idx = torch.tensor(fp16_pos)
        low_idx = torch.tensor(low_pos) if low_pos else None

        K_fp16 = K[:, :, fp16_idx, :]
        V_fp16 = V[:, :, fp16_idx, :]

        if low_idx is not None:
            K_low = K[:, :, low_idx, :]
            V_low = V[:, :, low_idx, :]
            # Quantize low-precision part.
            Kq, Ks, Kz, _ = _quantize_groups(K_low, bits_low, group_size, dim=2)
            K_low_hat = _dequantize_groups(Kq, Ks, Kz, bits_low, group_size, dim=2).to(K.dtype)
            Vq, Vs, Vz, _ = _quantize_groups(V_low, bits_low, group_size, dim=-1)
            V_low_hat = _dequantize_groups(Vq, Vs, Vz, bits_low, group_size, dim=-1).to(V.dtype)
            # Reconstruct combined cache (interleave positions).
            K_combined = torch.empty_like(K)
            V_combined = torch.empty_like(V)
            K_combined[:, :, fp16_idx, :] = K_fp16
            K_combined[:, :, low_idx, :] = K_low_hat
            V_combined[:, :, fp16_idx, :] = V_fp16
            V_combined[:, :, low_idx, :] = V_low_hat
        else:
            K_combined = K_fp16
            V_combined = V_fp16

        new_cache.update(K_combined, V_combined, layer_idx)

    n_low = len(low_pos)
    return new_cache, {
        "method": "hybrid_int2_fp16",
        "n_fp16": len(fp16_pos),
        "n_low": n_low,
        "fp16_positions": fp16_pos,
        "bits_low": bits_low,
    }
