"""
Part C: KV cache compression for Qwen/Qwen2.5-1.5B-Instruct.

Five methods compared at one matched INT4 reference-byte budget (the actual
serialized-byte size of the uniform channel-wise-K / token-wise-V INT4 cache):

  C2.i  sink_plus_recent_eviction   – 8 sink + 64 recent + fill to budget (FP16)
  C2.ii question_aware_eviction     – 8 sink + 64 recent + top question-scored (FP16)
  C3.i  quantize_kv_tokenwise       – INT4 token-wise for both K and V
  C3.ii quantize_kv_channelwise_k   – INT4 channel-wise-K / token-wise-V
  C4    hybrid_kv                   – FP16 selected + INT2 channel-wise for rest

PDF §2 C1 (READ, p.9):
  "Cache editing must preserve the original absolute rotary position IDs of retained
   tokens and use a consistent physical cache length when appending new tokens."
Retained tokens were rotated at prefill at their ORIGINAL absolute positions and are
NEVER re-rotated.  New tokens appended after eviction take position_ids that continue
from the ORIGINAL context length, while their physical cache index is the (shorter)
retained length — see `forward_with_positions` / `greedy_generate_edited`.

Retention scope (PDF §2 C2 asks us to state this explicitly): we use GLOBAL
retention — a single position set shared across all layers and heads.  A standard
`DynamicCache` forward needs a consistent physical length per layer, and eviction
physically removes a token from every layer, so per-layer retention is incompatible
with ordinary decoding.  Importance is summed over all layers and query heads, and
the retained position indices are stored ONCE (counted as 2 bytes / retained
position), not once per layer/head.

transformers 4.57.1 cache API (INFERRED — transformers is NOT installed in this
local env; verify on Colab):
  * `DynamicCache` has NO `key_cache` / `value_cache` attributes.  It stores per-layer
    objects in `cache.layers[i]` exposing `.keys` / `.values` tensors of shape
    (B, H_kv, L, d_h).  `cache.crop(n)` truncates to the first n tokens.
  * We build an edited cache with an empty `DynamicCache()` and per-layer
    `cache.update(K, V, layer_idx)` (lazily appends layers, returns the full tensors).
  * Qwen2 RoPE helpers live in `transformers.models.qwen2.modeling_qwen2`
    (`apply_rotary_pos_emb`) and `model.model.rotary_emb(x, position_ids) -> (cos, sin)`.

Quantisation convention (fixed across C3/C4):
  group_size=64, FP16 scale + FP16 zero-point (4 bytes metadata per group),
  asymmetric integer quantisation.
  Zero-point: z = clip(round(-wmin / s), 0, 2^b - 1).
  Trailing r = L mod 64 tokens remain FP16 for channel-wise-K layout — for BOTH K
  and V, and this holds at the LIVE length during decode (freshly generated tokens
  stay FP16 until they complete a 64-token group; see `quantize_kv_channelwise_k`).
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from common.byte_accounting import kv_cache_bytes_int


GROUP_SIZE = 64
SINK_TOKENS = 8
RECENT_TOKENS = 64
META_BYTES_PER_GROUP = 4   # FP16 scale (2 B) + FP16 zero-point (2 B)
INDEX_BYTES = 2            # 1 uint16 position index per retained/selected token


# ── transformers 4.57.1 DynamicCache accessors (INFERRED; verify on Colab) ──────

def _num_layers(cache: DynamicCache) -> int:
    return len(cache.layers)


def _layer_keys(cache: DynamicCache, i: int) -> torch.Tensor:
    return cache.layers[i].keys


def _layer_values(cache: DynamicCache, i: int) -> torch.Tensor:
    return cache.layers[i].values


def _seq_len(cache: DynamicCache) -> int:
    if len(cache.layers) == 0:
        return 0
    k = cache.layers[0].keys
    return 0 if k is None else k.shape[2]


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA key/value head expansion, matching transformers' repeat_kv.
    (B, H_kv, L, d_h) -> (B, H_kv*n_rep, L, d_h) where query head i attends to
    KV head i // n_rep."""
    if n_rep == 1:
        return x
    B, H, L, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, L, D).reshape(B, H * n_rep, L, D)


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


def channelwise_split_lengths(L: int, group_size: int = GROUP_SIZE) -> Tuple[int, int]:
    """
    Split a channel-wise-K cache of length L into (n_quantized, n_fp16_tail).

    Only COMPLETE 64-token groups are quantised; the trailing r = L mod 64 tokens
    stay FP16 for both K and V.  Call this at the LIVE length during decode: as L
    crosses each multiple of 64 the FP16 tail completes a group and shrinks, so
    newly generated tokens remain FP16 only until they finish a group (PDF §2 C3).
    """
    tail = L % group_size
    return L - tail, tail


# ── C1: full-precision reference + position-correct decode primitives ──────────

def forward_with_positions(
    model,
    cache: DynamicCache,
    input_ids: torch.Tensor,
    abs_start_pos: int,
    device: Optional[str] = None,
):
    """
    Append `input_ids` to `cache` with position IDs decoupled from the physical index.

    PDF §2 C1 (READ): retained tokens keep their ORIGINAL absolute RoPE positions; new
    tokens continue the absolute-position sequence from the ORIGINAL context length,
    while the physical cache index is the (possibly shorter) retained length.

    So for each appended token at offset o in [0, T):
        position_ids[o]  = abs_start_pos + o     (TRUE absolute RoPE position)
        cache_position[o] = physical_len + o     (where it is written in the cache)

    When nothing was evicted, abs_start_pos == physical_len and position_ids ==
    cache_position, i.e. this reduces to ordinary generation (the C1 100%-retention
    check).  Retained keys are used as-is and never re-rotated.

    Qwen2ForCausalLM.forward signature relied on (INFERRED; verify on Colab):
        forward(input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None, cache_position=None, ...)
    `position_ids` drives RoPE; `cache_position` indexes the cache / causal mask.
    """
    device = device or next(model.parameters()).device
    T = input_ids.shape[1]
    physical_len = _seq_len(cache)
    position_ids = torch.arange(
        abs_start_pos, abs_start_pos + T, device=device
    ).unsqueeze(0)
    cache_position = torch.arange(physical_len, physical_len + T, device=device)
    return model(
        input_ids=input_ids.to(device),
        past_key_values=cache,
        use_cache=True,
        position_ids=position_ids,
        cache_position=cache_position,
    )


@torch.no_grad()
def greedy_generate_edited(
    model,
    cache: DynamicCache,
    priming_ids: torch.Tensor,
    original_context_len: int,
    max_new_tokens: int,
    eos_id: Optional[int] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Greedy decode on top of an edited cache, keeping position IDs correct (C1).

    `priming_ids`     the suffix appended after eviction (e.g. the question tokens),
                      whose first token sits at absolute position
                      `original_context_len`.
    `original_context_len`
                      the ORIGINAL (pre-eviction) context length — new absolute
                      positions continue from here, NOT from the retained length.

    Physical cache indices continue from the current (retained) cache length via
    `forward_with_positions`, so the two never get confused.
    """
    device = device or next(model.parameters()).device
    out = forward_with_positions(model, cache, priming_ids, original_context_len, device)
    next_abs = original_context_len + priming_ids.shape[1]
    logits = out.logits[:, -1, :]

    generated = []
    for _ in range(max_new_tokens):
        next_id = logits.argmax(dim=-1, keepdim=True)   # (B, 1)
        generated.append(next_id)
        if eos_id is not None and next_id.item() == eos_id:
            break
        out = forward_with_positions(model, cache, next_id, next_abs, device)
        next_abs += 1
        logits = out.logits[:, -1, :]

    return torch.cat(generated, dim=1) if generated else priming_ids[:, :0]


# ── C2: Token eviction ─────────────────────────────────────────────────────────

def _int4_reference_budget(cache: DynamicCache, n_layers: int, n_kv_heads: int, head_dim: int) -> int:
    """Compute the matched INT4 budget = channel-wise-K/token-wise-V INT4 bytes."""
    L = _seq_len(cache)
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
    Budget is FP16 bytes of retained positions PLUS 2 bytes per stored position index
    (retained positions are non-contiguous, so indices are materialised).

    Returns (new_cache, metadata).
    metadata contains: n_retained, retained_positions (list), bytes_used.
    """
    n_layers = _num_layers(cache)
    L = _seq_len(cache)
    bytes_per_pos = 4 * n_layers * n_kv_heads * head_dim + INDEX_BYTES  # FP16 K+V + index
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
    C2.ii: Question-aware retention (SnapKV-inspired, simplified fixed policy).
    Score past positions using attention from the question suffix to the context.
    Fix 8 sink + 64 recent as safeguards; fill with top-scored positions.

    Byte budget counts FP16 K+V per retained position plus 2 bytes per stored position
    index (GLOBAL index set, stored once — see module docstring on retention scope).

    NOTE: This is a simplified fixed policy, NOT a full reproduction of the published
    head-specific SnapKV algorithm.  Attributed accordingly.
    """
    n_layers = _num_layers(cache)
    L = _seq_len(cache)
    bytes_per_pos = 4 * n_layers * n_kv_heads * head_dim + INDEX_BYTES
    max_positions = budget_bytes // bytes_per_pos

    safeguard_pos = set(range(min(sink, L))) | set(range(max(sink, L - recent), L))
    n_fill = max(0, min(L, max_positions) - len(safeguard_pos))

    # Importance over past positions from the question suffix's attention to the cache.
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
        "importance": importance.detach().cpu(),
        "method": "question_aware",
    }


def _score_positions_by_attention(model, cache, question_ids: torch.Tensor, device: str) -> torch.Tensor:
    """
    Question-aware importance scoring for C2.ii / C4 (SnapKV-inspired, simplified).

    READ (PDF §2 C2, p.9): "fill it with the highest-ranked positions ... using
    attention from the final question/observation suffix to rank earlier positions.
    Keep the working SDPA model for generation; extract query projections and compute
    masked FP32 attention against existing rotary-positioned cached keys for importance
    scores. Do not switch to an unvalidated eager inference path solely to collect
    attention."

    Implementation (all choices stated per the handout):
      * Keep the SDPA model.  Register forward hooks on every layer's
        `self_attn.q_proj` to capture the question suffix's RAW (pre-RoPE) queries —
        SDPA does not return attention weights, so we recompute them ourselves.
      * Apply RoPE to those queries at their TRUE absolute positions [L, L+Q) using
        the model's own `rotary_emb` + `apply_rotary_pos_emb`.  The cached keys are
        already rotated (at prefill) and are used as-is.
      * FP32 masked softmax(QK^T / sqrt(d_h)) over the cached context keys, with GQA
        head expansion (num_attention_heads query heads -> num_key_value_heads KV
        heads, both READ from the config, never hard-coded).  Every question query
        attends to every cached context key (all context precedes the suffix), so the
        softmax normalises over the context keys; no extra causal mask is needed.
      * GLOBAL retention: importance is SUMMED over all layers and all query heads to
        a single (L,) vector (see module docstring).
      * The cache is NOT modified: the suffix is run as a STANDALONE forward (no past
        cache), and we assert the cache length is unchanged (before == after).  This
        standalone pass is the documented simplification vs. true SnapKV.
      * Gold evidence offsets are NEVER read here.

    INFERRED (verify on Colab; transformers 4.57.1 not installed locally):
      `model.model` is the Qwen2Model; `model.model.layers[i].self_attn.q_proj`;
      `model.model.rotary_emb(x, position_ids) -> (cos, sin)`;
      `transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb(q, k, cos, sin)`.
    """
    base = getattr(model, "model", model)       # Qwen2Model
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d_h = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)
    n_rep = n_heads // n_kv

    L_before = _seq_len(cache)

    # 1) Capture raw (pre-RoPE) q_proj activations per layer via forward hooks.
    captured = {}
    handles = []

    def _make_hook(idx):
        def hook(module, inp, out):
            captured[idx] = out.detach()
        return hook

    layers = base.layers
    for li, layer in enumerate(layers):
        handles.append(layer.self_attn.q_proj.register_forward_hook(_make_hook(li)))

    try:
        with torch.no_grad():
            # Standalone forward over the question suffix → original cache untouched.
            base(input_ids=question_ids.to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()

    assert _seq_len(cache) == L_before, "scoring must not modify the cache"

    Q = question_ids.shape[1]

    # 2) RoPE cos/sin at the TRUE absolute positions of the question suffix.
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    pos_ids = torch.arange(L_before, L_before + Q, device=device).unsqueeze(0)
    any_q = next(iter(captured.values()))
    cos, sin = base.rotary_emb(any_q.float(), pos_ids)

    importance = torch.zeros(L_before, device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(d_h)

    for li in range(len(layers)):
        q = captured[li].to(torch.float32)              # (B, Q, n_heads*d_h)
        B = q.shape[0]
        q = q.view(B, Q, n_heads, d_h).transpose(1, 2)  # (B, n_heads, Q, d_h)
        q_rot, _ = apply_rotary_pos_emb(q, q, cos.to(q.dtype), sin.to(q.dtype))

        K = _layer_keys(cache, li).to(torch.float32)    # (B, n_kv, L, d_h), rotated
        K = _repeat_kv(K, n_rep)                         # (B, n_heads, L, d_h)

        scores = torch.matmul(q_rot, K.transpose(-1, -2)) * scale  # (B, n_heads, Q, L)
        attn = torch.softmax(scores, dim=-1)            # over cached context keys
        importance += attn.sum(dim=(0, 1, 2))           # (L,)
        del q, q_rot, K, scores, attn

    assert torch.isfinite(importance).all(), "importance has non-finite values"
    assert importance.abs().sum() > 0, "importance scores are all zero"
    return importance


def _evict_to_positions(cache: DynamicCache, retain_pos: List[int]) -> DynamicCache:
    """
    Create a new DynamicCache retaining only `retain_pos`, IN THE GIVEN ORDER.

    Uses the transformers 4.57.1 layered API (`cache.layers[i].keys/.values`) and
    builds the edited cache with per-layer `update`.  `index_select` preserves the
    requested order, so callers pass ascending positions to keep absolute order.
    Retained tokens' rotary positions are unchanged (they are not re-rotated here);
    absolute positions are reasserted via `forward_with_positions` on the next append.
    """
    new_cache = DynamicCache()
    for layer_idx in range(_num_layers(cache)):
        K = _layer_keys(cache, layer_idx)      # (B, H_kv, L, d_h)
        V = _layer_values(cache, layer_idx)
        idx = torch.tensor(retain_pos, dtype=torch.long, device=K.device)
        new_cache.update(K.index_select(2, idx), V.index_select(2, idx), layer_idx)
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

    Decode-time contract (PDF §2 C3): apply this consistently to prompt-cache entries
    AND to subsequently generated entries.  Token-wise groups lie entirely within each
    token's feature vector, so every token (including freshly generated ones) is fully
    quantised the moment it is appended — there is no FP16 trailing tail for this
    layout (unlike channel-wise-K).
    """
    new_cache = DynamicCache()
    total_bytes = 0
    for layer_idx in range(_num_layers(cache)):
        K = _layer_keys(cache, layer_idx)  # (B, H_kv, L, d_h)
        V = _layer_values(cache, layer_idx)
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
    Trailing r = L mod 64 tokens remain FP16 for BOTH K and V.

    Decode-time contract (PDF §2 C3, verified here, not only at prefill): keys are
    grouped along the TOKEN axis, so a token only becomes quantisable once its
    64-token group is complete.  `channelwise_split_lengths(L)` computes
    (L - L%64, L%64) at the LIVE length, and this function quantises exactly the first
    `L - L%64` tokens of K while keeping the last `L%64` in FP16; V's trailing r tokens
    are kept FP16 the same way.  Re-invoking after each decode step (or batch of steps)
    therefore quantises any newly completed group while fresh tokens stay FP16 until
    they finish a group.  Byte accounting uses the handout's M_b(L) at the live length
    via `common.byte_accounting.kv_cache_bytes_int`.
    """
    new_cache = DynamicCache()
    total_bytes = 0

    for layer_idx in range(_num_layers(cache)):
        K = _layer_keys(cache, layer_idx)  # (B, H_kv, L, d_h)
        V = _layer_values(cache, layer_idx)
        B, H_kv, L, d_h = K.shape

        L_full, tail = channelwise_split_lengths(L, group_size)

        # Key: quantize complete groups along token dim, keep trailing r tokens FP16.
        if L_full > 0:
            K_full = K[:, :, :L_full, :]
            K_q, Ks, Kz, _ = _quantize_groups(K_full, bits, group_size, dim=2)
            K_full_hat = _dequantize_groups(K_q, Ks, Kz, bits, group_size, dim=2).to(K.dtype)
            K_hat = torch.cat([K_full_hat, K[:, :, L_full:, :]], dim=2)
        else:
            K_hat = K

        # Value: token-wise (feature-grouped); trailing r tokens kept FP16 too.
        if L_full > 0:
            V_full = V[:, :, :L_full, :]
            V_q, Vs, Vz, _ = _quantize_groups(V_full, bits, group_size, dim=-1)
            V_full_hat = _dequantize_groups(V_q, Vs, Vz, bits, group_size, dim=-1).to(V.dtype)
            V_hat = torch.cat([V_full_hat, V[:, :, L_full:, :]], dim=2)
        else:
            V_hat = V

        new_cache.update(K_hat, V_hat, layer_idx)

    # Count bytes with the handout formula M_b(L) (K + V combined, trailing r FP16).
    L = _seq_len(cache)
    n_layers = _num_layers(cache)
    H_kv = _layer_keys(cache, 0).shape[1]
    d_h = _layer_keys(cache, 0).shape[3]
    total_bytes = kv_cache_bytes_int(L, n_layers, H_kv, d_h, bits=bits, group_size=group_size)

    return new_cache, {
        "method": "channelwise_k_tokenwise_v_int4",
        "total_bytes": total_bytes,
        "bits": bits,
        "n_fp16_tail": L % group_size,
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
    5. Accumulate bytes (low-bit groups + metadata + 2-byte FP16 position indices +
       FP16 payload); stop adding FP16 positions if budget would be exceeded.
    """
    n_layers = _num_layers(cache)
    L = _seq_len(cache)
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
        idx_bytes = len(fp16_positions) * INDEX_BYTES
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
    for layer_idx in range(_num_layers(cache)):
        K = _layer_keys(cache, layer_idx)
        V = _layer_values(cache, layer_idx)

        fp16_idx = torch.tensor(fp16_pos, dtype=torch.long, device=K.device)
        low_idx = torch.tensor(low_pos, dtype=torch.long, device=K.device) if low_pos else None

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
