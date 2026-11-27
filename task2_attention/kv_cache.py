"""
Part C: KV-cache memory for Qwen/Qwen2.5-1.5B-Instruct (PDF §2 Part C).

Five compressed-cache configurations at ONE case-specific budget, the actual
serialized bytes of the uniform INT4 channel-wise-K / token-wise-V cache of the
pre-question context (M_4(L_ctx)):

  sink_recent      C2.i  FP16 eviction: 8 sink + most-recent positions
  question_aware   C2.ii FP16 eviction: 8 sink + 64 recent + top question-scored
  int4_tokenwise   C3.i  INT4, token-wise groups for K and V
  int4_channelwise C3.ii INT4, channel-wise-K / token-wise-V groups
  hybrid           C4    FP16 (8 sink + 64 recent + question-scored) + INT2
                         channel-wise-K / token-wise-V for the rest

Conventions fixed across C3/C4 (reported in every result file):
  * group size 64; per group an FP16 scale and an FP16 offset (4 metadata bytes) plus
    the bit-packed integer payload (2 values/byte at INT4, 4 values/byte at INT2);
  * asymmetric quantisation with the offset = FP16 group MINIMUM:
        s = fp16((max - min) / (2^b - 1)),  o = fp16(min),
        q = clamp(round((x - o) / s), 0, 2^b - 1),  x_hat = o + s * q.
    (An integer zero-point clamped to [0, 2^b-1] would collapse every all-positive
    group to its max; the FP16-offset form has no such clamp.)
  * only COMPLETE 64-token groups are quantised: the trailing r = L mod 64 tokens stay
    FP16 for K and V, for BOTH INT4 layouts, at the live length during decode
    (`requantize_completed_groups`).  With d_h = 128 the token-wise layout has the
    same byte count as the channel-wise one, M_b(L), which the packed encoders verify;
  * evaluation uses the DEQUANTISED values in an ordinary FP16 DynamicCache with SDPA:
    packed bytes are the storage cost, allocated GPU memory is reported separately.

Retention scope (C2): GLOBAL — one position set shared by all layers and heads
(eviction physically removes a position from every layer so the standard cache keeps
one physical length).  Importance is summed over layers, query heads and question
tokens.  Retained-position indices are stored once, 2 bytes (uint16) per FP16 position,
and are counted in every eviction / hybrid total.

Position handling (C1): retained tokens keep the RoPE rotation they received at their
ORIGINAL absolute position; appended tokens take position_ids continuing from the
original context length while their physical cache index is the (shorter) retained
length — `forward_with_positions`.

transformers 4.57.1 cache API: `cache.layers[i].keys / .values` (B, H_kv, L, d_h),
`cache.crop(n)`, `DynamicCache().update(K, V, layer_idx)`.  SDPA returns no attention
weights, so question-aware scoring hooks `q_proj`, applies RoPE at the true
positions and computes FP32 attention itself (`question_importance`), and C1 checks
that this recomputation matches the SDPA attention output.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import DynamicCache

from common.byte_accounting import kv_cache_bytes_int, kv_cache_bytes_fp16


GROUP_SIZE = 64
SINK_TOKENS = 8
RECENT_TOKENS = 64
META_BYTES_PER_GROUP = 4   # FP16 scale (2 B) + FP16 offset (2 B)
INDEX_BYTES = 2            # uint16 position index per FP16-retained / FP16-selected token
ZERO_POINT_CONVENTION = ("asymmetric; FP16 scale s=(max-min)/(2^b-1) and FP16 offset o=min per 64-element "
                         "group; q=clamp(round((x-o)/s),0,2^b-1); x_hat=o+s*q; no integer zero-point")


# ── cache accessors (transformers 4.57.1 layered DynamicCache) ────────────────

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
    """GQA expansion matching transformers' repeat_kv: query head h uses KV head h // n_rep."""
    if n_rep == 1:
        return x
    B, H, L, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, L, D).reshape(B, H * n_rep, L, D)


def cache_from_tensors(layers: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    c = DynamicCache()
    for i, (K, V) in enumerate(layers):
        c.update(K, V, i)
    return c


def clone_cache(cache: DynamicCache) -> DynamicCache:
    return cache_from_tensors([(_layer_keys(cache, i).clone(), _layer_values(cache, i).clone())
                               for i in range(_num_layers(cache))])


def cache_shape(cache: DynamicCache) -> dict:
    K = _layer_keys(cache, 0)
    return {"n_layers": _num_layers(cache), "B": K.shape[0], "H_kv": K.shape[1],
            "L": K.shape[2], "d_h": K.shape[3]}


def channelwise_split_lengths(L: int, group_size: int = GROUP_SIZE) -> Tuple[int, int]:
    """(n_quantised, n_fp16_tail) for a cache of live length L: only complete groups."""
    tail = L % group_size
    return L - tail, tail


# ── group quantisation and bit packing ────────────────────────────────────────

def quantize_groups(x: torch.Tensor, bits: int, group_size: int = GROUP_SIZE
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantise along the LAST dim in complete groups of `group_size`.

    x (..., n) with n % group_size == 0.  Returns
      q      uint8 (..., n)            integer codes in [0, 2^bits - 1]
      scale  fp16  (..., n/group_size)
      offset fp16  (..., n/group_size) (the group minimum)
    """
    n = x.shape[-1]
    if n % group_size:
        raise ValueError(f"last dim {n} is not a multiple of the group size {group_size}")
    qmax = 2 ** bits - 1
    xg = x.float().reshape(*x.shape[:-1], n // group_size, group_size)
    lo = xg.amin(-1)
    hi = xg.amax(-1)
    scale = ((hi - lo) / qmax).to(torch.float16)
    offset = lo.to(torch.float16)
    s = scale.float().unsqueeze(-1)
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = torch.round((xg - offset.float().unsqueeze(-1)) / safe)
    q = torch.where(s > 0, q, torch.zeros_like(q)).clamp_(0, qmax).to(torch.uint8)
    return q.reshape(x.shape), scale, offset


def dequantize_groups(q: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
                      group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Inverse of quantize_groups (FP32 output)."""
    n = q.shape[-1]
    qg = q.float().reshape(*q.shape[:-1], n // group_size, group_size)
    x = offset.float().unsqueeze(-1) + scale.float().unsqueeze(-1) * qg
    return x.reshape(q.shape)


def pack_bits(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 codes (..., n) with values < 2^bits into uint8 (..., n*bits/8)."""
    if bits not in (2, 4, 8):
        raise ValueError(f"unsupported bit width {bits}")
    if bits == 8:
        return q.contiguous()
    per = 8 // bits
    n = q.shape[-1]
    if n % per:
        raise ValueError(f"{n} codes do not fill whole bytes at {bits} bits")
    g = q.reshape(*q.shape[:-1], n // per, per).to(torch.int32)
    shifts = torch.arange(per, device=q.device, dtype=torch.int32) * bits
    return (g << shifts).sum(-1).to(torch.uint8)


def unpack_bits(p: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of pack_bits."""
    if bits == 8:
        return p
    per = 8 // bits
    shifts = torch.arange(per, device=p.device, dtype=torch.int32) * bits
    v = (p.to(torch.int32).unsqueeze(-1) >> shifts) & (2 ** bits - 1)
    return v.reshape(*p.shape[:-1], p.shape[-1] * per).to(torch.uint8)


@dataclass
class PackedTensor:
    """One K or V tensor (B, H, L, d) in a given layout: complete groups packed, tail FP16."""
    layout: str               # 'token' (groups along d) or 'channel' (groups along tokens)
    bits: int
    shape: Tuple[int, int, int, int]
    n_quant: int              # tokens covered by complete groups
    payload: torch.Tensor     # uint8
    scale: torch.Tensor       # fp16
    offset: torch.Tensor      # fp16
    tail: torch.Tensor        # fp16 (B, H, L - n_quant, d)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.payload, self.scale, self.offset, self.tail))


def pack_tensor(X: torch.Tensor, layout: str, bits: int, group_size: int = GROUP_SIZE) -> PackedTensor:
    B, H, L, d = X.shape
    Lq, _ = channelwise_split_lengths(L, group_size)
    body = X[:, :, :Lq, :]
    if layout == "token":
        if d % group_size:
            raise ValueError(f"head_dim {d} not a multiple of {group_size}")
        q, s, o = quantize_groups(body, bits, group_size)                       # groups along d
    elif layout == "channel":
        G = Lq // group_size
        xb = body.reshape(B, H, G, group_size, d).transpose(-1, -2)            # (B,H,G,d,64)
        q, s, o = quantize_groups(xb, bits, group_size)
    else:
        raise ValueError(layout)
    return PackedTensor(layout, bits, (B, H, L, d), Lq, pack_bits(q, bits), s, o,
                        X[:, :, Lq:, :].to(torch.float16).contiguous())


def unpack_tensor(p: PackedTensor, dtype=torch.float16, group_size: int = GROUP_SIZE) -> torch.Tensor:
    B, H, L, d = p.shape
    q = unpack_bits(p.payload, p.bits)
    x = dequantize_groups(q, p.scale, p.offset, group_size)
    if p.layout == "channel":
        G = p.n_quant // group_size
        x = x.transpose(-1, -2).reshape(B, H, G * group_size, d)
    return torch.cat([x.to(dtype), p.tail.to(dtype)], dim=2)


LAYOUTS = {"tokenwise": ("token", "token"), "channelwise": ("channel", "token")}


@dataclass
class PackedLayer:
    K: PackedTensor
    V: PackedTensor

    @property
    def nbytes(self) -> int:
        return self.K.nbytes + self.V.nbytes


def encode_cache(cache: DynamicCache, layout: str, bits: int) -> List[PackedLayer]:
    """Pack every layer. layout 'tokenwise' (K and V token-wise) or 'channelwise'
    (channel-wise K, token-wise V)."""
    kl, vl = LAYOUTS[layout]
    return [PackedLayer(pack_tensor(_layer_keys(cache, i), kl, bits),
                        pack_tensor(_layer_values(cache, i), vl, bits))
            for i in range(_num_layers(cache))]


def decode_cache(packed: Sequence[PackedLayer], dtype=torch.float16) -> DynamicCache:
    return cache_from_tensors([(unpack_tensor(p.K, dtype), unpack_tensor(p.V, dtype)) for p in packed])


def packed_nbytes(packed: Sequence) -> int:
    return int(sum(p.nbytes for p in packed))


def fake_quantize_(X: torch.Tensor, start: int, end: int, layout: str, bits: int):
    """In place: replace tokens [start, end) of X (B,H,L,d) by their quantise->dequantise
    values (start, end multiples of 64)."""
    if end <= start:
        return
    slab = X[:, :, start:end, :]
    p = pack_tensor(slab, layout, bits)
    assert p.n_quant == end - start
    X[:, :, start:end, :] = unpack_tensor(p, X.dtype)


def requantize_completed_groups(cache: DynamicCache, layout: str, bits: int, done_upto: int) -> int:
    """Decode-time contract (C3): quantise any 64-token group completed since `done_upto`
    so newly generated tokens stay FP16 only until their group is complete.
    Returns the new quantised prefix length."""
    L = _seq_len(cache)
    upto, _ = channelwise_split_lengths(L)
    if upto > done_upto:
        kl, vl = LAYOUTS[layout]
        for i in range(_num_layers(cache)):
            fake_quantize_(_layer_keys(cache, i), done_upto, upto, kl, bits)
            fake_quantize_(_layer_values(cache, i), done_upto, upto, vl, bits)
    return max(upto, done_upto)


# ── byte budgets and position selection ───────────────────────────────────────

def int_bytes(L: int, n_layers: int, n_kv: int, d_h: int, bits: int) -> int:
    """M_b(L) (handout formula; channel-wise or token-wise with FP16 tail, d_h % 64 == 0)."""
    return kv_cache_bytes_int(L, n_layers, n_kv, d_h, bits=bits, group_size=GROUP_SIZE)


def fp16_position_bytes(n_layers: int, n_kv: int, d_h: int) -> int:
    return 4 * n_layers * n_kv * d_h


def eviction_bytes(n: int, n_layers: int, n_kv: int, d_h: int) -> int:
    return n * (fp16_position_bytes(n_layers, n_kv, d_h) + INDEX_BYTES)


def max_eviction_positions(budget: int, L: int, n_layers: int, n_kv: int, d_h: int) -> int:
    """Largest number of FP16 positions (payload + index) within the budget."""
    return min(L, budget // (fp16_position_bytes(n_layers, n_kv, d_h) + INDEX_BYTES))


def safeguard_positions(L: int, sink: int = SINK_TOKENS, recent: int = RECENT_TOKENS) -> List[int]:
    return sorted(set(range(min(sink, L))) | set(range(max(0, L - recent), L)))


def select_sink_recent(L: int, n: int, sink: int = SINK_TOKENS, recent: int = RECENT_TOKENS) -> List[int]:
    """8 sink + 64 recent safeguards, remaining budget filled with the most recent positions."""
    safe = safeguard_positions(L, sink, recent)
    if n < len(safe):
        raise ValueError(f"budget of {n} positions is below the {len(safe)} safeguard positions")
    n = min(n, L)
    return sorted(set(range(min(sink, L))) | set(range(L - (n - min(sink, L)), L)))


def rank_by_importance(importance: torch.Tensor, exclude: Sequence[int]) -> List[int]:
    """Positions sorted by descending importance (ties -> earlier position), excluding `exclude`."""
    imp = importance.detach().float().cpu()
    ex = torch.zeros(imp.numel(), dtype=torch.bool)
    if len(exclude):
        ex[torch.as_tensor(list(exclude))] = True
    idx = torch.arange(imp.numel())
    order = sorted(idx[~ex].tolist(), key=lambda i: (-float(imp[i]), i))
    return order


def select_question_aware(L: int, n: int, importance: torch.Tensor,
                          sink: int = SINK_TOKENS, recent: int = RECENT_TOKENS) -> List[int]:
    safe = safeguard_positions(L, sink, recent)
    if n < len(safe):
        raise ValueError(f"budget of {n} positions is below the {len(safe)} safeguard positions")
    fill = rank_by_importance(importance[:L], safe)[:max(0, min(n, L) - len(safe))]
    return sorted(set(safe) | set(fill))


def hybrid_bytes(L: int, n_fp16: int, n_layers: int, n_kv: int, d_h: int, bits_low: int = 2) -> dict:
    """FP16 payload + FP16 indices + low-bit part (complete groups + metadata + FP16 residual)."""
    n_low = L - n_fp16
    fp16 = n_fp16 * fp16_position_bytes(n_layers, n_kv, d_h)
    idx = n_fp16 * INDEX_BYTES
    low = int_bytes(n_low, n_layers, n_kv, d_h, bits_low)
    groups, resid = channelwise_split_lengths(n_low)
    return {"total": fp16 + idx + low, "fp16_payload": fp16, "indices": idx, "low_bit_incl_residual": low,
            "n_fp16": n_fp16, "n_low": n_low, "n_low_quantised": groups, "n_low_fp16_residual": resid}


def select_hybrid(L: int, importance: torch.Tensor, budget: int, n_layers: int, n_kv: int, d_h: int,
                  bits_low: int = 2, sink: int = SINK_TOKENS, recent: int = RECENT_TOKENS
                  ) -> Tuple[List[int], dict]:
    """Largest number of FP16 positions (safeguards + top question-scored) whose total
    actual bytes stay within `budget`.  Byte cost is NOT monotone in n_fp16 (removing a
    low-bit token can break a complete group), so every n is checked."""
    safe = safeguard_positions(L, sink, recent)
    ranked = rank_by_importance(importance[:L], safe)
    best = None
    for n_f in range(L, len(safe) - 1, -1):
        b = hybrid_bytes(L, n_f, n_layers, n_kv, d_h, bits_low)
        if b["total"] <= budget:
            best = n_f
            break
    if best is None:
        raise ValueError("hybrid safeguards alone exceed the budget")
    fp16 = sorted(set(safe) | set(ranked[:best - len(safe)]))
    return fp16, hybrid_bytes(L, best, n_layers, n_kv, d_h, bits_low)


@dataclass
class PackedHybrid:
    fp16_idx: torch.Tensor            # int16 (n_fp16,)  (uint16 semantics, L < 32768)
    fp16: List[Tuple[torch.Tensor, torch.Tensor]]
    low: List[PackedLayer]            # channel-wise K / token-wise V low-bit over low positions
    L: int

    @property
    def nbytes(self) -> int:
        b = self.fp16_idx.numel() * 2
        b += sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in self.fp16)
        return int(b + packed_nbytes(self.low))


def encode_hybrid(cache: DynamicCache, fp16_positions: Sequence[int], bits_low: int = 2) -> PackedHybrid:
    L = _seq_len(cache)
    dev = _layer_keys(cache, 0).device
    f = torch.as_tensor(sorted(fp16_positions), dtype=torch.long, device=dev)
    mask = torch.ones(L, dtype=torch.bool, device=dev)
    mask[f] = False
    low_idx = mask.nonzero().flatten()
    fp16, low = [], []
    for i in range(_num_layers(cache)):
        K, V = _layer_keys(cache, i), _layer_values(cache, i)
        fp16.append((K.index_select(2, f).to(torch.float16).contiguous(),
                     V.index_select(2, f).to(torch.float16).contiguous()))
        low.append(PackedLayer(pack_tensor(K.index_select(2, low_idx), "channel", bits_low),
                               pack_tensor(V.index_select(2, low_idx), "token", bits_low)))
    return PackedHybrid(f.to(torch.int16), fp16, low, L)


def decode_hybrid(p: PackedHybrid, dtype=torch.float16) -> DynamicCache:
    f = p.fp16_idx.long()
    dev = f.device
    mask = torch.ones(p.L, dtype=torch.bool, device=dev)
    mask[f] = False
    low_idx = mask.nonzero().flatten()
    layers = []
    for (Kf, Vf), pl in zip(p.fp16, p.low):
        B, H, _, d = Kf.shape
        K = torch.empty(B, H, p.L, d, dtype=dtype, device=dev)
        V = torch.empty_like(K)
        K[:, :, f] = Kf.to(dtype)
        V[:, :, f] = Vf.to(dtype)
        K[:, :, low_idx] = unpack_tensor(pl.K, dtype)
        V[:, :, low_idx] = unpack_tensor(pl.V, dtype)
        layers.append((K, V))
    return cache_from_tensors(layers)


def reconstruction_error(ref: DynamicCache, approx: DynamicCache) -> dict:
    """Relative Frobenius error for K and V separately over all layers (same length caches)."""
    nk = dk = nv = dv = 0.0
    for i in range(_num_layers(ref)):
        K, V = _layer_keys(ref, i).float(), _layer_values(ref, i).float()
        Ka, Va = _layer_keys(approx, i).float(), _layer_values(approx, i).float()
        nk += float((K - Ka).pow(2).sum()); dk += float(K.pow(2).sum())
        nv += float((V - Va).pow(2).sum()); dv += float(V.pow(2).sum())
    return {"K_rel_fro": math.sqrt(nk / dk) if dk else 0.0, "V_rel_fro": math.sqrt(nv / dv) if dv else 0.0}


# ── C1: position-correct cache editing and decoding ───────────────────────────

def forward_with_positions(model, cache: DynamicCache, input_ids: torch.Tensor, abs_start_pos: int,
                           device=None, logits_to_keep: int = 0):
    """Append `input_ids` to `cache`: position_ids (RoPE) continue from `abs_start_pos`
    (the ORIGINAL absolute position), cache_position continues from the physical cache
    length.  With no eviction these coincide and this is ordinary decoding.
    `logits_to_keep=k` computes only the last k logits (0 = all)."""
    device = device or next(model.parameters()).device
    T = input_ids.shape[1]
    phys = _seq_len(cache)
    position_ids = torch.arange(abs_start_pos, abs_start_pos + T, device=device).unsqueeze(0)
    position_ids = position_ids.expand(input_ids.shape[0], -1)
    cache_position = torch.arange(phys, phys + T, device=device)
    return model(input_ids=input_ids.to(device), past_key_values=cache, use_cache=True,
                 position_ids=position_ids, cache_position=cache_position,
                 logits_to_keep=logits_to_keep)


@torch.no_grad()
def greedy_generate_edited(model, cache: DynamicCache, priming_ids: torch.Tensor, original_context_len: int,
                           max_new_tokens: int, eos_ids: Sequence[int] = (), device=None,
                           after_step=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy decoding on an edited cache with correct absolute positions.

    Feeds `priming_ids` (first token at absolute position original_context_len), then
    decodes.  `after_step(cache)` is called after every forward (used to quantise
    completed groups).  Returns (generated ids (1, n), first-step logits (1, V) fp32)."""
    device = device or next(model.parameters()).device
    out = forward_with_positions(model, cache, priming_ids, original_context_len, device, logits_to_keep=1)
    if after_step:
        after_step(cache)
    next_abs = original_context_len + priming_ids.shape[1]
    logits = out.logits[:, -1, :].float()
    first = logits.clone()
    gen = []
    eos = set(int(e) for e in eos_ids)
    for _ in range(max_new_tokens):
        nxt = logits.argmax(-1, keepdim=True)
        gen.append(nxt)
        if int(nxt.item()) in eos:
            break
        out = forward_with_positions(model, cache, nxt, next_abs, device, logits_to_keep=1)
        if after_step:
            after_step(cache)
        next_abs += 1
        logits = out.logits[:, -1, :].float()
    ids = torch.cat(gen, dim=1) if gen else priming_ids[:, :0]
    return ids, first


def _evict_to_positions(cache: DynamicCache, retain_pos: Sequence[int]) -> DynamicCache:
    """New cache with only `retain_pos` (in the given order; pass ascending positions).
    Retained keys are not re-rotated."""
    new_cache = DynamicCache()
    for li in range(_num_layers(cache)):
        K, V = _layer_keys(cache, li), _layer_values(cache, li)
        idx = torch.as_tensor(list(retain_pos), dtype=torch.long, device=K.device)
        new_cache.update(K.index_select(2, idx), V.index_select(2, idx), li)
    return new_cache


# ── C2/C4: question-aware importance (SDPA model kept) ────────────────────────

@torch.no_grad()
def question_importance(model, cache: DynamicCache, suffix_ids: torch.Tensor, context_len: int,
                        validate: bool = False) -> Tuple[torch.Tensor, dict]:
    """Importance of each context position from the question suffix's attention.

    The suffix is run through the WORKING SDPA model on top of the full FP16 context
    cache (so every layer's queries are the real ones), `q_proj` outputs are captured
    with forward hooks, RoPE is applied at the true absolute positions
    [context_len, context_len + Q), and masked FP32 softmax(q K^T / sqrt(d)) is computed
    against the rotary-positioned cached keys (context keys + causal suffix keys).
    Importance = attention mass on each CONTEXT position, summed over layers, query heads
    and suffix tokens (global retention).  The cache is cropped back to `context_len`
    afterwards, so it is unchanged.  Gold evidence offsets / the answer are never read.

    validate=True also captures the `o_proj` input (the SDPA attention output) and
    reports the max |manual attention @ V - SDPA output| per layer.
    """
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    base = getattr(model, "model", model)
    cfg = model.config
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    d_h = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_h
    n_rep = n_h // n_kv
    assert _seq_len(cache) == context_len, (_seq_len(cache), context_len)
    dev = suffix_ids.device

    q_cap, o_cap, handles = {}, {}, []
    for li, layer in enumerate(base.layers):
        handles.append(layer.self_attn.q_proj.register_forward_hook(
            lambda m, i, o, li=li: q_cap.__setitem__(li, o.detach())))
        if validate:
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda m, i, li=li: o_cap.__setitem__(li, i[0].detach())))
    try:
        forward_with_positions(model, cache, suffix_ids, context_len, dev, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()

    Q = suffix_ids.shape[1]
    pos = torch.arange(context_len, context_len + Q, device=dev).unsqueeze(0)
    cos, sin = base.rotary_emb(q_cap[0].float(), pos)
    total = context_len + Q
    allowed = torch.ones(Q, total, dtype=torch.bool, device=dev)
    allowed[:, context_len:] = torch.ones(Q, Q, dtype=torch.bool, device=dev).tril()
    importance = torch.zeros(context_len, dtype=torch.float32, device=dev)
    max_err = []
    scale = 1.0 / math.sqrt(d_h)
    for li in range(len(base.layers)):
        q = q_cap[li].float()
        B = q.shape[0]
        q = q.view(B, Q, n_h, d_h).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos.float(), sin.float())
        K = _repeat_kv(_layer_keys(cache, li).float(), n_rep)              # (B, n_h, total, d)
        s = torch.matmul(q, K.transpose(-1, -2)) * scale
        s = s.masked_fill(~allowed, float("-inf"))
        a = torch.softmax(s, dim=-1)
        importance += a[..., :context_len].sum(dim=(0, 1, 2))
        if validate:
            V = _repeat_kv(_layer_values(cache, li).float(), n_rep)
            o = torch.matmul(a, V).transpose(1, 2).reshape(B, Q, n_h * d_h)
            max_err.append(float((o - o_cap[li].float()).abs().max()))
    cache.crop(context_len)
    assert _seq_len(cache) == context_len
    assert torch.isfinite(importance).all() and importance.sum() > 0
    info = {"n_suffix_tokens": Q, "retention_scope": "global (summed over layers, heads, question tokens)"}
    if validate:
        info["manual_vs_sdpa_attention_max_abs_err_per_layer"] = max_err
    return importance, info
