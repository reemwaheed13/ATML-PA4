"""
Storage accounting for compressed model weights and KV caches.

All formulas are taken verbatim from the PDF.  Results are in BYTES unless
the function name says otherwise.

Key rule (PDF "What to Watch For"):
  "Zero-valued weights and fake-quantized FP16 weights still use dense operations
   unless an explicitly measured sparse/low-bit kernel is used.  Do not infer
   FLOP, VRAM, or latency reductions from the number of zeros or nominal bits alone."

Label theoretical estimates clearly as such; never equate them with measured
allocated GPU memory.
"""

import math


# ── Weight storage ─────────────────────────────────────────────────────────────

def fp16_weight_bytes(n_params: int) -> int:
    """Dense FP16 storage: 2 bytes per parameter."""
    return n_params * 2


def sparse_fp16_bytes(n_eligible: int, nonzero_fraction: float) -> int:
    """
    Theoretical 16-bit sparse format used in PDF "Shared resources":
      1 mask bit per eligible weight + 16 bits per nonzero.
    Result is byte-aligned (ceil to nearest byte) for the mask.

    This is a *representation estimate*, not an allocated GPU tensor.
    """
    n_nonzero = round(n_eligible * nonzero_fraction)
    mask_bytes = math.ceil(n_eligible / 8)
    value_bytes = n_nonzero * 2
    return mask_bytes + value_bytes


def quant_packed_bytes(
    n_eligible: int,
    bits: int,
    group_size: int = 128,
    n_metadata_bytes_per_group: int = 4,  # FP16 scale (2 B) + FP16 zero-point (2 B)
) -> int:
    """
    Packed storage for group-wise weight-only quantisation.

    n_eligible              total number of eligible (compressed) scalar weights
    bits                    quantisation bit-width {2, 3, 4, 8}
    group_size              number of weights per quantisation group
    n_metadata_bytes_per_group  2 (scale FP16) + 2 (zero-point FP16) = 4 B

    Weight payload is ceil(n_eligible * bits / 8) bytes.
    Metadata is ceil(n_eligible / group_size) groups × metadata bytes/group.
    Total does NOT include uncompressed parameters (embeddings, norms, LM head).
    """
    payload_bytes = math.ceil(n_eligible * bits / 8)
    n_groups = math.ceil(n_eligible / group_size)
    metadata_bytes = n_groups * n_metadata_bytes_per_group
    return payload_bytes + metadata_bytes


def model_total_bytes(
    n_eligible: int,
    n_uncompressed: int,
    bits: int,
    group_size: int = 128,
) -> int:
    """
    Total model storage: compressed eligible weights + FP16 uncompressed
    (embeddings, norms, biases, LM head — counting tied params once).
    """
    return quant_packed_bytes(n_eligible, bits, group_size) + fp16_weight_bytes(n_uncompressed)


# ── KV cache storage ───────────────────────────────────────────────────────────

def kv_cache_bytes_fp16(L: int, n_layers: int, n_kv_heads: int, head_dim: int) -> int:
    """
    Full FP16 KV cache:  M_16(L) = 4 * n_l * H_kv * d_h * L  bytes.
    (Factor 4 = 2 for K+V × 2 bytes per FP16 element.)
    """
    return 4 * n_layers * n_kv_heads * head_dim * L


def kv_cache_bytes_int(
    L: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    bits: int = 4,
    group_size: int = 64,
    metadata_bytes_per_group: int = 4,  # FP16 scale + FP16 offset
) -> int:
    """
    Channel-wise-K / token-wise-V INT quantisation with FP16 trailing tokens.

    From PDF:
      M_b(L) = 2*n_l*H_kv*d_h * [ floor(L/64)*(64*b/8 + 4) + 2*(L mod 64) ]

    For each of the 2*n_l*H_kv*d_h stored KV channels:
      - Complete groups of 64 tokens: packed payload + 4 bytes metadata.
      - Trailing r = L mod 64 tokens: 2 bytes each (FP16).
    """
    n_channels = 2 * n_layers * n_kv_heads * head_dim
    full_groups = L // group_size
    tail_tokens = L % group_size
    per_channel = full_groups * (group_size * bits // 8 + metadata_bytes_per_group) + 2 * tail_tokens
    return n_channels * per_channel


def kv_eviction_bytes(
    n_retained: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
) -> int:
    """FP16 storage for n_retained cache positions (after eviction)."""
    return 4 * n_layers * n_kv_heads * head_dim * n_retained


def max_fp16_positions_within_budget(
    budget_bytes: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
) -> int:
    """
    Largest number of FP16 KV positions whose actual bytes ≤ budget_bytes.
    Used for eviction policies: 'retain the largest legal number of FP16 positions
    whose actual bytes do not exceed the reference INT4 budget.'
    """
    bytes_per_position = 4 * n_layers * n_kv_heads * head_dim
    return budget_bytes // bytes_per_position


# ── Diagnostic helpers ─────────────────────────────────────────────────────────

def summarise_compression(
    n_eligible: int,
    n_uncompressed: int,
    bits: int,
    group_size: int = 128,
    label: str = "",
) -> dict:
    fp16_eligible = fp16_weight_bytes(n_eligible)
    fp16_total = fp16_weight_bytes(n_eligible + n_uncompressed)
    quant_total = model_total_bytes(n_eligible, n_uncompressed, bits, group_size)
    return {
        "label": label,
        "bits": bits,
        "group_size": group_size,
        "n_eligible": n_eligible,
        "n_uncompressed": n_uncompressed,
        "fp16_eligible_bytes": fp16_eligible,
        "fp16_total_bytes": fp16_total,
        "quant_total_bytes": quant_total,
        "compression_ratio": fp16_total / quant_total if quant_total else float("inf"),
        "nominal_bits_per_eligible": bits,
        "effective_bits_per_eligible": (quant_packed_bytes(n_eligible, bits, group_size) * 8) / n_eligible,
    }
