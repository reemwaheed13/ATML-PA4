"""
MQAR Transformer model (PDF §2 Part B).

Fixed architecture (identical for every attention operator; only `attn` changes):
  - 2 pre-normalized Transformer blocks (LayerNorm -> attn -> residual, LayerNorm -> MLP -> residual)
  - hidden_dim = 128, n_heads = 4, head_dim = 32, 4x MLP (512, GELU)
  - learned token embeddings (vocab 256) + fixed sinusoidal positional embeddings
  - classification head: hidden -> 64 logits (value IDs 128..191) read at the K marked
    query positions (query-position classification, NOT next-token LM)

Attention operators (`attn_type`):
  'dense'         causal softmax attention (PyTorch SDPA, is_causal=True)
  'window'        causal sliding window, query i attends j in [i-W+1, i], W=64
  'linear'        causal ELU+1 linear attention (Katharopoulos et al. 2020)
  'dilated'       heads 0,1: preceding 64 positions; heads 2,3: 64 positions at offsets
                  {0,4,...,252} (j = i - offset, j >= 0)
  'local_linear'  B3 intervention: heads 0,1 local softmax W=33 over an FP16 K/V buffer,
                  heads 2,3 ELU+1 linear attention with an FP32 state.  Same projections,
                  so the trainable parameter count equals every other operator's.  W=33
                  is the unique window that makes the persistent state bytes per layer
                  equal to the 'linear' comparator (see `persistent_state_bytes`).

Implementation choices that the PDF asks to be stated explicitly (B2, B-R3):
  * 'window', 'dilated' and the local heads of 'local_linear' use an explicit dense
    boolean (T, T) / (H, T, T) mask passed to SDPA.  This is a CORRECTNESS
    implementation: it computes and stores dense scores, it is not an efficient sparse
    kernel, and its measured memory must not be read as sparse-attention memory.
  * 'linear' trains with the exact PARALLEL form  A = tril(phi(Q) phi(K)^T),
    out = A V / (A 1 + eps), which materialises a (T, T) matrix per head.  The
    recurrent O(d^2)-state form (`linear_mode='recurrent'`) is mathematically identical
    (unit-tested) and is used in the forward-only sweep to show streaming cost.

Test-time controls (B1 isolation control, RQ2): `set_eval_override(('window', 64))`
imposes a W=64 mask on a dense-trained model; `('topk', 64)` keeps only the 64
highest-scoring causal keys per query (content-selected, post-hoc).  Overrides apply to
'dense' blocks only and never during training.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB_SIZE = 256
HIDDEN_DIM = 128
N_HEADS = 4
HEAD_DIM = 32
MLP_MULT = 4
N_LAYERS = 2
MAX_SEQ_LEN = 256
VALUE_CLASSES = 64      # value IDs 128..191
VALUE_OFFSET = 128

WINDOW = 64
DILATION = 4
N_DILATED = 64          # offsets {0, 4, ..., 252}
B3_WINDOW = 33          # state-byte-matched local window of the B3 intervention
LINEAR_EPS = 1e-6

ATTN_TYPES = ("dense", "window", "linear", "dilated", "local_linear")


def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding, shape (1, max_len, d_model)."""
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


# ── Boolean masks (True = may attend) ─────────────────────────────────────────

def causal_mask(T: int, device=None) -> torch.Tensor:
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def window_mask(T: int, W: int, device=None) -> torch.Tensor:
    """Query i attends keys j with i-W+1 <= j <= i."""
    i = torch.arange(T, device=device).unsqueeze(1)
    j = torch.arange(T, device=device).unsqueeze(0)
    return (j <= i) & (j >= i - W + 1)


def dilated_offset_mask(T: int, dilation: int = DILATION, n_offsets: int = N_DILATED,
                        device=None) -> torch.Tensor:
    """Query i attends j = i - o for o in {0, d, 2d, ..., (n-1)d}, j >= 0."""
    i = torch.arange(T, device=device).unsqueeze(1)
    j = torch.arange(T, device=device).unsqueeze(0)
    dist = i - j
    return (dist >= 0) & (dist % dilation == 0) & (dist <= dilation * (n_offsets - 1))


def mixed_dilated_mask(T: int, device=None) -> torch.Tensor:
    """(4, T, T): heads 0,1 local W=64, heads 2,3 dilated offsets {0,4,...,252}."""
    loc = window_mask(T, WINDOW, device)
    dil = dilated_offset_mask(T, device=device)
    return torch.stack([loc, loc, dil, dil], dim=0)


def topk_causal_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Keep, per query row, the k largest causal scores. scores (..., T, T)."""
    T = scores.shape[-1]
    c = causal_mask(T, scores.device)
    s = scores.masked_fill(~c, float("-inf"))
    kk = min(k, T)
    thresh = s.topk(kk, dim=-1).values[..., -1:]            # k-th largest per row
    return (s >= thresh) & c


# ── Linear attention kernels ──────────────────────────────────────────────────

def elu1(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


def linear_attention_parallel(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                              eps: float = LINEAR_EPS) -> torch.Tensor:
    """Exact causal ELU+1 linear attention, parallel form. Q,K,V (B,H,T,d).
    Materialises the (T, T) kernel matrix per head (stated in the module docstring)."""
    q, k = elu1(Q), elu1(K)
    A = torch.matmul(q, k.transpose(-2, -1))                  # (B,H,T,T)
    A = A * causal_mask(Q.shape[-2], Q.device).to(A.dtype)
    num = torch.matmul(A, V)
    den = A.sum(dim=-1, keepdim=True) + eps
    return num / den


def linear_attention_recurrent(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                               eps: float = LINEAR_EPS) -> torch.Tensor:
    """Same computation as a recurrence with an O(d^2) state:
    S_t = S_{t-1} + phi(k_t) v_t^T, z_t = z_{t-1} + phi(k_t), a_t = phi(q_t)^T S_t / (phi(q_t)^T z_t + eps)."""
    q, k = elu1(Q), elu1(K)
    B, H, T, d = Q.shape
    S = Q.new_zeros(B, H, d, V.shape[-1])
    z = Q.new_zeros(B, H, d)
    outs = []
    for t in range(T):
        kt, vt, qt = k[:, :, t], V[:, :, t], q[:, :, t]
        S = S + kt.unsqueeze(-1) * vt.unsqueeze(-2)
        z = z + kt
        num = torch.einsum("bhd,bhde->bhe", qt, S)
        den = (qt * z).sum(-1, keepdim=True) + eps
        outs.append(num / den)
    return torch.stack(outs, dim=2)


# ── Attention module ──────────────────────────────────────────────────────────

class MQARAttention(nn.Module):
    """Shared q/k/v/o projections; `attn_type` selects only the operator."""

    def __init__(self, hidden_dim: int, n_heads: int, attn_type: str,
                 window: int = WINDOW, linear_mode: str = "parallel"):
        super().__init__()
        if attn_type not in ATTN_TYPES:
            raise ValueError(f"unknown attn_type {attn_type!r}; known {ATTN_TYPES}")
        if attn_type in ("dilated", "local_linear") and n_heads != 4:
            raise ValueError(f"{attn_type} requires exactly 4 heads")
        self.attn_type = attn_type
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.window = window
        self.linear_mode = linear_mode
        self.eval_override: Optional[Tuple[str, int]] = None
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _linear(self, Q, K, V):
        if self.linear_mode == "recurrent":
            return linear_attention_recurrent(Q, K, V)
        return linear_attention_parallel(Q, K, V)

    def _explicit_masked(self, Q, K, V, mask):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~mask, float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), V)

    def attend(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        T = Q.shape[-2]
        dev = Q.device
        t = self.attn_type
        if t == "dense":
            ov = None if self.training else self.eval_override
            if ov is None:
                return F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=self.scale)
            kind, n = ov
            if kind == "window":
                return self._explicit_masked(Q, K, V, window_mask(T, n, dev))
            if kind == "topk":
                scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
                return self._explicit_masked(Q, K, V, topk_causal_mask(scores, n))
            raise ValueError(f"unknown eval override {ov}")
        if t == "window":
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=window_mask(T, self.window, dev),
                                                  scale=self.scale)
        if t == "dilated":
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=mixed_dilated_mask(T, dev).unsqueeze(0),
                                                  scale=self.scale)
        if t == "linear":
            return self._linear(Q, K, V)
        # local_linear (B3): heads 0,1 local softmax over an FP16 K/V buffer, heads 2,3 linear.
        Ql, Kl, Vl = Q[:, :2], K[:, :2], V[:, :2]
        Kl = Kl.to(torch.float16).to(Q.dtype)     # persistent local buffer is stored in FP16
        Vl = Vl.to(torch.float16).to(Q.dtype)
        loc = F.scaled_dot_product_attention(Ql, Kl, Vl, attn_mask=window_mask(T, B3_WINDOW, dev),
                                             scale=self.scale)
        lin = self._linear(Q[:, 2:], K[:, 2:], V[:, 2:])
        return torch.cat([loc, lin], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.n_heads, self.head_dim

        def split(lin):
            return lin(x).reshape(B, T, H, d).transpose(1, 2)

        out = self.attend(split(self.q), split(self.k), split(self.v))
        return self.o(out.transpose(1, 2).reshape(B, T, D))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, mlp_mult: int, attn_type: str,
                 window: int = WINDOW, linear_mode: str = "parallel"):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.attn = MQARAttention(hidden_dim, n_heads, attn_type, window, linear_mode)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_mult, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_mult, hidden_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MQARTransformer(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden_dim: int = HIDDEN_DIM,
                 n_heads: int = N_HEADS, n_layers: int = N_LAYERS, mlp_mult: int = MLP_MULT,
                 max_len: int = MAX_SEQ_LEN, attn_type: str = "dense", window: int = WINDOW,
                 linear_mode: str = "parallel"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_type = attn_type
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.register_buffer("pos_enc", sinusoidal_pe(max_len, hidden_dim), persistent=False)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, mlp_mult, attn_type, window, linear_mode)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, VALUE_CLASSES, bias=True)

    def set_eval_override(self, override: Optional[Tuple[str, int]]):
        """Test-time-only mask control for dense blocks (None restores ordinary attention)."""
        for b in self.blocks:
            b.attn.eval_override = override

    def set_linear_mode(self, mode: str):
        for b in self.blocks:
            b.attn.linear_mode = mode

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Run the blocks on input embeddings h (B, T, hidden)."""
        for block in self.blocks:
            h = block(h)
        return self.ln_out(h)

    def forward(self, token_ids: torch.Tensor,
                query_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, K, 64) logits at query_positions, or (B, T, 64) if None."""
        T = token_ids.shape[1]
        h = self.encode(self.embedding(token_ids) + self.pos_enc[:, :T, :])
        if query_positions is not None:
            idx = query_positions.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            h = h.gather(dim=1, index=idx)
        return self.classifier(h)


def make_mqar_model(attn_type: str, window: int = WINDOW, device: str = "cuda",
                    max_len: int = MAX_SEQ_LEN, linear_mode: str = "parallel") -> MQARTransformer:
    return MQARTransformer(attn_type=attn_type, window=window, max_len=max_len,
                           linear_mode=linear_mode).to(device)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Theoretical costs (PDF B-R1, B2-4, B3-5) ──────────────────────────────────

def accessible_positions_per_query(attn_type: str, T: int = MAX_SEQ_LEN) -> dict:
    """Candidate key/value positions a single query can read in ONE layer, per head,
    averaged over query positions 0..T-1, plus the attended-pair count summed over
    heads and layers.  Linear heads read a fixed-size state (reported as 'state')."""
    i = torch.arange(T)
    dense = (i + 1).float()
    win = torch.clamp(i + 1, max=WINDOW).float()
    dil = torch.clamp(i // DILATION + 1, max=N_DILATED).float()
    b3w = torch.clamp(i + 1, max=B3_WINDOW).float()
    per_head = {
        "dense": [dense] * 4,
        "window": [win] * 4,
        "dilated": [win, win, dil, dil],
        "linear": [None] * 4,
        "local_linear": [b3w, b3w, None, None],
    }[attn_type]
    pairs = sum(float(h.sum()) for h in per_head if h is not None) * N_LAYERS
    mean_per_head = [None if h is None else float(h.mean()) for h in per_head]
    return {
        "T": T,
        "mean_positions_per_query_per_head": mean_per_head,
        "max_positions_per_query_per_head": [None if h is None else float(h.max()) for h in per_head],
        "attended_pairs_total": pairs,
        "linear_state_heads": sum(h is None for h in per_head),
        "two_layer_receptive_field": {
            "dense": T - 1, "window": 2 * (WINDOW - 1), "dilated": T - 1,
            "linear": T - 1, "local_linear": T - 1,
        }[attn_type],
    }


def persistent_state_bytes(attn_type: str, T: int = MAX_SEQ_LEN, act_bytes: int = 4) -> dict:
    """Persistent decode-time state per layer per sequence (bytes).

    dense       K,V cache for all T tokens:           2 * T * H * d * act_bytes
    window      ring buffer of the last W tokens:      2 * W * H * d * act_bytes
    dilated     local heads keep W tokens; dilated heads must keep the last
                d*(n-1)+1 = 253 tokens (every residue class is needed over time)
    linear      per head S (d x d) + z (d), FP32:      H * (d^2 + d) * 4
    local_linear  2 linear heads FP32 state + 2 local heads FP16 K/V buffer of W=33:
                2*(d^2+d)*4 + 2*33*2*d*2  == linear's bytes exactly (16,896 B)
    """
    H, d = N_HEADS, HEAD_DIM
    span_dil = DILATION * (N_DILATED - 1) + 1
    if attn_type == "dense":
        b = 2 * T * H * d * act_bytes
    elif attn_type == "window":
        b = 2 * min(T, WINDOW) * H * d * act_bytes
    elif attn_type == "dilated":
        b = 2 * min(T, WINDOW) * 2 * d * act_bytes + 2 * min(T, span_dil) * 2 * d * act_bytes
    elif attn_type == "linear":
        b = H * (d * d + d) * 4
    elif attn_type == "local_linear":
        b = 2 * (d * d + d) * 4 + 2 * min(T, B3_WINDOW) * 2 * d * 2
    else:
        raise ValueError(attn_type)
    return {"bytes_per_layer": int(b), "bytes_all_layers": int(b * N_LAYERS), "T": T}
