"""
MQAR Transformer model (Part B).

Fixed architecture (PDF §2 Part B):
  - 2 pre-normalized Transformer blocks
  - hidden_dim = 128, n_heads = 4, head_dim = 32
  - 4× MLP width (MLP hidden = 512)
  - Learned token embeddings, 256-token vocabulary
  - Sinusoidal positional embeddings
  - Classification head: projects hidden_dim → 64 at query positions
    (64 possible value IDs: 128..191)

Attention variants:
  'dense'       Standard causal self-attention.
  'window'      Causal sliding-window, W=64 preceding positions.
  'linear'      Causal ELU+1 linear attention (recurrent summaries).
  'dilated'     Mixed: 2 heads use W=64 local, 2 heads use 64 dilated offsets.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


VOCAB_SIZE = 256
HIDDEN_DIM = 128
N_HEADS = 4
HEAD_DIM = 32          # hidden_dim / n_heads
MLP_MULT = 4
N_LAYERS = 2
MAX_SEQ_LEN = 256
VALUE_CLASSES = 64     # value IDs 128..191


def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding, shape (1, max_len, d_model)."""
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, max_len, d_model)


# ── Attention modules ──────────────────────────────────────────────────────────

class DenseCausalAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, d = self.n_heads, self.head_dim

        def proj_split(linear):
            return linear(x).reshape(B, L, H, d).transpose(1, 2)  # (B, H, L, d)

        Q, K, V = proj_split(self.q), proj_split(self.k), proj_split(self.v)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.o(out)


class SlidingWindowAttention(nn.Module):
    """Causal sliding-window attention: attend to at most W preceding positions."""
    def __init__(self, hidden_dim: int, n_heads: int, window: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.window = window
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, d, W = self.n_heads, self.head_dim, self.window

        def proj_split(linear):
            return linear(x).reshape(B, L, H, d).transpose(1, 2)

        Q, K, V = proj_split(self.q), proj_split(self.k), proj_split(self.v)
        # Build causal + window mask: position i attends to j in [i-W+1, i].
        pos = torch.arange(L, device=x.device)
        # Boolean mask: True where attention is FORBIDDEN.
        mask = (pos.unsqueeze(0) < pos.unsqueeze(1) - W + 1) | (pos.unsqueeze(0) > pos.unsqueeze(1))
        # (L, L); upper triangle = future, lower-left = too far back.
        attn_bias = mask.float() * float("-inf")
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale + attn_bias
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, L, D)
        return self.o(out)


class LinearAttention(nn.Module):
    """
    Causal ELU+1 linear attention.
    Feature map φ(x) = ELU(x) + 1 (component-wise positive).
    Recurrent: S_t = S_{t-1} + φ(k_t) v_t^T, z_t = z_{t-1} + φ(k_t).
    At inference: â_t = φ(q_t)^T S_t / (φ(q_t)^T z_t + ε).
    """
    def __init__(self, hidden_dim: int, n_heads: int, eps: float = 1e-6):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.eps = eps
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o = nn.Linear(hidden_dim, hidden_dim, bias=False)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, d = self.n_heads, self.head_dim

        def proj_split(linear):
            return linear(x).reshape(B, L, H, d)  # (B, L, H, d)

        Qf = self._phi(proj_split(self.q))  # (B, L, H, d)
        Kf = self._phi(proj_split(self.k))
        V_ = proj_split(self.v)             # (B, L, H, d)

        # Recurrent computation — parallelisable via cumsum for training.
        S = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)   # (B, H, d, d)
        z = torch.zeros(B, H, d, device=x.device, dtype=x.dtype)       # (B, H, d)
        outputs = []

        for t in range(L):
            k_t = Kf[:, t, :, :]    # (B, H, d)
            v_t = V_[:, t, :, :]    # (B, H, d)
            q_t = Qf[:, t, :, :]

            # State update.
            S = S + torch.einsum("bhd,bhe->bhde", k_t, v_t)
            z = z + k_t

            # Output.
            num = torch.einsum("bhd,bhde->bhe", q_t, S)   # (B, H, d)
            denom = (torch.einsum("bhd,bhd->bh", q_t, z) + self.eps).unsqueeze(-1)
            outputs.append((num / denom).unsqueeze(1))    # (B, 1, H, d)

        out = torch.cat(outputs, dim=1).reshape(B, L, D)  # (B, L, D)
        return self.o(out)


class MixedDilatedAttention(nn.Module):
    """
    Mixed local/dilated sparse attention.
    2 heads: preceding W=64 positions (local causal).
    2 heads: 64 positions at dilated offsets {0, 4, 8, ..., 252}, respecting causality.

    Implemented with explicit boolean masks (a correctness implementation;
    the dense score materialisation cost is noted, not elided).
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4, window: int = 64, dilation: int = 4):
        super().__init__()
        assert n_heads == 4, "MixedDilatedAttention requires exactly 4 heads"
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.window = window
        self.dilation = dilation
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _build_masks(self, L: int, device) -> torch.Tensor:
        """Return (n_heads, L, L) additive masks (-inf for forbidden)."""
        pos = torch.arange(L, device=device)

        # Local heads (0, 1): attend j in [i-W+1, i].
        local_forbidden = (pos.unsqueeze(0) < pos.unsqueeze(1) - self.window + 1) | \
                          (pos.unsqueeze(0) > pos.unsqueeze(1))

        # Dilated heads (2, 3): attend j where (i - j) in {0, 4, 8, ..., 252} and j <= i.
        offsets = torch.arange(0, L, self.dilation, device=device)  # 0, 4, 8, ...
        # For each (i, j): distance = i - j; allowed if distance in offsets and j <= i.
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)  # (L, L): dist[i,j] = i - j
        allowed_offset = torch.isin(dist, offsets) & (dist >= 0)
        dilated_forbidden = ~allowed_offset

        local_mask = local_forbidden.float() * float("-inf")        # (L, L)
        dilated_mask = dilated_forbidden.float() * float("-inf")

        # Stack: heads 0,1 = local; heads 2,3 = dilated.
        masks = torch.stack([local_mask, local_mask, dilated_mask, dilated_mask], dim=0)
        return masks  # (4, L, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, d = self.n_heads, self.head_dim

        def proj_split(linear):
            return linear(x).reshape(B, L, H, d).transpose(1, 2)  # (B, H, L, d)

        Q, K, V = proj_split(self.q), proj_split(self.k), proj_split(self.v)
        masks = self._build_masks(L, x.device).unsqueeze(0)  # (1, H, L, L)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale + masks
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, L, D)
        return self.o(out)


# ── Transformer block (pre-norm) ───────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, mlp_mult: int, attn_type: str, window: int = 64):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        if attn_type == "dense":
            self.attn = DenseCausalAttention(hidden_dim, n_heads)
        elif attn_type == "window":
            self.attn = SlidingWindowAttention(hidden_dim, n_heads, window)
        elif attn_type == "linear":
            self.attn = LinearAttention(hidden_dim, n_heads)
        elif attn_type == "dilated":
            self.attn = MixedDilatedAttention(hidden_dim, n_heads, window)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        mlp_hidden = hidden_dim * mlp_mult
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden, bias=False),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ── Full MQAR model ────────────────────────────────────────────────────────────

class MQARTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        mlp_mult: int = MLP_MULT,
        max_len: int = MAX_SEQ_LEN,
        attn_type: str = "dense",
        window: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_type = attn_type

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.register_buffer("pos_enc", sinusoidal_pe(max_len, hidden_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, mlp_mult, attn_type, window)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(hidden_dim)
        # 64 value classes: IDs 128..191.
        self.classifier = nn.Linear(hidden_dim, VALUE_CLASSES, bias=True)

    def forward(
        self,
        token_ids: torch.Tensor,                  # (B, T)
        query_positions: Optional[torch.Tensor] = None,  # (B, K) int indices
    ) -> torch.Tensor:
        """
        Returns
        -------
        If query_positions is given: (B, K, 64) logits at query positions.
        Otherwise: (B, T, 64) logits at every position.
        """
        B, T = token_ids.shape
        h = self.embedding(token_ids) + self.pos_enc[:, :T, :]
        for block in self.blocks:
            h = block(h)
        h = self.ln_out(h)

        if query_positions is not None:
            # Gather hidden states at query positions.
            idx = query_positions.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            h_q = h.gather(dim=1, index=idx)   # (B, K, hidden_dim)
            return self.classifier(h_q)         # (B, K, 64)

        return self.classifier(h)               # (B, T, 64)


def make_mqar_model(attn_type: str, window: int = 64, device: str = "cuda") -> MQARTransformer:
    model = MQARTransformer(attn_type=attn_type, window=window)
    return model.to(device)
