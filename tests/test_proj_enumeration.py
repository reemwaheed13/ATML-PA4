"""
Unit test: eligible-projection enumeration.

Requirement (PDF §1 step 1):
  "Test that eligible matrices and tied weights are enumerated correctly."

Expected:
  - Exactly 7 eligible projection types per Transformer block:
    q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Tied parameters (lm_head.weight == model.embed_tokens.weight in Qwen2.5)
    counted ONCE — lm_head is excluded from the eligible set.
  - Embedding, norm, bias, and lm_head layers are excluded.

This test uses a small synthetic Qwen2.5-style model defined in-process so it
can run on CPU without downloading the full checkpoint.
"""

import unittest
import torch
import torch.nn as nn
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from task1_compression.baseline import enumerate_eligible_projections, ELIGIBLE_SUFFIXES, count_eligible_params


# ── Synthetic Qwen2.5-style model ─────────────────────────────────────────────

class FakeMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate_proj = nn.Linear(d, d * 2, bias=False)
        self.up_proj   = nn.Linear(d, d * 2, bias=False)
        self.down_proj = nn.Linear(d * 2, d, bias=False)


class FakeAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)


class FakeLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = FakeAttention(d)
        self.mlp = FakeMLP(d)
        self.input_layernorm = nn.LayerNorm(d)
        self.post_attention_layernorm = nn.LayerNorm(d)


class FakeQwenModel(nn.Module):
    def __init__(self, n_layers=4, vocab=512, d=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([FakeLayer(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return x


class FakeQwen(nn.Module):
    """Mimics the top-level structure of Qwen2.5."""
    def __init__(self, n_layers=4, vocab=512, d=64):
        super().__init__()
        self.model = FakeQwenModel(n_layers, vocab, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        # Tie weights exactly as Qwen2.5 does.
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, x):
        return x


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestEligibleProjectionEnumeration(unittest.TestCase):
    def setUp(self):
        self.n_layers = 4
        self.model = FakeQwen(n_layers=self.n_layers)

    def test_count_per_block(self):
        """Each Transformer block must contribute exactly 7 eligible projections."""
        projs = enumerate_eligible_projections(self.model)
        # Group by layer.
        from collections import defaultdict
        by_layer = defaultdict(list)
        for name, _ in projs:
            # Extract layer index from name like "model.layers.2.self_attn.q_proj"
            parts = name.split(".")
            if "layers" in parts:
                li = parts[parts.index("layers") + 1]
                by_layer[li].append(name)

        for layer_idx, names in by_layer.items():
            self.assertEqual(
                len(names), 7,
                f"Layer {layer_idx} has {len(names)} projections, expected 7: {names}"
            )

    def test_all_suffix_types_present_per_block(self):
        """All 7 eligible suffixes must appear in each block."""
        projs = enumerate_eligible_projections(self.model)
        from collections import defaultdict
        by_layer = defaultdict(set)
        for name, _ in projs:
            parts = name.split(".")
            if "layers" in parts:
                li = parts[parts.index("layers") + 1]
                suffix = name.rsplit(".", 1)[-1]
                by_layer[li].add(suffix)

        expected = set(ELIGIBLE_SUFFIXES)
        for layer_idx, suffixes in by_layer.items():
            self.assertEqual(
                suffixes, expected,
                f"Layer {layer_idx}: got {suffixes}, expected {expected}"
            )

    def test_lm_head_excluded(self):
        """lm_head must NOT appear in eligible projections (tied weight)."""
        projs = enumerate_eligible_projections(self.model)
        names = [n for n, _ in projs]
        self.assertFalse(
            any("lm_head" in n for n in names),
            f"lm_head appeared in eligible projections: {[n for n in names if 'lm_head' in n]}"
        )

    def test_tied_weight_counted_once(self):
        """
        lm_head.weight is tied to embed_tokens.weight.
        The lm_head module must be excluded from eligible projections entirely,
        so the tied weight is counted only once (via embed_tokens, which is also
        excluded — the point is that lm_head is not double-counted).
        """
        projs = enumerate_eligible_projections(self.model)
        # Verify: id of lm_head weight should not appear among eligible modules.
        lm_weight_id = id(self.model.lm_head.weight)
        for _, module in projs:
            self.assertNotEqual(
                id(module.weight), lm_weight_id,
                "lm_head's (tied) weight appeared as an eligible projection weight"
            )

    def test_embedding_excluded(self):
        """embed_tokens must NOT be in eligible projections."""
        projs = enumerate_eligible_projections(self.model)
        names = [n for n, _ in projs]
        self.assertFalse(
            any("embed_tokens" in n for n in names),
            "embed_tokens appeared in eligible projections"
        )

    def test_norm_excluded(self):
        """LayerNorm layers must NOT appear in eligible projections."""
        projs = enumerate_eligible_projections(self.model)
        names = [n for n, _ in projs]
        self.assertFalse(
            any("norm" in n for n in names),
            f"Norm layer appeared: {[n for n in names if 'norm' in n]}"
        )

    def test_total_projection_count(self):
        """Total eligible projections = n_layers × 7."""
        projs = enumerate_eligible_projections(self.model)
        self.assertEqual(len(projs), self.n_layers * 7)

    def test_count_eligible_params(self):
        """count_eligible_params returns a dict with 'total' key and per-suffix counts."""
        counts = count_eligible_params(self.model)
        self.assertIn("total", counts)
        for suffix in ELIGIBLE_SUFFIXES:
            self.assertIn(suffix, counts)
            self.assertGreater(counts[suffix], 0)
        # Sum of per-suffix counts equals total.
        self.assertEqual(
            sum(counts[s] for s in ELIGIBLE_SUFFIXES), counts["total"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
