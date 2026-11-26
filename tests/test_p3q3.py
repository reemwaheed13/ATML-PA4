"""
CPU tests for the P3 (nonuniform Wanda) and Q3 (mixed precision) allocators, on a tiny
random Qwen2 (no downloads).

- P3 allocate() hits the exact global zero count (== P2 at the same S) and keeps every
  block / projection inside [0.10, 0.85]; the applied masks reproduce the counts.
- Q3 allocate() never exceeds the budget, and with an unlimited budget gives all 8-bit.
- The Gram error tr(D G D^T) equals the explicit ||(W' - W) X^T||_F^2.
"""

import os
import random
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from transformers import Qwen2Config, Qwen2ForCausalLM
    HAS_TF = True
except Exception:  # pragma: no cover
    HAS_TF = False


def tiny_qwen2(n_layers=6):
    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=128, intermediate_size=384, num_hidden_layers=n_layers,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
                      tie_word_embeddings=True, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


class TestGramError(unittest.TestCase):
    def test_gram_error_matches_explicit(self):
        from task1_compression.prune_nonuniform import gram_error
        g = torch.Generator().manual_seed(7)
        for d_out, d_in, n in ((16, 32, 100), (40, 24, 7), (8, 128, 300)):
            X = torch.randn(n, d_in, generator=g, dtype=torch.float64)
            W = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
            W2 = W + 0.1 * torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
            G = X.t() @ X
            explicit = ((W2 - W) @ X.t()).pow(2).sum().item()
            self.assertAlmostEqual(gram_error(W2 - W, G) / explicit, 1.0, places=10)


@unittest.skipUnless(HAS_TF, "transformers not installed")
class TestGramErrorOnModel(unittest.TestCase):
    def test_block_sensitivity_matches_raw_activations(self):
        from task1_compression.baseline import collect_calibration_stats, enumerate_eligible_projections
        from task1_compression.prune_nonuniform import block_sensitivity, wanda_scores_for
        from task1_compression.prune_wanda import collect_calibration_inputs
        from task1_compression.baseline import exact_row_prune_mask

        model = tiny_qwen2(2)
        ids = torch.randint(0, 256, (3, 24), generator=torch.Generator().manual_seed(3)).tolist()
        with torch.no_grad():
            stats = collect_calibration_stats(model, ids, "cpu", with_gram=True)
            X = collect_calibration_inputs(model, ids, "cpu")
        sens = block_sensitivity(model, stats, 0.5, "cpu")
        for name, mod in enumerate_eligible_projections(model):
            W = mod.weight.data.float()
            mask = exact_row_prune_mask(wanda_scores_for(W, stats[name]), int(0.5 * W.shape[1]))
            D = (W * mask - W).double()
            explicit = (D @ X[name].double().t()).pow(2).sum().item()
            self.assertAlmostEqual(sens["per_projection"][name]["err"] / explicit, 1.0, places=3)


@unittest.skipUnless(HAS_TF, "transformers not installed")
class TestP3Allocate(unittest.TestCase):
    def setUp(self):
        from task1_compression.prune_nonuniform import projection_shapes
        self.model = tiny_qwen2(6)
        self.shapes = projection_shapes(self.model)
        self.N = sum(o * i for o, i in self.shapes.values())

    def _check(self, sens, S, lam):
        from task1_compression.prune_nonuniform import S_MAX, S_MIN, allocate, block_index, p2_zero_count
        a = allocate(sens, S, lam, self.shapes)
        target = int(round(S * self.N))
        self.assertEqual(a["achieved_zeros"], target)
        self.assertEqual(sum(a["proj_zeros"].values()), target)
        self.assertEqual(target, p2_zero_count(self.shapes, S))
        for l, s in a["s_block"].items():
            self.assertGreaterEqual(s, S_MIN - 1e-12)
            self.assertLessEqual(s, S_MAX + 1e-12)
        for n, z in a["proj_zeros"].items():
            o, i = self.shapes[n]
            self.assertGreaterEqual(z / (o * i), S_MIN - 1e-12, n)
            self.assertLessEqual(z / (o * i), S_MAX + 1e-12, n)
            base, extra = a["row_counts"][n]
            self.assertEqual(base * o + extra, z)
            self.assertTrue(0 <= extra < o)
            # Close to the block's continuous target (within one weight per row + rounding).
            self.assertLess(abs(z - a["s_block"][block_index(n)] * o * i), o + 2)
        return a

    def test_exact_zero_count_and_bounds(self):
        rng = random.Random(0)
        for S in (0.5, 0.75):
            for lam in (0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.5, 2.0):
                sens = {l: rng.random() for l in range(6)}
                a = self._check(sens, S, lam)
                if lam == 0.0:
                    for s in a["s_block"].values():
                        self.assertAlmostEqual(s, S, places=9)

    def test_direction_and_ties(self):
        from task1_compression.prune_nonuniform import allocate
        sens = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0, 5: 0.5}
        a = allocate(sens, 0.5, 0.1, self.shapes)
        # Most sensitive block (0) is pruned least, least sensitive (5) most.
        self.assertLess(a["s_block"][0], a["s_block"][5])
        self._check({l: 1.0 for l in range(6)}, 0.5, 0.16)  # all tied

    def test_masks_reproduce_counts(self):
        from task1_compression.baseline import collect_calibration_stats, enumerate_eligible_projections
        from task1_compression.prune_nonuniform import allocate, apply_allocation
        ids = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(1)).tolist()
        with torch.no_grad():
            stats = collect_calibration_stats(self.model, ids, "cpu", with_gram=False)
        originals = {n: m.weight.data.clone() for n, m in enumerate_eligible_projections(self.model)}
        a = allocate({l: float(l) for l in range(6)}, 0.75, 0.16, self.shapes)
        masks = apply_allocation(self.model, a, stats, originals, "cpu")
        total = 0
        for n, m in enumerate_eligible_projections(self.model):
            z = int((~masks[n]["mask"]).sum())
            self.assertEqual(z, a["proj_zeros"][n])
            rows = (~masks[n]["mask"]).sum(dim=1)
            self.assertLessEqual(int(rows.max() - rows.min()), 1)
            total += z
        self.assertEqual(total, a["target_zeros"])
        # Restoring from originals then re-applying is idempotent on the counts.
        masks2 = apply_allocation(self.model, a, stats, originals, "cpu")
        for n in masks:
            self.assertTrue(torch.equal(masks[n]["mask"], masks2[n]["mask"]))


class TestQ3Allocate(unittest.TestCase):
    def _table(self, seed, n=20):
        from task1_compression.quant_mixed import projection_bytes
        rng = random.Random(seed)
        tab = {}
        for k in range(n):
            d_out, d_in = rng.choice([(128, 128), (384, 128), (128, 384), (256, 256)])
            e2 = rng.uniform(0.05, 0.5)
            errs = {2: e2, 3: e2 * rng.uniform(0.2, 0.5), 4: e2 * rng.uniform(0.02, 0.15),
                    8: e2 * rng.uniform(1e-5, 1e-3)}
            tab[f"model.layers.{k // 7}.mlp.p{k}"] = {
                b: {"rel_err": errs[b], "bytes": projection_bytes(d_out, d_in, b)} for b in (2, 3, 4, 8)}
        return tab

    def test_never_exceeds_budget(self):
        from task1_compression.quant_mixed import allocate
        for seed in range(10):
            tab = self._table(seed)
            fixed = 12345
            lo = fixed + sum(r[2]["bytes"] for r in tab.values())
            hi = fixed + sum(r[8]["bytes"] for r in tab.values())
            for frac in (0.0, 0.1, 0.3, 0.5, 0.77, 1.0):
                budget = int(lo + frac * (hi - lo))
                a = allocate(tab, budget, 2, fixed)
                total = fixed + sum(tab[n][b]["bytes"] for n, b in a["bits"].items())
                self.assertEqual(total, a["total_bytes"])
                self.assertLessEqual(total, budget)
                self.assertTrue(all(b in (2, 3, 4, 8) for b in a["bits"].values()))
                # Greedy stops only when no single upgrade fits.
                for n, b in a["bits"].items():
                    for b2 in (3, 4, 8):
                        if b2 > b:
                            self.assertGreater(total + tab[n][b2]["bytes"] - tab[n][b]["bytes"], budget)
            with self.assertRaises(ValueError):
                allocate(tab, lo - 1, 2, fixed)

    def test_unlimited_budget_all_8bit(self):
        from task1_compression.quant_mixed import allocate
        for b_min in (2, 3):
            tab = self._table(3)
            a = allocate(tab, float("inf"), b_min, 0)
            self.assertTrue(all(b == 8 for b in a["bits"].values()))

    def test_uniform_budget_b3_gives_uniform(self):
        from task1_compression.quant_mixed import allocate
        tab = self._table(5)
        budget = sum(r[3]["bytes"] for r in tab.values())
        a = allocate(tab, budget, 3, 0)
        self.assertEqual(a["total_bytes"], budget)
        self.assertTrue(all(b == 3 for b in a["bits"].values()))


@unittest.skipUnless(HAS_TF, "transformers not installed")
class TestQ3TableOnModel(unittest.TestCase):
    def test_table_matches_q2_and_budget(self):
        from task1_compression.baseline import collect_calibration_stats, enumerate_eligible_projections
        from task1_compression.quant_awq import apply_awq_quantization
        from task1_compression.quant_mixed import allocate, bit_error_table, quantize_with_allocation

        model = tiny_qwen2(2)
        ids = torch.randint(0, 256, (3, 24), generator=torch.Generator().manual_seed(2)).tolist()
        with torch.no_grad():
            stats = collect_calibration_stats(model, ids, "cpu", with_gram=True)
        tab = bit_error_table(model, stats, device="cpu")["table"]
        originals = {n: m.weight.data.clone() for n, m in enumerate_eligible_projections(model)}
        for n, row in tab.items():
            self.assertLessEqual(row[8]["rel_err"], row[2]["rel_err"])
        # An all-4 allocation reproduces Q2-INT4 exactly (same quantizer, unchanged).
        quantize_with_allocation(model, tab, {n: 4 for n in tab}, originals, device="cpu")
        q3w = {n: m.weight.data.clone() for n, m in enumerate_eligible_projections(model)}
        for n, m in enumerate_eligible_projections(model):
            m.weight.data.copy_(originals[n])
        apply_awq_quantization(model, stats, 4, device="cpu")
        for n, m in enumerate_eligible_projections(model):
            self.assertTrue(torch.equal(q3w[n], m.weight.data), n)
        budget = sum(r[4]["bytes"] for r in tab.values())
        a = allocate(tab, budget, 2, 0)
        self.assertLessEqual(a["total_bytes"], budget)


if __name__ == "__main__":
    unittest.main(verbosity=2)
