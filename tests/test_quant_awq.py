"""
Unit test: AWQ-style quantization (serialized compensation) on a tiny random Qwen2.

No downloads: the model is built from a Qwen2Config (hidden 128, intermediate 384,
2 layers, 4 heads, 2 KV heads, vocab 256, tied embeddings).  Checks:
  - apply_awq_quantization runs (the old folding code crashed on this shape);
  - non-target weights (norms, embeddings) are unchanged;
  - each eligible weight equals dequant(quant(W * alpha)) * diag(1/alpha) for its
    recorded alpha, and the byte accounting includes 2*d_in compensating bytes;
  - quant_mixed budgets include the same compensating bytes.
"""

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from transformers import Qwen2Config, Qwen2ForCausalLM
    HAS_TF = True
except Exception:  # pragma: no cover
    HAS_TF = False

TF_SKIP = "transformers not installed"


def tiny_qwen2(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        hidden_size=128, intermediate_size=384, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
        tie_word_embeddings=True, max_position_embeddings=2048,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    # Non-trivial norm weights so "unchanged" is a meaningful check.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5)
    return model


@unittest.skipUnless(HAS_TF, TF_SKIP)
class TestAWQ(unittest.TestCase):
    def setUp(self):
        from task1_compression.baseline import collect_calibration_stats
        self.model = tiny_qwen2()
        g = torch.Generator().manual_seed(1)
        self.calib_ids = torch.randint(0, 256, (4, 32), generator=g).tolist()
        self.stats = collect_calibration_stats(self.model, self.calib_ids, "cpu", with_gram=True)

    def test_awq_no_crash_targets_only_and_exact_reconstruction(self):
        from task1_compression.baseline import enumerate_eligible_projections
        from task1_compression.quant_awq import apply_awq_quantization, awq_compensated_weight

        eligible = {n for n, _ in enumerate_eligible_projections(self.model)}
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}

        res = apply_awq_quantization(self.model, self.stats, bits=4, group_size=128, device="cpu")
        self.assertEqual(set(res), eligible)

        after = dict(self.model.named_parameters())
        for n, p in after.items():
            mod_name = n.rsplit(".", 1)[0]
            if mod_name in eligible and n.endswith("weight"):
                continue
            self.assertTrue(torch.equal(p, before[n]), f"non-target parameter changed: {n}")

        for name, mod in enumerate_eligible_projections(self.model):
            info = res[name]
            W0 = before[name + ".weight"].float()
            alpha = info["awq_scale"]
            expected, *_ = awq_compensated_weight(W0, alpha, 4, 128)
            torch.testing.assert_close(mod.weight.data.float(), expected.to(mod.weight.dtype).float(),
                                       rtol=0, atol=1e-6)
            d_out, d_in = W0.shape
            self.assertIsNone(info["folded_into"])
            self.assertEqual(info["compensating_scale_bytes"], 2 * d_in)
            self.assertEqual(info["total_bytes"],
                             math.ceil(d_out * d_in * 4 / 8) + math.ceil(d_in / 128) * 4 * d_out + 2 * d_in)
            self.assertLess(info["prerounding_rel_output_err"], 1e-10)
            self.assertNotIn("q_int", info)

        out = self.model(torch.tensor(self.calib_ids[:1])).logits
        self.assertTrue(torch.isfinite(out).all())

    def test_mixed_budget_includes_compensating_bytes(self):
        from task1_compression.baseline import enumerate_eligible_projections
        from task1_compression.quant_mixed import (allocate, bit_error_table,
                                                   quantize_with_allocation)

        projs = enumerate_eligible_projections(self.model)
        uniform4 = sum(math.ceil(m.weight.numel() * 4 / 8)
                       + math.ceil(m.weight.shape[1] / 128) * 4 * m.weight.shape[0]
                       + 2 * m.weight.shape[1] for _, m in projs)
        bits = (4, 8)
        table = bit_error_table(self.model, self.stats, bits=bits, group_size=128, device="cpu")["table"]
        for n, m in projs:
            self.assertEqual(table[n][4]["bytes"],
                             math.ceil(m.weight.numel() * 4 / 8)
                             + math.ceil(m.weight.shape[1] / 128) * 4 * m.weight.shape[0]
                             + 2 * m.weight.shape[1])

        # With the compensating bytes in both budget and cost, uniform INT4 fits
        # exactly and no upgrade fits on top of it.
        alloc = allocate(table, uniform4, b_min=4, allowed_bits=bits)
        self.assertEqual(alloc["bits"], {n: 4 for n, _ in projs})
        self.assertLessEqual(alloc["total_bytes"], uniform4)

        res = quantize_with_allocation(self.model, table, alloc["bits"], group_size=128, device="cpu")
        self.assertEqual(set(res), {n for n, _ in projs})
        for n, m in projs:
            self.assertEqual(res[n]["compensating_scale_bytes"], 2 * m.weight.shape[1])
        self.assertLessEqual(sum(r["total_bytes"] for r in res.values()), uniform4)

if __name__ == "__main__":
    unittest.main(verbosity=2)
