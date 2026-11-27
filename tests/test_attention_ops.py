"""CPU tests for Task 2 Part A operators and the Part A driver (seed 6304, no GPU)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from task2_attention import attention_ops as ao  # noqa: E402
from task2_attention import part_a  # noqa: E402

SEED = 6304


def _rand(*shape, dtype=torch.float64):
    g = torch.Generator().manual_seed(SEED)
    return torch.randn(*shape, generator=g, dtype=dtype)


class TestOps(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(SEED)
        self.Q, self.K, self.V = (torch.randn(2, 3, 70, 16, generator=g, dtype=torch.float64) for _ in range(3))

    def test_tiled_equals_dense_all_tile_sizes(self):
        ref = ao.dense_causal_attention(self.Q, self.K, self.V)
        for ts in (1, 7, 16, 64, 128):
            out = ao.tiled_causal_attention(self.Q, self.K, self.V, tile_size=ts)
            self.assertLess((out - ref).abs().max().item(), 1e-6, ts)   # FP32 accumulators

    def test_dense_matches_sdpa(self):
        ref = ao.sdpa_attention(self.Q.float(), self.K.float(), self.V.float())
        out = ao.dense_causal_attention(self.Q.float(), self.K.float(), self.V.float())
        self.assertLess((out - ref).abs().max().item(), 1e-5)

    def test_tiled_gradients_match_dense(self):
        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (self.Q, self.K, self.V))
        q2, k2, v2 = (t.clone().requires_grad_(True) for t in (self.Q, self.K, self.V))
        dO = torch.randn_like(self.Q)
        ao.dense_causal_attention(q1, k1, v1).backward(dO)
        ao.tiled_causal_attention(q2, k2, v2, tile_size=16).backward(dO)
        for a, b in ((q1, q2), (k1, k2), (v1, v2)):
            self.assertLess((a.grad - b.grad).abs().max().item(), 1e-6)

    def test_causality_every_position(self):
        for name, fn in (("dense", ao.dense_causal_attention),
                         ("tiled", lambda q, k, v: ao.tiled_causal_attention(q, k, v, tile_size=8)),
                         ("sdpa", ao.sdpa_attention)):
            for j in (1, 17, 40):
                r = ao.verify_causality(fn, L=48, j=j)
                self.assertTrue(r["causal"], (name, j, r))

    def test_all_masked_rows_are_zero_with_finite_grads(self):
        km = torch.ones(2, 70, dtype=torch.bool)
        km[0, :5] = False
        for fn in (ao.dense_causal_attention, lambda q, k, v, **kw: ao.tiled_causal_attention(q, k, v, 16, **kw)):
            q, k, v = (t.clone().requires_grad_(True) for t in (self.Q, self.K, self.V))
            out = fn(q, k, v, key_mask=km)
            self.assertTrue(torch.isfinite(out).all())
            self.assertEqual(out[0, :, :5].abs().max().item(), 0.0)
            out.sum().backward()
            self.assertTrue(all(torch.isfinite(t.grad).all() for t in (q, k, v)))
        ref = ao.dense_causal_attention(self.Q, self.K, self.V, key_mask=km)
        til = ao.tiled_causal_attention(self.Q, self.K, self.V, tile_size=16, key_mask=km)
        self.assertLess((ref - til).abs().max().item(), 1e-6)

    def test_extreme_scores_stable(self):
        Qs = self.Q * 300
        ref = ao.dense_causal_attention(Qs, self.K, self.V)
        out = ao.tiled_causal_attention(Qs.float(), self.K.float(), self.V.float(), tile_size=16)
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess((out.double() - ref).abs().max().item(), 1e-3)

    def test_kernel_classifier(self):
        self.assertEqual(ao.classify_kernels(["pytorch_flash::flash_fwd_kernel"]), "flash")
        self.assertEqual(ao.classify_kernels(["fmha_cutlassF_f32_aligned_64x64"]), "efficient")
        self.assertEqual(ao.classify_kernels(["softmax_warp_forward", "ampere_sgemm"]), "math")


class TestPartADriver(unittest.TestCase):
    def test_sweep_resumable_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            st = part_a.run_sweep(td, "cpu", lengths=(64,), dtypes=("fp32",), n_warmup=0, n_rep=1,
                                  log=lambda *_: None)
            self.assertEqual(st["fp32_L64_dense"], "ok")
            self.assertEqual(st["fp32_L64_tiled"], "ok")
            self.assertEqual(st["fp32_L64_sdpa"], "ok")
            self.assertIn(st["fp32_L64_sdpa_cudnn"], ("ok", "unsupported"))
            rec = json.load(open(Path(td) / "conditions" / "fp32_L64_tiled.json"))
            self.assertLess(rec["output_error"]["max_abs"], 1e-5)
            self.assertLess(rec["grad_error"]["dQ"]["max_abs"], 1e-4)
            self.assertEqual(rec["forward"]["n_rep"], 1)
            st2 = part_a.run_sweep(td, "cpu", lengths=(64,), dtypes=("fp32",), n_warmup=0, n_rep=1,
                                   log=lambda *_: None)
            self.assertEqual(set(st2.values()), {"skipped"})

    def test_checks_cpu(self):
        r = part_a.run_checks("cpu", lengths_storage=(256,))
        self.assertTrue(r["causality"]["dense_fp32"]["causal"])
        self.assertTrue(r["causality"]["tiled_fp32"]["causal"])
        self.assertTrue(all(v["max_abs"] < 1e-4 for v in r["tile_sizes_L1024"].values()))
        self.assertTrue(r["edge_cases"]["allmasked_dense_fp32"]["masked_rows_finite"])
        self.assertTrue(r["edge_cases"]["allmasked_tiled_fp32"]["grads_finite"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
