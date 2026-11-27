"""
CPU tests for Task 2 Part B (MQAR): masks, receptive field, causality, linear-attention
equivalence, B3 state/parameter matching, LR schedule, the 42-run grid, and an
end-to-end smoke run of the driver on a fake data repo (no downloads).
Seed 6304 everywhere.
"""

import gzip
import json
import math
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from task2_attention import mqar_model as mm  # noqa: E402
from task2_attention import mqar_train as mt  # noqa: E402
from task2_attention import part_b  # noqa: E402

SEED = 6304


class TestMasks(unittest.TestCase):
    def test_window_mask(self):
        m = mm.window_mask(256, 64)
        self.assertEqual(int(m[255].sum()), 64)
        self.assertEqual(int(m[10].sum()), 11)
        self.assertTrue(bool(m[255, 192]) and not bool(m[255, 191]))
        self.assertFalse(bool(m.triu(1).any()))

    def test_dilated_offsets_capped_at_64(self):
        for T in (256, 1024):
            m = mm.dilated_offset_mask(T)
            row = m[T - 1]
            offs = sorted((T - 1 - row.nonzero().flatten()).tolist())
            self.assertEqual(offs, list(range(0, 253, 4)))
            self.assertEqual(len(offs), 64)
        m = mm.mixed_dilated_mask(256)
        self.assertEqual(tuple(m.shape), (4, 256, 256))
        self.assertTrue(torch.equal(m[0], mm.window_mask(256, 64)))

    def test_topk_mask(self):
        torch.manual_seed(SEED)
        s = torch.randn(2, 3, 20, 20)
        m = mm.topk_causal_mask(s, 5)
        self.assertFalse(bool((m & ~mm.causal_mask(20)).any()))
        counts = m.sum(-1)
        self.assertTrue(torch.all(counts[..., 5:] == 5))
        self.assertTrue(torch.all(counts[..., :5] == torch.arange(1, 6)))


class TestModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        self.tok = torch.randint(2, 256, (2, 256))

    def test_all_types_finite_and_shapes(self):
        qp = torch.tensor([[100, 200], [150, 255]])
        for t in mm.ATTN_TYPES:
            torch.manual_seed(SEED)
            m = mm.make_mqar_model(t, device="cpu")
            out = m(self.tok, qp)
            self.assertEqual(tuple(out.shape), (2, 2, 64), t)
            self.assertTrue(torch.isfinite(out).all(), f"{t} produced non-finite logits")

    def test_causality_every_type(self):
        for t in mm.ATTN_TYPES:
            torch.manual_seed(SEED)
            m = mm.make_mqar_model(t, device="cpu").eval()
            tok2 = self.tok.clone()
            tok2[:, 180:] = (tok2[:, 180:] + 7) % 254 + 2
            with torch.no_grad():
                a, b = m(self.tok), m(tok2)
            self.assertLess((a[:, :180] - b[:, :180]).abs().max().item(), 1e-5, t)
            self.assertGreater((a[:, 180:] - b[:, 180:]).abs().max().item(), 0.0, t)

    def test_window_two_layer_receptive_field_is_126(self):
        torch.manual_seed(SEED)
        m = mm.make_mqar_model("window", device="cpu").eval()
        j = 50
        tok2 = self.tok.clone()
        tok2[:, j] = (tok2[:, j] + 5) % 254 + 2
        with torch.no_grad():
            d = (m(self.tok) - m(tok2)).abs().amax(dim=(0, 2))
        self.assertGreater(d[j + 126].item(), 0.0)          # distance 126 reachable
        self.assertEqual(d[j + 127:].max().item(), 0.0)     # distance >= 127 unreachable

    def test_linear_parallel_equals_recurrent(self):
        torch.manual_seed(SEED)
        Q, K, V = (torch.randn(2, 3, 40, 8, dtype=torch.float64) for _ in range(3))
        a = mm.linear_attention_parallel(Q, K, V)
        b = mm.linear_attention_recurrent(Q, K, V)
        self.assertLess((a - b).abs().max().item(), 1e-10)

    def test_dense_eval_override_matches_window_model(self):
        torch.manual_seed(SEED)
        dense = mm.make_mqar_model("dense", device="cpu").eval()
        win = mm.make_mqar_model("window", device="cpu").eval()
        win.load_state_dict(dense.state_dict())
        with torch.no_grad():
            dense.set_eval_override(("window", 64))
            a = dense(self.tok)
            dense.set_eval_override(None)
            b = win(self.tok)
            c = dense(self.tok)
        self.assertLess((a - b).abs().max().item(), 1e-4)
        self.assertGreater((a - c).abs().max().item(), 1e-4)

    def test_topk_override_with_full_k_equals_dense(self):
        torch.manual_seed(SEED)
        dense = mm.make_mqar_model("dense", device="cpu").eval()
        with torch.no_grad():
            ref = dense(self.tok)
            dense.set_eval_override(("topk", 256))
            out = dense(self.tok)
        self.assertLess((ref - out).abs().max().item(), 1e-4)

    def test_override_ignored_in_training_mode(self):
        torch.manual_seed(SEED)
        dense = mm.make_mqar_model("dense", device="cpu").train()
        ref = dense(self.tok)
        dense.set_eval_override(("window", 64))
        out = dense(self.tok)
        self.assertLess((ref - out).abs().max().item(), 1e-6)


class TestB3Matching(unittest.TestCase):
    def test_same_params_as_baselines(self):
        counts = {t: mm.count_trainable_params(mm.make_mqar_model(t, device="cpu")) for t in mm.ATTN_TYPES}
        base = counts["linear"]
        self.assertLessEqual(abs(counts["local_linear"] - base) / base, 0.05)
        self.assertEqual(len(set(counts.values())), 1, counts)

    def test_state_bytes_equal_linear(self):
        lin = mm.persistent_state_bytes("linear")["bytes_per_layer"]
        b3 = mm.persistent_state_bytes("local_linear")["bytes_per_layer"]
        self.assertEqual(lin, 4 * (32 * 32 + 32) * 4)
        self.assertEqual(b3, lin)

    def test_pair_counts_T256(self):
        self.assertEqual(mm.accessible_positions_per_query("dense")["attended_pairs_total"],
                         4 * 2 * 256 * 257 / 2)
        win = sum(min(i + 1, 64) for i in range(256))
        self.assertEqual(mm.accessible_positions_per_query("window")["attended_pairs_total"], 8 * win)
        self.assertEqual(mm.accessible_positions_per_query("window")["two_layer_receptive_field"], 126)


class TestSchedule(unittest.TestCase):
    def test_warmup_and_decay_to_15pct(self):
        f = [mt.linear_warmup_decay(s) for s in range(mt.N_UPDATES)]
        self.assertAlmostEqual(f[0], 1 / 200)
        self.assertAlmostEqual(f[199], 1.0)
        self.assertAlmostEqual(f[200], 1.0)
        self.assertAlmostEqual(f[-1], 0.15)
        self.assertTrue(all(a >= b for a, b in zip(f[200:], f[201:])))


class TestGrid(unittest.TestCase):
    def test_42_runs(self):
        g = part_b.build_grid()
        self.assertEqual(len(g), 42)
        by_step = {s: sum(1 for r in g if r["step"] == s) for s in ("B1", "B2", "B3")}
        self.assertEqual(by_step, {"B1": 24, "B2": 15, "B3": 3})
        self.assertEqual({r["seed"] for r in g}, {6304, 6305, 6306})
        self.assertIn("dense_K16_near_s6305", {r["run"] for r in g})
        self.assertEqual(part_b.select_runs("B3"),
                         [f"local_linear_K16_mixed_s{s}" for s in (6304, 6305, 6306)])


# ── driver smoke test on a fake data repo ─────────────────────────────────────

# Stdlib-only copy of the supplied generator's algorithm (data repo scripts/mqar_generator.py),
# used only so the test needs no download.
FAKE_GEN = textwrap.dedent('''
    import hashlib, random
    T = 256
    def stable_seed(split, k, regime, index):
        msg = f"FAKE|{split}|{k}|{regime}|{index}".encode()
        return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")
    def make_example(split, k, regime, index):
        rng = random.Random(stable_seed(split, k, regime, index))
        keys = rng.sample(range(64, 128), k); vals = rng.sample(range(128, 192), k)
        tokens = [rng.randrange(2, 64) for _ in range(T)]
        for j in range(k):
            tokens[2*j] = keys[j]; tokens[2*j+1] = vals[j]
        order = list(range(k)); rng.shuffle(order); used = set(range(2*k)); pos = {}
        for j in order:
            src = 2*j
            c = list(range(2*k+1, T))
            if regime == "near": c = [q for q in c if 1 <= q-src <= 63]
            elif regime == "far": c = [q for q in c if 128 <= q-src <= 250]
            else: c = [q for q in c if 1 <= q-src <= 250]
            rng.shuffle(c); ch = [q for q in c if q not in used]
            pos[j] = ch[0]; used.add(ch[0]); tokens[ch[0]] = keys[j]
        q = sorted((p, vals[j], 2*j, p-2*j) for j, p in pos.items())
        ex = {"id": f"{split}-{regime}-K{k}-{index:05d}", "split": split, "regime": regime, "K": k,
              "tokens": tokens, "query_positions": [x[0] for x in q], "answer_ids": [x[1] for x in q],
              "source_positions": [x[2] for x in q], "distances": [x[3] for x in q]}
        validate(ex); return ex
    def validate(ex):
        x = ex["tokens"]
        assert len(x) == T and len(set(ex["query_positions"])) == ex["K"]
        for p, s, y, d in zip(ex["query_positions"], ex["source_positions"], ex["answer_ids"], ex["distances"]):
            assert x[s] == x[p] and x[s+1] == y and p - s == d
''')


def make_fake_repo(root: Path, n_per_cell: int = 3):
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "mqar_generator.py").write_text(FAKE_GEN)
    ns = {}
    exec(FAKE_GEN, ns)  # noqa: S102 - test-local generator text defined above
    (root / "data" / "mqar").mkdir(parents=True)
    for split, fname in (("test", "mqar_test.jsonl.gz"), ("development", "mqar_development.jsonl.gz")):
        with gzip.open(root / "data" / "mqar" / fname, "wt") as fh:
            for K in (4, 8, 16):
                for reg in ("near", "far", "mixed"):
                    for i in range(n_per_cell):
                        fh.write(json.dumps(ns["make_example"](split, K, reg, i)) + "\n")


class TestDriverSmoke(unittest.TestCase):
    def test_grid_runs_resume_and_error_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            make_fake_repo(td / "repo")
            res = td / "res"
            runs = ["dense_K4_mixed_s6304", "local_linear_K16_mixed_s6304"]
            st = part_b.run_grid(str(res), str(td / "repo"), runs, "cpu", n_updates=2,
                                 cache_dir=str(td / "cache"), log=lambda *_: None)
            self.assertEqual(st, {r: "done" for r in runs})
            s = json.load(open(res / "runs" / runs[0] / "summary.json"))
            self.assertFalse(s["full_protocol"])
            self.assertEqual(set(s["test"]["cells"]), {f"K{k}_{r}" for k in (4, 8, 16)
                                                       for r in ("near", "far", "mixed")})
            self.assertIn("test_time_window64", s["test_time_controls"])
            self.assertIn("test_time_topk64", s["test_time_controls"])
            self.assertTrue((res / "runs" / runs[0] / "ckpt_step2.pt").exists())
            man = json.load(open(res / "grid_manifest.json"))
            self.assertEqual(man["n_runs"], 42)
            # resume: finished runs are skipped
            st2 = part_b.run_grid(str(res), str(td / "repo"), runs, "cpu", n_updates=2,
                                  cache_dir=str(td / "cache"), log=lambda *_: None)
            self.assertEqual(set(st2.values()), {"skipped"})
            # a failing run writes error.json and the driver continues
            orig = part_b.train_one

            def boom(cfg, *a, **k):
                if cfg["attn_type"] == "window":
                    raise RuntimeError("injected")
                return orig(cfg, *a, **k)
            part_b.train_one = boom
            try:
                st3 = part_b.run_grid(str(res), str(td / "repo"),
                                      ["window_K4_near_s6304", "linear_K4_mixed_s6304"], "cpu",
                                      n_updates=2, cache_dir=str(td / "cache"), log=lambda *_: None)
            finally:
                part_b.train_one = orig
            self.assertEqual(st3["window_K4_near_s6304"], "failed")
            self.assertEqual(st3["linear_K4_mixed_s6304"], "done")
            self.assertTrue((res / "runs" / "window_K4_near_s6304" / "error.json").exists())

    def test_aggregate_on_smoke_runs(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        from task2_attention.aggregate import aggregate
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            make_fake_repo(td / "repo")
            runs = ["dense_K16_mixed_s6304", "window_K16_mixed_s6304", "linear_K16_mixed_s6304",
                    "local_linear_K16_mixed_s6304"]
            part_b.run_grid(str(td / "t2" / "b"), str(td / "repo"), runs, "cpu", n_updates=2,
                            cache_dir=str(td / "cache"), log=lambda *_: None)
            st = aggregate(str(td / "t2"))
            self.assertEqual(st["b"]["status"], "ok", st["b"])
            self.assertEqual(st["b"]["n_runs"], 4)
            self.assertEqual(st["a"]["status"], "missing")
            self.assertEqual(st["c"]["status"], "missing")
            import csv
            rows = list(csv.DictReader(open(td / "t2" / "tables" / "b_runs.csv")))
            self.assertEqual(len(rows), 4)
            ctrl = list(csv.DictReader(open(td / "t2" / "tables" / "b_mask_controls.csv")))
            self.assertEqual(len(ctrl), 3)                      # dense K16 mixed x 3 regimes
            self.assertTrue((td / "t2" / "tables" / "b3_intervention.csv").exists())

    def test_sweep_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            st = part_b.run_sweep(td, "cpu", Ts=(16,), n_warmup=0, n_rep=1, log=lambda *_: None)
            self.assertEqual(set(st.values()), {"ok"})
            self.assertEqual(len(st), len(part_b.SWEEP_METHODS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
