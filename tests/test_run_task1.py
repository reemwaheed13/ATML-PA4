"""
CPU smoke test for task1_compression.run_task1 (no downloads).

Builds a tiny random Qwen2 (fresh, identically seeded weights per load), a fake
word-level tokenizer and tiny synthetic locked data files, then runs
fp16, p1_s50, p2_s50, p3_s50, p2_24, q1_int4, q2_int4, q3_int4 end to end on CPU,
checks the output files and summary fields (incl. P3/Q3 matched budgets and dev
selection files), resumability, condition groups, and aggregate.
"""

import csv
import gzip
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from transformers import Qwen2Config, Qwen2ForCausalLM
    HAS_TF = True
except Exception:  # pragma: no cover
    HAS_TF = False

VOCAB = 256
SMOKE = ["fp16", "p1_s50", "p2_s50", "p3_s50", "p2_24", "q1_int4", "q2_int4", "q3_int4"]


class WordTokenizer:
    """Qwen-like spacing: a space attaches to the following word."""
    pad_token_id = 0
    eos_token_id = 0
    vocab_size = VOCAB

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False, **kw):
        ids = []
        for piece in re.findall(r" ?[^ ]+| +", text):
            if piece not in self.vocab:
                self.vocab[piece] = 1 + (len(self.vocab) % (VOCAB - 1))
            ids.append(self.vocab[piece])
        return {"input_ids": ids}


def tiny_loader(device):
    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=128, intermediate_size=384, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
                      tie_word_embeddings=True, max_position_embeddings=2048)
    model = Qwen2ForCausalLM(cfg).to(device).eval()
    return model, WordTokenizer()


def _write_gz(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def make_data(root):
    g = torch.Generator().manual_seed(123)

    def ids(n):
        return torch.randint(1, VOCAB, (n,), generator=g).tolist()

    _write_gz(os.path.join(root, "c4_calibration_locked.jsonl.gz"),
              [{"token_ids": ids(32), "source_row": i, "token_offset": 0} for i in range(6)])
    _write_gz(os.path.join(root, "c4_development_locked.jsonl.gz"),
              [{"token_ids": ids(32), "source_row": i, "token_offset": 0} for i in range(2)])

    def win(n):
        return {"token_ids": ids(n), "score_target_offsets": [1, n], "scored_tokens": n - 1}

    # Both files total 96 scored tokens; WT2 has variable-length windows.
    _write_gz(os.path.join(root, "c4_validation_windows_locked.jsonl.gz"), [win(33) for _ in range(3)])
    _write_gz(os.path.join(root, "wikitext2_test_windows_locked.jsonl.gz"),
              [win(33), win(33), win(17), win(17)])
    ex = {"activity_label": "Cooking", "ctx": "A man cracks eggs into a bowl. He",
          "ctx_a": "A man cracks eggs into a bowl.", "ctx_b": "he",
          "endings": ["whisks them.", "throws the bowl away.", "eats.", "bakes it for an hour."]}
    _write_gz(os.path.join(root, "hellaswag_val_locked.jsonl.gz"),
              [{"order": o, "example": {**ex, "label": str(o % 4)}, "source_row_index": 10 + o}
               for o in range(512)])
    return 96


@unittest.skipUnless(HAS_TF, "transformers not installed")
class TestRunTask1Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from task1_compression import run_task1
        cls.rt = run_task1
        cls.tmp = tempfile.mkdtemp()
        cls.data = os.path.join(cls.tmp, "data")
        cls.results = os.path.join(cls.tmp, "results")
        os.makedirs(cls.data)
        cls.n = make_data(cls.data)
        cls.base_args = ["run", "--data-root", cls.data, "--results-root", cls.results,
                         "--device", "cpu", "--expected-n", str(cls.n),
                         "--calib-cache", os.path.join(cls.tmp, "calib_stats.pt"),
                         "--q3-table-cache", os.path.join(cls.tmp, "q3_table.pt")]
        cls.status = run_task1.main(cls.base_args + ["--conditions", ",".join(SMOKE)],
                                    load_model=tiny_loader)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _summary(self, c):
        with open(os.path.join(self.results, c, "summary.json")) as fh:
            return json.load(fh)

    def test_registry(self):
        self.assertEqual(len(self.rt.CONDITIONS), 18)
        self.assertEqual(len(self.rt.OVERNIGHT), 14)
        self.assertEqual(self.rt.ADAPTIVE, ["p3_s50", "p3_s75", "q3_int4", "q3_int3"])
        full = {c for c, v in self.rt.CONDITIONS.items() if v["hs_full"]}
        self.assertEqual(full, {"fp16", "p1_s50", "p2_s50", "p3_s50", "q1_int4", "q1_int3",
                                "q2_int4", "q2_int3", "q3_int4", "q3_int3"})

    def test_all_conditions_ran_and_wrote_files(self):
        self.assertEqual(self.status, {c: "done" for c in SMOKE})
        for c in SMOKE:
            for f in ("summary.json", "ppl_c4_per_window.json", "ppl_wt2_per_window.json",
                      "hellaswag_per_example.json"):
                self.assertTrue(os.path.exists(os.path.join(self.results, c, f)), f"{c}/{f}")
            s = self._summary(c)
            self.assertEqual(s["c4_val"]["N"], self.n)
            self.assertEqual(s["wt2_test"]["N"], self.n)
            self.assertEqual(s["hellaswag"]["n_examples"], 512)
            self.assertIn("task_version", s["hellaswag"])
            lat = s["latency"]
            for k in ("prefill_512", "prefill_2048", "decode_64_after_prefill_512"):
                self.assertEqual(lat[k]["n_rep"], 10)
                self.assertEqual(lat[k]["n_warmup"], 3)
            self.assertIn("dense FP16 execution", lat["label"])
            st = s["storage"]
            self.assertEqual(st["dense_fp16_bytes"]["label"], "dense")
            self.assertEqual(st["theoretical_sparse_fp16_bytes"]["label"], "theoretical")
            self.assertIn("calibration_output_error", s)
        self.assertTrue(os.path.exists(os.path.join(self.results, "latency_input_ids.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "calib_stats.pt")))

    def test_condition_specific_fields(self):
        fp = self._summary("fp16")
        self.assertEqual(fp["calibration_output_error"]["global_token_weighted_mean_rel_output_err"], 0.0)
        for c in ("p1_s50", "p2_s50", "p2_24"):
            s = self._summary(c)
            self.assertAlmostEqual(s["sparsity"]["mask_sparsity"]["global"], 0.5, places=6)
            self.assertGreaterEqual(s["sparsity"]["achieved_eligible_sparsity"], 0.5)
            self.assertGreater(s["calibration_output_error"]["global_token_weighted_mean_rel_output_err"], 0.0)
        self.assertIn("score_distribution", self._summary("p1_s50")["details"])
        self.assertIn("score_distribution", self._summary("p2_s50")["details"])
        ov = self._summary("p2_24")["details"]["overlap_with_p2_s50"]
        self.assertTrue(ov["identical_eligible_sparsity"])
        self.assertTrue(0.0 < ov["global_jaccard"] <= 1.0)
        for c in ("q1_int4", "q2_int4"):
            st = self._summary(c)["storage"]
            self.assertEqual(st["packed_quant_bytes"]["label"], "packed")
            self.assertEqual(st["nominal_bits_per_eligible"], 4.0)
        q2 = self._summary("q2_int4")
        self.assertGreater(q2["storage"]["packed_quant_bytes"]["compensating_scale_bytes"], 0)
        pr = q2["details"]["prerounding_check"]
        self.assertLess(pr["projection_level"]["max_rel_output_err_fp32"], 1e-10)
        self.assertLess(pr["model_level"]["max_abs_logit_diff"], 1e-3)

    def test_p3_q3_fields(self):
        p3 = self._summary("p3_s50")
        d = p3["details"]["p3"]
        self.assertTrue(d["matches_p2_zero_count"])
        p2 = self._summary("p2_s50")
        self.assertEqual(p3["sparsity"]["mask_sparsity"]["zeros"], p2["sparsity"]["mask_sparsity"]["zeros"])
        self.assertEqual(len(d["dev_selection"]), 5)
        self.assertIn(d["lambda"], d["lambda_grid"])
        for v in d["block_sparsity"].values():
            self.assertTrue(0.10 - 1e-9 <= v <= 0.85 + 1e-9)
        self.assertEqual(set(d["rejected_outlier_proxy"]["per_block"]), set(d["block_sensitivity"]))
        self.assertTrue(os.path.exists(os.path.join(self.results, "p3_s50", "p3_s50_selection.json")))
        q3 = self._summary("q3_int4")
        bc = q3["details"]["q3"]["budget_check"]
        self.assertTrue(bc["within_budget"])
        self.assertLessEqual(q3["storage"]["packed_quant_bytes"]["value"],
                             self._summary("q2_int4")["storage"]["packed_quant_bytes"]["value"])
        self.assertEqual(bc["budget_bytes_q2_uniform"],
                         self._summary("q2_int4")["storage"]["packed_quant_bytes"]["value"])
        self.assertIn(q3["details"]["q3"]["b_min"], (2, 3))
        self.assertTrue(os.path.exists(os.path.join(self.results, "q3_int4", "q3_int4_selection.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "q3_table.pt")))

    def test_resume_skips_and_condition_groups(self):
        status = self.rt.main(self.base_args + ["--conditions", "fp16,p3_s50"], load_model=tiny_loader)
        self.assertEqual(status, {"fp16": "skipped", "p3_s50": "skipped"})
        self.assertEqual(self.rt.parse_conditions("adaptive"), self.rt.ADAPTIVE)
        self.assertEqual(len(self.rt.parse_conditions("all")), 18)
        with self.assertRaises(ValueError):
            self.rt.parse_conditions("p9_s50")

    def test_aggregate(self):
        res = self.rt.main(["aggregate", "--results-root", self.results, "--data-root", self.data])
        self.assertEqual(res["n_conditions"], len(SMOKE))
        with open(os.path.join(self.results, "table.csv")) as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual({r["condition"] for r in rows}, set(SMOKE))
        with open(os.path.join(self.results, "qualitative.csv")) as fh:
            q = list(csv.DictReader(fh))
        self.assertEqual(len({r["order"] for r in q}), 8)
        self.assertTrue(os.path.exists(os.path.join(self.results, "qualitative_orders.json")))
        self.assertTrue(os.path.exists(os.path.join(self.results, "bootstrap.json")))
        with open(os.path.join(self.results, "bootstrap.json")) as fh:
            boot = json.load(fh)
        self.assertIn("p3_s50_minus_p2_s50", boot)
        self.assertIn("q3_int4_minus_q2_int4", boot)
        mb = {r["pair"]: r for r in res["matched_budget"]}
        self.assertTrue(mb["p3_s50 vs p2_s50"]["matched"])
        self.assertTrue(mb["q3_int4 vs q2_int4"]["matched"])
        try:
            import matplotlib  # noqa: F401
            self.assertIn("pruning_quality_vs_sparsity.png", res["plots"])
            self.assertIn("quant_quality_vs_packed_bytes.png", res["plots"])
            for f in ("p3_sensitivity_allocation.png", "q3_sensitivity_bits.png", "dev_selection_curves.png",
                      "calib_error_vs_quality.png"):
                self.assertIn(f, res["plots"])
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
