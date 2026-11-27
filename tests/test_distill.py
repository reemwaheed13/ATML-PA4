"""
CPU tests for Task 3 distillation, prefix banks, LoRA merge and evaluation helpers.
Tiny random Qwen2 teacher/student + character tokenizer; no downloads; seed 6304.
"""

import json
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import peft  # noqa: F401
    import transformers  # noqa: F401
    from tests.tiny import CharTokenizer, tiny_qwen
    from task3_specdec import distill as ds
    from task3_specdec import evaluate as ev
    from task3_specdec import merge as mg
    from task3_specdec import prefix_eval as pe
    from common.runio import read_jsonl
    HAS, SKIP = True, ""
except Exception as exc:  # pragma: no cover
    HAS, SKIP = False, f"transformers/peft unavailable: {exc}"

SEED = 6304


def _prompts():
    math_ = [{"id": f"GSM8K-train-{i:04d}", "category": "math", "text": f"Q{i}: what is {i} + {i}?",
              "answer": f"#### {2 * i}"} for i in range(3)]
    short = [{"id": f"SHORT-train-{i:04d}", "category": "short", "text": f"Return code {i}.", "answer": str(i),
              "family": i % 3} for i in range(3)]
    return {"math": math_, "short": short}


@unittest.skipUnless(HAS, SKIP)
class TestSchedule(unittest.TestCase):
    def test_round_robin_reuse(self):
        s = ds.PromptSchedule(_prompts(), SEED)
        first = [s.next("math")[0]["id"] for _ in range(3)]
        second = [s.next("math")[0]["id"] for _ in range(3)]
        self.assertEqual(sorted(first), sorted(p["id"] for p in _prompts()["math"]))
        self.assertEqual(first, second)                      # same permutation reused
        self.assertEqual(s.next("math")[1], 2)               # third visit of that prompt
        self.assertNotEqual(ds.PromptSchedule(_prompts(), SEED + 1).order, s.order)

    def test_next_category(self):
        q = {"math": 30, "short": 10}
        self.assertEqual(ds.next_category({"math": 0, "short": 0}, q), "math")
        self.assertEqual(ds.next_category({"math": 3, "short": 0}, q), "short")
        self.assertEqual(ds.next_category({"math": 30, "short": 9}, q), "short")
        self.assertIsNone(ds.next_category({"math": 30, "short": 10}, q))

    def test_kl_normalisation_is_mean_token_kl(self):
        g = torch.Generator().manual_seed(SEED)
        t = torch.log_softmax(torch.randn(7, 11, generator=g), -1)
        s = torch.log_softmax(torch.randn(7, 11, generator=g), -1)
        per = torch.nn.functional.kl_div(s, t, log_target=True, reduction="none").sum(-1)
        self.assertAlmostEqual(float(ds.kl_sum(t, s) / 7), float(per.mean()), places=5)


@unittest.skipUnless(HAS, SKIP)
class TestDistillTiny(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = CharTokenizer()
        cls.teacher = tiny_qwen(SEED)

    def _check_run(self, s, out, quota, tpu):
        total = sum(quota.values())
        self.assertEqual(s["scored_tokens"], quota)
        self.assertEqual(s["updates"], math.ceil(total / tpu))
        curves = json.load(open(Path(out) / "curves.json"))
        sizes = [c["n_supervised"] for c in curves]
        self.assertEqual(sizes, [tpu] * (total // tpu) + ([total % tpu] if total % tpu else []))
        self.assertTrue(all(math.isfinite(c["loss_mean_token_kl"]) and c["loss_mean_token_kl"] >= -1e-6
                            for c in curves))
        self.assertTrue((Path(out) / "adapter").exists())
        pres = read_jsonl(Path(out) / "presentations.jsonl.gz")
        self.assertEqual(sum(p["teacher_scored_tokens"] for p in pres), total)
        for p in pres:
            for k in ("source_id", "rollout_seed", "response_ids", "termination", "teacher_scored_tokens"):
                self.assertIn(k, p)
        self.assertGreater(s["params"]["trainable_adapter_params"], 0)
        return pres

    def test_off_policy(self):
        with tempfile.TemporaryDirectory() as td:
            meta = ds.build_teacher_pool(self.teacher, self.tok, _prompts(), str(Path(td) / "pool.jsonl.gz"),
                                         "cpu", batch_size=8, log=lambda *_: None)
            self.assertEqual(meta["n_rollouts"], 3 * 1 + 3 * 8)
            pool = ds.load_pool(str(Path(td) / "pool.jsonl.gz"))
            quota, tpu = {"math": 40, "short": 20}, 16
            s = ds.distill("off", self.teacher, tiny_qwen(SEED + 7), self.tok, _prompts(), SEED,
                           str(Path(td) / "off"), "cpu", pool, meta, quota=quota, tokens_per_update=tpu,
                           log=lambda *_: None)
            pres = self._check_run(s, Path(td) / "off", quota, tpu)
            self.assertGreaterEqual(s["truncated_by_quota"], 1)        # a 192-token math rollout > 40 quota
            ids = {(r["source_id"], r["rollout"]): r["response_ids"] for r in read_jsonl(Path(td) / "pool.jsonl.gz")}
            for p in pres:                                               # off-policy presents frozen pool rollouts
                self.assertEqual(p["response_ids"], ids[(p["source_id"], p["visit"] % ds.POOL_ROLLOUTS[p["category"]])])

    def test_on_policy(self):
        with tempfile.TemporaryDirectory() as td:
            quota, tpu = {"math": 24, "short": 16}, 16
            s = ds.distill("on", self.teacher, tiny_qwen(SEED + 7), self.tok, _prompts(), SEED,
                           str(Path(td) / "on"), "cpu", quota=quota, tokens_per_update=tpu, log=lambda *_: None)
            self._check_run(s, Path(td) / "on", quota, tpu)
            self.assertGreater(s["costs_s"]["student_rollout_s"], 0.0)


@unittest.skipUnless(HAS, SKIP)
class TestBanks(unittest.TestCase):
    def test_positions_match_setup_data(self):
        # verbatim logic of setup_data.py prefix_banks()
        def ref(pid, which, seq_len):
            rng = random.Random(f'PA4-banks-v1:6304:{pid}:{which}')
            positions = []
            for lo, hi, n in [(1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)]:
                candidates = list(range(lo, min(hi, seq_len) + 1))
                positions.extend(sorted(rng.sample(candidates, min(n, len(candidates)))))
            return positions
        for L in (5, 40, 192):
            self.assertEqual(pe.select_positions("GSM8K-0001", pe.SPEC_BANK, L), ref("GSM8K-0001", pe.SPEC_BANK, L))
        recs = pe.make_records({"id": "X", "domain": "math", "text": "t"}, "b", list(range(100, 140)), 1)
        self.assertTrue(all(len(r["response_prefix_token_ids"]) == r["position"] - 1 for r in recs))
        self.assertEqual(len(recs), 8 + 8 + 8)

    def test_alignment_identities_and_eval(self):
        tok = CharTokenizer()
        t, d = tiny_qwen(SEED), tiny_qwen(SEED + 3)
        prompt = {"id": "GSM8K-0000", "domain": "math", "text": "What is 2 + 2?"}
        recs = pe.make_records(prompt, "teacher", list(range(40, 90)), 1)
        rows = pe.evaluate_banks(t, {"self": t, "other": d}, tok, recs, "cpu")
        self.assertEqual(len(rows), 2 * len(recs))
        for r in rows:
            self.assertAlmostEqual(r["expected_accept"], 1 - r["tv"], places=5)
            self.assertGreaterEqual(r["fwd_kl"], -1e-5)
            self.assertGreaterEqual(r["rev_kl"], -1e-5)
            if r["draft"] == "self":
                self.assertLess(r["tv"], 1e-5)
                self.assertEqual(r["greedy_agree"], 1.0)
        # same distribution as a per-record forward
        r0 = [r for r in recs if r["position"] == max(x["position"] for x in recs)][0]
        ids = tok.apply_chat_template([{"role": "user", "content": prompt["text"]}], tokenize=True) + \
            r0["response_prefix_token_ids"]
        with torch.no_grad():
            p = torch.softmax(t(input_ids=torch.tensor([ids])).logits[0, -1], -1)
            q = torch.softmax(d(input_ids=torch.tensor([ids])).logits[0, -1], -1)
        tv = 0.5 * float((p - q).abs().sum())
        got = [r for r in rows if r["draft"] == "other" and r["position"] == r0["position"]][0]["tv"]
        self.assertAlmostEqual(got, tv, places=4)

    def test_group_records_rejects_inconsistent_prefixes(self):
        a = {"bank": "b", "prompt_id": "p", "position": 3, "response_prefix_token_ids": [1, 2]}
        b = {"bank": "b", "prompt_id": "p", "position": 4, "response_prefix_token_ids": [1, 9, 3]}
        with self.assertRaises(AssertionError):
            pe.group_records([a, b])


@unittest.skipUnless(HAS, SKIP)
class TestMergeAndEval(unittest.TestCase):
    def test_merge_verifies_and_saves(self):
        with tempfile.TemporaryDirectory() as td:
            student = ds.make_lora_student(tiny_qwen(SEED))
            for n, p in student.named_parameters():
                if "lora_B" in n:
                    torch.nn.init.normal_(p, std=0.05)      # non-zero delta
            student.save_pretrained(str(Path(td) / "adapter"))
            batch = [[5, 6, 7, 8, 9], [10, 11, 12]]
            res = mg.merge_adapter(str(Path(td) / "adapter"), str(Path(td) / "merged"), "tiny", None, batch,
                                   "cpu", load_base_fn=lambda: tiny_qwen(SEED), dtype=torch.float32)
            self.assertLess(res["verify"]["max_abs_logit_diff"], 1e-4)
            self.assertEqual(res["verify"]["argmax_agreement"], 1.0)
            self.assertTrue((Path(td) / "merged" / "config.json").exists())
            base = tiny_qwen(SEED)
            with torch.no_grad():
                d = (base(input_ids=torch.tensor([batch[0]])).logits -
                     transformers.AutoModelForCausalLM.from_pretrained(str(Path(td) / "merged"))(
                         input_ids=torch.tensor([batch[0]])).logits).abs().max().item()
            self.assertGreater(d, 1e-4)                       # the adapter actually changed the model

    def test_cost_model(self):
        cm = ev.cost_model(0.1, 1.2, 4, alpha=0.8)
        a = cm["break_even_alpha"]
        self.assertTrue(cm["break_even_exists"])
        self.assertAlmostEqual(1 + sum(a ** j for j in range(1, 5)), 4 * 0.1 + 1.2, places=6)
        self.assertGreater(cm["predicted_speedup"], 1.0)
        self.assertIsNone(ev.cost_model(0.5, 2.0, 2)["break_even_alpha"])     # 1+1+1 = 3 is not > 3
        self.assertEqual(ev.cost_model(0.0, 0.5, 2)["break_even_alpha"], 0.0)

    def test_alpha_from_rounds(self):
        a = ev.alpha_from_rounds([2, 2, 2], 2)
        self.assertAlmostEqual(a, 1.0, places=6)
        a = ev.alpha_from_rounds([1, 1], 4)
        self.assertAlmostEqual(a + a ** 2 + a ** 3 + a ** 4, 1.0, places=6)

    def test_spec_condition_and_costs_tiny(self):
        tok = CharTokenizer()
        t, d = tiny_qwen(SEED), tiny_qwen(SEED + 1)
        prompts = [{"id": "GSM8K-0000", "domain": "math", "workload": "math", "text": "2+2?", "answer": "#### 4"},
                   {"id": "SHORT-familiar-0000", "domain": "short", "workload": "familiar", "text": "Code 7.",
                    "answer": "7"}]
        res = ev.run_spec_condition(t, d, tok, prompts, 2, "tiny", "cpu", n_reps=2, cap=6, n_fixed64=1,
                                    log=lambda *_: None)
        self.assertEqual(len(res["primary"]), 4)
        self.assertEqual(len(res["fixed64"]), 2)
        self.assertTrue(all(len(r["spec_tokens"]) == 64 for r in res["fixed64"]))
        self.assertEqual({r["spec_first"] for r in res["primary"]}, {True, False})   # counterbalanced
        for r in res["primary"]:
            self.assertLessEqual(r["spec_n_output_tokens"], 6)
            self.assertGreater(r["time_ratio_target_over_spec"], 0)
        rows = ev.isolated_costs({"t": t, "d": d}, "cpu", lengths=(8,), n_rep=1)
        self.assertEqual(len(rows), 2)
        self.assertIn("verify_5_tokens_ms", rows[0])
        acc = ev.standalone_accuracy(d, tok, prompts, "tiny", "cpu", n_reps=1)
        self.assertEqual(len(acc), 2)


class TestAggregateEmpty(unittest.TestCase):
    def test_missing_inputs_reported(self):
        if not HAS:
            self.skipTest(SKIP)
        from task3_specdec.aggregate import aggregate
        with tempfile.TemporaryDirectory() as td:
            st = aggregate(td)
            self.assertEqual({k: v for k, v in st.items()},
                             {"validation": "missing", "distill": "missing", "alignment": "missing",
                              "spec": "missing", "standalone": "missing"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
