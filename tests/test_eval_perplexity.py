"""
Unit test: perplexity evaluator (CPU, no model download).

Checks the half-open `score_target_offsets` convention and the pooled NLL/PPL
math against a hand-computed value, using a tiny dummy model that returns a
fixed all-zero logit tensor.

Record under test:
  token_ids = 10 ids, score_target_offsets = [1, 10] (half-open → positions
  1..9), scored_tokens = 9.  With all-zero logits over a vocab of size V, every
  log-softmax entry equals -log(V), so each scored token's NLL is log(V):
    N        = 9
    mean_nll = log(V)
    ppl      = V
"""

import json
import math
import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.eval_perplexity import evaluate_perplexity, _scored_range


VOCAB = 16


class _Output:
    def __init__(self, logits):
        self.logits = logits


class ConstantLogitModel(nn.Module):
    """Dummy CausalLM: returns fixed all-zero logits of shape (B, L, VOCAB)."""

    def __init__(self, vocab=VOCAB):
        super().__init__()
        self.vocab = vocab

    def forward(self, ids):
        B, L = ids.shape
        return _Output(torch.zeros(B, L, self.vocab, dtype=torch.float32))


class TestEvaluatePerplexity(unittest.TestCase):
    def _write_records(self, records):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        self.addCleanup(os.remove, path)
        return path

    def test_hand_computed_nll_and_N(self):
        rec = {
            "token_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # 10 ids
            "score_target_offsets": [1, 10],                   # half-open → 1..9
            "scored_tokens": 9,
        }
        path = self._write_records([rec])
        model = ConstantLogitModel()

        result = evaluate_perplexity(
            model, path, device="cpu", dtype=torch.bfloat16,
            expected_total_tokens=9,
        )

        self.assertEqual(result["N"], 9)
        self.assertAlmostEqual(result["mean_nll"], math.log(VOCAB), places=5)
        self.assertAlmostEqual(result["ppl"], float(VOCAB), places=3)
        self.assertEqual(len(result["per_window"]), 1)
        self.assertEqual(result["per_window"][0]["n"], 9)
        self.assertAlmostEqual(
            result["per_window"][0]["nll"], 9 * math.log(VOCAB), places=4
        )

    def test_expected_total_tokens_mismatch_raises(self):
        rec = {
            "token_ids": list(range(10)),
            "score_target_offsets": [1, 10],
            "scored_tokens": 9,
        }
        path = self._write_records([rec])
        model = ConstantLogitModel()
        with self.assertRaises(AssertionError):
            evaluate_perplexity(
                model, path, device="cpu", dtype=torch.bfloat16,
                expected_total_tokens=50000,
            )


class TestScoredRange(unittest.TestCase):
    def test_half_open_range_is_not_a_position_list(self):
        # [1, 10] must mean positions 1..9 (9 tokens), NOT the two points {1, 10}.
        start, end = _scored_range(
            {"score_target_offsets": [1, 10], "scored_tokens": 9}, L=10
        )
        self.assertEqual((start, end), (1, 10))
        self.assertEqual(len(range(start, end)), 9)

    def test_default_range_excludes_position_zero(self):
        start, end = _scored_range({}, L=10)
        self.assertEqual((start, end), (1, 10))

    def test_scored_tokens_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            _scored_range({"score_target_offsets": [1, 10], "scored_tokens": 8}, L=10)

    def test_start_zero_rejected(self):
        # Position 0 is context-only; a range starting at 0 is invalid.
        with self.assertRaises(ValueError):
            _scored_range({"score_target_offsets": [0, 10]}, L=10)

    def test_end_past_length_rejected(self):
        with self.assertRaises(ValueError):
            _scored_range({"score_target_offsets": [1, 11]}, L=10)

    def test_empty_or_reversed_range_rejected(self):
        with self.assertRaises(ValueError):
            _scored_range({"score_target_offsets": [5, 5]}, L=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
