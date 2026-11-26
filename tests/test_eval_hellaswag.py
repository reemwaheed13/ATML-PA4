"""
Unit test: HellaSwag token boundary and batched scoring (CPU, no downloads).

The fake tokenizer mimics Qwen's BPE behaviour that caused the original bug:
a space merges into the FOLLOWING word ("a b" -> ["a", " b"]), but a trailing
space with nothing after it is its own token ("a " -> ["a", " "]).  The old
boundary (len(tok(context + " "))) therefore skipped the first ending token.
"""

import gzip
import json
import math
import os
import re
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.eval_hellaswag import evaluate_hellaswag, score_example, _build_context

VOCAB = 64


class FakeBPETokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self.vocab = {}
        self.calls = []

    def _piece_id(self, piece):
        if piece not in self.vocab:
            self.vocab[piece] = 1 + (len(self.vocab) % (VOCAB - 1))
        return self.vocab[piece]

    def tokenize(self, text):
        return re.findall(r" ?[^ ]+| +", text)

    def __call__(self, text, add_special_tokens=False, **kw):
        self.calls.append(text)
        return {"input_ids": [self._piece_id(p) for p in self.tokenize(text)]}


class _Out:
    def __init__(self, logits):
        self.logits = logits


class NextTokenModel(nn.Module):
    """Causal toy LM: logits at position t depend only on token t.

    logits[b, t, v] = 3.0 * cos(v * (ids[b, t] + 1)), so every position has a
    distinct, non-uniform distribution; padding to the right cannot affect earlier
    positions.  Records every call's inputs.
    """

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids=None, attention_mask=None):
        self.calls.append((input_ids.clone(), None if attention_mask is None else attention_mask.clone()))
        v = torch.arange(VOCAB, dtype=torch.float32)
        logits = 3.0 * torch.cos(v.view(1, 1, -1) * (input_ids.float().unsqueeze(-1) + 1))
        return _Out(logits)


def reference_loglik(model, tok, context, ending):
    """Unbatched reference: score every token of tok(' ' + ending) after tok(context)."""
    ctx = tok(context)["input_ids"]
    cont = tok(" " + ending)["input_ids"]
    ids = torch.tensor([ctx + cont])
    lp = torch.log_softmax(model(input_ids=ids).logits[0], dim=-1)
    return sum(lp[len(ctx) + k - 1, cont[k]].item() for k in range(len(cont))), len(cont)


EXAMPLE = {
    "activity_label": "Cooking",
    "ctx": "A man cracks eggs into a bowl. He",
    "ctx_a": "A man cracks eggs into a bowl.",
    "ctx_b": "he",
    "endings": [
        "whisks them quickly.",
        "throws the bowl out the window and leaves",
        "eats.",
        "puts the bowl in the oven for an hour.",
    ],
    "label": "0",
}


class TestHellaSwagBoundary(unittest.TestCase):
    def test_fake_tokenizer_reproduces_bug_condition(self):
        tok = FakeBPETokenizer()
        ctx = _build_context(EXAMPLE)
        # Trailing space is its own token in isolation ...
        self.assertEqual(tok.tokenize(ctx + " ")[-1], " ")
        # ... but merges into the ending's first token in the full string.
        full = tok.tokenize(ctx + " " + EXAMPLE["endings"][0])
        self.assertEqual(full[len(tok.tokenize(ctx))], " whisks")
        # So the old boundary skipped exactly that first ending token.
        old_ctx_len = len(tok.tokenize(ctx + " "))
        self.assertEqual(old_ctx_len, len(tok.tokenize(ctx)) + 1)

    def test_every_continuation_token_scored_once(self):
        tok = FakeBPETokenizer()
        model = NextTokenModel()
        ctx = _build_context(EXAMPLE)
        raw, norm, n_cont = score_example(model, tok, ctx, EXAMPLE["endings"], "cpu", pad_id=0)

        # One batched forward for all 4 endings, right-padded with a mask.
        self.assertEqual(len(model.calls), 1)
        ids, mask = model.calls[0]
        self.assertEqual(ids.shape[0], 4)
        self.assertIsNotNone(mask)

        for k, ending in enumerate(EXAMPLE["endings"]):
            expected_n = len(tok.tokenize(" " + ending))
            self.assertEqual(n_cont[k], expected_n)
            self.assertEqual(n_cont[k], len(ending.split(" ")))  # one token per word
            self.assertEqual(int(mask[k].sum()), len(tok.tokenize(ctx)) + expected_n)
            ref_ll, ref_n = reference_loglik(NextTokenModel(), tok, ctx, ending)
            self.assertEqual(ref_n, n_cont[k])
            self.assertAlmostEqual(raw[k], ref_ll, places=4)
            self.assertAlmostEqual(norm[k], ref_ll / len(ending), places=6)

    def test_first_ending_token_contributes(self):
        """Changing only the first word of an ending must change its score."""
        tok = FakeBPETokenizer()
        model = NextTokenModel()
        ctx = _build_context(EXAMPLE)
        raw, _, _ = score_example(model, tok, ctx, ["whisks them.", "stirs them."], "cpu")
        self.assertNotAlmostEqual(raw[0], raw[1], places=6)

    def test_evaluate_file_and_subset(self):
        tok = FakeBPETokenizer()
        model = NextTokenModel()
        fd, path = tempfile.mkstemp(suffix=".jsonl.gz")
        os.close(fd)
        self.addCleanup(os.remove, path)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for order in range(5):
                fh.write(json.dumps({"order": order, "example": EXAMPLE, "source_row_index": 100 + order}) + "\n")
        res = evaluate_hellaswag(model, tok, path, "cpu", subset_ids=range(3))
        self.assertEqual(res["n_examples"], 3)
        self.assertEqual([r["order"] for r in res["per_example"]], [0, 1, 2])
        r0 = res["per_example"][0]
        self.assertEqual(len(r0["n_cont_tokens"]), 4)
        self.assertEqual(r0["norm_pred"], max(range(4), key=lambda k: r0["norm_scores"][k]))
        self.assertIn("format", res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
