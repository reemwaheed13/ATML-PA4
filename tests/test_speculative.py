"""
Exact speculative sampling tests (PDF §3 step 1), CPU, seed 6304.

χ²: the SAME pure decision function `verify_round` that the real decoder calls is driven
on a 3-symbol Markov target/draft; 100,000 length-3 sequences at γ=2 and γ=4; every
expected count >= 10; the correct sampler must NOT be rejected at p < 0.01 and the
negative control (residual replaced by p) MUST be rejected.

Deterministic: one-token round, full acceptance with bonus, final-token truncation,
EOS mid-proposal, rejection -> residual support, split-prefill equivalence, cache
invariant / verification logits vs a full forward, left-padded batch prefill.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from task3_specdec import speculative as sp  # noqa: E402

try:
    import transformers  # noqa: F401
    from tests.tiny import VOCAB, tiny_qwen
    from task3_specdec import generation as gen
    HAS_TF, SKIP = True, ""
except Exception as exc:  # pragma: no cover
    HAS_TF, SKIP = False, f"transformers unavailable: {exc}"

SEED = 6304


class TestChi2(unittest.TestCase):
    def test_toy_joint_is_a_distribution(self):
        toy = sp.MarkovToy()
        import itertools
        tot = sum(toy.joint(s) for s in itertools.product(range(3), repeat=3))
        self.assertAlmostEqual(tot, 1.0, places=12)
        self.assertGreaterEqual(min(toy.joint(s) for s in itertools.product(range(3), repeat=3)) * 100_000, 10)

    def test_correct_sampler_not_rejected_gamma2_gamma4(self):
        for g in (2, 4):
            r = sp.chi2_test(g, n_sequences=100_000, seed=SEED, correct=True)
            self.assertGreaterEqual(r["min_expected"], 10)
            self.assertEqual(r["dof"], 26)
            self.assertFalse(r["rejected_at_0.01"], f"gamma={g}: p={r['p_value']:.4g} chi2={r['chi2']:.2f}")

    def test_negative_control_rejected_gamma2_gamma4(self):
        for g in (2, 4):
            r = sp.chi2_test(g, n_sequences=100_000, seed=SEED, correct=False)
            self.assertTrue(r["rejected_at_0.01"], f"gamma={g}: p={r['p_value']:.4g}")


class TestVerifyRound(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)

    def test_zero_proposals_is_a_target_sample(self):
        p = torch.tensor([[0.0, 1.0, 0.0]])
        r = sp.verify_round(p, p[:0], [])
        self.assertEqual((r.tokens, r.n_accepted, r.n_examined, r.bonus), ([1], 0, 0, True))

    def test_full_acceptance_adds_bonus(self):
        q = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
        p = torch.cat([q, torch.tensor([[0.0, 0.0, 1.0]])])         # p == q on proposals -> always accept
        for _ in range(50):
            r = sp.verify_round(p, q, [1, 2])
            self.assertEqual(r.tokens, [1, 2, 2])
            self.assertTrue(r.bonus and not r.rejected and r.n_accepted == 2)

    def test_zero_target_mass_always_rejected_and_residual_avoids_it(self):
        q = torch.tensor([[0.5, 0.5, 0.0]])
        p = torch.tensor([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0]])
        for _ in range(200):
            r = sp.verify_round(p, q, [0])
            self.assertTrue(r.rejected)
            self.assertEqual(r.n_examined, 1)
            self.assertEqual(r.tokens, [2])            # residual = max(p - q, 0) = [0, 0, 0.5]

    def test_eos_mid_proposal_stops_without_bonus(self):
        q = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        p = torch.cat([q, torch.tensor([[1.0, 0.0, 0.0]])])
        r = sp.verify_round(p, q, [1, 2], eos_ids={2})
        self.assertEqual(r.tokens, [1, 2])
        self.assertTrue(r.eos and not r.bonus)
        r = sp.verify_round(p, q, [1, 2], eos_ids={1})
        self.assertEqual(r.tokens, [1])

    def test_proposal_length(self):
        self.assertEqual(sp.proposal_length(4, 1), 0)
        self.assertEqual(sp.proposal_length(4, 3), 2)
        self.assertEqual(sp.proposal_length(4, 64), 4)


@unittest.skipUnless(HAS_TF, SKIP)
class TestRealDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = tiny_qwen(SEED)
        cls.draft = tiny_qwen(SEED + 1)
        g = torch.Generator().manual_seed(SEED)
        cls.prompt = torch.randint(2, VOCAB, (23,), generator=g).tolist()

    def test_one_token_round(self):
        torch.manual_seed(SEED)
        st = sp.speculative_generate(self.target, self.draft, self.prompt, 1, 4, frozenset(), "cpu")
        self.assertEqual(len(st.tokens), 1)
        self.assertEqual((st.n_rounds, st.n_draft_forwards, st.n_proposed), (1, 0, 0))
        self.assertTrue(st.truncated)

    def test_full_acceptance_bonus_and_truncation(self):
        torch.manual_seed(SEED)
        st = sp.speculative_generate(self.target, self.target, self.prompt, 7, 4, frozenset(), "cpu",
                                     debug_check=True)
        # draft == target: round 1 proposes 4 (+bonus) = 5 tokens; round 2 has 2 left -> 1 proposal + bonus
        self.assertEqual(len(st.tokens), 7)
        self.assertEqual(st.n_rounds, 2)
        self.assertEqual(st.n_proposed, 5)
        self.assertEqual(st.n_accepted, 5)
        self.assertEqual(st.n_bonus, 2)
        self.assertLess(st.debug_max_logit_err, 1e-4)

    def test_eos_stops_generation(self):
        torch.manual_seed(SEED)
        st = sp.speculative_generate(self.target, self.draft, self.prompt, 20, 4, frozenset(range(VOCAB)), "cpu")
        self.assertEqual(len(st.tokens), 1)
        self.assertTrue(st.eos and not st.truncated)

    def test_cache_invariant_and_verification_logits(self):
        for g in (2, 4):
            torch.manual_seed(SEED + g)
            st = sp.speculative_generate(self.target, self.draft, self.prompt, 30, g, frozenset(), "cpu",
                                         debug_check=True)
            self.assertEqual(len(st.tokens), 30)
            self.assertLess(st.debug_max_logit_err, 1e-4)
            self.assertEqual(st.n_accepted + st.n_rounds, 30)      # each round emits accepted + 1
            self.assertLessEqual(st.n_accepted, st.n_examined)
            self.assertLessEqual(st.n_examined, st.n_proposed)

    def test_prompt_of_length_one(self):
        torch.manual_seed(SEED)
        st = sp.speculative_generate(self.target, self.draft, self.prompt[:1], 6, 2, frozenset(), "cpu",
                                     debug_check=True)
        self.assertEqual(len(st.tokens), 6)

    def test_split_prefill_equivalence(self):
        one = sp.prefill(self.target, self.prompt, "cpu")
        chunked = sp.prefill(self.target, self.prompt, "cpu", chunk=5)
        self.assertEqual(one.get_seq_length(), len(self.prompt) - 1)
        self.assertEqual(chunked.get_seq_length(), len(self.prompt) - 1)
        for i in range(len(one.layers)):
            self.assertLess((one.layers[i].keys - chunked.layers[i].keys).abs().max().item(), 1e-5)
            self.assertLess((one.layers[i].values - chunked.layers[i].values).abs().max().item(), 1e-5)
        with torch.no_grad():
            a = sp.forward_tokens(self.target, one, self.prompt[-1:], "cpu")
            b = sp.forward_tokens(self.target, chunked, self.prompt[-1:], "cpu")
            full = self.target(input_ids=torch.tensor([self.prompt])).logits[0, -1]
        self.assertLess((a - b).abs().max().item(), 1e-4)
        self.assertLess((a[-1] - full).abs().max().item(), 1e-4)

    def test_target_only_reference(self):
        torch.manual_seed(SEED)
        st = sp.target_only_generate(self.target, self.prompt, 9, frozenset(), "cpu")
        self.assertEqual(len(st.tokens), 9)
        self.assertEqual(st.n_rounds, 9)

    def test_left_padded_batch_prefill_matches_unpadded(self):
        prompts = [self.prompt, self.prompt[:9], self.prompt[3:17]]
        with torch.no_grad():
            _, logits, _, _ = gen.padded_prefill(self.target, prompts, "cpu", pad_id=0)
            for i, p in enumerate(prompts):
                ref = self.target(input_ids=torch.tensor([p])).logits[0, -1]
                self.assertLess((logits[i] - ref).abs().max().item(), 1e-4, i)
        torch.manual_seed(SEED)
        outs = gen.sample_batch(self.target, prompts, 5, eos_ids=frozenset({7}), device="cpu", pad_id=0)
        for o in outs:
            self.assertTrue(1 <= len(o["tokens"]) <= 5)
            self.assertEqual(o["termination"] == "eos", o["tokens"][-1] == 7)
            self.assertNotIn(7, o["tokens"][:-1])


class TestScoring(unittest.TestCase):
    def test_math_extraction(self):
        if not HAS_TF:
            self.skipTest(SKIP)
        self.assertEqual(gen.extract_math("so 3 + 4 = 7. The answer is \\boxed{1,234}"), 1234.0)
        self.assertEqual(gen.extract_math("She has 12 apples, then 15."), 15.0)
        self.assertIsNone(gen.extract_math("no digits here"))
        self.assertEqual(gen.gsm8k_gold("blah\n#### 42"), 42.0)

    def test_short_scoring(self):
        if not HAS_TF:
            self.skipTest(SKIP)
        p = {"domain": "short", "answer": "May"}
        s = gen.score_response(p, "May", "eos")
        self.assertEqual((s["correct"], s["correct_strict"], s["format_compliant"]), (1, 1, 1))
        s = gen.score_response(p, "The month is May.", "eos")
        self.assertEqual((s["correct"], s["correct_strict"], s["format_compliant"]), (1, 0, 0))
        s = gen.score_response({"domain": "short", "answer": "DE"}, "DE", "cap")
        self.assertEqual((s["correct"], s["format_compliant"], s["truncated"]), (1, 1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
