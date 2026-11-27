"""
CPU tests for task2_attention/kv_cache.py and part_c.py (PDF §2 Part C).

No downloads: the real-model paths run on a tiny random Qwen2ForCausalLM
(Qwen2Config, 2 layers, GQA 2 query heads / 1 KV head, head_dim 64) with a
character-level fake tokenizer that implements the chat template and offset mapping.
Seed 6304.

Covered: handout byte formula M_b(L); FP16-offset group quantisation (incl. the
all-positive-group regression); bit packing; packed bytes == M_b(L) for both INT4
layouts; decode-time requantisation of completed groups only; eviction / hybrid
selection within budget (hybrid maximal by brute force); hybrid encode/decode;
C1 split-prefill equivalence, 100%-retention greedy equivalence, scoring attention
vs SDPA and cache unchanged after scoring; an end-to-end eval_prompt on all 6
methods; the packed encode/decode benchmark.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.byte_accounting import kv_cache_bytes_int  # noqa: E402

try:
    from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM
    from task2_attention import kv_cache as kvc
    from task2_attention import part_c
    HAS_TF, TF_SKIP = True, ""
except Exception as exc:  # pragma: no cover - env dependent
    HAS_TF, TF_SKIP = False, f"transformers not importable: {exc}"

SEED = 6304


class TestByteFormula(unittest.TestCase):
    """M_b(L) = 2*n_l*H_kv*d_h * [ floor(L/64)*(64*b/8 + 4) + 2*(L mod 64) ]."""

    def test_matches_handout_values(self):
        cases = {2048: 16_515_072, 2100: 18_006_016, 4096: 33_030_144}
        for L, expected in cases.items():
            self.assertEqual(kv_cache_bytes_int(L, 28, 2, 128, bits=4, group_size=64), expected)


def tiny_model(vocab=300):
    torch.manual_seed(SEED)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
                      tie_word_embeddings=False, attn_implementation="sdpa")
    return Qwen2ForCausalLM(cfg).eval()


def rand_cache(n_layers=2, B=1, H=2, L=150, d=128, shift=0.0):
    g = torch.Generator().manual_seed(SEED)
    return kvc.cache_from_tensors([(torch.randn(B, H, L, d, generator=g) + shift,
                                    torch.randn(B, H, L, d, generator=g)) for _ in range(n_layers)])


@unittest.skipUnless(HAS_TF, TF_SKIP)
class TestQuantAndPacking(unittest.TestCase):
    def test_roundtrip_error_bound(self):
        g = torch.Generator().manual_seed(SEED)
        x = torch.randn(4, 3, 128, generator=g)
        for bits in (2, 4, 8):
            q, s, o = kvc.quantize_groups(x, bits)
            xh = kvc.dequantize_groups(q, s, o)
            bound = s.float().repeat_interleave(64, -1) / 2 + 1e-3 * x.abs().max()
            self.assertTrue(torch.all((x - xh).abs() <= bound), bits)

    def test_all_positive_group_regression(self):
        # an integer zero-point clamped to [0, 15] collapses this group; the FP16 offset does not
        x = torch.linspace(5.0, 6.0, 64).view(1, 64)
        q, s, o = kvc.quantize_groups(x, 4)
        err = (kvc.dequantize_groups(q, s, o) - x).abs().max().item()
        self.assertLess(err, (1.0 / 15) / 2 + 1e-2)

    def test_constant_group(self):
        x = torch.full((2, 64), 3.0)
        q, s, o = kvc.quantize_groups(x, 4)
        self.assertTrue(torch.equal(kvc.dequantize_groups(q, s, o), x))

    def test_pack_unpack(self):
        g = torch.Generator().manual_seed(SEED)
        for bits in (2, 4, 8):
            q = torch.randint(0, 2 ** bits, (3, 128), generator=g).to(torch.uint8)
            p = kvc.pack_bits(q, bits)
            self.assertEqual(p.shape[-1], 128 * bits // 8)
            self.assertTrue(torch.equal(kvc.unpack_bits(p, bits), q))

    def test_packed_bytes_equal_formula_both_layouts(self):
        for L in (63, 64, 150, 256):
            c = rand_cache(L=L)
            for layout in ("tokenwise", "channelwise"):
                for bits in (2, 4):
                    p = kvc.encode_cache(c, layout, bits)
                    self.assertEqual(kvc.packed_nbytes(p), kv_cache_bytes_int(L, 2, 2, 128, bits=bits), (L, layout))
                    d = kvc.decode_cache(p)
                    self.assertEqual(kvc._seq_len(d), L)
                    tail = L % 64
                    if tail:   # trailing tokens stay exact (FP16)
                        Kt = kvc._layer_keys(d, 0)[:, :, L - tail:].float()
                        self.assertLess((Kt - kvc._layer_keys(c, 0)[:, :, L - tail:]).abs().max().item(), 1e-2)

    def test_channelwise_lower_key_error_on_channel_outliers(self):
        c = rand_cache(L=128)
        K = kvc._layer_keys(c, 0)
        K[..., 3] += 20.0   # an outlier channel, as in real keys
        errs = {}
        for layout in ("tokenwise", "channelwise"):
            d = kvc.decode_cache(kvc.encode_cache(c, layout, 4))
            errs[layout] = kvc.reconstruction_error(c, d)["K_rel_fro"]
        self.assertLess(errs["channelwise"], errs["tokenwise"])

    def test_requantize_completed_groups_only(self):
        c = rand_cache(L=100)
        upto = kvc.channelwise_split_lengths(100)[0]
        self.assertEqual(upto, 64)
        before = kvc._layer_keys(c, 0).clone()
        # append 20 tokens -> live length 120: no new complete group
        for li in range(2):
            c.update(torch.randn(1, 2, 20, 128), torch.randn(1, 2, 20, 128), li)
        upto = kvc.requantize_completed_groups(c, "channelwise", 4, upto)
        self.assertEqual(upto, 64)
        self.assertTrue(torch.equal(kvc._layer_keys(c, 0)[:, :, 64:100], before[:, :, 64:100]))
        # 10 more -> 130: group [64,128) completes and is quantised, [128,130) stays FP16
        for li in range(2):
            c.update(torch.randn(1, 2, 10, 128), torch.randn(1, 2, 10, 128), li)
        tail_before = kvc._layer_keys(c, 0)[:, :, 128:].clone()
        upto = kvc.requantize_completed_groups(c, "channelwise", 4, upto)
        self.assertEqual(upto, 128)
        self.assertFalse(torch.equal(kvc._layer_keys(c, 0)[:, :, 64:100], before[:, :, 64:100]))
        self.assertTrue(torch.equal(kvc._layer_keys(c, 0)[:, :, 128:], tail_before))


@unittest.skipUnless(HAS_TF, TF_SKIP)
class TestSelection(unittest.TestCase):
    dims = (28, 2, 128)

    def test_eviction_budget(self):
        L = 2100
        budget = kvc.int_bytes(L, *self.dims, 4)
        n = kvc.max_eviction_positions(budget, L, *self.dims)
        self.assertLessEqual(kvc.eviction_bytes(n, *self.dims), budget)
        self.assertGreater(kvc.eviction_bytes(n + 1, *self.dims), budget)
        pos = kvc.select_sink_recent(L, n)
        self.assertEqual(len(pos), n)
        self.assertTrue(set(range(8)) <= set(pos) and set(range(L - 64, L)) <= set(pos))
        self.assertEqual(pos[8], L - (n - 8))
        imp = torch.zeros(L)
        imp[1000:1010] = 5.0
        qa = kvc.select_question_aware(L, n, imp)
        self.assertEqual(len(qa), n)
        self.assertTrue(set(range(1000, 1010)) <= set(qa))

    def test_hybrid_within_budget_and_maximal(self):
        for L in (2048, 2100, 4170):
            budget = kvc.int_bytes(L, *self.dims, 4)
            imp = torch.arange(L, dtype=torch.float32).flip(0)
            fp16, hb = kvc.select_hybrid(L, imp, budget, *self.dims)
            self.assertLessEqual(hb["total"], budget)
            self.assertEqual(len(fp16), hb["n_fp16"])
            for n in range(hb["n_fp16"] + 1, L + 1):     # brute force: nothing larger fits
                self.assertGreater(kvc.hybrid_bytes(L, n, *self.dims)["total"], budget)
            self.assertTrue(set(range(8)) <= set(fp16))

    def test_hybrid_encode_decode_bytes_and_exact_fp16(self):
        c = rand_cache(L=300)
        L = 300
        budget = kvc.int_bytes(L, 2, 2, 128, 4)
        imp = torch.rand(L, generator=torch.Generator().manual_seed(SEED))
        fp16, hb = kvc.select_hybrid(L, imp, budget, 2, 2, 128)
        p = kvc.encode_hybrid(c, fp16)
        self.assertEqual(p.nbytes, hb["total"])
        d = kvc.decode_hybrid(p, dtype=torch.float32)
        idx = torch.tensor(fp16)
        self.assertLess((kvc._layer_keys(d, 1)[:, :, idx] - kvc._layer_keys(c, 1)[:, :, idx]).abs().max().item(), 1e-2)

    def test_evict_keeps_positions_in_order(self):
        c = rand_cache(L=16, d=64)
        keep = [0, 2, 3, 7, 15]
        e = kvc._evict_to_positions(c, keep)
        self.assertTrue(torch.equal(kvc._layer_values(e, 1), kvc._layer_values(c, 1)[:, :, keep]))
        self.assertEqual(kvc._seq_len(c), 16)


# ── tiny real-model paths ─────────────────────────────────────────────────────

class CharTokenizer:
    """Character-level stand-in for the Qwen tokenizer: chat template + offsets."""
    eos_token_id = 1
    pad_token_id = 1

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self(text)["input_ids"] if tokenize else text

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False, return_tensors=None):
        ids = [ord(ch) % 250 + 2 for ch in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(chr(i - 2) if 48 <= i - 2 < 58 else "x" for i in ids)


def fake_case():
    lines = [f"Ledger line {i:04d}: market shipped {i * 7 % 900} parcels." for i in range(6)]
    ev = "IMPORTANT RECORD: access code for CASE-1 is 26836."
    lines2 = [f"Ledger line {i:04d}: station shipped {i * 3 % 900} parcels." for i in range(6, 12)]
    body = "\n".join(lines) + "\n" + ev + "\n" + "\n".join(lines2) + "\n"
    user = body + "QUESTION_BEGIN\nWhat is the access code for CASE-1? Reply with only five digits."
    s = user.index(ev)
    return {"id": "KV-test-start-0-00", "user_text": user, "code": "26836", "position": "start",
            "distractors": 0, "target_tokens": 2048, "evidence_char_start": s, "evidence_char_end": s + len(ev)}


@unittest.skipUnless(HAS_TF, TF_SKIP)
class TestTinyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.tok = CharTokenizer()

    def test_build_prompt_split(self):
        P = part_c.build_prompt(self.tok, fake_case())
        text = self.tok.apply_chat_template([{"role": "user", "content": fake_case()["user_text"]}])
        self.assertEqual(text[P["context_len"]:P["context_len"] + len("QUESTION_BEGIN")], "QUESTION_BEGIN")
        self.assertTrue(text.endswith("assistant\n"))
        self.assertEqual(len(P["evidence_tokens"]), len("IMPORTANT RECORD: access code for CASE-1 is 26836."))

    def test_split_prefill_equals_full_forward(self):
        torch.manual_seed(SEED)
        ids = torch.randint(2, 300, (1, 90))
        with torch.no_grad():
            full = self.model(ids).logits
            c = DynamicCache()
            a = kvc.forward_with_positions(self.model, c, ids[:, :37], 0).logits
            b = kvc.forward_with_positions(self.model, c, ids[:, 37:60], 37).logits
            rest = [kvc.forward_with_positions(self.model, c, ids[:, t:t + 1], t).logits for t in range(60, 90)]
        split = torch.cat([a, b] + rest, dim=1)
        self.assertLess((split - full).abs().max().item(), 1e-4)
        self.assertTrue(torch.isfinite(split).all())

    def test_full_retention_greedy_equals_generate(self):
        torch.manual_seed(SEED)
        ids = torch.randint(2, 300, (1, 40))
        with torch.no_grad():
            ref = self.model.generate(ids, max_new_tokens=12, do_sample=False, num_beams=1,
                                      pad_token_id=1, eos_token_id=[299])[0, 40:].tolist()
            c = DynamicCache()
            kvc.forward_with_positions(self.model, c, ids[:, :39], 0)
            c = kvc._evict_to_positions(c, list(range(39)))
            got, _ = kvc.greedy_generate_edited(self.model, c, ids[:, 39:], 39, 12, eos_ids=[299])
        self.assertEqual(got[0].tolist(), ref)

    def test_eviction_keeps_absolute_positions(self):
        """After evicting middle positions, appended tokens use positions continuing from the
        ORIGINAL length: retained-only attention through the edited cache must equal a full
        forward whose evicted keys are masked out."""
        torch.manual_seed(SEED)
        ids = torch.randint(2, 300, (1, 50))
        keep = list(range(10)) + list(range(30, 45))
        with torch.no_grad():
            c = DynamicCache()
            kvc.forward_with_positions(self.model, c, ids[:, :45], 0)
            e = kvc._evict_to_positions(c, keep)
            out_e = kvc.forward_with_positions(self.model, e, ids[:, 45:], 45).logits
            # reference: the SAME full prefill cache, evicted keys masked out by HF's own
            # 2D attention mask, suffix at its true absolute positions 45..49
            mask = torch.zeros(1, 50, dtype=torch.long)
            mask[0, keep] = 1
            mask[0, 45:] = 1
            out_m = self.model(ids[:, 45:], past_key_values=c, attention_mask=mask,
                               position_ids=torch.arange(45, 50).unsqueeze(0), use_cache=True).logits
        self.assertLess((out_e - out_m).abs().max().item(), 1e-4)

    def test_question_importance_matches_sdpa_and_restores_cache(self):
        torch.manual_seed(SEED)
        ids = torch.randint(2, 300, (1, 80))
        with torch.no_grad():
            c = DynamicCache()
            kvc.forward_with_positions(self.model, c, ids[:, :70], 0)
            k0 = kvc._layer_keys(c, 1).clone()
            imp, info = kvc.question_importance(self.model, c, ids[:, 70:], 70, validate=True)
        self.assertEqual(kvc._seq_len(c), 70)
        self.assertTrue(torch.equal(kvc._layer_keys(c, 1), k0))
        self.assertEqual(tuple(imp.shape), (70,))
        self.assertLess(max(info["manual_vs_sdpa_attention_max_abs_err_per_layer"]), 1e-4)
        # total attention mass on context <= (#layers * #heads * #question tokens)
        self.assertLessEqual(imp.sum().item(), 2 * 2 * 10 + 1e-3)

    def test_eval_prompt_all_methods(self):
        r = part_c.eval_prompt(self.model, self.tok, fake_case(), "cpu", max_new_tokens=4)
        self.assertEqual(list(r["methods"]), part_c.METHODS)
        budget = r["budget_bytes_int4_channelwise"]
        for m, rec in r["methods"].items():
            if m != "fp16":
                self.assertLessEqual(rec["bytes"], budget, m)
            self.assertTrue(rec["post_prefill_kl"] >= -1e-6)
            self.assertIn(rec["exact_match"], (0, 1))
        self.assertEqual(r["methods"]["int4_channelwise"]["bytes"], budget)
        self.assertEqual(r["methods"]["int4_tokenwise"]["bytes"], budget)
        self.assertEqual(r["methods"]["fp16"]["post_prefill_kl"], 0.0)
        for m in ("sink_recent", "question_aware", "hybrid"):
            self.assertTrue(0.0 <= r["methods"][m]["evidence_retained_frac"] <= 1.0)
        self.assertGreater(r["methods"]["fp16"]["bytes"], budget)

    def test_c1_validate(self):
        res = part_c.c1_validate(self.model, self.tok, [fake_case()], "cpu", max_new_tokens=4)
        c = res["cases"][0]
        self.assertTrue(c["all_logits_finite"])
        self.assertTrue(c["full_retention_matches_greedy"])
        self.assertTrue(c["scoring_cache_unchanged"])
        self.assertLess(c["split_vs_full_max_abs_logit_diff"], 1e-3)

    def test_packed_bench_cpu(self):
        rows = part_c.bench_packed(self.model, "cpu", batches=(1, 2), lengths=(128,), n_warmup=0, n_rep=1,
                                   log=lambda *_: None)
        self.assertEqual(len(rows), 2 * 3)
        self.assertTrue(all(r["bytes_match_formula"] for r in rows))

    def test_extract_code(self):
        self.assertEqual(part_c.extract_code("The code is 26836."), "26836")
        self.assertIsNone(part_c.extract_code("1234 or 123456"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
