"""
Unit tests for task2_attention/kv_cache.py  (PDF §2 Part C).

CPU-runnable tests (no GPU, tiny random tensors):
  * group quantize/dequantize round-trip error bounds (token-wise and channel-wise),
  * byte formula M_b(L) matches the handout for L in {2048, 2100, 4096} with
    n_l=28, H_kv=2, d_h=128 (constants used ONLY in the test; the real path reads
    these from the model config),
  * channel-wise split keeps complete 64-token groups and an FP16 tail,
  * eviction keeps exactly the requested positions, in order.

The byte-formula test needs only `common.byte_accounting` and runs anywhere.  The
quantisation / eviction tests import `task2_attention.kv_cache`, which imports
`transformers` (DynamicCache); they are skipped if transformers is not installed
locally and are intended to run on Colab CPU.

GPU tests for Colab are DEFINED but NOT run here (see TestKVCacheGPU, skipped):
  * full forward logits vs split prefill+decode,
  * 100%-retention via the eviction path reproduces greedy generation exactly.
"""

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.byte_accounting import kv_cache_bytes_int

try:
    import transformers  # noqa: F401
    from transformers import DynamicCache
    from task2_attention import kv_cache as kvc
    HAS_TF = True
    TF_SKIP = ""
except Exception as exc:  # pragma: no cover - env dependent
    HAS_TF = False
    TF_SKIP = f"transformers / kv_cache not importable: {exc}"


# ── Byte accounting: M_b(L) (no transformers needed) ───────────────────────────

class TestByteFormula(unittest.TestCase):
    """M_b(L) = 2*n_l*H_kv*d_h * [ floor(L/64)*(64*b/8 + 4) + 2*(L mod 64) ]."""

    N_L, H_KV, D_H, BITS, GS = 28, 2, 128, 4, 64

    def _expected(self, L):
        n_channels = 2 * self.N_L * self.H_KV * self.D_H
        per_channel = (L // self.GS) * (self.GS * self.BITS // 8 + 4) + 2 * (L % self.GS)
        return n_channels * per_channel

    def test_matches_handout_values(self):
        # Hand-computed: n_channels = 2*28*2*128 = 14336.
        cases = {
            2048: 16_515_072,   # 32 groups * 36 B/group, no tail
            2100: 18_006_016,   # 32 groups * 36 + 2*52 tail
            4096: 33_030_144,   # 64 groups * 36, no tail
        }
        for L, expected in cases.items():
            self.assertEqual(self._expected(L), expected, f"self-check failed at L={L}")
            got = kv_cache_bytes_int(
                L, self.N_L, self.H_KV, self.D_H, bits=self.BITS, group_size=self.GS
            )
            self.assertEqual(got, expected, f"kv_cache_bytes_int wrong at L={L}")


@unittest.skipUnless(HAS_TF, TF_SKIP)
class TestChannelwiseSplit(unittest.TestCase):
    def test_split_keeps_complete_groups_and_fp16_tail(self):
        self.assertEqual(kvc.channelwise_split_lengths(2048), (2048, 0))
        self.assertEqual(kvc.channelwise_split_lengths(2100), (2048, 52))
        self.assertEqual(kvc.channelwise_split_lengths(4096), (4096, 0))
        self.assertEqual(kvc.channelwise_split_lengths(63), (0, 63))
        self.assertEqual(kvc.channelwise_split_lengths(64), (64, 0))
        self.assertEqual(kvc.channelwise_split_lengths(65), (64, 1))


# ── Quantisation round-trip ────────────────────────────────────────────────────

@unittest.skipUnless(HAS_TF, TF_SKIP)
class TestQuantRoundTrip(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _group_scale_bound(self, x, bits, dim):
        # Max per-element reconstruction error for in-range asymmetric quant is s/2,
        # where s = (max - min) / (2^bits - 1) is the group scale.
        _, scales, _, _ = kvc._quantize_groups(x, bits, kvc.GROUP_SIZE, dim=dim)
        return float(scales.max().item()) / 2.0

    def test_tokenwise_roundtrip_error_bound(self):
        x = torch.randn(4, 3, 2 * kvc.GROUP_SIZE)   # group along last dim (feature)
        q, s, z, _ = kvc._quantize_groups(x, 4, kvc.GROUP_SIZE, dim=-1)
        x_hat = kvc._dequantize_groups(q, s, z, 4, kvc.GROUP_SIZE, dim=-1)
        bound = self._group_scale_bound(x, 4, dim=-1)
        self.assertLessEqual((x - x_hat).abs().max().item(), bound * (1 + 1e-3) + 1e-6)

    def test_channelwise_roundtrip_error_bound(self):
        # (B, H_kv, L, d_h): group K along the token axis (dim=2).
        x = torch.randn(1, 2, 2 * kvc.GROUP_SIZE, 4)
        q, s, z, _ = kvc._quantize_groups(x, 4, kvc.GROUP_SIZE, dim=2)
        x_hat = kvc._dequantize_groups(q, s, z, 4, kvc.GROUP_SIZE, dim=2)
        bound = self._group_scale_bound(x, 4, dim=2)
        self.assertLessEqual((x - x_hat).abs().max().item(), bound * (1 + 1e-3) + 1e-6)

    def test_more_bits_lower_error(self):
        x = torch.randn(4, 2 * kvc.GROUP_SIZE)
        err = {}
        for bits in (2, 4, 8):
            q, s, z, _ = kvc._quantize_groups(x, bits, kvc.GROUP_SIZE, dim=-1)
            x_hat = kvc._dequantize_groups(q, s, z, bits, kvc.GROUP_SIZE, dim=-1)
            err[bits] = (x - x_hat).abs().max().item()
        self.assertGreater(err[2], err[4])
        self.assertGreater(err[4], err[8])


# ── Eviction keeps exactly the requested positions, in order ──────────────────

@unittest.skipUnless(HAS_TF, TF_SKIP)
class TestEviction(unittest.TestCase):
    def _make_cache(self, n_layers=3, B=1, H_kv=2, L=16, d_h=4):
        """Each K/V column encodes its absolute position so retention is checkable:
        K[..., p, :] = p ; V[..., p, :] = p + 100."""
        cache = DynamicCache()
        pos = torch.arange(L).float()
        for li in range(n_layers):
            K = pos.view(1, 1, L, 1).expand(B, H_kv, L, d_h).clone()
            V = (pos + 100).view(1, 1, L, 1).expand(B, H_kv, L, d_h).clone()
            cache.update(K, V, li)
        return cache, n_layers, L

    def test_keeps_exactly_requested_positions_in_order(self):
        cache, n_layers, L = self._make_cache()
        retain = [0, 2, 3, 7, 15]
        new_cache = kvc._evict_to_positions(cache, retain)

        self.assertEqual(kvc._seq_len(new_cache), len(retain))
        self.assertEqual(kvc._num_layers(new_cache), n_layers)
        for li in range(n_layers):
            K = kvc._layer_keys(new_cache, li)
            V = kvc._layer_values(new_cache, li)
            self.assertEqual(K.shape[2], len(retain))
            # Column j must equal the original column at retain[j].
            for j, p in enumerate(retain):
                self.assertTrue(torch.all(K[:, :, j, :] == float(p)))
                self.assertTrue(torch.all(V[:, :, j, :] == float(p + 100)))

    def test_original_cache_unchanged(self):
        cache, _, L = self._make_cache()
        _ = kvc._evict_to_positions(cache, [0, 1, 2])
        self.assertEqual(kvc._seq_len(cache), L)


# ── GPU tests for Colab: DEFINED but NOT run (require CUDA + real Qwen model) ──

def gpu_test_full_vs_split_prefill_decode(model, tokenizer, device="cuda",
                                          prompt="The quick brown fox jumps over the lazy dog."):
    """
    C1: direct full-forward logits must match split prefill+decode logits.

    Call from a Colab cell with a loaded Qwen2.5-1.5B-Instruct model.  Returns the
    max argmax disagreement count (0 == exact next-token agreement).
    """
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    P = ids.shape[1]
    with torch.no_grad():
        full = model(ids, use_cache=False).logits  # (1, P, V)

    half = max(1, P // 2)
    cache = DynamicCache()
    with torch.no_grad():
        out = kvc.forward_with_positions(model, cache, ids[:, :half], 0, device)
    logits_list = [out.logits]                     # (1, half, V)
    abs_pos = half
    for t in range(half, P):
        with torch.no_grad():
            out = kvc.forward_with_positions(model, cache, ids[:, t:t + 1], abs_pos, device)
        abs_pos += 1
        logits_list.append(out.logits)
    split = torch.cat(logits_list, dim=1)          # (1, P, V)

    assert split.shape == full.shape, (split.shape, full.shape)
    disagree = (full.argmax(-1) != split.argmax(-1)).sum().item()
    assert disagree == 0, f"full vs split next-token mismatch at {disagree} positions"
    return disagree


def gpu_test_full_retention_matches_greedy(model, tokenizer,
                                           prompts=("What is the capital of France?",
                                                    "List three prime numbers."),
                                           device="cuda", max_new_tokens=20):
    """
    C1: 100% retention through the eviction path reproduces ordinary greedy decoding.

    Prefill the whole prompt, evict to ALL positions (100% retention), then greedy
    decode via `greedy_generate_edited`; the token ids must match HF greedy generate.
    """
    model.eval()
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        P = ids.shape[1]

        # Pure argmax greedy: pin out any repetition penalty / n-gram blocking from
        # the model's generation_config so it matches greedy_generate_edited exactly.
        gen = model.generate(ids, max_new_tokens=max_new_tokens,
                             do_sample=False, num_beams=1,
                             repetition_penalty=1.0, no_repeat_ngram_size=0)
        ref = gen[0, P:]

        # Prefill positions 0..P-2; prime decoding with the last prompt token.
        cache = DynamicCache()
        with torch.no_grad():
            kvc.forward_with_positions(model, cache, ids[:, :P - 1], 0, device)
        full_retain = list(range(kvc._seq_len(cache)))      # 100% retention
        cache = kvc._evict_to_positions(cache, full_retain)

        out = kvc.greedy_generate_edited(
            model, cache, ids[:, P - 1:P], original_context_len=P - 1,
            max_new_tokens=max_new_tokens,
            eos_id=getattr(tokenizer, "eos_token_id", None), device=device,
        )
        got = out[0]
        n = min(len(ref), len(got))
        assert torch.equal(ref[:n].cpu(), got[:n].cpu()), (
            f"greedy mismatch for prompt {prompt!r}: ref={ref[:n].tolist()} "
            f"got={got[:n].tolist()}"
        )
    return True


@unittest.skip("GPU/Colab only: requires CUDA and a loaded Qwen2.5-1.5B-Instruct model")
class TestKVCacheGPU(unittest.TestCase):
    def test_full_vs_split(self):
        raise NotImplementedError("call gpu_test_full_vs_split_prefill_decode(model, tokenizer)")

    def test_full_retention_greedy(self):
        raise NotImplementedError("call gpu_test_full_retention_matches_greedy(model, tokenizer)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
