"""
Exact speculative sampling (Leviathan et al. 2023) — PDF §3 step 1.

ONE pure decision function, `verify_round`, implements accept/reject
(accept x_i with probability min(1, p_i(x_i)/q_i(x_i))), positive-residual sampling on
rejection (r ∝ max(p_i - q_i, 0)) and the full-acceptance bonus token (from p_n).  It is
called unchanged by the real-model decoder (`speculative_generate`) and by the
multi-step Markov χ² harness (`toy_speculative_sequence`), so the χ² test exercises the
exact code that produces the real tokens.  Randomness: torch.rand / torch.multinomial
on the global generators (seed with torch.manual_seed).

Cache discipline of `speculative_generate` (batch 1):
  * INVARIANT: after every round, the target cache holds the committed sequence
    (prompt + emitted tokens) EXCEPT its last token (keep = len(committed) - 1 positions).
    The draft cache never exceeds keep; after a round that ran the draft (n_max > 0) it
    holds at least keep - 1.  A one-token round (n_max == 0) skips the draft, so after a
    bonus round it may lag by 2; the next round's feed = committed[cache_len(dc):]
    catches up any lag.
  * prefill: prompt[:-1] into both caches (one forward each).
  * a round: the draft extends from its cache, proposing n = min(γ, remaining-1) tokens
    (stopping early at an EOS proposal).  The target receives
    [committed tokens it has not cached, normally just the last emitted token] + the n
    proposals in ONE forward, giving the n+1 distributions p_0..p_n.  `verify_round`
    decides.  Both caches are then `crop()`ped to len(committed) - 1.
  * n = 0 (one token left before the cap) is a plain target step ("one-token round");
    the cap can therefore never be overshot.
  * EOS: an accepted EOS proposal, or an EOS drawn from the residual or bonus, ends
    generation; nothing after an EOS is emitted.
Sampling: temperature 1, top-p 1, top-k 0 (full softmax), repetition penalty 1.0.
"""

from collections import Counter
from dataclasses import dataclass, field
import itertools
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

QWEN_EOS_IDS = frozenset({151645, 151643})


# ── the pure round decision ───────────────────────────────────────────────────

def positive_residual(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Normalised max(p - q, 0).  If p <= q everywhere (p == q) a rejection has
    probability 0, so the fallback to p is unreachable in exact arithmetic."""
    r = (p - q).clamp_min(0.0)
    s = r.sum()
    return r / s if float(s) > 0.0 else p


@dataclass
class RoundResult:
    tokens: List[int]          # emitted this round (accepted proposals + residual/bonus token)
    n_accepted: int            # accepted proposals
    n_examined: int            # proposals that went through the accept test
    n_proposed: int            # proposals drafted
    bonus: bool                # full acceptance -> bonus token from p_n
    rejected: bool             # a proposal was rejected -> residual token
    eos: bool                  # an EOS token was emitted (generation ends)


def verify_round(p: torch.Tensor, q: torch.Tensor, proposals: Sequence[int],
                 eos_ids=frozenset(), residual_fn: Callable = positive_residual) -> RoundResult:
    """Exact speculative decision for one round.

    p          (n+1, V) target probabilities; p[i] is the distribution of the token that
               follows prefix + proposals[:i]
    q          (n, V)   draft probabilities from which proposals[i] was drawn
    proposals  n draft tokens (n may be 0: the round is a plain target sample)
    residual_fn  positive_residual for the exact sampler; the χ² negative control
               passes `lambda p, q: p`.
    """
    n = len(proposals)
    assert p.shape[0] == n + 1 and (n == 0 or q.shape[0] == n)
    out: List[int] = []
    for i, x in enumerate(proposals):
        qx = float(q[i, x])
        a = 1.0 if qx <= 0.0 else min(1.0, float(p[i, x]) / qx)
        if float(torch.rand(())) < a:
            out.append(int(x))
            if int(x) in eos_ids:
                return RoundResult(out, i + 1, i + 1, n, False, False, True)
        else:
            y = int(torch.multinomial(residual_fn(p[i], q[i]), 1))
            out.append(y)
            return RoundResult(out, i, i + 1, n, False, True, y in eos_ids)
    y = int(torch.multinomial(p[n], 1))
    out.append(y)
    return RoundResult(out, n, n, n, True, False, y in eos_ids)


def proposal_length(gamma: int, remaining: int) -> int:
    """Proposals in a round with `remaining` tokens left before the cap (>= 1)."""
    return max(0, min(gamma, remaining - 1))


# ── χ² harness: 3-symbol Markov target/draft driving verify_round ─────────────

TOY_P_START = [0.5, 0.3, 0.2]
TOY_P = [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]]
TOY_Q_START = [0.2, 0.3, 0.5]
TOY_Q = [[0.2, 0.4, 0.4], [0.5, 0.2, 0.3], [0.4, 0.4, 0.2]]


class MarkovToy:
    """First-order Markov chains over 3 symbols; the empty prefix uses the start law."""

    def __init__(self, p_start=TOY_P_START, P=TOY_P, q_start=TOY_Q_START, Q=TOY_Q):
        self.p_start = torch.tensor(p_start, dtype=torch.float64)
        self.P = torch.tensor(P, dtype=torch.float64)
        self.q_start = torch.tensor(q_start, dtype=torch.float64)
        self.Q = torch.tensor(Q, dtype=torch.float64)

    def target(self, prefix: Sequence[int]) -> torch.Tensor:
        return self.p_start if not prefix else self.P[prefix[-1]]

    def draft(self, prefix: Sequence[int]) -> torch.Tensor:
        return self.q_start if not prefix else self.Q[prefix[-1]]

    def joint(self, seq: Sequence[int]) -> float:
        pr, prefix = 1.0, []
        for s in seq:
            pr *= float(self.target(prefix)[s])
            prefix.append(s)
        return pr


def toy_speculative_sequence(toy: MarkovToy, length: int, gamma: int,
                             residual_fn: Callable = positive_residual) -> List[int]:
    """One length-`length` sequence by speculative rounds of γ proposals each.
    The harness always drafts γ proposals and truncates the final round's output to
    `length` (a prefix of an exactly sampled sequence is exactly distributed), so γ=4
    exercises full 4-proposal rounds even for length-3 sequences."""
    seq: List[int] = []
    while len(seq) < length:
        props, qs, ctx = [], [], list(seq)
        for _ in range(gamma):
            qd = toy.draft(ctx)
            x = int(torch.multinomial(qd, 1))
            props.append(x)
            qs.append(qd)
            ctx.append(x)
        ps = torch.stack([toy.target(seq + props[:i]) for i in range(gamma + 1)])
        res = verify_round(ps, torch.stack(qs), props, residual_fn=residual_fn)
        seq.extend(res.tokens)
    return seq[:length]


def chi2_test(gamma: int, n_sequences: int = 100_000, length: int = 3, seed: int = 6304,
              correct: bool = True, toy: Optional[MarkovToy] = None) -> dict:
    """Pearson χ² goodness of fit of speculative sequence frequencies against the exact
    target joint distribution over all 3^length sequences."""
    from scipy.stats import chisquare
    toy = toy or MarkovToy()
    torch.manual_seed(seed)
    res_fn = positive_residual if correct else (lambda p, q: p)
    counts = Counter(tuple(toy_speculative_sequence(toy, length, gamma, res_fn)) for _ in range(n_sequences))
    cats = list(itertools.product(range(3), repeat=length))
    probs = [toy.joint(c) for c in cats]
    expected = [pr * n_sequences for pr in probs]
    observed = [counts.get(c, 0) for c in cats]
    stat, pval = chisquare(observed, f_exp=expected)
    return {"gamma": gamma, "correct_sampler": correct, "n_sequences": n_sequences, "length": length,
            "seed": seed, "chi2": float(stat), "dof": len(cats) - 1, "p_value": float(pval),
            "min_expected": float(min(expected)), "rejected_at_0.01": bool(pval < 0.01),
            "table": [{"sequence": "".join(map(str, c)), "target_joint_prob": pr, "expected": e,
                       "observed": o, "empirical_prob": o / n_sequences}
                      for c, pr, e, o in zip(cats, probs, expected, observed)]}


# ── real-model decoding ───────────────────────────────────────────────────────

def _sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def cache_len(cache) -> int:
    return cache.get_seq_length() if cache is not None else 0


def forward_tokens(model, cache, tokens: Sequence[int], device, n_logits: int = 1):
    """Append `tokens` to `cache` (positions continue from the cache length) and
    return fp32 logits for the last `n_logits` positions, shape (n_logits, V)."""
    ids = torch.tensor([list(tokens)], dtype=torch.long, device=device)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=n_logits)
    return out.logits[0].float()


def prefill(model, prompt_ids: Sequence[int], device, chunk: Optional[int] = None):
    """Cache of prompt_ids[:-1] (the cache invariant); chunked prefill if `chunk`."""
    from transformers import DynamicCache
    cache = DynamicCache()
    body = list(prompt_ids[:-1])
    step = chunk or max(1, len(body))
    for s in range(0, len(body), step):
        forward_tokens(model, cache, body[s:s + step], device)
    return cache


@dataclass
class GenStats:
    tokens: List[int] = field(default_factory=list)
    n_rounds: int = 0                  # target verification passes (excl. prefill)
    n_proposed: int = 0
    n_examined: int = 0
    n_accepted: int = 0
    n_bonus: int = 0
    n_draft_forwards: int = 0
    accepted_per_round: List[int] = field(default_factory=list)
    eos: bool = False
    truncated: bool = False
    t_prefill_target_ms: float = 0.0
    t_prefill_draft_ms: float = 0.0
    t_draft_ms: float = 0.0
    t_verify_ms: float = 0.0
    t_accept_ms: float = 0.0
    t_rollback_ms: float = 0.0
    t_total_ms: float = 0.0
    peak_alloc_mb: Optional[float] = None
    debug_max_logit_err: float = 0.0

    def summary(self) -> dict:
        n_out = len(self.tokens)
        d = {k: v for k, v in self.__dict__.items() if k not in ("tokens", "accepted_per_round")}
        decode_ms = self.t_total_ms - self.t_prefill_target_ms - self.t_prefill_draft_ms
        phases = self.t_draft_ms + self.t_verify_ms + self.t_accept_ms + self.t_rollback_ms
        d.update(n_output_tokens=n_out,
                 accepted_over_examined=self.n_accepted / self.n_examined if self.n_examined else None,
                 accepted_over_proposed=self.n_accepted / self.n_proposed if self.n_proposed else None,
                 mean_accepted_per_verification=self.n_accepted / self.n_rounds if self.n_rounds else None,
                 target_passes_per_output_token=self.n_rounds / n_out if n_out else None,
                 draft_forwards_per_output_token=self.n_draft_forwards / n_out if n_out else None,
                 t_decode_ms=decode_ms, t_host_overhead_ms=decode_ms - phases,
                 tokens_per_s=n_out / (self.t_total_ms / 1000) if self.t_total_ms > 0 else None,
                 accepted_per_round=self.accepted_per_round)
        return d


def _peak_reset(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2 ** 20
    return None


@torch.no_grad()
def speculative_generate(target, draft, prompt_ids: Sequence[int], max_new_tokens: int, gamma: int,
                         eos_ids=QWEN_EOS_IDS, device="cuda", debug_check: bool = False,
                         prefill_chunk: Optional[int] = None) -> GenStats:
    """Exact speculative sampling with the cache discipline in the module docstring.
    Per-round phase timers are CUDA-synchronised.  debug_check=True (tests) compares
    every verification against a fresh full forward and asserts the cache invariant."""
    assert len(prompt_ids) >= 1 and max_new_tokens >= 1 and gamma >= 1
    st = GenStats()
    _peak_reset(device)
    _sync(device)
    t_start = time.perf_counter()
    tc = prefill(target, prompt_ids, device, prefill_chunk)
    _sync(device)
    t1 = time.perf_counter()
    st.t_prefill_target_ms = 1000 * (t1 - t_start)
    dc = prefill(draft, prompt_ids, device, prefill_chunk)
    _sync(device)
    st.t_prefill_draft_ms = 1000 * (time.perf_counter() - t1)
    committed = list(prompt_ids)
    while len(st.tokens) < max_new_tokens:
        n_max = proposal_length(gamma, max_new_tokens - len(st.tokens))
        # draft proposals
        t0 = time.perf_counter()
        feed = committed[cache_len(dc):]
        props, qs = [], []
        for _ in range(n_max):
            q = torch.softmax(forward_tokens(draft, dc, feed, device)[-1], dim=-1)
            st.n_draft_forwards += 1
            x = int(torch.multinomial(q, 1))
            props.append(x)
            qs.append(q)
            feed = [x]
            if x in eos_ids:
                break
        _sync(device)
        t1 = time.perf_counter()
        # one target verification pass over [uncached committed tokens] + proposals
        t_feed = committed[cache_len(tc):] + props
        logits = forward_tokens(target, tc, t_feed, device, n_logits=len(props) + 1)
        _sync(device)
        t2 = time.perf_counter()
        if debug_check:
            full = target(input_ids=torch.tensor([committed + props], device=device)).logits[0].float()
            ref = full[len(committed) - 1:]
            st.debug_max_logit_err = max(st.debug_max_logit_err, float((ref - logits).abs().max()))
        p = torch.softmax(logits, dim=-1)
        res = verify_round(p, torch.stack(qs) if qs else p[:0], props, eos_ids)
        _sync(device)
        t3 = time.perf_counter()
        st.tokens.extend(res.tokens)
        committed.extend(res.tokens)
        keep = len(committed) - 1
        tc.crop(keep)
        if cache_len(dc) > keep:
            dc.crop(keep)
        _sync(device)
        t4 = time.perf_counter()
        if debug_check:
            assert cache_len(tc) == keep, (cache_len(tc), keep)
            assert cache_len(dc) <= keep and (n_max == 0 or cache_len(dc) >= keep - 1), (
                cache_len(dc), keep, n_max)
        st.n_rounds += 1
        st.n_proposed += res.n_proposed
        st.n_examined += res.n_examined
        st.n_accepted += res.n_accepted
        st.n_bonus += int(res.bonus)
        st.accepted_per_round.append(res.n_accepted)
        st.t_draft_ms += 1000 * (t1 - t0)
        st.t_verify_ms += 1000 * (t2 - t1)
        st.t_accept_ms += 1000 * (t3 - t2)
        st.t_rollback_ms += 1000 * (t4 - t3)
        if res.eos:
            st.eos = True
            break
    assert len(st.tokens) <= max_new_tokens
    st.truncated = not st.eos and len(st.tokens) >= max_new_tokens
    _sync(device)
    st.t_total_ms = 1000 * (time.perf_counter() - t_start)
    st.peak_alloc_mb = _peak(device)
    return st


@torch.no_grad()
def target_only_generate(model, prompt_ids: Sequence[int], max_new_tokens: int, eos_ids=QWEN_EOS_IDS,
                         device="cuda", prefill_chunk: Optional[int] = None) -> GenStats:
    """Same-condition reference: one target forward per token, same cache discipline,
    full-softmax temperature-1 sampling.  t_verify_ms = per-token target forwards."""
    st = GenStats()
    _peak_reset(device)
    _sync(device)
    t_start = time.perf_counter()
    cache = prefill(model, prompt_ids, device, prefill_chunk)
    _sync(device)
    st.t_prefill_target_ms = 1000 * (time.perf_counter() - t_start)
    committed = list(prompt_ids)
    while len(st.tokens) < max_new_tokens:
        t1 = time.perf_counter()
        logits = forward_tokens(model, cache, committed[cache_len(cache):], device)
        _sync(device)
        t2 = time.perf_counter()
        y = int(torch.multinomial(torch.softmax(logits[-1], dim=-1), 1))
        _sync(device)
        st.t_verify_ms += 1000 * (t2 - t1)
        st.t_accept_ms += 1000 * (time.perf_counter() - t2)
        st.tokens.append(y)
        committed.append(y)
        st.n_rounds += 1
        if y in eos_ids:
            st.eos = True
            break
    st.truncated = not st.eos and len(st.tokens) >= max_new_tokens
    _sync(device)
    st.t_total_ms = 1000 * (time.perf_counter() - t_start)
    st.peak_alloc_mb = _peak(device)
    return st
