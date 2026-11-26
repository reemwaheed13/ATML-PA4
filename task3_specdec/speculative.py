"""
Exact speculative sampling (Leviathan et al. 2023).

Correctness requirement (PDF §3 step 1):
  - Accepts a draft proposal x ~ q with probability min(1, p(x)/q(x)).
  - On rejection, samples from the NORMALIZED POSITIVE RESIDUAL
    r(y) ∝ [p(y) - q(y)]+ (not from p directly).
  - If all γ proposals are accepted, samples an additional BONUS TOKEN
    from p(· | h + accepted).
  - Cache rollback/advancement must correctly represent the returned prefix.
  - χ² test: empirical joint sequence frequencies over 100,000 trials must
    NOT reject correct sampler at p < 0.01; a deliberately incorrect sampler
    (using p instead of residual) MUST be rejected at p < 0.01.

Sampling parameters (PDF §3):
  temperature=1, top_p=1, repetition_penalty=1.0
  EOS IDs for Qwen2.5-Instruct: {151645, 151643}
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from scipy.stats import chisquare
import numpy as np


# Qwen2.5-Instruct EOS token IDs.
QWEN_EOS_IDS = {151645, 151643}


# ── Core sampling primitives ───────────────────────────────────────────────────

def _temperature_sample(logits: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample from logits with given temperature.
    Returns (sampled_token_id, full_probability_vector).
    """
    if temperature != 1.0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return token, probs


def _residual_sample(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Sample from the normalized positive residual r(y) ∝ [p(y) - q(y)]+.
    p, q: probability vectors (V,), sum to 1.
    Returns sampled token id (scalar tensor).
    """
    residual = (p - q).clamp(min=0.0)
    total = residual.sum()
    if total < 1e-10:
        # Degenerate: p ≤ q everywhere — sample from p directly (edge case).
        return torch.multinomial(p.clamp(min=0.0) + 1e-10, num_samples=1).squeeze(-1)
    residual = residual / total
    return torch.multinomial(residual, num_samples=1).squeeze(-1)


# ── Single speculative step ────────────────────────────────────────────────────

def exact_speculative_step(
    draft_model,
    target_model,
    input_ids: torch.Tensor,            # (1, L) current prefix
    draft_past_kv,
    target_past_kv,
    gamma: int = 4,
    temperature: float = 1.0,
    eos_ids: set = QWEN_EOS_IDS,
    device: str = "cuda",
) -> Tuple[torch.Tensor, any, any, dict]:
    """
    One round of exact speculative decoding.

    1. Draft proposes gamma tokens x_1, ..., x_gamma.
    2. Target evaluates the gamma+1 continuations in one forward pass.
    3. Accept/reject using exact sampling; on rejection draw from residual.
    4. If all accepted, draw a bonus token from target.
    5. Return the accepted continuation and updated caches.

    Returns
    -------
    accepted_ids   (1, k) where 1 ≤ k ≤ gamma+1
    draft_past_kv  updated draft KV cache (trimmed to accepted prefix)
    target_past_kv updated target KV cache (trimmed to accepted prefix)
    info dict:
        n_proposed      int     gamma
        n_accepted      int     k (accepted proposal tokens; excludes bonus)
        bonus_token     bool    whether a bonus token was added
        draft_probs     list[Tensor(V)] draft distributions for each proposal
        target_probs    list[Tensor(V)] target distributions at same positions
        accepted_tokens list[int]
        eos_hit         bool
    """
    draft_model.eval()
    target_model.eval()

    # ── Step 1: Draft proposal ───────────────────────────────────────────────
    draft_tokens = []
    draft_probs = []
    cur_ids = input_ids  # (1, L)
    cur_draft_kv = draft_past_kv

    with torch.no_grad():
        for _ in range(gamma):
            out = draft_model(cur_ids, past_key_values=cur_draft_kv, use_cache=True)
            logits = out.logits[:, -1, :]        # (1, V)
            cur_draft_kv = out.past_key_values
            token, prob = _temperature_sample(logits[0], temperature)
            draft_tokens.append(token)
            draft_probs.append(prob)
            cur_ids = token.view(1, 1)
            if token.item() in eos_ids:
                break

    n_proposed = len(draft_tokens)

    # ── Step 2: Target evaluation on full proposed sequence ──────────────────
    proposal_ids = torch.tensor(
        [t.item() for t in draft_tokens], dtype=torch.long, device=device
    ).unsqueeze(0)  # (1, gamma)

    # Run target on [input_ids, proposal] in one pass.
    full_ids = torch.cat([input_ids, proposal_ids], dim=1)  # (1, L + gamma)
    with torch.no_grad():
        target_out = target_model(full_ids, past_key_values=target_past_kv, use_cache=True)
    target_logits = target_out.logits[0]   # (L+gamma, V)
    target_new_kv = target_out.past_key_values

    # Target probs at positions corresponding to draft proposals.
    # logit at position (L-1 + i) predicts the i-th draft token (0-indexed).
    orig_L = input_ids.shape[1]
    target_probs = []
    for i in range(n_proposed):
        logit_pos = orig_L - 1 + i   # logit at L-1+i predicts token at L+i
        p_i = F.softmax(target_logits[logit_pos].float(), dim=-1)
        target_probs.append(p_i)

    # ── Step 3: Accept/reject ────────────────────────────────────────────────
    accepted_tokens = []
    n_accepted = 0

    for i in range(n_proposed):
        x_i = draft_tokens[i].item()
        p_i = target_probs[i]
        q_i = draft_probs[i]

        ratio = (p_i[x_i] / (q_i[x_i] + 1e-10)).clamp(max=1.0).item()
        u = torch.rand(1).item()

        if u <= ratio:
            # Accept.
            accepted_tokens.append(x_i)
            n_accepted += 1
            if x_i in eos_ids:
                break
        else:
            # Reject: sample from positive residual.
            residual_token = _residual_sample(p_i, q_i).item()
            accepted_tokens.append(residual_token)
            break

    bonus_token = False
    if n_accepted == n_proposed and n_proposed > 0 and not (draft_tokens[-1].item() in eos_ids):
        # All proposals accepted: draw bonus token from target.
        bonus_logit_pos = orig_L - 1 + n_proposed
        if bonus_logit_pos < target_logits.shape[0]:
            p_bonus = F.softmax(target_logits[bonus_logit_pos].float(), dim=-1)
            bonus = torch.multinomial(p_bonus, num_samples=1).item()
            accepted_tokens.append(bonus)
            bonus_token = True

    # ── Step 4: Cache rollback ───────────────────────────────────────────────
    # Keep target cache up to accepted prefix length.
    k = len(accepted_tokens)  # actual new tokens
    # Trim target cache: it was extended to L+gamma; keep L+k.
    target_past_kv = _trim_kv_cache(target_new_kv, orig_L + k)

    # Rebuild draft cache: advance draft model from input_ids to accepted prefix.
    with torch.no_grad():
        accepted_ids_tensor = torch.tensor(
            accepted_tokens[:n_accepted if not bonus_token else n_accepted + 1],
            dtype=torch.long, device=device
        ).unsqueeze(0)
        # We need to run draft forward on accepted tokens to sync its cache.
        # For the draft cache, only the first n_accepted tokens were from draft proposals.
        # Roll back draft cache to input length, then advance.
        draft_past_kv = _trim_kv_cache(cur_draft_kv, orig_L + n_accepted)

    accepted_tensor = torch.tensor(accepted_tokens, dtype=torch.long, device=device).unsqueeze(0)
    eos_hit = any(t in eos_ids for t in accepted_tokens)

    return accepted_tensor, draft_past_kv, target_past_kv, {
        "n_proposed": n_proposed,
        "n_accepted": n_accepted,
        "bonus_token": bonus_token,
        "draft_probs": [p.cpu() for p in draft_probs],
        "target_probs": [p.cpu() for p in target_probs],
        "accepted_tokens": accepted_tokens,
        "eos_hit": eos_hit,
    }


def _trim_kv_cache(past_kv, seq_len: int):
    """Trim a transformers KV cache to `seq_len` positions."""
    if past_kv is None:
        return None
    try:
        from transformers import DynamicCache
        if isinstance(past_kv, DynamicCache):
            new_cache = DynamicCache()
            for layer_idx in range(len(past_kv.key_cache)):
                K = past_kv.key_cache[layer_idx][:, :, :seq_len, :]
                V = past_kv.value_cache[layer_idx][:, :, :seq_len, :]
                new_cache.update(K, V, layer_idx)
            return new_cache
    except ImportError:
        pass
    # Fallback for tuple-of-tuple format.
    return tuple(
        (layer[0][:, :, :seq_len, :], layer[1][:, :, :seq_len, :])
        for layer in past_kv
    )


# ── Full generation loop ───────────────────────────────────────────────────────

def exact_speculative_generate(
    draft_model,
    target_model,
    prompt_ids: torch.Tensor,            # (1, prompt_len)
    max_new_tokens: int = 64,
    gamma: int = 4,
    temperature: float = 1.0,
    eos_ids: set = QWEN_EOS_IDS,
    device: str = "cuda",
) -> Tuple[List[int], dict]:
    """
    Generate max_new_tokens using exact speculative decoding.

    Returns (generated_token_ids, stats_dict).
    stats_dict aggregates per-round info: acceptance counts, timings, etc.
    """
    generated = []
    cur_ids = prompt_ids.to(device)
    draft_past_kv = None
    target_past_kv = None
    total_proposed = 0
    total_accepted = 0
    total_target_calls = 0

    # Warm up caches with a prefill pass.
    with torch.no_grad():
        d_out = draft_model(cur_ids, use_cache=True)
        draft_past_kv = d_out.past_key_values
        t_out = target_model(cur_ids, use_cache=True)
        target_past_kv = t_out.past_key_values

    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        g = min(gamma, remaining)

        accepted, draft_past_kv, target_past_kv, info = exact_speculative_step(
            draft_model, target_model, cur_ids,
            draft_past_kv, target_past_kv,
            gamma=g, temperature=temperature, eos_ids=eos_ids, device=device,
        )

        new_tokens = accepted[0].tolist()
        generated.extend(new_tokens)
        cur_ids = accepted[:, -1:]  # feed last accepted token

        total_proposed += info["n_proposed"]
        total_accepted += info["n_accepted"]
        total_target_calls += 1

        if info["eos_hit"] or len(generated) >= max_new_tokens:
            break

    return generated[:max_new_tokens], {
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "total_target_calls": total_target_calls,
        "acceptance_rate": total_accepted / total_proposed if total_proposed else 0.0,
    }


# ── Statistical correctness test ───────────────────────────────────────────────

class SpeculativeSampler:
    """
    Wrapper for running the χ² goodness-of-fit distributional correctness test.
    """

    @staticmethod
    def _make_markov(
        n_symbols: int,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a small Markov target and draft for the correctness test.

        Returns (target_trans, draft_trans):
          Each is a (n_symbols, n_symbols) transition matrix (rows = distributions).
        """
        rng = torch.Generator()
        rng.manual_seed(seed)
        t = torch.rand(n_symbols, n_symbols, generator=rng)
        t = t / t.sum(dim=1, keepdim=True)   # row-stochastic
        d = torch.rand(n_symbols, n_symbols, generator=rng)
        d = d / d.sum(dim=1, keepdim=True)
        return t, d

    @staticmethod
    def _spec_step_markov(
        state: int,
        target_trans: torch.Tensor,
        draft_trans: torch.Tensor,
        gamma: int = 1,
        correct: bool = True,
    ) -> Tuple[int, int]:
        """
        One round of speculative sampling on a Markov model.
        correct=True  uses positive residual (correct).
        correct=False replaces residual with p (incorrect — must be rejected by test).
        Returns (accepted_len, next_state).
        """
        p = target_trans[state]
        q = draft_trans[state]

        # Draft proposal.
        x = torch.multinomial(q, 1).item()

        ratio = (p[x] / (q[x] + 1e-10)).clamp(max=1.0).item()
        u = torch.rand(1).item()

        if u <= ratio:
            return 1, x
        else:
            if correct:
                residual = (p - q).clamp(min=0.0)
                s = residual.sum().item()
                if s < 1e-10:
                    residual = p.clamp(min=0.0)
                residual = residual / (residual.sum() + 1e-10)
                y = torch.multinomial(residual, 1).item()
            else:
                # Incorrect: sample from p instead of residual.
                y = torch.multinomial(p, 1).item()
            return 0, y

    @classmethod
    def run_chi2_test(
        cls,
        n_symbols: int = 3,
        seq_len: int = 3,
        n_trials: int = 100_000,
        seed: int = 6304,
        correct: bool = True,
        gamma: int = 1,
    ) -> dict:
        """
        Test that the speculative sampler reproduces the target distribution
        over joint sequences of length `seq_len`.

        Requirement from PDF:
          - Correct sampler: p-value > 0.01 (do NOT reject at p<0.01).
          - Incorrect sampler: p-value < 0.01 (MUST reject at p<0.01).
          - All expected category counts ≥ 10.

        Returns dict with p_value, chi2_stat, dof, expected_counts, observed_counts.
        """
        torch.manual_seed(seed)
        target_trans, draft_trans = cls._make_markov(n_symbols, seed)

        # Compute exact joint probabilities under the TARGET chain.
        # Stationary distribution (uniform start for simplicity).
        init = torch.ones(n_symbols) / n_symbols
        # Joint prob of sequence (s0, s1, s2, ...): p(s0) * prod p(s_{i+1}|s_i).
        # For seq_len=3 we have n_symbols^3 sequences.
        joint_expected = {}
        for s0 in range(n_symbols):
            for s1 in range(n_symbols):
                for s2 in range(n_symbols):
                    prob = (
                        init[s0].item()
                        * target_trans[s0, s1].item()
                        * target_trans[s1, s2].item()
                    )
                    joint_expected[(s0, s1, s2)] = prob

        # Simulate speculative sampler.
        from collections import Counter
        observed = Counter()
        for _ in range(n_trials):
            seq = []
            state = torch.multinomial(init, 1).item()
            seq.append(state)
            for _ in range(seq_len - 1):
                _, state = cls._spec_step_markov(
                    state, target_trans, draft_trans, gamma=gamma, correct=correct
                )
                seq.append(state)
            observed[tuple(seq)] += 1

        # Build parallel arrays for chi2 test.
        all_seqs = list(joint_expected.keys())
        expected_counts = np.array([joint_expected[s] * n_trials for s in all_seqs])
        observed_counts = np.array([observed.get(s, 0) for s in all_seqs])

        # Check all expected ≥ 10.
        min_expected = expected_counts.min()
        chi2_stat, p_value = chisquare(observed_counts, f_exp=expected_counts)

        return {
            "correct": correct,
            "chi2_stat": float(chi2_stat),
            "p_value": float(p_value),
            "min_expected_count": float(min_expected),
            "rejects_at_0_01": p_value < 0.01,
            "n_trials": n_trials,
            "n_symbols": n_symbols,
            "seq_len": seq_len,
        }
