"""
Prefix banks and distribution alignment (PDF §3 "Fixed prefix banks", step 5).

Bank record format (identical to the supplied banks from setup_data.py --prefix-banks):
  prompt_id, domain, bank, position, prompt, response_prefix_token_ids, source_length,
  source_generation_seed (+ producing_checkpoint for our banks)
  `position` p is the 1-based index of the response token whose distribution is
  evaluated; response_prefix_token_ids = seq[:p-1] (setup_data convention).
Position selection (`select_positions`) reproduces setup_data.py exactly: per source
sequence, rng = random.Random(f"PA4-banks-v1:6304:{pid}:{bank}"), bins
(1,16,8), (17,32,8), (33,64,16), (65,192,16) -> up to 32 positions in 1..64 and up to 16
in 65..192, each sampled without replacement from the positions the sequence reaches.

Banks:
  teacher, original_draft          supplied (data/prefix_banks/prefix_banks_locked.jsonl.gz)
  exact_spec_original_gamma4       generated here with OUR exact sampler (original 0.5B
                                   draft, gamma 4, temperature 1, 192-token cap)
  student_<method>_s<seed>         each trained student's own rollouts (merged checkpoint)

Metrics at every saved prefix, for every draft, against the target p (`alignment`):
forward KL(p||q), reverse KL(q||p), TV = 1/2 sum|p-q|, expected one-token acceptance
sum min(p,q) = 1 - TV, greedy agreement argmax p == argmax q.  Each record is one
response token, so means over records are token-weighted.  All records of one source
sequence are prefixes of each other, so each (bank, prompt) group needs ONE forward per
model over prompt + longest prefix.
"""

import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from common.runio import read_jsonl, write_jsonl
from .generation import chat_ids, sample_batch
from .speculative import QWEN_EOS_IDS, speculative_generate

BANK_FILE = "data/prefix_banks/prefix_banks_locked.jsonl.gz"
BINS = ((1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16))
POSITION_BINS = ((1, 16), (17, 32), (33, 64), (65, 192))
SPEC_BANK = "exact_spec_original_gamma4"
EPS = 1e-12


def select_positions(pid: str, bank: str, seq_len: int) -> List[int]:
    rng = random.Random(f"PA4-banks-v1:6304:{pid}:{bank}")
    pos = []
    for lo, hi, n in BINS:
        cand = list(range(lo, min(hi, seq_len) + 1))
        pos.extend(sorted(rng.sample(cand, min(n, len(cand)))))
    return pos


def make_records(prompt: dict, bank: str, seq: Sequence[int], seed: int, checkpoint: Optional[str] = None) -> List[dict]:
    recs = []
    for p in select_positions(prompt["id"], bank, len(seq)):
        r = {"prompt_id": prompt["id"], "domain": prompt["domain"], "bank": bank, "position": p,
             "prompt": prompt["text"], "response_prefix_token_ids": list(seq[:p - 1]),
             "source_length": len(seq), "source_generation_seed": seed}
        if checkpoint:
            r["producing_checkpoint"] = checkpoint
        recs.append(r)
    return recs


def position_bin(p: int) -> str:
    for lo, hi in POSITION_BINS:
        if lo <= p <= hi:
            return f"[{lo},{hi}]"
    return "out"


def load_supplied_banks(data_repo: str) -> List[dict]:
    return read_jsonl(Path(data_repo) / BANK_FILE)


def check_bank_prompts(records: Sequence[dict], test_prompts: Sequence[dict]) -> dict:
    """The banks' prompt text must equal our test prompt text for every prompt id."""
    by_id = {p["id"]: p["text"] for p in test_prompts}
    bad = sorted({r["prompt_id"] for r in records if by_id.get(r["prompt_id"]) != r["prompt"]})
    counts = defaultdict(int)
    for r in records:
        counts[f"{r['bank']}/{r['domain']}"] += 1
    return {"mismatched_prompt_ids": bad, "counts": dict(counts)}


# ── bank generation ───────────────────────────────────────────────────────────

def generate_spec_bank(target, draft, tokenizer, prompts: Sequence[dict], out_path: str, device="cuda",
                       gamma: int = 4, max_new: int = 192, log=print) -> dict:
    """Exact-speculative consistency bank with our sampler (one sequence per prompt,
    torch.manual_seed(6304 + 10007*j + 777) for prompt j in bank order)."""
    recs, stats = [], []
    t0 = time.perf_counter()
    for j, p in enumerate(prompts):
        seed = 6304 + 10007 * j + 777
        torch.manual_seed(seed)
        st = speculative_generate(target, draft, chat_ids(tokenizer, p["text"]), max_new, gamma, QWEN_EOS_IDS, device)
        recs += make_records(p, SPEC_BANK, st.tokens, seed, "original Qwen2.5-0.5B-Instruct draft, gamma=4")
        stats.append({"prompt_id": p["id"], "len": len(st.tokens), "eos": st.eos,
                      "acc_over_examined": st.summary()["accepted_over_examined"]})
        if (j + 1) % 20 == 0:
            log(f"[spec bank] {j + 1}/{len(prompts)}")
    write_jsonl(out_path, recs)
    return {"n_records": len(recs), "seconds": time.perf_counter() - t0, "per_prompt": stats}


def generate_student_bank(student, tokenizer, prompts: Sequence[dict], bank: str, checkpoint: str,
                          out_path: str, device="cuda", max_new: int = 192, batch_size: int = 30,
                          seed: int = 6304) -> dict:
    """Frozen own-policy rollouts of one trained student (batched, seeded per batch)."""
    recs = []
    t0 = time.perf_counter()
    for b in range(0, len(prompts), batch_size):
        chunk = prompts[b:b + batch_size]
        bseed = seed + 1000003 * (b // batch_size + 1)
        torch.manual_seed(bseed)
        outs = sample_batch(student, [chat_ids(tokenizer, p["text"]) for p in chunk], max_new, device=device)
        for p, o in zip(chunk, outs):
            recs += make_records(p, bank, o["tokens"], bseed, checkpoint)
    write_jsonl(out_path, recs)
    return {"n_records": len(recs), "seconds": time.perf_counter() - t0}


# ── alignment metrics ─────────────────────────────────────────────────────────

def alignment(p_logp: torch.Tensor, q_logp: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Row-wise metrics from log-probabilities (N, V) fp32."""
    p, q = p_logp.exp(), q_logp.exp()
    return {"fwd_kl": (p * (p_logp - q_logp)).sum(-1),
            "rev_kl": (q * (q_logp - p_logp)).sum(-1),
            "tv": 0.5 * (p - q).abs().sum(-1),
            "expected_accept": torch.minimum(p, q).sum(-1),
            "greedy_agree": (p_logp.argmax(-1) == q_logp.argmax(-1)).float(),
            "p_top": p_logp.argmax(-1), "q_top": q_logp.argmax(-1)}


def group_records(records: Sequence[dict]) -> Dict[tuple, List[dict]]:
    g = defaultdict(list)
    for r in records:
        g[(r["bank"], r["prompt_id"])].append(r)
    for k, rs in g.items():
        longest = max(rs, key=lambda r: r["position"])["response_prefix_token_ids"]
        for r in rs:
            pre = r["response_prefix_token_ids"]
            assert len(pre) == r["position"] - 1, (k, r["position"], len(pre))
            assert longest[:len(pre)] == pre, f"records of {k} are not prefixes of one sequence"
    return g


@torch.no_grad()
def evaluate_banks(target, drafts: Dict[str, torch.nn.Module], tokenizer, records: Sequence[dict],
                   device="cuda", log=print) -> List[dict]:
    """One output row per (record, draft)."""
    groups = group_records(records)
    rows = []
    for gi, ((bank, pid), rs) in enumerate(sorted(groups.items())):
        rs = sorted(rs, key=lambda r: r["position"])
        prompt_ids = chat_ids(tokenizer, rs[0]["prompt"])
        longest = rs[-1]["response_prefix_token_ids"]
        ids = torch.tensor([prompt_ids + longest], device=device)
        P = len(prompt_ids)
        idx = torch.tensor([P + r["position"] - 2 for r in rs], device=device)
        t_logp = F.log_softmax(target(input_ids=ids, use_cache=False).logits[0][idx].float(), -1)
        for name, d in drafts.items():
            q_logp = F.log_softmax(d(input_ids=ids, use_cache=False).logits[0][idx].float(), -1)
            m = alignment(t_logp, q_logp)
            for j, r in enumerate(rs):
                rows.append({"bank": bank, "prompt_id": pid, "domain": r["domain"], "position": r["position"],
                             "position_bin": position_bin(r["position"]), "draft": name,
                             **{k: float(v[j]) if k not in ("p_top", "q_top") else int(v[j])
                                for k, v in m.items()}})
            del q_logp
        if (gi + 1) % 100 == 0:
            log(f"[banks] {gi + 1}/{len(groups)} groups")
    return rows
