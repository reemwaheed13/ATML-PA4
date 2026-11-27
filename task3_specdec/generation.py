"""
Shared Task 3 utilities: prompt sets, chat formatting, plain sampling, answer checks.

Prompt format (every condition, both models): the task text as the single user turn of
the tokenizer's chat template with add_generation_prompt=True — Qwen2.5's default system
prompt is inserted by the template (the same call setup_data.py used for the banks).
Sampling everywhere: temperature 1, top-p 1, top-k 0 (full softmax), repetition penalty
1.0, implemented directly on logits (no HF generation_config is consulted), seeded with
torch.manual_seed.  EOS = {151645 <|im_end|>, 151643 <|endoftext|>}.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from common.runio import read_jsonl
from .speculative import QWEN_EOS_IDS

TARGET_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MATH_CAP, SHORT_CAP = 192, 32


def chat_ids(tokenizer, text: str) -> List[int]:
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                        add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)


# ── prompt sets ───────────────────────────────────────────────────────────────

def load_train_prompts(data_repo: str) -> Dict[str, List[dict]]:
    """500 GSM8K training questions + 300 short-form training instructions."""
    root = Path(data_repo) / "data"
    gsm = sorted(read_jsonl(root / "public_locked" / "gsm8k_train_locked.jsonl.gz"), key=lambda r: r["order"])
    short = read_jsonl(root / "short_form" / "short_train.jsonl")
    math = [{"id": f"GSM8K-train-{r['order']:04d}", "category": "math", "text": r["example"]["question"],
             "answer": r["example"]["answer"]} for r in gsm]
    sh = [{"id": r["id"], "category": "short", "text": r["input"], "answer": r["answer"],
           "family": r["family"]} for r in short]
    assert len(math) == 500 and len(sh) == 300, (len(math), len(sh))
    return {"math": math, "short": sh}


def load_test_prompts(data_repo: str) -> List[dict]:
    """The 120 held-out prompts in setup_data.py's bank order: 60 GSM8K test, then the
    60 short-form test prompts (30 familiar, 30 shifted templates)."""
    root = Path(data_repo) / "data"
    gsm = sorted(read_jsonl(root / "public_locked" / "gsm8k_test_locked.jsonl.gz"), key=lambda r: r["order"])
    short = read_jsonl(root / "short_form" / "short_test.jsonl")
    out = [{"id": f"GSM8K-{r['order']:04d}", "domain": "math", "workload": "math",
            "text": r["example"]["question"], "answer": r["example"]["answer"]} for r in gsm]
    out += [{"id": r["id"], "domain": "short", "workload": r["split"], "text": r["input"],
             "answer": r["answer"], "family": r["family"]} for r in short]
    assert len(out) == 120 and sum(p["workload"] == "familiar" for p in out) == 30
    return out


def check_disjoint(train: Dict[str, List[dict]], test: List[dict]) -> dict:
    tr = {p["text"] for c in train.values() for p in c}
    te = {p["text"] for p in test}
    fam_train = {p["family"] for p in train["short"]}
    return {"train_test_text_overlap": len(tr & te), "short_train_families": sorted(fam_train),
            "note": "shifted test templates are distinct wordings of the training families "
                    "(held-out template families for format transfer)"}


# ── sampling ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_one(model, prompt_ids: Sequence[int], max_new_tokens: int, eos_ids=QWEN_EOS_IDS,
               device="cuda") -> dict:
    """Plain autoregressive temperature-1 sampling for one sequence (cache discipline of
    speculative.target_only_generate).  Returns {tokens, termination}."""
    from .speculative import target_only_generate
    st = target_only_generate(model, prompt_ids, max_new_tokens, eos_ids, device)
    return {"tokens": st.tokens, "termination": "eos" if st.eos else "cap"}


@torch.no_grad()
def padded_prefill(model, prompts: Sequence[Sequence[int]], device="cuda", pad_id: int = 151643):
    """Left-padded batch prefill. Returns (cache, last-position fp32 logits (B, V),
    attention mask, position ids); row i's logits equal an unpadded forward of prompt i."""
    from transformers import DynamicCache
    B = len(prompts)
    Lmax = max(len(p) for p in prompts)
    ids = torch.full((B, Lmax), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros(B, Lmax, dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        ids[i, Lmax - len(p):] = torch.tensor(list(p), device=device)
        mask[i, Lmax - len(p):] = 1
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    cache = DynamicCache()
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache,
                use_cache=True, logits_to_keep=1)
    return cache, out.logits[:, -1].float(), mask, pos


@torch.no_grad()
def sample_batch(model, prompts: Sequence[Sequence[int]], max_new_tokens: int, eos_ids=QWEN_EOS_IDS,
                 device="cuda", pad_id: int = 151643) -> List[dict]:
    """Batched temperature-1 sampling with LEFT padding (attention mask + explicit
    position ids, so every row sees exactly its own prompt).  Rows stop at EOS."""
    B = len(prompts)
    cache, logits, mask, pos = padded_prefill(model, prompts, device, pad_id)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    toks = [[] for _ in range(B)]
    term = ["cap"] * B
    eos = torch.tensor(sorted(eos_ids), device=device) if eos_ids else None
    next_pos = pos[:, -1:] + 1
    for _ in range(max_new_tokens):
        y = torch.multinomial(torch.softmax(logits, -1), 1)          # (B, 1)
        for i in torch.nonzero(~done).flatten().tolist():
            toks[i].append(int(y[i]))
        if eos is not None:
            hit = torch.isin(y[:, 0], eos) & ~done
            for i in torch.nonzero(hit).flatten().tolist():
                term[i] = "eos"
            done = done | hit
        if bool(done.all()):
            break
        mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.long, device=device)], dim=1)
        out = model(input_ids=y, attention_mask=mask, position_ids=next_pos, past_key_values=cache,
                    use_cache=True)
        next_pos = next_pos + 1
        logits = out.logits[:, -1].float()
    return [{"tokens": t, "termination": tm} for t, tm in zip(toks, term)]


# ── answer extraction ─────────────────────────────────────────────────────────

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december")


def _num(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def gsm8k_gold(answer: str) -> Optional[float]:
    return _num(answer.split("####")[-1].strip())


def extract_math(text: str) -> Optional[float]:
    """Fixed rule: the number inside the last \\boxed{...} if present, else the last
    number in the response."""
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    src = boxed[-1] if boxed else text
    nums = NUM_RE.findall(src)
    return _num(nums[-1]) if nums else None


def short_format_ok(answer: str, response: str) -> bool:
    """Format compliance: the stripped response is a bare answer of the expected kind."""
    r = response.strip().strip(".").strip()
    if answer.isdigit():
        return r.isdigit()
    if answer.lower() in MONTHS:
        return r.lower() in MONTHS
    return bool(re.fullmatch(r"[A-Z]{2}", r))


def score_response(prompt: dict, text: str, termination: str) -> dict:
    if prompt["domain"] == "math":
        pred, gold = extract_math(text), gsm8k_gold(prompt["answer"])
        return {"correct": int(pred is not None and gold is not None and abs(pred - gold) < 1e-6),
                "extracted": pred, "unextractable": int(pred is None), "truncated": int(termination == "cap")}
    ans = prompt["answer"]
    strict = text.strip().strip(".").strip() == ans
    lenient = re.search(rf"(?<![A-Za-z0-9]){re.escape(ans)}(?![A-Za-z0-9])", text, re.I) is not None
    return {"correct": int(lenient), "correct_strict": int(strict), "format_compliant": int(short_format_ok(ans, text)),
            "truncated": int(termination == "cap")}
