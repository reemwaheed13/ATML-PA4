"""
Shared HellaSwag evaluator.

Scoring convention (PDF §1 "Quality evaluation", lm-evaluation-harness acc_norm):
  - For each of the 4 candidate endings, compute the conditional log-likelihood of
    the continuation given the shared context.
  - raw_acc : argmax(loglik) == label
  - acc_norm: argmax(loglik / len(ending)) == label, where len is the NUMBER OF
    CHARACTERS of the ending string (PDF: "do not normalize by tokenizer token count").

Token boundary
--------------
Context and continuation are tokenized SEPARATELY and concatenated:

    ctx_ids  = tok(context,      add_special_tokens=False)
    cont_ids = tok(" " + ending, add_special_tokens=False)
    input    = ctx_ids + cont_ids

and exactly the len(cont_ids) continuation positions are scored.  The previous
version tokenized ``context + " "`` to find the boundary; Qwen's BPE gives that
trailing space its own token, whereas in the full string the space merges into the
ending's first token, so the first ending token was silently never scored.

The four endings of one example are scored in ONE right-padded batched forward
pass with an attention mask (right padding never affects earlier positions under
a causal mask).

Prompt format (recorded in HELLASWAG_FORMAT): context = activity_label + ": " + ctx,
continuation = " " + ending.  No further text preprocessing is applied (PDF §1
does not specify any).
"""

import gzip
import json
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


HELLASWAG_FORMAT = {
    "context": "activity_label + ': ' + ctx  (fallback ctx_a + ' ' + ctx_b if ctx is empty)",
    "continuation": "' ' + ending",
    "tokenization": "tok(context) + tok(' ' + ending), add_special_tokens=False, "
                    "tokenized separately and concatenated; every continuation token scored once",
    "text_preprocessing": "none",
    "metrics": "raw_acc = argmax sum loglik; acc_norm = argmax loglik / len(ending in characters)",
    "shots": 0,
}


def _iter_records(path: str):
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_context(example: dict) -> str:
    """Context string: activity_label + ": " + ctx."""
    ctx = example.get("ctx", "")
    if not ctx:
        ctx = (example.get("ctx_a", "") + " " + example.get("ctx_b", "")).strip()
    activity = example.get("activity_label", "")
    if activity:
        return activity + ": " + ctx
    return ctx


def _ids(tokenizer, text: str):
    out = tokenizer(text, add_special_tokens=False)
    ids = out["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return list(ids)


@torch.no_grad()
def score_example(model, tokenizer, context: str, endings, device: str, pad_id: int = 0):
    """
    Score all endings of one example in a single batched forward pass.

    Returns (raw_scores, norm_scores, n_cont_tokens): lists of length len(endings).
    """
    ctx_ids = _ids(tokenizer, context)
    cont_list = [_ids(tokenizer, " " + e) for e in endings]
    seqs = [ctx_ids + c for c in cont_list]
    max_len = max(len(s) for s in seqs)

    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    input_ids = input_ids.to(device)
    attn = attn.to(device)

    logits = model(input_ids=input_ids, attention_mask=attn).logits  # (4, L, V)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    n_ctx = len(ctx_ids)
    raw_scores, norm_scores, n_cont = [], [], []
    for i, (ending, cont) in enumerate(zip(endings, cont_list)):
        # Continuation token k sits at position n_ctx + k and is predicted by the
        # logits at position n_ctx + k - 1.
        pos = torch.arange(n_ctx, n_ctx + len(cont), device=log_probs.device)
        tgt = input_ids[i, pos]
        ll = log_probs[i, pos - 1].gather(-1, tgt.unsqueeze(-1)).sum().item()
        raw_scores.append(ll)
        norm_scores.append(ll / len(ending) if len(ending) > 0 else ll)
        n_cont.append(len(cont))
    return raw_scores, norm_scores, n_cont


@torch.no_grad()
def evaluate_hellaswag(
    model,
    tokenizer,
    locked_hellaswag_path: str,
    device: str,
    dtype=torch.float16,
    save_path: str = None,
    subset_ids: Optional[Iterable[int]] = None,
) -> dict:
    """
    Args
    ----
    model                 HF CausalLM already on `device`, in eval mode.
    tokenizer             Matching tokenizer.
    locked_hellaswag_path Path to hellaswag_val_locked.jsonl(.gz).
    device                e.g. "cuda:0".
    dtype                 Unused (kept for API compatibility; the model's own dtype is used).
    save_path             If given, save per-example scores as JSON.
    subset_ids            If given, only evaluate records whose `order` is in this set
                          (e.g. range(512) for the diagnostic subset).

    Returns
    -------
    dict: n_examples, raw_acc, acc_norm, format, per_example (list of dicts with
    order, label, raw_scores, norm_scores, n_cont_tokens, raw_pred, norm_pred).
    """
    model.eval()
    subset = set(subset_ids) if subset_ids is not None else None
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0

    per_example = []
    n_correct_raw = 0
    n_correct_norm = 0

    for rec in _iter_records(locked_hellaswag_path):
        order = rec["order"]
        if subset is not None and order not in subset:
            continue
        example = rec["example"]
        label = int(example["label"])
        context = _build_context(example)
        endings = example["endings"]

        raw_scores, norm_scores, n_cont = score_example(
            model, tokenizer, context, endings, device, pad_id
        )
        raw_pred = int(max(range(len(raw_scores)), key=lambda k: raw_scores[k]))
        norm_pred = int(max(range(len(norm_scores)), key=lambda k: norm_scores[k]))
        n_correct_raw += int(raw_pred == label)
        n_correct_norm += int(norm_pred == label)

        per_example.append({
            "order": order,
            "label": label,
            "raw_scores": raw_scores,
            "norm_scores": norm_scores,
            "n_cont_tokens": n_cont,
            "raw_pred": raw_pred,
            "norm_pred": norm_pred,
        })

    n_total = len(per_example)
    result = {
        "n_examples": n_total,
        "raw_acc": n_correct_raw / n_total if n_total else 0.0,
        "acc_norm": n_correct_norm / n_total if n_total else 0.0,
        "format": HELLASWAG_FORMAT,
        "per_example": per_example,
    }

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            json.dump(result, fh)

    return result
