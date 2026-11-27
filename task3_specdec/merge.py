"""
LoRA merge (PDF §3 step 6): merge each trained adapter into a FRESH FP16 0.5B backbone,
keep one merged checkpoint per (method, seed), and verify merged vs unmerged logits on a
fixed prompt batch (max |Δlogit|, mean KL, argmax agreement).  Merge time is recorded for
the end-to-end / deployment cost.
"""

import time
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F

from common.runio import write_json


def _sync(dev):
    if str(dev).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_base(model_id: str, revision, device, dtype=torch.float16):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype,
                                                attn_implementation="sdpa").to(device).eval()


@torch.no_grad()
def _logits(model, batch: List[List[int]], device) -> List[torch.Tensor]:
    return [model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0].float() for ids in batch]


def merge_adapter(adapter_dir: str, out_dir: str, model_id: str, revision, verify_batch: Sequence[List[int]],
                  device="cuda", load_base_fn=None, dtype=torch.float16) -> dict:
    """Returns verification + timing; saves the merged FP16 model to out_dir."""
    from peft import PeftModel
    load_base_fn = load_base_fn or (lambda: load_base(model_id, revision, device))
    _sync(device)
    t0 = time.perf_counter()
    base = load_base_fn()
    _sync(device)
    t_load = time.perf_counter() - t0
    peft_model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()
    unmerged = _logits(peft_model, verify_batch, device)
    _sync(device)
    t1 = time.perf_counter()
    merged = peft_model.merge_and_unload()
    _sync(device)
    t_merge = time.perf_counter() - t1
    merged = merged.to(dtype).eval()
    after = _logits(merged, verify_batch, device)
    max_abs, kls, agree, n = 0.0, [], 0, 0
    for a, b in zip(unmerged, after):
        max_abs = max(max_abs, float((a - b).abs().max()))
        la, lb = F.log_softmax(a, -1), F.log_softmax(b, -1)
        kls.append(float((la.exp() * (la - lb)).sum(-1).mean()))
        agree += int((a.argmax(-1) == b.argmax(-1)).sum())
        n += a.shape[0]
    t2 = time.perf_counter()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    t_save = time.perf_counter() - t2
    res = {"adapter_dir": adapter_dir, "merged_dir": out_dir, "dtype": str(dtype),
           "verify": {"n_prompts": len(verify_batch), "n_positions": n, "max_abs_logit_diff": max_abs,
                      "mean_kl_unmerged_to_merged": sum(kls) / len(kls), "argmax_agreement": agree / n},
           "seconds": {"load_fresh_backbone": t_load, "merge_and_unload": t_merge, "save": t_save}}
    del peft_model, merged, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res
