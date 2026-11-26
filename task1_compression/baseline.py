"""
FP16 baseline: load, enumerate eligible projections, verify tied-weight counting.

Eligible projections per Transformer block (PDF §1 "Dataset, Models, and Experimental Setup"):
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj  → 7 per block.

Excluded from compression:
  embed_tokens, norm layers, biases, lm_head.
  Shared/tied parameters counted once (embed_tokens.weight == lm_head.weight in Qwen2.5).
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ELIGIBLE_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def load_fp16_baseline(
    model_id: str = "Qwen/Qwen2.5-1.5B",
    revision: str = None,
    device: str = "cuda:0",
) -> Tuple:
    """
    Load a fresh FP16 copy of the model and its tokenizer.

    Returns (model, tokenizer).
    Sets model to eval mode; does NOT call .half() — model is loaded in fp16 via
    torch_dtype=torch.float16 so weights are genuinely fp16, not cast post-load.
    """
    kwargs = dict(
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    if revision:
        kwargs["revision"] = revision

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    model.eval()
    return model, tokenizer


def enumerate_eligible_projections(model) -> List[Tuple[str, torch.nn.Linear]]:
    """
    Return [(full_param_name, module)] for every eligible projection.

    Rules:
      - Only nn.Linear whose name ends with one of ELIGIBLE_SUFFIXES.
      - Tied parameters (lm_head.weight is model.embed_tokens.weight) are excluded
        from the eligible set; the tie is detected by id() equality.
      - Does NOT include embedding or lm_head modules.

    The returned list preserves model order (layer 0 → layer N-1 × 7 projections).
    """
    # Collect tied weight ids to skip.
    tied_ids = set()
    embed = getattr(model, "model", model)
    embed_w = getattr(getattr(embed, "embed_tokens", None), "weight", None)
    lm_head_w = getattr(getattr(model, "lm_head", None), "weight", None)
    if embed_w is not None and lm_head_w is not None and id(embed_w) == id(lm_head_w):
        tied_ids.add(id(lm_head_w))

    eligible = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.endswith(ELIGIBLE_SUFFIXES):
            continue
        if id(module.weight) in tied_ids:
            continue
        eligible.append((name, module))

    return eligible


def count_eligible_params(model) -> Dict[str, int]:
    """
    Count scalar parameters in eligible projections.
    Returns dict with per-projection-type counts and a total.
    Tied parameters counted once.
    """
    seen_ids = set()
    counts: Dict[str, int] = {s: 0 for s in ELIGIBLE_SUFFIXES}
    total = 0
    for name, module in enumerate_eligible_projections(model):
        if id(module.weight) in seen_ids:
            continue
        seen_ids.add(id(module.weight))
        suffix = next(s for s in ELIGIBLE_SUFFIXES if name.endswith(s))
        n = module.weight.numel()
        counts[suffix] += n
        total += n
    counts["total"] = total
    return counts


def save_layer_output_error(
    model,
    calibration_ids: List[List[int]],
    baseline_outputs: Dict[str, torch.Tensor],
    device: str,
    result_path: str,
):
    """
    Measure per-projection output error on calibration activations versus a
    reference (the unchanged FP16 model's saved outputs).

    baseline_outputs: dict mapping projection name → reference output tensor (CPU).
    Returns dict mapping name → {"mse": float, "cos_sim": float}.
    """
    errors = {}
    hooks = []
    actual = {}

    def make_hook(name):
        def hook(module, inp, out):
            actual[name] = out.detach().cpu()
        return hook

    for name, module in enumerate_eligible_projections(model):
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for ids in calibration_ids:
            t = torch.tensor([ids], dtype=torch.long, device=device)
            model(t)

    for h in hooks:
        h.remove()

    for name in baseline_outputs:
        if name not in actual:
            continue
        ref = baseline_outputs[name].float()
        cur = actual[name].float()
        mse = ((ref - cur) ** 2).mean().item()
        cos = torch.nn.functional.cosine_similarity(
            ref.reshape(-1), cur.reshape(-1), dim=0
        ).item()
        errors[name] = {"mse": mse, "cos_sim": cos}

    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as fh:
        json.dump(errors, fh, indent=2)
    return errors
