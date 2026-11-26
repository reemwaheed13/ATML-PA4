"""
FP16 baseline: load, enumerate eligible projections, verify tied-weight counting.

Eligible projections per Transformer block (PDF §1 "Dataset, Models, and Experimental Setup"):
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj  → 7 per block.

Excluded from compression:
  embed_tokens, norm layers, biases, lm_head.
  Shared/tied parameters counted once (embed_tokens.weight == lm_head.weight in Qwen2.5).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
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


def exact_row_prune_mask(scores: torch.Tensor, n_zero: int) -> torch.Tensor:
    """
    Boolean keep-mask (True = keep) that zeros EXACTLY ``n_zero`` entries per row:
    the ``n_zero`` smallest scores in each row.

    This replaces the previous threshold-plus-tie-fixup logic, which over-pruned
    whenever the per-row threshold value was shared by more columns than needed
    (e.g. dead input channels scoring exactly 0): ``scores > threshold`` zeroed
    *all* ties, and the fix-up branch (which only ever *added* zeros) never ran
    because its counter was already ≤ 0.  The result was achieved-sparsity drift,
    which breaks the matched-sparsity P1/P2/P3 comparison required by PDF §1
    ("What to Watch For": eligible-weight sparsity of P3 must equal P1/P2).

    ``torch.topk(..., largest=False)`` selects exactly ``n_zero`` indices per row,
    so the achieved per-row sparsity is exactly ``n_zero / d_in`` regardless of ties.
    """
    d_out, d_in = scores.shape
    if n_zero <= 0:
        return torch.ones(d_out, d_in, dtype=torch.bool, device=scores.device)
    if n_zero >= d_in:
        return torch.zeros(d_out, d_in, dtype=torch.bool, device=scores.device)
    _, zero_idx = torch.topk(scores, n_zero, dim=1, largest=False, sorted=False)
    mask = torch.ones(d_out, d_in, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, zero_idx, False)
    return mask


class CalibStats:
    """
    Memory-efficient sufficient statistics of the calibration inputs to ONE
    projection, replacing storage of the full (n_tokens, d_in) activation matrix.

    Storing raw activations for every eligible projection at the PDF §1 calibration
    size (128 seqs × 512 tokens ≈ 65k rows) costs ~Σ d_in × 65k × 4 B — well over
    100 GB for Qwen2.5-1.5B and a guaranteed OOM.  Every downstream consumer only
    needs a reduction over the token axis, so we accumulate those reductions online:

      col_sqnorm  (d_in,)       sqrt(Σ_t x_t²)  == Wanda column norm ||X_{:,j}||₂
      col_absmean (d_in,)       (1/N) Σ_t |x_t|  (AWQ activation scale / Q3 sensitivity)
      gram        (d_in, d_in)  Σ_t x_t x_tᵀ     (exact AWQ reconstruction error) or None
      n_tokens    int

    The Gram matrix is an exact sufficient statistic for the AWQ Frobenius error
    (‖X Dᵀ‖_F² = tr(D·gram·Dᵀ)); it is only needed for quantization, so pruning
    collects with ``with_gram=False`` (O(d_in) memory per projection).
    """

    __slots__ = ("col_sqnorm", "col_absmean", "gram", "n_tokens")

    def __init__(self, col_sqnorm, col_absmean, gram, n_tokens):
        self.col_sqnorm = col_sqnorm
        self.col_absmean = col_absmean
        self.gram = gram
        self.n_tokens = n_tokens


@torch.no_grad()
def collect_calibration_stats(
    model,
    calibration_token_ids: List[List[int]],
    device: str,
    with_gram: bool = True,
) -> Dict[str, "CalibStats"]:
    """
    Run the calibration sequences once and accumulate per-projection sufficient
    statistics (see :class:`CalibStats`) instead of the full activation matrices.

    with_gram  If True, also accumulate the d_in×d_in Gram matrix per projection
               (needed by AWQ).  For Wanda / sensitivity probes pass False to keep
               memory at O(d_in) per projection.

    Statistics are accumulated on CPU in fp32; the per-batch Gram is formed on the
    activation's device and moved to CPU, so peak GPU memory stays bounded.
    """
    accum: Dict[str, dict] = {}
    hooks = []

    for name, module in enumerate_eligible_projections(model):
        accum[name] = {"sqsum": None, "abssum": None, "gram": None, "n": 0}

        def make_hook(n):
            def hook(mod, inp, out):
                x = inp[0]
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                x = x.float()
                a = accum[n]
                sq = (x * x).sum(dim=0).cpu()
                ab = x.abs().sum(dim=0).cpu()
                a["sqsum"] = sq if a["sqsum"] is None else a["sqsum"] + sq
                a["abssum"] = ab if a["abssum"] is None else a["abssum"] + ab
                a["n"] += x.shape[0]
                if with_gram:
                    g = (x.t() @ x).cpu()
                    a["gram"] = g if a["gram"] is None else a["gram"] + g
            return hook

        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    for ids in calibration_token_ids:
        t = torch.tensor([ids], dtype=torch.long, device=device)
        model(t)

    for h in hooks:
        h.remove()

    stats: Dict[str, CalibStats] = {}
    for name, a in accum.items():
        n = max(a["n"], 1)
        stats[name] = CalibStats(
            col_sqnorm=a["sqsum"].sqrt(),
            col_absmean=a["abssum"] / n,
            gram=a["gram"],
            n_tokens=a["n"],
        )
    return stats


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
