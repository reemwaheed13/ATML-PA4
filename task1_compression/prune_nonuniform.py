"""
P3: nonuniform Wanda pruning with a calibration-measured, block-level allocation.

Allocation unit
    One Transformer block (28 in Qwen2.5-1.5B).  All 7 eligible projections of block
    l share one target sparsity s_l.

Sensitivity (calibration set only, 128 C4 sequences)
    For each block l and each of its projections p, build the P2 Wanda mask at the
    probe sparsity s_probe (same score |W_ij| * ||X_:,j||_2, same per-row
    exact_row_prune_mask, same n_zero = int(s_probe * d_in)), set D_p = W_p*mask - W_p
    and measure the calibration output error through the projection's Gram matrix
    G_p = sum_t x_t x_t^T (inputs from the UNMODIFIED FP16 model):
        err(D) = tr(D G D^T) = ||D X^T||_F^2
        sensitivity_l = sum_p tr(D_p G_p D_p^T) / sum_p tr(W_p G_p W_p^T)
    s_probe = the target global sparsity S (0.50 or 0.75).

Allocation rule
    r_l  = rank-normalised sensitivity in [0, 1] (0 = least sensitive; ties averaged)
    s_l  = S + lam * (1 - 2 r_l)                 (robust blocks are pruned more)
    s_l <- clip(s_l + c, 0.10, 0.85), with a common offset c found by bisection so
           that sum_l s_l N_l = round(S * N_eligible)
    integerise: per projection a total zero count (largest remainder, so the GLOBAL
    count equals round(S * N_eligible) exactly and each projection stays inside
    [0.10, 0.85]); within a projection the per-row counts differ by at most 1.

Hyperparameter (development set only, 32 C4 sequences)
    lam in LAMBDA_GRID, chosen by lowest development PPL (all tokens but position 0).
    WikiText-2, C4 validation and HellaSwag are never used to choose anything.

Rejected alternative
    outlier_proxy_sensitivity(): OWL-style weight-only ratio max|W| / mean|W| per
    block.  It ignores the activations and the mask actually applied, so it does not
    measure what pruning does to the block's outputs; recorded for comparison only.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .baseline import CalibStats, enumerate_eligible_projections, exact_row_prune_mask


S_MIN = 0.10
S_MAX = 0.85
LAMBDA_GRID = [0.02, 0.05, 0.08, 0.12, 0.16]


# ── helpers ───────────────────────────────────────────────────────────────────

def _dev(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def block_index(name: str) -> int:
    return int(name.split(".layers.")[1].split(".")[0])


def gram_error(D: torch.Tensor, G: torch.Tensor) -> float:
    """tr(D G D^T) == ||D X^T||_F^2 when G = X^T X."""
    return float(((D @ G) * D).sum().item())


def wanda_scores_for(W: torch.Tensor, stats: CalibStats) -> torch.Tensor:
    """|W_ij| * ||X_:,j||_2 (fp32, on W's device) -- identical to P2."""
    return W.abs() * stats.col_sqnorm.to(W.device, torch.float32).unsqueeze(0)


def p2_zero_count(proj_shapes: Dict[str, Tuple[int, int]], S: float) -> int:
    """Eligible zeros produced by P2 (uniform Wanda) at sparsity S."""
    return sum(int(S * d_in) * d_out for d_out, d_in in proj_shapes.values())


def projection_shapes(model) -> Dict[str, Tuple[int, int]]:
    return {n: tuple(m.weight.shape) for n, m in enumerate_eligible_projections(model)}


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra, rb = torch.tensor(_avg_ranks(a)), torch.tensor(_avg_ranks(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = (ra.norm() * rb.norm()).item()
    return float((ra * rb).sum().item() / den) if den > 0 else float("nan")


def _avg_ranks(x: Sequence[float]) -> List[float]:
    """0-based ranks with ties averaged."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


# ── sensitivity ───────────────────────────────────────────────────────────────

@torch.no_grad()
def block_sensitivity(model, stats: Dict[str, CalibStats], s_probe: float,
                      device: Optional[str] = None) -> Dict[str, dict]:
    """
    Per-block Gram-based relative output error of the P2 Wanda mask at s_probe.
    Returns {"per_block": {l: sens}, "per_projection": {name: {err, ref, rel}}}.
    The model is not modified.
    """
    dev = _dev(device)
    num: Dict[int, float] = {}
    den: Dict[int, float] = {}
    per_proj = {}
    for name, mod in enumerate_eligible_projections(model):
        s = stats[name]
        if s.gram is None:
            raise ValueError("block_sensitivity needs with_gram=True calibration stats")
        W = mod.weight.data.to(dev, torch.float32)
        G = s.gram.to(dev, torch.float32)
        mask = exact_row_prune_mask(wanda_scores_for(W, s), int(s_probe * W.shape[1]))
        D = W * mask - W
        e, r = gram_error(D, G), gram_error(W, G)
        l = block_index(name)
        num[l] = num.get(l, 0.0) + e
        den[l] = den.get(l, 0.0) + r
        per_proj[name] = {"err": e, "ref": r, "rel": e / r if r > 0 else 0.0}
        del W, G, mask, D
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    per_block = {l: (num[l] / den[l] if den[l] > 0 else 0.0) for l in sorted(num)}
    return {"per_block": per_block, "per_projection": per_proj, "s_probe": s_probe,
            "definition": "sum_p tr(D_p G_p D_p^T) / sum_p tr(W_p G_p W_p^T), D_p = W_p*mask_p - W_p, "
                          "mask_p = P2 Wanda mask at s_probe, G_p from the unmodified FP16 model"}


@torch.no_grad()
def outlier_proxy_sensitivity(model) -> Dict[int, float]:
    """REJECTED alternative (OWL-style, weight only): mean over the block's
    projections of max|W| / mean|W|.  Recorded for comparison, never used."""
    acc: Dict[int, List[float]] = {}
    for name, mod in enumerate_eligible_projections(model):
        W = mod.weight.data.float().abs()
        acc.setdefault(block_index(name), []).append(W.max().item() / (W.mean().item() + 1e-8))
    return {l: sum(v) / len(v) for l, v in sorted(acc.items())}


# ── allocation ────────────────────────────────────────────────────────────────

def allocate(sens: Dict[int, float], S: float, lam: float,
             proj_shapes: Dict[str, Tuple[int, int]],
             s_min: float = S_MIN, s_max: float = S_MAX,
             target_zeros: Optional[int] = None) -> dict:
    """
    sens         block -> sensitivity (higher = more sensitive)
    proj_shapes  projection name -> (d_out, d_in), every eligible projection

    Returns {"s_block", "offset", "proj_zeros", "row_counts", "target_zeros",
             "achieved_zeros", "rank", "s_raw"}; row_counts[name] = (base, n_extra):
    n_extra rows get base+1 zeros, the rest base.
    """
    blocks = sorted(sens)
    L = len(blocks)
    N_l = {l: 0 for l in blocks}
    for n, (o, i) in proj_shapes.items():
        N_l[block_index(n)] += o * i
    N = sum(N_l.values())
    if target_zeros is None:
        target_zeros = int(round(S * N))
    if not (s_min * N <= target_zeros <= s_max * N):
        raise ValueError("target sparsity outside the permitted [s_min, s_max] range")

    ranks = _avg_ranks([sens[l] for l in blocks])
    r = {l: (ranks[k] / (L - 1) if L > 1 else 0.5) for k, l in enumerate(blocks)}
    s_raw = {l: S + lam * (1.0 - 2.0 * r[l]) for l in blocks}

    def total(c):
        return sum(min(s_max, max(s_min, s_raw[l] + c)) * N_l[l] for l in blocks)

    # Bracket wide enough that every block clamps to s_min at lo and s_max at hi,
    # since s_raw ranges over [S - |lam|, S + |lam|] with S in [0, 1].
    lo, hi = -(1.0 + abs(lam)), 1.0 + abs(lam)
    assert total(lo) <= target_zeros <= total(hi), "bisection bracket does not contain target"
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < target_zeros:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    s_block = {l: min(s_max, max(s_min, s_raw[l] + c)) for l in blocks}

    # Integerise per projection with largest remainder, inside [s_min, s_max].
    names = list(proj_shapes)
    want, fl, lo_b, hi_b = {}, {}, {}, {}
    for n in names:
        o, i = proj_shapes[n]
        sz = o * i
        want[n] = s_block[block_index(n)] * sz
        lo_b[n] = math.ceil(s_min * sz)
        hi_b[n] = math.floor(s_max * sz)
        fl[n] = min(hi_b[n], max(lo_b[n], math.floor(want[n])))
    deficit = target_zeros - sum(fl.values())
    if deficit > 0:
        cand = sorted((n for n in names if fl[n] < hi_b[n]),
                      key=lambda n: (-(want[n] - fl[n]), names.index(n)))
        while deficit > 0:
            if not cand:
                raise RuntimeError("cannot integerise allocation within bounds")
            progressed = False
            for n in cand:
                if deficit == 0:
                    break
                if fl[n] < hi_b[n]:
                    fl[n] += 1
                    deficit -= 1
                    progressed = True
            if not progressed:
                raise RuntimeError("cannot integerise allocation within bounds")
    elif deficit < 0:
        cand = sorted((n for n in names if fl[n] > lo_b[n]),
                      key=lambda n: ((want[n] - fl[n]), names.index(n)))
        while deficit < 0:
            progressed = False
            for n in cand:
                if deficit == 0:
                    break
                if fl[n] > lo_b[n]:
                    fl[n] -= 1
                    deficit += 1
                    progressed = True
            if not progressed:
                raise RuntimeError("cannot integerise allocation within bounds")

    row_counts = {}
    for n in names:
        o, _ = proj_shapes[n]
        row_counts[n] = (fl[n] // o, fl[n] % o)
    achieved = sum(fl.values())
    assert achieved == target_zeros, (achieved, target_zeros)
    return {"s_block": s_block, "s_raw": s_raw, "rank": r, "offset": c, "lam": lam, "S": S,
            "proj_zeros": fl, "row_counts": row_counts, "target_zeros": target_zeros,
            "achieved_zeros": achieved, "n_eligible": N}


def variable_row_mask(scores: torch.Tensor, base: int, n_extra: int) -> torch.Tensor:
    """Keep-mask with `base` zeros in every row and one more zero in `n_extra` rows:
    the rows whose (base+1)-th smallest score is smallest.  Built from the same
    exact_row_prune_mask as P2."""
    if n_extra == 0:
        return exact_row_prune_mask(scores, base)
    kth = torch.kthvalue(scores, base + 1, dim=1).values
    extra_rows = torch.topk(kth, n_extra, largest=False, sorted=False).indices
    is_extra = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    is_extra[extra_rows] = True
    mask = torch.empty_like(scores, dtype=torch.bool)
    mask[~is_extra] = exact_row_prune_mask(scores[~is_extra], base)
    mask[is_extra] = exact_row_prune_mask(scores[is_extra], base + 1)
    return mask


@torch.no_grad()
def apply_allocation(model, alloc: dict, stats: Dict[str, CalibStats],
                     originals: Optional[Dict[str, torch.Tensor]] = None,
                     device: Optional[str] = None, keep_masks: bool = True) -> Dict[str, dict]:
    """Prune every eligible projection to its allocated counts.  If `originals`
    (CPU FP16 copies) is given, weights are first restored from it, so repeated calls
    always start from the unmodified FP16 weights.  keep_masks=False skips copying
    the boolean masks to CPU (used during the development grid)."""
    dev = _dev(device)
    out = {}
    for name, mod in enumerate_eligible_projections(model):
        if originals is not None:
            mod.weight.data.copy_(originals[name].to(mod.weight.device))
        W = mod.weight.data.to(dev, torch.float32)
        base, extra = alloc["row_counts"][name]
        mask = variable_row_mask(wanda_scores_for(W, stats[name]), base, extra)
        mod.weight.data = (W * mask).to(mod.weight.dtype).to(mod.weight.device)
        zeros = int((~mask).sum().item())
        assert zeros == alloc["proj_zeros"][name], (name, zeros, alloc["proj_zeros"][name])
        out[name] = {"mask": mask.cpu() if keep_masks else None, "zeros": zeros, "achieved_sparsity": zeros / mask.numel(),
                     "block_sparsity_target": alloc["s_block"][block_index(name)]}
        del W
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


# ── development-set selection ────────────────────────────────────────────────

@torch.no_grad()
def select_lambda(model, stats: Dict[str, CalibStats], S: float, dev_ids: List[List[int]],
                  originals: Dict[str, torch.Tensor], device: Optional[str] = None,
                  grid: Sequence[float] = LAMBDA_GRID, ppl_dtype=torch.float16,
                  batch_size: int = 1, sens: Optional[dict] = None) -> dict:
    """
    For each lam in grid: allocate, apply masks to the FP16 weights restored from
    the CPU copy `originals`, and score development PPL.  Picks the lowest dev PPL
    (ties -> smaller lam).  The model is left pruned at the chosen lam.
    """
    from common.eval_perplexity import perplexity_on_token_ids

    dev = _dev(device)
    shapes = projection_shapes(model)
    if sens is None:
        sens = block_sensitivity(model, stats, S, dev)
    target = int(round(S * sum(o * i for o, i in shapes.values())))
    p2 = p2_zero_count(shapes, S)
    assert target == p2, f"round(S*N_eligible)={target} != P2 zero count {p2}"

    points = []
    for lam in grid:
        alloc = allocate(sens["per_block"], S, lam, shapes)
        apply_allocation(model, alloc, stats, originals, dev, keep_masks=False)
        r = perplexity_on_token_ids(model, dev_ids, dev, dtype=ppl_dtype, batch_size=batch_size)
        points.append({"lam": lam, "dev_ppl": r["ppl"], "dev_mean_nll": r["mean_nll"], "dev_N": r["N"],
                       "s_block": {str(l): v for l, v in alloc["s_block"].items()},
                       "offset": alloc["offset"], "achieved_zeros": alloc["achieved_zeros"]})
    best = min(points, key=lambda p: (p["dev_ppl"], p["lam"]))
    alloc = allocate(sens["per_block"], S, best["lam"], shapes)
    masks = apply_allocation(model, alloc, stats, originals, dev)
    zeros = sum(v["zeros"] for v in masks.values())
    assert zeros == p2, f"P3 zeros {zeros} != P2 zeros {p2}"
    return {"lam": best["lam"], "grid": points, "alloc": alloc, "masks": masks, "sens": sens,
            "p2_zero_count": p2, "achieved_zeros": zeros}
