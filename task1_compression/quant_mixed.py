"""
Q3: mixed-precision weight-only PTQ at the Q2 uniform budgets.

Quantizer
    Exactly the Q2 quantizer from quant_awq.py, called unchanged:
    awq_scale_search (per-projection exponent grid, Gram-based error) followed by
    awq_compensated_weight (group-128 asymmetric codes of W*diag(alpha), serialized
    FP16 compensating scales, no folding).  Only the bit width differs per projection.
    Consequence: an all-b allocation reproduces Q2-INTb bit for bit.

Allocation unit
    Individual eligible projection (196 in Qwen2.5-1.5B).  Allowed bits {2, 3, 4, 8}.

Sensitivity table (calibration set only)
    For each projection p and bit width b: alpha_b = awq_scale_search(W, stats, b),
    D = W_hat_b - W, rel_err[p][b] = tr(D G D^T) / tr(W G W^T), with G the Gram of the
    projection's inputs on the UNMODIFIED FP16 model, and
        bytes[p][b] = ceil(n*b/8) + ceil(d_in/128)*4*d_out + 2*d_in
    (packed codes + FP16 scale/zero per group + FP16 compensating scales).

Allocation (greedy marginal efficiency)
    Start every projection at b_min; repeatedly take the single upgrade (to any
    higher allowed bit) with the largest (rel-error decrease)/(byte increase) that
    keeps the TOTAL within budget; stop when no upgrade fits.
    Budget = TOTAL bytes of Q2 uniform INT4 / INT3, including metadata, compensating
    scales and the FP16 uncompressed parameters (tied embedding counted once).  The
    Q3 total includes the same fixed bytes.

Hyperparameter (development set only)
    b_min in {2, 3}: both allocations are quantized from the FP16 weights and the
    lower development PPL wins.
"""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .baseline import CalibStats, enumerate_eligible_projections
from .quant_awq import SCALE_EXPONENTS, awq_compensated_weight, awq_scale_search
from .prune_nonuniform import gram_error


ALLOWED_BITS = [2, 3, 4, 8]
BMIN_GRID = [2, 3]
GROUP_SIZE = 128


def _dev(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def projection_bytes(d_out: int, d_in: int, bits: int, group_size: int = GROUP_SIZE) -> int:
    return math.ceil(d_out * d_in * bits / 8) + math.ceil(d_in / group_size) * 4 * d_out + 2 * d_in


@torch.no_grad()
def bit_error_table(model, stats: Dict[str, CalibStats], bits: Sequence[int] = ALLOWED_BITS,
                    group_size: int = GROUP_SIZE, device: Optional[str] = None,
                    cache_path: Optional[str] = None, cache_meta: Optional[dict] = None) -> dict:
    """
    {"meta", "build_seconds", "table": {name: {b: {rel_err, err, bytes, best_exp, alpha}}}}
    Loaded from `cache_path` when its meta matches `cache_meta`; otherwise built on
    `device` from the (unmodified) model weights and saved there.
    """
    dev = _dev(device)
    meta = dict(cache_meta or {})
    meta.update({"bits": list(bits), "group_size": group_size, "exponents": list(SCALE_EXPONENTS)})
    if cache_path and Path(cache_path).exists():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            if payload.get("meta") == meta:
                print(f"[q3] reusing cached bit-error table {cache_path}")
                return payload
            print("[q3] table cache metadata mismatch; rebuilding")
        except Exception as e:
            print(f"[q3] could not load table cache ({e}); rebuilding")

    t0 = time.perf_counter()
    table = {}
    for name, mod in enumerate_eligible_projections(model):
        s = stats[name]
        if s.gram is None:
            raise ValueError("bit_error_table needs with_gram=True calibration stats")
        W = mod.weight.data.to(dev, torch.float32)
        G = s.gram.to(dev, torch.float32)
        ref = gram_error(W, G)
        d_out, d_in = W.shape
        row = {}
        for b in bits:
            alpha, best_exp, _ = awq_scale_search(mod.weight.data, s, b, group_size, SCALE_EXPONENTS, dev)
            W_hat, *_ = awq_compensated_weight(W, alpha.to(dev), b, group_size)
            e = gram_error(W_hat - W, G)
            row[int(b)] = {"rel_err": e / ref if ref > 0 else 0.0, "err": e,
                           "bytes": projection_bytes(d_out, d_in, b, group_size),
                           "best_exp": float(best_exp), "alpha": alpha.cpu()}
            del W_hat
        table[name] = row
        del W, G
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    payload = {"meta": meta, "build_seconds": time.perf_counter() - t0, "table": table}
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = str(cache_path) + ".tmp"
        torch.save(payload, tmp)
        Path(tmp).replace(cache_path)
        print(f"[q3] cached bit-error table to {cache_path}")
    return payload


def allocate(table: Dict[str, Dict[int, dict]], budget: float, b_min: int,
             fixed_bytes: int = 0, allowed_bits: Sequence[int] = ALLOWED_BITS) -> dict:
    """
    Greedy: start at b_min; apply the single feasible upgrade with the largest
    (rel_err decrease)/(byte increase) until none fits.  `budget` is a TOTAL byte
    budget; `fixed_bytes` (FP16 uncompressed params) is counted inside it.
    """
    bits_sorted = sorted(int(b) for b in allowed_bits)
    assign = {n: int(b_min) for n in table}
    total = fixed_bytes + sum(table[n][b_min]["bytes"] for n in table)
    if total > budget:
        raise ValueError(f"b_min={b_min} alone ({total} B) exceeds the budget ({budget} B)")
    steps = 0
    while True:
        best = None
        for n, row in table.items():
            cur = assign[n]
            for b in bits_sorted:
                if b <= cur:
                    continue
                db = row[b]["bytes"] - row[cur]["bytes"]
                if total + db > budget:
                    continue
                de = row[cur]["rel_err"] - row[b]["rel_err"]
                ratio = de / db if db > 0 else float("inf")
                key = (ratio, -db)
                if best is None or key > best[0]:
                    best = (key, n, b, db)
        if best is None:
            break
        _, n, b, db = best
        assign[n] = b
        total += db
        steps += 1
    assert total <= budget
    err = sum(table[n][assign[n]]["rel_err"] for n in table)
    return {"bits": assign, "total_bytes": total, "budget_bytes": budget, "fixed_bytes": fixed_bytes,
            "b_min": int(b_min), "n_upgrades": steps, "sum_rel_err": err}


@torch.no_grad()
def quantize_with_allocation(model, table: Dict[str, Dict[int, dict]], bits: Dict[str, int],
                             originals: Optional[Dict[str, torch.Tensor]] = None,
                             group_size: int = GROUP_SIZE, device: Optional[str] = None) -> Dict[str, dict]:
    """Quantize each projection with the Q2 quantizer at its assigned bit width,
    reusing the alpha searched (by awq_scale_search) for that bit width."""
    dev = _dev(device)
    out = {}
    for name, mod in enumerate_eligible_projections(model):
        if originals is not None:
            mod.weight.data.copy_(originals[name].to(mod.weight.device))
        b = int(bits[name])
        W = mod.weight.data.to(dev, torch.float32)
        d_out, d_in = W.shape
        alpha = table[name][b]["alpha"].to(dev, torch.float32)
        W_hat, *_ = awq_compensated_weight(W, alpha, b, group_size)
        mod.weight.data = W_hat.to(mod.weight.dtype).to(mod.weight.device)
        payload = math.ceil(d_out * d_in * b / 8)
        meta = math.ceil(d_in / group_size) * 4 * d_out
        comp = 2 * d_in
        out[name] = {"bits": b, "packed_bytes": payload, "metadata_bytes": meta,
                     "compensating_scale_bytes": comp, "total_bytes": payload + meta + comp,
                     "best_exp": table[name][b]["best_exp"], "rel_err": table[name][b]["rel_err"]}
        del W, W_hat, alpha
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def select_bmin(model, table: Dict[str, Dict[int, dict]], budget: int, fixed_bytes: int,
                dev_ids: List[List[int]], originals: Dict[str, torch.Tensor],
                device: Optional[str] = None, grid: Sequence[int] = BMIN_GRID,
                ppl_dtype=torch.float16, batch_size: int = 1) -> dict:
    """Allocate for each b_min, quantize from the FP16 weights, score development
    PPL; keep the lower (ties -> larger b_min).  The model is left quantized with the
    chosen allocation."""
    from common.eval_perplexity import perplexity_on_token_ids

    dev = _dev(device)
    points, allocs = [], {}
    for bm in grid:
        try:
            a = allocate(table, budget, bm, fixed_bytes)
        except ValueError as e:
            points.append({"b_min": bm, "feasible": False, "reason": str(e)})
            continue
        quantize_with_allocation(model, table, a["bits"], originals, device=dev)
        r = perplexity_on_token_ids(model, dev_ids, dev, dtype=ppl_dtype, batch_size=batch_size)
        allocs[bm] = a
        points.append({"b_min": bm, "feasible": True, "dev_ppl": r["ppl"], "dev_mean_nll": r["mean_nll"],
                       "dev_N": r["N"], "total_bytes": a["total_bytes"], "sum_rel_err": a["sum_rel_err"],
                       "n_upgrades": a["n_upgrades"],
                       "bit_histogram": {str(b): sum(v == b for v in a["bits"].values()) for b in ALLOWED_BITS}})
    feas = [p for p in points if p["feasible"]]
    if not feas:
        raise ValueError("no feasible b_min")
    best = min(feas, key=lambda p: (p["dev_ppl"], -p["b_min"]))
    alloc = allocs[best["b_min"]]
    res = quantize_with_allocation(model, table, alloc["bits"], originals, device=dev)
    return {"b_min": best["b_min"], "grid": points, "alloc": alloc, "per_projection": res}
