"""
Task 1 driver: run every PDF §1 compression condition end-to-end, resumably.

Usage (Colab):
    python -m task1_compression.run_task1 run --conditions overnight
    python -m task1_compression.run_task1 run --conditions fp16
    python -m task1_compression.run_task1 aggregate

Conditions (PDF §1 steps 1-8, 18 total):
    fp16                          unchanged FP16 baseline
    p1_s25 / p1_s50 / p1_s75      magnitude pruning
    p2_s25 / p2_s50 / p2_s75      Wanda pruning
    p3_s50 / p3_s75               nonuniform Wanda (DRAFT design, needs --allow-draft-p3q3)
    p2_24                         2:4 Wanda structured diagnostic at 50%
    q1_int8 / q1_int4 / q1_int3   uniform group-wise asymmetric PTQ (group 128)
    q2_int8 / q2_int4 / q2_int3   AWQ-style PTQ (serialized FP16 compensating scales)
    q3_int4 / q3_int3             mixed precision at the Q2 INT4 / INT3 packed budgets (DRAFT)

Full 2,000-example HellaSwag: fp16, the three 50%-sparsity pruning methods
(p1_s50, p2_s50, p3_s50) and all INT4/INT3 quantization variants; every other
condition uses the first 512 IDs (orders 0..511) of the locked manifest.

Per condition the driver loads a FRESH FP16 model (pinned revision), seeds 6304,
compresses (timed), measures calibration output error against the unmodified
weights, evaluates both perplexity suites and HellaSwag, measures dense-FP16
execution latency and peak allocated GPU memory, computes labelled storage
(dense / theoretical sparse / packed), and writes
    <results_root>/<cond>/{summary.json, ppl_c4_per_window.json,
                           ppl_wt2_per_window.json, hellaswag_per_example.json}
summary.json is written last (atomically), so its existence marks completion;
completed conditions are skipped unless --overwrite.
"""

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.bootstrap import bootstrap_paired_hellaswag, bootstrap_paired_ppl  # noqa: E402
from common.byte_accounting import sparse_fp16_bytes  # noqa: E402
from common.eval_hellaswag import HELLASWAG_FORMAT, evaluate_hellaswag  # noqa: E402
from common.eval_perplexity import evaluate_perplexity  # noqa: E402
from common.timing import peak_gpu_memory_mb, reset_peak_memory, timed, timed_steps  # noqa: E402
from task1_compression.baseline import (  # noqa: E402
    ELIGIBLE_SUFFIXES,
    CalibStats,
    collect_calibration_stats,
    count_eligible_params,
    enumerate_eligible_projections,
)

SEED = 6304
GROUP_SIZE = 128
DEFAULT_DATA_ROOT = "/content/drive/MyDrive/atml-pa4-datasets/data/public_locked"
DEFAULT_RESULTS_ROOT = "/content/drive/MyDrive/atml_pa4_results/task1"
DEFAULT_CALIB_CACHE = "/content/calib_stats.pt"
DEFAULT_REVISIONS = str(_REPO_ROOT / "data" / "local_source_revisions.json")

FILES = {
    "calibration": "c4_calibration_locked.jsonl.gz",
    "development": "c4_development_locked.jsonl.gz",
    "c4_val": "c4_validation_windows_locked.jsonl.gz",
    "wt2_test": "wikitext2_test_windows_locked.jsonl.gz",
    "hellaswag": "hellaswag_val_locked.jsonl.gz",
}

HS_SUBSET_N = 512
HS_FULL_N = 2000
LATENCY = {"n_warmup": 3, "n_rep": 10, "prefill_lengths": [512, 2048],
           "decode_prefill_len": 512, "decode_steps": 64}
QUANTILES = [0, 1, 5, 25, 50, 75, 95, 99, 100]
N_QUALITATIVE = 8


# ── Condition registry ────────────────────────────────────────────────────────

def _c(family, method, sparsity=None, bits=None, hs_full=False, draft=False):
    return {"family": family, "method": method, "sparsity": sparsity, "bits": bits,
            "hs_full": hs_full, "draft": draft}


CONDITIONS: Dict[str, dict] = {
    "fp16": _c("baseline", "fp16", hs_full=True),
    "p1_s25": _c("pruning", "magnitude", sparsity=0.25),
    "p1_s50": _c("pruning", "magnitude", sparsity=0.50, hs_full=True),
    "p1_s75": _c("pruning", "magnitude", sparsity=0.75),
    "p2_s25": _c("pruning", "wanda", sparsity=0.25),
    "p2_s50": _c("pruning", "wanda", sparsity=0.50, hs_full=True),
    "p2_s75": _c("pruning", "wanda", sparsity=0.75),
    "p3_s50": _c("pruning", "nonuniform_wanda", sparsity=0.50, hs_full=True, draft=True),
    "p3_s75": _c("pruning", "nonuniform_wanda", sparsity=0.75, draft=True),
    "p2_24": _c("pruning", "wanda_2of4", sparsity=0.50),
    "q1_int8": _c("quantization", "uniform", bits=8),
    "q1_int4": _c("quantization", "uniform", bits=4, hs_full=True),
    "q1_int3": _c("quantization", "uniform", bits=3, hs_full=True),
    "q2_int8": _c("quantization", "awq", bits=8),
    "q2_int4": _c("quantization", "awq", bits=4, hs_full=True),
    "q2_int3": _c("quantization", "awq", bits=3, hs_full=True),
    "q3_int4": _c("quantization", "mixed", bits=4, hs_full=True, draft=True),
    "q3_int3": _c("quantization", "mixed", bits=3, hs_full=True, draft=True),
}
OVERNIGHT = [c for c, v in CONDITIONS.items() if not v["draft"]]


def parse_conditions(spec: str, allow_draft: bool) -> List[str]:
    if spec == "all":
        names = list(CONDITIONS)
    elif spec == "overnight":
        names = list(OVERNIGHT)
    else:
        names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown condition(s) {unknown}; known: {list(CONDITIONS)}")
    drafts = [n for n in names if CONDITIONS[n]["draft"]]
    if drafts and not allow_draft:
        raise ValueError(
            f"{drafts}: P3/Q3 sensitivity/allocation design is pending; pass "
            "--allow-draft-p3q3 to run the current draft implementation."
        )
    return names


# ── Small utilities ───────────────────────────────────────────────────────────

def _iter_jsonl(path):
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, obj, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=indent)
    os.replace(tmp, path)


def _sync(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        reset_peak_memory(device)


def _free(device: str):
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _proj_type(name: str) -> str:
    return next(s for s in ELIGIBLE_SUFFIXES if name.endswith(s))


def quantile_summary(t: torch.Tensor, device: str = "cpu") -> dict:
    """Quantiles (linear interpolation) + mean of a tensor, via a full sort."""
    x = t.detach().reshape(-1).to(device, torch.float32)
    s, _ = torch.sort(x)
    n = s.numel()
    out = {}
    for q in QUANTILES:
        pos = q / 100.0 * (n - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        v = s[lo] + (s[hi] - s[lo]) * (pos - lo)
        out[f"q{q}"] = float(v)
    out["mean"] = float(x.mean())
    del x, s
    return out


def load_revisions(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ── Calibration statistics (one with_gram=True collection per process) ───────

def _stats_to_payload(stats: Dict[str, CalibStats], meta: dict) -> dict:
    return {
        "meta": meta,
        "stats": {
            n: {"col_sqnorm": s.col_sqnorm, "col_absmean": s.col_absmean,
                "gram": s.gram, "n_tokens": int(s.n_tokens)}
            for n, s in stats.items()
        },
    }


def _payload_to_stats(payload: dict) -> Dict[str, CalibStats]:
    return {
        n: CalibStats(d["col_sqnorm"], d["col_absmean"], d["gram"], int(d["n_tokens"]))
        for n, d in payload["stats"].items()
    }


def get_calibration_stats(cfg, load_model, calib_ids) -> Dict[str, CalibStats]:
    """Load cached with_gram=True stats if their metadata matches, else collect
    them from a FRESH unmodified FP16 model and cache them to local disk."""
    meta = {
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "calibration_sha256": cfg["calibration_sha256"],
        "n_seqs": len(calib_ids),
        "with_gram": True,
    }
    cache = cfg.get("calib_cache")
    if cache and Path(cache).exists():
        try:
            payload = torch.load(cache, map_location="cpu", weights_only=True)
            if payload.get("meta") == meta:
                print(f"[calib] reusing cached stats {cache}")
                return _payload_to_stats(payload)
            print("[calib] cache metadata mismatch; recollecting")
        except Exception as e:  # corrupt/partial cache
            print(f"[calib] could not load cache ({e}); recollecting")

    print(f"[calib] collecting with_gram=True stats over {len(calib_ids)} sequences "
          "from a fresh FP16 model")
    model, _ = load_model(cfg["device"])
    t0 = time.perf_counter()
    stats = collect_calibration_stats(model, calib_ids, cfg["device"], with_gram=True)
    print(f"[calib] done in {time.perf_counter() - t0:.1f}s")
    del model
    _free(cfg["device"])
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        tmp = str(cache) + ".tmp"
        torch.save(_stats_to_payload(stats, meta), tmp)
        os.replace(tmp, cache)
        print(f"[calib] cached to {cache}")
    return stats


# ── Measurements ──────────────────────────────────────────────────────────────

@torch.no_grad()
def calibration_output_error(model, originals: Dict[str, torch.Tensor],
                             stats: Optional[Dict[str, CalibStats]], device: str) -> dict:
    """
    Per projection: D = W_compressed - W_fp16 (fp32), G = calibration Gram of that
    projection's inputs taken from the UNMODIFIED FP16 model.
        output_err     = tr(D G D^T) / n_tokens
        rel_output_err = tr(D G D^T) / tr(W G W^T)
    """
    per = {}
    for name, mod in enumerate_eligible_projections(model):
        if stats is None:  # fp16 baseline: weights unchanged
            per[name] = {"output_err": 0.0, "rel_output_err": 0.0, "n_tokens": 0}
            continue
        s = stats[name]
        G = s.gram.to(device, torch.float32)
        W0 = originals[name].to(device, torch.float32)
        D = mod.weight.data.to(device, torch.float32) - W0
        num = ((D @ G) * D).sum().item()
        den = ((W0 @ G) * W0).sum().item()
        per[name] = {"output_err": num / s.n_tokens,
                     "rel_output_err": num / den if den > 0 else 0.0,
                     "n_tokens": int(s.n_tokens)}
        del G, W0, D
    _free(device)

    w = [v["n_tokens"] for v in per.values()]
    tot = sum(w)

    def wmean(key, items):
        ww = sum(v["n_tokens"] for v in items)
        if ww == 0:
            return sum(v[key] for v in items) / max(len(items), 1)
        return sum(v[key] * v["n_tokens"] for v in items) / ww

    by_type = {}
    for t in ELIGIBLE_SUFFIXES:
        items = [v for n, v in per.items() if _proj_type(n) == t]
        by_type[t] = {"output_err": wmean("output_err", items),
                      "rel_output_err": wmean("rel_output_err", items)}
    return {
        "definition": ("D = W_compressed - W_fp16; output_err = tr(D G D^T)/n_tokens; "
                       "rel_output_err = tr(D G D^T)/tr(W G W^T); G = sum_t x_t x_t^T over the "
                       "128 calibration sequences, with inputs x_t taken from the UNMODIFIED FP16 "
                       "model (not the compressed model's own inputs)."),
        "global_token_weighted_mean_output_err": wmean("output_err", list(per.values())),
        "global_token_weighted_mean_rel_output_err": wmean("rel_output_err", list(per.values())),
        "total_calibration_tokens_per_projection": (w[0] if w else 0),
        "by_type": by_type,
        "per_projection": per,
        "_tot_weight": tot,
    }


@torch.no_grad()
def awq_prerounding_logit_check(model, searched: Dict[str, dict], calib_ids, device) -> dict:
    """Run calibration sequences through the model with rescaled-but-unrounded
    weights W_r = (W * diag(a)) * diag(1/a) (cast to the model dtype) and report the
    max |logit diff| vs the unmodified model.  Original tensors are restored."""
    seqs = [torch.tensor([ids], dtype=torch.long, device=device) for ids in calib_ids]
    ref = [model(input_ids=t).logits.float().cpu() for t in seqs]
    saved = {}
    for name, mod in enumerate_eligible_projections(model):
        a = searched[name]["alpha"].to(mod.weight.device, torch.float32)
        saved[name] = mod.weight.data
        W = saved[name].float()
        mod.weight.data = ((W * a.unsqueeze(0)) * (1.0 / a).unsqueeze(0)).to(saved[name].dtype)
        del W, a
    try:
        diffs = [(model(input_ids=t).logits.float().cpu() - r).abs().max().item()
                 for t, r in zip(seqs, ref)]
    finally:
        for name, mod in enumerate_eligible_projections(model):
            mod.weight.data = saved[name]
    del ref, saved
    _free(device)
    return {"n_sequences": len(seqs), "max_abs_logit_diff": max(diffs),
            "per_sequence_max_abs_logit_diff": diffs,
            "note": "rescaled-but-unrounded weights vs unmodified FP16 model"}


def latency_input_ids(results_root: Path, model, tokenizer) -> dict:
    """One fixed seeded id sequence (length 2048) shared by every condition."""
    hi = getattr(tokenizer, "vocab_size", None) or model.config.vocab_size
    hi = int(min(hi, model.config.vocab_size))
    n = max(LATENCY["prefill_lengths"])
    path = results_root / "latency_input_ids.json"
    if path.exists():
        rec = json.load(open(path))
        if rec["seed"] == SEED and rec["high"] == hi and rec["length"] == n:
            return rec
    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, hi, (n,), generator=g).tolist()
    rec = {"seed": SEED, "low": 0, "high": hi, "length": n, "ids": ids,
           "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
           "generator": "torch.Generator().manual_seed(6304); torch.randint(0, high, (length,))"}
    _write_json(path, rec)
    return rec


def _tstats(r: dict) -> dict:
    t = sorted(r["times_ms"])
    return {"mean_ms": r["mean_ms"], "std_ms": r["std_ms"],
            "median_ms": t[len(t) // 2], "times_ms": r["times_ms"],
            "n_warmup": r["n_warmup"], "n_rep": r["n_rep"]}


@torch.no_grad()
def measure_latency(model, ids: List[int], device: str) -> dict:
    nw, nr = LATENCY["n_warmup"], LATENCY["n_rep"]
    out = {"label": "dense FP16 execution (no sparse or low-bit kernels; zeros and "
                    "dequantized weights are stored and multiplied densely)",
           "unit": "ms", "batch_size": 1}

    def prefill(x):
        model(input_ids=x, use_cache=False)

    _reset_peak(device)
    for L in LATENCY["prefill_lengths"]:
        x = torch.tensor([ids[:L]], dtype=torch.long, device=device)
        out[f"prefill_{L}"] = _tstats(timed(prefill, x, n_warmup=nw, n_rep=nr))
        out[f"prefill_{L}"]["definition"] = f"one forward over 1x{L} tokens, use_cache=False"
        del x

    P, K = LATENCY["decode_prefill_len"], LATENCY["decode_steps"]
    x = torch.tensor([ids[:P]], dtype=torch.long, device=device)
    state = {}

    def do_prefill():
        o = model(input_ids=x, use_cache=True)
        state["pkv"] = o.past_key_values
        state["tok"] = o.logits[:, -1:].argmax(-1)

    def do_decode():
        pkv, tok = state.pop("pkv"), state.pop("tok")
        for _ in range(K):
            o = model(input_ids=tok, past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            tok = o.logits[:, -1:].argmax(-1)
        del pkv

    r = timed_steps([do_prefill, do_decode], ["prefill", "decode"], n_warmup=nw, n_rep=nr)
    dec = _tstats(r["decode"])
    dec["ms_per_token"] = dec["mean_ms"] / K
    dec["definition"] = (f"{K} greedy decode forward passes (1 token each, KV cache) after an "
                         f"untimed-in-this-number {P}-token cached prefill; timed separately")
    out[f"decode_{K}_after_prefill_{P}"] = dec
    out[f"decode_prefill_{P}_cached"] = _tstats(r["prefill"])
    out["peak_alloc_mb_latency"] = peak_gpu_memory_mb(device) if str(device).startswith("cuda") else None
    del x, state
    _free(device)
    return out


def eligible_weight_stats(model) -> dict:
    per_type_n = {t: 0 for t in ELIGIBLE_SUFFIXES}
    per_type_nz = {t: 0 for t in ELIGIBLE_SUFFIXES}
    for name, mod in enumerate_eligible_projections(model):
        t = _proj_type(name)
        per_type_n[t] += mod.weight.numel()
        per_type_nz[t] += int((mod.weight.data != 0).sum().item())
    n = sum(per_type_n.values())
    nz = sum(per_type_nz.values())
    return {
        "n_eligible": n,
        "effective_nonzeros": nz,
        "achieved_eligible_sparsity": 1.0 - nz / n,
        "achieved_sparsity_by_type": {t: 1.0 - per_type_nz[t] / per_type_n[t] for t in per_type_n},
        "definition": "fraction of eligible weights that are exactly zero after compression",
    }


def storage_accounting(model, cond: dict, ws: dict, quant_bytes: Optional[dict],
                       nominal_bits: float) -> dict:
    n_total = sum(p.numel() for p in model.parameters())  # tied params counted once
    n_elig = count_eligible_params(model)["total"]
    n_unc = n_total - n_elig
    dense = 2 * n_total
    sparse_elig = sparse_fp16_bytes(n_elig, ws["effective_nonzeros"] / n_elig)
    out = {
        "n_params_total_unique": n_total,
        "n_eligible": n_elig,
        "n_uncompressed": n_unc,
        "dense_fp16_bytes": {"value": dense, "label": "dense",
                             "definition": "2 bytes x every unique parameter (tied counted once)"},
        "theoretical_sparse_fp16_bytes": {
            "value": sparse_elig + 2 * n_unc, "eligible_part": sparse_elig, "label": "theoretical",
            "definition": ("1 mask bit per eligible weight (byte-aligned) + 16 bits per nonzero "
                           "eligible weight + FP16 uncompressed params; a representation estimate, "
                           "not allocated GPU memory")},
        "nominal_bits_per_eligible": nominal_bits,
    }
    if quant_bytes is not None:
        elig = quant_bytes["payload"] + quant_bytes["metadata"] + quant_bytes["compensating"]
        out["packed_quant_bytes"] = {
            "value": elig + 2 * n_unc, "eligible_part": elig, "label": "packed",
            "payload_bytes": quant_bytes["payload"], "metadata_bytes": quant_bytes["metadata"],
            "compensating_scale_bytes": quant_bytes["compensating"],
            "uncompressed_fp16_bytes": 2 * n_unc,
            "definition": ("bit-packed integer codes + FP16 scale and FP16 zero-point per group of "
                           "128 + serialized FP16 AWQ compensating scales (2*d_in per projection, Q2/Q3) "
                           "+ FP16 uncompressed params; theoretical packed format, evaluated via "
                           "dequantized FP16 weights")}
        out["packed_bits_per_eligible"] = elig * 8 / n_elig
    else:
        out["theoretical_sparse_bits_per_eligible"] = sparse_elig * 8 / n_elig
    return out


def q2_uniform_budget_bytes(model, bits: int, n_uncompressed: int) -> int:
    """Total packed bytes of Q2 uniform `bits`: payload + metadata + 2*d_in + FP16 rest."""
    total = 2 * n_uncompressed
    for _, m in enumerate_eligible_projections(model):
        d_out, d_in = m.weight.shape
        total += math.ceil(d_out * d_in * bits / 8) + math.ceil(d_in / GROUP_SIZE) * 4 * d_out + 2 * d_in
    return total


# ── Compression dispatch ──────────────────────────────────────────────────────

def compress(name: str, cond: dict, model, stats, calib_ids, cfg) -> dict:
    """Apply the condition's compression in place.  Returns a small JSON-able
    `info` dict plus timing; large return dicts are reduced and deleted here."""
    from task1_compression.prune_magnitude import magnitude_prune
    from task1_compression.prune_wanda import wanda_prune
    from task1_compression.structured_diag import apply_24_mask
    from task1_compression.quant_uniform import quantize_model_uniform
    from task1_compression.quant_awq import awq_search_scales, apply_awq_quantization, SCALE_EXPONENTS

    dev = cfg["device"]
    method = cond["method"]
    info: dict = {}
    quant_bytes = None
    nominal_bits = 16.0
    timing_note = ""

    def tic():
        _sync(dev)
        return time.perf_counter()

    def toc(t0):
        _sync(dev)
        return time.perf_counter() - t0

    if method == "fp16":
        secs = 0.0

    elif method in ("magnitude", "wanda"):
        t0 = tic()
        if method == "magnitude":
            res = magnitude_prune(model, cond["sparsity"])
        else:
            res = wanda_prune(model, cond["sparsity"], calibration_inputs=stats, device=dev)
        secs = toc(t0)
        timing_note = "pruning only (calibration statistics collected once per process, excluded)"
        key = "abs_weight" if method == "magnitude" else "wanda_score"
        info["score_distribution"] = {
            "score": "|W_ij|" if method == "magnitude" else "|W_ij| * ||X_:,j||_2",
            "quantiles": QUANTILES,
            "per_projection": {n: quantile_summary(v["scores"], dev) for n, v in res.items()},
        }
        info["score_distribution"]["name"] = key
        info["mask_sparsity"] = _mask_sparsity(res)
        del res

    elif method == "wanda_2of4":
        # Untimed: the unstructured 50% Wanda mask on the same fresh weights (overlap study).
        ref = wanda_prune(model, 0.5, calibration_inputs=stats, device=dev, inplace=False)
        umask = {n: v["mask"] for n, v in ref.items()}
        del ref
        _free(dev)
        t0 = tic()
        res = apply_24_mask(model, stats)
        secs = toc(t0)
        timing_note = "2:4 masking only (unstructured reference mask for the overlap study excluded)"
        info["mask_sparsity"] = _mask_sparsity(res)
        info["overlap_with_p2_s50"] = _mask_overlap(res, umask)
        del res, umask

    elif method == "nonuniform_wanda":
        from task1_compression.prune_nonuniform import measure_block_sensitivity, nonuniform_prune
        t0 = tic()
        sens = measure_block_sensitivity(model, stats, probe_sparsity=0.5, device=dev)
        masks, block_s = nonuniform_prune(model, cond["sparsity"], sens, stats, device=dev)
        secs = toc(t0)
        timing_note = "DRAFT P3: sensitivity measurement + allocation + pruning"
        info["draft_design"] = True
        info["block_sensitivity"] = {str(k): float(v) for k, v in sens.items()}
        info["block_sparsity"] = {str(k): float(v) for k, v in block_s.items()}
        info["mask_sparsity"] = _mask_sparsity(masks)
        del masks

    elif method == "uniform":
        b = cond["bits"]
        t0 = tic()
        res = quantize_model_uniform(model, b, GROUP_SIZE)
        secs = toc(t0)
        quant_bytes = {"payload": sum(v["packed_bytes"] for v in res.values()),
                       "metadata": sum(v["metadata_bytes"] for v in res.values()),
                       "compensating": 0}
        nominal_bits = float(b)
        info["quantizer"] = _quantizer_desc(b)
        del res

    elif method == "awq":
        b = cond["bits"]
        t0 = tic()
        searched = awq_search_scales(model, stats, b, GROUP_SIZE, SCALE_EXPONENTS, dev)
        t_search = toc(t0)
        # Pre-rounding check (excluded from compression time).
        info["prerounding_check"] = {
            "projection_level": {
                "max_abs_weight_diff_fp32": max(v["prerounding_max_abs_diff"] for v in searched.values()),
                "max_rel_output_err_fp32": max(v["prerounding_rel_output_err"] for v in searched.values()),
                "per_projection": {n: {"max_abs_diff": v["prerounding_max_abs_diff"],
                                       "rel_output_err": v["prerounding_rel_output_err"]}
                                   for n, v in searched.items()},
                "definition": "W_r = (W*diag(a))*diag(1/a) vs W in fp32; rel err via the calibration Gram",
            },
            "model_level": awq_prerounding_logit_check(model, searched, calib_ids[:4], dev),
        }
        t1 = tic()
        res = apply_awq_quantization(model, stats, b, GROUP_SIZE, SCALE_EXPONENTS, dev, searched=searched)
        secs = t_search + toc(t1)
        timing_note = "scale search + quantization (pre-rounding check excluded)"
        quant_bytes = {"payload": sum(v["packed_bytes"] for v in res.values()),
                       "metadata": sum(v["metadata_bytes"] for v in res.values()),
                       "compensating": sum(v["compensating_scale_bytes"] for v in res.values())}
        nominal_bits = float(b)
        info["quantizer"] = _quantizer_desc(b)
        info["awq"] = {
            "exponent_grid": list(SCALE_EXPONENTS),
            "compensation": "serialized FP16 per-input-channel scales (2*d_in bytes per projection), no folding",
            "best_exp": {n: v["best_exp"] for n, v in res.items()},
            "search_calibration_mse": {n: v["calibration_error"] for n, v in res.items()},
        }
        del res, searched

    elif method == "mixed":
        from task1_compression.quant_mixed import (allocate_bits, measure_all_sensitivities,
                                                   mixed_precision_quantize)
        b = cond["bits"]
        n_unc = sum(p.numel() for p in model.parameters()) - count_eligible_params(model)["total"]
        budget = q2_uniform_budget_bytes(model, b, n_unc)
        t0 = tic()
        sens = measure_all_sensitivities(model, stats)
        alloc = allocate_bits(sens, model, budget, n_unc)
        res = mixed_precision_quantize(model, stats, alloc)
        secs = toc(t0)
        timing_note = "DRAFT Q3: sensitivity + allocation + AWQ-style quantization"
        quant_bytes = {"payload": sum(v["packed_bytes"] for v in res.values()),
                       "metadata": sum(v["metadata_bytes"] for v in res.values()),
                       "compensating": sum(v["compensating_scale_bytes"] for v in res.values())}
        sizes = {n: m.weight.numel() for n, m in enumerate_eligible_projections(model)}
        nominal_bits = sum(sizes[n] * alloc[n] for n in alloc) / sum(sizes.values())
        total = sum(quant_bytes.values()) + 2 * n_unc
        info["draft_design"] = True
        info["quantizer"] = _quantizer_desc(None)
        info["bit_assignment"] = dict(alloc)
        info["sensitivity"] = {n: float(v) for n, v in sens.items()}
        info["matched_budget_check"] = {"budget_bytes_q2_uniform": budget, "total_packed_bytes": total,
                                        "within_budget": total <= budget}
        del res
    else:
        raise ValueError(method)

    _free(dev)
    return {"info": info, "compression_seconds": secs, "compression_timing_note": timing_note,
            "quant_bytes": quant_bytes, "nominal_bits": nominal_bits}


def _quantizer_desc(bits):
    return {"bits": bits, "group_size": GROUP_SIZE, "groups_along": "input columns",
            "asymmetric": True, "rounding": "round-half-to-even (torch.round)",
            "scale": "(w_max - w_min) / (2^b - 1), clamped >= 1e-8, stored FP16",
            "zero_point": "clip(round(-w_min / s), 0, 2^b - 1), stored FP16",
            "padding": "last group zero-padded to 128 (Qwen2.5 d_in are multiples of 128)",
            "packing": "ceil(n_weights * b / 8) bytes of bit-packed codes per projection"}


def _mask_sparsity(res: dict) -> dict:
    n = sum(v["mask"].numel() for v in res.values())
    kept = sum(int(v["mask"].sum()) for v in res.values())
    by_type = {}
    for t in ELIGIBLE_SUFFIXES:
        items = [v["mask"] for k, v in res.items() if _proj_type(k) == t]
        nt = sum(m.numel() for m in items)
        by_type[t] = 1.0 - sum(int(m.sum()) for m in items) / nt if nt else None
    return {"global": 1.0 - kept / n, "by_type": by_type}


def _mask_overlap(res24: dict, umask: dict) -> dict:
    per = {}
    tot_inter = tot_union = tot_k24 = tot_ku = 0
    for n, v in res24.items():
        a, b = v["mask"], umask[n]
        inter = int((a & b).sum())
        union = int((a | b).sum())
        k24, ku = int(a.sum()), int(b.sum())
        per[n] = {"shared_kept_fraction": inter / k24 if k24 else 0.0,
                  "jaccard": inter / union if union else 1.0,
                  "kept_2of4": k24, "kept_unstructured": ku}
        tot_inter += inter
        tot_union += union
        tot_k24 += k24
        tot_ku += ku
    same = tot_k24 == tot_ku
    if not same:
        warnings.warn(f"2:4 kept {tot_k24} != unstructured kept {tot_ku}")
    return {"definition": "kept = mask True; shared_kept_fraction = |K24 & Ku| / |K24|; "
                          "jaccard = |K24 & Ku| / |K24 | Ku|; Ku = unstructured Wanda 50% (p2_s50) mask",
            "global_shared_kept_fraction": tot_inter / tot_k24 if tot_k24 else 0.0,
            "global_jaccard": tot_inter / tot_union if tot_union else 1.0,
            "kept_2of4_total": tot_k24, "kept_unstructured_total": tot_ku,
            "identical_eligible_sparsity": same,
            "per_projection": per}


# ── Per-condition run ─────────────────────────────────────────────────────────

def run_condition(name: str, cfg: dict, load_model: Callable, get_stats: Callable,
                  calib_ids) -> dict:
    cond = CONDITIONS[name]
    dev = cfg["device"]
    out_dir = Path(cfg["results_root"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ({cond['method']}) ===")

    stats = None if cond["method"] == "fp16" else get_stats()

    model, tokenizer = load_model(dev)
    model.eval()
    torch.manual_seed(SEED)

    originals = None
    if cond["method"] != "fp16":
        originals = {n: m.weight.data.detach().clone() for n, m in enumerate_eligible_projections(model)}

    comp = compress(name, cond, model, stats, calib_ids, cfg)
    print(f"[{name}] compression {comp['compression_seconds']:.1f}s")

    calib_err = calibration_output_error(model, originals, stats, dev)
    calib_err.pop("_tot_weight", None)
    del originals
    _free(dev)

    ws = eligible_weight_stats(model)
    storage = storage_accounting(model, cond, ws, comp["quant_bytes"], comp["nominal_bits"])

    # ---- quality evaluation (peak memory measured from here) ----
    _reset_peak(dev)
    ppl_dtype = torch.float16 if str(dev).startswith("cuda") else torch.bfloat16
    data = Path(cfg["data_root"])
    ppl = {}
    for key, fname, out_name in (("c4", FILES["c4_val"], "ppl_c4_per_window.json"),
                                 ("wt2", FILES["wt2_test"], "ppl_wt2_per_window.json")):
        r = evaluate_perplexity(model, str(data / fname), dev, dtype=ppl_dtype,
                                batch_size=cfg["ppl_batch_size"],
                                expected_total_tokens=cfg["expected_n"])
        _write_json(out_dir / out_name, r["per_window"])
        ppl[key] = {"N": r["N"], "mean_nll": r["mean_nll"], "ppl": r["ppl"], "n_windows": len(r["per_window"])}
        print(f"[{name}] {key} PPL {r['ppl']:.4f} (N={r['N']})")
        del r

    subset = None if cond["hs_full"] else range(HS_SUBSET_N)
    hs = evaluate_hellaswag(model, tokenizer, str(data / FILES["hellaswag"]), dev, subset_ids=subset)
    _write_json(out_dir / "hellaswag_per_example.json", hs["per_example"])
    hs_expected = HS_FULL_N if cond["hs_full"] else HS_SUBSET_N
    print(f"[{name}] HellaSwag n={hs['n_examples']} acc={hs['raw_acc']:.4f} acc_norm={hs['acc_norm']:.4f}")
    peak_eval = peak_gpu_memory_mb(dev) if str(dev).startswith("cuda") else None

    # ---- latency (dense FP16 execution) ----
    ids_rec = latency_input_ids(Path(cfg["results_root"]), model, tokenizer)
    lat = measure_latency(model, ids_rec["ids"], dev)
    lat["input_ids_file"] = "latency_input_ids.json"
    lat["input_ids_sha256"] = ids_rec["sha256"]

    summary = {
        "condition": name,
        "family": cond["family"], "method": cond["method"], "target_sparsity": cond["sparsity"],
        "bits": cond["bits"], "draft": cond["draft"], "hs_full": cond["hs_full"],
        "seed": SEED,
        "env": _env(cfg, model),
        "compression_seconds": comp["compression_seconds"],
        "compression_timing_note": comp["compression_timing_note"],
        "c4_val": ppl["c4"],
        "wt2_test": ppl["wt2"],
        "hellaswag": {
            "task_version": (f"Rowan/hellaswag@{cfg['revisions']['revisions'].get('hellaswag')} "
                             f"split=validation, {HS_FULL_N} locked IDs ({FILES['hellaswag']}, seed {SEED}), "
                             "zero-shot, lm-evaluation-harness acc_norm convention "
                             "(loglik / len(ending) in characters)"),
            "subset": "all 2000 locked IDs" if cond["hs_full"] else f"orders 0..{HS_SUBSET_N - 1} (first 512 IDs)",
            "n_examples": hs["n_examples"],
            "n_matches_spec": hs["n_examples"] == hs_expected,
            "raw_acc": hs["raw_acc"],
            "acc_norm": hs["acc_norm"],
            "format": HELLASWAG_FORMAT,
        },
        "peak_alloc_mb_eval": peak_eval,
        "peak_alloc_mb_eval_note": f"torch.cuda.max_memory_allocated over PPL (batch {cfg['ppl_batch_size']}) + HellaSwag",
        "latency": lat,
        "sparsity": {**ws, **({"mask_sparsity": comp["info"].pop("mask_sparsity")}
                               if "mask_sparsity" in comp["info"] else {})},
        "storage": storage,
        "calibration_output_error": calib_err,
        "details": comp["info"],
    }
    _write_json(out_dir / "summary.json", summary, indent=1)
    del model, tokenizer, comp
    _free(dev)
    return summary


def _env(cfg, model) -> dict:
    import transformers
    try:
        import huggingface_hub
        hub_v = huggingface_hub.__version__
    except Exception:
        hub_v = None
    dev = cfg["device"]
    return {
        "model_id": cfg["model_id"], "model_revision": cfg["model_revision"],
        "torch": torch.__version__, "transformers": transformers.__version__,
        "huggingface_hub": hub_v, "device": dev,
        "cuda_device": torch.cuda.get_device_name(0) if str(dev).startswith("cuda") and torch.cuda.is_available() else None,
        "dtype": str(next(model.parameters()).dtype),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "calibration_file_sha256": cfg["calibration_sha256"],
        "data_root": cfg["data_root"],
    }


def run(cfg: dict, conditions: List[str], load_model: Optional[Callable] = None) -> Dict[str, str]:
    if load_model is None:
        from task1_compression.baseline import load_fp16_baseline

        def load_model(device):
            return load_fp16_baseline(cfg["model_id"], cfg["model_revision"], device)

    root = Path(cfg["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    calib_path = Path(cfg["data_root"]) / FILES["calibration"]
    calib_ids = [r["token_ids"] for r in _iter_jsonl(calib_path)]
    cfg["calibration_sha256"] = _sha256_file(calib_path)

    holder = {}

    def get_stats():
        if "stats" not in holder:
            holder["stats"] = get_calibration_stats(cfg, load_model, calib_ids)
        return holder["stats"]

    status = {}
    for name in conditions:
        if (root / name / "summary.json").exists() and not cfg["overwrite"]:
            print(f"[skip] {name}: summary.json exists (use --overwrite)")
            status[name] = "skipped"
            continue
        err_path = root / name / "error.json"
        try:
            run_condition(name, cfg, load_model, get_stats, calib_ids)
        except Exception as e:
            err = {"condition": name, "exception_type": type(e).__name__,
                   "message": str(e), "traceback": traceback.format_exc()}
            _write_json(err_path, err, indent=1)
            print(f"[FAILED] {name}: {err['exception_type']}: {err['message']}\n{err['traceback']}")
            _free(cfg["device"])
            status[name] = "failed"
            continue
        if err_path.exists():
            err_path.unlink()
        status[name] = "done"
    for name, st in status.items():
        print(f"[status] {name}: {st}")
    holder.clear()
    _free(cfg["device"])
    return status


# ── Aggregation ───────────────────────────────────────────────────────────────

BOOTSTRAP_PAIRS = [("p3_s50", "p2_s50"), ("p3_s75", "p2_s75"),
                   ("q3_int4", "q2_int4"), ("q3_int3", "q2_int3")]


def _load(root: Path, cond: str, fname: str):
    p = root / cond / fname
    return json.load(open(p)) if p.exists() else None


def _acc(pe, key="norm_pred"):
    return sum(r["label"] == r[key] for r in pe) / len(pe) if pe else float("nan")


def aggregate(results_root: str, data_root: Optional[str] = None) -> dict:
    root = Path(results_root)
    summ = {c: _load(root, c, "summary.json") for c in CONDITIONS}
    summ = {c: s for c, s in summ.items() if s is not None}
    hs = {c: _load(root, c, "hellaswag_per_example.json") for c in summ}
    base = summ.get("fp16")
    base_hs = {r["order"]: r for r in (hs.get("fp16") or [])}

    rows = []
    for c, s in summ.items():
        pe = hs[c] or []
        pe512 = [r for r in pe if r["order"] < HS_SUBSET_N]
        st = s["storage"]
        lat = s["latency"]
        row = {
            "condition": c, "family": s["family"], "method": s["method"],
            "target_sparsity": s["target_sparsity"], "bits": s["bits"], "draft": s["draft"],
            "achieved_eligible_sparsity": s["sparsity"]["achieved_eligible_sparsity"],
            "effective_nonzeros": s["sparsity"]["effective_nonzeros"],
            "nominal_bits_per_eligible": st["nominal_bits_per_eligible"],
            "dense_fp16_bytes": st["dense_fp16_bytes"]["value"],
            "theoretical_sparse_fp16_bytes": st["theoretical_sparse_fp16_bytes"]["value"],
            "packed_quant_bytes": st.get("packed_quant_bytes", {}).get("value"),
            "c4_ppl": s["c4_val"]["ppl"], "c4_mean_nll": s["c4_val"]["mean_nll"], "c4_N": s["c4_val"]["N"],
            "wt2_ppl": s["wt2_test"]["ppl"], "wt2_mean_nll": s["wt2_test"]["mean_nll"], "wt2_N": s["wt2_test"]["N"],
            "hs_n": s["hellaswag"]["n_examples"], "hs_raw_acc": s["hellaswag"]["raw_acc"],
            "hs_acc_norm": s["hellaswag"]["acc_norm"],
            "hs512_raw_acc": _acc(pe512, "raw_pred"), "hs512_acc_norm": _acc(pe512),
            "calib_rel_output_err": s["calibration_output_error"]["global_token_weighted_mean_rel_output_err"],
            "calib_output_err": s["calibration_output_error"]["global_token_weighted_mean_output_err"],
            "compression_seconds": s["compression_seconds"],
            "prefill_512_ms": lat["prefill_512"]["mean_ms"],
            "prefill_2048_ms": lat["prefill_2048"]["mean_ms"],
            "decode_64_ms": lat["decode_64_after_prefill_512"]["mean_ms"],
            "decode_ms_per_token": lat["decode_64_after_prefill_512"]["ms_per_token"],
            "peak_alloc_mb_eval": s["peak_alloc_mb_eval"],
            "peak_alloc_mb_latency": lat["peak_alloc_mb_latency"],
        }
        if base is not None:
            row["delta_c4_ppl_vs_fp16"] = row["c4_ppl"] - base["c4_val"]["ppl"]
            row["delta_wt2_ppl_vs_fp16"] = row["wt2_ppl"] - base["wt2_test"]["ppl"]
            b512 = [base_hs[r["order"]] for r in pe512 if r["order"] in base_hs]
            row["delta_hs512_acc_norm_vs_fp16"] = row["hs512_acc_norm"] - _acc(b512) if b512 else None
            matched = [base_hs[r["order"]] for r in pe if r["order"] in base_hs]
            row["delta_hs_acc_norm_vs_fp16_matched"] = (row["hs_acc_norm"] - _acc(matched)) if matched else None
        rows.append(row)

    if rows:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(root / "table.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Paired bootstrap P3-P2 and Q3-Q2.
    boot = {}
    for b_name, a_name in BOOTSTRAP_PAIRS:
        if b_name not in summ or a_name not in summ:
            continue
        a_map = {r["order"]: r for r in hs[a_name]}
        b_map = {r["order"]: r for r in hs[b_name]}
        orders = sorted(set(a_map) & set(b_map))
        entry = {"difference": f"{b_name} - {a_name}", "hellaswag_n": len(orders)}
        for metric in ("acc_norm", "raw_acc"):
            entry[f"hellaswag_{metric}"] = bootstrap_paired_hellaswag(
                [a_map[o] for o in orders], [b_map[o] for o in orders], metric=metric, seed=SEED)
        for key, fname in (("c4", "ppl_c4_per_window.json"), ("wt2", "ppl_wt2_per_window.json")):
            pa, pb = _load(root, a_name, fname), _load(root, b_name, fname)
            entry[f"{key}_ppl"] = bootstrap_paired_ppl(pa, pb, metric="ppl", seed=SEED)
        boot[f"{b_name}_minus_{a_name}"] = entry
    _write_json(root / "bootstrap.json", boot, indent=1)

    qual = _qualitative(root, summ, hs, data_root)
    plots = _plots(root, summ, rows)
    return {"n_conditions": len(summ), "table": str(root / "table.csv"), "bootstrap": list(boot),
            "qualitative_orders": qual, "plots": plots}


def _qualitative(root: Path, summ, hs, data_root) -> List[int]:
    p = root / "qualitative_orders.json"
    if p.exists():
        orders = json.load(open(p))["orders"]
    else:
        orders = random.Random(SEED).sample(range(HS_SUBSET_N), N_QUALITATIVE)
        _write_json(p, {"seed": SEED, "population": f"orders 0..{HS_SUBSET_N - 1}",
                        "method": "random.Random(6304).sample(range(512), 8)", "orders": orders})
    with open(root / "qualitative.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "condition", "label"] + [f"raw_{k}" for k in range(4)]
                   + [f"norm_{k}" for k in range(4)] + ["raw_pred", "norm_pred", "raw_correct", "norm_correct"])
        for o in orders:
            for c in summ:
                rec = next((r for r in (hs[c] or []) if r["order"] == o), None)
                if rec is None:
                    continue
                w.writerow([o, c, rec["label"]] + rec["raw_scores"] + rec["norm_scores"]
                           + [rec["raw_pred"], rec["norm_pred"], int(rec["raw_pred"] == rec["label"]),
                              int(rec["norm_pred"] == rec["label"])])
    if data_root:
        hp = Path(data_root) / FILES["hellaswag"]
        if hp.exists():
            want = set(orders)
            ex = {r["order"]: {"context": r["example"].get("activity_label", "") + ": " + r["example"].get("ctx", ""),
                               "endings": r["example"]["endings"], "label": r["example"]["label"]}
                  for r in _iter_jsonl(hp) if r["order"] in want}
            _write_json(root / "qualitative_examples.json", {str(o): ex.get(o) for o in orders}, indent=1)
    return orders


# Palette: validated categorical slots 1-3 (blue, orange, aqua) + marker shapes as
# secondary encoding; ink/grid tokens from the reference chart palette.
_SERIES = {"P1 magnitude": ("#2a78d6", "o"), "P2 Wanda": ("#eb6834", "s"), "P3 nonuniform": ("#1baf7a", "^"),
           "Q1 uniform": ("#2a78d6", "o"), "Q2 AWQ": ("#eb6834", "s"), "Q3 mixed": ("#1baf7a", "^")}
_INK, _INK2, _MUTED, _GRID, _AXIS, _SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _style_ax(ax):
    ax.set_facecolor(_SURF)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_INK2, labelsize=8)
    ax.xaxis.label.set_color(_INK2)
    ax.yaxis.label.set_color(_INK2)
    ax.title.set_color(_INK)


def _plots(root: Path, summ: dict, rows: List[dict]) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[aggregate] matplotlib unavailable ({e}); skipping plots")
        return []
    pdir = root / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    by = {r["condition"]: r for r in rows}
    made = []
    metrics = [("c4_ppl", "C4 validation PPL", True), ("wt2_ppl", "WikiText-2 test PPL", True),
               ("hs512_acc_norm", "HellaSwag acc_norm (first 512 IDs)", False)]
    base = by.get("fp16")

    def save(fig, name):
        fig.patch.set_facecolor(_SURF)
        fig.tight_layout()
        fig.savefig(pdir / name, dpi=150, facecolor=_SURF)
        plt.close(fig)
        made.append(name)

    def ref_line(ax, key):
        if base is not None and base.get(key) is not None:
            ax.axhline(base[key], color=_MUTED, linestyle="--", linewidth=1.2, label="FP16 baseline")

    # 1. Pruning: quality vs achieved global sparsity.
    groups = {"P1 magnitude": ["p1_s25", "p1_s50", "p1_s75"], "P2 Wanda": ["p2_s25", "p2_s50", "p2_s75"],
              "P3 nonuniform": ["p3_s50", "p3_s75"]}
    if any(c in by for g in groups.values() for c in g):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, (key, label, logy) in zip(axes, metrics):
            _style_ax(ax)
            for g, conds in groups.items():
                pts = [(by[c]["achieved_eligible_sparsity"], by[c][key]) for c in conds if c in by]
                if pts:
                    col, mk = _SERIES[g]
                    ax.plot(*zip(*pts), color=col, marker=mk, markersize=7, linewidth=2, label=g)
            if "p2_24" in by:
                ax.plot(by["p2_24"]["achieved_eligible_sparsity"], by["p2_24"][key], linestyle="none",
                        marker="X", markersize=9, color=_INK, label="2:4 Wanda")
            ref_line(ax, key)
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("achieved eligible-weight sparsity")
            ax.set_title(label, fontsize=10)
        axes[0].legend(fontsize=8, frameon=False)
        save(fig, "pruning_quality_vs_sparsity.png")

    # 2. Quantization: quality vs packed bytes.
    qgroups = {"Q1 uniform": ["q1_int8", "q1_int4", "q1_int3"], "Q2 AWQ": ["q2_int8", "q2_int4", "q2_int3"],
               "Q3 mixed": ["q3_int4", "q3_int3"]}
    if any(c in by for g in qgroups.values() for c in g):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, (key, label, logy) in zip(axes, metrics):
            _style_ax(ax)
            for g, conds in qgroups.items():
                pts = sorted((by[c]["packed_quant_bytes"] / 1e9, by[c][key]) for c in conds if c in by)
                if pts:
                    col, mk = _SERIES[g]
                    ax.plot(*zip(*pts), color=col, marker=mk, markersize=7, linewidth=2, label=g)
            ref_line(ax, key)
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("packed model bytes (GB, incl. metadata + FP16 rest)")
            ax.set_title(label, fontsize=10)
        axes[0].legend(fontsize=8, frameon=False)
        save(fig, "quant_quality_vs_packed_bytes.png")

    # 3. Calibration error vs held-out quality.
    pts = [r for r in rows if r["condition"] != "fp16" and r["calib_rel_output_err"] > 0]
    if pts:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fam = {"pruning": ("#2a78d6", "o", "pruning"), "quantization": ("#eb6834", "s", "quantization")}
        for ax, (key, label) in zip(axes, [("c4_ppl", "C4 validation PPL"),
                                           ("hs512_acc_norm", "HellaSwag acc_norm (first 512)")]):
            _style_ax(ax)
            for f, (col, mk, lab) in fam.items():
                sel = [r for r in pts if r["family"] == f]
                if sel:
                    ax.scatter([r["calib_rel_output_err"] for r in sel], [r[key] for r in sel],
                               color=col, marker=mk, s=40, label=lab, edgecolors=_SURF, linewidths=1.5)
                    for r in sel:
                        ax.annotate(r["condition"], (r["calib_rel_output_err"], r[key]), fontsize=7,
                                    color=_INK2, xytext=(3, 3), textcoords="offset points")
            ref_line(ax, key)
            ax.set_xscale("log")
            if key == "c4_ppl":
                ax.set_yscale("log")
            ax.set_xlabel("relative calibration output error (token-weighted mean)")
            ax.set_title(label, fontsize=10)
        axes[0].legend(fontsize=8, frameon=False)
        save(fig, "calib_error_vs_quality.png")

    # 4. Per-layer x projection-type relative calibration error heatmaps.
    from matplotlib.colors import LogNorm
    for c, s in summ.items():
        per = s["calibration_output_error"]["per_projection"]
        if c == "fp16" or not per:
            continue
        layers = sorted({int(n.split(".layers.")[1].split(".")[0]) for n in per})
        grid = [[max(per.get(_pname(l, t), {}).get("rel_output_err", 0.0), 1e-12) for t in ELIGIBLE_SUFFIXES]
                for l in layers]
        fig, ax = plt.subplots(figsize=(5, max(3, 0.18 * len(layers) + 1)))
        im = ax.imshow(grid, aspect="auto", cmap="Blues", norm=LogNorm())
        ax.set_xticks(range(len(ELIGIBLE_SUFFIXES)), ELIGIBLE_SUFFIXES, rotation=45, fontsize=7)
        ax.set_yticks(range(len(layers)), layers, fontsize=6)
        ax.set_ylabel("layer")
        ax.set_title(f"{c}: relative calibration output error", fontsize=9)
        fig.colorbar(im, ax=ax)
        save(fig, f"calib_error_heatmap_{c}.png")

    # 5. Adaptive-method sensitivity / allocation maps (P3, Q3) when present.
    for c in ("p3_s50", "p3_s75"):
        s = summ.get(c)
        if s and "block_sparsity" in s["details"]:
            bs = s["details"]["block_sparsity"]
            sens = s["details"].get("block_sensitivity", {})
            ks = sorted(bs, key=int)
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
            for ax, vals, lab in ((axes[0], [sens.get(k, 0.0) for k in ks], "block sensitivity"),
                                  (axes[1], [bs[k] for k in ks], "allocated block sparsity")):
                _style_ax(ax)
                ax.bar([int(k) for k in ks], vals, color="#2a78d6", width=0.8)
                ax.set_xlabel("layer")
                ax.set_title(f"{c}: {lab}", fontsize=10)
            save(fig, f"{c}_sensitivity_allocation.png")
    for c in ("q3_int4", "q3_int3"):
        s = summ.get(c)
        if s and "bit_assignment" in s["details"]:
            ba = s["details"]["bit_assignment"]
            sens = s["details"].get("sensitivity", {})
            layers = sorted({int(n.split(".layers.")[1].split(".")[0]) for n in ba})
            fig, axes = plt.subplots(1, 2, figsize=(10, max(3, 0.18 * len(layers) + 1)))
            for ax, src, lab, norm in ((axes[0], sens, "sensitivity", LogNorm()),
                                       (axes[1], ba, "assigned bits", None)):
                grid = [[max(float(src.get(_pname(l, t), 0.0)), 1e-12) for t in ELIGIBLE_SUFFIXES] for l in layers]
                im = ax.imshow(grid, aspect="auto", cmap="Blues", norm=norm)
                ax.set_xticks(range(len(ELIGIBLE_SUFFIXES)), ELIGIBLE_SUFFIXES, rotation=45, fontsize=7)
                ax.set_yticks(range(len(layers)), layers, fontsize=6)
                ax.set_title(f"{c}: {lab}", fontsize=9)
                fig.colorbar(im, ax=ax)
            save(fig, f"{c}_sensitivity_bits.png")
    return made


def _pname(layer: int, t: str) -> str:
    sub = "self_attn" if t in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
    return f"model.layers.{layer}.{sub}.{t}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run conditions")
    r.add_argument("--conditions", default="overnight",
                   help="comma list, 'overnight' (14 non-draft) or 'all' (18; drafts need the flag)")
    r.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    r.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    r.add_argument("--device", default=None, help="default: cuda if available else cpu")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--allow-draft-p3q3", action="store_true")
    r.add_argument("--calib-cache", default=DEFAULT_CALIB_CACHE,
                   help="local (not Drive) path for the cached with_gram=True stats; '' disables")
    r.add_argument("--expected-n", type=int, default=50000,
                   help="expected scored tokens per perplexity file (override only for tests)")
    r.add_argument("--revisions", default=DEFAULT_REVISIONS)
    r.add_argument("--ppl-batch-size", type=int, default=None, help="default 4 on cuda, 1 on cpu")
    a = sub.add_parser("aggregate", help="build table.csv, bootstrap.json, plots, qualitative.csv")
    a.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    a.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    return p


def main(argv=None, load_model: Optional[Callable] = None):
    args = build_parser().parse_args(argv)
    if args.cmd == "aggregate":
        res = aggregate(args.results_root, args.data_root)
        print(json.dumps(res, indent=1))
        return res
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    conditions = parse_conditions(args.conditions, args.allow_draft_p3q3)
    revs = load_revisions(args.revisions)
    cfg = {
        "device": device,
        "data_root": args.data_root,
        "results_root": args.results_root,
        "overwrite": args.overwrite,
        "calib_cache": args.calib_cache or None,
        "expected_n": args.expected_n,
        "ppl_batch_size": args.ppl_batch_size or (4 if device.startswith("cuda") else 1),
        "revisions": revs,
        "model_id": revs["repos"]["model_base"],
        "model_revision": revs["revisions"]["model_base"],
    }
    status = run(cfg, conditions, load_model)
    print(json.dumps(status, indent=1))
    return status


if __name__ == "__main__":
    main()
