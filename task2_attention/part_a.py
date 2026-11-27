"""
Task 2 Part A driver logic: dense vs tiled vs SDPA on identical synthetic tensors.

Shapes (PDF): B=2, H=4, d_h=64, L in {256, 512, 1024, 2048, 4096, 8192}, FP32 and FP16.
Per (dtype, L, method) — one resumable JSON each under <results>/a/conditions/:
  * forward-only latency and forward+backward latency: 3 warm-up + 10 timed,
    CUDA-synchronised repetitions each;
  * peak allocated AND reserved GPU memory for the forward and for forward+backward
    (reset before each measurement; inputs/reference are resident and reported);
  * output and gradient (dQ, dK, dV) error against the dense FP32 reference computed on
    the same FP32 tensors (FP16 methods get the FP16-cast copies);
  * status 'ok' | 'unsupported' (RuntimeError from a forced SDPA backend) | 'oom'.
Methods: dense (A1), tiled (A2, tile 128), sdpa (default dispatch, with the dispatched
kernel recorded from the profiler), sdpa_flash / sdpa_efficient / sdpa_math / sdpa_cudnn
(forced).  The dense FP32 reference is skipped (and errors left null) if it does not
fit — larger shapes are still benchmarked.

Extra checks (<results>/a/checks.json): causality by perturbing future K/V for every
method; tile sizes {16,32,64,128,256} x dtypes at L=1024; all-masked query rows
(key padding) and extreme scores (Q scaled x30, x300); theoretical intermediate
storage of dense vs tiled.
"""

import math
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from common.runio import read_json, write_json
from . import attention_ops as ao

B, H, D = 2, 4, 64
LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
TILE = 128
METHODS = ["dense", "tiled", "sdpa", "sdpa_flash", "sdpa_efficient", "sdpa_math", "sdpa_cudnn"]
LIMITATION = ("dense and tiled are Python reference implementations; the tiled backward is autograd "
              "through the Python loop and stores every tile's intermediates - it is not FlashAttention "
              "and its speed/memory do not characterise a fused tiled kernel")


def _fn(method: str, tile: int = TILE) -> Callable:
    if method == "dense":
        return ao.dense_causal_attention
    if method == "tiled":
        return lambda q, k, v, **kw: ao.tiled_causal_attention(q, k, v, tile_size=tile, **kw)
    if method == "sdpa":
        return ao.sdpa_attention
    if method.startswith("sdpa_"):
        be = method.split("_", 1)[1]
        return lambda q, k, v, **kw: ao.sdpa_attention(q, k, v, backend=be, **kw)
    raise ValueError(method)


def make_inputs(L: int, device, seed: int = 6304):
    g = torch.Generator(device="cpu").manual_seed(seed + L)
    Q, K, V, dO = (torch.randn(B, H, L, D, generator=g) for _ in range(4))
    return [t.to(device) for t in (Q, K, V, dO)]


def _cuda(device) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def _sync(device):
    if _cuda(device):
        torch.cuda.synchronize()


def _time(fn, device, n_warmup: int, n_rep: int) -> dict:
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_rep):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(1000 * (time.perf_counter() - t0))
    ts = sorted(times)
    mean = sum(times) / len(times)
    sd = math.sqrt(sum((t - mean) ** 2 for t in times) / max(1, len(times) - 1))
    return {"times_ms": times, "mean_ms": mean, "std_ms": sd, "median_ms": ts[len(ts) // 2],
            "n_warmup": n_warmup, "n_rep": n_rep}


def _mem(fn, device) -> dict:
    if not _cuda(device):
        return {}
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return {"peak_alloc_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "peak_alloc_above_baseline_mb": (torch.cuda.max_memory_allocated() - base) / 2 ** 20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20,
            "baseline_alloc_mb": base / 2 ** 20}


def reference(L: int, device) -> Optional[dict]:
    """Dense FP32 output and gradients, or None if it does not fit."""
    Q, K, V, dO = make_inputs(L, device)
    q, k, v = (t.clone().requires_grad_(True) for t in (Q, K, V))
    try:
        out = ao.dense_causal_attention(q, k, v)
        out.backward(dO)
        return {"out": out.detach(), "dQ": q.grad.detach(), "dK": k.grad.detach(), "dV": v.grad.detach()}
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        del q, k, v
        if _cuda(device):
            torch.cuda.empty_cache()


def run_condition(method: str, dtype_name: str, L: int, device, ref: Optional[dict],
                  n_warmup: int = 3, n_rep: int = 10) -> dict:
    dt = DTYPES[dtype_name]
    fn = _fn(method)
    Q, K, V, dO = (t.to(dt) for t in make_inputs(L, device))
    rec = {"method": method, "dtype": dtype_name, "L": L, "B": B, "H": H, "d": D,
           "tile_size": TILE if method == "tiled" else None, "limitation": LIMITATION
           if method in ("dense", "tiled") else None,
           "reference": "dense FP32 on the same FP32 tensors" if ref is not None else None}
    try:
        with torch.no_grad():
            out = fn(Q, K, V)
        rec["output_error"] = ao.compare(out, ref["out"]) if ref is not None else None
        rec["output_finite"] = bool(torch.isfinite(out).all())
        del out
        with torch.no_grad():
            rec["forward"] = _time(lambda: fn(Q, K, V), device, n_warmup, n_rep)
            rec["forward_mem"] = _mem(lambda: fn(Q, K, V), device)
        q, k, v = (t.clone().requires_grad_(True) for t in (Q, K, V))

        def fb():
            for t in (q, k, v):
                t.grad = None
            fn(q, k, v).backward(dO)

        fb()
        if ref is not None:
            rec["grad_error"] = {g: ao.compare(t.grad, ref[g]) for g, t in (("dQ", q), ("dK", k), ("dV", v))}
        rec["forward_backward"] = _time(fb, device, n_warmup, n_rep)
        rec["forward_backward_mem"] = _mem(fb, device)
        if method == "sdpa" and _cuda(device):
            rec["dispatched_forward"] = ao.dispatched_sdpa_backend(Q, K, V)
            rec["dispatched_backward"] = ao.dispatched_sdpa_backend(Q, K, V, backward=True)
        rec["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        rec.update(status="oom", message=str(e)[:400])
    except RuntimeError as e:
        msg = str(e)
        if method.startswith("sdpa_") and ("No available kernel" in msg or "not supported" in msg.lower()
                                           or "Aborting" in msg or "kernel" in msg.lower()):
            rec.update(status="unsupported", message=msg[:400])
        else:
            raise
    finally:
        if _cuda(device):
            torch.cuda.empty_cache()
    return rec


def run_sweep(results_root: str, device, lengths=LENGTHS, dtypes=("fp32", "fp16"), methods=METHODS,
              overwrite=False, n_warmup=3, n_rep=10, log=print) -> Dict[str, str]:
    out = Path(results_root) / "conditions"
    out.mkdir(parents=True, exist_ok=True)
    status = {}
    for L in lengths:
        todo = [(dn, m) for dn in dtypes for m in methods
                if overwrite or not (out / f"{dn}_L{L}_{m}.json").exists()]
        if not todo:
            status.update({f"{dn}_L{L}_{m}": "skipped" for dn in dtypes for m in methods})
            continue
        t0 = time.time()
        ref = reference(L, device)
        log(f"[A] L={L}: dense FP32 reference {'ok' if ref is not None else 'OOM (errors left null)'}"
            f" ({time.time() - t0:.1f}s)")
        for dn, m in todo:
            name = f"{dn}_L{L}_{m}"
            try:
                rec = run_condition(m, dn, L, device, ref, n_warmup, n_rep)
                rec["reference_available"] = ref is not None
                write_json(out / f"{name}.json", rec)
                status[name] = rec["status"]
                fw = rec.get("forward", {}).get("mean_ms")
                fb = rec.get("forward_backward", {}).get("mean_ms")
                err = (rec.get("output_error") or {}).get("max_abs")
                log(f"[A] {name}: {rec['status']} fwd {fw} ms fwd+bwd {fb} ms max|err| {err}")
            except Exception as e:  # noqa: BLE001
                write_json(out / f"{name}.error.json", {"exception_type": type(e).__name__,
                                                        "message": str(e), "traceback": traceback.format_exc()})
                status[name] = "failed"
                log(f"[A FAILED] {name}: {e}")
        del ref
        if _cuda(device):
            torch.cuda.empty_cache()
    return status


def storage_theory(L: int, tile: int = TILE, bytes_per: int = 4) -> dict:
    """Intermediate storage (forward, inference): dense keeps the (B,H,L,L) score and
    probability matrices; tiled keeps one (B,H,tile,tile) block plus FP32 running
    m, l (B,H,tile) and o (B,H,tile,d) for the current query tile."""
    dense = 2 * B * H * L * L * bytes_per
    tiled = B * H * tile * tile * 4 * 2 + B * H * tile * (2 + D) * 4
    return {"L": L, "dense_scores_and_probs_bytes": dense, "tiled_live_block_bytes": tiled,
            "note": "autograd backward of the tiled reference additionally stores every visited tile"}


def run_checks(device, lengths_storage=LENGTHS) -> dict:
    res = {"limitation": LIMITATION}
    # causality, every method
    caus = {}
    for m in METHODS:
        for dn, dt in DTYPES.items():
            if dn == "fp16" and not _cuda(device):
                continue
            try:
                caus[f"{m}_{dn}"] = ao.verify_causality(_fn(m, tile=8), L=64, device=device, dtype=dt)
            except RuntimeError as e:
                caus[f"{m}_{dn}"] = {"status": "unsupported", "message": str(e)[:200]}
    res["causality"] = caus
    # tile sizes
    L = 1024
    ref = reference(L, device)
    tiles = {}
    for dn, dt in DTYPES.items():
        if dn == "fp16" and not _cuda(device):
            continue
        Q, K, V, _ = (t.to(dt) for t in make_inputs(L, device))
        for ts in (16, 32, 64, 128, 256):
            with torch.no_grad():
                out = ao.tiled_causal_attention(Q, K, V, tile_size=ts)
            tiles[f"{dn}_tile{ts}"] = ao.compare(out, ref["out"])
    res["tile_sizes_L1024"] = tiles
    # edge cases at L=256
    L = 256
    Q, K, V, dO = make_inputs(L, device)
    km = torch.ones(B, L, dtype=torch.bool, device=device)
    km[0, :4] = False                     # batch 0, queries 0..3 have no valid key
    edge = {}
    ref_m = ao.dense_causal_attention(Q.double(), K.double(), V.double(), key_mask=km)
    for m in METHODS:
        for dn, dt in DTYPES.items():
            if dn == "fp16" and not _cuda(device):
                continue
            try:
                q, k, v = (t.to(dt).clone().requires_grad_(True) for t in (Q, K, V))
                out = _fn(m)(q, k, v, key_mask=km)
                out.float().backward(dO.float())
                rows = out[0, :, :4].float()
                edge[f"allmasked_{m}_{dn}"] = {
                    "masked_rows_finite": bool(torch.isfinite(rows).all()),
                    "masked_rows_max_abs": float(rows.abs().nan_to_num(nan=float("inf")).max()),
                    "other_rows_err_vs_fp64": ao.compare(out[:, :, 4:].double(), ref_m[:, :, 4:]),
                    "grads_finite": bool(all(torch.isfinite(t.grad).all() for t in (q, k, v)))}
            except RuntimeError as e:
                edge[f"allmasked_{m}_{dn}"] = {"status": "unsupported", "message": str(e)[:200]}
    for factor in (30.0, 300.0):
        Qs = Q * factor
        ref64 = ao.dense_causal_attention(Qs.double(), K.double(), V.double())
        for m in METHODS:
            for dn, dt in DTYPES.items():
                if dn == "fp16" and not _cuda(device):
                    continue
                try:
                    with torch.no_grad():
                        out = _fn(m)(Qs.to(dt), K.to(dt), V.to(dt))
                    edge[f"extreme_x{int(factor)}_{m}_{dn}"] = ao.compare(out.double(), ref64)
                except RuntimeError as e:
                    edge[f"extreme_x{int(factor)}_{m}_{dn}"] = {"status": "unsupported", "message": str(e)[:200]}
    res["edge_cases"] = edge
    res["edge_case_note"] = ("all-masked: key padding makes batch 0 queries 0..3 attend to nothing; A1/A2 "
                             "return 0 there by construction; SDPA behaviour recorded as observed. "
                             "Extreme: Q scaled x30 / x300, compared with a float64 dense reference.")
    res["storage_theory"] = [storage_theory(L) for L in lengths_storage]
    return res
