"""
MQAR data, training and evaluation (PDF §2 Part B).

Fixed protocol (PDF table, p.7) — identical for every attention operator:
  AdamW, peak LR 8e-4, weight decay 0.01, batch 32, 4,500 optimizer updates,
  200 linear-warmup updates then linear decay to 15% of peak (reached at update 4,500),
  global grad-norm clip 1.0, checkpoint evaluation at 500/1000/1500/2000/3000/4500,
  seeds 6304/6305/6306.

Data (PDF p.7 + data repo docs/DATA_FORMATS.md):
  * training stream: the SUPPLIED generator `scripts/mqar_generator.py::make_example`
    with split='train' (hash-seeded, disjoint from development/test by construction),
    indices 0 .. 4500*32-1 of the (K, regime) the run is trained on.  Every example is
    seen once.  Seed controls initialisation (torch.manual_seed) and the order of the
    144,000 examples (a seeded permutation), never the examples themselves.
  * development: frozen `data/mqar/mqar_development.jsonl.gz` (256 per K x regime) —
    checkpoint learning curves only, never model selection.
  * final test: frozen `data/mqar/mqar_test.jsonl.gz` (1,024 per K x regime).
  The generator's own `validate()` invariants are run on every training example
  (inside make_example), on every frozen dev/test record, and a sample of test indices
  is regenerated and compared to the frozen file before any training (`verify_data`).

Loss / metric: cross-entropy over the K marked query positions only (64 value classes,
label = answer_id - 128); accuracy pooled over supervised query positions.
"""

import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .mqar_model import (VALUE_OFFSET, count_trainable_params, make_mqar_model,
                         MQARTransformer)


CHECKPOINT_STEPS = [500, 1000, 1500, 2000, 3000, 4500]
N_UPDATES = 4500
BATCH_SIZE = 32
PEAK_LR = 8e-4
WEIGHT_DECAY = 0.01
WARMUP = 200
MIN_LR_FRAC = 0.15
GRAD_CLIP = 1.0
SEEDS = (6304, 6305, 6306)
KS = (4, 8, 16)
REGIMES = ("near", "far", "mixed")
DIST_BINS = ((0, 32), (32, 64), (64, 128), (128, 256))
T_SEQ = 256


# ── Schedule ──────────────────────────────────────────────────────────────────

def linear_warmup_decay(step: int, warmup_steps: int = WARMUP, total_steps: int = N_UPDATES,
                        min_frac: float = MIN_LR_FRAC) -> float:
    """LR multiplier for the update with 0-based index `step`.
    Linear warmup to 1.0 over `warmup_steps` updates, then linear decay reaching
    exactly `min_frac` at the last update (index total_steps-1)."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - 1 - warmup_steps)
    return max(min_frac, 1.0 - t * (1.0 - min_frac))


# ── Data ──────────────────────────────────────────────────────────────────────

def load_generator(data_repo: str):
    """Import the supplied `scripts/mqar_generator.py` from the data repository."""
    path = Path(data_repo) / "scripts" / "mqar_generator.py"
    if not path.exists():
        raise FileNotFoundError(f"supplied MQAR generator not found at {path}")
    spec = importlib.util.spec_from_file_location("pa4_mqar_generator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.T == T_SEQ, f"generator T={mod.T}, expected {T_SEQ}"
    return mod


def read_jsonl(path) -> List[dict]:
    import gzip
    p = Path(path)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def records_to_arrays(records: Sequence[dict]) -> Dict[str, np.ndarray]:
    """Pack records with a common K into arrays."""
    Ks = {r["K"] for r in records}
    assert len(Ks) == 1, f"records mix K values {Ks}"
    return {
        "tokens": np.asarray([r["tokens"] for r in records], dtype=np.int16),
        "query_positions": np.asarray([r["query_positions"] for r in records], dtype=np.int16),
        "answer_ids": np.asarray([r["answer_ids"] for r in records], dtype=np.int16),
        "distances": np.asarray([r["distances"] for r in records], dtype=np.int16),
        "source_positions": np.asarray([r["source_positions"] for r in records], dtype=np.int16),
    }


def group_by_cell(records: Sequence[dict]) -> Dict[tuple, Dict[str, np.ndarray]]:
    cells: Dict[tuple, list] = {}
    for r in records:
        cells.setdefault((r["K"], r["regime"]), []).append(r)
    return {k: records_to_arrays(v) for k, v in sorted(cells.items())}


def training_stream(gen, K: int, regime: str, n: int = N_UPDATES * BATCH_SIZE,
                    cache_dir: Optional[str] = None,
                    progress: Optional[Callable[[str], None]] = None) -> Dict[str, np.ndarray]:
    """Examples make_example('train', K, regime, i) for i in [0, n), as arrays.
    Cached as .npz (generation is CPU-bound, ~1 ms/example at K=16)."""
    cache = None
    if cache_dir:
        cache = Path(cache_dir) / f"train_K{K}_{regime}_n{n}.npz"
        if cache.exists():
            z = np.load(cache)
            return {k: z[k] for k in z.files}
    recs = []
    t0 = time.time()
    for i in range(n):
        recs.append(gen.make_example("train", K, regime, i))   # validate() runs inside
        if progress and (i + 1) % 20000 == 0:
            progress(f"train stream K={K} {regime}: {i + 1}/{n} ({time.time() - t0:.0f}s)")
    arr = records_to_arrays(recs)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + ".tmp.npz")
        np.savez(tmp, **arr)
        os.replace(tmp, cache)
    return arr


def verify_data(gen, test_records: Sequence[dict], dev_records: Sequence[dict],
                n_regen: int = 4) -> dict:
    """Run the supplied invariant checks on every frozen record and confirm that the
    generator reproduces a sample of the frozen test records exactly."""
    for r in list(test_records) + list(dev_records):
        gen.validate(r)
    cells_t = {}
    for r in test_records:
        cells_t.setdefault((r["K"], r["regime"]), []).append(r)
    regen_ok = 0
    for (K, regime), rs in sorted(cells_t.items()):
        for r in rs[:n_regen]:
            idx = int(r["id"].rsplit("-", 1)[1])
            g = gen.make_example("test", K, regime, idx)
            for f in ("tokens", "query_positions", "answer_ids", "source_positions", "distances"):
                if g[f] != r[f]:
                    raise AssertionError(f"generator does not reproduce frozen {r['id']} field {f}")
            regen_ok += 1
    counts = {f"K{K}_{reg}": len(v) for (K, reg), v in sorted(cells_t.items())}
    return {"n_test": len(test_records), "n_dev": len(dev_records),
            "test_counts": counts, "regenerated_and_matched": regen_ok, "invariants": "passed"}


def _to_batch(arr: Dict[str, np.ndarray], idx: np.ndarray, device):
    tok = torch.as_tensor(arr["tokens"][idx].astype(np.int64), device=device)
    qp = torch.as_tensor(arr["query_positions"][idx].astype(np.int64), device=device)
    lab = torch.as_tensor(arr["answer_ids"][idx].astype(np.int64), device=device) - VALUE_OFFSET
    return tok, qp, lab


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_cells(model: MQARTransformer, cells: Dict[tuple, Dict[str, np.ndarray]], device,
                   batch_size: int = 256, override=None) -> dict:
    """Accuracy per (K, regime), per distance bin, and an exact-distance histogram.

    Distances are the SAVED source-query distances (query_pos - source_pos), never
    inferred from the query position.  Returns JSON-serialisable aggregates:
      cells[f"K{K}_{regime}"] = {n, correct, acc, loss, bins: {"[a,b)": {n, correct, acc}}}
      by_K_distance[f"K{K}"] = {"n": [256 ints], "correct": [256 ints]}  (index = distance)
    """
    was_training = model.training
    model.eval()
    model.set_eval_override(override)
    ce = nn.CrossEntropyLoss(reduction="sum")
    out_cells, by_kd = {}, {}
    try:
        for (K, regime), arr in cells.items():
            N = arr["tokens"].shape[0]
            corr_all = np.zeros((N, K), dtype=bool)
            loss_sum = 0.0
            for s in range(0, N, batch_size):
                idx = np.arange(s, min(N, s + batch_size))
                tok, qp, lab = _to_batch(arr, idx, device)
                logits = model(tok, qp).float()
                loss_sum += ce(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1)).item()
                corr_all[idx] = (logits.argmax(-1) == lab).cpu().numpy()
            dist = arr["distances"].astype(np.int64)
            bins = {}
            for lo, hi in DIST_BINS:
                m = (dist >= lo) & (dist < hi)
                n = int(m.sum())
                c = int(corr_all[m].sum())
                bins[f"[{lo},{hi})"] = {"n": n, "correct": c, "acc": c / n if n else None}
            n_tot = int(corr_all.size)
            c_tot = int(corr_all.sum())
            out_cells[f"K{K}_{regime}"] = {"K": K, "regime": regime, "n_sequences": N,
                                           "n": n_tot, "correct": c_tot, "acc": c_tot / n_tot,
                                           "loss": loss_sum / n_tot, "bins": bins}
            kd = by_kd.setdefault(f"K{K}", {"n": [0] * T_SEQ, "correct": [0] * T_SEQ})
            hist_n = np.bincount(dist.reshape(-1), minlength=T_SEQ)
            hist_c = np.bincount(dist.reshape(-1), weights=corr_all.reshape(-1).astype(np.float64),
                                 minlength=T_SEQ)
            kd["n"] = [int(a + b) for a, b in zip(kd["n"], hist_n[:T_SEQ])]
            kd["correct"] = [int(a + b) for a, b in zip(kd["correct"], hist_c[:T_SEQ])]
    finally:
        model.set_eval_override(None)
        model.train(was_training)
    return {"cells": out_cells, "by_K_distance": by_kd,
            "override": None if override is None else list(override)}


def summarize_regimes(ev: dict) -> Dict[str, float]:
    """Flat {K{K}_{regime}: acc} view for logs."""
    return {k: round(v["acc"], 4) for k, v in ev["cells"].items()}


# ── Training ──────────────────────────────────────────────────────────────────

def _sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def train_mqar(
    attn_type: str,
    train_arr: Dict[str, np.ndarray],
    dev_cells: Dict[tuple, Dict[str, np.ndarray]],
    seed: int,
    out_dir: str,
    device: str = "cuda",
    n_updates: int = N_UPDATES,
    batch_size: int = BATCH_SIZE,
    checkpoint_steps: Sequence[int] = CHECKPOINT_STEPS,
    log: Callable[[str], None] = print,
) -> dict:
    """Train one MQAR model from scratch with the fixed protocol.

    Writes to out_dir: curves.json, ckpt_step{N}.pt (every checkpoint step, the last is
    the final checkpoint).  Returns {model, curves, train_stats}.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_mqar_model(attn_type, device=device)
    n_params = count_trainable_params(model)
    opt = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    sched = LambdaLR(opt, lr_lambda=lambda s: linear_warmup_decay(s, WARMUP, n_updates, MIN_LR_FRAC))
    loss_fn = nn.CrossEntropyLoss()

    N = train_arr["tokens"].shape[0]
    need = n_updates * batch_size
    if N < need:
        raise ValueError(f"training stream has {N} examples, need {need}")
    order = np.random.RandomState(seed).permutation(need)

    curves = {"loss": [], "lr": [], "grad_norm": [], "checkpoints": []}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.train()
    train_time = 0.0
    _sync(device)
    t_last = time.perf_counter()
    for step in range(n_updates):
        idx = order[step * batch_size:(step + 1) * batch_size]
        tok, qp, lab = _to_batch(train_arr, idx, device)
        logits = model(tok, qp)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        curves["lr"].append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
        curves["loss"].append(float(loss.item()))
        curves["grad_norm"].append(float(gn))
        s1 = step + 1
        if s1 in checkpoint_steps:
            _sync(device)
            train_time += time.perf_counter() - t_last
            ev = evaluate_cells(model, dev_cells, device)
            torch.save(model.state_dict(), out / f"ckpt_step{s1}.pt")
            curves["checkpoints"].append({"step": s1, "dev": ev["cells"],
                                          "train_seconds_so_far": train_time})
            log(f"[{attn_type} s{seed}] step {s1} loss {np.mean(curves['loss'][-100:]):.4f} "
                f"dev {summarize_regimes(ev)}")
            _sync(device)
            t_last = time.perf_counter()
    peak = (torch.cuda.max_memory_allocated() / 2 ** 20
            if str(device).startswith("cuda") and torch.cuda.is_available() else None)
    stats = {"n_trainable_params": n_params, "train_seconds": train_time,
             "ms_per_update": 1000.0 * train_time / n_updates, "peak_alloc_mb_train": peak,
             "n_updates": n_updates, "batch_size": batch_size, "examples_seen": need,
             "final_train_loss_mean_last100": float(np.mean(curves["loss"][-100:]))}
    with open(out / "curves.json", "w") as fh:
        json.dump(curves, fh)
    return {"model": model, "curves": curves, "train_stats": stats}


def failed_to_learn(curves: dict, chance_loss: float = math.log(64), tol: float = 0.05) -> bool:
    """Flag a run whose final training loss is still within `tol` (relative) of the
    uniform-label loss ln 64 — reported, never used to drop a run."""
    last = float(np.mean(curves["loss"][-100:]))
    return last >= (1 - tol) * chance_loss
