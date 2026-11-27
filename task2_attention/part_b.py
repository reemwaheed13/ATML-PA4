"""
Task 2 Part B driver logic: the 42-run MQAR grid and the forward-only sweep.

Grid (PDF §2 steps B1, B2, B3; every run x seeds 6304, 6305, 6306):
  B1  dense   K=4,8,16 mixed  + dense K=16 near (control)                4 configs
      window  K=4,8,16 near   + window K=16 mixed                        4 configs
  B2  linear  K=4,8,16 mixed                                             3 configs
      dilated K=4,16 mixed                                               2 configs
  B3  local_linear K=16 mixed (intervention, see B3_SPEC)                1 config
  => 14 configs x 3 seeds = 42 runs (asserted in build_grid()).

Every model is evaluated on the SAME frozen final-test set (all 9 K x regime cells,
1,024 sequences each).  Dense models are additionally evaluated with a test-time-only
W=64 window mask (B1 isolation control) and a test-time-only content top-64 mask
(RQ2: content-selected positions, post-hoc).

Outputs (results_root = .../task2/b):
  grid_manifest.json           the 42 runs, protocol, B3 spec/prediction, data checks
  runs/<run>/summary.json      written last; marks completion
  runs/<run>/curves.json, ckpt_step{500..4500}.pt, error.json on failure
  forward_sweep/<attn>_T<T>.json
"""

import math
import platform
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from common.runio import read_json, run_resumable, write_json
from .mqar_model import (ATTN_TYPES, B3_WINDOW, N_HEADS, HEAD_DIM, HIDDEN_DIM, N_LAYERS,
                         accessible_positions_per_query, count_trainable_params,
                         make_mqar_model, persistent_state_bytes)
from . import mqar_train as mt

GRID_CONFIGS = [
    # (step, attn_type, K, train regime)
    ("B1", "dense", 4, "mixed"), ("B1", "dense", 8, "mixed"), ("B1", "dense", 16, "mixed"),
    ("B1", "dense", 16, "near"),
    ("B1", "window", 4, "near"), ("B1", "window", 8, "near"), ("B1", "window", 16, "near"),
    ("B1", "window", 16, "mixed"),
    ("B2", "linear", 4, "mixed"), ("B2", "linear", 8, "mixed"), ("B2", "linear", 16, "mixed"),
    ("B2", "dilated", 4, "mixed"), ("B2", "dilated", 16, "mixed"),
    ("B3", "local_linear", 16, "mixed"),
]

B3_SPEC = {
    "name": "local_linear",
    "description": ("Head-split local/global hybrid: in every block heads 0,1 are causal softmax "
                    f"attention over the preceding W={B3_WINDOW} positions (K/V buffer stored in FP16), "
                    "heads 2,3 are causal ELU+1 linear attention (FP32 state). Same q/k/v/o projections, "
                    "norms, MLP, optimizer and 4,500-update budget as every other run."),
    "motivation": ("B2 linear attention compresses all K associations into a d x d state per head, "
                   "so recall should degrade with K (associative interference). An exact local path "
                   "gives softmax retrieval for nearby pairs at no extra persistent state if the "
                   "state budget is re-split between a small FP16 buffer and fewer linear heads "
                   "(cf. Based, Arora et al. 2024 / Zoology)."),
    "comparator": "linear (B2) at K=16 mixed, seeds 6304-6306",
    "state_matching": ("linear: 4 heads x (32*32+32) x 4 B = 16,896 B/layer. local_linear: 2 linear heads "
                       "x 1,056 x 4 B = 8,448 B + 2 local heads x 33 tokens x (K+V) x 32 x 2 B (FP16) = "
                       "8,448 B -> 16,896 B/layer, exactly equal. Allowed adjustment: FP16 precision for "
                       "the local K/V buffer; W=33 is the window that makes the totals equal."),
    "trainable_params": "identical to every other operator (no new parameters; 0% difference, within the 5% limit)",
    "attended_positions": f"local heads: min(i+1, {B3_WINDOW}) positions; linear heads: fixed state of 32x32+32",
    "prediction": ("PREDICTION (recorded before evaluation): relative to linear K=16 mixed, local_linear "
                   "will raise accuracy on the near regime and the [0,32) distance bin substantially "
                   "(exact softmax retrieval inside the 33-token window), help [32,64) partially through "
                   "the two-layer local path (receptive field 64), and leave the far regime / [128,256) "
                   "bin within a few points of the linear baseline because the remaining linear state "
                   "is half as large. It will stay below dense K=16 mixed overall."),
    "dev_selection_rule": ("No hyperparameter is selected on the development set: W=33 is fixed by the "
                           "state-byte equality, head split fixed a priori, final (update-4,500) checkpoint "
                           "reported. Development accuracy is used only for learning curves."),
}


def run_name(attn: str, K: int, regime: str, seed: int) -> str:
    return f"{attn}_K{K}_{regime}_s{seed}"


def build_grid() -> List[dict]:
    grid = []
    for step, attn, K, regime in GRID_CONFIGS:
        for seed in mt.SEEDS:
            grid.append({"run": run_name(attn, K, regime, seed), "step": step, "attn_type": attn,
                         "K": K, "train_regime": regime, "seed": seed})
    assert len(grid) == 42, f"grid has {len(grid)} runs, PDF requires 42"
    assert len({g["run"] for g in grid}) == 42
    return grid


GRID = build_grid()
GRID_BY_NAME = {g["run"]: g for g in GRID}


def select_runs(spec: str) -> List[str]:
    if spec in ("all", ""):
        return [g["run"] for g in GRID]
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok in ("B1", "B2", "B3"):
            out += [g["run"] for g in GRID if g["step"] == tok]
        elif tok in ATTN_TYPES:
            out += [g["run"] for g in GRID if g["attn_type"] == tok]
        elif tok in GRID_BY_NAME:
            out.append(tok)
        else:
            raise ValueError(f"unknown run/group {tok!r}")
    return list(dict.fromkeys(out))


# ── Data loading shared by all runs ───────────────────────────────────────────

class MQARData:
    def __init__(self, data_repo: str, cache_dir: Optional[str], n_train: int, log=print):
        self.gen = mt.load_generator(data_repo)
        root = Path(data_repo) / "data" / "mqar"
        self.test_records = mt.read_jsonl(root / "mqar_test.jsonl.gz")
        self.dev_records = mt.read_jsonl(root / "mqar_development.jsonl.gz")
        self.check = mt.verify_data(self.gen, self.test_records, self.dev_records)
        self.test_cells = mt.group_by_cell(self.test_records)
        self.dev_cells = mt.group_by_cell(self.dev_records)
        self.cache_dir = cache_dir
        self.n_train = n_train
        self.log = log
        self._streams: Dict[tuple, dict] = {}
        log(f"[data] {self.check}")

    def stream(self, K: int, regime: str) -> dict:
        key = (K, regime)
        if key not in self._streams:
            t0 = time.time()
            self._streams[key] = mt.training_stream(self.gen, K, regime, self.n_train,
                                                    self.cache_dir, self.log)
            self.log(f"[data] train stream K={K} {regime}: {self.n_train} examples ({time.time() - t0:.0f}s)")
        return self._streams[key]


def _chance(K: int) -> dict:
    return {"label_space": 1.0 / 64, "random_choice_present_values": 1.0 / K}


def _env(device) -> dict:
    return {"torch": torch.__version__, "python": platform.python_version(), "device": str(device),
            "cuda_device": (torch.cuda.get_device_name(0)
                            if str(device).startswith("cuda") and torch.cuda.is_available() else None)}


def train_one(cfg: dict, data: MQARData, out: Path, device: str, n_updates: int, log=print) -> dict:
    stream = data.stream(cfg["K"], cfg["train_regime"])
    ckpts = sorted(set([s for s in mt.CHECKPOINT_STEPS if s <= n_updates] + [n_updates]))
    t0 = time.time()
    res = mt.train_mqar(cfg["attn_type"], stream, data.dev_cells, cfg["seed"], str(out), device,
                        n_updates=n_updates, checkpoint_steps=ckpts, log=log)
    model = res["model"]
    t_eval = time.time()
    test = mt.evaluate_cells(model, data.test_cells, device)
    controls = {}
    if cfg["attn_type"] == "dense":
        controls["test_time_window64"] = mt.evaluate_cells(model, data.test_cells, device,
                                                           override=("window", 64))
        controls["test_time_topk64"] = mt.evaluate_cells(model, data.test_cells, device,
                                                         override=("topk", 64))
    eval_seconds = time.time() - t_eval
    dense_params = count_trainable_params(make_mqar_model("dense", device="cpu"))
    n_params = res["train_stats"]["n_trainable_params"]
    summary = {
        **cfg,
        "protocol": {"n_updates": n_updates, "batch_size": mt.BATCH_SIZE, "peak_lr": mt.PEAK_LR,
                     "weight_decay": mt.WEIGHT_DECAY, "warmup": mt.WARMUP, "min_lr_frac": mt.MIN_LR_FRAC,
                     "grad_clip": mt.GRAD_CLIP, "checkpoint_steps": ckpts,
                     "train_examples": f"make_example('train', {cfg['K']}, '{cfg['train_regime']}', i), "
                                       f"i < {n_updates * mt.BATCH_SIZE}, seeded order"},
        "full_protocol": n_updates == mt.N_UPDATES,
        "train_stats": res["train_stats"],
        "param_ratio_vs_dense": n_params / dense_params,
        "failed_to_learn": mt.failed_to_learn(res["curves"]),
        "test": test,
        "test_time_controls": controls,
        "eval_seconds": eval_seconds,
        "wall_seconds": time.time() - t0,
        "chance": _chance(cfg["K"]),
        "theory_T256": {"access": accessible_positions_per_query(cfg["attn_type"]),
                        "state": persistent_state_bytes(cfg["attn_type"])},
        "implementation_note": {
            "dense": "SDPA is_causal=True (no explicit mask tensor)",
            "window": "SDPA with an explicit dense (T,T) boolean mask: correctness implementation, not a sparse kernel",
            "dilated": "SDPA with an explicit dense (H,T,T) boolean mask: correctness implementation, not a sparse kernel",
            "linear": "exact parallel form materialising a (T,T) kernel matrix per head (training); recurrent form identical",
            "local_linear": "local heads: SDPA + dense (T,T) boolean mask, FP16-rounded K/V; linear heads: parallel (T,T) form",
        }[cfg["attn_type"]],
        "final_checkpoint": f"ckpt_step{n_updates}.pt",
        "env": _env(device),
    }
    write_json(out / "summary.json", summary)
    log(f"[{cfg['run']}] test {mt.summarize_regimes(test)}  ({summary['wall_seconds']:.0f}s)")
    return summary


def run_grid(results_root: str, data_repo: str, runs: List[str], device: str,
             overwrite: bool = False, n_updates: int = mt.N_UPDATES, cache_dir: Optional[str] = None,
             log=print) -> Dict[str, str]:
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    data = MQARData(data_repo, cache_dir, n_updates * mt.BATCH_SIZE, log)
    write_json(root / "grid_manifest.json", {
        "n_runs": len(GRID), "grid": GRID, "configs": GRID_CONFIGS, "seeds": list(mt.SEEDS),
        "b3": B3_SPEC, "data_check": data.check, "n_updates": n_updates,
        "theory_T256": {a: {"access": accessible_positions_per_query(a), "state": persistent_state_bytes(a)}
                        for a in ATTN_TYPES},
    })
    # Order runs so that runs sharing a (K, regime) stream are adjacent.
    runs = sorted(runs, key=lambda r: (GRID_BY_NAME[r]["K"], GRID_BY_NAME[r]["train_regime"],
                                       [g["run"] for g in GRID].index(r)))
    return run_resumable(runs, lambda n: root / "runs" / n,
                         lambda n, out: train_one(GRID_BY_NAME[n], data, out, device, n_updates, log),
                         overwrite=overwrite, log=log)


# ── Forward-only sweep (B-R2) ─────────────────────────────────────────────────

SWEEP_T = (256, 1024, 4096, 8192)
SWEEP_B = 4
SWEEP_METHODS = {
    # name: (attn_type, linear_mode, what it materialises)
    "dense": ("dense", "parallel", "nothing beyond SDPA (is_causal=True)"),
    "window": ("window", "parallel", "dense (T,T) boolean mask + dense scores inside SDPA"),
    "dilated": ("dilated", "parallel", "dense (H,T,T) boolean mask + dense scores inside SDPA"),
    "linear_parallel": ("linear", "parallel", "(B,H,T,T) linear kernel matrix (linear prefix as a dense matrix)"),
    "linear_recurrent": ("linear", "recurrent", "O(d^2) running state only; Python loop over T"),
    "local_linear": ("local_linear", "parallel", "dense (T,T) mask for local heads + (B,2,T,T) linear kernel matrix"),
}


def _sweep_one(method: str, T: int, device: str, n_warmup: int, n_rep: int) -> dict:
    attn, mode, mat = SWEEP_METHODS[method]
    torch.manual_seed(6304)
    model = make_mqar_model(attn, device=device, linear_mode=mode).eval()
    g = torch.Generator(device="cpu").manual_seed(6304 + T)
    h = torch.randn(SWEEP_B, T, HIDDEN_DIM, generator=g).to(device)   # fixed random embeddings
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    rec = {"method": method, "attn_type": attn, "linear_mode": mode, "T": T, "B": SWEEP_B,
           "H": N_HEADS, "d": HEAD_DIM, "layers": N_LAYERS, "materializes": mat,
           "n_warmup": n_warmup, "n_rep": n_rep, "dtype": "float32"}
    try:
        with torch.no_grad():
            for _ in range(n_warmup):
                model.encode(h)
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
            times = []
            for _ in range(n_rep):
                if cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.encode(h)
                if cuda:
                    torch.cuda.synchronize()
                times.append(1000 * (time.perf_counter() - t0))
        rec.update(status="ok", times_ms=times, mean_ms=float(np.mean(times)),
                   std_ms=float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
                   median_ms=float(np.median(times)),
                   tokens_per_s=SWEEP_B * T / (np.mean(times) / 1000))
        if cuda:
            rec["peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            rec["peak_alloc_above_inputs_mb"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
            rec["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2 ** 20
    except torch.cuda.OutOfMemoryError as e:
        rec.update(status="oom", message=str(e)[:500])
    finally:
        del model, h
        if cuda:
            torch.cuda.empty_cache()
    rec["theory"] = {"pairs_per_layer_all_heads": accessible_positions_per_query(attn, T)["attended_pairs_total"] / N_LAYERS,
                     "state_bytes_per_layer": persistent_state_bytes(attn, T)["bytes_per_layer"],
                     "mask_bytes": {"dense": 0, "window": T * T, "dilated": 4 * T * T, "linear": 0,
                                    "local_linear": T * T}[attn],
                     "linear_kernel_matrix_bytes": (SWEEP_B * {"linear": 4, "local_linear": 2}.get(attn, 0)
                                                    * T * T * 4 if mode == "parallel" else 0)}
    return rec


def run_sweep(results_root: str, device: str, overwrite: bool = False, Ts=SWEEP_T,
              methods=None, n_warmup: int = 3, n_rep: int = 10, log=print) -> Dict[str, str]:
    out = Path(results_root) / "forward_sweep"
    out.mkdir(parents=True, exist_ok=True)
    methods = list(methods or SWEEP_METHODS)
    status = {}
    for m in methods:
        for T in Ts:
            f = out / f"{m}_T{T}.json"
            if f.exists() and not overwrite:
                status[f.stem] = "skipped"
                continue
            try:
                rec = _sweep_one(m, T, device, n_warmup, n_rep)
                rec["env"] = _env(device)
                write_json(f, rec)
                status[f.stem] = rec["status"]
                log(f"[sweep] {m} T={T}: {rec['status']} "
                    f"{rec.get('mean_ms', float('nan')):.2f} ms, peak {rec.get('peak_alloc_mb', float('nan')):.0f} MB")
            except Exception as e:  # noqa: BLE001
                import traceback
                write_json(out / f"{f.stem}.error.json", {"exception_type": type(e).__name__,
                                                          "message": str(e), "traceback": traceback.format_exc()})
                status[f.stem] = "failed"
                log(f"[sweep FAILED] {m} T={T}: {e}")
    return status
