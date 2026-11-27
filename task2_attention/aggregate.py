"""
Task 2 aggregation: every table / plot listed in PDF §2 "Required Evidence".

Inputs  <root>/a/{conditions/*.json, checks.json}
        <root>/b/{grid_manifest.json, runs/*/summary.json, runs/*/curves.json, forward_sweep/*.json}
        <root>/c/{c1_validation.json, prompts/*/summary.json, packed_bench.json}
Outputs <root>/tables/*.csv|json and <root>/plots/*.png  (missing inputs are reported
        in the returned dict, never silently treated as zero).
"""

import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from common.bootstrap import bootstrap_paired_accuracy, bootstrap_paired_mean
from common.runio import read_json, write_json


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path: Path, out: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    out.append(path.name)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ── Part A ────────────────────────────────────────────────────────────────────

def aggregate_a(root: Path, tables: Path, plots: Path) -> dict:
    files = sorted(glob.glob(str(root / "a" / "conditions" / "*.json")))
    recs = [read_json(f) for f in files if not f.endswith(".error.json")]
    errors = [Path(f).name for f in files if f.endswith(".error.json")]
    if not recs:
        return {"status": "missing"}
    rows = []
    for r in recs:
        oe = r.get("output_error") or {}
        ge = r.get("grad_error") or {}
        rows.append({
            "dtype": r["dtype"], "L": r["L"], "method": r["method"], "status": r["status"],
            "fwd_ms": (r.get("forward") or {}).get("mean_ms"), "fwd_std_ms": (r.get("forward") or {}).get("std_ms"),
            "fwdbwd_ms": (r.get("forward_backward") or {}).get("mean_ms"),
            "fwdbwd_std_ms": (r.get("forward_backward") or {}).get("std_ms"),
            "n_rep": (r.get("forward") or {}).get("n_rep"),
            "fwd_peak_alloc_mb": (r.get("forward_mem") or {}).get("peak_alloc_mb"),
            "fwd_peak_reserved_mb": (r.get("forward_mem") or {}).get("peak_reserved_mb"),
            "fwdbwd_peak_alloc_mb": (r.get("forward_backward_mem") or {}).get("peak_alloc_mb"),
            "fwdbwd_peak_reserved_mb": (r.get("forward_backward_mem") or {}).get("peak_reserved_mb"),
            "out_max_abs_err": oe.get("max_abs"), "out_rel_fro_err": oe.get("rel_fro"),
            "dQ_max_abs_err": (ge.get("dQ") or {}).get("max_abs"), "dK_max_abs_err": (ge.get("dK") or {}).get("max_abs"),
            "dV_max_abs_err": (ge.get("dV") or {}).get("max_abs"),
            "dQ_rel_fro_err": (ge.get("dQ") or {}).get("rel_fro"),
            "dispatched_fwd": (r.get("dispatched_forward") or {}).get("backend"),
            "dispatched_bwd": (r.get("dispatched_backward") or {}).get("backend"),
            "reference_available": r.get("reference_available"), "message": r.get("message"),
        })
    rows.sort(key=lambda x: (x["dtype"], x["method"], x["L"]))
    _write_csv(tables / "a_conditions.csv", rows)
    # compact backend / unsupported-operation table
    back = defaultdict(dict)
    for x in rows:
        k = (x["dtype"], x["L"])
        back[k]["dtype"], back[k]["L"] = k
        v = x["status"]
        if x["method"] == "sdpa" and x["dispatched_fwd"]:
            v = f"ok fwd={x['dispatched_fwd']} bwd={x['dispatched_bwd']}"
        back[k][x["method"]] = v
    _write_csv(tables / "a_backends.csv", sorted(back.values(), key=lambda r: (r["dtype"], r["L"])))
    made = []
    plt = _plt()
    for metric, ylabel, fname in (("fwd_ms", "forward latency (ms)", "a_latency_forward.png"),
                                  ("fwdbwd_ms", "forward+backward latency (ms)", "a_latency_fwd_bwd.png"),
                                  ("fwd_peak_alloc_mb", "peak allocated, forward (MB)", "a_memory_forward.png"),
                                  ("fwdbwd_peak_alloc_mb", "peak allocated, fwd+bwd (MB)", "a_memory_fwd_bwd.png"),
                                  ("out_max_abs_err", "max |out - dense FP32|", "a_output_error.png"),
                                  ("dQ_max_abs_err", "max |dQ - dense FP32|", "a_grad_error.png")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, dt in zip(axes, ("fp32", "fp16")):
            for m in sorted({x["method"] for x in rows}):
                pts = [(x["L"], x[metric]) for x in rows if x["dtype"] == dt and x["method"] == m
                       and x["status"] == "ok" and x[metric] is not None and x[metric] > 0]
                if pts:
                    ax.plot(*zip(*sorted(pts)), marker="o", label=m)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{dt}, B=2 H=4 d=64, n=10 reps")
            ax.set_xlabel("sequence length L (tokens)")
            ax.grid(alpha=.3)
        axes[0].set_ylabel(ylabel)
        axes[1].legend(fontsize=7)
        _save(fig, plots / fname, made)
    checks = root / "a" / "checks.json"
    return {"status": "ok", "n_conditions": len(rows), "errors": errors, "plots": made,
            "checks": str(checks) if checks.exists() else "missing"}


# ── Part B ────────────────────────────────────────────────────────────────────

FINE_BIN = 16


def aggregate_b(root: Path, tables: Path, plots: Path) -> dict:
    man_p = root / "b" / "grid_manifest.json"
    if not man_p.exists():
        return {"status": "missing"}
    man = read_json(man_p)
    runs, missing = {}, []
    for g in man["grid"]:
        p = root / "b" / "runs" / g["run"] / "summary.json"
        if p.exists():
            runs[g["run"]] = read_json(p)
        else:
            missing.append(g["run"])
    rows = []
    for name, s in runs.items():
        K = s["K"]
        c = s["test"]["cells"]
        row = {"run": name, "step": s["step"], "attn_type": s["attn_type"], "K": K, "train_regime": s["train_regime"],
               "seed": s["seed"], "full_protocol": s["full_protocol"], "failed_to_learn": s["failed_to_learn"],
               "params": s["train_stats"]["n_trainable_params"], "param_ratio_vs_dense": s["param_ratio_vs_dense"],
               "ms_per_update": s["train_stats"]["ms_per_update"], "peak_alloc_mb_train": s["train_stats"]["peak_alloc_mb_train"],
               "final_train_loss": s["train_stats"]["final_train_loss_mean_last100"],
               "chance_label": 1 / 64, "chance_1_over_K": 1 / K}
        for k, v in c.items():
            row[f"acc_{k}"] = v["acc"]
            row[f"loss_{k}"] = v["loss"]
        for reg in ("near", "far", "mixed"):
            for b, bv in c[f"K{K}_{reg}"]["bins"].items():
                row[f"acc_trainK_{reg}_{b}"] = bv["acc"]
        allbins = defaultdict(lambda: [0, 0])
        for reg in ("near", "far", "mixed"):
            for b, bv in c[f"K{K}_{reg}"]["bins"].items():
                allbins[b][0] += bv["n"]
                allbins[b][1] += bv["correct"]
        for b, (n, cc) in allbins.items():
            row[f"acc_trainK_allregimes_{b}"] = cc / n if n else None
        rows.append(row)
    rows.sort(key=lambda r: r["run"])
    _write_csv(tables / "b_runs.csv", rows)
    # config-level mean/std over seeds
    cfg = defaultdict(list)
    for r in rows:
        cfg[(r["step"], r["attn_type"], r["K"], r["train_regime"])].append(r)
    crow = []
    for (step, a, K, reg), rs in sorted(cfg.items()):
        d = {"step": step, "attn_type": a, "K": K, "train_regime": reg, "n_seeds": len(rs),
             "seeds": ",".join(str(r["seed"]) for r in rs),
             "n_failed_to_learn": sum(bool(r["failed_to_learn"]) for r in rs)}
        for k in [k for k in rs[0] if k.startswith("acc_") or k in ("ms_per_update", "peak_alloc_mb_train", "final_train_loss")]:
            d[f"{k}_mean"] = _mean([r.get(k) for r in rs])
            d[f"{k}_std"] = _std([r.get(k) for r in rs])
        crow.append(d)
    _write_csv(tables / "b_configs.csv", crow)
    # train-vs-test mask controls
    ctrl = []
    partner = {("dense", 4, "mixed"): ("window", 4, "near"), ("dense", 8, "mixed"): ("window", 8, "near"),
               ("dense", 16, "mixed"): ("window", 16, "mixed"), ("dense", 16, "near"): ("window", 16, "near")}
    for name, s in runs.items():
        if s["attn_type"] != "dense":
            continue
        K = s["K"]
        pa, pk, preg = partner[("dense", K, s["train_regime"])]
        pname = f"{pa}_K{pk}_{preg}_s{s['seed']}"
        for reg in ("near", "far", "mixed"):
            cell = f"K{K}_{reg}"
            d = {"dense_run": name, "test_cell": cell,
                 "dense": s["test"]["cells"][cell]["acc"],
                 "dense_testtime_window64": s["test_time_controls"]["test_time_window64"]["cells"][cell]["acc"],
                 "dense_testtime_topk64": s["test_time_controls"]["test_time_topk64"]["cells"][cell]["acc"],
                 "trained_window_run": pname,
                 "trained_window": runs[pname]["test"]["cells"][cell]["acc"] if pname in runs else None,
                 "partner_note": "same K and training regime" if preg == s["train_regime"]
                                 else f"window partner trained on {preg} (PDF grid)"}
            ctrl.append(d)
    _write_csv(tables / "b_mask_controls.csv", ctrl)
    # theoretical vs measured costs
    cost = []
    for a, th in man["theory_T256"].items():
        rs = [r for r in rows if r["attn_type"] == a]
        d = {"attn_type": a, "theory_attended_pairs_T256": th["access"]["attended_pairs_total"],
             "theory_mean_positions_per_query_per_head": json.dumps(th["access"]["mean_positions_per_query_per_head"]),
             "theory_two_layer_receptive_field": th["access"]["two_layer_receptive_field"],
             "theory_state_bytes_per_layer_T256": th["state"]["bytes_per_layer"],
             "measured_ms_per_update_mean": _mean([r["ms_per_update"] for r in rs]),
             "measured_peak_alloc_mb_train_mean": _mean([r["peak_alloc_mb_train"] for r in rs]),
             "n_runs": len(rs)}
        cost.append(d)
    sweep = [read_json(f) for f in sorted(glob.glob(str(root / "b" / "forward_sweep" / "*_T*.json")))
             if not f.endswith(".error.json")]
    for r in sweep:
        r.pop("times_ms", None)
        r.pop("env", None)
    _write_csv(tables / "b_costs_theory_vs_measured.csv", cost)
    _write_csv(tables / "b_forward_sweep.csv", [{**{k: v for k, v in r.items() if k != "theory"},
                                                 **{f"theory_{k}": v for k, v in r.get("theory", {}).items()}}
                                                for r in sweep])
    # B3: intervention vs its comparator (linear K16 mixed) and dense K16 mixed, per seed
    b3 = {"spec": man.get("b3"), "rows": []}
    for seed in man["seeds"]:
        row = {"seed": seed}
        for tag, a in (("intervention", "local_linear"), ("comparator_linear", "linear"), ("dense", "dense")):
            s = runs.get(f"{a}_K16_mixed_s{seed}")
            if s is None:
                continue
            for reg in ("near", "far", "mixed"):
                row[f"{tag}_{reg}"] = s["test"]["cells"][f"K16_{reg}"]["acc"]
            for b, bv in s["test"]["cells"]["K16_mixed"]["bins"].items():
                row[f"{tag}_mixed_{b}"] = bv["acc"]
            row[f"{tag}_params"] = s["train_stats"]["n_trainable_params"]
            row[f"{tag}_state_bytes_per_layer"] = s["theory_T256"]["state"]["bytes_per_layer"]
        b3["rows"].append(row)
    write_json(tables / "b3_intervention.json", b3)
    _write_csv(tables / "b3_intervention.csv", b3["rows"])
    made = _plots_b(root, man, runs, plots, sweep)
    return {"status": "ok", "n_runs": len(runs), "missing_runs": missing, "plots": made}


def _plots_b(root: Path, man: dict, runs: Dict[str, dict], plots: Path, sweep) -> list:
    plt = _plt()
    import numpy as np
    made = []
    by_cfg = defaultdict(list)
    for name, s in runs.items():
        by_cfg[(s["attn_type"], s["K"], s["train_regime"])].append(name)
    # learning curves, one figure per mechanism
    for a in sorted({k[0] for k in by_cfg}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for (aa, K, reg), names in sorted(by_cfg.items()):
            if aa != a:
                continue
            for n in sorted(names):
                cp = root / "b" / "runs" / n / "curves.json"
                if not cp.exists():
                    continue
                cv = read_json(cp)
                loss = np.asarray(cv["loss"])
                w = max(1, min(50, len(loss)))
                sm = np.convolve(loss, np.ones(w) / w, mode="valid")
                ax1.plot(np.arange(w, len(loss) + 1), sm, lw=.8, label=f"K={K} {reg} s{runs[n]['seed']}")
                steps = [c["step"] for c in cv["checkpoints"]]
                acc = [c["dev"][f"K{K}_{reg}"]["acc"] for c in cv["checkpoints"]]
                ax2.plot(steps, acc, marker="o", ms=3, lw=.8, label=f"K={K} {reg} s{runs[n]['seed']}")
        ax1.axhline(math.log(64), ls="--", c="gray", lw=.8, label="ln 64 (uniform)")
        ax1.set_xlabel("optimizer update"); ax1.set_ylabel("train CE (50-update mean)")
        ax2.axhline(1 / 64, ls="--", c="gray", lw=.8)
        ax2.set_xlabel("optimizer update"); ax2.set_ylabel("dev accuracy (training K, regime; 256 seqs)")
        ax1.set_title(f"{a}: training loss"); ax2.set_title(f"{a}: dev accuracy at checkpoints")
        ax2.legend(fontsize=6, ncol=2)
        _save(fig, plots / f"b_curves_{a}.png", made)
    # distance x load heatmap per config (seed-mean)
    for (a, K, reg), names in sorted(by_cfg.items()):
        nb = 256 // FINE_BIN
        acc = np.full((3, nb), np.nan)
        for ki, kk in enumerate((4, 8, 16)):
            n = np.zeros(256); c = np.zeros(256)
            for nm in names:
                kd = runs[nm]["test"]["by_K_distance"][f"K{kk}"]
                n += np.asarray(kd["n"]); c += np.asarray(kd["correct"])
            for b in range(nb):
                sl = slice(b * FINE_BIN, (b + 1) * FINE_BIN)
                if n[sl].sum() > 0:
                    acc[ki, b] = c[sl].sum() / n[sl].sum()
        fig, ax = plt.subplots(figsize=(9, 2.8))
        im = ax.imshow(acc, aspect="auto", vmin=0, vmax=1, cmap="viridis",
                       extent=[0, 256, 2.5, -0.5])
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels([f"K=4{' (trained)' if K == 4 else ''}",
                                                     f"K=8{' (trained)' if K == 8 else ''}",
                                                     f"K=16{' (trained)' if K == 16 else ''}"])
        ax.axvline(126, c="w", ls=":", lw=1)
        ax.set_xlabel("source-query distance (tokens; 16-token bins; dotted = 126)")
        ax.set_title(f"{a} trained K={K} {reg}: test accuracy (all regimes pooled, {len(names)} seeds)")
        fig.colorbar(im, ax=ax, fraction=.03)
        _save(fig, plots / f"b_heatmap_{a}_K{K}_{reg}.png", made)
    # near/far/mixed by K bars (at training K)
    fig, ax = plt.subplots(figsize=(12, 4))
    labels, xs = [], 0
    for (a, K, reg), names in sorted(by_cfg.items()):
        for j, rg in enumerate(("near", "far", "mixed")):
            v = [runs[n]["test"]["cells"][f"K{K}_{rg}"]["acc"] for n in names]
            ax.bar(xs + j * 0.27, np.mean(v), 0.25, yerr=np.std(v, ddof=1) if len(v) > 1 else 0,
                   color=["C0", "C1", "C2"][j], label=rg if xs == 0 else None)
        labels.append(f"{a}\nK{K} {reg}")
        xs += 1
    ax.set_xticks(np.arange(xs) + .27); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("test accuracy at training K (mean ± sd over seeds)")
    ax.axhline(1 / 64, ls="--", c="gray", lw=.8)
    ax.legend()
    _save(fig, plots / "b_regimes_by_config.png", made)
    if sweep:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for m in sorted({r["method"] for r in sweep}):
            pts = sorted((r["T"], r.get("mean_ms"), r.get("peak_alloc_mb")) for r in sweep
                         if r["method"] == m and r["status"] == "ok")
            if pts:
                ax1.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=m)
                if pts[0][2] is not None:
                    ax2.plot([p[0] for p in pts], [p[2] for p in pts], marker="o", label=m)
        for ax, yl in ((ax1, "forward latency (ms, 10 reps)"), (ax2, "peak allocated (MB)")):
            ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("T (tokens), B=4 H=4 d=32")
            ax.set_ylabel(yl); ax.grid(alpha=.3)
        ax1.legend(fontsize=7)
        _save(fig, plots / "b_forward_sweep.png", made)
    return made


# ── Part C ────────────────────────────────────────────────────────────────────

C_METHODS = ["fp16", "sink_recent", "question_aware", "int4_tokenwise", "int4_channelwise", "hybrid"]
PRINCIPAL = [("fp16", m) for m in C_METHODS[1:]] + [
    ("sink_recent", "question_aware"), ("int4_tokenwise", "int4_channelwise"),
    ("question_aware", "hybrid"), ("int4_channelwise", "hybrid")]


def aggregate_c(root: Path, tables: Path, plots: Path, n_resamples: int = 2000) -> dict:
    files = sorted(glob.glob(str(root / "c" / "prompts" / "*" / "summary.json")))
    P = [read_json(f) for f in files]
    if not P:
        return {"status": "missing"}
    P.sort(key=lambda p: p["id"])
    methods = [m for m in C_METHODS if all(m in p["methods"] for p in P)]
    main = []
    for m in methods:
        R = [p["methods"][m] for p in P]
        ref_ok = [p["methods"]["fp16"]["exact_match"] for p in P]
        cond = [r["exact_match"] for r, ok in zip(R, ref_ok) if ok]
        main.append({
            "method": m, "n_prompts": len(R),
            "exact_match": _mean([r["exact_match"] for r in R]),
            "exact_match_given_fp16_correct": _mean(cond), "n_fp16_correct": sum(ref_ok),
            "ref_answer_nll_mean_per_token": _mean([r["ref_answer_nll_mean"] for r in R]),
            "ref_answer_nll_sum": _mean([r["ref_answer_nll_sum"] for r in R]),
            "post_prefill_kl_from_fp16": _mean([r["post_prefill_kl"] for r in R]),
            "teacher_forced_kl_from_fp16": _mean([r["tf_kl_mean"] for r in R]),
            "evidence_retained_frac": _mean([r.get("evidence_retained_frac") for r in R]),
            "packed_bytes_mean": _mean([r["bytes"] for r in R]),
            "realized_fraction_of_fp16_mean": _mean([r["realized_fraction_of_fp16"] for r in R]),
            "realized_fraction_min": min(r["realized_fraction_of_fp16"] for r in R),
            "realized_fraction_max": max(r["realized_fraction_of_fp16"] for r in R),
            "max_bytes_over_budget": max(r["bytes"] / p["budget_bytes_int4_channelwise"] for r, p in zip(R, P)),
            "all_within_budget": all(r["within_budget"] in (None, True) for r in R),
            "n_fp16_positions_mean": _mean([r.get("n_fp16_positions") for r in R]),
            "K_rel_fro_err": _mean([(r.get("reconstruction") or {}).get("K_rel_fro") for r in R]),
            "V_rel_fro_err": _mean([(r.get("reconstruction") or {}).get("V_rel_fro") for r in R]),
        })
    _write_csv(tables / "c_main_matched_int4_budget.csv", main)
    # per-prompt machine-readable table
    per = []
    for p in P:
        for m in methods:
            r = p["methods"][m]
            per.append({"id": p["id"], "target_tokens": p["target_tokens"], "position": p["position"],
                        "distractors": p["distractors"], "context_len": p["context_len"], "method": m,
                        "exact_match": r["exact_match"], "extracted": r["extracted"], "code": p["code"],
                        "ref_answer_nll_mean": r["ref_answer_nll_mean"], "post_prefill_kl": r["post_prefill_kl"],
                        "tf_kl_mean": r["tf_kl_mean"], "evidence_retained_frac": r.get("evidence_retained_frac"),
                        "bytes": r["bytes"], "budget": p["budget_bytes_int4_channelwise"],
                        "realized_fraction_of_fp16": r["realized_fraction_of_fp16"],
                        "prefill_ms": p["prefill_ms"], "preprocess_ms": r["preprocess_ms"],
                        "decode_ms_per_token": r["decode_ms_per_token"], "peak_alloc_mb": r["peak_alloc_mb"]})
    _write_csv(tables / "c_per_prompt.csv", per)
    # breakdowns by factor
    brk = []
    for factor in ("target_tokens", "position", "distractors"):
        for lvl in sorted({p[factor] for p in P}, key=str):
            sub = [p for p in P if p[factor] == lvl]
            for m in methods:
                R = [p["methods"][m] for p in sub]
                brk.append({"factor": factor, "level": lvl, "method": m, "n": len(R),
                            "exact_match": _mean([r["exact_match"] for r in R]),
                            "ref_answer_nll_mean": _mean([r["ref_answer_nll_mean"] for r in R]),
                            "post_prefill_kl": _mean([r["post_prefill_kl"] for r in R]),
                            "evidence_retained_frac": _mean([r.get("evidence_retained_frac") for r in R])})
    _write_csv(tables / "c_breakdown_by_factor.csv", brk)
    cells = []
    for key in sorted({(p["target_tokens"], p["position"], p["distractors"]) for p in P}, key=str):
        sub = [p for p in P if (p["target_tokens"], p["position"], p["distractors"]) == key]
        d = {"target_tokens": key[0], "position": key[1], "distractors": key[2], "n": len(sub),
             "note": "descriptive (10 prompts per cell)"}
        for m in methods:
            d[m] = _mean([p["methods"][m]["exact_match"] for p in sub])
        cells.append(d)
    _write_csv(tables / "c_cells_descriptive.csv", cells)
    # paired bootstrap over prompts
    boot = []
    for a, b in PRINCIPAL:
        if a not in methods or b not in methods:
            continue
        ea = [p["methods"][a]["exact_match"] for p in P]
        eb = [p["methods"][b]["exact_match"] for p in P]
        acc = bootstrap_paired_accuracy(ea, eb, n_resamples=n_resamples)
        nll = bootstrap_paired_mean([p["methods"][a]["ref_answer_nll_mean"] for p in P],
                                    [p["methods"][b]["ref_answer_nll_mean"] for p in P], n_resamples=n_resamples)
        boot.append({"a": a, "b": b, "metric": "exact_match (b - a, pp/100)", **acc})
        boot.append({"a": a, "b": b, "metric": "ref_answer_nll_mean (b - a)", **nll})
    write_json(tables / "c_bootstrap.json", boot)
    _write_csv(tables / "c_bootstrap.csv", boot)
    # systems
    sysrows = []
    for m in methods:
        R = [p["methods"][m] for p in P]
        sysrows.append({
            "method": m, "prefill_ms_mean_shared": _mean([p["prefill_ms"] for p in P]),
            "prefill_peak_alloc_mb_mean": _mean([p["prefill_peak_alloc_mb"] for p in P]),
            "scoring_ms_mean": _mean([p["scoring_ms"] for p in P]) if m in ("question_aware", "hybrid") else 0.0,
            "build_ms_mean": _mean([r["build_ms"] for r in R]), "encode_ms_mean": _mean([r.get("encode_ms") for r in R]),
            "decode_unpack_ms_mean": _mean([r.get("decode_ms") for r in R]),
            "preprocess_ms_mean": _mean([r["preprocess_ms"] for r in R]),
            "suffix_plus_decode_ms_mean": _mean([r["suffix_plus_decode_ms"] for r in R]),
            "decode_ms_per_token_mean": _mean([r["decode_ms_per_token"] for r in R]),
            "calculated_bytes_mean": _mean([r["bytes"] for r in R]),
            "shared_decode_fp16_bytes_mean": _mean([r["shared_decode_fp16_bytes"] for r in R]),
            "total_live_bytes_mean": _mean([r["total_live_bytes"] for r in R]),
            "total_live_bytes_requantised_mean": _mean([r.get("total_live_bytes_actual_requantised") for r in R]),
            "peak_alloc_mb_mean": _mean([r["peak_alloc_mb"] for r in R]),
            "note": "bytes = serialized packed size (asserted equal to the formula); peak_alloc = allocated GPU "
                    "memory with the dequantised FP16 cache used by ordinary SDPA"})
    _write_csv(tables / "c_systems.csv", sysrows)
    bench_p = root / "c" / "packed_bench.json"
    if bench_p.exists():
        bench = read_json(bench_p)
        _write_csv(tables / "c_packed_encode_decode.csv", [
            {"layout": r["layout"], "B": r["B"], "L": r["L"], "serialized_bytes": r["serialized_bytes"],
             "formula_bytes": r["formula_bytes"], "fp16_bytes": r["fp16_bytes"],
             "pack_ms": r["pack_ms"]["mean_ms"], "pack_std_ms": r["pack_ms"]["std_ms"],
             "unpack_ms": r["unpack_dequant_ms"]["mean_ms"], "unpack_std_ms": r["unpack_dequant_ms"]["std_ms"],
             "decode_step_with_unpack_ms": r["decode_step_with_unpack_ms"]["mean_ms"],
             "decode_step_fp16_ms": r["decode_step_fp16_ms"]["mean_ms"],
             "peak_alloc_mb_unpack_step": r["peak_alloc_mb_unpack_step"],
             "peak_alloc_mb_fp16_step": r["peak_alloc_mb_fp16_step"], "n_rep": r["n_rep"]} for r in bench])
    # aligned generations: fixed rule, first case (case index 00) of each (length, position) at 8 distractors
    chosen = [p for p in P if p["distractors"] == 8 and p["id"].endswith("-00")]
    aligned = []
    for p in chosen:
        ev = p["evidence_tokens"]
        lo, hi = max(0, min(ev) - 16), min(p["context_len"], max(ev) + 17)
        entry = {"id": p["id"], "code": p["code"], "evidence_token_span": [min(ev), max(ev)],
                 "mask_window": [lo, hi], "methods": {}}
        for m in methods:
            r = p["methods"][m]
            sel = r.get("retained_positions") or r.get("fp16_positions")
            entry["methods"][m] = {
                "generation": r["generation"], "extracted": r["extracted"], "exact_match": r["exact_match"],
                "n_fp16_positions": r.get("n_fp16_positions"),
                "selection_mask_window": None if sel is None else "".join("1" if i in set(sel) else "0"
                                                                          for i in range(lo, hi)),
                "evidence_retained_frac": r.get("evidence_retained_frac")}
        aligned.append(entry)
    write_json(tables / "c_aligned_generations.json", {"selection_rule": "first case (-00) of every "
                                                       "(length, position) cell with 8 distractors", "cases": aligned})
    made = []
    plt = _plt()
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, factor in zip(axes, ("target_tokens", "position", "distractors")):
        lv = [r for r in brk if r["factor"] == factor]
        levels = list(dict.fromkeys(r["level"] for r in lv))
        for j, m in enumerate(methods):
            vals = [next(r["exact_match"] for r in lv if r["level"] == L and r["method"] == m) for L in levels]
            ax.bar(np.arange(len(levels)) + j * 0.13, vals, 0.13, label=m)
        ax.set_xticks(np.arange(len(levels)) + 0.33); ax.set_xticklabels(levels)
        ax.set_xlabel(factor)
    axes[0].set_ylabel("exact match (n per level shown in c_breakdown_by_factor.csv)")
    axes[2].legend(fontsize=7)
    _save(fig, plots / "c_accuracy_by_factor.png", made)
    c1 = root / "c" / "c1_validation.json"
    return {"status": "ok", "n_prompts": len(P), "methods": methods, "plots": made,
            "c1_validation": read_json(c1)["pass"] if c1.exists() else "missing",
            "packed_bench": "ok" if bench_p.exists() else "missing"}


def aggregate(results_root: str) -> dict:
    root = Path(results_root)
    tables, plots = root / "tables", root / "plots"
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    res = {}
    for part, fn in (("a", aggregate_a), ("b", aggregate_b), ("c", aggregate_c)):
        try:
            res[part] = fn(root, tables, plots)
        except Exception as e:  # noqa: BLE001 - one part's failure must not block the others
            import traceback
            res[part] = {"status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    write_json(tables / "aggregate_status.json", res)
    return res
