"""
Task 3 aggregation: every table / plot in PDF §3 "Required Evidence".

Writes <root>/tables/*.csv|json and <root>/plots/*.png; each section is independent and
reports 'missing' inputs instead of failing the rest.
"""

import glob
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from common.bootstrap import bootstrap_paired_mean
from common.runio import read_json, read_jsonl, write_json
from task2_attention.aggregate import _mean, _plt, _save, _std, _write_csv
from task3_specdec import evaluate as ev

TRAINED = ["off_s6304", "off_s6305", "on_s6304", "on_s6305"]
DRAFTS = ["original"] + TRAINED
METRICS = ["fwd_kl", "rev_kl", "tv", "expected_accept", "greedy_agree"]
LEN_BINS = ((1, 16), (17, 32), (33, 64))


def _workload(prompt_id: str) -> str:
    if prompt_id.startswith("GSM8K"):
        return "math"
    return "familiar" if "familiar" in prompt_id else "shifted"


def _tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    except Exception:  # noqa: BLE001 - decoded text is a convenience only
        return None


def sec_validation(root, T):
    p = root / "sampler_validation.json"
    if not p.exists():
        return "missing"
    v = read_json(p)
    _write_csv(T / "t3_chi2.csv", [{k: r[k] for k in r if k != "table"} for r in v["chi2"]])
    joint = defaultdict(dict)
    for r in v["chi2"]:
        tag = f"g{r['gamma']}_{'correct' if r['correct_sampler'] else 'negative_control'}"
        for row in r["table"]:
            joint[row["sequence"]]["sequence"] = row["sequence"]
            joint[row["sequence"]]["target_joint_prob"] = row["target_joint_prob"]
            joint[row["sequence"]][f"empirical_{tag}"] = row["empirical_prob"]
    _write_csv(T / "t3_chi2_joint_probabilities.csv", sorted(joint.values(), key=lambda r: r["sequence"]))
    _write_csv(T / "t3_real_model_checks.csv", v["real_model_checks"])
    return {"chi2_pass": v["chi2_pass"]}


def sec_distill(root, T, P):
    rows, curves = [], {}
    merge = {}
    for n in TRAINED:
        mp = root / "merge_logs" / n / "merge_summary.json"
        if mp.exists():
            merge[n] = read_json(mp)
    for n in TRAINED:
        sp = root / "distill" / n / "summary.json"
        if not sp.exists():
            continue
        s = read_json(sp)
        c = s["costs_s"]
        mg = merge.get(n, {}).get("seconds", {})
        merge_s = sum(mg.values()) if mg else None
        total = (c.get("teacher_generation_s_pool") or 0) + c["teacher_scoring_s"] + c["student_rollout_s"] + \
            c["student_fwd_bwd_s"] + c["optimizer_s"] + (merge_s or 0)
        rows.append({"run": n, "method": s["method"], "seed": s["seed"], "updates": s["updates"],
                     "skipped_updates": s["skipped_updates"], "presentations": s["n_presentations"],
                     "scored_math": s["scored_tokens"]["math"], "scored_short": s["scored_tokens"]["short"],
                     "truncated_by_quota": s["truncated_by_quota"],
                     "mean_len_math": s["mean_response_len"]["math"], "mean_len_short": s["mean_response_len"]["short"],
                     "eos_rate_math": s["eos_rate"]["math"], "eos_rate_short": s["eos_rate"]["short"],
                     "teacher_generation_s": c.get("teacher_generation_s_pool"), "teacher_scoring_s": c["teacher_scoring_s"],
                     "student_rollout_s": c["student_rollout_s"], "student_fwd_bwd_s": c["student_fwd_bwd_s"],
                     "optimizer_s": c["optimizer_s"], "wall_train_s": c["wall_train_s"], "merge_s": merge_s,
                     "total_cost_s": total, "peak_alloc_mb": s["peak_alloc_mb"],
                     "backbone_params": s["params"]["backbone_params"],
                     "adapter_params": s["params"]["trainable_adapter_params"],
                     "merge_max_abs_logit_diff": merge.get(n, {}).get("verify", {}).get("max_abs_logit_diff"),
                     "merge_argmax_agreement": merge.get(n, {}).get("verify", {}).get("argmax_agreement")})
        curves[n] = read_json(root / "distill" / n / "curves.json")
    if not rows:
        return "missing"
    _write_csv(T / "t3_distill_runs_costs.csv", rows)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, cv in curves.items():
        ax.plot([c["teacher_scored_tokens"] for c in cv], [c["loss_mean_token_kl"] for c in cv],
                lw=.9, ls="-" if n.startswith("off") else "--", label=n)
    ax.set_xlabel("teacher-scored response tokens"); ax.set_ylabel("mean token KL(p_T || q) per update")
    ax.set_title("distillation loss (1 point per 256-token update)"); ax.legend(); ax.grid(alpha=.3)
    made = []
    _save(fig, P / "t3_distill_loss.png", made)
    return {"runs": len(rows), "plots": made}


def sec_alignment(root, T, P):
    p = root / "banks" / "alignment" / "rows.jsonl.gz"
    if not p.exists():
        return "missing"
    rows = read_jsonl(p)
    for r in rows:
        r["workload"] = _workload(r["prompt_id"])
        r["span"] = "short [1,64]" if r["position"] <= 64 else "long [65,192]"
    out = []
    keys = [("draft", "bank"), ("draft", "bank", "workload"), ("draft", "bank", "position_bin"),
            ("draft", "bank", "workload", "span")]
    for ks in keys:
        g = defaultdict(list)
        for r in rows:
            g[tuple(r[k] for k in ks)].append(r)
        for k, rs in sorted(g.items()):
            d = {kk: v for kk, v in zip(ks, k)}
            d["grouping"] = "+".join(ks)
            d["n_tokens"] = len(rs)
            for m in METRICS:
                d[m] = _mean([r[m] for r in rs])
            out.append(d)
    _write_csv(T / "t3_alignment.csv", out)
    # seed dispersion for each method on each bank
    disp = []
    by = {(r["draft"], r["bank"]): r for r in out if r["grouping"] == "draft+bank"}
    for bank in sorted({r["bank"] for r in rows}):
        for meth in ("off", "on"):
            vals = [by.get((f"{meth}_s{s}", bank)) for s in (6304, 6305)]
            vals = [v for v in vals if v]
            d = {"bank": bank, "method": meth, "n_seeds": len(vals)}
            for m in METRICS:
                d[f"{m}_mean"] = _mean([v[m] for v in vals])
                d[f"{m}_sd"] = _std([v[m] for v in vals])
            disp.append(d)
    _write_csv(T / "t3_alignment_seed_dispersion.csv", disp)
    # target vs exact-speculative consistency: per-prompt means, paired over prompts
    cons = []
    for d in DRAFTS:
        per = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["draft"] == d and r["bank"] in ("teacher", "exact_spec_original_gamma4"):
                per[r["prompt_id"]][r["bank"]].append(r)
        pids = [k for k, v in per.items() if len(v) == 2]
        for m in ("tv", "fwd_kl", "expected_accept"):
            a = [_mean([x[m] for x in per[k]["teacher"]]) for k in pids]
            b = [_mean([x[m] for x in per[k]["exact_spec_original_gamma4"]]) for k in pids]
            if pids:
                cons.append({"draft": d, "metric": m, "teacher_bank": _mean(a), "exact_spec_bank": _mean(b),
                             **{f"diff_{k}": v for k, v in bootstrap_paired_mean(a, b).items()}})
    bank_meta = []
    banks = read_json(root / "banks" / "alignment" / "summary.json").get("bank_source_lengths", {})
    for b, d in banks.items():
        bank_meta.append({"bank": b, "n_prompts": len(d), "mean_source_length": _mean(list(d.values())),
                          "frac_full_192": _mean([int(v >= 192) for v in d.values()])})
    write_json(T / "t3_target_vs_spec_consistency.json", {"paired_over_prompts": cons, "bank_source_stats": bank_meta,
                                                          "note": "teacher and exact_spec banks sample the same "
                                                                  "target-prefix law; differences should be within "
                                                                  "sampling uncertainty (CIs cover 0)"})
    # aligned disagreements: largest |TV(off) - TV(on)| on shared teacher-bank prefixes, seed 6304
    idx = defaultdict(dict)
    for r in rows:
        if r["bank"] == "teacher" and r["draft"] in ("off_s6304", "on_s6304", "original"):
            idx[(r["prompt_id"], r["position"])][r["draft"]] = r
    dis = sorted((v for v in idx.values() if len(v) == 3),
                 key=lambda v: -abs(v["off_s6304"]["tv"] - v["on_s6304"]["tv"]))[:12]
    tok = _tok()
    dec = (lambda i: tok.decode([i])) if tok else (lambda i: i)
    write_json(T / "t3_prefix_disagreements.json", [
        {"prompt_id": v["original"]["prompt_id"], "position": v["original"]["position"],
         "target_top": dec(v["original"]["p_top"]),
         **{f"{d}_top": dec(v[d]["q_top"]) for d in ("original", "off_s6304", "on_s6304")},
         **{f"{d}_tv": v[d]["tv"] for d in ("original", "off_s6304", "on_s6304")}} for v in dis])
    # plot: expected acceptance by position bin per draft (teacher bank)
    plt = _plt()
    made = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, wl in zip(axes, ("math", "familiar", "shifted")):
        for d in DRAFTS:
            pts = []
            for lo, hi in ((1, 16), (17, 32), (33, 64), (65, 192)):
                v = [r["expected_accept"] for r in rows if r["draft"] == d and r["bank"] == "teacher"
                     and r["workload"] == wl and lo <= r["position"] <= hi]
                if v:
                    pts.append((f"[{lo},{hi}]", sum(v) / len(v)))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=d)
        ax.set_title(f"{wl} (teacher bank)"); ax.set_xlabel("response position bin")
    axes[0].set_ylabel("expected one-token acceptance 1 - TV"); axes[2].legend(fontsize=7)
    _save(fig, P / "t3_expected_acceptance_by_position.png", made)
    return {"n_rows": len(rows), "plots": made}


def _pool(rows, num, den):
    a, b = sum(r[num] or 0 for r in rows), sum(r[den] or 0 for r in rows)
    return a / b if b else None


def _spec_summary(rows: List[dict]) -> dict:
    nr = sum(r["spec_n_rounds"] for r in rows)
    return {
        "n_generations": len(rows),
        "accepted_over_examined": _pool(rows, "spec_n_accepted", "spec_n_examined"),
        "accepted_over_proposed": _pool(rows, "spec_n_accepted", "spec_n_proposed"),
        "mean_accepted_per_verification": _pool(rows, "spec_n_accepted", "spec_n_rounds"),
        "target_passes_per_output_token": _pool(rows, "spec_n_rounds", "spec_n_output_tokens"),
        "target_only_passes_per_output_token": _pool(rows, "target_n_rounds", "target_n_output_tokens"),
        "draft_forwards_per_output_token": _pool(rows, "spec_n_draft_forwards", "spec_n_output_tokens"),
        "spec_tokens_per_s_mean": _mean([r["spec_tokens_per_s"] for r in rows]),
        "target_tokens_per_s_mean": _mean([r["target_tokens_per_s"] for r in rows]),
        "spec_latency_ms_mean": _mean([r["spec_t_total_ms"] for r in rows]),
        "target_latency_ms_mean": _mean([r["target_t_total_ms"] for r in rows]),
        "measured_speedup_pooled": (sum(r["target_t_total_ms"] for r in rows) / sum(r["spec_t_total_ms"] for r in rows)),
        "per_prompt_time_ratio_mean": _mean([r["time_ratio_target_over_spec"] for r in rows]),
        "spec_output_tokens_mean": _mean([r["spec_n_output_tokens"] for r in rows]),
        "target_output_tokens_mean": _mean([r["target_n_output_tokens"] for r in rows]),
        "spec_eos_rate": _mean([int(r["spec_eos"]) for r in rows]),
        "spec_truncated_rate": _mean([int(r["spec_truncated"]) for r in rows]),
        "target_eos_rate": _mean([int(r["target_eos"]) for r in rows]),
        "spec_peak_alloc_mb_mean": _mean([r["spec_peak_alloc_mb"] for r in rows]),
        "target_peak_alloc_mb_mean": _mean([r["target_peak_alloc_mb"] for r in rows]),
        "round_draft_ms": sum(r["spec_t_draft_ms"] for r in rows) / nr if nr else None,
        "round_verify_ms": sum(r["spec_t_verify_ms"] for r in rows) / nr if nr else None,
        "round_accept_residual_ms": sum(r["spec_t_accept_ms"] for r in rows) / nr if nr else None,
        "round_rollback_ms": sum(r["spec_t_rollback_ms"] for r in rows) / nr if nr else None,
        "prefill_ms_mean": _mean([r["spec_t_prefill_target_ms"] + r["spec_t_prefill_draft_ms"] for r in rows]),
        "host_overhead_ms_mean": _mean([r["spec_t_host_overhead_ms"] for r in rows]),
    }


def _per_prompt(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r["prompt_id"]].append(r[key])
    return {k: sum(v) / len(v) for k, v in g.items()}


def sec_spec(root, T, P):
    conds = {}
    for f in glob.glob(str(root / "spec" / "*" / "primary.jsonl.gz")):
        name = Path(f).parent.name
        d, g = name.rsplit("__g", 1)
        conds[(d, int(g))] = {"primary": read_jsonl(f), "fixed64": read_jsonl(Path(f).parent / "fixed64.jsonl.gz")}
    if not conds:
        return "missing"
    costs = read_json(root / "costs" / "summary.json")["rows"] if (root / "costs" / "summary.json").exists() else []
    cl = {(r["model"], r["cache_len"]): r for r in costs}
    bank_rows = {}
    ap = T / "t3_alignment.csv"
    import csv
    if ap.exists():
        for r in csv.DictReader(open(ap)):
            if r["grouping"] == "draft+bank+workload" and r["bank"] == "teacher":
                bank_rows[(r["draft"], r["workload"])] = float(r["expected_accept"])
            if r["grouping"] == "draft+bank" and r["bank"] == "teacher":
                bank_rows[(r["draft"], "all")] = float(r["expected_accept"])
    main, work, lens, fixed = [], [], [], []
    for (d, g), c in sorted(conds.items()):
        pr = c["primary"]
        s = {"draft": d, "gamma": g, **_spec_summary(pr)}
        alpha = s["accepted_over_examined"]
        for L in ev.COST_LENGTHS:
            t, dr = cl.get(("target", L)), cl.get(("original", L))
            if t and dr:
                cm = ev.cost_model(dr["decode_step_ms"] / t["decode_step_ms"],
                                   t[f"verify_{g + 1}_tokens_ms"] / t["decode_step_ms"], g, alpha)
                s[f"c_L{L}"], s[f"v_gamma_L{L}"] = cm["c"], cm["v_gamma"]
                s[f"predicted_speedup_L{L}"] = cm.get("predicted_speedup")
                s[f"break_even_alpha_L{L}"] = cm["break_even_alpha"]
        s["expected_accept_teacher_bank"] = bank_rows.get((d, "all"))
        # bootstrap over prompts of the mean per-prompt speedup (reps averaged within a prompt)
        rp = list(_per_prompt(pr, "time_ratio_target_over_spec").values())
        bs = bootstrap_paired_mean([0.0] * len(rp), rp)
        s["per_prompt_speedup_mean"] = bs["observed"]
        s["per_prompt_speedup_ci_lo"] = bs["ci_lo"]
        s["per_prompt_speedup_ci_hi"] = bs["ci_hi"]
        main.append(s)
        for wl in ("math", "familiar", "shifted"):
            sub = [r for r in pr if r["workload"] == wl]
            if sub:
                w = {"draft": d, "gamma": g, "workload": wl, **_spec_summary(sub),
                     "expected_accept_teacher_bank": bank_rows.get((d, wl))}
                work.append(w)
        for lo, hi in LEN_BINS:
            sub = [r for r in pr if lo <= r["spec_n_output_tokens"] <= hi]
            if sub:
                lens.append({"draft": d, "gamma": g, "output_len_bin": f"[{lo},{hi}]", **_spec_summary(sub)})
        if c["fixed64"]:
            fixed.append({"draft": d, "gamma": g, "diagnostic": "24 math prompts, 64 tokens, EOS ignored",
                          **_spec_summary(c["fixed64"])})
    _write_csv(T / "t3_spec_main.csv", main)
    _write_csv(T / "t3_spec_by_workload.csv", work)
    _write_csv(T / "t3_spec_by_output_length.csv", lens)
    _write_csv(T / "t3_spec_fixed64_throughput.csv", fixed)
    _write_csv(T / "t3_isolated_costs.csv", costs)
    # paired bootstrap: each trained draft vs original at the same gamma (prompts paired)
    boot = []
    for g in ev.GAMMAS:
        if ("original", g) not in conds:
            continue
        base = conds[("original", g)]["primary"]
        for d in TRAINED:
            if (d, g) not in conds:
                continue
            other = conds[(d, g)]["primary"]
            for key, label in (("time_ratio_target_over_spec", "speedup (target/spec time)"),
                               ("spec_accepted_over_examined", "accepted/examined")):
                a, b = _per_prompt(base, key), _per_prompt(other, key)
                ks = sorted(set(a) & set(b))
                boot.append({"gamma": g, "a": "original", "b": d, "metric": label,
                             **bootstrap_paired_mean([a[k] for k in ks], [b[k] for k in ks])})
    write_json(T / "t3_spec_bootstrap.json", boot)
    _write_csv(T / "t3_spec_bootstrap.csv", boot)
    # unmerged wrapper overhead vs the same merged adapter on the same prompts/rep
    un = [k for k in conds if k[0].startswith("unmerged")]
    wrap = []
    for d, g in un:
        name = d.split(":", 1)[-1] if ":" in d else d.replace("unmerged_", "")
        if (name, g) in conds:
            u = conds[(d, g)]["primary"]
            ids = {(r["prompt_id"], r["rep"]) for r in u}
            m = [r for r in conds[(name, g)]["primary"] if (r["prompt_id"], r["rep"]) in ids]
            wrap.append({"adapter": name, "gamma": g, "n": len(u),
                         "unmerged_round_draft_ms": _spec_summary(u)["round_draft_ms"],
                         "merged_round_draft_ms": _spec_summary(m)["round_draft_ms"],
                         "unmerged_spec_latency_ms": _spec_summary(u)["spec_latency_ms_mean"],
                         "merged_spec_latency_ms": _spec_summary(m)["spec_latency_ms_mean"]})
    _write_csv(T / "t3_unmerged_vs_merged.csv", wrap)
    # deployment break-even including distillation + merge cost
    dist = {}
    if (T / "t3_distill_runs_costs.csv").exists():
        for r in csv.DictReader(open(T / "t3_distill_runs_costs.csv")):
            dist[r["run"]] = float(r["total_cost_s"])
    dep = []
    for (d, g), c in sorted(conds.items()):
        if d not in TRAINED or ("original", g) not in conds:
            continue
        save = _mean([r["target_t_total_ms"] - r["spec_t_total_ms"] for r in c["primary"]])
        save0 = _mean([r["target_t_total_ms"] - r["spec_t_total_ms"] for r in conds[("original", g)]["primary"]])
        cost = dist.get(d)
        inc = save - save0
        dep.append({"draft": d, "gamma": g, "saving_vs_target_ms_per_request": save,
                    "saving_vs_original_draft_ms_per_request": inc, "upfront_cost_s": cost,
                    "requests_to_break_even_vs_original_draft": (cost * 1000 / inc) if cost and inc > 0 else None,
                    "note": "None = no measured per-request gain over the original draft"})
    _write_csv(T / "t3_deployment_break_even.csv", dep)
    # representative matched-prompt continuations (fixed rule: first prompt of each workload, rep 0, gamma 4)
    tok = _tok()
    rep = []
    for wl in ("math", "familiar", "shifted"):
        for d in DRAFTS:
            c = conds.get((d, 4))
            if not c:
                continue
            r = next((x for x in c["primary"] if x["workload"] == wl and x["rep"] == 0), None)
            if r:
                rep.append({"workload": wl, "prompt_id": r["prompt_id"], "draft": d,
                            "spec_text": tok.decode(r["spec_tokens"]) if tok else r["spec_tokens"],
                            "accepted_per_round": r["spec_accepted_per_round"], "eos": r["spec_eos"]})
    write_json(T / "t3_representative_continuations.json", rep)
    plt = _plt()
    made = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for g, mk in ((2, "o"), (4, "s")):
        ss = [s for s in main if s["gamma"] == g and not s["draft"].startswith("unmerged")]
        ax1.bar([f"{s['draft']}\ng{g}" for s in ss], [s["accepted_over_examined"] for s in ss])
        ax2.errorbar([f"{s['draft']}\ng{g}" for s in ss], [s["per_prompt_speedup_mean"] for s in ss],
                     yerr=[[s["per_prompt_speedup_mean"] - s["per_prompt_speedup_ci_lo"] for s in ss],
                           [s["per_prompt_speedup_ci_hi"] - s["per_prompt_speedup_mean"] for s in ss]],
                     fmt=mk, capsize=3)
    ax1.set_ylabel("accepted / examined proposals"); ax2.axhline(1, c="gray", ls="--")
    ax2.set_ylabel("per-prompt target/spec time (95% bootstrap CI)")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelsize=6)
    _save(fig, P / "t3_acceptance_and_speedup.png", made)
    return {"conditions": len(conds), "plots": made}


def sec_standalone(root, T):
    rows = []
    for d in DRAFTS:
        f = root / "standalone" / d / "rows.jsonl.gz"
        if f.exists():
            rows += read_jsonl(f)
    if not rows:
        return "missing"
    out = []
    for d in DRAFTS:
        for wl in ("math", "familiar", "shifted"):
            rs = [r for r in rows if r["draft"] == d and r["workload"] == wl]
            if rs:
                out.append({"draft": d, "workload": wl, "n": len(rs), "accuracy": _mean([r["correct"] for r in rs]),
                            "strict_exact": _mean([r.get("correct_strict") for r in rs]),
                            "format_compliance": _mean([r.get("format_compliant") for r in rs]),
                            "unextractable_rate": _mean([r.get("unextractable") for r in rs]),
                            "truncated_rate": _mean([r["truncated"] for r in rs]),
                            "mean_tokens": _mean([r["n_tokens"] for r in rs])})
    _write_csv(T / "t3_standalone_accuracy.csv", out)
    return {"rows": len(out)}


def aggregate(results_root: str) -> dict:
    root = Path(results_root)
    T, P = root / "tables", root / "plots"
    T.mkdir(parents=True, exist_ok=True)
    P.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, fn in (("validation", lambda: sec_validation(root, T)), ("distill", lambda: sec_distill(root, T, P)),
                     ("alignment", lambda: sec_alignment(root, T, P)), ("spec", lambda: sec_spec(root, T, P)),
                     ("standalone", lambda: sec_standalone(root, T))):
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            res[name] = {"status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    write_json(T / "aggregate_status.json", res)
    return res
