"""
Task 3 driver (PDF §3). Resumable stages; results go to Drive, never the repo.
Every stage skips work whose marker file exists; a failed unit writes error.json and
the driver continues with the next unit.

    python -m task3_specdec.run_task3 check          # tokenizer/template agreement, splits, bank prompts
    python -m task3_specdec.run_task3 validate       # chi^2 (+negative control), split prefill, cache checks
    python -m task3_specdec.run_task3 bank_spec      # exact_spec_original_gamma4 bank with our sampler
    python -m task3_specdec.run_task3 pool           # frozen teacher rollouts (off-policy)
    python -m task3_specdec.run_task3 distill --methods off,on --seeds 6304,6305
    python -m task3_specdec.run_task3 merge          # LoRA -> fresh FP16 backbone, verify, time
    python -m task3_specdec.run_task3 bank_students  # each trained student's own rollout bank
    python -m task3_specdec.run_task3 banks_eval     # KL / TV / acceptance on every bank x draft
    python -m task3_specdec.run_task3 costs          # isolated draft/target step costs (c, v_gamma)
    python -m task3_specdec.run_task3 spec --drafts original,off_s6304,...   # steps 2 and 6
    python -m task3_specdec.run_task3 unmerged       # one unmerged adapter: wrapper overhead
    python -m task3_specdec.run_task3 standalone     # each draft as a model (step 7)
    python -m task3_specdec.run_task3 aggregate
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.runio import free_cuda, read_json, read_jsonl, run_resumable, write_json, write_jsonl  # noqa: E402
from task3_specdec import distill as ds  # noqa: E402
from task3_specdec import evaluate as ev  # noqa: E402
from task3_specdec import generation as gen  # noqa: E402
from task3_specdec import prefix_eval as pe  # noqa: E402
from task3_specdec import speculative as sp  # noqa: E402

DEFAULT_DATA_REPO = "/content/drive/MyDrive/atml-pa4-datasets"
DEFAULT_RESULTS = "/content/drive/MyDrive/atml_pa4_results/task3"
TRAINED = [f"{m}_s{s}" for m in ("off", "on") for s in ds.SEEDS]
DRAFTS = ["original"] + TRAINED


class Ctx:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.results_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.revs = read_json(_REPO_ROOT / "data" / "local_source_revisions.json")["revisions"]
        self._tok = None

    @property
    def tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(gen.TARGET_ID, revision=self.revs["model_target"])
        return self._tok

    def load(self, which: str):
        from transformers import AutoModelForCausalLM
        kw = dict(torch_dtype=torch.float16, attn_implementation="sdpa")
        if which == "target":
            m = AutoModelForCausalLM.from_pretrained(gen.TARGET_ID, revision=self.revs["model_target"], **kw)
        elif which == "original":
            m = AutoModelForCausalLM.from_pretrained(gen.DRAFT_ID, revision=self.revs["model_draft"], **kw)
        elif which.startswith("unmerged:"):
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(gen.DRAFT_ID, revision=self.revs["model_draft"], **kw)
            m = PeftModel.from_pretrained(base, str(self.root / "distill" / which.split(":", 1)[1] / "adapter"))
        else:
            m = AutoModelForCausalLM.from_pretrained(str(self.root / "merged" / which), **kw)
        return m.to(self.dev).eval()

    def test_prompts(self):
        return gen.load_test_prompts(self.args.data_repo)

    def train_prompts(self):
        return gen.load_train_prompts(self.args.data_repo)


def _timed_load(ctx, which):
    t0 = time.perf_counter()
    m = ctx.load(which)
    if ctx.dev.startswith("cuda"):
        torch.cuda.synchronize()
    return m, time.perf_counter() - t0


# ── stages ────────────────────────────────────────────────────────────────────

def st_check(ctx):
    from transformers import AutoTokenizer
    out = ctx.root / "check.json"
    if out.exists() and not ctx.args.overwrite:
        return read_json(out)
    dtok = AutoTokenizer.from_pretrained(gen.DRAFT_ID, revision=ctx.revs["model_draft"])
    test, train = ctx.test_prompts(), ctx.train_prompts()
    sample = [p["text"] for p in test] + [p["text"] for p in train["math"][:25] + train["short"][:25]]
    mism = [i for i, t in enumerate(sample) if gen.chat_ids(ctx.tok, t) != gen.chat_ids(dtok, t)]
    eos_tok = {i: ctx.tok.convert_ids_to_tokens(i) for i in sorted(sp.QWEN_EOS_IDS)}
    res = {"n_prompts_checked": len(sample), "chat_template_token_mismatches": mism,
           "vocab_equal": ctx.tok.get_vocab() == dtok.get_vocab(), "eos_tokens": eos_tok,
           "example_prompt": ctx.tok.apply_chat_template([{"role": "user", "content": test[0]["text"]}],
                                                         tokenize=False, add_generation_prompt=True),
           "splits": gen.check_disjoint(train, test)}
    try:
        res["supplied_banks"] = pe.check_bank_prompts(pe.load_supplied_banks(ctx.args.data_repo), test)
    except FileNotFoundError as e:
        res["supplied_banks"] = f"missing: {e}"
    res["pass"] = (not mism and res["vocab_equal"] and res["splits"]["train_test_text_overlap"] == 0
                   and isinstance(res["supplied_banks"], dict) and not res["supplied_banks"]["mismatched_prompt_ids"])
    write_json(out, res)
    return res


def st_validate(ctx):
    out = ctx.root / "sampler_validation.json"
    if out.exists() and not ctx.args.overwrite:
        return read_json(out)
    res = {"chi2": [sp.chi2_test(g, correct=c) for g in (2, 4) for c in (True, False)]}
    res["chi2_pass"] = all((r["rejected_at_0.01"] != r["correct_sampler"]) and r["min_expected"] >= 10
                           for r in res["chi2"])
    target, draft = ctx.load("target"), ctx.load("original")
    test = ctx.test_prompts()
    checks = []
    for p in (test[0], test[60], test[90]):
        ids = gen.chat_ids(ctx.tok, p["text"])
        for name, m in (("target", target), ("draft", draft)):
            one = sp.prefill(m, ids, ctx.dev)
            chunked = sp.prefill(m, ids, ctx.dev, chunk=7)
            kdiff = max(float((one.layers[i].keys - chunked.layers[i].keys).abs().max()) for i in range(len(one.layers)))
            l1 = sp.forward_tokens(m, one, ids[-1:], ctx.dev)
            l2 = sp.forward_tokens(m, chunked, ids[-1:], ctx.dev)
            full = m(input_ids=torch.tensor([ids], device=ctx.dev)).logits[0, -1].float()
            checks.append({"prompt_id": p["id"], "model": name, "cache_len_one": one.get_seq_length(),
                           "cache_len_chunked": chunked.get_seq_length(), "max_key_diff": kdiff,
                           "max_logit_diff_chunked_vs_one": float((l1 - l2).abs().max()),
                           "max_logit_diff_cached_vs_full": float((l1[-1] - full).abs().max()),
                           "argmax_equal": int(l1[-1].argmax()) == int(l2[-1].argmax()) == int(full.argmax())})
        torch.manual_seed(6304)
        st = sp.speculative_generate(target, draft, ids, 48, 4, sp.QWEN_EOS_IDS, ctx.dev, debug_check=True)
        checks.append({"prompt_id": p["id"], "model": "speculative_debug", "rounds": st.n_rounds,
                       "max_verify_logit_err_vs_full_forward": st.debug_max_logit_err,
                       "n_tokens": len(st.tokens)})
    res["real_model_checks"] = checks
    res["real_model_note"] = "FP16 SDPA: chunked-prefill differences at the 1e-2 logit level are numerical; argmax must agree"
    write_json(out, res)
    del target, draft
    free_cuda()
    return {"chi2_pass": res["chi2_pass"], "n_checks": len(checks)}


def st_bank_spec(ctx):
    out = ctx.root / "banks" / f"{pe.SPEC_BANK}.jsonl.gz"
    meta_p = ctx.root / "banks" / f"{pe.SPEC_BANK}.meta.json"
    if meta_p.exists() and not ctx.args.overwrite:
        return read_json(meta_p)
    target, draft = ctx.load("target"), ctx.load("original")
    meta = pe.generate_spec_bank(target, draft, ctx.tok, ctx.test_prompts(), str(out), ctx.dev)
    write_json(meta_p, meta)
    del target, draft
    free_cuda()
    return {"n_records": meta["n_records"], "seconds": meta["seconds"]}


def st_pool(ctx):
    meta_p = ctx.root / "distill" / "teacher_pool.meta.json"
    if meta_p.exists() and not ctx.args.overwrite:
        return read_json(meta_p)
    teacher = ctx.load("target")
    meta = ds.build_teacher_pool(teacher, ctx.tok, ctx.train_prompts(), str(ctx.root / "distill" / "teacher_pool.jsonl.gz"),
                                 ctx.dev)
    write_json(meta_p, meta)
    del teacher
    free_cuda()
    return meta


def st_distill(ctx):
    methods = ctx.args.methods.split(",")
    seeds = [int(s) for s in ctx.args.seeds.split(",")]
    runs = [f"{m}_s{s}" for m in methods for s in seeds]
    train = ctx.train_prompts()
    holder = {}

    def one(name, out):
        m, s = name.split("_s")
        if "teacher" not in holder:
            holder["teacher"] = ctx.load("target")
        pool = pool_meta = None
        if m == "off":
            pool = ds.load_pool(str(ctx.root / "distill" / "teacher_pool.jsonl.gz"))
            pool_meta = read_json(ctx.root / "distill" / "teacher_pool.meta.json")
        student_base = ctx.load("original")
        ds.distill(m, holder["teacher"], student_base, ctx.tok, train, int(s), str(out), ctx.dev, pool, pool_meta)
        del student_base
        free_cuda()

    st = run_resumable(runs, lambda n: ctx.root / "distill" / n, one, ctx.args.overwrite)
    holder.clear()
    free_cuda()
    return st


def st_merge(ctx):
    from task3_specdec.merge import merge_adapter
    train = ctx.train_prompts()
    # fixed verification batch: 4 training prompts of each category (no evaluation prompts)
    batch = [gen.chat_ids(ctx.tok, p["text"]) for p in train["math"][:4] + train["short"][:4]]

    def one(name, out):
        res = merge_adapter(str(ctx.root / "distill" / name / "adapter"), str(ctx.root / "merged" / name),
                            gen.DRAFT_ID, ctx.revs["model_draft"], batch, ctx.dev)
        write_json(out / "merge_summary.json", res)

    return run_resumable(TRAINED, lambda n: ctx.root / "merge_logs" / n, one, ctx.args.overwrite,
                         marker="merge_summary.json")


def st_bank_students(ctx):
    test = ctx.test_prompts()

    def one(name, out):
        m = ctx.load(name)
        meta = pe.generate_student_bank(m, ctx.tok, test, f"student_{name}", f"merged/{name}",
                                        str(ctx.root / "banks" / f"student_{name}.jsonl.gz"), ctx.dev)
        write_json(out / "summary.json", meta)
        del m

    return run_resumable(TRAINED, lambda n: ctx.root / "banks" / "student_logs" / n, one, ctx.args.overwrite)


def st_banks_eval(ctx):
    out = ctx.root / "banks" / "alignment"

    def one(_, o):
        recs = pe.load_supplied_banks(ctx.args.data_repo)
        recs += read_jsonl(ctx.root / "banks" / f"{pe.SPEC_BANK}.jsonl.gz")
        for n in TRAINED:
            recs += read_jsonl(ctx.root / "banks" / f"student_{n}.jsonl.gz")
        target = ctx.load("target")
        drafts = {n: ctx.load(n) for n in DRAFTS}
        rows = pe.evaluate_banks(target, drafts, ctx.tok, recs, ctx.dev)
        write_jsonl(o / "rows.jsonl.gz", rows)
        src = {}
        for r in recs:
            src.setdefault(r["bank"], {})[r["prompt_id"]] = r["source_length"]
        write_json(o / "summary.json", {"n_records": len(recs), "n_rows": len(rows), "drafts": DRAFTS,
                                        "banks": sorted(src), "bank_source_lengths": src})
        del target, drafts

    return run_resumable(["alignment"], lambda n: out, one, ctx.args.overwrite)


def st_costs(ctx):
    out = ctx.root / "costs"

    def one(_, o):
        target = ctx.load("target")
        models = {"target": target, "original": ctx.load("original")}
        if (ctx.root / "merged" / TRAINED[0]).exists():
            models[f"merged_{TRAINED[0]}"] = ctx.load(TRAINED[0])
        if (ctx.root / "distill" / TRAINED[0] / "adapter").exists():
            models[f"unmerged_{TRAINED[0]}"] = ctx.load(f"unmerged:{TRAINED[0]}")
        rows = ev.isolated_costs(models, ctx.dev)
        write_json(o / "summary.json", {"rows": rows, "lengths": list(ev.COST_LENGTHS),
                                        "note": "median of 10 synchronized reps after 3 warm-ups"})
        del models, target

    return run_resumable(["costs"], lambda n: out, one, ctx.args.overwrite)


def _spec_units(ctx, drafts, gammas):
    return [f"{d}__g{g}" for d in drafts for g in gammas]


def st_spec(ctx, drafts=None, unmerged=False):
    drafts = drafts or ctx.args.drafts.split(",")
    gammas = [int(g) for g in ctx.args.gammas.split(",")]
    test = ctx.test_prompts()
    holder = {}

    def one(unit, out):
        d, g = unit.split("__g")
        if "target" not in holder:
            holder["target"], holder["target_load_s"] = _timed_load(ctx, "target")
        if holder.get("draft_name") != d:
            holder.pop("draft", None)
            free_cuda()
            holder["draft"], holder["draft_load_s"] = _timed_load(ctx, d)
            holder["draft_name"] = d
        prompts = test
        n_reps, n_fixed = ev.N_REPS, ev.N_FIXED64
        if d.startswith("unmerged:"):
            prompts, n_reps, n_fixed = [p for p in test if p["domain"] == "math"][:24], 1, 0
        t0 = time.perf_counter()
        res = ev.run_spec_condition(holder["target"], holder["draft"], ctx.tok, prompts, int(g), d, ctx.dev,
                                    n_reps=n_reps, n_fixed64=n_fixed)
        write_jsonl(out / "primary.jsonl.gz", res["primary"])
        write_jsonl(out / "fixed64.jsonl.gz", res["fixed64"])
        write_json(out / "summary.json", {"draft": d, "gamma": int(g), "n_primary": len(res["primary"]),
                                          "n_fixed64": len(res["fixed64"]), "seconds": time.perf_counter() - t0,
                                          "warm_setup_s": {"target_load": holder["target_load_s"],
                                                           "draft_load": holder["draft_load_s"]},
                                          "cap": ev.SPEED_CAP, "n_reps": n_reps,
                                          "counterbalance": "(prompt index + rep + gamma) even -> speculative first"})

    units = _spec_units(ctx, drafts, gammas)
    st = run_resumable(units, lambda n: ctx.root / "spec" / n.replace(":", "_"), one, ctx.args.overwrite)
    holder.clear()
    free_cuda()
    return st


def st_unmerged(ctx):
    ctx.args.gammas = "4"
    return st_spec(ctx, drafts=[f"unmerged:{TRAINED[0]}"])


def st_standalone(ctx):
    test = ctx.test_prompts()

    def one(name, out):
        m = ctx.load(name)
        rows = ev.standalone_accuracy(m, ctx.tok, test, name, ctx.dev)
        write_jsonl(out / "rows.jsonl.gz", rows)
        write_json(out / "summary.json", {"draft": name, "n_rows": len(rows), "caps": ev.QUALITY_CAP})
        del m

    return run_resumable(DRAFTS, lambda n: ctx.root / "standalone" / n, one, ctx.args.overwrite)


def st_aggregate(ctx):
    from task3_specdec.aggregate import aggregate
    return aggregate(str(ctx.root))


STAGES = {"check": st_check, "validate": st_validate, "bank_spec": st_bank_spec, "pool": st_pool,
          "distill": st_distill, "merge": st_merge, "bank_students": st_bank_students,
          "banks_eval": st_banks_eval, "costs": st_costs, "spec": st_spec, "unmerged": st_unmerged,
          "standalone": st_standalone, "aggregate": st_aggregate}
ALL_ORDER = ["check", "validate", "bank_spec", "pool", "distill", "merge", "bank_students", "banks_eval",
             "costs", "spec", "unmerged", "standalone", "aggregate"]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=list(STAGES) + ["all"])
    p.add_argument("--data-repo", default=DEFAULT_DATA_REPO)
    p.add_argument("--results-root", default=DEFAULT_RESULTS)
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--methods", default="off,on")
    p.add_argument("--seeds", default=",".join(str(s) for s in ds.SEEDS))
    p.add_argument("--drafts", default=",".join(DRAFTS))
    p.add_argument("--gammas", default=",".join(str(g) for g in ev.GAMMAS))
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    ctx = Ctx(args)
    stages = ALL_ORDER if args.stage == "all" else [args.stage]
    res = {}
    for s in stages:
        print(f"===== stage {s} =====", flush=True)
        res[s] = STAGES[s](ctx)
    print(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    main()
