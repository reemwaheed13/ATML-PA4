"""
Task 2 driver (PDF §2).  Every stage is resumable: finished runs are skipped, a failed
run writes error.json and the driver continues.  Results go to Drive, never the repo.

Usage (Colab):
    python -m task2_attention.run_task2 2a                    # Part A sweep + checks
    python -m task2_attention.run_task2 2b --stage train      # 42-run MQAR grid
    python -m task2_attention.run_task2 2b --stage sweep      # forward-only sweep
    python -m task2_attention.run_task2 2b --runs B1          # subset (B1|B2|B3|attn|run names)
    python -m task2_attention.run_task2 2c --stage c1         # C1 cache validation
    python -m task2_attention.run_task2 2c --stage eval       # C2-C5: 180 prompts x 6 methods
    python -m task2_attention.run_task2 2c --stage bench      # C5 packed encode/decode benchmark
    python -m task2_attention.run_task2 aggregate             # every table / plot
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from task2_attention import part_b  # noqa: E402
from task2_attention import mqar_train as mt  # noqa: E402

DEFAULT_DATA_REPO = "/content/drive/MyDrive/atml-pa4-datasets"
DEFAULT_RESULTS = "/content/drive/MyDrive/atml_pa4_results/task2"


def _device(arg):
    return arg or ("cuda" if torch.cuda.is_available() else "cpu")


def cmd_2b(args):
    root = Path(args.results_root) / "b"
    dev = _device(args.device)
    out = {}
    if args.stage in ("train", "all"):
        runs = part_b.select_runs(args.runs)
        cache = args.train_cache if args.train_cache else str(root / "_train_cache")
        out["train"] = part_b.run_grid(str(root), args.data_repo, runs, dev, args.overwrite,
                                       n_updates=args.n_updates, cache_dir=cache)
    if args.stage in ("sweep", "all"):
        Ts = tuple(int(t) for t in args.sweep_T.split(","))
        out["sweep"] = part_b.run_sweep(str(root), dev, args.overwrite, Ts=Ts,
                                        n_rep=args.sweep_reps)
    return out


def cmd_2a(args):
    from task2_attention import part_a
    from common.runio import write_json
    root = Path(args.results_root) / "a"
    dev = _device(args.device)
    Ls = tuple(int(x) for x in args.lengths.split(","))
    out = {"sweep": part_a.run_sweep(str(root), dev, lengths=Ls, overwrite=args.overwrite, n_rep=args.reps)}
    chk = root / "checks.json"
    if args.overwrite or not chk.exists():
        write_json(chk, part_a.run_checks(dev))
    out["checks"] = str(chk)
    return out


def cmd_2c(args):
    from task2_attention import part_c
    from common.runio import read_json, write_json
    root = Path(args.results_root) / "c"
    root.mkdir(parents=True, exist_ok=True)
    dev = _device(args.device)
    revs = read_json(_REPO_ROOT / "data" / "local_source_revisions.json")
    model, tok = part_c.load_model(dev, revs["revisions"]["model_target"])
    out = {}
    if args.stage in ("c1", "all"):
        p = root / "c1_validation.json"
        if args.overwrite or not p.exists():
            cases = part_c.load_cases(args.data_repo)
            pick = [next(c for c in cases if c["target_tokens"] == t and c["position"] == pos)
                    for t in (2048, 4096) for pos in ("start", "end")]
            write_json(p, part_c.c1_validate(model, tok, pick, dev))
        out["c1_pass"] = read_json(p)["pass"]
    if args.stage in ("eval", "all"):
        out["eval"] = part_c.run_eval(str(root), args.data_repo, dev, args.overwrite, args.limit,
                                      model=model, tokenizer=tok)
    if args.stage in ("bench", "all"):
        p = root / "packed_bench.json"
        if args.overwrite or not p.exists():
            write_json(p, part_c.bench_packed(model, dev))
        out["bench"] = str(p)
    return out


def cmd_aggregate(args):
    from task2_attention.aggregate import aggregate
    return aggregate(args.results_root)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("2b", help="MQAR: 42-run grid (B1/B2/B3) and forward-only sweep")
    b.add_argument("--stage", choices=["train", "sweep", "all"], default="all")
    b.add_argument("--runs", default="all", help="'all', B1/B2/B3, an attention type, or run names (comma list)")
    b.add_argument("--data-repo", default=DEFAULT_DATA_REPO)
    b.add_argument("--results-root", default=DEFAULT_RESULTS)
    b.add_argument("--train-cache", default="", help="dir for cached generator streams (default <results>/b/_train_cache)")
    b.add_argument("--device", default=None)
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--n-updates", type=int, default=mt.N_UPDATES,
                   help="TESTS ONLY: the PDF protocol is 4,500; summaries record full_protocol")
    b.add_argument("--sweep-T", default=",".join(str(t) for t in part_b.SWEEP_T))
    b.add_argument("--sweep-reps", type=int, default=10)
    b.set_defaults(func=cmd_2b)

    a = sub.add_parser("2a", help="exact attention: dense / tiled / SDPA sweep and checks")
    a.add_argument("--results-root", default=DEFAULT_RESULTS)
    a.add_argument("--lengths", default="256,512,1024,2048,4096,8192")
    a.add_argument("--reps", type=int, default=10)
    a.add_argument("--device", default=None)
    a.add_argument("--overwrite", action="store_true")
    a.set_defaults(func=cmd_2a)

    c = sub.add_parser("2c", help="KV memory on Qwen2.5-1.5B-Instruct")
    c.add_argument("--stage", choices=["c1", "eval", "bench", "all"], default="all")
    c.add_argument("--data-repo", default=DEFAULT_DATA_REPO)
    c.add_argument("--results-root", default=DEFAULT_RESULTS)
    c.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke)")
    c.add_argument("--device", default=None)
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_2c)

    g = sub.add_parser("aggregate", help="write every Task 2 table and plot")
    g.add_argument("--results-root", default=DEFAULT_RESULTS)
    g.set_defaults(func=cmd_aggregate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    res = args.func(args)
    print(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    main()
