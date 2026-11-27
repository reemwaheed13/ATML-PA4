"""
Task 2 driver (PDF §2).  Every stage is resumable: finished runs are skipped, a failed
run writes error.json and the driver continues.  Results go to Drive, never the repo.

Usage (Colab):
    python -m task2_attention.run_task2 2b --stage train      # 42-run MQAR grid
    python -m task2_attention.run_task2 2b --stage sweep      # forward-only sweep
    python -m task2_attention.run_task2 2b --runs B1          # subset (B1|B2|B3|attn|run names)
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
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    res = args.func(args)
    print(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    main()
