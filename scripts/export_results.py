"""Export the machine-readable Task 2 / Task 3 results from a Drive results root into a
staging directory that mirrors the repo's results/task{2,3}/, then zip it.

    python scripts/export_results.py --results-root /content/drive/MyDrive/atml_pa4_results \
        --task 2 --staging /content/export_task2 --zip /content/drive/MyDrive/atml_pa4_results/export_task2.zip

Only whitelisted files are copied (tables, plots, summaries, curves, manifests, validation
outputs and per-condition / per-example / per-prompt results).  Model checkpoints, merged
weights, LoRA adapters, optimizer states and .npz MQAR streams are never copied, nor any
teacher-pool rollout file > 5 MB or any file > 20 MB.  Every skipped file is printed with its
size.  results/task{N}/EXPORT_MANIFEST.json lists each copied file with its sha256 and size.
Unzip the archive at the repo root to populate results/task{N}/.
"""
import argparse
import fnmatch
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

MAX_BYTES = 20 * 1024 * 1024
TEACHER_POOL_MAX_BYTES = 5 * 1024 * 1024
MANIFEST = "EXPORT_MANIFEST.json"

# Never copied, whatever the whitelist says.
DENY_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".npz", ".npy", ".gguf", ".tmp")
DENY_DIRS = {"adapter", "merged", "_train_cache", "checkpoints"}
DENY_NAME_PATTERNS = ("optimizer*", "scheduler*", "ckpt_*", "adapter_model*", "*.error.json", "error.json")

# Relative-path globs (fnmatch, '*' also matches '/') of what is exported.
WHITELIST = {
    2: [
        "tables/*.csv", "tables/*.json", "plots/*.png",
        "*summary.json", "*curves.json", "b/grid_manifest.json",
        "c/c1_validation.json",                 # C1 cache validation + KV quant diagnostic
        "a/checks.json",                        # 2A causality / tile / edge-case validation
        "a/conditions/*.json",                  # 2A per-condition records
        "b/forward_sweep/*.json",               # 2B forward-only sweep records
        "c/packed_bench.json",                  # C5 per-condition benchmark
    ],
    3: [
        "tables/*.csv", "tables/*.json", "plots/*.png",
        "*summary.json", "*curves.json", "*.meta.json",
        "check.json", "sampler_validation.json",          # check / validate (chi^2) outputs
        "spec/*/primary.jsonl.gz", "spec/*/fixed64.jsonl.gz",  # per-prompt speculative runs
        "standalone/*/rows.jsonl.gz",                     # per-prompt standalone accuracy
        "banks/alignment/*/rows.jsonl.gz", "banks/alignment/rows.jsonl.gz",  # per-prefix alignment
        "banks/*.jsonl.gz",                               # frozen prefix banks (per prompt)
        "distill/teacher_pool.jsonl.gz",                  # teacher pool (only if <= 5 MB)
    ],
}


def _match(rel, patterns):
    return any(fnmatch.fnmatch(rel, p) for p in patterns)


def classify(rel: str, size: int, task: int):
    """Return None if `rel` (posix path relative to the task root) is exported, else a skip reason."""
    parts = rel.split("/")
    name = parts[-1]
    if rel == MANIFEST:
        return "previous export manifest"
    if name.lower().endswith(DENY_SUFFIXES):
        return "weights / optimizer / stream / temp file"
    if any(p in DENY_DIRS for p in parts[:-1]):
        return "checkpoint / adapter / merged-weights / training-cache directory"
    if any(fnmatch.fnmatch(name, p) for p in DENY_NAME_PATTERNS):
        return "checkpoint / optimizer / error file"
    if not _match(rel, WHITELIST[task]):
        return "not whitelisted"
    if "teacher_pool" in name and not name.endswith(".meta.json") and size > TEACHER_POOL_MAX_BYTES:
        return "teacher-pool rollouts > 5 MB"
    if size > MAX_BYTES:
        return "file > 20 MB"
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mb(n):
    return f"{n / 1e6:.2f} MB"


def export(results_root, task: int, staging, zip_path=None, log=print) -> dict:
    task = int(task)
    src = Path(results_root) / f"task{task}"
    if not src.is_dir():
        raise FileNotFoundError(f"no results directory {src}")
    staging = Path(staging)
    dst = staging / "results" / f"task{task}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    copied, skipped = [], []
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        size = p.stat().st_size
        reason = classify(rel, size, task)
        if reason is not None:
            skipped.append({"path": rel, "size": size, "reason": reason})
            log(f"[skip] {rel} ({_mb(size)}): {reason}")
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        copied.append({"path": rel, "size": size, "sha256": sha256(out)})

    manifest = {"task": task, "source": str(src), "n_files": len(copied),
                "total_bytes": sum(f["size"] for f in copied),
                "max_file_bytes": MAX_BYTES, "teacher_pool_max_bytes": TEACHER_POOL_MAX_BYTES,
                "files": copied, "skipped": skipped}
    (dst / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"copied {len(copied)} files ({_mb(manifest['total_bytes'])}), skipped {len(skipped)} -> {dst}")

    if zip_path is not None:
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = zip_path.with_name(zip_path.name + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(dst.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(staging).as_posix())
        tmp.replace(zip_path)
        log(f"wrote {zip_path} ({_mb(zip_path.stat().st_size)})")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-root", required=True, help="dir containing task2/ and task3/")
    ap.add_argument("--task", type=int, choices=[2, 3], required=True)
    ap.add_argument("--staging", default=None, help="staging dir (default ./export_task{N})")
    ap.add_argument("--zip", default=None, help="zip path (default <staging>.zip)")
    a = ap.parse_args(argv)
    staging = Path(a.staging or f"export_task{a.task}")
    zip_path = Path(a.zip) if a.zip else staging.with_name(staging.name + ".zip")
    export(a.results_root, a.task, staging, zip_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
