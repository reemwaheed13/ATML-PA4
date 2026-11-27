"""
Small I/O helpers shared by the Task 2 / Task 3 drivers.

Resumability contract (same as task1_compression/run_task1.py):
  * a run's `summary.json` is written LAST and atomically; its existence marks the
    run complete and the driver skips it unless overwrite=True;
  * a run that raises writes `error.json` (type, message, traceback) next to where its
    summary would be, the driver logs it and continues with the next run;
  * a later successful run removes a stale error.json.
"""

import gc
import gzip
import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

SEED = 6304


def iter_jsonl(path) -> Iterable[dict]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path) -> List[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path, rows: Iterable[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(tmp, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path, obj, indent: Optional[int] = 1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def free_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def run_resumable(names: List[str], out_dir_of: Callable[[str], Path],
                  run_one: Callable[[str, Path], dict], overwrite: bool = False,
                  log: Callable[[str], None] = print,
                  marker: str = "summary.json") -> Dict[str, str]:
    """Run `run_one(name, out_dir)` for each name; `run_one` must write `marker` last.

    Returns {name: 'done' | 'skipped' | 'failed'}.
    """
    status: Dict[str, str] = {}
    for name in names:
        out = Path(out_dir_of(name))
        if (out / marker).exists() and not overwrite:
            log(f"[skip] {name}: {marker} exists")
            status[name] = "skipped"
            continue
        out.mkdir(parents=True, exist_ok=True)
        err_path = out / "error.json"
        try:
            run_one(name, out)
            if not (out / marker).exists():
                raise RuntimeError(f"{name}: run finished without writing {marker}")
        except Exception as e:  # noqa: BLE001 - isolate per-run failures by design
            err = {"run": name, "exception_type": type(e).__name__, "message": str(e),
                   "traceback": traceback.format_exc()}
            write_json(err_path, err)
            log(f"[FAILED] {name}: {type(e).__name__}: {e}\n{err['traceback']}")
            status[name] = "failed"
            free_cuda()
            continue
        if err_path.exists():
            err_path.unlink()
        status[name] = "done"
        free_cuda()
    for n, s in status.items():
        log(f"[status] {n}: {s}")
    return status
