"""
Timing harness for latency measurements.

Requirements from PDF "Shared Experimental Conventions":
  - Warm up the relevant path before timing.
  - Synchronize CUDA around timed operations.
  - Keep input shapes fixed for matched methods.
  - Report measurement unit and number of repetitions.
  - Separate prefill, decode, and postprocessing where relevant.
"""

import contextlib
import time
import statistics
from typing import Callable, Dict, List, Optional

import torch


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def _cuda_timer():
    """Returns a dict filled with 'elapsed_ms' on exit."""
    info = {}
    _cuda_sync()
    t0 = time.perf_counter()
    yield info
    _cuda_sync()
    info["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0


def timed(
    fn: Callable,
    *args,
    n_warmup: int = 3,
    n_rep: int = 5,
    **kwargs,
) -> Dict:
    """
    Run `fn(*args, **kwargs)` with warmup and CUDA-synchronised timing.

    Returns
    -------
    dict:
      times_ms  list of float   individual repetition times
      mean_ms   float
      std_ms    float
      n_warmup  int
      n_rep     int
    """
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(n_rep):
        with _cuda_timer() as info:
            fn(*args, **kwargs)
        times.append(info["elapsed_ms"])

    return {
        "times_ms": times,
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "n_warmup": n_warmup,
        "n_rep": n_rep,
    }


def timed_steps(
    steps: List[Callable],
    step_names: List[str],
    *,
    n_warmup: int = 3,
    n_rep: int = 5,
) -> Dict:
    """
    Time a sequence of steps separately.  Useful for decomposing a speculative
    round into: draft_propose | target_verify | accept_sample | cache_rollback.

    Each callable in `steps` takes no arguments; use functools.partial or
    lambdas to bind arguments.

    Returns dict mapping step_name → timed() result dict.
    """
    # Warmup full pipeline.
    for _ in range(n_warmup):
        for fn in steps:
            fn()

    results = {name: {"times_ms": [], "n_warmup": n_warmup, "n_rep": n_rep}
               for name in step_names}

    for _ in range(n_rep):
        for fn, name in zip(steps, step_names):
            with _cuda_timer() as info:
                fn()
            results[name]["times_ms"].append(info["elapsed_ms"])

    for name in step_names:
        t = results[name]["times_ms"]
        results[name]["mean_ms"] = statistics.mean(t)
        results[name]["std_ms"] = statistics.stdev(t) if len(t) > 1 else 0.0

    return results


def peak_gpu_memory_mb(device: str = "cuda:0") -> float:
    """Return peak allocated GPU memory in MB since the last reset."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def reset_peak_memory(device: str = "cuda:0"):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
