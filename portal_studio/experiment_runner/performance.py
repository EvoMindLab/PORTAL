"""PORTAL optimizer-performance characterization utilities.

Collect best-so-far values at fixed evaluation checkpoints and summarize
repeated-run errors as median log errors. These PORTAL-owned utilities define
the behavioral coordinates used for curation; no instance-space embedding or
algorithm-footprint computation is performed here.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


PERFORMANCE_CHECKPOINTS_FE = (30000, 50000, 100000, 200000, 300000)
PERFORMANCE_LOG_EPS = 1e-10


def checkpoint_best_so_far(fe_count: Iterable[int],
                           best_f: Iterable[float],
                           checkpoints: Iterable[int] = PERFORMANCE_CHECKPOINTS_FE,
                           *, fitness_batches: Iterable[Iterable[float]] | None = None,
                           ) -> dict[int, float]:
    """Best-so-far value at each fixed FE checkpoint.

    Checkpoints beyond the tape's last FE carry the final best-so-far value
    forward, matching the repository performance-collection convention.

    Recorded batch fitnesses, when supplied, recover the running minimum at
    the exact evaluation inside a batch. They never trigger new evaluations.
    Sparse histories without those values use the last completed
    record; a checkpoint before the first record has no known value (inf).
    """
    fe_arr = np.asarray(list(fe_count), dtype=int)
    bf_arr = np.asarray(list(best_f), dtype=float)
    if fe_arr.size == 0:
        return {}
    if fitness_batches is not None:
        batches = [np.asarray(values, dtype=float).reshape(-1)
                   for values in fitness_batches]
        if (len(batches) != len(fe_arr)
                or not np.array_equal(np.cumsum([len(values) for values in batches]),
                                      fe_arr)):
            raise ValueError('Recorded batch sizes do not match evaluation counts.')
        bf_arr = np.concatenate(batches)
        fe_arr = np.arange(1, len(bf_arr) + 1)
    running_min = np.minimum.accumulate(bf_arr)
    out: dict[int, float] = {}
    for checkpoint in checkpoints:
        idx = np.searchsorted(fe_arr, int(checkpoint), side="right") - 1
        out[int(checkpoint)] = float(running_min[idx]) if idx >= 0 else float('inf')
    return out


def median_log_error(errors: Iterable[float],
                     eps: float = PERFORMANCE_LOG_EPS) -> float:
    """Return median_r log(error_r + eps), one algorithm-performance coordinate."""
    arr = np.asarray(list(errors), dtype=float)
    return float(np.median(np.log(arr + eps)))
