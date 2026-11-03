"""Per-run executor — the worker entry point.

One call = one (PORTAL instance, algorithm, run_index, seed) tuple → one row in
``summary.sqlite::runs`` plus its trace folded into ``trend.sqlite``. The
function is module-level (not nested in a class) so it pickles cleanly
into a :class:`concurrent.futures.ProcessPoolExecutor` worker.

We reuse the existing single-run runner at
:mod:`portal_studio.algorithms.runner` rather than re-implementing it —
that runner already wires the :class:`RecordingProblem` tape, the
solver registry, and the global-optimum lookup. Our job here is just
to:

1. Resolve the (algorithm-name, params) → solver call arguments.
2. Invoke ``run_solver_on_instance`` with the right seed/budget.
3. Convert the recorded tape into the downsampled best-so-far trace.
4. Hand the per-run scalars + trace to :class:`ResultStore`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from portal_studio.experiment_runner.performance import (
    PERFORMANCE_CHECKPOINTS_FE, checkpoint_best_so_far,
)
from portal_studio.experiment_runner.store import (
    ResultStore, RunRecord,
)
from portal_studio.experiment_runner.convergence import checkpoint_grid
from portal_studio.experiment_runner.provenance import pairing_key
from portal_studio.experiment_runner.normalisation import reference_scale


@dataclass
class RunUnit:
    """One unit of work — the smallest thing the scheduler hands a worker."""
    runset_uid: str          # task_label::algorithm
    task_label: str
    algorithm: str
    algorithm_params: dict[str, Any]
    run_index: int           # 0-based index within the runset
    seed: int                # the seed for this specific run
    fe_budget: int
    realization_type: str = "none"
    realization_base_seed: int | str = "train"


def execute_run(unit: RunUnit, composite: dict[str, Any],
                output_dir: str) -> dict[str, Any]:
    """Worker entry: run one (algorithm, seed) and persist to the store.

    Returns a small dict so the main process can log the outcome without
    re-reading the DB. The :class:`ResultStore` write happens inside the
    worker; SQLite's file lock serialises concurrent writers.
    """
    # Import lazily so worker cold-start stays minimal — no Qt, no GUI.
    from portal_studio.algorithms.runner import run_solver_on_instance

    result = run_solver_on_instance(
        instance=composite,
        solver_name=unit.algorithm,
        budget=unit.fe_budget,
        seed=unit.seed,
        params=unit.algorithm_params or None,
        realization_type=unit.realization_type,
        realization_base_seed=unit.realization_base_seed,
        realization_sample_id=int(unit.run_index) + 1,
    )

    # Translate the RecordingProblem tape into a downsampled best-so-far
    # convergence curve.
    tape = result.problem.tape
    fe_counts = [frame.fe_count for frame in tape]
    best_seen = [frame.best_f for frame in tape]

    # Absolute minimization error vs the known global optimum. PORTAL problems
    # should not beat F*, but abs() matches the paper scripts and keeps plots
    # and statistics nonnegative if numerical noise crosses it.
    final_best_f = float(tape[-1].best_f) if tape else float("inf")
    global_opt = float(result.global_optimum)
    final_error = abs(final_best_f - global_opt)
    if not np.isfinite(final_error):
        raise ValueError('The run produced no finite final error; it cannot be stored as successful.')
    fitness_batches = ([frame.fitnesses for frame in tape]
                       if all(hasattr(frame, 'fitnesses') for frame in tape) else None)
    # Recover exact within-batch best values on a budget-based common grid.
    # Keep the actual stopping point too; readers carry it forward afterwards.
    grid = checkpoint_grid(unit.fe_budget)
    grid = np.unique(np.append(grid[grid <= fe_counts[-1]], fe_counts[-1]))
    trace = checkpoint_best_so_far(fe_counts, best_seen, grid, fitness_batches=fitness_batches)
    trace = {fe: value for fe, value in trace.items() if np.isfinite(value)}
    fe_ckpt = np.asarray(list(trace), dtype=int)
    best_ckpt = np.asarray(list(trace.values()), dtype=float)
    performance_vals = checkpoint_best_so_far(
        fe_counts, best_seen, PERFORMANCE_CHECKPOINTS_FE, fitness_batches=fitness_batches)
    performance_checkpoints = {
        fe: abs(float(value) - global_opt)
        for fe, value in performance_vals.items()
    }
    # Convert trace from raw best_f into best-so-far absolute error vs global opt
    # so plotting and statistical comparison work on the same scale across
    # instances with different scale offsets.
    best_err_ckpt = np.abs(best_ckpt - global_opt)

    evaluated = getattr(result.problem, 'instance', None)
    scale, scale_reason = reference_scale(evaluated)
    identity = None if evaluated is None else pairing_key(
        composite, evaluated, realization_type=unit.realization_type,
        base_seed=unit.realization_base_seed, sample_id=int(unit.run_index) + 1,
        fe_budget=unit.fe_budget,
    )
    rec = RunRecord(
        runset_uid=unit.runset_uid,
        run_index=unit.run_index,
        seed=unit.seed,
        final_error=final_error,
        fe_used=int(tape[-1].fe_count) if tape else 0,
        trace_fe=fe_ckpt,
        trace_best=best_err_ckpt,
        performance_checkpoints=performance_checkpoints,
        pairing_key=identity,
        reference_scale=scale,
        normalisation_reason=scale_reason,
    )
    store = ResultStore(output_dir)
    store.record_run(rec, finished_at=_utcnow())

    return {
        "runset_uid": unit.runset_uid,
        "run_index": unit.run_index,
        "seed": unit.seed,
        "final_error": final_error,
        "fe_used": rec.fe_used,
    }


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
