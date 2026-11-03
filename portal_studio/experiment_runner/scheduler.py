"""Per-batch ProcessPoolExecutor wrapper.

Each batch owns one pool, uses a serial path when ``max_parallel_runs == 1``,
and supports cooperative cancellation: cancel queued units where possible,
then collect in-flight outcomes before reporting terminal counts.

Two public entry points:

- :class:`build_run_units` — pure-Python expansion of a
  :class:`BatchConfig` into a flat list of :class:`RunUnit` work items
  (handles the multi-algorithm fan-out and the per-run seed strategy).
- :class:`BatchScheduler` — owns the pool, the cancel flag, and the
  progress callbacks. The Qt layer wraps this in a ``QThread``.
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from concurrent.futures import (
    Future, ProcessPoolExecutor, as_completed,
)
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import numpy as np

from portal_studio.algorithms.runner import RunCanceled
from portal_studio.experiment_runner.config import BatchConfig
from portal_studio.experiment_runner.runner import RunUnit, execute_run
from portal_studio.experiment_runner.store import ResultStore


@dataclass(frozen=True, slots=True)
class RunFailure:
    """A failed attempt, never a successful observation in the result tables."""
    runset_uid: str
    task_label: str
    algorithm: str
    run_index: int
    seed: int
    error: str

    @property
    def message(self) -> str:
        return (f"{self.task_label} / {self.algorithm}, run {self.run_index + 1} "
                f"(seed {self.seed}): {self.error}")


@dataclass(slots=True)
class BatchProgress:
    """One snapshot the GUI shows in the Jobs table."""
    batch_id: str
    status: str               # 'queued' | 'running' | 'finished' | 'failed' | 'canceled'
    total_runs: int
    completed_runs: int
    message: str = ""
    failures: tuple[RunFailure, ...] = ()
    runset_counts: dict[str, int] | None = None  # Settled persisted counts on terminal snapshots.

    @property
    def failed_runs(self) -> int:
        return len(self.failures)


def build_run_units(cfg: BatchConfig) -> list[RunUnit]:
    """Expand the batch into a flat list of (algorithm, run_index) units.

    Multi-algorithm tasks fan out: ``algorithm: ["PSO", "DE"]`` and
    ``runs=3`` → 6 units, all sharing the same task label. Seeds are
    drawn from the task's ``master_seed`` per a deterministic algorithm
    that depends on (task_label, algorithm, run_index) so that re-runs
    are reproducible *and* every (algorithm, run) gets a distinct seed.
    """
    units: list[RunUnit] = []
    for task in cfg.tasks:
        for algo in task.algorithm_list():
            # Merge shared params with per-algo overrides.
            params = dict(task.algorithm_params)
            params.update(task.per_algo_params.get(algo, {}))
            runset_uid = f"{task.label}::{algo}"
            seeds = _draw_seeds(task.master_seed, task.label, algo,
                                task.runs, task.seed_strategy)
            for run_index, seed in enumerate(seeds):
                units.append(RunUnit(
                    runset_uid=runset_uid,
                    task_label=task.label,
                    algorithm=algo,
                    algorithm_params=params,
                    run_index=run_index,
                    seed=seed,
                    fe_budget=cfg.fe_budget,
                    realization_type=task.realization_type,
                ))
    return units


def _draw_seeds(master_seed: int, task_label: str, algorithm: str,
                runs: int, strategy: str) -> list[int]:
    """Deterministic per-run seed sequence.

    For ``fixed``: use the contiguous sequence
    ``master_seed, master_seed+1, …`` so each run has a distinct seed.
    This supports reproducible repeated runs with different random streams.
    For ``random``: a numpy ``SeedSequence`` keyed by (task_label,
    algorithm, master_seed) spawns ``runs`` independent child seeds.
    """
    if strategy == "fixed":
        return [int(master_seed) + i for i in range(int(runs))]
    # SeedSequence accepts a tuple-like; we fold task+algo into an entropy mix.
    entropy = (int(master_seed), _stable_hash(task_label),
               _stable_hash(algorithm))
    ss = np.random.SeedSequence(entropy=entropy)
    children = ss.spawn(int(runs))
    return [int(c.generate_state(1)[0]) for c in children]


def _stable_hash(text: str) -> int:
    """Cheap deterministic int hash; ``hash()`` is randomised per process."""
    h = 1469598103934665603
    for ch in text.encode("utf-8"):
        h ^= ch
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h & 0x7FFFFFFF


class BatchScheduler:
    """One scheduler per batch. Spawns the pool, drives the units,
    serialises progress via callbacks.

    Lifecycle::

        scheduler = BatchScheduler(cfg, composite, output_dir)
        scheduler.run(progress=on_progress, finished=on_finished)
        # ... from another thread, possibly:
        scheduler.cancel()
    """

    def __init__(self, cfg: BatchConfig, composite: dict[str, Any],
                 output_dir: str, *, batch_id: str | None = None) -> None:
        self.cfg = cfg
        self.composite = composite
        self.output_dir = output_dir
        self.batch_id = batch_id or _new_batch_id()
        self._cancel = threading.Event()
        self._units = build_run_units(cfg)
        self._completed = 0
        self._failures: list[RunFailure] = []
        self._runset_expected = {
            uid: len(units) for uid, units in _group_by_runset(self._units).items()
        }
        self._successful_indices: dict[str, set[int]] = {}
        self._finished_runsets: set[str] = set()

    # ─────────────────────────── public surface ─────────────────────

    @property
    def total_runs(self) -> int:
        return len(self._units)

    def cancel(self) -> None:
        """Cooperative cancel — flips the flag; in-flight units finish."""
        self._cancel.set()

    def run(self, *, progress: Callable[[BatchProgress], None] | None = None,
            unit_finished: Callable[[dict[str, Any]], None] | None = None,
            ) -> BatchProgress:
        """Run the batch to completion (or cancel). Returns the final state.

        Individual run errors are reported and do not stop other runs.
        ``completed_runs`` counts successes only; ``failed_runs`` counts errors.
        ``progress`` is invoked after every run finishes with a fresh
        :class:`BatchProgress` snapshot. ``unit_finished`` is invoked with
        the small dict :func:`execute_run` returns, so callers can stream
        per-run results into a UI without re-querying the store.
        """
        store = self._init_store()
        cfg_pool = max(1, min(self.cfg.max_parallel_runs,
                              (os.cpu_count() or 2) - 1))

        self._emit_progress(progress, "running")

        try:
            if cfg_pool == 1:
                self._run_serially(store, progress, unit_finished)
            else:
                self._run_parallel(cfg_pool, progress, unit_finished)
        except Exception as exc:    # pragma: no cover
            return self._finalise(progress, status="failed", message=str(exc))

        if self._cancel.is_set():
            return self._finalise(progress, status="canceled",
                                  message="canceled by user")
        message = (f"{self._completed}/{self.total_runs} runs succeeded; "
                   f"{len(self._failures)} runs failed.") if self._failures else ""
        return self._finalise(progress, status="finished", message=message)

    # ─────────────────────────── internals ──────────────────────────

    def _init_store(self) -> ResultStore:
        store = ResultStore(self.output_dir)
        store.ensure_schema()
        store.write_meta({
            "batch_id": self.batch_id,
            "fe_budget": self.cfg.fe_budget,
            "metric_outputs": self.cfg.metric_outputs,
            "composite_path": self.cfg.composite_path,
            "started_at": _utcnow(),
            "run_failures": [],
        })
        # Pre-declare every runset so the GUI can render rows immediately.
        for runset_uid, units in _group_by_runset(self._units).items():
            store.declare_runset(
                runset_uid=runset_uid,
                batch_id=self.batch_id,
                task_label=units[0].task_label,
                algorithm=units[0].algorithm,
                algorithm_params=units[0].algorithm_params,
                runs_expected=len(units),
            )
            store.set_runset_status(runset_uid, 'running', started_at=_utcnow())
        return store

    def _run_serially(self, store: ResultStore,
                      progress: Callable[[BatchProgress], None] | None,
                      unit_finished: Callable[[dict[str, Any]], None] | None,
                      ) -> None:
        self._run_units_serially(self._units, progress, unit_finished)

    def _run_units_serially(self, units: list[RunUnit],
                            progress: Callable[[BatchProgress], None] | None,
                            unit_finished: Callable[[dict[str, Any]], None] | None,
                            ) -> None:
        for unit in units:
            if self._cancel.is_set():
                return
            try:
                out = execute_run(unit, self.composite, self.output_dir)
            except RunCanceled:
                self.cancel()
                return
            except Exception as exc:
                self._record_failure(unit, exc, progress)
                continue
            self._completed += 1
            out = self._finish_successful_runset(unit, out)
            self._emit_progress(progress, "running")
            if unit_finished is not None:
                unit_finished(out)

    def _run_parallel(self, max_workers: int,
                      progress: Callable[[BatchProgress], None] | None,
                      unit_finished: Callable[[dict[str, Any]], None] | None,
                      ) -> None:
        from concurrent.futures.process import BrokenProcessPool

        if self._cancel.is_set():
            return
        done: set[int] = set()      # id() of units whose result we collected
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures: dict[Future, RunUnit] = {
                    pool.submit(execute_run, u, self.composite, self.output_dir): u
                    for u in self._units
                }
                for fut in as_completed(futures):
                    if self._cancel.is_set():
                        self._settle_canceled_futures(futures, done, progress, unit_finished)
                        return
                    self._collect_future(fut, futures[fut], done, progress, unit_finished)
                    if self._cancel.is_set():
                        self._settle_canceled_futures(futures, done, progress, unit_finished)
                        return
        except BrokenProcessPool:
            # If the process pool fails, retry unsettled units serially.
            # ``record_run`` replaces rows by run_uid, so a result persisted
            # before a lost worker response does not create a duplicate.
            if self._cancel.is_set():
                return
            remaining = [u for u in self._units if id(u) not in done]
            self._run_units_serially(remaining, progress, unit_finished)

    def _collect_future(self, future: Future, unit: RunUnit, done: set[int],
                        progress, unit_finished) -> None:
        from concurrent.futures import CancelledError
        from concurrent.futures.process import BrokenProcessPool

        if id(unit) in done:
            return
        try:
            out = future.result()
        except CancelledError:
            done.add(id(unit))
            return
        except BrokenProcessPool:
            raise
        except RunCanceled:
            done.add(id(unit))
            self.cancel()
            return
        except Exception as exc:
            done.add(id(unit))
            self._record_failure(unit, exc, progress)
            return
        done.add(id(unit))
        self._completed += 1
        out = self._finish_successful_runset(unit, out)
        self._emit_progress(progress, 'running')
        if unit_finished is not None:
            unit_finished(out)

    def _finish_successful_runset(self, unit: RunUnit, out: dict) -> dict:
        """Commit completion before notifying the GUI, independently of peers."""
        indices = self._successful_indices.setdefault(unit.runset_uid, set())
        indices.add(unit.run_index)
        if (len(indices) == self._runset_expected[unit.runset_uid]
                and unit.runset_uid not in self._finished_runsets
                and not self._cancel.is_set()
                and not any(f.runset_uid == unit.runset_uid for f in self._failures)):
            store = ResultStore(self.output_dir)
            persisted = store.load_run_scalars(unit.runset_uid)
            if len(persisted.get('final_error', [])) == len(indices):
                store.set_runset_status(unit.runset_uid, 'finished', finished_at=_utcnow())
                self._finished_runsets.add(unit.runset_uid)
                return {**out, 'runset_finished': True}
        return out

    def _settle_canceled_futures(self, futures: dict[Future, RunUnit], done: set[int],
                                 progress, unit_finished) -> None:
        from concurrent.futures.process import BrokenProcessPool

        self._emit_progress(progress, 'running',
                            'Cancellation requested; waiting for in-flight runs.')
        for future, unit in futures.items():
            if id(unit) not in done:
                future.cancel()  # Best effort: running or dispatched work can still finish.
        for future, unit in futures.items():
            try:
                self._collect_future(future, unit, done, progress, unit_finished)
            except BrokenProcessPool:
                # Do not restart work after cancellation. A result may already
                # be persisted despite a lost worker response; final counts
                # are reconciled against the store after the pool settles.
                continue

    def _record_failure(self, unit: RunUnit, exc: Exception, progress) -> None:
        failure = RunFailure(
            unit.runset_uid, unit.task_label, unit.algorithm,
            unit.run_index, unit.seed, f"{type(exc).__name__}: {exc}",
        )
        self._failures.append(failure)
        # Preserve diagnostics separately from the successful-run measurements.
        ResultStore(self.output_dir).write_meta({
            "run_failures": [asdict(item) for item in self._failures],
        })
        logging.getLogger(__name__).warning("Run failed: %s", failure.message)
        self._emit_progress(progress, "running", failure.message)

    def _emit_progress(self,
                       progress: Callable[[BatchProgress], None] | None,
                       status: str,
                       message: str = "") -> None:
        if progress is None:
            return
        progress(BatchProgress(
            batch_id=self.batch_id,
            status=status,
            total_runs=self.total_runs,
            completed_runs=self._completed,
            message=message,
            failures=tuple(self._failures),
        ))

    def _finalise(self,
                  progress: Callable[[BatchProgress], None] | None,
                  *, status: str, message: str = "") -> BatchProgress:
        store = ResultStore(self.output_dir)
        failed_runsets = {failure.runset_uid for failure in self._failures}
        runset_counts = {}
        for uid, units in _group_by_runset(self._units).items():
            scalars = store.load_run_scalars(uid)
            runset_counts[uid] = len(scalars.get('final_error', []))
            if uid in self._finished_runsets:
                # A later cancellation or failure belongs to unfinished peers.
                continue
            complete = runset_counts[uid] == len(units)
            terminal = ('canceled' if status == 'canceled' else
                        'finished' if complete and uid not in failed_runsets else 'failed')
            store.set_runset_status(uid, terminal, finished_at=_utcnow())
        if status == 'canceled':
            self._completed = sum(runset_counts.values())
        final = BatchProgress(
            batch_id=self.batch_id, status=status,
            total_runs=self.total_runs,
            completed_runs=self._completed, message=message,
            failures=tuple(self._failures), runset_counts=runset_counts)
        if progress is not None:
            progress(final)
        return final


def _group_by_runset(units: Iterable[RunUnit]) -> dict[str, list[RunUnit]]:
    out: dict[str, list[RunUnit]] = {}
    for u in units:
        out.setdefault(u.runset_uid, []).append(u)
    return out


def _new_batch_id() -> str:
    return uuid.uuid4().hex[:12]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
