"""Per-batch sqlite result store.

Two databases are written per batch directory:

- ``summary.sqlite``: per-run scalar metrics, individual sparse traces,
  per-runset row and batch metadata. One row per individual run; one row per runset (a runset is
  the cross-product of a task label × one algorithm).
- ``trend.sqlite``: aggregated (mean ± stderr) trend curves keyed by
  ``runset_uid`` and ``metric`` so plotting reads a small fixed-size set
  of rows per series even when ``runs`` is 31.

This module is intentionally pure-Python (no Qt). It is imported by both
the in-process Qt code (Results sub-tab) and by worker processes (which
write per-run rows from inside :func:`runner.execute_run`). Concurrent
writes from N workers are serialised by SQLite's file lock. Scalar and
trace replacement share a transaction across the attached databases.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from portal_studio.experiment_runner.convergence import read_aligned_moments, standard_error


SUMMARY_DB_NAME = "summary.sqlite"
TREND_DB_NAME = "trend.sqlite"


_SUMMARY_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runsets (
    runset_uid     TEXT PRIMARY KEY,
    batch_id       TEXT NOT NULL,
    task_label     TEXT NOT NULL,
    algorithm      TEXT NOT NULL,
    algorithm_params TEXT NOT NULL,    -- JSON
    runs_expected  INTEGER NOT NULL,
    runs_done      INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'queued',
    started_at     TEXT,
    finished_at    TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_uid        TEXT PRIMARY KEY,
    runset_uid     TEXT NOT NULL REFERENCES runsets(runset_uid),
    run_index      INTEGER NOT NULL,
    seed           INTEGER NOT NULL,
    final_error    REAL,
    fe_used        INTEGER,
    finished_at    TEXT
);

CREATE INDEX IF NOT EXISTS runs_by_runset ON runs(runset_uid);

CREATE TABLE IF NOT EXISTS performance_checkpoints (
    run_uid        TEXT NOT NULL,
    runset_uid     TEXT NOT NULL,
    checkpoint_fe  INTEGER NOT NULL,
    absolute_error REAL NOT NULL,
    PRIMARY KEY (run_uid, checkpoint_fe)
);

CREATE INDEX IF NOT EXISTS performance_checkpoints_by_runset ON performance_checkpoints(runset_uid);
"""

_TREND_SCHEMA = """
CREATE TABLE IF NOT EXISTS trends (
    runset_uid TEXT NOT NULL,
    metric     TEXT NOT NULL,
    fe         INTEGER NOT NULL,
    mean       REAL NOT NULL,
    stderr     REAL NOT NULL,
    count      INTEGER NOT NULL,
    PRIMARY KEY (runset_uid, metric, fe)
);
"""


@dataclass
class RunRecord:
    """One run's scalar outcome — written from a worker."""
    runset_uid: str
    run_index: int
    seed: int
    final_error: float
    fe_used: int
    # Sparse best-so-far trace as paired arrays (FE checkpoints + best_f at
    # each checkpoint). Worker downsamples to ~200 buckets before sending,
    # so this stays small.
    trace_fe: np.ndarray
    trace_best: np.ndarray
    # Per-run minimization error at each canonical Performance checkpoint. Empty for
    # callers that do not compute Performance data. The SQLite column is still named
    # absolute_error for compatibility with existing result bundles.
    performance_checkpoints: dict[int, float] = field(default_factory=dict)
    # Verified identity of the source and evaluated realisation, not an
    # optimiser seed. None denotes a legacy/unverified observation.
    pairing_key: str | None = None
    reference_scale: float | None = None
    normalisation_reason: str | None = None


def migrate_performance_checkpoints(cx: sqlite3.Connection) -> None:
    """Preserve checkpoint rows from the historical table on a writable copy.

    The legacy SQL name is retained here solely to read older saved results.
    New databases and all current writes use performance_checkpoints.
    """
    tables = {row[0] for row in cx.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "isa_checkpoints" not in tables:
        return
    if "performance_checkpoints" not in tables:
        cx.execute("ALTER TABLE isa_checkpoints RENAME TO performance_checkpoints")
    else:
        conflict = cx.execute(
            "SELECT 1 FROM isa_checkpoints old JOIN performance_checkpoints new "
            "USING(run_uid, checkpoint_fe) WHERE old.runset_uid IS NOT new.runset_uid "
            "OR old.absolute_error IS NOT new.absolute_error LIMIT 1"
        ).fetchone()
        if conflict:
            raise ValueError("Legacy and canonical performance checkpoints conflict; both tables were retained.")
        cx.execute("INSERT OR IGNORE INTO performance_checkpoints "
                   "(run_uid,runset_uid,checkpoint_fe,absolute_error) "
                   "SELECT run_uid,runset_uid,checkpoint_fe,absolute_error FROM isa_checkpoints")
        cx.execute("DROP TABLE isa_checkpoints")
    cx.execute("DROP INDEX IF EXISTS isa_checkpoints_by_runset")
    cx.execute("CREATE INDEX IF NOT EXISTS performance_checkpoints_by_runset "
               "ON performance_checkpoints(runset_uid)")


class ResultStore:
    """File-backed store for one batch.

    Use :meth:`ensure_schema` once when the batch starts. Workers call
    :meth:`record_run` from inside their process. Reads
    (:meth:`list_runsets`, :meth:`load_trend`) are GUI-side.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self._summary_path = self.output_dir / SUMMARY_DB_NAME
        self._trend_path = self.output_dir / TREND_DB_NAME
        # SQLite's per-process connection is fine; we don't share connections
        # across threads — every method opens-uses-closes.
        self._lock = threading.Lock()

    # ───────────────────────── schema setup ─────────────────────────

    def ensure_schema(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self._open(self._summary_path) as cx:
            cx.execute('BEGIN IMMEDIATE')
            migrate_performance_checkpoints(cx)
            cx.commit()
            cx.executescript(_SUMMARY_SCHEMA)
            columns = {row[1] for row in cx.execute('PRAGMA table_info(runs)')}
            if 'pairing_key' not in columns:
                cx.execute('ALTER TABLE runs ADD COLUMN pairing_key TEXT')
            for name, kind in [('reference_scale', 'REAL'), ('normalisation_reason', 'TEXT')]:
                if name not in columns:
                    cx.execute(f'ALTER TABLE runs ADD COLUMN {name} {kind}')
            cx.execute('CREATE TABLE IF NOT EXISTS normalised_traces ('
                       'run_uid TEXT NOT NULL, fe INTEGER NOT NULL, value REAL NOT NULL, '
                       'PRIMARY KEY(run_uid,fe))')
            # A marker is needed even for runs with an empty trace. Its rowid
            # retains write order for the original Welford accumulation order.
            cx.execute('CREATE TABLE IF NOT EXISTS raw_trace_runs ('
                       'run_uid TEXT PRIMARY KEY)')
            cx.execute('CREATE TABLE IF NOT EXISTS raw_traces ('
                       'run_uid TEXT NOT NULL, fe INTEGER NOT NULL, value REAL NOT NULL, '
                       'PRIMARY KEY(run_uid,fe))')
        with self._open(self._trend_path) as cx:
            cx.executescript(_TREND_SCHEMA)

    def write_meta(self, items: dict[str, Any]) -> None:
        with self._open(self._summary_path) as cx:
            cx.executemany(
                "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                [(k, json.dumps(v)) for k, v in items.items()],
            )

    def declare_runset(self, *, runset_uid: str, batch_id: str,
                       task_label: str, algorithm: str,
                       algorithm_params: dict[str, Any],
                       runs_expected: int) -> None:
        """Insert a runset row in ``queued`` state."""
        with self._open(self._summary_path) as cx:
            cx.execute(
                """INSERT OR REPLACE INTO runsets
                   (runset_uid, batch_id, task_label, algorithm,
                    algorithm_params, runs_expected, runs_done, status)
                   VALUES (?, ?, ?, ?, ?, ?, 0, 'queued')""",
                (runset_uid, batch_id, task_label, algorithm,
                 json.dumps(algorithm_params), runs_expected),
            )

    def set_runset_status(self, runset_uid: str, status: str,
                          *, started_at: str | None = None,
                          finished_at: str | None = None) -> None:
        sets = ["status = ?"]
        args: list[Any] = [status]
        if started_at is not None:
            sets.append("started_at = ?")
            args.append(started_at)
        if finished_at is not None:
            sets.append("finished_at = ?")
            args.append(finished_at)
        args.append(runset_uid)
        with self._open(self._summary_path) as cx:
            cx.execute(
                f"UPDATE runsets SET {', '.join(sets)} WHERE runset_uid = ?",
                args,
            )

    # ───────────────────────── per-run writes ───────────────────────

    def record_run(self, rec: RunRecord, finished_at: str) -> None:
        """Replace one run and rebuild its runset's trace from unique runs.

        Lock both databases before reading. Ordinary write exceptions roll
        back scalars, checkpoints, raw/derived traces and the aggregate together.
        """
        run_uid = f"{rec.runset_uid}::{rec.run_index}"
        trace_fe = np.asarray(rec.trace_fe)
        trace_best = np.asarray(rec.trace_best)
        if trace_fe.ndim != 1 or trace_best.ndim != 1:
            raise ValueError('Run trace arrays must be one-dimensional.')
        trace = [(int(fe), float(value))
                 for fe, value in zip(trace_fe, trace_best, strict=True)]
        if any(not np.isfinite(value) for _, value in trace):
            raise ValueError('Run trace values must be finite.')
        with self._lock:
            with self._open(self._summary_path) as cx:
                cx.execute('ATTACH DATABASE ? AS trend_db', (str(self._trend_path),))
                cx.execute('BEGIN IMMEDIATE')
                legacy = cx.execute(
                    'SELECT 1 FROM runs r LEFT JOIN raw_trace_runs t USING(run_uid) '
                    'WHERE r.runset_uid=? AND t.run_uid IS NULL LIMIT 1',
                    (rec.runset_uid,)).fetchone()
                if legacy:
                    raise ValueError(
                        'This runset has legacy results without individual raw traces. '
                        'Start a new batch to save or retry runs; existing results remain readable.')
                cx.execute(
                    """INSERT OR REPLACE INTO runs
                       (run_uid, runset_uid, run_index, seed,
                        final_error, fe_used, finished_at, pairing_key)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (run_uid, rec.runset_uid, rec.run_index, rec.seed,
                     float(rec.final_error), int(rec.fe_used), finished_at,
                     rec.pairing_key),
                )
                cx.execute(
                    """UPDATE runsets SET runs_done =
                       (SELECT COUNT(*) FROM runs WHERE runset_uid = ?)
                       WHERE runset_uid = ?""",
                    (rec.runset_uid, rec.runset_uid),
                )
                cx.execute('UPDATE runs SET reference_scale=?, normalisation_reason=? WHERE run_uid=?',
                           (rec.reference_scale, rec.normalisation_reason, run_uid))
                cx.execute('DELETE FROM normalised_traces WHERE run_uid=?', (run_uid,))
                if rec.reference_scale is not None:
                    cx.executemany('INSERT OR REPLACE INTO normalised_traces VALUES (?,?,?)',
                                   [(run_uid, int(fe), float(value) / rec.reference_scale)
                                    for fe, value in trace])
                cx.execute('DELETE FROM performance_checkpoints WHERE run_uid=?', (run_uid,))
                if rec.performance_checkpoints:
                    cx.executemany(
                        """INSERT OR REPLACE INTO performance_checkpoints
                           (run_uid, runset_uid, checkpoint_fe, absolute_error)
                           VALUES (?, ?, ?, ?)""",
                        [
                            (run_uid, rec.runset_uid, int(fe), float(err))
                            for fe, err in rec.performance_checkpoints.items()
                        ],
                    )
                cx.execute('INSERT OR IGNORE INTO raw_trace_runs VALUES (?)', (run_uid,))
                cx.execute('DELETE FROM raw_traces WHERE run_uid=?', (run_uid,))
                cx.executemany('INSERT OR REPLACE INTO raw_traces VALUES (?,?,?)',
                               [(run_uid, fe, value) for fe, value in trace])
                self._rebuild_trace(cx, rec.runset_uid)

    @staticmethod
    def _rebuild_trace(cx: sqlite3.Connection, runset_uid: str, *, budget=None) -> None:
        """Align every run, then compute Welford moments in arrival order.

        ``stderr`` stores the sum of squared deviations;
        readers convert it to standard error. All writes use the caller's
        transaction, including deletion of checkpoints absent after replacement.
        """
        result = read_aligned_moments(cx, runset_uid, budget=budget)
        if result is None:
            return  # Preserve legacy aggregates when individual traces are absent.
        fe, mean, m2, count = result
        cx.execute('DELETE FROM trend_db.trends WHERE runset_uid=? AND metric=?',
                   (runset_uid, 'best_so_far'))
        cx.executemany('INSERT INTO trend_db.trends VALUES (?,?,?,?,?,?)',
                       [(runset_uid, 'best_so_far', int(x), float(y), float(s), count)
                        for x, y, s in zip(fe, mean, m2)])

    # ───────────────────────── reads (GUI side) ─────────────────────

    def list_runsets(self) -> list[dict[str, Any]]:
        with self._open(self._summary_path) as cx:
            cur = cx.execute(
                """SELECT runset_uid, batch_id, task_label, algorithm,
                          algorithm_params, runs_expected, runs_done,
                          status, started_at, finished_at
                   FROM runsets ORDER BY task_label, algorithm""")
            out = []
            for row in cur:
                out.append({
                    "runset_uid": row[0], "batch_id": row[1],
                    "task_label": row[2], "algorithm": row[3],
                    "algorithm_params": json.loads(row[4]),
                    "runs_expected": row[5], "runs_done": row[6],
                    "status": row[7],
                    "started_at": row[8], "finished_at": row[9],
                })
            return out

    def load_run_scalars(self, runset_uid: str) -> dict[str, list[float]]:
        """Per-run final_error and fe_used for one runset (for boxplots)."""
        with self._open(self._summary_path) as cx:
            cur = cx.execute(
                """SELECT final_error, fe_used FROM runs
                   WHERE runset_uid = ? ORDER BY run_index""",
                (runset_uid,))
            errs, fes = [], []
            for fe_err, fe_used in cur:
                errs.append(float(fe_err))
                fes.append(int(fe_used))
            return {"final_error": errs, "fe_used": fes}

    def load_trend(self, runset_uid: str, metric: str
                   ) -> dict[str, np.ndarray]:
        """Aggregated mean ± stderr trend curve for one runset/metric."""
        if metric in ('best_so_far', 'final_error'):
            with closing(self._open(self._summary_path)) as cx:
                result = read_aligned_moments(cx, runset_uid)
            if result is not None:
                fe, mean, m2, count = result
                return dict(fe=fe, mean=mean, stderr=standard_error(m2, count),
                            count=np.full(len(fe), count, dtype=int))
        with self._open(self._trend_path) as cx:
            cur = cx.execute(
                """SELECT fe, mean, stderr, count FROM trends
                   WHERE runset_uid = ? AND metric = ?
                   ORDER BY fe""",
                (runset_uid, metric))
            fe, mean, s, count = [], [], [], []
            for r in cur:
                fe.append(r[0]); mean.append(r[1])
                s.append(r[2]); count.append(r[3])
        s_arr = np.asarray(s, dtype=float)
        n_arr = np.asarray(count, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            stderr = np.where(
                n_arr > 1,
                np.sqrt(s_arr / (n_arr * (n_arr - 1))),
                0.0,
            )
        return {
            "fe": np.asarray(fe, dtype=int),
            "mean": np.asarray(mean, dtype=float),
            "stderr": stderr,
            "count": n_arr.astype(int),
        }

    # ─────────────────────── connection helpers ─────────────────────

    @staticmethod
    def _open(path: Path) -> sqlite3.Connection:
        cx = sqlite3.connect(path, timeout=30.0)
        cx.execute("PRAGMA journal_mode = WAL")
        cx.execute("PRAGMA synchronous = NORMAL")
        return cx


# ─────────────────── trace downsampling utility ────────────────────

def downsample_best_so_far(fe_count: Iterable[int],
                           best_f: Iterable[float],
                           target_buckets: int = 200
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Pick ~``target_buckets`` (FE, best_so_far) checkpoints from a tape.

    PORTAL's :class:`RecordingProblem` records one Frame per evaluation,
    so a 100k-FE run yields a 100k-long tape. We pick log-spaced FE
    checkpoints and keep the *best* (minimum) value seen up to each
    checkpoint, which is the canonical convergence-curve summary.
    """
    fe_arr = np.asarray(list(fe_count), dtype=int)
    bf_arr = np.asarray(list(best_f), dtype=float)
    if fe_arr.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    # Use the running minimum to represent best-so-far convergence.
    running_min = np.minimum.accumulate(bf_arr)
    fe_max = int(fe_arr[-1])
    if fe_max <= 0 or fe_arr.size <= target_buckets:
        return fe_arr, running_min
    # Log-spaced checkpoints between FE=1 and fe_max.
    checkpoints = np.unique(
        np.round(np.geomspace(1, fe_max, target_buckets)).astype(int))
    # For each checkpoint, find the rightmost tape index where fe <= cp.
    idx = np.searchsorted(fe_arr, checkpoints, side="right") - 1
    idx = np.clip(idx, 0, fe_arr.size - 1)
    return fe_arr[idx], running_min[idx]
