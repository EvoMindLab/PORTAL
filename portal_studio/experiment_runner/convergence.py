"""Align saved best-so-far traces without dropping runs at sparse checkpoints."""
from __future__ import annotations

import json
import numpy as np


def checkpoint_grid(budget: int, buckets: int = 200) -> np.ndarray:
    """A recording grid determined by the requested budget, never stopping time."""
    if budget < 1:
        raise ValueError('The evaluation budget must be positive.')
    return np.unique(np.round(np.geomspace(1, budget, buckets)).astype(int))


def aligned_moments(traces, *, end=None):
    """Union grid, last observation carried forward, constant run count.

    Start only when every run has an observation. No interpolation, future
    observations, or invented values before a run's first point are used.
    Returns FE, mean, sum of squared deviations (M2), and sample count.
    """
    empty = (np.array([], dtype=int), np.array([]), np.array([]), 0)
    if not traces or any(not len(trace) for trace in traces):
        return empty
    arrays = []
    for trace in traces:
        points = np.asarray(trace, dtype=float)
        fe, values = points[:, 0], points[:, 1]
        if (np.any(~np.isfinite(points)) or np.any(fe < 1)
                or np.any(fe != np.floor(fe)) or np.any(np.diff(fe) <= 0)
                or np.any(values < 0)):
            raise ValueError('Invalid individual convergence trace.')
        arrays.append((fe.astype(int), values))
    first = max(fe[0] for fe, _ in arrays)
    last = max(fe[-1] for fe, _ in arrays) if end is None else int(end)
    if last < first:
        return empty
    grid = np.unique(np.concatenate([fe for fe, _ in arrays] + [np.array([last])]))
    grid = grid[(grid >= first) & (grid <= last)]
    mean = np.zeros(len(grid))
    m2 = np.zeros(len(grid))
    for n, (fe, values) in enumerate(arrays, 1):
        values = values[np.searchsorted(fe, grid, side='right') - 1]
        delta = values - mean
        mean += delta / n
        m2 += delta * (values - mean)
    return grid, mean, m2, len(arrays)


def read_aligned_moments(cx, uid, *, normalised=False, budget=None):
    """Read individual traces; None identifies unreconstructable legacy data."""
    tables = {row[0] for row in cx.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    table = 'normalised_traces' if normalised else 'raw_traces'
    if table not in tables:
        return None
    if not normalised:
        if 'raw_trace_runs' not in tables:
            return None
        missing = cx.execute(
            'SELECT COUNT(*) FROM runs r LEFT JOIN raw_trace_runs j USING(run_uid) '
            'WHERE r.runset_uid=? AND j.run_uid IS NULL', (uid,)).fetchone()[0]
        total = cx.execute('SELECT COUNT(*) FROM runs WHERE runset_uid=?', (uid,)).fetchone()[0]
        if missing == total and total:
            return None
        if missing:
            raise ValueError('Individual convergence traces are missing for some runs.')
    runs = cx.execute('SELECT run_uid,fe_used FROM runs WHERE runset_uid=? ORDER BY run_index',
                      (uid,)).fetchall()
    if 'raw_trace_runs' in tables:
        order = dict(cx.execute('SELECT run_uid,rowid FROM raw_trace_runs'))
        runs.sort(key=lambda row: order.get(row[0], 0))
    points = {row[0]: [] for row in runs}
    for run, fe, value in cx.execute(
            f'SELECT t.run_uid,t.fe,t.value FROM {table} t JOIN runs r USING(run_uid) '
            'WHERE r.runset_uid=? ORDER BY t.fe', (uid,)):
        points[run].append((fe, value))
    if budget is None and 'meta' in tables:
        row = cx.execute("SELECT value FROM meta WHERE key='fe_budget'").fetchone()
        if row:
            budget = int(json.loads(row[0]))
    if budget is None:
        budget = max([r[1] or 0 for r in runs] +
                     [p[-1][0] for p in points.values() if p], default=0)
    return aligned_moments([points[row[0]] for row in runs], end=budget)


def standard_error(m2, count):
    return np.sqrt(np.maximum(m2, 0.) / (count * (count - 1))) if count > 1 else np.zeros_like(m2)
