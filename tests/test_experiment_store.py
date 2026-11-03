import sqlite3
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from portal_studio.experiment_runner.store import ResultStore, RunRecord


def _record(runset_uid="case::PSO", run_index=0, performance_checkpoints=None):
    return RunRecord(
        runset_uid=runset_uid,
        run_index=run_index,
        seed=100 + run_index,
        final_error=0.01,
        fe_used=300000,
        trace_fe=np.asarray([300000], dtype=int),
        trace_best=np.asarray([0.01], dtype=float),
        performance_checkpoints=performance_checkpoints or {},
    )


def test_ensure_schema_creates_performance_checkpoints_table(tmp_path):
    store = ResultStore(tmp_path)
    store.ensure_schema()

    with sqlite3.connect(tmp_path / "summary.sqlite") as cx:
        row = cx.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='performance_checkpoints'"
        ).fetchone()

    assert row is not None


def test_record_run_persists_performance_checkpoints(tmp_path):
    store = ResultStore(tmp_path)
    store.ensure_schema()
    rec = _record(performance_checkpoints={30000: 5.0, 50000: 3.0})

    store.record_run(rec, finished_at="2026-07-07T00:00:00+00:00")

    with sqlite3.connect(tmp_path / "summary.sqlite") as cx:
        rows = cx.execute(
            "SELECT checkpoint_fe, absolute_error FROM performance_checkpoints "
            "WHERE runset_uid = ? ORDER BY checkpoint_fe",
            ("case::PSO",),
        ).fetchall()

    assert rows == [(30000, 5.0), (50000, 3.0)]


def test_record_run_with_no_performance_checkpoints_writes_nothing(tmp_path):
    store = ResultStore(tmp_path)
    store.ensure_schema()
    rec = _record(performance_checkpoints={})

    store.record_run(rec, finished_at="2026-07-07T00:00:00+00:00")

    with sqlite3.connect(tmp_path / "summary.sqlite") as cx:
        count = cx.execute("SELECT COUNT(*) FROM performance_checkpoints").fetchone()[0]

    assert count == 0


def test_record_run_still_writes_final_error(tmp_path):
    store = ResultStore(tmp_path)
    store.ensure_schema()
    rec = _record()

    store.record_run(rec, finished_at="2026-07-07T00:00:00+00:00")

    scalars = store.load_run_scalars("case::PSO")
    assert scalars["final_error"] == [0.01]
    assert scalars["fe_used"] == [300000]


def _write_parallel_record(output_dir, run_index):
    rec = _record(run_index=run_index)
    rec.final_error = float(run_index)
    rec.trace_fe = np.arange(1, 201, dtype=int)
    rec.trace_best = np.full(200, float(run_index))
    ResultStore(output_dir).record_run(rec, finished_at="test")


def test_parallel_trace_aggregation_keeps_every_sample(tmp_path):
    store = ResultStore(tmp_path)
    store.ensure_schema()
    count = 16
    store.declare_runset(runset_uid="case::PSO", batch_id="test", task_label="case",
                         algorithm="PSO", algorithm_params={}, runs_expected=count)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_write_parallel_record, str(tmp_path), i) for i in range(count)]
        for future in futures:
            future.result(timeout=30)
    with sqlite3.connect(tmp_path / "summary.sqlite") as cx:
        assert cx.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == count
        assert cx.execute("SELECT runs_done FROM runsets").fetchone()[0] == count
    with sqlite3.connect(tmp_path / "trend.sqlite") as cx:
        rows = cx.execute("SELECT mean, stderr, count FROM trends").fetchall()
    values = np.arange(count, dtype=float)
    # The final FE used is beyond this fixture's sparse trace, so its best
    # value is also carried to that endpoint rather than ending at FE 200.
    assert len(rows) == 201
    with sqlite3.connect(tmp_path / "trend.sqlite") as cx:
        assert cx.execute("SELECT MAX(fe) FROM trends").fetchone()[0] == _record().fe_used
    for mean, squared_deviations, samples in rows:
        assert samples == count
        assert mean == pytest.approx(values.mean())
        assert squared_deviations == pytest.approx(np.sum((values - values.mean()) ** 2))
