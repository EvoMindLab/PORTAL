"""Run failures are visible, isolated, and excluded from successful results."""
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
import json
import os
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtTest import QTest

from portal_studio.algorithms.runner import RunCanceled
from portal_studio.experiment_runner.config import BatchConfig, TaskConfig
from portal_studio.experiment_runner import scheduler as scheduler_module
from portal_studio.experiment_runner.scheduler import BatchProgress, BatchScheduler, RunFailure
from portal_studio.experiment_runner.store import ResultStore, RunRecord
from portal_studio.gui.educational import tab as education_module
from portal_studio.gui.experimental.batch_service import BatchService
from portal_studio.gui.experimental.pending_tasks import PendingTasksPanel
from portal_studio.gui.experimental.task_model import Task, TaskRegistry


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _mixed_execute(unit, composite, output_dir):
    """Picklable worker: fail the middle attempt after a successful first run."""
    if unit.run_index == 1:
        raise ValueError("injected optimizer error")
    ResultStore(output_dir).record_run(RunRecord(
        runset_uid=unit.runset_uid, run_index=unit.run_index, seed=unit.seed,
        final_error=0.5, fe_used=unit.fe_budget,
        trace_fe=np.array([1, unit.fe_budget]), trace_best=np.array([1.0, 0.5]),
    ), finished_at="test")
    return {"runset_uid": unit.runset_uid, "run_index": unit.run_index}


@pytest.mark.parametrize("workers", [1, 2])
def test_batch_continues_after_error_and_preserves_only_successes(
    tmp_path, monkeypatch, workers,
):
    monkeypatch.setattr(scheduler_module, "execute_run", _mixed_execute)
    monkeypatch.setattr(scheduler_module.os, "cpu_count", lambda: 4)
    cfg = BatchConfig(tasks=[TaskConfig(
        label="trial", algorithm="ABC", runs=3,
        seed_strategy="fixed", master_seed=42,
    )], max_parallel_runs=workers, fe_budget=20)
    updates, successes = [], []
    final = BatchScheduler(cfg, {}, str(tmp_path)).run(
        progress=updates.append, unit_finished=successes.append)
    assert final.status == "finished"
    assert final.completed_runs == 2
    assert final.failed_runs == 1
    assert sorted(item["run_index"] for item in successes) == [0, 2]
    assert "ABC, run 2 (seed 43): ValueError: injected" in final.failures[0].message
    assert any("injected optimizer error" in item.message for item in updates)
    assert updates[0].failures == ()  # Earlier snapshots are not mutated.
    with sqlite3.connect(tmp_path / "summary.sqlite") as cx:
        assert cx.execute("SELECT run_index FROM runs ORDER BY run_index").fetchall() == [(0,), (2,)]
        failures = json.loads(cx.execute(
            "SELECT value FROM meta WHERE key = 'run_failures'").fetchone()[0])
        assert failures[0]["run_index"] == 1
        assert failures[0]["seed"] == 43
    with sqlite3.connect(tmp_path / "trend.sqlite") as cx:
        assert cx.execute("SELECT DISTINCT count FROM trends").fetchall() == [(2,)]


@pytest.mark.parametrize("error", [ValueError("all failed"), ModuleNotFoundError("niapy")])
def test_all_failed_runs_are_not_marked_as_success(tmp_path, monkeypatch, error):
    execute = Mock(side_effect=error)
    monkeypatch.setattr(scheduler_module, "execute_run", execute)
    cfg = BatchConfig(tasks=[TaskConfig(label="all", algorithm="ABC", runs=3)],
                      max_parallel_runs=1)
    final = BatchScheduler(cfg, {}, str(tmp_path)).run()
    assert execute.call_count == 3
    assert final.completed_runs == 0
    assert final.failed_runs == 3
    assert "0/3 runs succeeded" in final.message


def test_batch_cancellation_is_not_reported_as_an_algorithm_error(tmp_path, monkeypatch):
    execute = Mock(side_effect=RunCanceled("canceled"))
    monkeypatch.setattr(scheduler_module, "execute_run", execute)
    cfg = BatchConfig(tasks=[TaskConfig(label="cancel", algorithm="ABC", runs=3)],
                      max_parallel_runs=1)
    final = BatchScheduler(cfg, {}, str(tmp_path)).run()
    assert execute.call_count == 1
    assert final.status == "canceled"
    assert final.completed_runs == final.failed_runs == 0


def test_broken_pool_recovery_does_not_retry_known_run_errors(tmp_path, monkeypatch):
    class BrokenPool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, fn, unit, *args):
            future = Future()
            future.set_exception(ValueError("known run error") if unit.run_index == 0
                                 else BrokenProcessPool("worker exited"))
            return future

    monkeypatch.setattr(scheduler_module, "ProcessPoolExecutor", BrokenPool)
    monkeypatch.setattr(scheduler_module, "as_completed", lambda futures: iter(futures))
    monkeypatch.setattr(scheduler_module.os, "cpu_count", lambda: 4)
    execute = Mock(side_effect=lambda unit, *_args: {"run_index": unit.run_index})
    monkeypatch.setattr(scheduler_module, "execute_run", execute)
    cfg = BatchConfig(tasks=[TaskConfig(label="pool", algorithm="ABC", runs=3)],
                      max_parallel_runs=2)
    final = BatchScheduler(cfg, {}, str(tmp_path)).run()
    assert [call.args[0].run_index for call in execute.call_args_list] == [1, 2]
    assert (final.completed_runs, final.failed_runs) == (2, 1)


def test_batch_gui_shows_errors_without_failing_unrelated_tasks(app, monkeypatch):
    warning = Mock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    registry, service = TaskRegistry(), BatchService()
    panel = PendingTasksPanel(registry=registry, batch_service=service)
    good = Task(label="good", algorithm="PSO", runs=2, runs_done=2,
                status="running", runset_uid="good::PSO")
    bad = Task(label="bad", algorithm="ABC", runs=2, runs_done=1,
               status="running", runset_uid="bad::ABC")
    for task in (good, bad):
        registry.add(task)
    panel._batch_to_tasks["batch"] = [good.task_id, bad.task_id]
    failure = RunFailure("bad::ABC", "bad", "ABC", 1, 43, "ValueError: failed")
    snapshot = BatchProgress("batch", "running", 4, 3, failures=(failure,))
    service.batch_progress.emit(snapshot)
    assert bad.status == "running" and bad.runs_failed == 1
    index = panel.model.index(1, 3)
    assert "1 failed" in panel.model.data(index)
    assert "run 2" in panel.model.data(index, Qt.ItemDataRole.ToolTipRole)
    service.batch_finished.emit(BatchProgress("batch", "finished", 4, 3, failures=(failure,)))
    assert good.status == "completed"
    assert bad.status == "failed"
    assert "1/2 runs succeeded; 1 failed" in bad.message
    warning.assert_called_once()
    assert "Successful results are preserved" in warning.call_args.args[2]
    panel.close()


def test_batch_service_thread_delivers_failure_summary_and_cleans_up(app, tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler_module, "execute_run", _mixed_execute)
    monkeypatch.setattr(scheduler_module.os, "cpu_count", lambda: 4)
    warning = Mock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    registry, service = TaskRegistry(), BatchService()
    panel = PendingTasksPanel(registry=registry, batch_service=service)
    task = Task(label="thread", algorithm="ABC", runs=3,
                status="running", runset_uid="thread::ABC")
    registry.add(task)
    panel._runset_to_task[task.runset_uid] = task.task_id
    cfg = BatchConfig(tasks=[TaskConfig(label=task.label, algorithm=task.algorithm, runs=3)],
                      max_parallel_runs=2, fe_budget=20)
    finals = []
    service.batch_finished.connect(finals.append)
    submission = service.submit(cfg, {}, str(tmp_path), label="thread test")
    panel._batch_to_tasks[submission.batch_id] = [task.task_id]
    try:
        deadline = time.monotonic() + 30
        while service._active and time.monotonic() < deadline:
            QTest.qWait(10)
        assert not service._active, "Batch thread did not clean up"
        assert len(finals) == 1
        assert task.status == "failed"
        assert (task.runs_done, task.runs_failed) == (2, 1)
        warning.assert_called_once()
    finally:
        service.shutdown()
        panel.close()


def _mixed_education_job(request):
    if request["request_index"] == 0:
        raise ValueError("education optimizer error")
    return {"request_index": request["request_index"], "setup_label": "Setup B", "tape": [1]}


def test_education_process_failure_preserves_successful_peer(monkeypatch):
    monkeypatch.setattr(education_module, "_run_education_solver_job", _mixed_education_job)
    requests = [dict(request_index=i, setup_label=f"Setup {i}", solver_name="ABC", seed=42)
                for i in range(2)]
    progress = Mock()
    results = education_module._run_education_requests_in_processes(
        requests, max_workers=2, progress=progress)
    assert len(results) == 2
    assert "ABC (seed 42): ValueError: education optimizer error" == results[0]["error"]
    assert "tape" not in results[0]
    assert results[1]["tape"] == [1]
    assert progress.call_count == 2
    assert progress.call_args.args[:2] == (2, 2)


@pytest.mark.parametrize("all_failed", [False, True])
def test_education_gui_clears_failed_tape_but_keeps_success(app, monkeypatch, all_failed):
    warning = Mock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    sides = [Mock(), Mock()]
    sides[1].max_fe.return_value = 20
    # Exercise the real slot without rendering an unrelated landscape.
    tab = SimpleNamespace(
        _close_run_dialog=Mock(), run_btn=Mock(), _run_canceled=False,
        _run_runnable=sides, frame_slider=Mock(), _draw_all_at_fe=Mock(),
        _refresh_fe_label=Mock(), on_play_clicked=Mock(),
    )
    results = [{"request_index": 0, "setup_label": "Setup A", "error": "ABC: failed"},
               {"request_index": 1, "setup_label": "Setup B", "tape": [1]}]
    if all_failed:
        results[1] = {"request_index": 1, "error": "ABC: also failed"}
    education_module.EducationTab._on_background_run_finished(tab, results)
    tab.run_btn.setEnabled.assert_called_once_with(True)
    sides[0].clear_run.assert_called_once()
    sides[0].apply_run_result.assert_not_called()
    if all_failed:
        sides[1].clear_run.assert_called_once()
        tab.on_play_clicked.assert_not_called()
    else:
        sides[1].apply_run_result.assert_called_once_with(results[1])
        tab.on_play_clicked.assert_called_once()
    warning.assert_called_once()
