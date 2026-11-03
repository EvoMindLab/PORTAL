import csv
import gc
import io
import json
import os
import shutil
import sqlite3
import zipfile
from dataclasses import asdict

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from portal_studio.experiment_runner.finished_archive import export_finished_tasks, import_finished_tasks
from portal_studio.experiment_runner.store import ResultStore, RunRecord
from portal_studio.experiment_runner.analysis.stats import SelectionItem, _gather_matrix
from portal_studio.experiment_runner.analysis.plotting import load_trend_series_from_selection
from portal_studio.gui.experimental.task_model import Task, TaskRegistry
from portal_studio.gui.experimental.completed_tasks import CompletedTasksPanel
from portal_studio.gui.analysis.statistical_analysis import SaPanel, sa_group_key


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _task(root, label, algorithm="PSO", *, problem="problem", status="completed"):
    store = ResultStore(root)
    store.ensure_schema()
    uid = f"{label}::{algorithm}"
    store.declare_runset(runset_uid=uid, batch_id="batch", task_label=label,
                        algorithm=algorithm, algorithm_params={"pop_size": 10}, runs_expected=2)
    for i, error in enumerate([2., 4.]):
        store.record_run(RunRecord(
            runset_uid=uid, run_index=i, seed=42+i, final_error=error, fe_used=500,
            trace_fe=np.array([1, 100, 500]), trace_best=np.array([10., 6., error]),
            pairing_key=f"{problem}-trial-{i}", reference_scale=2.,
            performance_checkpoints={100: 6.},
        ), finished_at="test")
    store.set_runset_status(uid, "finished" if status == "completed" else status)
    return Task(label=label, algorithm=algorithm, algorithm_params={"pop_size": 10}, runs=2,
                fe_budget=500, seed_strategy="fixed", master_seed=42,
                composite_label=problem, composite_source_kind="file", composite_path="missing-problem.json",
                status=status, output_dir=str(root), runset_uid=uid, runs_done=2)


def _selection(tasks):
    return [SelectionItem(t.output_dir, t.runset_uid, t.label, t.algorithm, t.label) for t in tasks]


def test_results_survive_without_original_files_and_keep_sa_pairing(tmp_path):
    source = tmp_path / "original"
    tasks = [_task(source, "a"), _task(source, "b", "DE")]
    _task(source, "unselected")
    before_keys = [sa_group_key(t) for t in tasks]
    before, _ = _gather_matrix(_selection(tasks), metric="normalised_error")
    before_trend = load_trend_series_from_selection(_selection(tasks), metric="final_error")
    archive = export_finished_tasks([asdict(t) for t in tasks], tmp_path / "results.zip")
    with zipfile.ZipFile(archive) as z:
        errors = list(csv.DictReader(io.StringIO(z.read("run_errors.csv").decode("utf-8-sig"))))
        traces = list(csv.DictReader(io.StringIO(z.read("convergence.csv").decode("utf-8-sig"))))
    assert len(errors) == 4 and len(traces) == 12
    assert {row["task"] for row in errors} == {"a", "b"}
    assert [float(row["final_error"]) for row in errors] == [2., 4., 2., 4.]
    gc.collect()  # Existing ResultStore connections close when collected.
    shutil.rmtree(source)
    restored = [Task(**r) for r in import_finished_tasks(archive, tmp_path / "imports")]
    assert [sa_group_key(t) for t in restored] == before_keys
    after, _ = _gather_matrix(_selection(restored), metric="normalised_error")
    np.testing.assert_array_equal(after, before)
    after_trend = load_trend_series_from_selection(_selection(restored), metric="final_error")
    for a, b in zip(before_trend, after_trend):
        np.testing.assert_array_equal(a.fe, b.fe)
        np.testing.assert_array_equal(a.mean, b.mean)
        np.testing.assert_array_equal(a.stderr, b.stderr)
    with sqlite3.connect(restored[0].output_dir + "/summary.sqlite") as cx:
        assert cx.execute("SELECT COUNT(*) FROM runsets").fetchone()[0] == 2
        assert cx.execute("SELECT COUNT(*) FROM performance_checkpoints").fetchone()[0] == 4


def test_different_sources_and_settings_remain_distinct(tmp_path):
    tasks = [_task(tmp_path / "one", "same"),
             _task(tmp_path / "two", "same", problem="different")]
    tasks[1].master_seed = 123
    path = export_finished_tasks([asdict(t) for t in tasks], tmp_path / "results.zip")
    restored = [Task(**r) for r in import_finished_tasks(path, tmp_path / "imports")]
    assert restored[0].output_dir != restored[1].output_dir
    assert restored[1].master_seed == 123
    assert sa_group_key(restored[0]) != sa_group_key(restored[1])
    with pytest.raises(ValueError, match="different problem"):
        _gather_matrix(_selection(restored))


def test_failed_without_results_restores_as_failed(app, tmp_path):
    task = Task(label="failed", algorithm="unavailable", runs=2, status="failed", message="No instance")
    path = export_finished_tasks([asdict(task)], tmp_path / "failed.zip")
    record, = import_finished_tasks(path, tmp_path / "imports")
    registry = TaskRegistry()
    panel = SaPanel(registry=registry)
    registry.add(Task(**record))
    assert registry.completed()[0].status == "failed"
    assert not registry.pending()
    assert panel.model.rowCount() == 0
    panel.close()


def test_rejects_missing_success_data_and_incomplete_results(tmp_path):
    task = Task(label="missing", algorithm="PSO", status="completed", runs_done=2, runs=2)
    with pytest.raises(ValueError, match="missing its result"):
        export_finished_tasks([asdict(task)], tmp_path / "bad.zip")
    task = _task(tmp_path / "source", "task")
    task.runs_done = 1
    with pytest.raises(ValueError, match="run counts"):
        export_finished_tasks([asdict(task)], tmp_path / "bad.zip")
    assert not (tmp_path / "bad.zip").exists()


def test_invalid_archive_does_not_restore_partial_results(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": "portal-finished-results", "version": 1,
            "tasks": [{"task": {}, "bundle": "../../escape"}]}))
    with pytest.raises(ValueError, match="bundle path"):
        import_finished_tasks(path, tmp_path / "imports")
    assert list((tmp_path / "imports").iterdir()) == []


def test_gui_import_restores_finished_and_sa_without_pending(app, tmp_path, monkeypatch):
    task = _task(tmp_path / "source", "task")
    archive = export_finished_tasks([asdict(task)], tmp_path / "results.zip")
    gc.collect()
    shutil.rmtree(tmp_path / "source")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(archive), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    registry = TaskRegistry()
    panel = CompletedTasksPanel(registry=registry, batch_service=None)
    sa = SaPanel(registry=registry)
    panel._on_import_results()
    assert not registry.pending()
    assert len(registry.completed()) == 1
    assert panel.model.rowCount() == 1
    assert sa.model.rowCount() == 1
    restored, = registry.completed()
    assert restored.metrics["final_error_mean"] == 3.
    exported = tmp_path / "gui-export.zip"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(exported), ""))
    panel.model.toggle_selection(restored.task_id, True)
    panel._on_export_results()
    assert len(import_finished_tasks(exported, tmp_path / "check")) == 1
    panel._on_import_results()
    assert len(registry.completed()) == 2
    assert len({t.task_id for t in registry.completed()}) == 2
    panel.close()
    sa.close()


def test_partial_failure_keeps_successful_errors_without_becoming_sa_ready(app, tmp_path):
    task = _task(tmp_path / "source", "partial", status="failed")
    task.runs = 3
    task.runs_failed = 1
    task.message = "Run 3 failed"
    with sqlite3.connect(tmp_path / "source" / "summary.sqlite") as cx:
        cx.execute("UPDATE runsets SET runs_expected=3")
    path = export_finished_tasks([asdict(task)], tmp_path / "partial.zip")
    restored, = [Task(**r) for r in import_finished_tasks(path, tmp_path / "imports")]
    assert restored.status == "failed" and restored.runs_failed == 1
    assert restored.message == task.message
    assert restored.runs_done == 2
    with pytest.raises(ValueError, match="complete successful"):
        _gather_matrix(_selection([restored]))


def test_archive_preserves_raw_and_normalised_traces_and_settings(tmp_path):
    task = _task(tmp_path / "source", "custom")
    task.seed_strategy = "random"
    task.master_seed = 789
    task.realization_type = "type2"
    task.metric_outputs = ["final_error"]
    path = export_finished_tasks([asdict(task)], tmp_path / "custom.zip")
    restored, = [Task(**r) for r in import_finished_tasks(path, tmp_path / "imports")]
    for name in ("raw_traces", "normalised_traces", "raw_trace_runs", "performance_checkpoints", "runs"):
        from contextlib import closing
        with closing(sqlite3.connect(task.output_dir + "/summary.sqlite")) as a, \
                closing(sqlite3.connect(restored.output_dir + "/summary.sqlite")) as b:
            assert a.execute(f"SELECT * FROM {name}").fetchall() == b.execute(f"SELECT * FROM {name}").fetchall()
    for key in ("algorithm_params", "runs", "fe_budget", "seed_strategy", "master_seed",
                "realization_type", "metric_outputs", "composite_label", "composite_source_kind", "composite_path"):
        assert getattr(restored, key) == getattr(task, key)
