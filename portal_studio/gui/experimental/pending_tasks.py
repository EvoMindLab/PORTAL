"""Column 2 — Pending Tasks.

Authored tasks live here until they're run. Five-column table
(Select / Task ID / Task Info / Status / Delete) on top, parallel-pool
config + Run Selected Tasks at the bottom. Tasks pop out of this panel when
their status flips to
``completed`` / ``failed`` / ``canceled`` (the Finished Tasks panel
picks them up there).
"""
from __future__ import annotations

import datetime
import os
from collections import defaultdict
from pathlib import Path

from PySide6 import QtCore
from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, Qt, Signal,
)
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QPushButton, QSpinBox, QTableView,
    QVBoxLayout, QWidget,
)

from portal_studio.experiment_runner.config import BatchConfig, TaskConfig
from portal_studio.gui.experimental.batch_service import BatchService
from portal_studio.gui.experimental.task_model import Task, TaskRegistry
from portal_studio.gui.common.help import help_body, make_help_button, show_help_dialog, term
from portal_studio.gui.common.icons import make_icon_button, make_x_icon


class _PendingTasksModel(QAbstractTableModel):
    """Custom model with a Select checkbox in column 0."""

    HEADERS = ["Select", "Task ID", "Task Info", "Status", "Delete"]

    def __init__(self, registry: TaskRegistry,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._registry = registry
        self._tasks: list[Task] = []
        self._selected: set[str] = set()
        registry.pending_changed.connect(self._refresh)
        self._refresh()

    # ───────────────────────── public ──────────────────────────────

    def selected_ids(self) -> list[str]:
        return [tid for tid in self._selected if any(
            t.task_id == tid for t in self._tasks)]

    def task_at(self, row: int) -> Task | None:
        if 0 <= row < len(self._tasks):
            return self._tasks[row]
        return None

    def select_all(self, checked: bool) -> None:
        if checked:
            self._selected = {t.task_id for t in self._tasks
                              if t.status == "pending"}
        else:
            self._selected.clear()
        if self._tasks:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self._tasks) - 1, 0))

    def toggle_selection(self, task_id: str, checked: bool) -> None:
        if checked:
            self._selected.add(task_id)
        else:
            self._selected.discard(task_id)

    # ───────────────────────── Qt overrides ────────────────────────

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._tasks)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole):
        if (role == Qt.ItemDataRole.DisplayRole
                and orientation == Qt.Orientation.Horizontal):
            return self.HEADERS[section]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def data(self, index: QModelIndex,
             role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        task = self._tasks[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return [
                "",          # Select rendered via setIndexWidget
                task.task_id,
                task.info(),
                _pending_status_text(task),
                "",          # Delete rendered via setIndexWidget
            ][col]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        if role == Qt.ItemDataRole.ForegroundRole and col == 3:
            return QBrush(QColor(_status_color(task.status)))
        if role == Qt.ItemDataRole.ToolTipRole and task.message:
            return task.message
        return None

    # ───────────────────────── refresh ─────────────────────────────

    def _refresh(self) -> None:
        # Show pending + running rows. Terminal-status tasks have moved to
        # the Completed column.
        self.beginResetModel()
        self._tasks = [
            t for t in self._registry.all_tasks()
            if t.status in ("pending", "running")
        ]
        live_ids = {t.task_id for t in self._tasks}
        self._selected &= live_ids
        self.endResetModel()


def _status_text(status: str) -> str:
    return {"pending": "New", "running": "Running",
            "completed": "Done", "failed": "Failed",
            "canceled": "Canceled"}.get(status, status)


def _pending_status_text(task: Task) -> str:
    if task.status == "running":
        text = f"Running: {task.runs_done}/{task.runs} succeeded"
        if task.runs_failed:
            text += f"; {task.runs_failed} failed"
        return text
    return _status_text(task.status)


def _status_color(status: str) -> str:
    return {"pending": "#6b7280", "running": "#2563eb",
            "completed": "#16a34a", "failed": "#b91c1c",
            "canceled": "#9a3412"}.get(status, "#374151")


class PendingTasksPanel(QFrame):
    """Column 2. Owns the Pending table + Run / Delete controls."""

    def __init__(self, *, registry: TaskRegistry,
                 batch_service: BatchService,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("projectPanel")
        self._registry = registry
        self._batch_service = batch_service
        # Track which task_id each runset_uid maps to, so unit_finished
        # signals can be routed back to the right task.
        self._runset_to_task: dict[str, str] = {}
        # Track which batch contains which task_ids (for terminal events).
        self._batch_to_tasks: dict[str, list[str]] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        self.title_label = QLabel("Pending Tasks")
        self.title_label.setObjectName("panelTitle")
        title_row.addWidget(self.title_label)
        self.help_btn = make_help_button(
            "Explain pending tasks and the worker pool",
            self._show_pending_tasks_help,
        )
        title_row.addWidget(self.help_btn)
        title_row.addStretch(1)
        outer.addLayout(title_row)

        self.select_all_cb = QCheckBox("Select All")
        self.select_all_cb.toggled.connect(self._on_select_all)
        outer.addWidget(self.select_all_cb)

        self.model = _PendingTasksModel(registry, self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        outer.addWidget(self.table, stretch=1)

        # Pool configuration.
        pool_row = QHBoxLayout()
        pool_row.setSpacing(8)
        pool_row.addWidget(QLabel("Parallel Pool Type:"))
        self.pool_type_combo = QComboBox()
        self.pool_type_combo.addItem("Process Pool", userData="process")
        self.pool_type_combo.addItem("Serial (1 worker)", userData="serial")
        self.pool_type_combo.currentIndexChanged.connect(
            self._sync_worker_visibility)
        pool_row.addWidget(self.pool_type_combo)
        self.workers_label = QLabel("Workers:")
        pool_row.addWidget(self.workers_label)
        self.workers_spin = QSpinBox()
        cores = max(1, (os.cpu_count() or 2) - 1)
        self.workers_spin.setRange(1, max(64, cores))
        self.workers_spin.setValue(cores)
        pool_row.addWidget(self.workers_spin)
        pool_row.addStretch(1)
        outer.addLayout(pool_row)
        self._sync_worker_visibility()

        self.run_btn = QPushButton("▶ Run Selected Tasks")
        self.run_btn.setProperty("variant", "primary")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._on_run_clicked)
        outer.addWidget(self.run_btn)
        registry.pending_changed.connect(self._sync_run_button)
        self._sync_run_button()

        self.delete_btn = None

        # Wire BatchService.
        batch_service.unit_finished.connect(self._on_unit_finished)
        batch_service.batch_progress.connect(self._on_batch_progress)
        batch_service.batch_finished.connect(self._on_batch_terminal)
        batch_service.batch_failed.connect(self._on_batch_failed)

        self.model.modelReset.connect(self._install_row_checkboxes)
        self.model.dataChanged.connect(self._install_row_checkboxes)
        self._install_row_checkboxes()

    # ───────────────────────── slots ───────────────────────────────

    def _on_select_all(self, checked: bool) -> None:
        self.model.select_all(checked)
        self._install_row_checkboxes()

    def _sync_worker_visibility(self) -> None:
        is_process_pool = self.pool_type_combo.currentData() == "process"
        self.workers_label.setVisible(is_process_pool)
        self.workers_spin.setVisible(is_process_pool)

    def _sync_run_button(self) -> None:
        self.run_btn.setEnabled(bool(self._registry.pending()))

    def _show_pending_tasks_help(self) -> None:
        body = help_body("Pending tasks", [
            f"{term('Pending Tasks')} lists the task configurations waiting "
            f"to run. Select individual rows, or use {term('Select All')} to "
            "select every pending task.",
            f"{term('Run Selected Tasks')} starts the selected tasks. Each "
            "task is expanded into its requested algorithm runs, and the "
            "Status column shows how many runs have finished. When a task "
            "stops, it moves to Finished Tasks.",
            f"{term('Parallel Pool Type')} controls how those runs are "
            f"executed. {term('Serial (1 worker)')} processes one algorithm "
            f"run at a time. {term('Process Pool')} can process several "
            "independent runs at the same time.",
            f"{term('Workers')} appears for Process Pool and sets the maximum "
            "number of runs that can execute at the same time. More workers "
            "may reduce the total runtime, but they also use more processor "
            "time and memory.",
            "The row delete icon removes the task from the list. It is "
            "intended for a task that has not started; deleting a row does "
            "not delete saved result files.",
        ])
        show_help_dialog(self, "Pending tasks help", body)

    def _on_run_clicked(self) -> None:
        selected = self.model.selected_ids()
        if not selected:
            QMessageBox.information(
                self, "Run Selected Tasks",
                "Tick at least one task before clicking Run.")
            return
        tasks = [self._registry.by_id(tid) for tid in selected]
        tasks = [t for t in tasks if t is not None and t.status == "pending"]
        if not tasks:
            return

        # One BatchConfig has one source, FE budget and metric selection.
        groups: dict[tuple[str, str, int, tuple[str, ...]], list[Task]] = defaultdict(list)
        for t in tasks:
            key = (t.composite_label, t.composite_path or "<generator>",
                   int(t.fe_budget), tuple(t.metric_outputs))
            groups[key].append(t)

        composites, resolve_errors = self._resolve_composites(tasks)
        for group_key, grp in groups.items():
            source_key = group_key[:2]
            composite_label, _path = source_key
            composite = composites.get(source_key)
            if composite is None:
                message = resolve_errors.get(
                    source_key,
                    f"Instance '{composite_label}' is no longer available.",
                )
                for t in grp:
                    self._registry.set_status(
                        t.task_id, "failed", message=message
                    )
                QMessageBox.critical(self, "Run failed", message)
                continue
            cfg = self._build_batch_config(grp)
            output_dir = self._autoname_output_dir()
            cfg.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
            for t in grp:
                t.runset_uid = f"{t.label}::{t.algorithm}"
                t.output_dir = output_dir
                self._runset_to_task[t.runset_uid] = t.task_id
                self._registry.set_status(t.task_id, "running")
            submission = self._batch_service.submit(
                cfg, composite, output_dir,
                label=Path(output_dir).name)
            self._batch_to_tasks[submission.batch_id] = [t.task_id for t in grp]

        # Reset Select All toggle.
        self.select_all_cb.setChecked(False)

    def _delete_pending_task(self, task_id: str) -> None:
        task = self._registry.by_id(task_id)
        if task is None or task.status != "pending":
            return
        ok = QMessageBox.question(
            self, "Delete Pending Tasks",
            "Drop this pending task from the list?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ok != QMessageBox.StandardButton.Yes:
            return
        self._registry.remove(task_id)

    def _on_unit_finished(self, batch_id: str, payload: dict) -> None:
        runset_uid = payload.get("runset_uid")
        if not runset_uid:
            return
        task_id = self._runset_to_task.get(runset_uid)
        if task_id is None:
            return
        task = self._registry.by_id(task_id)
        if task is None or task.status != "running":
            return
        new_done = task.runs_done + 1
        self._registry.update_progress(task_id, new_done)
        if (payload.get("runset_finished") and new_done == task.runs
                and not task.runs_failed):
            self._registry.set_status(task_id, "completed")

    def _on_batch_progress(self, snapshot) -> None:
        failures = defaultdict(list)
        for failure in getattr(snapshot, "failures", ()):
            failures[failure.runset_uid].append(failure)
        for tid in self._batch_to_tasks.get(snapshot.batch_id, []):
            task = self._registry.by_id(tid)
            if task is None:
                continue
            errors = failures.get(task.runset_uid, [])
            if len(errors) != task.runs_failed:
                self._registry.update_progress(
                    tid, task.runs_done, runs_failed=len(errors),
                    message="\n".join(error.message for error in errors))

    def _on_batch_terminal(self, snapshot) -> None:
        self._on_batch_progress(snapshot)
        ids = self._batch_to_tasks.pop(snapshot.batch_id, [])
        status_map = {"canceled": "canceled", "failed": "failed"}
        new_status = status_map.get(snapshot.status, "completed")
        message = getattr(snapshot, "message", "") if new_status == "failed" else ""
        for tid in ids:
            task = self._registry.by_id(tid)
            if task is None or task.status != "running":
                continue
            counts = getattr(snapshot, 'runset_counts', None) or {}
            if task.runset_uid in counts:
                self._registry.update_progress(tid, counts[task.runset_uid])
            task_status = new_status
            task_message = message
            if task.runs_failed and new_status != 'canceled':
                # Do not present a partially successful runset as a complete
                # experiment, or automatically analyze just its successful runs.
                task_status = "failed"
                task_message = (f"{task.runs_done}/{task.runs} runs succeeded; "
                                f"{task.runs_failed} failed.\n{task.message}")
            self._registry.set_status(tid, task_status, message=task_message)
        if new_status == "failed" and message:
            QMessageBox.critical(self, "Run failed", message)
        elif getattr(snapshot, "failures", ()):
            details = "\n".join(f.message for f in snapshot.failures[:5])
            if len(snapshot.failures) > 5:
                details += "\nFurther errors are listed in the task tooltips."
            QMessageBox.warning(
                self, "Run errors",
                f"{snapshot.completed_runs}/{snapshot.total_runs} runs succeeded; "
                f"{len(snapshot.failures)} failed. Successful results are preserved.\n\n"
                f"{details}")

    def _on_batch_failed(self, batch_id: str, message: str) -> None:
        for tid in self._batch_to_tasks.pop(batch_id, []):
            task = self._registry.by_id(tid)
            if task is not None and task.status == "running":
                self._registry.set_status(tid, "failed", message=message)
        if message:
            QMessageBox.critical(self, "Run failed", message)

    # ───────────────────────── internals ───────────────────────────

    def _resolve_composites(
        self, tasks: list[Task]
    ) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], str]]:
        """Resolve task sources and return both instances and visible errors.

        For ``generator`` sources we read the live inventory; for file
        sources we re-read the unified JSON from disk. Keys include both the
        label and path so two files with the same display label cannot collide.
        """
        out: dict[tuple[str, str], dict] = {}
        errors: dict[tuple[str, str], str] = {}
        for t in tasks:
            source_key = (
                t.composite_label,
                t.composite_path or "<generator>",
            )
            if source_key in out or source_key in errors:
                continue
            if t.composite_source_kind == "generator":
                try:
                    comp = self._lookup_generator_composite(t.composite_label)
                    if comp is not None:
                        out[source_key] = comp
                    else:
                        errors[source_key] = (
                            f"Generated instance '{t.composite_label}' is no "
                            "longer available."
                        )
                except Exception as exc:  # noqa: BLE001
                    errors[source_key] = (
                        f"Failed to resolve generated instance "
                        f"'{t.composite_label}': {exc}"
                    )
            elif t.composite_source_kind == "file" and t.composite_path:
                try:
                    from portal_core.load_instance import load_instance
                    from portal_studio.validation import validate_instance_for_gui
                    instance = load_instance(t.composite_path)
                    validate_instance_for_gui(instance)
                    out[source_key] = instance
                except Exception as exc:  # noqa: BLE001
                    errors[source_key] = (
                        f"Failed to load instance '{t.composite_label}' from "
                        f"'{t.composite_path}': {exc}"
                    )
            else:
                errors[source_key] = (
                    f"Instance source for '{t.composite_label}' is incomplete."
                )
        return out, errors

    def _lookup_generator_composite(self, label: str | None = None) -> dict | None:
        # The parent ExperimentationTab installs this hook on us; called
        # whenever we need the live composite at Run time.
        if hasattr(self, "_generator_composite_provider"):
            try:
                return self._generator_composite_provider(label)
            except TypeError:
                return self._generator_composite_provider()
        return None

    def install_generator_composite_provider(self, provider) -> None:
        self._generator_composite_provider = provider

    def _build_batch_config(self, tasks: list[Task]) -> BatchConfig:
        # Every task is single-algorithm; one TaskConfig per task.
        # Submission groups share these batch-wide settings.
        cfgs = [
            TaskConfig(
                label=t.label, algorithm=t.algorithm,
                algorithm_params=dict(t.algorithm_params),
                runs=t.runs, seed_strategy=t.seed_strategy,
                master_seed=t.master_seed,
                realization_type=t.realization_type,
            )
            for t in tasks
        ]
        return BatchConfig(
            tasks=cfgs,
            fe_budget=tasks[0].fe_budget,
            metric_outputs=list(tasks[0].metric_outputs),
            max_parallel_runs=int(self.workers_spin.value())
            if self.pool_type_combo.currentData() == "process" else 1,
            output_dir=None,
            composite_path=tasks[0].composite_path or "<generator>",
        )

    def _autoname_output_dir(self) -> str:
        # Use an absolute bundle path so the scheduler and workers access
        # the same result databases regardless of their working directories.
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return str((Path("results") / "raw_runs" / f"batch_{stamp}").resolve())

    def _install_row_checkboxes(self) -> None:
        for row in range(self.model.rowCount()):
            task = self.model.task_at(row)
            if task is None:
                continue
            select_cell = self.model.index(row, 0)
            existing = self.table.indexWidget(select_cell)
            checked = task.task_id in self.model.selected_ids()
            if existing is not None and isinstance(existing, QCheckBox):
                existing.blockSignals(True)
                existing.setChecked(checked)
                existing.setEnabled(task.status == "pending")
                existing.blockSignals(False)
            else:
                cb = QCheckBox()
                cb.setChecked(checked)
                cb.setEnabled(task.status == "pending")
                tid = task.task_id
                cb.toggled.connect(
                    lambda checked, t=tid: self.model.toggle_selection(t, checked))
                self.table.setIndexWidget(select_cell, cb)

            delete_cell = self.model.index(row, 4)
            existing_delete = self.table.indexWidget(delete_cell)
            if existing_delete is not None:
                existing_delete.setEnabled(task.status == "pending")
                continue
            delete_btn = make_icon_button(
                make_x_icon(), "Delete this pending task",
                lambda t=task.task_id: self._delete_pending_task(t),
                size=22,
            )
            delete_btn.setEnabled(task.status == "pending")
            self.table.setIndexWidget(delete_cell, delete_btn)
