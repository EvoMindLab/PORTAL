"""Finished Tasks panel for the Experiment tab.

Completed and terminal tasks land here after leaving Pending. The main table
shows task status and final-absolute-error metrics inline.
"""
from __future__ import annotations

import sqlite3
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
    QTableView, QVBoxLayout,
)

from portal_studio.experiment_runner.finished_archive import (
    export_finished_tasks, import_finished_tasks,
)
from portal_studio.gui.experimental.batch_service import BatchService
from portal_studio.gui.experimental.task_model import Task, TaskRegistry
from portal_studio.gui.common.help import help_body, make_help_button, show_help_dialog, term
from portal_studio.gui.common.icons import make_icon_button, make_x_icon
from portal_studio.experiment_runner.normalisation import METRIC_CHOICES, read_normalised


class _CompletedTasksModel(QAbstractTableModel):
    """Select / Task ID / Info / metrics / Status / Delete."""

    HEADERS = [
        "Select",
        "Task ID",
        "Task Info",
        "Mean",
        "Median",
        "SE",
        "Status",
        "Delete",
    ]

    def __init__(self, registry: TaskRegistry,
                 parent=None) -> None:
        super().__init__(parent)
        self._registry = registry
        self.metric = 'final_error'
        self._tasks: list[Task] = []
        self._selected: set[str] = set()
        registry.completed_changed.connect(self._refresh)
        self._refresh()

    def selected_ids(self) -> list[str]:
        return [tid for tid in self._selected if any(
            t.task_id == tid for t in self._tasks)]

    def set_metric(self, metric):
        self.metric = metric
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 3, 5)
        self._refresh()

    def task_at(self, row: int) -> Task | None:
        if 0 <= row < len(self._tasks):
            return self._tasks[row]
        return None

    def select_all(self, checked: bool) -> None:
        if checked:
            self._selected = {t.task_id for t in self._tasks}
        else:
            self._selected.clear()
        self._refresh_checkboxes_only()

    def toggle_selection(self, task_id: str, checked: bool) -> None:
        if checked:
            self._selected.add(task_id)
        else:
            self._selected.discard(task_id)

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
            if col == 0:
                return ""
            if col == 1:
                return task.task_id
            if col == 2:
                return task.info()
            if col == 3:
                return _fmt_metric(task, self.metric + "_mean")
            if col == 4:
                return _fmt_metric(task, self.metric + "_median")
            if col == 5:
                return _fmt_metric(task, self.metric + "_se")
            if col == 6:
                return _status_text(task.status)
            return ""
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        if role == Qt.ItemDataRole.ForegroundRole and col == 6:
            return QBrush(QColor(_status_color(task.status)))
        if role == Qt.ItemDataRole.ToolTipRole and col == 6:
            if task.status == "failed" and task.message:
                return task.message
            return None
        return None

    def _refresh(self) -> None:
        self.beginResetModel()
        self._tasks = self._registry.completed()
        self._selected &= {t.task_id for t in self._tasks}
        self.endResetModel()

    def _refresh_checkboxes_only(self) -> None:
        if self._tasks:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self._tasks) - 1, 0),
            )


def _status_text(status: str) -> str:
    return {"completed": "Done", "failed": "Failed",
            "canceled": "Canceled"}.get(status, status)


def _status_color(status: str) -> str:
    return {"completed": "#16a34a", "failed": "#b91c1c",
            "canceled": "#9a3412"}.get(status, "#374151")


class CompletedTasksPanel(QFrame):
    """Finished-task table with portable result import/export."""

    def __init__(self, *, registry: TaskRegistry,
                 batch_service: BatchService,
                 parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("projectPanel")
        self._registry = registry

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        self.title_label = QLabel("Finished Tasks")
        self.title_label.setObjectName("panelTitle")
        title_row.addWidget(self.title_label)
        self.help_btn = make_help_button(
            "Explain finished tasks",
            self._show_completed_tasks_help,
        )
        title_row.addWidget(self.help_btn)
        title_row.addStretch(1)
        outer.addLayout(title_row)

        self.select_all_cb = QCheckBox("Select All")
        self.select_all_cb.toggled.connect(self._on_select_all)
        head_row = QHBoxLayout()
        head_row.addWidget(self.select_all_cb)
        head_row.addWidget(QLabel('Error measure:'))
        self.metric_combo = QComboBox()
        for label, key in METRIC_CHOICES:
            self.metric_combo.addItem(label, key)
        self.metric_combo.setToolTip('Normalised measures use the evaluated recipe scale; unavailable values show a dash.')
        head_row.addWidget(self.metric_combo)
        head_row.addStretch(1)
        outer.addLayout(head_row)

        self.model = _CompletedTasksModel(registry, self)
        self.metric_combo.currentIndexChanged.connect(
            lambda: self.model.set_metric(self.metric_combo.currentData()))
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        outer.addWidget(self.table, stretch=1)

        io_row = QHBoxLayout()
        io_row.setSpacing(8)
        self.import_btn = QPushButton("Import Finished Results")
        self.import_btn.setProperty("variant", "primary")
        self.import_btn.setMinimumHeight(32)
        self.import_btn.clicked.connect(self._on_import_results)
        io_row.addWidget(self.import_btn)

        self.export_btn = QPushButton("Export Selected Results")
        self.export_btn.setProperty("variant", "primary")
        self.export_btn.setMinimumHeight(32)
        self.export_btn.clicked.connect(self._on_export_results)
        io_row.addWidget(self.export_btn)
        registry.completed_changed.connect(self._sync_export_button)
        self._sync_export_button()
        outer.addLayout(io_row)

        self.delete_btn = None

        registry.completed_changed.connect(self._refresh_metrics_for_batch)
        self.model.modelReset.connect(self._install_row_widgets)
        self.model.dataChanged.connect(self._install_row_widgets)
        self._install_row_widgets()

    def _on_select_all(self, checked: bool) -> None:
        self.model.select_all(checked)
        self._install_row_widgets()

    def _sync_export_button(self) -> None:
        self.export_btn.setEnabled(bool(self._registry.completed()))

    def _delete_completed_task(self, task_id: str) -> None:
        task = self._registry.by_id(task_id)
        if task is None or task.status not in ("completed", "failed", "canceled"):
            return
        ok = QMessageBox.question(
            self, "Delete Finished Task",
            "Drop this finished task from the list? "
            "The on-disk result bundle is not affected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ok == QMessageBox.StandardButton.Yes:
            self._registry.remove(task_id)

    def _show_completed_tasks_help(self) -> None:
        body = help_body("Finished tasks", [
            f"{term('Finished Tasks')} lists tasks that have completed, "
            "failed, or been canceled. The Status column shows the final "
            "state of each task.",
            f"{term('Select All')} and the row checkboxes choose tasks for "
            "export. Export uses the selected rows; Import does not require "
            "a selection.",
            "Choose Raw error, Normalised error, or Reference-scale error (%). "
            "Normalised error divides each run's raw error by its evaluated recipe scale. "
            "Percentages can exceed 100%. Missing scales show a dash; raw results remain available.",
            f"{term('Import Finished Results')} restores saved results into "
            "Finished Tasks without running an optimiser or loading a Generator "
            "instance. Completed tasks are then available for Statistical Analysis.",
            f"{term('Export Selected Results')} saves the selected finished tasks, "
            "their individual settings, errors, convergence data and analysis identities "
            "in a portable ZIP archive. It also includes CSV tables readable in Excel. "
            "Failed and canceled tasks keep their status and any available results.",
            "Successfully completed tasks are available in Statistical "
            "Analysis automatically. Tasks that failed or were canceled are "
            "not included.",
            "The row delete icon removes the task from this list. Result files "
            "already written to disk are not deleted.",
        ])
        show_help_dialog(self, "Finished tasks help", body)

    def _refresh_metrics_for_batch(self, _snapshot=None) -> None:
        for task in self._registry.all_tasks():
            if task.status not in ("completed",):
                continue
            if task.runset_uid is None or task.output_dir is None:
                continue
            if task.metrics:
                continue
            metrics = _summarise(task.output_dir, task.runset_uid)
            if metrics:
                self._registry.set_metrics(task.task_id, metrics)

    def _on_export_results(self) -> None:
        tasks = [self._registry.by_id(tid) for tid in self.model.selected_ids()]
        tasks = [t for t in tasks if t is not None]
        if not tasks:
            QMessageBox.information(self, "Export Results", "Tick at least one finished task to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export finished results", "finished_results.portal-results.zip",
            "PORTAL results (*.zip)")
        if not path:
            return
        try:
            export_finished_tasks([asdict(task) for task in tasks], path)
        except (OSError, ValueError, sqlite3.Error) as exc:
            QMessageBox.critical(self, "Export Results", str(exc))
            return
        QMessageBox.information(self, "Export Results", f"Saved {len(tasks)} finished task(s) to {path}.")

    def _on_import_results(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import finished results", "", "PORTAL results (*.zip)")
        if not path:
            return
        try:
            records = import_finished_tasks(path, Path("results") / "imported")
            tasks = [Task(**record) for record in records]
            for task in tasks:
                if task.status == "completed" and task.output_dir:
                    task.metrics = _summarise(task.output_dir, task.runset_uid)
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, zipfile.BadZipFile) as exc:
            QMessageBox.critical(self, "Import Results", str(exc))
            return
        existing = {t.task_id for t in self._registry.all_tasks()}
        for task in tasks:
            while task.task_id in existing:
                task.task_id = Task(label=task.label, algorithm=task.algorithm).task_id
            existing.add(task.task_id)
            self._registry.add(task)
        QMessageBox.information(
            self, "Import Results",
            f"Restored {len(tasks)} task(s) into Finished Tasks. "
            "Successful tasks are available in Statistical Analysis; no tasks were queued.")

    def _install_row_widgets(self) -> None:
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
                existing.blockSignals(False)
            else:
                cb = QCheckBox()
                cb.setChecked(checked)
                tid = task.task_id
                cb.toggled.connect(
                    lambda checked, t=tid: self.model.toggle_selection(t, checked))
                self.table.setIndexWidget(select_cell, cb)

            delete_cell = self.model.index(row, 7)
            if self.table.indexWidget(delete_cell) is None:
                delete_btn = make_icon_button(
                    make_x_icon(), "Delete this finished task",
                    lambda t=task.task_id: self._delete_completed_task(t),
                    size=22,
                )
                self.table.setIndexWidget(delete_cell, delete_btn)


def _fmt_metric(task: Task, key: str) -> str:
    val = task.metrics.get(key)
    return f"{val:.4g}" if val is not None else "-"


def _summarise(output_dir: str, runset_uid: str) -> dict[str, float]:
    """Pull per-run final_error from summary.sqlite, fold to mean/median/SE."""
    p = Path(output_dir) / "summary.sqlite"
    if not p.is_file():
        return {}
    try:
        cx = sqlite3.connect(p, timeout=5.0)
    except sqlite3.DatabaseError:
        return {}
    try:
        cur = cx.execute(
            """SELECT final_error FROM runs
               WHERE runset_uid = ? AND final_error IS NOT NULL""",
            (runset_uid,))
        values = np.asarray([r[0] for r in cur], dtype=float)
    finally:
        cx.close()
    if values.size == 0:
        return {}
    mean = float(np.mean(values))
    median = float(np.median(values))
    se = float(np.std(values, ddof=1) / np.sqrt(values.size)
               if values.size > 1 else 0.0)
    result = {
        "final_error_mean": mean,
        "final_error_median": median,
        "final_error_se": se,
    }
    try:
        values = read_normalised(output_dir, runset_uid)
        for key, multiplier in [('normalised_error', 1.), ('percentage_error', 100.)]:
            scaled = values * multiplier
            if not np.all(np.isfinite(scaled)):
                continue
            result.update({key + '_mean': float(np.mean(scaled)),
                           key + '_median': float(np.median(scaled)),
                           key + '_se': float(np.std(scaled, ddof=1) / np.sqrt(scaled.size))
                           if scaled.size > 1 else 0.})
    except ValueError:
        pass
    return result
