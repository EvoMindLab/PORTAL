"""Statistical analysis of completed tasks grouped by verified run identities."""
from __future__ import annotations

import sqlite3

from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, Qt,
)
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableView, QVBoxLayout, QWidget,
)

_DEFAULT_ACCEPTANCE = 1e-8

from portal_studio.experiment_runner.analysis.plotting import (
    load_boxplot_series_from_selection,
    load_trend_series_from_selection,
    render_boxplot, render_trend,
)
from portal_studio.experiment_runner.analysis.stats import (
    SaResult, SaTable, SelectionItem, run_sa, validated_run_data,
)
from portal_studio.gui.analysis.plot_dialog import PlotDialog
from portal_studio.experiment_runner.normalisation import METRIC_CHOICES
from portal_studio.gui.experimental.task_model import Task, TaskRegistry
from portal_studio.gui.common.help import help_body, make_help_button, show_help_dialog, term


_METHOD_CHOICES = [
    ("friedman", "Friedman test"),
    ("signed-rank", "Wilcoxon signed-rank"),
]


def sa_group_key(task: Task, metric: str = 'final_error') -> tuple:
    """Use recorded per-run identities, never display names or source paths."""
    if task.status != "completed":
        raise ValueError("Only successfully completed tasks are eligible.")
    if not task.output_dir or not task.runset_uid:
        raise ValueError("Recorded results are unavailable.")
    item = SelectionItem(task.output_dir, task.runset_uid, task.label, task.algorithm, task.label)
    values, identities = validated_run_data(item, metric, minimum_runs=1)
    if len(values) != task.runs or task.runs_failed:
        raise ValueError("Recorded runs do not match the requested successful run count.")
    return (task.realization_type, int(task.fe_budget), int(task.runs), identities)


def _sa_group_label(key: tuple, tasks: list[Task]) -> str:
    realization_type, budget, runs, _identities = key
    type_label = {"none": "Type 0", "type1": "Type 1", "type2": "Type 2"}.get(
        realization_type, realization_type)
    return f"{tasks[0].composite_label} | {type_label} | FE {budget:g} | runs {runs:g}"


def _normalise_group_key(value):
    """Qt can round-trip nested tuples as lists in combo-box item data."""
    return tuple(_normalise_group_key(v) for v in value) if isinstance(value, (list, tuple)) else value


class _SaTasksModel(QAbstractTableModel):
    HEADERS = ["Select", "Task ID", "Task Info"]

    def __init__(self, registry: TaskRegistry,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._registry = registry
        self._tasks: list[Task] = []
        self._selected: set[str] = set()

    def selected_ids(self) -> list[str]:
        return [tid for tid in self._selected if any(
            t.task_id == tid for t in self._tasks)]

    def task_at(self, row: int) -> Task | None:
        if 0 <= row < len(self._tasks):
            return self._tasks[row]
        return None

    def select_all(self, checked: bool) -> None:
        if checked:
            self._selected = {t.task_id for t in self._tasks}
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

    def all_tasks(self) -> list[Task]:
        return list(self._tasks)

    def set_tasks(self, tasks: list[Task]) -> None:
        self.beginResetModel()
        self._tasks = list(tasks)
        self._selected &= {t.task_id for t in tasks}
        self.endResetModel()

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
        if role == Qt.ItemDataRole.DisplayRole:
            task = self._tasks[index.row()]
            return ["", task.task_id, task.info()][index.column()]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None

class _SaTableViewModel(QAbstractTableModel):
    """Single SA-result table — used inline below the Run button."""

    def __init__(self, table: SaTable,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._table = table

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._table.rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._table.columns)

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole):
        if (role == Qt.ItemDataRole.DisplayRole
                and orientation == Qt.Orientation.Horizontal):
            return self._table.columns[section]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def data(self, index: QModelIndex,
             role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self._table.rows[index.row()][index.column()]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        if role == Qt.ItemDataRole.BackgroundRole:
            sig = self._table.significant
            if (index.row() < len(sig)
                    and index.column() < len(sig[index.row()])
                    and sig[index.row()][index.column()]):
                return QBrush(QColor("#fef3c7"))
        return None


class _SaResultDialog(QDialog):
    """Modal table viewer for one statistical-test result."""

    def __init__(self, result: SaResult,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Statistical test data")
        self.setModal(True)
        self.resize(720, 420)
        self.table_views: list[QTableView] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        if result.message:
            msg = QLabel(f"<i>{result.message}</i>")
            msg.setWordWrap(True)
            msg.setStyleSheet("color: #6b7280;")
            layout.addWidget(msg)

        for table in result.tables:
            if table.caption:
                cap = QLabel(f"<b>{table.caption}</b>")
                cap.setWordWrap(True)
                layout.addWidget(cap)
            if not table.columns:
                continue
            tv = QTableView()
            tv.setModel(_SaTableViewModel(table, tv))
            tv.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
            tv.verticalHeader().setVisible(False)
            tv.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents)
            tv.horizontalHeader().setStretchLastSection(True)
            layout.addWidget(tv)
            self.table_views.append(tv)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class SaPanel(QFrame):
    """Column 4 — Selected tasks + method + run + plot buttons."""

    def __init__(self, *, registry: TaskRegistry,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("projectPanel")
        self._registry = registry
        self._groups: dict[tuple, list[Task]] = {}
        self._latest: SaResult | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        self.title_label = QLabel("Statistical Analysis (SA)")
        self.title_label.setObjectName("panelTitle")
        title_row.addWidget(self.title_label)
        self.help_btn = make_help_button(
            "Explain statistical analysis",
            self._show_sa_help,
        )
        title_row.addWidget(self.help_btn)
        title_row.addStretch(1)
        outer.addLayout(title_row)

        self.metric_combo = QComboBox()
        for label, key in METRIC_CHOICES:
            self.metric_combo.addItem(label, key)
        self.metric_combo.setToolTip('Error measure for tests and plots. Acceptance threshold uses the selected units.')
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        self.metric_combo.currentIndexChanged.connect(lambda: self.acceptance_edit.setText(''))
        metric_row = QHBoxLayout()
        metric_row.addWidget(QLabel('Error measure:'))
        metric_row.addWidget(self.metric_combo, stretch=1)
        outer.addLayout(metric_row)

        group_row = QHBoxLayout()
        group_row.addWidget(QLabel("Compatible group:"))
        self.group_combo = QComboBox()
        self.group_combo.currentIndexChanged.connect(self._on_group_changed)
        self.group_combo.currentIndexChanged.connect(self._sync_group_combo_tooltip)
        group_row.addWidget(self.group_combo, stretch=1)
        outer.addLayout(group_row)


        head = QHBoxLayout()
        self.select_all_cb = QCheckBox("Select All")
        self.select_all_cb.toggled.connect(self._on_select_all)
        head.addWidget(self.select_all_cb)
        head.addStretch(1)
        outer.addLayout(head)

        self.model = _SaTasksModel(registry, self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        outer.addWidget(self.table, stretch=1)

        # Grid keeps the method/threshold row's two halves aligned with the
        # two button columns below it (same column boundary throughout).
        self.controls_grid = QGridLayout()
        controls_grid = self.controls_grid
        controls_grid.setHorizontalSpacing(8)
        controls_grid.setVerticalSpacing(8)
        controls_grid.setColumnStretch(0, 1)
        controls_grid.setColumnStretch(1, 1)

        self.method_row = QHBoxLayout()
        self.method_row.setSpacing(8)
        self.method_label = QLabel("SA Method:")
        self.method_row.addWidget(self.method_label)
        self.method_combo = QComboBox()
        for key, label in _METHOD_CHOICES:
            self.method_combo.addItem(label, userData=key)
            self.method_combo.setItemData(
                self.method_combo.count() - 1,
                label,
                Qt.ItemDataRole.ToolTipRole,
            )
        self._sync_method_tooltip()
        self.method_combo.currentIndexChanged.connect(self._sync_method_tooltip)
        self.method_combo.currentIndexChanged.connect(self._clear_sa_result)
        self.method_row.addWidget(self.method_combo, stretch=1)
        controls_grid.addLayout(self.method_row, 0, 0)

        self.threshold_row = QHBoxLayout()
        self.threshold_row.setSpacing(8)
        self.acceptance_label = QLabel("Acceptance threshold:")
        self.threshold_row.addWidget(self.acceptance_label)
        self.acceptance_edit = QLineEdit(repr(_DEFAULT_ACCEPTANCE))
        self.acceptance_edit.setToolTip(
            "Error below this value counts as 'solved' (drawn as a dashed line on trend plots).")
        self.acceptance_edit.setMaximumWidth(90)
        self.threshold_row.addWidget(self.acceptance_edit, stretch=1)
        controls_grid.addLayout(self.threshold_row, 0, 1)

        self.run_btn = QPushButton("📊 Run Statistical Test")
        self.run_btn.setMinimumHeight(32)
        self.run_btn.clicked.connect(self._on_run_clicked)
        controls_grid.addWidget(self.run_btn, 1, 0)

        self.result_data_btn = QPushButton("Statistical Test Data")
        self.result_data_btn.setMinimumHeight(32)
        self.result_data_btn.setEnabled(False)
        self.result_data_btn.clicked.connect(self._on_result_data_clicked)
        controls_grid.addWidget(self.result_data_btn, 1, 1)

        self.trend_btn = QPushButton("📈 Trend Plots")
        self.trend_btn.clicked.connect(self._on_trend_clicked)
        controls_grid.addWidget(self.trend_btn, 2, 0)

        self.box_btn = QPushButton("📦 Box Plots")
        self.box_btn.clicked.connect(self._on_box_clicked)
        controls_grid.addWidget(self.box_btn, 2, 1)

        outer.addLayout(controls_grid)

        # Inline result-table region.
        self._results_container = QVBoxLayout()
        outer.addLayout(self._results_container)

        self.delete_btn = None

        registry.completed_changed.connect(self._refresh_groups)
        self.model.modelReset.connect(self._install_row_checkboxes)
        self.model.dataChanged.connect(self._install_row_checkboxes)
        self._refresh_groups()
        self._install_row_checkboxes()

    # ───────────────────────── slots ───────────────────────────────

    def _on_select_all(self, checked: bool) -> None:
        self._clear_sa_result()
        self.model.select_all(checked)
        self._install_row_checkboxes()
        self._sync_select_all()

    def _sync_select_all(self) -> None:
        self.select_all_cb.blockSignals(True)
        self.select_all_cb.setChecked(
            bool(self.model.rowCount()) and len(self.model.selected_ids()) == self.model.rowCount())
        self.select_all_cb.blockSignals(False)

    def _clear_sa_result(self, *_args) -> None:
        self._latest = None
        if hasattr(self, "result_data_btn"):
            self.result_data_btn.setEnabled(False)
        while hasattr(self, "_results_container") and self._results_container.count():
            item = self._results_container.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _show_sa_help(self) -> None:
        body = help_body("Statistical analysis", [
            "Error measure selects raw error, normalised error, or reference-scale error (%). "
            "Normalised values use each evaluated recipe's scale and require enabled neutralisation. "
            "Missing scales are unavailable; percentages can exceed 100%.",
            f"{term('Statistical Analysis (SA)')} compares final errors in the "
            "selected measure. Raw error is |final best objective - known global "
            "optimum|. Successfully completed tasks appear here automatically; "
            "smaller errors indicate solutions closer to the optimum.",
            f"Choose {term('Error measure')} first. Groups update automatically "
            "to include complete successful tasks with valid data for that measure. "
            "Excluded tasks are listed with their reasons. Normalised and percentage "
            "measures require recorded positive reference scales for every run.",
            f"{term('Compatible group')} matches recorded problem identities for "
            "corresponding runs, realisation type, evaluation budget and run count. "
            "Algorithms and algorithm seeds may differ. Display labels, source paths "
            "and optimiser names do not establish problem identity. Single-run tasks "
            "can be plotted; paired tests require at least two runs. Studio validates "
            "the data again before testing.",
            f"{term('Select All')} and the row checkboxes choose the "
            "algorithms to include in the analysis or plots.",
            f"{term('SA Method')} selects the comparison. The Friedman test "
            "compares three or more algorithms and uses Holm-corrected "
            "pairwise Wilcoxon signed-rank post-hoc rows when significant. "
            "Wilcoxon signed-rank compares verified paired runs. "
            "Pairwise rows report the raw p-value, Holm-adjusted p-value, "
            "and significance based on the adjusted value. An n/a comparison "
            "is undefined, for example when paired samples are identical.",
            f"{term('Acceptance threshold')} uses the selected error measure's units. "
            "Enter it again after changing measures. Trend Plots display it as a reference line.",
            f"{term('Run Statistical Test')} analyses the selected algorithms. "
            f"After a successful run, {term('Statistical Test Data')} opens "
            "the result tables in a separate window.",
            f"{term('Trend Plots')} and {term('Box Plots')} summarize the "
            "selected task results directly. They do not require a "
            "statistical test first.",
        ])
        show_help_dialog(self, "Statistical analysis help", body)

    def _on_metric_changed(self, *_args) -> None:
        self.model.select_all(False)
        self._refresh_groups()

    def _refresh_groups(self) -> None:
        self._clear_sa_result()
        current = _normalise_group_key(self.group_combo.currentData())
        groups: dict[tuple, list[Task]] = {}
        metric = self.metric_combo.currentData()
        for task in self._registry.completed():
            try:
                key = sa_group_key(task, metric)
            except (ValueError, TypeError, OSError, sqlite3.Error):
                continue
            else:
                groups.setdefault(key, []).append(task)
        self._groups = groups
        ordered = sorted(groups, key=lambda k: (_sa_group_label(k, groups[k]), k))

        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        for number, key in enumerate(ordered, 1):
            label = f"Group {number}: {_sa_group_label(key, groups[key])}"
            self.group_combo.addItem(label, userData=key)
            self.group_combo.setItemData(self.group_combo.count() - 1, label, Qt.ItemDataRole.ToolTipRole)
        if current in ordered:
            self.group_combo.setCurrentIndex(ordered.index(current))
        elif ordered:
            self.group_combo.setCurrentIndex(0)
        self.group_combo.blockSignals(False)
        self._sync_group_combo_tooltip()
        self.model.set_tasks(groups.get(_normalise_group_key(self.group_combo.currentData()), []))
        self._install_row_checkboxes()
        self._sync_select_all()
        has_groups = bool(ordered)
        for widget in (self.group_combo, self.select_all_cb, self.table,
                       self.run_btn, self.trend_btn, self.box_btn):
            widget.setEnabled(has_groups)
        # Keep Error measure enabled even if every task lacks this measure.

    def _sync_group_combo_tooltip(self, *_args) -> None:
        index = self.group_combo.currentIndex()
        if 0 <= index < self.group_combo.count():
            self.group_combo.setToolTip(self.group_combo.itemText(index))
        else:
            self.group_combo.setToolTip("")

    def _sync_method_tooltip(self, *_args) -> None:
        index = self.method_combo.currentIndex()
        if 0 <= index < self.method_combo.count():
            self.method_combo.setToolTip(self.method_combo.itemText(index))
        else:
            self.method_combo.setToolTip("")

    def _on_group_changed(self, _index: int) -> None:
        self._clear_sa_result()
        self.model.set_tasks(self._groups.get(_normalise_group_key(self.group_combo.currentData()), []))
        self._install_row_checkboxes()
        self._sync_select_all()

    def _selected_tasks(self) -> list[Task]:
        ids = self.model.selected_ids()
        return [t for t in self.model.all_tasks() if t.task_id in ids]

    def _on_run_clicked(self) -> None:
        selection = self._build_selection()
        if len(selection) < 2:
            QMessageBox.information(
                self, "Run Statistical Test",
                "Tick at least 2 SA tasks before running a test.")
            return
        method = self.method_combo.currentData()
        try:
            result = run_sa(selection, method=method, metric=self.metric_combo.currentData())
        except Exception as exc:                                  # noqa: BLE001
            QMessageBox.critical(self, "Run Statistical Test", str(exc))
            return
        self._latest = result
        self.result_data_btn.setEnabled(True)

    def _on_result_data_clicked(self) -> None:
        if self._latest is None:
            QMessageBox.information(
                self, "Statistical Test Data",
                "Run a statistical test before opening the result data.")
            return
        _SaResultDialog(self._latest, self).exec()

    def _acceptance_threshold(self) -> float | None:
        try:
            v = float(self.acceptance_edit.text().strip())
            return v if v > 0 else None
        except ValueError:
            return None

    def _on_trend_clicked(self) -> None:
        selection = self._build_selection()
        if not selection:
            QMessageBox.information(
                self, "Trend Plots",
                "Tick at least 1 SA task before rendering a plot.")
            return
        metric = self.metric_combo.currentData()
        try:
            series = load_trend_series_from_selection(
                selection, metric='best_so_far' if metric == 'final_error' else metric,
                legend_label_mode="algorithm")
        except ValueError as exc:
            QMessageBox.information(self, 'Trend Plots', str(exc))
            return
        if not series:
            QMessageBox.information(
                self, "Trend Plots",
                "No trend data found on disk for the selected tasks.")
            return
        fig = render_trend(series, metric=metric,
                           acceptance_threshold=self._acceptance_threshold())
        dlg = PlotDialog(fig, title=f"Trend — {metric}", parent=self)
        dlg.exec()

    def _on_box_clicked(self) -> None:
        selection = self._build_selection()
        if not selection:
            QMessageBox.information(
                self, "Box Plots",
                "Tick at least 1 SA task before rendering a plot.")
            return
        metric = self.metric_combo.currentData()
        try:
            series = load_boxplot_series_from_selection(
                selection, metric=metric, legend_label_mode="algorithm")
        except ValueError as exc:
            QMessageBox.information(self, 'Box Plots', str(exc))
            return
        if not series:
            QMessageBox.information(
                self, "Box Plots",
                "No scalar samples found on disk for the selected tasks.")
            return
        fig = render_boxplot(series, metric=metric)
        dlg = PlotDialog(fig, title=f"Box — {metric}", parent=self)
        dlg.exec()

    # ───────────────────────── internals ───────────────────────────

    def _build_selection(self) -> list[SelectionItem]:
        out: list[SelectionItem] = []
        for task in self._selected_tasks():
            if not (task.output_dir and task.runset_uid):
                continue
            out.append(SelectionItem(
                bundle_path=task.output_dir,
                runset_uid=task.runset_uid,
                label=f"{task.label}::{task.algorithm}",
                algorithm=task.algorithm,
                task_label=task.label,
            ))
        return out

    def _render_result(self, result: SaResult) -> None:
        while self._results_container.count():
            item = self._results_container.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if result.message:
            msg = QLabel(f"<i>{result.message}</i>")
            msg.setStyleSheet("color: #9aa0a6;")
            self._results_container.addWidget(msg)
        for table in result.tables:
            if not table.columns:
                lbl = QLabel(f"<i>{table.caption}</i>")
                lbl.setStyleSheet("color: #9aa0a6;")
                self._results_container.addWidget(lbl)
                continue
            cap = QLabel(f"<b>{table.caption}</b>")
            cap.setWordWrap(True)
            self._results_container.addWidget(cap)
            tv = QTableView()
            tv.setModel(_SaTableViewModel(table, tv))
            tv.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
            tv.verticalHeader().setVisible(False)
            tv.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents)
            tv.setMaximumHeight(min(36 * (len(table.rows) + 1) + 40, 240))
            self._results_container.addWidget(tv)

    def _install_row_checkboxes(self) -> None:
        for row in range(self.model.rowCount()):
            task = self.model.task_at(row)
            if task is None:
                continue
            cell = self.model.index(row, 0)
            existing = self.table.indexWidget(cell)
            checked = task.task_id in self.model.selected_ids()
            if existing is not None and isinstance(existing, QCheckBox):
                existing.blockSignals(True)
                existing.setChecked(checked)
                existing.blockSignals(False)
                continue
            cb = QCheckBox()
            cb.setChecked(checked)
            tid = task.task_id
            cb.toggled.connect(
                lambda checked, t=tid: self._on_task_checked(t, checked))
            self.table.setIndexWidget(cell, cb)

    def _on_task_checked(self, task_id: str, checked: bool) -> None:
        self._clear_sa_result()
        self.model.toggle_selection(task_id, checked)
        self._sync_select_all()
