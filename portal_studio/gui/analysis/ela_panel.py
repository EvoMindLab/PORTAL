"""ELA panel for the Generator tab's right column.

User flow:
1. Set the sample factor (×d): it sizes the sample used for every feature.
2. Tick the feature groups to compute. The two computationally intensive groups (ela_conv,
   gradient) are unchecked by default.
3. Select one or more prepared Generator instances from the list, which
   tracks the inventory automatically.
4. Click 'Run ELA': a popup confirms results are saved as Excel and a save
   destination is prompted upfront, then the compute runs in a background
   QThread and the .xlsx is written when it finishes. The sampling seed is
   fixed internally (no user-facing seed control).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PySide6 import QtCore
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from portal_studio.experiment.ela import (
    ELAComputationCanceled, FEATURE_GROUPS, SLOW_GROUPS,
    compute_ela_batch, save_ela_xlsx,
)
from portal_studio.gui.common.composite_labels import generator_composite_label
from portal_studio.gui.common.help import help_body, make_parameter_help_button, show_help_dialog


_ROLE = QtCore.Qt.ItemDataRole.UserRole

# ELA sampling uses a fixed internal seed for reproducible sampling.
_ELA_SEED = 42

FEATURE_GROUP_HELP: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "ela_meta": (
        "Meta-model",
        (
            "Measures how well simple regression models (linear and quadratic) "
            "can approximate the landscape using the sampled points. It reports "
            "how much of the fitness variation is explained by these models "
            "(adjusted R<sup>2</sup>) and how stretched or skewed the quadratic "
            "model&rsquo;s geometry is (condition number).",
            "A high fit score indicates agreement with the fitted model over "
            "the sample. Model geometry describes that fitted approximation.",
        ),
    ),
    "ela_distribution": (
        "Y-distribution",
        (
            "Describes the statistical shape of the fitness value distribution "
            "across all sampled points: specifically its skewness "
            "(asymmetry), kurtosis (tail weight), and estimated number of "
            "peaks.",
            "These features describe sampled objective values. Peaks in their "
            "distribution are not a count of landscape optima.",
        ),
    ),
    "ela_level": (
        "Level-set",
        (
            "Divides sampled points into &ldquo;good&rdquo; and &ldquo;bad&rdquo;"
            " groups based on fitness quantiles, then measures how "
            "accurately a linear or quadratic classifier separates them in the "
            "decision space.",
            "Low classification error means the fitted classifier separates "
            "the sampled fitness classes well. High error indicates less "
            "separation under that classifier.",
        ),
    ),
    "pca": (
        "Principal components",
        (
            "Applies Principal Component Analysis to the decision space and to "
            "the combined sample (inputs + fitness values), reporting how much "
            "variance is captured by the leading components.",
            "Variance concentration depends on the sample and its scaling. "
            "It does not by itself establish the objective's effective "
            "dimension or identify irrelevant variables.",
        ),
    ),
    "nbc": (
        "Nearest-better clustering",
        (
            "For each sampled point, finds the nearest neighbour with a "
            "better fitness value. Summaries compare nearest-better and "
            "nearest-neighbour distances and their relationship to fitness.",
            "Use these sample-based descriptors alongside other features to "
            "investigate clustering and possible funnel structure.",
        ),
    ),
    "dispersion": (
        "Dispersion",
        (
            "Compares the average pairwise distance among the top-performing "
            "samples against the average pairwise distance across all samples. "
            "A small ratio means high performers are tightly clustered; a "
            "large ratio means they are spread out.",
            "Use this to compare the spatial concentration of good sampled "
            "solutions at different fitness quantiles.",
        ),
    ),
    "information_content": (
        "Information content",
        (
            "Measures the information-theoretic structure of the landscape by "
            "analysing fitness sequences along sorted sample paths. Produces "
            "an entropy score (overall variability), a smoothness index (how "
            "gradually fitness changes), and a ruggedness index (how many "
            "local reversals occur).",
            "These descriptors characterise sampled fitness changes at "
            "different sensitivity thresholds. Compare them under consistent sampling.",
        ),
    ),
    "fitness_distance": (
        "Fitness-distance correlation",
        (
            "Measures the correlation between each sample point&rsquo;s fitness "
            "value and its distance to the best solution found in the sample. "
            "Also reports the mean and standard deviation of fitness values "
            "and distances.",
            "For minimisation, positive correlation associates larger distances "
            "with worse fitness in the sample. The reference is the best sampled "
            "point, not necessarily a global optimum; weak correlation alone "
            "does not establish deception.",
        ),
    ),
    "ela_conv": (
        "ELA convexity",
        (
            "Evaluates random convex combinations of sampled point pairs and "
            "compares their fitness with linearly interpolated endpoint values. "
            "This feature group requires additional function "
            "evaluations beyond the initial sample.",
            "These are sampled convexity and linearity indicators, not a proof "
            "of global convexity. Enabling this group increases computation time.",
        ),
    ),
    "gradient": (
        "Gradient",
        (
            "Performs a random walk through the search space and estimates the "
            "normalised slope magnitude between consecutive steps, summarising its mean and "
            "standard deviation across the walk.",
            "These values summarise fitness-change magnitudes along the walk. "
            "They do not recover the full gradient direction or establish "
            "multimodality. This group uses additional objective evaluations.",
        ),
    ),
}


@dataclass
class _InstanceEntry:
    """One prepared Generator instance in the inventory-backed list.

    ``checked`` tracks inclusion through the row's custom checkbox widget.
    """
    label: str
    composite: Dict[str, Any]
    checked: bool = True


class _ELAWorker(QObject):
    """QObject that runs compute_ela_batch in a background thread."""

    progress = Signal(int, int, str)   # i, n, label
    finished = Signal(object)          # pandas.DataFrame
    canceled = Signal()
    failed = Signal(str)               # error message

    def __init__(self, items: List[Tuple[str, Dict[str, Any]]],
                 groups: List[str], sample_factor: int, seed: int) -> None:
        super().__init__()
        self._items = items
        self._groups = groups
        self._sample_factor = sample_factor
        self._seed = seed
        self._stop_requested = False

    def cancel(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            df = compute_ela_batch(
                self._items, groups=self._groups,
                sample_factor=self._sample_factor, seed=self._seed,
                progress=lambda i, n, label: self.progress.emit(i, n, label),
                should_stop=lambda: self._stop_requested,
            )
            self.finished.emit(df)
        except ELAComputationCanceled:
            self.canceled.emit()
        except Exception as exc:   # noqa: BLE001 - surface any failure
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _IntegerLineEdit(QLineEdit):
    """Plain numeric text input with spin-box-like value helpers."""

    def __init__(self, minimum: int, maximum: int, value: int,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._minimum = int(minimum)
        self._maximum = int(maximum)
        self._default = int(value)
        self.setValidator(QIntValidator(self._minimum, self._maximum, self))
        self.setValue(value)

    def value(self) -> int:
        text = self.text().strip()
        if not text:
            self.setValue(self._default)
            return self._default
        value = max(self._minimum, min(self._maximum, int(text)))
        if text != str(value):
            self.setValue(value)
        return value

    def setValue(self, value: int) -> None:
        self.setText(str(max(self._minimum, min(self._maximum, int(value)))))


class ELAPanel(QWidget):
    """Right-column ELA panel backed by the live Generator inventory.

    Reads prepared instances from ``.composites`` and includes the current
    ``.composite`` if it is not already listed.
    """

    composite_selected = Signal(int)    # emits composite index when row is clicked
    instance_deleted = Signal(int)      # emits removed index after a row is deleted

    def __init__(self, inventory, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._inventory = inventory
        self._results = None            # last pandas.DataFrame
        self._worker: Optional[_ELAWorker] = None
        self._thread: Optional[QThread] = None
        self._closing = False
        self._checked_states: dict[int, bool] = {}  # composite index -> checked

        # The theme stylesheet targets ``QWidget#projectPanel``
        # — a literal-class selector that doesn't always cascade onto a
        # subclass like ELAPanel. Host the content in an inner plain QWidget
        # so the white-card look matches the other two columns exactly.
        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.setSpacing(0)
        card = QWidget()
        card.setObjectName("projectPanel")
        wrap.addWidget(card)
        self._card = card

        outer = QVBoxLayout(card)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        title = QLabel("Exploratory Landscape Analysis (ELA)")
        title.setObjectName("panelTitle")
        outer.addWidget(title)

        # Sample factor first: it sizes the sample used for every feature below.
        sf_header = QLabel("Sample factor")
        sf_header.setStyleSheet("font-weight: 600;")
        outer.addWidget(sf_header)
        ss_row = QHBoxLayout()
        self.sample_settings_row = ss_row
        ss_row.addWidget(QLabel("Factor:"))
        self.sample_spin = _IntegerLineEdit(10, 10_000, 1000)
        self.sample_spin.setMaximumWidth(96)
        ss_row.addWidget(self.sample_spin)
        ss_row.addWidget(QLabel("× dimension"))    # total samples = factor × dimension
        self.sample_factor_help_button = make_parameter_help_button(
            "Explain sample factor",
            self._show_sample_factor_help,
        )
        ss_row.addWidget(self.sample_factor_help_button)
        ss_row.addStretch(1)
        outer.addLayout(ss_row)

        # Feature groups
        groups_header = QLabel("Feature groups")
        groups_header.setStyleSheet("font-weight: 600;")
        outer.addWidget(groups_header)
        groups_box = QFrame()
        gg = QGridLayout(groups_box)
        gg.setContentsMargins(0, 0, 0, 0)
        gg.setSpacing(4)
        self.feature_groups_layout = gg
        self.group_checks: Dict[str, QCheckBox] = {}
        self.feature_help_buttons: Dict[str, QPushButton] = {}
        self.feature_group_rows: Dict[str, QWidget] = {}
        for i, g in enumerate(FEATURE_GROUPS):
            label, _description = FEATURE_GROUP_HELP[g]
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            cb = QCheckBox(
                label + ("  (computationally intensive)" if g in SLOW_GROUPS else ""))
            cb.setChecked(g not in SLOW_GROUPS)
            cb.setToolTip(f"Include the {label} ELA feature group in this run.")
            self.group_checks[g] = cb
            help_btn = make_parameter_help_button(
                f"Explain {label} ({g})",
                lambda _checked=False, group=g: self._show_feature_group_help(group),
            )
            self.feature_help_buttons[g] = help_btn
            row_layout.addWidget(cb)
            row_layout.addWidget(help_btn)
            row_layout.addStretch(1)
            self.feature_group_rows[g] = row_widget
            gg.addWidget(row_widget, i, 0)
        outer.addWidget(groups_box)

        # Instances
        inst_header = QLabel("Prepared instances")
        inst_header.setObjectName("elaInstancesHeader")
        inst_header.setStyleSheet("font-weight: 600;")
        outer.addWidget(inst_header)
        self.inst_list = QListWidget()
        self.inst_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.inst_list.currentRowChanged.connect(self._on_row_selected)
        outer.addWidget(self.inst_list, stretch=1)
        # Empty-state hint, toggled by refresh_instance_list.
        self.inst_empty_hint = QLabel(
            "<i>No prepared instances yet. In the Generator, create a block, "
            "then click <b>Prepare Instance for ELA</b> to add it here.</i>")
        self.inst_empty_hint.setWordWrap(True)
        self.inst_empty_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.inst_empty_hint.setStyleSheet("color: #9aa0a6; padding: 18px;")
        outer.addWidget(self.inst_empty_hint, stretch=1)

        self.excel_note = QLabel("Results are saved automatically as an Excel (.xlsx) file.")
        self.excel_note.setObjectName("elaExcelSaveNote")
        self.excel_note.setWordWrap(True)
        self.excel_note.setStyleSheet("color: #6b7280;")
        outer.addWidget(self.excel_note)

        # Progress + actions
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        outer.addWidget(self.progress_bar)
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #6b7280;")
        outer.addWidget(self.progress_label)

        # Single 'Run ELA' button — the user is prompted for a save destination
        # right when they click, and the .xlsx is written automatically when
        # the run finishes (no separate Save button).
        self.run_btn = QPushButton("Run ELA")
        self.run_btn.setProperty("variant", "primary")
        outer.addWidget(self.run_btn)
        self.stop_btn = QPushButton("Stop ELA")
        self.stop_btn.setVisible(False)
        self.stop_btn.setEnabled(False)
        outer.addWidget(self.stop_btn)

        # Wire signals
        self.run_btn.clicked.connect(self.on_run_clicked)
        self.stop_btn.clicked.connect(self._on_stop_clicked)

        # Path the worker will write to once compute finishes. Set in
        # on_run_clicked; cleared on cancellation / failure.
        self._pending_save_path: Optional[str] = None

        self.refresh_instance_list()

    # ------------------------------------------------------------- helpers
    def selected_groups(self) -> List[str]:
        return [g for g, cb in self.group_checks.items() if cb.isChecked()]

    def _show_feature_group_help(self, group: str) -> None:
        label, paragraphs = FEATURE_GROUP_HELP[group]
        notes = [f"<b>{label}</b>", *paragraphs]
        if group in SLOW_GROUPS:
            notes.append(
                "This group is unchecked by default because it performs additional function evaluations."
            )
        self._show_help_dialog(f"{label} help", help_body(label, notes))

    def _show_sample_factor_help(self) -> None:
        self._show_help_dialog(
            "Sample factor help",
            help_body("Sample factor", [
                "<b>Sample Factor</b>",
                "Controls how many points are sampled to analyse the "
                "landscape. The total number of samples is this value "
                "multiplied by the problem dimension. A larger factor gives "
                "more accurate feature estimates but takes longer to compute.",
            ]),
        )

    def _show_help_dialog(self, title: str, body: str) -> None:
        show_help_dialog(self, title, body)

    def selected_instances(self) -> List[_InstanceEntry]:
        """Return entries whose per-row checkbox is ticked."""
        out: List[_InstanceEntry] = []
        for i in range(self.inst_list.count()):
            it = self.inst_list.item(i)
            entry = it.data(_ROLE)
            if entry is not None and entry.checked:
                out.append(entry)
        return out

    def refresh_instance_list(self) -> None:
        """Rebuild the instance list from all built composites in the inventory."""
        prev_row = self.inst_list.currentRow()
        self.inst_list.blockSignals(True)
        self.inst_list.clear()

        composites = self._inventory_composites()
        for idx, comp in enumerate(composites):
            label = generator_composite_label(comp)
            checked = self._checked_states.get(idx, True)
            entry = _InstanceEntry(label=label, composite=comp, checked=checked)
            self._add_instance_item(entry, idx)

        self.inst_list.blockSignals(False)

        # Restore or default selection to the last composite (suppress
        # composite_selected so the tree doesn't reload on every refresh).
        count = self.inst_list.count()
        if count > 0:
            restore = prev_row if 0 <= prev_row < count else count - 1
            self.inst_list.blockSignals(True)
            self.inst_list.setCurrentRow(restore)
            self.inst_list.blockSignals(False)

        # Toggle the empty-state hint.
        if hasattr(self, "inst_empty_hint"):
            self.inst_list.setVisible(count > 0)
            self.inst_empty_hint.setVisible(count == 0)

    def _inventory_composites(self) -> List[Dict[str, Any]]:
        """Return generated composites in display order."""
        out: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for comp in getattr(self._inventory, "composites", []) or []:
            if isinstance(comp, dict) and id(comp) not in seen:
                out.append(comp)
                seen.add(id(comp))
        latest = getattr(self._inventory, "composite", None)
        if isinstance(latest, dict) and id(latest) not in seen:
            out.append(latest)
        return out

    def _add_instance_item(self, entry: _InstanceEntry, idx: int) -> None:
        """Add one row using a custom widget: checkbox + label."""
        from PySide6.QtWidgets import QFrame
        item = QListWidgetItem()
        item.setData(_ROLE, entry)
        row = QFrame()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(6)

        cb = QCheckBox()
        cb.setChecked(entry.checked)
        cb.toggled.connect(lambda checked, e=entry, i=idx: self._on_entry_check_toggled(e, i, checked))
        lay.addWidget(cb)

        label = QLabel(entry.label)
        label.setWordWrap(False)
        lay.addWidget(label, stretch=1)

        from portal_studio.gui.common.icons import make_x_icon, make_icon_button
        delete_btn = make_icon_button(
            make_x_icon(), "Delete this prepared instance",
            lambda i=idx: self._on_delete_instance(i), size=22)
        lay.addWidget(delete_btn)

        item.setSizeHint(row.sizeHint())
        self.inst_list.addItem(item)
        self.inst_list.setItemWidget(item, row)

    def _on_entry_check_toggled(self, entry: _InstanceEntry, idx: int, checked: bool) -> None:
        entry.checked = bool(checked)
        self._checked_states[idx] = bool(checked)

    def _on_row_selected(self, row: int) -> None:
        """Emit the composite index so the Generator tab can show its tree."""
        if row >= 0:
            self.composite_selected.emit(row)

    def _on_delete_instance(self, idx: int) -> None:
        """Confirm, then permanently remove prepared instance ``idx``."""
        answer = QMessageBox.question(
            self, "Delete instance",
            "Delete this prepared instance? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        composites = getattr(self._inventory, "composites", []) or []
        if 0 <= idx < len(composites):
            self._inventory.remove_composite(idx)
        else:
            self._inventory.composite = None
        self._reindex_checked_states(idx)
        self.refresh_instance_list()
        self.instance_deleted.emit(idx)

    def _reindex_checked_states(self, removed_idx: int) -> None:
        """Shift the index-keyed check states down past the removed row."""
        new: dict[int, bool] = {}
        for k, v in self._checked_states.items():
            if k == removed_idx:
                continue
            new[k - 1 if k > removed_idx else k] = v
        self._checked_states = new

    # ------------------------------------------------------------- actions
    def on_run_clicked(self) -> None:
        if self._closing:
            return
        groups = self.selected_groups()
        if not groups:
            QMessageBox.information(self, "Run ELA",
                                    "Pick at least one feature group.")
            return
        instances = self.selected_instances()
        if not instances:
            QMessageBox.information(self, "Run ELA",
                                    "Tick at least one instance to analyse.")
            return
        QMessageBox.information(
            self, "Run ELA",
            "ELA will analyse the selected instances. The results will be "
            "saved automatically as an Excel (.xlsx) file once the run "
            "finishes. Choose where to save it next.")
        # Prompt for the save destination upfront, so once the run completes
        # the .xlsx is written automatically (no separate Save click).
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ELA results", "ela_results.xlsx",
            "Excel files (*.xlsx)")
        if not path:
            return                          # user cancelled — don't run
        self._pending_save_path = path
        items = [(e.label, e.composite) for e in instances]

        self._set_running(True, total=len(items) * len(groups))
        self._thread = QThread(self)
        self._worker = _ELAWorker(
            items=items, groups=groups,
            sample_factor=int(self.sample_spin.value()),
            seed=_ELA_SEED,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.canceled.connect(self._on_canceled)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit, QtCore.Qt.ConnectionType.DirectConnection)
        self._worker.canceled.connect(self._thread.quit, QtCore.Qt.ConnectionType.DirectConnection)
        self._worker.failed.connect(self._thread.quit, QtCore.Qt.ConnectionType.DirectConnection)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _set_running(self, running: bool, *, total: int = 0) -> None:
        self.run_btn.setEnabled(not running)
        self.inst_list.setEnabled(not running)
        self.stop_btn.setVisible(running)
        self.stop_btn.setEnabled(running)
        self.progress_bar.setVisible(running)
        if running:
            self.stop_btn.setText("Stop ELA")
            self.progress_bar.setRange(0, max(1, int(total)))
            self.progress_bar.setValue(0)
            self.progress_label.setText(f"Running ELA…  0 / {max(1, int(total))}")

    def _on_stop_clicked(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stopping...")
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText(
            "Stop requested. Waiting for the current ELA feature to finish...")

    def _on_progress(self, i: int, n: int, label: str) -> None:
        if self._closing:
            return
        self.progress_bar.setRange(0, max(1, int(n)))
        self.progress_bar.setValue(max(0, min(int(i), self.progress_bar.maximum())))
        self.progress_label.setText(f"Running ELA…  {i} / {n}: {label}")

    def _on_finished(self, df) -> None:
        if self._closing:
            return
        self._results = df
        self.progress_bar.setValue(self.progress_bar.maximum())
        n_features = sum(c.startswith("ela_") for c in df.columns)
        # Auto-write to the path the user picked when they clicked Run.
        written = None
        save_path = self._pending_save_path
        if save_path:
            try:
                written = save_ela_xlsx(df, save_path)
            except Exception as exc:   # noqa: BLE001
                QMessageBox.critical(self, "Save ELA results", str(exc))
        self._pending_save_path = None
        if written is not None:
            self.progress_label.setText(
                f"Done: {len(df)} instance(s), {n_features} features. Saved.")
            QMessageBox.information(self, "ELA done", f"Results saved to:\n{written}")
        else:
            self.progress_label.setText(
                f"Done: {len(df)} instance(s), {n_features} features.")
        self._set_running(False)

    def _on_failed(self, msg: str) -> None:
        if self._closing:
            return
        self._pending_save_path = None
        self._set_running(False)
        self.progress_label.setText(f"Failed: {msg}")
        QMessageBox.critical(self, "ELA run failed", msg)

    def _on_canceled(self) -> None:
        if self._closing:
            return
        self._pending_save_path = None
        self._set_running(False)
        self.stop_btn.setText("Stop ELA")
        self.progress_label.setText("Canceled. No ELA file was saved.")

    def _cleanup_thread(self) -> None:
        if self._thread is not None:
            self._thread.wait()
            self._thread.deleteLater()
            self._thread = None
        self._worker = None

    def shutdown(self) -> None:
        """Cooperatively cancel and wait before the panel is destroyed."""
        self._closing = True
        self._pending_save_path = None
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._cleanup_thread()
