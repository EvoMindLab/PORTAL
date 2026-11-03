"""Education benchmarking tab with optional side-by-side comparison mode.

Hosts one EducationSide in single mode or two in comparison mode, plus a shared
Run button and an FE-synced playback bar.
"""
from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from PySide6 import QtCore
from PySide6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from portal_studio.gui.educational.solver_setup import (
    EducationSide, _DEFAULT_BUDGET, _DEFAULT_ACCEPTANCE, _FE_BUDGET_PER_DIM)
from portal_studio.gui.common.widgets import NoWheelComboBox, NoWheelSpinBox
from portal_studio.gui.common.help import help_body, make_help_button, show_help_dialog
from portal_studio.gui.common.theme import LIGHT
from portal_studio.session import SessionInventory

_SPEEDS = {"1x": 1, "2x": 2, "5x": 5, "10x": 10, "20x": 20, "50x": 50}
_BASE_INTERVAL_MS = 200      # 1x timer interval; divided by the speed multiplier.
_FE_STEPS = 200              # ~200 timer ticks span the whole FE range.


def _run_education_solver_job(request: dict) -> dict:
    """Process-pool worker for one Education benchmarking setup."""
    from portal_studio.algorithms.runner import run_solver_on_instance

    result = run_solver_on_instance(
        instance=request["instance"],
        solver_name=request["solver_name"],
        budget=int(request["budget"]),
        seed=int(request["seed"]),
        params=dict(request.get("params") or {}),
        progress=None,
    )
    return {
        "request_index": int(request["request_index"]),
        "setup_label": request.get("setup_label", ""),
        "tape": result.problem.tape,
        "global_optimum": result.global_optimum,
        "playback_particle_window": int(
            request.get("playback_particle_window", 1) or 1
        ),
        "tape_source": request.get("tape_source"),
    }


def _run_education_requests_in_processes(
    requests: list[dict],
    *,
    max_workers: int,
    poll_events=None,
    is_canceled=None,
    progress=None,
) -> list[dict]:
    """Run Education comparison requests concurrently in worker processes."""
    if not requests:
        return []
    total = len(requests)
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        pending = {
            pool.submit(_run_education_solver_job, request): request
            for request in requests
        }
        while pending:
            if is_canceled is not None and is_canceled():
                pool.shutdown(wait=False, cancel_futures=True)
                return []
            done, _not_done = wait(
                tuple(pending), timeout=0.05, return_when=FIRST_COMPLETED)
            if poll_events is not None:
                poll_events()
            for future in done:
                request = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    # One failed setup must not discard its successful peer.
                    result = {
                        "request_index": int(request["request_index"]),
                        "setup_label": request.get("setup_label", ""),
                        "error": (f"{request['solver_name']} (seed {request['seed']}): "
                                  f"{type(exc).__name__}: {exc}"),
                    }
                results.append(result)
                if progress is not None:
                    progress(len(results), total, result.get("setup_label", ""))
    return sorted(results, key=lambda item: int(item.get("request_index", 0)))


class _EducationRunWorker(QtCore.QObject):
    """Runs Education benchmarking requests away from the GUI thread."""

    progress = QtCore.Signal(int, int, str)
    finished = QtCore.Signal(object)
    canceled = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, requests: list[dict], *, max_workers: int) -> None:
        super().__init__()
        self._requests = list(requests)
        self._max_workers = max(1, int(max_workers))
        self._canceled = False

    @QtCore.Slot()
    def run(self) -> None:
        try:
            results = _run_education_requests_in_processes(
                self._requests,
                max_workers=self._max_workers,
                is_canceled=lambda: self._canceled,
                progress=self.progress.emit,
            )
        except Exception as exc:  # noqa: BLE001 - surface worker errors to GUI
            self.failed.emit(str(exc))
            return
        if self._canceled:
            self.canceled.emit()
            return
        self.finished.emit(results)

    def cancel(self) -> None:
        self._canceled = True


class EducationTab(QWidget):
    """One or two EducationSides + shared Run and FE-synced playback."""

    def __init__(self, inventory: SessionInventory, parent=None) -> None:
        super().__init__(parent)
        self.inventory = inventory
        self.current_fe: int = 0
        self._run_canceled = False
        self._run_thread = None
        self._run_worker = None
        self._closing = False
        self._run_dialog_ref = None
        self._run_runnable = []
        self._run_budget = 0
        self._run_total_fe = 0
        self._budget_user_edited = False
        self._syncing_budget = False
        self.sides = [EducationSide(inventory, "Setup A"),
                      EducationSide(inventory, "Setup B")]

        outer = QVBoxLayout(self)

        # The main window places this next to the tour/help toolbar action.
        from portal_studio.gui.common.icons import make_back_arrow_icon
        self.back_btn = QPushButton(self)
        self.back_btn.setIcon(make_back_arrow_icon())
        self.back_btn.setFixedSize(32, 28)
        self.back_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.back_btn.setStyleSheet(
            "QPushButton {"
            f"  background: {LIGHT['back_btn_bg']};"
            f"  border: 1px solid {LIGHT['back_btn_border']};"
            "  border-radius: 6px;"
            "}"
            f"QPushButton:hover {{ background: {LIGHT['back_btn_bg_hover']}; }}"
            f"QPushButton:pressed {{ background: {LIGHT['back_btn_bg_pressed']}; }}"
        )
        self.back_btn.setToolTip(
            "Back to 2-D Authoring - return to the component/instance "
            "authoring view.")
        # Comparison-mode toggle is constructed here (the left panel adds it
        # to its layout below) so signal wiring later in __init__ stays in
        # one place.
        self.comparison_check = QPushButton("⇆  Comparison mode")
        self.comparison_check.setCheckable(True)
        self.comparison_check.setProperty("variant", "toggle")
        self.comparison_check.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.comparison_check.setToolTip(
            "Compare two solver/instance setups side by side.")

        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([300, 1050])
        outer.addWidget(splitter, stretch=1)

        self.comparison_check.toggled.connect(self._on_comparison_toggled)
        self.run_btn.clicked.connect(self.on_run_clicked)
        self.play_btn.clicked.connect(self.on_play_clicked)
        self.reset_btn.clicked.connect(self.on_reset_clicked)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(_BASE_INTERVAL_MS // _SPEEDS[self.speed_combo.currentText()])
        self._timer.timeout.connect(self._advance)

        for side in self.sides:
            side.on_run_cleared = self._on_side_run_cleared
            side.source_combo.currentIndexChanged.connect(self._sync_default_budget)

        self._on_comparison_toggled(False)   # start in single mode
        self._sync_default_budget()

    # ------------------------------------------------------------------ layout
    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(330)              # fixed in both modes; no horizontal scroll
        v = QVBoxLayout(panel)
        v.setContentsMargins(4, 6, 4, 6)
        v.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        bv = QVBoxLayout(body)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(6)
        # Each side is one elevated card titled "Setup A"/"Setup B"; the title
        # lives inside the card (side.setup_title) and is shown only in
        # comparison mode (see _on_comparison_toggled).
        bv.addWidget(self.sides[0].controls)
        bv.addWidget(self.sides[1].controls)
        bv.addStretch(1)
        scroll.setWidget(body)
        v.addWidget(scroll, stretch=1)

        # The per-side budget controls are hidden; a single shared FE budget
        # (below) governs every setup so the user can't set different budgets.
        for side in self.sides:
            side.budget_row.setVisible(False)

        # Shared acceptance threshold and FE budget, laid out on a grid so the
        # two input boxes line up in a column regardless of label width. A run's
        # error below the acceptance value counts as solved (the chart's line).
        controls = QGridLayout()
        controls.setColumnStretch(3, 1)

        controls.addWidget(QLabel("Acceptance rate:"), 0, 0)
        self.acceptance_edit = QLineEdit(self._format_acceptance(_DEFAULT_ACCEPTANCE))
        self.acceptance_edit.setMaximumWidth(96)
        self.acceptance_edit.setToolTip(
            "The 'solved' error threshold: a run whose error drops below this "
            "counts as solved (the green acceptance line on the chart). "
            "Default 1e-8.")
        controls.addWidget(self.acceptance_edit, 0, 1)
        self.acceptance_help_btn = make_help_button(
            "Explain the acceptance rate", self._show_acceptance_help)
        controls.addWidget(self.acceptance_help_btn, 0, 2)

        controls.addWidget(QLabel("Budget (FE):"), 1, 0)
        self.budget_spin = NoWheelSpinBox()
        self.budget_spin.setRange(10, 1_000_000)
        self.budget_spin.setValue(_DEFAULT_BUDGET)
        self.budget_spin.setMaximumWidth(96)
        self.budget_spin.valueChanged.connect(self._on_budget_changed)
        controls.addWidget(self.budget_spin, 1, 1)
        self.budget_help_btn = make_help_button(
            "Explain the FE budget", self._show_budget_help)
        controls.addWidget(self.budget_help_btn, 1, 2)
        v.addLayout(controls)

        # Keep Run and Comparison mode adjacent at the bottom of the column.
        self.run_btn = QPushButton("Run")
        self.run_btn.setProperty("variant", "primary")
        v.addWidget(self.run_btn)
        v.addWidget(self.comparison_check)
        return panel

    @staticmethod
    def _format_acceptance(value: float) -> str:
        return f"{value:g}"

    def _acceptance_value(self) -> float:
        """The shared acceptance threshold, or the default if the field is blank
        or not a positive number."""
        try:
            v = float(self.acceptance_edit.text())
        except (ValueError, TypeError):
            v = 0.0
        return v if v > 0 else _DEFAULT_ACCEPTANCE

    def _show_acceptance_help(self) -> None:
        show_help_dialog(
            self, "Acceptance rate help",
            help_body("Acceptance rate", [
                "The <b>Acceptance rate</b> field is an absolute-error threshold, "
                "not a proportion of successful runs. "
                "Once a run's best error drops below this value, the problem "
                "counts as solved, and the chart draws a green acceptance line "
                "at this level.",
                "A smaller value (for example <b>1e-8</b>) is a stricter target "
                "and is harder to reach; a larger value is easier.",
                "In <b>Comparison mode</b> this single threshold applies to "
                "<b>both setups</b>, so the two runs are judged solved by the "
                "same standard.",
            ]))

    def _show_budget_help(self) -> None:
        show_help_dialog(
            self, "FE budget help",
            help_body("Budget (FE)", [
                "The <b>function-evaluation budget</b> is how many fitness "
                "evaluations each algorithm run is allowed before it stops. A "
                "larger budget lets the search run longer. "
                "The budget initially follows 10,000 times the problem dimension; "
                "editing it stops automatic updates. Education uses optimiser "
                "seed 42 with no seed control.",
                "In <b>Comparison mode</b> this single budget is used by "
                "<b>both setups</b>, so the two runs are compared over the same "
                "number of evaluations.",
            ]))

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        plots = QSplitter(QtCore.Qt.Orientation.Vertical)
        plots.addWidget(self.sides[0].plot_strip)
        plots.addWidget(self.sides[1].plot_strip)
        plots.setSizes([500, 500])
        v.addWidget(plots, stretch=1)

        bar = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.reset_btn = QPushButton("Reset")
        bar.addWidget(self.play_btn)
        bar.addWidget(self.reset_btn)
        bar.addWidget(QLabel("Speed:"))
        self.speed_combo = NoWheelComboBox()
        self.speed_combo.addItems(list(_SPEEDS.keys()))
        self.speed_combo.setCurrentText("10x")
        bar.addWidget(self.speed_combo)
        self.frame_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        bar.addWidget(self.frame_slider, stretch=1)
        self.fe_label = QLabel("FE: 0/0")
        bar.addWidget(self.fe_label)
        v.addLayout(bar)
        return panel

    # ------------------------------------------------------------------- modes
    def active_sides(self):
        if self.comparison_check.isChecked():
            return list(self.sides)
        return [self.sides[0]]

    def _on_comparison_toggled(self, on: bool) -> None:
        # The "Setup A"/"Setup B" card titles only matter when comparing. Side
        # B's title hides with its whole card; Side A's must be toggled directly.
        self.sides[0].setup_title.setVisible(on)
        self.sides[1].controls.setVisible(on)
        self.sides[1].plot_strip.setVisible(on)
        if on:
            # Fresh start for the comparison: reset both sides to defaults so the
            # previous single-mode setup/results don't carry over.
            for side in self.sides:
                side.refresh_sources()
                side.reset_to_defaults()
        else:
            self.sides[1].clear_run()
        self._sync_default_budget()

    def refresh_sources(self) -> None:
        """Forward to the active sides (called by main_window on tab-switch)."""
        for side in self.active_sides():
            side.refresh_sources()
        self._sync_default_budget()

    def _on_budget_changed(self, _value: int) -> None:
        if not self._syncing_budget:
            self._budget_user_edited = True

    def _sync_default_budget(self, *_args) -> None:
        if self._budget_user_edited:
            return
        dims = [
            side._current_dimension()
            for side in self.active_sides()
            if side.source_combo.currentData() is not None
        ]
        dims = [d for d in dims if d is not None]
        budget = (
            _DEFAULT_BUDGET if not dims
            else _FE_BUDGET_PER_DIM * max(int(d) for d in dims)
        )
        budget = max(
            self.budget_spin.minimum(),
            min(self.budget_spin.maximum(), int(budget)),
        )
        self._syncing_budget = True
        try:
            self.budget_spin.setValue(budget)
        finally:
            self._syncing_budget = False

    # --------------------------------------------------------------------- run
    def _running_dialog(self, total_fe: int) -> "QDialog":
        from PySide6.QtWidgets import QProgressBar
        dlg = QDialog(self)
        dlg.setWindowTitle("Running")
        dlg.setModal(True)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(14)
        msg = QLabel(
            "Running the algorithm — please wait.\n\n"
            "The search is computing for the selected budget; this can take a "
            "few seconds for large FE budgets. This window closes automatically "
            "when the run finishes.")
        msg.setWordWrap(True)
        msg.setMinimumWidth(340)
        lay.addWidget(msg)
        self._run_progress_label = QLabel("Preparing run...")
        self._run_progress_label.setStyleSheet(f"color: {LIGHT['progress_label']};")
        lay.addWidget(self._run_progress_label)
        bar = QProgressBar()
        bar.setRange(0, max(1, int(total_fe)))
        bar.setValue(0)
        lay.addWidget(bar)
        self._run_progress_bar = bar

        # Cancel requests cooperative cancellation through the active worker.
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._run_cancel_btn = QPushButton("Cancel")
        self._run_cancel_btn.clicked.connect(self._on_cancel_run)
        btn_row.addWidget(self._run_cancel_btn)
        lay.addLayout(btn_row)
        return dlg

    def _on_cancel_run(self) -> None:
        self._run_canceled = True
        self._run_cancel_btn.setEnabled(False)
        if self._run_worker is not None:
            self._run_worker.cancel()
        self._run_cancel_btn.setText("Cancelling…")

    def _update_run_progress(self, fe: int, total_fe: int, label: str) -> None:
        if not hasattr(self, "_run_progress_bar"):
            return
        total = max(1, int(total_fe))
        value = max(0, min(int(fe), total))
        self._run_progress_bar.setRange(0, total)
        self._run_progress_bar.setValue(value)
        self._run_progress_label.setText(f"{label}: FE {value}/{total}")

    def on_run_clicked(self) -> None:
        from PySide6.QtWidgets import QApplication
        self._timer.stop()
        self.play_btn.setText("Play")
        if self._run_thread is not None:
            return
        runnable = [s for s in self.active_sides()
                    if s.source_combo.currentData() is not None]
        if not runnable:
            return   # nothing selected to run
        # One shared FE budget and acceptance threshold govern every setup.
        budget = int(self.budget_spin.value())
        accept = self._acceptance_value()
        for s in runnable:
            s.budget_spin.setValue(budget)
            s.set_acceptance_threshold(accept)
        # Display the progress dialog before starting the background worker.
        self._run_canceled = False
        total_fe = budget * len(runnable)
        dlg = self._running_dialog(total_fe)
        dlg.show()
        QApplication.processEvents()
        requests = []
        for idx, side in enumerate(runnable):
            setup_label = f"Setup {chr(ord('A') + idx)}"
            request = side.build_run_request(idx, setup_label, budget)
            if request is not None:
                requests.append(request)
        if not requests:
            dlg.close()
            dlg.deleteLater()
            return
        self._start_background_run(requests, runnable, budget, total_fe, dlg)

    def _start_background_run(self, requests: list[dict], runnable: list,
                              budget: int, total_fe: int, dlg) -> None:
        if self._closing:
            return
        thread = QtCore.QThread(self)
        worker = _EducationRunWorker(
            requests,
            max_workers=min(len(requests), os.cpu_count() or 1),
        )
        worker.moveToThread(thread)
        self._run_thread = thread
        self._run_worker = worker
        self._run_dialog_ref = dlg
        self._run_runnable = list(runnable)
        self._run_budget = int(budget)
        self._run_total_fe = int(total_fe)
        self.run_btn.setEnabled(False)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_background_run_progress)
        worker.finished.connect(self._on_background_run_finished)
        worker.canceled.connect(self._on_background_run_canceled)
        worker.failed.connect(self._on_background_run_failed)
        worker.finished.connect(thread.quit, QtCore.Qt.ConnectionType.DirectConnection)
        worker.canceled.connect(thread.quit, QtCore.Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, QtCore.Qt.ConnectionType.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_background_run_refs)
        thread.start()

    @QtCore.Slot(int, int, str)
    def _on_background_run_progress(self, completed: int, _total: int,
                                    label: str) -> None:
        if self._closing:
            return
        self._update_run_progress(
            int(completed) * int(self._run_budget),
            int(self._run_total_fe),
            label,
        )

    @QtCore.Slot(object)
    def _on_background_run_finished(self, results: list[dict]) -> None:
        if getattr(self, '_closing', False):
            return
        self._close_run_dialog()
        self.run_btn.setEnabled(True)
        if self._run_canceled:
            return
        ran = []
        errors = []
        runnable = list(self._run_runnable)
        for result in results:
            idx = int(result["request_index"])
            if 0 <= idx < len(runnable):
                if "error" in result:
                    runnable[idx].clear_run()
                    errors.append(f"{result.get('setup_label', '')}: {result['error']}")
                    continue
                runnable[idx].apply_run_result(result)
                ran.append(runnable[idx])
        if errors:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Run errors", "\n\n".join(errors))
        if not ran:
            return
        self.current_fe = 0
        max_fe = max(s.max_fe() for s in ran)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max_fe)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self._draw_all_at_fe(0)
        self._refresh_fe_label()
        self.on_play_clicked()   # auto-start

    @QtCore.Slot()
    def _on_background_run_canceled(self) -> None:
        if self._closing:
            return
        self._close_run_dialog()
        self.run_btn.setEnabled(True)

    @QtCore.Slot(str)
    def _on_background_run_failed(self, message: str) -> None:
        if self._closing:
            return
        from PySide6.QtWidgets import QMessageBox

        self._close_run_dialog()
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Run failed", message)

    def _close_run_dialog(self) -> None:
        dlg = self._run_dialog_ref
        self._run_dialog_ref = None
        if dlg is not None:
            dlg.close()
            dlg.deleteLater()

    @QtCore.Slot()
    def _clear_background_run_refs(self) -> None:
        if self._run_thread is not None:
            self._run_thread.wait()
        self._run_thread = None
        self._run_worker = None
        self._run_runnable = []
        self._run_budget = 0
        self._run_total_fe = 0

    def shutdown(self) -> None:
        """Wait for in-flight solver work before releasing its GUI owner."""
        self._closing = True
        self._run_canceled = True
        self._timer.stop()
        if self._run_worker is not None:
            self._run_worker.cancel()
        if self._run_thread is not None:
            self._run_thread.quit()
            self._run_thread.wait()
            self._clear_background_run_refs()
        self._close_run_dialog()

    def _draw_all_at_fe(self, fe: int) -> None:
        for s in self.active_sides():
            s.draw_at_fe(fe)

    def _on_side_run_cleared(self) -> None:
        # A side dropped its run (e.g. its instance changed). Stop shared
        # playback and reset the slider so it never runs against a stale tape.
        self._timer.stop()
        self.play_btn.setText("Play")
        self.current_fe = 0
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self._refresh_fe_label()

    def _refresh_fe_label(self) -> None:
        self.fe_label.setText(f"FE: {self.current_fe}/{self.frame_slider.maximum()}")

    # ---------------------------------------------------------------- playback
    def on_play_clicked(self) -> None:
        if self.frame_slider.maximum() <= 0:
            return
        if self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("Play")
        else:
            if self.current_fe >= self.frame_slider.maximum():
                self.current_fe = 0
                self.frame_slider.setValue(0)
            self._timer.start()
            self.play_btn.setText("Pause")

    def on_reset_clicked(self) -> None:
        self._timer.stop()
        self.play_btn.setText("Play")
        self.current_fe = 0
        self.frame_slider.setValue(0)
        self._draw_all_at_fe(0)
        self._refresh_fe_label()

    def _advance(self) -> None:
        max_fe = self.frame_slider.maximum()
        if max_fe <= 0:
            self._timer.stop()
            return
        step = max(1, max_fe // _FE_STEPS)
        nxt = self.current_fe + step
        if nxt >= max_fe:
            self._timer.stop()
            self.play_btn.setText("Play")
            self.frame_slider.setValue(max_fe)
            return
        self.frame_slider.setValue(nxt)   # drives _on_slider_changed

    def _on_slider_changed(self, fe: int) -> None:
        self.current_fe = int(fe)
        self._draw_all_at_fe(self.current_fe)
        self._refresh_fe_label()

    def _on_speed_changed(self, _text: str) -> None:
        self._timer.setInterval(_BASE_INTERVAL_MS // _SPEEDS[self.speed_combo.currentText()])
