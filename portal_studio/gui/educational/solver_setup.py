"""One side of the Education tab: controls + figures + run state for a single
solver/instance pairing. EducationTab composes one of these (single mode) or two
(comparison mode), plus a shared Run button and an FE-synced playback bar.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
from PySide6 import QtCore
from PySide6.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QSizePolicy, QSplitter,
    QVBoxLayout, QWidget,
)

from portal_studio.gui.common.widgets import (
    NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox,
)
from portal_studio.gui.common.icons import make_icon_button, make_reset_icon, make_save_icon
from portal_studio.gui.common.help import (
    help_body, make_help_button, show_help_dialog, term,
)
from portal_studio.gui.common.plot_style import style_compact_axes
from portal_studio.gui.common.theme import LIGHT
from portal_studio.gui.educational import figure_io
from portal_studio.algorithms.registry import (
    get_algorithm_meta,
    get_source_version,
    list_algorithm_sources,
    list_solvers,
)
from portal_studio.algorithms.solver_params import get_solver_params
from portal_studio.session import SessionInventory
from portal_core.fitness import fitness_batch

_NO_BUTTONS = QAbstractSpinBox.ButtonSymbols.NoButtons
_DEFAULT_ACCEPTANCE = 1e-8     # default "solved" error threshold (horizontal line)
_FE_BUDGET_PER_DIM = 10_000
_DEFAULT_BUDGET = 20_000
_FIXED_SEED = 42               # runs use a fixed seed (no user-facing seed control)


@lru_cache(maxsize=4)
def _cached_contour_grid(n: int, bounds_min: float, bounds_max: float) -> tuple:
    xs = np.linspace(float(bounds_min), float(bounds_max), int(n))
    ys = np.linspace(float(bounds_min), float(bounds_max), int(n))
    Xg, Yg = np.meshgrid(xs, ys)
    X = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    for arr in (Xg, Yg, X):
        arr.setflags(write=False)
    return Xg, Yg, X


class EducationSide:
    """Controls, figures, and run state for one solver/instance pairing."""

    def __init__(self, inventory: SessionInventory, title: str = "Setup A") -> None:
        self.inventory = inventory
        self._title = title
        self.tape = None
        self.global_optimum: Optional[float] = None
        self.tape_source = None
        self.current_frame: int = 0
        self.contour_ax = None
        self._particles = None
        self._best_marker = None
        self._error_marker = None
        self._contour_home_limits = None
        self._error_home_limits = None
        self._param_widgets: dict = {}
        self._playback_particle_window = 1
        self._budget_user_edited = False
        self._syncing_budget = False
        self.on_run_cleared = None   # tab sets this; called when the run is dropped

        self.controls = self._build_controls()
        self.plot_strip = self._build_plots()

        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.algorithm_source_combo.currentIndexChanged.connect(
            self._on_algorithm_source_changed)
        self.algorithm_combo.currentIndexChanged.connect(self.rebuild_param_panel)
        self.reset_params_btn.clicked.connect(self.reset_run_settings)

        self.rebuild_param_panel()
        self.refresh_sources()

    # ----------------------------------------------------------- construction
    def _section_card(self, title: str, help_handler=None) -> QFrame:
        """A soft-elevated white card with a title and a ``body`` layout.

        Callers attach widgets to the returned frame's ``body`` layout. When
        ``help_handler`` is given, a "?" help button sits next to the title and
        is exposed as ``card.help_btn``.
        """
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        card = QFrame()
        card.setObjectName("sectionCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        header = QLabel(title)
        header.setObjectName("panelTitle")   # match the Generator card headers
        card.title_label = header
        header_row.addWidget(header)
        card.help_btn = None
        if help_handler is not None:
            card.help_btn = make_help_button(f"Explain {title}", help_handler)
            header_row.addWidget(card.help_btn)
        header_row.addStretch(1)
        lay.addLayout(header_row)
        card.body = QVBoxLayout()
        card.body.setContentsMargins(0, 0, 0, 0)
        card.body.setSpacing(6)
        lay.addLayout(card.body)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(14)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(15, 23, 42, 32))   # ~12% alpha cool-grey
        card.setGraphicsEffect(shadow)
        return card

    def _sub_section(self, title: str, help_handler=None) -> QFrame:
        """A flat, borderless sub-section inside the setup card: a smaller
        header (slightly smaller than the card's "Setup A/B" title) over a
        ``body`` layout. Used so one elevated card holds both the Problem
        Instance and Algorithm Parameters groups instead of separate windows.
        """
        sub = QFrame()
        sub.setObjectName("setupSubSection")
        sub.setStyleSheet(
            "QFrame#setupSubSection { background: transparent; border: none; }")
        lay = QVBoxLayout(sub)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        header = QLabel(title)
        header.setObjectName("subSectionTitle")   # smaller than the card title
        sub.title_label = header
        header_row.addWidget(header)
        sub.help_btn = None
        if help_handler is not None:
            sub.help_btn = make_help_button(f"Explain {title}", help_handler)
            header_row.addWidget(sub.help_btn)
        header_row.addStretch(1)
        lay.addLayout(header_row)
        sub.body = QVBoxLayout()
        sub.body.setContentsMargins(0, 0, 0, 0)
        sub.body.setSpacing(6)
        lay.addLayout(sub.body)
        return sub

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("educationControls")
        panel.setStyleSheet(
            "QWidget#educationControls { background: transparent; }")
        v = QVBoxLayout(panel)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(10)

        # One elevated "window" card per setup, titled "Setup A"/"Setup B",
        # holding both sub-sections (Problem Instance + Algorithm Parameters)
        # so the column reads as a single card per setup instead of several.
        setup_card = self._section_card(self._title)
        self.setup_title = setup_card.title_label
        cv = setup_card.body
        cv.setSpacing(12)

        # ---- Sub-section: problem instance + algorithm selectors ----
        instance_section = self._sub_section(
            "Problem Instance", help_handler=self._show_problem_help)
        self.problem_help_btn = instance_section.help_btn
        iv = instance_section.body
        row = QHBoxLayout()
        lbl = QLabel("Problem instance:"); lbl.setFixedWidth(110)
        row.addWidget(lbl)
        self.source_combo = NoWheelComboBox()
        row.addWidget(self.source_combo, stretch=1)
        iv.addLayout(row)

        asrow = QHBoxLayout()
        aslbl = QLabel("Source:"); aslbl.setFixedWidth(110)
        asrow.addWidget(aslbl)
        self.algorithm_source_combo = NoWheelComboBox()
        self.algorithm_source_combo.addItems(list_algorithm_sources())
        asrow.addWidget(self.algorithm_source_combo, stretch=1)
        self.algorithm_source_help_btn = make_help_button(
            "Explain Source", self._show_algorithm_source_help)
        asrow.addWidget(self.algorithm_source_help_btn)
        iv.addLayout(asrow)

        arow = QHBoxLayout()
        albl = QLabel("Algorithm:"); albl.setFixedWidth(110)
        arow.addWidget(albl)
        self.algorithm_combo = NoWheelComboBox()
        self._refresh_algorithm_combo()
        arow.addWidget(self.algorithm_combo, stretch=1)
        iv.addLayout(arow)
        cv.addWidget(instance_section)

        # Hairline divider between the two sub-sections so the single card still
        # reads as two distinct groups.
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {LIGHT['card_border']};")
        cv.addWidget(divider)

        # ---- Sub-section: solver params + budget/seed/acceptance + reset ----
        params_section = self._sub_section("Algorithm Parameters")
        pv = params_section.body

        self.param_card = QFrame()
        # Flush inside the card (the card supplies the framing); no nested box.
        self.param_card.setObjectName("educationParamGrid")
        self.param_card.setStyleSheet(
            "QFrame#educationParamGrid { background: transparent; border: none; }")
        self.param_card.setSizePolicy(QSizePolicy.Policy.Preferred,
                                      QSizePolicy.Policy.Maximum)
        self.param_layout = QGridLayout(self.param_card)
        self.param_layout.setContentsMargins(0, 0, 0, 0)
        self.param_layout.setHorizontalSpacing(8)
        self.param_layout.setVerticalSpacing(10)
        self.param_layout.setColumnMinimumWidth(0, 90)
        self.param_layout.setColumnStretch(3, 1)
        pv.addWidget(self.param_card)

        # Budget row, wrapped in a widget so the Education tab can hide it and
        # provide a single shared FE budget for both comparison setups.
        self.budget_row = QWidget()
        brow = QHBoxLayout(self.budget_row)
        brow.setContentsMargins(0, 0, 0, 0)
        brow.addWidget(QLabel("Budget (FE):"))
        self.budget_spin = NoWheelSpinBox()
        self.budget_spin.setButtonSymbols(_NO_BUTTONS)
        self.budget_spin.setRange(10, 1_000_000)
        self.budget_spin.setValue(_DEFAULT_BUDGET)
        self.budget_spin.setMaximumWidth(72)   # keep the row within the fixed width
        self.budget_spin.valueChanged.connect(self._on_budget_changed)
        brow.addWidget(self.budget_spin)
        brow.addStretch(1)
        pv.addWidget(self.budget_row)

        self.reset_params_btn = QPushButton("Reset to defaults")
        self.reset_params_btn.setToolTip(
            "Restore the algorithm parameters to their default values.")
        pv.addWidget(self.reset_params_btn)
        cv.addWidget(params_section)
        v.addWidget(setup_card)

        v.addStretch(1)
        return panel

    def _show_help_dialog(self, title: str, body: str, formula: str | None = None) -> None:
        show_help_dialog(self.controls, title, body, formula)

    def _show_problem_help(self) -> None:
        body = help_body("Benchmarking controls", [
            f"{term('Problem instance')} is the menu that picks the landscape to "
            "optimize: choose a component or an instance you built. Build one "
            "first if the list is empty.",
            f"{term('Algorithm')} chooses the optimizer that runs on the selected "
            "problem; the settings below configure that run, and the charts "
            "replay its search over the chosen budget.",
            f"{term('Comparison mode')} splits the view into two sides so you can "
            "run a different instance/algorithm setup on each and compare their "
            "searches side by side.",
        ])
        self._show_help_dialog("Problem instance help", body)

    def _build_plots(self) -> QWidget:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FC
        from matplotlib.figure import Figure
        strip = QSplitter(QtCore.Qt.Orientation.Horizontal)
        # constrained_layout reserves room for axis labels/titles so they are
        # never clipped, even when the canvas is short.
        self.contour_fig = Figure(figsize=(5, 5), constrained_layout=True)
        self.contour_canvas = FC(self.contour_fig)
        self.contour_plot_panel = self._plot_panel(
            self.contour_canvas,
        )
        strip.addWidget(self.contour_plot_panel)
        self.error_fig = Figure(figsize=(5, 5), constrained_layout=True)
        self.error_canvas = FC(self.error_fig)
        self.error_plot_panel = self._plot_panel(
            self.error_canvas,
            save_attr="error_save_pdf_btn",
            save_tip="Save convergence figure as PDF",
            save_handler=lambda: self._save_figure_pdf(
                self.error_fig, "convergence_figure.pdf"),
        )
        strip.addWidget(self.error_plot_panel)
        strip.setSizes([500, 500])

        # Wrap the charts so a live best-fitness readout can sit beneath them.
        container = QWidget()
        cv = QVBoxLayout(container)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(2)
        cv.addWidget(strip, stretch=1)
        self.fitness_label = QLabel()
        self.fitness_label.setStyleSheet(f"font-weight: 600; color: {LIGHT['fitness_text']};")
        self.fitness_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        cv.addWidget(self.fitness_label)
        self.update_fitness_readout()
        return container

    def _plot_panel(self, canvas: QWidget, *, save_attr: str | None = None,
                    reset_attr: str | None = None, save_tip: str = "",
                    reset_tip: str = "", save_handler=None,
                    reset_handler=None) -> QWidget:
        panel = QWidget()
        panel.setObjectName("benchmarkPlotPanel")
        panel.setStyleSheet(
            f"QWidget#benchmarkPlotPanel {{ background: {LIGHT['surface']}; }}"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 8, 0)
        controls.addStretch(1)
        has_controls = False
        if save_handler is not None:
            save_btn = make_icon_button(
                make_save_icon(), save_tip, save_handler,
                size=26, object_name="plotActionBtn")
            controls.addWidget(save_btn)
            has_controls = True
            if save_attr is not None:
                setattr(self, save_attr, save_btn)
        if reset_handler is not None:
            reset_btn = make_icon_button(
                make_reset_icon(), reset_tip, reset_handler,
                size=26, object_name="plotActionBtn")
            controls.addWidget(reset_btn)
            has_controls = True
            if reset_attr is not None:
                setattr(self, reset_attr, reset_btn)
        if has_controls:
            layout.addLayout(controls)
        layout.addWidget(canvas, stretch=1)
        return panel

    def _save_figure_pdf(self, fig, default_name: str) -> None:
        figure_io.save_figure_pdf(self.controls, fig, default_name)

    @staticmethod
    def _capture_axes_limits(ax):
        if ax is None:
            return None
        return (tuple(ax.get_xlim()), tuple(ax.get_ylim()))

    @staticmethod
    def _restore_axes_limits(ax, limits) -> None:
        if ax is None or limits is None:
            return
        xlim, ylim = limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    def _reset_contour_view(self) -> None:
        self._restore_axes_limits(self.contour_ax, self._contour_home_limits)
        self.contour_canvas.draw_idle()

    def _reset_error_view(self) -> None:
        if not self.error_fig.axes:
            return
        self._restore_axes_limits(self.error_fig.axes[0], self._error_home_limits)
        self.error_canvas.draw_idle()

    # --------------------------------------------------------------- instances
    def refresh_sources(self) -> None:
        prev = self.source_combo.currentData()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        for i, comp in enumerate(self.inventory.staged_components):
            self.source_combo.addItem(comp.label, ("component", i))
        for i, blk in enumerate(self.inventory.staged_blocks):
            if blk.components:
                # Labelled "Instance_N" to match the 2-D Authoring tree, rather
                # than the internal "block_N" label.
                self.source_combo.addItem(f"Instance_{i + 1}", ("block", i))
        if prev is not None:
            for i in range(self.source_combo.count()):
                if self.source_combo.itemData(i) == prev:
                    self.source_combo.setCurrentIndex(i)
                    break
        self.source_combo.blockSignals(False)
        self._sync_default_budget()
        self._on_source_changed()

    def _on_source_changed(self) -> None:
        if self.tape is not None and self.source_combo.currentData() != self.tape_source:
            self.clear_run()
        self._sync_default_budget()
        self.render_contour()
        # pop_size defaults track the instance dimension, so refresh the panel.
        self.rebuild_param_panel()

    def resolve_instance(self) -> Optional[dict]:
        data = self.source_combo.currentData()
        if data is None:
            return None
        from portal_studio.builder.assembly import assemble_block, assemble_composite
        kind, idx = data
        if kind == "component":
            comp = self.inventory.staged_components[idx]
            blk = assemble_block(components=[comp], weight=1.0, label="(preview)",
                                 indices_strategy="auto", explicit_indices=None)
        else:
            blk = self.inventory.staged_blocks[idx]
            if not blk.components:
                return None
        return assemble_composite([blk])["blocks"][0]

    # ----------------------------------------------------------------- params
    def _current_source(self) -> Optional[str]:
        text = self.algorithm_source_combo.currentText().strip()
        return text or None

    def _refresh_algorithm_combo(self, preferred: str | None = None) -> None:
        current = preferred or self.algorithm_combo.currentText()
        names = list_solvers(source=self._current_source())
        self.algorithm_combo.blockSignals(True)
        self.algorithm_combo.clear()
        self.algorithm_combo.addItems(names)
        if current in names:
            self.algorithm_combo.setCurrentText(current)
        self._set_algorithm_combo_full_name_tooltips()
        self.algorithm_combo.blockSignals(False)
        if hasattr(self, "param_layout"):
            self.rebuild_param_panel()

    def _on_algorithm_source_changed(self, _index: int) -> None:
        self._refresh_algorithm_combo()

    def _set_algorithm_combo_full_name_tooltips(self) -> None:
        for index in range(self.algorithm_combo.count()):
            name = self.algorithm_combo.itemText(index)
            full_name = get_algorithm_meta(name).description or name
            self.algorithm_combo.setItemData(
                index, full_name, QtCore.Qt.ItemDataRole.ToolTipRole)

    def _show_algorithm_source_help(self) -> None:
        source = self._current_source() or "PORTAL"
        version = get_source_version(source)
        if version:
            detail = f"The installed {source} version is {version}."
        elif source == "PORTAL":
            detail = "Native PORTAL wrappers have no external package version."
        else:
            detail = f"{source} does not appear to be installed."
        body = help_body("Source", [
            f"{term('Source')} selects the origin of the algorithm "
            "implementation: native PORTAL wrappers, or an integrated "
            "third-party package.",
            detail,
        ])
        self._show_help_dialog("Source help", body)

    def rebuild_param_panel(self) -> None:
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._param_widgets = {}
        solver = self.algorithm_combo.currentText()
        specs = get_solver_params(solver)
        d = self._current_dimension()
        visible = []
        for spec in specs.values():
            # Auto (no-default) parameters other than pop_size are computed by
            # the algorithm itself from the problem (dimension/bounds); they are
            # not exposed for editing.
            if spec.default is None and spec.name != "pop_size":
                continue
            # pop_size: pre-fill the same dimension-based value the solver would
            # choose when left blank (see algorithms/*.py), still editable.
            if spec.name == "pop_size" and spec.default is None and d is not None:
                spec.default = self._auto_pop_size(solver, d)
                spec.kind = "int"
            visible.append(spec)
        if not visible:
            self.param_layout.addWidget(
                QLabel("This solver has no configurable parameters."), 0, 0, 1, 3)
            return
        for row, spec in enumerate(visible):
            label = QLabel(spec.name)
            label.setToolTip(spec.tooltip)
            self.param_layout.addWidget(label, row, 0)
            widget = self._make_param_widget(spec)
            widget.setToolTip(spec.tooltip)
            if not isinstance(widget, QCheckBox):
                widget.setFixedWidth(90)
            self.param_layout.addWidget(widget, row, 1)
            self._param_widgets[spec.name] = (spec.kind, widget)

    def _current_dimension(self) -> Optional[int]:
        """Dimension d of the currently selected instance, or None."""
        inst = self.resolve_instance()
        if inst is None:
            return None
        try:
            return int(inst.get("d"))
        except (TypeError, ValueError):
            return None

    def _on_budget_changed(self, _value: int) -> None:
        if not self._syncing_budget:
            self._budget_user_edited = True

    def _sync_default_budget(self) -> None:
        if self._budget_user_edited:
            return
        d = self._current_dimension()
        budget = _DEFAULT_BUDGET if d is None else _FE_BUDGET_PER_DIM * d
        budget = max(
            self.budget_spin.minimum(),
            min(self.budget_spin.maximum(), int(budget)),
        )
        self._syncing_budget = True
        try:
            self.budget_spin.setValue(budget)
        finally:
            self._syncing_budget = False

    @staticmethod
    def _auto_pop_size(solver: str, d: int) -> int:
        """Population size a solver picks for a blank pop_size, by dimension.

        Mirrors each solver's own fallback in algorithms/*.py: jDE uses 10*d;
        BBPSO/FIPS/QPSO cap it at 50 (min(10*d, 50)).
        """
        if solver == "jDE":
            return 10 * d
        return min(10 * d, 50)

    def _make_param_widget(self, spec) -> QWidget:
        if spec.default is None:
            w = QLineEdit()
            w.setPlaceholderText("auto")
            return w
        if spec.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(spec.default))
            return w
        if spec.kind == "choice":
            w = NoWheelComboBox()
            w.addItems([str(c) for c in (spec.choices or [])])
            if spec.default is not None:
                w.setCurrentText(str(spec.default))
            return w
        if spec.kind == "int":
            w = NoWheelSpinBox()
            w.setButtonSymbols(_NO_BUTTONS)
            w.setRange(int(spec.min) if spec.min is not None else 0,
                       int(spec.max) if spec.max is not None else 1_000_000)
            w.setSingleStep(int(spec.step) if spec.step else 1)
            if spec.default is not None:
                w.setValue(int(spec.default))
            return w
        if spec.min is not None and spec.max is not None:
            w = NoWheelDoubleSpinBox()
            w.setButtonSymbols(_NO_BUTTONS)
            w.setDecimals(spec.decimals if spec.decimals is not None else 4)
            w.setRange(float(spec.min), float(spec.max))
            w.setSingleStep(float(spec.step) if spec.step else 0.1)
            if spec.default is not None:
                w.setValue(float(spec.default))
            return w
        w = QLineEdit()
        w.setPlaceholderText("number")
        if spec.default is not None:
            w.setText(repr(spec.default))
        return w

    def collect_params(self) -> dict:
        params: dict = {}
        for name, (kind, widget) in self._param_widgets.items():
            if isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if not text:
                    continue
                params[name] = int(float(text)) if kind == "int" else float(text)
            elif kind == "bool":
                params[name] = bool(widget.isChecked())
            elif kind == "choice":
                params[name] = widget.currentText()
            elif kind == "int":
                params[name] = int(widget.value())
            else:
                params[name] = float(widget.value())
        return params

    @staticmethod
    def _particle_window_from_params(params: dict) -> int:
        try:
            return max(1, int(params.get("pop_size", 1)))
        except (TypeError, ValueError):
            return 1

    # ----------------------------------------------------------------- contour
    def render_contour(self) -> None:
        inst = self.resolve_instance()
        self.contour_fig.clear()
        self.contour_ax = None
        self._particles = None
        self._best_marker = None
        if inst is None:
            ax = self.contour_fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5,
                    "No problem instance yet.\n\n"
                    "There are currently no components or blocks built in the "
                    "Builder tab.\nBuild one there first, then pick it here.",
                    ha="center", va="center", wrap=True, fontsize=11, color="#667085")
            self.contour_ax = ax
            self._contour_home_limits = self._capture_axes_limits(ax)
            self.contour_canvas.draw_idle()
            return
        ax = self.contour_fig.add_subplot(111)
        Xg, Yg, X = _cached_contour_grid(
            100,
            float(inst["bounds_min"]),
            float(inst["bounds_max"]),
        )
        Z = np.asarray(
            fitness_batch(X, inst, evaluation_device="numpy"),
            dtype=float,
        ).reshape(Xg.shape)
        ax.contour(Xg, Yg, Z, levels=25, cmap="jet")
        ax.set_aspect("equal")
        ax.set_title(self.source_combo.currentText())
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        style_compact_axes(ax)
        self.contour_ax = ax
        self._contour_home_limits = self._capture_axes_limits(ax)
        self.contour_canvas.draw_idle()

    # --------------------------------------------------------------------- run
    def run(self, progress=None) -> bool:
        """Compute the tape for the current selection. Returns True on success."""
        from portal_studio.algorithms.runner import (
            run_solver_on_instance, RunCanceled)
        inst = self.resolve_instance()
        if inst is None:
            return False
        params = self.collect_params()
        try:
            result = run_solver_on_instance(
                instance=inst,
                solver_name=self.algorithm_combo.currentText(),
                budget=int(self.budget_spin.value()),
                seed=_FIXED_SEED,
                params=params,
                progress=progress,
            )
        except RunCanceled:
            return False           # user cancelled — leave the prior run untouched
        except Exception as exc:   # noqa: BLE001 - surface solver/config errors
            QMessageBox.critical(self.controls, "Run failed", str(exc))
            return False
        self.tape = result.problem.tape
        self.global_optimum = result.global_optimum
        self.tape_source = self.source_combo.currentData()
        self._playback_particle_window = self._particle_window_from_params(params)
        self.current_frame = 0
        self.render_contour()
        self.render_error_chart()
        self.update_fitness_readout()
        return True

    def build_run_request(self, request_index: int, setup_label: str,
                          budget: int) -> Optional[dict]:
        """Serializable solver request for EducationTab's process pool."""
        inst = self.resolve_instance()
        if inst is None:
            return None
        params = self.collect_params()
        return {
            "request_index": int(request_index),
            "setup_label": str(setup_label),
            "instance": inst,
            "solver_name": self.algorithm_combo.currentText(),
            "budget": int(budget),
            "seed": _FIXED_SEED,
            "params": params,
            "playback_particle_window": self._particle_window_from_params(params),
            "tape_source": self.source_combo.currentData(),
        }

    def apply_run_result(self, result: dict) -> None:
        """Install a worker result into this side and redraw its charts."""
        self.tape = result["tape"]
        self.global_optimum = result["global_optimum"]
        self.tape_source = result.get("tape_source", self.source_combo.currentData())
        self._playback_particle_window = int(
            result.get("playback_particle_window", 1) or 1
        )
        self.current_frame = 0
        self.render_contour()
        self.render_error_chart()
        self.update_fitness_readout()

    def clear_run(self) -> None:
        self.tape = None
        self.global_optimum = None
        self.tape_source = None
        self._playback_particle_window = 1
        self.current_frame = 0
        self.render_contour()
        self.render_error_chart()
        self.update_fitness_readout()
        if self.on_run_cleared is not None:
            self.on_run_cleared()

    def reset_run_settings(self) -> None:
        """Reset the run settings (parameters, budget, seed) to defaults,
        keeping the chosen instance and algorithm; drops any run.
        This is what the per-side 'Reset to defaults' button does."""
        self._budget_user_edited = False
        self._sync_default_budget()
        self.rebuild_param_panel()
        self.clear_run()

    def reset_to_defaults(self) -> None:
        """Full reset: instance and algorithm to the first option plus the run
        settings. Used when Comparison mode turns on."""
        if self.source_combo.count():
            self.source_combo.setCurrentIndex(0)
        self.algorithm_combo.setCurrentIndex(0)
        self.reset_run_settings()

    def acceptance_threshold(self) -> float:
        """The 'solved' error threshold drawn on the convergence chart.

        Driven by the Education tab's shared acceptance control; falls back to
        the internal default until one is set.
        """
        return getattr(self, "_acceptance", _DEFAULT_ACCEPTANCE)

    def set_acceptance_threshold(self, value: float) -> None:
        """Set the 'solved' error threshold (from the shared Education control)."""
        self._acceptance = float(value)

    def max_fe(self) -> int:
        return self.tape[-1].fe_count if self.tape else 0

    def update_fitness_readout(self) -> None:
        """Show the best-so-far fitness at the current frame (— when no run)."""
        if self.tape:
            best_f = float(self.tape[self.current_frame].best_f)
            self.fitness_label.setText(f"Best fitness: {best_f:.4g}")
        else:
            self.fitness_label.setText("Best fitness: —")

    # ------------------------------------------------------ frame lookup + draw
    def frame_index_at_fe(self, fe: int) -> int:
        """Index of the last frame whose fe_count <= fe (0 if none)."""
        if not self.tape:
            return 0
        idx = 0
        for i, f in enumerate(self.tape):
            if f.fe_count <= fe:
                idx = i
            else:
                break
        return idx

    @staticmethod
    def _safe_remove(artist) -> None:
        """Remove a matplotlib artist if present, ignoring already-removed ones."""
        if artist is not None:
            try:
                artist.remove()
            except Exception:
                pass

    def _display_positions_for_frame(self, frame_index: int) -> np.ndarray:
        frame = self.tape[frame_index]
        positions = np.asarray(frame.positions, dtype=float).reshape(-1, 2)
        window = max(1, int(self._playback_particle_window or 1))
        if len(positions) != 1 or window == 1:
            return positions
        start = max(0, int(frame_index) - window + 1)
        chunks = [
            np.asarray(f.positions, dtype=float).reshape(-1, 2)
            for f in self.tape[start:int(frame_index) + 1]
        ]
        return np.vstack(chunks)

    def draw_at_fe(self, fe: int) -> None:
        if not self.tape or self.contour_ax is None:
            return
        self.current_frame = self.frame_index_at_fe(fe)
        frame = self.tape[self.current_frame]
        positions = self._display_positions_for_frame(self.current_frame)
        self._safe_remove(self._particles)
        self._particles = self.contour_ax.scatter(
            positions[:, 0], positions[:, 1],
            c="limegreen", edgecolors="black", linewidths=0.4, s=40, zorder=5)
        if frame.best_x is not None:
            self._safe_remove(self._best_marker)
            self._best_marker = self.contour_ax.scatter(
                [frame.best_x[0]], [frame.best_x[1]],
                marker="*", c="red", s=180, zorder=6)
        self.contour_canvas.draw_idle()
        self._move_error_marker_to(fe)
        self.update_fitness_readout()

    # ----------------------------------------------------------- error chart
    def render_error_chart(self) -> None:
        # Current Error vs FE = best-so-far absolute minimization error in a
        # static landscape: |B_t - F*|.
        self.error_fig.clear()
        self._error_marker = None
        if not self.tape:
            self.error_canvas.draw_idle()
            return
        ax = self.error_fig.add_subplot(111)
        ax.set_box_aspect(1)
        fes = [f.fe_count for f in self.tape]
        errs = [abs(float(f.best_f) - self.global_optimum) for f in self.tape]
        ax.plot(fes, errs, color="#2e86ab", linewidth=1.5)
        ax.set_yscale("symlog")
        ax.set_title("Convergence")
        ax.set_xlabel("Fitness Evaluation")
        ax.set_ylabel("Current Error")
        style_compact_axes(ax)
        ax.grid(True)
        # Choose approximately five FE ticks based on the evaluation budget.
        from matplotlib.ticker import FuncFormatter, MultipleLocator
        step = self._nice_fe_step(fes[-1])
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.xaxis.set_major_formatter(FuncFormatter(self._fmt_fe))

        # Faint vertical reference line at each FE checkpoint (one per tick step).
        for cp in range(step, fes[-1] + 1, step):
            ax.axvline(cp, color="#b0b7c3", linestyle=":", linewidth=0.8, zorder=0)
        # Acceptance threshold: error below this line counts as 'solved'.
        thr = self.acceptance_threshold()
        ax.axhline(thr, color="#2eb872", linestyle="--", linewidth=1.0,
                   label=f"acceptance ({thr:g})")
        ax.legend(loc="upper right", fontsize=8)

        self._error_marker = ax.axvline(fes[0], color="red", linewidth=1.0, alpha=0.8)
        self._error_home_limits = self._capture_axes_limits(ax)
        self.error_canvas.draw_idle()

    @staticmethod
    def _nice_fe_step(max_fe: int) -> int:
        """A readable FE tick interval (~5 ticks) for a budget of ``max_fe``."""
        import math
        if max_fe <= 5:
            return 1
        raw = max_fe / 5.0
        mag = 10 ** int(math.floor(math.log10(raw)))
        for m in (1, 2, 5):
            if raw <= m * mag:
                return int(m * mag)
        return int(10 * mag)

    @staticmethod
    def _fmt_fe(value, _pos=None) -> str:
        """Format an FE tick: 0, 5000 -> '5k', 20000 -> '20k', 300 -> '300'."""
        if value == 0:
            return "0"
        if abs(value) >= 1000:
            return f"{value / 1000:g}k"
        return f"{int(value)}"

    def _move_error_marker_to(self, fe: int) -> None:
        if self._error_marker is None:
            return
        fe = min(fe, self.max_fe())     # park on the last point once finished
        self._error_marker.set_xdata([fe, fe])
        self.error_canvas.draw_idle()
