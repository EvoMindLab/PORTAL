"""Column 1 — Task Settings.

Single-task author surface styled like the Education benchmarking setup card:

    Task Settings <card>
        Load instance / problem / algorithm
        Algorithm Parameters
        Run Settings
    [ + Add Task ]

Clicking **+ Add Task** stamps a new :class:`Task` into the
:class:`TaskRegistry` with ``status='pending'``; it shows up in the
Pending Tasks column.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractSpinBox, QButtonGroup, QCheckBox, QFileDialog, QFrame,
    QGraphicsDropShadowEffect, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QRadioButton, QSizePolicy, QComboBox,
    QVBoxLayout, QWidget,
)

from portal_studio.algorithms.registry import (
    get_algorithm_meta,
    get_source_version,
    list_algorithm_sources,
    list_solvers,
)
from portal_studio.algorithms.solver_params import (
    ParamSpec, get_solver_params,
)
from portal_studio.gui.common.composite_labels import generator_composite_label
from portal_studio.gui.experimental.task_model import Task, TaskRegistry
from portal_studio.gui.common.help import (
    help_body, make_help_button, show_help_dialog, term,
)
from portal_studio.gui.common.widgets import (
    NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox,
)


_NO_BUTTONS = QAbstractSpinBox.ButtonSymbols.NoButtons
_FORM_LABEL_WIDTH = 92
_FORM_INPUT_WIDTH = 112
_SETUP_COMBO_WIDTH = 200
_FE_BUDGET_PER_DIM = 10_000
_FALLBACK_FE_BUDGET = 300_000


@dataclass
class CompositeSource:
    """One entry in the Problem dropdown."""
    label: str
    composite: dict[str, Any]
    source_kind: str        # 'generator' | 'file'
    path: str | None = None


class TaskSettingsPanel(QFrame):
    """Column 1 — author a single task and push it to the registry."""

    composite_changed = Signal()       # for external listeners

    def __init__(self, *, inventory, registry: TaskRegistry,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("educationControls")
        self.setStyleSheet(
            "QFrame#educationControls { background: transparent; border: none; }")
        self._inventory = inventory
        self._registry = registry
        self._file_sources: list[CompositeSource] = []
        self._fe_budget_user_edited = False
        self._syncing_fe_budget = False
        # name → (kind, widget) for the algorithm-parameter card.
        self._param_widgets: dict[str, tuple[str, QWidget]] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(10)

        self.setup_card = self._section_card("Task Settings")
        self.setup_title = self.setup_card.title_label
        self.task_settings_help_btn = self.setup_card.help_btn
        card_body = self.setup_card.body
        card_body.setSpacing(12)

        # ── Problem source and algorithm, directly under Task Settings ──
        self.problem_setup = QFrame()
        self.problem_setup.setObjectName("setupSubSection")
        self.problem_setup.setStyleSheet(
            "QFrame#setupSubSection { background: transparent; border: none; }")
        setup_grid = QGridLayout(self.problem_setup)
        setup_grid.setContentsMargins(0, 0, 0, 0)
        setup_grid.setHorizontalSpacing(8)
        setup_grid.setVerticalSpacing(8)
        setup_grid.setColumnMinimumWidth(0, _FORM_LABEL_WIDTH)
        setup_grid.setColumnStretch(0, 0)
        setup_grid.setColumnStretch(1, 0)
        setup_grid.setColumnStretch(2, 0)
        setup_grid.setColumnStretch(3, 1)

        setup_grid.addWidget(QLabel("Problem instance:"), 0, 0)
        self.problem_combo = NoWheelComboBox()
        self.problem_combo.setFixedWidth(_SETUP_COMBO_WIDTH)
        self.problem_combo.setMinimumContentsLength(6)
        self.problem_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.problem_combo.currentIndexChanged.connect(
            self._on_composite_changed)
        setup_grid.addWidget(self.problem_combo, 0, 1)
        self.problem_help_btn = make_help_button(
            "Explain Problem Instance",
            self._show_problem_instance_help,
        )
        setup_grid.addWidget(self.problem_help_btn, 0, 2)

        self.load_btn = QPushButton("Load Instance")
        self.load_btn.setFixedWidth(_FORM_INPUT_WIDTH)
        self.load_btn.setToolTip(
            "Load one or more unified PORTAL instance JSON files.")
        self.load_btn.clicked.connect(self._on_load_clicked)
        setup_grid.addWidget(self.load_btn, 1, 1)

        setup_grid.addWidget(QLabel("Source:"), 2, 0)
        self.algorithm_source_combo = NoWheelComboBox()
        self.algorithm_source_combo.setFixedWidth(_SETUP_COMBO_WIDTH)
        self.algorithm_source_combo.addItems(list_algorithm_sources())
        self.algorithm_source_combo.currentIndexChanged.connect(
            self._on_algorithm_source_changed)
        setup_grid.addWidget(self.algorithm_source_combo, 2, 1)
        self.algorithm_source_help_btn = make_help_button(
            "Explain Source",
            self._show_algorithm_source_help,
        )
        setup_grid.addWidget(self.algorithm_source_help_btn, 2, 2)

        setup_grid.addWidget(QLabel("Algorithm:"), 3, 0)
        self.algorithm_combo = NoWheelComboBox()
        self.algorithm_combo.setFixedWidth(_SETUP_COMBO_WIDTH)
        self._refresh_algorithm_combo()
        self.algorithm_combo.currentIndexChanged.connect(
            self._rebuild_param_panel)
        self.algorithm_combo.currentIndexChanged.connect(
            self._update_algorithm_tooltip)
        setup_grid.addWidget(self.algorithm_combo, 3, 1)
        self.algorithm_help_btn = make_help_button(
            "Explain Algorithm",
            self._show_algorithm_help,
        )
        setup_grid.addWidget(self.algorithm_help_btn, 3, 2)
        self._update_algorithm_tooltip()

        card_body.addWidget(self.problem_setup)

        card_body.addWidget(self._divider())

        # ── Sub-section 2: algorithm parameters ───────────────────
        self.params_section = self._sub_section(
            "Algorithm Parameters",
            help_handler=self._show_algorithm_parameters_help,
        )
        params_body = self.params_section.body
        self.param_card = QFrame()
        self.param_card.setObjectName("educationParamGrid")
        self.param_card.setStyleSheet(
            "QFrame#educationParamGrid { background: transparent; border: none; }")
        self.param_card.setSizePolicy(QSizePolicy.Policy.Preferred,
                                      QSizePolicy.Policy.Maximum)
        self.param_layout = QGridLayout(self.param_card)
        self.param_layout.setContentsMargins(0, 0, 0, 0)
        self.param_layout.setHorizontalSpacing(8)
        self.param_layout.setVerticalSpacing(10)
        self.param_layout.setColumnMinimumWidth(0, _FORM_LABEL_WIDTH)
        self.param_layout.setColumnStretch(0, 0)
        self.param_layout.setColumnStretch(1, 0)
        self.param_layout.setColumnStretch(2, 1)
        params_body.addWidget(self.param_card)
        card_body.addWidget(self.params_section)

        card_body.addWidget(self._divider())

        # ── Sub-section 3: run settings ────────────────────────────
        self.run_section = self._sub_section(
            "Run Settings",
            help_handler=self._show_run_settings_help,
        )
        rl = self.run_section.body
        rl.setSpacing(6)

        run_grid = QGridLayout()
        self.run_settings_grid = run_grid
        run_grid.setContentsMargins(0, 0, 0, 0)
        run_grid.setHorizontalSpacing(8)
        run_grid.setVerticalSpacing(8)
        run_grid.setColumnMinimumWidth(0, _FORM_LABEL_WIDTH)
        run_grid.setColumnMinimumWidth(1, _FORM_INPUT_WIDTH)
        run_grid.setColumnStretch(0, 0)
        run_grid.setColumnStretch(1, 0)
        run_grid.setColumnStretch(2, 0)
        run_grid.setColumnStretch(3, 1)

        run_grid.addWidget(QLabel("Number of runs:"), 0, 0)
        self.runs_spin = NoWheelSpinBox()
        self.runs_spin.setButtonSymbols(_NO_BUTTONS)
        self.runs_spin.setRange(1, 200)
        self.runs_spin.setValue(31)
        self.runs_spin.setFixedWidth(_FORM_INPUT_WIDTH)
        run_grid.addWidget(self.runs_spin, 0, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        run_grid.addWidget(QLabel("FE Budget:"), 1, 0)
        self.fe_spin = NoWheelSpinBox()
        self.fe_spin.setButtonSymbols(_NO_BUTTONS)
        self.fe_spin.setRange(100, 10_000_000)
        self.fe_spin.setSingleStep(1000)
        self.fe_spin.setValue(_FALLBACK_FE_BUDGET)
        self.fe_spin.setFixedWidth(_FORM_INPUT_WIDTH)
        self.fe_spin.valueChanged.connect(self._on_fe_budget_changed)
        run_grid.addWidget(self.fe_spin, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        run_grid.addWidget(QLabel("Seed:"), 2, 0)
        self.seed_choices = QWidget()
        self.seed_choices.setObjectName("seedChoices")
        self.seed_choices.setStyleSheet("QWidget#seedChoices { background: transparent; }")
        seed_row = QHBoxLayout(self.seed_choices)
        seed_row.setContentsMargins(0, 0, 0, 0)
        seed_row.setSpacing(10)
        self.seed_random = QRadioButton("Random")
        self.seed_random.setStyleSheet("QRadioButton { background: transparent; font-size: 10pt; }")
        self.seed_manual = QRadioButton("Manual")
        self.seed_manual.setStyleSheet("QRadioButton { background: transparent; font-size: 10pt; }")
        self.seed_random.setChecked(True)
        self._seed_group = QButtonGroup(self)
        self._seed_group.addButton(self.seed_random)
        self._seed_group.addButton(self.seed_manual)
        seed_row.addWidget(self.seed_random)
        seed_row.addWidget(self.seed_manual)
        seed_row.addStretch(1)
        run_grid.addWidget(
            self.seed_choices, 2, 1, 1, 3,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        self.manual_seed_row = QWidget()
        self.manual_seed_row.setObjectName("manualSeedRow")
        self.manual_seed_row.setStyleSheet(
            "QWidget#manualSeedRow { background: transparent; }")
        manual_seed_layout = QGridLayout(self.manual_seed_row)
        manual_seed_layout.setContentsMargins(0, 0, 0, 0)
        manual_seed_layout.setHorizontalSpacing(8)
        manual_seed_layout.setColumnMinimumWidth(0, _FORM_LABEL_WIDTH)
        manual_seed_layout.setColumnStretch(0, 0)
        manual_seed_layout.setColumnStretch(1, 0)
        manual_seed_layout.setColumnStretch(2, 1)
        self.manual_seed_label = QLabel("Algorithm seed:")
        manual_seed_layout.addWidget(self.manual_seed_label, 0, 0)
        self.master_seed_spin = NoWheelSpinBox()
        self.master_seed_spin.setButtonSymbols(_NO_BUTTONS)
        self.master_seed_spin.setRange(0, 999_999)
        self.master_seed_spin.setValue(42)
        self.master_seed_spin.setFixedWidth(_FORM_INPUT_WIDTH)
        manual_seed_layout.addWidget(self.master_seed_spin, 0, 1)
        run_grid.addWidget(self.manual_seed_row, 3, 0, 1, 4)

        run_grid.addWidget(QLabel("Realisation:"), 4, 0)
        self.realisation_combo = NoWheelComboBox()
        self.realisation_combo.setFixedWidth(_FORM_INPUT_WIDTH)
        self.realisation_combo.addItem("Type 0", userData="none")
        self.realisation_combo.addItem("Type 1", userData="type1")
        self.realisation_combo.addItem("Type 2", userData="type2")
        self.realisation_combo.setToolTip(
            "Choose whether runs use the anchor instance or generated realisations.")
        run_grid.addWidget(
            self.realisation_combo, 4, 1,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        self.realisation_help_btn = make_help_button(
            "Explain Realisation",
            self._show_realisation_help,
        )
        run_grid.addWidget(
            self.realisation_help_btn, 4, 2,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        rl.addLayout(run_grid)
        self.manual_seed_row.setVisible(False)
        self.seed_manual.toggled.connect(self.manual_seed_row.setVisible)

        self.label_input = QLineEdit()   # kept for _on_add_clicked; not shown

        card_body.addWidget(self.run_section)
        outer.addWidget(self.setup_card)
        outer.addStretch(1)

        self.add_btn = QPushButton("+ Add Task")
        self.add_btn.setProperty("variant", "primary")
        self.add_btn.setMinimumHeight(36)
        self.add_btn.clicked.connect(self._on_add_clicked)
        outer.addWidget(self.add_btn)

        self._refresh_problem_combo()
        self._rebuild_param_panel()
        self._update_add_btn_enabled()

    # ───────────────────────────────────────────────────────── construction

    def _section_card(self, title: str, help_handler=None) -> QFrame:
        card = QFrame()
        card.setObjectName("sectionCard")
        card.setStyleSheet(
            "QFrame#sectionCard {"
            "  background: #ffffff;"
            "  border: 1px solid #e4e7ec;"
            "  border-radius: 8px;"
            "}")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        header = QLabel(title)
        header.setObjectName("panelTitle")
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
        shadow.setColor(QColor(15, 23, 42, 32))
        card.setGraphicsEffect(shadow)
        return card

    def _sub_section(self, title: str, help_handler=None) -> QFrame:
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
        header.setObjectName("subSectionTitle")
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

    @staticmethod
    def _divider() -> QFrame:
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #e4e7ec;")
        return divider

    # ───────────────────────── public ──────────────────────────────

    def refresh_problem_combo(self) -> None:
        """Hook for the outer tab when the user may have rebuilt the generated instance."""
        self._refresh_problem_combo()

    def current_source(self) -> CompositeSource | None:
        if self.problem_combo.count() == 0:
            return None
        return self.problem_combo.itemData(self.problem_combo.currentIndex())

    # ───────────────────────── internals ───────────────────────────

    def _refresh_problem_combo(self) -> None:
        prior = self.current_source()
        self.problem_combo.blockSignals(True)
        self.problem_combo.clear()
        for gen_comp in self._generator_composites():
            label = generator_composite_label(gen_comp)
            self.problem_combo.addItem(
                label,
                userData=CompositeSource(
                    label=label,
                    composite=gen_comp,
                    source_kind="generator",
                ))
        for fs in self._file_sources:
            self.problem_combo.addItem(fs.label, userData=fs)
        self._set_combo_item_tooltips(self.problem_combo)
        if prior is not None:
            for i in range(self.problem_combo.count()):
                src: CompositeSource = self.problem_combo.itemData(i)
                if src.label == prior.label:
                    self.problem_combo.setCurrentIndex(i)
                    break
        self.problem_combo.blockSignals(False)
        self._update_add_btn_enabled()
        self._update_problem_tooltip()
        self._sync_default_fe_budget()
        self._rebuild_param_panel()
        self.composite_changed.emit()

    def _generator_composites(self) -> list[dict[str, Any]]:
        """Generated composites in creation order."""
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for comp in getattr(self._inventory, "composites", []) or []:
            if isinstance(comp, dict) and id(comp) not in seen:
                out.append(comp)
                seen.add(id(comp))
        latest = getattr(self._inventory, "composite", None)
        if isinstance(latest, dict) and id(latest) not in seen:
            out.append(latest)
        return out

    def _on_composite_changed(self, _index: int) -> None:
        self._update_add_btn_enabled()
        self._update_problem_tooltip()
        self._sync_default_fe_budget()
        self._rebuild_param_panel()
        self.composite_changed.emit()

    def _update_problem_tooltip(self) -> None:
        self.problem_combo.setToolTip(self.problem_combo.currentText())

    def _update_algorithm_tooltip(self, _index: int = -1) -> None:
        self.algorithm_combo.setToolTip(self.algorithm_combo.currentText())

    def _current_source(self) -> str | None:
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
        self._update_algorithm_tooltip()
        if hasattr(self, "param_layout"):
            self._rebuild_param_panel()

    def _on_algorithm_source_changed(self, _index: int) -> None:
        self._refresh_algorithm_combo()

    def _set_algorithm_combo_full_name_tooltips(self) -> None:
        for index in range(self.algorithm_combo.count()):
            name = self.algorithm_combo.itemText(index)
            full_name = get_algorithm_meta(name).description or name
            self.algorithm_combo.setItemData(
                index, full_name, Qt.ItemDataRole.ToolTipRole)

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

    @staticmethod
    def _set_combo_item_tooltips(combo: QComboBox) -> None:
        for index in range(combo.count()):
            combo.setItemData(
                index,
                combo.itemText(index),
                Qt.ItemDataRole.ToolTipRole,
            )

    def _on_load_clicked(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load PORTAL instance JSON(s)", "", "JSON Files (*.json)")
        if not paths:
            return
        errors: list[str] = []
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                from portal_core.load_instance import load_instance_payload
                comp = load_instance_payload(payload)
                from portal_studio.validation import validate_instance_for_gui
                validate_instance_for_gui(comp)
                self._file_sources.append(CompositeSource(
                    label=f"{os.path.basename(p)} (d={comp.get('d', '?')})",
                    composite=comp,
                    source_kind="file",
                    path=p,
                ))
            except Exception as exc:                              # noqa: BLE001
                errors.append(f"{p}: {exc}")
        self._refresh_problem_combo()
        if errors:
            QMessageBox.critical(
                self, "Load PORTAL instances",
                "Some files could not be loaded:\n\n" + "\n".join(errors))

    def _show_help_dialog(self, title: str, body: str) -> None:
        show_help_dialog(self, title, body)

    def _show_problem_instance_help(self) -> None:
        body = help_body("Problem instance", [
            f"{term('Problem instance')} selects the landscape the algorithm "
            "will optimize. Choose an instance created in Generator, or "
            f"select {term('Load Instance')} to add a saved PORTAL JSON "
            "instance.",
        ])
        self._show_help_dialog("Problem instance help", body)

    def _show_algorithm_help(self) -> None:
        body = help_body("Algorithm", [
            f"{term('Algorithm')} selects the optimization method for this "
            "task. Choose one algorithm, then review its settings before "
            "adding the task.",
        ])
        self._show_help_dialog("Algorithm help", body)

    def _show_algorithm_parameters_help(self) -> None:
        body = help_body("Algorithm parameters", [
            f"{term('Algorithm Parameters')} contains only the settings used "
            "by the selected algorithm. The fields change when you choose a "
            "different algorithm; keeping the displayed values uses its "
            "standard setup.",
        ])
        self._show_help_dialog("Algorithm parameters help", body)

    def _show_run_settings_help(self) -> None:
        body = help_body("Run settings", [
            f"{term('Number of runs')} sets how many independent runs are "
            "performed. Multiple runs show how consistently the algorithm "
            "performs on the same selected problem source; Type 1 and Type 2 "
            "realisation modes create controlled variants for those runs.",
            f"{term('FE Budget')} is the maximum number of function "
            "evaluations allowed per run. It initially follows 10,000 times "
            "the problem dimension, within the input limits. Editing it stops "
            "automatic updates when the selected problem changes.",
            f"{term('Seed')} controls the random choices made by the "
            f"algorithm. {term('Random')} derives a deterministic seed sequence "
            f"from the task label, algorithm and stored master seed. {term('Manual')} uses the entered value for the "
            "first run and increases it by one for each later run, making "
            "the experiment repeatable.",
            f"{term('Realisation')} controls the problem instance used by "
            "each run. Use the Realisation help button next to that row for "
            "the Type 0, Type 1, and Type 2 details.",
        ])
        self._show_help_dialog("Run settings help", body)

    def _show_realisation_help(self) -> None:
        type_list = (
            "<ul style=\"margin: 0 0 10px 18px; padding: 0;\">"
            f"<li>{term('Type 0')}: use the selected problem instance directly.</li>"
            f"<li>{term('Type 1')}: apply a rigid centre translation and a "
            "common beta offset while keeping basin shapes and relative basin "
            "depths unchanged.</li>"
            f"<li>{term('Type 2')}: creates stronger within-recipe variants; it "
            "reassigns existing signed parameter values and may reassign block weights. "
            "Centres are rotated and translated, with translation alone if no suitable "
            "rotation fits. Multi-block coordinate labels may change; block sizes "
            "and component membership are retained.</li>"
            "</ul>"
        )
        body = help_body("Realisation", [
            f"{term('Realisation')} chooses how the selected problem instance "
            "is used for repeated runs.",
            type_list,
            "The realisation base seed is fixed at 42, with sample identifiers "
            "1, 2, and so on across runs. These settings are separate from the "
            "optimiser seed. Relocation can change difficulty within fixed bounds.",
        ])
        self._show_help_dialog("Realisation help", body)

    def _rebuild_param_panel(self) -> None:
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                # Hide controls immediately while they await deferred deletion
                # so only active controls remain visible.
                w.hide()
                w.deleteLater()
        self._param_widgets = {}
        algo = self.algorithm_combo.currentText()
        if not algo:
            return
        specs = get_solver_params(algo)
        d = self._current_dimension()
        visible: list[ParamSpec] = []
        for spec in specs.values():
            # Match Education: auto-only parameters stay with the solver.
            # pop_size is the useful exception, prefilled from dimension but
            # still editable for experiments.
            if spec.default is None and spec.name != "pop_size":
                continue
            if spec.name == "pop_size" and spec.default is None and d is not None:
                spec.default = self._auto_pop_size(algo, d)
                spec.kind = "int"
            visible.append(spec)
        if not visible:
            self.param_layout.addWidget(
                QLabel("This solver has no configurable parameters."),
                0, 0, 1, 3)
            return
        for row, spec in enumerate(visible):
            label = QLabel(spec.name)
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
            self.param_layout.addWidget(label, row, 0)
            widget = self._make_param_widget(spec)
            if spec.tooltip:
                widget.setToolTip(spec.tooltip)
            if not isinstance(widget, QCheckBox):
                widget.setFixedWidth(_FORM_INPUT_WIDTH)
            self.param_layout.addWidget(widget, row, 1)
            self._param_widgets[spec.name] = (spec.kind, widget)

    def _current_dimension(self) -> int | None:
        source = self.current_source()
        if source is None:
            return None
        try:
            return int(source.composite.get("d"))
        except (AttributeError, TypeError, ValueError):
            return None

    def _on_fe_budget_changed(self, _value: int) -> None:
        if not self._syncing_fe_budget:
            self._fe_budget_user_edited = True

    def _sync_default_fe_budget(self) -> None:
        if self._fe_budget_user_edited:
            return
        d = self._current_dimension()
        budget = _FALLBACK_FE_BUDGET if d is None else _FE_BUDGET_PER_DIM * d
        budget = max(self.fe_spin.minimum(), min(self.fe_spin.maximum(), int(budget)))
        self._syncing_fe_budget = True
        try:
            self.fe_spin.setValue(budget)
        finally:
            self._syncing_fe_budget = False

    @staticmethod
    def _auto_pop_size(solver: str, d: int) -> int:
        if solver == "jDE":
            return 10 * d
        return min(10 * d, 50)

    def _make_param_widget(self, spec: ParamSpec) -> QWidget:
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
            w.setRange(
                int(spec.min) if spec.min is not None else 0,
                int(spec.max) if spec.max is not None else 1_000_000,
            )
            w.setSingleStep(int(spec.step) if spec.step else 1)
            w.setValue(int(spec.default))
            return w
        if spec.min is not None and spec.max is not None:
            w = NoWheelDoubleSpinBox()
            w.setButtonSymbols(_NO_BUTTONS)
            w.setDecimals(spec.decimals if spec.decimals is not None else 4)
            w.setRange(float(spec.min), float(spec.max))
            w.setSingleStep(float(spec.step) if spec.step else 0.1)
            w.setValue(float(spec.default))
            return w
        w = QLineEdit()
        w.setPlaceholderText("number")
        if spec.default is not None:
            w.setText(repr(spec.default))
        return w

    def _collect_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, (kind, widget) in self._param_widgets.items():
            if isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if not text:
                    continue
                try:
                    out[name] = int(float(text)) if kind == "int" else float(text)
                except ValueError:
                    continue
            elif isinstance(widget, QCheckBox):
                out[name] = bool(widget.isChecked())
            elif isinstance(widget, NoWheelComboBox):
                out[name] = widget.currentText()
            elif isinstance(widget, NoWheelSpinBox):
                out[name] = int(widget.value())
            elif isinstance(widget, NoWheelDoubleSpinBox):
                out[name] = float(widget.value())
        return out

    def _update_add_btn_enabled(self) -> None:
        self.add_btn.setEnabled(self.current_source() is not None)

    def _on_add_clicked(self) -> None:
        source = self.current_source()
        if source is None:
            QMessageBox.information(
                self, "Add Task",
                "Pick a PORTAL instance (Generator or loaded JSON) first.")
            return
        algo = self.algorithm_combo.currentText()
        label = self.label_input.text().strip() or _default_label(
            algo, source.label)
        # Disambiguate against existing tasks with the same label.
        existing = {t.label for t in self._registry.all_tasks()}
        unique_label = label
        n = 2
        while unique_label in existing:
            unique_label = f"{label}_{n}"
            n += 1
        task = Task(
            label=unique_label,
            algorithm=algo,
            algorithm_params=self._collect_params(),
            runs=int(self.runs_spin.value()),
            fe_budget=int(self.fe_spin.value()),
            seed_strategy="random" if self.seed_random.isChecked() else "fixed",
            master_seed=int(self.master_seed_spin.value()),
            realization_type=str(self.realisation_combo.currentData() or "none"),
            composite_label=source.label,
            composite_source_kind=source.source_kind,
            composite_path=source.path,
        )
        self._registry.add(task)
        self.label_input.clear()


def _default_label(algo: str, composite_label: str) -> str:
    """auto-label: <algo>_<short composite token>."""
    short = (
        "gen" if "Generator" in composite_label
        else composite_label.split()[0].split(".")[0].lower()
    )
    return f"{algo.lower().replace('-', '').replace(' ', '')}_{short}"
