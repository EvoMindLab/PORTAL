"""Generator tab: high-dimensional, paper-aligned authoring of PORTAL composites.
Author components with paper-aligned mode menus in collapsible cards, group
same-dimension components into blocks, compose (mixed dimension) and save.
High-dimensional landscapes are inspected through parameter tables and ELA.

Component and block models live in ``portal_studio.experiment``. Solver
benchmarking is handled separately by the Experiment workspace.
"""
from __future__ import annotations

import logging
import os
from html import escape
from typing import List, Optional

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QMenu, QPushButton, QScrollArea, QSizePolicy, QSplitter,
    QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from portal_studio.experiment.modes import (
    TRANSFORM_PARAM_MODES,
    canonical_transform_defaults,
    sample_transform_inner_param,
)
from portal_studio.experiment.session import ExpBlock, ExpComponent, ExperimentInventory
from portal_studio.validation import GUI_BLOCK_DIMENSION_MESSAGE, MIN_GUI_BLOCK_DIMENSION
from portal_core.transform_registry import TRANSFORM_DISPLAY_NAMES
from portal_studio.gui.generator.rotation_dialogs import (
    BlockDiagonalDialog, CONTROLLED_PARTIAL_SEPERATE_LABEL, block_diagonal_display)
from portal_studio.gui.generator import display as generator_display
from portal_studio.gui.generator import sampling as generator_sampling
from portal_studio.gui.generator import tree_model
from portal_studio.gui.common.collapsible import CollapsibleSection
from portal_studio.gui.common.angles import DegreeLineEdit
from portal_studio.gui.generator.parameter_rows import (
    EXTRA_DEFAULTS, EXTRAS_FOR, MODES_FOR, PARAM_TOOLTIPS,
    ParamRowFactory, default_range, mode_int,
)
from portal_studio.gui.common.help import (
    build_help_dialog, base_form_help_paragraphs, centre_help_paragraphs,
    seed_help_paragraphs, exponent_help_paragraphs,
    component_table_help_paragraphs,
    kappa_help_paragraphs, rotation_help_paragraphs,
    neutralization_help_paragraphs, transform_type_help_paragraphs,
    transform_param_help_paragraphs,
    formula_image_label, formula_intro,
    formula_tokens, help_body, make_help_button, make_parameter_help_button, math_text,
    mode_entries, parameter_help_formula, parameter_help_text,
    randomisation_modes, scalar_range_help, show_help_dialog, term,
)
# Inputs that ignore mouse-wheel scrolling: values change only by typing/clicking,
# so hovering over a field while scrolling the panel never silently edits it.
from portal_studio.gui.common.widgets import (
    NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox,
)

logger = logging.getLogger(__name__)

_NO_BUTTONS = QAbstractSpinBox.ButtonSymbols.NoButtons
_ROLE = QtCore.Qt.ItemDataRole.UserRole
_LABEL_W = 64          # compact leading-label column; math notation is short
# Left-panel base minimum width — fits the narrowest common state (all rows
# in their minimum mode). _recompute_panel_min_width() raises this as needed.
_PANEL_BASE_MIN = 430
_DROPDOWN_W = 180      # uniform combo width across all parameter / mode menus
                       # so vector, baseline, rotation, and transform menus
                       # line up vertically.
_SPIN_W = 92           # cap spin-box width; their huge ranges otherwise demand
                       # ~150px each and bloat (then clip) the whole column
_WIDGET_MAX_HEIGHT = 16777215
# Active fields show near-black text; disabled fields go gray.
_FIELD_QSS = (
    "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { color: #1a1a1a; }"
    "QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,"
    " QDoubleSpinBox:disabled { color: #9aa0a6; }"
    # Primary-variant buttons (Add Block / Add Component) go grey when
    # disabled so the frozen state reads clearly.
    " QPushButton[variant=\"primary\"]:disabled {"
    "   background: #e6e8ec; color: #9aa0a6;"
    "   border: 1px solid #d6d9de;"
    " }"
)
# Section card (Add Block, Component Parameters): an elevated white container
# with a soft border so its body visually groups together.
_SECTION_CARD_QSS = (
    "QFrame#sectionCard {"
    "  background: #ffffff;"
    "  border: 1px solid #e4e7ec;"
    "  border-radius: 8px;"
    "}"
)
# Section sub-header chip (e.g. "Components", "Blocks"): a soft gray bar that
# pops off the white card — matches the Builder tab, where the gray comes from
# the section container showing behind the label.
_SECTION_HEADER_QSS = (
    "font-weight: 700; background: #f4f6fa;"
    " border-radius: 6px; padding: 5px 8px;"
)
_VECTORS = ["c", "kappa_plus", "kappa_minus", "p_plus", "p_minus"]
# Effect-first tooltips for the controls built directly in this tab (the row
# parameters c/κ/p and the transform inner-params get theirs from the factory's
# PARAM_TOOLTIPS). They answer "what does this do?" in one line; the section "?"
# buttons carry the full paper theory.
_TIP_BASELINE = (
    "Two component forms — A gives every direction its own exponent p±; B uses "
    "one exponent for the whole basin. Same centre, anisotropy and rotation "
    "either way.")
_TIP_BETA = (
    "Vertical offset — the function's value at the centre. The component with "
    "the lowest β sets its block's minimum when its centre is in the domain.")
_TIP_ROTATION = (
    "Sets the internal coordinate orientation. Interactions also depend on "
    "the base form, anisotropy and transformations.")
_TIP_THETA = "Range the rotation angles are drawn from (degrees)."
_TIP_PROB = "Chance each variable pair gets a non-zero rotation angle."
_TIP_ANGLE = "Fixed rotation angle applied to every rotated pair (degrees)."
_TIP_BLOCK_SIZES = (
    "Comma-separated sizes for each block in block-diagonal rotation, "
    "e.g. '10, 10, 10'. Leave empty to auto-split dimensions equally.")
_TIP_BLOCK_ANGLES = (
    "Comma-separated rotation angle per block (degrees), "
    "e.g. '45, 90, 135'. Leave empty to use supplied angles.")
_TIP_NEUTRALIZE = (
    "Calibrates the baseline scale using exponents, sign-averaged anisotropy, "
    "Delta and the per-coordinate reference displacement. "
    "Off = neutralizing factors fixed at 1.")
_TIP_DELTA = (
    "Positive vertical reference scale. For Form A, the rise at |z_i| = r_ref "
    "averaged over sign choices equals Delta.")
_TIP_RREF = "Positive reference displacement in each internal coordinate; the reference norm is sqrt(d) times r_ref."
_TIP_PSINGLE = (
    "One curvature exponent for the whole basin: 0.5 linear, 1 quadratic, "
    ">1 superquadratic growth.")
_TIP_TRANSFORM_TYPE = (
    "A ruggedness operator. Element-wise ones (additive periodic, "
    "log-sinusoidal, wavelet) act coordinate-wise; coupling ones (tensor, "
    "radial) can add interactions. All preserve the component centre.")
_ROTATION_MODES = [
    "No Rotation", "Fully Connected", "Probabilistic Coupling", "Uniform Random Angle",
    "Fixed Angle", "Chain-like Coupling", "Block-Diagonal",
]
# Block-diagonal rotation is the explicit partial-separability mode; the other
# modes are selected from the standard rotation dropdown.
_FULL_ROTATION_MODES = [
    (2, "Fully Connected"),
    (3, "Probabilistic Coupling"),
    (4, "Uniform Random Angle"),
    (5, "Fixed Angle"),
    (6, "Chain-like Coupling"),
]
_PARTIAL_ROTATION_MODES = [
    (7, "Block-Diagonal"),
]
_ADDITIVE_PERIODIC_PARAM_MODES = [
    "Uniform Random",
    "Dimension-shared",
    "Beta Distribution",
    "Linear Spread (Shuffled)",
]
_ADDITIVE_PERIODIC_PARAM_MODE_VALUES = [1, 3, 5, 6]
# Transform types -> per-dimension vector params (each with a default [min, max]
# range) and per-component scalar params (each a [min, max] range; a single value
# is drawn per component, or fixed when min == max). Defaults follow the
# paper's canonical ranges sourced from ``canonical_transform_defaults()``
# (which reads ``_get_parameter_ranges`` in the engine).


class GeneratorTab(QWidget):
    """Author high-d composites from parametric components/blocks."""

    _TRANSFORM_DISPLAY_NAMES = dict(TRANSFORM_DISPLAY_NAMES)

    def __init__(self, inventory, parent=None) -> None:
        super().__init__(parent)
        self.session_inventory = inventory      # shared 2-D inventory (unused here)
        self.inventory = ExperimentInventory()
        self.vector_rows: dict = {}
        self._pending_transforms: List[dict] = []   # specs, realized on Add Component
        self._tf_vec_rows: dict = {}
        self._tf_scalar_widgets: dict = {}
        # ± pair coupling. When a pair is coupled (default), the minus parameter
        # is an exact copy of the plus parameter — the paper's "matched ±"
        # case (κ⁺=κ⁻ etc.), which yields sign-symmetric components. Ticking the
        # row's "Separate ± directions" checkbox samples them separately
        # (deliberate asymmetry). ``_pairs`` holds the persistent κ/p pair widgets;
        # ``_tf_pairs`` the per-transform pairs (rebuilt with the transform type).
        self._pairs: dict = {}
        self._tf_pairs: dict = {}
        self.parameter_help_buttons: dict[str, QPushButton] = {}
        self._pair_coupled: dict = {}        # base -> coupled? for κ / p
        self._tf_pair_coupled: dict = {}     # base -> coupled? for the active transform
        # Block-Diagonal configuration (grouping + angle mode), driven by the
        # Configure dialog. Defaults to automatic grouping and fixed angles.
        self._block_diag_spec = {"grouping": "auto", "angle_mode": "fixed"}
        # Seed the editor session started on (New Component's auto-assigned seed,
        # or a selected component's saved seed). Reset Parameters restores this
        # rather than a hardcoded 42, so a reset returns to the session default.
        self._editor_seed_default = 42
        # Mode-aware row factory: builds both vector-parameter rows and
        # transform-parameter rows, owns the per-row {mode, vmin, vmax, extras}
        # widget store.
        self._row_factory = ParamRowFactory(
            label_factory=self._lbl,
            on_visibility_change=self._on_parameter_row_visibility_changed,
        )
        self._extra_widgets = self._row_factory.extras   # alias for the factory's mode-specific widgets
        # Tracks the component currently loaded into the editor for the
        # select -> tweak -> Apply Changes flow. None unless a component row
        # is selected.
        self._component_apply_target: Optional[tuple] = None
        self._block_apply_target: Optional[ExpBlock] = None
        self._editor_mode: str = "none"
        self._new_component_block: Optional[ExpBlock] = None
        # Index of the composite whose blocks are currently shown in the tree.
        self._active_composite_idx: Optional[int] = None
        # Transform row selected for the select -> tweak -> Apply Changes flow.
        self._transform_apply_target: Optional[int] = None
        # Cache engine-provided transform defaults once per tab so the GUI
        # uses the same parameter ranges as the backend.
        self._transforms = canonical_transform_defaults()

        outer = QVBoxLayout(self)
        split = QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self._build_right_panel())
        split.addWidget(self._build_left_panel())
        from portal_studio.gui.analysis.ela_panel import ELAPanel
        self.ela_panel = ELAPanel(self.inventory, parent=self)
        split.addWidget(self.ela_panel)
        # Layout priority: the side panes should balance visually, while the
        # Component Parameters pane keeps extra room for the denser editor rows.
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 2)
        split.setSizes([760, 980, 760])
        outer.addWidget(split, stretch=1)

        self.baseline_combo.currentTextChanged.connect(self._on_baseline_changed_with_resize)
        self.add_btn.clicked.connect(self.on_add_component)
        self.apply_changes_btn.clicked.connect(self._on_apply_changes)
        self.reset_btn.clicked.connect(self.on_reset)
        self.add_block_btn.clicked.connect(self.on_add_block_clicked)
        self.build_btn.clicked.connect(self.on_build_composite)
        self.save_btn.clicked.connect(self.on_save_composite)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.ela_panel.composite_selected.connect(self._on_composite_selected)
        self.ela_panel.instance_deleted.connect(self._on_instance_deleted)
        self.transform_combo.currentTextChanged.connect(self._rebuild_transform_params)
        self.add_transform_btn.clicked.connect(self.on_add_transform)
        self.remove_transform_btn.clicked.connect(self.on_remove_transform)

        self._rebuild_transform_params()
        self._on_baseline_changed(self.baseline_combo.currentText())
        self._refresh_tree()
        self._on_tree_selection_changed()   # init Add-Component button + hint state
        self._recompute_panel_min_width()   # initial fit

    @classmethod
    def _transform_display_name(cls, name: str) -> str:
        return cls._TRANSFORM_DISPLAY_NAMES.get(
            str(name), str(name).replace("_", " ").title())

    @classmethod
    def _transform_internal_name(cls, label: str) -> str:
        text = str(label)
        for internal, display in cls._TRANSFORM_DISPLAY_NAMES.items():
            if text in {internal, display}:
                return internal
        return text

    def _current_transform_name(self) -> str:
        data = self.transform_combo.currentData()
        if data is not None:
            return str(data)
        return self._transform_internal_name(
            NoWheelComboBox.currentText(self.transform_combo))

    def _set_current_transform_name(self, name: str) -> None:
        internal = self._transform_internal_name(name)
        index = self.transform_combo.findData(internal)
        if index < 0:
            index = self.transform_combo.findText(
                self._transform_display_name(internal))
        if index >= 0:
            self.transform_combo.setCurrentIndex(index)

    def _sync_transform_combo_tooltip(self, *args) -> None:
        self.transform_combo.setToolTip(
            self._transform_display_name(self._current_transform_name()))

    # ------------------------------------------------------------- left panel
    @staticmethod
    def _lbl(text: str, w: int = _LABEL_W) -> QLabel:
        """A fixed-width leading label so every card row shares one left column."""
        display = math_text(text)
        lab = QLabel(display)
        if "<" in display:
            lab.setTextFormat(Qt.TextFormat.RichText)
        lab.setFixedWidth(w)
        return lab

    @staticmethod
    def _math_label(text: str) -> QLabel:
        display = math_text(text)
        lab = QLabel(display)
        if "<" in display:
            lab.setTextFormat(Qt.TextFormat.RichText)
        return lab

    @staticmethod
    def _right_aligned_widget(widget: QWidget) -> QWidget:
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(widget)
        return row_widget

    @staticmethod
    def _attach_row_help(row_widget: QWidget, help_button: QPushButton) -> None:
        layout = row_widget.layout()
        if isinstance(layout, QGridLayout):
            col = 7 if layout.itemAtPosition(0, 6) is not None else 6
            layout.addWidget(
                help_button, 0, col,
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            layout.setColumnStretch(col + 1, 1)

    @staticmethod
    def _attach_pair_checkbox(row_widget: QWidget, checkbox: QCheckBox) -> None:
        layout = row_widget.layout()
        if isinstance(layout, QGridLayout):
            layout.addWidget(
                checkbox, 0, 6,
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    @staticmethod
    def _anchor_row_tail(row_widget: QWidget, col: int = 6) -> None:
        """Keep free horizontal space after the last content column."""
        layout = row_widget.layout()
        if isinstance(layout, QGridLayout):
            layout.setColumnStretch(col, 1)

    # ----------------------------------------------------------- ± pairs
    @staticmethod
    def _tf_display_name(transform_name: str, pname: str) -> str:
        """Label override for a transform parameter where the user value differs
        from the effective one. Tensor interference: the user sets the base
        amplitude μ₀, so ``mu_plus``/``mu_minus`` render as ``mu0_plus``/
        ``mu0_minus`` (math_text turns those into μ₀⁺/μ₀⁻); the data key is
        unchanged."""
        if transform_name == "tensor_interference" and pname in (
                "mu_plus", "mu_minus"):
            return "mu0" + pname[2:]
        return pname

    @staticmethod
    def _tf_table_label(transform_name: str, name: str, base: str,
                        sign: str) -> str:
        """Plain-text data-table label. Tensor interference's μ is the base
        amplitude μ₀; plain_math_header would flatten the subscript, so emit
        μ₀⁺/μ₀⁻ literally here."""
        if transform_name == "tensor_interference" and base == "mu":
            sup = {"plus": "⁺", "minus": "⁻"}.get(sign, "")
            return f"μ₀{sup}"
        return generator_display.plain_math_header(name)

    @staticmethod
    def _split_pair(name: str):
        """Return ``(base, sign)`` for a ± parameter name, else ``(name, None)``."""
        if name.endswith("_plus"):
            return name[:-5], "plus"
        if name.endswith("_minus"):
            return name[:-6], "minus"
        return name, None

    @staticmethod
    def _strip_pair_sign(html: str) -> str:
        """Drop the ⁺/⁻ superscript from a math label, leaving the base symbol."""
        return (html.replace("<sup>+</sup>", "")
                    .replace("<sup>-</sup>", "")
                    .replace("<sup>−</sup>", ""))

    def _make_pair_checkbox(self, context: str = "component") -> QCheckBox:
        """A small checkbox for splitting matched +/- direction parameters."""
        cb = QCheckBox("Separate ± directions")
        cb.setStyleSheet("color: #6b7280; font-size: 11px;")
        if context == "transform":
            cb.setToolTip(
                "Check Separate ± directions to sample each sign independently "
                "using the shared mode and bounds. Leave unchecked for matched values."
            )
        else:
            cb.setToolTip(
                "Check Separate ± directions to sample each sign independently "
                "using the shared mode and bounds. Leave unchecked for matched values."
            )
        return cb

    _COMPONENT_PAIR_NO_EFFECT_MODES = {
        "kappa": {5},      # Fixed Unit Scale (κ=1)
        "p": {2, 3},       # Fixed Value, Fixed Gradient
    }

    def _register_pair(self, base, plus, minus, checkbox,
                       rows, pairs, coupled_store,
                       show_minus_row: bool = True) -> None:
        """Wire a ± pair: relabel the primary row + toggle the minus row."""
        # Use the label the factory already rendered (honours any display_name
        # override, e.g. tensor interference's μ₀⁺) rather than recomputing from
        # the raw key, which would drop the override.
        plus_disp = rows[plus]["_name_label"].text()
        pairs[base] = {
            "plus": plus, "minus": minus, "checkbox": checkbox,
            "label": rows[plus]["_name_label"],
            "plus_disp": plus_disp,
            "combined": self._strip_pair_sign(plus_disp),
            "rows": rows, "coupled_store": coupled_store,
            "show_minus_row": show_minus_row,
        }
        checkbox.toggled.connect(
            lambda _c, b=base, p=pairs: self._apply_pair_visibility(b, p))
        if not show_minus_row:
            rows[plus]["mode"].currentTextChanged.connect(
                lambda _t, b=base: self._sync_component_pair_checkbox_state(b))
        self._apply_pair_visibility(base, pairs)
        if not show_minus_row:
            self._sync_component_pair_checkbox_state(base)

    def _sync_component_pair_checkbox_state(self, base: str) -> None:
        """Hide component pair separation when the selected mode cannot differ."""
        info = self._pairs.get(base)
        if not info:
            return
        mode = self._mode_int(info["rows"][info["plus"]]["mode"])
        has_effect = mode not in self._COMPONENT_PAIR_NO_EFFECT_MODES.get(base, set())
        cb = info["checkbox"]
        if not has_effect and cb.isChecked():
            cb.setChecked(False)
        cb.setVisible(has_effect)
        self._apply_pair_visibility(base, self._pairs)

    def _apply_pair_visibility(self, base, pairs) -> None:
        """Show one combined row (coupled) or split ⁺/⁻ rows (independent)."""
        info = pairs[base]
        is_coupled = not info["checkbox"].isChecked()
        info["coupled_store"][base] = is_coupled
        lbl = info["label"]
        lbl.setTextFormat(Qt.TextFormat.RichText)
        show_minus_row = bool(info.get("show_minus_row", True))
        lbl.setText(info["combined"] if (is_coupled or not show_minus_row) else info["plus_disp"])
        minus_widget = info["rows"][info["minus"]]["widget"]
        if is_coupled or not show_minus_row:
            minus_widget.setMaximumHeight(0)
            minus_widget.setVisible(False)
        else:
            minus_widget.setMaximumHeight(_WIDGET_MAX_HEIGHT)
            minus_widget.setVisible(True)
        if pairs is getattr(self, "_tf_pairs", None):
            self._sync_pending_transform_coupled_state()
        self._recompute_panel_min_width()
        self._refresh_component_body_height()

    def _sync_pending_transform_coupled_state(self) -> None:
        """Keep staged transform pair state in step with the visible editor."""
        if getattr(self, "_loading_transform_row", False):
            return
        if not getattr(self, "_pending_transforms", None):
            return
        idx = self._transform_apply_target
        if idx is None:
            current_name = self._current_transform_name()
            last_idx = len(self._pending_transforms) - 1
            if self._pending_transforms[last_idx].get("name") != current_name:
                return
            idx = last_idx
        if 0 <= idx < len(self._pending_transforms):
            self._pending_transforms[idx]["coupled"] = dict(self._tf_pair_coupled)

    def _build_section_card(self, title: Optional[str]) -> QFrame:
        """Build a soft-elevated card with a section title and a body layout.

        The returned QFrame exposes ``body_layout`` (a ``QVBoxLayout``) so
        callers can attach their widgets. The card has a thin border and a
        subtle drop shadow so adjacent cards in the same column read as
        separate visual groups.
        """
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        card = QFrame()
        card.setObjectName("sectionCard")
        card.setStyleSheet(_SECTION_CARD_QSS)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
        if title:
            header = QLabel(title)
            header.setObjectName("panelTitle")   # match the ELA panel header size
            lay.addWidget(header)
        card.body_layout = QVBoxLayout()
        card.body_layout.setContentsMargins(0, 0, 0, 0)
        card.body_layout.setSpacing(6)
        lay.addLayout(card.body_layout)
        # Soft drop shadow for visual separation between cards.
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(14)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(15, 23, 42, 32))   # ~12% alpha cool-grey
        card.setGraphicsEffect(shadow)
        return card

    def _build_add_block_card(self) -> QFrame:
        """The left-panel new-block action: a single Add Block button.

        New blocks start with dimension 30 and weight 1.0 and are edited in the
        middle Block Parameters panel. Hidden ``dim_input`` and ``weight_input``
        controls supply the creation values to ``on_add_block_clicked``.
        """
        add_block_card = self._build_section_card(None)
        ac = add_block_card.body_layout
        self.dim_input = NoWheelSpinBox(add_block_card)
        self.dim_input.setRange(1, 1000)
        self.dim_input.setValue(30)
        self.dim_input.hide()
        self.weight_input = NoWheelDoubleSpinBox(add_block_card)
        self.weight_input.setRange(0.001, 1e6)
        self.weight_input.setDecimals(3)
        self.weight_input.setValue(1.0)
        self.weight_input.hide()
        self.add_block_btn = QPushButton("+ Add Block")
        self.add_block_btn.setProperty("variant", "primary")
        ac.addWidget(self.add_block_btn)
        return add_block_card

    def _build_placeholder_panel(self, title: str, subtitle: str) -> QWidget:
        """A titled column reserved for future analysis tooling (ELA / Performance)."""
        panel = QWidget()
        panel.setObjectName("projectPanel")       # white surface card (elevated look)
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)
        head = QLabel(title)
        head.setObjectName("panelTitle")
        v.addWidget(head)
        sub = QLabel(subtitle)
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #9aa0a6;")
        v.addWidget(sub)
        v.addStretch(1)
        return panel

    def _help_button(self, tip: str, handler) -> QPushButton:
        return make_help_button(tip, handler)

    def _parameter_help_button(self, name: str, modes: Optional[List[str]] = None) -> QPushButton:
        btn = make_parameter_help_button(
            f"Explain {self._plain_math_header(name)}",
            lambda n=name, m=modes: self._show_parameter_help(n, m),
        )
        self.parameter_help_buttons[name] = btn
        return btn

    @staticmethod
    def _mode_entries(modes: Optional[List[str]]) -> list[tuple[str, str]]:
        return mode_entries(modes)

    def _parameter_help_text(self, name: str, modes: Optional[List[str]]) -> tuple[str, str]:
        return parameter_help_text(name, modes, self._current_transform_name())

    def _scalar_range_help(self, name: str) -> str:
        return scalar_range_help(name)

    @staticmethod
    def _parameter_help_formula(name: str) -> str | None:
        return parameter_help_formula(name)

    def _centre_help_paragraphs(self) -> List[str]:
        return centre_help_paragraphs()

    def _show_parameter_help(self, name: str, modes: Optional[List[str]] = None) -> None:
        if name == "seed":
            self._show_help_dialog(
                "Seed help",
                self._help_body("Seed", seed_help_paragraphs()),
                None,
            )
            return
        if name == "base_form":
            self._show_help_dialog(
                "Base form help",
                self._help_body("Base form", base_form_help_paragraphs()),
                None,
            )
            return
        if name == "c":
            self._show_help_dialog(
                "Component Center help",
                self._help_body("Component Center", self._centre_help_paragraphs()),
                None,
            )
            return
        if name == "p_plus":
            self._show_help_dialog(
                "p help",
                self._help_body("p", exponent_help_paragraphs()),
                None,
            )
            return
        if name == "kappa_plus":
            self._show_help_dialog(
                "κ help",
                self._help_body("κ", kappa_help_paragraphs()),
                None,
            )
            return
        if name == "rotation":
            self._show_help_dialog(
                "Rotation help",
                self._help_body("Rotation", rotation_help_paragraphs()),
                self._parameter_help_formula("rotation"),
            )
            return
        if name == "neutralization":
            self._show_help_dialog(
                "Neutralization help",
                self._help_body("Neutralization", neutralization_help_paragraphs()),
                self._parameter_help_formula("neutralization"),
            )
            return
        if name == "transform_type":
            self._show_help_dialog(
                "Transform type help",
                self._help_body("Transform type", transform_type_help_paragraphs()),
                self._parameter_help_formula("transform_type"),
            )
            return
        title, definition = self._parameter_help_text(name, modes)
        base = name[:-5] if name.endswith("_plus") else (
            name[:-6] if name.endswith("_minus") else name)
        tf_help = transform_param_help_paragraphs(
            self._current_transform_name(), base)
        if tf_help is not None:
            self._show_help_dialog(
                f"{title} help", self._help_body(title, tf_help), None)
            return
        paragraphs = [definition]
        mode_entries = self._mode_entries(modes)
        if mode_entries and name != "base_form":
            paragraphs += self._randomisation_modes(
                mode_entries, self._randomisation_intro(name))
        elif name in {"sigma", "eta", "eps", "mu", "p", "q", "omega"}:
            paragraphs.append(self._scalar_range_help(name))
        self._show_help_dialog(
            f"{title} help",
            self._help_body(title, paragraphs),
            self._parameter_help_formula(name),
        )

    @staticmethod
    def _term(text: str) -> str:
        return term(text)

    def _help_body(self, heading: str, paragraphs: List[str]) -> str:
        return help_body(heading, paragraphs)

    def _formula_intro(self, purpose: str, variables: str) -> list[str]:
        return formula_intro(purpose, variables)

    def _randomisation_modes(
        self,
        entries: list[tuple[str, str]],
        intro: str | None = None,
    ) -> list[str]:
        return randomisation_modes(entries, intro)

    def _randomisation_intro(self, name: str) -> str:
        display = self._term(name)
        if name == "c":
            return (
                f"{self._term('Randomisation modes')} for {display} define how "
                "PORTAL chooses the component centre."
            )
        if name == "rotation":
            return (
                f"{self._term('Randomisation modes')} for {display} define how "
                f"PORTAL fills the rotation angle matrix {self._term('Psi_k')}."
            )
        if name.startswith("kappa_"):
            return (
                f"{self._term('Randomisation modes')} for {self._term('kappa')} "
                "define how PORTAL chooses the anisotropy values across "
                "dimensions and positive/negative directions."
            )
        if name.startswith("p_"):
            return (
                f"{self._term('Randomisation modes')} for {self._term('p_single')} "
                "define how PORTAL chooses the exponent values across "
                "dimensions and positive/negative directions."
            )
        if name.endswith(("_plus", "_minus")):
            base_name, _sign = self._split_pair(name)
            base_display = self._term(base_name)
            transform_name = self._current_transform_name()
            transform_randomisation = {
                ("additive_periodic", "mu"): "additive-ripple amplitude values",
                ("additive_periodic", "gamma"): "additive-ripple switch-on values",
                ("additive_periodic", "omega"): "additive-ripple frequency values",
                ("log_sinusoidal", "mu"): "log-space modulation amplitude values",
                ("log_sinusoidal", "omega1"): "first log-space frequency values",
                ("log_sinusoidal", "omega2"): "second log-space frequency values",
                ("wavelet_mod", "mu"): "wavelet amplitude values",
                ("wavelet_mod", "omega"): "wavelet frequency values",
                ("tensor_interference", "mu"): "tensor-interference strength values",
                ("tensor_interference", "omega"): "tensor-interference frequency values",
            }
            specific = transform_randomisation.get((transform_name, base_name))
            if specific:
                return (
                    f"{self._term('Randomisation modes')} for {base_display} define how "
                    f"PORTAL chooses the {specific} across dimensions."
                )
            return (
                f"{self._term('Randomisation modes')} for {base_display} define how "
                "PORTAL chooses the transform parameter across dimensions."
            )
        return (
            f"{self._term('Randomisation modes')} for {display} define how "
            "PORTAL chooses this value."
        )

    @staticmethod
    def _formula_tokens(formula: str) -> list[tuple[str, str]]:
        return formula_tokens(formula)

    def _formula_image_label(self, formula: str) -> QLabel:
        return formula_image_label(formula)

    def _build_help_dialog(self, title: str, body: str, formula: str | None = None) -> QDialog:
        return build_help_dialog(self, title, body, formula)

    def _show_help_dialog(self, title: str, body: str, formula: str | None = None) -> None:
        show_help_dialog(self, title, body, formula)

    def _show_baseline_help(self) -> None:
        self._show_help_dialog(
            "Base Form help",
            self._help_body("Base Form", [
                "Base Form A and Base Form B are the two PORTAL component forms. "
                "They use the same centre, scaling, rotation, and transformation "
                "ideas, but differ in how the exponent terms are applied.",
                f"Use {self._term('A')} when you want separate positive and "
                f"negative exponent vectors {self._term('p_plus')} and "
                f"{self._term('p_minus')}. Use {self._term('B')} when you want "
                f"the single exponent form {self._term('p_single')}.",
            ] + self._formula_intro(
                "shows how Layer 2 combines component functions using the "
                "lower envelope in Part 1, Eq. (13)",
                f"{self._term('n_k')} selects Base Form A "
                f"({self._term('n_k')} = 1) or Base Form B "
                f"({self._term('n_k')} = 2), and {self._term('C')} is the "
                "component set",
            )),
            "f(x) = min_k∈C φ_k^(n_k)(x)",
        )

    def _show_geometry_help(self) -> None:
        self._show_help_dialog(
            "Position & Orientation help",
            self._help_body("Position & Orientation", [
                f"{self._term('c')} sets the component centre. In random modes "
                "it is sampled from the visible min/max range; exclusion mode "
                "keeps it away from the forbidden interval.",
                f"{self._term('beta')} shifts the component value vertically.",
                f"{self._term('Rotation')} controls the orientation of the local "
                "coordinate system. The angle range appears for modes that "
                "sample angles. "
                f"{self._term('prob')} appears for Probabilistic mode; "
                f"{self._term('angle')} appears for Fixed angle mode.",
            ] + self._randomisation_modes([
                ("Uniform Random", "samples every component centre coordinate "
                 "independently between min and max."),
                ("With Exclusion Zone", "samples each centre coordinate from "
                 "the lower or upper side of the excluded interval."),
            ]) + self._formula_intro(
                "shows the internal coordinate definition from Part 1, Eq. (5)",
                f"{self._term('x')} is the input point, {self._term('c')} is "
                f"the component centre, {self._term('R_k')} is the rotation, "
                f"{self._term('T_k')} is the optional transformation, and "
                f"{self._term('z_k')} is the internal coordinate",
            )),
            "z_k = T_k(R_k(x - c_k))",
        )

    def _show_neutralization_help(self) -> None:
        self._show_help_dialog(
            "Neutralization help",
            self._help_body("Neutralization", [
                "Neutralization calibrates the baseline scale. "
                f"Part 1 defines neutralizing factors {self._term('rho_plus')} "
                f"and {self._term('rho_minus')} for Baseline A, and "
                f"{self._term('rho')} for Baseline B, so changes to "
                f"{self._term('p_plus')}, {self._term('p_minus')}, or "
                f"{self._term('kappa_plus')}/{self._term('kappa_minus')} are "
                "accounted for at the reference displacement. This does not "
                "guarantee equal basin exposure or optimizer difficulty.",
                f"{self._term('Delta')} controls the vertical reference scale; "
                f"{self._term('r_ref')} is a displacement in each internal coordinate. "
                "For Form A, the rise averaged over sign choices at this reference equals Delta. "
                "For Baseline A the factors follow Eq. (8); for Baseline B "
                "the single neutralizing factor follows Eq. (12). When "
                "neutralization is off, PORTAL uses neutralizing factors equal "
                "to 1.",
            ] + self._formula_intro(
                "shows the Baseline A neutralizing factor from Part 1, Eq. (8)",
                f"{self._term('rho_plus')}/{self._term('rho_minus')} are "
                "direction-specific neutralizing factors, "
                f"{self._term('Delta')} is the target rise, "
                f"{self._term('r_ref')} is the per-coordinate reference displacement, and "
                f"{self._term('kappa')} with a bar is the component-level "
                "average anisotropy",
            )),
            "ρ_k,i^± = (Δ/d) / (κ̄_k r_ref^(2p_k,i^±))",
        )

    def _show_advanced_help(self) -> None:
        self._show_help_dialog(
            "Shape Parameters help",
            self._help_body("Shape Parameters", [
                f"{self._term('kappa_plus')} and {self._term('kappa_minus')} "
                "are the positive and negative anisotropy factors from Part 1. "
                "They control basin elongation and conditioning.",
                f"{self._term('p_plus')} and {self._term('p_minus')} are the "
                "positive and negative exponents for Baseline A. They control "
                "directional curvature through the power term "
                f"{self._term('2p_plus')}/{self._term('2p_minus')}. For "
                f"Baseline B, the editor uses the single exponent "
                f"{self._term('p_single')}.",
                "Each row's mode menu decides whether the value is fixed, "
                "sampled from a range, isotropic, or drawn from the selected "
                "distribution.",
            ] + self._randomisation_modes([
                ("Uniform Random", "for κ rows, samples each component and "
                 "dimension uniformly between min and max."),
                ("Beta Distribution", "samples beta-distributed values and "
                 "rescales them to the visible min/max range."),
                ("Isotropic Random", "for κ rows, samples one "
                 "value per component and repeats it across dimensions."),
                ("Linear Spread (Shuffled)", "builds a linear grid across "
                 "the range and randomly permutes it per component."),
                ("Fixed Unit Scale (κ=1)", "for κ rows, sets every value to 1."),
                ("Uniform Random", "for p rows, samples exponents uniformly "
                 "between min and max."),
                ("Fixed Value", "for p rows, repeats the fixed exponent "
                 "entered by the user."),
                ("Fixed Gradient", "for p rows, assigns a "
                 "left-to-right linear gradient across dimensions."),
            ]) + self._formula_intro(
                "shows the Baseline A component structure from Part 1, Eq. (6)",
                f"{self._term('kappa_plus')}/{self._term('kappa_minus')} "
                "set anisotropy, "
                f"{self._term('p_plus')}/{self._term('p_minus')} set "
                f"curvature, {self._term('rho_plus')}/{self._term('rho_minus')} "
                f"are neutralizing factors, and {self._term('beta_k')} is the "
                "vertical offset",
            )),
            "φ_k^(1)(x) = β_k + Σ_i ρ_k,i^± κ_k,i^± |z_k,i|^(2p_k,i^±)",
        )

    def _show_transforms_help(self) -> None:
        self._show_help_dialog(
            "Transform Sequence help",
            self._help_body("Transform Sequence", [
                f"{self._term('Type')} selects the transformation added to the "
                "component's transform sequence.",
                "Vector inner parameters use the same mode/min/max pattern as "
                "the main component parameters. Scalar values are sampled from "
                "their visible ranges when the transform is added.",
                f"Use {self._term('Add Transform')} to append the "
                "configured transform; the sequence order is the order applied "
                "to the component.",
            ] + self._randomisation_modes([
                ("Independent random", "samples each transform inner parameter "
                 "from its visible range."),
                ("Component-shared", "uses one sampled value across all "
                 "dimensions of one component."),
                ("Dimension-shared", "uses one sampled value per dimension, "
                 "shared across components."),
                ("Global shared", "uses one sampled value everywhere."),
                ("Beta distribution", "draws transform values using alpha and "
                 "beta shape controls."),
                ("Linspace with permutation", "spreads values across the range "
                 "and shuffles their order."),
            ]) + self._formula_intro(
                "shows that transforms are applied one after another in the "
                "saved sequence",
                f"{self._term('T_1')} is the first transform, "
                f"{self._term('T_2')} is the second transform, and "
                f"{self._term('T_n')} is the final transform in the sequence",
            )),
            "x' = T_n(...T_2(T_1(x)))",
        )

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        # Own object name (not the shared grey "parameterPanel") so the editor
        # pane gets a white ELA-style surface without affecting other tabs.
        panel.setObjectName("generatorEditorPanel")
        panel.setStyleSheet(_FIELD_QSS)
        # Pin a minimum width that fits the widest card row (label + combo + three
        # number cells) plus margins, so the splitter can never
        # shrink this column enough to clip the trailing "value" field.
        # Base minimum fits the common case (rows in their narrowest mode).
        # _recompute_panel_min_width() raises the minimum dynamically when a
        # wider mode (e.g. kappa Beta distribution) becomes visible and lowers
        # it again when the user switches back.
        self._param_panel = panel
        panel.setMinimumWidth(_PANEL_BASE_MIN)
        v = QVBoxLayout(panel)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)
        panel_layout = v

        # ---- Parameter editor card ----
        components_card = self._build_section_card(None)
        self.component_parameters_card = components_card
        # Sit flush on the white pane (no inner box): drop the card border/shadow.
        components_card.setStyleSheet(
            "QFrame#sectionCard { background: transparent; border: none; }")
        components_card.setGraphicsEffect(None)
        panel_layout.addWidget(components_card)
        # All subsequent component-editor widgets attach to this card's body
        # layout instead of the panel's outer layout.
        v = components_card.body_layout

        self.block_edit_card = QWidget()
        be = QVBoxLayout(self.block_edit_card)
        be.setContentsMargins(0, 0, 0, 6)
        be.setSpacing(8)
        block_header = QLabel("Block Parameters")
        block_header.setObjectName("panelTitle")   # match the ELA header size
        be.addWidget(block_header)
        dim_row = QHBoxLayout()
        dim_row.setContentsMargins(0, 0, 0, 0)
        dim_row.setSpacing(10)
        dim_row.addWidget(QLabel("Dimension:"))
        self.block_dim_input = NoWheelSpinBox()
        self.block_dim_input.setButtonSymbols(_NO_BUTTONS)
        self.block_dim_input.setRange(1, 1000)
        self.block_dim_input.setMaximumWidth(_SPIN_W)
        dim_row.addWidget(self.block_dim_input)
        dim_row.addWidget(self._parameter_help_button("block_dim"))
        dim_row.addWidget(QLabel("Weight:"))
        self.block_weight_input = NoWheelDoubleSpinBox()
        self.block_weight_input.setButtonSymbols(_NO_BUTTONS)
        self.block_weight_input.setRange(0.001, 1e6)
        self.block_weight_input.setDecimals(3)
        self.block_weight_input.setMaximumWidth(_SPIN_W)
        dim_row.addWidget(self.block_weight_input)
        dim_row.addWidget(self._parameter_help_button("block_weight"))
        dim_row.addStretch(1)
        be.addLayout(dim_row)
        # New components are started from the per-instance "+" in the Instance
        # Layers tree (see _make_node_actions).
        self.block_edit_card.setVisible(False)
        v.addWidget(self.block_edit_card)

        # Empty-state / block-mode hint, centred in the middle of the pane
        # (like the ELA panel's placeholder) rather than reserving a grey box.
        self.active_block_hint = QLabel(self._default_hint_html())
        self.active_block_hint.setWordWrap(True)
        self.active_block_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.active_block_hint.setStyleSheet("color: #9aa0a6; padding: 18px;")
        panel_layout.addWidget(self.active_block_hint, stretch=1)

        body = QWidget()
        body.setObjectName("componentEditorBody")
        body.setStyleSheet(
            "QWidget#componentEditorBody { background: transparent; border: none; }")
        bv = QVBoxLayout(body)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(6)

        self.base_section = CollapsibleSection(
            "Component Parameters", flat=True,
            collapsible=False)
        self.base_section._toggle.setStyleSheet(   # match the ELA header size
            "font-size: 13pt; font-weight: 700;")
        seed_row = QHBoxLayout()
        seed_row.setContentsMargins(0, 0, 0, 0)
        seed_row.addWidget(self._lbl("Seed"))
        self.seed_spin = NoWheelSpinBox()
        self.seed_spin.setButtonSymbols(_NO_BUTTONS)
        self.seed_spin.setRange(0, 1_000_000)
        self.seed_spin.setValue(42)
        self.seed_spin.setMaximumWidth(_SPIN_W)
        seed_row.addWidget(self.seed_spin)
        seed_row.addWidget(self._parameter_help_button("seed"))
        seed_row.addStretch(1)
        self.base_section.body_layout.addLayout(seed_row)

        brow = QHBoxLayout()
        base_lbl = self._lbl("Base form")
        base_lbl.setToolTip(_TIP_BASELINE)
        brow.addWidget(base_lbl)
        self.baseline_combo = NoWheelComboBox()
        self.baseline_combo.addItems(["A", "B"])
        self.baseline_combo.setFixedWidth(_DROPDOWN_W)
        self.baseline_combo.setToolTip(_TIP_BASELINE)
        brow.addWidget(self.baseline_combo)
        brow.addWidget(self._parameter_help_button("base_form", ["A", "B"]))
        brow.addStretch(1)
        self.base_section.body_layout.addLayout(brow)
        bv.addWidget(self.base_section)

        self.position_section = CollapsibleSection(
            "Position & Orientation",
            flat=True, collapsible=False)
        self.position_section._header.hide()
        c_row = self._mode_row_widget("c", self.vector_rows, *default_range("c"),
                                      modes=MODES_FOR["c"],
                                      extras_spec=EXTRAS_FOR["c"])
        self._attach_row_help(c_row, self._parameter_help_button("c", MODES_FOR["c"]))
        self.position_section.body_layout.addWidget(c_row)
        beta_row = QHBoxLayout()
        beta_lbl = self._lbl("beta")
        beta_lbl.setToolTip(_TIP_BETA)
        beta_row.addWidget(beta_lbl)
        self.beta_spin = NoWheelDoubleSpinBox()
        self.beta_spin.setButtonSymbols(_NO_BUTTONS)
        self.beta_spin.setRange(-1e6, 1e6)
        self.beta_spin.setDecimals(3)
        self.beta_spin.setMaximumWidth(_SPIN_W)
        self.beta_spin.setToolTip(_TIP_BETA)
        beta_row.addWidget(self.beta_spin)
        beta_row.addWidget(self._parameter_help_button("beta"))
        beta_row.addStretch(1)
        self.position_section.body_layout.addLayout(beta_row)
        self._long_parameter_label_w = max(
            _LABEL_W,
            self.fontMetrics().horizontalAdvance("Neutralization") + 8,
        )
        # Rotation row, single-line like the vector parameter rows.
        # Layout: [label] [mode combo] prob:[ ] angle:[ ]
        # θ_min/θ_max are hardcoded to [-180, 180] (per benchmark_generator.py)
        # and are not exposed as user inputs.
        # Each per-mode input shows ONLY when the selected RotationMode uses
        # it (per generate_rotation_angles in benchmark_generator.py).
        self._rotation_row_widget = QWidget()
        rotation_outer = QVBoxLayout(self._rotation_row_widget)
        rotation_outer.setContentsMargins(0, 0, 0, 0)
        rotation_outer.setSpacing(4)
        rrow = QHBoxLayout()
        rrow.setContentsMargins(0, 0, 0, 0)
        rrow.setSpacing(8)
        rot_lbl = self._lbl("Rotation", w=self._long_parameter_label_w)
        rot_lbl.setToolTip(_TIP_ROTATION)
        rrow.addWidget(rot_lbl)
        self._rotation_header_layout = rrow
        self._rotation_row_layout = rrow
        self.rotation_enable = QCheckBox("Enable")
        self.rotation_enable.setToolTip(
            "Unchecked means No Rotation; the base form and transforms can still couple variables.")
        rrow.addWidget(self.rotation_enable)
        # "Controlled Partial Separable" sits on the Enable row and shows only
        # when rotation is enabled. Checked → Block-diagonal only, configured via
        # the dialog (the mode dropdown is hidden); unchecked → the standard
        # rotation modes selectable from the dropdown.
        self.rotation_partial_check = QCheckBox("Controlled Partial Separable")
        self.rotation_partial_check.setToolTip(
            "Checked: Block-Diagonal rotation only, set up in the Configure "
            "dialog (no mode dropdown). Unchecked: choose a standard rotation "
            "mode from the dropdown (Fully Connected, Probabilistic, Uniform "
            "angle, Fixed angle, or Chain-like).")
        rrow.addWidget(self.rotation_partial_check)
        # Help icon sits next to the Enable checkbox so it stays visible even
        # when rotation is disabled (the mode/controls row is hidden then).
        rrow.addWidget(self._parameter_help_button("rotation", list(_ROTATION_MODES)))
        rrow.addStretch(1)
        rotation_outer.addLayout(rrow)

        # The Configure button sits on its own line, indented to line up under
        # the "Controlled Partial Separable" checkbox. It replaces the mode
        # dropdown when that checkbox is ticked (Block-Diagonal rotation).
        self.block_diag_config_btn = QPushButton("Configure")
        self.block_diag_config_btn.clicked.connect(self._open_block_diagonal_dialog)
        # Stretch the button so its right edge lines up with the end of the
        # "Controlled Partial Separable" label above it.
        self.block_diag_config_btn.setFixedWidth(
            self.rotation_partial_check.sizeHint().width())
        self._block_diag_config_row_widget = QWidget()
        cfg_row = QHBoxLayout(self._block_diag_config_row_widget)
        cfg_indent = (self._long_parameter_label_w + 8
                      + self.rotation_enable.sizeHint().width() + 8)
        cfg_row.setContentsMargins(cfg_indent, 0, 0, 0)
        cfg_row.setSpacing(4)
        cfg_row.addWidget(self.block_diag_config_btn)
        cfg_row.addStretch(1)
        rotation_outer.addWidget(self._block_diag_config_row_widget)

        self._rotation_controls_row_widget = QWidget()
        controls_row = QHBoxLayout(self._rotation_controls_row_widget)
        controls_row.setContentsMargins(self._long_parameter_label_w + 4, 0, 0, 0)
        controls_row.setSpacing(4)
        self._rotation_controls_layout = controls_row
        rotation_outer.addWidget(self._rotation_controls_row_widget)
        self.rotation_combo = NoWheelComboBox()
        self.rotation_combo.setFixedWidth(_DROPDOWN_W)
        self.rotation_combo.setToolTip(_TIP_ROTATION)
        controls_row.addWidget(self.rotation_combo)
        # Per-mode inputs, each with its inline label so the user can read it.
        # θ_min/θ_max are hardcoded to [-180, 180] and not shown here.
        _rot_tips = {
            "theta_min": _TIP_THETA,
            "theta_max": _TIP_THETA,
            "prob": _TIP_PROB,
            "angle": _TIP_ANGLE,
        }
        self._rot_widgets: dict = {}
        for kind, default_text, label_text, field_width in (
            ("theta_min",    "-180", "theta_min", 56),
            ("theta_max",    "180",  "theta_max", 56),
            ("prob",         "0.5", "prob",   56),
            ("angle",        "45",  "angle",  56),
        ):
            lab = self._math_label(label_text)
            lab.setStyleSheet("color: #6b7280; font-size: 11px;")
            le = (QLineEdit(default_text) if kind == "prob"
                  else DegreeLineEdit(float(default_text)))
            le.setFixedWidth(field_width if kind == "prob" else 86)
            if kind != "prob":
                lab.setText(lab.text() + " (deg)")
            le.setPlaceholderText(label_text)
            lab.setToolTip(_rot_tips[kind]); le.setToolTip(_rot_tips[kind])
            controls_row.addWidget(lab); controls_row.addWidget(le)
            self._rot_widgets[kind] = (lab, le)
        controls_row.addStretch(1)
        # Position rotation controls after the exponent and kappa rows.

        self.theta_min = self._rot_widgets["theta_min"][1]
        self.theta_max = self._rot_widgets["theta_max"][1]
        self.rot_prob = self._rot_widgets["prob"][1]
        self.rot_angle = self._rot_widgets["angle"][1]

        self._populate_rotation_modes()
        self.rotation_enable.toggled.connect(self._sync_rotation_visibility)
        self.rotation_partial_check.toggled.connect(self._on_rotation_category_toggled)
        self.rotation_combo.currentIndexChanged.connect(self._sync_rotation_visibility)
        self._sync_rotation_visibility()
        bv.addWidget(self.position_section)

        # Neutralization on a single line: [✓ Enable]  Delta:[ ]  r_ref:[ ]
        # When Enable is off, Delta and r_ref grey out and become read-only —
        # they're paper-meaningful only when neutralisation is on (Eq 8 / 12).
        self.shape_section = CollapsibleSection(
            "Shape Parameters", flat=True, collapsible=False)
        self.shape_section._header.hide()
        # Each row shows ONLY what the current mode uses (per the paper). Row
        # widths therefore vary by mode — fewer columns when the mode needs
        # fewer inputs (e.g. kappa mode 5 'All-isotropic 1.0' shows nothing
        # after the dropdown).
        # κ and p are ± pairs. Each renders coupled by default (one row whose
        # settings drive both directions; minus = copy of plus), with an
        # "Separate ± directions" checkbox to split them into separate κ⁺/κ⁻ rows.
        for base in ("p", "kappa"):
            plus, minus = f"{base}_plus", f"{base}_minus"
            cb = self._make_pair_checkbox()
            plus_row = self._mode_row_widget(
                plus, self.vector_rows, *default_range(plus),
                modes=MODES_FOR[plus], extras_spec=EXTRAS_FOR[plus])
            self._attach_pair_checkbox(plus_row, cb)
            self._attach_row_help(
                plus_row, self._parameter_help_button(plus, MODES_FOR[plus]))
            self.shape_section.body_layout.addWidget(plus_row)
            minus_row = self._mode_row_widget(
                minus, self.vector_rows, *default_range(minus),
                modes=MODES_FOR[minus], extras_spec=EXTRAS_FOR[minus])
            # Keep one visible help icon per ± pair to avoid duplicated hints.
            self._parameter_help_button(minus, MODES_FOR[minus])
            self._anchor_row_tail(minus_row)
            self.shape_section.body_layout.addWidget(minus_row)
            self._register_pair(base, plus, minus, cb,
                                self.vector_rows, self._pairs,
                                self._pair_coupled,
                                show_minus_row=False)
            if base == "p":
                self.p_single_row = QWidget()
                ps = QHBoxLayout(self.p_single_row)
                ps.setContentsMargins(0, 0, 0, 0)
                p_single_lbl = self._lbl("p_single")
                p_single_lbl.setToolTip(_TIP_PSINGLE)
                ps.addWidget(p_single_lbl)
                self.p_single_spin = NoWheelDoubleSpinBox()
                self.p_single_spin.setButtonSymbols(_NO_BUTTONS)
                self.p_single_spin.setDecimals(3); self.p_single_spin.setRange(0.001, 1e6)
                # Form B's single exponent defaults to p=0.5 (a linear/cone basin).
                self.p_single_spin.setValue(0.5)
                self.p_single_spin.setMaximumWidth(_SPIN_W)
                self.p_single_spin.setToolTip(_TIP_PSINGLE)
                ps.addWidget(self.p_single_spin)
                ps.addWidget(self._parameter_help_button("p_single"))
                ps.addStretch(1)
                self.shape_section.body_layout.addWidget(self.p_single_row)
        self.shape_section.body_layout.addWidget(self._rotation_row_widget)
        bv.addWidget(self.shape_section)

        self.neutralization_section = CollapsibleSection(
            "Neutralization", flat=True, collapsible=False)
        self.neutralization_section._header.hide()
        nrow = QHBoxLayout()
        nrow.setContentsMargins(0, 0, 0, 0)
        nrow.setSpacing(8)
        self._neutralization_header_layout = nrow
        self._neutralization_label_w = self._long_parameter_label_w
        self.neutralization_inline_title = self._lbl(
            "Neutralization", w=self._neutralization_label_w)
        self.neutralization_inline_title.setStyleSheet("font-weight: 400;")
        nrow.addWidget(self.neutralization_inline_title)
        self.neutralize_enable = QCheckBox("Enable")
        self.neutralize_enable.setChecked(False)
        self.neutralize_enable.setToolTip(_TIP_NEUTRALIZE)
        nrow.addWidget(self.neutralize_enable)
        # Help icon sits next to the Enable checkbox so it stays visible even
        # when neutralization is disabled (the Delta/r_ref row is hidden then).
        nrow.addWidget(self._parameter_help_button("neutralization"))
        nrow.addStretch(1)
        self.neutralization_section.body_layout.addLayout(nrow)

        self._neutralization_controls_row_widget = QWidget()
        controls_row = QHBoxLayout(self._neutralization_controls_row_widget)
        controls_row.setContentsMargins(self._neutralization_label_w + 4, 0, 0, 0)
        controls_row.setSpacing(8)
        self._neutralization_controls_layout = controls_row
        delta_lbl = self._math_label("Delta:")
        delta_lbl.setToolTip(_TIP_DELTA)
        controls_row.addWidget(delta_lbl)
        self.delta_spin = NoWheelDoubleSpinBox()
        self.delta_spin.setButtonSymbols(_NO_BUTTONS)
        self.delta_spin.setDecimals(3); self.delta_spin.setRange(0.001, 1e6)
        self.delta_spin.setValue(100.0)
        self.delta_spin.setMaximumWidth(_SPIN_W)
        self.delta_spin.setToolTip(_TIP_DELTA)
        controls_row.addWidget(self.delta_spin)
        rref_lbl = self._math_label("r_ref:")
        rref_lbl.setToolTip(_TIP_RREF)
        controls_row.addWidget(rref_lbl)
        self.rref_spin = NoWheelDoubleSpinBox()
        self.rref_spin.setButtonSymbols(_NO_BUTTONS)
        self.rref_spin.setDecimals(3); self.rref_spin.setRange(0.001, 1e6)
        self.rref_spin.setValue(100.0)
        self.rref_spin.setMaximumWidth(_SPIN_W)
        self.rref_spin.setToolTip(_TIP_RREF)
        controls_row.addWidget(self.rref_spin)
        controls_row.addStretch(1)
        self.neutralization_section.body_layout.addWidget(
            self._neutralization_controls_row_widget)
        self.neutralize_enable.toggled.connect(self._sync_neutralize_inputs)
        self._sync_neutralize_inputs(self.neutralize_enable.isChecked())
        bv.addWidget(self.neutralization_section)

        # Transform Sequence lives inside the Component Parameters window (one
        # scrolling editor), not a separate card below it.
        tf = CollapsibleSection(
            "Transform Sequence", flat=True, collapsible=False)
        trow = QHBoxLayout()
        type_lbl = self._lbl("Type")
        type_lbl.setToolTip(_TIP_TRANSFORM_TYPE)
        trow.addWidget(type_lbl)
        self.transform_combo = NoWheelComboBox()
        for name in self._transforms:
            display_name = self._transform_display_name(name)
            self.transform_combo.addItem(display_name, name)
            self.transform_combo.setItemData(
                self.transform_combo.count() - 1,
                display_name,
                Qt.ItemDataRole.ToolTipRole,
            )
        self.transform_combo.currentText = self._current_transform_name
        self.transform_combo.setCurrentText = self._set_current_transform_name
        self.transform_combo.setFixedWidth(_DROPDOWN_W)
        self._sync_transform_combo_tooltip()
        self.transform_combo.currentIndexChanged.connect(
            self._sync_transform_combo_tooltip)
        trow.addWidget(self.transform_combo)
        trow.addWidget(self._parameter_help_button("transform_type"))
        trow.addStretch(1)
        tf.body_layout.addLayout(trow)
        self._tf_params_box = QWidget()
        self._tf_params_layout = QVBoxLayout(self._tf_params_box)
        self._tf_params_layout.setContentsMargins(0, 0, 0, 0)
        self._tf_params_layout.setSpacing(4)
        self._tf_params_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        tf.body_layout.addWidget(self._tf_params_box)
        self.add_transform_btn = QPushButton("+ Add Transform")
        tf.body_layout.addWidget(self.add_transform_btn)

        # Transform sequence list. Each row carries a drag-handle, order number,
        # transform name, and Delete icon. Row click loads it into the fields
        # above; Apply Changes writes the edited values back.
        from PySide6.QtWidgets import QAbstractItemView as _AIV
        self.transform_list = QListWidget()
        # The transform list expands to fit its rows within the component editor.
        self.transform_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.transform_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.transform_list.setSelectionMode(_AIV.SelectionMode.SingleSelection)
        self.transform_list.setDragDropMode(_AIV.DragDropMode.InternalMove)
        self.transform_list.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.transform_list.model().rowsMoved.connect(self._on_transform_rows_moved)
        # Row click loads the row's data into the editor fields above and makes
        # it the current transform Apply Changes target.
        self.transform_list.itemClicked.connect(self._on_transform_row_clicked)
        tf.body_layout.addWidget(self.transform_list)
        # Hidden removal control for signal wiring and programmatic access.
        # Visible per-row × icons remove individual transformations.
        self.remove_transform_btn = QPushButton()
        self.remove_transform_btn.hide()
        bv.addWidget(tf)
        self.transform_card = tf   # toggled with the other editor sections

        # No trailing stretch: the editor hugs its sections so short/hidden
        # states don't leave a big empty grey box (the scroll handles overflow).
        # Expose the editor body so _on_tree_selection_changed can lock the
        # component-editing controls when no block is active while keeping
        # contextual help buttons clickable.
        self._editor_body = body
        v.addWidget(body)

        self.add_btn = QPushButton("+ Add Component")
        self.add_btn.setProperty("variant", "primary")
        self.apply_changes_btn = QPushButton("Apply Changes")
        self.apply_changes_btn.setProperty("variant", "primary")
        self.reset_btn = QPushButton("Reset Parameters")
        component_actions = QHBoxLayout()
        component_actions.setContentsMargins(0, 0, 0, 0)
        component_actions.setSpacing(8)
        component_actions.addWidget(self.add_btn)
        component_actions.addWidget(self.apply_changes_btn)
        self.component_actions_layout = component_actions
        self._refresh_component_body_height()
        # Wrap the tall parameter column in a vertical scroll area so a short
        # window scrolls the cards instead of compressing them — without this
        # the section cards (which carry hard minimum heights) collide with the
        # action buttons below. Mirrors the Builder/Education tabs.
        scroll = QScrollArea()
        scroll.setObjectName("paramScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        self._param_scroll = scroll

        # Outer pane: scrollable cards on top, action buttons pinned in a bar
        # below the scroll area so they stay visible at any scroll position.
        button_bar = QWidget()
        bb = QVBoxLayout(button_bar)
        bb.setContentsMargins(10, 8, 10, 10)   # align with the cards' inset
        bb.setSpacing(8)
        bb.addLayout(component_actions)
        bb.addWidget(self.reset_btn)

        container = QWidget()
        container.setObjectName("parameterPane")
        cv = QVBoxLayout(container)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(scroll, stretch=1)
        cv.addWidget(button_bar)
        self._param_pane = container
        return container

    def _mode_row_widget(self, name: str, store: dict,
                         lo: float, hi: float,
                         modes: List[str],
                         extras_spec: Optional[dict] = None,
                         mode_values: Optional[List[int]] = None,
                         display_name: Optional[str] = None) -> QWidget:
        """Build a mode-aware row through the factory and mirror into ``store``.

        ``store`` receives the row for access through ``self.vector_rows`` or
        ``self._tf_vec_rows``; the factory's ``rows`` map owns the row metadata.
        ``display_name`` overrides the leading label while retaining ``name``
        as the data key.
        """
        w = self._row_factory.build(
            name, lo, hi, modes, extras_spec, mode_values=mode_values,
            display_name=display_name)
        w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        store[name] = self._row_factory.rows[name]
        return w

    def _on_mode_changed(self, name: str) -> None:
        """Delegate row visibility to ``ParamRowFactory.sync_visibility``."""
        self._row_factory.sync_visibility(name)

    @staticmethod
    def _mode_int(combo) -> int:
        """Read the selected engine mode through ``param_rows.mode_int``."""
        return mode_int(combo)

    @staticmethod
    def _set_mode_combo_value(combo, mode: int) -> None:
        """Select a combo item by stored engine mode, falling back to index order."""
        for i in range(combo.count()):
            data = combo.itemData(i)
            if data is not None and int(data) == int(mode):
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(min(max(0, int(mode) - 1), combo.count() - 1))

    def _transform_param_mode_options(self):
        if self._current_transform_name() in {
            "additive_periodic", "log_sinusoidal", "wavelet_mod",
            "tensor_interference"}:
            return _ADDITIVE_PERIODIC_PARAM_MODES, _ADDITIVE_PERIODIC_PARAM_MODE_VALUES
        return TRANSFORM_PARAM_MODES, None

    def _hidden_transform_scalar_specs(self, existing: Optional[dict] = None) -> dict:
        """Scalar specs used by the engine but intentionally hidden in the GUI."""
        transform_name = self._current_transform_name()
        hidden_by_transform = {
            "log_sinusoidal": {"sigma"},
            "wavelet_mod": {"eps"},
            "radial_polytrig": {"eps"},
        }
        hidden_names = hidden_by_transform.get(transform_name, set())
        if not hidden_names:
            return {}
        specs: dict = {}
        for scalar_name in hidden_names:
            if existing and scalar_name in existing:
                scalar = existing[scalar_name]
                if isinstance(scalar, dict):
                    specs[scalar_name] = dict(scalar)
                else:
                    specs[scalar_name] = {"lo": str(scalar), "hi": str(scalar)}
                continue
            lo, hi = self._transforms[transform_name]["scalar"][scalar_name]
            specs[scalar_name] = {"lo": str(lo), "hi": str(hi)}
        return specs

    def _on_baseline_changed_with_resize(self, text: str) -> None:
        """Wrapper: baseline change toggles p_+/p_-/p_single, then resize."""
        self._on_baseline_changed(text)
        self._refresh_component_body_height()
        self._recompute_panel_min_width()

    def _on_baseline_changed(self, text: str) -> None:
        is_a = (text == "A")
        # Baseline A uses the p⁺/p⁻ pair; Baseline B uses the single exponent p.
        self.vector_rows["p_plus"]["widget"].setVisible(is_a)
        self._pairs["p"]["checkbox"].setVisible(is_a)
        if is_a:
            self._on_mode_changed("p_plus")
            self._on_mode_changed("p_minus")
            # Let the pair's coupled state decide whether p⁻ shows.
            self._apply_pair_visibility("p", self._pairs)
        else:
            self.vector_rows["p_minus"]["widget"].setVisible(False)
        self.p_single_row.setVisible(not is_a)
        self.p_single_spin.setEnabled(not is_a)

    def _refresh_component_body_height(self) -> None:
        """Prevent Qt from vertically squeezing the integrated parameter body."""
        body = getattr(self, "_editor_body", None)
        if body is None:
            return
        for section in (
            getattr(self, "base_section", None),
            getattr(self, "position_section", None),
            getattr(self, "shape_section", None),
            getattr(self, "neutralization_section", None),
        ):
            if section is None or not hasattr(section, "_body"):
                continue
            section_body = section._body
            if section_body.layout() is not None:
                section_body.layout().invalidate()
                section_body.layout().activate()
            section_body.setMinimumHeight(section_body.sizeHint().height())
            section_layout = section.layout()
            if section_layout is not None:
                section_layout.invalidate()
                section_layout.activate()
            section.setMinimumHeight(section.sizeHint().height())
        body.layout().invalidate()
        body.layout().activate()
        body.setMinimumHeight(body.sizeHint().height())

    def _set_component_editor_visible(self, visible: bool) -> None:
        """Show/hide the component-editor sections as one group."""
        for section in (
            getattr(self, "base_section", None),
            getattr(self, "position_section", None),
            getattr(self, "shape_section", None),
            getattr(self, "neutralization_section", None),
            getattr(self, "transform_card", None),
        ):
            if section is not None:
                section.setVisible(visible)

    def _set_component_editor_enabled(self, active: bool) -> None:
        """Lock editor controls until a block exists, without locking help."""
        for widget in (
            self.seed_spin, self.baseline_combo, self.beta_spin,
            self.rotation_combo, self.neutralize_enable, self.transform_combo,
            self.add_transform_btn, self.transform_list, self.remove_transform_btn,
        ):
            widget.setEnabled(active)

        for pairs in (self._pairs, self._tf_pairs):
            for info in pairs.values():
                info["checkbox"].setEnabled(active)

        for rows in (self.vector_rows, self._tf_vec_rows):
            for entry in rows.values():
                for key in ("mode", "vmin", "vmax", "_vmin_label", "_vmax_label"):
                    control = entry.get(key)
                    if control is not None:
                        control.setEnabled(active)
                for control in entry.get("_extra_labels", {}).values():
                    control.setEnabled(active)
                for control in entry.get("_extras", {}).values():
                    control.setEnabled(active)

        for lab, edit in self._rot_widgets.values():
            lab.setEnabled(active)
            edit.setEnabled(active)

        for widgets in self._tf_scalar_widgets.values():
            widgets["lo"].setEnabled(active)
            widgets["hi"].setEnabled(active)

        self.p_single_spin.setEnabled(active and self.baseline_combo.currentText() != "A")
        self.delta_spin.setEnabled(active and self.neutralize_enable.isChecked())
        self.rref_spin.setEnabled(active and self.neutralize_enable.isChecked())

    def _reset_parameter_fields(self) -> None:
        # Restore field defaults, including the seed the current editor session
        # started on (New Component's auto-assigned seed, or a selected
        # component's saved seed) -- not a hardcoded 42 and not the value the
        # user may have typed since. No popup: callers that want the "reset"
        # confirmation (on_reset) show it themselves.
        self.baseline_combo.setCurrentText("A")
        self.seed_spin.setValue(self._editor_seed_default)
        self.beta_spin.setValue(0.0)
        self.neutralize_enable.setChecked(False)
        self.delta_spin.setValue(100.0)
        self.rref_spin.setValue(100.0)
        self.p_single_spin.setValue(0.5)        # Form B default exponent
        for name in _VECTORS:
            r = self.vector_rows[name]
            r["mode"].setCurrentIndex(0)            # mode 1 = first item in every menu
            lo, hi = default_range(name)
            r["vmin"].setText(str(lo)); r["vmax"].setText(str(hi))
            for k, w in self._extra_widgets.get(name, {}).items():
                w.setText(EXTRA_DEFAULTS.get(k, ""))
        # Recouple every ± pair (the default), so reset returns to matched ±.
        for info in self._pairs.values():
            info["checkbox"].setChecked(False)
        self.rotation_enable.setChecked(False)
        self.rotation_partial_check.setChecked(False)
        self._populate_rotation_modes(preferred_mode=2)
        self.theta_min.set_degrees(-180); self.theta_max.set_degrees(180)
        self.rot_prob.setText("0.5"); self.rot_angle.set_degrees(45)
        self._block_diag_spec = {"grouping": "auto", "angle_mode": "fixed"}
        self._sync_rotation_visibility()
        self._pending_transforms.clear()
        self._refresh_transform_list()
        self.transform_combo.setCurrentIndex(0)
        self._rebuild_transform_params()

    def on_reset(self) -> None:
        """Reset Parameters button: blank the editor and confirm to the user."""
        self._reset_parameter_fields()
        QMessageBox.information(
            self, "Reset Parameters", "Parameters have been reset.")

    # ------------------------------------------------------------- generation
    @staticmethod
    def _f(text: str, default: float = 0.0) -> float:
        """Parse a QLineEdit string to float, using ``default`` for empty input."""
        t = (text or "").strip()
        return float(t) if t else float(default)

    def _vector_spec_from_row(self, name: str) -> dict:
        """Capture the full mode + range + extras spec for one parameter row."""
        entry = self.vector_rows[name]
        spec = {
            "mode": self._mode_int(entry["mode"]),
            "vmin": self._f(entry["vmin"].text(), 0.0),
            "vmax": self._f(entry["vmax"].text(), 0.0),
        }
        for k, w in self._extra_widgets.get(name, {}).items():
            spec[k] = self._f(w.text(), float(EXTRA_DEFAULTS.get(k, "0") or "0"))
        return spec

    @staticmethod
    def _rng_for(seed: int, counters: dict, copies: dict = None,
                 seed_overrides: dict = None):
        """Return ``key -> RandomState``: one independent stream per parameter.

        Each stream is reproducible from ``(seed, key, counter)`` so a component
        regenerates identically, a matching seed reproduces it exactly, and
        bumping one ``counter`` rerolls that one stream alone. A key present in
        ``copies`` (``{key: {"seed", "counter"}}``) instead derives from that
        source component's seed+counter — the "copy a parameter" override.
        """
        return generator_sampling.rng_for(seed, counters, copies, seed_overrides)

    def _component_rng_for(self, component):
        """Stream factory for a stored component, honoring its copy overrides."""
        return self._rng_for(
            int(component.seed),
            getattr(component, "reroll_counters", None) or {},
            getattr(component, "param_copies", None) or {},
            getattr(component, "param_seed_overrides", None) or {})

    def _generate_component_vectors(self, d: int, rng_for, baseline: str) -> dict:
        """Sample c and the κ / p ± pairs, honouring each pair's coupled state.

        A coupled pair (the default) draws the ``+`` vector once and copies it
        to ``−`` — the paper's matched ± case (κ⁺=κ⁻). An independent pair
        draws each direction separately.
        """
        generated = {"c": self._sample_vector_from_spec(
            "c", self._vector_spec_from_row("c"), d, rng_for("c"))}
        self._sample_component_pair(generated, "kappa", d, rng_for)
        if baseline != "B":
            self._sample_component_pair(generated, "p", d, rng_for)
        return generated

    def _sample_component_pair(self, generated: dict, base: str, d: int, rng_for) -> None:
        plus, minus = f"{base}_plus", f"{base}_minus"
        plus_spec = self._vector_spec_from_row(plus)
        generated[plus] = self._sample_vector_from_spec(plus, plus_spec, d, rng_for(plus))
        if self._pair_coupled.get(base, True):
            generated[minus] = generated[plus].copy()
        else:
            generated[minus] = self._sample_vector_from_spec(minus, plus_spec, d, rng_for(minus))

    def _copy_candidates(self, component, key: str) -> list:
        """Other components a parameter ``key`` can be copied from: same block
        dimension and baseline, excluding self, that actually contain ``key``.

        For a transform key ``tf{i}:{pname}`` the candidate must also have the
        same transform *type* at index ``i`` with that parameter, so the copy is
        meaningful.
        """
        block = self._block_of(component)
        if block is None:
            return []
        d_b = int(block.d_b)
        is_tf = key.startswith("tf")
        if is_tf:
            head, _, pname = key.partition(":")
            t_index = int(head[2:])
            my_specs = getattr(component, "transform_specs", None) or []
            if t_index >= len(my_specs):
                return []
            my_tf_name = my_specs[t_index].get("name")
        out = []
        for b in self.inventory.blocks:
            if int(b.d_b) != d_b:
                continue
            for c in b.components:
                if c is component or c.baseline != component.baseline:
                    continue
                if is_tf:
                    c_specs = getattr(c, "transform_specs", None) or []
                    if t_index >= len(c_specs):
                        continue
                    if c_specs[t_index].get("name") != my_tf_name:
                        continue
                    if pname not in (c_specs[t_index].get("vec", {}) or {}):
                        continue
                    out.append(c)
                elif key in (c.generated or {}):
                    out.append(c)
        return out

    def _block_of(self, component):
        """The block owning ``component``, or None."""
        for b in self.inventory.blocks:
            if component in b.components:
                return b
        return None

    def _component_label(self, block) -> str:
        """Next component label as ``component<block#>_<comp#>`` (1-based)."""
        try:
            bnum = self.inventory.blocks.index(block) + 1
        except ValueError:
            bnum = 1
        return f"component{bnum}_{len(block.components) + 1}"

    def _reindex_component_labels(self) -> None:
        """Renumber every component label to ``component<block#>_<comp#>`` so
        deletions never leave gaps or duplicate numbers."""
        for bi, block in enumerate(self.inventory.blocks):
            for ci, comp in enumerate(block.components):
                comp.label = f"component{bi + 1}_{ci + 1}"

    def _advance_seed_box(self) -> None:
        """Move the seed box to the next free seed (one past the highest existing).

        Called when starting a New Component so each fresh component gets a
        distinct seed. With no components yet, the current seed is kept (the
        first component uses whatever the box shows, default 42).
        """
        existing = [int(c.seed) for b in self.inventory.blocks for c in b.components]
        if existing:
            self.seed_spin.setValue(max(existing) + 1)

    # Vector modes that draw no randomness: rerolling them changes nothing, so
    # they get no reroll button.
    _VECTOR_DETERMINISTIC_MODES = {
        "c": set(),
        "kappa_plus": {5}, "kappa_minus": {5},
        "p_plus": {2, 3}, "p_minus": {2, 3},
    }

    def _vector_key_rerollable(self, component, key: str) -> bool:
        """True when ``key`` has a live random stream the user can reroll."""
        spec = (component.vector_specs or {}).get(key)
        if spec is None:
            return False
        mode = int(spec.get("mode", 1))
        if mode in self._VECTOR_DETERMINISTIC_MODES.get(key, set()):
            return False
        base, sign = self._split_pair(key)
        if sign == "minus" and base in {"kappa", "p"}:
            # Coupled − mirrors +, so reroll happens on the + row instead.
            coupled = getattr(component, "pair_coupled", None) or {}
            if coupled.get(base, True):
                return False
        return True

    def _populate_value_row(self, table, row_idx, generated, key, d, value_col=1):
        """Refresh one parameter row's value cells from ``generated``."""
        for dim_idx in range(d):
            table.setItem(
                row_idx, value_col + dim_idx,
                self._centered_table_item(
                    self._vector_value_at(generated, key, dim_idx)))

    def _make_action_column_toggles(self):
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        checks = {
            "reroll": QCheckBox("Reroll"),
            "seed": QCheckBox("Seeded reroll"),
            "share": QCheckBox("Copy"),
        }
        checks["reroll"].setChecked(True)
        checks["seed"].setChecked(False)
        checks["share"].setChecked(False)
        for cb in checks.values():
            row.addWidget(cb)
        row.addStretch(1)
        tables = []

        def apply_to(table):
            table.setColumnHidden(0, not checks["reroll"].isChecked())
            table.setColumnHidden(1, not checks["seed"].isChecked())
            table.setColumnHidden(2, not checks["share"].isChecked())
            self._fit_table_to_rows(table)

        def apply_all(*_args):
            for table in tables:
                apply_to(table)

        for cb in checks.values():
            cb.toggled.connect(apply_all)

        def register(table):
            tables.append(table)
            apply_to(table)

        return holder, register

    def _make_vector_reroll_button(self, component, key, label, table, row_of, d,
                                   value_col=1, tint=None, on_changed=None):
        """An icon button that rerolls ``key`` and live-refreshes its (and mirror) cells."""
        from portal_studio.gui.common.icons import make_reroll_icon
        btn = QPushButton()
        btn.setIcon(make_reroll_icon())
        btn.setFixedSize(22, 22)
        btn.setObjectName("nodeActionBtn")
        base = label.rstrip("+-⁺⁻")   # name the variable, not its ± side
        btn.setToolTip(
            f"Reroll {base}: redraw fresh values for this parameter only, "
            "from its current mode and range, leaving every other parameter "
            "unchanged.")
        btn.setAccessibleName(btn.toolTip())
        btn.setAccessibleDescription(btn.toolTip())
        ncols = value_col + d

        def _refresh(k):
            self._populate_value_row(table, row_of[k], component.generated, k, d, value_col)
            if tint is not None:   # new items lose the tint → re-apply it
                self._tint_row(table, row_of[k], tint, ncols)

        def _refresh_after_change():
            _refresh(key)
            base, sign = self._split_pair(key)
            if base in {"kappa", "p"} and sign == "plus":
                mkey = f"{base}_minus"
                if mkey in row_of:
                    _refresh(mkey)

        def _do_reroll():
            self._reroll_param(component, key)
            _refresh_after_change()
            if on_changed is not None:
                on_changed()

        btn.clicked.connect(_do_reroll)
        return btn

    def _make_vector_seed_button(self, component, key, label, table, row_of, d,
                                 value_col=1, tint=None, on_changed=None):
        """An icon button that rerolls ``key`` from a user-entered row seed."""
        from portal_studio.gui.common.icons import make_seed_icon
        base = label.replace("+", "").replace("-", "")
        ncols = value_col + d

        def _refresh(k):
            self._populate_value_row(table, row_of[k], component.generated, k, d, value_col)
            if tint is not None:
                self._tint_row(table, row_of[k], tint, ncols)

        def _refresh_after_change():
            _refresh(key)
            base_name, sign = self._split_pair(key)
            if base_name in {"kappa", "p"} and sign == "plus":
                mkey = f"{base_name}_minus"
                if mkey in row_of:
                    _refresh(mkey)

        def _do_seeded_reroll():
            from PySide6.QtWidgets import QInputDialog
            seed_overrides = getattr(component, "param_seed_overrides", None) or {}
            current = int(seed_overrides.get(key, component.seed))
            seed, ok = QInputDialog.getInt(
                self, "Reroll with seed", "Seed:", current, 0, 1_000_000, 1)
            if ok:
                self._set_param_seed_override(component, key, seed)
                _refresh_after_change()
                if on_changed is not None:
                    on_changed()

        def _do_clear_seed():
            self._clear_param_seed_override(component, key)
            _refresh_after_change()
            if on_changed is not None:
                on_changed()

        btn = QPushButton()
        btn.setIcon(make_seed_icon())
        btn.setFixedSize(22, 22)
        btn.setObjectName("rowSeedBtn")
        btn.setToolTip(
            f"Reroll {base} with a seed; manual seed overrides auto-reroll; "
            "right-click to clear this row seed"
        )
        btn.setAccessibleName(btn.toolTip())
        btn.setAccessibleDescription(btn.toolTip())
        btn.setStyleSheet(
            "QPushButton#rowSeedBtn {"
            " border: 1px solid #cbd5e1; border-radius: 4px;"
            " background: #ffffff; color: #4c5870;"
            " padding: 0;"
            "}"
            "QPushButton#rowSeedBtn:hover { background: #eaf3ff; }")
        btn.clicked.connect(_do_seeded_reroll)
        btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        def _show_seed_context(pos):
            if key not in (getattr(component, "param_seed_overrides", None) or {}):
                return
            menu = QMenu(btn)
            menu.addAction("Clear row seed", _do_clear_seed)
            menu.exec(btn.mapToGlobal(pos))

        btn.customContextMenuRequested.connect(_show_seed_context)
        return btn

    @staticmethod
    def _centered_icon_cell(btn, bg=None) -> QWidget:
        """A table cell holding a single icon button, centred (optional tint)."""
        cell = QWidget()
        if bg is not None:
            cell.setAutoFillBackground(True)
            cell.setStyleSheet(f"background-color: {bg.name()};")
        lay = QHBoxLayout(cell)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addStretch(1)
        lay.addWidget(btn)
        lay.addStretch(1)
        return cell

    @staticmethod
    def _tint_row(table, row_idx: int, color, ncols: int) -> None:
        """Tint a row's label + value cells (column 0 holds a widget, skipped)."""
        for col in range(1, ncols):
            item = table.item(row_idx, col)
            if item is not None:
                item.setBackground(color)

    def _regenerate_param(self, component, key: str) -> None:
        """Resample one parameter (vector or transform) from the component's
        current seed / counters / param_copies. A copy override may carry the
        source's spec so the value matches the source even when this component's
        own range/mode differs. Mirrors coupled ± pairs."""
        block = self._block_of(component)
        if block is None:
            return
        d = int(block.d_b)
        rng_for = self._component_rng_for(component)
        override = (getattr(component, "param_copies", None) or {}).get(key) or {}
        if key.startswith("tf"):
            head, _, pname = key.partition(":")
            t_index = int(head[2:])
            specs = getattr(component, "transform_specs", None) or []
            if t_index >= len(specs):
                return
            vec = specs[t_index].get("vec", {}) or {}
            if pname not in vec:
                return
            spec = override.get("spec") or vec[pname]
            arr = sample_transform_inner_param(
                rng_for(key), K=1, d=d,
                vmin=self._f(spec.get("vmin", "0"), 0.0),
                vmax=self._f(spec.get("vmax", "0"), 0.0),
                mode=int(spec.get("mode", 1)),
                beta_alpha=self._f(spec.get("alpha", "0.2"), 0.2),
                beta_beta=self._f(spec.get("beta", "0.2"), 0.2),
            )
            params = component.transforms[t_index].setdefault("params", {})
            params[pname] = arr.reshape(-1)
            base, sign = self._split_pair(pname)
            coupled = specs[t_index].get("coupled", {}) or {}
            if sign == "plus" and base in coupled and coupled.get(base, True):
                params[f"{base}_minus"] = params[pname].copy()
        else:
            spec = override.get("spec") or (component.vector_specs or {}).get(key)
            if spec is None:
                return
            component.generated[key] = self._sample_vector_from_spec(
                key, spec, d, rng_for(key))
            base, sign = self._split_pair(key)
            if base in {"kappa", "p"} and sign == "plus":
                coupled = getattr(component, "pair_coupled", None) or {}
                if coupled.get(base, True):
                    component.generated[f"{base}_minus"] = \
                        component.generated[key].copy()

    def _effective_source_key(self, source, key: str) -> str:
        """The stream key that actually produced ``source``'s value for ``key``.

        A coupled ``−`` side is a mirror of its ``+`` side, so its value comes
        from the ``+`` stream, not its own. Return the ``+`` key in that case.
        """
        base, sign = self._split_pair(key)
        if sign != "minus":
            return key
        if key.startswith("tf"):
            head, _, pname = key.partition(":")     # head="tf0", pname="mu_minus"
            pbase, _psign = self._split_pair(pname)  # pbase="mu" (strip the tf prefix)
            i = int(head[2:])
            specs = getattr(source, "transform_specs", None) or []
            t_coupled = (specs[i].get("coupled", {}) if i < len(specs) else {}) or {}
            if t_coupled.get(pbase, True):
                return f"{head}:{pbase}_plus"
        else:
            coupled = getattr(source, "pair_coupled", None) or {}
            if base in {"kappa", "p"} and coupled.get(base, True):
                return f"{base}_plus"
        return key

    @staticmethod
    def _source_spec_for(source, key: str) -> dict:
        """The source component's spec dict for parameter ``key``."""
        if key.startswith("tf"):
            head, _, pname = key.partition(":")
            i = int(head[2:])
            specs = getattr(source, "transform_specs", None) or []
            if i < len(specs):
                return dict((specs[i].get("vec", {}) or {}).get(pname, {}) or {})
            return {}
        return dict((getattr(source, "vector_specs", None) or {}).get(key, {}) or {})

    def _stream_descriptor_for(self, source, key: str) -> dict:
        src_key = self._effective_source_key(source, key)
        copy = (getattr(source, "param_copies", None) or {}).get(src_key)
        if copy:
            return {
                "seed": int(copy["seed"]),
                "counter": int(copy.get("counter", 0)),
                "src_key": copy.get("src_key", src_key),
                "spec": dict(copy.get("spec") or self._source_spec_for(source, src_key)),
            }
        seed_overrides = getattr(source, "param_seed_overrides", None) or {}
        counters = getattr(source, "reroll_counters", None) or {}
        return {
            "seed": int(seed_overrides.get(src_key, source.seed)),
            "counter": int(counters.get(src_key, 0)),
            "src_key": src_key,
            "spec": self._source_spec_for(source, src_key),
        }

    def _param_value_array(self, component, key: str):
        """Current value array for parameter ``key`` (vector or transform)."""
        if key.startswith("tf"):
            head, _, pname = key.partition(":")
            i = int(head[2:])
            tfs = getattr(component, "transforms", None) or []
            if i < len(tfs):
                return (tfs[i].get("params", {}) or {}).get(pname)
            return None
        return (getattr(component, "generated", None) or {}).get(key)

    @staticmethod
    def _mirror_data_key(data_key: str, base: str) -> str:
        """The ``−`` side data key for a coupled ``+`` data key."""
        if data_key.startswith("tf"):
            prefix = data_key.rsplit(":", 1)[0]
            return f"{prefix}:{base}_minus"
        return f"{base}_minus"

    def _copy_param_from(self, component, key: str, source) -> None:
        """Snapshot-copy ``key`` from ``source``: borrow the source's actual
        stream (seed+counter+spec) for that value and regenerate this
        component's parameter to match the source exactly."""
        if getattr(component, "param_copies", None) is None:
            component.param_copies = {}
        stream = self._stream_descriptor_for(source, key)
        component.param_copies[key] = {
            "from": source.label, "seed": int(stream["seed"]),
            "counter": int(stream["counter"]), "spec": dict(stream["spec"]),
            "src_key": stream["src_key"]}
        self._regenerate_param(component, key)

    def _share_pair_from(self, component, base: str, source,
                         transform_index: int = None) -> bool:
        """Make a ± pair identical to ``source``: adopt the source's coupling
        (separate/coupled) and copy the value(s). Returns whether the target's
        direction changed."""
        if transform_index is None:
            plus, minus = f"{base}_plus", f"{base}_minus"
            src_coupled = bool((getattr(source, "pair_coupled", None) or {}).get(base, True))
            if getattr(component, "pair_coupled", None) is None:
                component.pair_coupled = {}
            before = bool(component.pair_coupled.get(base, True))
            component.pair_coupled[base] = src_coupled
        else:
            i = transform_index
            plus, minus = f"tf{i}:{base}_plus", f"tf{i}:{base}_minus"
            s_specs = getattr(source, "transform_specs", None) or []
            src_coupled = bool(
                (s_specs[i].get("coupled", {}) if i < len(s_specs) else {}).get(base, True))
            t_specs = getattr(component, "transform_specs", None) or []
            if i < len(t_specs):
                # transform_specs are shallow-copied across components, so the
                # nested 'coupled' dict can be shared. Privatise it before
                # mutating so sharing never changes another component.
                t_specs[i] = dict(t_specs[i])
                t_coupled = dict(t_specs[i].get("coupled", {}) or {})
                t_specs[i]["coupled"] = t_coupled
            else:
                t_coupled = {}
            before = bool(t_coupled.get(base, True))
            t_coupled[base] = src_coupled
        if src_coupled:
            (getattr(component, "param_copies", None) or {}).pop(minus, None)
            self._copy_param_from(component, plus, source)   # regenerate + and mirror −
        else:
            self._copy_param_from(component, plus, source)
            self._copy_param_from(component, minus, source)
        return before != src_coupled

    def _make_copy_button(self, component, key, table, row_of, d, value_col,
                          tint=None, row_key=None):
        """An icon button that shares ``key``'s value from a chosen component."""
        from portal_studio.gui.common.icons import make_clone_icon
        btn = QPushButton()
        btn.setIcon(make_clone_icon())
        btn.setFixedSize(22, 22)
        btn.setObjectName("nodeActionBtn")
        btn.setToolTip("Copy this parameter value from another component")
        btn.setAccessibleName(btn.toolTip())
        btn.setAccessibleDescription(btn.toolTip())
        btn.clicked.connect(
            lambda: self._do_copy(component, key, table, row_of, d, value_col, row_key))
        return btn

    def _make_share_pair_button(self, component, base, table, row_of, d, value_col,
                                tint=None, dialog=None, transform_index=None):
        """An icon button that shares a whole ± pair (direction + values)."""
        from portal_studio.gui.common.icons import make_clone_icon
        btn = QPushButton()
        btn.setIcon(make_clone_icon())
        btn.setFixedSize(22, 22)
        btn.setObjectName("nodeActionBtn")
        btn.setToolTip("Copy this +/- pair from another component, including direction and values")
        btn.setAccessibleName(btn.toolTip())
        btn.setAccessibleDescription(btn.toolTip())
        btn.clicked.connect(
            lambda: self._do_share_pair(component, base, table, row_of, d,
                                        value_col, dialog, transform_index))
        return btn

    def _do_share_pair(self, component, base, table, row_of, d, value_col,
                       dialog, transform_index=None):
        plus_key = (f"{base}_plus" if transform_index is None
                    else f"tf{transform_index}:{base}_plus")
        candidates = self._copy_candidates(component, plus_key)
        if not candidates:
            QMessageBox.information(
                self, "Copy parameter",
                "No compatible component (same dimension, baseline, and "
                "transform) to copy from.")
            return
        source = self._pick_copy_source(component, plus_key, candidates)
        if source is None:
            return
        changed = self._share_pair_from(component, base, source, transform_index)
        if changed and dialog is not None:
            self._rebuild_parameters_body(dialog)
            return
        plus_row, minus_row = f"{base}_plus", f"{base}_minus"
        if plus_row in row_of:
            self._refresh_value_cells(table, component, plus_key, row_of[plus_row], d, value_col)
        if minus_row in row_of:
            minus_key = (f"{base}_minus" if transform_index is None
                         else f"tf{transform_index}:{base}_minus")
            self._refresh_value_cells(table, component, minus_key, row_of[minus_row], d, value_col)

    def _pick_copy_source(self, component, key, candidates):
        """Show a list of candidate components; return the chosen one or None."""
        from PySide6.QtWidgets import QInputDialog
        labels = [c.label for c in candidates]
        choice, ok = QInputDialog.getItem(
            self, "Copy from component", "Component:", labels, 0, False)
        if not ok:
            return None
        return candidates[labels.index(choice)]

    def _do_copy(self, component, key, table, row_of, d, value_col, row_key=None):
        row_key = row_key or key
        candidates = self._copy_candidates(component, key)
        if not candidates:
            QMessageBox.information(
                self, "Copy parameter",
                "No compatible component (same dimension, baseline, and "
                "transform) to copy from.")
            return
        source = self._pick_copy_source(component, key, candidates)
        if source is None:
            return
        self._copy_param_from(component, key, source)
        if row_key in row_of:
            self._refresh_value_cells(
                table, component, key, row_of[row_key], d, value_col)
        base, sign = self._split_pair(row_key)
        minus_row = f"{base}_minus"
        if sign == "plus" and minus_row in row_of:
            self._refresh_value_cells(
                table, component, self._mirror_data_key(key, base),
                row_of[minus_row], d, value_col)

    def _refresh_value_cells(self, table, component, key, row_idx, d, value_col):
        arr = self._param_value_array(component, key)
        for dim_idx in range(d):
            txt = "" if arr is None else self._value_at(arr, dim_idx)
            table.setItem(
                row_idx, value_col + dim_idx, self._centered_table_item(txt))

    def _reroll_param(self, component, key: str) -> None:
        """Bump one stream's reroll counter and regenerate only that parameter.

        Rerolling also detaches any copy override on that key, so the row
        returns to the component's own seed.
        """
        block = self._block_of(component)
        if block is None or (component.vector_specs or {}).get(key) is None:
            return
        (getattr(component, "param_copies", None) or {}).pop(key, None)
        counters = component.reroll_counters
        counters[key] = int(counters.get(key, 0)) + 1
        self._regenerate_param(component, key)

    def _set_param_seed_override(self, component, key: str, seed: int) -> None:
        if getattr(component, "param_seed_overrides", None) is None:
            component.param_seed_overrides = {}
        (getattr(component, "param_copies", None) or {}).pop(key, None)
        component.param_seed_overrides[key] = int(seed)
        component.reroll_counters[key] = 0
        self._regenerate_param(component, key)

    def _clear_param_seed_override(self, component, key: str) -> None:
        (getattr(component, "param_copies", None) or {}).pop(key, None)
        (getattr(component, "param_seed_overrides", None) or {}).pop(key, None)
        component.reroll_counters[key] = 0
        self._regenerate_param(component, key)

    def _transform_param_rerollable(self, component, t_index: int, pname: str) -> bool:
        """True for a transform's vector inner-param with a live random stream."""
        specs = getattr(component, "transform_specs", None) or []
        if t_index >= len(specs):
            return False
        tspec = specs[t_index]
        if pname not in (tspec.get("vec", {}) or {}):
            return False   # only vector inner-params draw per-dimension randomness
        base, sign = self._split_pair(pname)
        coupled = tspec.get("coupled", {}) or {}
        if sign == "minus" and base in coupled and coupled.get(base, True):
            return False   # coupled − mirrors +, reroll happens on the + row
        return True

    def _reroll_transform_param(self, component, t_index: int, pname: str) -> None:
        """Bump and regenerate one transform vector inner-param's stream.

        Rerolling also detaches any copy override on that key.
        """
        block = self._block_of(component)
        specs = getattr(component, "transform_specs", None) or []
        if block is None or t_index >= len(specs):
            return
        if pname not in (specs[t_index].get("vec", {}) or {}):
            return
        key = f"tf{t_index}:{pname}"
        (getattr(component, "param_copies", None) or {}).pop(key, None)
        counters = component.reroll_counters
        counters[key] = int(counters.get(key, 0)) + 1
        self._regenerate_param(component, key)

    def _set_transform_seed_override(self, component, t_index: int, pname: str,
                                     seed: int) -> None:
        key = f"tf{t_index}:{pname}"
        if getattr(component, "param_seed_overrides", None) is None:
            component.param_seed_overrides = {}
        (getattr(component, "param_copies", None) or {}).pop(key, None)
        component.param_seed_overrides[key] = int(seed)
        component.reroll_counters[key] = 0
        self._regenerate_param(component, key)

    def _clear_transform_seed_override(self, component, t_index: int,
                                       pname: str) -> None:
        key = f"tf{t_index}:{pname}"
        (getattr(component, "param_copies", None) or {}).pop(key, None)
        (getattr(component, "param_seed_overrides", None) or {}).pop(key, None)
        component.reroll_counters[key] = 0
        self._regenerate_param(component, key)

    def _populate_transform_value_row(self, table, row_idx, params, pname, d, value_col=1):
        val = params.get(pname)
        for dim_idx in range(d):
            table.setItem(
                row_idx, value_col + dim_idx,
                self._centered_table_item("" if val is None else self._value_at(val, dim_idx)))

    def _make_transform_reroll_button(self, component, t_index, pname, label,
                                      table, row_of, d, value_col=1, tint=None,
                                      on_changed=None):
        from portal_studio.gui.common.icons import make_reroll_icon
        btn = QPushButton()
        btn.setIcon(make_reroll_icon())
        btn.setFixedSize(22, 22)
        btn.setObjectName("nodeActionBtn")
        base = label.rstrip("+-⁺⁻")   # name the variable, not its ± side
        btn.setToolTip(
            f"Reroll {base}: redraw fresh values for this parameter only, "
            "from its current mode and range, leaving every other parameter "
            "unchanged.")
        btn.setAccessibleName(btn.toolTip())
        btn.setAccessibleDescription(btn.toolTip())
        ncols = value_col + d

        def _refresh(k, params):
            self._populate_transform_value_row(table, row_of[k], params, k, d, value_col)
            if tint is not None:   # new items lose the tint → re-apply it
                self._tint_row(table, row_of[k], tint, ncols)

        def _refresh_after_change():
            params = component.transforms[t_index].get("params", {})
            _refresh(pname, params)
            base, sign = self._split_pair(pname)
            if sign == "plus":
                mkey = f"{base}_minus"
                if mkey in row_of:
                    _refresh(mkey, params)

        def _do_reroll():
            self._reroll_transform_param(component, t_index, pname)
            _refresh_after_change()
            if on_changed is not None:
                on_changed()

        btn.clicked.connect(_do_reroll)
        return btn

    def _make_transform_seed_button(self, component, t_index, pname, label,
                                    table, row_of, d, value_col=1, tint=None,
                                    on_changed=None):
        from portal_studio.gui.common.icons import make_seed_icon
        base = label.replace("+", "").replace("-", "")
        ncols = value_col + d

        def _refresh(k, params):
            self._populate_transform_value_row(table, row_of[k], params, k, d, value_col)
            if tint is not None:
                self._tint_row(table, row_of[k], tint, ncols)

        def _refresh_after_change():
            params = component.transforms[t_index].get("params", {})
            _refresh(pname, params)
            base_name, sign = self._split_pair(pname)
            if sign == "plus":
                mkey = f"{base_name}_minus"
                if mkey in row_of:
                    _refresh(mkey, params)

        def _do_seeded_reroll():
            from PySide6.QtWidgets import QInputDialog
            key = f"tf{t_index}:{pname}"
            seed_overrides = getattr(component, "param_seed_overrides", None) or {}
            current = int(seed_overrides.get(key, component.seed))
            seed, ok = QInputDialog.getInt(
                self, "Reroll with seed", "Seed:", current, 0, 1_000_000, 1)
            if ok:
                self._set_transform_seed_override(component, t_index, pname, seed)
                _refresh_after_change()
                if on_changed is not None:
                    on_changed()

        def _do_clear_seed():
            self._clear_transform_seed_override(component, t_index, pname)
            _refresh_after_change()
            if on_changed is not None:
                on_changed()

        btn = QPushButton()
        btn.setIcon(make_seed_icon())
        btn.setFixedSize(22, 22)
        btn.setObjectName("rowSeedBtn")
        btn.setToolTip(
            f"Reroll {base} with a seed; manual seed overrides auto-reroll; "
            "right-click to clear this row seed"
        )
        btn.setAccessibleName(btn.toolTip())
        btn.setAccessibleDescription(btn.toolTip())
        btn.setStyleSheet(
            "QPushButton#rowSeedBtn {"
            " border: 1px solid #cbd5e1; border-radius: 4px;"
            " background: #ffffff; color: #4c5870;"
            " padding: 0;"
            "}"
            "QPushButton#rowSeedBtn:hover { background: #eaf3ff; }")
        btn.clicked.connect(_do_seeded_reroll)
        btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        def _show_seed_context(pos):
            key = f"tf{t_index}:{pname}"
            if key not in (getattr(component, "param_seed_overrides", None) or {}):
                return
            menu = QMenu(btn)
            menu.addAction("Clear row seed", _do_clear_seed)
            menu.exec(btn.mapToGlobal(pos))

        btn.customContextMenuRequested.connect(_show_seed_context)
        return btn

    @staticmethod
    def _sample_vector_from_spec(name: str, spec: dict, d: int, rng) -> np.ndarray:
        return generator_sampling.sample_vector_from_spec(name, spec, d, rng)

    @staticmethod
    def _rotation_from_mode(d: int, rng, mode: int):
        """Regenerate rotation from a stored mode.

        The current data model stores the rotation mode but not every optional
        per-mode field, so dimension-change regeneration uses the editor's
        default ranges.
        """
        return GeneratorTab._rotation_from_spec(d, rng, {"mode": mode})

    @staticmethod
    def _rotation_from_spec(d: int, rng, spec: dict):
        """Regenerate rotation from a saved mode plus its mode-specific args."""
        from portal_core.benchmark_generator import generate_rotation_angles, givens_from_psi
        mode = int((spec or {}).get("mode", 1))
        ranges = {
            "theta_min": float((spec or {}).get("theta_min", -180.0)),
            "theta_max": float((spec or {}).get("theta_max", 180.0)),
        }
        kwargs = {}
        if mode == 3:
            kwargs["prob"] = float((spec or {}).get("prob", 0.5))
        if mode == 5:
            kwargs["specific_angle"] = float((spec or {}).get("angle", 45.0))
        if mode == 7:
            if (spec or {}).get("block_sizes"):
                kwargs["block_sizes"] = list((spec or {}).get("block_sizes", []))
            if (spec or {}).get("angle_mode") == "randomize":
                kwargs["randomize_angles"] = True
            elif (spec or {}).get("block_angles"):
                kwargs["block_angles"] = list((spec or {}).get("block_angles", []))
        psi = generate_rotation_angles(rng, d, ranges, mode, **kwargs)
        R = givens_from_psi(np.deg2rad(psi))
        return psi, R

    # Inputs each RotationMode actually consumes
    # (per generate_rotation_angles in benchmark_generator.py).
    _ROTATION_INPUTS = {
        1: set(),                                  # No rotation
        2: {"theta_min", "theta_max"},             # Fully connected
        3: {"theta_min", "theta_max", "prob"},     # Probabilistic
        4: {"theta_min", "theta_max"},             # Uniform angle
        5: {"angle"},                              # Fixed angle (specific_angle)
        6: {"theta_min", "theta_max"},             # Chain-like
        7: set(),                                  # Block-diagonal (uses dialog)
    }

    _ROTATION_SEPARABILITY = {
        1: "Fully separable",
        2: "Fully non-separable",
        3: "Partially separable",
        4: "Fully non-separable",
        5: "Fully non-separable",
        6: "Partially separable",
        7: "Partially separable",
    }

    def _rotation_category_modes(self):
        return _PARTIAL_ROTATION_MODES if self.rotation_partial_check.isChecked() else _FULL_ROTATION_MODES

    def _populate_rotation_modes(self, preferred_mode: Optional[int] = None) -> None:
        if not hasattr(self, "rotation_combo"):
            return
        modes = self._rotation_category_modes()
        current_mode = preferred_mode if preferred_mode is not None else self._selected_rotation_mode()
        self.rotation_combo.blockSignals(True)
        try:
            self.rotation_combo.clear()
            for mode, label in modes:
                self.rotation_combo.addItem(label, mode)
            mode_values = [mode for mode, _label in modes]
            if current_mode in mode_values:
                self.rotation_combo.setCurrentIndex(mode_values.index(current_mode))
            else:
                self.rotation_combo.setCurrentIndex(0)
        finally:
            self.rotation_combo.blockSignals(False)

    def _on_rotation_category_toggled(self, checked: bool) -> None:
        # A checkbox fires on both check and uncheck, so repopulate either way.
        self._populate_rotation_modes()
        self._sync_rotation_visibility()

    def _selected_rotation_mode(self) -> int:
        if not getattr(self, "rotation_enable", None) or not self.rotation_enable.isChecked():
            return 1
        data = self.rotation_combo.currentData()
        return int(data) if data is not None else 1

    def _set_rotation_mode_controls(self, mode: int) -> None:
        mode = int(mode)
        if mode == 1:
            self.rotation_enable.setChecked(False)
            self._sync_rotation_visibility()
            return
        self.rotation_enable.setChecked(True)
        # The partial-separability control selects block-diagonal rotation;
        # other rotation modes use the standard dropdown.
        self.rotation_partial_check.setChecked(mode == 7)
        self._populate_rotation_modes(preferred_mode=mode)
        self._sync_rotation_visibility()

    def _recompute_panel_min_width(self) -> None:
        """Resize the left panel to fit the widest currently-visible row.

        Iterates every parameter row in ``_row_factory.rows`` plus the
        rotation row, takes the max ``sizeHint().width()`` across them, and
        sets the panel minimum to that plus a fixed allowance for the
        section-card padding and outer margins.
        """
        if not hasattr(self, "_param_panel"):
            return
        max_row_w = 0
        for entry in self._row_factory.rows.values():
            # Skip deleted Qt widgets if a rebuild or another update leaves
            # a stale reference in the row factory.
            try:
                w = entry["widget"].sizeHint().width()
            except RuntimeError:
                continue
            if w > max_row_w:
                max_row_w = w
        # Rotation row is built outside the factory; include it if present.
        rot_row = getattr(self, "_rotation_row_widget", None)
        if rot_row is not None:
            try:
                max_row_w = max(max_row_w, rot_row.sizeHint().width())
            except RuntimeError:
                pass
        # +88 ≈ section-card content margins (10+10) + outer panel margins
        # (10+10) + card border padding budget.
        needed = max_row_w + 72
        target = max(_PANEL_BASE_MIN, needed)
        self._param_panel.setMinimumWidth(target)
        # Include the vertical scrollbar width in the scroll area's minimum
        # width so the viewport can display the full parameter panel.
        scroll = getattr(self, "_param_scroll", None)
        if scroll is not None:
            sb = scroll.verticalScrollBar().sizeHint().width()
            scroll.setMinimumWidth(target + sb + 2)

    def _on_parameter_row_visibility_changed(self) -> None:
        """Refresh editor geometry after mode-dependent row content changes."""
        self._recompute_panel_min_width()
        self._refresh_component_body_height()

    def _sync_neutralize_inputs(self, enabled: bool) -> None:
        """Show/hide Delta and r_ref when the Enable checkbox toggles.

        Per Part I (Eq 8 for Form A, Eq 12 for Form B), Delta and r_ref drive
        the neutralisation term; they are paper-meaningful only when
        neutralisation is enabled.
        """
        self._neutralization_controls_row_widget.setVisible(bool(enabled))
        self.delta_spin.setEnabled(bool(enabled))
        self.rref_spin.setEnabled(bool(enabled))
        self._refresh_component_body_height()

    def _sync_rotation_visibility(self) -> None:
        """Show only the controls the selected RotationMode uses."""
        enabled = self.rotation_enable.isChecked()
        partial = self.rotation_partial_check.isChecked()
        self.rotation_partial_check.setVisible(enabled)
        # Controlled Partial Separable replaces the whole dropdown row with the
        # Configure button (which lives up on the Enable row); the non-separable
        # group keeps the dropdown row.
        self._rotation_controls_row_widget.setVisible(enabled and not partial)
        self.rotation_combo.setVisible(enabled and not partial)
        mode = self._selected_rotation_mode()
        self._block_diag_config_row_widget.setVisible(enabled and partial)
        self.block_diag_config_btn.setVisible(enabled and partial)
        wanted = self._ROTATION_INPUTS.get(mode, set()) if enabled else set()
        for kind, (lab, le) in self._rot_widgets.items():
            visible = kind in wanted
            lab.setVisible(visible)
            le.setVisible(visible)
        # The rotation row width changes with the mode; let the panel
        # recompute its minimum accordingly.
        self._recompute_panel_min_width()

    def _open_block_diagonal_dialog(self) -> None:
        """Open the Configure Block-Diagonal dialog and store its spec fragment."""
        block = self.active_block()
        d = int(block.d_b) if block is not None else 2
        spec = dict(self._block_diag_spec); spec.setdefault("mode", 7)
        dlg = BlockDiagonalDialog(d, spec, self)
        if dlg.exec():
            frag = dlg.result_spec()
            self._block_diag_spec = {k: v for k, v in frag.items() if k != "mode"}

    def _rotation_spec_from_widgets(self) -> dict:
        """Read the editor's rotation controls into a regeneratable spec."""
        mode = self._selected_rotation_mode()
        spec = {"mode": mode}
        if mode in {2, 3, 4, 6}:
            spec["theta_min"] = self.theta_min.degrees_value(-180.0)
            spec["theta_max"] = self.theta_max.degrees_value(180.0)
        if mode == 3:
            spec["prob"] = float(self.rot_prob.text() or 0.5)
        if mode == 5:
            spec["angle"] = self.rot_angle.degrees_value(45.0)
        if mode == 7:
            spec.update(self._block_diag_spec)
        return spec

    def _read_rotation(self, d: int, rng_for):
        """Materialise the rotation Psi and R for the selected RotationMode.

        Only forwards kwargs the chosen mode actually consumes — matches what
        ``generate_rotation_angles`` looks at per its mode branch.
        """
        spec = self._rotation_spec_from_widgets()
        psi, R = self._rotation_from_spec(d, rng_for("rotation"), spec)
        return psi, R, spec

    # -------------------------------------------------------------- transforms
    def _rebuild_transform_params(self, *args) -> None:
        """Repopulate the per-type parameter controls for the chosen transform.

        Vector inner-parameters (``mu_plus``, ``omega_plus`` etc.) use the paper's
        6-mode transform-param menu (``TRANSFORM_PARAM_MODES``). At K=1 (the tab
        authors one component per click) modes 2-4 collapse to mode-1 outputs, but
        the mode integer still rides into the saved JSON for engine fidelity.
        """
        while self._tf_params_layout.count():
            item = self._tf_params_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        # Remove transform-specific factory entries before rebuilding the
        # controls. Persistent component rows share the same factory, whose
        # geometry calculations require valid widget references.
        for pname in list(self._tf_vec_rows):
            self._row_factory.rows.pop(pname, None)
            self._row_factory.extras.pop(pname, None)
        self._tf_vec_rows = {}
        self._tf_scalar_widgets = {}
        self._tf_pairs = {}
        self._tf_pair_coupled = {}
        spec = self._transforms[self._current_transform_name()]
        mode_labels, mode_values = self._transform_param_mode_options()
        # Vector inner-params come in ± pairs (μ⁺/μ⁻, ω⁺/ω⁻, …); render each
        # pair coupled-by-default with its own "Separate ± directions" checkbox, exactly
        # like the κ/p pairs in Shape Parameters.
        vec = spec["vec"]
        handled: set = set()
        for pname in vec:
            if pname in handled:
                continue
            base, sign = self._split_pair(pname)
            plus, minus = f"{base}_plus", f"{base}_minus"
            if sign in ("plus", "minus") and plus in vec and minus in vec:
                tf_name = self._current_transform_name()
                cb = self._make_pair_checkbox("transform")
                plus_row = self._mode_row_widget(
                    plus, self._tf_vec_rows, *vec[plus],
                    modes=mode_labels, extras_spec={5: ("alpha", "beta")},
                    mode_values=mode_values,
                    display_name=self._tf_display_name(tf_name, plus))
                self._attach_pair_checkbox(plus_row, cb)
                self._attach_row_help(
                    plus_row, self._parameter_help_button(plus, mode_labels))
                self._tf_params_layout.addWidget(plus_row)
                minus_row = self._mode_row_widget(
                    minus, self._tf_vec_rows, *vec[minus],
                    modes=mode_labels, extras_spec={5: ("alpha", "beta")},
                    mode_values=mode_values,
                    display_name=self._tf_display_name(tf_name, minus))
                # Keep one visible help icon per +/- transform pair, matching
                # the main kappa/p parameter rows.
                self._parameter_help_button(minus, mode_labels)
                self._anchor_row_tail(minus_row)
                self._tf_params_layout.addWidget(minus_row)
                self._register_pair(base, plus, minus, cb,
                                    self._tf_vec_rows, self._tf_pairs,
                                    self._tf_pair_coupled,
                                    show_minus_row=False)
                handled.update((plus, minus))
            else:
                lo, hi = vec[pname]
                row = self._mode_row_widget(pname, self._tf_vec_rows, lo, hi,
                                            modes=mode_labels,
                                            extras_spec={5: ("alpha", "beta")},
                                            mode_values=mode_values)
                self._attach_row_help(
                    row, self._parameter_help_button(pname, mode_labels))
                self._tf_params_layout.addWidget(row)
                handled.add(pname)
        hidden_scalars = self._hidden_transform_scalar_specs()
        for pname, (lo, hi) in spec["scalar"].items():
            if pname in hidden_scalars:
                continue
            cont = QWidget()
            h = QHBoxLayout(cont)
            h.setContentsMargins(0, 0, 0, 0)
            scalar_lbl = self._lbl(pname)
            scalar_tip = PARAM_TOOLTIPS.get(pname, "")
            if scalar_tip:
                scalar_lbl.setToolTip(scalar_tip)
            h.addWidget(scalar_lbl)
            lo_e = QLineEdit(str(lo)); lo_e.setFixedWidth(56)
            hi_e = QLineEdit(str(hi)); hi_e.setFixedWidth(56)
            if scalar_tip:
                lo_e.setToolTip(scalar_tip); hi_e.setToolTip(scalar_tip)
            min_lbl = QLabel("min")
            min_lbl.setStyleSheet("color: #6b7280; font-size: 11px;")
            max_lbl = QLabel("max")
            max_lbl.setStyleSheet("color: #6b7280; font-size: 11px;")
            h.addWidget(min_lbl)
            h.addWidget(lo_e)
            h.addWidget(max_lbl)
            h.addWidget(hi_e)
            h.addWidget(self._parameter_help_button(pname))
            h.addStretch(1)
            # One scalar drawn per component from [min, max] (fixed when min == max).
            self._tf_scalar_widgets[pname] = {"lo": lo_e, "hi": hi_e}
            self._tf_params_layout.addWidget(cont)

    def _capture_transform_spec(self, existing_scalar: Optional[dict] = None) -> dict:
        """Snapshot the transform editor (vec rows + scalars) into a pending spec.

        ``existing_scalar`` carries forward GUI-hidden scalars (e.g. ``sigma``,
        ``eps``) when overwriting an existing row; omit it for a brand-new row.
        """
        vec: dict = {}
        for pname, entry in self._tf_vec_rows.items():
            spec = {
                "mode": self._mode_int(entry["mode"]),
                "vmin": entry["vmin"].text(),
                "vmax": entry["vmax"].text(),
            }
            for k, w in self._extra_widgets.get(pname, {}).items():
                spec[k] = w.text()
            vec[pname] = spec
        # The − row is hidden and never independently editable, so its widgets
        # hold stale defaults. Mirror each + spec onto its − so the stored spec
        # matches what generation actually uses (a separate − is drawn from the
        # + spec). This keeps reroll/share faithful when the + range was edited.
        for pname in list(vec):
            base, sign = self._split_pair(pname)
            if sign == "plus":
                minus = f"{base}_minus"
                if minus in vec:
                    vec[minus] = dict(vec[pname])
        scal = {p: {"lo": w["lo"].text(), "hi": w["hi"].text()}
                for p, w in self._tf_scalar_widgets.items()}
        scal.update(self._hidden_transform_scalar_specs(existing_scalar))
        return {
            "name": self._current_transform_name(),
            "vec": vec,
            "scalar": scal,
            "coupled": dict(self._tf_pair_coupled),
        }

    def on_add_transform(self) -> None:
        block = self.active_block()
        if block is not None and not self._check_block_dimension(int(block.d_b)):
            return
        self._pending_transforms.append(self._capture_transform_spec())
        # Adding a new row exits any in-progress transform apply target.
        self._transform_apply_target = None
        self._refresh_transform_list()

    def on_remove_transform(self) -> None:
        """Remove the selected transform row for programmatic callers.

        The interface provides a × icon on each row for direct removal.
        """
        idx = self.transform_list.currentRow()
        if 0 <= idx < len(self._pending_transforms):
            self._delete_transform_row(idx)

    def _refresh_transform_list(self) -> None:
        """Render _pending_transforms as draggable rows with inline action icons.

        After re-populating, the list is resized to the sum of its row
        heights so it claims exactly the space it needs — no internal
        scrollbar.
        """
        from PySide6.QtWidgets import QListWidgetItem
        self._ensure_icons()
        tl = self.transform_list
        tl.blockSignals(True)
        tl.clear()
        for idx, t in enumerate(self._pending_transforms):
            item = QListWidgetItem()
            item.setData(_ROLE, t)              # reference kept for drag reorder
            widget = self._make_transform_row(idx, t)
            item.setSizeHint(widget.sizeHint())
            tl.addItem(item)
            tl.setItemWidget(item, widget)
        tl.blockSignals(False)
        # Auto-size: sum of every row's height + frame padding. When the list
        # is empty the widget collapses to zero height (no empty box).
        if tl.count() == 0:
            tl.setFixedHeight(0)
        else:
            total = sum(tl.sizeHintForRow(i) for i in range(tl.count()))
            total += 2 * tl.frameWidth() + 2
            tl.setFixedHeight(total)

    def _make_transform_row(self, idx: int, transform: dict) -> QWidget:
        """One transform row: drag-handle, number, name, and Delete icon."""
        row = QFrame()
        row.setObjectName("componentLibraryRow")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(6, 3, 4, 3)
        lay.setSpacing(6)
        handle = QLabel("☰")
        handle.setStyleSheet("color: #9aa0a6;")
        handle.setToolTip("Drag to reorder")
        lay.addWidget(handle)
        num = QLabel(str(idx + 1))
        num.setStyleSheet("color: #2563eb; font-weight: 700;")
        num.setFixedWidth(14)
        lay.addWidget(num)
        name = QLabel(self._transform_display_name(str(transform.get("name", ""))))
        name.setStyleSheet("font-weight: 600;")
        lay.addWidget(name, stretch=1)
        lay.addWidget(self._mini_button(
            self._icon_cross, "Remove this transformation",
            lambda i=idx: self._delete_transform_row(i)))
        return row

    # ---- Per-row Select / Apply / Delete --------------------------------
    def _load_transform_row_data(self, idx: int) -> None:
        """Pure data-load: replay row ``idx``'s spec into the editor fields.

        No side effect on ``_transform_apply_target``; callers decide whether
        the loaded row should become the current Apply Changes target.
        """
        if not (0 <= idx < len(self._pending_transforms)):
            return
        t = self._pending_transforms[idx]
        self._loading_transform_row = True
        self._set_current_transform_name(t.get("name", self._current_transform_name()))
        self._rebuild_transform_params()
        # Restore each ± pair's coupled state (checking the box = independent).
        for base, is_coupled in (t.get("coupled") or {}).items():
            info = self._tf_pairs.get(base)
            if info is not None:
                info["checkbox"].setChecked(not is_coupled)
        self._loading_transform_row = False
        # Vec rows: restore mode + min/max + extras text.
        for pname, spec in (t.get("vec") or {}).items():
            entry = self._tf_vec_rows.get(pname)
            if entry is None:
                continue
            self._set_mode_combo_value(entry["mode"], int(spec.get("mode", 1)))
            if "vmin" in spec:
                entry["vmin"].setText(str(spec["vmin"]))
            if "vmax" in spec:
                entry["vmax"].setText(str(spec["vmax"]))
            for k, w in self._extra_widgets.get(pname, {}).items():
                if k in spec:
                    w.setText(str(spec[k]))
        # Scalar rows: restore lo/hi.
        for pname, spec in (t.get("scalar") or {}).items():
            widget = self._tf_scalar_widgets.get(pname)
            if widget is None:
                continue
            if "lo" in spec:
                widget["lo"].setText(str(spec["lo"]))
            if "hi" in spec:
                widget["hi"].setText(str(spec["hi"]))

    def _load_transform_row(self, idx: int) -> None:
        """Compatibility wrapper: load row ``idx`` and target Apply Changes."""
        if not (0 <= idx < len(self._pending_transforms)):
            return
        self._load_transform_row_data(idx)
        self._transform_apply_target = idx
        self.apply_changes_btn.setEnabled(True)

    def _on_transform_row_clicked(self, item) -> None:
        """Row-click: load the row's data and target Apply Changes."""
        idx = self.transform_list.row(item)
        if not (0 <= idx < len(self._pending_transforms)):
            return
        self._load_transform_row_data(idx)
        self._transform_apply_target = idx
        self.apply_changes_btn.setEnabled(True)

    def _save_transform_row(self, idx: int) -> bool:
        """Capture the current editor state and overwrite row ``idx`` in place."""
        if not (0 <= idx < len(self._pending_transforms)):
            return False
        block = self.active_block()
        if block is not None and not self._check_block_dimension(int(block.d_b)):
            return False
        existing_scalar = (self._pending_transforms[idx].get("scalar") or {})
        self._pending_transforms[idx] = self._capture_transform_spec(existing_scalar)
        # Exit transform-target mode after apply.
        self._transform_apply_target = None
        self._refresh_transform_list()
        return True

    def _delete_transform_row(self, idx: int) -> None:
        if not (0 <= idx < len(self._pending_transforms)):
            return
        # Adjust the apply cursor: if selected row is deleted, release;
        # if a later selected row shifts down, keep tracking the same row.
        if self._transform_apply_target == idx:
            self._transform_apply_target = None
        elif (self._transform_apply_target is not None
              and self._transform_apply_target > idx):
            self._transform_apply_target -= 1
        self._pending_transforms.pop(idx)
        self._refresh_transform_list()

    def _on_transform_rows_moved(self, *_args) -> None:
        """After a drag-drop reorder, rebuild ``_pending_transforms`` from the
        list's new row order (each item still carries its dict in ItemDataRole)."""
        new_order: List[dict] = []
        for i in range(self.transform_list.count()):
            it = self.transform_list.item(i)
            data = it.data(_ROLE)
            if isinstance(data, dict):
                new_order.append(data)
        if len(new_order) == len(self._pending_transforms):
            self._pending_transforms = new_order
            # Indices shifted by the reorder; release the apply target and let
            # the user click the row again.
            self._transform_apply_target = None
            # Re-render so the row numbers update.
            self._refresh_transform_list()

    @staticmethod
    def _draw_scalar(spec: dict, rng) -> float:
        return generator_sampling._draw_scalar(spec, rng)

    def _realize_transforms(self, d: int, rng_for, specs: Optional[List[dict]] = None) -> List[dict]:
        source_specs = self._pending_transforms if specs is None else specs
        if d < MIN_GUI_BLOCK_DIMENSION:
            raise ValueError(GUI_BLOCK_DIMENSION_MESSAGE)
        return generator_sampling.realize_transforms(d, rng_for, source_specs)

    def _check_block_dimension(self, d: int) -> bool:
        if d < MIN_GUI_BLOCK_DIMENSION:
            QMessageBox.warning(self, "Unsupported block dimension", GUI_BLOCK_DIMENSION_MESSAGE)
            return False
        return True

    def _editor_component_vectors(self, d: int, rng_for, baseline: str):
        """Sample c/κ/p, p_single, and rotation from the current editor state.

        Returns ``(generated, rotation_spec)``. Raises ``ValueError`` on
        invalid rotation inputs; callers surface it in a dialog.
        """
        generated = self._generate_component_vectors(d, rng_for, baseline)
        if baseline == "B":
            generated["p_single"] = float(self.p_single_spin.value())
        psi, R, rotation_spec = self._read_rotation(d, rng_for)
        generated["Psi_deg"] = psi
        generated["R"] = R
        return generated, rotation_spec

    def _editor_scalars(self) -> dict:
        """Beta + neutralization scalars as currently set in the editor."""
        return {
            "beta": float(self.beta_spin.value()),
            "neutralize": {"enable": self.neutralize_enable.isChecked(),
                           "Delta": float(self.delta_spin.value()),
                           "r_ref": float(self.rref_spin.value())},
        }

    def _apply_editor_state(self, comp: "ExpComponent", *, baseline: str,
                            seed: int, generated: dict, rotation_spec: dict,
                            counters: dict, transforms: list,
                            seed_overrides: dict = None) -> None:
        """Write current editor-derived state onto ``comp`` (add or edit).

        Persists the full vector spec per parameter (mode + min/max + extras)
        and the editor-side transform specs so the component is regeneratable
        and re-selectable from its own state.
        """
        comp.baseline = baseline
        comp.seed = seed
        comp.scalars = self._editor_scalars()
        comp.rotation = rotation_spec
        comp.generated = generated
        comp.vector_specs = {
            name: self._vector_spec_from_row(name)
            for name in self.vector_rows
            if name in generated
        }
        comp.pair_coupled = dict(self._pair_coupled)
        comp.reroll_counters = counters
        comp.param_seed_overrides = dict(seed_overrides or {})
        comp.transform_specs = [dict(t) for t in self._pending_transforms]
        comp.transforms = transforms

    def on_add_component(self) -> None:
        # Per Eq 13, d is the parent block's d_b. The button is disabled when
        # there is no active block (no place to put the new φ_k).
        blk = self.active_block()
        if blk is None:
            return
        d = int(blk.d_b)
        if not self._check_block_dimension(d):
            return
        seed = int(self.seed_spin.value())
        counters: dict = {}             # fresh component: every stream at counter 0
        rng_for = self._rng_for(seed, counters)
        baseline = self.baseline_combo.currentText()
        try:
            generated, rotation_spec = self._editor_component_vectors(d, rng_for, baseline)
            transforms = self._realize_transforms(d, rng_for)
        except ValueError as exc:
            QMessageBox.critical(self, "Add component", str(exc))
            return
        comp = ExpComponent(label=self._component_label(blk),
                            baseline=baseline, seed=seed)
        self._apply_editor_state(
            comp, baseline=baseline, seed=seed, generated=generated,
            rotation_spec=rotation_spec, counters=counters, transforms=transforms)
        blk.components.append(comp)
        self._refresh_tree()
        # Leave "new" before re-selecting so the discard guard can't misfire.
        self._new_component_block = None
        self._set_editor_mode("block")   # back to the block detail after adding
        # Keep the just-added block as the active node so the user can keep
        # adding components without re-selecting.
        for i in range(self.tree.topLevelItemCount()):
            if self.tree.topLevelItem(i).data(0, _ROLE) is blk:
                self.tree.setCurrentItem(self.tree.topLevelItem(i))
                break
        self._warn_if_block_has_hidden_component_centres(blk)

    def _warn_if_block_has_hidden_component_centres(self, block: "ExpBlock") -> bool:
        """Show a plain-language warning when component centres are hidden."""
        if block is None or len(getattr(block, "components", []) or []) < 2:
            return False
        try:
            from portal_core.validate_no_cover import validate_no_cover_with_sources
            from portal_studio.experiment.assemble import block_to_instance

            ok, offenders = validate_no_cover_with_sources(block_to_instance(block))
        except Exception:  # noqa: BLE001 - advisory check must not block editing
            logger.debug(
                "Unable to evaluate the advisory hidden-center warning.",
                exc_info=True,
            )
            return False
        if ok:
            return False

        lines = []
        seen = set()
        components = list(block.components)
        for comp_idx, covering_indices, _observed, _expected, _gap in offenders:
            idx = int(comp_idx)
            label = (
                components[idx].label
                if 0 <= idx < len(components)
                else f"component{idx + 1}"
            )
            if label not in seen:
                covering_labels = [
                    components[int(cover_idx)].label
                    if 0 <= int(cover_idx) < len(components)
                    else f"component{int(cover_idx) + 1}"
                    for cover_idx in covering_indices
                ]
                covered_html = (
                    f"<span style='color:#2563eb; font-weight:600;'>{escape(label)}</span>"
                )
                if covering_labels:
                    coverers_html = ", ".join(
                        f"<span style='color:#7a2e0e; font-weight:600;'>{escape(cover_label)}</span>"
                        for cover_label in covering_labels
                    )
                    lines.append(f"{covered_html} <span style='color:#475467;'>(covered by {coverers_html})</span>")
                else:
                    lines.append(covered_html)
                seen.add(label)

        hidden_html = "<br>".join(lines) if lines else (
            "<span style='color:#2563eb; font-weight:600;'>one or more components</span>"
        )
        QMessageBox.warning(
            self,
            "Check component centres",
            "<p><span style='color:#b42318; font-weight:600;'>Some component "
            "centres are not visible in this block.</span></p>"
            f"<p><b>Covered components:</b> {hidden_html}</p>"
            "<p><span style='color:#475467;'>This can happen when another "
            "component is lower at the same location. To make the centres "
            "visible, edit the centre, height, shape, rotation, or "
            "transformation values in the component editor, or open the "
            "component data table from Instance Layers.</span></p>",
        )
        return True

    # --------------------------------------------------- block-first hierarchy
    def _build_right_panel(self) -> QWidget:
        """Top-down composite tree: composite → blocks → components.

        Per Eq 13 and Eq 14, blocks own components and the composite owns
        blocks. The user authors top-down: declare a block (label, d_b,
        weight), then add components inside it. No flat pool, no checkbox
        filter when composing.
        """
        panel = QWidget()
        self.project_panel = panel
        panel.setObjectName("projectPanel")
        panel.setStyleSheet(
            _FIELD_QSS
            + " QWidget#projectPanel { background: #ffffff;"
            " border: 1px solid #e4e7ec; border-radius: 8px; }"
        )
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        layers_card = self._build_section_card(None)
        self.instance_layers_card = layers_card
        layers_card.setStyleSheet(
            "QFrame#sectionCard { background: transparent; border: none; border-radius: 0; }")
        layers_card.setGraphicsEffect(None)
        layers_layout = layers_card.body_layout
        layers_header = QHBoxLayout()
        layers_title = QLabel("Instance Layers")
        layers_title.setObjectName("panelTitle")   # match other section headers
        layers_header.addWidget(layers_title)
        self.instance_layers_help_btn = make_parameter_help_button(
            "Explain Instance Layers", self._show_instance_layers_help)
        layers_header.addWidget(self.instance_layers_help_btn)
        layers_header.addStretch(1)
        layers_layout.addLayout(layers_header)
        v.addWidget(layers_card, stretch=1)

        # Block-first tree: column 0 contains labels; column 1 contains
        # actions for adding, inspecting, cloning, or deleting nodes.
        from PySide6.QtWidgets import QAbstractItemView, QHeaderView
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setFrameShape(QFrame.Shape.NoFrame)
        self.tree.setStyleSheet(
            "QTreeWidget { background: transparent; border: none; padding: 0; }")
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layers_layout.addWidget(self.tree, stretch=1)

        # Empty-state hint: shown when the tree has no blocks yet. Hidden
        # automatically by _refresh_tree() once the inventory has any block.
        self.tree_empty_hint = QLabel(
            "<i>No blocks yet. Create one with <b>+ Add Block</b> on the "
            "top of this pane to start authoring components.</i>")
        self.tree_empty_hint.setWordWrap(True)
        self.tree_empty_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.tree_empty_hint.setStyleSheet("color: #9aa0a6; padding: 24px;")
        layers_layout.addWidget(self.tree_empty_hint, stretch=1)

        # Instance-level total dimension readout.
        self.total_d_label = QLabel("Instance overall dimension = 0")
        self.total_d_label.setStyleSheet(_SECTION_HEADER_QSS)
        layers_layout.addWidget(self.total_d_label)

        # New-block inputs sit below the tree, above the instance actions.
        layers_layout.addWidget(self._build_add_block_card())

        # Instance actions: an even pair sits side by side.
        self.build_btn = QPushButton("Prepare Instance for ELA")
        self.build_btn.setProperty("variant", "primary")
        self.save_btn = QPushButton("Save Instance JSON")
        self.instance_actions_widget = QWidget()
        self.instance_actions_widget.setObjectName("instanceActionsWidget")
        self.instance_actions_widget.setStyleSheet(
            "QWidget#instanceActionsWidget { background: transparent; }")
        instance_actions = QHBoxLayout(self.instance_actions_widget)
        instance_actions.setContentsMargins(0, 0, 0, 0)
        instance_actions.setSpacing(8)
        instance_actions.addWidget(self.build_btn)
        instance_actions.addWidget(self.save_btn)
        self.instance_actions_layout = instance_actions
        layers_layout.addWidget(self.instance_actions_widget)
        return panel

    def on_add_block_clicked(self) -> None:
        """Create a new block with the strip's (d_b, weight) values.

        Label auto-generates as ``block_<N>`` — paper-level bookkeeping only.
        """
        label = f"block_{len(self.inventory.blocks) + 1}"
        d_b = int(self.dim_input.value())
        if not self._check_block_dimension(d_b):
            return
        weight = float(self.weight_input.value())
        block = ExpBlock(label=label, d_b=d_b, weight=weight, components=[])
        self.inventory.blocks.append(block)
        self._refresh_tree()
        # Auto-select the new block so the user can immediately add components.
        node = self.tree.topLevelItem(self.tree.topLevelItemCount() - 1)
        self.tree.setCurrentItem(node)

    def _refresh_tree(self) -> None:
        """Rebuild the block-first tree from the inventory."""
        self.tree.blockSignals(True)
        self.tree.clear()
        for blk in self.inventory.blocks:
            label = (
                f"{blk.label}  (dimension={blk.d_b}, "
                f"weight={self._format_tree_number(blk.weight)}, "
                f"components={len(blk.components)})"
            )
            node = QTreeWidgetItem([label, ""])
            node.setData(0, _ROLE, blk)
            for comp in blk.components:
                child = QTreeWidgetItem([f"{comp.label}  (baseline={comp.baseline})", ""])
                child.setData(0, _ROLE, comp)
                node.addChild(child)
            self.tree.addTopLevelItem(node)
            node.setExpanded(True)
            # Per-row icon action widgets, set after the item is in the tree
            # (setItemWidget requires the item to have a parent first).
            self.tree.setItemWidget(node, 1, self._make_node_actions(block=blk))
            for i, comp in enumerate(blk.components):
                child = node.child(i)
                self.tree.setItemWidget(
                    child, 1,
                    self._make_node_actions(block=blk, component=comp))
        self.tree.blockSignals(False)
        self.total_d_label.setText(
            f"Instance overall dimension = {self.inventory.total_dimension()}")
        # Toggle the empty-state hint based on whether there are any blocks.
        if hasattr(self, "tree_empty_hint"):
            has_blocks = len(self.inventory.blocks) > 0
            self.tree.setVisible(has_blocks)
            self.tree_empty_hint.setVisible(not has_blocks)

    @staticmethod
    def _format_tree_number(value: float) -> str:
        return generator_display.format_tree_number(value)

    def _ensure_icons(self) -> None:
        """Lazy-init the tree row icons (one-shot, cached on the tab)."""
        if not hasattr(self, "_icon_cross"):
            from portal_studio.gui.common.icons import (
                make_table_icon, make_trash_icon, make_clone_icon)
            self._icon_table = make_table_icon()
            self._icon_cross = make_trash_icon()
            self._icon_clone = make_clone_icon()

    def _mini_button(self, icon: QIcon, tip: str, handler) -> QPushButton:
        btn = QPushButton()
        btn.setIcon(icon)
        btn.setFixedSize(24, 24)
        btn.setToolTip(tip)
        btn.setAccessibleName(tip)
        btn.setAccessibleDescription(tip)
        btn.setObjectName("nodeActionBtn")
        btn.clicked.connect(lambda _checked=False: handler())
        return btn

    def _make_node_actions(self, *, block: "ExpBlock",
                           component: Optional["ExpComponent"] = None) -> QWidget:
        """Build a row-of-icons action widget for one tree row.

        Component rows provide parameter-table, clone, and delete actions; block
        rows provide add-component and delete actions. Select a row to edit its
        parameters, then save with the middle panel's Apply Changes button.
        """
        self._ensure_icons()
        holder = QWidget()
        holder.setObjectName("nodeActions")
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 6, 0)
        row.setSpacing(1)
        if component is not None:
            row.addWidget(self._mini_button(
                self._icon_table,
                "Open the generated-parameter table: the actual sampled "
                "values for every dimension of this component.",
                lambda b=block, c=component: self._open_component_table(b, c)))
            row.addWidget(self._mini_button(
                self._icon_clone,
                "Clone this component into a block of your choice, as an "
                "independent copy.",
                lambda b=block, c=component: self._prompt_clone_component(b, c)))
        else:
            plus = QPushButton("+")
            plus.setFixedSize(24, 24)
            plus.setObjectName("nodeActionBtn")
            plus.setToolTip("Add a new component to this instance")
            plus.setAccessibleName(plus.toolTip())
            plus.setAccessibleDescription(plus.toolTip())
            plus.clicked.connect(
                lambda _checked=False, b=block: self._on_new_component_for_block(b))
            row.addWidget(plus)
        delete_tip = (
            "Delete this component from its block."
            if component is not None
            else "Delete this block and all of its components.")
        row.addWidget(self._mini_button(
            self._icon_cross, delete_tip,
            (lambda b=block, c=component: self.delete_component(b, c))
            if component is not None
            else (lambda b=block: self.delete_block(b))))
        return holder

    @staticmethod
    def _format_component_value(value) -> str:
        return generator_display.format_component_value(value)

    @staticmethod
    def _plain_math_header(text: str) -> str:
        return generator_display.plain_math_header(text)

    @staticmethod
    def _fit_table_to_rows(table: QTableWidget) -> None:
        table.resizeRowsToContents()
        table.resizeColumnsToContents()
        height = table.horizontalHeader().height()
        height += sum(table.rowHeight(i) for i in range(table.rowCount()))
        height += 2 * table.frameWidth() + 2
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setMinimumHeight(height)
        table.setMaximumHeight(height)
        width = table.verticalHeader().width()
        width += sum(table.columnWidth(i) for i in range(table.columnCount()))
        width += 2 * table.frameWidth() + 2
        table.setMinimumWidth(width)

    @staticmethod
    def _centered_table_item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item

    @classmethod
    def _rotation_triangle_table_data(cls, generated: dict, d: int):
        return generator_display.rotation_triangle_table_data(generated, d)

    @classmethod
    def _rotation_pair_table_data(cls, generated: dict, d: int):
        return generator_display.rotation_pair_table_data(generated, d)

    @staticmethod
    def _block_diagonal_rotation_heading(grouping: str, angle_mode: str) -> str:
        return generator_display.block_diagonal_rotation_heading(grouping, angle_mode)

    @staticmethod
    def _rotation_mode_heading(mode: int) -> str:
        return generator_display.rotation_mode_heading(mode)

    def _add_block_diagonal_section(self, layout, component, rot: dict, d: int) -> None:
        """Show the Block-Diagonal grouping (block size + angle) in the data table
        when the component uses partial-separation (mode 7) rotation."""
        grouping, angle_mode, _rows, remaining = block_diagonal_display(rot, d)
        heading = QLabel(self._block_diagonal_rotation_heading(grouping, angle_mode))
        heading.setStyleSheet(
            "font-weight: 700; color: #1f2a44; margin-top: 8px;")
        layout.addWidget(heading)

        # Only the upper triangle carries pair angles; diagonal/lower cells are redundant.
        headers, triangle_rows = self._rotation_triangle_table_data(
            getattr(component, "generated", None) or {}, d)
        ncols = len(headers)
        tbl = QTableWidget(len(triangle_rows), ncols)
        tbl.setObjectName("componentRotationTriangleTable")
        tbl.setHorizontalHeaderLabels(headers)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl.setAlternatingRowColors(False)
        tbl.verticalHeader().setVisible(False)
        tint = QColor("#efeaff")     # lavender, distinct from the c/κ/p groups
        for r, row_values in enumerate(triangle_rows):
            for c, value in enumerate(row_values):
                tbl.setItem(r, c, self._centered_table_item(value))
            self._tint_row(tbl, r, tint, ncols)
        self._fit_table_to_rows(tbl)
        layout.addWidget(tbl)

        if remaining > 0:
            note = QLabel(
                f"remaining: {remaining} dim(s), not mixed by this rotation")
            note.setStyleSheet("color: #667085; font-size: 11px;")
            layout.addWidget(note)

    def _add_rotation_pairs_section(self, layout, component, d: int) -> None:
        headers, rows = self._rotation_pair_table_data(
            getattr(component, "generated", None) or {}, d)
        if not rows:
            return
        rot = getattr(component, "rotation", None) or {}
        if int(rot.get("mode", 1)) == 7:
            grouping, angle_mode, _rows, _remaining = block_diagonal_display(rot, d)
            heading_text = self._block_diagonal_rotation_heading(grouping, angle_mode)
        else:
            heading_text = self._rotation_mode_heading(int(rot.get("mode", 1)))
        heading = QLabel(heading_text)
        heading.setStyleSheet(
            "font-weight: 700; color: #1f2a44; margin-top: 6px;")
        layout.addWidget(heading)

        tbl = QTableWidget(len(rows), len(headers))
        tbl.setObjectName("componentRotationPairTable")
        tbl.setHorizontalHeaderLabels(headers)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl.setAlternatingRowColors(False)
        tbl.horizontalHeader().setVisible(False)
        tbl.verticalHeader().setVisible(False)
        tint = QColor("#efeaff")
        for row_idx, row_values in enumerate(rows):
            for col_idx, value in enumerate(row_values):
                item = self._centered_table_item(value)
                if row_values and row_values[0] == "pair":
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                tbl.setItem(row_idx, col_idx, item)
            self._tint_row(tbl, row_idx, tint, len(headers))
        self._fit_table_to_rows(tbl)
        layout.addWidget(tbl)

    @classmethod
    def _vector_value_at(cls, generated: dict, name: str, index: int) -> str:
        if name not in generated:
            return ""
        return cls._value_at(generated[name], index)

    @classmethod
    def _value_at(cls, value, index: int) -> str:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return cls._format_component_value(arr.item())
        if index >= arr.shape[0]:
            return ""
        return cls._format_component_value(arr.reshape(-1)[index])

    def _build_component_parameters_dialog(
            self, block: "ExpBlock", component: "ExpComponent") -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{component.label} parameters")
        dialog.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        dialog.resize(760, 460)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setObjectName("componentParametersScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        scroll_layout = QVBoxLayout(content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)
        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

        dialog._pp_block = block
        dialog._pp_component = component
        dialog._pp_scroll_layout = scroll_layout
        dialog._data_changed = False     # set by reroll / share / copy actions
        self._populate_parameters_dialog_body(dialog, scroll_layout, block, component)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)

        def _on_ok():
            if getattr(dialog, "_data_changed", False):
                warned = self._warn_if_block_has_hidden_component_centres(block)
                if not warned:
                    QMessageBox.information(
                        self, "Saved", "Updated data has been saved.")
            dialog.accept()
        buttons.accepted.connect(_on_ok)
        layout.addWidget(buttons)
        return dialog

    def _populate_parameters_dialog_body(self, dialog, scroll_layout, block, component):
        """Fill the parameter dialog's scroll body. Re-invokable to rebuild it
        after a share changes a pair's direction (which changes the layout)."""
        title_row = QHBoxLayout()
        title = QLabel(f"{component.label} in {block.label}")
        title.setStyleSheet("font-weight: 700; font-size: 15px; color: #0f172a;")
        title_row.addWidget(title)
        title_row.addWidget(make_parameter_help_button(
            "Explain the generated-parameter table",
            self._show_component_table_help))
        title_row.addStretch(1)
        scroll_layout.addLayout(title_row)

        def mark_changed(*_a):
            dialog._data_changed = True

        scalars = component.scalars or {}
        neutralize = scalars.get("neutralize", {}) or {}
        summary_items = [
            ("base form", component.baseline),
            ("component seed", component.seed),
            ("beta", scalars.get("beta", 0.0)),
            ("neutralization", "on" if neutralize.get("enable") else "off"),
        ]
        # Delta and r_ref are paper-meaningful only when neutralization is on
        # (Eq 8 / 12); mirror the editor, which hides them when it's off.
        if neutralize.get("enable"):
            summary_items += [
                ("Delta", neutralize.get("Delta", "")),
                ("r_ref", neutralize.get("r_ref", "")),
            ]
        summary_text = "; ".join(
            f"{math_text(name)} = {self._format_component_value(value)}"
            for name, value in summary_items
        )
        summary = QLabel(summary_text)
        summary.setObjectName("componentParameterSummary")
        summary.setTextFormat(Qt.TextFormat.RichText)
        summary.setWordWrap(True)
        summary.setStyleSheet("color: #0f172a;")
        scroll_layout.addWidget(summary)
        d = int(block.d_b)
        action_toggles, register_action_table = self._make_action_column_toggles()
        scroll_layout.addWidget(action_toggles)
        value_col = 4

        generated = component.generated or {}
        # Row order: centre, then the exponent p, then the curvature kappa
        # (beta is a scalar shown in the summary).
        if component.baseline == "B":
            parameter_rows = [("c", "c"), ("p", "p_single"),
                              ("\u03ba+", "kappa_plus"), ("\u03ba-", "kappa_minus")]
        else:
            parameter_rows = [("c", "c"), ("p+", "p_plus"), ("p-", "p_minus"),
                              ("\u03ba+", "kappa_plus"), ("\u03ba-", "kappa_minus")]
        dialog._reroll_buttons = {}
        row_of = {name: row_idx for row_idx, (_lbl, name) in enumerate(parameter_rows)}
        # Each parameter group gets its own colour: the centre, and each ± pair.
        tint_of = {
            "c": QColor("#fff3e6"),                                          # centre — peach
            "kappa_plus": QColor("#eaf2ff"), "kappa_minus": QColor("#eaf2ff"),  # κ — blue
            "p_plus": QColor("#eafaf0"), "p_minus": QColor("#eafaf0"),          # p — green
            "p_single": QColor("#eafaf0"),
        }
        table = QTableWidget(len(parameter_rows), d + value_col)
        table.setObjectName("componentParameterTable")
        table.setHorizontalHeaderLabels(
            ["↻", "seed", "copy", "parameter"] + [str(i + 1) for i in range(d)])
        # Define what the reroll and copy columns do, on their headers.
        table.horizontalHeaderItem(0).setToolTip(
            "Reroll: redraw a parameter's random values from its current mode "
            "and range, leaving every other parameter unchanged. Reproducible "
            "from the seed.")
        table.horizontalHeaderItem(1).setToolTip(
            "Seeded reroll: enter a manual seed for this parameter row and "
            "redraw it from that seed.")
        table.horizontalHeaderItem(2).setToolTip(
            "Copy: set this parameter to another component's value (same "
            "dimension and baseline). Reproducible from its seed.")
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(False)
        table.verticalHeader().setVisible(False)   # row numbers add no info here
        coupled = getattr(component, "pair_coupled", None) or {}
        for row_idx, (label, name) in enumerate(parameter_rows):
            tint = tint_of.get(name)
            table.setItem(row_idx, 3, self._centered_table_item(label))
            if self._vector_key_rerollable(component, name):
                btn = self._make_vector_reroll_button(
                    component, name, label, table, row_of, d, value_col=value_col,
                    tint=tint, on_changed=mark_changed)
                seed_btn = self._make_vector_seed_button(
                    component, name, label, table, row_of, d, value_col=value_col,
                    tint=tint, on_changed=mark_changed)
                dialog._reroll_buttons[name] = btn
                base, sign = self._split_pair(name)
                # Coupled pair → one centred reroll control spanning the +/- rows.
                if (base in {"kappa", "p"} and sign == "plus"
                        and coupled.get(base, True) and f"{base}_minus" in row_of):
                    table.setSpan(row_idx, 0, 2, 1)
                    table.setSpan(row_idx, 1, 2, 1)
                table.setCellWidget(row_idx, 0, self._centered_icon_cell(btn, bg=tint))
                table.setCellWidget(row_idx, 1, self._centered_icon_cell(seed_btn, bg=tint))
                # Copy control: one pair-level button per ± pair (always spans
                # the two rows); single params get a value-only copy button.
                is_pair = base in {"kappa", "p"}
                if is_pair and sign == "minus":
                    pass   # shared from the + row; no share button on −
                else:
                    if is_pair and sign == "plus" and f"{base}_minus" in row_of:
                        table.setSpan(row_idx, 2, 2, 1)
                        share_btn = self._make_share_pair_button(
                            component, base, table, row_of, d, value_col=value_col,
                            tint=tint, dialog=dialog)
                    else:
                        share_btn = self._make_copy_button(
                            component, name, table, row_of, d,
                            value_col=value_col, tint=tint)
                    share_btn.clicked.connect(mark_changed)
                    table.setCellWidget(
                        row_idx, 2, self._centered_icon_cell(share_btn, bg=tint))
            for dim_idx in range(d):
                table.setItem(
                    row_idx, dim_idx + value_col,
                    self._centered_table_item(
                        self._vector_value_at(generated, name, dim_idx)))
            if tint is not None:
                self._tint_row(table, row_idx, tint, d + value_col)
        self._fit_table_to_rows(table)
        register_action_table(table)
        scroll_layout.addWidget(table)

        for order, transform in enumerate(component.transforms or [], start=1):
            t_name = str(transform.get("name", "transform"))
            params = list((transform.get("params", {}) or {}).items())
            if not params:
                continue
            transform_label = QLabel(
                f"{order}. {self._transform_display_name(t_name)}")
            transform_label.setStyleSheet(
                "font-weight: 700; color: #0f172a; margin-top: 8px;")
            scroll_layout.addWidget(transform_label)
            transform_table = QTableWidget(len(params), d + value_col)
            transform_table.setObjectName("componentTransformTable")
            transform_table.setHorizontalHeaderLabels(
                ["↻", "seed", "copy", "parameter"] + [str(i + 1) for i in range(d)])
            transform_table.horizontalHeaderItem(0).setToolTip(
                "Reroll: redraw a parameter's random values from its current "
                "mode and range, leaving every other parameter unchanged. "
                "Reproducible from the seed.")
            transform_table.horizontalHeaderItem(1).setToolTip(
                "Seeded reroll: enter a manual seed for this transform row "
                "and redraw it from that seed.")
            transform_table.horizontalHeaderItem(2).setToolTip(
                "Copy: set this parameter to another component's value with the "
                "same transform (same dimension and baseline). Reproducible from its seed.")
            transform_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            transform_table.setAlternatingRowColors(False)
            transform_table.verticalHeader().setVisible(False)
            t_index = order - 1
            trow_of = {name: row_idx for row_idx, (name, _v) in enumerate(params)}
            t_coupled = (component.transform_specs[t_index].get("coupled", {})
                         if t_index < len(getattr(component, "transform_specs", []) or [])
                         else {}) or {}
            # Each ± pair (mu, gamma, omega, …) gets its own group colour.
            tf_palette = ["#eaf2ff", "#eafaf0", "#fdf0f6", "#fff7e8", "#f0f0fb"]
            base_color: dict = {}
            for nm, _v in params:
                b, _s = self._split_pair(nm)
                if b not in base_color:
                    base_color[b] = QColor(tf_palette[len(base_color) % len(tf_palette)])
            for row_idx, (name, value) in enumerate(params):
                base, sign = self._split_pair(name)
                label = self._tf_table_label(t_name, name, base, sign)
                tint = base_color[base]
                transform_table.setItem(row_idx, 3, self._centered_table_item(label))
                if self._transform_param_rerollable(component, t_index, name):
                    btn = self._make_transform_reroll_button(
                        component, t_index, name, label, transform_table, trow_of, d,
                        value_col=value_col, tint=tint, on_changed=mark_changed)
                    seed_btn = self._make_transform_seed_button(
                        component, t_index, name, label, transform_table, trow_of, d,
                        value_col=value_col, tint=tint, on_changed=mark_changed)
                    dialog._reroll_buttons[f"tf{t_index}:{name}"] = btn
                    if (sign == "plus" and base in t_coupled
                            and t_coupled.get(base, True) and f"{base}_minus" in trow_of):
                        transform_table.setSpan(row_idx, 0, 2, 1)
                        transform_table.setSpan(row_idx, 1, 2, 1)
                    transform_table.setCellWidget(
                        row_idx, 0, self._centered_icon_cell(btn, bg=tint))
                    transform_table.setCellWidget(
                        row_idx, 1, self._centered_icon_cell(seed_btn, bg=tint))
                    if sign == "minus":
                        pass   # shared from the + row
                    else:
                        if sign == "plus" and f"{base}_minus" in trow_of:
                            transform_table.setSpan(row_idx, 2, 2, 1)
                            share_btn = self._make_share_pair_button(
                                component, base, transform_table, trow_of, d,
                                value_col=value_col, tint=tint, dialog=dialog,
                                transform_index=t_index)
                        else:
                            share_btn = self._make_copy_button(
                                component, f"tf{t_index}:{name}", transform_table,
                                trow_of, d, value_col=value_col, tint=tint, row_key=name)
                        share_btn.clicked.connect(mark_changed)
                        transform_table.setCellWidget(
                            row_idx, 2, self._centered_icon_cell(share_btn, bg=tint))
                for dim_idx in range(d):
                    transform_table.setItem(
                        row_idx, dim_idx + value_col,
                        self._centered_table_item(self._value_at(value, dim_idx)))
                self._tint_row(transform_table, row_idx, tint, d + value_col)
            self._fit_table_to_rows(transform_table)
            register_action_table(transform_table)
            scroll_layout.addWidget(transform_table)

        self._add_rotation_pairs_section(scroll_layout, component, d)
        scroll_layout.addStretch(1)

    def _rebuild_parameters_body(self, dialog):
        """Clear and repopulate the dialog body (after a direction change)."""
        sl = dialog._pp_scroll_layout
        while sl.count():
            item = sl.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._populate_parameters_dialog_body(
            dialog, sl, dialog._pp_block, dialog._pp_component)

    def _open_component_table(
            self, block: "ExpBlock", component: "ExpComponent") -> None:
        """Data-table icon: select the component's tree row (so it highlights
        like a click), then open its generated-parameter table."""
        for i in range(self.tree.topLevelItemCount()):
            node = self.tree.topLevelItem(i)
            if node.data(0, _ROLE) is not block:
                continue
            for j in range(node.childCount()):
                child = node.child(j)
                if child.data(0, _ROLE) is component:
                    self.tree.setCurrentItem(child)
                    break
            break
        self._show_component_parameters_dialog(block, component)

    def _show_component_parameters_dialog(
            self, block: "ExpBlock", component: "ExpComponent") -> None:
        self._build_component_parameters_dialog(block, component).exec()

    def _build_instance_layers_help_dialog(self) -> QDialog:
        """Help dialog for the Instance Layers panel, showing each row icon
        beside what it does so the icons are self-explanatory."""
        self._ensure_icons()
        dialog = QDialog(self)
        dialog.setWindowTitle("Instance Layers help")
        dialog.setObjectName("instanceLayersHelpDialog")
        dialog.setModal(True)
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        heading = QLabel("<b>Instance Layers</b>")
        heading.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(heading)
        intro = QLabel(
            "Lists every block in the composite and the components inside each "
            "one. Select a row to load it into the editor, or use the row "
            "buttons:")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # The "+" is a text button rather than an icon, so describe it first
        # with a matching "+" glyph in the icon column.
        plus_row = QHBoxLayout()
        plus_row.setSpacing(8)
        plus_lbl = QLabel("+")
        plus_lbl.setFixedWidth(22)
        plus_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        plus_lbl.setStyleSheet("font-weight: bold; font-size: 15px;")
        plus_row.addWidget(plus_lbl)
        plus_text = QLabel(
            "<b>Add component</b>: on an instance (block) row, opens a blank "
            "component editor for that instance. Adjust its parameters, then "
            "press <b>+ Add Component</b> to add it.")
        plus_text.setWordWrap(True)
        plus_text.setTextFormat(Qt.TextFormat.RichText)
        plus_row.addWidget(plus_text, stretch=1)
        layout.addLayout(plus_row)

        rows = [
            (self._icon_table, "View parameters",
             "opens the generated-parameter table for a component, showing its "
             "actual sampled values for every dimension."),
            (self._icon_clone, "Clone",
             "copies a component into a block of your choice, as an "
             "independent duplicate."),
            (self._icon_cross, "Delete",
             "removes a component from its block, or an entire block and all "
             "of its components."),
        ]
        for icon, name, desc in rows:
            row = QHBoxLayout()
            row.setSpacing(8)
            icon_lbl = QLabel()
            icon_lbl.setPixmap(icon.pixmap(18, 18))
            icon_lbl.setFixedWidth(22)
            icon_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
            row.addWidget(icon_lbl)
            text = QLabel(f"<b>{name}</b>: {desc}")
            text.setWordWrap(True)
            text.setTextFormat(Qt.TextFormat.RichText)
            row.addWidget(text, stretch=1)
            layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        return dialog

    def _show_instance_layers_help(self) -> None:
        self._build_instance_layers_help_dialog().exec()

    def _show_component_table_help(self) -> None:
        self._show_help_dialog(
            "Generated parameters help",
            self._help_body("Generated parameters", component_table_help_paragraphs()),
            None,
        )

    def delete_block(self, block: "ExpBlock") -> None:
        """Remove a block (and its components) from the inventory."""
        if block in self.inventory.blocks:
            if (self._component_apply_target is not None
                    and self._component_apply_target[0] is block):
                self._component_apply_target = None
            if self._block_apply_target is block:
                self._block_apply_target = None
            self.inventory.blocks.remove(block)
            self._reindex_component_labels()
            self._refresh_tree()
            self._on_tree_selection_changed()
            self._reset_seed_when_empty()

    def delete_component(self, block: "ExpBlock",
                         component: "ExpComponent") -> None:
        """Remove one component from a block."""
        if block in self.inventory.blocks and component in block.components:
            block.components.remove(component)
            self._reindex_component_labels()
            # If the editor was tracking this component, stop tracking it.
            if self._component_apply_target == (block, component):
                self._component_apply_target = None
                self.active_block_hint.setText(self._default_hint_html())
            self._refresh_tree()
            self._on_tree_selection_changed()
            self._reset_seed_when_empty()

    def _reset_seed_when_empty(self) -> None:
        """Return the seed box to its default (42) once the Instance Layers are
        empty, so the next component starts from a clean seed."""
        if not self.inventory.blocks:
            self.seed_spin.setValue(42)
            self._editor_seed_default = 42

    def _clone_component(self, src_block, component, target_block) -> None:
        """Deep-copy ``component`` into ``target_block`` (regenerating if d differs)."""
        if not self._check_block_dimension(int(target_block.d_b)):
            return
        import copy
        twin = copy.deepcopy(component)
        twin.label = self._component_label(target_block)
        target_block.components.append(twin)
        if int(target_block.d_b) != int(src_block.d_b):
            # Same seed → deterministic, correctly-sized regeneration for the new d.
            self._regenerate_component_for_dimension(twin, int(target_block.d_b))
        self._refresh_tree()

    def _prompt_clone_component(self, src_block, component) -> None:
        """Ask which block to clone into, then clone."""
        from PySide6.QtWidgets import QInputDialog
        blocks = list(self.inventory.blocks)
        if not blocks:
            return
        labels = [f"{b.label} (d={b.d_b})" for b in blocks]
        default = blocks.index(src_block) if src_block in blocks else 0
        choice, ok = QInputDialog.getItem(
            self, "Clone component",
            "Add the copy to which block?", labels, default, editable=False)
        if not ok:
            return
        self._clone_component(src_block, component, blocks[labels.index(choice)])

    # ----- Per-component loading / Apply Changes ---------------------------

    @staticmethod
    def _default_hint_html() -> str:
        return ("<i>Create a block with <b>+ Add Block</b> above, "
                "or click an existing block in the tree, before adding components.</i>")

    def _load_component_spec_into_editor(self, block: "ExpBlock",
                                         component: "ExpComponent") -> None:
        """Replay the component's saved spec into the editor controls.

        Pure data-load; the selection handler decides which component, if any,
        is the current Apply Changes target.
        """
        self.baseline_combo.setCurrentText(component.baseline)
        scalars = component.scalars or {}
        self.beta_spin.setValue(float(scalars.get("beta", 0.0)))
        neut = scalars.get("neutralize", {})
        self.neutralize_enable.setChecked(bool(neut.get("enable", False)))
        if "Delta" in neut:
            self.delta_spin.setValue(float(neut["Delta"]))
        if "r_ref" in neut:
            self.rref_spin.setValue(float(neut["r_ref"]))
        self.seed_spin.setValue(int(component.seed))
        # The loaded component's seed is the session default a later Reset returns to.
        self._editor_seed_default = int(component.seed)

        # Replay every saved vector spec back into its mode dropdown + extras.
        # Map the saved mode number to the dropdown index
        # (mode 1 = index 0, mode 2 = index 1, ...).
        for name, spec in (component.vector_specs or {}).items():
            entry = self.vector_rows.get(name)
            if entry is None:
                continue
            mode_idx = max(0, int(spec.get("mode", 1)) - 1)
            entry["mode"].setCurrentIndex(min(mode_idx, entry["mode"].count() - 1))
            if "vmin" in spec:
                entry["vmin"].setText(str(spec["vmin"]))
            if "vmax" in spec:
                entry["vmax"].setText(str(spec["vmax"]))
            for k, w in self._extra_widgets.get(name, {}).items():
                if k in spec:
                    w.setText(str(spec[k]))

        # Restore each κ/p pair's coupled state; an absent setting means coupled.
        pc = getattr(component, "pair_coupled", None) or {}
        for base, info in self._pairs.items():
            info["checkbox"].setChecked(not pc.get(base, True))

        # Rotation: re-apply mode integer.
        rot = component.rotation or {}
        rot_mode = int(rot.get("mode", 1))
        self._set_rotation_mode_controls(rot_mode)
        if "theta_min" in rot:
            self.theta_min.set_degrees(rot["theta_min"])
        if "theta_max" in rot:
            self.theta_max.set_degrees(rot["theta_max"])
        if "prob" in rot:
            self.rot_prob.setText(str(rot["prob"]))
        if "angle" in rot:
            self.rot_angle.set_degrees(rot["angle"])
        if rot.get("mode") == 7:
            self._block_diag_spec = {k: v for k, v in rot.items() if k != "mode"}

        # Restore saved transform specifications into _pending_transforms
        # and re-render the list. An absent or empty specification clears it.
        specs = getattr(component, "transform_specs", None) or []
        self._pending_transforms = [dict(t) for t in specs]
        # Selecting a component starts with no transform row targeted.
        self._transform_apply_target = None
        self._refresh_transform_list()
        # Refresh the transform parameter editor for the selected component.
        # Load the first transform's spec if present; otherwise rebuild to the
        # coupled defaults so stale checkboxes cannot linger.
        if self._pending_transforms:
            self._load_transform_row_data(0)
        else:
            self._rebuild_transform_params()

    def _load_block_into_editor(self, block: "ExpBlock") -> None:
        """Replay block-level values into the middle panel."""
        self.block_dim_input.setValue(int(block.d_b))
        self.block_weight_input.setValue(float(block.weight))

    def _on_apply_changes(self) -> None:
        """Apply editor values to the selected block or component."""
        if self._transform_apply_target is not None:
            if not self._save_transform_row(self._transform_apply_target):
                return
            if self._component_apply_target is not None:
                block, component = self._component_apply_target
                if not self._on_save_component(block, component):
                    return
            self._notify_changes_applied()
            return
        if self._block_apply_target is not None:
            if self._on_apply_block_changes(self._block_apply_target):
                self._notify_changes_applied()
            return
        if self._component_apply_target is None:
            return
        block, component = self._component_apply_target
        if self._on_save_component(block, component):
            self._notify_changes_applied()

    def _notify_changes_applied(self) -> None:
        """Confirm a successful Apply Changes so the user knows it worked."""
        QMessageBox.information(
            self, "Apply Changes", "The changes have been applied.")

    def _on_apply_block_changes(self, block: "ExpBlock") -> bool:
        """Apply dimension/weight edits to a block, regenerating components.

        Returns True when the edits were applied, False when nothing changed
        (block gone, user cancelled the dimension change, or regen failed).
        """
        if block not in self.inventory.blocks:
            return False
        old_d = int(block.d_b)
        new_d = int(self.block_dim_input.value())
        new_weight = float(self.block_weight_input.value())
        dimension_changed = new_d != old_d

        if not self._check_block_dimension(new_d):
            return False

        if dimension_changed and block.components:
            result = QMessageBox.question(
                self,
                "Apply Changes",
                "Changing this block dimension will regenerate "
                f"{len(block.components)} component(s) in this block. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if result != QMessageBox.StandardButton.Yes:
                self._load_block_into_editor(block)
                return False

        block.weight = new_weight
        if dimension_changed:
            block.d_b = new_d
            try:
                for component in block.components:
                    self._regenerate_component_for_dimension(component, new_d)
            except ValueError as exc:
                block.d_b = old_d
                QMessageBox.critical(self, "Apply Changes", str(exc))
                return False

        self._refresh_tree()
        self._block_apply_target = block
        self._select_block_item(block)
        self.total_d_label.setText(
            f"Instance overall dimension = {self.inventory.total_dimension()}")
        self.active_block_hint.setText(
            f"<i>Applied changes to <b>{block.label}</b>.</i>")
        self.active_block_hint.setVisible(True)
        return True

    def _regenerate_component_for_dimension(
            self, component: "ExpComponent", d: int) -> None:
        """Regenerate a stored component for a new parent block dimension."""
        if d < MIN_GUI_BLOCK_DIMENSION:
            raise ValueError(GUI_BLOCK_DIMENSION_MESSAGE)
        seed = int(component.seed)
        rng_for = self._component_rng_for(component)
        baseline = component.baseline
        coupled = getattr(component, "pair_coupled", None) or {}
        generated = {}
        for name, spec in (component.vector_specs or {}).items():
            if baseline == "B" and name in ("p_plus", "p_minus"):
                continue
            base, sign = self._split_pair(name)
            # Coupled pairs: skip the − draw and copy from + afterwards, so the
            # regenerated component keeps κ⁻ = κ⁺ (matched ±).
            if sign == "minus" and base in {"kappa", "p"}:
                continue
            generated[name] = self._sample_vector_from_spec(name, spec, d, rng_for(name))
            if sign == "plus" and base in {"kappa", "p"}:
                minus = f"{base}_minus"
                if coupled.get(base, True):
                    generated[minus] = generated[name].copy()
                else:
                    generated[minus] = self._sample_vector_from_spec(minus, spec, d, rng_for(minus))
        if baseline == "B":
            generated["p_single"] = float(
                component.generated.get("p_single", 1.0))
        psi, R = self._rotation_from_spec(d, rng_for("rotation"), component.rotation or {"mode": 1})
        generated["Psi_deg"] = psi
        generated["R"] = R
        component.generated = generated
        component.transforms = self._realize_transforms(
            d, rng_for, getattr(component, "transform_specs", None) or [])

    def _on_save_component(self, block: "ExpBlock",
                           component: "ExpComponent") -> bool:
        """Re-sample the component using current editor state, mutate in place.

        Component identity (and place in ``block.components``) is preserved,
        so any references (e.g. a future composite already built) still
        point at the same object. Transforms are left untouched.
        """
        if block not in self.inventory.blocks or component not in block.components:
            return False
        d = int(block.d_b)
        if not self._check_block_dimension(d):
            return False
        seed = int(self.seed_spin.value())
        # Editing preserves the component's identity, so keep its reroll counters
        # and any copy overrides; only the (possibly changed) seed and specs
        # drive the redraw.
        counters = getattr(component, "reroll_counters", None) or {}
        seed_overrides = getattr(component, "param_seed_overrides", None) or {}
        rng_for = self._rng_for(
            seed, counters, getattr(component, "param_copies", None) or {},
            seed_overrides)
        baseline = self.baseline_combo.currentText()
        try:
            generated, rotation_spec = self._editor_component_vectors(d, rng_for, baseline)
        except ValueError as exc:
            QMessageBox.critical(self, "Apply Changes", str(exc))
            return False
        transforms = self._realize_transforms(d, rng_for)
        self._apply_editor_state(
            component, baseline=baseline, seed=seed, generated=generated,
            rotation_spec=rotation_spec, counters=counters, transforms=transforms,
            seed_overrides=seed_overrides)

        self._refresh_tree()         # rebuilds the edit-button index cleanly
        self._component_apply_target = (block, component)
        self._block_apply_target = None
        self._select_component_item(block, component)
        self.active_block_hint.setText(
            f"<i>Applied changes to <b>{component.label}</b>.</i>")
        self._warn_if_block_has_hidden_component_centres(block)
        return True

    def _select_component_item(self, block: "ExpBlock",
                               component: "ExpComponent") -> None:
        """Restore tree selection to a component after a tree refresh."""
        self.tree.blockSignals(True)
        try:
            child = tree_model.find_component_item(self.tree, block, component)
            if child is not None:
                self.tree.setCurrentItem(child)
        finally:
            self.tree.blockSignals(False)

    def _select_block_item(self, block: "ExpBlock") -> None:
        """Restore tree selection to a block after a tree refresh."""
        self.tree.blockSignals(True)
        try:
            node = tree_model.find_block_item(self.tree, block)
            if node is not None:
                self.tree.setCurrentItem(node)
        finally:
            self.tree.blockSignals(False)

    def active_block(self) -> Optional["ExpBlock"]:
        """Return the block whose tree node is currently selected (or its parent for components).

        Uses ``selectedItems`` rather than ``currentItem`` because Qt's
        ``currentItem`` can lag behind selection clears.
        """
        return tree_model.active_block_from_tree(self.tree)

    def _block_has_unapplied_edits(self, block: "ExpBlock") -> bool:
        """True when the block editor's dimension/weight differ from what is
        saved on ``block`` (i.e. the user changed them but did not Apply)."""
        if block is None or block not in self.inventory.blocks:
            return False
        return (int(self.block_dim_input.value()) != int(block.d_b)
                or float(self.block_weight_input.value()) != float(block.weight))

    def _on_new_component_for_block(self, blk: "ExpBlock") -> None:
        """Instance-row '+': select this block, then open a blank component
        editor for it."""
        # If the block editor has dimension/weight changes that were not saved
        # yet, stop and tell the user to Apply Changes first (otherwise opening
        # the component editor would silently drop those edits).
        target = self._block_apply_target
        if (self._editor_mode == "block" and target is not None
                and self._block_has_unapplied_edits(target)):
            QMessageBox.information(
                self, "Apply Changes",
                "This instance has dimension or weight changes that have not "
                "been saved. Press Apply Changes first, then add a component.")
            return
        for i in range(self.tree.topLevelItemCount()):
            node = self.tree.topLevelItem(i)
            if node.data(0, _ROLE) is blk:
                self.tree.setCurrentItem(node)
                break
        self.on_new_component()

    def on_new_component(self) -> None:
        """Open a blank component editor for the selected block (no commit yet)."""
        blk = self.active_block()
        if blk is None:
            return
        self._new_component_block = blk
        self._reset_parameter_fields()   # blank the editor (no reset popup)
        self._advance_seed_box()     # a fresh component gets the next free seed
        # This auto-assigned seed is the session default a later Reset returns to.
        self._editor_seed_default = int(self.seed_spin.value())
        self._set_editor_mode("new")
        # Keep focus inside the component editor; otherwise it can drift to the
        # next focusable widget (the ELA sample-factor box) and select its text.
        self.baseline_combo.setFocus()

    def _set_editor_mode(self, mode: str) -> None:
        """Single source of truth for what the middle panel shows + which
        bottom buttons are live. mode in {"none","block","new","edit"}."""
        self._editor_mode = mode
        is_component = mode in ("new", "edit")
        self.block_edit_card.setVisible(mode == "block")
        self._set_component_editor_visible(is_component)
        self.add_btn.setEnabled(mode == "new")
        self.apply_changes_btn.setEnabled(mode in ("block", "edit"))
        # Reset Parameters only applies to the component editor; it is disabled
        # (greyed out) while editing block parameters.
        self.reset_btn.setEnabled(is_component)
        if is_component:
            self._set_component_editor_enabled(True)
        if mode in ("none", "block"):
            self.active_block_hint.setText(
                self._default_hint_html() if mode == "none"
                else "<i>Use the <b>+</b> on an instance in the Instance Layers "
                     "list to add a new component, or select a component there "
                     "to edit it here.</i>")
            self.active_block_hint.setVisible(True)
        else:
            self.active_block_hint.setVisible(False)
        self._refresh_component_body_height()

    def _confirm_discard_new_component(self) -> bool:
        """Ask before abandoning an uncommitted new component. True = discard."""
        result = QMessageBox.question(
            self, "Discard component",
            "Discard the unsaved new component?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return result == QMessageBox.StandardButton.Yes

    def _on_tree_selection_changed(self) -> None:
        """Drive editor mode from the tree selection."""
        if self._editor_mode == "new":
            if not self._confirm_discard_new_component():
                # Keep editing: restore the selection to the new component's
                # block so active_block() still resolves (Add Component reads
                # the tree selection), and stay in "new".
                self.tree.blockSignals(True)
                self.tree.clearSelection()
                blk = self._new_component_block
                if blk is not None:
                    for i in range(self.tree.topLevelItemCount()):
                        node = self.tree.topLevelItem(i)
                        if node.data(0, _ROLE) is blk:
                            self.tree.setCurrentItem(node)
                            break
                self.tree.blockSignals(False)
                self._set_editor_mode("new")
                return
            self._new_component_block = None
        self._sync_rotation_visibility()
        items = self.tree.selectedItems()
        if not items:
            self._component_apply_target = None
            self._block_apply_target = None
            self._set_editor_mode("none")
            return
        data = tree_model.item_role_data(items[0])
        if isinstance(data, ExpComponent):
            parent = items[0].parent()
            blk = parent.data(0, _ROLE) if parent is not None else None
            if isinstance(blk, ExpBlock):
                self._load_component_spec_into_editor(blk, data)
                self._component_apply_target = (blk, data)
                self._block_apply_target = None
                self.active_block_hint.setText(
                    f"<i>Loaded <b>{data.label}</b>. Adjust fields and press "
                    "<b>Apply Changes</b> to update it.</i>")
                self._set_editor_mode("edit")
        elif isinstance(data, ExpBlock):
            self._component_apply_target = None
            self._block_apply_target = data
            self._load_block_into_editor(data)
            self._set_editor_mode("block")

    # ------------------------------------------------------------- composite
    def _on_composite_selected(self, idx: int) -> None:
        """Load the blocks snapshot for composite[idx] into the tree."""
        import copy
        snapshots = getattr(self.inventory, "blocks_snapshots", [])
        if idx < 0 or idx >= len(snapshots):
            return
        self._active_composite_idx = idx
        self.inventory.blocks = copy.deepcopy(snapshots[idx])
        self._refresh_tree()
        self._on_tree_selection_changed()

    def _on_instance_deleted(self, idx: int) -> None:
        """Keep ``_active_composite_idx`` aligned after the ELA panel removes
        prepared instance ``idx`` (the inventory lists have already shifted)."""
        if self._active_composite_idx is None:
            return
        if self._active_composite_idx == idx:
            self._active_composite_idx = None
        elif self._active_composite_idx > idx:
            self._active_composite_idx -= 1

    def on_build_composite(self) -> None:
        from portal_studio.experiment.assemble import build_composite
        if not self.inventory.blocks:
            QMessageBox.information(self, "Prepare Instance for ELA",
                                    "Create at least one block first.")
            return
        for block in self.inventory.blocks:
            if not self._check_block_dimension(int(block.d_b)):
                return
        # Eq 13 requires K >= 1 per block; empty blocks would have undefined min_k.
        empty = [b.label for b in self.inventory.blocks if not b.components]
        if empty:
            QMessageBox.critical(
                self, "Prepare Instance for ELA",
                f"Add at least one component to each block before preparing "
                f"the instance. Empty block(s): {', '.join(empty)}.")
            return
        for block in self.inventory.blocks:
            self._warn_if_block_has_hidden_component_centres(block)
        try:
            composite = build_composite(self.inventory.blocks, seed=0)
            self._active_composite_idx = self.inventory.add_composite(composite)
        except Exception as exc:   # noqa: BLE001 - surface assembly errors
            QMessageBox.critical(self, "Prepare Instance for ELA", str(exc))
            return
        # Surface the new composite in the ELA panel's instance list.
        if hasattr(self, "ela_panel"):
            self.ela_panel.refresh_instance_list()
        QMessageBox.information(
            self, "Prepare Instance for ELA",
            "Instance prepared for ELA with overall dimension "
            f"{self.inventory.composite['d']}.")

    def on_save_composite(self) -> None:
        from portal_studio.experiment.assemble import save_studio_composite
        if self.inventory.composite is None:
            QMessageBox.information(
                self, "Save Instance JSON", "Prepare an instance for ELA first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Instance JSON", "PORTAL_instance.json", "JSON Files (*.json)")
        if not path:
            return
        try:
            written = save_studio_composite(
                self.inventory.composite,
                filename=os.path.basename(path),
                output_dir=os.path.dirname(path) or ".")
        except Exception as exc:   # noqa: BLE001
            QMessageBox.critical(self, "Save Instance JSON", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Instance JSON saved to:\n{written}")
