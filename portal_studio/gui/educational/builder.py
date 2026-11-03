"""Builder tab: workflow-oriented component, block, and composite authoring."""
from __future__ import annotations

import copy
import re

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portal_studio.builder.assembly import (
    BlockAssemblyError,
    CompositeAssemblyError,
    assemble_block,
    assemble_composite,
)
from portal_core.save_instance import save_composite_as_unified
from portal_studio.gui.educational.component_editor import ComponentEditorWidget
from portal_studio.gui.educational import figure_io
from portal_studio.gui.educational import landscape as educational_landscape
from portal_studio.gui.common.icons import make_icon_button, make_reset_icon, make_save_icon
from portal_studio.gui.common.help import (
    formula_intro, help_body, make_help_button, make_parameter_help_button,
    math_text, show_help_dialog, term,
    base_form_help_paragraphs, centre_help_paragraphs,
    kappa_help_paragraphs, exponent_help_paragraphs,
    neutralization_help_paragraphs, delta_help_paragraphs,
    r_ref_help_paragraphs, transform_type_help_paragraphs,
    transform_param_help_paragraphs,
)
from portal_studio.gui.common.plot_style import (
    style_compact_axes,
    style_compact_colorbar,
)
from portal_studio.gui.common.widgets import NoWheelDoubleSpinBox
from portal_studio.gui.common.theme import LIGHT, PARAM_TABLE_TINTS, style_figure
from portal_studio.session import BlockSpec, ComponentSpec, CompositeRef, SessionInventory
from portal_studio.validation import validate_instance_for_gui

_NO_BUTTONS = QAbstractSpinBox.ButtonSymbols.NoButtons
_SIDE_PANEL_MIN_WIDTH = 320
_PARAM_PANEL_MIN_WIDTH = 340
_EDUCATIONAL_SPLITTER_SIZES = [560, 680, 1000]
_EDU_LABEL_W = 64
_EDU_SPIN_W = 76        # hidden per-axis transform grid only (see _fit_transform_param_widgets)

# Public save entry point using the unified instance format.
def save_composite(composite, *args, **kwargs):
    validate_instance_for_gui(composite)
    return save_composite_as_unified(composite, *args, **kwargs)

# The parameter / transform "bar" menus keep this familiar dropdown width; the
# value boxes are deliberately narrower (see _edu_field_width).
_EDU_DROPDOWN_W = 180
_EDU_XY_SAMPLE = "(-100, -100)"   # widest typical value shown (matches "(0.15, 0.15)")


def _edu_field_width(widget: QWidget) -> int:
    """Width for a value box (scalar spin box or combined "(x, y)" field): just
    wide enough to show ``_EDU_XY_SAMPLE`` in ``widget``'s font, plus the QSS
    chrome (6px padding + 1px border each side) and a little caret room. This is
    smaller than the menus, but derived from font metrics so values like
    "(0.15, 0.15)" never clip — even on a larger font / higher-DPI display."""
    text_px = widget.fontMetrics().horizontalAdvance(_EDU_XY_SAMPLE)
    return text_px + 2 * (6 + 1) + 6


class _XYParenEdit(QLineEdit):
    """A '(x, y)' field whose structural characters (the parentheses and the
    comma) can't be deleted — only the numbers between them are editable."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._guarding = False
        self.textChanged.connect(self._reguard)

    def keyPressEvent(self, event) -> None:
        # Block Backspace/Delete from removing a structural ( , ) character, so
        # the user can never delete them while editing the numbers.
        if not self.hasSelectedText():
            text = self.text()
            pos = self.cursorPosition()
            if (event.key() == Qt.Key.Key_Backspace
                    and pos > 0 and text[pos - 1] in "(),"):
                return   # the character before the caret is structural: keep it
            if (event.key() == Qt.Key.Key_Delete
                    and pos < len(text) and text[pos] in "(),"):
                return   # the character at the caret is structural: keep it
        super().keyPressEvent(event)

    def _reguard(self, text: str) -> None:
        if self._guarding:
            return
        inner = text.replace("(", "").replace(")", "")
        # Keep exactly one comma so the "(x, y)" structure (and its comma)
        # survives even a select-all delete, which the per-key guard can't catch.
        commas = inner.count(",")
        if commas == 0:
            inner = inner + ", "
        elif commas > 1:
            head, _, tail = inner.partition(",")
            inner = head + "," + tail.replace(",", "")
        new = f"({inner})"
        if new == text:
            return
        self._guarding = True
        caret = self.cursorPosition()
        self.setText(new)
        # Keep the caret between the parentheses.
        self.setCursorPosition(min(max(1, caret), max(1, len(new) - 1)))
        self._guarding = False


class SegmentedSwitch(QWidget):
    """A compact two-option sliding switch (e.g. Surface / Contour).

    The selected option is shown by an accent-filled pill that slides between
    the two halves. Emits ``valueChanged`` with 0 (first) or 1 (second).
    """

    valueChanged = Signal(int)

    def __init__(self, options, parent=None) -> None:
        super().__init__(parent)
        self._options = list(options)
        self._value = 0
        self._track = QColor("#e2e7ef")
        self._accent = QColor("#0b66d8")
        self._on_text = QColor("#ffffff")
        self._off_text = QColor("#667085")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_palette(self, palette: dict) -> None:
        self._track = QColor(palette["border_soft"])
        self._accent = QColor(palette["accent"])
        self._off_text = QColor(palette["muted"])
        self.update()

    def value(self) -> int:
        return self._value

    def is_first(self) -> bool:
        return self._value == 0

    def setValue(self, v) -> None:
        v = 1 if v else 0
        if v != self._value:
            self._value = v
            self.update()
            self.valueChanged.emit(v)

    def sizeHint(self) -> QSize:
        return QSize(190, 30)

    def minimumSizeHint(self) -> QSize:
        return QSize(150, 26)

    def mousePressEvent(self, event) -> None:
        self.setValue(0 if event.position().x() < self.width() / 2 else 1)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        radius = h / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self._track))
        p.drawRoundedRect(QRectF(0, 0, w, h), radius, radius)
        half = w / 2
        thumb_x = 2.0 if self._value == 0 else half
        p.setBrush(QBrush(self._accent))
        p.drawRoundedRect(QRectF(thumb_x, 2, half - 2, h - 4), (h - 4) / 2, (h - 4) / 2)
        for i, label in enumerate(self._options):
            p.setPen(self._on_text if i == self._value else self._off_text)
            p.drawText(QRectF(i * half, 0, half, h), Qt.AlignmentFlag.AlignCenter, label)


class BuilderTab(QWidget):
    """2D authoring surface: layers on the left, parameters in the middle."""

    @staticmethod
    def _palette_bootstrap() -> dict:
        """The palette, usable before ``self._palette`` is assigned in __init__."""
        return LIGHT

    def __init__(self, inventory: SessionInventory, parent=None) -> None:
        super().__init__(parent)
        self.inventory = inventory

        self.editor = ComponentEditorWidget()
        self.editor.setObjectName("educationalEditorShell")
        self.editor.left_panel.setObjectName("educationalEditorLeftPanel")
        p = self._palette_bootstrap()
        self.editor.setStyleSheet(
            "QWidget#educationalEditorShell,"
            " QWidget#educationalEditorLeftPanel {"
            f" background: {p['surface']}; border: none;"
            " }"
            f" QSplitter {{ background: {p['surface']}; border: none; }}"
            # Active fields show near-black text; disabled fields go gray —
            # scoped to this editor shell so it matches Generator's fields
            # without leaking into the app-wide base rule.
            f" QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ color: {p['field_text_dark']}; }}"
            f" QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,"
            f" QDoubleSpinBox:disabled {{ color: {p['hint']}; }}"
            # Primary-variant buttons (Add Block / Add Component) go grey
            # when disabled so the frozen state reads clearly.
            f" QPushButton[variant=\"primary\"]:disabled {{"
            f"   background: {p['field_disabled_primary_bg']}; color: {p['hint']};"
            f"   border: 1px solid {p['field_disabled_primary_border']};"
            f" }}"
        )
        self.editor.left_panel.setStyleSheet(
            "QWidget#educationalEditorLeftPanel {"
            f" background: {self._palette_bootstrap()['surface']}; border: none;"
            "}"
        )
        self.editor.hide_legacy_panel()
        self.editor.right_panel.hide()
        if hasattr(self.editor, "transform_controls_row"):
            self.editor.transform_controls_row.hide()
        if self.editor.layout() is not None:
            self.editor.layout().setContentsMargins(0, 0, 0, 0)
            self.editor.layout().setSpacing(0)
        self.editor.left_layout.setContentsMargins(0, 0, 0, 0)
        self._modernize_embedded_editor()
        self._make_transform_section_collapsible()
        self._build_tree_icons()
        self._active_block_idx: int | None = None
        self._active_composite_idx: int | None = None
        self._prepared_block_instances: dict[int, dict] = {}
        self._component_apply_idx: int | None = None
        self._block_apply_idx: int | None = None
        self._transform_apply_idx: int | None = None

        left_panel = self._build_parameter_panel()
        center_panel = self._build_visualization_panel()
        right_panel = self._build_project_panel()
        self._build_compatibility_widgets()

        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.main_splitter = splitter
        splitter.addWidget(right_panel)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        # Keep Instance Layers and Component Parameters visually balanced.
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 3)
        splitter.setSizes(_EDUCATIONAL_SPLITTER_SIZES)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        outer.addWidget(splitter)
        # Keep status labels for internal updates, but do not show the footer
        # strip in the Education tab UI.
        self._status_bar_widget = self._build_status_bar()
        self._status_bar_widget.hide()

        self._refresh_components_table()
        self._refresh_blocks_list()
        self._refresh_project_tree()
        self._refresh_summary()

        # Live, debounced visualization: any parameter change re-renders the
        # current component's landscape ~200 ms after the last edit.
        self._live_timer = QtCore.QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(200)
        self._live_timer.timeout.connect(self._do_live_plot)
        self._wire_live_preview()
        self._setup_transform_list()
        self._show_current_landscape_hint()

        self._palette = LIGHT
        self.apply_palette(LIGHT)
        QtCore.QTimer.singleShot(0, self._fit_editor)

    # -------------------- UI construction --------------------
    def _modernize_embedded_editor(self) -> None:
        """Restyle the embedded editor controls for the builder shell."""
        self._replace_parameter_layout()
        for panel in (self.editor.baseline_params_panel, self.editor.transform_panel):
            panel.setObjectName("embeddedEditorCard")
            for label in panel.findChildren(QLabel):
                if label.text() in ("Baseline Parameters", "Transformation Configuration"):
                    label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    label.setObjectName("panelTitle")
                    label.setStyleSheet("")
        self._sanitize_editor_labels()
        self.editor.transform_combo.currentTextChanged.connect(self._sanitize_editor_labels)

    def _replace_parameter_layout(self) -> None:
        """Group existing parameter widgets into workflow sections without rewiring them."""
        if hasattr(self.editor, "workflow_params_panel"):
            return
        self._build_flat_parameter_layout()
        return

        self.editor.baseline_params_panel.hide()
        panel = QWidget()
        self.editor.workflow_params_panel = panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(
            self._make_parameter_section(
                "Baseline",
                [("Baseline", self.editor.baseline_combo, None, None)],
                help_button=self._section_help_button(
                    "baseline_help_btn",
                    "Explain the Baseline section",
                    self._show_baseline_help,
                ),
            )
        )
        layout.addWidget(
            self._make_parameter_section(
                "Geometry",
                [
                    ("c<sub>x</sub>", self.editor.center_x, "c<sub>y</sub>", self.editor.center_y),
                    ("Rotation", self.editor.rotation_spin, "β", self.editor.beta_spin),
                ],
                two_column=True,
                help_button=self._section_help_button(
                    "geometry_help_btn",
                    "Explain the Geometry section",
                    self._show_geometry_help,
                ),
            )
        )
        self.editor.neutral_enable.setText("Enable")  # the section title already says Neutralization
        layout.addWidget(
            self._make_parameter_section(
                "Neutralization",
                [
                    ("", self.editor.neutral_enable, None, None),
                    ("Δ", self.editor.neutral_delta, "r<sub>ref</sub>", self.editor.neutral_rref),
                ],
                two_column=True,
                help_button=self._section_help_button(
                    "neutralization_help_btn",
                    "Explain the Neutralization section",
                    self._show_neutralization_help,
                ),
            )
        )

        self.editor.p_plus_x_label = QLabel("p<sup>+</sup><sub>x</sub>")
        self.editor.p_plus_y_label = QLabel("p<sup>+</sup><sub>y</sub>")
        self.editor.p_minus_x_label = QLabel("p<sup>-</sup><sub>x</sub>")
        self.editor.p_minus_y_label = QLabel("p<sup>-</sup><sub>y</sub>")
        self.editor.p_single_label = QLabel("p")
        layout.addWidget(
            self._make_parameter_section(
                "Advanced",
                [
                    ("κ<sup>+</sup><sub>x</sub>", self.editor.kappa_plus_x, "κ<sup>+</sup><sub>y</sub>", self.editor.kappa_plus_y),
                    ("κ<sup>-</sup><sub>x</sub>", self.editor.kappa_minus_x, "κ<sup>-</sup><sub>y</sub>", self.editor.kappa_minus_y),
                    (self.editor.p_plus_x_label, self.editor.p_plus_x, self.editor.p_plus_y_label, self.editor.p_plus_y),
                    (self.editor.p_minus_x_label, self.editor.p_minus_x, self.editor.p_minus_y_label, self.editor.p_minus_y),
                    (self.editor.p_single_label, self.editor.p_single, None, None),
                ],
                two_column=True,
                help_button=self._section_help_button(
                    "advanced_help_btn",
                    "Explain the Advanced section",
                    self._show_advanced_help,
                ),
            )
        )
        self.editor.left_layout.insertWidget(0, panel)
        # Each card sizes to its content (so collapsing shrinks it to the
        # header); leftover vertical space then pools at the bottom instead of
        # inflating a card.
        from PySide6.QtWidgets import QSizePolicy

        for section in (panel, self.editor.transform_panel):
            section.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        # Set hidden panels to zero height so they do not affect the visible
        # column's size hint.
        self.editor.baseline_params_panel.setMaximumHeight(0)
        if hasattr(self.editor, "legacy_actions_row"):
            self.editor.legacy_actions_row.setMaximumHeight(0)
        self.editor._on_baseline_changed(self.editor.baseline_combo.currentText())

    def _build_flat_parameter_layout(self) -> None:
        """Build the Education component editor as a Generator-style form."""
        self.editor.baseline_params_panel.hide()

        card = QFrame()
        card.setObjectName("sectionCard")
        # Flush white surface (no border, no shadow) so the editor reads as one
        # continuous white column like the Generator — not a card floating on grey.
        card.setStyleSheet(
            "QFrame#sectionCard, QWidget#sectionCard {"
            " background: transparent; border: none; }")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(6)
        card.setGraphicsEffect(None)
        self.component_parameters_card = card

        panel = QWidget()
        panel.setObjectName("parameterPane")
        panel.setStyleSheet(
            f"QWidget#parameterPane {{ background: {self._palette_bootstrap()['surface']}; border: none; }}")
        self.editor.workflow_params_panel = panel
        self.educational_parameter_form = panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        title = QLabel("Component Parameters")
        # 13pt matches the Generator tab's "Component Parameters" header. The
        # title is added into the first shaded band below (with Base form), so it
        # sits inside the grey window like the Generator's section header.
        title.setObjectName("panelHeading")
        self.educational_parameter_title = title

        card_layout.addWidget(panel)

        self.parameter_help_buttons: dict[str, QPushButton] = {}
        self._educational_rows: dict[str, QWidget] = {}

        self.editor.p_single_label = QLabel("p")

        # State for the combined "x, y" fields and the Separate ± pairs.
        self._edu_xy_refreshers: list = []
        self._edu_minus_refresh: dict = {}
        self._edu_minus_spins: dict = {}
        self._edu_coupled: dict = {"kappa": True, "p": True}
        self._edu_pair_checkboxes: dict = {}
        self._edu_plus_fields: dict = {}
        self._edu_tf_refreshers: list = []

        base_row = self._educational_param_row(
            ("Base form", self.editor.baseline_combo),
            help_keys=("base_form",),
        )
        center_row = self._make_edu_vector_row(
            "c", self.editor.center_x, self.editor.center_y, help_key="c")
        beta_row = self._educational_param_row(
            ("β", self.editor.beta_spin),
            help_keys=("beta",),
        )
        rotation_row = self._educational_param_row(
            ("Rotation", self.editor.rotation_spin),
            help_keys=("rotation",),
        )
        kappa_plus_row, kappa_minus_row = self._make_edu_pair_rows(
            "kappa", "κ",
            self.editor.kappa_plus_x, self.editor.kappa_plus_y,
            self.editor.kappa_minus_x, self.editor.kappa_minus_y,
            help_key="kappa_plus")
        p_plus_row, p_minus_row = self._make_edu_pair_rows(
            "p", "p",
            self.editor.p_plus_x, self.editor.p_plus_y,
            self.editor.p_minus_x, self.editor.p_minus_y,
            help_key="p_plus")
        p_single_row = self._educational_param_row(
            (self.editor.p_single_label, self.editor.p_single),
            help_keys=("p_single",),
        )

        # Group rows into shaded bands like the Generator panel: Base form,
        # then {c, β}, then {exponent p, κ, Rotation}. Neutralization gets its
        # own band below. The white panel shows between bands as a gap, so each
        # group reads as one card instead of every variable in its own window.
        layout.addWidget(self._edu_param_band(title, base_row))
        layout.addWidget(self._edu_param_band(center_row, beta_row))
        layout.addWidget(self._edu_param_band(
            p_plus_row, p_minus_row, p_single_row,
            kappa_plus_row, kappa_minus_row, rotation_row))

        self._educational_rows = {
            "p_plus": p_plus_row,
            "p_minus": p_minus_row,
            "p_single": p_single_row,
        }
        # Keep generator-compatible registry entries without duplicate visible
        # minus-side help icons.
        for hidden_key in ("kappa_minus", "p_minus"):
            self._educational_hidden_help_button(hidden_key)

        # Neutralization, Generator-style: the Enable checkbox and help icon
        # sit on the header row; Δ / r_ref move to a second row that is shown
        # only when Enable is checked (default off).
        neutral_group = QWidget()
        neutral_group.setObjectName("educationalNeutralizationRow")
        neutral_v = QVBoxLayout(neutral_group)
        neutral_v.setContentsMargins(0, 0, 0, 0)
        neutral_v.setSpacing(4)

        neutral_header_layout = QHBoxLayout()
        neutral_header_layout.setContentsMargins(0, 0, 0, 0)
        neutral_header_layout.setSpacing(6)
        neutral_label = QLabel("Neutralization")
        # "Neutralization" is wider than the _EDU_LABEL_W param-label column, so
        # size it to its own text (+padding) and let it push the Enable checkbox
        # right — otherwise the word is clipped.
        neutral_label.setFixedWidth(
            neutral_label.fontMetrics().horizontalAdvance("Neutralization") + 8)
        neutral_header_layout.addWidget(neutral_label)
        self.editor.neutral_enable.setText("Enable")
        neutral_header_layout.addWidget(self.editor.neutral_enable)
        neutral_header_layout.addWidget(
            self._educational_help_button("neutralization"))
        neutral_header_layout.addStretch(1)
        neutral_v.addLayout(neutral_header_layout)

        neutral_values = QWidget()
        neutral_values_layout = QHBoxLayout(neutral_values)
        neutral_values_layout.setContentsMargins(_EDU_LABEL_W + 4, 0, 0, 0)
        neutral_values_layout.setSpacing(6)
        self._add_inline_pair(
            neutral_values_layout, "Δ", self.editor.neutral_delta,
            label_width=20)
        self._add_inline_pair(
            neutral_values_layout, "r<sub>ref</sub>", self.editor.neutral_rref,
            label_width=30)
        neutral_values_layout.addStretch(1)
        neutral_v.addWidget(neutral_values)

        layout.addWidget(self._edu_param_band(neutral_group))
        self.educational_neutralization_row = neutral_group
        self.educational_neutralization_values_row = neutral_values
        # Default on, and Δ / r_ref follow the Enable checkbox.
        self.editor.neutral_enable.setChecked(True)
        neutral_values.setVisible(True)
        self.editor.neutral_enable.toggled.connect(neutral_values.setVisible)
        # Showing the Δ / r_ref row grows the card, so the editor must be
        # re-fitted — otherwise the extra height is stolen from the rows below
        # (the Transform Sequence), which overlap.
        self.editor.neutral_enable.toggled.connect(lambda *_: self._fit_editor())
        self.editor.left_layout.insertWidget(0, card)
        from PySide6.QtWidgets import QSizePolicy

        for section in (card, panel, self.editor.transform_panel):
            section.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.editor.baseline_params_panel.setMaximumHeight(0)
        if hasattr(self.editor, "legacy_actions_row"):
            self.editor.legacy_actions_row.setMaximumHeight(0)
        self.editor.baseline_combo.currentTextChanged.connect(
            self._sync_educational_baseline_rows)
        self._sync_edu_from_editor()
        self._sync_educational_baseline_rows(self.editor.baseline_combo.currentText())
        self.editor._on_baseline_changed(self.editor.baseline_combo.currentText())

    def _sync_educational_baseline_rows(self, baseline: str) -> None:
        """Mirror Generator ordering by toggling p rows as baseline changes."""
        if not hasattr(self, "_educational_rows"):
            return
        is_b = str(baseline).strip().upper() == "B"
        self._educational_rows.get("p_plus", QWidget()).setVisible(not is_b)
        # The p− row also needs the pair to be in 'Separate ±' mode.
        self._educational_rows.get("p_minus", QWidget()).setVisible(
            (not is_b) and not self._edu_coupled.get("p", True))
        self._educational_rows.get("p_single", QWidget()).setVisible(is_b)
        self._fit_editor()

    @staticmethod
    def _parse_xy(text: str, fallback_x: float, fallback_y: float) -> tuple[float, float]:
        """Parse a '(x, y)' string (parens optional). A single number fills both
        axes. Falls back to the given values if the text can't be parsed."""
        text = str(text).replace("(", "").replace(")", "").replace(";", ",")
        parts = [p.strip() for p in text.split(",")]
        nums = [p for p in parts if p != ""]
        try:
            if len(nums) == 1:
                v = float(nums[0])
                return v, v
            return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            return fallback_x, fallback_y

    def _make_edu_xy_field(self, spin_x, spin_y, base=None, is_minus=False,
                           refreshers=None) -> QLineEdit:
        """A 'x, y' text box backed by two (hidden) spinboxes. When ``base`` is a
        coupled pair and this is the '+' field, edits mirror onto the '−' spins.
        ``refreshers`` is the list this field's refresher registers in (lets the
        dynamically-rebuilt transform fields use a separate list)."""
        if refreshers is None:
            refreshers = self._edu_xy_refreshers
        field = _XYParenEdit()
        field.setObjectName("educationalXYField")
        field.setFixedWidth(_edu_field_width(field))

        def refresh() -> None:
            field.setText(f"({spin_x.value():g}, {spin_y.value():g})")

        def commit() -> None:
            x, y = self._parse_xy(field.text(), spin_x.value(), spin_y.value())
            spin_x.setValue(x)
            spin_y.setValue(y)
            if base is not None and not is_minus and self._edu_coupled.get(base, True):
                mx, my = self._edu_minus_spins[base]
                mx.setValue(spin_x.value())
                my.setValue(spin_y.value())
                if base in self._edu_minus_refresh:
                    self._edu_minus_refresh[base]()
            refresh()
            self._schedule_live_plot()

        field.editingFinished.connect(commit)
        refresh()
        # Register in the caller's list, not always the persistent one. Transform
        # fields pass _edu_tf_refreshers (cleared on every transform-type rebuild),
        # so their refreshers don't linger in _edu_xy_refreshers and fire against
        # spinboxes that were deleted when the transform type changed.
        refreshers.append(refresh)
        if is_minus and base is not None:
            self._edu_minus_refresh[base] = refresh
        return field

    def _make_edu_vector_row(self, symbol: str, spin_x, spin_y, help_key: str) -> QWidget:
        """A single combined-(x,y) row (no ± pair), e.g. the centre c."""
        row = QWidget()
        row.setObjectName("educationalParameterRow")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        label = QLabel(symbol)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setFixedWidth(_EDU_LABEL_W)
        rl.addWidget(label)
        rl.addWidget(self._make_edu_xy_field(spin_x, spin_y))
        rl.addWidget(self._educational_help_button(help_key))
        rl.addStretch(1)
        return row

    def _make_edu_pair_rows(self, base: str, symbol: str, px, py, mx, my,
                            help_key: str, refreshers=None) -> tuple[QWidget, QWidget]:
        """Build the '+'/'−' combined-(x,y) rows for a vector pair with a
        Separate ± checkbox. The '−' row is hidden until the box is ticked.
        ``refreshers`` lets the transform pairs use a separate refresher list."""
        self._edu_minus_spins[base] = (mx, my)
        self._edu_coupled.setdefault(base, True)

        plus_row = QWidget()
        plus_row.setObjectName("educationalParameterRow")
        prv = QVBoxLayout(plus_row)
        prv.setContentsMargins(0, 0, 0, 0)
        prv.setSpacing(2)

        plus_top = QHBoxLayout()
        plus_top.setContentsMargins(0, 0, 0, 0)
        plus_top.setSpacing(6)
        plus_label = QLabel(symbol)
        plus_label.setTextFormat(Qt.TextFormat.RichText)
        plus_label.setFixedWidth(_EDU_LABEL_W)
        plus_top.addWidget(plus_label)
        plus_field = self._make_edu_xy_field(px, py, base=base, refreshers=refreshers)
        self._edu_plus_fields[base] = plus_field
        plus_top.addWidget(plus_field)
        # "Separate ± directions" sits inline, right of the value box — matched
        # to the Generator panel's pair checkbox label (muted grey, 11px), so it
        # reads as a subordinate toggle on the row rather than a separate line.
        cb = QCheckBox("Separate ± directions")
        cb.setStyleSheet(f"color: {self._palette_bootstrap()['progress_label']}; font-size: 11px;")
        cb.setChecked(not self._edu_coupled.get(base, True))
        plus_top.addWidget(cb)
        plus_top.addWidget(self._educational_help_button(help_key))
        plus_top.addStretch(1)
        prv.addLayout(plus_top)

        minus_row = QWidget()
        minus_row.setObjectName("educationalParameterRow")
        ml = QHBoxLayout(minus_row)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(6)
        minus_label = QLabel(f"{symbol}<sup>-</sup>")
        minus_label.setTextFormat(Qt.TextFormat.RichText)
        minus_label.setFixedWidth(_EDU_LABEL_W)
        ml.addWidget(minus_label)
        ml.addWidget(self._make_edu_xy_field(
            mx, my, base=base, is_minus=True, refreshers=refreshers))
        ml.addStretch(1)
        minus_row.setVisible(not self._edu_coupled.get(base, True))
        plus_label.setText(symbol if self._edu_coupled.get(base, True)
                           else f"{symbol}<sup>+</sup>")

        def toggled(checked: bool) -> None:
            self._edu_coupled[base] = not checked
            minus_row.setVisible(checked)
            plus_label.setText(f"{symbol}<sup>+</sup>" if checked else symbol)
            if not checked:                       # recouple: mirror − = +
                mx.setValue(px.value())
                my.setValue(py.value())
                if base in self._edu_minus_refresh:
                    self._edu_minus_refresh[base]()
            self._fit_editor()
            self._schedule_live_plot()

        cb.toggled.connect(toggled)
        self._edu_pair_checkboxes[base] = cb
        return plus_row, minus_row

    def _sync_edu_from_editor(self) -> None:
        """Set ± coupling from the spin values and refresh the combined fields.
        Call after loading a component or resetting the editor."""
        e = self.editor
        pairs = {
            "kappa": (e.kappa_plus_x, e.kappa_plus_y, e.kappa_minus_x, e.kappa_minus_y),
            "p": (e.p_plus_x, e.p_plus_y, e.p_minus_x, e.p_minus_y),
        }
        for base, (px, py, mx, my) in pairs.items():
            cb = self._edu_pair_checkboxes.get(base)
            if cb is None:
                continue
            separated = (px.value() != mx.value()) or (py.value() != my.value())
            cb.setChecked(separated)              # fires toggled → rows/labels
        for refresh in getattr(self, "_edu_xy_refreshers", []):
            refresh()
        # Rebuild transform rows from the loaded spin values.
        self._rebuild_edu_transform_rows()

    def _educational_help_button(self, key: str) -> QPushButton:
        button = make_parameter_help_button(
            f"Explain {key.replace('_', ' ')}",
            lambda k=key: self._show_educational_parameter_help(k),
        )
        self.parameter_help_buttons[key] = button
        return button

    def _educational_hidden_help_button(self, key: str) -> QPushButton:
        button = make_parameter_help_button(
            f"Explain {key.replace('_', ' ')}",
            lambda _checked=False, k=key: self._show_educational_parameter_help(k),
        )
        self.parameter_help_buttons[key] = button
        return button

    def _edu_param_band(self, *rows: QWidget) -> QWidget:
        """Wrap parameter rows in one shaded band, matching the Generator panel
        where each section (Base form / c,β / p,κ,Rotation / Neutralization)
        reads as a single grey card. Bands sit on the white pane, so the panel
        shows through the layout spacing as a white gap between them."""
        band = QWidget()
        band.setObjectName("educationalParamBand")
        bl = QVBoxLayout(band)
        # Match the Generator's flat section body: a white pane inset around
        # each grey group, then 10px horizontal padding inside the grey group.
        bl.setContentsMargins(10, 6, 10, 6)
        bl.setSpacing(4)
        for row in rows:
            bl.addWidget(row)
        return band

    def _educational_param_row(self, *pairs: tuple, help_keys: tuple[str, ...]) -> QWidget:
        row = QWidget()
        row.setObjectName("educationalParameterRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for idx, (label, widget) in enumerate(pairs):
            # Keep a consistent two-column rhythm across rows so text boxes
            # align vertically like the Generator panel.
            label_width = _EDU_LABEL_W if idx == 0 else 34
            if idx > 0:
                layout.addSpacing(24)
            self._add_inline_pair(layout, label, widget, label_width=label_width)
        for key in help_keys:
            layout.addWidget(self._educational_help_button(key))
        layout.addStretch(1)
        return row

    def _add_inline_pair(
        self,
        layout: QHBoxLayout,
        label,
        widget: QWidget,
        label_width: int = _EDU_LABEL_W,
        field_width: int | None = None,
    ) -> None:
        from PySide6.QtWidgets import QSizePolicy

        if field_width is None:
            # Menus keep the wider "bar" width; value boxes are narrower.
            field_width = (_EDU_DROPDOWN_W if isinstance(widget, QComboBox)
                           else _edu_field_width(widget))
        if isinstance(label, QLabel):
            label_widget = label
        else:
            label_widget = QLabel(str(label))
        label_widget.setTextFormat(Qt.TextFormat.RichText)
        label_widget.setFixedWidth(label_width)
        label_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        layout.addWidget(label_widget)
        widget.setFixedWidth(field_width)
        layout.addWidget(widget)

    def _fit_transform_param_widgets(self) -> None:
        """Keep Education transform fields as compact as Generator fields."""
        if not hasattr(self.editor, "transform_param_widgets"):
            return
        grid = self.editor.transform_params_layout
        grid.setColumnMinimumWidth(0, _EDU_LABEL_W)
        grid.setColumnMinimumWidth(1, _EDU_SPIN_W)
        grid.setColumnMinimumWidth(2, 58)
        grid.setColumnMinimumWidth(3, _EDU_SPIN_W)
        for column in range(4):
            grid.setColumnStretch(column, 0)
        grid.setColumnStretch(4, 1)
        for widget in self.editor.transform_param_widgets.values():
            widget.setFixedWidth(_EDU_SPIN_W)
            grid.setAlignment(widget, Qt.AlignmentFlag.AlignLeft)
        for idx in range(grid.count()):
            item = grid.itemAt(idx)
            widget = item.widget() if item is not None else None
            if not isinstance(widget, QLabel):
                continue
            widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            row, column, _row_span, _column_span = grid.getItemPosition(idx)
            if column == 0:
                widget.setFixedWidth(_EDU_LABEL_W)
                grid.setAlignment(
                    widget, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            elif column == 2:
                widget.setFixedWidth(34)
                grid.setAlignment(
                    widget, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._sync_transform_help_buttons()

    @staticmethod
    def _transform_help_key_from_widget_name(name: str) -> str:
        for suffix in ("_x", "_y"):
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name

    def _sync_transform_help_buttons(self) -> None:
        if not hasattr(self.editor, "transform_param_widgets"):
            return
        grid = self.editor.transform_params_layout
        for idx in reversed(range(grid.count())):
            item = grid.itemAt(idx)
            widget = item.widget() if item is not None else None
            if isinstance(widget, QPushButton) and widget.objectName() == "parameterHelpButton":
                grid.takeAt(idx)
                widget.setParent(None)
                widget.deleteLater()
        widget_to_name = {
            widget: self._transform_help_key_from_widget_name(name)
            for name, widget in self.editor.transform_param_widgets.items()
        }
        row_to_keys: dict[int, list[str]] = {}
        for idx in range(grid.count()):
            item = grid.itemAt(idx)
            widget = item.widget() if item is not None else None
            key = widget_to_name.get(widget)
            if key is None:
                continue
            row, _column, _row_span, _column_span = grid.getItemPosition(idx)
            row_to_keys.setdefault(row, [])
            if key not in row_to_keys[row]:
                row_to_keys[row].append(key)

        for row, keys in row_to_keys.items():
            visible_key = None
            for key in keys:
                if key.endswith("_minus"):
                    self._educational_hidden_help_button(key)
                    continue
                visible_key = key
                break
            if visible_key is not None:
                button = self._educational_help_button(visible_key)
                grid.addWidget(button, row, 4)
                grid.setAlignment(
                    button, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def _section_help_button(self, attr: str, tip: str, handler) -> QPushButton:
        button = make_parameter_help_button(tip, handler)
        setattr(self, attr, button)
        return button

    def _show_help_dialog(self, title: str, body: str, formula: str | None = None) -> None:
        show_help_dialog(self, title, body, formula)

    def _show_instance_layers_help(self) -> None:
        """Explain the Instance Layers tree and its per-row action buttons.

        Mirrors the Generator tab's instance-layers help, adapted to the 2D
        authoring terms (instances/components) and this tree's icons.
        """
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dialog = QDialog(self)
        dialog.setWindowTitle("Instance Layers help")
        dialog.setModal(True)
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        heading = QLabel("<b>Instance Layers</b>")
        heading.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(heading)
        intro = QLabel(
            "Each instance is a PORTAL landscape you author here in 2D, made of "
            "one or more components. Select a component to load it into the "
            "Component Parameters editor on the right, or use the row buttons:")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        def icon_row(pixmap_source, name: str, desc: str) -> None:
            row = QHBoxLayout()
            row.setSpacing(8)
            icon_lbl = QLabel()
            if isinstance(pixmap_source, str):
                icon_lbl.setText(pixmap_source)
                icon_lbl.setStyleSheet("font-weight: 700; font-size: 15px;")
            else:
                icon_lbl.setPixmap(pixmap_source.pixmap(18, 18))
            icon_lbl.setFixedWidth(22)
            icon_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
            row.addWidget(icon_lbl)
            text = QLabel(f"<b>{name}</b>: {desc}")
            text.setWordWrap(True)
            text.setTextFormat(Qt.TextFormat.RichText)
            row.addWidget(text, stretch=1)
            layout.addLayout(row)

        icon_row("+", "Add component",
                 "opens a new component editor; configure its fields, then "
                 "select + Add Component to store it in that instance.")
        icon_row(self._icon_table, "View component data table",
                 "opens a read-only table of the component's actual parameter "
                 "values for every dimension.")
        icon_row(self._icon_clone, "Clone",
                 "copies a component into an instance of your choice, as an "
                 "independent duplicate.")
        icon_row(self._icon_cross, "Delete",
                 "removes a component from its instance, or an entire instance "
                 "and all of its components.")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _show_educational_parameter_help(self, key: str) -> None:
        entries = {
            "base_form": (
                "Base form help",
                help_body("Base form", base_form_help_paragraphs()),
                None,
            ),
            "rotation": (
                "Rotation help",
                help_body("Rotation", [
                    "<b>Rotation</b>",
                    "Sets the component's internal coordinate orientation. Enter "
                    "the angle in degrees. Zero removes rotation-induced mixing; "
                    "the base form, anisotropy, transformations and component "
                    "competition also determine variable interactions.",
                ]),
                "z_k = T_k(R_k(x - c_k))",
            ),
            "c": (
                "c help",
                help_body("c", centre_help_paragraphs(educational=True)),
                None,
            ),
            "beta": (
                "Beta help",
                help_body("Beta", [
                    "<b>Beta (&beta;)</b>",
                    "Sets the component value at its centre. The lowest &beta; "
                    "sets the landscape's minimum when that centre is in the "
                    "domain. Higher-offset components can be hidden by the "
                    "lower envelope; they need not form visible local optima.",
                ]),
                None,
            ),
            "neutralization": (
                "Neutralization help",
                help_body("Neutralization", neutralization_help_paragraphs()),
                "|z_i| = r_ref",
            ),
            "Delta": (
                "Delta help",
                help_body("Delta", delta_help_paragraphs()),
                "Form A: rho_k,i,+/- = (Delta/d)/(kappa_bar_k r_ref^(2 p_k,i,+/-))",
            ),
            "r_ref": (
                "Reference displacement help",
                help_body("Reference displacement", r_ref_help_paragraphs()),
                "Form B: rho_k = Delta/(r_ref^2 sum_i kappa_bar_k,i^(1/p_k))^(p_k)",
            ),
            "kappa_plus": (
                "κ help",
                help_body("κ", kappa_help_paragraphs(educational=True)),
                None,
            ),
            "kappa_minus": (
                "κ⁻ help",
                help_body("κ⁻", [
                    f"{term('kappa_minus')} is the negative-direction anisotropy "
                    f"factor. It can be matched with or separated from {term('kappa_plus')}.",
                ]),
                None,
            ),
            "p_plus": (
                "p help",
                help_body("p", exponent_help_paragraphs(educational=True)),
                None,
            ),
            "p_minus": (
                "p⁻ help",
                help_body("p⁻", [
                    f"{term('p_minus')} is the negative-direction exponent for "
                    f"Base Form A. It can be matched with or separated from {term('p_plus')}.",
                ]),
                None,
            ),
            "p_single": (
                "p help",
                help_body("p", [
                    f"{term('p_single')} is the single exponent used by Base Form B.",
                ]),
                None,
            ),
            "transform_type": (
                "Transform type help",
                help_body("Transform type", transform_type_help_paragraphs()),
                "z' = T_n(...T_2(T_1(z)))",
            ),
            "mu_plus": (
                "μ help",
                help_body("μ", [
                    f"{term('mu_plus')} and {term('mu_minus')} control "
                    "transform perturbation strength on positive and negative sides.",
                ]),
                None,
            ),
            "mu_minus": (
                "μ⁻ help",
                help_body("μ⁻", [
                    f"{term('mu_minus')} controls transform strength on the "
                    "negative direction.",
                ]),
                None,
            ),
            "gamma_plus": (
                "γ help",
                help_body("γ", [
                    f"{term('gamma_plus')} and {term('gamma_minus')} "
                    "control how quickly the transform envelope ramps away from the centre "
                    "on positive and negative sides.",
                ]),
                None,
            ),
            "gamma_minus": (
                "γ⁻ help",
                help_body("γ⁻", [
                    f"{term('gamma_minus')} controls the negative-direction "
                    "envelope ramp.",
                ]),
                None,
            ),
            "omega_plus": (
                "ω help",
                help_body("ω", [
                    f"{term('omega_plus')} and {term('omega_minus')} "
                    "control spacing or frequency of transform oscillations on positive "
                    "and negative sides.",
                ]),
                None,
            ),
            "omega_minus": (
                "ω⁻ help",
                help_body("ω⁻", [
                    f"{term('omega_minus')} controls negative-direction "
                    "transform spacing or frequency.",
                ]),
                None,
            ),
            "omega1_plus": (
                "ω₁ help",
                help_body("ω₁", [
                    f"{term('omega1_plus')} and {term('omega1_minus')} "
                    "are the first Log-Sinusoidal log-space frequencies on positive "
                    "and negative sides. They set one oscillation rate in "
                    f"{term('sin')}({term('omega1')} log|a|), and work "
                    f"with {term('omega2_plus')} and {term('omega2_minus')} "
                    "to create multi-scale beating ruggedness.",
                ]),
                None,
            ),
            "omega1_minus": (
                "ω₁⁻ help",
                help_body("ω₁⁻", [
                    f"{term('omega1_minus')} controls the negative-side first "
                    "Log-Sinusoidal log-space frequency.",
                ]),
                None,
            ),
            "omega2_plus": (
                "ω₂ help",
                help_body("ω₂", [
                    f"{term('omega2_plus')} and {term('omega2_minus')} "
                    "are the second Log-Sinusoidal log-space frequencies on positive "
                    "and negative sides. They set the companion oscillation rate in "
                    f"{term('sin')}({term('omega2')} log|a|), and work "
                    f"with {term('omega1_plus')} and {term('omega1_minus')} "
                    "to create multi-scale beating ruggedness.",
                ]),
                None,
            ),
            "omega2_minus": (
                "ω₂⁻ help",
                help_body("ω₂⁻", [
                    f"{term('omega2_minus')} controls the negative-side second "
                    "Log-Sinusoidal log-space frequency.",
                ]),
                None,
            ),
            "sigma": (
                "σ help",
                help_body("σ", [
                    f"{term('sigma')} is the Log-Sinusoidal origin-smoothing scale. It "
                    "controls how the log-sinusoidal pattern is softened near the "
                    "component centre.",
                ]),
                None,
            ),
            "eta": (
                "η help",
                help_body("η", [
                    f"{term('eta')} is the Wavelet-Inspired extent scale. PORTAL uses "
                    "it with the selected frequency to set the spatial extent of "
                    "the wavelet envelope.",
                ]),
                None,
            ),
            "eps": (
                "eps help",
                help_body("eps", [
                    f"{term('eps')} is a numerical-stability constant used as a near-zero "
                    "guard in transform calculations.",
                ]),
                None,
            ),
            "mu": (
                "μ help",
                help_body("μ", [
                    f"{term('mu')} is the Radial Polynomial-Trigonometric radial "
                    "perturbation strength.",
                ]),
                None,
            ),
            "p": (
                "p help",
                help_body("p", [
                    f"{term('p')} is the radial polynomial power in the Radial "
                    "Polynomial-Trigonometric transform.",
                ]),
                None,
            ),
            "q": (
                "q help",
                help_body("q", [
                    f"{term('q')} is the radial trigonometric power in the Radial "
                    "Polynomial-Trigonometric transform.",
                ]),
                None,
            ),
            "omega": (
                "ω help",
                help_body("ω", [
                    f"{term('omega')} is the radial oscillation frequency in the Radial "
                    "Polynomial-Trigonometric transform.",
                ]),
                None,
            ),
        }
        # Transform inner parameters reuse the Generator's per-transform copy in
        # 2D/manual (educational) form: no randomisation modes, manual ± note.
        base = key[:-5] if key.endswith("_plus") else (
            key[:-6] if key.endswith("_minus") else key)
        tf_paras = transform_param_help_paragraphs(
            self.editor._current_transform_name(), base, educational=True)
        if tf_paras is not None:
            title = entries[key][0] if key in entries else f"{base} help"
            self._show_help_dialog(
                title, help_body(title.replace(" help", ""), tf_paras), None)
            return
        title, body, formula = entries[key]
        self._show_help_dialog(title, body, formula)

    def _show_baseline_help(self) -> None:
        body = help_body("Baseline", [
            "Baseline controls which PORTAL component form is used.",
            f"{term('Baseline A')} uses separate positive and negative exponent "
            f"vectors, {term('p_plus')} and {term('p_minus')}. "
            f"{term('Baseline B')} uses a single exponent value, "
            f"{term('p_single')}.",
        ] + formula_intro(
            "shows the two component forms used by the Education editor",
            f"{term('phi_k')} is component k, {term('beta_k')} is its vertical "
            f"offset, {term('rho_plus')}/{term('rho_minus')} are the "
            "neutralizing factors for Form A, and z is the transformed local "
            "coordinate",
        ))
        self._show_help_dialog(
            "Baseline help", body,
            "Form A: use + parameters when z_k,i >= 0, and - parameters when z_k,i < 0\n"
            "phi_k(x) = beta_k + sum_i rho_k,i,+/- kappa_k,i,+/- |z_k,i|^(2 p_k,i,+/-)\n"
            "Form B: phi_k(x) = beta_k + rho_k (sum_i (kappa_k,i,+/-)^(1/p_k) |z_k,i|^2)^(p_k)",
        )

    def _show_geometry_help(self) -> None:
        body = help_body("Geometry", [
            "Geometry describes where the component is placed and how its "
            "local coordinate system is oriented.",
            f"{term('c')} is the component centre. {term('beta')} shifts the "
            f"component value vertically. {term('Rotation')} changes the local "
            "coordinate system before the shape is evaluated.",
        ] + formula_intro(
            "shows the local coordinate used by the component",
            f"{term('x')} is the input point, {term('c')} is the centre, "
            f"{term('R')} is the rotation matrix, {term('T_k')} is the optional "
            f"transform sequence, and {term('z')} is the transformed local coordinate",
        ))
        self._show_help_dialog("Geometry help", body, "z_k = T_k(R_k(x - c_k))")

    def _show_neutralization_help(self) -> None:
        body = help_body("Neutralization", [
            "Neutralization calibrates the baseline scale using the exponents "
            "and sign-averaged anisotropy. "
            f"It computes {term('rho_plus')} and {term('rho_minus')} for "
            f"Baseline A, and {term('rho')} for Baseline B.",
            f"{term('Delta')} is the vertical reference scale; "
            f"{term('r_ref')} is the reference displacement in each internal coordinate. "
            "For Form A, the rise averaged over sign choices at this reference equals Delta. "
            "This does not guarantee equal basin exposure or optimizer difficulty. "
            "When neutralization is off, PORTAL uses "
            "neutralizing factors equal to 1.",
        ] + formula_intro(
            "shows the neutralizing factors implemented by the engine",
            f"{term('rho_plus')}/{term('rho_minus')} are the Form A "
            f"direction-specific factors, {term('rho')} is the Form B factor, "
            f"{term('Delta')} is the target rise, and {term('r_ref')} is the "
            "per-coordinate reference displacement",
        ))
        self._show_help_dialog(
            "Neutralization help", body,
            "Form A: rho_k,i,+/- = (Delta/d)/(kappa_bar_k r_ref^(2 p_k,i,+/-))\n"
            "Form B: rho_k = Delta/(r_ref^2 sum_i kappa_bar_k,i^(1/p_k))^(p_k)",
        )

    def _show_advanced_help(self) -> None:
        body = help_body("Advanced", [
            "Advanced parameters control the shape and directional scaling of "
            "the component.",
            f"{term('kappa_plus')} and {term('kappa_minus')} control "
            f"directional scaling. {term('p_plus')} and {term('p_minus')} "
            f"control exponent shape for Baseline A. For Baseline B, "
            f"{term('p_single')} is used instead.",
        ] + formula_intro(
            "shows how shape parameters enter the component formula",
            f"{term('kappa_plus')}/{term('kappa_minus')} set directional "
            f"anisotropy, {term('p_plus')}/{term('p_minus')} set Form A "
            f"curvature, and {term('p_single')} sets Form B curvature",
        ))
        self._show_help_dialog(
            "Advanced help", body,
            "Form A: phi_k(x) = beta_k + sum_i rho_k,i,+/- kappa_k,i,+/- |z_k,i|^(2 p_k,i,+/-)\n"
            "Form B: phi_k(x) = beta_k + rho_k (sum_i (kappa_k,i,+/-)^(1/p_k) |z_k,i|^2)^(p_k)",
        )

    def _make_parameter_section(self, title: str, rows: list[tuple],
                                two_column: bool = False,
                                help_button: QPushButton | None = None) -> QWidget:
        from PySide6.QtWidgets import QSizePolicy

        container = QFrame()
        container.setObjectName("parameterSection")
        # Size to content so a collapsed section shrinks to just its header.
        container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 8)
        outer.setSpacing(0)

        header_row = QWidget()
        header_layout = QHBoxLayout(header_row)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        header = QPushButton(f"v  {title}")
        header.setObjectName("sectionHeader")
        header.setCheckable(True)
        header.setChecked(True)
        header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        header_layout.addWidget(header)
        if help_button is not None:
            header_layout.addWidget(help_button)
        outer.addWidget(header_row)

        body = QWidget()
        grid = QGridLayout(body)
        grid.setContentsMargins(12, 2, 12, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        if two_column:
            # X and Y of each parameter share a row (like the transform fields).
            grid.setColumnStretch(4, 1)
            for r, (label_a, widget_a, label_b, widget_b) in enumerate(rows):
                self._add_labeled_widget(grid, r, 0, label_a, widget_a)
                if widget_b is not None:
                    self._add_labeled_widget(grid, r, 2, label_b, widget_b)
        else:
            # One field per row (single column) so the panel stays narrow.
            grid.setColumnStretch(2, 1)
            r = 0
            for label_a, widget_a, label_b, widget_b in rows:
                self._add_labeled_widget(grid, r, 0, label_a, widget_a)
                r += 1
                if widget_b is not None:
                    self._add_labeled_widget(grid, r, 0, label_b, widget_b)
                    r += 1
        outer.addWidget(body)

        def _toggle_section(checked, c=container, b=body, h=header, t=title):
            b.setVisible(checked)
            # Cap the whole card to the header height when collapsed so it
            # actually shrinks (size hints ignore a body that is only hidden).
            c.setMaximumHeight(16777215 if checked else h.sizeHint().height() + 14)
            h.setText(("v  " if checked else ">  ") + t)
            self._fit_editor()

        header.toggled.connect(_toggle_section)
        return container

    @staticmethod
    def _add_labeled_widget(grid: QGridLayout, row: int, column: int, label, widget: QWidget) -> None:
        if isinstance(label, QLabel):
            label_widget = label
        else:
            label_widget = QLabel(str(label))
        label_widget.setTextFormat(Qt.TextFormat.RichText)
        grid.addWidget(label_widget, row, column)
        grid.addWidget(widget, row, column + 1)

    def _sanitize_editor_labels(self, *_args) -> None:
        replacements = {
            "Beta (Î²):": "Beta (beta):",
            "Rotation (Â°):": "Rotation (deg):",
            "Îºâº X:": "Kappa+ X:",
            "Îºâº Y:": "Kappa+ Y:",
            "Îºâ» X:": "Kappa- X:",
            "Îºâ» Y:": "Kappa- Y:",
            "pâº X:": "p+ X:",
            "pâº Y:": "p+ Y:",
            "pâ» X:": "p- X:",
            "pâ» Y:": "p- Y:",
            "Î¼âº X:": "mu+ X:",
            "Î¼âº Y:": "mu+ Y:",
            "Î¼â» X:": "mu- X:",
            "Î¼â» Y:": "mu- Y:",
            "Î³âº X:": "gamma+ X:",
            "Î³âº Y:": "gamma+ Y:",
            "Î³â» X:": "gamma- X:",
            "Î³â» Y:": "gamma- Y:",
            "Ï‰âº X:": "omega+ X:",
            "Ï‰âº Y:": "omega+ Y:",
            "Ï‰â» X:": "omega- X:",
            "Ï‰â» Y:": "omega- Y:",
            "Ï‰1âº X:": "omega1+ X:",
            "Ï‰1âº Y:": "omega1+ Y:",
            "Ï‰1â» X:": "omega1- X:",
            "Ï‰1â» Y:": "omega1- Y:",
            "Ï‰2âº X:": "omega2+ X:",
            "Ï‰2âº Y:": "omega2+ Y:",
            "Ï‰2â» X:": "omega2- X:",
            "Ï‰2â» Y:": "omega2- Y:",
            "Ïƒ:": "sigma:",
            "Îµ:": "eps:",
            "Lâº X:": "L+ X:",
            "Lâº Y:": "L+ Y:",
            "Lâ» X:": "L- X:",
            "Lâ» Y:": "L- Y:",
            "Î·:": "eta:",
            "Î¼:": "mu:",
            "Ï‰:": "omega:",
        }
        for label in self.editor.findChildren(QLabel):
            clean = replacements.get(label.text())
            if clean is not None:
                label.setText(clean)

    def _make_transform_section_collapsible(self) -> None:
        """Restyle the transform panel as a Generator-style section."""
        panel = self.editor.transform_panel
        grid = panel.layout()
        panel.setObjectName("sectionCard")
        grid.setContentsMargins(10, 8, 10, 10)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        self.editor.transform_params_layout.setContentsMargins(0, 0, 0, 0)
        self.editor.transform_params_layout.setHorizontalSpacing(6)
        self.editor.transform_params_layout.setVerticalSpacing(6)
        self.editor.transform_combo.setFixedWidth(_EDU_DROPDOWN_W)
        self._configure_transform_combo_tooltips()
        self.editor.btn_add_transform.setStyleSheet("")
        self.editor.btn_add_transform.setMinimumHeight(34)
        header = None
        type_label = None
        for label in panel.findChildren(QLabel):
            if label.text() == "Transformation Configuration":
                header = label
            elif label.text() == "Type":
                type_label = label
        if type_label is None:
            type_label = QLabel("Type")
        if header is not None:
            header.hide()

        body = QWidget()
        body.setObjectName("parameterPane")
        # The transform section reads as one grey band like the Component
        # Parameter groups: the Type row, the μ/γ/ω rows and the Add Transform
        # button all share the shaded background with Generator-style padding.
        body.setStyleSheet(f"QWidget#parameterPane {{ background: {self._palette_bootstrap()['bg']}; }}")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(10, 6, 10, 6)
        body_layout.setSpacing(6)
        self.educational_transform_body = body

        title = QLabel("Transform Sequence")
        title.setStyleSheet(f"font-weight: 700; color: {self._palette_bootstrap()['heading']};")
        self._transform_section_title = title
        body_layout.addWidget(title)

        type_row = QHBoxLayout()
        type_row.setContentsMargins(0, 0, 0, 0)
        type_row.setSpacing(6)
        type_label.setFixedWidth(_EDU_LABEL_W)
        self.educational_transform_type_label = type_label
        type_row.addWidget(type_label)
        type_row.addWidget(self.editor.transform_combo)
        type_row.addWidget(self._educational_help_button("transform_type"))
        type_row.addStretch(1)
        body_layout.addLayout(type_row)

        # The per-axis grid stays as the hidden data source; the educational
        # combined-(x,y) rows are rebuilt into _edu_tf_rows on each type change.
        body_layout.addWidget(self.editor.transform_params_panel)
        self.editor.transform_params_panel.hide()
        self._edu_tf_rows = QWidget()
        self._edu_tf_rows_layout = QVBoxLayout(self._edu_tf_rows)
        self._edu_tf_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._edu_tf_rows_layout.setSpacing(6)
        body_layout.addWidget(self._edu_tf_rows)
        body_layout.addWidget(self.editor.btn_add_transform)
        body_layout.addWidget(self.editor.transform_list)
        self.editor.transform_combo.currentIndexChanged.connect(
            lambda *_: self._rebuild_edu_transform_rows())
        # Place transform controls inside the Component Parameters card.
        if hasattr(self, "educational_parameter_form"):
            self.educational_parameter_form.layout().addWidget(body)
            self.editor.transform_panel.hide()
        else:
            grid.addWidget(body, 0, 0, 1, 2)

        self.editor.btn_add_transform.setText("+ Add Transform")
        # Match the Generator's transform button: a plain (default) button, not
        # the big primary blue.
        self.editor.btn_add_transform.setProperty("variant", None)
        self._rebuild_edu_transform_rows()
        self._fit_editor()

    _TF_SYMBOLS = {
        "mu": "μ", "gamma": "γ", "omega": "ω", "omega1": "ω₁", "omega2": "ω₂",
        "eta": "η", "eps": "eps", "sigma": "σ", "p": "p", "q": "q", "L": "L",
    }
    # Internal numerical scalars hidden from the Education transform GUI.
    _EDU_HIDDEN_TF_SCALARS = {"sigma", "eps"}

    def _tf_symbol(self, transform_name: str, base: str) -> str:
        """Display symbol for a transform parameter, transform-aware where the
        user value differs from the effective one. For tensor interference the
        user sets the base amplitude μ₀ (the effective μ is scaled internally),
        so it is labelled μ₀ rather than μ."""
        if transform_name == "tensor_interference" and base == "mu":
            return "μ₀"
        return self._TF_SYMBOLS.get(base, base)

    def _make_edu_scalar_row(self, symbol: str, spin, help_key: str) -> QWidget:
        """A single-value transform row (no x/y), showing the spinbox directly."""
        row = QWidget()
        row.setObjectName("educationalParameterRow")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        label = QLabel(symbol)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setFixedWidth(_EDU_LABEL_W)
        rl.addWidget(label)
        spin.setFixedWidth(_edu_field_width(spin))
        rl.addWidget(spin)
        rl.addWidget(self._educational_help_button(help_key))
        rl.addStretch(1)
        return row

    def _rebuild_edu_transform_rows(self) -> None:
        """Rebuild the educational combined-(x,y) transform rows from the
        (hidden) per-axis spinboxes after a transform-type change."""
        if not hasattr(self, "_edu_tf_rows"):
            return
        lay = self._edu_tf_rows_layout
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                # Hide children before reparenting so they remain invisible until
                # deferred deletion.
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self._edu_tf_refreshers = []
        widgets = dict(getattr(self.editor, "transform_param_widgets", {}))
        pairs: dict = {}
        order: list = []
        scalars: list = []
        for key, w in widgets.items():
            for suf, slot in (("_plus_x", "px"), ("_plus_y", "py"),
                              ("_minus_x", "mx"), ("_minus_y", "my")):
                if key.endswith(suf):
                    base = key[: -len(suf)]
                    if base not in pairs:
                        pairs[base] = {}
                        order.append(base)
                    pairs[base][slot] = w
                    break
            else:
                scalars.append((key, w))
        tf_name = self.editor._current_transform_name()
        for base in order:
            d = pairs[base]
            if not all(k in d for k in ("px", "py", "mx", "my")):
                continue
            symbol = self._tf_symbol(tf_name, base)
            self._edu_coupled[base] = not (
                d["px"].value() != d["mx"].value()
                or d["py"].value() != d["my"].value())
            plus_row, minus_row = self._make_edu_pair_rows(
                base, symbol, d["px"], d["py"], d["mx"], d["my"],
                help_key=f"{base}_plus", refreshers=self._edu_tf_refreshers)
            lay.addWidget(plus_row)
            lay.addWidget(minus_row)
        for key, w in scalars:
            # Internal numerical-stability scalars (σ for Log-Sinusoidal, eps
            # for Wavelet/Radial) are not user-facing in the Education tab; the
            # spinboxes keep their defaults as the hidden data source.
            if key in self._EDU_HIDDEN_TF_SCALARS:
                continue
            lay.addWidget(self._make_edu_scalar_row(
                self._TF_SYMBOLS.get(key, key), w, help_key=key))
        # Defer sizing until the next event-loop turn so the new rows have
        # completed their polish and layout pass.
        QtCore.QTimer.singleShot(0, self._fit_editor)
        self._fit_editor()

    def _configure_transform_combo_tooltips(self) -> None:
        """Show full transform names on hover when combo entries are truncated."""
        combo = self.editor.transform_combo
        for idx in range(combo.count()):
            text = combo.itemText(idx)
            combo.setItemData(idx, text, Qt.ItemDataRole.ToolTipRole)

        view = combo.view()
        if view is not None:
            view.setMouseTracking(True)

        def _sync_from_index(index: int) -> None:
            if 0 <= index < combo.count():
                combo.setToolTip(combo.itemText(index))

        combo.highlighted.connect(_sync_from_index)
        combo.currentIndexChanged.connect(_sync_from_index)
        _sync_from_index(combo.currentIndex())

    def _build_parameter_panel(self) -> QWidget:
        panel = QWidget()
        self.parameter_panel = panel
        panel.setObjectName("parameterPanel")
        # White editor column like the Generator, so the parameter card sits on
        # one continuous white surface instead of a grey backdrop.
        panel.setStyleSheet(
            "QWidget#parameterPanel { background: transparent;"
            " border: none; border-radius: 0; }")
        panel.setMinimumWidth(_PARAM_PANEL_MIN_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 6, 2, 6)
        layout.setSpacing(6)

        # Single-block 2D instances carry no weight, so there is no per-block
        # editor; the placeholder keeps the visibility wiring simple.
        self.selected_block_editor = QWidget()
        self.selected_block_editor.setVisible(False)
        layout.addWidget(self.selected_block_editor)

        self.component_empty_surface = QFrame()
        self.component_empty_surface.setObjectName("componentEmptySurface")
        p = self._palette_bootstrap()
        self.component_empty_surface.setStyleSheet(
            "QFrame#componentEmptySurface {"
            f" background: {p['surface']};"
            " border: none;"
            " border-radius: 8px;"
            "}")
        empty_layout = QVBoxLayout(self.component_empty_surface)
        empty_layout.setContentsMargins(10, 10, 10, 10)
        empty_layout.setSpacing(0)

        self.active_block_hint = QLabel(self._default_hint_html())
        self.active_block_hint.setWordWrap(True)
        self.active_block_hint.setObjectName("hintLabel")
        # Centre the hint so it fills the space the editor would occupy.
        self.active_block_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(self.active_block_hint, stretch=1)
        layout.addWidget(self.component_empty_surface, stretch=1)

        # Scroll the parameter form so a tall component (many sections +
        # transforms) never forces the whole window taller than the screen.
        # The action buttons below stay pinned and always visible.
        param_scroll = QScrollArea()
        param_scroll.setObjectName("paramScroll")
        # Let the form keep its natural content height and stay pinned to the
        # top: collapsing a section then frees space at the bottom instead of
        # stretching the cards apart. A resize event keeps its width synced to
        # the viewport (see eventFilter), and it scrolls vertically when tall.
        param_scroll.setWidgetResizable(False)
        param_scroll.setFrameShape(QFrame.Shape.NoFrame)
        param_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        param_scroll.setAlignment(Qt.AlignmentFlag.AlignTop)
        param_scroll.setWidget(self.editor)
        param_scroll.viewport().installEventFilter(self)
        # White viewport so the area below the (content-height) editor reads as
        # the same continuous white surface, not a grey gap. The selector is
        # scoped to the viewport by objectName: a bare "background" would bleed
        # onto every descendant row and wash the grey parameter bands white.
        param_scroll.viewport().setObjectName("paramScrollViewport")
        param_scroll.viewport().setStyleSheet(
            f"QWidget#paramScrollViewport {{ background: {self._palette_bootstrap()['surface']}; }}")
        self._param_scroll = param_scroll
        layout.addWidget(param_scroll, stretch=1)

        self.add_btn = QPushButton("+ Add Component")
        self.add_btn.setProperty("variant", "primary")
        self.add_btn.clicked.connect(self.on_add_component_clicked)
        self.add_btn.setEnabled(False)
        self.add_btn.setToolTip("Add an instance first, then add components to the selected instance.")
        self.apply_changes_btn = QPushButton("Apply Changes")
        self.apply_changes_btn.setProperty("variant", "primary")
        self.apply_changes_btn.clicked.connect(self.on_apply_changes_clicked)
        self.apply_changes_btn.setEnabled(False)
        self.new_component_btn = QPushButton("Reset Component Parameters")
        self.new_component_btn.clicked.connect(self.on_reset_parameters_clicked)
        self.new_component_btn.setText("Reset Parameters")
        for btn in (self.add_btn, self.apply_changes_btn, self.new_component_btn):
            btn.setMinimumHeight(34)
        component_actions = QWidget()
        self.component_actions = component_actions
        component_actions_col = QVBoxLayout(component_actions)
        component_actions_col.setContentsMargins(0, 0, 0, 0)
        component_actions_col.setSpacing(8)
        component_actions_top = QHBoxLayout()
        component_actions_top.setContentsMargins(0, 0, 0, 0)
        component_actions_top.setSpacing(8)
        component_actions_top.addWidget(self.add_btn, 1)
        component_actions_top.addWidget(self.apply_changes_btn, 1)
        component_actions_col.addLayout(component_actions_top)
        component_actions_col.addWidget(self.new_component_btn)
        layout.addWidget(component_actions)
        self._set_component_editor_enabled(False)
        return panel

    @staticmethod
    def _default_hint_html() -> str:
        return ("<i>Create an instance with <b>+ Add Instance</b> below, "
                "or click an existing instance in the tree, before adding components.</i>")

    @staticmethod
    def _add_component_hint_html() -> str:
        """Shown when an instance exists but no component is open for editing:
        points the user at an existing component or the per-instance '+'."""
        return ("<i>Click a component in <b>Instance Layers</b> to edit it, "
                "or create one with the <b>+</b> next to your instance.</i>")

    def _set_component_editor_enabled(self, active: bool) -> None:
        help_buttons = set(getattr(self, "parameter_help_buttons", {}).values())
        lockable_types = (
            QAbstractSpinBox, QCheckBox, QComboBox, QLineEdit,
            QListWidget, QPushButton,
        )
        for widget in self.editor.findChildren(QWidget):
            if widget in help_buttons:
                continue
            if isinstance(widget, lockable_types):
                widget.setEnabled(active)
        self.add_btn.setEnabled(active)
        self.new_component_btn.setEnabled(active)
        if not active:
            # Point at the per-instance "+" once an instance exists; otherwise the
            # generic "add an instance first" hint.
            self.active_block_hint.setText(
                self._add_component_hint_html() if self.inventory.staged_blocks
                else self._default_hint_html())
        self.active_block_hint.setVisible(not active)
        if hasattr(self, "component_empty_surface"):
            self.component_empty_surface.setVisible(not active)
        # Hide the component editor until an instance/component is active. Keep
        # the disabled action buttons visible so users can see what becomes
        # available after selecting or creating a component.
        for section in (
            getattr(self, "component_parameters_card", None),
            getattr(self, "_param_scroll", None),
        ):
            if section is not None:
                section.setVisible(active)
        if hasattr(self, "component_actions"):
            self.component_actions.setVisible(True)
        if not active:
            self.apply_changes_btn.setEnabled(False)
            self.selected_block_editor.setVisible(False)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent

        if (
            hasattr(self, "_param_scroll")
            and obj is self._param_scroll.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            self._fit_editor()
        return super().eventFilter(obj, event)

    def _fit_editor(self) -> None:
        """Size the parameter form to the panel width and its own content height.

        With a non-resizable scroll area this keeps the form pinned to the top:
        collapsing a section frees space at the bottom instead of stretching the
        cards apart, while the width still tracks the panel.
        """
        if not hasattr(self, "_param_scroll"):
            return
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QApplication
        # Process pending layout requests so size hints reflect row visibility
        # through the nested parameter bands, pane, and component card.
        for _ in range(3):
            QApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
        try:
            width = max(self._param_scroll.viewport().width(), 1)
            # Sum the visible cards' heights, clamped by any collapse cap, so a
            # collapsed section truly frees its space (sizeHint alone ignores the
            # maximum-height cap and the hidden right panel is much taller).
            ll = self.editor.left_layout
            height = 24
            for i in range(ll.count()):
                w = ll.itemAt(i).widget()
                if w is not None and w.isVisible():
                    height += min(w.sizeHint().height(), w.maximumHeight()) + ll.spacing()
            self.editor.setFixedWidth(width)
            self.editor.resize(width, height)
        except RuntimeError:
            return   # a queued resize fired while the widget was being torn down

    def _build_visualization_panel(self) -> QWidget:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        panel = QWidget()
        panel.setObjectName("visualizationPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 8, 0)

        plot_frame = QFrame()
        plot_frame.setObjectName("plotFrame")
        plot_frame.setFrameShape(QFrame.Shape.StyledPanel)
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(8, 8, 8, 8)

        self.preview_title = QLabel("")
        self.preview_title.hide()

        self.visualization_tabs = QTabWidget()

        # Tab 1: surface (top) and contour (bottom) of the current landscape,
        # stacked together so both are visible at once.
        sc_tab = QWidget()
        sc_layout = QVBoxLayout(sc_tab)
        sc_layout.setContentsMargins(0, 0, 0, 0)
        self.editor.canvas.setParent(sc_tab)
        self.current_surface_panel = self._build_plot_panel(
            self.editor.canvas,
            self.editor.fig,
            default_name="current_landscape_surface.pdf",
            save_tip="Save current landscape surface as PDF",
            reset_tip="Reset current landscape surface view",
            reset_handler=self._reset_current_surface_view,
            save_attr="current_surface_save_pdf_btn",
            reset_attr="current_surface_reset_view_btn",
        )
        sc_layout.addWidget(self.current_surface_panel, stretch=1)
        self.contour_fig = Figure(figsize=(8, 4))
        self.contour_canvas = FigureCanvas(self.contour_fig)
        self.current_contour_panel = self._build_plot_panel(
            self.contour_canvas,
            self.contour_fig,
            default_name="current_landscape_contour.pdf",
            save_tip="Save current landscape contour as PDF",
            save_attr="current_contour_save_pdf_btn",
        )
        sc_layout.addWidget(self.current_contour_panel, stretch=1)
        self.current_landscape_hint = QLabel(
            "Build a component or instance first, then select it here to visualize its landscape."
        )
        self.current_landscape_hint.setObjectName("mutedLabel")
        self.current_landscape_hint.setWordWrap(True)
        self.current_landscape_hint.setAlignment(Qt.AlignCenter)
        sc_layout.addWidget(self.current_landscape_hint, stretch=1)
        self.visualization_tabs.addTab(sc_tab, "Current Landscape")

        # Tabs 2 & 3: every component / every block rendered two-per-row, as
        # surface or contour.
        self._cmp = {}
        self.visualization_tabs.addTab(self._build_comparison_tab("components"), "Visualize Components")
        self.visualization_tabs.addTab(self._build_comparison_tab("blocks"), "Visualize Instances")
        self.visualization_tabs.currentChanged.connect(self._on_viz_tab_changed)

        plot_layout.addWidget(self.visualization_tabs)
        layout.addWidget(plot_frame, stretch=1)
        return panel

    def _build_plot_panel(self, canvas: QWidget, fig, *, default_name: str,
                          save_tip: str, reset_tip: str = "",
                          reset_handler=None,
                          save_attr: str | None = None,
                          reset_attr: str | None = None,
                          title_widget: QWidget | None = None) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(4)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        if title_widget is not None:
            controls.addWidget(title_widget)
        controls.addStretch(1)
        save_btn = make_icon_button(
            make_save_icon(), save_tip,
            lambda f=fig, name=default_name: self._save_figure_pdf(f, name),
            size=26, object_name="plotActionBtn")
        controls.addWidget(save_btn)
        reset_btn = None
        if reset_handler is not None:
            reset_btn = make_icon_button(
                make_reset_icon(), reset_tip, reset_handler,
                size=26, object_name="plotActionBtn")
            controls.addWidget(reset_btn)
        panel_layout.addLayout(controls)
        panel_layout.addWidget(canvas, stretch=1)

        if save_attr is not None:
            setattr(self, save_attr, save_btn)
        if reset_attr is not None and reset_btn is not None:
            setattr(self, reset_attr, reset_btn)
        return panel

    @staticmethod
    def _safe_pdf_name(label: str) -> str:
        return figure_io.safe_pdf_name(label)

    def _save_figure_pdf(self, fig, default_name: str) -> None:
        figure_io.save_figure_pdf(self, fig, default_name)

    @staticmethod
    def _capture_axes_state(ax):
        if ax is None:
            return None
        state = {
            "xlim": tuple(ax.get_xlim()),
            "ylim": tuple(ax.get_ylim()),
        }
        if hasattr(ax, "get_zlim"):
            state["zlim"] = tuple(ax.get_zlim())
        if hasattr(ax, "elev") and hasattr(ax, "azim"):
            state["view"] = (ax.elev, ax.azim)
        return state

    @staticmethod
    def _restore_axes_state(ax, state) -> None:
        if ax is None or state is None:
            return
        ax.set_xlim(*state["xlim"])
        ax.set_ylim(*state["ylim"])
        if "zlim" in state and hasattr(ax, "set_zlim"):
            ax.set_zlim(*state["zlim"])
        if "view" in state and hasattr(ax, "view_init"):
            ax.view_init(elev=state["view"][0], azim=state["view"][1])

    def _show_current_landscape_hint(self) -> None:
        if not hasattr(self, "current_landscape_hint"):
            return
        self.editor.fig.clear()
        self.contour_fig.clear()
        self.current_surface_panel.hide()
        self.current_contour_panel.hide()
        self.current_landscape_hint.show()
        self.editor.canvas.draw_idle()
        self.contour_canvas.draw_idle()

    def _show_current_landscape_plots(self) -> None:
        self.current_landscape_hint.hide()
        self.current_surface_panel.show()
        self.current_contour_panel.show()

    def _reset_current_surface_view(self) -> None:
        if not self.editor.fig.axes:
            return
        self._restore_axes_state(
            self.editor.fig.axes[0],
            getattr(self, "_current_surface_home_state", None),
        )
        self.editor.canvas.draw_idle()

    def _reset_current_contour_view(self) -> None:
        if not self.contour_fig.axes:
            return
        self._restore_axes_state(
            self.contour_fig.axes[0],
            getattr(self, "_current_contour_home_state", None),
        )
        self.contour_canvas.draw_idle()

    def _build_comparison_tab(self, kind: str) -> QWidget:
        """Build a 'compare all' tab for ``kind`` ('blocks' or 'components')."""
        tab = QWidget()
        v = QVBoxLayout(tab)
        v.setContentsMargins(0, 6, 0, 0)
        v.setSpacing(6)

        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(2, 0, 2, 0)
        toggle_row.addWidget(QLabel("View:"))
        switch = SegmentedSwitch(["Surface", "Contour"])
        switch.setFixedSize(190, 30)
        switch.valueChanged.connect(lambda _v, k=kind: self._refresh_comparison_grid(k))
        toggle_row.addWidget(switch)
        toggle_row.addStretch(1)
        v.addLayout(toggle_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        contents = QWidget()
        outer = QVBoxLayout(contents)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(0)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        # Two equal columns always, so a single item stays half-width (not full),
        # and a trailing stretch keeps the cells compact instead of inflating them.
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        outer.addLayout(grid)
        outer.addStretch(1)
        scroll.setWidget(contents)
        v.addWidget(scroll, stretch=1)

        self._cmp[kind] = {"switch": switch, "grid": grid}
        return tab

    def _on_viz_tab_changed(self, index: int) -> None:
        text = self.visualization_tabs.tabText(index)
        if text == "Visualize Instances":
            self._refresh_comparison_grid("blocks")
        elif text == "Visualize Components":
            self._refresh_comparison_grid("components")

    def _refresh_comparison_grid(self, kind: str = "blocks") -> None:
        """Render every staged block/component 2-per-row as surface or contour."""
        panel = getattr(self, "_cmp", {}).get(kind)
        if panel is None:
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        grid = panel["grid"]
        while grid.count():
            item = grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if kind == "components":
            items = list(self.inventory.staged_components)
            empty_msg = "Add one or more components to compare them here."
            captions = [f"{c.label}  (Baseline {c.baseline})" for c in items]
            landscapes = lambda c, X: self._compute_component_landscape(c, X)
        else:
            items = [
                (idx, b)
                for idx, b in enumerate(self.inventory.staged_blocks)
                if b.components
            ]
            empty_msg = "Add components to one or more instances to visualize them here."
            captions = [
                f"Instance_{idx + 1}  ({len(b.components)} "
                f"{'component' if len(b.components) == 1 else 'components'})"
                for idx, b in items
            ]
            landscapes = lambda b, X: self._compute_block_landscape(b.components, X)

        if not items:
            empty = QLabel(empty_msg)
            empty.setObjectName("mutedLabel")
            empty.setAlignment(Qt.AlignCenter)
            grid.addWidget(empty, 0, 0)
            return

        surface_mode = panel["switch"].is_first()
        palette = getattr(self, "_palette", LIGHT)
        Xg, Yg, X = self._grid_2d(n=70)
        for i, item in enumerate(items):
            target = item[1] if kind != "components" else item
            Z = landscapes(target, X).reshape(Xg.shape)
            cell = QFrame()
            cell.setObjectName("plotFrame")
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(6, 6, 6, 6)
            caption = QLabel(captions[i])
            caption.setStyleSheet("font-weight: 700;")
            fig = Figure(figsize=(4, 3))
            if surface_mode:
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_surface(Xg, Yg, Z, cmap="jet")
                ax.view_init(elev=30, azim=225)
            else:
                ax = fig.add_subplot(111)
                ax.set_aspect("equal")
                ax.contourf(Xg, Yg, Z, levels=20, cmap="jet")
                ax.set_anchor("C")  # keep the square contour centred
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            style_compact_axes(ax)
            fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)
            style_figure(fig, palette)
            canvas = FigureCanvas(fig)
            # Fixed height -> every cell is the same compact size regardless of
            # how many items there are (1, 2, or many).
            canvas.setFixedHeight(360)
            home_state = self._capture_axes_state(ax)
            reset_handler = None
            if surface_mode:
                reset_handler = (
                    lambda _checked=False, a=ax, state=home_state, c=canvas:
                    (self._restore_axes_state(a, state), c.draw_idle())
                )
            plot_panel = self._build_plot_panel(
                canvas,
                fig,
                default_name=self._safe_pdf_name(captions[i]),
                save_tip=f"Save {captions[i].split('  ')[0]} figure as PDF",
                reset_tip=f"Reset {captions[i].split('  ')[0]} figure view",
                reset_handler=reset_handler,
                title_widget=caption,
            )
            cell_layout.addWidget(plot_panel)
            row, col = divmod(i, 2)
            grid.addWidget(cell, row, col)

    def _build_project_panel(self) -> QWidget:
        panel = QWidget()
        self.project_panel = panel
        panel.setObjectName("projectPanel")
        p = self._palette_bootstrap()
        panel.setStyleSheet(
            f"QWidget#projectPanel {{ background: {p['surface']};"
            f" border: 1px solid {p['card_border']}; border-radius: 8px; }}"
            # Same scoped field-color rule as the editor shell above, so
            # parameter fields under the project panel match Generator's.
            f" QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ color: {p['field_text_dark']}; }}"
            f" QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,"
            f" QDoubleSpinBox:disabled {{ color: {p['hint']}; }}"
            f" QPushButton[variant=\"primary\"]:disabled {{"
            f"   background: {p['field_disabled_primary_bg']}; color: {p['hint']};"
            f"   border: 1px solid {p['field_disabled_primary_border']};"
            f" }}"
        )
        panel.setMinimumWidth(_SIDE_PANEL_MIN_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Inner content only: the outer projectPanel is the single visible window.
        tree_section = QFrame()
        self.instance_layers_card = tree_section
        tree_section.setObjectName("sectionCard")
        tree_section.setStyleSheet(
            "QFrame#sectionCard {"
            "  background: transparent;"
            "  border: none;"
            "  border-radius: 0;"
            "}")
        tree_section.setGraphicsEffect(None)
        tv = QVBoxLayout(tree_section)
        tv.setContentsMargins(10, 8, 10, 10)
        tv.setSpacing(6)
        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        tree_label = QLabel("Instance Layers")
        # 13pt matches the Generator tab's "Instance Layers" (panelTitle) header.
        tree_label.setStyleSheet("font-weight: 700; font-size: 13pt;")
        header_row.addWidget(tree_label)
        self.instance_layers_help_btn = make_parameter_help_button(
            "Explain Instance Layers", self._show_instance_layers_help)
        header_row.addWidget(self.instance_layers_help_btn)
        header_row.addStretch(1)
        tv.addLayout(header_row)
        self.project_tree = QTreeWidget()
        self.project_tree.setHeaderHidden(True)
        # Drop the theme's bordered "inner window" box and enlarge the font.
        self.project_tree.setFrameShape(QFrame.Shape.NoFrame)
        self.project_tree.setStyleSheet(
            "QTreeWidget { background: transparent; border: none; padding: 0;"
            " font-size: 14px; }")
        # Column 0: checkbox + icon + label. Column 1: per-row action buttons.
        self.project_tree.setColumnCount(2)
        header = self.project_tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.project_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.project_tree.setIndentation(18)
        self.project_tree.itemSelectionChanged.connect(self.on_project_tree_selection_changed)
        self.project_tree.currentItemChanged.connect(self.on_project_tree_current_item_changed)
        tv.addWidget(self.project_tree)

        self.tree_empty_hint = QLabel(
            "<i>No instances yet. Create one with <b>+ Add Instance</b> below "
            "to start authoring components.</i>"
        )
        self.tree_empty_hint.setWordWrap(True)
        self.tree_empty_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.tree_empty_hint.setObjectName("hintLabel")
        self.tree_empty_hint.setStyleSheet("padding: 24px;")
        tv.addWidget(self.tree_empty_hint, stretch=1)

        # Add Instance sits at the bottom of the Instance Layers, above the
        # Benchmark / Save row.
        self.add_block_btn = QPushButton("+ Add Instance")
        self.add_block_btn.setProperty("variant", "primary")
        self.add_block_btn.clicked.connect(self.on_add_block_clicked)
        self.add_block_btn.setMinimumHeight(34)
        tv.addWidget(self.add_block_btn)

        # Cross-view navigation: send the user to the Benchmarking view to run
        # algorithms on whatever they've built here. Wired by ``main_window``.
        self.try_algorithm_btn = QPushButton("Benchmark What You Built")
        self.try_algorithm_btn.setProperty("variant", "primary")
        self.try_algorithm_btn.setMinimumHeight(32)
        self.try_algorithm_btn.setEnabled(False)
        self.try_algorithm_btn.setToolTip(
            "Switch to the Benchmarking view to run algorithms on your blocks.")
        instance_actions = QWidget()
        instance_actions.setObjectName("instanceActionsRow")
        instance_actions.setStyleSheet(
            "QWidget#instanceActionsRow { background: transparent; }")
        instance_actions_row = QHBoxLayout(instance_actions)
        instance_actions_row.setContentsMargins(0, 0, 0, 0)
        instance_actions_row.setSpacing(8)
        instance_actions_row.addWidget(self.try_algorithm_btn, 1)
        tv.addWidget(instance_actions)
        layout.addWidget(tree_section, stretch=2)
        return panel

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 2, 8, 2)
        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        self.footer_counts_label = QLabel()
        layout.addWidget(self.footer_counts_label)
        return bar

    def _build_compatibility_widgets(self) -> None:
        """Create hidden controls referenced by component and block handlers."""
        self.components_table = QTableWidget(0, 4)
        self.components_table.setHorizontalHeaderLabels(["Select", "Label", "Baseline", "Centre"])
        self.components_table.setColumnWidth(0, 60)
        self.components_table.hide()

        self.blocks_list = QListWidget()
        self.blocks_list.hide()

        self.component_tree = QTreeWidget()
        self.component_tree.setHeaderHidden(True)
        self.component_tree.setColumnCount(2)
        self.component_tree.setRootIsDecorated(False)
        self.component_tree.itemSelectionChanged.connect(self.on_component_tree_selection_changed)
        self.component_tree.hide()

        self.build_block_btn = QPushButton("Build Block from Selected Components")
        self.build_block_btn.setProperty("variant", "primary")
        self.build_block_btn.clicked.connect(self.on_build_block_clicked)
        self.build_block_btn.hide()

    # -------------------- Component actions --------------------
    def _mark_instance_dirty(self) -> None:
        self._active_composite_idx = None
        self._prepared_block_instances.clear()

    def _prepared_block_instance(self, block: BlockSpec) -> dict:
        """Freeze one core composition for both preview and export."""
        key = id(block)
        if key not in self._prepared_block_instances:
            self._prepared_block_instances[key] = assemble_composite(
                [copy.deepcopy(block)])
        return self._prepared_block_instances[key]

    def on_add_component_clicked(self) -> None:
        block_idx = self._current_block_index()
        if block_idx is None:
            QMessageBox.information(
                self,
                "Add component",
                "Add or select a block before adding components.",
            )
            self.add_btn.setEnabled(False)
            return
        self.editor._refresh_params_to_controller()
        cfg = self.editor.controller.build_cfg()
        block = self.inventory.staged_blocks[block_idx]
        spec = ComponentSpec(
            # Match the Generator: component<instance#>_<component#>.
            label=f"component{block_idx + 1}_{len(block.components) + 1}",
            baseline=str(cfg.get("baseline", "A")),
            rotation_deg=float(cfg.get("rotation_deg", 0.0)),
            neutralize=dict(cfg.get("neutralize", {})),
            params=dict(cfg.get("params", {})),
            T_seq=list(cfg.get("T_seq", [])),
        )
        self.inventory.add_component(spec)
        self.inventory.staged_blocks[block_idx].components.append(spec)
        self._mark_instance_dirty()
        self._refresh_components_table()
        self._refresh_project_tree()
        # Mirror the Generator: adding a component returns the middle panel to the
        # instance view. The component editor disappears and the centred hint
        # invites the user to click a component (to edit) or add another with the
        # per-instance "+". Selecting the instance row drives that state and plots
        # the instance landscape including the component just added.
        self._active_block_idx = block_idx
        self._component_apply_idx = None
        self._editing_component_idx = None
        self._set_component_edit_state(None)
        block_item = self.project_tree.topLevelItem(block_idx)
        if block_item is not None:
            self.project_tree.setCurrentItem(block_item)
        self._set_component_editor_enabled(False)

    def on_reset_parameters_clicked(self) -> None:
        """The 'Reset Parameters' button: reset the form, then confirm."""
        self.on_create_new_component_clicked()
        QMessageBox.information(
            self, "Reset Parameters", "Parameters have been reset.")

    def on_create_new_component_clicked(self) -> None:
        self.editor.reset_editor()
        # Education default: neutralization on.
        self.editor.neutral_enable.setChecked(True)
        self._sync_edu_from_editor()
        self._transform_apply_idx = None
        # A fresh component has nothing to apply yet: Apply Changes is disabled
        # (greyed) until an existing component is selected for editing.
        self._editing_component_idx = None
        self._component_apply_idx = None
        self.apply_changes_btn.setEnabled(False)
        self._set_component_edit_state(None)
        self._refresh_transform_rows()
        self._schedule_live_plot()

    def _on_new_component_for_block(self, block_idx: int) -> None:
        """The per-instance '+' action: focus that instance and start a fresh
        component form for it."""
        self._select_project_block(block_idx)
        self.on_create_new_component_clicked()

    def on_apply_changes_clicked(self) -> None:
        if (
            self._transform_apply_idx is not None
            and 0 <= self._transform_apply_idx < len(self.editor.controller.state.T_seq)
        ):
            if self._save_transform_row(self._transform_apply_idx):
                self._notify_changes_applied()
            return

        if (
            self._block_apply_idx is not None
            and 0 <= self._block_apply_idx < len(self.inventory.staged_blocks)
        ):
            block = self.inventory.staged_blocks[self._block_apply_idx]
            self._mark_instance_dirty()
            self._refresh_blocks_list()
            self._refresh_project_tree()
            self._select_project_block(self._block_apply_idx)
            self.status_label.setText(f"Status: Applied changes to {block.label}")
            self._notify_changes_applied()
            return

        if (
            self._component_apply_idx is not None
            and 0 <= self._component_apply_idx < len(self.inventory.staged_components)
        ):
            self._on_save_component(self._component_apply_idx)
            self._loaded_cfg_repr = self._editor_cfg_repr()   # saved state matches the editor
            self._notify_changes_applied()

    def _notify_changes_applied(self) -> None:
        """Confirm a successful Apply Changes so the user knows it worked."""
        QMessageBox.information(
            self, "Apply Changes", "The changes have been applied.")

    # -------------------- Per-node actions (edit / save / delete) --------------------
    def _load_component_into_editor(self, idx: int) -> None:
        """Pure data-load helper — replays a staged component into the editor.

        No edit-state side effect. Used by both the row-selection path and
        the pencil click; only the pencil sets _editing_component_idx and
        pushes the row's pencil button.
        """
        if not (0 <= idx < len(self.inventory.staged_components)):
            return
        spec = self.inventory.staged_components[idx]
        self.editor.load_component_config(
            {
                "baseline": spec.baseline,
                "rotation_deg": spec.rotation_deg,
                "neutralize": spec.neutralize,
                "params": spec.params,
                "T_seq": spec.T_seq,
            }
        )
        self._sync_edu_from_editor()
        self._refresh_transform_rows()
        self._schedule_live_plot()

    def _on_edit_component(self, idx: int) -> None:
        """Pencil click: load the component AND enter edit mode."""
        if not (0 <= idx < len(self.inventory.staged_components)):
            return
        self._load_component_into_editor(idx)
        self._editing_component_idx = idx
        self._set_component_edit_state(idx)
        # Switching to component edit releases any transform edit state — the
        # transform list is about to be rewritten by load_component_config.
        self._editing_transform_idx = None
        self._transform_apply_idx = None
        self._set_transform_edit_state(None)
        spec = self.inventory.staged_components[idx]
        self.status_label.setText(f"Status: Editing {spec.label}")

    def _on_save_component(self, idx: int) -> None:
        """Write the current editor state back into component ``idx`` in place.

        Mutating the existing spec (rather than replacing it) means blocks that
        already reference this component see the change too.
        """
        if not (0 <= idx < len(self.inventory.staged_components)):
            return
        self.editor._refresh_params_to_controller()
        cfg = self.editor.controller.build_cfg()
        spec = self.inventory.staged_components[idx]
        spec.baseline = str(cfg.get("baseline", "A"))
        spec.rotation_deg = float(cfg.get("rotation_deg", 0.0))
        spec.neutralize = dict(cfg.get("neutralize", {}))
        spec.params = dict(cfg.get("params", {}))
        spec.T_seq = list(cfg.get("T_seq", []))
        self._mark_instance_dirty()
        # Exit edit mode.
        self._editing_component_idx = None
        self._refresh_components_table()
        self._refresh_project_tree()
        self._set_component_edit_state(None)
        self.status_label.setText(f"Status: Saved {spec.label}")

    def _reindex_component_labels(self) -> None:
        """Renumber every component label to ``component<block#>_<comp#>`` so
        deletions never leave gaps or duplicate numbers."""
        for bi, block in enumerate(self.inventory.staged_blocks):
            for ci, comp in enumerate(block.components):
                comp.label = f"component{bi + 1}_{ci + 1}"

    def _on_delete_node(self, kind: str, idx: int) -> None:
        if kind == "component" and 0 <= idx < len(self.inventory.staged_components):
            removed = self.inventory.staged_components.pop(idx)
            for block in self.inventory.staged_blocks:
                block.components = [comp for comp in block.components if comp is not removed]
            self._mark_instance_dirty()
            if self._component_apply_idx == idx:
                self._component_apply_idx = None
                self.apply_changes_btn.setEnabled(self._block_apply_idx is not None)
            elif self._component_apply_idx is not None and self._component_apply_idx > idx:
                self._component_apply_idx -= 1
            # Adjust the edit cursor: release if it was this row, shift down if later.
            if self._editing_component_idx == idx:
                self._editing_component_idx = None
            elif (self._editing_component_idx is not None
                  and self._editing_component_idx > idx):
                self._editing_component_idx -= 1
        elif kind == "block" and 0 <= idx < len(self.inventory.staged_blocks):
            self.inventory.staged_blocks.pop(idx)
            self._mark_instance_dirty()
            if self._block_apply_idx == idx:
                self._block_apply_idx = None
                self.apply_changes_btn.setEnabled(self._component_apply_idx is not None)
                self.selected_block_editor.setVisible(False)
            elif self._block_apply_idx is not None and self._block_apply_idx > idx:
                self._block_apply_idx -= 1
            if self._active_block_idx == idx:
                self._active_block_idx = None
                self.add_btn.setEnabled(False)
            elif self._active_block_idx is not None and self._active_block_idx > idx:
                self._active_block_idx -= 1
        self._reindex_component_labels()
        self._refresh_components_table()
        self._refresh_blocks_list()
        self._refresh_project_tree()
        self._refresh_summary()
        self._set_component_edit_state(self._editing_component_idx)
        self._set_component_editor_enabled(bool(self.inventory.staged_blocks))

    def _on_clone_component(self, idx: int) -> None:
        """Duplicate a component as an independent copy.

        The copy lands in the component's own instance, or, when several
        instances exist, in the one the user picks.
        """
        if not (0 <= idx < len(self.inventory.staged_components)):
            return
        import copy
        src = self.inventory.staged_components[idx]
        blocks = self.inventory.staged_blocks
        owner_idx = next(
            (bi for bi, b in enumerate(blocks) if any(c is src for c in b.components)),
            None,
        )
        if owner_idx is None:
            return
        target_idx = owner_idx
        # With more than one instance, ask which to copy into (default: the owner).
        if len(blocks) > 1:
            from PySide6.QtWidgets import QInputDialog
            labels = [f"Instance_{bi + 1}" for bi in range(len(blocks))]
            choice, ok = QInputDialog.getItem(
                self, "Clone component",
                "Add the copy to which instance?", labels, owner_idx, editable=False)
            if not ok:
                return
            target_idx = labels.index(choice)
        target = blocks[target_idx]
        twin = copy.deepcopy(src)
        twin.label = f"component{target_idx + 1}_{len(target.components) + 1}"
        self.inventory.add_component(twin)
        target.components.append(twin)
        self._mark_instance_dirty()
        self._refresh_components_table()
        self._refresh_project_tree()
        self._refresh_summary()
        self.status_label.setText(f"Status: Cloned {src.label} -> {twin.label}")

    # -------------------- Block actions --------------------
    def on_add_block_clicked(self) -> None:
        label = f"block_{len(self.inventory.staged_blocks) + 1}"
        block = BlockSpec(
            label=label,
            components=[],
            weight=1.0,             # single-block 2D instances are unweighted
            indices_strategy="auto",
            explicit_indices=None,
        )
        self.inventory.add_block(block)
        self._mark_instance_dirty()
        self._active_block_idx = len(self.inventory.staged_blocks) - 1
        self.add_btn.setToolTip("Add the current component parameters to the selected block.")
        self._refresh_blocks_list()
        self._refresh_project_tree()
        # Highlight the new instance WITHOUT opening the component form: the middle
        # panel shows the "click the + next to your instance" guidance instead, and
        # the editor opens only when the user creates a component (the tree "+").
        node = self.project_tree.topLevelItem(self._active_block_idx)
        if node is not None:
            self.project_tree.blockSignals(True)
            self.project_tree.setCurrentItem(node)
            self.project_tree.blockSignals(False)
        self._set_component_editor_enabled(False)
        self.status_label.setText(f"Status: Added {label}")

    def on_build_block_clicked(self) -> None:
        rows = self._selected_component_rows() or self._checked_component_rows()
        if not rows:
            return
        components = [self.inventory.staged_components[r] for r in rows]
        label = f"block_{len(self.inventory.staged_blocks) + 1}"
        try:
            blk = assemble_block(
                components=components,
                weight=1.0,
                label=label,
                indices_strategy="auto",
                explicit_indices=None,
            )
        except BlockAssemblyError as e:
            QMessageBox.critical(self, "Block assembly", str(e))
            return
        self.inventory.add_block(blk)
        self._uncheck_all_components()
        self._refresh_blocks_list()
        self._refresh_project_tree()
        # Auto-preview the block that was just built.
        self._active_block_idx = len(self.inventory.staged_blocks) - 1
        self._select_project_block(self._active_block_idx)
        self.blocks_list.setCurrentRow(self._active_block_idx)
        self.on_plot_block_clicked()

    def _current_block_index(self) -> int | None:
        if (
            self._active_block_idx is not None
            and 0 <= self._active_block_idx < len(self.inventory.staged_blocks)
        ):
            return self._active_block_idx
        item = self.project_tree.currentItem() if hasattr(self, "project_tree") else None
        if item is not None:
            data = item.data(0, Qt.UserRole)
            if isinstance(data, (tuple, list)):
                if data[0] == "block":
                    return int(data[1])
                if data[0] == "component" and item.parent() is not None:
                    parent_data = item.parent().data(0, Qt.UserRole)
                    if isinstance(parent_data, (tuple, list)) and parent_data[0] == "block":
                        return int(parent_data[1])
        return None

    def _select_project_block(self, idx: int | None) -> None:
        if idx is None or not (0 <= idx < self.project_tree.topLevelItemCount()):
            return
        item = self.project_tree.topLevelItem(idx)
        self.project_tree.setCurrentItem(item)
        self._active_block_idx = idx
        self._set_component_editor_enabled(True)
        self.add_btn.setEnabled(True)
        self.add_btn.setToolTip("Add the current component parameters to the selected block.")

    def on_build_instance_clicked(self) -> None:
        if not self.inventory.staged_blocks:
            QMessageBox.information(self, "Build instance", "Add at least one block first.")
            return
        empty = [blk.label for blk in self.inventory.staged_blocks if not blk.components]
        if empty:
            QMessageBox.information(
                self,
                "Build instance",
                "Add at least one component to each block before building the instance.",
            )
            return
        try:
            if len(self.inventory.staged_blocks) == 1:
                composite = self._prepared_block_instance(self.inventory.staged_blocks[0])
            else:
                composite = assemble_composite(copy.deepcopy(self.inventory.staged_blocks))
        except (BlockAssemblyError, CompositeAssemblyError, ValueError) as exc:
            QMessageBox.critical(self, "Build instance", str(exc))
            return
        label = f"educational_instance_{len(self.inventory.composites) + 1}"
        ref = CompositeRef(
            label=label,
            composite=composite,
            source_blocks=copy.deepcopy(self.inventory.staged_blocks),
        )
        self._active_composite_idx = self.inventory.add_composite(ref)
        self._refresh_summary()
        self.status_label.setText(f"Status: Built {label}")
        Xg, Yg, X = self._grid_2d()
        Z = self._compute_instance_landscape(composite, X).reshape(Xg.shape)
        self._render_landscape(Xg, Yg, Z, "Built instance")

    def on_save_instance_clicked(self) -> None:
        if self._active_composite_idx is None:
            self.on_build_instance_clicked()
        if (
            self._active_composite_idx is None
            or not (0 <= self._active_composite_idx < len(self.inventory.composites))
        ):
            return
        import os

        ref = self.inventory.composites[self._active_composite_idx]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save instance",
            f"{ref.label}.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            written = save_composite(
                ref.composite,
                filename=os.path.basename(path),
                output_dir=os.path.dirname(path) or None,
            )
        except Exception as exc:  # noqa: BLE001 - present persistence errors to the user
            QMessageBox.critical(self, "Save instance", str(exc))
            return
        ref.saved_path = written
        ref.label = os.path.basename(written)
        self._refresh_summary()
        self.status_label.setText(f"Status: Saved {ref.label}")
        QMessageBox.information(self, "Saved", f"Saved instance to {written}.")

    def _on_save_block_instance(self, idx: int) -> None:
        if not (0 <= idx < len(self.inventory.staged_blocks)):
            return
        block = self.inventory.staged_blocks[idx]
        if not block.components:
            QMessageBox.information(
                self,
                "Save instance",
                "Add at least one component to this instance before saving it.",
            )
            return
        try:
            composite = self._prepared_block_instance(block)
        except (BlockAssemblyError, CompositeAssemblyError, ValueError) as exc:
            QMessageBox.critical(self, "Save instance", str(exc))
            return

        import os

        default_name = f"Instance_{idx + 1}.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save instance",
            default_name,
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            written = save_composite(
                composite,
                filename=os.path.basename(path),
                output_dir=os.path.dirname(path) or None,
            )
        except Exception as exc:  # noqa: BLE001 - present persistence errors to the user
            QMessageBox.critical(self, "Save instance", str(exc))
            return
        label = os.path.basename(written)
        self.status_label.setText(f"Status: Saved {label}")
        QMessageBox.information(self, "Saved", f"Saved instance to {written}.")

    # -------------------- Selection helpers --------------------
    def _checked_component_rows(self) -> list[int]:
        rows: list[int] = []
        for r in range(self.components_table.rowCount()):
            item = self.components_table.item(r, 0)
            if item is not None and item.checkState() == Qt.Checked:
                rows.append(r)
        return rows

    def check_all_components(self) -> None:
        """Select every staged component using its checkbox."""
        for r in range(self.components_table.rowCount()):
            item = self.components_table.item(r, 0)
            if item is not None:
                item.setCheckState(Qt.Checked)

    def _uncheck_all_components(self) -> None:
        for r in range(self.components_table.rowCount()):
            item = self.components_table.item(r, 0)
            if item is not None:
                item.setCheckState(Qt.Unchecked)

    def _selected_component_rows(self) -> list[int]:
        """Component indices ticked in the top Components list."""
        rows: list[int] = []
        if hasattr(self, "component_tree"):
            for i in range(self.component_tree.topLevelItemCount()):
                item = self.component_tree.topLevelItem(i)
                data = item.data(0, Qt.UserRole)
                if (
                    isinstance(data, (tuple, list))
                    and data[0] == "component"
                    and item.checkState(0) == Qt.Checked
                ):
                    rows.append(int(data[1]))
        return sorted(set(rows))

    def _selected_block_rows(self) -> list[int]:
        """Block indices whose tree checkbox is ticked."""
        rows: list[int] = []
        for item in self._iter_tree_items():
            data = item.data(0, Qt.UserRole)
            if (
                isinstance(data, (tuple, list))
                and data[0] == "block"
                and item.checkState(0) == Qt.Checked
            ):
                rows.append(int(data[1]))
        return sorted(set(rows))

    # -------------------- Refresh helpers --------------------
    def _refresh_components_table(self) -> None:
        self.components_table.setRowCount(len(self.inventory.staged_components))
        for row, spec in enumerate(self.inventory.staged_components):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Unchecked)
            check.setTextAlignment(Qt.AlignCenter)
            self.components_table.setItem(row, 0, check)
            self.components_table.setItem(row, 1, self._centered_table_item(spec.label))
            self.components_table.setItem(row, 2, self._centered_table_item(spec.baseline))
            centre = spec.params.get("c", [0.0, 0.0])
            self.components_table.setItem(row, 3, self._centered_table_item(str(centre)))
        self._refresh_summary()

    def _refresh_blocks_list(self) -> None:
        self.blocks_list.clear()
        for blk in self.inventory.staged_blocks:
            self.blocks_list.addItem(
                QListWidgetItem(
                    f"{blk.label}  (weight={blk.weight}, components={len(blk.components)})"
                )
            )
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        n_c = len(self.inventory.staged_components)
        n_b = len(self.inventory.staged_blocks)
        n_i = len(self.inventory.composites)
        self.footer_counts_label.setText(
            f"{n_c} components  -  {n_b} blocks  -  {n_i} instances"
        )

    @staticmethod
    def _format_component_value(value) -> str:
        try:
            import numpy as np

            if isinstance(value, np.generic):
                value = value.item()
        except Exception:
            pass
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    @classmethod
    def _value_at(cls, value, index: int) -> str:
        try:
            import numpy as np

            arr = np.asarray(value)
            if arr.ndim == 0:
                return cls._format_component_value(arr.item())
            if index >= arr.reshape(-1).shape[0]:
                return ""
            return cls._format_component_value(arr.reshape(-1)[index])
        except Exception:
            if isinstance(value, (list, tuple)) and index < len(value):
                return cls._format_component_value(value[index])
            return cls._format_component_value(value)

    @staticmethod
    def _plain_math_header(text: str) -> str:
        display = math_text(text)
        display = display.replace("<sup>+</sup>", "+")
        display = display.replace("<sup>-</sup>", "-")
        return re.sub(r"<sub>(.*?)</sub>", r"\1", display)

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

    def _build_component_parameters_table(self, spec: ComponentSpec) -> QTableWidget:
        params = spec.params or {}
        if spec.baseline == "B":
            rows = [("c", "c"), ("κ+", "kappa_plus"), ("κ-", "kappa_minus"), ("p", "p_single")]
        else:
            rows = [
                ("c", "c"),
                ("κ+", "kappa_plus"),
                ("κ-", "kappa_minus"),
                ("p+", "p_plus"),
                ("p-", "p_minus"),
            ]
        table = QTableWidget(len(rows), 3)
        table.setObjectName("componentParameterTable")
        table.setHorizontalHeaderLabels(["parameter", "x", "y"])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(False)
        # Per-group tints matching the Generator's parameter table (centre =
        # peach, κ = blue, p = green). No reroll column in the Education tab.
        tint_of = {
            "c": QColor(PARAM_TABLE_TINTS["c"]),
            "kappa_plus": QColor(PARAM_TABLE_TINTS["kappa_plus"]),
            "kappa_minus": QColor(PARAM_TABLE_TINTS["kappa_minus"]),
            "p_plus": QColor(PARAM_TABLE_TINTS["p_plus"]),
            "p_minus": QColor(PARAM_TABLE_TINTS["p_minus"]),
            "p_single": QColor(PARAM_TABLE_TINTS["p_single"]),
        }
        for row_idx, (label, key) in enumerate(rows):
            value = params.get(key, "")
            cells = [
                self._centered_table_item(label),
                self._centered_table_item(self._value_at(value, 0)),
                self._centered_table_item(self._value_at(value, 1)),
            ]
            tint = tint_of.get(key)
            for col, item in enumerate(cells):
                if tint is not None:
                    item.setBackground(tint)
                table.setItem(row_idx, col, item)
        self._fit_table_to_rows(table)
        return table

    @staticmethod
    def _transform_param_base(name: str) -> str:
        """Drop a ±-direction suffix so μ⁺/μ⁻ share one group key."""
        if name.endswith("_plus"):
            return name[:-5]
        if name.endswith("_minus"):
            return name[:-6]
        return name

    def _tf_table_label(self, transform_name: str, name: str) -> str:
        """Plain-text data-table label; tensor interference's μ is the base
        amplitude μ₀, which plain_math_header would flatten, so emit μ₀⁺/μ₀⁻
        literally."""
        if transform_name == "tensor_interference" and \
                self._transform_param_base(name) == "mu":
            sup = "⁺" if name.endswith("_plus") else (
                "⁻" if name.endswith("_minus") else "")
            return f"μ₀{sup}"
        return self._plain_math_header(name)

    def _build_component_transform_table(self, params: dict,
                                         transform_name: str = "") -> QTableWidget:
        items = list((params or {}).items())
        table = QTableWidget(len(items), 3)
        table.setObjectName("componentTransformTable")
        table.setHorizontalHeaderLabels(["parameter", "x", "y"])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(False)
        # Each ± pair (μ, γ, ω, …) gets its own group colour, matching the
        # Generator's transform table palette.
        tf_palette = [
            PARAM_TABLE_TINTS["tf_blue"], PARAM_TABLE_TINTS["tf_green"],
            PARAM_TABLE_TINTS["tf_pink"], PARAM_TABLE_TINTS["tf_yellow"],
            PARAM_TABLE_TINTS["tf_lavender"],
        ]
        base_color: dict = {}
        for name, _value in items:
            base = self._transform_param_base(name)
            if base not in base_color:
                base_color[base] = QColor(
                    tf_palette[len(base_color) % len(tf_palette)])
        for row_idx, (name, value) in enumerate(items):
            tint = base_color.get(self._transform_param_base(name))
            cells = [
                self._centered_table_item(
                    self._tf_table_label(transform_name, name)),
                self._centered_table_item(self._value_at(value, 0)),
                self._centered_table_item(self._value_at(value, 1)),
            ]
            for col, item in enumerate(cells):
                if tint is not None:
                    item.setBackground(tint)
                table.setItem(row_idx, col, item)
        self._fit_table_to_rows(table)
        return table

    def _build_component_parameters_dialog(self, spec: ComponentSpec) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{spec.label} parameters")
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

        title = QLabel(spec.label)
        title.setStyleSheet(
            f"font-weight: 700; font-size: 15px; color: {self._palette['emphasis']};")
        scroll_layout.addWidget(title)
        neutral_on = bool(spec.neutralize.get('enable'))
        summary_text = (
            f"base form = {spec.baseline}; "
            f"beta = {self._format_component_value(spec.params.get('beta', 0.0))}; "
            f"neutralization = {'on' if neutral_on else 'off'}"
        )
        # Delta / r_ref only matter when neutralization is on; omit them when off.
        if neutral_on:
            summary_text += (
                f"; Delta = {self._format_component_value(spec.neutralize.get('Delta', ''))}"
                f"; r_ref = {self._format_component_value(spec.neutralize.get('r_ref', ''))}"
            )
        summary = QLabel(summary_text)
        summary.setObjectName("componentParameterSummary")
        summary.setWordWrap(True)
        summary.setStyleSheet(f"color: {self._palette['emphasis']};")
        scroll_layout.addWidget(summary)
        scroll_layout.addWidget(self._build_component_parameters_table(spec))

        for order, transform in enumerate(spec.T_seq or [], start=1):
            t_name = str(transform.get("name", "transform"))
            params = transform.get("params", {}) or {}
            if not params:
                continue
            transform_label = QLabel(
                f"{order}. {self.editor._transform_display_name(t_name)}")
            transform_label.setStyleSheet(
                f"font-weight: 700; color: {self._palette['emphasis']}; margin-top: 8px;")
            scroll_layout.addWidget(transform_label)
            scroll_layout.addWidget(
                self._build_component_transform_table(params, t_name))

        scroll_layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        return dialog

    def _open_component_table(self, idx: int) -> None:
        """Data-table icon: select the component's tree row (so it highlights
        like a click), then open its data table."""
        for item in self._iter_tree_items():
            data = item.data(0, Qt.UserRole)
            # Qt returns the stored ("component", idx) tuple as a list.
            if data and tuple(data) == ("component", idx):
                self.project_tree.setCurrentItem(item)
                break
        self._show_component_parameters_dialog(idx)

    def _show_component_parameters_dialog(self, idx: int) -> None:
        if not (0 <= idx < len(self.inventory.staged_components)):
            return
        spec = self.inventory.staged_components[idx]
        self._build_component_parameters_dialog(spec).exec()

    # -------------------- Tree icons --------------------
    def _build_tree_icons(self) -> None:
        """Pre-render the small vector icons shown in the model-hierarchy tree.

        Drawn with QPainter (no asset files): a shaded circle per component
        (shade depends on baseline), an isometric cube per block, and a
        stacked-layers glyph per composite.
        """
        self._icon_component_a = self._make_circle_icon("#5b9bd5")
        self._icon_component_b = self._make_circle_icon("#7e8aa2")
        self._icon_block = self._make_cube_icon()
        self._icon_composite = self._make_layers_icon()
        self._icon_pencil = self._make_pencil_icon()
        self._icon_table = self._make_table_icon()
        from portal_studio.gui.common.icons import make_clone_icon
        self._icon_clone = make_clone_icon()
        # Edit-state tracking for the per-row pencil 'pressed' look. Mirrors
        # the Generator tab pattern: separate selecting from entering edit
        # mode; only an explicit pencil click pushes the button.
        self._editing_component_idx: "int | None" = None
        self._component_edit_buttons: dict = {}
        self._editing_transform_idx: "int | None" = None
        self._transform_edit_buttons: dict = {}
        self._icon_save = self._make_save_icon()
        self._icon_cross = self._make_cross_icon()

    def _make_pencil_icon(self, color: str = "#4c5870", size: int = 16) -> QIcon:
        pm, p = self._begin_icon(size)
        p.setPen(QPen(QColor(color), 2.2))
        p.drawLine(QPointF(4.5, size - 4.5), QPointF(size - 5, 5))  # pencil body
        p.setPen(QPen(QColor(color), 1.3))
        p.drawLine(QPointF(3, size - 3), QPointF(5.5, size - 5.5))  # tip
        p.end()
        return QIcon(pm)

    def _make_save_icon(self, color: str = "#4c5870", size: int = 16) -> QIcon:
        pm, p = self._begin_icon(size)
        m = 2.5
        p.setPen(QPen(QColor(color), 1.4))
        p.setBrush(QBrush(QColor(0, 0, 0, 0)))
        p.drawRoundedRect(QRectF(m, m, size - 2 * m, size - 2 * m), 2, 2)  # disk body
        p.drawRect(QRectF(m + 2.5, m, size - 2 * m - 5, 3.2))  # top slot
        p.drawRect(QRectF(m + 2.5, size - m - 4.5, size - 2 * m - 5, 3.0))  # window
        p.end()
        return QIcon(pm)

    def _make_table_icon(self, color: str = "#4c5870", size: int = 16) -> QIcon:
        pm, p = self._begin_icon(size)
        m = 2.5
        p.setPen(QPen(QColor(color), 1.3))
        p.setBrush(QBrush(QColor(0, 0, 0, 0)))
        rect = QRectF(m, m, size - 2 * m, size - 2 * m)
        p.drawRoundedRect(rect, 1.5, 1.5)
        for frac in (1 / 3, 2 / 3):
            x = m + (size - 2 * m) * frac
            y = m + (size - 2 * m) * frac
            p.drawLine(QPointF(x, m), QPointF(x, size - m))
            p.drawLine(QPointF(m, y), QPointF(size - m, y))
        p.end()
        return QIcon(pm)

    def _make_cross_icon(self, color: str = "#d64545", size: int = 16) -> QIcon:
        pm, p = self._begin_icon(size)
        m = 4.5
        p.setPen(QPen(QColor(color), 2.3))
        p.drawLine(QPointF(m, m), QPointF(size - m, size - m))
        p.drawLine(QPointF(size - m, m), QPointF(m, size - m))
        p.end()
        return QIcon(pm)

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

    # Edit pencil 'pressed look' (soft blue card + inset border).
    _EDIT_BTN_QSS = (
        "QPushButton#nodeActionBtn { background: transparent; border: none; }"
        "QPushButton#nodeActionBtn:checked {"
        "  background: #dbeafe;"
        "  border: 1px solid #93c5fd;"
        "  border-radius: 4px;"
        "}"
    )

    def _make_node_actions(self, kind: str, idx: int) -> QWidget:
        holder = QWidget()
        holder.setObjectName("nodeActions")
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 6, 0)
        row.setSpacing(1)
        if kind == "component":
            row.addWidget(self._mini_button(
                self._icon_table, "Open the generated-parameter table for this component",
                lambda i=idx: self._open_component_table(i)))
            row.addWidget(self._mini_button(
                self._icon_clone, "Clone this component as an independent copy.",
                lambda i=idx: self._on_clone_component(i)))
        if kind == "block":
            plus = QPushButton("+")
            plus.setFixedSize(24, 24)
            plus.setObjectName("nodeActionBtn")
            plus.setToolTip("Add a new component to this instance")
            plus.setAccessibleName(plus.toolTip())
            plus.setAccessibleDescription(plus.toolTip())
            plus.clicked.connect(
                lambda _checked=False, i=idx: self._on_new_component_for_block(i))
            row.addWidget(plus)
            row.addWidget(self._mini_button(
                self._icon_save, "Save this instance as JSON",
                lambda i=idx: self._on_save_block_instance(i)))
        delete_tip = "Delete this component" if kind == "component" else "Delete this instance"
        row.addWidget(self._mini_button(self._icon_cross, delete_tip,
                                        lambda k=kind, i=idx: self._on_delete_node(k, i)))
        return holder

    def _set_component_edit_state(self, idx: "int | None") -> None:
        for btn in self._component_edit_buttons.values():
            try:
                btn.setChecked(False)
            except RuntimeError:
                continue
        if idx is None:
            return
        btn = self._component_edit_buttons.get(idx)
        if btn is not None:
            try:
                btn.setChecked(True)
            except RuntimeError:
                pass

    def _set_transform_edit_state(self, idx: "int | None") -> None:
        for btn in self._transform_edit_buttons.values():
            try:
                btn.setChecked(False)
            except RuntimeError:
                continue
        if idx is None:
            return
        btn = self._transform_edit_buttons.get(idx)
        if btn is not None:
            try:
                btn.setChecked(True)
            except RuntimeError:
                pass

    @staticmethod
    def _begin_icon(size: int = 18):
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        return pm, painter

    def _make_circle_icon(self, fill: str, size: int = 18) -> QIcon:
        pm, p = self._begin_icon(size)
        m = 3.5
        p.setBrush(QBrush(QColor(fill)))
        p.setPen(QPen(QColor(fill).darker(140), 1.3))
        p.drawEllipse(QRectF(m, m, size - 2 * m, size - 2 * m))
        p.end()
        return QIcon(pm)

    def _make_cube_icon(self, color: str = "#3b82f6", size: int = 18) -> QIcon:
        pm, p = self._begin_icon(size)
        m = 2.5
        cx = size / 2.0
        top = m
        bottom = size - m
        h = bottom - top
        apex_top = QPointF(cx, top)
        upper_right = QPointF(size - m, top + h * 0.25)
        lower_right = QPointF(size - m, top + h * 0.75)
        apex_bottom = QPointF(cx, bottom)
        lower_left = QPointF(m, top + h * 0.75)
        upper_left = QPointF(m, top + h * 0.25)
        center = QPointF(cx, top + h * 0.5)
        outline = QPolygonF([apex_top, upper_right, lower_right, apex_bottom, lower_left, upper_left])
        p.setPen(QPen(QColor(color), 1.4))
        p.setBrush(QBrush(QColor(color).lighter(168)))
        p.drawPolygon(outline)
        p.drawLine(center, apex_top)
        p.drawLine(center, lower_right)
        p.drawLine(center, lower_left)
        p.end()
        return QIcon(pm)

    def _make_layers_icon(self, color: str = "#3b82f6", size: int = 18) -> QIcon:
        pm, p = self._begin_icon(size)
        m = 2.0
        cx = size / 2.0
        hw = (size - 2 * m) / 2.0
        hh = hw * 0.55

        def rhombus(cy: float) -> QPolygonF:
            return QPolygonF([
                QPointF(cx, cy - hh),
                QPointF(cx + hw, cy),
                QPointF(cx, cy + hh),
                QPointF(cx - hw, cy),
            ])

        p.setPen(QPen(QColor(color), 1.3))
        p.setBrush(QBrush(QColor(0, 0, 0, 0)))  # outline only for lower layers
        p.drawPolygon(rhombus(size * 0.66))
        p.drawPolygon(rhombus(size * 0.50))
        p.setBrush(QBrush(QColor(color).lighter(150)))  # filled top layer
        p.drawPolygon(rhombus(size * 0.34))
        p.end()
        return QIcon(pm)

    def _iter_tree_items(self):
        """Yield every item in the project tree, depth-first."""
        def walk(item):
            yield item
            for i in range(item.childCount()):
                yield from walk(item.child(i))
        for i in range(self.project_tree.topLevelItemCount()):
            yield from walk(self.project_tree.topLevelItem(i))

    def _component_icon(self, baseline: str) -> QIcon:
        return (
            self._icon_component_a
            if str(baseline).upper() == "A"
            else self._icon_component_b
        )

    def _refresh_component_list(self) -> None:
        """Populate the top Components pool: every component, always present."""
        if not hasattr(self, "component_tree"):
            return
        # Discard stale per-row button references — about to be re-created.
        self._component_edit_buttons = {}
        self.component_tree.blockSignals(True)
        self.component_tree.clear()
        for idx, spec in enumerate(self.inventory.staged_components):
            item = QTreeWidgetItem([f"{spec.label}    Baseline: {spec.baseline}"])
            item.setIcon(0, self._component_icon(spec.baseline))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Unchecked)
            item.setData(0, Qt.UserRole, ("component", idx))
            item.setToolTip(0, "Tick to include in a block; click to load into the editor")
            self.component_tree.addTopLevelItem(item)
            self.component_tree.setItemWidget(item, 1, self._make_node_actions("component", idx))
        self.component_tree.blockSignals(False)

    def _block_index(self, block: BlockSpec) -> int | None:
        for i, candidate in enumerate(self.inventory.staged_blocks):
            if candidate is block:
                return i
        return None

    def _component_index(self, component: ComponentSpec) -> int | None:
        for i, candidate in enumerate(self.inventory.staged_components):
            if candidate is component:
                return i
        return None

    def _add_component_node(self, parent: QTreeWidgetItem, component: ComponentSpec) -> None:
        node = QTreeWidgetItem([f"{component.label}    Baseline: {component.baseline}"])
        node.setIcon(0, self._component_icon(component.baseline))
        component_index = self._component_index(component)
        if component_index is not None:
            node.setData(0, Qt.UserRole, ("component", component_index))
            node.setToolTip(0, "Click to preview; use the actions to view data, clone, or delete")
        parent.addChild(node)
        if component_index is not None:
            self.project_tree.setItemWidget(
                node,
                1,
                self._make_node_actions("component", component_index),
            )

    def _add_block_node(self, block: BlockSpec) -> None:
        block_index = self._block_index(block)
        number = (block_index + 1) if block_index is not None else "?"
        count = len(block.components)
        unit = "component" if count == 1 else "components"
        node = QTreeWidgetItem([f"Instance_{number}  ({count} {unit})"])
        if block_index is not None:
            node.setData(0, Qt.UserRole, ("block", block_index))
            node.setToolTip(0, "Select this instance, then add components to it")
        self.project_tree.addTopLevelItem(node)
        if block_index is not None:
            self.project_tree.setItemWidget(
                node,
                1,
                self._make_node_actions("block", block_index),
            )
        for component in block.components:
            self._add_component_node(node, component)
        node.setExpanded(True)

    def _refresh_project_tree(self) -> None:
        """Render the editable block-first instance tree."""
        self._refresh_component_list()
        self.project_tree.blockSignals(True)
        self.project_tree.clear()

        blocks = self.inventory.staged_blocks

        # Every staged block shown as a top-level interactive node.
        for block in blocks:
            self._add_block_node(block)

        self.project_tree.blockSignals(False)
        has_blocks = len(self.inventory.staged_blocks) > 0
        self.project_tree.setVisible(has_blocks)
        if hasattr(self, "tree_empty_hint"):
            self.tree_empty_hint.setVisible(not has_blocks)
        has_benchmarkable_block = any(len(block.components) > 0 for block in blocks)
        self.try_algorithm_btn.setEnabled(has_benchmarkable_block)

    def on_component_tree_selection_changed(self) -> None:
        """Row-selection only LOADS data — does NOT enter edit mode.

        Edit mode is entered exclusively by clicking the pencil icon on a
        row (see ``_on_edit_component``). Selecting a different row also
        releases any in-progress edit so the pencil-pressed visual matches
        the actual editing target.
        """
        items = self.component_tree.selectedItems()
        if not items:
            return
        data = items[0].data(0, Qt.UserRole)
        if isinstance(data, (tuple, list)) and data[0] == "component":
            idx = int(data[1])
            self._load_component_into_editor(idx)
            # Released if the user moved off the currently-edited row.
            if (self._editing_component_idx is not None
                    and self._editing_component_idx != idx):
                self._editing_component_idx = None
                self._set_component_edit_state(None)

    # -------------------- Transform list (left panel) --------------------
    def _setup_transform_list(self) -> None:
        """Turn the editor's transform list into a draggable, in-panel list.

        The list supports drag-to-reorder, and every change is routed through
        the live re-render.
        """
        tl = self.editor.transform_list
        tl.setStyleSheet("")  # use the global themed QListWidget look
        tl.show()
        tl.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        tl.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        tl.setDefaultDropAction(Qt.MoveAction)
        # No internal scroll: the list auto-resizes to fit every row and the
        # outer editor scroll area handles overflow.
        tl.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tl.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tl.itemClicked.connect(self._on_transform_row_clicked)
        tl.model().rowsMoved.connect(self._on_transform_rows_moved)
        # Adding a transform goes through the editor's button; refresh + replot.
        self.editor.btn_add_transform.clicked.connect(self._sync_transforms_and_plot)
        # Changing the transform type rebuilds the parameter fields (different
        # height), so re-fit the form afterwards.
        self.editor.transform_combo.currentTextChanged.connect(lambda *_: self._fit_editor())
        self._building_transform_rows = False
        self._refresh_transform_rows()

    def _refresh_transform_rows(self) -> None:
        """Render controller.state.T_seq as compact, draggable rows.

        After repopulating, the list is resized to the sum of its row
        heights so it claims exactly the space it needs and the outer
        editor scroll handles overflow (no inner scrollbar).
        """
        # Drop stale per-row button references — the QWidgets are about to be
        # rebuilt; they would soon be dangling Qt references otherwise.
        self._transform_edit_buttons = {}
        tl = self.editor.transform_list
        self._building_transform_rows = True
        tl.blockSignals(True)
        tl.clear()
        for idx, transform in enumerate(self.editor.controller.state.T_seq):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, transform)  # ref kept for drag reorder
            widget = self._make_transform_row(idx, transform)
            item.setSizeHint(widget.sizeHint())
            tl.addItem(item)
            tl.setItemWidget(item, widget)
        tl.blockSignals(False)
        self._building_transform_rows = False
        # Include stylesheet padding, item margins, and inter-item spacing
        # in the list height; sizeHintForRow excludes these allowances.
        n = tl.count()
        if n == 0:
            tl.setFixedHeight(0)
        else:
            QSS_LIST_PADDING = 12   # 6px top + 6px bottom
            QSS_ITEM_MARGIN = 6     # 3px top + 3px bottom per item
            rows = sum(tl.sizeHintForRow(i) for i in range(n))
            total = (rows
                     + 2 * tl.frameWidth()
                     + QSS_LIST_PADDING
                     + n * QSS_ITEM_MARGIN
                     + max(0, n - 1) * tl.spacing()
                     + 4)  # small safety buffer for sub-pixel rounding
            tl.setFixedHeight(total)
        # Re-fit the outer editor scroll so its content height tracks the
        # newly-resized list.
        QtCore.QTimer.singleShot(0, self._fit_editor)

    def _make_transform_row(self, idx: int, transform: dict) -> QWidget:
        row = QFrame()
        row.setObjectName("componentLibraryRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(8, 4, 6, 4)
        layout.setSpacing(6)
        handle = QLabel("☰")  # drag-handle hint
        handle.setObjectName("mutedLabel")
        handle.setToolTip("Drag to reorder")
        layout.addWidget(handle)
        number = QLabel(str(idx + 1))
        number.setObjectName("accentLabel")
        number.setFixedWidth(12)
        layout.addWidget(number)
        from PySide6.QtWidgets import QSizePolicy

        name = QLabel(self.editor._transform_display_name(str(transform.get("name", ""))))
        name.setStyleSheet("font-weight: 700;")
        # Flex and shrink so a long name never pushes the action icons out of
        # the row (it elides/clips instead of overflowing).
        name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        name.setMinimumWidth(0)
        layout.addWidget(name, stretch=1)
        layout.addWidget(self._mini_button(
            self._icon_cross, "Remove this transformation",
            lambda i=idx: self._delete_transform_row(i)))
        return row

    def _sync_transforms_and_plot(self, *_args) -> None:
        """Re-render the transform rows and trigger a live re-plot."""
        self._refresh_transform_rows()
        self._schedule_live_plot()

    def _on_transform_row_clicked(self, item) -> None:
        """Row click only LOADS data — does NOT enter edit mode (matches the
        component-tree behaviour). If a DIFFERENT transform was being edited,
        release its pressed pencil."""
        idx = self.editor.transform_list.row(item)
        seq = self.editor.controller.state.T_seq
        if 0 <= idx < len(seq):
            self._load_transform_into_fields(seq[idx])
            self._transform_apply_idx = idx
            self.apply_changes_btn.setEnabled(True)
            if self._editing_transform_idx is not None:
                self._editing_transform_idx = None
                self._set_transform_edit_state(None)

    def _load_transform_row(self, idx: int) -> None:
        """Pencil-icon click: load the spec AND enter edit mode (pressed look)."""
        seq = self.editor.controller.state.T_seq
        if 0 <= idx < len(seq):
            self._load_transform_into_fields(seq[idx])
            self._transform_apply_idx = idx
            self.apply_changes_btn.setEnabled(True)
            self.status_label.setText(f"Status: Editing transformation {idx + 1}")

    def _save_transform_row(self, idx: int) -> bool:
        """Write the current parameter fields back into transform ``idx``."""
        if not (0 <= idx < len(self.editor.controller.state.T_seq)):
            return False
        self.editor.transform_list.setCurrentRow(idx)
        if not self.editor._on_edit_transform():
            return False
        self._editing_transform_idx = None
        self._transform_apply_idx = None
        self._sync_transforms_and_plot()
        self._set_transform_edit_state(None)
        self.status_label.setText(f"Status: Saved transformation {idx + 1}")
        return True

    def _load_transform_into_fields(self, transform: dict) -> None:
        """Populate the transform type + parameter fields from ``transform``."""
        self.editor.set_transform_type(str(transform.get("name", "")))
        widgets = self.editor.transform_param_widgets
        for key, value in dict(transform.get("params", {})).items():
            if isinstance(value, (list, tuple)):
                for suffix, item in zip(("_x", "_y"), value):
                    self._set_widget_value(widgets.get(key + suffix), item)
            else:
                self._set_widget_value(widgets.get(key), value)

    @staticmethod
    def _set_widget_value(widget, value) -> None:
        if widget is None:
            return
        if hasattr(widget, "setValue"):
            widget.setValue(float(value))
        elif hasattr(widget, "setText"):
            widget.setText(str(value))

    def _delete_transform_row(self, idx: int) -> None:
        # Adjust the edit/apply cursors before the row is removed.
        if self._editing_transform_idx == idx:
            self._editing_transform_idx = None
        elif (self._editing_transform_idx is not None
              and self._editing_transform_idx > idx):
            self._editing_transform_idx -= 1
        if self._transform_apply_idx == idx:
            self._transform_apply_idx = None
        elif (self._transform_apply_idx is not None
              and self._transform_apply_idx > idx):
            self._transform_apply_idx -= 1
        self.editor.transform_list.setCurrentRow(idx)
        self.editor._on_delete_transform()
        self._sync_transforms_and_plot()
        self._set_transform_edit_state(self._editing_transform_idx)

    def _on_transform_rows_moved(self, *_args) -> None:
        """After a drag, rebuild T_seq from the new visual order and re-plot."""
        if self._building_transform_rows:
            return
        tl = self.editor.transform_list
        new_seq = []
        for i in range(tl.count()):
            data = tl.item(i).data(Qt.UserRole)
            if isinstance(data, dict):
                new_seq.append(data)
        self.editor.controller.state.T_seq = new_seq
        # Indices shifted by the reorder; release edit/apply state.
        self._editing_transform_idx = None
        self._transform_apply_idx = None
        self._sync_transforms_and_plot()

    def on_project_tree_selection_changed(self) -> None:
        items = self.project_tree.selectedItems()
        if not items and self.project_tree.currentItem() is not None:
            items = [self.project_tree.currentItem()]
        if not items:
            return
        self._handle_project_tree_item(items[0])

    def on_project_tree_current_item_changed(self, current, _previous) -> None:
        # Read the live current item (not the passed arg) so a guard-driven
        # re-selection is respected and the echo signal dedups correctly.
        item = self.project_tree.currentItem()
        if item is not None:
            self._handle_project_tree_item(item)

    def _editor_cfg_repr(self) -> str:
        """A stable representation of the editor's current component config,
        used to detect unapplied edits."""
        self.editor._refresh_params_to_controller()
        return repr(self.editor.controller.build_cfg())

    def _has_unapplied_component_edit(self) -> bool:
        """True when an existing component is loaded and the editor differs
        from what was last loaded or applied."""
        if getattr(self, "_component_apply_idx", None) is None:
            return False
        snap = getattr(self, "_loaded_cfg_repr", None)
        return snap is not None and self._editor_cfg_repr() != snap

    def _confirm_discard_component_edit(self) -> bool:
        """Warn that edits weren't applied; returns True to discard and switch."""
        if getattr(self, "_discard_prompt_active", False):
            return False
        self._discard_prompt_active = True
        try:
            result = QMessageBox.question(
                self, "Unapplied changes",
                "Your changes to this component have not been applied. "
                "Discard them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return result == QMessageBox.StandardButton.Yes
        finally:
            self._discard_prompt_active = False

    def _reselect_current_component(self) -> None:
        """Restore the tree selection to the component being edited (no signals)."""
        idx = getattr(self, "_component_apply_idx", None)
        if idx is None:
            return
        self.project_tree.blockSignals(True)
        try:
            for i in range(self.project_tree.topLevelItemCount()):
                node = self.project_tree.topLevelItem(i)
                for j in range(node.childCount()):
                    child = node.child(j)
                    d = child.data(0, Qt.UserRole)
                    if (isinstance(d, (tuple, list)) and d[0] == "component"
                            and int(d[1]) == idx):
                        self.project_tree.setCurrentItem(child)
                        return
        finally:
            self.project_tree.blockSignals(False)

    def _handle_project_tree_item(self, item: QTreeWidgetItem) -> None:
        data = item.data(0, Qt.UserRole)
        if not isinstance(data, (tuple, list)):
            return
        kind, idx = data
        target = (kind, int(idx))
        if target == getattr(self, "_loaded_target", None):
            return   # the tree fires current + selection signals; ignore the echo
        if getattr(self, "_discard_prompt_active", False):
            return
        # Warn before leaving a component that has unapplied edits.
        cur_idx = getattr(self, "_component_apply_idx", None)
        if (cur_idx is not None and target != ("component", cur_idx)
                and self._has_unapplied_component_edit()):
            prev_target = getattr(self, "_loaded_target", None)
            self._loaded_target = target   # target for duplicate selection signals
            if not self._confirm_discard_component_edit():
                self._loaded_target = prev_target
                self._reselect_current_component()
                return
        self._loaded_target = target
        if kind == "component":
            if item.parent() is not None:
                parent_data = item.parent().data(0, Qt.UserRole)
                if isinstance(parent_data, (tuple, list)) and parent_data[0] == "block":
                    self._active_block_idx = int(parent_data[1])
                    self._set_component_editor_enabled(True)
                    # Viewing an existing component: Add Component is disabled
                    # (greyed); add a new one by selecting the instance row.
                    self.add_btn.setEnabled(False)
            self._component_apply_idx = int(idx)
            self._block_apply_idx = None
            self.apply_changes_btn.setEnabled(True)
            self.selected_block_editor.setVisible(False)
            spec = self.inventory.staged_components[int(idx)]
            self.editor.load_component_config(
                {
                    "baseline": spec.baseline,
                    "rotation_deg": spec.rotation_deg,
                    "neutralize": spec.neutralize,
                    "params": spec.params,
                    "T_seq": spec.T_seq,
                }
            )
            self._sync_edu_from_editor()
            self._refresh_transform_rows()
            Xg, Yg, X = self._grid_2d()
            y = self.editor.controller.compute_points(X)
            self._render_landscape(Xg, Yg, y.reshape(Xg.shape), spec.label)
            # Snapshot the loaded state so later edits can be detected.
            self._loaded_cfg_repr = self._editor_cfg_repr()
        elif kind == "block":
            block_idx = int(idx)
            self._active_block_idx = block_idx
            self._block_apply_idx = None
            self._component_apply_idx = None
            # Selecting an instance row only arms the "+" target — it does NOT open
            # the component editor. Keep the editor hidden and show the "click the
            # + to add a component" guidance (same as adding an instance). The
            # editor reappears only when the user hits the per-instance "+" or
            # clicks an existing component row.
            self._set_component_editor_enabled(False)
            self.blocks_list.setCurrentRow(block_idx)
            self.selected_block_editor.setVisible(False)
            if self.inventory.staged_blocks[block_idx].components:
                self.on_plot_block_clicked()
            else:
                self.status_label.setText(
                    f"Status: Selected {self.inventory.staged_blocks[block_idx].label}"
                )

    # -------------------- Live preview --------------------
    def _wire_live_preview(self) -> None:
        """Connect every editor parameter widget to the debounced re-render."""
        e = self.editor
        for spin in (
            e.center_x, e.center_y, e.beta_spin, e.rotation_spin,
            e.kappa_plus_x, e.kappa_plus_y, e.kappa_minus_x, e.kappa_minus_y,
            e.p_plus_x, e.p_plus_y, e.p_minus_x, e.p_minus_y, e.p_single,
            e.neutral_delta, e.neutral_rref,
        ):
            spin.valueChanged.connect(self._schedule_live_plot)
        e.baseline_combo.currentTextChanged.connect(self._schedule_live_plot)
        e.neutral_enable.toggled.connect(self._schedule_live_plot)

    def _schedule_live_plot(self, *_args) -> None:
        """(Re)start the debounce timer; coalesces rapid edits into one render."""
        if not self.inventory.staged_components and not any(
            block.components for block in self.inventory.staged_blocks
        ):
            self._live_timer.stop()
            self._show_current_landscape_hint()
            return
        self._live_timer.start()

    def _do_live_plot(self) -> None:
        """Render the editor's current component without stealing the active tab."""
        self.editor._refresh_params_to_controller()
        Xg, Yg, X = self._grid_2d()
        try:
            y = self.editor.controller.compute_points(X)
        except ValueError as exc:
            self.status_label.setText(f"Status: Cannot preview component: {exc}")
            return
        self._render_landscape(
            Xg, Yg, y.reshape(Xg.shape), "Current component", switch_tab=False
        )

    # -------------------- Plot handlers --------------------
    def on_plot_current_clicked(self) -> None:
        """Plot the editor's current single-component landscape."""
        Xg, Yg, X = self._grid_2d()
        try:
            y = self.editor.controller.compute_points(X)
        except ValueError as exc:
            QMessageBox.critical(self, "Component preview", str(exc))
            return
        self._render_landscape(Xg, Yg, y.reshape(Xg.shape), "Current component")

    def on_plot_block_clicked(self) -> None:
        """Plot the selected block's lower envelope."""
        row = self.blocks_list.currentRow()
        if row < 0:
            selected = self._selected_block_rows()
            row = selected[0] if selected else -1
        if row < 0:
            return
        block = self.inventory.staged_blocks[row]
        if not block.components:
            return
        Xg, Yg, X = self._grid_2d()
        try:
            instance = self._prepared_block_instance(block)
            Z = self._compute_instance_landscape(instance, X).reshape(Xg.shape)
        except (BlockAssemblyError, CompositeAssemblyError, ValueError) as exc:
            QMessageBox.critical(self, "Instance preview", str(exc))
            return
        self._render_landscape(Xg, Yg, Z, f"Instance: Instance_{row + 1}")

    # -------------------- Plot helpers --------------------
    @staticmethod
    def _grid_2d(n: int = 120):
        return educational_landscape.grid_2d(int(n))

    @staticmethod
    def _compute_block_landscape(components, X):
        return educational_landscape.compute_block_landscape(components, X)

    @staticmethod
    def _compute_instance_landscape(inst, X):
        return educational_landscape.compute_instance_landscape(inst, X)

    @staticmethod
    def _compute_component_landscape(spec, X):
        return educational_landscape.compute_component_landscape(spec, X)

    def _render_landscape(self, Xg, Yg, Z, title: str, switch_tab: bool = True) -> None:
        """Render the surface (top) and contour (bottom) of one landscape."""
        self._show_current_landscape_plots()
        palette = getattr(self, "_palette", LIGHT)
        fig = self.editor.fig
        fig.clear()
        fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.08)
        ax_s = fig.add_subplot(111, projection="3d")
        ax_s.plot_surface(Xg, Yg, Z, cmap="jet")
        ax_s.set_title(f"Surface - {title}")
        ax_s.set_xlabel("x1")
        ax_s.set_ylabel("x2")
        ax_s.set_zlabel("f")
        ax_s.view_init(elev=30, azim=225)
        style_compact_axes(ax_s)
        style_figure(fig, palette)
        self._current_surface_home_state = self._capture_axes_state(ax_s)
        self.editor.canvas.draw_idle()

        cf = self.contour_fig
        cf.clear()
        cf.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.14)
        ax_c = cf.add_subplot(111)
        ax_c.set_aspect("equal")
        contour = ax_c.contourf(Xg, Yg, Z, levels=24, cmap="jet")
        ax_c.set_title(f"Contour - {title}")
        ax_c.set_xlabel("x1")
        ax_c.set_ylabel("x2")
        cbar = cf.colorbar(contour, ax=ax_c, fraction=0.046, pad=0.04)
        style_compact_axes(ax_c)
        style_compact_colorbar(cbar)
        ax_c.set_anchor("C")  # keep the square contour horizontally centred
        style_figure(cf, palette)
        self._current_contour_home_state = self._capture_axes_state(ax_c)
        self.contour_canvas.draw_idle()
        if switch_tab:
            self.visualization_tabs.setCurrentIndex(0)

    def apply_palette(self, palette: dict) -> None:
        """Recolor the matplotlib canvases to match the active theme.

        Qt's stylesheet does not reach matplotlib figures, so the Builder's
        surface and contour canvases are recolored here whenever the theme
        changes (the comparison thumbnails are rebuilt below).
        """
        self._palette = palette
        for fig, canvas in (
            (self.editor.fig, self.editor.canvas),
            (getattr(self, "contour_fig", None), getattr(self, "contour_canvas", None)),
        ):
            if fig is None or canvas is None:
                continue
            style_figure(fig, palette)
            canvas.draw_idle()
        for panel in getattr(self, "_cmp", {}).values():
            panel["switch"].set_palette(palette)
        if hasattr(self, "visualization_tabs"):
            text = self.visualization_tabs.tabText(self.visualization_tabs.currentIndex())
            if text == "Visualize Instances":
                self._refresh_comparison_grid("blocks")
            elif text == "Visualize Components":
                self._refresh_comparison_grid("components")
