"""Component editor widgets for PORTAL Studio."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

import numpy as np

try:
        from PySide6 import QtCore, QtWidgets
        from PySide6.QtWidgets import (
            QMainWindow,
            QWidget,
            QGridLayout,
            QSplitter,
            QComboBox,
            QDoubleSpinBox,
            QSpinBox,
            QLabel,
            QPushButton,
            QTableWidget,
            QTableWidgetItem,
            QListWidget,
            QListWidgetItem,
            QPlainTextEdit,
            QLineEdit,
            QFormLayout,
            QVBoxLayout,
            QHBoxLayout,
            QCheckBox,
            QFileDialog,
        )
except Exception:  # pragma: no cover - optional GUI dependency
        QtWidgets = None
        QMainWindow = object
        QWidget = object

import os
import sys

from portal_core.transform_registry import TRANSFORM_DISPLAY_NAMES
from portal_studio.validation import validate_component_config
from portal_studio.gui.common.angles import DegreeSpinBox, display_degrees


class CleanDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that displays clean numbers without trailing zeros."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Allow high precision input (up to 8 decimal places)
        self.setDecimals(8)
        # Remove up/down arrows
        self.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        # Enable keyboard tracking for proper editing
        self.setKeyboardTracking(True)
        # Don't grab the mouse wheel just by hovering: the value only changes
        # when the user clicks into the field and types.
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        # Keep the number fields compact instead of stretching across the cell.
        self.setMaximumWidth(78)
        self.setMinimumWidth(44)

    def wheelEvent(self, event):
        # Ignore scroll so the value never changes on hover-scroll; let the
        # event propagate so the surrounding scroll area scrolls instead.
        event.ignore()

    def textFromValue(self, value: float) -> str:
        """Override to display clean numbers without unnecessary trailing zeros."""
        # Format with full precision, then strip trailing zeros
        text = f"{value:.8f}".rstrip('0').rstrip('.')
        # If it's just an integer, show at least one decimal place for clarity
        if '.' not in text and value != int(value):
            text = f"{value:.1f}"
        return text

    def valueFromText(self, text: str) -> float:
        """Override to parse user input properly."""
        try:
            return float(text)
        except ValueError:
            return 0.0


class ScientificNotationLineEdit(QLineEdit):
    """QLineEdit that auto-formats numeric input to scientific notation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Keep it as compact as the spin boxes so it doesn't widen the panel.
        self.setMaximumWidth(78)
        self.setMinimumWidth(44)
        self.editingFinished.connect(self._format_to_scientific)

    def _format_to_scientific(self):
        """Convert entered value to scientific notation."""
        text = self.text().strip()
        if not text:
            return
        try:
            value = float(text)
            # Format as scientific notation with 6 decimal places for precision (e.g., 5.0e+01, 1.234567e-12)
            self.setText(f"{value:.6e}")
        except ValueError:
            # If not a valid number, clear the field
            pass

    def get_value(self) -> float:
        """Get the numeric value from the field."""
        try:
            return float(self.text())
        except ValueError:
            return 0.0


# Support package imports and direct-script execution.
try:
    from .controller import PortalController, gui_cfg_to_instance
    from .._io import load_config_json, save_config_json
    from ..common.plot_style import style_compact_axes, style_compact_colorbar
    from ..common.theme import LIGHT
except Exception:
    # Add the package directory to sys.path for absolute imports.
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(here, "..", ".."))  # portal_studio package directory
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from portal_studio.adapters.gui_to_instance import PortalController, gui_cfg_to_instance
    from portal_studio._io import load_config_json, save_config_json
    from portal_studio.gui.common.plot_style import (
        style_compact_axes,
        style_compact_colorbar,
    )
    from portal_studio.gui.common.theme import LIGHT


class ComponentEditorWidget(QWidget):  # pragma: no cover - UI exercised manually
    _TRANSFORM_DISPLAY_NAMES = dict(TRANSFORM_DISPLAY_NAMES)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        if QtWidgets is None:
            raise RuntimeError("PySide6 is required to run the GUI application")


        self.controller = PortalController()
        self.d = 2
        self.cbar = None  # Store colorbar reference
        self.editing_component_idx = None  # Track which component is being edited

        # Central layout: left editor, right list+plot stacked vertically
        top_layout = QGridLayout(self)

        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        top_layout.addWidget(splitter, 0, 0)

        # Left panel
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        splitter.addWidget(self.left_panel)

        # Baseline & params
        self._build_baseline_params()
        # Transform panel
        self._build_transform_panel()
        # Add/Reset row
        self._build_actions_row()

        # Right panel container
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        splitter.addWidget(self.right_panel)
        # Component list and plot buttons
        self._build_component_list_panel()
        # Plot panel
        self._build_plot_panel()

        splitter.setSizes([400, 1000])  # Left panel smaller: 400 for params, 1000 for plots/components

    # -------------------- UI Builders --------------------
    def _build_baseline_params(self) -> None:
        box = QWidget()
        self.baseline_params_panel = box
        grid = QGridLayout(box)
        row = 0

        # Section header
        header_label = QLabel("Baseline Parameters")
        header_label.setStyleSheet(f"""
            font-weight: bold;
            font-size: 13pt;
            color: white;
            background-color: {LIGHT['legacy_header_baseline']};
            padding: 10px;
            border-radius: 5px;
            margin: 5px;
        """)
        header_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(header_label, row, 0, 1, 4)
        row += 1

        # Baseline selection (spans both columns)
        self.baseline_combo = QComboBox()
        self.baseline_combo.addItems(["A", "B"])
        self.baseline_combo.setCurrentText("A")
        self.baseline_combo.setMinimumWidth(0)
        self.baseline_combo.currentTextChanged.connect(self._on_baseline_changed)
        grid.addWidget(QLabel("Baseline:"), row, 0)
        grid.addWidget(self.baseline_combo, row, 1)
        row += 1

        # Centers c_x, c_y (2 per row)
        self.center_x = CleanDoubleSpinBox(); self.center_x.setRange(-1e6, 1e6)
        self.center_y = CleanDoubleSpinBox(); self.center_y.setRange(-1e6, 1e6)
        grid.addWidget(QLabel("Center X:"), row, 0)
        grid.addWidget(self.center_x, row, 1)
        grid.addWidget(QLabel("Center Y:"), row, 2)
        grid.addWidget(self.center_y, row, 3)
        row += 1

        # Beta and Rotation (2 per row)
        self.beta_spin = CleanDoubleSpinBox(); self.beta_spin.setRange(-1e6, 1e6)
        self.rotation_spin = DegreeSpinBox()
        self.rotation_spin.set_degrees(0.0)
        self.rotation_spin.valueChanged.connect(self._on_rotation_changed)
        grid.addWidget(QLabel("Beta (β):"), row, 0)
        grid.addWidget(self.beta_spin, row, 1)
        grid.addWidget(QLabel("Rotation (deg):"), row, 2)
        grid.addWidget(self.rotation_spin, row, 3)
        row += 1

        # Kappa plus X,Y (2 per row)
        self.kappa_plus_x = CleanDoubleSpinBox(); self.kappa_plus_x.setRange(1e-8, 1e6); self.kappa_plus_x.setValue(10.0)
        self.kappa_plus_y = CleanDoubleSpinBox(); self.kappa_plus_y.setRange(1e-8, 1e6); self.kappa_plus_y.setValue(10.0)
        grid.addWidget(QLabel("κ⁺ X:"), row, 0)
        grid.addWidget(self.kappa_plus_x, row, 1)
        grid.addWidget(QLabel("κ⁺ Y:"), row, 2)
        grid.addWidget(self.kappa_plus_y, row, 3)
        row += 1

        # Kappa minus X,Y (2 per row)
        self.kappa_minus_x = CleanDoubleSpinBox(); self.kappa_minus_x.setRange(1e-8, 1e6); self.kappa_minus_x.setValue(10.0)
        self.kappa_minus_y = CleanDoubleSpinBox(); self.kappa_minus_y.setRange(1e-8, 1e6); self.kappa_minus_y.setValue(10.0)
        grid.addWidget(QLabel("κ⁻ X:"), row, 0)
        grid.addWidget(self.kappa_minus_x, row, 1)
        grid.addWidget(QLabel("κ⁻ Y:"), row, 2)
        grid.addWidget(self.kappa_minus_y, row, 3)
        row += 1

        # Exponents p+ X,Y (2 per row) - Form A only
        self.p_plus_x = CleanDoubleSpinBox(); self.p_plus_x.setRange(0.01, 1e6); self.p_plus_x.setValue(1.0)
        self.p_plus_y = CleanDoubleSpinBox(); self.p_plus_y.setRange(0.01, 1e6); self.p_plus_y.setValue(1.0)
        self.p_plus_x_label = QLabel("p⁺ X:")
        self.p_plus_y_label = QLabel("p⁺ Y:")
        grid.addWidget(self.p_plus_x_label, row, 0)
        grid.addWidget(self.p_plus_x, row, 1)
        grid.addWidget(self.p_plus_y_label, row, 2)
        grid.addWidget(self.p_plus_y, row, 3)
        row += 1

        # Exponents p- X,Y (2 per row) - Form A only
        self.p_minus_x = CleanDoubleSpinBox(); self.p_minus_x.setRange(0.01, 1e6); self.p_minus_x.setValue(1.0)
        self.p_minus_y = CleanDoubleSpinBox(); self.p_minus_y.setRange(0.01, 1e6); self.p_minus_y.setValue(1.0)
        self.p_minus_x_label = QLabel("p⁻ X:")
        self.p_minus_y_label = QLabel("p⁻ Y:")
        grid.addWidget(self.p_minus_x_label, row, 0)
        grid.addWidget(self.p_minus_x, row, 1)
        grid.addWidget(self.p_minus_y_label, row, 2)
        grid.addWidget(self.p_minus_y, row, 3)
        row += 1

        # p (single) - Form B only
        self.p_single = CleanDoubleSpinBox(); self.p_single.setRange(0.01, 1e6); self.p_single.setValue(0.5)  # Form B default exponent
        self.p_single_label = QLabel("p (single):")
        grid.addWidget(self.p_single_label, row, 0)
        grid.addWidget(self.p_single, row, 1)
        row += 1

        # Neutralization section
        self.neutral_enable = QCheckBox("Enable neutralization"); self.neutral_enable.setChecked(True)
        grid.addWidget(self.neutral_enable, row, 0, 1, 4)
        row += 1

        # Delta and r_ref (2 per row)
        self.neutral_delta = CleanDoubleSpinBox(); self.neutral_delta.setRange(1e-8, 1e9); self.neutral_delta.setValue(100.0)
        self.neutral_rref = CleanDoubleSpinBox(); self.neutral_rref.setRange(1e-8, 1e9); self.neutral_rref.setValue(100.0)
        grid.addWidget(QLabel("Delta:"), row, 0)
        grid.addWidget(self.neutral_delta, row, 1)
        grid.addWidget(QLabel("r_ref:"), row, 2)
        grid.addWidget(self.neutral_rref, row, 3)

        self.left_layout.addWidget(box)
        self._refresh_params_to_controller()
        # Initialize visibility based on default baseline (A)
        self._on_baseline_changed("A")

    def _build_transform_panel(self) -> None:
        panel = QWidget()
        self.transform_panel = panel
        grid = QGridLayout(panel)

        # Section header
        header_label = QLabel("Transformation Configuration")
        header_label.setStyleSheet(f"""
            font-weight: bold;
            font-size: 13pt;
            color: white;
            background-color: {LIGHT['legacy_header_transform']};
            padding: 10px;
            border-radius: 5px;
            margin: 5px;
        """)
        header_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(header_label, 0, 0, 1, 2)

        # Selection
        grid.addWidget(QLabel("Type"), 1, 0)
        self.transform_combo = QComboBox()
        for internal, display in self._TRANSFORM_DISPLAY_NAMES.items():
            self.transform_combo.addItem(display, internal)
        # Let the combo shrink with the panel; the popup still lists full names.
        self.transform_combo.setMinimumWidth(0)
        self.transform_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.transform_combo.setMinimumContentsLength(4)
        self.transform_combo.currentIndexChanged.connect(self._on_transform_type_changed)
        grid.addWidget(self.transform_combo, 1, 1)

        # Dynamic parameter panel.
        self.transform_params_panel = QWidget()
        self.transform_params_layout = QGridLayout(self.transform_params_panel)
        self.transform_params_layout.setColumnStretch(1, 1)
        self.transform_params_layout.setColumnStretch(3, 1)
        grid.addWidget(self.transform_params_panel, 2, 0, 1, 2)

        # Store current transform param widgets
        self.transform_param_widgets = {}

        # Buttons
        self.btn_add_transform = QPushButton("Add Transformation")
        self.btn_add_transform.setProperty("variant", "legacySuccess")
        self.btn_add_transform.clicked.connect(self._on_add_transform)
        grid.addWidget(self.btn_add_transform, 3, 0, 1, 2)

        # Transform cards
        self.transform_list = QListWidget()
        self.transform_list.setSpacing(8)
        self.transform_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.transform_list.setStyleSheet(f"""
            QListWidget {{
                background: {LIGHT['legacy_transform_list_bg']};
                border: 1px solid {LIGHT['border']};
                border-radius: 6px;
                padding: 6px;
            }}
            QListWidget::item {{
                border: none;
                margin: 3px;
            }}
        """)
        grid.addWidget(self.transform_list, 4, 0, 1, 2)

        # Transform cards expose these actions inline, so this row stays
        # hidden; the controls are kept only for callers that reference them.
        ctrl_row = QWidget(); h = QHBoxLayout(ctrl_row)
        self.transform_controls_row = ctrl_row
        self.btn_delete = QPushButton("Delete")
        self.btn_up = QPushButton("Move Up")
        self.btn_down = QPushButton("Move Down")
        self.btn_edit = QPushButton("Edit")
        h.addWidget(self.btn_delete); h.addWidget(self.btn_up); h.addWidget(self.btn_down); h.addWidget(self.btn_edit)
        self.btn_delete.clicked.connect(self._on_delete_transform)
        self.btn_up.clicked.connect(self._on_move_up)
        self.btn_down.clicked.connect(self._on_move_down)
        self.btn_edit.clicked.connect(self._on_edit_transform)
        ctrl_row.hide()
        grid.addWidget(ctrl_row, 5, 0, 1, 2)

        self.left_layout.addWidget(panel)
        # Initialize with default transform type fields
        self._build_transform_param_fields(self._current_transform_name())

    def _build_actions_row(self) -> None:
        row = QWidget(); h = QHBoxLayout(row)
        self.legacy_actions_row = row
        self.btn_add_component = QPushButton("➤ Add Current Component")
        self.btn_update_component = QPushButton("✓ Update Component")
        self.btn_update_component.setProperty("variant", "legacyWarning")
        self.btn_update_component.clicked.connect(self._on_update_component)
        self.btn_update_component.setVisible(False)  # Hidden by default
        self.btn_reset = QPushButton("⟳ Reset")
        self.btn_add_component.clicked.connect(self._on_add_component)
        self.btn_reset.clicked.connect(self._on_reset)
        h.addWidget(self.btn_add_component)
        h.addWidget(self.btn_update_component)
        h.addWidget(self.btn_reset)
        self.left_layout.addWidget(row)

    def _build_component_list_panel(self) -> None:
        box = QWidget(); v = QVBoxLayout(box)
        self.legacy_components_panel = box
        head = QWidget(); h = QHBoxLayout(head)
        self.btn_plot_selected = QPushButton("Plot Selected Component")
        self.btn_plot_selected.clicked.connect(self._on_plot_selected)
        self.btn_save = QPushButton("Save Components")
        self.btn_save.clicked.connect(self._on_save_components)
        self.btn_load = QPushButton("Load Components")
        self.btn_load.clicked.connect(self._on_load_components)
        self.btn_validate = QPushButton("Validate Components")
        self.btn_validate.clicked.connect(self._on_validate_components)
        h.addWidget(self.btn_plot_selected)
        h.addWidget(self.btn_save)
        h.addWidget(self.btn_load)
        h.addWidget(self.btn_validate)
        v.addWidget(head)

        # Combine Select+ID into one column (radio button + ID label)
        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([
            "Select/ID", "Baseline", "Center", "Beta", "Kappa⁺", "Kappa⁻",
            "P", "Rotation(deg)", "Transforms", "Edit", "Delete",
        ])
        # Make Select/ID column wider and auto-size other columns to contents
        self.table.setColumnWidth(0, 120)
        header = self.table.horizontalHeader()
        try:
            # Allow manual resizing (Interactive) while keeping Select/ID fixed
            header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
            # Stretch the last section to avoid large blank gaps
            header.setStretchLastSection(True)
            # Center header text for readability
            header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        except Exception:
            # Keep default header sizing when QHeaderView is unavailable.
            pass
        v.addWidget(self.table)
        # Add component list with stretch factor 1 (plot has stretch 2, so plot is 2x bigger)
        self.right_layout.addWidget(box, stretch=1)

    def hide_legacy_panel(self) -> None:
        """Hide the standalone component table and action rows."""
        if hasattr(self, "legacy_components_panel"):
            self.legacy_components_panel.hide()
        if hasattr(self, "legacy_actions_row"):
            self.legacy_actions_row.hide()

    def _build_plot_panel(self) -> None:
        # Import Matplotlib when constructing the plot panel.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        # Slightly larger figure and use more canvas area (less margins)
        self.fig = Figure(figsize=(12, 8))
        # Use tighter margins so plots use blank space near borders
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        self.canvas = FigureCanvas(self.fig)
        # Use a GridSpec with width ratios so both plots are larger and use border space
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.20)
        self.ax_surface = self.fig.add_subplot(gs[0], projection="3d")
        self.ax_contour = self.fig.add_subplot(gs[1])
        # Set contour plot to square aspect ratio
        self.ax_contour.set_aspect('equal')
        # Add plot with stretch factor to make it take more vertical space
        self.right_layout.addWidget(self.canvas, stretch=2)

    # -------------------- Event Handlers --------------------
    def _on_baseline_changed(self, text: str) -> None:
        self.controller.set_baseline(text)
        # Show/hide p parameters based on baseline
        if text == "A":
            # Form A: show p_plus/p_minus, hide p_single
            self.p_plus_x_label.show(); self.p_plus_x.show()
            self.p_plus_y_label.show(); self.p_plus_y.show()
            self.p_minus_x_label.show(); self.p_minus_x.show()
            self.p_minus_y_label.show(); self.p_minus_y.show()
            self.p_single_label.hide(); self.p_single.hide()
        else:
            # Form B: hide p_plus/p_minus, show p_single
            self.p_plus_x_label.hide(); self.p_plus_x.hide()
            self.p_plus_y_label.hide(); self.p_plus_y.hide()
            self.p_minus_x_label.hide(); self.p_minus_x.hide()
            self.p_minus_y_label.hide(); self.p_minus_y.hide()
            self.p_single_label.show(); self.p_single.show()
        self._refresh_params_to_controller()

    def _on_rotation_changed(self, val: float) -> None:
        self.controller.set_rotation_deg(self.rotation_spin.degrees_value())

    def _on_add_transform(self) -> None:
        name = self._current_transform_name()
        try:
            params = self._extract_transform_params_from_widgets()
            self.controller.add_transform(name, params, d=self.d)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Transform Error", str(e))
            return
        self._refresh_transform_list()

    def _on_delete_transform(self) -> None:
        idx = self.transform_list.currentRow()
        if idx < 0:
            return
        try:
            self.controller.delete_transform(idx)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Delete Error", str(e))
            return
        self._refresh_transform_list()

    def _on_move_up(self) -> None:
        idx = self.transform_list.currentRow()
        if idx < 0:
            return
        self.controller.move_transform_up(idx)
        self._refresh_transform_list()
        self.transform_list.setCurrentRow(max(0, idx - 1))

    def _on_move_down(self) -> None:
        idx = self.transform_list.currentRow()
        if idx < 0:
            return
        self.controller.move_transform_down(idx)
        self._refresh_transform_list()
        self.transform_list.setCurrentRow(min(self.transform_list.count() - 1, idx + 1))

    def _on_edit_transform(self) -> bool:
        idx = self.transform_list.currentRow()
        if idx < 0:
            return False
        name = self._current_transform_name()
        try:
            params = self._extract_transform_params_from_widgets()
            self.controller.edit_transform(idx, name=name, params=params)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Edit Error", str(e))
            return False
        self._refresh_transform_list()
        return True

    def _on_add_component(self) -> None:
        # Single component entry reflecting current editor state
        # Ensure controller state matches current UI inputs before snapshot
        self._refresh_params_to_controller()
        cfg = self.controller.build_cfg()
        # Snapshot into controller-managed components list
        comp_idx = self.controller.add_component_from_current()
        p = cfg.get("params", {})
        center = [float(x) for x in p.get("c", [0.0, 0.0])]
        beta = float(p.get("beta", 0.0))
        kp = [float(x) for x in p.get("kappa_plus", [0.0, 0.0])]
        km = [float(x) for x in p.get("kappa_minus", [0.0, 0.0])]
        # P-Params summary
        if self.controller.state.baseline == "A":
            pp = p.get("p_plus", [1.0, 1.0]); pm = p.get("p_minus", [1.0, 1.0])
            p_summary = f"p⁺={pp}, p⁻={pm}"
        else:
            p_summary = f"p={p.get('p_single', 1.0)}"
        # Transforms summary
        t_summary = ", ".join([t["name"] for t in self.controller.state.T_seq])
        if not t_summary:
            t_summary = "None"

        row = self.table.rowCount()
        self.table.insertRow(row)

        # Add radio button in first column
        radio_widget = QWidget()
        radio_layout = QHBoxLayout(radio_widget)
        radio_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        radio_layout.setContentsMargins(0, 0, 0, 0)
        radio_btn = QtWidgets.QRadioButton()
        if row == 0:  # Select first row by default
            radio_btn.setChecked(True)
        radio_btn.toggled.connect(lambda checked, r=row: self._on_radio_selected(r, checked))
        radio_layout.addWidget(radio_btn)
        # Add ID label next to radio button in same cell
        id_label = QLabel(str(comp_idx + 1))
        id_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        radio_layout.addWidget(id_label)
        self.table.setCellWidget(row, 0, radio_widget)

        items = [
            QTableWidgetItem(self.controller.state.baseline),
            QTableWidgetItem(str(center)),
            QTableWidgetItem(str(beta)),
            QTableWidgetItem(str(kp)),
            QTableWidgetItem(str(km)),
            QTableWidgetItem(p_summary),
            QTableWidgetItem(display_degrees(self.controller.state.rotation_deg)),
            QTableWidgetItem(t_summary),
        ]
        for col, it in enumerate(items):
            it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, col + 1, it)  # Shift by 1 for radio button column
        # Clickable Edit/Delete controls
        btn_edit = QPushButton("Edit")
        btn_edit.clicked.connect(lambda _=False, r=row: self._on_edit_component_row(r))
        self.table.setCellWidget(row, 9, btn_edit)
        btn_del = QPushButton("Delete")
        btn_del.clicked.connect(lambda _=False, r=row: self._on_delete_component_row(r))
        self.table.setCellWidget(row, 10, btn_del)

    def _on_update_component(self) -> None:
        """Update the component being edited by deleting old and adding new."""
        if self.editing_component_idx is None:
            return

        # Delete the old component
        try:
            self.controller.state.components.pop(self.editing_component_idx)
        except Exception:
            pass

        # Ensure controller state matches UI before adding the edited component
        self._refresh_params_to_controller()
        # Add the edited component as new
        self.controller.add_component_from_current()

        # Clear edit mode
        self.editing_component_idx = None
        self.btn_update_component.setVisible(False)
        self.btn_add_component.setVisible(True)

        # Refresh table and remove highlighting
        self._refresh_component_table()

    def _on_radio_selected(self, row: int, checked: bool) -> None:
        """Handle radio button selection - ensure only one is selected."""
        if checked:
            # Uncheck all other radio buttons
            for r in range(self.table.rowCount()):
                if r != row:
                    widget = self.table.cellWidget(r, 0)
                    if widget:
                        radio = widget.findChild(QtWidgets.QRadioButton)
                        if radio and radio.isChecked():
                            radio.setChecked(False)

    def _on_reset(self) -> None:
        # Reset the editor inputs and transform sequence to defaults while
        # preserving existing components.
        self.baseline_combo.setCurrentText("A")
        self.rotation_spin.set_degrees(0.0)
        self.neutral_enable.setChecked(True)
        self.neutral_delta.setValue(100.0)
        self.neutral_rref.setValue(100.0)
        self.center_x.setValue(0.0); self.center_y.setValue(0.0)
        self.beta_spin.setValue(0.0)
        self.kappa_plus_x.setValue(10.0); self.kappa_plus_y.setValue(10.0)
        self.kappa_minus_x.setValue(10.0); self.kappa_minus_y.setValue(10.0)
        self.p_plus_x.setValue(1.0); self.p_plus_y.setValue(1.0)
        self.p_minus_x.setValue(1.0); self.p_minus_y.setValue(1.0)
        self.p_single.setValue(0.5)        # Form B default exponent
        # Clear transform list UI and controller transforms only
        self.transform_list.clear()
        self.controller.state.T_seq = []
        # Reset transform type selector to first item and rebuild param widgets
        try:
            self.transform_combo.setCurrentIndex(0)
        except Exception:
            pass
        # Rebuild transform parameter input fields to their default values
        self._build_transform_param_fields(self._current_transform_name())
        # Clear edit mode but keep component list intact
        self.editing_component_idx = None
        self.btn_update_component.setVisible(False)
        self.btn_add_component.setVisible(True)
        # Push these default params into the controller so editor and controller are synced
        self._refresh_params_to_controller()
        # Refresh transform UI and component table to reflect preserved components
        self._refresh_transform_list()
        self._refresh_component_table()

    def _on_plot_selected(self) -> None:
        # Compute on grid and plot surface and contour
        # Ensure controller state reflects UI for the current editor
        self._refresh_params_to_controller()
        # Build simple grid
        xs = np.linspace(-100.0, 100.0, 100)
        ys = np.linspace(-100.0, 100.0, 100)
        Xg, Yg = np.meshgrid(xs, ys)
        A = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
        # Determine selected component from radio button
        selected_row = None
        for r in range(self.table.rowCount()):
            widget = self.table.cellWidget(r, 0)
            if widget:
                radio = widget.findChild(QtWidgets.QRadioButton)
                if radio and radio.isChecked():
                    selected_row = r
                    break

        if selected_row is not None:
            # Get component ID from selected row
            widget = self.table.cellWidget(selected_row, 0)
            if widget:
                lbl = widget.findChild(QLabel)
                try:
                    comp_id = int(lbl.text())
                    indices = [comp_id - 1]
                except Exception:
                    indices = None
            else:
                indices = None
        else:
            indices = None

        # Compute aggregated result across selected components
        y = self.controller.compute_aggregated_points(A, indices)
        Z = y.reshape(Xg.shape)
        # Clear figure completely and recreate subplots to avoid stacking colorbars
        self.fig.clear()
        # Recreate with larger figure margins so subplots use available canvas
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        # Recreate with GridSpec so both plots are larger and use border space
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.20)
        self.ax_surface = self.fig.add_subplot(gs[0], projection="3d")
        self.ax_contour = self.fig.add_subplot(gs[1])

        # Plot surface
        surf = self.ax_surface.plot_surface(Xg, Yg, Z, cmap="jet")
        self.ax_surface.set_title("Surface Plot")
        self.ax_surface.set_xlabel("x1"); self.ax_surface.set_ylabel("x2"); self.ax_surface.set_zlabel("f")
        # Keep both axes showing -100 on the lower side.
        self.ax_surface.view_init(elev=30, azim=225)
        style_compact_axes(self.ax_surface)

        # Plot contour
        contour = self.ax_contour.contourf(Xg, Yg, Z, levels=20, cmap="jet")
        self.ax_contour.set_title("Contour Plot")
        self.ax_contour.set_xlabel("x1"); self.ax_contour.set_ylabel("x2")
        # Set square aspect ratio for contour plot
        self.ax_contour.set_aspect('equal')
        # Add colorbar to contour plot
        self.cbar = self.fig.colorbar(contour, ax=self.ax_contour, fraction=0.046, pad=0.04)
        style_compact_axes(self.ax_contour)
        style_compact_colorbar(self.cbar)
        self.canvas.draw_idle()

    def _on_save_components(self) -> None:
        # Save current components list to a JSON file
        path, _ = QFileDialog.getSaveFileName(self, "Save Components", "", "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            obj = {"components": self.controller.get_components()}
            for config in obj['components']:
                validate_component_config(config)
            save_config_json(path, obj)
            QtWidgets.QMessageBox.information(self, "Saved", f"Components saved to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def _on_load_components(self) -> None:
        # Load components list from a JSON file
        path, _ = QFileDialog.getOpenFileName(self, "Load Components", "", "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            obj = load_config_json(path)
            comps = obj.get("components", None)
            if not isinstance(comps, list):
                raise ValueError("JSON must contain a 'components' list")
            for config in comps:
                validate_component_config(config)
            self.controller.state.components = copy.deepcopy(comps)
            self._refresh_component_table()
            QtWidgets.QMessageBox.information(self, "Loaded", f"Loaded {len(comps)} components from {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", str(e))

    def _on_validate_components(self) -> None:
        # Validate each loaded component by parsing models and checking shapes
        try:
            comps = self.controller.get_components()
            if not comps:
                QtWidgets.QMessageBox.information(self, "Validate", "No components to validate.")
                return
            for i, cfg in enumerate(comps):
                # Build the instance dict; invalid parameters raise ValueError.
                gui_cfg_to_instance(cfg, self.d)
            QtWidgets.QMessageBox.information(self, "Validate", f"All {len(comps)} components valid for d={self.d}.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Validation Error", str(e))

    # -------------------- Synchronization --------------------
    def _refresh_params_to_controller(self) -> None:
        self.controller.set_baseline(self.baseline_combo.currentText())
        self.controller.set_rotation_deg(self.rotation_spin.degrees_value())
        self.controller.set_neutralize(
            enable=self.neutral_enable.isChecked(),
            Delta=self.neutral_delta.value(),
            r_ref=self.neutral_rref.value(),
        )
        params = {
            "c": [self.center_x.value(), self.center_y.value()],
            "beta": self.beta_spin.value(),
            "kappa_plus": [self.kappa_plus_x.value(), self.kappa_plus_y.value()],
            "kappa_minus": [self.kappa_minus_x.value(), self.kappa_minus_y.value()],
            "p_plus": [self.p_plus_x.value(), self.p_plus_y.value()],
            "p_minus": [self.p_minus_x.value(), self.p_minus_y.value()],
            "p_single": self.p_single.value(),
        }
        self.controller.set_params_from_dict(params, d=self.d)

    def _refresh_transform_list(self) -> None:
        self.transform_list.clear()
        for idx, t in enumerate(self.controller.state.T_seq):
            item = QListWidgetItem()
            card = self._make_transform_card(idx, t)
            item.setSizeHint(card.sizeHint())
            self.transform_list.addItem(item)
            self.transform_list.setItemWidget(item, card)

    def _make_transform_card(self, idx: int, transform: Dict[str, Any]) -> QWidget:
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget {{
                background: {LIGHT['surface']};
                border: 1px solid {LIGHT['legacy_transform_card_border']};
                border-radius: 6px;
            }}
            QLabel {{
                border: none;
                background: transparent;
            }}
            QPushButton {{
                padding: 4px 7px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        title = QLabel(self._transform_display_name(str(transform.get("name", ""))))
        title.setStyleSheet(f"font-weight: bold; color: {LIGHT['text']}; font-size: 11pt;")
        layout.addWidget(title)

        summary = QLabel(self._transform_summary(dict(transform.get("params", {}))))
        summary.setWordWrap(True)
        summary.setStyleSheet(f"color: {LIGHT['caption']};")
        layout.addWidget(summary)

        actions = QWidget()
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for text, handler in (
            ("Edit", lambda _=False, i=idx: self._edit_transform_at(i)),
            ("Delete", lambda _=False, i=idx: self._delete_transform_at(i)),
            ("Up", lambda _=False, i=idx: self._move_transform_at(i, -1)),
            ("Down", lambda _=False, i=idx: self._move_transform_at(i, 1)),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            row.addWidget(btn)
        row.addStretch(1)
        layout.addWidget(actions)
        return card

    @staticmethod
    def _transform_display_name(name: str) -> str:
        return ComponentEditorWidget._TRANSFORM_DISPLAY_NAMES.get(
            name, name.replace("_", " ").title())

    def _current_transform_name(self) -> str:
        data = self.transform_combo.currentData()
        if data is not None:
            return str(data)
        return self._transform_internal_name(self.transform_combo.currentText())

    def set_transform_type(self, name: str) -> None:
        internal = self._transform_internal_name(name)
        index = self.transform_combo.findData(internal)
        if index >= 0:
            self.transform_combo.setCurrentIndex(index)

    @classmethod
    def _transform_internal_name(cls, label: str) -> str:
        text = str(label)
        if text in cls._TRANSFORM_DISPLAY_NAMES:
            return text
        for internal, display in cls._TRANSFORM_DISPLAY_NAMES.items():
            if text == display:
                return internal
        return text

    @staticmethod
    def _format_param_value(value: Any) -> str:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            vals = []
            for item in value[:2]:
                try:
                    vals.append(f"{float(item):.3g}")
                except Exception:
                    vals.append(str(item))
            suffix = ", ..." if len(value) > 2 else ""
            return "[" + ", ".join(vals) + suffix + "]"
        try:
            return f"{float(value):.3g}"
        except Exception:
            return str(value)

    def _transform_summary(self, params: Dict[str, Any]) -> str:
        if not params:
            return "Default parameters"
        parts = []
        for key in sorted(params.keys()):
            label = (
                key.replace("mu", "mu")
                .replace("gamma", "gamma")
                .replace("omega", "omega")
                .replace("_plus", "+")
                .replace("_minus", "-")
                .replace("_", " ")
            )
            parts.append(f"{label}: {self._format_param_value(params[key])}")
            if len(parts) >= 4:
                break
        if len(params) > 4:
            parts.append("...")
        return "\n".join(parts)

    def _edit_transform_at(self, idx: int) -> None:
        self.transform_list.setCurrentRow(idx)
        self._on_edit_transform()

    def _delete_transform_at(self, idx: int) -> None:
        self.transform_list.setCurrentRow(idx)
        self._on_delete_transform()

    def _move_transform_at(self, idx: int, delta: int) -> None:
        self.transform_list.setCurrentRow(idx)
        if delta < 0:
            self._on_move_up()
        else:
            self._on_move_down()

    def load_component_config(self, cfg: Dict[str, Any]) -> None:
        """Load a saved component config into the editor without changing inventory."""
        validate_component_config(cfg)
        cfg = copy.deepcopy(cfg)
        baseline = str(cfg.get("baseline", "A"))
        self.baseline_combo.setCurrentText(baseline)
        self.rotation_spin.set_degrees(float(cfg.get("rotation_deg", 0.0)))
        neutral = dict(cfg.get("neutralize", {}))
        self.neutral_enable.setChecked(bool(neutral.get("enable", False)))
        self.neutral_delta.setValue(float(neutral.get("Delta", 100.0)))
        self.neutral_rref.setValue(float(neutral.get("r_ref", 100.0)))

        p = dict(cfg.get("params", {}))
        center = list(p.get("c", [0.0, 0.0]))
        self.center_x.setValue(float(center[0]))
        self.center_y.setValue(float(center[1]))
        self.beta_spin.setValue(float(p.get("beta", 0.0)))
        kp = list(p.get("kappa_plus", [10.0, 10.0]))
        km = list(p.get("kappa_minus", [10.0, 10.0]))
        self.kappa_plus_x.setValue(float(kp[0]))
        self.kappa_plus_y.setValue(float(kp[1]))
        self.kappa_minus_x.setValue(float(km[0]))
        self.kappa_minus_y.setValue(float(km[1]))
        if baseline == "A":
            pp = list(p.get("p_plus", [1.0, 1.0]))
            pm = list(p.get("p_minus", [1.0, 1.0]))
            self.p_plus_x.setValue(float(pp[0]))
            self.p_plus_y.setValue(float(pp[1]))
            self.p_minus_x.setValue(float(pm[0]))
            self.p_minus_y.setValue(float(pm[1]))
        else:
            self.p_single.setValue(float(p.get("p_single", 1.0)))
        self.controller.state.T_seq = list(cfg.get("T_seq", []))
        self._refresh_params_to_controller()
        self._refresh_transform_list()

    def reset_editor(self) -> None:
        """Public wrapper used by BuilderTab for the Create New Component action."""
        self._on_reset()

    def _refresh_component_table(self) -> None:
        # Rebuild the components table from the controller state
        self.table.setRowCount(0)
        comps = self.controller.get_components()
        for idx, cfg in enumerate(comps):
            p = dict(cfg.get("params", {}))
            center = [float(x) for x in p.get("c", [0.0, 0.0])]
            beta = float(p.get("beta", 0.0))
            kp = [float(x) for x in p.get("kappa_plus", [0.0, 0.0])]
            km = [float(x) for x in p.get("kappa_minus", [0.0, 0.0])]
            baseline = str(cfg.get("baseline", "A"))
            rot = float(cfg.get("rotation_deg", 0.0))
            t_seq = list(cfg.get("T_seq", []))
            if baseline.upper() == "A":
                pp = p.get("p_plus", [1.0, 1.0]); pm = p.get("p_minus", [1.0, 1.0])
                p_summary = f"p⁺={pp}, p⁻={pm}"
            else:
                p_summary = f"p={p.get('p_single', 1.0)}"
            t_summary = ", ".join([t.get("name", "?") for t in t_seq])
            if not t_summary:
                t_summary = "None"
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Add radio button in first column
            radio_widget = QWidget()
            radio_layout = QHBoxLayout(radio_widget)
            radio_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            radio_layout.setContentsMargins(0, 0, 0, 0)
            radio_btn = QtWidgets.QRadioButton()
            if row == 0:  # Select first row by default
                radio_btn.setChecked(True)
            radio_btn.toggled.connect(lambda checked, r=row: self._on_radio_selected(r, checked))
            radio_layout.addWidget(radio_btn)
            # Add ID label next to radio button in same cell
            id_label = QLabel(str(idx + 1))
            id_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            radio_layout.addWidget(id_label)
            self.table.setCellWidget(row, 0, radio_widget)

            items = [
                QTableWidgetItem(baseline),
                QTableWidgetItem(str(center)),
                QTableWidgetItem(str(beta)),
                QTableWidgetItem(str(kp)),
                QTableWidgetItem(str(km)),
                QTableWidgetItem(p_summary),
                QTableWidgetItem(display_degrees(rot)),
                QTableWidgetItem(t_summary),
            ]
            for col, it in enumerate(items):
                it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                # Highlight row if it's being edited with a soft pastel color
                if self.editing_component_idx == idx:
                    from PySide6.QtGui import QColor
                    it.setBackground(QColor(255, 235, 205))  # Soft peach/apricot color
                self.table.setItem(row, col + 1, it)  # Shift by 1 for radio button column
            # Buttons for Edit/Delete
            btn_edit = QPushButton("Edit")
            btn_edit.clicked.connect(lambda _=False, r=row: self._on_edit_component_row(r))
            self.table.setCellWidget(row, 9, btn_edit)
            btn_del = QPushButton("Delete")
            btn_del.clicked.connect(lambda _=False, r=row: self._on_delete_component_row(r))
            self.table.setCellWidget(row, 10, btn_del)

    def _on_transform_type_changed(self, _index=None) -> None:
        self._build_transform_param_fields(self._current_transform_name())

    def _build_transform_param_fields(self, transform_type: str) -> None:
        """Build individual parameter input fields based on transform type."""
        # Clear existing widgets
        while self.transform_params_layout.count():
            item = self.transform_params_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.transform_param_widgets.clear()

        row = 0
        if transform_type == "additive_periodic":
            # mu_plus X, Y
            self.transform_params_layout.addWidget(QLabel("μ⁺ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(1.5)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["mu_plus_x"] = w
            self.transform_params_layout.addWidget(QLabel("μ⁺ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(1.5)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["mu_plus_y"] = w
            row += 1
            # mu_minus X, Y
            self.transform_params_layout.addWidget(QLabel("μ⁻ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(1.5)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["mu_minus_x"] = w
            self.transform_params_layout.addWidget(QLabel("μ⁻ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(1.5)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["mu_minus_y"] = w
            row += 1
            # gamma_plus X, Y
            self.transform_params_layout.addWidget(QLabel("γ⁺ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.3)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["gamma_plus_x"] = w
            self.transform_params_layout.addWidget(QLabel("γ⁺ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.3)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["gamma_plus_y"] = w
            row += 1
            # gamma_minus X, Y
            self.transform_params_layout.addWidget(QLabel("γ⁻ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.3)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["gamma_minus_x"] = w
            self.transform_params_layout.addWidget(QLabel("γ⁻ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.3)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["gamma_minus_y"] = w
            row += 1
            # omega_plus X, Y
            self.transform_params_layout.addWidget(QLabel("ω⁺ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.15)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["omega_plus_x"] = w
            self.transform_params_layout.addWidget(QLabel("ω⁺ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.15)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["omega_plus_y"] = w
            row += 1
            # omega_minus X, Y
            self.transform_params_layout.addWidget(QLabel("ω⁻ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.15)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["omega_minus_x"] = w
            self.transform_params_layout.addWidget(QLabel("ω⁻ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.15)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["omega_minus_y"] = w

        elif transform_type == "log_sinusoidal":
            # mu_plus X, Y
            self.transform_params_layout.addWidget(QLabel("μ⁺ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.2)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["mu_plus_x"] = w
            self.transform_params_layout.addWidget(QLabel("μ⁺ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.2)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["mu_plus_y"] = w
            row += 1
            # mu_minus X, Y
            self.transform_params_layout.addWidget(QLabel("μ⁻ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.2)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["mu_minus_x"] = w
            self.transform_params_layout.addWidget(QLabel("μ⁻ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.2)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["mu_minus_y"] = w
            row += 1
            # omega1_plus X, Y
            self.transform_params_layout.addWidget(QLabel("ω1⁺ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(7.0)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["omega1_plus_x"] = w
            self.transform_params_layout.addWidget(QLabel("ω1⁺ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(7.0)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["omega1_plus_y"] = w
            row += 1
            # omega1_minus X, Y
            self.transform_params_layout.addWidget(QLabel("ω1⁻ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(7.0)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["omega1_minus_x"] = w
            self.transform_params_layout.addWidget(QLabel("ω1⁻ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(7.0)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["omega1_minus_y"] = w
            row += 1
            # omega2_plus X, Y
            self.transform_params_layout.addWidget(QLabel("ω2⁺ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(7.0)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["omega2_plus_x"] = w
            self.transform_params_layout.addWidget(QLabel("ω2⁺ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(7.0)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["omega2_plus_y"] = w
            row += 1
            # omega2_minus X, Y
            self.transform_params_layout.addWidget(QLabel("ω2⁻ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(7.0)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["omega2_minus_x"] = w
            self.transform_params_layout.addWidget(QLabel("ω2⁻ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(7.0)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["omega2_minus_y"] = w
            row += 1
            # sigma (origin-smoothing scale, Eq. 19)
            self.transform_params_layout.addWidget(QLabel("σ:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(0.0, 1e6); w.setValue(0.1)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["sigma"] = w

        elif transform_type == "wavelet_mod":
            # mu_plus X, Y
            self.transform_params_layout.addWidget(QLabel("μ⁺ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(50.0)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["mu_plus_x"] = w
            self.transform_params_layout.addWidget(QLabel("μ⁺ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(50.0)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["mu_plus_y"] = w
            row += 1
            # mu_minus X, Y
            self.transform_params_layout.addWidget(QLabel("μ⁻ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(50.0)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["mu_minus_x"] = w
            self.transform_params_layout.addWidget(QLabel("μ⁻ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(50.0)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["mu_minus_y"] = w
            row += 1
            # omega_plus X, Y
            self.transform_params_layout.addWidget(QLabel("ω⁺ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.8)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["omega_plus_x"] = w
            self.transform_params_layout.addWidget(QLabel("ω⁺ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.8)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["omega_plus_y"] = w
            row += 1
            # omega_minus X, Y
            self.transform_params_layout.addWidget(QLabel("ω⁻ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.8)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["omega_minus_x"] = w
            self.transform_params_layout.addWidget(QLabel("ω⁻ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.8)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["omega_minus_y"] = w
            row += 1
            # eta
            self.transform_params_layout.addWidget(QLabel("η:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(20.0)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["eta"] = w
            # eps (use ScientificNotationLineEdit for auto-formatting)
            self.transform_params_layout.addWidget(QLabel("eps:"), row, 2)
            w = ScientificNotationLineEdit(); w.setText("1e-12")
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["eps"] = w

        elif transform_type == "tensor_interference":
            # mu_plus X, Y  (the user-set value is the base amplitude mu_0)
            self.transform_params_layout.addWidget(QLabel("μ₀⁺ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(22.0)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["mu_plus_x"] = w
            self.transform_params_layout.addWidget(QLabel("μ₀⁺ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(22.0)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["mu_plus_y"] = w
            row += 1
            # mu_minus X, Y
            self.transform_params_layout.addWidget(QLabel("μ₀⁻ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(22.0)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["mu_minus_x"] = w
            self.transform_params_layout.addWidget(QLabel("μ₀⁻ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(22.0)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["mu_minus_y"] = w
            row += 1
            # omega_plus X, Y
            self.transform_params_layout.addWidget(QLabel("ω⁺ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.20)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["omega_plus_x"] = w
            self.transform_params_layout.addWidget(QLabel("ω⁺ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.20)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["omega_plus_y"] = w
            row += 1
            # omega_minus X, Y
            self.transform_params_layout.addWidget(QLabel("ω⁻ X:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.20)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["omega_minus_x"] = w
            self.transform_params_layout.addWidget(QLabel("ω⁻ Y:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.20)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["omega_minus_y"] = w

        elif transform_type == "radial_polytrig":
            # mu, p
            self.transform_params_layout.addWidget(QLabel("μ:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(1.8)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["mu"] = w
            self.transform_params_layout.addWidget(QLabel("p:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.5)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["p"] = w
            row += 1
            # q, omega
            self.transform_params_layout.addWidget(QLabel("q:"), row, 0)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(1.4)
            self.transform_params_layout.addWidget(w, row, 1)
            self.transform_param_widgets["q"] = w
            self.transform_params_layout.addWidget(QLabel("ω:"), row, 2)
            w = CleanDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setValue(0.05)
            self.transform_params_layout.addWidget(w, row, 3)
            self.transform_param_widgets["omega"] = w
            row += 1
            # eps (use ScientificNotationLineEdit for auto-formatting)
            self.transform_params_layout.addWidget(QLabel("ε:"), row, 0)
            w = ScientificNotationLineEdit(); w.setText("1e-12")
            self.transform_params_layout.addWidget(w, row, 1, 1, 3)
            self.transform_param_widgets["eps"] = w

        self._style_transform_param_labels()

    def _style_transform_param_labels(self) -> None:
        for label in self.transform_params_panel.findChildren(QLabel):
            display = self._transform_param_label_text(label.text())
            if display is None:
                continue
            label.setText(display)
            label.setTextFormat(QtCore.Qt.TextFormat.RichText)

    @staticmethod
    def _transform_param_label_text(text: str) -> str | None:
        raw = str(text).strip().rstrip(":")
        direct = {
            "σ": "σ",
            "Ïƒ": "σ",
            "ε": "eps",
            "Îµ": "eps",
            "η": "η",
            "Î·": "η",
            "μ": "μ",
            "Î¼": "μ",
            "p": "p",
            "q": "q",
            "ω": "ω",
            "Ï‰": "ω",
        }
        direct["eps"] = "eps"
        if raw in direct:
            return direct[raw]

        parts = raw.split()
        if len(parts) != 2 or parts[1] not in {"X", "Y"}:
            return None
        symbol = parts[0]
        axis = parts[1].lower()
        sign = ""
        plus_suffixes = ("⁺", "âº")
        minus_suffixes = ("⁻", "â»")
        if symbol.endswith(plus_suffixes):
            symbol = symbol[:-1]
            sign = "+"
        elif symbol.endswith(minus_suffixes):
            symbol = symbol[:-1]
            sign = "-"

        replacements = {
            "Î¼": "μ",
            "Î³": "γ",
            "Ï‰": "ω",
        }
        for bad, good in replacements.items():
            symbol = symbol.replace(bad, good)

        if symbol.startswith("ω") and len(symbol) > 1:
            base = f"ω<sub>{symbol[1:]}</sub>"
        else:
            base = symbol
        if sign:
            base += f"<sup>{sign}</sup>"
        return f"{base}<sub>{axis}</sub>"

    def _extract_transform_params_from_widgets(self) -> Dict[str, Any]:
        """Extract transform parameters from individual input fields."""
        params: Dict[str, Any] = {}
        transform_type = self._current_transform_name()

        if transform_type == "additive_periodic":
            params["mu_plus"] = [self.transform_param_widgets["mu_plus_x"].value(),
                                  self.transform_param_widgets["mu_plus_y"].value()]
            params["mu_minus"] = [self.transform_param_widgets["mu_minus_x"].value(),
                                   self.transform_param_widgets["mu_minus_y"].value()]
            params["gamma_plus"] = [self.transform_param_widgets["gamma_plus_x"].value(),
                                     self.transform_param_widgets["gamma_plus_y"].value()]
            params["gamma_minus"] = [self.transform_param_widgets["gamma_minus_x"].value(),
                                      self.transform_param_widgets["gamma_minus_y"].value()]
            params["omega_plus"] = [self.transform_param_widgets["omega_plus_x"].value(),
                                     self.transform_param_widgets["omega_plus_y"].value()]
            params["omega_minus"] = [self.transform_param_widgets["omega_minus_x"].value(),
                                      self.transform_param_widgets["omega_minus_y"].value()]
        elif transform_type == "log_sinusoidal":
            params["mu_plus"] = [self.transform_param_widgets["mu_plus_x"].value(),
                                  self.transform_param_widgets["mu_plus_y"].value()]
            params["mu_minus"] = [self.transform_param_widgets["mu_minus_x"].value(),
                                   self.transform_param_widgets["mu_minus_y"].value()]
            params["omega1_plus"] = [self.transform_param_widgets["omega1_plus_x"].value(),
                                      self.transform_param_widgets["omega1_plus_y"].value()]
            params["omega1_minus"] = [self.transform_param_widgets["omega1_minus_x"].value(),
                                       self.transform_param_widgets["omega1_minus_y"].value()]
            params["omega2_plus"] = [self.transform_param_widgets["omega2_plus_x"].value(),
                                      self.transform_param_widgets["omega2_plus_y"].value()]
            params["omega2_minus"] = [self.transform_param_widgets["omega2_minus_x"].value(),
                                       self.transform_param_widgets["omega2_minus_y"].value()]
            params["sigma"] = self.transform_param_widgets["sigma"].value()
        elif transform_type == "wavelet_mod":
            params["mu_plus"] = [self.transform_param_widgets["mu_plus_x"].value(),
                                  self.transform_param_widgets["mu_plus_y"].value()]
            params["mu_minus"] = [self.transform_param_widgets["mu_minus_x"].value(),
                                   self.transform_param_widgets["mu_minus_y"].value()]
            params["omega_plus"] = [self.transform_param_widgets["omega_plus_x"].value(),
                                     self.transform_param_widgets["omega_plus_y"].value()]
            params["omega_minus"] = [self.transform_param_widgets["omega_minus_x"].value(),
                                      self.transform_param_widgets["omega_minus_y"].value()]
            params["eta"] = self.transform_param_widgets["eta"].value()
            if not np.isfinite(params["eta"]) or params["eta"] <= 0:
                raise ValueError("Wavelet extent eta must be finite and greater than zero.")
            params["eps"] = self.transform_param_widgets["eps"].get_value()
        elif transform_type == "tensor_interference":
            params["mu_plus"] = [self.transform_param_widgets["mu_plus_x"].value(),
                                  self.transform_param_widgets["mu_plus_y"].value()]
            params["mu_minus"] = [self.transform_param_widgets["mu_minus_x"].value(),
                                   self.transform_param_widgets["mu_minus_y"].value()]
            params["omega_plus"] = [self.transform_param_widgets["omega_plus_x"].value(),
                                     self.transform_param_widgets["omega_plus_y"].value()]
            params["omega_minus"] = [self.transform_param_widgets["omega_minus_x"].value(),
                                      self.transform_param_widgets["omega_minus_y"].value()]
        elif transform_type == "radial_polytrig":
            params["mu"] = self.transform_param_widgets["mu"].value()
            params["p"] = self.transform_param_widgets["p"].value()
            params["q"] = self.transform_param_widgets["q"].value()
            params["omega"] = self.transform_param_widgets["omega"].value()
            params["eps"] = self.transform_param_widgets["eps"].get_value()

        return params

    def _on_delete_component_row(self, row: int) -> None:
        comps = self.controller.get_components()
        if 0 <= row < len(comps):
            # Show confirmation dialog
            reply = QtWidgets.QMessageBox.question(
                self,
                'Delete Component',
                f'Are you sure you want to delete Component {row + 1}?',
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                try:
                    self.controller.state.components.pop(row)
                except Exception:
                    pass
                self._refresh_component_table()

    def _on_edit_component_row(self, row: int) -> None:
        comps = self.controller.get_components()
        if not (0 <= row < len(comps)):
            return

        # Enter edit mode
        self.editing_component_idx = row
        self.btn_update_component.setVisible(True)
        self.btn_add_component.setVisible(False)

        cfg = comps[row]
        baseline = str(cfg.get("baseline", "A"))
        self.baseline_combo.setCurrentText(baseline)
        self.rotation_spin.set_degrees(float(cfg.get("rotation_deg", 0.0)))
        neutral = dict(cfg.get("neutralize", {}))
        self.neutral_enable.setChecked(bool(neutral.get("enable", False)))
        self.neutral_delta.setValue(float(neutral.get("Delta", 0.0)))
        self.neutral_rref.setValue(float(neutral.get("r_ref", 1.0)))
        p = dict(cfg.get("params", {}))
        c = list(p.get("c", [0.0, 0.0])); self.center_x.setValue(float(c[0])); self.center_y.setValue(float(c[1]))
        self.beta_spin.setValue(float(p.get("beta", 0.0)))
        kp = list(p.get("kappa_plus", [10.0, 10.0])); self.kappa_plus_x.setValue(float(kp[0])); self.kappa_plus_y.setValue(float(kp[1]))
        km = list(p.get("kappa_minus", [10.0, 10.0])); self.kappa_minus_x.setValue(float(km[0])); self.kappa_minus_y.setValue(float(km[1]))
        if baseline == "A":
            pp = list(p.get("p_plus", [1.0, 1.0])); pm = list(p.get("p_minus", [1.0, 1.0]))
            self.p_plus_x.setValue(float(pp[0])); self.p_plus_y.setValue(float(pp[1]))
            self.p_minus_x.setValue(float(pm[0])); self.p_minus_y.setValue(float(pm[1]))
        else:
            self.p_single.setValue(float(p.get("p_single", 1.0)))
        # Load transforms into list UI
        t_seq = list(cfg.get("T_seq", []))
        self.controller.state.T_seq = t_seq
        self._refresh_transform_list()

        # Refresh table to show highlighting
        self._refresh_component_table()
