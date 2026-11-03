"""Configure Block-Diagonal rotation: Auto/Manual group sizes and
Fixed/Randomize per-group angle. result_spec() returns a spec fragment.

Membership is engine-assigned; the user only sets group sizes. Dimensions not
covered by any group are each fully separable."""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QLabel,
    QPushButton, QLineEdit, QSpinBox, QWidget, QFrame,
)

from portal_studio.gui.common.icons import make_x_icon
from portal_studio.gui.common.angles import DegreeLineEdit, display_degrees

_DEFAULT_GROUP_ANGLE = 45.0
_AUTO_ANGLES = [45.0, 135.0, 22.5, 112.5]
CONTROLLED_PARTIAL_SEPERATE_LABEL = "Controlled Partial Separable"
# Per-group chip colors, cycled.
_GROUP_COLORS = ["#0b66d8", "#2eb872", "#d97706", "#a855f7", "#db2777", "#0891b2"]

_STYLE = """
QDialog { background: #f4f6fa; }
QLabel#dlgTitle { font-size: 15px; font-weight: 700; color: #1f2a44; }
QLabel#sectionLabel { font-weight: 600; color: #1f2a44; }
QLabel#muted { color: #667085; font-size: 11px; }
QLabel#errorLabel { color: #d64545; font-size: 12px; }
QRadioButton { color: #1f2a44; spacing: 6px; padding: 2px 0; }
QLineEdit, QSpinBox {
    border: 1px solid #d8dde6; border-radius: 6px; padding: 3px 6px;
    background: #ffffff; color: #1f2a44;
}
QLineEdit:focus, QSpinBox:focus { border: 1px solid #0b66d8; }
QPushButton {
    border: 1px solid #d8dde6; border-radius: 8px; padding: 6px 14px;
    background: #ffffff; color: #1f2a44;
}
QPushButton:hover { background: #eaf3ff; }
QPushButton#primary { background: #0b66d8; color: #ffffff; border: none; }
QPushButton#primary:hover { background: #0a5cc4; }
QPushButton#danger { border: none; background: transparent; color: #d64545; padding: 2px 8px; }
QPushButton#danger:hover { color: #b91c1c; }
QFrame#card { background: #ffffff; border: 1px solid #e2e7ef; border-radius: 10px; }
QFrame#groupCard { background: #ffffff; border: 1px solid #e2e7ef; border-radius: 8px; }
QFrame#sep { background: #e2e7ef; max-height: 1px; min-height: 1px; }
QComboBox {
    border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px 8px;
    background: #ffffff; color: #1f2a44;
}
QComboBox:focus { border: 1px solid #0b66d8; }
"""


def _auto_sizes(d: int) -> List[int]:
    n = 3 if d >= 30 else 2
    base, rem = d // n, d % n
    sizes = [base] * n
    for i in range(rem):
        sizes[i] += 1
    return sizes


def _fmt_angle(value: float) -> str:
    return display_degrees(value)


def block_diagonal_display(spec: dict, d: int):
    """Describe a mode-7 rotation spec for read-only display (the data table).

    Returns ``(grouping_label, angle_mode, rows, remaining)`` where ``rows`` is a
    list of ``(block_label, size, angle_str)``. Falls back to the engine's auto
    sizes/angles when the spec omits them (a never-configured block-diagonal)."""
    grouping = "Manual" if spec.get("grouping") == "manual" else "Auto"
    angle_mode = "randomize" if spec.get("angle_mode") == "randomize" else "fixed"
    sizes = [int(s) for s in (spec.get("block_sizes") or [])]
    if not sizes:
        sizes = _auto_sizes(d)
    angles = [float(a) for a in (spec.get("block_angles") or [])]
    rows = []
    for b, size in enumerate(sizes):
        if angle_mode == "randomize":
            angle_str = "random per pair"
        elif b < len(angles):
            angle_str = _fmt_angle(angles[b])
        else:
            angle_str = _fmt_angle(_AUTO_ANGLES[b % len(_AUTO_ANGLES)])
        rows.append((f"Block {b + 1}", size, angle_str))
    remaining = d - sum(sizes)
    return grouping, angle_mode, rows, remaining


class BlockDiagonalDialog(QDialog):
    def __init__(self, d: int, spec: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"{CONTROLLED_PARTIAL_SEPERATE_LABEL} - Dimension={int(d)}")
        self._d = int(d)
        self._grouping = "auto"
        self._angle_mode = "fixed"
        self._sizes: List[int] = []
        self._angles: List[float] = []

        self.setStyleSheet(_STYLE)
        self.setMinimumWidth(440)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(12)

        # Mode selector card: Grouping + Angle as segmented toggles.
        mode_card = QFrame(); mode_card.setObjectName("card")
        modes = QHBoxLayout(mode_card)
        modes.setContentsMargins(14, 12, 14, 12); modes.setSpacing(8)
        self.grouping_combo = QComboBox(); self.grouping_combo.addItems(["Auto", "Manual"])
        self.angle_combo = QComboBox(); self.angle_combo.addItems(["Fixed", "Randomize"])
        grouping_lbl = QLabel("Grouping:"); grouping_lbl.setObjectName("sectionLabel")
        angle_lbl = QLabel("Angle:"); angle_lbl.setObjectName("sectionLabel")
        modes.addWidget(grouping_lbl); modes.addWidget(self.grouping_combo)
        modes.addSpacing(24)
        modes.addWidget(angle_lbl); modes.addWidget(self.angle_combo)
        modes.addStretch(1)
        root.addWidget(mode_card)

        self.angle_range_row = QWidget()
        rr = QHBoxLayout(self.angle_range_row)
        rr.setContentsMargins(2, 0, 2, 0)
        rr.setSpacing(8)
        range_lbl = QLabel("Angle range (deg):")
        range_lbl.setObjectName("sectionLabel")
        theta_min_lbl = QLabel("min")
        theta_min_lbl.setObjectName("muted")
        theta_max_lbl = QLabel("max")
        theta_max_lbl.setObjectName("muted")
        self.theta_min_edit = DegreeLineEdit(-180)
        self.theta_min_edit.setFixedWidth(100)
        self.theta_max_edit = DegreeLineEdit(180)
        self.theta_max_edit.setFixedWidth(100)
        rr.addWidget(range_lbl)
        rr.addWidget(theta_min_lbl)
        rr.addWidget(self.theta_min_edit)
        rr.addWidget(theta_max_lbl)
        rr.addWidget(self.theta_max_edit)
        rr.addStretch(1)
        root.addWidget(self.angle_range_row)

        # add-group row (manual only)
        self.add_row = QWidget(); ar = QHBoxLayout(self.add_row); ar.setContentsMargins(2, 0, 2, 0)
        ar.setSpacing(8)
        size_lbl = QLabel("Group size:"); size_lbl.setObjectName("sectionLabel")
        ar.addWidget(size_lbl)
        self.size_spin = QSpinBox(); self.size_spin.setMinimum(2); self.size_spin.setMaximum(self._d)
        self.size_spin.setFixedWidth(70)
        ar.addWidget(self.size_spin)
        self.size_status = QLabel()
        self.size_status.setObjectName("muted")
        ar.addWidget(self.size_status)
        self.add_btn = QPushButton("+ Add group"); self.add_btn.clicked.connect(self._on_add_clicked)
        ar.addWidget(self.add_btn); ar.addStretch(1)
        root.addWidget(self.add_row)

        self.error_label = QLabel(""); self.error_label.setObjectName("errorLabel")
        self.error_label.setVisible(False)
        root.addWidget(self.error_label)

        self.groups_host = QWidget(); self.groups_layout = QVBoxLayout(self.groups_host)
        self.groups_layout.setContentsMargins(0, 0, 0, 0); self.groups_layout.setSpacing(8)
        root.addWidget(self.groups_host)

        self.remaining_label = QLabel(""); self.remaining_label.setObjectName("muted")
        root.addWidget(self.remaining_label)

        root.addStretch(1)
        sep = QFrame(); sep.setObjectName("sep")
        root.addWidget(sep)

        foot = QHBoxLayout(); foot.addStretch(1); foot.setSpacing(8)
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        done = QPushButton("Done"); done.setObjectName("primary"); done.clicked.connect(self._on_done)
        foot.addWidget(cancel); foot.addWidget(done)
        root.addLayout(foot)

        self.grouping_combo.currentTextChanged.connect(
            lambda t: self.set_grouping("manual" if t == "Manual" else "auto"))
        self.angle_combo.currentTextChanged.connect(
            lambda t: self.set_angle_mode("randomize" if t == "Randomize" else "fixed"))

        if spec:
            self._load_spec(spec)
        else:
            self._apply_auto_sizes()
        self._refresh()

    # ---- public API ----
    def set_grouping(self, value: str) -> None:
        self._grouping = "manual" if value == "manual" else "auto"
        self.grouping_combo.blockSignals(True)
        self.grouping_combo.setCurrentText(
            "Manual" if self._grouping == "manual" else "Auto")
        self.grouping_combo.blockSignals(False)
        if self._grouping == "auto":
            self._apply_auto_sizes()
        else:
            self._sizes, self._angles = [], []
        self._show_error("")
        self._refresh()

    def set_angle_mode(self, value: str) -> None:
        self._angle_mode = "randomize" if value == "randomize" else "fixed"
        self.angle_combo.blockSignals(True)
        self.angle_combo.setCurrentText(
            "Randomize" if self._angle_mode == "randomize" else "Fixed")
        self.angle_combo.blockSignals(False)
        self._refresh()

    def add_group(self, size: int) -> bool:
        if self._grouping != "manual":
            return False
        size = int(size)
        if size < 2:
            self._show_error(
                "Group size must be at least 2 — a size-1 group couples "
                "nothing; that coordinate is not mixed by rotation.")
            return False
        if sum(self._sizes) + size > self._d:
            self._show_error(f"Only {self.remaining()} dimension(s) remaining.")
            return False
        self._sizes.append(size)
        self._angles.append(_AUTO_ANGLES[len(self._angles) % len(_AUTO_ANGLES)])
        self._show_error("")
        self._refresh()
        return True

    def remove_group(self, index: int) -> None:
        if 0 <= index < len(self._sizes):
            self._sizes.pop(index); self._angles.pop(index)
            self._show_error("")
            self._refresh()

    def set_group_angle(self, index: int, angle: float) -> None:
        if 0 <= index < len(self._angles):
            self._angles[index] = float(angle)

    def remaining(self) -> int:
        return self._d - sum(self._sizes)

    def error_text(self) -> str:
        return self.error_label.text()

    def result_spec(self) -> dict:
        spec = {"mode": 7, "grouping": self._grouping, "angle_mode": self._angle_mode}
        spec["block_sizes"] = list(self._sizes)
        if self._angle_mode == "fixed":
            spec["block_angles"] = list(self._angles)
        else:
            theta_min, theta_max = self._angle_range_values()
            spec["theta_min"] = theta_min
            spec["theta_max"] = theta_max
        return spec

    # ---- internal ----
    def _on_done(self) -> None:
        if self._angle_mode == "randomize" and self._validate_angle_range() is None:
            return
        # Manual grouping with no groups leaves every dimension separable
        # (no rotation), which is not partial separability — reject it so the
        # mode actually couples something. Auto always has system sizes.
        if self._grouping == "manual" and sum(self._sizes) == 0:
            self._show_error(
                "Add at least one group — an empty manual configuration "
                "leaves all rotation angles zero.")
            return
        self.accept()

    def _show_error(self, text: str) -> None:
        self.error_label.setText(text)
        self.error_label.setVisible(bool(text))

    def _on_add_clicked(self) -> None:
        self.add_group(self.size_spin.value())

    def _apply_auto_sizes(self) -> None:
        self._sizes = _auto_sizes(self._d)
        self._angles = [_AUTO_ANGLES[b % len(_AUTO_ANGLES)]
                        for b in range(len(self._sizes))]

    def _refresh(self) -> None:
        rem = self.remaining()
        self.add_row.setVisible(self._grouping == "manual")
        self.angle_range_row.setVisible(self._angle_mode == "randomize")
        # Offer the remaining dimensions as the next group size. When fewer
        # than two remain, replace the input with an explanation.
        # A group needs at least 2 dims to couple anything, so adding is only
        # possible while 2+ dims remain; a lone leftover dim stays separable.
        can_add = self._grouping == "manual" and rem >= 2
        self.size_spin.setEnabled(can_add)
        self.size_spin.setVisible(rem >= 2)
        self.size_status.setVisible(rem < 2)
        self.size_status.setText(
            "No dimensions remaining" if rem == 0 else
            "1 dimension remaining; a group needs at least 2")
        self.add_btn.setEnabled(can_add)
        self.size_spin.setMaximum(max(2, rem))
        self.size_spin.setValue(rem if rem >= 2 else 2)
        # rebuild group rows as cards
        while self.groups_layout.count():
            item = self.groups_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for gi, size in enumerate(self._sizes):
            card = QFrame(); card.setObjectName("groupCard")
            row = QHBoxLayout(card); row.setContentsMargins(12, 8, 10, 8); row.setSpacing(10)
            chip = QLabel()
            chip.setFixedSize(12, 12)
            chip.setStyleSheet(
                f"background: {_GROUP_COLORS[gi % len(_GROUP_COLORS)]};"
                "border-radius: 6px;")
            row.addWidget(chip)
            name = QLabel(f"<b>Group {gi + 1}</b> &nbsp;·&nbsp; size {size}")
            name.setTextFormat(Qt.TextFormat.RichText)
            row.addWidget(name)
            row.addStretch(1)
            if self._angle_mode == "fixed":
                ang_lbl = QLabel("angle (deg)"); ang_lbl.setObjectName("muted")
                edit = DegreeLineEdit(self._angles[gi]); edit.setFixedWidth(100)
                edit.textChanged.connect(
                    lambda text, idx=gi, field=edit: self._on_angle_edited(idx, field))
                row.addWidget(ang_lbl); row.addWidget(edit)
            else:
                rnd = QLabel("angle: random"); rnd.setObjectName("muted")
                row.addWidget(rnd)
            if self._grouping == "manual":
                rm = QPushButton(); rm.setObjectName("danger")
                rm.setIcon(make_x_icon())
                rm.setFixedSize(24, 24)
                rm.setToolTip("Ungroup these dimensions")
                rm.setAccessibleName(rm.toolTip())
                rm.setAccessibleDescription(rm.toolTip())
                rm.clicked.connect(lambda _c, idx=gi: self.remove_group(idx))
                row.addWidget(rm)
            self.groups_layout.addWidget(card)
        self.remaining_label.setText(
            f"remaining: {self.remaining()} dim(s), not mixed by this rotation")

    def _on_angle_edited(self, index: int, field: DegreeLineEdit) -> None:
        try:
            self._angles[index] = field.degrees_value()
        except (ValueError, IndexError):
            pass

    def _angle_range_values(self):
        return self.theta_min_edit.degrees_value(), self.theta_max_edit.degrees_value()

    def _validate_angle_range(self):
        try:
            theta_min, theta_max = self._angle_range_values()
        except ValueError:
            self._show_error("Angle range must use finite numeric values in degrees.")
            return None
        if theta_min < -180 or theta_min > 180 or theta_max < -180 or theta_max > 180:
            self._show_error("Angle range values must be between -180 and 180 degrees.")
            return None
        if theta_min > theta_max:
            self._show_error("Angle range minimum must be less than or equal to maximum.")
            return None
        self._show_error("")
        return theta_min, theta_max

    def _load_spec(self, spec: dict) -> None:
        grouping = "manual" if spec.get("grouping") == "manual" else "auto"
        angle_mode = "randomize" if spec.get("angle_mode") == "randomize" else "fixed"
        # Set the dropdowns without firing their handlers, then assign the loaded
        # sizes/angles directly.
        self.grouping_combo.blockSignals(True)
        self.grouping_combo.setCurrentText("Manual" if grouping == "manual" else "Auto")
        self.grouping_combo.blockSignals(False)
        self.angle_combo.blockSignals(True)
        self.angle_combo.setCurrentText("Randomize" if angle_mode == "randomize" else "Fixed")
        self.angle_combo.blockSignals(False)
        self._grouping = grouping
        self._angle_mode = angle_mode
        self.theta_min_edit.set_degrees(spec.get("theta_min", -180.0))
        self.theta_max_edit.set_degrees(spec.get("theta_max", 180.0))
        if grouping == "manual":
            self._sizes = [int(s) for s in spec.get("block_sizes", [])]
            angles = list(spec.get("block_angles", []))
            self._angles = [float(angles[i]) if i < len(angles) else _DEFAULT_GROUP_ANGLE
                            for i in range(len(self._sizes))]
        else:
            self._apply_auto_sizes()
            # Auto chooses group sizes, but saved per-group edits remain valid.
            for index, angle in enumerate(spec.get("block_angles", [])):
                if index < len(self._angles):
                    self._angles[index] = float(angle)
