"""Spin boxes and combo boxes that ignore mouse-wheel scrolling.

Values change only by clicking/typing (or clicking the dropdown list) — scrolling
the panel never silently changes a value. Matches the Builder tab's inputs (see
``component_editor.CleanDoubleSpinBox``).
"""
from __future__ import annotations

from PySide6 import QtCore
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class NoWheelComboBox(QComboBox):
    """A combo box whose value never changes on hover-scroll."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event) -> None:
        event.ignore()   # let the surrounding scroll area scroll instead


class NoWheelSpinBox(QSpinBox):
    """An integer spin box whose value never changes on hover-scroll."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """A float spin box whose value never changes on hover-scroll."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event) -> None:
        event.ignore()
