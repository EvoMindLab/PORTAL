"""Degree-based rotation controls matching the saved settings.

Keep untouched loaded values exactly, including values displayed in a spinbox.
"""
from __future__ import annotations

import math

from PySide6.QtWidgets import QLineEdit

from .widgets import NoWheelDoubleSpinBox


def degrees_text(degrees: float) -> str:
    return format(float(degrees), '.17g')


def display_degrees(degrees: float) -> str:
    return format(float(degrees), '.6g')


class DegreeLineEdit(QLineEdit):
    """Accept and return degrees without a unit conversion."""

    def __init__(self, degrees: float = 0.0, parent=None):
        super().__init__(parent)
        self.set_degrees(degrees)

    def set_degrees(self, degrees: float) -> None:
        self._loaded_degrees = float(degrees)
        self._loaded_text = degrees_text(degrees)
        self.setText(self._loaded_text)
        # Show the sign and leading digits when a precise value is wider
        # than the field; QLineEdit otherwise scrolls to its trailing digits.
        self.setCursorPosition(0)

    def degrees_value(self, default=None) -> float:
        text = self.text().strip()
        if text == self._loaded_text:
            return self._loaded_degrees
        if not text and default is not None:
            return float(default)
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('Angles must be finite numbers in degrees.')
        return value


class DegreeSpinBox(NoWheelDoubleSpinBox):
    """The widget value and accessors use degrees, preserving loaded settings."""

    def __init__(self, parent=None):
        self._loading_degrees = False
        self._loaded_degrees = 0.0
        self._loaded_value = 0.0
        super().__init__(parent)
        self.setDecimals(17)
        self.setRange(-360.0, 360.0)
        self.setSingleStep(1.0)
        self.setButtonSymbols(self.ButtonSymbols.NoButtons)
        self.setMinimumWidth(78)
        self.setMaximumWidth(120)

    def textFromValue(self, value: float) -> str:
        return format(value, '.17g')

    def set_degrees(self, degrees: float) -> None:
        self._loaded_degrees = float(degrees)
        self._loading_degrees = True
        try:
            self.setValue(degrees)
            self._loaded_value = self.value()
        finally:
            self._loading_degrees = False

    def degrees_value(self) -> float:
        if self._loading_degrees or self.value() == self._loaded_value:
            return self._loaded_degrees
        return self.value()
