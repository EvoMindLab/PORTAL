"""A small collapsible 'card' section, styled like the Builder's parameter sections.

A bold header button toggles a body widget; callers add content to ``body_layout``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)


class CollapsibleSection(QFrame):
    """A titled section whose body collapses when the header is clicked."""

    def __init__(
            self, title: str, parent=None,
            help_button: QPushButton | None = None,
            flat: bool = False, collapsible: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("parameterSubsection" if flat else "parameterSection")
        self._title = title
        self._collapsible = collapsible
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 2 if flat else 6)
        outer.setSpacing(0)

        self._header = QWidget()
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        # When collapsible, the header is a toggle button with a caret that
        # shows/hides the body. When not, it's a static title that always
        # stays open (no caret, no click-to-collapse).
        self._toggle = QPushButton((f"v  {title}") if collapsible else title)
        self._toggle.setObjectName("sectionHeader")
        self._toggle.setCheckable(collapsible)
        self._toggle.setChecked(True)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if collapsible:
            self._toggle.toggled.connect(self._on_toggle)
        else:
            from PySide6.QtCore import Qt
            self._toggle.setCursor(Qt.CursorShape.ArrowCursor)
            self._toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header_layout.addWidget(self._toggle)

        self._help_button = help_button
        if self._help_button is not None:
            header_layout.addWidget(self._help_button)
        outer.addWidget(self._header)

        self._body = QWidget()
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(10, 2 if flat else 4, 10, 3 if flat else 8)
        self.body_layout.setSpacing(4 if flat else 6)
        outer.addWidget(self._body)

    def _on_toggle(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._toggle.setText(("v  " if checked else ">  ") + self._title)
