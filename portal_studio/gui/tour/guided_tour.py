"""Lightweight guided-tour overlay for PORTAL Studio.

The tour darkens the active window, outlines one target widget, and shows a
small message card with Back / Next / Skip controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6 import QtCore
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _noop() -> None:
    pass


@dataclass(frozen=True)
class TourStep:
    target: QWidget
    title: str
    body: str
    activate: Callable[[], None] = _noop


class GuidedTourOverlay(QWidget):
    """Semi-transparent coach-mark overlay for a sequence of widgets."""

    finished = QtCore.Signal()

    def __init__(self, host: QWidget, steps: list[TourStep]) -> None:
        super().__init__(host)
        self._host = host
        self._steps = list(steps)
        self._index = 0
        self._target_rect = QRect()
        self._target_pixmap = QPixmap()
        self._updating_step = False

        self.setObjectName("guidedTourOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.card = QFrame(self)
        self.card.setObjectName("guidedTourCard")
        self.card.setStyleSheet(
            "QFrame#guidedTourCard {"
            " background: #ffffff;"
            " border: 1px solid #d0d5dd;"
            " border-radius: 8px;"
            "}"
        )
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(8)

        self.title_label = QLabel()
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-weight: 700; font-size: 12pt; color: #111827;")
        card_layout.addWidget(self.title_label)

        self.body_label = QLabel()
        self.body_label.setWordWrap(True)
        self.body_label.setStyleSheet("color: #374151;")
        card_layout.addWidget(self.body_label)

        self.step_label = QLabel()
        self.step_label.setStyleSheet("color: #667085;")

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 4, 0, 0)
        button_row.addWidget(self.step_label)
        button_row.addStretch(1)
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self.previous_step)
        button_row.addWidget(self.back_btn)
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self.finish)
        button_row.addWidget(self.skip_btn)
        self.next_btn = QPushButton("Next")
        self.next_btn.setProperty("variant", "primary")
        self.next_btn.clicked.connect(self.next_step)
        button_row.addWidget(self.next_btn)
        card_layout.addLayout(button_row)

    def start(self) -> None:
        if not self._steps:
            self.finish()
            return
        self.setGeometry(self._host.rect())
        self._host.installEventFilter(self)
        self.raise_()
        self.show()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        self._show_step(0)

    def next_step(self) -> None:
        if self._index >= len(self._steps) - 1:
            self.finish()
            return
        self._show_step(self._index + 1)

    def previous_step(self) -> None:
        if self._index > 0:
            self._show_step(self._index - 1)

    def finish(self) -> None:
        self.finished.emit()
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._host.removeEventFilter(self)
        super().closeEvent(event)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched is self._host and event.type() == QtCore.QEvent.Type.Resize:
            self.setGeometry(self._host.rect())
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._steps:
            self._refresh_current_step(capture_pixmap=False)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.finish()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Right):
            self.next_step()
            return
        if event.key() == Qt.Key.Key_Left:
            self.previous_step()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._target_pixmap.isNull():
            painter.drawPixmap(self._target_rect, self._target_pixmap)

        overlay = QPainterPath()
        overlay.addRect(self.rect())
        target = QPainterPath()
        target.addRoundedRect(self._target_rect.adjusted(-6, -6, 6, 6), 10, 10)
        overlay = overlay.subtracted(target)
        painter.fillPath(overlay, QColor(15, 23, 42, 175))

        painter.setPen(QPen(QColor("#1677ff"), 3))
        painter.drawRoundedRect(self._target_rect.adjusted(-6, -6, 6, 6), 10, 10)

    def _show_step(self, index: int) -> None:
        self._index = max(0, min(index, len(self._steps) - 1))
        step = self._steps[self._index]
        step.activate()
        QtCore.QCoreApplication.processEvents()
        self.title_label.setText(step.title)
        self.body_label.setText(step.body)
        self.step_label.setText(f"{self._index + 1} of {len(self._steps)}")
        self.back_btn.setEnabled(self._index > 0)
        self.next_btn.setText("Finish" if self._index == len(self._steps) - 1 else "Next")

        self._refresh_current_step(capture_pixmap=True)

    def _refresh_current_step(self, *, capture_pixmap: bool) -> None:
        if not self._steps or self._updating_step:
            return
        self._updating_step = True
        try:
            step = self._steps[self._index]
            self._target_rect = self._target_rect_for(step.target)
            if capture_pixmap:
                self._target_pixmap = (
                    step.target.grab()
                    if step.target is not None and step.target.isVisible()
                    else QPixmap()
                )
            self._position_card()
            self.update()
        finally:
            self._updating_step = False

    def _target_rect_for(self, widget: QWidget) -> QRect:
        if widget is None or not widget.isVisible():
            return QRect(24, 24, 1, 1)
        top_left = widget.mapTo(self._host, QPoint(0, 0))
        return QRect(top_left, widget.size()).intersected(self.rect())

    def _position_card(self) -> None:
        self.card.setFixedWidth(min(360, max(280, self.width() - 48)))
        self.card.adjustSize()
        margin = 18
        panel_target = (
            self._target_rect.width() >= 260
            and self._target_rect.height() >= 180
        )
        below = self._target_rect.bottom() + margin
        above = self._target_rect.top() - self.card.height() - margin
        default_x = min(
            max(self._target_rect.left(), margin),
            max(margin, self.width() - self.card.width() - margin),
        )
        if panel_target:
            right_x = self._target_rect.right() + margin
            left_x = self._target_rect.left() - self.card.width() - margin
            y = min(
                max(self._target_rect.top(), margin),
                max(margin, self.height() - self.card.height() - margin),
            )
            if right_x + self.card.width() <= self.width() - margin:
                x = right_x
            elif left_x >= margin:
                x = left_x
            elif below + self.card.height() <= self.height() - margin:
                x = default_x
                y = below
            elif above >= margin:
                x = default_x
                y = above
            else:
                # Full-height middle panels may leave slightly less than the
                # normal card width on either side. Fit that space rather than
                # covering the highlighted controls.
                left_space = self._target_rect.left() - 2 * margin
                right_space = self.width() - self._target_rect.right() - 2 * margin
                side_space = max(left_space, right_space)
                if side_space >= 280:
                    self.card.setFixedWidth(min(360, side_space))
                    self.card.adjustSize()
                    x = (right_x if right_space >= left_space else
                         self._target_rect.left() - self.card.width() - margin)
                    y = min(
                        max(self._target_rect.top(), margin),
                        max(margin, self.height() - self.card.height() - margin),
                    )
                else:
                    x = default_x
        elif below + self.card.height() <= self.height() - margin:
            x = default_x
            y = below
        elif above >= margin:
            x = default_x
            y = above
        else:
            x = default_x
            y = max(margin, self.height() - self.card.height() - margin)
        self.card.move(x, y)
