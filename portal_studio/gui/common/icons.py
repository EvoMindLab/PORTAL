"""Shared vector icon factory for all PORTAL Studio icon buttons."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap

if TYPE_CHECKING:
    from PySide6.QtWidgets import QPushButton


def _begin(size: int = 16) -> tuple[QPixmap, QPainter]:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    return pm, p


def make_app_icon() -> QIcon:
    """Title-bar icon for PORTAL Studio."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128):
        icon.addPixmap(_make_app_pixmap(size))
    return icon


def _make_app_pixmap(size: int) -> QPixmap:
    pm, p = _begin(size)
    pad = max(1.0, size * 0.08)
    rect = QRectF(pad, pad, size - 2 * pad, size - 2 * pad)

    p.setPen(QPen(QColor("#08234a"), max(1.0, size * 0.045)))
    p.setBrush(QBrush(QColor("#0b2d5c")))
    p.drawRoundedRect(rect, size * 0.16, size * 0.16)

    accent = QColor("#1f7ae0")
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(accent))
    p.drawRoundedRect(
        QRectF(size * 0.18, size * 0.17, size * 0.22, size * 0.66),
        size * 0.08,
        size * 0.08,
    )
    p.drawRoundedRect(
        QRectF(size * 0.42, size * 0.17, size * 0.40, size * 0.20),
        size * 0.08,
        size * 0.08,
    )

    font = QFont()
    font.setBold(True)
    font.setPointSize(max(8, int(size * 0.52)))
    p.setFont(font)
    p.setPen(QPen(QColor("#ffffff"), 1.0))
    p.drawText(QRectF(size * 0.18, size * 0.12, size * 0.64, size * 0.76),
               Qt.AlignmentFlag.AlignCenter, "P")
    p.end()
    return pm


def make_pencil_icon(color: str = "#4c5870", size: int = 16) -> QIcon:
    pm, p = _begin(size)
    p.setPen(QPen(QColor(color), 2.2))
    p.drawLine(QPointF(4.5, size - 4.5), QPointF(size - 5, 5))
    p.setPen(QPen(QColor(color), 1.3))
    p.drawLine(QPointF(3, size - 3), QPointF(5.5, size - 5.5))
    p.end()
    return QIcon(pm)


def make_save_icon(color: str = "#4c5870", size: int = 16) -> QIcon:
    pm, p = _begin(size)
    m = 2.5
    p.setPen(QPen(QColor(color), 1.4))
    p.setBrush(QBrush(QColor(0, 0, 0, 0)))
    p.drawRoundedRect(QRectF(m, m, size - 2 * m, size - 2 * m), 2, 2)
    p.drawRect(QRectF(m + 2.5, m, size - 2 * m - 5, 3.2))
    p.drawRect(QRectF(m + 2.5, size - m - 4.5, size - 2 * m - 5, 3.0))
    p.end()
    return QIcon(pm)


def make_reset_icon(color: str = "#4c5870", size: int = 16) -> QIcon:
    """Circular arrow for returning a plot to its original view."""
    pm, p = _begin(size)
    ink = QColor(color)
    p.setPen(QPen(ink, 1.7, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    rect = QRectF(3.2, 3.2, size - 6.4, size - 6.4)
    p.drawArc(rect, 35 * 16, 285 * 16)
    p.drawLine(QPointF(4.0, 5.2), QPointF(3.0, 2.4))
    p.drawLine(QPointF(4.0, 5.2), QPointF(6.8, 4.3))
    p.end()
    return QIcon(pm)


def make_x_icon(color: str = "#d64545", size: int = 16) -> QIcon:
    pm, p = _begin(size)
    m = 4.5
    p.setPen(QPen(QColor(color), 2.3))
    p.drawLine(QPointF(m, m), QPointF(size - m, size - m))
    p.drawLine(QPointF(size - m, m), QPointF(m, size - m))
    p.end()
    return QIcon(pm)


def make_question_icon(color: str = "#4c5870", size: int = 16) -> QIcon:
    pm, p = _begin(size)
    font = QFont()
    font.setBold(True)
    font.setPointSize(max(9, int(size * 0.72)))
    p.setFont(font)
    p.setPen(QPen(QColor(color), 1.0))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "?")
    p.end()
    return QIcon(pm)


def make_exclamation_icon(color: str = "#4c5870", size: int = 16) -> QIcon:
    pm, p = _begin(size)
    font = QFont()
    font.setBold(True)
    font.setPointSize(max(9, int(size * 0.72)))
    p.setFont(font)
    p.setPen(QPen(QColor(color), 1.0))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "!")
    p.end()
    return QIcon(pm)


def make_table_icon(color: str = "#4c5870", size: int = 16) -> QIcon:
    pm, p = _begin(size)
    pen = QPen(QColor(color), 1.2)
    p.setPen(pen)
    p.setBrush(QBrush(QColor(0, 0, 0, 0)))
    m = 2.5
    rect = QRectF(m, m, size - 2 * m, size - 2 * m)
    p.drawRoundedRect(rect, 1.5, 1.5)
    p.drawLine(QPointF(m, m + 4), QPointF(size - m, m + 4))
    p.drawLine(QPointF(m + 4, m), QPointF(m + 4, size - m))
    p.drawLine(QPointF(m, m + 8), QPointF(size - m, m + 8))
    p.drawLine(QPointF(m + 8, m), QPointF(m + 8, size - m))
    p.end()
    return QIcon(pm)


def make_clone_icon(color: str = "#4c5870", size: int = 16) -> QIcon:
    """Two overlapping rounded squares — the conventional duplicate glyph."""
    pm, p = _begin(size)
    p.setPen(QPen(QColor(color), 1.3))
    p.setBrush(QBrush(QColor(0, 0, 0, 0)))
    s = size - 2
    p.drawRoundedRect(QRectF(2, 4, s - 5, s - 5), 1.6, 1.6)        # back square
    p.fillRect(QRectF(5.2, 7.2, s - 9, s - 9), QColor(0, 0, 0, 0))
    p.drawRoundedRect(QRectF(5, 7, s - 5, s - 5), 1.6, 1.6)        # front square
    p.end()
    return QIcon(pm)


def make_reroll_icon(color: str = "#4c5870", size: int = 16) -> QIcon:
    """A die showing five pips — the 'reroll the dice' affordance."""
    pm, p = _begin(size)
    m = 2.0
    p.setPen(QPen(QColor(color), 1.3))
    p.setBrush(QBrush(QColor(0, 0, 0, 0)))
    p.drawRoundedRect(QRectF(m, m, size - 2 * m, size - 2 * m), 3, 3)
    # Five pips (four corners + centre).
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(color)))
    r = 1.3
    lo, hi, mid = m + 3.0, size - m - 3.0, size / 2.0
    for x, y in [(lo, lo), (hi, lo), (mid, mid), (lo, hi), (hi, hi)]:
        p.drawEllipse(QPointF(x, y), r, r)
    p.end()
    return QIcon(pm)


def make_seed_icon(color: str = "#4c5870", size: int = 16) -> QIcon:
    """A compact seed badge for deterministic seed controls."""
    pm, p = _begin(size)
    ink = QColor(color)
    p.setPen(QPen(ink, 1.2))
    p.setBrush(QBrush(QColor("#eef6ff")))
    m = 2.4
    p.drawRoundedRect(QRectF(m, m, size - 2 * m, size - 2 * m), 3, 3)

    font = QFont()
    font.setBold(True)
    font.setPointSize(max(7, int(size * 0.58)))
    p.setFont(font)
    p.setPen(QPen(ink, 1.0))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "S")

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor("#1f7ae0")))
    p.drawEllipse(QPointF(size - 4.4, 4.3), 1.7, 1.7)
    p.end()
    return QIcon(pm)


def make_back_arrow_icon(color: str = "#1f5130", size: int = 16) -> QIcon:
    """A leftward arrow — the 'go back to the previous page' affordance."""
    pm, p = _begin(size)
    p.setPen(QPen(QColor(color), 2.0, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    mid = size / 2.0
    left, right = 3.5, size - 3.5
    p.drawLine(QPointF(left, mid), QPointF(right, mid))      # shaft
    p.drawLine(QPointF(left, mid), QPointF(left + 4.5, mid - 4.5))   # upper barb
    p.drawLine(QPointF(left, mid), QPointF(left + 4.5, mid + 4.5))   # lower barb
    p.end()
    return QIcon(pm)


# Aliases used by all call-sites
make_edit_icon  = make_pencil_icon
make_check_icon = make_save_icon
make_trash_icon = make_x_icon
make_help_icon  = make_question_icon
make_alert_help_icon = make_exclamation_icon
make_restore_icon = make_reset_icon


def make_icon_button(icon: QIcon, tip: str, handler,
                     size: int = 24, object_name: str = "nodeActionBtn") -> "QPushButton":
    from PySide6.QtWidgets import QPushButton
    btn = QPushButton()
    btn.setIcon(icon)
    btn.setFixedSize(size, size)
    btn.setToolTip(tip)
    btn.setAccessibleName(tip)
    btn.setAccessibleDescription(tip)
    btn.setObjectName(object_name)
    btn.clicked.connect(lambda _checked=False: handler())
    return btn
