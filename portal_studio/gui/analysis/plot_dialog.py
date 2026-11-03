"""Modal dialog that hosts a matplotlib chart + Export button.

The SA column launches one of these for Trend / Box plots so the chart does
not have to share screen real estate with the SA controls.
"""
from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget,
)

from portal_studio.experiment_runner.analysis.plotting import (
    EXPORT_FORMATS, export_figure,
)


class PlotDialog(QDialog):
    """Modal viewer for a matplotlib Figure + an Export button."""

    def __init__(self, fig: Figure, title: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 600)
        self._fig = fig

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        canvas = FigureCanvasQTAgg(fig)
        outer.addWidget(canvas, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.export_btn = QPushButton("Export…")
        self.export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self.export_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.close_btn)
        outer.addLayout(btn_row)

    def _on_export(self) -> None:
        out_path, fmt_choice = QFileDialog.getSaveFileName(
            self, "Export figure", "figure.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not out_path:
            return
        ext = Path(out_path).suffix.lstrip(".").lower()
        if ext not in EXPORT_FORMATS:
            ext = (
                "pdf" if "PDF" in fmt_choice
                else "svg" if "SVG" in fmt_choice
                else "png"
            )
        export_figure(
            self._fig, Path(out_path).parent,
            basename=Path(out_path).stem, formats=[ext],
        )
