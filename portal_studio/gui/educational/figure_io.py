"""Figure export helpers for Education plotting widgets."""
from __future__ import annotations

from pathlib import Path
import re

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget


def safe_pdf_name(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
    return f"{safe or 'figure'}.pdf"


def choose_pdf_path(parent: QWidget, default_name: str) -> Path | None:
    out_path, _fmt = QFileDialog.getSaveFileName(
        parent, "Save figure as PDF", default_name, "PDF (*.pdf)")
    if not out_path:
        return None
    path = Path(out_path)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    return path


def save_figure_pdf(parent: QWidget, fig, default_name: str) -> Path | None:
    path = choose_pdf_path(parent, default_name)
    if path is None:
        return None
    try:
        fig.savefig(path, format="pdf", bbox_inches="tight")
    except Exception as exc:  # noqa: BLE001 - show export failure to user
        QMessageBox.critical(parent, "Save failed", f"Could not save PDF:\n{exc}")
        return None
    return path
