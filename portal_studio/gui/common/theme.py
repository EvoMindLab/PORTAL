"""Centralized light visual theme for PORTAL Studio.

Owns the design tokens, builds one application-wide Qt stylesheet from the
palette, styles matplotlib figures to match, and exposes a ``ThemeManager``
that applies the light style.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

# --- Design tokens -----------------------------------------------------------
LIGHT: dict[str, str] = {
    "bg": "#f4f6fa",
    "surface": "#ffffff",
    "border": "#d8dde6",
    "border_soft": "#e2e7ef",
    "text": "#1f2a44",
    "muted": "#667085",
    "accent": "#0b66d8",
    "accent_hover": "#0a5cc4",
    "accent_active": "#094fa8",
    "accent_soft": "#eaf3ff",
    "success": "#2eb872",
    "danger": "#d64545",
    "dashed": "#cfd6e2",
    "field_bg": "#ffffff",
    "row_bg": "#f8fafc",
    "fig_bg": "#ffffff",
    "axes_bg": "#ffffff",
    "axes_fg": "#1f2a44",
    # Workspace-specific tokens.
    "card_border": "#e4e7ec",   # sectionCard border (distinct from the app-wide `border` token)
    "heading": "#2c3140",       # card/section title text
    "hint": "#9aa0a6",          # placeholder/empty-state hint text
    "emphasis": "#0f172a",      # component-parameters dialog title/summary text
    "legacy_success": "#18A558",
    "legacy_warning": "#FF9500",
    "caption": "#4c5870",
    "legacy_header_baseline": "#2E86AB",
    "legacy_header_transform": "#A23B72",
    "legacy_transform_list_bg": "#f7f8fa",
    "legacy_transform_card_border": "#ccd3df",
    "back_btn_bg": "#d7f4df",
    "back_btn_border": "#9fd6ab",
    "back_btn_bg_hover": "#c9edd3",
    "back_btn_bg_pressed": "#bce6c8",
    "progress_label": "#6b7280",
    "field_text_dark": "#1a1a1a",
    "fitness_text": "#1f2937",
    "field_disabled_primary_bg": "#e6e8ec",
    "field_disabled_primary_border": "#d6d9de",
}

# Table-cell tint colors shared by the Education and Generator
# parameter/transform tables (QTableWidgetItem backgrounds). A named lookup
# rather than a single design token because callers index it by tint name.
PARAM_TABLE_TINTS: dict[str, str] = {
    # Component-parameter table group tints (centre = peach, kappa = blue,
    # p = green).
    "c": "#fff3e6",
    "kappa_plus": "#eaf2ff",
    "kappa_minus": "#eaf2ff",
    "p_plus": "#eafaf0",
    "p_minus": "#eafaf0",
    "p_single": "#eafaf0",
    # Component-transform table per-± -pair palette, cycled in this order.
    "tf_blue": "#eaf2ff",
    "tf_green": "#eafaf0",
    "tf_pink": "#fdf0f6",
    "tf_yellow": "#fff7e8",
    "tf_lavender": "#f0f0fb",
}

CHECKBOX_CHECKMARK = Path(__file__).resolve().parent.parent / "assets" / "checkbox-check.svg"


# --- Stylesheet builder ------------------------------------------------------
def build_qss(p: dict[str, str]) -> str:
    """Return a complete application-wide Qt stylesheet for palette ``p``."""
    checkbox_checkmark = CHECKBOX_CHECKMARK.as_posix()
    return f"""
    QWidget {{
        background: {p['bg']};
        color: {p['text']};
        font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif;
        font-size: 10pt;
    }}

    /* Cards / panels -------------------------------------------------------*/
    QFrame#parameterPanel, QWidget#parameterPanel {{
        background: {p['bg']};
        border: 1px solid {p['border']};
        border-radius: 10px;
    }}
    QFrame#parameterSection, QFrame#plotControlCard, QFrame#pipelineCard,
    QFrame#plotFrame, QFrame#pipelinePanel, QWidget#projectPanel,
    QWidget#embeddedEditorCard, QWidget#generatorEditorPanel {{
        background: {p['surface']};
        border: 1px solid {p['border']};
        border-radius: 8px;
    }}
    QFrame#parameterSubsection {{
        background: transparent;
        border: none;
        border-radius: 0;
    }}
    QFrame#componentLibraryRow {{
        background: {p['row_bg']};
        border: 1px solid {p['border_soft']};
        border-radius: 7px;
    }}
    QFrame#sectionCard, QWidget#sectionCard {{
        background: {p['surface']};
        border: 1px solid {p['card_border']};
        border-radius: 8px;
    }}
    QWidget#educationalParamBand {{
        background: {p['bg']};
    }}
    QLabel#panelHeading {{
        font-weight: 700;
        font-size: 13pt;
        color: {p['heading']};
    }}
    QLabel#hintLabel {{
        color: {p['hint']};
    }}

    QLabel {{ background: transparent; border: none; color: {p['text']}; }}
    QLabel#mutedLabel {{ color: {p['muted']}; }}
    QLabel#accentLabel {{ color: {p['accent']}; font-weight: 600; }}
    QLabel#statusDot {{ color: {p['success']}; font-weight: 700; }}
    QLabel#panelTitle {{ font-size: 13pt; font-weight: 700; color: {p['text']}; }}
    QLabel#subSectionTitle {{ font-size: 10.5pt; font-weight: 600; color: {p['text']}; }}
    QPushButton#sectionHeader {{
        background: transparent; border: none; color: {p['text']};
        font-weight: 700; text-align: left; padding: 8px 10px;
    }}

    /* Buttons --------------------------------------------------------------*/
    QPushButton {{
        background: {p['surface']};
        border: 1px solid {p['border']};
        border-radius: 6px;
        color: {p['text']};
        font-weight: 600;
        padding: 7px 10px;
    }}
    QPushButton:hover {{
        background: {p['accent_soft']};
        border-color: {p['accent']};
    }}
    QPushButton:pressed {{
        background: {p['border']};
        border-color: {p['accent']};
    }}
    QPushButton[variant="primary"] {{
        background: {p['accent']};
        border: 1px solid {p['accent']};
        color: #ffffff;
        font-weight: 700;
    }}
    QPushButton[variant="primary"]:hover {{
        background: {p['accent_hover']};
        border-color: {p['accent_hover']};
    }}
    QPushButton[variant="primary"]:pressed {{
        background: {p['accent_active']};
        border-color: {p['accent_active']};
    }}
    QPushButton:disabled {{
        background: #eef1f5;
        border-color: #dde1e8;
        color: #a6acb8;
    }}
    QPushButton[variant="primary"]:disabled {{
        background: #c4cad5;
        border-color: #c4cad5;
        color: #eef1f5;
    }}
    QPushButton[variant="dashed"] {{
        background: transparent;
        border: 1px dashed {p['dashed']};
        color: {p['text']};
    }}
    QPushButton[variant="legacySuccess"] {{
        background-color: {p['legacy_success']};
        color: #ffffff;
        font-weight: bold;
        padding: 8px;
    }}
    QPushButton[variant="legacyWarning"] {{
        background-color: {p['legacy_warning']};
        color: #ffffff;
        font-weight: bold;
        padding: 8px;
    }}
    QPushButton[variant="toggle"] {{
        padding: 5px 14px;
    }}
    QPushButton[variant="toggle"]:checked {{
        background: {p['accent']};
        border: 1px solid {p['accent']};
        color: #ffffff;
        font-weight: 700;
    }}
    QPushButton[variant="toggle"]:pressed {{
        background: {p['accent_active']};
        border: 1px solid {p['accent_active']};
        color: #ffffff;
    }}
    QWidget#nodeActions {{ background: transparent; }}
    QPushButton#nodeActionBtn {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 2px;
    }}
    QPushButton#nodeActionBtn:hover {{
        background: {p['accent_soft']};
        border-color: {p['accent']};
    }}
    QPushButton#nodeActionBtn:pressed {{
        background: {p['accent']};
        border-color: {p['accent']};
    }}

    /* Inputs ---------------------------------------------------------------*/
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {{
        background: {p['field_bg']};
        border: 1px solid {p['border']};
        border-radius: 5px;
        min-height: 22px;
        padding: 3px 6px;
        color: {p['text']};
    }}
    QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {{
        border-color: {p['accent']};
    }}
    QCheckBox {{ color: {p['text']}; padding: 4px 0; background: transparent; }}
    QCheckBox::indicator {{
        width: 14px; height: 14px;
        border: 1.5px solid {p['border']};
        border-radius: 3px;
        background: {p['surface']};
    }}
    QCheckBox::indicator:checked {{
        background: {p['accent']};
        border-color: {p['accent']};
        image: url("{checkbox_checkmark}");
    }}
    QRadioButton {{ color: {p['text']}; padding: 4px 0; background: transparent; }}
    QRadioButton::indicator {{
        width: 14px; height: 14px;
        border: 1.5px solid {p['border']};
        border-radius: 7px;
        background: {p['surface']};
    }}
    QRadioButton::indicator:checked {{
        background: {p['accent']};
        border-color: {p['accent']};
    }}

    /* Lists / trees --------------------------------------------------------*/
    QTreeWidget, QListWidget {{
        background: {p['surface']};
        border: 1px solid {p['border_soft']};
        border-radius: 8px;
        padding: 6px;
        outline: 0;
    }}
    QTreeWidget::item {{ min-height: 26px; padding: 3px; }}
    QTreeWidget::item:selected, QListWidget::item:selected {{
        background: {p['accent_soft']};
        color: {p['accent']};
        border-radius: 5px;
    }}

    /* Tabs -----------------------------------------------------------------*/
    QTabWidget::pane {{ border: none; border-top: 1px solid {p['border_soft']}; }}
    QTabBar::tab {{
        padding: 8px 16px; color: {p['muted']};
        background: transparent; border: none;
    }}
    QTabBar::tab:selected {{
        color: {p['accent']};
        border-bottom: 2px solid {p['accent']};
        font-weight: 600;
    }}

    /* Stepper badges -------------------------------------------------------*/
    QFrame#stepChip {{ background: transparent; border-radius: 8px; }}
    QFrame#stepChip[state="active"] {{ background: {p['accent_soft']}; }}
    QLabel#stepBadge {{
        min-width: 24px; max-width: 24px; min-height: 24px; max-height: 24px;
        border-radius: 12px; background: #c2c9d6; color: #ffffff;
        font-weight: 700;
    }}
    QLabel#stepBadge[state="active"] {{ background: {p['accent']}; color: #ffffff; }}
    QLabel#stepBadge[state="done"] {{ background: {p['success']}; color: #ffffff; }}
    QLabel#stepLabel[state="active"] {{ color: {p['accent']}; font-weight: 700; }}
    QLabel#stepLabel {{ color: {p['muted']}; }}

    /* Top toolbar ----------------------------------------------------------*/
    QToolBar {{
        background: {p['surface']};
        border-bottom: 1px solid {p['border']};
        spacing: 10px; padding: 6px 10px;
    }}
    QToolButton {{ padding: 5px 8px; border-radius: 5px; color: {p['text']}; }}
    QToolButton:hover {{ background: {p['accent_soft']}; }}

    /* Scrollbars -----------------------------------------------------------*/
    QScrollBar:vertical, QScrollBar:horizontal {{
        background: transparent; border: none;
    }}
    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
        background: {p['border']}; border-radius: 5px; min-height: 24px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    """


# --- Matplotlib styling ------------------------------------------------------
def style_figure(fig, p: dict[str, str]) -> None:
    """Recolor a matplotlib ``Figure`` (and its axes) to match palette ``p``."""
    fig.set_facecolor(p["fig_bg"])
    for ax in fig.get_axes():
        ax.set_facecolor(p["axes_bg"])
        for spine in ax.spines.values():
            spine.set_color(p["axes_fg"])
        ax.tick_params(colors=p["axes_fg"])
        ax.xaxis.label.set_color(p["axes_fg"])
        ax.yaxis.label.set_color(p["axes_fg"])
        if ax.get_title():
            ax.title.set_color(p["axes_fg"])


# --- Theme manager -----------------------------------------------------------
class ThemeManager(QObject):
    """Applies the light palette to the QApplication."""

    mode_changed = Signal(dict)

    def __init__(self, app: QApplication, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._app = app
        self.mode = "light"
        self.palette = LIGHT

    def set_mode(self, mode: str) -> None:
        """Apply the light style."""
        self.palette = LIGHT
        self.mode = "light"
        self._app.setStyleSheet(build_qss(self.palette))
        self._app.setFont(QFont("Segoe UI", 10))
        self.mode_changed.emit(self.palette)
