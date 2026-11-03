"""Small rich-help popups shared by Generator and Education panels."""
from __future__ import annotations

import re
from typing import Callable, List, Optional

from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QPushButton, QVBoxLayout,
)

from portal_studio.gui.common.icons import make_alert_help_icon, make_help_icon, make_icon_button


HELP_BUTTON_QSS = (
    "QPushButton#sectionHelpButton {"
    "  border: 1px solid #cfd6e4;"
    "  border-radius: 10px;"
    "  background: #f8fafc;"
    "  color: #334155;"
    "  font-weight: 700;"
    "  padding: 0px;"
    "}"
    "QPushButton#sectionHelpButton:hover {"
    "  background: #e8f0fe;"
    "  border-color: #8ab4f8;"
    "}"
)
PARAMETER_HELP_BUTTON_QSS = (
    "QPushButton#parameterHelpButton {"
    "  border: 1px solid #cfd6e4;"
    "  border-radius: 10px;"
    "  background: #f8fafc;"
    "  color: #334155;"
    "  font-weight: 700;"
    "  padding: 0px;"
    "}"
    "QPushButton#parameterHelpButton:hover {"
    "  background: #fff7ed;"
    "  border-color: #f59e0b;"
    "}"
)
HELP_TEXT_QSS = (
    "font-size: 13px;"
    "line-height: 1.45;"
    "color: #0f172a;"
)
HELP_TERM_STYLE = "font-weight: 600; color: #0f172a;"
FORMULA_SYMBOLS = {
    "phi": "φ",
    "rho": "ρ",
    "beta": "β",
    "Delta": "Δ",
    "theta": "θ",
    "kappa": "κ",
    "mu": "μ",
    "gamma": "γ",
    "omega": "ω",
    "sigma": "σ",
    "eta": "η",
}
MATH_LABELS = {
    "base_form": "Base form",
    "block_dim": "Dimension",
    "block_weight": "Weight",
    "beta": "β",
    "Delta": "Δ",
    "r_ref": "r<sub>ref</sub>",
    "theta_min": "θ<sub>min</sub>",
    "theta_max": "θ<sub>max</sub>",
    "phi_k": "φ<sub>k</sub>",
    "beta_k": "β<sub>k</sub>",
    "c_k": "c<sub>k</sub>",
    "T_k": "T<sub>k</sub>",
    "R_k": "R<sub>k</sub>",
    "Psi_k": "Ψ<sub>k</sub>",
    "z_k": "z<sub>k</sub>",
    "p_single": "p",
    "transform_type": "Transform type",
}
GREEK_BASES = {
    "kappa": "κ",
    "rho": "ρ",
    "mu": "μ",
    "gamma": "γ",
    "omega": "ω",
    "sigma": "σ",
    "eta": "η",
    "theta": "θ",
    "beta": "β",
    "phi": "φ",
}
SIGN_SUFFIXES = {"plus": "+", "minus": "-"}


def math_text(text: str) -> str:
    """Return paper-style display text for internal PORTAL parameter names."""
    raw = str(text)
    suffix = ":" if raw.endswith(":") else ""
    key = raw[:-1] if suffix else raw
    if key in MATH_LABELS:
        return MATH_LABELS[key] + suffix

    if "_" not in key:
        for name, symbol in GREEK_BASES.items():
            if key.startswith(name) and key[len(name):].isdigit():
                return f"{symbol}<sub>{key[len(name):]}</sub>{suffix}"
        return GREEK_BASES.get(key, key) + suffix

    base, tail = key.rsplit("_", 1)
    base_symbol = GREEK_BASES.get(base, base)
    if base_symbol == base:
        for name, symbol in GREEK_BASES.items():
            if base.startswith(name):
                remainder = base[len(name):]
                base_symbol = symbol
                if remainder:
                    base_symbol += f"<sub>{remainder}</sub>"
                break
    if tail in SIGN_SUFFIXES:
        return f"{base_symbol}<sup>{SIGN_SUFFIXES[tail]}</sup>{suffix}"
    if tail:
        return f"{base_symbol}<sub>{tail}</sub>{suffix}"
    return base_symbol + suffix


def make_help_button(tip: str, handler: Callable[[], None]) -> QPushButton:
    btn = make_icon_button(
        make_help_icon(),
        tip,
        handler,
        size=22,
        object_name="sectionHelpButton",
    )
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(HELP_BUTTON_QSS)
    return btn


def make_parameter_help_button(tip: str, handler: Callable[[], None]) -> QPushButton:
    btn = make_icon_button(
        make_help_icon(),
        tip,
        handler,
        size=22,
        object_name="parameterHelpButton",
    )
    btn.setIcon(QIcon())
    btn.setText("?")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(PARAMETER_HELP_BUTTON_QSS)
    return btn


def term(text: str) -> str:
    return f"<span style=\"{HELP_TERM_STYLE}\">{math_text(text)}</span>"


def help_body(heading: str, paragraphs: list[str]) -> str:
    parts = []
    for p in paragraphs:
        # A paragraph wrapped in <center>...</center> renders centred (e.g. a
        # standalone formula line); everything else stays left-aligned.
        if p.startswith("<center>") and p.endswith("</center>"):
            inner = p[len("<center>"):-len("</center>")]
            parts.append(
                f"<p align=\"center\" style=\"margin: 0 0 10px 0;\">{inner}</p>")
        else:
            parts.append(f"<p style=\"margin: 0 0 10px 0;\">{p}</p>")
    return (
        f"<div style=\"{HELP_TEXT_QSS}\">"
        f"{''.join(parts)}</div>"
    )


def formula_intro(purpose: str, variables: str) -> list[str]:
    return [
        f"The formula below {purpose}; here, {variables}.",
    ]


def randomisation_modes(
    entries: list[tuple[str, str]],
    intro: str | None = None,
) -> list[str]:
    return [
        intro or (
            f"{term('Randomisation modes')} define how PORTAL chooses values "
            "when a component is generated."
        ),
        "<ul style=\"margin: 0 0 10px 18px; padding: 0;\">"
        + "".join(
            f"<li>{term(name)}: {description}</li>"
            for name, description in entries
        )
        + "</ul>",
    ]


def formula_tokens(formula: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []

    def append_text(text: str) -> None:
        if not text:
            return
        symbol_pattern = re.compile(
            r"\b(" + "|".join(map(re.escape, FORMULA_SYMBOLS)) + r")\b")
        pos = 0
        for symbol_match in symbol_pattern.finditer(text):
            if symbol_match.start() > pos:
                tokens.append(("text", text[pos:symbol_match.start()]))
            tokens.append(("text", FORMULA_SYMBOLS[symbol_match.group(1)]))
            pos = symbol_match.end()
        if pos < len(text):
            tokens.append(("text", text[pos:]))

    pos = 0
    pattern = re.compile(r"([A-Za-zφρβθκμγωσηΔ\u0304]+)_([A-Za-z0-9,]+)")
    for match in pattern.finditer(formula):
        if match.start() > pos:
            append_text(formula[pos:match.start()])
        base = FORMULA_SYMBOLS.get(match.group(1), match.group(1))
        tokens.append(("text", base))
        tokens.append(("sub", match.group(2)))
        pos = match.end()
    if pos < len(formula):
        append_text(formula[pos:])
    return tokens


def formula_image_label(formula: str) -> QLabel:
    font = QFont("Cambria Math")
    font.setPointSize(13)
    font.setItalic(True)
    sub_font = QFont(font)
    sub_font.setPointSize(9)
    line_tokens = [formula_tokens(line) for line in str(formula).splitlines()]
    metrics = QFontMetrics(font)
    sub_metrics = QFontMetrics(sub_font)
    text_width = max(
        sum(
            (sub_metrics if kind == "sub" else metrics).horizontalAdvance(text)
            for kind, text in tokens
        )
        for tokens in line_tokens
    )
    width = max(420, text_width + 32)
    line_height = metrics.height() + 8
    height = max(50, line_height * len(line_tokens) + 18)
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#f8fafc"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.setPen(QColor("#0f172a"))
    baseline = 12 + metrics.ascent()
    for tokens in line_tokens:
        x = 16
        sub_baseline = baseline + 7
        for kind, text in tokens:
            if kind == "sub":
                painter.setFont(sub_font)
                painter.drawText(x, sub_baseline, text)
                x += sub_metrics.horizontalAdvance(text)
            else:
                painter.setFont(font)
                painter.drawText(x, baseline, text)
                x += metrics.horizontalAdvance(text)
        baseline += line_height
    painter.end()

    label = QLabel()
    label.setObjectName("formulaImageLabel")
    label.setPixmap(pixmap)
    label.setStyleSheet(
        "QLabel#formulaImageLabel {"
        "  background: #f8fafc;"
        "  border: 1px solid #d9e2ef;"
        "  border-radius: 6px;"
        "  padding: 4px;"
        "}"
    )
    return label


def build_help_dialog(parent, title: str, body: str,
                      formula: str | None = None) -> QDialog:
    # Accept ``formula`` for call-site compatibility. Help dialogs render
    # prose only, with no formula image.
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setModal(True)
    dialog.setMinimumWidth(560)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(18, 16, 18, 14)
    layout.setSpacing(12)

    text = QLabel(body)
    text.setWordWrap(True)
    text.setTextFormat(Qt.TextFormat.RichText)
    text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(text)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
    buttons.accepted.connect(dialog.accept)
    layout.addWidget(buttons)
    return dialog


def show_help_dialog(parent, title: str, body: str,
                     formula: str | None = None) -> None:
    dialog = build_help_dialog(parent, title, body, formula)
    dialog.exec()


# --------------------------------------------------------------------------
# Generator parameter help shared by the tab's help handlers.
# --------------------------------------------------------------------------

def plain_math_header(text: str) -> str:
    """``math_text`` with sign superscripts and subscripts flattened to plain text."""
    display = math_text(text)
    display = display.replace("<sup>+</sup>", "+")
    display = display.replace("<sup>-</sup>", "-")
    return re.sub(r"<sub>(.*?)</sub>", r"", display)


def mode_entries(modes: Optional[List[str]]) -> list[tuple[str, str]]:
    descriptions = {
        # Centre modes
        "With Exclusion Zone": "samples the centre outside the excluded interval.",
    # Kappa / exponent / centre modes (GUI labels)
        "Uniform Random": "samples each value uniformly from the visible min/max range.",
        "Fixed Value": "uses the entered value without random sampling.",
        "Fixed Gradient": "spreads exponent values smoothly across dimensions.",
        "Beta Distribution": "draws values from a beta-shaped distribution between min and max.",
        "Linear Spread (Shuffled)": "spreads values across the range and shuffles their order.",
        "Isotropic Random": "samples one isotropic value for the component and repeats it across every dimension.",
        "Fixed Unit Scale (κ=1)": "sets every anisotropy value to 1.0.",
        # Transform inner-parameter modes
        "Independent random": "samples each component and dimension independently.",
        "Component-shared": "uses one sampled value across all dimensions of one component.",
        "Dimension-shared": "uses one sampled value per dimension, shared across components.",
        "Global shared": "uses one sampled value everywhere.",
        "Beta distribution": "draws values from a beta-shaped distribution between min and max.",
        "Linspace with permutation": "spreads values across the range and shuffles their order.",
        "Linear spread (shuffled)": "spreads values across the range and shuffles their order.",
    # Rotation modes (GUI labels)
        "No Rotation": "sets all pairwise angles to zero, so the local axes stay aligned with the original axes.",
        "Fully Connected": "samples an angle for every variable pair.",
        "Probabilistic Coupling": "samples each variable-pair angle with the chosen probability; unselected pairs stay at zero.",
        "Uniform Random Angle": "samples one angle and applies it to every variable pair.",
        "Fixed Angle": "uses the entered angle for every variable pair.",
        "Chain-like Coupling": "rotates neighbouring coordinate pairs only.",
        "Block-Diagonal": "rotates variables within blocks without cross-block mixing.",
    }
    return [(mode, descriptions.get(mode, "follows this parameter's selected sampling rule."))
            for mode in (modes or [])]


def parameter_help_text(name: str, modes: Optional[List[str]], transform_name: str) -> tuple[str, str]:
    display = term(name)
    transform_definitions = {
        ("additive_periodic", "mu_plus"): (
            "μ",
            f"{term('mu_plus')} and {term('mu_minus')} set the "
            "perturbation strength and amplitude of the additive ripple "
            "added to each coordinate on positive and negative sides. "
            "In code this multiplies a gated sinusoid, so larger μ makes "
            "the induced bumps deeper or taller."),
        ("additive_periodic", "mu_minus"): (
            "μ⁻",
            f"{term('mu_minus')} sets the negative-side amplitude of "
            "the additive ripple."),
        ("additive_periodic", "gamma_plus"): (
            "γ",
            f"{term('gamma_plus')} and {term('gamma_minus')} set "
            "the switch-on rate of the additive ripple. Larger γ makes the "
            "factor 1 - exp(-γ |a|) approach full strength closer to the centre."),
        ("additive_periodic", "gamma_minus"): (
            "γ⁻",
            f"{term('gamma_minus')} sets the negative-side switch-on "
            "rate of the additive ripple."),
        ("additive_periodic", "omega_plus"): (
            "ω",
            f"{term('omega_plus')} and {term('omega_minus')} set "
            "the ripple frequency in sin(ω |a|). Larger ω packs extrema "
            "closer together along the coordinate axis."),
        ("additive_periodic", "omega_minus"): (
            "ω⁻",
            f"{term('omega_minus')} sets the negative-side ripple "
            "frequency in sin(ω |a|)."),
        ("log_sinusoidal", "mu_plus"): (
            "μ",
            f"{term('mu_plus')} and {term('mu_minus')} set the "
            "amplitude of the log-space modulation. Larger μ strengthens "
            "the exp(log|a| + modulation) distortion away from the centre."),
        ("log_sinusoidal", "mu_minus"): (
            "μ⁻",
            f"{term('mu_minus')} sets the negative-side amplitude of "
            "the log-space modulation."),
        ("wavelet_mod", "mu_plus"): (
            "μ",
            f"{term('mu_plus')} and {term('mu_minus')} set the "
            "amplitude of the wavelet lobe added along each coordinate. "
            "μ = 0 leaves that side unchanged."),
        ("wavelet_mod", "mu_minus"): (
            "μ⁻",
            f"{term('mu_minus')} sets the negative-side amplitude of "
            "the wavelet lobe."),
        ("wavelet_mod", "omega_plus"): (
            "ω",
            f"{term('omega_plus')} and {term('omega_minus')} set "
            "the oscillation frequency in sin(ω |a| - π/2). The envelope "
            "length is derived from it as L = η / ω."),
        ("wavelet_mod", "omega_minus"): (
            "ω⁻",
            f"{term('omega_minus')} sets the negative-side wavelet "
            "frequency in sin(ω |a| - π/2)."),
        ("tensor_interference", "mu_plus"): (
            "μ",
            f"{term('mu_plus')} and {term('mu_minus')} set the "
            "base amplitude of tensor interference. Enter base values; "
            "the evaluator applies the local dimension-dependent factor "
            "2^((d-1)/2) automatically."),
        ("tensor_interference", "mu_minus"): (
            "μ⁻",
            f"{term('mu_minus')} sets the negative-side strength of "
            "the tensor-interference perturbation."),
        ("tensor_interference", "omega_plus"): (
            "ω",
            f"{term('omega_plus')} and {term('omega_minus')} set "
            "the sine frequency for each dimension. A coordinate is modulated "
            "by its own sin(ω a) term and sin² gates from the other dimensions."),
        ("tensor_interference", "omega_minus"): (
            "ω⁻",
            f"{term('omega_minus')} sets the negative-side sine "
            "frequency used in the tensor-interference gates."),
        ("wavelet_mod", "eta"): (
            "η",
            f"{display} is the Wavelet-Inspired extent scale. The code uses "
            "L = η / ω, so η widens or narrows the wavelet envelope relative "
            "to the selected frequency."),
        ("wavelet_mod", "eps"): (
            "eps",
            f"{display} is the numerical-stability constant and near-zero "
            "guard for the Wavelet-Inspired transform. Values with |a| "
            "below eps are left unchanged."),
        ("radial_polytrig", "mu"): (
            "μ",
            f"{display} controls the radial perturbation strength and "
            "radial displacement. The radial distance is the distance from "
            "the component centre; larger μ makes the concentric-ring "
            "perturbation stronger."),
        ("radial_polytrig", "p"): (
            "p",
            f"{display} is the radial polynomial power. It controls how "
            "quickly the perturbation grows as distance from the centre "
            "increases."),
        ("radial_polytrig", "q"): (
            "q",
            f"{display} is the radial trigonometric power: it controls how "
            "distance enters the sine phase, changing how ring spacing "
            "varies as you move away from the centre."),
        ("radial_polytrig", "omega"): (
            "ω",
            f"{display} is the radial oscillation frequency. "
            "Larger ω packs radial rings closer together."),
        ("radial_polytrig", "eps"): (
            "eps",
            f"{display} is the near-zero radius guard. If the distance from "
            "the component centre is below eps, the transform leaves "
            "near-origin points unchanged."),
    }
    if (transform_name, name) in transform_definitions:
        title, definition = transform_definitions[(transform_name, name)]
        if name.endswith("_plus"):
            definition += (
                f" Leave {term('Separate ± directions')} unchecked to "
                "use the same settings on both sides; check it when you want "
                "independent draws from the shared settings."
            )
        return title, definition
    definitions = {
        "base_form": (
            "Base form",
            "<b>Base form</b><br><br>"
            "<b>Base Form A</b><br>"
            "Controls how each dimension and direction shapes the basin "
            "independently. The curvature and stretch along the positive side "
            "of an axis can differ from the negative side, and each axis can "
            "behave differently from the others. Use this when you want "
            "fine-grained, asymmetric control over basin geometry."
            "<br><br>"
            "<b>Base Form B</b><br>"
            "Controls the basin using a single curvature that applies to the "
            "whole component at once. All dimensions contribute to one shared "
            "powered sum, so the basin grows as a unified shape. Use this when "
            "you want simpler, globally coherent curvature without per-axis "
            "detail."
            "<br><br>"
            "<b>Key difference</b><br>"
            "Form A gives you per-axis, per-direction control (more expressive "
            "but more parameters to configure). Form B gives you one knob for "
            "whole basin; directional anisotropy remains available in both forms."),
        "block_dim": (
            "Dimension",
            "<b>Dimension</b><br>"
            "Controls how many search variables this block uses. The composite "
            "landscape&rsquo;s total dimensionality is the sum of dimensions "
            "across all blocks. Each block requires at least two dimensions."),
        "block_weight": (
            "Weight",
            "<b>Weight</b><br>"
            "Controls how strongly this block&rsquo;s fitness contributes to "
            "the composite score. The composite evaluates a weighted sum across "
            "all blocks: a higher weight means this block has more influence on "
            "the overall landscape. The default is 1.0."),
        "c": (
            "c",
            f"{display} is the component centre. It defines where the basin is located "
            "before rotation and transformation are applied."),
        "beta": (
            "β",
            "<b>Beta (&beta;)</b><br>"
            "Sets the component value at its centre. Within a block, the lowest "
            "&beta; gives the minimum value when its centre lies in the domain. "
            "Other components may be hidden by the lower envelope; a higher "
            "&beta; does not guarantee a visible local optimum."),
        "rotation": (
            "Rotation",
            f"{display} controls the orientation of the local coordinate system. "
            f"The generator fills the angle matrix {term('Psi_k')}, "
            f"which is used to construct {term('R_k')} in the internal "
            "coordinate definition. Angle controls and tables use degrees."),
        "kappa_plus": (
            "κ",
            f"{term('kappa_plus')} and {term('kappa_minus')} are "
            "positive and negative anisotropy factors. They control elongation "
            "and conditioning along each dimension and direction. Leave "
            f"{term('Separate ± directions')} unchecked to use the same "
            "κ settings for positive and negative directions; check it when "
            "you want independent draws from the shared settings."),
        "kappa_minus": (
            "κ⁻",
            f"{term('kappa_minus')} is the negative-direction anisotropy "
            "factor. It can be matched with or separated from "
            f"{term('kappa_plus')}."),
        "p_plus": (
            "p",
            f"{term('p_plus')} and {term('p_minus')} are the "
            "positive and negative exponents for Base Form A. They control "
            "directional curvature. Leave "
            f"{term('Separate ± directions')} unchecked to use the same "
            "exponent settings for positive and negative directions; check it "
            "when you want independent draws from the shared settings."),
        "p_minus": (
            "p⁻",
            f"{term('p_minus')} is the negative-direction exponent for "
            "Base Form A. It can be matched with or separated from "
            f"{term('p_plus')}."),
        "p_single": (
            "p",
            f"{display} is the single exponent used by Base Form B."),
        "neutralization": (
            "Neutralization",
            f"Neutralization calibrates a baseline reference rise using {term('Delta')} "
            f"and the per-coordinate displacement {term('r_ref')}. "
            "Directional differences remain; disabling it sets the scaling factors to 1."),
        "transform_type": (
            "Transform type",
            f"{display} selects the nonlinear transformation added to the "
            "component's internal coordinates. Use Transform Sequence to "
            "choose and order transforms; they are applied in order to the "
            "rotated component coordinates."),
        "mu_plus": (
            "μ",
            f"{term('mu_plus')} and {term('mu_minus')} control "
            "transform perturbation strength on positive and negative sides. "
            f"Leave {term('Separate ± directions')} unchecked to use the same "
            "μ settings on both sides; check it when you want independent draws "
            "from the shared settings."),
        "mu_minus": (
            "μ⁻",
            f"{term('mu_minus')} controls transform strength on the "
            "negative direction."),
        "gamma_plus": (
            "γ",
            f"{term('gamma_plus')} and {term('gamma_minus')} "
            "control how quickly the transform envelope ramps away from the centre "
            "on positive and negative sides. "
            f"Leave {term('Separate ± directions')} unchecked to use the same "
            "γ settings on both sides; check it when you want independent draws "
            "from the shared settings."),
        "gamma_minus": (
            "γ⁻",
            f"{term('gamma_minus')} controls the negative-direction "
            "envelope ramp."),
        "omega_plus": (
            "ω",
            f"{term('omega_plus')} and {term('omega_minus')} "
            "control spacing or frequency of transform oscillations on positive "
            "and negative sides. "
            f"Leave {term('Separate ± directions')} unchecked to use the same "
            "ω settings on both sides; check it when you want independent draws "
            "from the shared settings."),
        "omega_minus": (
            "ω⁻",
            f"{term('omega_minus')} controls negative-direction "
            "transform spacing or frequency."),
        "omega1_plus": (
            "ω₁",
            f"{term('omega1_plus')} and {term('omega1_minus')} "
            "are the first Log-Sinusoidal log-space frequencies on positive "
            "and negative sides. They set one oscillation rate in "
            f"{term('sin')}({term('omega1')} log|a|), and work "
            f"with {term('omega2_plus')} and {term('omega2_minus')} "
            "to create multi-scale beating ruggedness. "
            f"Leave {term('Separate ± directions')} unchecked to use the same "
            "ω₁ settings on both sides; check it when you want independent draws "
            "from the shared settings."),
        "omega1_minus": (
            "ω₁⁻",
            f"{term('omega1_minus')} controls the negative-side first "
            "Log-Sinusoidal log-space frequency."),
        "omega2_plus": (
            "ω₂",
            f"{term('omega2_plus')} and {term('omega2_minus')} "
            "are the second Log-Sinusoidal log-space frequencies on positive "
            "and negative sides. They set the companion oscillation rate in "
            f"{term('sin')}({term('omega2')} log|a|), and work "
            f"with {term('omega1_plus')} and {term('omega1_minus')} "
            "to create multi-scale beating ruggedness. "
            f"Leave {term('Separate ± directions')} unchecked to use the same "
            "ω₂ settings on both sides; check it when you want independent draws "
            "from the shared settings."),
        "omega2_minus": (
            "ω₂⁻",
            f"{term('omega2_minus')} controls the negative-side second "
            "Log-Sinusoidal log-space frequency."),
        "sigma": (
            "σ",
            f"{display} is the Log-Sinusoidal origin-smoothing scale. It "
            "controls how the log-sinusoidal pattern is softened near the "
            "component centre."),
        "eta": (
            "η",
            f"{display} is the Wavelet-Inspired extent scale. PORTAL uses "
            "it with the selected frequency to set the spatial extent of "
            "the wavelet envelope."),
        "eps": (
            "eps",
            f"{display} is a numerical-stability constant used as a near-zero "
            "guard in transform calculations. Keep min and max equal unless you "
            "are deliberately testing numerical sensitivity."),
        "mu": (
            "μ",
            f"{display} is the Radial Polynomial-Trigonometric radial "
            "perturbation strength."),
        "p": (
            "p",
            f"{display} is the radial polynomial power in the Radial "
            "Polynomial-Trigonometric transform."),
        "q": (
            "q",
            f"{display} is the radial trigonometric power in the Radial "
            "Polynomial-Trigonometric transform."),
        "omega": (
            "ω",
            f"{display} is the radial oscillation frequency in the Radial "
            "Polynomial-Trigonometric transform."),
    }
    return definitions.get(name, (plain_math_header(name), f"{display} is a PORTAL parameter."))


def scalar_range_help(name: str) -> str:
    display = term(name)
    if name == "eps":
        return (
            f"{display} is normally fixed: set min and max to the same "
            "value to keep the engine default."
        )
    return (
        f"The Generator draws one scalar value for {display} uniformly "
        "between min and max when the component is generated. If min equals max, "
        "the value is fixed."
    )


def parameter_help_formula(name: str) -> str | None:
    formulas = {
        "rotation": "z_k = T_k(R_k(x - c_k))",
        "neutralization": "|z_i| = r_ref",
        "transform_type": "z' = T_n(...T_2(T_1(z)))",
    }
    return formulas.get(name)


# Coloured sub-headings shared by the self-contained parameter helps. The
# QLabel help body is RichText, so inline <span style="color"> is honoured.
_SEPARATE_DIR_HEADING = (
    '<b><span style="color: #8b0000;">Separate &plusmn; Directions</span></b>')

# Education (2D, manual-entry) variant of the ± explanation: there are no
# sampling modes here, so it is about typing one pair of numbers vs two.
_EDU_SEP_DIR_NOTE = (
    "By default, the + and &minus; side of each axis use the same value, so "
    "you enter one pair of numbers. Tick this to enter the positive and "
    "negative side of each axis independently.")


def _section_title(text: str) -> str:
    """Coloured, bold sub-heading that labels a list of options."""
    return f'<b><span style="color: #1d4ed8;">{text}</span></b>'


def _modes_title(count_word: str) -> str:
    """Coloured 'There are <N> modes to set it:' sub-heading."""
    return _section_title(f"There are {count_word} modes to set it:")


def seed_help_paragraphs() -> List[str]:
    """Self-contained help body for the component seed."""
    items = [
        ("New Component",
         "the first component uses the displayed seed (default 42). "
         "Subsequent components start with one above the highest existing seed."),
        ("Identical components",
         "use Clone to duplicate the realised component, or save the prepared "
         "instance JSON to preserve it for evaluation. Rerolls, row seeds and "
         "copied values also affect regeneration."),
    ]
    bullets = "".join(f"<li><b>{n}</b>: {d}</li>" for n, d in items)
    return [
        "<b>Seed</b>",
        "Sets the component's base random seed for centre, curvature, "
        "anisotropy, rotation and transformation parameters. Reproducing "
        "generated values also requires matching settings, dimension, reroll "
        "state, row-seed overrides and copied values.",
        _section_title("How the seed is used:"),
        "<ul style=\"margin: 0 0 10px 18px; padding: 0;\">" + bullets + "</ul>",
    ]


def component_table_help_paragraphs() -> List[str]:
    """Self-contained help body for the generated-parameter data table."""
    items = [
        ("Reroll",
         "shows the dice column. Clicking it redraws one parameter row from "
         "its mode and range. A matched negative-side row follows the positive "
         "row. Rerolling a copied value removes its copy override."),
        ("Seeded reroll",
         "shows the manual-seed column. Clicking it asks for a seed for this "
         "parameter row and redraws the row from that seed. Use the same "
         "manual seed on the same row in another compatible component when "
         "you need the same stream and generation settings. Right-click the "
         "seed button and select Clear row seed to remove an override."),
        ("Copy",
         "shows the copy column. It sets a parameter to another "
         "component&rsquo;s value. The list only shows compatible components "
         "(same dimension and base form, and the same transformation type at "
         "the same sequence position for transformation parameters). This is "
         "a snapshot; later source edits do not update it. Copying a &plusmn; pair also copies the source&rsquo;s "
         "direction, so the whole pair becomes identical."),
    ]
    bullets = "".join(f"<li><b>{n}</b>: {d}</li>" for n, d in items)
    return [
        "<b>Generated parameters</b>",
        "Displays the component's realised values. Cells are read-only; the "
        "action buttons regenerate or copy values. Numbered columns show "
        "values by dimension.",
        _section_title("Action checkboxes:"),
        "The checkboxes above the table choose which action columns are shown. "
        "Leave a checkbox off when you only want to inspect values.",
        "<ul style=\"margin: 0 0 10px 18px; padding: 0;\">" + bullets + "</ul>",
    ]


def base_form_help_paragraphs() -> List[str]:
    """Self-contained help body for the base form selector."""
    forms = [
        ("Base Form A",
         "Controls how each dimension and direction shapes the basin "
         "independently. The curvature and stretch along the positive side of "
         "an axis can differ from the negative side, and each axis can behave "
         "differently from the others. Use this when you want fine-grained, "
         "asymmetric control over basin geometry."),
        ("Base Form B",
         "Controls the basin using a single curvature that applies to the "
         "whole component at once. All dimensions contribute to one shared "
         "powered sum, so the basin grows as a unified shape. Use this when "
         "you want simpler, globally coherent curvature without per-axis "
         "detail."),
    ]
    bullets = "".join(f"<li><b>{n}</b>: {d}</li>" for n, d in forms)
    return [
        "<b>Base form</b>",
        "Selects how the component&rsquo;s basin is built.",
        _section_title("There are two forms to choose from:"),
        "<ul style=\"margin: 0 0 10px 18px; padding: 0;\">" + bullets + "</ul>",
        "<b>Key difference</b>",
        "Form A gives you per-axis, per-direction control (more expressive but "
        "more parameters to configure). Form B gives you one knob for the "
        "whole basin. Form A is additively separable before rotation with "
        "identity or coordinate-wise transformations; Form B generally couples "
        "coordinates when its exponent differs from 1. Both support directional anisotropy.",
    ]


def centre_help_paragraphs(educational: bool = False) -> List[str]:
    """Self-contained help body for the component centre (c).

    ``educational`` drops the randomisation modes (the Education tab enters the
    centre manually in 2D).
    """
    paras = [
        "<b>Component Center (c)</b>",
        "Sets where this component&rsquo;s optimal point sits in the search "
        "space.",
    ]
    if educational:
        return paras
    modes = [
        ("Uniform Random",
         "Samples each centre coordinate uniformly between the displayed "
         "minimum and maximum."),
        ("With Exclusion Zone",
         "Samples each centre coordinate outside the specified interval, "
         "within the sampling bounds. For example, [&minus;20,20] excludes "
         "that coordinate band; the interval need not be centred at zero."),
    ]
    bullets = "".join(f"<li><b>{n}</b>: {d}</li>" for n, d in modes)
    paras += [
        _section_title("There are two modes to place it:"),
        "<ul style=\"margin: 0 0 10px 18px; padding: 0;\">" + bullets + "</ul>",
    ]
    return paras


def exponent_help_paragraphs(educational: bool = False) -> List[str]:
    """Self-contained help body for the Base Form A exponents (p).

    ``educational`` drops the randomisation modes and uses the manual-entry ±
    note (the Education tab types values directly in 2D)."""
    modes = [
        ("Uniform Random",
         "Assigns a different random curvature to each dimension "
         "independently. Use this if you want a varied basin shape with no "
         "special structure."),
        ("Fixed Value",
         "Applies the same curvature to every dimension. Use this if you want "
         "a common exponent; anisotropy and transformations can still produce "
         "directional differences."),
        ("Fixed Gradient",
         "Spreads curvature values evenly across dimensions in a fixed order: "
         "the first dimension always gets the lowest curvature, the "
         "last always gets the highest. Use this if you want a predictable, "
         "structured progression across axes."),
        ("Beta Distribution",
         "Samples exponents within the bounds using positive shape controls "
         "alpha and beta. Equal controls give a symmetric distribution; "
         "unequal controls favour one end."),
        ("Linear Spread (Shuffled)",
         "Same as Fixed Gradient (the same spread of values from "
         "lowest to highest), but the assignment to dimensions is "
         "randomized. So the full range of curvatures is always represented, "
         "but you cannot predict which axis gets which value. Use this if you "
         "want coverage of the full curvature range without a structured axis "
         "ordering."),
    ]
    bullets = "".join(
        f"<li><b>{name}</b>: {desc}</li>" for name, desc in modes)
    paras = [
        "<b>Exponent (p)</b>",
        "Controls growth away from the component centre: p = 0.5 gives linear "
        "growth and p = 1 gives quadratic growth in the baseline. Values below "
        "0.5 give sublinear growth; values above 0.5 give superlinear growth. "
        "Generator's default sampling range is 0.2&ndash;1.2.",
    ]
    if not educational:
        paras += [
            _modes_title("five"),
            "<ul style=\"margin: 0 0 10px 18px; padding: 0;\">" + bullets + "</ul>",
        ]
    paras += [
        _SEPARATE_DIR_HEADING,
        _EDU_SEP_DIR_NOTE if educational else
        "By default, the positive and negative sides of each axis share the "
        "same curvature value. Enabling this allows the two sides to differ. "
        "The same mode and range apply, but independent values are drawn for "
        "each side. Use this if you want an asymmetric basin where, for "
        "example, the landscape rises steeply on one side of the optimum but "
        "gently on the other.",
    ]
    return paras


def kappa_help_paragraphs(educational: bool = False) -> List[str]:
    """Self-contained help body for the anisotropy factors (kappa).

    ``educational`` drops the randomisation modes and uses the manual-entry ±
    note (the Education tab types values directly in 2D)."""
    modes = [
        ("Uniform Random",
         "Assigns a different random stretch to each dimension independently. "
         "Use this if you want an irregular, anisotropic basin with no special "
         "structure."),
        ("Beta Distribution",
         "Samples stretch values from a skewed distribution, pushing most "
         "values toward either the low or high end of the range. Use this if "
         "you want to control whether most axes are similarly scaled or "
         "whether extreme elongation dominates."),
        ("Isotropic Random",
         "Assigns one sampled anisotropy value to every dimension. Exponents, "
         "directional settings and transformations can still affect geometry."),
        ("Linear Spread (Shuffled)",
         "Spreads stretch values evenly across the full range and assigns them "
         "to dimensions randomly. Guarantees the full contrast between the "
         "flattest and steepest axes is always present. Use this if you want a "
         "specified directional contrast with shuffled axis assignments."),
        ("Fixed Unit Scale (κ=1)",
         "Sets all anisotropy values to 1. This removes directional contrast "
         "from &kappa;, while exponents and transformations still shape the basin."),
    ]
    bullets = "".join(
        f"<li><b>{name}</b>: {desc}</li>" for name, desc in modes)
    paras = [
        "<b>Kappa (&kappa;)</b>",
        "Sets directional scaling. The largest-to-smallest &kappa; ratio is "
        "the prescribed directional conditioning ratio; exponents, transformations "
        "and component competition also affect geometry and difficulty. "
        "Generator's default sampling range is 1&ndash;100.",
    ]
    if not educational:
        paras += [
            _modes_title("five"),
            "<ul style=\"margin: 0 0 10px 18px; padding: 0;\">" + bullets + "</ul>",
        ]
    paras += [
        _SEPARATE_DIR_HEADING,
        _EDU_SEP_DIR_NOTE if educational else
        "By default, the positive and negative sides of each axis share the "
        "same stretch value. Enabling this allows the two sides to differ. The "
        "same mode and range apply, but independent values are drawn for each "
        "side. Use this if you want an asymmetric basin where the landscape is "
        "more elongated on one side of the optimum than the other.",
    ]
    return paras


def rotation_help_paragraphs() -> List[str]:
    """Self-contained help body for the rotation parameter."""
    # The modes selectable from the dropdown when the "Controlled Partial
    # Separable" box is unchecked. How separable each leaves the problem
    # depends on the mode and the chosen angle, so they are not labelled
    # "fully non-separable".
    non_separable = [
        ("Fully Connected",
         "Samples an angle for every coordinate pair."),
        ("Probabilistic Coupling",
         "Selects each coordinate pair with the given probability, then assigns "
         "angles to selected pairs. A probability of 0.5 selects each pair "
         "with a 50% chance."),
        ("Uniform Random Angle",
         "Applies one sampled angle to every coordinate pair."),
        ("Fixed Angle",
         "Applies one user-entered angle to every coordinate pair."),
        ("Chain-like Coupling",
         "Assigns angles to neighbouring pairs: 1&harr;2, 2&harr;3, and so on. "
         "Accumulated rotations can mix more distant coordinates."),
    ]
    def bullets(modes):
        return ("<ul style=\"margin: 0 0 10px 18px; padding: 0;\">"
                + "".join(f"<li><b>{n}</b>: {d}</li>" for n, d in modes)
                + "</ul>")

    return [
        "<b>Rotation</b>",
        "Enter angles in degrees: a quarter turn is 90&deg; and a half turn is 180&deg;.",
        "Sets the orientation of the component's internal coordinates. Zero "
        "angles remove rotation-induced mixing. Interactions also depend on "
        "the base form, anisotropy, transformations and component competition; "
        "rotation alone does not determine separability or interaction strength.",
        '<b><span style="color: #1d4ed8;">Tick <b>Enable</b>, then choose one '
        'of two categories:</span></b>',
        "<b>Standard rotation modes</b>",
        "Leave <b>Controlled Partial Separable</b> unchecked and pick a mode "
        "from the dropdown. These modes specify which coordinate planes receive "
        "rotation angles:",
        bullets(non_separable),
        "<b>Controlled Partial Separable</b>",
        "Check this box to rotate coordinates only within groups, leaving "
        "cross-group rotation entries zero. Other construction choices can "
        "still introduce cross-group interactions. Press <b>Configure</b> to set the "
        "group sizes (Auto or Manual) and the per-group angle (Fixed or "
        "Randomize).",
    ]


_DELTA_DESC = (
    "Sets the target increase in objective value above the component's value "
    "at its centre, using r<sub>ref</sub> as the reference displacement. "
    "A larger &Delta; gives the baseline component a larger vertical scale.")
_R_REF_DESC = (
    "Sets the positive reference displacement in each internal coordinate.")


def neutralization_help_paragraphs() -> List[str]:
    """Self-contained help body for neutralization (with its Δ / r_ref knobs)."""
    bullets = (
        f"<li><b>&Delta; (Delta)</b>: {_DELTA_DESC}</li>"
        f"<li><b>r<sub>ref</sub> (Reference Displacement)</b>: {_R_REF_DESC}</li>")
    return [
        "<b>Neutralization</b>",
        "When enabled, PORTAL computes scaling factors from the exponents, "
        "sign-averaged anisotropy, &Delta; and r<sub>ref</sub>. This calibrates "
        "the baseline at a common internal-coordinate reference. It does not "
        "make all directions or all displacements equally high, and does not "
        "guarantee equal basin exposure or optimizer difficulty. When disabled, "
        "the scaling factors are 1.",
        _section_title("It is controlled by two parameters:"),
        "<ul style=\"margin: 0 0 10px 18px; padding: 0;\">" + bullets + "</ul>",
    ]


def delta_help_paragraphs() -> List[str]:
    """The Δ (Delta) section."""
    return ["<b>&Delta; (Delta)</b>", _DELTA_DESC]


def r_ref_help_paragraphs() -> List[str]:
    """The r_ref (reference displacement) section."""
    return ["<b>r<sub>ref</sub> (Reference Displacement)</b>", _R_REF_DESC]


def transform_type_help_paragraphs() -> List[str]:
    """Self-contained help body for the transform type selector."""
    transforms = [
        ("Additive Periodic",
         "Creates periodic perturbations along each internal coordinate. "
         "The perturbation builds up gradually from "
         "the center and reaches full effect further out. Use this for "
         "regular, axis-aligned local optima with uniform density."),
        ("Log-Sinusoidal",
         "Creates oscillatory modulation across logarithmic coordinate scales, "
         "with smoothing near the centre."),
        ("Wavelet-Inspired Modulation",
         "Creates ruggedness concentrated in a specific band around the center: "
         "strongest in the middle zone, fading both toward the center "
         "and toward the edges. Use this for localized ruggedness that leaves "
         "the outer basin smooth."),
        ("Tensor Interference",
         "Introduces cross-dimensional coupling by creating a checkerboard "
         "pattern of local optima across the basin. Unlike the element-wise "
         "transforms above, this one makes dimensions interact. Use this for "
         "grid-like interference patterns that span the full basin."),
        ("Radial Polynomial-Trigonometric",
         "Applies a radial perturbation with no directional preference of its "
         "own. Baseline anisotropy and other transformations can still make "
         "the completed landscape direction-dependent."),
    ]
    bullets = "".join(
        f"<li><b>{name}</b>: {desc}</li>" for name, desc in transforms)
    return [
        "<b>Transform Type</b>",
        "Adds ruggedness to the baseline basin. Transformations preserve the "
        "component centre but can introduce additional minima and alter the "
        "surface at finite displacements. They are generally non-commutative, "
        "so their sequence can change the landscape.",
        _section_title("There are five transforms to choose from:"),
        "<ul style=\"margin: 0 0 10px 18px; padding: 0;\">" + bullets + "</ul>",
    ]


# Per-(transform, base-parameter) self-contained help. Each entry drives a
# self-contained body: a "<symbol> (name)" heading, an effect-first intro with
# the range, a bulleted mode list, and the Separate-± note. Modes here are the
# parameter's real menu, in menu order.
_TRANSFORM_PARAM_HELP: dict = {
    ("additive_periodic", "mu"): {
        "heading": "&mu; (Amplitude)",
        "intro": "Controls the depth and height of the ripples: how "
                 "pronounced the coordinate perturbations are. Larger values "
                 "strengthen those perturbations. Range: 0.1&ndash;0.7.",
        "modes": [
            ("Uniform Random",
             "assigns a different random amplitude to each dimension. Use this "
             "for varied ripple strength across axes."),
            ("Dimension-shared",
             "one random value applied to all dimensions equally. Use this if "
             "you want ripples of the same strength in every direction."),
            ("Beta Distribution",
             "concentrates values toward the low or high end of the range. Use "
             "this to bias most dimensions toward weak or strong ripples."),
            ("Linear Spread (Shuffled)",
             "spreads values evenly across the full range, randomly assigned "
             "to dimensions. Use this to guarantee the full range of ripple "
             "strengths is represented."),
        ],
        "pm_note": "by default, ripples are sign-symmetric: the pattern "
                   "looks the same on both sides of the optimum. When enabled, "
                   "the same range and sampling mode apply to both + and "
                   "&minus;, but independent values are drawn, so the "
                   "ripple depth can differ between the positive and negative "
                   "sides of each axis.",
    },
    ("additive_periodic", "gamma"): {
        "heading": "&gamma; (Envelope Rate)",
        "intro": "Controls how quickly ripples activate away from the center. "
                 "Small values delay onset: ripples only appear further "
                 "out. Large values bring them closer in. Range: "
                 "0.002&ndash;0.2.",
        "modes": [
            ("Uniform Random",
             "assigns a different random onset rate to each dimension. Use "
             "this for varied ripple activation across axes."),
            ("Dimension-shared",
             "one random value applied to all dimensions equally. Use this if "
             "you want ripples to activate at the same distance in every "
             "direction."),
            ("Beta Distribution",
             "concentrates values toward slow or fast onset. Use this to bias "
             "most dimensions toward early or late ripple activation."),
            ("Linear Spread (Shuffled)",
             "spreads values evenly across the full range, randomly assigned "
             "to dimensions. Use this to guarantee the full range of onset "
             "rates is represented."),
        ],
        "pm_note": "by default, the ripple onset is sign-symmetric. When "
                   "enabled, the same range and sampling mode apply to both + "
                   "and &minus;, but independent values are drawn, so "
                   "ripples can activate sooner on one side than the other.",
    },
    ("additive_periodic", "omega"): {
        "heading": "&omega; (Frequency)",
        "intro": "Controls the spacing between ripples. Higher values pack "
                 "more ripples closer together; lower values spread them "
                 "apart. Range: 0.05&ndash;1.0.",
        "modes": [
            ("Uniform Random",
             "assigns a different random frequency to each dimension. Use this "
             "for varied ripple density across axes."),
            ("Dimension-shared",
             "one random value applied to all dimensions equally. Use this if "
             "you want the same ripple spacing in every direction."),
            ("Beta Distribution",
             "concentrates values toward sparse or dense ripples. Use this to "
             "bias most dimensions toward wide or tight spacing."),
            ("Linear Spread (Shuffled)",
             "spreads values evenly across the full range, randomly assigned "
             "to dimensions. Use this to guarantee the full range of ripple "
             "densities is represented."),
        ],
        "pm_note": "by default, ripple spacing is sign-symmetric. When "
                   "enabled, the same range and sampling mode apply to both + "
                   "and &minus;, but independent values are drawn, so "
                   "one side can have denser ripples than the other.",
    },
    ("log_sinusoidal", "mu"): {
        "heading": "&mu; (Amplitude)",
        "intro": "Controls the strength of the modulation: how much "
                 "the pattern deviates from the smooth basin. Larger values "
                 "create deeper, more prominent irregularities. Range: "
                 "0.05&ndash;0.5.",
        "modes": [
            ("Uniform Random",
             "assigns a different random strength to each dimension. Use this "
             "for varied modulation across axes."),
            ("Dimension-shared",
             "one random value applied to all dimensions equally. Use this if "
             "you want the same modulation strength in every direction."),
            ("Beta Distribution",
             "concentrates values toward weak or strong modulation. Use this "
             "to bias most dimensions toward subtle or prominent "
             "irregularities."),
            ("Linear Spread (Shuffled)",
             "spreads values evenly across the full range, randomly assigned "
             "to dimensions. Use this to guarantee the full range of "
             "modulation strengths is represented."),
        ],
        "pm_note": "by default, the modulation is sign-symmetric. When "
                   "enabled, the same range and sampling mode apply to both + "
                   "and &minus;, but independent values are drawn, so "
                   "the pattern can be stronger on one side than the other.",
    },
    ("log_sinusoidal", "omega1"): {
        "heading": "&omega;<sub>1</sub>, &omega;<sub>2</sub> "
                   "(Log-space Frequencies)",
        "intro": "Control how many oscillations occur per decade of distance "
                 "from the center. Using two different frequencies creates a "
                 "richer, multi-scale interference pattern. Range: 5&ndash;50.",
        "modes": [
            ("Uniform Random",
             "assigns different random frequencies to each dimension. Use this "
             "for varied fractal patterns across axes."),
            ("Dimension-shared",
             "one random value applied to all dimensions equally. Use this if "
             "you want the same fractal scaling in every direction."),
            ("Beta Distribution",
             "concentrates values toward low or high frequencies. Use this to "
             "bias most dimensions toward coarse or fine fractal detail."),
            ("Linear Spread (Shuffled)",
             "spreads values evenly across the full range, randomly assigned "
             "to dimensions. Use this to guarantee the full range of "
             "frequencies is represented."),
        ],
        "pm_note": "by default, frequencies are sign-symmetric. When enabled, "
                   "the same range and sampling mode apply to both + and "
                   "&minus;, but independent values are drawn, so the "
                   "fractal pattern can scale differently on each side.",
    },
    ("wavelet_mod", "mu"): {
        "heading": "&mu; (Amplitude)",
        "intro": "Controls the depth and height of the local optima within "
                 "the active band. Larger values create stronger "
                 "perturbations. Range: 10&ndash;50.",
        "modes": [
            ("Uniform Random",
             "assigns a different random amplitude to each dimension. Use this "
             "for varied band strength across axes."),
            ("Dimension-shared",
             "one random value applied to all dimensions equally. Use this if "
             "you want the same band strength in every direction."),
            ("Beta Distribution",
             "concentrates values toward weak or strong perturbations. Use "
             "this to bias most dimensions toward subtle or prominent local "
             "optima."),
            ("Linear Spread (Shuffled)",
             "spreads values evenly across the full range, randomly assigned "
             "to dimensions. Use this to guarantee the full range of band "
             "strengths is represented."),
        ],
        "pm_note": "by default, the band is sign-symmetric. When enabled, the "
                   "same range and sampling mode apply to both + and &minus;, "
                   "but independent values are drawn, so local optima "
                   "can be stronger on one side than the other.",
    },
    ("wavelet_mod", "omega"): {
        "heading": "&omega; (Frequency)",
        "intro": "Controls the spacing of local optima within the active "
                 "band. Higher values pack more features into the band; lower "
                 "values space them apart. Range: 0.3&ndash;1.0.",
        "modes": [
            ("Uniform Random",
             "assigns a different random frequency to each dimension. Use this "
             "for varied feature density across axes."),
            ("Dimension-shared",
             "one random value applied to all dimensions equally. Use this if "
             "you want the same feature spacing in every direction."),
            ("Beta Distribution",
             "concentrates values toward sparse or dense features. Use this to "
             "bias most dimensions toward wide or tight spacing within the "
             "band."),
            ("Linear Spread (Shuffled)",
             "spreads values evenly across the full range, randomly assigned "
             "to dimensions. Use this to guarantee the full range of feature "
             "densities is represented."),
        ],
        "pm_note": "by default, spacing within the band is sign-symmetric. "
                   "When enabled, the same range and sampling mode apply to "
                   "both + and &minus;, but independent values are drawn, "
                   "so one side can have denser features than the "
                   "other.",
    },
    ("wavelet_mod", "eta"): {
        "heading": "&eta; (Extent Scale)",
        "intro": "Controls how far from the center the active band is located. "
                 "Larger values push the band further out; smaller values "
                 "bring it closer in. Range: 10&ndash;24.",
    },
    ("tensor_interference", "mu"): {
        "heading": "&mu;<sub>0</sub> (Base amplitude)",
        "intro": "Controls the strength of the checkerboard interference "
                 "pattern. Larger values create deeper, more prominent "
                 "grid-like traps. Range: 10&ndash;20.",
        "modes": [
            ("Uniform Random",
             "assigns a different random strength to each dimension. Use this "
             "for varied interference intensity across axes."),
            ("Dimension-shared",
             "one random value applied to all dimensions equally. Use this if "
             "you want the same interference strength in every direction."),
            ("Beta Distribution",
             "concentrates values toward weak or strong interference. Use this "
             "to bias most dimensions toward subtle or prominent grid traps."),
            ("Linear Spread (Shuffled)",
             "spreads values evenly across the full range, randomly assigned "
             "to dimensions. Use this to guarantee the full range of "
             "interference strengths is represented."),
        ],
        "pm_note": "by default, the interference pattern is sign-symmetric. "
                   "When enabled, the same range and sampling mode apply to "
                   "both + and &minus;, but independent values are drawn, "
                   "so the pattern can be stronger on one side of each "
                   "axis.",
    },
    ("tensor_interference", "omega"): {
        "heading": "&omega; (Frequency)",
        "intro": "Controls the spacing of the checkerboard grid. Higher values "
                 "create a finer, denser grid; lower values produce a coarser "
                 "pattern. Range: 0.1&ndash;0.7.",
        "modes": [
            ("Uniform Random",
             "assigns a different random frequency to each dimension. Use this "
             "for varied grid spacing across axes."),
            ("Dimension-shared",
             "one random value applied to all dimensions equally. Use this if "
             "you want the same grid spacing in every direction."),
            ("Beta Distribution",
             "concentrates values toward coarse or fine grids. Use this to "
             "bias most dimensions toward wide or tight checkerboard cells."),
            ("Linear Spread (Shuffled)",
             "spreads values evenly across the full range, randomly assigned "
             "to dimensions. Use this to guarantee the full range of grid "
             "spacings is represented."),
        ],
        "pm_note": "by default, the grid spacing is sign-symmetric. When "
                   "enabled, the same range and sampling mode apply to both + "
                   "and &minus;, but independent values are drawn, so "
                   "the grid can be finer on one side than the other.",
    },
}
# ω1 and ω2 share one combined description (the two Log-space Frequencies).
_TRANSFORM_PARAM_HELP[("log_sinusoidal", "omega2")] = \
    _TRANSFORM_PARAM_HELP[("log_sinusoidal", "omega1")]

# Radial Polynomial-Trigonometric parameters are all scalars: no randomisation
# modes and no ± directions. Each carries the same closing note explaining why.
_RADIAL_NOTE = (
    "Radial parameters are scalars with no per-dimension generation modes or "
    "separate &plusmn; values. Generator samples each scalar once per component "
    "from its bounds; equal bounds fix it. The operator is radial, while the "
    "baseline and other transformations can retain directional differences.")
for _rp_name, _rp_heading, _rp_intro in [
    ("mu", "&mu; (Amplitude)",
     "Controls the overall strength of the concentric rings. Larger values "
     "create deeper, more prominent rings. Range: 0.4&ndash;2.0."),
    ("p", "p (Radial Ramp)",
     "Controls the radial growth profile, with 0 &lt; p &lt; 1. This is distinct "
     "from the component curvature exponent. Range: 0.4&ndash;0.7."),
    ("q", "q (Ring Spacing)",
     "Controls how ring spacing changes with distance. q = 1 gives evenly "
     "spaced rings; q &lt; 1 spreads rings further apart outward; q &gt; 1 "
     "compresses them. Range: 0.4&ndash;1.2."),
    ("omega", "&omega; (Frequency)",
     "Controls how many rings appear across the basin. Higher values produce "
     "more rings; lower values produce fewer, wider ones. Range: 0.1&ndash;10."),
]:
    _TRANSFORM_PARAM_HELP[("radial_polytrig", _rp_name)] = {
        "heading": _rp_heading,
        "intro": _rp_intro,
        "note": _RADIAL_NOTE,
    }


def transform_param_help_paragraphs(transform_name: str,
                                    base_name: str,
                                    educational: bool = False) -> Optional[List[str]]:
    """Self-contained help body for one transform inner parameter, or ``None``.

    ``None`` means no specialised text is defined for
    ``(transform_name, base_name)``; callers use the generic definition.
    ``educational`` drops the randomisation modes and uses the manual-entry ±
    note (the Education tab types values directly in 2D).
    """
    data = _TRANSFORM_PARAM_HELP.get((transform_name, base_name))
    if data is None:
        return None
    # Standardise the range wording: every "Range: a–b" reads "The default
    # range is a–b".
    intro = data["intro"].replace("Range:", "The default range is")
    paras = [f"<b>{data['heading']}</b>", intro]
    modes = data.get("modes")
    if modes and not educational:
        bullets = "".join(f"<li><b>{n}</b>: {d}</li>" for n, d in modes)
        count = {4: "four", 5: "five", 6: "six"}.get(len(modes), "several")
        paras.append(_modes_title(count))
        paras.append(
            "<ul style=\"margin: 0 0 10px 18px; padding: 0;\">" + bullets
            + "</ul>")
    if data.get("pm_note"):
        paras.append(_SEPARATE_DIR_HEADING)
        if educational:
            paras.append(_EDU_SEP_DIR_NOTE)
        else:
            # Capitalise the first letter so the ± paragraph starts as a sentence.
            pm = data["pm_note"]
            paras.append(pm[:1].upper() + pm[1:])
    if data.get("note"):
        paras.append(data["note"])
    return paras
