"""Reusable parameter-row factory for the Generator tab.

Each row is: ``label  [mode dropdown]  [vmin] [vmax]  [extras...]``. The mode
menu and the visible extras follow the paper's per-parameter mode set, so the
same factory powers both vector parameters (``c``, ``kappa_+``, ``p_+``) and
transform inner parameters (``mu_+``, ``omega_+``, …) without per-call
specialisation.

The factory owns a tiny store mapping parameter name → its mode-extra
``QLineEdit`` widgets (alpha, beta, exclusion_lo, exclusion_hi, fixed). The
caller only holds the resulting ``QWidget`` for layout.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QLineEdit, QWidget

from portal_studio.experiment.modes import (
    CENTER_MODES, KAPPA_MODES, P_MODES, canonical_ranges,
)
from portal_studio.gui.common.widgets import NoWheelComboBox


# ─── parameter → mode menu ────────────────────────────────────────────────
MODES_FOR: Dict[str, List[str]] = {
    "c": CENTER_MODES,
    "kappa_plus": KAPPA_MODES, "kappa_minus": KAPPA_MODES,
    "p_plus": P_MODES, "p_minus": P_MODES,
}

# ─── parameter → {mode integer: tuple of extra-widget kinds} ──────────────
EXTRAS_FOR: Dict[str, Dict[int, Tuple[str, ...]]] = {
    "c": {2: ("exclusion_lo", "exclusion_hi")},
    "kappa_plus":  {2: ("alpha", "beta")},
    "kappa_minus": {2: ("alpha", "beta")},
    "p_plus":  {2: ("fixed",), 4: ("alpha", "beta")},
    "p_minus": {2: ("fixed",), 4: ("alpha", "beta")},
}
EXTRA_DEFAULTS: Dict[str, str] = {
    "exclusion_lo": "-20", "exclusion_hi": "20",
    "alpha": "0.5", "beta": "0.5",
    "fixed": "1.0",
}
# Inline labels shown right before each extra input, in paper-style math
# notation (rendered as RichText when they carry markup).
_EXTRA_DISPLAY_LABEL: Dict[str, str] = {
    "exclusion_lo": "excl<sub>lo</sub>",
    "exclusion_hi": "excl<sub>hi</sub>",
    "alpha": "α",
    "beta": "β",
    "fixed": "value",
}
# Brief tooltips describe parameter effects; section help provides
# the mathematical context from PORTAL Part I.
PARAM_TOOLTIPS: Dict[str, str] = {
    "c": "Where the basin sits in the search space, sampled per coordinate "
         "from the min/max range.",
    "kappa_plus": "Anisotropy — how stretched the basin is per direction. "
                  "The anisotropy ratio sets directional contrast; exponents "
                  "and transformations also affect geometry.",
    "p_plus": "Curvature per direction. p=1 is the standard quadratic bowl, "
              "p=0.5 gives linear growth; smaller values are sublinear and "
              "larger values are superlinear.",
    # --- transform vector inner-parameters ---
    "mu_plus": "Ripple strength — how deep/tall the induced local optima are; "
               "0 means no perturbation.",
    "gamma_plus": "How quickly the ripples switch on away from the centre; "
                  "larger values make them start closer in.",
    "omega_plus": "Ripple spacing along the axis — larger ω packs the extrema "
                  "closer together (spacing ≈ π/ω).",
    "omega1_plus": "First log-space frequency; combined with ω₂ it produces "
                   "multi-scale 'beating' ruggedness.",
    "omega2_plus": "Second log-space frequency; combined with ω₁ it produces "
                   "multi-scale 'beating' ruggedness.",
    # --- transform scalar parameters ---
    "sigma": "Smoothing scale for Log-Sinusoidal modulation near the centre.",
    "eps": "Small numerical-stability constant; leave at the default.",
    "eta": "Overall reach of the wavelet lobes from the centre (extent scale).",
    "mu": "Strength of the radial ring perturbation; 0 means none.",
    "p": "Polynomial order shaping the radial ring profile.",
    "q": "Trigonometric order shaping the radial ring profile.",
    "omega": "Spacing of the concentric rings — larger ω packs them closer.",
}
# Per-extra tooltips, keyed by extra kind (not parameter name).
_EXTRA_TOOLTIPS: Dict[str, str] = {
    "exclusion_lo": "Lower edge of the interval the centre is kept out of.",
    "exclusion_hi": "Upper edge of the interval the centre is kept out of.",
    "alpha": "First Beta-distribution shape control for the sampled values.",
    "beta": "Second Beta-distribution shape control for the sampled values.",
    "fixed": "Fixed value reused for every direction.",
}


def param_tooltip(name: str) -> str:
    """Effect tooltip for a parameter row; the − direction reuses its + text."""
    if name in PARAM_TOOLTIPS:
        return PARAM_TOOLTIPS[name]
    if name.endswith("_minus"):
        return PARAM_TOOLTIPS.get(name[:-6] + "_plus", "")
    return ""


# Uniform widths so input columns align across all parameter rows.
_DROPDOWN_W = 180     # mode combo; fits the longest mode label
_NUM_W = 56           # numeric inputs (min, max, alpha, beta, value, …)
# Grid columns: 0=name 1=dropdown 2=min-label 3=min-input 4=max-label
# 5=max-input. Mode-specific extras wrap to row 1 under min/max.
# they sit directly beneath those inputs. (label_col, input_col) per extra
# slot — first extra under min, second under max.
_EXTRA_COL_PAIRS = [(2, 3), (4, 5)]
# Modes that don't sample from [vmin, vmax]: hide the range cells.
MODES_NO_RANGE: Dict[str, set] = {
    "c": set(),                               # both centre modes use the range
    "kappa_plus": {5},                        # mode 5 (all-ones) ignores range
    "kappa_minus": {5},
    "p_plus": {2},                            # mode 2 (Fixed value) ignores range
    "p_minus": {2},
}
# (min_key, max_key) lookups in ``canonical_ranges()`` per parameter.
_RANGE_KEYS: Dict[str, Tuple[str, str]] = {
    "c": ("c_min", "c_max"),
    "kappa_plus": ("kappa_min", "kappa_max"),
    "kappa_minus": ("kappa_min", "kappa_max"),
    "p_plus": ("p_min", "p_max"),
    "p_minus": ("p_min", "p_max"),
}


def default_range(name: str) -> Tuple[float, float]:
    """Engine-canonical ``(min, max)`` for ``name``.

    The centre ``c`` uses the engine's ``c_min``/``c_max`` — the domain bounds
    inset by 10 (e.g. -90..90 for a -100..100 domain) — matching the range the
    benchmark generator actually samples centres from, so the GUI default does
    not exceed the engine's range. (Users can still widen or narrow it.)
    """
    R = canonical_ranges()
    lo_key, hi_key = _RANGE_KEYS[name]
    return float(R[lo_key]), float(R[hi_key])


def mode_int(combo) -> int:
    """Return the engine's mode integer for the combo's current item.

    Most ``*_MODES`` lists are in engine mode-1..N order. Some GUI-filtered
    menus hide duplicate one-component modes but store the original engine
    mode integer in the combo item's UserRole.
    """
    data = combo.currentData()
    if data is not None:
        return int(data)
    return int(combo.currentIndex()) + 1


class ParamRowFactory:
    """Builds + manages mode-aware parameter rows. One factory per tab is enough.

    The factory keeps:
    - ``rows[name]``: ``{"mode", "vmin", "vmax", "widget"}`` for each parameter row
      it built.
    - ``extras[name]``: ``{kind: QLineEdit}`` for that row's mode-specific extras.
    Callers reach into ``factory.rows[name]["mode"]`` to read or set the mode.
    """

    def __init__(self, label_factory: Callable[[str], QLabel],
                 on_visibility_change: Optional[Callable[[], None]] = None) -> None:
        self._label = label_factory
        self.rows: Dict[str, dict] = {}
        self.extras: Dict[str, Dict[str, QLineEdit]] = {}
        # Optional hook the tab uses to recompute the left-panel minimum width
        # once a row's visible content changes (so a Beta-distribution mode
        # automatically widens the column, and switching back narrows it).
        self._on_visibility_change = on_visibility_change

    def build(self, name: str, lo: float, hi: float,
              modes: List[str],
              extras_spec: Optional[Dict[int, Tuple[str, ...]]] = None,
              mode_values: Optional[List[int]] = None,
              display_name: Optional[str] = None) -> QWidget:
        """Build one row and register it in ``rows``/``extras``.

        Layout (a 2-row grid)::

            [name]  [mode dropdown]  min:[ ]    max:[ ]
                                     excl_lo:[ ] excl_hi:[ ]

        Each numeric input has an inline label so the user can read what the
        field is. **The row shows only what the current mode uses** —
        min/max disappear when the mode ignores ranges (e.g. kappa mode 5
        Fixed Unit Scale κ=1), and only the extras paper-defined for the
        current mode appear.

        Mode-specific extras (e.g. the exclusion-zone bounds) live on a second
        grid row that shares the min/max columns, so they sit directly beneath
        those inputs and never extend the first line — enabling them adds a
        row of height, not width.
        """
        w = QWidget()
        grid = QGridLayout(w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)

        name_label = self._label(display_name or name)
        tip = param_tooltip(name)
        if tip:
            name_label.setToolTip(tip)
        grid.addWidget(name_label, 0, 0)

        mode_cb = NoWheelComboBox()
        mode_cb.addItems(modes)
        for i, mode_name in enumerate(modes):
            mode_cb.setItemData(i, mode_name, Qt.ItemDataRole.ToolTipRole)
            if mode_values is not None:
                mode_cb.setItemData(i, int(mode_values[i]), Qt.ItemDataRole.UserRole)
        mode_cb.setToolTip(mode_cb.currentText())
        mode_cb.currentTextChanged.connect(mode_cb.setToolTip)
        # Uniform width so the inputs that follow line up vertically across
        # every parameter row regardless of the longest mode label.
        mode_cb.setFixedWidth(_DROPDOWN_W)
        grid.addWidget(mode_cb, 0, 1)

        # min / max with their own inline labels. Both the label and the
        # input are stored so they can be hidden together when the current
        # mode doesn't sample from a range.
        vmin_label = self._inline_label("min")
        grid.addWidget(vmin_label, 0, 2)
        vmin = QLineEdit(str(lo)); vmin.setFixedWidth(_NUM_W)
        grid.addWidget(vmin, 0, 3)
        vmax_label = self._inline_label("max")
        grid.addWidget(vmax_label, 0, 4)
        vmax = QLineEdit(str(hi)); vmax.setFixedWidth(_NUM_W)
        grid.addWidget(vmax, 0, 5)
        if tip:
            vmin.setToolTip(tip)
            vmax.setToolTip(tip)
        # Each extra is positioned by its slot WITHIN its mode's tuple, so the
        # first extra sits under min and the second under max. Different modes'
        # extras may share a cell (they are never visible simultaneously).
        extra_col: Dict[str, Tuple[int, int]] = {}
        for kinds in (extras_spec or {}).values():
            for slot, k in enumerate(kinds):
                if k not in extra_col:
                    extra_col[k] = (
                        _EXTRA_COL_PAIRS[slot] if slot < len(_EXTRA_COL_PAIRS)
                        else (2 + 2 * slot, 3 + 2 * slot))

        # Every kind that ANY of this row's modes can enable. Only these
        # widgets are constructed; modes the paper doesn't define for this
        # parameter do not create extra fields.
        extras: Dict[str, QLineEdit] = {}
        extra_labels: Dict[str, QLabel] = {}
        row_kinds: List[str] = []
        for kinds in (extras_spec or {}).values():
            for k in kinds:
                if k not in row_kinds:
                    row_kinds.append(k)
        # An extra sits inline on row 0 (next to the dropdown) when every mode
        # that enables it also hides the min/max range — there's nothing on the
        # first line for it to collide with, so it shouldn't wrap to a second
        # line (e.g. a Fixed Value's "value" box). Extras that coexist with a
        # visible range (alpha/beta, exclusion bounds) stay on row 1, beneath it.
        no_range = MODES_NO_RANGE.get(name, set())
        for k in row_kinds:
            lab_col, in_col = extra_col[k]
            modes_enabling = {m for m, kinds in (extras_spec or {}).items()
                              if k in kinds}
            extra_row = 0 if modes_enabling and modes_enabling <= no_range else 1
            extra_tip = _EXTRA_TOOLTIPS.get(k, k)
            lab = self._inline_label(_EXTRA_DISPLAY_LABEL.get(k, k))
            lab.setToolTip(extra_tip)
            grid.addWidget(lab, extra_row, lab_col)
            le = QLineEdit(EXTRA_DEFAULTS.get(k, ""))
            le.setFixedWidth(_NUM_W)
            le.setPlaceholderText(k)
            le.setToolTip(extra_tip)
            grid.addWidget(le, extra_row, in_col)
            extras[k] = le
            extra_labels[k] = lab

        entry = {"mode": mode_cb, "vmin": vmin, "vmax": vmax, "widget": w,
                 "_extras_spec": extras_spec or {},
                 "_extras": extras,
                 "_no_range": MODES_NO_RANGE.get(name, set()),
                 "_extra_labels": extra_labels,
                 "_name_label": name_label,
                 "_vmin_label": vmin_label, "_vmax_label": vmax_label}
        self.rows[name] = entry
        self.extras[name] = extras
        mode_cb.currentTextChanged.connect(
            lambda _t, n=name: self.sync_visibility(n)
        )
        self.sync_visibility(name)
        return w

    @staticmethod
    def _inline_label(text: str) -> QLabel:
        """Small grey leading label printed right before a numeric input.

        Labels carrying markup (e.g. ``excl<sub>lo</sub>``) render as
        RichText so paper-style math notation displays correctly.
        """
        lab = QLabel(text)
        if "<" in text:
            lab.setTextFormat(Qt.TextFormat.RichText)
        lab.setStyleSheet("color: #6b7280; font-size: 11px;")
        return lab

    def sync_visibility(self, name: str) -> None:
        """Show/hide range cells and extras to reflect the row's current mode.

        Hide inputs that the selected mode does not use. The visible row width
        adapts to the controls required by that mode.
        """
        entry = self.rows.get(name)
        if entry is None:
            return
        mode = mode_int(entry["mode"])
        uses_range = mode not in entry["_no_range"]
        entry["vmin"].setVisible(uses_range)
        entry["vmax"].setVisible(uses_range)
        entry["_vmin_label"].setVisible(uses_range)
        entry["_vmax_label"].setVisible(uses_range)
        wanted = set(entry["_extras_spec"].get(mode, ()))
        for k, w in self.extras[name].items():
            visible = k in wanted
            w.setVisible(visible)
            lab = entry["_extra_labels"].get(k)
            if lab is not None:
                lab.setVisible(visible)
        row = entry["widget"]
        if row.layout() is not None:
            row.layout().invalidate()
            row.layout().activate()
        row.setMinimumHeight(row.sizeHint().height())
        # The extras occupy grid row 1; when none are visible that row claims
        # no height, so no explicit container toggle is needed.
        # Notify the tab so the left-panel minimum width can fit (or shrink to)
        # whatever the visible content now needs.
        if self._on_visibility_change is not None:
            self._on_visibility_change()

    def reset_defaults(self, name: str) -> None:
        """Reset mode to first option, range to engine defaults, extras to defaults."""
        entry = self.rows[name]
        entry["mode"].setCurrentIndex(0)
        lo, hi = default_range(name)
        entry["vmin"].setText(str(lo))
        entry["vmax"].setText(str(hi))
        for k, w in self.extras.get(name, {}).items():
            w.setText(EXTRA_DEFAULTS.get(k, ""))
