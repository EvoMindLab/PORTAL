"""Display and table-formatting helpers for the Generator tab."""
from __future__ import annotations

from typing import Any

import numpy as np

from portal_studio.gui.common.help import plain_math_header as _plain_math_header
from portal_studio.gui.common.angles import display_degrees
from portal_studio.gui.generator.rotation_dialogs import (
    CONTROLLED_PARTIAL_SEPERATE_LABEL,
)


_ROTATION_MODES = [
    "No Rotation", "Fully Connected", "Probabilistic Coupling",
    "Uniform Random Angle", "Fixed Angle", "Chain-like Coupling",
    "Block-Diagonal",
]


def format_tree_number(value: float) -> str:
    """Show compact tree numbers: 10.0 -> 10, 1.5 -> 1.5."""
    return f"{float(value):g}"


def format_component_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def plain_math_header(text: str) -> str:
    return _plain_math_header(text)


def rotation_triangle_table_data(generated: dict, d: int):
    headers = [""] + [str(i + 1) for i in range(d)]
    psi = (generated or {}).get("Psi_deg")
    rows = []
    if psi is None:
        psi_arr = None
    else:
        psi_arr = np.asarray(psi)
        if psi_arr.ndim > 2:
            psi_arr = psi_arr[:, :, 0]
    for row_idx in range(d):
        row = [str(row_idx + 1)]
        for col_idx in range(d):
            if col_idx <= row_idx or psi_arr is None:
                row.append("")
            elif row_idx < psi_arr.shape[0] and col_idx < psi_arr.shape[1]:
                row.append(display_degrees(float(psi_arr[row_idx, col_idx])))
            else:
                row.append("")
        rows.append(row)
    return headers, rows


def rotation_pair_table_data(generated: dict, d: int):
    psi = (generated or {}).get("Psi_deg")
    if psi is None:
        return [""], []
    psi_arr = np.asarray(psi)
    if psi_arr.ndim > 2:
        psi_arr = psi_arr[:, :, 0]
    rows = []
    max_cols = 1
    for row_idx in range(d):
        pair_row = ["pair"]
        angle_row = ["angle (deg)"]
        for col_idx in range(row_idx + 1, d):
            if row_idx >= psi_arr.shape[0] or col_idx >= psi_arr.shape[1]:
                continue
            angle = float(psi_arr[row_idx, col_idx])
            if np.isclose(angle, 0.0):
                continue
            pair_row.append(f"{row_idx + 1}, {col_idx + 1}")
            angle_row.append(display_degrees(angle))
        if len(pair_row) > 1:
            rows.extend([pair_row, angle_row])
            max_cols = max(max_cols, len(pair_row))
    return [""] * max_cols, rows


def block_diagonal_rotation_heading(grouping: str, angle_mode: str) -> str:
    angle_word = "random angle" if angle_mode == "randomize" else "fixed angle"
    return f"Rotation: {CONTROLLED_PARTIAL_SEPERATE_LABEL} ({grouping}, {angle_word}, deg)"


def rotation_mode_heading(mode: int) -> str:
    if int(mode) == 7:
        return f"Rotation: {CONTROLLED_PARTIAL_SEPERATE_LABEL} (deg)"
    if 1 <= int(mode) <= len(_ROTATION_MODES):
        return f"Rotation: {_ROTATION_MODES[int(mode) - 1]} (deg)"
    return f"Rotation: mode {int(mode)}"
