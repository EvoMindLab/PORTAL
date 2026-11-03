"""GUI controller and compatibility adapter for PORTAL Studio.

The backend-owned instance construction lives in ``portal_core.instance_factory``.
This module adapts Studio controller state to that factory, keeping schema,
neutralization, and transform-packing knowledge out of the GUI layer.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from portal_core.fitness import fitness_batch
from portal_studio.validation import validate_component_config
from portal_core.instance_factory import (
    coerce_to_d as _coerce_to_d,
    component_config_to_instance,
)
from portal_core.transform_registry import (
    ALLOWED_TRANSFORM_NAMES,
    normalize_transform_name,
)


@dataclass
class ControllerState:
    baseline: str = "A"
    rotation_deg: float = 0.0
    neutralize: Dict[str, Any] = field(
        default_factory=lambda: {"enable": True, "Delta": 100.0, "r_ref": 100.0}
    )
    params: Dict[str, Any] = field(
        default_factory=lambda: {
            "c": None,
            "beta": 0.0,
            "kappa_plus": None,
            "kappa_minus": None,
            "p_single": None,
            "p_plus": None,
            "p_minus": None,
            "R": None,
        }
    )
    T_seq: List[Dict[str, Any]] = field(default_factory=list)
    components: List[Dict[str, Any]] = field(default_factory=list)


def gui_cfg_to_instance(cfg: Dict[str, Any], d: int) -> Dict[str, Any]:
    """Compatibility wrapper for the backend-owned component factory."""
    validate_component_config(cfg)
    return component_config_to_instance(copy.deepcopy(cfg), d)


class PortalController:
    """Holds GUI state; computes through ``portal_core``."""

    def __init__(self) -> None:
        self.state = ControllerState()
        self._d: Optional[int] = None

    def set_baseline(self, baseline: str) -> None:
        b = str(baseline).upper()
        if b not in {"A", "B"}:
            raise ValueError("baseline must be 'A' or 'B'")
        self.state.baseline = b

    def set_rotation_deg(self, deg: float) -> None:
        self.state.rotation_deg = float(deg)

    def set_neutralize(self, enable: bool, Delta: float | None = None,
                       r_ref: float | None = None) -> None:
        self.state.neutralize["enable"] = bool(enable)
        if Delta is not None:
            self.state.neutralize["Delta"] = float(Delta)
        if r_ref is not None:
            self.state.neutralize["r_ref"] = float(r_ref)

    def set_params_from_dict(self, params: Dict[str, Any], d: int) -> None:
        params = copy.deepcopy(params)
        self._d = int(d)
        p = self.state.params
        p["c"] = _coerce_to_d(params.get("c"), d, 0.0)
        p["beta"] = float(params.get("beta", 0.0))
        p["kappa_plus"] = _coerce_to_d(params.get("kappa_plus"), d, 1.0)
        p["kappa_minus"] = _coerce_to_d(params.get("kappa_minus"), d, 1.0)
        if self.state.baseline == "A":
            p["p_plus"] = _coerce_to_d(params.get("p_plus"), d, 1.0)
            p["p_minus"] = _coerce_to_d(params.get("p_minus"), d, 1.0)
            p["p_single"] = None
        else:
            p["p_plus"] = None
            p["p_minus"] = None
            value = params.get("p_single", 1.0)
            p["p_single"] = float(1.0 if value is None else value)

        R = params.get("R")
        if R is not None:
            R_arr = np.asarray(R, dtype=np.float64)
            if R_arr.shape != (d, d):
                raise ValueError(f"R must be shape ({d},{d}), got {R_arr.shape}")
            p["R"] = R_arr
        else:
            p["R"] = None

    def add_transform(self, name: str, params: Dict[str, Any],
                      d: Optional[int] = None) -> None:
        self.state.T_seq.append({
            "name": normalize_transform_name(name),
            "params": copy.deepcopy(params),
        })
        if d is not None:
            self._d = int(d)

    def edit_transform(self, index: int, name: Optional[str] = None,
                       params: Optional[Dict[str, Any]] = None) -> None:
        self._check_index(index)
        transform = self.state.T_seq[index]
        if name is not None:
            transform["name"] = normalize_transform_name(name)
        if params is not None:
            transform["params"] = copy.deepcopy(params)

    def delete_transform(self, index: int) -> None:
        self._check_index(index)
        del self.state.T_seq[index]

    def move_transform_up(self, index: int) -> None:
        self._check_index(index)
        if index <= 0:
            return
        seq = self.state.T_seq
        seq[index - 1], seq[index] = seq[index], seq[index - 1]

    def move_transform_down(self, index: int) -> None:
        self._check_index(index)
        seq = self.state.T_seq
        if index >= len(seq) - 1:
            return
        seq[index + 1], seq[index] = seq[index], seq[index + 1]

    def build_cfg(self) -> Dict[str, Any]:
        return copy.deepcopy({
            "baseline": self.state.baseline,
            "rotation_deg": self.state.rotation_deg,
            "neutralize": dict(self.state.neutralize),
            "params": dict(self.state.params),
            "T_seq": list(self.state.T_seq),
        })

    def compute_points(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        _n, d = X.shape
        self._d = d
        inst = gui_cfg_to_instance(self.build_cfg(), d)
        return np.asarray(
            fitness_batch(X, inst, evaluation_device="numpy"),
            dtype=np.float64,
        )

    def add_component_from_current(self) -> int:
        self.state.components.append(self.build_cfg())
        return len(self.state.components) - 1

    def clear_components(self) -> None:
        self.state.components.clear()

    def get_components(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.state.components)

    def compute_aggregated_points(self, X: np.ndarray,
                                  indices: Optional[List[int]] = None) -> np.ndarray:
        """Landscape = lower envelope over selected component instances."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n, d = X.shape
        self._d = d
        if not self.state.components:
            return self.compute_points(X)

        selected = indices if indices else list(range(len(self.state.components)))
        y_min: Optional[np.ndarray] = None
        for idx in selected:
            if idx < 0 or idx >= len(self.state.components):
                raise IndexError("component index out of range")
            inst = gui_cfg_to_instance(self.state.components[idx], d)
            y = np.asarray(
                fitness_batch(X, inst, evaluation_device="numpy"),
                dtype=np.float64,
            )
            y_min = y if y_min is None else np.minimum(y_min, y)
        return y_min if y_min is not None else np.zeros(n, dtype=np.float64)

    def _check_index(self, index: int) -> None:
        if index < 0 or index >= len(self.state.T_seq):
            raise IndexError("transform index out of range")
