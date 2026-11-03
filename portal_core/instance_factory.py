"""Backend-owned helpers for constructing PORTAL runtime instances.

The GUI supplies user-facing component configuration dictionaries; this module
turns them into canonical runtime instances used by ``portal_core.fitness``.
Keeping this logic in ``portal_core`` avoids Studio-side copies of schema,
transform packing, and neutralization behavior.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .benchmark_generator import precompute_scalings
from .transform_registry import (
    normalize_transform_name,
    transform_block_name,
)


def coerce_to_d(value: Any, d: int, default: float) -> np.ndarray:
    """Coerce a scalar or vector-like value to a length-``d`` float array."""
    arr = np.asarray(value if value is not None else default, dtype=np.float64)
    if arr.ndim == 0 or arr.size == 1:
        return np.full(d, float(arr.item()), dtype=np.float64)
    arr = arr.astype(np.float64, copy=False).reshape(-1)
    if arr.size != d:
        raise ValueError(f"Expected length {d}, got {arr.size}")
    return arr


def planar_rotation(d: int, angle_deg: float) -> np.ndarray:
    """Return a 2-D planar rotation embedded in a ``d``-dimensional identity."""
    theta = np.deg2rad(float(angle_deg))
    R = np.eye(d, dtype=np.float64)
    if d >= 2:
        R[:2, :2] = np.array(
            [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]],
            dtype=np.float64,
        )
    return R


def planar_psi(d: int, angle_deg: float) -> np.ndarray:
    """Return the upper-triangular angle matrix for ``planar_rotation``."""
    psi = np.zeros((d, d), dtype=np.float64)
    if d >= 2:
        psi[0, 1] = float(angle_deg)
    return psi


def pack_transform_block(name: str, params: Dict[str, Any], d: int) -> Dict[str, Any]:
    """Pack one transform's user parameters into canonical runtime layout."""
    name = normalize_transform_name(name)
    tp = dict(params)

    def arr(key: str, default: float) -> np.ndarray:
        return coerce_to_d(tp.get(key), d, default).reshape(1, d)

    if name == "additive_periodic":
        return {
            "mu_plus": arr("mu_plus", 0.0),
            "mu_minus": arr("mu_minus", 0.0),
            "gamma_plus": arr("gamma_plus", 0.0),
            "gamma_minus": arr("gamma_minus", 0.0),
            "omega_plus": arr("omega_plus", 0.0),
            "omega_minus": arr("omega_minus", 0.0),
        }
    if name == "log_sinusoidal":
        return {
            "mu_plus": arr("mu_plus", 0.0),
            "mu_minus": arr("mu_minus", 0.0),
            "omega1_plus": arr("omega1_plus", 0.0),
            "omega1_minus": arr("omega1_minus", 0.0),
            "omega2_plus": arr("omega2_plus", 0.0),
            "omega2_minus": arr("omega2_minus", 0.0),
            "sigma": np.array([float(tp.get("sigma", 0.1))], dtype=np.float64),
        }
    if name == "wavelet_mod":
        return {
            "mu_plus": arr("mu_plus", 0.0),
            "mu_minus": arr("mu_minus", 0.0),
            "omega_plus": arr("omega_plus", 0.0),
            "omega_minus": arr("omega_minus", 0.0),
            "eta": np.array([float(tp.get("eta", 0.0))], dtype=np.float64),
            "eps": float(tp.get("eps", 1e-12)),
        }
    if name == "tensor_interference":
        return {
            "mu_plus": arr("mu_plus", 0.0),
            "mu_minus": arr("mu_minus", 0.0),
            "omega_plus": arr("omega_plus", 0.0),
            "omega_minus": arr("omega_minus", 0.0),
        }
    if name == "radial_polytrig":
        return {
            "mu": np.array([float(tp.get("mu", 0.0))], dtype=np.float64),
            "p": np.array([float(tp.get("p", 0.5))], dtype=np.float64),
            "q": np.array([float(tp.get("q", 1.0))], dtype=np.float64),
            "omega": np.array([float(tp.get("omega", 0.0))], dtype=np.float64),
            "eps": float(tp.get("eps", 1e-12)),
        }
    raise ValueError(f"Unknown transform: {name}")


def component_config_to_instance(cfg: Dict[str, Any], d: int) -> Dict[str, Any]:
    """Build a canonical K=1 PORTAL runtime instance from a component config."""
    d = int(d)
    baseline = str(cfg.get("baseline", "A")).upper()
    if baseline not in {"A", "B"}:
        raise ValueError("baseline must be 'A' or 'B'")

    rotation_deg = float(cfg.get("rotation_deg", 0.0))
    neutralize = dict(cfg.get("neutralize", {"enable": False}))
    params_cfg = dict(cfg.get("params", {}))

    c = coerce_to_d(params_cfg.get("c"), d, 0.0)
    beta = float(params_cfg.get("beta", 0.0))
    kappa_plus = coerce_to_d(params_cfg.get("kappa_plus"), d, 1.0)
    kappa_minus = coerce_to_d(params_cfg.get("kappa_minus"), d, 1.0)

    form = 1 if baseline == "A" else 2
    if form == 1:
        p_plus = coerce_to_d(params_cfg.get("p_plus"), d, 1.0)
        p_minus = coerce_to_d(params_cfg.get("p_minus"), d, 1.0)
        p_single = 1.0
    else:
        p_plus = np.ones(d, dtype=np.float64)
        p_minus = np.ones(d, dtype=np.float64)
        p_single = float(params_cfg.get("p_single", 1.0))

    explicit_R = params_cfg.get("R")
    explicit_psi = params_cfg.get("Psi_deg")
    if explicit_R is not None:
        R = np.asarray(explicit_R, dtype=np.float64)
        if R.shape != (d, d):
            raise ValueError(f"R must be shape ({d},{d}), got {R.shape}")
    else:
        R = planar_rotation(d, rotation_deg)
    if explicit_psi is not None:
        Psi_deg = np.asarray(explicit_psi, dtype=np.float64)
        if Psi_deg.shape != (d, d):
            raise ValueError(
                f"Psi_deg must be shape ({d},{d}), got {Psi_deg.shape}"
            )
    else:
        Psi_deg = planar_psi(d, rotation_deg)

    P: Dict[str, Any] = {
        "form": np.array([form], dtype=int),
        "c": c.reshape(1, d),
        "beta": np.array([beta], dtype=np.float64),
        "kappa_plus": kappa_plus.reshape(1, d),
        "kappa_minus": kappa_minus.reshape(1, d),
        "p_single": np.array([p_single], dtype=np.float64),
        "p_plus": p_plus.reshape(1, d),
        "p_minus": p_minus.reshape(1, d),
        "R": R.reshape(d, d, 1),
        "Psi_deg": Psi_deg.reshape(d, d, 1),
        "neutralize_enable": np.array(
            [bool(neutralize.get("enable", False))], dtype=bool
        ),
        "Delta": np.array(
            [float(neutralize.get("Delta", neutralize.get("delta", 0.0)))],
            dtype=np.float64,
        ),
        "r_ref": np.array(
            [float(neutralize.get("r_ref", neutralize.get("rref", 1.0)))],
            dtype=np.float64,
        ),
    }

    transform_seq: List[str] = []
    occurrences: Dict[str, int] = {}
    for transform in cfg.get("T_seq", []):
        if not isinstance(transform, dict) or "name" not in transform:
            raise ValueError("Each transform must be a dict with a 'name' key")
        name = normalize_transform_name(str(transform["name"]))
        occurrences[name] = occurrences.get(name, 0) + 1
        reference = name if occurrences[name] == 1 else f"{name}#{occurrences[name]}"
        transform_seq.append(reference)
        P[transform_block_name(reference)] = pack_transform_block(
            name, dict(transform.get("params", {})), d
        )

    portal = {
        "d": d,
        "K": 1,
        "Seed": int(cfg.get("Seed", 0) or 0),
        "baseline": [baseline],
        "bounds_min": -100,
        "bounds_max": 100,
        "warn_out_of_bounds": False,
        "transform_seq": [transform_seq],
        "P": P,
    }
    _form, rho_single, rho_plus, rho_minus = precompute_scalings(portal)
    if form == 1:
        rho_single = np.ones((1,), dtype=np.float64)
    else:
        rho_plus = np.ones((1, d), dtype=np.float64)
        rho_minus = np.ones((1, d), dtype=np.float64)
    P["rho_single"] = rho_single
    P["rho_plus"] = rho_plus
    P["rho_minus"] = rho_minus
    return portal


def merge_k1_instances(insts: List[dict]) -> dict:
    """Stack K=1 runtime instances into one K=N block instance."""
    if not insts:
        raise ValueError("merge_k1_instances requires at least one instance")
    K = len(insts)
    d = int(insts[0]["d"])
    if any(int(inst["K"]) != 1 for inst in insts):
        raise ValueError("merge_k1_instances expects only K=1 instances")
    if any(int(inst["d"]) != d for inst in insts):
        raise ValueError("all K=1 instances must have the same dimension")

    out_P: dict = {}
    for key in (
        "form", "beta", "p_single", "rho_single", "Delta", "r_ref",
        "neutralize_enable",
    ):
        out_P[key] = np.concatenate([
            np.asarray(inst["P"][key]).reshape(-1) for inst in insts
        ])
    for key in (
        "c", "kappa_plus", "kappa_minus", "p_plus", "p_minus",
        "rho_plus", "rho_minus",
    ):
        out_P[key] = np.vstack([
            np.asarray(inst["P"][key]).reshape(1, d) for inst in insts
        ])
    out_P["R"] = np.concatenate([
        np.asarray(inst["P"]["R"]).reshape(d, d, 1) for inst in insts
    ], axis=-1)
    out_P["Psi_deg"] = np.concatenate([
        np.asarray(inst["P"]["Psi_deg"]).reshape(d, d, 1) for inst in insts
    ], axis=-1)

    transform_seq: List[List[str]] = []
    for inst in insts:
        seq = inst.get("transform_seq", [[]])[0]
        transform_seq.append(list(seq))

    transform_keys: List[str] = []
    for inst in insts:
        for key in inst["P"]:
            if key.startswith("T_") and key not in transform_keys:
                transform_keys.append(key)

    for tkey in transform_keys:
        template = next(inst["P"][tkey] for inst in insts if tkey in inst["P"])
        merged_sub: dict = {}
        for sub_key, example in template.items():
            example_arr = np.asarray(example)
            parts = []
            for inst in insts:
                sub = inst["P"].get(tkey)
                if sub is not None and sub_key in sub:
                    parts.append(np.asarray(sub[sub_key]))
                else:
                    parts.append(np.zeros_like(example_arr))
            if example_arr.ndim == 2:
                merged_sub[sub_key] = np.vstack(parts)
            elif example_arr.ndim == 1:
                merged_sub[sub_key] = np.concatenate([
                    part.reshape(-1) for part in parts
                ])
            else:
                active_values = [
                    float(np.asarray(inst["P"][tkey][sub_key]).reshape(-1)[0])
                    for inst in insts
                    if tkey in inst["P"] and sub_key in inst["P"][tkey]
                ]
                if active_values and not np.allclose(
                    active_values, active_values[0], rtol=0.0, atol=0.0
                ):
                    raise ValueError(
                        f"Shared transform scalar {tkey}.{sub_key} differs "
                        "between components."
                    )
                merged_sub[sub_key] = active_values[0]
        out_P[tkey] = merged_sub

    return {
        "d": d,
        "K": K,
        "Seed": 0,
        "baseline": [str(inst.get("baseline", ["A"])[0]).upper()
                     for inst in insts],
        "bounds_min": -100,
        "bounds_max": 100,
        "warn_out_of_bounds": False,
        "transform_seq": transform_seq,
        "P": out_P,
    }
