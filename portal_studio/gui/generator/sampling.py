"""Qt-free sampling and realization helpers for the Generator tab."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from portal_studio.experiment.modes import (
    sample_centers,
    sample_kappa,
    sample_p_exponents,
    sample_transform_inner_param,
)
from portal_studio.experiment.seeding import derive_param_rng


def rng_for(
    seed: int,
    counters: dict[str, int],
    copies: dict[str, dict[str, Any]] | None = None,
    seed_overrides: dict[str, int] | None = None,
) -> Callable[[str], np.random.RandomState]:
    """Return a deterministic, independent random stream for each parameter."""
    copies = copies or {}
    seed_overrides = seed_overrides or {}

    def _rng(key: str) -> np.random.RandomState:
        if key in copies:
            copy_spec = copies[key]
            return derive_param_rng(
                int(copy_spec["seed"]),
                copy_spec.get("src_key", key),
                int(copy_spec["counter"]),
            )
        effective_seed = int(seed_overrides.get(key, seed))
        return derive_param_rng(effective_seed, key, int((counters or {}).get(key, 0)))

    return _rng


def sample_vector_from_spec(
    name: str,
    spec: dict,
    d: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample one vector parameter from a stored mode/range spec."""
    mode = int(spec["mode"])
    if name == "c":
        arr = sample_centers(
            rng, K=1, d=d,
            c_min=spec["vmin"], c_max=spec["vmax"], mode=mode,
            exclusion_zone=([spec["exclusion_lo"], spec["exclusion_hi"]]
                            if mode == 2 else None),
        )
    elif name in ("kappa_plus", "kappa_minus"):
        arr = sample_kappa(
            rng, K=1, d=d,
            kappa_min=spec["vmin"], kappa_max=spec["vmax"], mode=mode,
            beta_alpha=spec.get("alpha", 0.2),
            beta_beta=spec.get("beta", 0.2),
        )
    elif name in ("p_plus", "p_minus"):
        arr = sample_p_exponents(
            rng, K=1, d=d,
            p_min=spec["vmin"], p_max=spec["vmax"], mode=mode,
            fixed_p=spec.get("fixed", 1.0),
            p_beta_alpha=spec.get("alpha", 0.5),
            p_beta_beta=spec.get("beta", 0.5),
        )
    else:
        raise ValueError(f"Unknown vector parameter {name!r}")
    values = arr.reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values.")
    if name != 'c' and np.any(values <= 0):
        raise ValueError(f"{name} must contain values > 0.")
    return values


def _split_pair(name: str) -> tuple[str, str | None]:
    if name.endswith("_plus"):
        return name[:-5], "plus"
    if name.endswith("_minus"):
        return name[:-6], "minus"
    return name, None


def _float_from_spec(value: Any, default: float = 0.0) -> float:
    text = (value or "").strip()
    return float(text) if text else default


def _draw_scalar(spec: dict, rng: np.random.RandomState) -> float:
    lo = _float_from_spec(spec["lo"])
    hi = _float_from_spec(spec["hi"], lo)
    if hi < lo:
        lo, hi = hi, lo
    return lo if hi == lo else float(rng.uniform(lo, hi))


def realize_transforms(
    d: int,
    rng_for_key: Callable[[str], np.random.RandomState],
    specs: list[dict] | None,
) -> list[dict]:
    """Freeze transform specs into engine-ready ``{name, params}`` mappings."""
    out: list[dict] = []
    for i, transform in enumerate(specs or []):
        coupled = transform.get("coupled", {})
        params: dict = {}
        for pname, spec in transform["vec"].items():
            base, sign = _split_pair(pname)
            if sign == "minus" and base in coupled:
                continue
            mode = int(spec.get("mode", 1))
            vmin = _float_from_spec(spec.get("vmin", "0"), 0.0)
            vmax = _float_from_spec(spec.get("vmax", "0"), 0.0)
            arr = sample_transform_inner_param(
                rng_for_key(f"tf{i}:{pname}"), K=1, d=d, vmin=vmin, vmax=vmax, mode=mode,
                beta_alpha=_float_from_spec(spec.get("alpha", "0.2"), 0.2),
                beta_beta=_float_from_spec(spec.get("beta", "0.2"), 0.2),
            )
            params[pname] = arr.reshape(-1)
            if sign == "plus" and base in coupled:
                minus = f"{base}_minus"
                if coupled.get(base, True):
                    params[minus] = params[pname].copy()
                else:
                    arr = sample_transform_inner_param(
                        rng_for_key(f"tf{i}:{minus}"), K=1, d=d,
                        vmin=vmin, vmax=vmax, mode=mode,
                        beta_alpha=_float_from_spec(spec.get("alpha", "0.2"), 0.2),
                        beta_beta=_float_from_spec(spec.get("beta", "0.2"), 0.2),
                    )
                    params[minus] = arr.reshape(-1)
        for pname, spec in transform["scalar"].items():
            params[pname] = _draw_scalar(spec, rng_for_key(f"tf{i}:{pname}"))
        out.append({"name": transform["name"], "params": params})
    return out
