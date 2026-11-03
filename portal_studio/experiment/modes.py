"""Paper-aligned per-parameter random-generation modes.

Every sampler exposes exactly the modes ``portal_core/benchmark_generator.py``
implements — no more, no fewer — so a Studio-authored component round-trips
through ``save_instance`` / ``load_instance`` byte-equivalently to a
generator-authored one.

Each module-level constant ``*_MODES`` is the user-facing menu for that
parameter, listed in the engine's mode-1..N order (e.g.
``"Uniform Random"``, ``"With Exclusion Zone"``, ...). The engine call's
``mode`` argument is simply the selected item's index + 1.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from portal_core.benchmark_generator import (
    sample_centers as _engine_sample_centers,
    sample_kappa as _engine_sample_kappa,
    sample_exponents as _engine_sample_exponents,
    sample_beta as _engine_sample_beta,
    sample_baseline as _engine_sample_baseline,
    sample_transform_seq as _engine_sample_transform_seq,
    sample_transform_param as _engine_sample_transform_param,
    _get_parameter_ranges as _engine_get_parameter_ranges,
)


def canonical_ranges(bounds_min: float = -100.0,
                     bounds_max: float = 100.0) -> dict:
    """Return the engine's default parameter ranges as a plain dict.

    Single source of truth for UI default values. The engine's
    ``_get_parameter_ranges`` is the canonical reference; this function wraps
    it so callers don't reach into a private symbol.
    """
    portal = {"bounds_min": float(bounds_min), "bounds_max": float(bounds_max)}
    ranges = _engine_get_parameter_ranges(portal)
    # Expose the raw domain bounds too (the engine returns the +/-10-margin
    # c_min/c_max; callers needing the full search box read these directly).
    ranges.setdefault("bounds_min", float(bounds_min))
    ranges.setdefault("bounds_max", float(bounds_max))
    return ranges


# Engine constants for the per-transform scalars that ``_get_parameter_ranges``
# does NOT cover — kept here as named constants with explicit engine
# references so any future drift is easy to spot.
LS_SIGMA_DEFAULT = 0.1     # benchmark_generator.py: ls_sigma kwarg default
EPS_DEFAULT = 1e-12        # benchmark_generator.py: T_WM['eps'] / T_RP['eps']


def canonical_transform_defaults() -> dict:
    """Per-transform default parameter spec, sourced from the engine.

    Shape: ``{transform_name: {"vec": {param: (min, max)}, "scalar": {...}}}``.
    Vector ranges are pulled from :func:`canonical_ranges`; scalar engine
    constants (``LS_SIGMA_DEFAULT``, ``EPS_DEFAULT``, plus ``WM`` ``eta``) are
    encoded as ``(value, value)`` pairs so the UI's "fixed when min == max"
    convention keeps working.
    """
    R = canonical_ranges()
    return {
        "additive_periodic": {
            "vec": {
                "mu_plus":     (R["AP_mu_min"],    R["AP_mu_max"]),
                "mu_minus":    (R["AP_mu_min"],    R["AP_mu_max"]),
                "gamma_plus":  (R["AP_gamma_min"], R["AP_gamma_max"]),
                "gamma_minus": (R["AP_gamma_min"], R["AP_gamma_max"]),
                "omega_plus":  (R["AP_omega_min"], R["AP_omega_max"]),
                "omega_minus": (R["AP_omega_min"], R["AP_omega_max"]),
            },
            "scalar": {},
        },
        "log_sinusoidal": {
            "vec": {
                "mu_plus":      (R["LS_mu_min"],    R["LS_mu_max"]),
                "mu_minus":     (R["LS_mu_min"],    R["LS_mu_max"]),
                "omega1_plus":  (R["LS_omega_min"], R["LS_omega_max"]),
                "omega1_minus": (R["LS_omega_min"], R["LS_omega_max"]),
                "omega2_plus":  (R["LS_omega_min"], R["LS_omega_max"]),
                "omega2_minus": (R["LS_omega_min"], R["LS_omega_max"]),
            },
            "scalar": {
                "sigma": (LS_SIGMA_DEFAULT, LS_SIGMA_DEFAULT),
            },
        },
        "wavelet_mod": {
            "vec": {
                "mu_plus":     (R["WM_mu_min"],    R["WM_mu_max"]),
                "mu_minus":    (R["WM_mu_min"],    R["WM_mu_max"]),
                "omega_plus":  (R["WM_omega_min"], R["WM_omega_max"]),
                "omega_minus": (R["WM_omega_min"], R["WM_omega_max"]),
            },
            "scalar": {
                "eta": (R["WM_eta_min"], R["WM_eta_max"]),
                "eps": (EPS_DEFAULT,     EPS_DEFAULT),
            },
        },
        "tensor_interference": {
            "vec": {
                "mu_plus":     (R["TI_mu0_min"],   R["TI_mu0_max"]),
                "mu_minus":    (R["TI_mu0_min"],   R["TI_mu0_max"]),
                "omega_plus":  (R["TI_omega_min"], R["TI_omega_max"]),
                "omega_minus": (R["TI_omega_min"], R["TI_omega_max"]),
            },
            "scalar": {},
        },
        "radial_polytrig": {
            "vec": {},
            "scalar": {
                "mu":    (R["RP_mu_min"],    R["RP_mu_max"]),
                "p":     (R["RP_p_min"],     R["RP_p_max"]),
                "q":     (R["RP_q_min"],     R["RP_q_max"]),
                "omega": (R["RP_omega_min"], R["RP_omega_max"]),
                "eps":   (EPS_DEFAULT,       EPS_DEFAULT),
            },
        },
    }


CENTER_MODES: List[str] = [
    "Uniform Random",
    "With Exclusion Zone",
]


def sample_centers(rng: np.random.RandomState, *,
                   K: int, d: int,
                   c_min: float, c_max: float,
                   mode: int,
                   exclusion_zone: Optional[List[float]] = None) -> np.ndarray:
    """Sample K centres in d dimensions per the engine's centre menu.

    Args:
        rng: random state (the component's seeded ``np.random.RandomState``).
        K, d: component count and dimensionality.
        c_min, c_max: per-coordinate sampling bounds (typically the recipe's
            bounds inset by 10).
        mode: 1-2 per ``CENTER_MODES`` in the Studio UI. The wrapper still
            forwards other engine-supported modes for legacy/programmatic use.
        exclusion_zone: ``[lo, hi]`` excluded interval; required for mode 2.

    Returns:
        Array of shape ``(K, d)``.
    """
    ranges = {"c_min": float(c_min), "c_max": float(c_max)}
    kwargs = {}
    if mode == 2:
        if exclusion_zone is None:
            raise ValueError("center mode 2 needs exclusion_zone=[lo, hi]")
        kwargs["exclusion_zone"] = list(exclusion_zone)
    return _engine_sample_centers(rng, int(K), int(d), ranges, int(mode), **kwargs)


# ─── kappa ────────────────────────────────────────────────────────────────

KAPPA_MODES: List[str] = [
    "Uniform Random",
    "Beta Distribution",
    "Isotropic Random",
    "Linear Spread (Shuffled)",
    "Fixed Unit Scale (κ=1)",
]


def sample_kappa(rng: np.random.RandomState, *,
                 K: int, d: int,
                 kappa_min: float, kappa_max: float,
                 mode: int,
                 beta_alpha: float = 0.2,
                 beta_beta: float = 0.2) -> np.ndarray:
    """Sample one curvature vector per component, per direction.

    ``mode`` ∈ {1..5} per ``KAPPA_MODES``. Mode 2 (Beta distribution) consumes
    ``beta_alpha`` and ``beta_beta``; the other modes ignore them.

    Returns an array of shape ``(K, d)``.
    """
    ranges = {"kappa_min": float(kappa_min), "kappa_max": float(kappa_max)}
    return _engine_sample_kappa(rng, int(K), int(d), ranges, int(mode),
                                beta_alpha=float(beta_alpha),
                                beta_beta=float(beta_beta))


# ─── exponents p ──────────────────────────────────────────────────────────

P_MODES: List[str] = [
    "Uniform Random",
    "Fixed Value",
    "Fixed Gradient",
    "Beta Distribution",
    "Linear Spread (Shuffled)",
]


def sample_p_exponents(rng: np.random.RandomState, *,
                       K: int, d: int,
                       p_min: float, p_max: float,
                       mode: int,
                       fixed_p: float = 0.8,
                       p_beta_alpha: float = 0.5,
                       p_beta_beta: float = 0.5) -> np.ndarray:
    """Sample exponent vectors per ``P_MODES``. Returns shape ``(K, d)``.

    Mode 2 consumes ``fixed_p``; mode 4 consumes ``p_beta_alpha`` /
    ``p_beta_beta``. Other modes ignore those kwargs.
    """
    ranges = {"p_min": float(p_min), "p_max": float(p_max)}
    return _engine_sample_exponents(rng, int(K), int(d), ranges, int(mode),
                                    fixed_p=float(fixed_p),
                                    p_beta_alpha=float(p_beta_alpha),
                                    p_beta_beta=float(p_beta_beta))


# ─── beta offsets ─────────────────────────────────────────────────────────

BETA_MODES: List[str] = [
    "Independent random",
    "Shared random value",
    "Fixed value",
    "Beta distribution",
    "Linspace with permutation",
    "Shared base + bounded offsets",
]


def sample_beta_offsets(rng: np.random.RandomState, *,
                        K: int,
                        beta_min: float, beta_max: float,
                        mode: int,
                        fixed_beta: float = 0.0,
                        beta_alpha: float = 0.2,
                        beta_beta_param: float = 0.2,
                        beta_gap: float = 20.0,
                        beta_opt_gap: float = 1.0) -> np.ndarray:
    """Sample component offsets per ``BETA_MODES``. Returns shape ``(K,)``.

    Mode 3 consumes ``fixed_beta``; mode 4 consumes ``beta_alpha`` /
    ``beta_beta_param``; mode 6 consumes ``beta_gap`` and ``beta_opt_gap``
    (one shared base in ``[beta_min, beta_max - gap]``; one random component
    sits at the base and the others draw offsets in ``[opt_gap, gap]``,
    keeping depth gaps no-cover-compatible and decoys away from the optimum).
    Other modes ignore those kwargs.
    """
    ranges = {"beta_min": float(beta_min), "beta_max": float(beta_max)}
    return _engine_sample_beta(rng, int(K), ranges, int(mode),
                               fixed_beta=float(fixed_beta),
                               beta_alpha=float(beta_alpha),
                               beta_beta_param=float(beta_beta_param),
                               beta_gap=float(beta_gap),
                               beta_opt_gap=float(beta_opt_gap))


# ─── baseline A/B ─────────────────────────────────────────────────────────

BASELINE_MODES: List[str] = [
    "Random 50/50",
    "Custom probability",
    "All same",
    "User-specified",
]


def sample_baselines(rng: np.random.RandomState, *,
                     K: int,
                     mode: int,
                     prob_a: float = 0.5,
                     baseline_type: str = "A",
                     user_seq: Optional[List[str]] = None) -> List[str]:
    """Return a length-K list of 'A'/'B' per ``BASELINE_MODES``.

    Mode 2 consumes ``prob_a``; mode 3 consumes ``baseline_type``;
    mode 4 consumes ``user_seq`` (length K of 'A'/'B').
    """
    return _engine_sample_baseline(rng, int(K), int(mode),
                                   prob_a=float(prob_a),
                                   baseline_type=str(baseline_type),
                                   user_seq=list(user_seq or []))


# ─── transform sequence ──────────────────────────────────────────────────

TRANSFORM_SEQ_MODES: List[str] = [
    "Fully random",
    "Fixed count",
    "No transformations",
    "User-specified",
]


def sample_transform_sequence(rng: np.random.RandomState, *,
                              K: int,
                              mode: int,
                              count_min: int = 0,
                              count_max: int = 1,
                              count: int = 1,
                              user_seq: Optional[List[List[str]]] = None) -> List[List[str]]:
    """Return a length-K list of transform-name lists per ``TRANSFORM_SEQ_MODES``.

    Mode 1 consumes ``count_min``/``count_max``; mode 2 consumes ``count``;
    mode 4 consumes ``user_seq``.
    """
    return _engine_sample_transform_seq(
        rng, int(K), int(mode),
        count_min=int(count_min), count_max=int(count_max),
        count=count,
        user_seq=list(user_seq) if user_seq is not None else [[]],
    )


# ─── transform inner parameters (mu_+, omega_+, ...) ─────────────────────

TRANSFORM_PARAM_MODES: List[str] = [
    "Independent random",
    "Component-shared",
    "Dimension-shared",
    "Global shared",
    "Beta distribution",
    "Linspace with permutation",
]


def sample_transform_inner_param(rng: np.random.RandomState, *,
                                 K: int, d: int,
                                 vmin: float, vmax: float,
                                 mode: int,
                                 beta_alpha: float = 0.2,
                                 beta_beta: float = 0.2) -> np.ndarray:
    """Sample one transform-inner parameter (e.g. ``mu_plus``) of shape ``(K, d)``.

    Mode 5 consumes ``beta_alpha`` / ``beta_beta``; the other modes ignore
    them.
    """
    return _engine_sample_transform_param(
        rng, int(K), int(d), float(vmin), float(vmax), int(mode),
        beta_alpha=float(beta_alpha), beta_beta=float(beta_beta),
    )
