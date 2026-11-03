"""Canonical PORTAL transform metadata.

This module is the shared source of truth for transform names, runtime ``P``
block names, schema field order, and scalar/vector field classification. GUI,
load/save, realization, and adapter code should import these constants instead
of copying transform tables locally.
"""
from __future__ import annotations

from typing import Dict, Tuple


TRANSFORM_ALIASES: Dict[str, str] = {
    "radial_logsin": "radial_polytrig",
}

TRANSFORM_DISPLAY_NAMES: Dict[str, str] = {
    "additive_periodic": "Additive Periodic",
    "log_sinusoidal": "Log-Sinusoidal",
    "wavelet_mod": "Wavelet-Inspired Modulation",
    "tensor_interference": "Tensor Interference",
    "radial_polytrig": "Radial Polynomial-Trigonometric",
}

TRANSFORM_BLOCK_TO_SEQ_NAME: Dict[str, str] = {
    "T_additive_periodic": "additive_periodic",
    "T_log_sinusoidal": "log_sinusoidal",
    "T_wavelet_mod": "wavelet_mod",
    "T_tensor_interference": "tensor_interference",
    "T_radial_polytrig": "radial_polytrig",
}

TRANSFORM_SEQ_TO_BLOCK_NAME: Dict[str, str] = {
    seq_name: block_name
    for block_name, seq_name in TRANSFORM_BLOCK_TO_SEQ_NAME.items()
}

ALLOWED_TRANSFORM_NAMES = frozenset(TRANSFORM_SEQ_TO_BLOCK_NAME)

TRANSFORM_BLOCK_FIELD_ORDER: Dict[str, Tuple[str, ...]] = {
    "T_additive_periodic": (
        "mu_plus", "mu_minus",
        "gamma_plus", "gamma_minus",
        "omega_plus", "omega_minus",
    ),
    "T_log_sinusoidal": (
        "mu_plus", "mu_minus",
        "omega1_plus", "omega2_plus",
        "omega1_minus", "omega2_minus",
        "sigma",
    ),
    "T_wavelet_mod": (
        "mu_plus", "mu_minus",
        "omega_plus", "omega_minus",
        "eta", "eps",
    ),
    "T_tensor_interference": (
        "mu_plus", "mu_minus",
        "omega_plus", "omega_minus",
    ),
    "T_radial_polytrig": (
        "mu", "p", "q", "omega", "eps",
    ),
}

TRANSFORM_PER_COMPONENT_SCALARS: Dict[str, Tuple[str, ...]] = {
    "T_log_sinusoidal": ("sigma",),
    "T_wavelet_mod": ("eta",),
    "T_radial_polytrig": ("mu", "p", "q", "omega"),
}

TRANSFORM_BLOCK_SCALARS: Dict[str, Tuple[str, ...]] = {
    "T_wavelet_mod": ("eps",),
    "T_radial_polytrig": ("eps",),
}

TRANSFORM_PER_DIM_FIELDS: Dict[str, Tuple[str, ...]] = {
    block_name: tuple(
        field
        for field in fields
        if (
            field not in TRANSFORM_PER_COMPONENT_SCALARS.get(block_name, ())
            and field not in TRANSFORM_BLOCK_SCALARS.get(block_name, ())
        )
    )
    for block_name, fields in TRANSFORM_BLOCK_FIELD_ORDER.items()
}


def normalize_transform_name(name: str) -> str:
    """Return a canonical type or occurrence reference (e.g. ``wavelet_mod#2``).

    Unsuffixed references retain their legacy shared-parameter meaning. A
    suffix identifies an independently configured occurrence of the same type.
    """
    base, separator, occurrence = str(name).strip().partition('#')
    canonical = TRANSFORM_ALIASES.get(base, base)
    if canonical not in ALLOWED_TRANSFORM_NAMES:
        raise ValueError(f"Unknown transform: {canonical}")
    if separator:
        if not occurrence.isdecimal() or int(occurrence) < 2:
            raise ValueError(f"Invalid transform occurrence: {name}")
        return f"{canonical}#{int(occurrence)}"
    return canonical


def transform_type(name: str) -> str:
    """Return the scientific operator type of a sequence reference."""
    return normalize_transform_name(name).partition('#')[0]


def transform_block_name(name: str) -> str:
    """Return the runtime ``P`` block name for a recipe transform name."""
    return 'T_' + normalize_transform_name(name)


def transform_block_type(block_name: str) -> str:
    """Return the metadata key for a possibly occurrence-specific block."""
    if not str(block_name).startswith('T_'):
        raise ValueError(f"Not a transform parameter block: {block_name}")
    return TRANSFORM_SEQ_TO_BLOCK_NAME[transform_type(str(block_name)[2:])]


def iter_transform_blocks(parameters):
    """Yield ``(storage key, metadata key, block)`` for all transform blocks."""
    for key, block in parameters.items():
        if not str(key).startswith('T_') or not isinstance(block, dict):
            continue
        try:
            base_key = transform_block_type(key)
        except ValueError:
            continue
        yield key, base_key, block


def parameters_for_transform(parameters, name: str):
    """Expose one occurrence under its operator's legacy evaluator key."""
    key = transform_block_name(name)
    base_key = transform_block_type(key)
    if key == base_key:
        return parameters
    if key not in parameters:
        raise ValueError(f"Missing parameters for transformation {name!r}")
    return {**parameters, base_key: parameters[key]}
