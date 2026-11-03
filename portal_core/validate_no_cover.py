"""Validate the component-center no-cover condition within one Layer-2 block.

For every component ``k`` in a block, the block-local minimum composition must
satisfy

    min_j phi_j(c_k) == beta_k

within a numerical tolerance.  A smaller value means that another component
covers the intended center.  This module does not validate a Layer-3 weighted
composition as a whole.  Multi-block generation must call
``validate_no_cover_block`` independently for each block before or during
composition.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Package import (normal PORTAL use).
    from .fitness import _slice_parameters_for_block, phi_component, seq_for_component
except ImportError:  # Direct-script compatibility.
    from fitness import (  # type: ignore
        _slice_parameters_for_block,
        phi_component,
        seq_for_component,
    )


Offender = Tuple[int, float, float, float]
CoveringOffender = Tuple[int, Tuple[int, ...], float, float, float]


def validate_no_cover_block(
    instance: Dict[str, Any],
    *,
    component_indices: Optional[Iterable[int]] = None,
    dimension_indices: Optional[Sequence[int]] = None,
    tol: float = 1e-6,
) -> Tuple[bool, List[Offender]]:
    """Validate one Layer-2 block and return its covered component centers.

    ``component_indices`` and ``dimension_indices`` identify a block-local view
    when Type 3 operates on a stored multi-block recipe.  They do not change or
    evaluate the Layer-3 composition.  With both omitted, the whole input is
    treated as an ordinary single-block instance.

    Each offender is ``(component_index, observed_value, beta, gap)`` using the
    component's zero-based index in ``instance``.
    """
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(f"tol must be finite and nonnegative; received {tol!r}.")

    P = instance["P"]
    d_global = int(instance["d"])
    k_total = int(instance["K"])

    if component_indices is None:
        components = list(range(k_total))
    else:
        components = [int(k) for k in component_indices]
    if not components:
        raise ValueError("A Layer-2 block must contain at least one component.")
    if any(k < 0 or k >= k_total for k in components):
        raise ValueError("component_indices contains an out-of-range component.")

    if dimension_indices is None:
        idx = np.arange(d_global, dtype=int)
    else:
        idx = np.asarray(list(dimension_indices), dtype=int)
    if idx.size == 0:
        raise ValueError("A Layer-2 block must contain at least one dimension.")
    if np.any(idx < 0) or np.any(idx >= d_global) or np.unique(idx).size != idx.size:
        raise ValueError("dimension_indices must be unique indices in [0, d).")

    centers = np.asarray(P["c"], dtype=float)
    betas = np.asarray(P["beta"], dtype=float).reshape(-1)
    p_local = _slice_parameters_for_block(P, idx, d_global)
    instance_local = {"d": int(idx.size)}
    seq_cfg = instance["transform_seq"]

    offenders: List[Offender] = []
    for k in components:
        x_local = centers[k, idx]
        values = [
            float(
                phi_component(
                    x_local,
                    p_local,
                    instance_local,
                    j,
                    seq_for_component(seq_cfg, j, k_total),
                )
            )
            for j in components
        ]
        observed = float(np.min(values))
        expected = float(betas[k])
        gap = observed - expected
        if gap < -tol:
            offenders.append((int(k), observed, expected, float(gap)))

    return len(offenders) == 0, offenders


def validate_no_cover_block_with_sources(
    instance: Dict[str, Any],
    *,
    component_indices: Optional[Iterable[int]] = None,
    dimension_indices: Optional[Sequence[int]] = None,
    tol: float = 1e-6,
) -> Tuple[bool, List[CoveringOffender]]:
    """Validate one Layer-2 block and include the covering component indices.

    Each offender is ``(component_index, covering_component_indices,
    observed_value, beta, gap)`` using zero-based indices in ``instance``.
    The coverage decision is identical to :func:`validate_no_cover_block`; this
    helper only exposes which component(s) attained the lower value.
    """
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(f"tol must be finite and nonnegative; received {tol!r}.")

    P = instance["P"]
    d_global = int(instance["d"])
    k_total = int(instance["K"])

    if component_indices is None:
        components = list(range(k_total))
    else:
        components = [int(k) for k in component_indices]
    if not components:
        raise ValueError("A Layer-2 block must contain at least one component.")
    if any(k < 0 or k >= k_total for k in components):
        raise ValueError("component_indices contains an out-of-range component.")

    if dimension_indices is None:
        idx = np.arange(d_global, dtype=int)
    else:
        idx = np.asarray(list(dimension_indices), dtype=int)
    if idx.size == 0:
        raise ValueError("A Layer-2 block must contain at least one dimension.")
    if np.any(idx < 0) or np.any(idx >= d_global) or np.unique(idx).size != idx.size:
        raise ValueError("dimension_indices must be unique indices in [0, d).")

    centers = np.asarray(P["c"], dtype=float)
    betas = np.asarray(P["beta"], dtype=float).reshape(-1)
    p_local = _slice_parameters_for_block(P, idx, d_global)
    instance_local = {"d": int(idx.size)}
    seq_cfg = instance["transform_seq"]

    offenders: List[CoveringOffender] = []
    for k in components:
        x_local = centers[k, idx]
        values = [
            float(
                phi_component(
                    x_local,
                    p_local,
                    instance_local,
                    j,
                    seq_for_component(seq_cfg, j, k_total),
                )
            )
            for j in components
        ]
        observed = float(np.min(values))
        expected = float(betas[k])
        gap = observed - expected
        if gap < -tol:
            covering = tuple(
                int(j)
                for j, value in zip(components, values)
                if int(j) != int(k) and float(value) <= observed + tol
            )
            offenders.append((int(k), covering, observed, expected, float(gap)))

    return len(offenders) == 0, offenders


def validate_no_cover(
    instance: Dict[str, Any], tol: float = 1e-6
) -> Tuple[bool, List[Offender]]:
    """Validate an ordinary single-block PORTAL instance.

    Layer-3 compositions must be checked while their constituent blocks are
    generated.  Type 3 uses ``validate_no_cover_block`` separately for each
    stored block.
    """
    if instance.get("composite", False):
        raise ValueError(
            "validate_no_cover expects one Layer-2 block, not a Layer-3 "
            "composition. Validate each block independently."
        )

    schema = instance.get("_schema_blocks")
    if isinstance(schema, list) and len(schema) > 1:
        raise ValueError(
            "validate_no_cover expects one Layer-2 block. For a stored "
            "multi-block recipe, validate each block independently with "
            "validate_no_cover_block."
        )

    return validate_no_cover_block(instance, tol=tol)


def validate_no_cover_with_sources(
    instance: Dict[str, Any], tol: float = 1e-6
) -> Tuple[bool, List[CoveringOffender]]:
    """Validate an ordinary single-block instance and include coverer indices."""
    if instance.get("composite", False):
        raise ValueError(
            "validate_no_cover expects one Layer-2 block, not a Layer-3 "
            "composition. Validate each block independently."
        )

    schema = instance.get("_schema_blocks")
    if isinstance(schema, list) and len(schema) > 1:
        raise ValueError(
            "validate_no_cover expects one Layer-2 block. For a stored "
            "multi-block recipe, validate each block independently with "
            "validate_no_cover_block."
        )

    return validate_no_cover_block_with_sources(instance, tol=tol)
