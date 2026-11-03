"""Turn ExpComponents/ExpBlocks into PORTAL instance dicts by reusing core.

Component packing and K=1 merging are backend-owned by
``portal_core.instance_factory``; this module only adapts Studio dataclasses to
that core interface and delegates block composition to ``portal_core``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from portal_core.compose_blocks import compose_blocks
from portal_core.instance_factory import (
    component_config_to_instance,
    merge_k1_instances,
)
from portal_core.save_instance import save_composite_as_unified
from portal_studio.experiment.session import ExpBlock, ExpComponent
from portal_studio.validation import validate_component_config, validate_instance_for_gui


def component_to_instance(comp: ExpComponent, d: Optional[int] = None) -> Dict[str, Any]:
    """Build a K=1 PORTAL instance dict from a generated component.

    ``d`` defaults to the length of the component's frozen ``c`` array — the
    component itself carries no dimension (per Eq 13). Callers who already
    know the parent block's ``d_b`` may pass it explicitly for stricter
    validation.
    """
    g = comp.generated
    inferred = int(np.asarray(g["c"]).shape[-1])
    if d is None:
        d = inferred
    elif int(d) != inferred:
        raise ValueError(
            f"component {comp.label!r} has d={inferred} but block requested d_b={d}"
        )
    params: Dict[str, Any] = {
        "c": np.asarray(g["c"], dtype=float),
        "beta": float(comp.scalars.get("beta", 0.0)),
        "kappa_plus": np.asarray(g["kappa_plus"], dtype=float),
        "kappa_minus": np.asarray(g["kappa_minus"], dtype=float),
        "R": np.asarray(g["R"], dtype=float),
    }
    if "Psi_deg" in g:
        params["Psi_deg"] = np.asarray(g["Psi_deg"], dtype=float)
    if str(comp.baseline).upper() == "A":
        params["p_plus"] = np.asarray(g["p_plus"], dtype=float)
        params["p_minus"] = np.asarray(g["p_minus"], dtype=float)
    else:
        params["p_single"] = float(g.get("p_single", 1.0))
    cfg = {
        "Seed": int(comp.seed),
        "baseline": comp.baseline,
        "rotation_deg": 0.0,   # ignored: explicit R provided
        "neutralize": dict(comp.scalars.get("neutralize", {"enable": False})),
        "params": params,
        "T_seq": list(comp.transforms),
    }
    validate_component_config(cfg)
    return component_config_to_instance(cfg, d)


def block_from_components(label: str, components: List[ExpComponent],
                          *, d_b: int, weight: float = 1.0) -> ExpBlock:
    """Wrap a list of components into an :class:`ExpBlock` of dimension ``d_b``.

    Validates that every component's frozen ``c`` array matches ``d_b``
    (Eq 13 requires shared dimension across all components in a block).
    Supports programmatic block construction; the Generator tab
    creates blocks via "Add Block" before any component exists.
    """
    if not components:
        raise ValueError("a block needs at least one component")
    if d_b <= 0:
        raise ValueError(f"d_b must be positive, got {d_b}")
    for c in components:
        inferred = int(np.asarray(c.generated["c"]).shape[-1])
        if inferred != d_b:
            raise ValueError(
                f"component {c.label!r} has d={inferred} but block requested d_b={d_b}"
            )
    return ExpBlock(label=label, d_b=int(d_b), weight=weight, components=list(components))


def block_to_instance(block: ExpBlock) -> Dict[str, Any]:
    """Stack a block's components (all dimension ``d_b``) into one K-component dict."""
    if not block.components:
        raise ValueError(f"block {block.label!r} has no components")
    insts = [component_to_instance(c, d=block.d_b) for c in block.components]
    return merge_k1_instances(insts)


def build_composite(blocks: List[ExpBlock], *, seed: Optional[int] = None) -> Dict[str, Any]:
    """Compose blocks (mixed d_b allowed) into a multi-block PORTAL instance.

    Internally this uses ``compose_blocks`` to resolve the global
    dimension layout and per-block weights, but the result is only an in-memory
    runtime staging structure -- it is *never* written to disk as a composite
    envelope. ``save_studio_composite`` flattens it into a single unified
    instance so the on-disk schema is identical to a single-block instance and
    to a script-generated multi-block anchor.

    Attaches a ``_studio`` provenance block describing the modes used for every
    component, so the saved JSON carries a full regeneration recipe alongside
    the frozen arrays.
    """
    if not blocks:
        raise ValueError("a composite needs at least one block")
    block_dicts = [block_to_instance(b) for b in blocks]
    weights = [b.weight for b in blocks]
    composite = compose_blocks(block_dicts, weights=weights, seed=seed)
    composite["_studio"] = _studio_provenance(blocks)
    return composite


def save_studio_composite(composite: Dict[str, Any], *,
                          filename: str,
                          output_dir: str) -> str:
    """Save a multi-block instance as a single unified JSON (not a composite).

    The Studio stages multi-block instances via ``build_composite`` (which
    returns a Layer-3 runtime structure), but persists them through
    ``save_composite_as_unified`` so the artifact is a standalone unified
    instance -- one file, block hierarchy inside, no ``composite: True`` flag
    and no external block references. The file has top-level ``blocks`` and loads through
    ``load_instance`` and evaluates identically. The Studio's ``_studio``
    provenance block is merged back in afterwards (loaders ignore unknown
    top-level keys).
    """
    validate_instance_for_gui(composite)
    return save_composite_as_unified(
        composite, filename=filename, output_dir=output_dir
    )


def _studio_provenance(blocks: List[ExpBlock]) -> Dict[str, Any]:
    """Per-component mode info, byte-stable for diffing and replay."""
    return {
        "schema": "portal_studio.experiment/1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"),
        "blocks": [
            {
                "label": b.label,
                "d_b": int(b.d_b),
                "weight": float(b.weight),
                "components": [
                    {
                        "label": c.label,
                        "baseline": c.baseline,
                        "seed": int(c.seed),
                        "reroll_counters": dict(getattr(c, "reroll_counters", {}) or {}),
                        "param_seed_overrides": dict(getattr(c, "param_seed_overrides", {}) or {}),
                        "param_copies": dict(getattr(c, "param_copies", {}) or {}),
                        "rotation_mode": int(c.rotation.get("mode", 1)),
                        "vector_specs": dict(c.vector_specs),
                        "transforms": [dict(t, params={}) for t in c.transforms],  # names only; arrays are frozen in P
                    }
                    for c in b.components
                ],
            }
            for b in blocks
        ],
    }
