"""Pure-Python assembly logic for blocks and composites.

No Qt imports. Functions raise ``BlockAssemblyError`` /
``CompositeAssemblyError`` on invalid input so the GUI layer can present
them as dialog messages.
"""
from __future__ import annotations

from typing import List, Optional

from portal_core.instance_factory import (
    component_config_to_instance,
    merge_k1_instances,
)
from portal_studio.session import BlockSpec, ComponentSpec


class BlockAssemblyError(ValueError):
    """Raised when block assembly inputs are invalid."""


def assemble_block(
    components: List[ComponentSpec],
    weight: float,
    label: str,
    indices_strategy: str = "auto",
    explicit_indices: Optional[List[int]] = None,
) -> BlockSpec:
    """Build a BlockSpec from staged components.

    The Education editor uses two-dimensional blocks,
    so explicit_indices, when given, must have length 2.
    """
    if not components:
        raise BlockAssemblyError("a block needs at least one component")
    if weight <= 0.0:
        raise BlockAssemblyError(f"weight must be positive, got {weight}")
    if indices_strategy not in {"auto", "explicit"}:
        raise BlockAssemblyError(
            f"indices_strategy must be 'auto' or 'explicit', got {indices_strategy!r}"
        )
    if indices_strategy == "explicit":
        if explicit_indices is None or len(explicit_indices) != 2:
            raise BlockAssemblyError(
                "explicit_indices must be a list of length 2 (v2 fixes d_b = 2)"
            )
    return BlockSpec(
        label=label,
        components=list(components),
        weight=float(weight),
        indices_strategy=indices_strategy,
        explicit_indices=list(explicit_indices) if explicit_indices is not None else None,
    )


class CompositeAssemblyError(ValueError):
    """Raised when composite assembly inputs are invalid."""


def assemble_composite(blocks: List[BlockSpec]) -> dict:
    """Build a PORTAL composite dict from staged blocks.

    Each block's components are compiled into a single multi-component
    instance, then combined with ``compose_blocks``. Automatic index
    assignment uses a random permutation when ``indices_strategy == 'auto'``.
    """
    if not blocks:
        raise CompositeAssemblyError("a composite needs at least one block")

    # Local import: avoid a hard runtime dep on Qt for pure-Python users.
    from portal_core.compose_blocks import compose_blocks

    def _block_to_instance(blk: BlockSpec) -> dict:
        per_comp_insts = [
            component_config_to_instance(
                {
                    "baseline": comp.baseline,
                    "rotation_deg": comp.rotation_deg,
                    "neutralize": comp.neutralize,
                    "params": comp.params,
                    "T_seq": comp.T_seq,
                },
                d=2,
            )
            for comp in blk.components
        ]
        return merge_k1_instances(per_comp_insts)

    block_instances = [_block_to_instance(blk) for blk in blocks]
    weights = [blk.weight for blk in blocks]
    return compose_blocks(block_instances, weights=weights)


_merge_K1_instances = merge_k1_instances
