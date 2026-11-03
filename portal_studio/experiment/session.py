"""Data model for the Generator tab (high-d composite authoring).

These dataclasses represent the Generator's components, blocks and working
composite tree. Solver benchmark tasks use ``portal_studio.experiment_runner``.

Paper grounding (PORTAL Part I):
- ``ExpComponent`` ↔ a basin function ``φ_k`` (Eq 6 / 9). A component has no
  intrinsic dimension; its parameter vectors are sized by the parent block.
- ``ExpBlock`` ↔ a block landscape ``φ_block = min_k φ_k`` (Eq 13). Every
  component inside must share the block's dimension ``d_b``.
- ``ExperimentInventory`` ↔ the working composite tree. Blocks compose
  into ``F(x) = Σ_i w_i · φ_block_i ( x_{I_i} )`` (Eq 14).

The Education 2-D Authoring view uses a separate, simpler model.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ExpComponent:
    """One basin function ``φ_k`` (Eq 6 / 9). Dimension is the parent block's d_b.

    Carries the user's mode-driven generation spec (``vector_specs``,
    ``rotation``) plus the frozen arrays the engine consumes
    (``generated``, ``transforms``). The frozen arrays are sized to the
    parent block's ``d_b`` at "Add Component" time; there is no per-component
    dimension field because Eq 13 forbids dimension mismatch within a block.
    """
    label: str
    baseline: str                       # 'A' or 'B' (per Eqs 6 vs 9)
    seed: int                           # one seed per component (Part II convention)
    scalars: Dict[str, Any] = field(default_factory=dict)        # beta, Delta, r_ref, neutralize
    vector_specs: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # mode-aware spec per parameter
    rotation: Dict[str, Any] = field(default_factory=dict)       # RotationMode + its mode-specific args
    generated: Dict[str, np.ndarray] = field(default_factory=dict)  # frozen on Add Component
    transforms: List[Dict[str, Any]] = field(default_factory=list)  # frozen T_seq [{name, params}]
    transform_specs: List[Dict[str, Any]] = field(default_factory=list)  # editor-side spec [{name, vec, scalar}]
    pair_coupled: Dict[str, bool] = field(default_factory=dict)  # ± pairs sharing values, e.g. {"kappa": True, "p": True}
    reroll_counters: Dict[str, int] = field(default_factory=dict)  # param stream key -> reroll count
    param_seed_overrides: Dict[str, int] = field(default_factory=dict)  # key -> user-specified row seed
    param_copies: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # key -> {"from", "seed", "counter"}


@dataclass
class ExpBlock:
    """A block landscape ``φ_block = min_k φ_k`` (Eq 13).

    ``d_b`` is the block's dimension; every component in ``components`` is
    shaped by ``d_b``. ``weight`` is the block's coefficient ``w_i`` in the
    composite sum (Eq 14); it is paper-meaningful only when the block is
    composed, but is declared at block-creation time as a UX convenience.
    """
    label: str
    d_b: int
    weight: float = 1.0
    components: List[ExpComponent] = field(default_factory=list)


class ExperimentInventory:
    """The working composite tree for the Generator tab.

    Blocks live as a flat list; the composite's structure ``F(x) = Σ_i w_i ·
    φ_block_i ( x_{I_i} )`` (Eq 14) is materialised by ``build_composite``
    from this list. There is **no** flat "staged_components" pool — every
    component must belong to a block (the paper has no notion of an
    orphan ``φ_k``).
    """

    def __init__(self) -> None:
        self.blocks: List[ExpBlock] = []
        self.composites: List[Dict[str, Any]] = []
        self.composite: Optional[Dict[str, Any]] = None
        self.blocks_snapshots: List[List[ExpBlock]] = []

    def total_dimension(self) -> int:
        """Composite dimension = sum of block dimensions (Eq 14 with disjoint I_i)."""
        return sum(b.d_b for b in self.blocks)

    def add_composite(self, composite: Dict[str, Any]) -> int:
        """Append a generated composite and mark it as the latest one."""
        self.composites.append(composite)
        self.composite = composite
        self.blocks_snapshots.append(copy.deepcopy(self.blocks))
        return len(self.composites) - 1

    def remove_composite(self, idx: int) -> None:
        """Drop composite ``idx`` and its parallel block snapshot.

        Keeps ``composite`` (the latest pointer) on a surviving composite, or
        ``None`` once the list is empty. Out-of-range indices are ignored.
        """
        if not (0 <= idx < len(self.composites)):
            return
        del self.composites[idx]
        if idx < len(self.blocks_snapshots):
            del self.blocks_snapshots[idx]
        self.composite = self.composites[-1] if self.composites else None
