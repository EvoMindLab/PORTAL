"""Session inventory: in-memory store of staged components, blocks, composites.

Pure-Python; Qt-agnostic. The GUI layer wraps this with signals via
``QObject`` adapters when it needs change notifications.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ComponentSpec:
    """A single component, the same shape the existing GUI cfg uses."""
    label: str
    baseline: str           # 'A' or 'B'
    rotation_deg: float
    neutralize: Dict[str, Any]
    params: Dict[str, Any]
    T_seq: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BlockSpec:
    """A staged block: a set of components + weight + variable strategy."""
    label: str
    components: List[ComponentSpec]
    weight: float
    indices_strategy: str   # 'auto' or 'explicit'
    explicit_indices: Optional[List[int]]  # used when indices_strategy == 'explicit'


@dataclass
class CompositeRef:
    """A built composite plus optional filesystem path if saved."""
    label: str
    composite: Dict[str, Any]
    saved_path: Optional[str] = None
    # The staged blocks this composite was assembled from, so the model
    # hierarchy can display composite -> block -> component nesting.
    source_blocks: Optional[List[BlockSpec]] = None


class SessionInventory:
    """In-memory store of the user's authoring artefacts.

    Methods mutate the lists in place; GUI adapters observe and refresh.
    """

    def __init__(self) -> None:
        self.staged_components: List[ComponentSpec] = []
        self.staged_blocks: List[BlockSpec] = []
        self.composites: List[CompositeRef] = []

    # -------- components --------
    def add_component(self, c: ComponentSpec) -> int:
        self.staged_components.append(c)
        return len(self.staged_components) - 1

    def remove_component(self, index: int) -> None:
        del self.staged_components[index]

    # -------- blocks --------
    def add_block(self, b: BlockSpec) -> int:
        self.staged_blocks.append(b)
        return len(self.staged_blocks) - 1

    def remove_block(self, index: int) -> None:
        del self.staged_blocks[index]

    # -------- composites --------
    def add_composite(self, comp: CompositeRef) -> int:
        self.composites.append(comp)
        return len(self.composites) - 1

    # -------- helpers --------
    def composite_total_d(self) -> int:
        """Total dimension of a composite formed from all staged blocks.

        Each Education block has two dimensions, so total d is twice the block count.
        """
        return 2 * len(self.staged_blocks)
