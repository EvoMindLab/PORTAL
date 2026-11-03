"""Human-readable labels for generated PORTAL composites."""
from __future__ import annotations

from typing import Any, Mapping


def generator_composite_label(composite: Mapping[str, Any]) -> str:
    """Return a stable label for the live generated instance."""
    d = int(composite.get("d", 0))
    j = int(composite.get("J", 0))
    block_dims = _block_dimensions(composite)
    dims_token = ",".join(str(dim) for dim in block_dims) if block_dims else "?"
    created = _created_token(composite)
    return f"PORTAL_d{d}_J{j}_blocks[{dims_token}]_{created}"


def _block_dimensions(composite: Mapping[str, Any]) -> list[int]:
    studio = composite.get("_studio")
    if isinstance(studio, Mapping):
        blocks = studio.get("blocks")
        if isinstance(blocks, list):
            dims = [
                int(block.get("d_b", 0))
                for block in blocks
                if isinstance(block, Mapping) and int(block.get("d_b", 0)) > 0
            ]
            if dims:
                return dims

    blocks = composite.get("blocks")
    if isinstance(blocks, list):
        dims = [
            int(block.get("d", 0))
            for block in blocks
            if isinstance(block, Mapping) and int(block.get("d", 0)) > 0
        ]
        if dims:
            return dims

    j = int(composite.get("J", 0))
    d = int(composite.get("d", 0))
    if j > 0 and d > 0 and d % j == 0:
        return [d // j] * j
    return []


def _created_token(composite: Mapping[str, Any]) -> str:
    studio = composite.get("_studio")
    if isinstance(studio, Mapping):
        created = str(studio.get("created_at", "")).strip()
        if created:
            # Drop sub-second precision — seconds are enough for uniqueness.
            created = created.split(".")[0]
            return _safe_label_token(created)
    return "legacy"


def _safe_label_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value)
