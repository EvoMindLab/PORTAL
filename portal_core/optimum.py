"""Analytic optimum helpers for PORTAL runtime instances."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def global_optimum_value(instance: Dict[str, Any]) -> float:
    """Return the known global minimum value of a PORTAL instance."""
    if _is_composite(instance):
        weights = instance["weights"]
        blocks = instance["blocks"]
        return float(sum(
            float(weight) * _single_block_min(block)
            for weight, block in zip(weights, blocks)
        ))

    schema_blocks = instance.get("_schema_blocks")
    if isinstance(schema_blocks, list) and schema_blocks:
        return _schema_blocks_min(instance, schema_blocks)

    return _single_block_min(instance)


def _single_block_min(block: Dict[str, Any]) -> float:
    return float(np.min(np.asarray(block["P"]["beta"], dtype=float)))


def _is_composite(instance: Dict[str, Any]) -> bool:
    return bool(instance.get("composite", False) or instance.get("_composite", False))


def _schema_blocks_min(
    instance: Dict[str, Any],
    schema_blocks: list[Dict[str, Any]],
) -> float:
    betas = np.asarray(instance["P"]["beta"], dtype=float).reshape(-1)
    k_total = int(instance["K"])
    if betas.size != k_total:
        raise ValueError(
            f"Invalid beta length: got {betas.size}, expected K={k_total}."
        )

    total = 0.0
    cursor = 0
    for index, block in enumerate(schema_blocks):
        k_block = int(block.get("K", 0))
        if k_block <= 0:
            raise ValueError(
                f"Invalid _schema_blocks[{index}].K={k_block}; expected positive."
            )
        if cursor + k_block > k_total:
            raise ValueError(
                "Invalid _schema_blocks partition: cumulative K exceeds total K."
            )
        weight = float(block.get("weight", 1.0))
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"Invalid _schema_blocks[{index}].weight={weight}; "
                "expected finite > 0."
            )

        block_betas = betas[cursor: cursor + k_block]
        total += weight * float(np.min(block_betas))
        cursor += k_block

    if cursor != k_total:
        raise ValueError(
            f"Invalid _schema_blocks partition: cumulative K={cursor}, "
            f"expected K={k_total}."
        )
    return float(total)
