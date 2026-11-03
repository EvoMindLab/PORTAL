"""Read-only checks before Studio uses a centre-based analytic optimum."""
from __future__ import annotations

import numpy as np


def validate_centres_in_domain(instance: dict) -> None:
    """Reject unsupported centres without moving them or consuming randomness.

    Unified multi-block instances pad centres outside each block's active
    coordinates. Only active coordinates belong to that component's domain.
    This checks feasibility, not every scientific assumption of an optimum.
    """
    dimension = int(instance['d'])
    count = int(instance['K'])
    lower = np.broadcast_to(np.asarray(instance['bounds_min'], dtype=float), (dimension,))
    upper = np.broadcast_to(np.asarray(instance['bounds_max'], dtype=float), (dimension,))
    if (not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)):
        raise ValueError('Evaluation bounds must be finite, with lower bounds below upper bounds.')
    centres = np.asarray(instance['P']['c'], dtype=float)
    if centres.shape != (count, dimension):
        raise ValueError('Cannot verify component centres: their dimensions do not match the instance.')

    blocks = instance.get('_schema_blocks') or [
        dict(block_id=1, K=count, d=dimension, dim_indices=list(range(dimension)))
    ]
    start = 0
    for block_index, block in enumerate(blocks):
        block_count = int(block['K'])
        indices = np.asarray(block.get('dim_indices', list(range(dimension))), dtype=int)
        if (block_count <= 0 or start + block_count > count
                or indices.ndim != 1 or not indices.size
                or np.any(indices < 0) or np.any(indices >= dimension)
                or len(np.unique(indices)) != len(indices)):
            raise ValueError('Cannot verify component centres: invalid block dimensions.')
        active = centres[start:start + block_count, :][:, indices]
        invalid = (~np.isfinite(active) | (active < lower[indices])
                   | (active > upper[indices]))
        if np.any(invalid):
            row, column = np.argwhere(invalid)[0]
            coordinate = int(indices[column])
            value = float(active[row, column])
            block_id = block.get('block_id', block_index + 1)
            raise ValueError(
                f'Cannot run this instance: block {block_id}, component {row + 1}, '
                f'coordinate {coordinate + 1} has centre {value:g} outside '
                f'the evaluation bounds [{lower[coordinate]:g}, {upper[coordinate]:g}]. '
                'Use a domain that contains the component centres, or a fixed '
                'instance whose centres are inside the domain.'
            )
        start += block_count
    if start != count:
        raise ValueError('Cannot verify component centres: the blocks do not cover all components.')
