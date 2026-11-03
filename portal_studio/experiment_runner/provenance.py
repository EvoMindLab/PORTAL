"""Stable identity of the concrete problems used in paired experiments."""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        # Unused rho entries in legacy/runtime layouts may be NaN. Preserve
        # their identity without emitting invalid JSON or discarding parameters.
        return {'__nonfinite_float__': str(value)}
    return value


def _digest(value) -> str:
    raw = json.dumps(_plain(value), sort_keys=True, separators=(',', ':'),
                     allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def problem_fingerprint(instance: dict) -> str:
    """Hash evaluation parameters/bounds, excluding labels, seeds and caches.

    Hashing frozen parameters (rather than file paths or display names) also
    detects a recipe that was edited between two algorithm submissions.
    """
    if instance.get('composite', False):
        payload = {key: instance[key] for key in (
            'd', 'bounds_min', 'bounds_max', 'weights', 'indices_per_block',
        )}
        payload['blocks'] = [problem_fingerprint(block) for block in instance['blocks']]
    else:
        payload = {key: instance[key] for key in (
            'd', 'K', 'bounds_min', 'bounds_max', 'transform_seq', 'P',
        )}
        payload['blocks'] = instance.get('_schema_blocks')
    return _digest(payload)


def pairing_key(source: dict, evaluated: dict, *, realization_type: str,
                base_seed, sample_id: int, fe_budget: int) -> str:
    """Identify a paired trial independently of the optimiser's random seed."""
    return _digest({
        'source': problem_fingerprint(source),
        'evaluated': problem_fingerprint(evaluated),
        'realization_type': realization_type,
        'base_seed': base_seed if realization_type != 'none' else None,
        'sample_id': int(sample_id),
        'fe_budget': int(fe_budget),
    })
