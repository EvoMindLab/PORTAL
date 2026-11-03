"""Codecs for unified instance files and version-2 containers.

Canonical writers save the unified payload with top-level ``blocks``. The
decoder accepts that layout and supported version-2 containers. Callers can
explicitly request a container through ``encode_instance_file``.
"""
from __future__ import annotations

FORMAT = 'PORTAL-instance'
VERSION = 2
INDEPENDENT_TRANSFORMS = 'independent_transform_occurrences'
EXPLICIT_ROTATIONS = 'explicit_rotation_matrices'
SUPPORTED_FEATURES = frozenset({INDEPENDENT_TRANSFORMS, EXPLICIT_ROTATIONS})


def needs_independent_transforms(payload: dict) -> bool:
    return any(
        '#' in str(reference)
        for block in payload.get('blocks', [])
        for component in block.get('components', [])
        for reference in component.get('transform_seq', [])
    )


def encode_instance_file(payload: dict) -> dict:
    """Wrap extension-bearing payloads in a version-2 container."""
    features = []
    if needs_independent_transforms(payload):
        features.append(INDEPENDENT_TRANSFORMS)
    if needs_explicit_rotations(payload):
        features.append(EXPLICIT_ROTATIONS)
    if not features:
        return payload
    return dict(portal_format=FORMAT, format_version=VERSION,
                required_features=features, instance=payload)


def needs_explicit_rotations(payload: dict) -> bool:
    return any(
        isinstance(component.get('rotation'), dict)
        and 'R' in component['rotation']
        for block in payload.get('blocks', [])
        for component in block.get('components', [])
    )


def decode_instance_file(raw):
    """Unwrap supported files, rejecting unknown versions/features before parsing."""
    if not isinstance(raw, dict) or 'portal_format' not in raw:
        return raw  # Unified payloads pass through unchanged.
    if raw['portal_format'] != FORMAT:
        raise ValueError('Unsupported PORTAL file format. Use a compatible PORTAL reader.')
    version = raw.get('format_version')
    if type(version) is not int or version != VERSION:
        raise ValueError(f'Unsupported PORTAL file format version {version!r}. '
                         'Use a compatible PORTAL reader.')
    if 'blocks' in raw:
        raise ValueError('Ambiguous PORTAL file: versioned files must not contain top-level blocks.')
    features = raw.get('required_features')
    if not isinstance(features, list) or any(not isinstance(f, str) for f in features):
        raise ValueError('PORTAL required_features must be a list of feature names.')
    unknown = set(features) - SUPPORTED_FEATURES
    if unknown:
        raise ValueError('Unsupported PORTAL file features: ' + ', '.join(sorted(unknown))
                         + '. Use a compatible PORTAL reader.')
    payload = raw.get('instance')
    if not isinstance(payload, dict) or 'blocks' not in payload or 'portal_format' in payload:
        raise ValueError('PORTAL versioned file must contain one unified instance with blocks.')
    if needs_independent_transforms(payload) and INDEPENDENT_TRANSFORMS not in features:
        raise ValueError('PORTAL file does not declare required independent transformation support.')
    if needs_explicit_rotations(payload) and EXPLICIT_ROTATIONS not in features:
        raise ValueError('PORTAL file does not declare required explicit rotation matrix support.')
    return payload
