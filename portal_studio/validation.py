"""GUI input/output checks; the scientific core remains the evaluator."""
import numpy as np

from portal_core.transform_registry import normalize_transform_name


MIN_GUI_BLOCK_DIMENSION = 2
GUI_BLOCK_DIMENSION_MESSAGE = (
    'PORTAL requires every block to have at least 2 dimensions. '
    'Use a block of dimension 2 or more.'
)


def _positive_extent(value):
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Wavelet extent eta must be finite and greater than zero.') from exc
    if not values.size or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('Wavelet extent eta must be finite and greater than zero.')


def validate_component_config(config):
    for transform in config.get('T_seq', []):
        name = normalize_transform_name(transform['name'])
        if name.partition('#')[0] == 'wavelet_mod':
            _positive_extent(transform.get('params', {}).get('eta', 0.))


def validate_instance_for_gui(instance):
    """Check GUI-supported block dimensions and active wavelet extents."""
    if instance.get('composite', False) or instance.get('_composite', False):
        for block in instance.get('blocks', []):
            validate_instance_for_gui(block)
        return
    if (instance.get('d') is not None
            and int(instance['d']) < MIN_GUI_BLOCK_DIMENSION) or any(
        int(block['d']) < MIN_GUI_BLOCK_DIMENSION
        for block in instance.get('_schema_blocks', [])
    ):
        raise ValueError(GUI_BLOCK_DIMENSION_MESSAGE)
    parameters = instance.get('P', {})
    for component, sequence in enumerate(instance.get('transform_seq', [])):
        for reference in sequence:
            # Runtime dispatch retains its canonical-name policy.
            if str(reference).partition('#')[0] != 'wavelet_mod':
                continue
            values = np.asarray(parameters.get('T_' + reference, {}).get('eta', 0.),
                                dtype=float).reshape(-1)
            if values.size not in (1, int(instance['K'])):
                raise ValueError('Wavelet extent eta has invalid component dimensions.')
            _positive_extent(values[0] if values.size == 1 else values[component])
