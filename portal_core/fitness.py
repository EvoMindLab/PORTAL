"""
Author: Delaram Yazdani, Danial Yazdani, Mai Peng
Email: delaram.yazdani@yahoo.com
       danial.yazdani@gmail.com
       pengmai1998@gmail.com
Title: PORTAL Fitness Evaluation

Description:
    Evaluates the fitness of a solution vector using the PORTAL benchmark

Inputs:
    x: Solution vector (d×1 or 1D array)
    portal_instance: PORTAL benchmark instance dictionary

Output:
    f: Fitness value (scalar)
"""

import os
import numpy as np
from typing import Dict, Any, List, Union

try:
    from .transform_registry import (
        iter_transform_blocks, parameters_for_transform, transform_type,
    )
except ImportError:
    from transform_registry import (
        iter_transform_blocks, parameters_for_transform, transform_type,
    )

_EVALUATION_DEVICES = {'auto', 'numpy', 'cpu', 'cuda', 'mps'}
_TORCH_BATCH_TRANSFORMS = {
    'none',
    'additive_periodic',
    'log_sinusoidal',
    'wavelet_mod',
    'tensor_interference',
    'radial_polytrig',
}


def _p_single_at(P: Dict[str, Any], k: int) -> float:
    """Return scalar p_single for component k from 0D/1D/2D storage."""
    arr = np.asarray(P['p_single'])
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[k])
    return float(arr[k, 0])


def _normalized_schema_blocks(portal_instance: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return validated block metadata with a synthetic single-block fallback."""
    schema_blocks = portal_instance.get('_schema_blocks')
    if not isinstance(schema_blocks, list) or len(schema_blocks) == 0:
        return [{
            'block_id': 1,
            'K': int(portal_instance['K']),
            'weight': 1.0,
            'dim_indices': list(range(int(portal_instance['d']))),
        }]

    K = int(portal_instance['K'])
    d = int(portal_instance['d'])
    cursor = 0
    out: List[Dict[str, Any]] = []
    all_dim_indices: List[int] = []
    block_ids: List[int] = []

    for bi, block in enumerate(schema_blocks, start=1):
        k_block = int(block.get('K', 0))
        if k_block <= 0:
            raise ValueError(f"Invalid _schema_blocks[{bi - 1}].K={k_block}; expected positive integer.")
        if cursor + k_block > K:
            raise ValueError("Invalid _schema_blocks partition: cumulative block K exceeds total K.")

        weight = float(block.get('weight', 1.0))
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Invalid _schema_blocks[{bi - 1}].weight={weight}; expected finite > 0.")

        dim_idx = [int(i) for i in block.get('dim_indices', [])]
        if len(dim_idx) == 0:
            dim_idx = list(range(d))
        if any(i < 0 or i >= d for i in dim_idx):
            raise ValueError(f"Invalid _schema_blocks[{bi - 1}].dim_indices; out of range for d={d}.")
        if len(set(dim_idx)) != len(dim_idx):
            raise ValueError(f"Invalid _schema_blocks[{bi - 1}].dim_indices; duplicate indices.")

        block_id = int(block.get('block_id', bi))
        block_ids.append(block_id)
        all_dim_indices.extend(dim_idx)

        out.append({
            'block_id': block_id,
            'K': k_block,
            'weight': weight,
            'dim_indices': dim_idx,
        })
        cursor += k_block

    if cursor != K:
        raise ValueError(
            f"Invalid _schema_blocks partition: cumulative block K={cursor}, expected K={K}."
        )
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("Invalid _schema_blocks: block IDs must be unique.")
    if sorted(all_dim_indices) != list(range(d)):
        raise ValueError(
            "Invalid _schema_blocks: dim_indices must form an exact, "
            f"non-overlapping partition of global dimensions [0, {d})."
        )

    return out


def _slice_parameters_for_block(P: Dict[str, Any], idx: np.ndarray, d_global: int) -> Dict[str, Any]:
    """Return a block-local parameter view preserving component axis K."""
    k_count = int(np.asarray(P['beta']).reshape(-1).size)
    k_axis = np.arange(k_count, dtype=int)

    out: Dict[str, Any] = {
        'c': np.asarray(P['c'])[:, idx],
        'R': np.asarray(P['R'])[np.ix_(idx, idx, k_axis)],
        'kappa_plus': np.asarray(P['kappa_plus'])[:, idx],
        'kappa_minus': np.asarray(P['kappa_minus'])[:, idx],
        'p_plus': np.asarray(P['p_plus'])[:, idx],
        'p_minus': np.asarray(P['p_minus'])[:, idx],
        'rho_plus': np.asarray(P['rho_plus'])[:, idx],
        'rho_minus': np.asarray(P['rho_minus'])[:, idx],
        'beta': np.asarray(P['beta']),
        'form': np.asarray(P['form']),
        'p_single': np.asarray(P['p_single']),
        'rho_single': np.asarray(P['rho_single']),
    }

    psi_deg = P.get('Psi_deg', None)
    if psi_deg is not None:
        out['Psi_deg'] = np.asarray(psi_deg)[np.ix_(idx, idx, k_axis)]

    for tname, _base_name, tb in iter_transform_blocks(P):
        t_out: Dict[str, Any] = {}
        for key, value in tb.items():
            arr = np.asarray(value)
            if arr.ndim == 2 and arr.shape[1] == d_global:
                t_out[key] = arr[:, idx]
            else:
                t_out[key] = value
        out[tname] = t_out

    return out


def _evaluate_layer3(x: np.ndarray, portal_instance: Dict[str, Any]) -> float:
    """Evaluate using block-local component values then weighted block minima."""
    p = portal_instance['P']
    d = int(portal_instance['d'])
    k_total = int(portal_instance['K'])
    seq_cfg = portal_instance['transform_seq']
    blocks = _normalized_schema_blocks(portal_instance)

    total = 0.0
    cursor = 0
    for block in blocks:
        k_block = int(block['K'])
        weight = float(block['weight'])
        idx = np.asarray(block['dim_indices'], dtype=int)
        x_local = x[idx]
        p_local = _slice_parameters_for_block(p, idx, d)
        portal_local = {'d': int(idx.size)}

        vals = np.zeros(k_block, dtype=float)
        for off, k in enumerate(range(cursor, cursor + k_block)):
            seq_k = seq_for_component(seq_cfg, k, k_total)
            vals[off] = phi_component(x_local, p_local, portal_local, k, seq_k)

        total += weight * float(np.min(vals))
        cursor += k_block

    return float(total)


def fitness(x: np.ndarray, portal_instance: Dict[str, Any]) -> float:
    """
    Evaluate the fitness of a solution using the PORTAL benchmark.

    Args:
        x: Solution vector (d×1 or 1D array)
        portal_instance: PORTAL benchmark instance dictionary

    Returns:
        Fitness value (scalar)
    """
    # Ensure x is a column vector
    x = np.atleast_1d(x).flatten()

    # Check dimension
    if len(x) != portal_instance['d']:
        raise ValueError(f"Solution dimension mismatch. Expected {portal_instance['d']}, got {len(x)}")

    # Check bounds: warn (configurable) but always evaluate without clipping
    bounds_min = portal_instance['bounds_min']
    bounds_max = portal_instance['bounds_max']
    if portal_instance.get('warn_out_of_bounds', True):
        if np.any(x < bounds_min) or np.any(x > bounds_max):
            print(f"Warning: Solution out of bounds [{bounds_min}, {bounds_max}]; "
                  f"evaluating anyway. Set 'warn_out_of_bounds' to False to silence.")

    return _evaluate_layer3(x, portal_instance)


def fitness_batch(
    x: np.ndarray,
    portal_instance: Dict[str, Any],
    evaluation_device: str = None,
) -> np.ndarray:
    """
    Evaluate a batch of candidate solutions.

    The batch fast path preserves the Layer-3 / Family4 semantics used by the
    scalar reference implementation:

        F(x) = sum_b weight_b * min_k f_{b,k}(x[I_b]).

    The torch fast path implements the active PORTAL transformations
    (AP/LS/WM/TI/RP). If torch is unavailable, or an unknown transform is
    encountered, the evaluator falls back to the scalar reference path.
    """
    samples = np.asarray(x, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    elif (
        samples.ndim == 2
        and samples.shape[1] != int(portal_instance['d'])
        and samples.shape[0] == int(portal_instance['d'])
    ):
        samples = samples.reshape(1, -1)
    samples = samples.reshape(samples.shape[0], -1)

    if samples.shape[1] != int(portal_instance['d']):
        raise ValueError(
            f"Solution dimension mismatch. Expected {portal_instance['d']}, "
            f"got {samples.shape[1]}"
        )

    bounds_min = portal_instance['bounds_min']
    bounds_max = portal_instance['bounds_max']
    if portal_instance.get('warn_out_of_bounds', True):
        if np.any(samples < bounds_min) or np.any(samples > bounds_max):
            print(f"Warning: Solution out of bounds [{bounds_min}, {bounds_max}]; "
                  f"evaluating anyway. Set 'warn_out_of_bounds' to False to silence.")

    device_policy = _normalise_evaluation_device(evaluation_device)
    supports_no_transform = _supports_no_transform_batch(portal_instance)
    if supports_no_transform and device_policy in {'auto', 'numpy'}:
        values = _evaluate_layer3_batch_numpy(samples, portal_instance)
        portal_instance['_last_evaluation_backend'] = 'numpy:cpu'
        return values

    failure_notes = []
    if _supports_torch_transform_batch(portal_instance):
        for torch, device, label in _select_torch_devices(portal_instance, device_policy):
            try:
                values = _evaluate_layer3_batch_torch(samples, portal_instance, torch, device)
                portal_instance['_last_evaluation_backend'] = label
                return values
            except Exception as exc:
                failure_notes.append(f'{label} failed: {type(exc).__name__}')
    else:
        failure_notes.append('unsupported transform')

    if supports_no_transform:
        values = _evaluate_layer3_batch_numpy(samples, portal_instance)
        if failure_notes:
            portal_instance['_last_evaluation_backend'] = (
                f"numpy:cpu ({'; '.join(failure_notes)})"
            )
        else:
            portal_instance['_last_evaluation_backend'] = 'numpy:cpu'
        return values

    if failure_notes:
        suffix = f" ({'; '.join(failure_notes)})"
    else:
        suffix = ''
    portal_instance['_last_evaluation_backend'] = f'numpy:scalar-fallback(transforms){suffix}'
    return np.asarray([fitness(sample, portal_instance) for sample in samples], dtype=float)


def _supports_no_transform_batch(portal_instance: Dict[str, Any]) -> bool:
    """Return True when every component has no transform sequence."""
    seq_cfg = portal_instance['transform_seq']
    K = int(portal_instance['K'])
    for k in range(K):
        seq_k = seq_for_component(seq_cfg, k, K)
        if any(str(item).lower() != 'none' for item in seq_k):
            return False
    return True


def _evaluation_transform_type(name: str) -> str:
    """Resolve canonical occurrence references without interpreting aliases."""
    base, separator, _occurrence = name.partition('#')
    if (separator and name == name.strip()
            and base in _TORCH_BATCH_TRANSFORMS and base != 'none'):
        return transform_type(name)
    return name


def _supports_torch_transform_batch(portal_instance: Dict[str, Any]) -> bool:
    """Return True when every transform has a torch batch implementation."""
    seq_cfg = portal_instance['transform_seq']
    K = int(portal_instance['K'])
    for k in range(K):
        seq_k = seq_for_component(seq_cfg, k, K)
        for item in seq_k:
            if _evaluation_transform_type(str(item).lower()) not in _TORCH_BATCH_TRANSFORMS:
                return False
    return True


def _normalise_evaluation_device(device: str = None) -> str:
    value = device
    if value is None:
        value = os.environ.get('PORTAL_EVALUATION_DEVICE', 'auto')
    value = str(value).strip().lower()
    aliases = {
        '': 'auto',
        'default': 'auto',
        'gpu': 'cuda',
        'nvidia': 'cuda',
        'metal': 'mps',
        'apple': 'mps',
    }
    value = aliases.get(value, value)
    if value not in _EVALUATION_DEVICES:
        raise ValueError(
            f"Unknown PORTAL evaluation device {device!r}. "
            f"Use one of {sorted(_EVALUATION_DEVICES)}."
        )
    return value


def _device_chain(policy: str) -> list:
    if policy == 'cuda':
        return ['cuda', 'cpu', 'mps']
    if policy == 'mps':
        return ['mps', 'cpu']
    if policy == 'cpu':
        return ['cpu', 'mps']
    if policy == 'numpy':
        return ['numpy']
    return ['cuda', 'cpu', 'mps']


def _select_torch_devices(
    portal_instance: Dict[str, Any],
    evaluation_device: str = None,
):
    policy = _normalise_evaluation_device(evaluation_device)
    if policy == 'numpy':
        portal_instance['_last_evaluation_backend'] = 'numpy:cpu'
        return []

    try:
        import torch
    except Exception:
        portal_instance['_last_evaluation_backend'] = 'numpy:cpu (torch unavailable)'
        return []

    devices = []
    for candidate in _device_chain(policy):
        if candidate == 'cuda' and torch.cuda.is_available():
            devices.append((torch, torch.device('cuda'), 'torch:cuda'))
        if candidate == 'mps':
            has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if has_mps:
                devices.append((torch, torch.device('mps'), 'torch:mps'))
        if candidate == 'cpu':
            devices.append((torch, torch.device('cpu'), 'torch:cpu'))

    if not devices:
        portal_instance['_last_evaluation_backend'] = 'numpy:cpu (no usable torch device)'
    return devices


def _torch_dtype(torch, device):
    if str(device).startswith('mps'):
        return torch.float32
    return torch.float64


def _torch_cache(portal_instance: Dict[str, Any], torch, device):
    dtype = _torch_dtype(torch, device)
    key = (str(device), str(dtype))
    cache = portal_instance.setdefault('_torch_cache', {})
    if key in cache:
        return cache[key], dtype

    P = portal_instance['P']
    tensor_cache = {
        'c': torch.as_tensor(np.ascontiguousarray(P['c']), dtype=dtype, device=device),
        'R': torch.as_tensor(np.ascontiguousarray(P['R']), dtype=dtype, device=device),
        'kappa_plus': torch.as_tensor(np.ascontiguousarray(P['kappa_plus']), dtype=dtype, device=device),
        'kappa_minus': torch.as_tensor(np.ascontiguousarray(P['kappa_minus']), dtype=dtype, device=device),
        'p_plus': torch.as_tensor(np.ascontiguousarray(P['p_plus']), dtype=dtype, device=device),
        'p_minus': torch.as_tensor(np.ascontiguousarray(P['p_minus']), dtype=dtype, device=device),
        'rho_plus': torch.as_tensor(np.ascontiguousarray(P['rho_plus']), dtype=dtype, device=device),
        'rho_minus': torch.as_tensor(np.ascontiguousarray(P['rho_minus']), dtype=dtype, device=device),
        'beta': torch.as_tensor(np.ascontiguousarray(P['beta']), dtype=dtype, device=device),
        'rho_single': torch.as_tensor(np.ascontiguousarray(P['rho_single']), dtype=dtype, device=device),
        'p_single': torch.as_tensor(np.ascontiguousarray(P['p_single']), dtype=dtype, device=device),
        'form': np.asarray(P['form'], dtype=int),
    }
    for tname, _base_name, tb in iter_transform_blocks(P):
        tensor_cache[tname] = _torch_transform_cache(tb, torch, dtype, device)
    cache[key] = tensor_cache
    return tensor_cache, dtype


def _torch_transform_cache(transform_block: Dict[str, Any], torch, dtype, device):
    out = {}
    for key, value in transform_block.items():
        arr = np.asarray(value)
        if arr.dtype == object:
            out[key] = value
        else:
            out[key] = torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype, device=device)
    return out


def _evaluate_layer3_batch_numpy(
    samples: np.ndarray,
    portal_instance: Dict[str, Any],
) -> np.ndarray:
    p = portal_instance['P']
    d = int(portal_instance['d'])
    blocks = _normalized_schema_blocks(portal_instance)

    total = np.zeros(samples.shape[0], dtype=float)
    cursor = 0
    for block in blocks:
        k_block = int(block['K'])
        weight = float(block['weight'])
        idx = np.asarray(block['dim_indices'], dtype=int)

        block_values = []
        for k in range(cursor, cursor + k_block):
            block_values.append(
                _phi_component_batch_no_transform_numpy(samples, p, idx, d, k)
            )
        total += weight * np.min(np.vstack(block_values), axis=0)
        cursor += k_block
    return total.astype(float, copy=False)


def _phi_component_batch_no_transform_numpy(
    samples: np.ndarray,
    P: Dict[str, Any],
    idx: np.ndarray,
    d_global: int,
    k: int,
) -> np.ndarray:
    x_local = samples[:, idx]
    ck = np.asarray(P['c'])[k, idx].reshape(1, -1)
    Rk = np.asarray(P['R'])[np.ix_(idx, idx, [k])][:, :, 0]
    z = (x_local - ck) @ Rk.T

    form = int(np.asarray(P['form']).reshape(-1)[k])
    if form == 1:
        positive = z >= 0
        h = np.where(positive, np.asarray(P['kappa_plus'])[k, idx],
                     np.asarray(P['kappa_minus'])[k, idx])
        p_i = np.where(positive, np.asarray(P['p_plus'])[k, idx],
                       np.asarray(P['p_minus'])[k, idx])
        rho_i = np.where(positive, np.asarray(P['rho_plus'])[k, idx],
                         np.asarray(P['rho_minus'])[k, idx])
        acc = np.sum(rho_i * h * (np.abs(z) ** (2 * p_i)), axis=1)
        return np.asarray(P['beta'])[k] + acc

    p_single = _p_single_at(P, k)
    positive = z >= 0
    h = np.where(positive, np.asarray(P['kappa_plus'])[k, idx],
                 np.asarray(P['kappa_minus'])[k, idx])
    qsum = np.sum(
        h ** (1 / max(p_single, np.finfo(float).eps)) * (np.abs(z) ** 2),
        axis=1,
    )
    return np.asarray(P['beta'])[k] + np.asarray(P['rho_single'])[k] * (qsum ** p_single)


def _evaluate_layer3_batch_torch(
    samples: np.ndarray,
    portal_instance: Dict[str, Any],
    torch,
    device,
) -> np.ndarray:
    P_t, dtype = _torch_cache(portal_instance, torch, device)
    blocks = _normalized_schema_blocks(portal_instance)

    with torch.inference_mode():
        sample_t = torch.as_tensor(
            np.ascontiguousarray(samples),
            dtype=dtype,
            device=device,
        )
        total = torch.zeros(sample_t.shape[0], dtype=dtype, device=device)
        cursor = 0
        for block in blocks:
            k_block = int(block['K'])
            weight = float(block['weight'])
            idx = torch.as_tensor(block['dim_indices'], dtype=torch.long, device=device)

            block_values = []
            for k in range(cursor, cursor + k_block):
                seq_k = seq_for_component(portal_instance['transform_seq'], k, int(portal_instance['K']))
                block_values.append(
                    _phi_component_batch_torch(sample_t, P_t, idx, dtype, torch, k, seq_k)
                )
            total = total + weight * torch.min(torch.stack(block_values, dim=0), dim=0).values
            cursor += k_block

        return total.detach().cpu().numpy().astype(float)


def _phi_component_batch_torch(sample_t, P_t, idx, dtype, torch, k: int, seq_k: List[str]):
    x_local = sample_t.index_select(1, idx)
    ck = P_t['c'][k].index_select(0, idx).reshape(1, -1)
    Rk = P_t['R'][:, :, k].index_select(0, idx).index_select(1, idx)
    z = (x_local - ck) @ Rk.T
    z = _apply_transform_seq_batch_torch(z, P_t, idx, dtype, torch, k, seq_k)

    if int(P_t['form'][k]) == 1:
        positive = z >= 0
        h = torch.where(
            positive,
            P_t['kappa_plus'][k].index_select(0, idx),
            P_t['kappa_minus'][k].index_select(0, idx),
        )
        p_i = torch.where(
            positive,
            P_t['p_plus'][k].index_select(0, idx),
            P_t['p_minus'][k].index_select(0, idx),
        )
        rho_i = torch.where(
            positive,
            P_t['rho_plus'][k].index_select(0, idx),
            P_t['rho_minus'][k].index_select(0, idx),
        )
        acc = torch.sum(rho_i * h * torch.pow(torch.abs(z), 2 * p_i), dim=1)
        return P_t['beta'][k] + acc

    p_single = P_t['p_single'][k].reshape(-1)[0]
    positive = z >= 0
    h = torch.where(
        positive,
        P_t['kappa_plus'][k].index_select(0, idx),
        P_t['kappa_minus'][k].index_select(0, idx),
    )
    qsum = torch.sum(
        torch.pow(h, 1 / torch.clamp(p_single, min=torch.finfo(dtype).eps))
        * (torch.abs(z) ** 2),
        dim=1,
    )
    return P_t['beta'][k] + P_t['rho_single'][k] * (qsum ** p_single)


def _phi_component_batch_no_transform_torch(sample_t, P_t, idx, dtype, torch, k: int):
    return _phi_component_batch_torch(sample_t, P_t, idx, dtype, torch, k, [])


def _apply_transform_seq_batch_torch(a, P_t, idx, dtype, torch, k: int, seq_k: List[str]):
    z = a
    for transform_name in seq_k:
        z = _apply_transform_once_batch_torch(z, P_t, idx, dtype, torch, k, transform_name)
    return z


def _transform_row(P_t, tname: str, field: str, idx, k: int):
    return P_t[tname][field][k].index_select(0, idx).reshape(1, -1)


def _transform_scalar(P_t, tname: str, field: str, k: int, default=None):
    tb = P_t.get(tname, {})
    value = tb.get(field)
    if value is None:
        return default
    if hasattr(value, 'ndim') and value.ndim > 0:
        flat = value.reshape(-1)
        return flat[k] if flat.numel() > 1 else flat[0]
    return value


def _apply_transform_once_batch_torch(a, P_t, idx, dtype, torch, k: int, name: str):
    reference = str(name).lower()
    name_lower = _evaluation_transform_type(reference)
    if name_lower == 'none':
        return a
    if name_lower in _TORCH_BATCH_TRANSFORMS:
        P_t = parameters_for_transform(P_t, reference)

    if name_lower == 'additive_periodic':
        return _additive_periodic_batch_torch(a, P_t, idx, torch, k)
    if name_lower == 'log_sinusoidal':
        return _log_sinusoidal_batch_torch(a, P_t, idx, dtype, torch, k)
    if name_lower == 'wavelet_mod':
        return _wavelet_mod_batch_torch(a, P_t, idx, torch, k)
    if name_lower == 'tensor_interference':
        return _tensor_interference_batch_torch(a, P_t, idx, torch, k)
    if name_lower == 'radial_polytrig':
        return _radial_polytrig_batch_torch(a, P_t, dtype, torch, k)

    raise ValueError(f"Unsupported transform for torch batch evaluation: {name!r}")


def _additive_periodic_batch_torch(a, P_t, idx, torch, k: int):
    tname = 'T_additive_periodic'
    positive = a >= 0
    mu = torch.where(
        positive,
        _transform_row(P_t, tname, 'mu_plus', idx, k),
        _transform_row(P_t, tname, 'mu_minus', idx, k),
    )
    gam = torch.where(
        positive,
        _transform_row(P_t, tname, 'gamma_plus', idx, k),
        _transform_row(P_t, tname, 'gamma_minus', idx, k),
    )
    om = torch.where(
        positive,
        _transform_row(P_t, tname, 'omega_plus', idx, k),
        _transform_row(P_t, tname, 'omega_minus', idx, k),
    )
    abs_a = torch.abs(a)
    g = a * (1 - torch.exp(-gam * abs_a)) * torch.sin(om * abs_a)
    return a + mu * g


def _log_sinusoidal_batch_torch(a, P_t, idx, dtype, torch, k: int):
    tname = 'T_log_sinusoidal'
    positive = a > 0
    mu = torch.where(
        positive,
        _transform_row(P_t, tname, 'mu_plus', idx, k),
        _transform_row(P_t, tname, 'mu_minus', idx, k),
    )
    o1 = torch.where(
        positive,
        _transform_row(P_t, tname, 'omega1_plus', idx, k),
        _transform_row(P_t, tname, 'omega1_minus', idx, k),
    )
    o2 = torch.where(
        positive,
        _transform_row(P_t, tname, 'omega2_plus', idx, k),
        _transform_row(P_t, tname, 'omega2_minus', idx, k),
    )
    sigma_k = _transform_scalar(P_t, tname, 'sigma', k)
    if sigma_k is None:
        sigma_k = a.new_tensor(0.1)
    sigma2 = torch.clamp(sigma_k * sigma_k, min=torch.finfo(dtype).tiny)
    abs_a = torch.abs(a)
    L = torch.log(abs_a + 1e-12)
    gate = 1.0 - torch.exp(-(a * a) / sigma2)
    modulation = gate * mu * (torch.sin(o1 * L) + torch.sin(o2 * L))
    # Regularize only the log phases; retain the coordinate as the prefactor.
    return a * torch.exp(modulation)


def _wavelet_mod_batch_torch(a, P_t, idx, torch, k: int):
    tname = 'T_wavelet_mod'
    positive = a >= 0
    mu = torch.where(
        positive,
        _transform_row(P_t, tname, 'mu_plus', idx, k),
        _transform_row(P_t, tname, 'mu_minus', idx, k),
    )
    omg = torch.where(
        positive,
        _transform_row(P_t, tname, 'omega_plus', idx, k),
        _transform_row(P_t, tname, 'omega_minus', idx, k),
    )
    eta = _transform_scalar(P_t, tname, 'eta', k)
    epss = _transform_scalar(P_t, tname, 'eps', k)
    t = torch.abs(a)
    active = (t >= epss) & (omg > 0) & (mu != 0)
    eps_tensor = epss if hasattr(epss, 'reshape') else a.new_tensor(float(epss))
    L = eta / torch.maximum(omg, eps_tensor)
    ratio = t / L
    env = (ratio ** 2) * torch.exp(-(ratio ** 2))
    phi = env * torch.sin(omg * t - np.pi / 2)
    candidate = a + mu * phi * (a / (t + epss))
    return torch.where(active, candidate, a)


def _tensor_interference_batch_torch(a, P_t, idx, torch, k: int):
    tname = 'T_tensor_interference'
    positive = a >= 0
    mu = torch.where(
        positive,
        _transform_row(P_t, tname, 'mu_plus', idx, k),
        _transform_row(P_t, tname, 'mu_minus', idx, k),
    )
    om = torch.where(
        positive,
        _transform_row(P_t, tname, 'omega_plus', idx, k),
        _transform_row(P_t, tname, 'omega_minus', idx, k),
    )
    sin2 = torch.sin(om * a) ** 2
    n_samples, d = a.shape
    phi_cols = []
    for i in range(d):
        phi_i = torch.ones(n_samples, dtype=a.dtype, device=a.device)
        for j in range(d):
            if j != i:
                phi_i = phi_i * sin2[:, j]
        phi_cols.append(phi_i)
    phi_even = torch.stack(phi_cols, dim=1)
    scaled_mu = mu * (2 ** ((d - 1) / 2.0))
    candidate = a + scaled_mu * torch.sin(om * a) * phi_even
    active = (mu != 0) & (om > 0)
    return torch.where(active, candidate, a)


def _radial_polytrig_batch_torch(a, P_t, dtype, torch, k: int):
    tname = 'T_radial_polytrig'
    mu = _transform_scalar(P_t, tname, 'mu', k)
    p = _transform_scalar(P_t, tname, 'p', k)
    q = _transform_scalar(P_t, tname, 'q', k)
    om = _transform_scalar(P_t, tname, 'omega', k)
    epss = _transform_scalar(P_t, tname, 'eps', k)
    r = torch.linalg.norm(a, dim=1, keepdim=True)
    active = r >= epss
    phi = torch.pow(r, p) * torch.sin(om * torch.pow(r, q))
    candidate = a + mu * phi * (a / (r + epss))
    return torch.where(active, candidate, a)


def phi_component(x: np.ndarray, P: Dict[str, Any], portal: Dict[str, Any],
                  k: int, seq_k: List[str]) -> float:
    """
    Evaluate a single component.

    Args:
        x: Solution vector
        P: Parameters dictionary
        portal: PORTAL instance
        k: Component index
        seq_k: Transform sequence for this component

    Returns:
        Component fitness value
    """
    d = portal['d']
    ck = P['c'][k, :].reshape(-1, 1)

    # Internal coordinate transformation
    x_col = x.reshape(-1, 1)
    a = P['R'][:, :, k] @ (x_col - ck)

    # Apply transformation sequence
    z = apply_transform_seq(a, P, seq_k, k)

    # Compute fitness based on form
    form = P['form'][k]

    if form == 1:  # Form A
        acc = 0.0
        for i in range(d):
            if z[i, 0] >= 0:
                h = P['kappa_plus'][k, i]
                p_i = P['p_plus'][k, i]
                rho_i = P['rho_plus'][k, i]
            else:
                h = P['kappa_minus'][k, i]
                p_i = P['p_minus'][k, i]
                rho_i = P['rho_minus'][k, i]
            acc += rho_i * h * (abs(z[i, 0]) ** (2 * p_i))
        return P['beta'][k] + acc

    else:  # Form B (form == 2)
        qsum = 0.0
        p_single = _p_single_at(P, k)
        for i in range(d):
            if z[i, 0] >= 0:
                h = P['kappa_plus'][k, i]
            else:
                h = P['kappa_minus'][k, i]
            qsum += h ** (1 / max(p_single, np.finfo(float).eps)) * (abs(z[i, 0]) ** 2)
        return P['beta'][k] + P['rho_single'][k] * (qsum ** p_single)


def apply_transform_seq(a: np.ndarray, P: Dict[str, Any], seq: List[str], k: int) -> np.ndarray:
    """
    Apply a sequence of transformations.

    Args:
        a: Input vector
        P: Parameters dictionary
        seq: List of transform names
        k: Component index

    Returns:
        Transformed vector
    """
    z = a.copy()
    for transform_name in seq:
        z = apply_transform_once(z, P, transform_name, k)
    return z


def apply_transform_once(a: np.ndarray, P: Dict[str, Any], name: str, k: int) -> np.ndarray:
    """
    Apply a single transformation.

    Args:
        a: Input vector (d×1)
        P: Parameters dictionary
        name: Transform name
        k: Component index

    Returns:
        Transformed vector (d×1)
    """
    d = a.shape[0]
    z = a.copy()
    epslog = 1e-12

    reference = name.lower()
    name_lower = _evaluation_transform_type(reference)
    if name_lower != 'none' and name_lower in _TORCH_BATCH_TRANSFORMS:
        P = parameters_for_transform(P, reference)

    if name_lower == 'none':
        return z

    elif name_lower == 'additive_periodic':
        for i in range(d):
            ai = a[i, 0]
            if ai >= 0:
                mu = P['T_additive_periodic']['mu_plus'][k, i]
                gam = P['T_additive_periodic']['gamma_plus'][k, i]
                om = P['T_additive_periodic']['omega_plus'][k, i]
            else:
                mu = P['T_additive_periodic']['mu_minus'][k, i]
                gam = P['T_additive_periodic']['gamma_minus'][k, i]
                om = P['T_additive_periodic']['omega_minus'][k, i]
            g = ai * (1 - np.exp(-gam * abs(ai))) * np.sin(om * abs(ai))
            z[i, 0] = ai + mu * g

    elif name_lower == 'log_sinusoidal':
        # Per-component smoothing scale for the origin gate g(a) = 1 - exp(-a^2 / sigma^2),
        # which guarantees T(0)=0 and grad T(0)=I exactly while preserving the
        # large-|a| log-sinusoidal behaviour. Use sigma=0.1 if absent.
        sigma_arr = P['T_log_sinusoidal'].get('sigma', None)
        if sigma_arr is None:
            sigma_k = 0.1
        else:
            sigma_k = float(np.atleast_1d(np.asarray(sigma_arr)).flatten()[k])
        sigma2 = max(sigma_k * sigma_k, np.finfo(float).tiny)
        for i in range(d):
            ai = a[i, 0]
            if ai == 0:
                z[i, 0] = 0
                continue

            abs_ai = abs(ai)
            L = np.log(abs_ai + epslog)
            if ai > 0:
                mu = P['T_log_sinusoidal']['mu_plus'][k, i]
                o1 = P['T_log_sinusoidal']['omega1_plus'][k, i]
                o2 = P['T_log_sinusoidal']['omega2_plus'][k, i]
            else:
                mu = P['T_log_sinusoidal']['mu_minus'][k, i]
                o1 = P['T_log_sinusoidal']['omega1_minus'][k, i]
                o2 = P['T_log_sinusoidal']['omega2_minus'][k, i]
            gate = 1.0 - np.exp(-(ai * ai) / sigma2)
            modulation = gate * mu * (np.sin(o1 * L) + np.sin(o2 * L))
            # Regularize only the log phases; retain the coordinate as the prefactor.
            z[i, 0] = ai * np.exp(modulation)

    elif name_lower == 'wavelet_mod':
        eta = P['T_wavelet_mod']['eta'][k]
        epss = P['T_wavelet_mod']['eps']
        for i in range(d):
            ai = a[i, 0]
            t = abs(ai)
            if t < epss:
                z[i, 0] = ai
                continue

            if ai >= 0:
                mu = P['T_wavelet_mod']['mu_plus'][k, i]
                omg = P['T_wavelet_mod']['omega_plus'][k, i]
            else:
                mu = P['T_wavelet_mod']['mu_minus'][k, i]
                omg = P['T_wavelet_mod']['omega_minus'][k, i]

            if omg <= 0 or mu == 0:
                z[i, 0] = ai
                continue

            # Envelope length is always derived as eta / omega (paper convention).
            L = eta / max(omg, epss)

            env = (t / L) ** 2 * np.exp(-(t / L) ** 2)
            phi = env * np.sin(omg * t - np.pi / 2)
            z[i, 0] = ai + mu * phi * (ai / (t + epss))

    elif name_lower == 'tensor_interference':
        for i in range(d):
            ai = a[i, 0]
            if ai >= 0:
                mu_i = P['T_tensor_interference']['mu_plus'][k, i]
                om_i = P['T_tensor_interference']['omega_plus'][k, i]
            else:
                mu_i = P['T_tensor_interference']['mu_minus'][k, i]
                om_i = P['T_tensor_interference']['omega_minus'][k, i]

            if mu_i == 0 or om_i <= 0:
                z[i, 0] = ai
                continue

            Phi_even = 1.0
            for j in range(d):
                if j == i:
                    continue
                aj = a[j, 0]
                if aj >= 0:
                    om_j = P['T_tensor_interference']['omega_plus'][k, j]
                else:
                    om_j = P['T_tensor_interference']['omega_minus'][k, j]
                Phi_even *= np.sin(om_j * aj) ** 2

            # Prescribed dimension factor. It does not keep the typical
            # interference magnitude constant as the dimension changes.
            scaled_mu = mu_i * (2 ** ((d - 1) / 2.0))
            z[i, 0] = ai + scaled_mu * np.sin(om_i * ai) * Phi_even

    elif name_lower == 'radial_polytrig':
        mu = P['T_radial_polytrig']['mu'][k]
        p = P['T_radial_polytrig']['p'][k]
        q = P['T_radial_polytrig']['q'][k]
        om = P['T_radial_polytrig']['omega'][k]
        epss = P['T_radial_polytrig']['eps']
        r = np.linalg.norm(a)
        if r < epss:
            return a
        phi = r ** p * np.sin(om * r ** q)
        z = a + mu * phi * (a / (r + epss))

    return z


def seq_for_component(seq_cfg: List[List[str]], k: int, K: int) -> List[str]:
    """
    Get transformation sequence for a component.

    Args:
        seq_cfg: List of transformation sequences (one per component)
        k: Component index
        K: Total number of components

    Returns:
        List of transform names for this component
    """
    if not isinstance(seq_cfg, list) or len(seq_cfg) != K:
        raise ValueError(f"transform_seq must be a list of length K={K}; each entry a list of transform names.")

    item = seq_cfg[k]
    if item is None or len(item) == 0:
        return []
    elif isinstance(item, list):
        return [str(t) for t in item]
    elif isinstance(item, str):
        return [item]
    else:
        raise ValueError(f"transform_seq[{k}] must be list, string, or empty.")
