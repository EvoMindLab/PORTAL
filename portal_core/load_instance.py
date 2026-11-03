"""
Author: Delaram Yazdani, Danial Yazdani, Mai Peng
Email: delaram.yazdani@yahoo.com
       danial.yazdani@gmail.com
       pengmai1998@gmail.com
Title: Load PORTAL Instance from JSON (unified blocks/components schema)

Description
-----------
Loads a PORTAL benchmark instance from a JSON file written under the unified
hierarchical schema (see ``save_instance.py``), optionally inside a supported
versioned file container, and reconstitutes the flat
runtime layout that ``fitness.py``, ``realization.py``, and the builders
expect:

    {
      'Seed': int, 'd': int, 'K': int,
      'bounds_min': ..., 'bounds_max': ...,
      'baseline': [...], 'transform_seq': [[...], ...],
      'warn_out_of_bounds': bool,
      'Rng': np.random.RandomState,
      'P': { c, beta, kappa_+/-, p_*, rho_*, Delta, r_ref, form,
             neutralize_enable, Psi_deg, R, T_additive_periodic, ... },
      '_suite': { ... }
    }

For Family 1 / 2 each anchor has ``B = 1`` block with ``K = 1`` component.
For ``B > 1``, blocks are flattened into a single runtime component list
(sum of all block K values), while preserving the original partition in
``portal['_schema_blocks']`` so save can round-trip the same block layout.
Each block may carry a scalar ``weight``; when missing, ``weight=1.0`` is
assumed.

In the unified schema, transformation order is component-level via
``components[k]['transform_seq']``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

try:
    from .instance_format import decode_instance_file
except ImportError:  # Direct-script compatibility.
    from instance_format import decode_instance_file  # type: ignore

try:
    from .transform_registry import (
        TRANSFORM_BLOCK_SCALARS as _T_BLOCK_SCALARS,
        TRANSFORM_PER_COMPONENT_SCALARS as _T_PER_COMPONENT_SCALARS,
        TRANSFORM_PER_DIM_FIELDS as _T_PER_DIM_FIELDS,
        iter_transform_blocks,
    )
except ImportError:
    from transform_registry import (  # type: ignore
        TRANSFORM_BLOCK_SCALARS as _T_BLOCK_SCALARS,
        TRANSFORM_PER_COMPONENT_SCALARS as _T_PER_COMPONENT_SCALARS,
        TRANSFORM_PER_DIM_FIELDS as _T_PER_DIM_FIELDS,
        iter_transform_blocks,
    )


def load_instance(filename: str) -> Dict[str, Any]:
    """Load a PORTAL benchmark instance from a JSON file."""
    file_path = Path(filename)
    if not file_path.exists():
        script_dir = Path(__file__).parent
        instances_dir = script_dir / "Instances"
        file_path = instances_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Instance file not found: {filename}")

    try:
        with open(file_path, "r", encoding="utf-8") as fid:
            raw = json.load(fid)
        print(f"Instance loaded successfully from:\n  {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load instance: {str(e)}")

    portal = load_instance_payload(raw)

    print(f"Loaded PORTAL Instance:")
    print(f"  Dimension: {portal['d']}")
    print(f"  Components: {portal['K']}")
    print(f"  Seed: {portal['Seed']}")
    if "baseline" in portal:
        print(f"  Baselines: {' '.join(portal['baseline'])}")
    if "transform_seq" in portal:
        print(f"  Transformations: ", end="")
        for k in range(portal["K"]):
            transforms = portal["transform_seq"][k]
            if isinstance(transforms, list):
                print("{", end="")
                print("->".join([f"'{t}'" for t in transforms]), end="")
                print("}", end="")
        print()

    return portal


def load_instance_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Build a runtime instance directly from a unified JSON payload.

    This is the in-memory counterpart of :func:`load_instance`. Studio code
    uses it after converting a transient composition to the canonical unified
    payload, so file-loaded and newly assembled instances share one runtime.
    """
    raw = decode_instance_file(raw)
    if not isinstance(raw, dict) or "blocks" not in raw:
        raise ValueError(
            "Unified schema required: missing top-level 'blocks' array."
        )
    if bool(raw.get("composite", False)):
        raise ValueError(
            "Legacy composite envelopes are not unified PORTAL instances."
        )
    portal = _from_unified_schema(raw)

    if "Seed" in portal:
        portal["Rng"] = np.random.RandomState(int(portal["Seed"]))
    else:
        print("Warning: Seed not found in loaded instance. Creating new RNG.")
        portal["Rng"] = np.random.RandomState(1234)

    portal["P"] = fix_parameter_dimensions(portal["P"], portal["K"], portal["d"])

    return portal


# ---------------------------------------------------------------------------
# Unified-schema -> flat-runtime conversion
# ---------------------------------------------------------------------------
def _validate_component_parameters(
    component: Dict[str, Any], block_number: int, component_number: int
) -> None:
    """Validate active shape/scaling inputs and enabled reference controls."""
    location = f"Block {block_number}, component {component_number}"
    fields = ['kappa_plus', 'kappa_minus']
    fields += ['p_plus', 'p_minus'] if component.get('form', 1) == 1 else ['p_single']
    fields += ['rho_plus', 'rho_minus'] if component.get('form', 1) == 1 else ['rho_single']
    for field in fields:
        try:
            values = np.asarray(component[field], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{location}: {field} must contain finite values > 0.") from exc
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{location}: {field} must contain finite values > 0.")
    if not bool(component.get("neutralize_enable", True)):
        return

    for field in ("Delta", "r_ref"):
        value = component.get(field)
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{location} enables neutralization but {field} is missing "
                "or non-numeric; expected a finite value greater than zero."
            ) from exc
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(
                f"{location} enables neutralization but has invalid "
                f"{field}={value!r}; expected a finite value greater than zero."
            )


def _from_unified_schema(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a unified-schema JSON dict into a flat runtime PORTAL instance."""
    d = int(raw["d"])
    blocks = raw.get("blocks") or []
    if not blocks:
        raise ValueError("Unified-schema JSON has no 'blocks' array.")
    if len(blocks) != int(raw.get("B", len(blocks))):
        raise ValueError("Mismatch between top-level 'B' and len(blocks).")

    block_dims_top = raw.get("block_dims")
    if block_dims_top is not None:
        if not isinstance(block_dims_top, list) or len(block_dims_top) != len(blocks):
            raise ValueError("Top-level 'block_dims' must be a list with length B.")
        block_dims_top = [int(x) for x in block_dims_top]
        if any(x <= 0 for x in block_dims_top):
            raise ValueError("Top-level 'block_dims' must contain positive integers.")
        if sum(block_dims_top) != d:
            raise ValueError(f"Top-level block_dims sum {sum(block_dims_top)} does not match d={d}.")

    suite = raw.get("_suite", {}) or {}
    suite_groups = suite.get("dimension_groups_one_based")

    component_records: List[Dict[str, Any]] = []
    baseline_flat: List[str] = []
    transform_seq_flat: List[List[str]] = []
    schema_blocks: List[Dict[str, Any]] = []
    all_dim_indices: List[int] = []
    block_ids: List[int] = []
    contiguous_cursor = 0

    for b_idx, block in enumerate(blocks):
        components: List[Dict[str, Any]] = list(block.get("components") or [])
        K_block = int(block.get("K", len(components)))
        if K_block != len(components):
            raise ValueError(
                f"Block {b_idx + 1} has K={K_block} but components list has {len(components)} entries."
            )
        if K_block <= 0:
            raise ValueError(f"Block {b_idx + 1} has non-positive K={K_block}.")

        block_id = int(block.get("block_id", b_idx + 1))
        block_ids.append(block_id)
        weight = float(block.get("weight", 1.0))
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Block {b_idx + 1} has invalid weight={weight}; expected finite > 0.")

        block_d = int(block.get("d", block_dims_top[b_idx] if block_dims_top is not None else d))
        if block_d <= 0:
            raise ValueError(f"Block {b_idx + 1} has non-positive d={block_d}.")
        if block_dims_top is not None and block_d != int(block_dims_top[b_idx]):
            raise ValueError(
                f"Block {b_idx + 1} has d={block_d}, expected {int(block_dims_top[b_idx])} from top-level block_dims."
            )

        dim_one_based = block.get("dim_indices_one_based")
        if dim_one_based is None and isinstance(suite_groups, list) and b_idx < len(suite_groups):
            dim_one_based = suite_groups[b_idx]
        if dim_one_based is None:
            if len(blocks) == 1 and block_d == d:
                dim_one_based = list(range(1, d + 1))
            else:
                start = contiguous_cursor + 1
                stop = contiguous_cursor + block_d
                if stop > d:
                    raise ValueError(
                        f"Block {b_idx + 1} cannot infer dim_indices_one_based: "
                        f"cursor overflow for d={d}, block_d={block_d}."
                    )
                dim_one_based = list(range(start, stop + 1))

        dim_idx = _validate_dim_indices(dim_one_based, d, block_d, b_idx)
        all_dim_indices.extend(dim_idx)
        contiguous_cursor += block_d

        schema_blocks.append({
            "block_id": block_id,
            "K": K_block,
            "weight": weight,
            "d": block_d,
            "dim_indices": dim_idx,
        })

        for c_idx, comp in enumerate(components):
            if not isinstance(comp, dict):
                raise ValueError(
                    f"Block {b_idx + 1}, component {c_idx + 1} must be an object."
                )
            _validate_component_parameters(comp, b_idx + 1, c_idx + 1)
            component_records.append({
                "component": comp,
                "block_d": block_d,
                "dim_idx": dim_idx,
            })
        baseline_flat.extend(_normalize_block_baseline(block.get("baseline"), components, K_block))
        transform_seq_flat.extend(_component_transform_seqs(components, K_block))

    if len(set(block_ids)) != len(block_ids):
        raise ValueError("Block IDs must be unique within a unified instance.")
    if sorted(all_dim_indices) != list(range(d)):
        raise ValueError(
            "Block dim_indices_one_based must form an exact, non-overlapping "
            f"partition of global dimensions [1, {d}]."
        )

    K = len(component_records)
    if K == 0:
        raise ValueError("Unified-schema JSON has no components across all blocks.")

    P = _flatten_components(component_records, d, K)

    portal: Dict[str, Any] = {
        "Seed":               int(raw["Seed"]),
        "d":                  d,
        "K":                  K,
        "bounds_min":         raw["bounds_min"],
        "bounds_max":         raw["bounds_max"],
        "baseline":           baseline_flat,
        "transform_seq":      transform_seq_flat,
        "warn_out_of_bounds": bool(raw.get("warn_out_of_bounds", False)),
        "P":                  P,
        "_suite":             raw.get("_suite", {}) or {},
        "_schema_blocks":     schema_blocks,
    }
    identity = {
        key: raw[key]
        for key in ("family", "recipe")
        if raw.get(key) is not None
    }
    if identity:
        portal["_identity"] = identity
    if isinstance(raw.get("_studio"), dict):
        portal["_studio"] = raw["_studio"]
    if isinstance(raw.get("_generation"), dict):
        portal["_generation"] = raw["_generation"]
    return portal


def _flatten_components(components: List[Dict[str, Any]],
                        d: int, K: int) -> Dict[str, Any]:
    """Assemble the flat P dictionary from per-component sub-objects."""
    records = components
    comp_list = [r["component"] for r in records]

    P: Dict[str, Any] = {}

    # ---- per-component scalars (length-K vectors) ----
    P["form"]       = np.array([_as_float_or_default(c.get("form"), 1.0) for c in comp_list], dtype=float)
    P["beta"]       = np.array([_as_float_or_default(c.get("beta"), np.nan) for c in comp_list], dtype=float)
    P["Delta"]      = np.array([_as_float_or_default(c.get("Delta"), np.nan) for c in comp_list], dtype=float)
    P["r_ref"]      = np.array([_as_float_or_default(c.get("r_ref"), np.nan) for c in comp_list], dtype=float)
    P["rho_single"] = np.array([_as_float_or_default(c.get("rho_single"), np.nan) for c in comp_list], dtype=float)
    P["p_single"]   = np.array([_as_float_or_default(c.get("p_single"), np.nan) for c in comp_list], dtype=float).reshape(K, 1)

    # Preserve each component's neutralization setting; a block may mix
    # neutralized and unneutralized components.
    P["neutralize_enable"] = np.array(
        [bool(c.get("neutralize_enable", True)) for c in comp_list],
        dtype=bool,
    )

    # ---- per-dimension fields (K x d) ----
    for fld in ("c", "kappa_plus", "kappa_minus",
                "p_plus", "p_minus", "rho_plus", "rho_minus"):
        rows: List[np.ndarray] = []
        for rec in records:
            comp = rec["component"]
            block_d = int(rec["block_d"])
            dim_idx = rec["dim_idx"]
            fill = 0.6 if fld in ("p_plus", "p_minus") else 0.0
            rows.append(_embed_local_vector(comp[fld], d, block_d, dim_idx, fill=fill))
        P[fld] = np.array(rows, dtype=float)

    # ---- rotation: angles (d, d, K) and rotation matrix R (d, d, K) ----
    Psi_deg = np.zeros((d, d, K), dtype=float)
    R = np.zeros((d, d, K), dtype=float)
    for k, rec in enumerate(records):
        comp = rec["component"]
        block_d = int(rec["block_d"])
        dim_idx = rec["dim_idx"]
        rot = comp.get("rotation") or {}
        Psi_local = np.asarray(rot.get("Psi_deg", np.zeros((block_d, block_d))), dtype=float)
        Psi_k = _embed_local_matrix(Psi_local, d, block_d, dim_idx)
        Psi_deg[:, :, k] = Psi_k
        # Build the local rotation first, then embed it at the block indices.
        # This preserves local axis ordering even when dim_idx is shuffled.
        if "R" in rot:
            R_local = np.asarray(rot["R"], dtype=float)
            if R_local.shape != (block_d, block_d) or not np.all(np.isfinite(R_local)):
                raise ValueError("Rotation R must be a finite matrix matching the block dimension.")
        else:
            R_local = givens_from_psi(np.deg2rad(Psi_local))
        R_k = np.eye(d, dtype=float)
        idx = np.asarray(dim_idx, dtype=int)
        R_k[np.ix_(idx, idx)] = R_local
        R[:, :, k] = R_k
    P["Psi_deg"] = Psi_deg
    P["R"] = R

    # ---- T_* transform blocks ----
    block_types = {
        key: base_key
        for component in comp_list
        for key, base_key, _block in iter_transform_blocks(component)
    }
    for blk_name, base_name in block_types.items():
        per_dim_fields = _T_PER_DIM_FIELDS[base_name]
        comp_blocks = [c.get(blk_name) if isinstance(c.get(blk_name), dict) else None
                       for c in comp_list]
        if all(b is None for b in comp_blocks):
            continue

        T_block: Dict[str, Any] = {}

        # Per-dim (K x d) fields
        for fld in per_dim_fields:
            rows: List[np.ndarray] = []
            has_any = False
            for rec, b in zip(records, comp_blocks):
                block_d = int(rec["block_d"])
                dim_idx = rec["dim_idx"]
                if b is not None and fld in b and b[fld] is not None:
                    rows.append(_embed_local_vector(b[fld], d, block_d, dim_idx, fill=0.0))
                    has_any = True
                else:
                    rows.append(np.zeros(d, dtype=float))
            if has_any:
                T_block[fld] = np.array(rows, dtype=float)

        # Per-component scalars (length-K vector)
        for fld in _T_PER_COMPONENT_SCALARS.get(base_name, ()):
            vals: List[float] = []
            has_any = False
            for b in comp_blocks:
                if b is not None and fld in b and b[fld] is not None:
                    vals.append(float(b[fld]))
                    has_any = True
                else:
                    vals.append(0.0)
            if has_any:
                T_block[fld] = np.array(vals, dtype=float)

        # Block-level (K-independent) scalars
        for fld in _T_BLOCK_SCALARS.get(base_name, ()):
            values: List[float] = []
            for b in comp_blocks:
                if b is not None and fld in b and b[fld] is not None:
                    values.append(float(b[fld]))
            if values:
                if not np.allclose(values, values[0], rtol=0.0, atol=0.0):
                    raise ValueError(
                        f"Shared transform scalar {blk_name}.{fld} differs "
                        "between components or blocks."
                    )
                T_block[fld] = values[0]

        P[blk_name] = T_block

    return P


def _as_float_or_default(value: Any, default: float) -> float:
    """Convert scalar to float; map None to default."""
    if value is None:
        return float(default)
    return float(value)


def _validate_dim_indices(dim_one_based: Any, d: int, block_d: int, b_idx: int) -> List[int]:
    """Validate block dim indices (1-based) and return 0-based list."""
    if not isinstance(dim_one_based, list):
        raise ValueError(f"Block {b_idx + 1} dim_indices_one_based must be a list.")
    if len(dim_one_based) != block_d:
        raise ValueError(
            f"Block {b_idx + 1} dim_indices_one_based length {len(dim_one_based)} != block d={block_d}."
        )
    try:
        dim_idx = [int(x) - 1 for x in dim_one_based]
    except Exception as exc:
        raise ValueError(f"Block {b_idx + 1} dim_indices_one_based has non-integer values.") from exc
    if any(i < 0 or i >= d for i in dim_idx):
        raise ValueError(f"Block {b_idx + 1} dim_indices_one_based values must be in [1, {d}].")
    if len(set(dim_idx)) != len(dim_idx):
        raise ValueError(f"Block {b_idx + 1} dim_indices_one_based contains duplicates.")
    return dim_idx


def _embed_local_vector(values: Any,
                        d: int,
                        block_d: int,
                        dim_idx: List[int],
                        *,
                        fill: float) -> np.ndarray:
    """Embed a block-local vector into a global-d vector."""
    src = np.asarray(values, dtype=float).reshape(-1)
    if src.size != block_d:
        raise ValueError(f"Expected local vector of length {block_d}, got {src.size}.")
    out = np.full(d, float(fill), dtype=float)
    out[np.asarray(dim_idx, dtype=int)] = src
    return out


def _embed_local_matrix(values: Any,
                        d: int,
                        block_d: int,
                        dim_idx: List[int]) -> np.ndarray:
    """Embed a block-local square matrix into a global dxd matrix."""
    src = np.asarray(values, dtype=float)
    if src.shape != (block_d, block_d):
        raise ValueError(f"Expected local matrix shape {(block_d, block_d)}, got {src.shape}.")
    out = np.zeros((d, d), dtype=float)
    idx = np.asarray(dim_idx, dtype=int)
    out[np.ix_(idx, idx)] = src
    return out


# ---------------------------------------------------------------------------
# Rotation, dimension-fix, and conversion helpers
# ---------------------------------------------------------------------------
def givens_from_psi(Psi: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix R from upper-triangular angle matrix Psi
    using Givens rotations.

    Args:
        Psi: d×d upper-triangular matrix of rotation angles (radians)

    Returns:
        R: d×d orthogonal rotation matrix
    """
    d = Psi.shape[0]
    R = np.eye(d)

    for u in range(d - 1):
        for v in range(u + 1, d):
            psi = Psi[u, v]
            if not np.isfinite(psi) or abs(psi) < 1e-15:
                continue

            G = np.eye(d)
            c = np.cos(psi)
            s = np.sin(psi)
            G[u, u] = c
            G[u, v] = -s
            G[v, u] = s
            G[v, v] = c
            R = R @ G

    return R


def fix_parameter_dimensions(P: Dict[str, Any], K: int, d: int) -> Dict[str, Any]:
    """
    Fix parameter dimensions after JSON load.
    When K=1, JSON may compress dimensions. This function ensures
    all parameters have the correct shape.

    Args:
        P: Parameter dictionary
        K: Number of components
        d: Dimension

    Returns:
        Fixed parameter dictionary
    """
    # Fix K×d matrices (should be K rows, d columns)
    kd_fields = ['c', 'kappa_plus', 'kappa_minus', 'p_plus', 'p_minus',
                 'rho_plus', 'rho_minus']
    for field in kd_fields:
        if field in P and P[field] is not None:
            P[field] = ensure_size(P[field], K, d)

    # Fix K×1 vectors (should be K elements)
    k1_fields = ['beta', 'Delta', 'r_ref', 'rho_single', 'form']
    for field in k1_fields:
        if field in P and P[field] is not None:
            arr = ensure_size(P[field], K, 1)
            P[field] = arr.ravel() if arr.size > 0 else arr

    # p_single is special: MATLAB stores as 1D array, but internally we use K×1
    if 'p_single' in P and P['p_single'] is not None:
        p_single = np.array(P['p_single'])
        # If it's already a simple K-element array, reshape to K×1 for internal use
        if p_single.ndim == 1 and len(p_single) == K:
            P['p_single'] = p_single.reshape(K, 1)
        else:
            # Otherwise ensure proper sizing
            P['p_single'] = ensure_size(P['p_single'], K, 1)

    # Fix 3D arrays: Psi_deg and R (d×d×K)
    if 'Psi_deg' in P:
        P['Psi_deg'] = ensure_size_3d(P['Psi_deg'], d, d, K)

    if 'neutralize_enable' in P:
        enabled = np.asarray(P['neutralize_enable'], dtype=bool).reshape(-1)
        if enabled.size == 1:
            enabled = np.repeat(enabled, K)
        elif enabled.size != K:
            raise ValueError(
                "neutralize_enable must be scalar or have one value per component."
            )
        P['neutralize_enable'] = enabled

    if 'R' in P:
        P['R'] = ensure_size_3d(P['R'], d, d, K)

    # Apply the same dimension policy to every independently stored occurrence.
    for _storage_name, base_name, T in iter_transform_blocks(P):
        for field in _T_PER_DIM_FIELDS[base_name]:
            T[field] = ensure_size(T[field], K, d)
        if base_name == 'T_log_sinusoidal' and 'sigma' not in T:
            T['sigma'] = 0.1 * np.ones(K)
        if base_name == 'T_wavelet_mod':
            T.pop('L_plus', None)
            T.pop('L_minus', None)
        for field in _T_PER_COMPONENT_SCALARS.get(base_name, ()):
            arr = ensure_size(T[field], K, 1)
            T[field] = arr.ravel() if arr.size > 0 else arr

    return P


def ensure_size(A: Any, rows: int, cols: int) -> np.ndarray:
    """
    Ensure matrix A has size [rows × cols].
    Handles cases where JSON compression changed dimensions.

    Args:
        A: Input array (can be list, scalar, or array)
        rows: Target number of rows
        cols: Target number of columns

    Returns:
        Array with correct shape
    """
    A = np.atleast_1d(np.array(A, dtype=float))

    if A.size == 0:
        return np.zeros((rows, cols))

    current_shape = A.shape

    # If already correct size
    if current_shape == (rows, cols):
        return A

    # If scalar and should be rows×cols
    if A.size == 1:
        return np.full((rows, cols), A.item())

    # If row vector when should be column vector
    if current_shape == (1, rows) and cols == 1:
        return A.T

    # If column vector when should be row vector
    if current_shape == (rows, 1) and cols > 1:
        return np.tile(A, (1, cols))

    # If dimensions are swapped
    if current_shape == (cols, rows):
        return A.T

    # If it's a row vector [1×cols] but should be [rows×cols] with rows=1
    if current_shape == (1, cols) and rows == 1:
        return A

    # If it's a 1D array of correct length for either dimension
    if A.ndim == 1:
        if A.size == rows * cols:
            return A.reshape(rows, cols)
        elif A.size == rows and cols == 1:
            return A.reshape(rows, 1)
        elif A.size == cols and rows == 1:
            return A.reshape(1, cols)

    # If total elements match, try to reshape
    if A.size == rows * cols:
        return A.reshape(rows, cols)

    # If we get here, something unexpected happened
    print(f"Warning: Cannot reshape array from {current_shape} to [{rows}×{cols}]")
    return A


def ensure_size_3d(A: Any, dim1: int, dim2: int, dim3: int) -> np.ndarray:
    """
    Ensure 3D array A has size [dim1 × dim2 × dim3].
    Handles cases where JSON compression changed dimensions for K=1.

    Args:
        A: Input array
        dim1, dim2, dim3: Target dimensions

    Returns:
        Reshaped array with correct dimensions
    """
    A = np.array(A, dtype=float)

    if A.size == 0:
        return np.zeros((dim1, dim2, dim3))

    current_shape = A.shape

    # If already correct size
    if current_shape == (dim1, dim2, dim3):
        return A

    # Case 1: K=1 and array was compressed from [d×d×1] to [d×d]
    if A.ndim == 2 and dim3 == 1 and current_shape == (dim1, dim2):
        return A.reshape(dim1, dim2, 1)

    # Case 2: Array is already 3D but dimensions match by element count
    if A.size == dim1 * dim2 * dim3:
        return A.reshape(dim1, dim2, dim3)

    # If we get here, something unexpected happened
    print(f"Warning: Cannot reshape 3D array from {current_shape} to [{dim1}×{dim2}×{dim3}]")
    return A


def _normalize_transform_seq(seq: Any, K: int) -> List[List[str]]:
    """Coerce transform_seq into a list of length-K lists of tf-name strings."""
    if seq is None:
        return [[] for _ in range(K)]
    out: List[List[str]] = []
    for k in range(K):
        item = seq[k] if isinstance(seq, (list, tuple)) and k < len(seq) else []
        if item is None:
            out.append([])
        elif isinstance(item, str):
            out.append([item])
        elif isinstance(item, (list, tuple)):
            out.append([str(t) for t in item])
        else:
            out.append([str(item)])
    return out


def _component_transform_seqs(components: List[Dict[str, Any]],
                              K: int) -> List[List[str]]:
    """Return per-component transform sequences from component-level field."""
    out: List[List[str]] = []
    for k in range(K):
        comp = components[k] if k < len(components) else {}
        if not (isinstance(comp, dict) and "transform_seq" in comp):
            raise ValueError(
                f"Unified schema requires components[{k}].transform_seq."
            )
        out.append(_normalize_transform_seq([comp.get("transform_seq")], 1)[0])
    return out


def _normalize_block_baseline(baseline: Any,
                              components: List[Dict[str, Any]],
                              K: int) -> List[str]:
    """Return a length-K baseline list, inferring missing labels from form."""
    out: List[str] = []
    for k in range(K):
        label = None
        if isinstance(baseline, (list, tuple)) and k < len(baseline):
            raw = str(baseline[k]).upper()
            if raw in ("A", "B"):
                label = raw
        if label is None:
            form = int(float(components[k].get("form", 1)))
            label = "A" if form == 1 else "B"
        out.append(label)
    return out
