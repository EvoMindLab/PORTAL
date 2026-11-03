"""
Author: Delaram Yazdani, Danial Yazdani, Mai Peng
Email: delaram.yazdani@yahoo.com
             danial.yazdani@gmail.com
             pengmai1998@gmail.com
Title: Save PORTAL Instance to JSON (unified blocks/components schema)

Serialize a runtime PORTAL instance to the unified hierarchical JSON schema.

The saved format is the shared family schema with top-level blocks/components,
component-level transform_seq, and optional suite provenance metadata.
Files with independent repeated transformations wrap this payload in a
versioned container so readers without support reject them before evaluation.

Notes:
- Runtime instances use flat in-memory arrays under inst['P']; this module
    performs runtime-to-schema conversion.
- Non-finite floats (NaN/Inf) are serialized as JSON null for strict parser
    compatibility.
- If inst['_schema_blocks'] is present, save preserves the block partition.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from .transform_registry import (
        TRANSFORM_BLOCK_FIELD_ORDER as _T_BLOCK_FIELD_ORDER,
        TRANSFORM_BLOCK_SCALARS as _T_BLOCK_SCALARS,
        TRANSFORM_PER_COMPONENT_SCALARS as _T_PER_COMPONENT_SCALARS,
        iter_transform_blocks, transform_block_type,
    )
except ImportError:
    from transform_registry import (  # type: ignore
        TRANSFORM_BLOCK_FIELD_ORDER as _T_BLOCK_FIELD_ORDER,
        TRANSFORM_BLOCK_SCALARS as _T_BLOCK_SCALARS,
        TRANSFORM_PER_COMPONENT_SCALARS as _T_PER_COMPONENT_SCALARS,
        iter_transform_blocks, transform_block_type,
    )


def save_instance(
    portal_instance: Dict[str, Any],
    filename: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Save a PORTAL benchmark instance to a JSON file under the unified schema."""
    if filename is None or filename == "":
        filename = (
            f"PORTAL_d{portal_instance['d']}_K{portal_instance['K']}"
            f"_seed{portal_instance['Seed']}.json"
        )
    if not filename.endswith(".json"):
        filename = Path(filename).stem + ".json"

    script_dir = Path(__file__).parent
    instances_dir = Path(output_dir) if output_dir else script_dir / "Instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    filepath = instances_dir / filename

    payload = _build_unified_payload(portal_instance)
    payload = _sanitize_for_json(payload)
    _validate_unified_payload(payload)

    try:
        with open(filepath, "w", encoding="utf-8") as fid:
            json.dump(payload, fid, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"Instance saved successfully to:\n  {filepath}")
        return str(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to save instance: {str(e)}")


# ---------------------------------------------------------------------------
# Flat-runtime -> unified-JSON conversion
# ---------------------------------------------------------------------------
def _build_unified_payload(portal_instance: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a flat runtime PORTAL instance into the unified JSON layout."""
    inst = copy.deepcopy(portal_instance)
    inst.pop("Rng", None)
    # Raw 'ranges' is stored inside the informational '_generation' block;
    # the bare key itself is not part of the unified schema.
    inst.pop("ranges", None)
    generation_meta = inst.pop("_generation", None)

    d = int(inst["d"])
    K = int(inst["K"])
    P = inst["P"]
    suite = inst.get("_suite", {}) or {}
    studio = inst.get("_studio")

    identity = inst.get("_identity", {}) if isinstance(inst.get("_identity"), dict) else {}
    family_id = suite.get("family_id", identity.get("family"))
    recipe_index = suite.get("recipe_index")
    instance_file = suite.get("instance_file") or ""
    recipe_str = identity.get("recipe") or (Path(instance_file).stem if instance_file else None)
    if recipe_str is None and recipe_index is not None and family_id is not None:
        recipe_str = f"f{int(family_id):02d}_r{int(recipe_index):03d}"

    baseline = inst.get("baseline", [])
    transform_seq = inst.get("transform_seq", [])
    schema_blocks = inst.get("_schema_blocks")

    blocks = _build_blocks(
        P=P,
        d=d,
        K=K,
        baseline=baseline,
        transform_seq=transform_seq,
        schema_blocks=schema_blocks,
    )
    block_dims = [int(b.get("d", d)) for b in blocks]

    payload: Dict[str, Any] = {
        "family":             int(family_id) if family_id is not None else None,
        "recipe":             recipe_str,
        "Seed":               int(inst["Seed"]),
        "d":                  d,
        "B":                  len(blocks),
        "block_dims":         block_dims,
        "bounds_min":         _to_py(inst["bounds_min"]),
        "bounds_max":         _to_py(inst["bounds_max"]),
        "warn_out_of_bounds": bool(inst.get("warn_out_of_bounds", False)),
        "blocks":             blocks,
        "_suite":             _convert_arrays_to_lists(suite),
    }
    if generation_meta:
        # Informational generation metadata (sampling ranges and resolved
        # config); ignored by loaders and the evaluation pipeline.
        payload["_generation"] = _convert_arrays_to_lists(generation_meta)
    if isinstance(studio, dict):
        payload["_studio"] = _convert_arrays_to_lists(studio)

    if payload["family"] is None:
        del payload["family"]
    if payload["recipe"] is None:
        del payload["recipe"]
    return payload


def composite_to_unified_payload(
    composite: Dict[str, Any],
    *,
    seed: Optional[int] = None,
    suite: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Repackage a Layer-3 composite into one unified multi-block payload.

    A composite (see ``compose_blocks``) holds independent single-block PORTAL
    instances plus a global dimension permutation. Each composite block is
    already a valid single-block instance, so this attaches that block's global
    dimension indices (``composite['indices_per_block'][j]``) and its Layer-3
    weight (``composite['weights'][j]``) and concatenates the blocks under one
    unified-schema payload -- the same schema a script-generated multi-block
    anchor uses. The result evaluates identically via ``fitness`` /
    ``_normalized_schema_blocks`` because unified aggregation
    ``sum_j weight_j * min_k phi_j`` matches ``fitness_composite``'s
    ``sum_j w_j * f_block_j`` when each block is single-block (internal weight 1).

    There is no ``composite: True`` flag and no nested payloads: a single JSON,
    readable by ``load_instance`` like any other instance.
    """
    if not composite.get("composite", False):
        raise ValueError(
            "composite_to_unified_payload expects a composite instance "
            "(missing 'composite': True)."
        )

    blocks_rt = composite["blocks"]
    indices_per_block = composite["indices_per_block"]
    weights = composite["weights"]
    block_ids = composite.get("block_ids", list(range(1, len(blocks_rt) + 1)))
    D = int(composite["d"])
    if not blocks_rt:
        raise ValueError("A composite must contain at least one block.")
    if not (len(blocks_rt) == len(indices_per_block) == len(weights) == len(block_ids)):
        raise ValueError(
            "Composite blocks / indices_per_block / weights / block_ids length mismatch."
        )
    normalized_block_ids = [int(value) for value in block_ids]
    if len(set(normalized_block_ids)) != len(normalized_block_ids):
        raise ValueError("Composite block_ids must be unique.")

    normalized_indices: List[List[int]] = []
    for j, indices in enumerate(indices_per_block):
        try:
            normalized = [int(i) for i in indices]
        except Exception as exc:
            raise ValueError(f"Composite block {j} has invalid global indices.") from exc
        normalized_indices.append(normalized)
    flattened_indices = [i for block in normalized_indices for i in block]
    if sorted(flattened_indices) != list(range(D)):
        raise ValueError(
            "Composite indices_per_block must form an exact, non-overlapping "
            f"partition of global dimensions [0, {D})."
        )

    normalized_weights = [float(value) for value in weights]
    if any(not np.isfinite(value) or value <= 0.0 for value in normalized_weights):
        raise ValueError("Composite block weights must be finite and strictly positive.")

    merged_blocks: List[Dict[str, Any]] = []
    block_dims: List[int] = []
    for j, block_rt in enumerate(blocks_rt):
        inner_payload = _build_unified_payload(block_rt)
        inner_blocks = inner_payload["blocks"]
        if len(inner_blocks) != 1:
            raise ValueError(
                f"Composite block {j} is itself multi-block; "
                "unified repackaging expects single-block instances."
            )
        blk = inner_blocks[0]
        gidx = normalized_indices[j]
        block_d = int(blk.get("d", len(gidx)))
        if len(gidx) != block_d:
            raise ValueError(
                f"Composite block {j}: indices_per_block has {len(gidx)} dims "
                f"but block dimension is {block_d}."
            )
        blk["block_id"] = normalized_block_ids[j]
        blk["weight"] = normalized_weights[j]
        blk["dim_indices_one_based"] = [i + 1 for i in gidx]
        merged_blocks.append(blk)
        block_dims.append(block_d)

    if sum(block_dims) != D:
        raise ValueError(
            f"Composite block dims {block_dims} sum to {sum(block_dims)}, expected d={D}."
        )

    if seed is None:
        seed = int(composite.get("seed", blocks_rt[0].get("Seed", 0)) or 0)

    suite_meta = composite.get("_suite", {}) if suite is None else suite
    suite_meta = suite_meta if isinstance(suite_meta, dict) else {}
    identity = composite.get("_identity", {})
    identity = identity if isinstance(identity, dict) else {}
    family_id = suite_meta.get("family_id", identity.get("family"))
    instance_file = suite_meta.get("instance_file") or ""
    recipe_index = suite_meta.get("recipe_index")
    recipe_str = identity.get("recipe") or (Path(instance_file).stem if instance_file else None)
    if recipe_str is None and recipe_index is not None and family_id is not None:
        recipe_str = f"f{int(family_id):02d}_r{int(recipe_index):03d}"

    payload: Dict[str, Any] = {
        "Seed":               int(seed),
        "d":                  D,
        "B":                  len(merged_blocks),
        "block_dims":         block_dims,
        "bounds_min":         _to_py(composite.get("bounds_min", blocks_rt[0].get("bounds_min", -100))),
        "bounds_max":         _to_py(composite.get("bounds_max", blocks_rt[0].get("bounds_max", 100))),
        "warn_out_of_bounds": bool(composite.get("warn_out_of_bounds", False)),
        "blocks":             merged_blocks,
        "_suite":             _convert_arrays_to_lists(suite_meta),
    }
    if family_id is not None:
        payload["family"] = int(family_id)
    if recipe_str is not None:
        payload["recipe"] = str(recipe_str)
    if isinstance(composite.get("_studio"), dict):
        payload["_studio"] = _convert_arrays_to_lists(composite["_studio"])
    if isinstance(composite.get("_generation"), dict):
        payload["_generation"] = _convert_arrays_to_lists(composite["_generation"])
    return payload


def save_composite_as_unified(
    composite: Dict[str, Any],
    filename: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    *,
    seed: Optional[int] = None,
    suite: Optional[Dict[str, Any]] = None,
    remove_block_files: bool = False,
) -> str:
    """Save a Layer-3 composite as a single unified multi-block JSON file.

    This is the multi-block counterpart of ``save_instance``: it flattens the
    composite (via ``composite_to_unified_payload``) so the on-disk artifact is
    a standalone unified instance, not a composite envelope. The evaluator loads
    it with ``load_instance`` like any single-block instance.
    """
    if filename is None or filename == "":
        filename = f"PORTAL_d{int(composite['d'])}_B{len(composite['blocks'])}.json"
    if not filename.endswith(".json"):
        filename = Path(filename).stem + ".json"

    script_dir = Path(__file__).parent
    instances_dir = Path(output_dir) if output_dir else script_dir / "Instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    filepath = instances_dir / filename

    payload = composite_to_unified_payload(composite, seed=seed, suite=suite)
    payload = _sanitize_for_json(payload)
    _validate_unified_payload(payload)

    try:
        with open(filepath, "w", encoding="utf-8") as fid:
            json.dump(payload, fid, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"Instance saved successfully to:\n  {filepath}")
        if remove_block_files:
            removed: List[str] = []
            for source in composite.get("_source_paths", []) or []:
                if not source:
                    continue
                source_path = Path(source)
                try:
                    if source_path.exists() and source_path.resolve() != filepath.resolve():
                        source_path.unlink()
                        removed.append(str(source_path))
                except OSError as exc:
                    # The unified artifact is already complete. A cleanup
                    # failure must not be reported as a failed save.
                    print(f"  Warning: could not remove source block {source_path}: {exc}")
            if removed:
                print(f"Removed {len(removed)} source block JSON file(s).")
        return str(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to save instance: {str(e)}")


def _build_blocks(
    P: Dict[str, Any],
    d: int,
    K: int,
    baseline: Any,
    transform_seq: Any,
    schema_blocks: Any,
) -> List[Dict[str, Any]]:
    """Build unified-schema blocks from flat runtime arrays.

    If ``schema_blocks`` is present, preserve its validated partitioning.
    If it is absent, emit one ordinary Layer-2 block. Invalid metadata is
    rejected rather than silently changing a multi-block landscape.
    """
    has_schema = schema_blocks is not None
    if has_schema and not _has_valid_schema_blocks(schema_blocks, K, d):
        raise ValueError(
            "Runtime _schema_blocks metadata is present but invalid; refusing "
            "to collapse it silently into a single block."
        )

    if has_schema:
        blocks: List[Dict[str, Any]] = []
        cursor = 0
        contiguous_cursor = 0
        for idx, meta in enumerate(schema_blocks):
            k_block = int(meta["K"])
            block_id = int(meta.get("block_id", idx + 1))
            block_weight = _coerce_block_weight(meta, default=1.0)
            block_d = int(meta.get("d", d))
            dim_idx = _coerce_dim_indices(meta.get("dim_indices"), d=d, block_d=block_d)
            if dim_idx is None:
                if len(schema_blocks) == 1 and block_d == d:
                    dim_idx = list(range(d))
                else:
                    start = contiguous_cursor
                    stop = contiguous_cursor + block_d
                    if stop > d:
                        raise ValueError(
                            f"Cannot infer contiguous dim_indices for block {idx + 1}: d={d}, block_d={block_d}."
                        )
                    dim_idx = list(range(start, stop))
            contiguous_cursor += block_d

            seq_slice = []
            if isinstance(transform_seq, (list, tuple)):
                seq_slice = list(transform_seq[cursor:cursor + k_block])
            seq_norm = _normalize_transform_seq(seq_slice, k_block)

            components = [
                _build_component(
                    k=cursor + local_k,
                    P=P,
                    d=d,
                    seq_k=seq_norm[local_k],
                    block_d=block_d,
                    dim_idx=dim_idx,
                    component_id=local_k + 1,
                )
                for local_k in range(k_block)
            ]
            baseline_block = _baseline_slice(baseline, P, cursor, k_block)

            blocks.append({
                "block_id":      block_id,
                "weight":        block_weight,
                "d":             block_d,
                "dim_indices_one_based": [int(i) + 1 for i in dim_idx],
                "baseline":      baseline_block,
                "K":             k_block,
                "components":    components,
            })
            cursor += k_block
        return blocks

    seq_norm = _normalize_transform_seq(transform_seq, K)
    components = [
        _build_component(
            k=k, P=P, d=d, seq_k=seq_norm[k], block_d=d,
            dim_idx=list(range(d)), component_id=k + 1,
        )
        for k in range(K)
    ]
    return [{
        "block_id":      1,
        "weight":        1.0,
        "d":             d,
        "dim_indices_one_based": list(range(1, d + 1)),
        "baseline":      _baseline_slice(baseline, P, 0, K),
        "K":             K,
        "components":    components,
    }]


def _has_valid_schema_blocks(schema_blocks: Any, K: int, d: int) -> bool:
    """Return True if schema_blocks is a valid partition of total K."""
    if not isinstance(schema_blocks, (list, tuple)) or len(schema_blocks) == 0:
        return False
    total = 0
    for item in schema_blocks:
        if not isinstance(item, dict) or "K" not in item:
            return False
        k_block = int(item["K"])
        if k_block <= 0:
            return False
        if "d" in item:
            try:
                block_d = int(item["d"])
            except Exception:
                return False
            if block_d <= 0 or block_d > int(d):
                return False
            dim_idx = item.get("dim_indices")
            if dim_idx is not None:
                idx = _coerce_dim_indices(dim_idx, d=int(d), block_d=block_d)
                if idx is None:
                    return False
        if "weight" in item:
            try:
                w = float(item["weight"])
            except Exception:
                return False
            if not np.isfinite(w) or w <= 0.0:
                return False
        total += k_block
    return total == int(K)


def _coerce_block_weight(meta: Dict[str, Any], default: float = 1.0) -> float:
    """Return a finite positive block weight, or default when absent/invalid."""
    try:
        w = float(meta.get("weight", default))
    except Exception:
        return float(default)
    if not np.isfinite(w) or w <= 0.0:
        return float(default)
    return w


def _baseline_slice(baseline: Any, P: Dict[str, Any], start: int, K: int) -> List[str]:
    """Return K baseline labels starting at index start, defaulting from form."""
    out: List[str] = []
    for offset in range(K):
        k = start + offset
        label = None
        if isinstance(baseline, (list, tuple)) and k < len(baseline):
            raw = str(baseline[k]).upper()
            if raw in ("A", "B"):
                label = raw
        if label is None:
            form_k = int(_row_scalar(P.get("form", [1]), k))
            label = "A" if form_k == 1 else "B"
        out.append(label)
    return out


def _build_component(k: int,
                     P: Dict[str, Any],
                     d: int,
                     seq_k: List[str],
                     block_d: int,
                     dim_idx: List[int],
                     component_id: Optional[int] = None) -> Dict[str, Any]:
    """Extract component k from the flat P dictionary."""
    comp: Dict[str, Any] = {
        "component_id":      int(component_id if component_id is not None else k + 1),
        "transform_seq":     list(seq_k),
        "form":              int(_row_scalar(P["form"], k)),
        "neutralize_enable": _bool_at(
            P.get("neutralize_enable", True), k, default=True
        ),
        "c":                 _row_local(P["c"], k, d, dim_idx),
        "beta":              float(_row_scalar(P["beta"], k)),
        "Delta":             float(_row_scalar(P["Delta"], k)),
        "r_ref":             float(_row_scalar(P["r_ref"], k)),
        "kappa_plus":        _row_local(P["kappa_plus"], k, d, dim_idx),
        "kappa_minus":       _row_local(P["kappa_minus"], k, d, dim_idx),
        "p_single":          float(_row_scalar(P["p_single"], k)),
        "p_plus":            _row_local(P["p_plus"], k, d, dim_idx),
        "p_minus":           _row_local(P["p_minus"], k, d, dim_idx),
        "rho_plus":          _row_local(P["rho_plus"], k, d, dim_idx),
        "rho_minus":         _row_local(P["rho_minus"], k, d, dim_idx),
        "rho_single":        float(_row_scalar(P["rho_single"], k)),
        "rotation":          _build_rotation(P, k, d, dim_idx),
    }
    # Only serialize a transform's parameter block when that transform is
    # actually named in this component's transform_seq. Presence of a T_* block
    # in the flat P (the generator populates all five for convenience) does not
    # mean the component uses it; the evaluator gates on transform_seq, so a
    # dormant block is unused payload. Gating here keeps the on-disk schema
    # identical between the script generator and the GUI, which only ever emits
    # the transforms it applied.
    seq_names = set(seq_k)
    for blk_name, _base_name, block in iter_transform_blocks(P):
        seq_name = blk_name[2:]
        if seq_name not in seq_names:
            continue
        comp[blk_name] = _build_transform_block(blk_name, block, k, d, dim_idx)
    return comp


def _build_rotation(P: Dict[str, Any], k: int, d: int, dim_idx: List[int]) -> Dict[str, Any]:
    """Store angles plus an authoritative matrix when angles cannot reproduce it."""
    psi = P.get("Psi_deg", None)
    if psi is None:
        Psi_k = np.zeros((d, d))
    else:
        Psi_arr = np.asarray(psi)
        if Psi_arr.ndim == 3:
            Psi_k = Psi_arr[:, :, k]
        elif Psi_arr.ndim == 2:
            Psi_k = Psi_arr
        else:
            Psi_k = np.zeros((d, d))
    idx = np.asarray(dim_idx, dtype=int)
    Psi_local = np.asarray(Psi_k, dtype=float)[np.ix_(idx, idx)]
    active = bool(np.any(np.abs(Psi_local) > 0.0))
    rotation = {"active": active, "Psi_deg": Psi_local.tolist()}
    if "R" in P:
        matrix = np.asarray(P["R"], dtype=float)
        if matrix.ndim == 3:
            matrix = matrix[:, :, k]
        if matrix.shape != (d, d) or not np.all(np.isfinite(matrix)):
            raise ValueError("Rotation R must be a finite matrix matching the instance dimension.")
        local_matrix = matrix[np.ix_(idx, idx)]
        try:
            from .load_instance import givens_from_psi
        except ImportError:
            from load_instance import givens_from_psi
        reconstructed = givens_from_psi(np.deg2rad(Psi_local))
        if not np.array_equal(local_matrix, reconstructed):
            # Preserve authoritative data; do not infer or approximate angles.
            rotation["R"] = local_matrix.tolist()
            rotation["active"] = not np.array_equal(local_matrix, np.eye(idx.size))
    return rotation


def _build_transform_block(blk_name: str,
                           T: Dict[str, Any],
                           k: int,
                           d: int,
                           dim_idx: List[int]) -> Dict[str, Any]:
    """Build a per-component transform-block sub-object."""
    out: Dict[str, Any] = {}
    base_name = transform_block_type(blk_name)
    order = _T_BLOCK_FIELD_ORDER[base_name]
    per_comp_scalars = _T_PER_COMPONENT_SCALARS.get(base_name, ())
    block_scalars = _T_BLOCK_SCALARS.get(base_name, ())
    for fld in order:
        if fld not in T:
            continue
        val = T[fld]
        if fld in block_scalars:
            arr = np.asarray(val)
            out[fld] = float(arr.item()) if arr.size == 1 else _to_py(val)
        elif fld in per_comp_scalars:
            out[fld] = float(_row_scalar(val, k))
        else:
            out[fld] = _row_local(val, k, d, dim_idx)
    return out


def _row(arr: Any, k: int, d: int) -> List[float]:
    """Return component-k row of a flat (K, d) field as a length-d list."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        if a.size == d:
            return a.tolist()
        if a.size == 1:
            return [float(a.item())] * d
        return [float(a[k])] * d
    if a.ndim == 2:
        return a[k, :].tolist()
    raise ValueError(f"Unexpected ndim={a.ndim} for per-dimension field")


def _bool_at(value: Any, k: int, *, default: bool) -> bool:
    """Return a scalar or component-indexed boolean without ambiguity."""
    if value is None:
        return bool(default)
    arr = np.asarray(value, dtype=bool).reshape(-1)
    if arr.size == 0:
        return bool(default)
    if arr.size == 1:
        return bool(arr[0])
    if k >= arr.size:
        raise ValueError(
            f"Boolean component field has length {arr.size}, cannot read index {k}."
        )
    return bool(arr[k])


def _row_local(arr: Any, k: int, d: int, dim_idx: List[int]) -> List[float]:
    """Return component-k row sliced to local block dimensions."""
    row = np.asarray(_row(arr, k, d), dtype=float)
    idx = np.asarray(dim_idx, dtype=int)
    return row[idx].tolist()


def _row_scalar(arr: Any, k: int) -> float:
    """Return component-k scalar from a length-K vector / scalar / (K,1) array."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 0 or a.size == 1:
        return float(a.item() if a.ndim else a)
    if a.ndim == 1:
        return float(a[k])
    if a.ndim == 2:
        return float(a[k, 0])
    raise ValueError(f"Unexpected ndim={a.ndim} for per-component scalar field")


def _coerce_dim_indices(dim_indices: Any, *, d: int, block_d: int) -> Optional[List[int]]:
    """Validate optional 0-based dim-indices list and return normalized list."""
    if dim_indices is None:
        return None
    if not isinstance(dim_indices, (list, tuple)):
        return None
    try:
        out = [int(x) for x in dim_indices]
    except Exception:
        return None
    if len(out) != int(block_d):
        return None
    if any(i < 0 or i >= int(d) for i in out):
        return None
    if len(set(out)) != len(out):
        return None
    return out


def _normalize_transform_seq(seq: Any, K: int) -> List[List[str]]:
    """Coerce transform_seq into a list of length-K lists of tf-name strings."""
    if seq is None:
        return [[] for _ in range(K)]
    out: List[List[str]] = []
    for k in range(K):
        item = seq[k] if k < len(seq) else []
        if item is None:
            out.append([])
        elif isinstance(item, str):
            out.append([item])
        elif isinstance(item, (list, tuple)):
            out.append([str(t) for t in item])
        else:
            out.append([str(item)])
    return out


def _to_py(x: Any) -> Any:
    """Convert numpy / nested-array values to native Python for JSON."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def _convert_arrays_to_lists(obj: Any) -> Any:
    """Recursively convert numpy arrays inside ``obj`` to plain lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_arrays_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_arrays_to_lists(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert non-finite floats to None for strict JSON output."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def _validate_unified_payload(payload: Dict[str, Any]) -> None:
    """Fail before writing if a generated payload cannot be canonically loaded."""
    try:
        from .load_instance import load_instance_payload
    except ImportError:  # Direct-script compatibility.
        from load_instance import load_instance_payload  # type: ignore

    load_instance_payload(copy.deepcopy(payload))
