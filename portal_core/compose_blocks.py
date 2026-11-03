"""
Author: Danial Yazdani, Delaram Yazdani, and Mai Peng
Email:  danial.yazdani@gmail.com
        delaram.yazdani@yahoo.com
        pengmai1998@gmail.com

Title:  PORTAL Layer-3 (Composition) Helper

Description:
    Implements PORTAL's third architectural layer (composition) as a thin wrapper
    over single-block instances. Given a list of independently-generated PORTAL
    block instances and optional per-block weights, this module produces a
    *composite* instance whose fitness is the weighted sum of the per-block
    landscapes:

        F(x) = sum_j  w_j * f_j( x[I_j] )

    where each f_j is the standard PORTAL fitness of the j-th block instance and
    I_j is the set of global variable indices assigned to block j.

    Variable layout. Algorithms operating on the composite see only a single
    D-dimensional black box (D = sum of block dimensions). Internally, the
    composite owns a random permutation pi of {0,...,D-1} that scatters each
    block's d_j local coordinates over arbitrary positions in the global vector
    (so block boundaries are not exposed by the index ordering). The map is
    persisted in `indices_per_block`, where `indices_per_block[j][a]` is the
    global index of block j's a-th local coordinate. fitness_composite uses this
    map to gather x[I_j] in block-local order before delegating to fitness().

    Persistence. save_composite writes the same unified hierarchical JSON used
    by save_instance, with all blocks embedded in one file. The original block
    JSON files used as input are kept by default; pass
    remove_block_files=True to delete them after a successful save. Legacy
    composite envelopes remain readable through load_composite.

License:
    MIT for PORTAL-owned code; see LICENSE and LICENSING.md at the repository root.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Make sibling core modules importable during direct-script execution.
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

try:  # Normal package imports.
    from .fitness import fitness as _block_fitness  # noqa: E402
    from .fitness import fitness_batch as _block_fitness_batch  # noqa: E402
    from .load_instance import load_instance as _load_block  # noqa: E402
    from .save_instance import save_instance as _save_block  # noqa: E402
    from .save_instance import save_composite_as_unified as _save_composite_unified  # noqa: E402
except ImportError:  # Direct-script compatibility.
    from fitness import fitness as _block_fitness  # type: ignore  # noqa: E402
    from fitness import fitness_batch as _block_fitness_batch  # type: ignore  # noqa: E402
    from load_instance import load_instance as _load_block  # type: ignore  # noqa: E402
    from save_instance import save_instance as _save_block  # type: ignore  # noqa: E402
    from save_instance import save_composite_as_unified as _save_composite_unified  # type: ignore  # noqa: E402


PathLike = Union[str, Path]
BlockSpec = Union[Dict[str, Any], PathLike]
WeightsSpec = Union[None, str, Sequence[float]]


# ---------------------------------------------------------------------------
# Internal helpers used only by the legacy envelope reader/writer: serialize or
# deserialize one nested block through the canonical unified block schema.
# ---------------------------------------------------------------------------

def _block_to_jsonable(block: Dict[str, Any]) -> Dict[str, Any]:
    """Return the JSON-ready dict that save_instance would write for `block`."""
    with tempfile.TemporaryDirectory() as td:
        path = _save_block(block, filename='_tmp_block.json', output_dir=td)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def _jsonable_to_block(data: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct a fully usable block instance from its JSON-form dict."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / '_tmp_block.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        return _load_block(str(path))


def _resolve_block(spec: BlockSpec) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Return (instance_dict, source_path_or_None). source_path is set when `spec`
    was a path to an existing JSON file (used by remove_block_files).
    """
    if isinstance(spec, dict):
        return spec, None
    p = Path(spec)
    inst = _load_block(str(p))
    return inst, p.resolve() if p.exists() else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compose_blocks(
    block_specs: Sequence[BlockSpec],
    weights: WeightsSpec = None,
    weight_random_range: Tuple[float, float] = (0.5, 2.0),
    permutation: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
    bounds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Build a PORTAL composite (Layer-3) instance from independent block instances.

    Args:
        block_specs:
            Ordered list of length J. Each entry is either
              * a path (str / Path) to a PORTAL block JSON file (preferred), or
              * an in-memory PORTAL instance dictionary.
        weights:
            * None (default): every block gets weight 1.0.
            * 'random': sample one weight per block uniformly from
              `weight_random_range` using `seed`.
            * Sequence[float]: explicit per-block weights, length must equal J,
              assigned by order. Use this for deliberate ill-balancing.
        weight_random_range:
            (low, high) used when weights == 'random'. Default (0.5, 2.0).
        permutation:
            Optional explicit length-D permutation of {0,...,D-1}, where
            D = sum(d_j). If None (default) a random permutation is drawn from
            `seed`. The permutation is split by block-size into the per-block
            global-index lists; algorithms see x in global order while
            fitness_composite uses the map to recover each block's local view.
        seed:
            RNG seed for random weights and/or random permutation.
        bounds:
            Optional [lower, upper] global box. If omitted, taken from the first
            block (PORTAL uses [-100, 100] by default).

    Returns:
        Composite instance dictionary with keys:
            'composite' (True), 'd', 'J', 'bounds_min', 'bounds_max',
            'warn_out_of_bounds', 'weights', 'permutation',
            'indices_per_block', 'blocks',
            '_source_paths'  (private; used by save_composite for cleanup).
    """
    if len(block_specs) == 0:
        raise ValueError("compose_blocks requires at least one block.")

    blocks: List[Dict[str, Any]] = []
    source_paths: List[Optional[str]] = []
    for spec in block_specs:
        inst, src = _resolve_block(spec)
        if bool(inst.get('composite', False)):
            raise ValueError(
                "compose_blocks expects standalone Layer-2 block instances, "
                "not an already composed transient structure."
            )
        schema_blocks = inst.get('_schema_blocks')
        if isinstance(schema_blocks, list) and len(schema_blocks) != 1:
            raise ValueError(
                "compose_blocks expects each source JSON to represent one "
                "standalone Layer-2 block. A unified multi-block instance "
                "cannot be nested as a source block."
            )
        if int(inst.get('d', 0)) <= 0:
            raise ValueError("Every source block must have a positive dimension.")
        blocks.append(inst)
        source_paths.append(str(src) if src is not None else None)
    J = len(blocks)

    block_dims = [int(b['d']) for b in blocks]
    D = int(sum(block_dims))

    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    # --- permutation ------------------------------------------------------
    if permutation is None:
        pi = rng.permutation(D).astype(int)
    else:
        pi = np.asarray(list(permutation), dtype=int)
        if pi.size != D:
            raise ValueError(
                f"permutation has length {pi.size} but composite dimension is {D}."
            )
        if sorted(pi.tolist()) != list(range(D)):
            raise ValueError("permutation must be a permutation of [0, D).")

    indices_per_block: List[List[int]] = []
    offset = 0
    for dj in block_dims:
        indices_per_block.append(pi[offset:offset + dj].tolist())
        offset += dj

    # --- weights ----------------------------------------------------------
    if weights is None:
        w = [1.0] * J
    elif isinstance(weights, str):
        if weights.lower() != 'random':
            raise ValueError(
                f"Unknown weights mode {weights!r}. Use None, 'random', or a list."
            )
        lo, hi = float(weight_random_range[0]), float(weight_random_range[1])
        if not (lo > 0 and hi >= lo):
            raise ValueError("weight_random_range must be (low, high) with 0 < low <= high.")
        w = rng.uniform(lo, hi, size=J).tolist()
    else:
        if len(weights) != J:
            raise ValueError(
                f"weights has length {len(weights)} but block_specs has length {J}."
            )
        w = [float(v) for v in weights]
    if any(not np.isfinite(value) or value <= 0.0 for value in w):
        raise ValueError("Every block weight must be finite and strictly positive.")

    # --- bounds -----------------------------------------------------------
    if bounds is None:
        bmin = float(blocks[0].get('bounds_min', -100))
        bmax = float(blocks[0].get('bounds_max', 100))
        for j, block in enumerate(blocks[1:], start=2):
            block_bounds = (
                float(block.get('bounds_min', -100)),
                float(block.get('bounds_max', 100)),
            )
            if block_bounds != (bmin, bmax):
                raise ValueError(
                    f"Source block {j} uses bounds {block_bounds}, whereas "
                    f"source block 1 uses {(bmin, bmax)}. Supply explicit "
                    "global bounds only if this difference is intentional."
                )
    else:
        if len(bounds) != 2:
            raise ValueError("bounds must contain exactly [lower, upper].")
        bmin, bmax = float(bounds[0]), float(bounds[1])
    if not np.isfinite(bmin) or not np.isfinite(bmax) or bmin >= bmax:
        raise ValueError("Global bounds must be finite and satisfy lower < upper.")

    return {
        'composite': True,
        'seed': int(seed) if seed is not None else int(blocks[0].get('Seed', 0) or 0),
        'd': D,
        'J': J,
        'bounds_min': bmin,
        'bounds_max': bmax,
        'warn_out_of_bounds': any(bool(b.get('warn_out_of_bounds', True)) for b in blocks),
        'weights': w,
        'permutation': pi.tolist(),
        'indices_per_block': indices_per_block,
        'blocks': blocks,
        '_source_paths': source_paths,
    }


def fitness_composite(x: np.ndarray, composite: Dict[str, Any]) -> float:
    """
    Evaluate a composite instance: F(x) = sum_j w_j * f_j(x[I_j]).

    `x` is the global D-dimensional solution vector (algorithm view, with the
    composite's permutation already applied implicitly: the algorithm never
    sees block boundaries). The per-block out-of-bounds warning behavior is
    delegated to fitness().

    Evaluation is serial; per-run / per-algorithm parallelism is intentionally
    left to a higher driver layer.
    """
    if not composite.get('composite', False):
        raise ValueError(
            "fitness_composite expects a composite instance "
            "(missing 'composite': True). Use fitness() for single-block instances."
        )

    x = np.asarray(x, dtype=float).ravel()
    if x.size != int(composite['d']):
        raise ValueError(
            f"Solution dimension mismatch. Expected {composite['d']}, got {x.size}."
        )

    total = 0.0
    weights = composite['weights']
    blocks = composite['blocks']
    for j, idx in enumerate(composite['indices_per_block']):
        sub_x = x[np.asarray(idx, dtype=int)]
        total += float(weights[j]) * float(_block_fitness(sub_x, blocks[j]))
    return total


def fitness_composite_batch(
    x: np.ndarray,
    composite: Dict[str, Any],
    evaluation_device: str = None,
) -> np.ndarray:
    """Evaluate a composite instance over a batch of global samples."""
    if not composite.get('composite', False):
        raise ValueError(
            "fitness_composite_batch expects a composite instance "
            "(missing 'composite': True). Use fitness_batch() for single-block instances."
        )

    samples = np.asarray(x, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    elif (
        samples.ndim == 2
        and samples.shape[1] != int(composite['d'])
        and samples.shape[0] == int(composite['d'])
    ):
        samples = samples.reshape(1, -1)
    samples = samples.reshape(samples.shape[0], -1)
    if samples.shape[1] != int(composite['d']):
        raise ValueError(
            f"Solution dimension mismatch. Expected {composite['d']}, got {samples.shape[1]}."
        )

    total = np.zeros(samples.shape[0], dtype=float)
    weights = composite['weights']
    blocks = composite['blocks']
    for j, idx in enumerate(composite['indices_per_block']):
        sub_x = samples[:, np.asarray(idx, dtype=int)]
        total += float(weights[j]) * _block_fitness_batch(
            sub_x,
            blocks[j],
            evaluation_device=evaluation_device,
        )
    return total


def save_composite(
    composite: Dict[str, Any],
    filename: Optional[str] = None,
    output_dir: Optional[PathLike] = None,
    remove_block_files: bool = False,
) -> str:
    """Save a composed Layer-3 instance as one unified JSON.

    This is the public script-facing composition writer.  It intentionally
    produces no ``composite: true`` envelope; the result loads through
    :func:`portal_core.load_instance.load_instance` and evaluates through the
    standard unified fitness path.  ``load_composite`` remains available only
    for backward-compatible reading of older envelope files.
    """
    if filename is None or filename == "":
        filename = f"PORTAL_d{composite['d']}_B{composite['J']}.json"
    return _save_composite_unified(
        composite,
        filename=filename,
        output_dir=output_dir,
        remove_block_files=remove_block_files,
    )


def save_legacy_composite(
    composite: Dict[str, Any],
    filename: Optional[str] = None,
    output_dir: Optional[PathLike] = None,
    remove_block_files: bool = False,
) -> str:
    """
    Persist the deprecated ``composite: true`` envelope format.

    The output JSON inlines every block's fully serialized form, so this single
    file is sufficient for downstream loading; no `blocks/` sidecar directory
    is created.

    Args:
        composite: dict produced by compose_blocks.
        filename: output file name (default auto-generated).
        output_dir: target directory (default Python/Utils/Instances).
        remove_block_files: if True, delete the original block JSON files that
            were passed to compose_blocks (only those given as paths and that
            still exist on disk). Default False keeps the originals.

    Returns:
        Full path to the written composite JSON.
    """
    if filename is None or filename == '':
        filename = f"PORTAL_composite_d{composite['d']}_J{composite['J']}.json"
    if not filename.endswith('.json'):
        filename = Path(filename).stem + '.json'

    instances_dir = Path(output_dir) if output_dir else _HERE / 'Instances'
    instances_dir.mkdir(parents=True, exist_ok=True)
    filepath = instances_dir / filename

    blocks_jsonable = [_block_to_jsonable(b) for b in composite['blocks']]

    payload = {
        'composite': True,
        'd': int(composite['d']),
        'J': int(composite['J']),
        'bounds_min': float(composite['bounds_min']),
        'bounds_max': float(composite['bounds_max']),
        'warn_out_of_bounds': bool(composite.get('warn_out_of_bounds', True)),
        'weights': [float(w) for w in composite['weights']],
        'permutation': [int(i) for i in composite['permutation']],
        'indices_per_block': [[int(i) for i in idx] for idx in composite['indices_per_block']],
        'blocks': blocks_jsonable,
    }

    # Forward optional provenance metadata so the saved JSON is self-describing
    if isinstance(composite.get('_suite'), dict):
        payload['_suite'] = dict(composite['_suite'])
    if isinstance(composite.get('_studio'), dict):
        payload['_studio'] = dict(composite['_studio'])
    if isinstance(composite.get('_generation'), dict):
        payload['_generation'] = dict(composite['_generation'])

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Legacy composite envelope saved to:\n  {filepath}")

    if remove_block_files:
        removed: List[str] = []
        for src in composite.get('_source_paths', []) or []:
            if not src:
                continue
            try:
                p = Path(src)
                if p.exists() and p.resolve() != filepath.resolve():
                    p.unlink()
                    removed.append(str(p))
            except OSError as e:
                print(f"  Warning: could not remove block file {src}: {e}")
        if removed:
            print(f"Removed {len(removed)} original block file(s):")
            for r in removed:
                print(f"  {r}")

    return str(filepath)


def load_composite(filepath: PathLike) -> Dict[str, Any]:
    """
    Load a legacy self-contained composite envelope.

    No external block files are required.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        candidate = _HERE / 'Instances' / filepath
        if candidate.exists():
            filepath = candidate
        else:
            raise FileNotFoundError(f"Composite file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    if not payload.get('composite', False):
        raise ValueError(f"{filepath} is not a composite file.")

    blocks = [_jsonable_to_block(b) for b in payload['blocks']]

    D = int(payload['d'])
    J = int(payload['J'])
    weights = [float(w) for w in payload['weights']]
    indices_per_block = [[int(i) for i in idx] for idx in payload['indices_per_block']]
    permutation = [int(i) for i in payload['permutation']]
    if J <= 0 or len(blocks) != J:
        raise ValueError("Legacy composite has an invalid block count.")
    if len(weights) != J or len(indices_per_block) != J:
        raise ValueError("Legacy composite blocks, weights, and indices have inconsistent lengths.")
    if any(not np.isfinite(w) or w <= 0.0 for w in weights):
        raise ValueError("Legacy composite weights must be finite and strictly positive.")
    if permutation != [i for idx in indices_per_block for i in idx]:
        raise ValueError("Legacy composite permutation does not match indices_per_block.")
    if sorted(permutation) != list(range(D)):
        raise ValueError(
            "Legacy composite dimensions must form an exact, non-overlapping "
            f"partition of [0, {D})."
        )
    for j, (block, indices) in enumerate(zip(blocks, indices_per_block)):
        if len(indices) != int(block['d']):
            raise ValueError(
                f"Legacy composite block {j} has {len(indices)} global indices "
                f"but dimension {int(block['d'])}."
            )

    composite = {
        'composite': True,
        'd': D,
        'J': J,
        'bounds_min': float(payload['bounds_min']),
        'bounds_max': float(payload['bounds_max']),
        'warn_out_of_bounds': bool(payload.get('warn_out_of_bounds', True)),
        'weights': weights,
        'permutation': permutation,
        'indices_per_block': indices_per_block,
        'block_ids': list(range(1, J + 1)),
        'blocks': blocks,
        '_source_paths': [None] * J,
    }
    for metadata_key in ('_suite', '_studio', '_generation'):
        if isinstance(payload.get(metadata_key), dict):
            composite[metadata_key] = payload[metadata_key]
    print(f"Composite instance loaded from:\n  {filepath}")
    return composite


def unified_payload_to_composite(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Split a unified multi-block instance payload into a composite runtime.

    Inverse of ``portal_core.save_instance.composite_to_unified_payload``: given
    a unified-schema JSON payload (top-level ``blocks``, each carrying its own
    ``dim_indices_one_based``, ``weight`` and locally-indexed components), rebuild
    the in-memory composite structure (independent single-block instances plus
    the global dimension map) that Studio tools consume.

    Each block is re-parsed through the canonical unified loader
    (``_from_unified_schema``) with its indices reset to a local ``1..d_b`` range,
    so no parameter slicing is re-implemented here. The result evaluates
    identically to ``fitness`` on the original unified instance, because
    ``fitness_composite`` reproduces ``sum_j w_j * min_k phi_j`` block-for-block.

    This lets GUI surfaces using a composite runtime (e.g. the 2-D editor's
    landscape preview) read Studio's unified JSON without a composite envelope
    on disk.
    """
    from portal_core.load_instance import _from_unified_schema, fix_parameter_dimensions

    if payload.get("composite", False):
        raise ValueError(
            "unified_payload_to_composite expects a unified instance payload, "
            "not a composite envelope."
        )
    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError("Unified payload has no 'blocks' array.")

    D = int(payload["d"])
    bmin = float(payload.get("bounds_min", -100))
    bmax = float(payload.get("bounds_max", 100))
    warn = bool(payload.get("warn_out_of_bounds", False))
    seed = int(payload.get("Seed", 0) or 0)

    block_runtimes: List[Dict[str, Any]] = []
    indices_per_block: List[List[int]] = []
    weights: List[float] = []
    block_ids: List[int] = []
    for j, blk in enumerate(raw_blocks):
        block_d = int(blk.get("d", len(blk.get("dim_indices_one_based", []))))
        gidx_one = blk.get("dim_indices_one_based")
        if not isinstance(gidx_one, list) or len(gidx_one) != block_d:
            raise ValueError(
                f"Block {j}: dim_indices_one_based missing or wrong length "
                f"(expected {block_d})."
            )
        gidx = [int(i) - 1 for i in gidx_one]
        if len(set(gidx)) != len(gidx) or any(i < 0 or i >= D for i in gidx):
            raise ValueError(
                f"Block {j}: dim_indices_one_based must contain unique values "
                f"in [1, {D}]."
            )
        block_ids.append(int(blk.get("block_id", j + 1)))

        weight = float(blk.get("weight", 1.0))
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Block {j}: weight must be finite and strictly positive.")

        # Re-index this block to a standalone local 1..d_b frame and parse it as
        # its own single-block unified instance. Its internal Layer-3 weight is
        # reset to 1.0: the block's real weight is carried in the composite's
        # ``weights`` list and applied by ``fitness_composite``. Leaving the
        # original weight here would double-count it (weight applied once inside
        # the single-block fitness and again by the composite aggregation).
        local_block = dict(blk)
        local_block["block_id"] = 1
        local_block["weight"] = 1.0
        local_block["dim_indices_one_based"] = list(range(1, block_d + 1))
        single_payload = {
            "Seed": seed,
            "d": block_d,
            "B": 1,
            "block_dims": [block_d],
            "bounds_min": bmin,
            "bounds_max": bmax,
            "warn_out_of_bounds": warn,
            "blocks": [local_block],
        }
        block_rt = _from_unified_schema(single_payload)
        block_rt["P"] = fix_parameter_dimensions(block_rt["P"], block_rt["K"], block_rt["d"])
        block_runtimes.append(block_rt)
        indices_per_block.append(gidx)
        weights.append(weight)

    permutation: List[int] = []
    for idx in indices_per_block:
        permutation.extend(idx)
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("Unified block IDs must be unique.")
    if sorted(permutation) != list(range(D)):
        raise ValueError(
            "Unified block dimensions must form an exact, non-overlapping "
            f"partition of global dimensions [1, {D}]."
        )

    composite: Dict[str, Any] = {
        "composite": True,
        "seed": seed,
        "d": D,
        "J": len(block_runtimes),
        "bounds_min": bmin,
        "bounds_max": bmax,
        "warn_out_of_bounds": warn,
        "weights": weights,
        "permutation": permutation,
        "indices_per_block": indices_per_block,
        "block_ids": block_ids,
        "blocks": block_runtimes,
        "_source_paths": [None] * len(block_runtimes),
    }
    if isinstance(payload.get("_studio"), dict):
        composite["_studio"] = payload["_studio"]
    if isinstance(payload.get("_suite"), dict):
        composite["_suite"] = payload["_suite"]
    if isinstance(payload.get("_generation"), dict):
        composite["_generation"] = payload["_generation"]
    identity = {
        key: payload[key]
        for key in ("family", "recipe")
        if payload.get(key) is not None
    }
    if identity:
        composite["_identity"] = identity
    return composite
