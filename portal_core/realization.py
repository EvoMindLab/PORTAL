"""
PORTAL Realization Generator
============================

Generates in-memory realizations (clones) of a PORTAL anchor instance.

Methods:
  1) method='center_beta'  (Type 1: minimal-perturbation repeated instances)
       - A single per-dimension shift vector is added to every component center
         within a block (a rigid translation of the landscape), repaired by
         reflection off the center bounds so that no center leaves
         [CENTER_MIN, CENTER_MAX].
       - A common vertical beta offset is applied, preserving relative
         inter-component basin-depth gaps.
       - No rotation or coordinate permutation is used, so basin shapes and
         variable interactions are unchanged. ELA values can nevertheless
         vary because translating an asymmetric landscape changes the portion
         exposed inside the finite search domain.

  2) method='diverse'  (Type 2: structure-diversifying instances)
       - For multi-block instances, a deterministic global coordinate
         permutation is applied first (a consistent isometry that relabels
         dimensions across centers, conditioning, exponents, interactions, and
         transform fields together).
       - A common beta offset and a per-block random orthogonal rotation of the
         component centers are applied; the rotation preserves pairwise center
         distances but not each basin's orientation relative to its position,
         so it deliberately changes landscape geometry.
       - Then, per component, signed per-dimension pairs are shuffled with random
         per-direction swaps (kappa/p and signed transform fields).
       - For multi-block recipes only, the existing block weights are
         reassigned by a deterministic derangement; no new weights are sampled.
       - rho fields are recomputed after shuffling.

  3) method='augment'  (Type 3: large-scale dataset augmentation, NOT a
     difficulty-preserving realization)
       - Same diversification machinery as Type 2 (global coordinate
         permutation and block-weight reassignment for multi-block instances,
         signed per-dimension shuffles, and rho recomputation), except that the
         Type 2 spatial step is replaced by RESAMPLING: every component
         center is drawn uniformly from a user-supplied center range, and
         basin depths follow the frozen-catalog rule of one shared base per
         block drawn from the user-supplied beta range plus per-component
         offsets bounded by ``beta_gap`` (default 20); one random component
         sits at the base and all other offsets stay at least ``beta_opt_gap``
         (default 1) above it, keeping within-block depth gaps compatible with
         the no-cover condition and decoy basins away from the optimum.
       - Because center placement and basin depths define the multimodal and
         deception structure, Type 3 generates DIFFERENT problems within the
         recipe's structural family; it is intended for building large
         meta-learning datasets from a bank of curated recipes, not for
         repeated-run evaluation on a fixed suite.
       - Each center/beta proposal is accepted only when every component center
         remains exposed in its Layer-2 block; covered proposals are redrawn
         under a finite, recorded attempt limit.

Types 1 and 2 are deterministically keyed by (base_seed, sample_id). Type 3
also incorporates a recipe-specific key so that equal sample identifiers from
different source recipes do not reuse the same augmentation stream.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_PYTHON = _HERE.parent
if str(_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYTHON))

try:  # Package imports (normal PORTAL use).
    from .benchmark_generator import precompute_scalings  # noqa: E402
    from .transform_registry import TRANSFORM_PER_DIM_FIELDS as _T_PER_DIM_FIELDS  # noqa: E402
    from .transform_registry import iter_transform_blocks  # noqa: E402
    from .validate_no_cover import validate_no_cover_block  # noqa: E402
except ImportError:  # Direct-script compatibility.
    from benchmark_generator import precompute_scalings  # type: ignore  # noqa: E402
    from transform_registry import TRANSFORM_PER_DIM_FIELDS as _T_PER_DIM_FIELDS  # type: ignore  # noqa: E402
    from transform_registry import iter_transform_blocks  # type: ignore  # noqa: E402
    from validate_no_cover import validate_no_cover_block  # type: ignore  # noqa: E402


PORTAL_SEED_TRAIN: int = 42
PORTAL_SEED_TEST: int = 2048

CENTER_MIN: float = -80.0
CENTER_MAX: float = +80.0
BETA_MIN: float = -1000.0
BETA_MAX: float = +1000.0
DEFAULT_BETA_GAP: float = 20.0  # frozen-catalog within-block depth span
DEFAULT_BETA_OPT_GAP: float = 1.0  # frozen-catalog optimum-protection margin
NO_COVER_MAX_ATTEMPTS: int = 1000
NO_COVER_TOLERANCE: float = 1e-6
_BLOCK_WEIGHT_STREAM_TAG: int = 0x42574D53  # "BWMS"

METHOD_CENTER_BETA = "center_beta"
METHOD_DIVERSE = "diverse"
METHOD_AUGMENT = "augment"
_VALID_METHODS = {METHOD_CENTER_BETA, METHOD_DIVERSE, METHOD_AUGMENT}


def _seed_words(base_seed: int,
                sample_id: int,
                recipe_seed_key: Optional[int] = None) -> List[int]:
    """Return the portable uint32 entropy words for a realization stream."""
    words = [int(base_seed) & 0xFFFFFFFF]
    if recipe_seed_key is not None:
        key = int(recipe_seed_key) & 0xFFFFFFFFFFFFFFFF
        words.extend([key & 0xFFFFFFFF, (key >> 32) & 0xFFFFFFFF])
    words.append(int(sample_id) & 0xFFFFFFFF)
    return words


def make_realization_seed(base_seed: int,
                          sample_id: int,
                          recipe_seed_key: Optional[int] = None) -> np.random.SeedSequence:
    """Build a deterministic SeedSequence for a realization stream.

    ``recipe_seed_key`` is used by Type 3 augmentation only. Its inclusion
    separates otherwise identical sample identifiers drawn from different
    source recipes, while Types 1 and 2 retain their established
    ``(base_seed, sample_id)`` schedule.
    """
    return np.random.SeedSequence(_seed_words(base_seed, sample_id, recipe_seed_key))


def _default_augmentation_recipe_key(anchor: Dict[str, Any],
                                     family_id: Optional[int],
                                     recipe_index: Optional[int]) -> int:
    """Derive a stable fallback Type 3 key from anchor identity metadata."""
    suite = anchor.get("_suite", {}) or {}
    family = family_id if family_id is not None else suite.get("family_id")
    recipe = recipe_index if recipe_index is not None else suite.get("recipe_index")
    if family is not None and recipe is not None:
        identity = f"family={int(family)};recipe={int(recipe)}"
    else:
        identity = (
            f"seed={anchor.get('Seed', '')};d={anchor.get('d', '')};"
            f"K={anchor.get('K', '')};file={suite.get('instance_file', '')}"
        )
    return int.from_bytes(
        hashlib.blake2b(identity.encode("utf-8"), digest_size=8).digest(),
        byteorder="big",
        signed=False,
    )


def _rng_from_anchor(anchor: Dict[str, Any],
                     base_seed: int,
                     sample_id: int,
                     family_id: Optional[int],
                     recipe_index: Optional[int],
                     method: str,
                     recipe_seed_key: Optional[int]) -> Tuple[np.random.Generator, List[int]]:
    """Return a deterministic RNG and its recorded seed key."""
    if method == METHOD_AUGMENT:
        key = recipe_seed_key
        if key is None:
            key = _default_augmentation_recipe_key(anchor, family_id, recipe_index)
        words = _seed_words(base_seed, sample_id, key)
    else:
        words = _seed_words(base_seed, sample_id)
    return np.random.default_rng(np.random.SeedSequence(words)), words


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.arange(n, dtype=int)
    for _ in range(256):
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p
    return np.roll(np.arange(n, dtype=int), 1)


def _random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    if d <= 1:
        return np.eye(max(d, 1), dtype=float)
    A = rng.normal(0.0, 1.0, size=(d, d))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    return Q


def _reflect_into_interval(x: np.ndarray,
                           lo: np.ndarray,
                           hi: np.ndarray) -> np.ndarray:
    """Fold ``x`` into ``[lo, hi]`` element-wise by reflection (mirror repair).

    An overshoot past a bound is reflected back inside rather than clamped to it,
    so values do not pile up on the boundary (as in reflecting boundary handling
    for particle swarms).  Arbitrary magnitudes are handled by periodic
    triangular folding.  Where ``hi == lo`` the only feasible value ``lo`` is
    returned.
    """
    x = np.asarray(x, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    out = lo.copy()
    width = hi - lo
    pos = width > 0.0
    if np.any(pos):
        y = np.mod(x[pos] - lo[pos], 2.0 * width[pos])
        y = np.where(y > width[pos], 2.0 * width[pos] - y, y)
        out[pos] = lo[pos] + y
    return out


def _normalize_schema_blocks(inst: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return normalized block metadata with component ranges and dim indices."""
    d = int(inst["d"])
    K = int(inst["K"])
    schema = inst.get("_schema_blocks")

    if not isinstance(schema, list) or len(schema) == 0:
        schema = [{"block_id": 1, "K": K, "weight": 1.0, "d": d, "dim_indices": list(range(d))}]
        # Keep the normalized records and the metadata stored on the instance
        # as the same objects. Later writes through ``blk['meta']`` must be
        # visible in ``inst['_schema_blocks']``.
        inst["_schema_blocks"] = schema

    blocks: List[Dict[str, Any]] = []
    cursor_k = 0
    cursor_d = 0
    for bi, meta in enumerate(schema):
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid _schema_blocks[{bi}] entry.")

        k_block = int(meta.get("K", 0))
        if k_block <= 0:
            raise ValueError(f"_schema_blocks[{bi}].K must be positive.")

        block_d = int(meta.get("d", d))
        if block_d <= 0 or block_d > d:
            raise ValueError(f"_schema_blocks[{bi}].d must be in [1, d].")

        dim_idx = meta.get("dim_indices")
        if dim_idx is None:
            if len(schema) == 1 and block_d == d:
                dim_idx = list(range(d))
            else:
                stop = cursor_d + block_d
                if stop > d:
                    raise ValueError("Cannot infer block dim indices from _schema_blocks.")
                dim_idx = list(range(cursor_d, stop))
        else:
            if not isinstance(dim_idx, (list, tuple)):
                raise ValueError(f"_schema_blocks[{bi}].dim_indices must be a list.")
            dim_idx = [int(x) for x in dim_idx]

        if len(dim_idx) != block_d:
            raise ValueError(f"_schema_blocks[{bi}] dim_indices length mismatch with block d.")
        if any(i < 0 or i >= d for i in dim_idx):
            raise ValueError(f"_schema_blocks[{bi}] dim_indices out of range.")
        if len(set(dim_idx)) != len(dim_idx):
            raise ValueError(f"_schema_blocks[{bi}] dim_indices must be unique.")

        block = {
            "meta": meta,
            "block_id": int(meta.get("block_id", bi + 1)),
            "K": k_block,
            "d": block_d,
            "weight": float(meta.get("weight", 1.0)),
            "start": cursor_k,
            "end": cursor_k + k_block,
            "dim_idx": np.asarray(dim_idx, dtype=int),
        }
        blocks.append(block)

        meta["d"] = block_d
        meta["dim_indices"] = [int(x) for x in dim_idx]

        cursor_k += k_block
        cursor_d += block_d

    if cursor_k != K:
        raise ValueError(f"_schema_blocks K-partition sums to {cursor_k}, expected K={K}.")

    return blocks


def _precompute_scalings_blockwise(
    inst: Dict[str, Any], blocks: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute neutralization factors in each block's local dimension.

    Multi-block runtime instances store component vectors in global
    coordinates, but neutralization is a Layer-2 operation. Constructing a
    minimal local instance for the shared scaling routine avoids depending on
    the values used to fill inactive, out-of-block coordinates.
    """
    P = inst["P"]
    d_global = int(inst["d"])
    K_global = int(inst["K"])

    form = np.zeros(K_global, dtype=int)
    rho_single = np.full(K_global, np.nan, dtype=float)
    # Zero is the canonical out-of-block fill used by the unified loader.
    rho_plus = np.zeros((K_global, d_global), dtype=float)
    rho_minus = np.zeros((K_global, d_global), dtype=float)

    neutralize = np.asarray(
        P.get("neutralize_enable", np.ones(K_global, dtype=bool)), dtype=bool
    ).reshape(-1)
    if neutralize.size == 1:
        neutralize = np.repeat(neutralize, K_global)
    if neutralize.size != K_global:
        raise ValueError(
            "neutralize_enable must be scalar or have one value per component."
        )

    p_single_all = np.asarray(P["p_single"], dtype=float)
    for blk in blocks:
        start = int(blk["start"])
        end = int(blk["end"])
        dim_idx = np.asarray(blk["dim_idx"], dtype=int)
        component_idx = np.arange(start, end, dtype=int)

        if p_single_all.ndim >= 2:
            p_single_local = p_single_all[component_idx, :]
        else:
            p_single_local = p_single_all[component_idx]

        local_instance = {
            "K": int(end - start),
            "d": int(blk["d"]),
            "baseline": list(inst["baseline"][start:end]),
            "P": {
                "kappa_plus": np.asarray(P["kappa_plus"], dtype=float)[
                    np.ix_(component_idx, dim_idx)
                ],
                "kappa_minus": np.asarray(P["kappa_minus"], dtype=float)[
                    np.ix_(component_idx, dim_idx)
                ],
                "p_plus": np.asarray(P["p_plus"], dtype=float)[
                    np.ix_(component_idx, dim_idx)
                ],
                "p_minus": np.asarray(P["p_minus"], dtype=float)[
                    np.ix_(component_idx, dim_idx)
                ],
                "p_single": p_single_local,
                "Delta": np.asarray(P["Delta"], dtype=float)[component_idx],
                "r_ref": np.asarray(P["r_ref"], dtype=float)[component_idx],
                "neutralize_enable": neutralize[component_idx],
            },
        }
        local_form, local_single, local_plus, local_minus = precompute_scalings(
            local_instance
        )
        form[component_idx] = local_form
        rho_single[component_idx] = local_single
        rho_plus[np.ix_(component_idx, dim_idx)] = local_plus
        rho_minus[np.ix_(component_idx, dim_idx)] = local_minus

    return form, rho_single, rho_plus, rho_minus


def _apply_global_dimension_permutation(inst: Dict[str, Any],
                                        blocks: List[Dict[str, Any]],
                                        rng: np.random.Generator) -> List[int]:
    """Apply one global coordinate permutation and update block dim-index metadata."""
    d = int(inst["d"])
    old_to_new = np.asarray(rng.permutation(d), dtype=int)

    P = inst["P"]

    for key in ("c", "kappa_plus", "kappa_minus", "p_plus", "p_minus", "rho_plus", "rho_minus"):
        if key not in P:
            continue
        arr = np.asarray(P[key], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != d:
            continue
        out = np.zeros_like(arr)
        out[:, old_to_new] = arr
        P[key] = out

    for key in ("Psi_deg", "R"):
        if key not in P:
            continue
        arr = np.asarray(P[key], dtype=float)
        if arr.ndim != 3 or arr.shape[0] != d or arr.shape[1] != d:
            continue
        out = np.zeros_like(arr)
        for k in range(arr.shape[2]):
            out[np.ix_(old_to_new, old_to_new, [k])] = arr[:, :, [k]]
        P[key] = out

    for blk_name, base_name, T in iter_transform_blocks(P):
        fields = _T_PER_DIM_FIELDS[base_name]
        for fld in fields:
            if fld not in T:
                continue
            arr = np.asarray(T[fld], dtype=float)
            if arr.ndim != 2 or arr.shape[1] != d:
                continue
            out = np.zeros_like(arr)
            out[:, old_to_new] = arr
            T[fld] = out

    for blk in blocks:
        new_idx = old_to_new[blk["dim_idx"]]
        blk["dim_idx"] = new_idx
        blk["meta"]["dim_indices"] = [int(x) for x in new_idx.tolist()]

    return [int(x) + 1 for x in old_to_new.tolist()]


def _shuffle_block_weights(blocks: List[Dict[str, Any]],
                           rng: np.random.Generator) -> Optional[Dict[str, Any]]:
    """Reassign the existing weight multiset among multiple Layer-3 blocks.

    A derangement prevents any block position from deliberately retaining its
    source assignment.  Equal-valued weights may still make some numerical
    assignments indistinguishable.  Single-block recipes return ``None`` and
    consume no random numbers.
    """
    if len(blocks) <= 1:
        return None

    old_weights = np.asarray([float(blk["weight"]) for blk in blocks], dtype=float)
    source_positions = _derangement(len(blocks), rng)
    new_weights = old_weights[source_positions]

    for blk, weight in zip(blocks, new_weights):
        value = float(weight)
        blk["weight"] = value
        blk["meta"]["weight"] = value

    return {
        "old_weights": old_weights.tolist(),
        "new_weights": new_weights.tolist(),
        "source_block_positions_one_based": [
            int(position) + 1 for position in source_positions.tolist()
        ],
        "numerically_changed": bool(not np.array_equal(old_weights, new_weights)),
    }


def _apply_beta_offset(P: Dict[str, Any],
                       start: int,
                       end: int,
                       rng: np.random.Generator) -> Dict[str, float]:
    """Apply one common vertical offset preserving inter-component basin-depth gaps."""
    betas = np.asarray(P["beta"][start:end], dtype=float)
    best_beta = float(np.min(betas))
    new_best = float(rng.uniform(BETA_MIN, BETA_MAX))
    delta = float(new_best - best_beta)
    P["beta"][start:end] = betas + delta
    return {"best_beta_old": best_beta, "best_beta_new": new_best, "beta_shift": delta}


def _block_record(blk: Dict[str, Any], mode: str) -> Dict[str, Any]:
    return {
        "block_id": int(blk["block_id"]),
        "K": int(int(blk["end"]) - int(blk["start"])),
        "d": int(blk["d"]),
        "dim_indices_one_based": [int(i) + 1 for i in blk["dim_idx"].tolist()],
        "mode": mode,
    }


def _apply_type1(inst: Dict[str, Any],
                 blocks: List[Dict[str, Any]],
                 rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Type 1 (center_beta): rigid bounded shift of centers plus a common beta offset.

    A single per-dimension shift vector is added to every component center in a
    block (a rigid translation), reflected into the feasible range so that no
    center leaves [CENTER_MIN, CENTER_MAX].  No rotation or permutation is used,
    so basin shapes and variable interactions are preserved.
    """
    P = inst["P"]
    details: List[Dict[str, Any]] = []

    for blk in blocks:
        start = int(blk["start"])
        end = int(blk["end"])
        idx = blk["dim_idx"]
        block_d = int(blk["d"])

        rec = _block_record(blk, "shift")
        rec.update(_apply_beta_offset(P, start, end, rng))

        centers = np.asarray(P["c"][start:end, :][:, idx], dtype=float)  # [k_block, block_d]
        cmin = centers.min(axis=0)
        cmax = centers.max(axis=0)
        t_lo = CENTER_MIN - cmin  # <= 0, feasible interval always contains 0
        t_hi = CENTER_MAX - cmax  # >= 0
        span = float(CENTER_MAX - CENTER_MIN)
        raw = rng.uniform(-span, span, size=block_d)
        shift = _reflect_into_interval(raw, t_lo, t_hi)

        for k in range(start, end):
            P["c"][k, idx] = np.asarray(P["c"][k, idx], dtype=float) + shift

        moved = np.asarray(P["c"][start:end, :][:, idx], dtype=float)
        viol = 0.0
        if moved.size:
            viol = float(max(0.0, moved.max() - CENTER_MAX, CENTER_MIN - moved.min()))
        rec["shift_vector"] = shift.tolist()
        rec["center_bound_max_violation"] = viol
        details.append(rec)

    return details


def _bounded_center_move(centers: np.ndarray,
                         rng: np.random.Generator,
                         lb: float,
                         ub: float,
                         attempts: int = 8) -> Tuple[np.ndarray, np.ndarray, str]:
    """Rotate a block's centers about the origin and repair them into [lb, ub].

    The rotation preserves pairwise center distances; a single common shift,
    reflected into the feasible range, then brings the whole (rigid) constellation
    inside the box without placing any center exactly on the boundary.  A rotation
    whose rotated per-dimension extent exceeds the box width cannot be fit by any
    rigid shift; such a rotation is rejected and re-drawn, and after ``attempts``
    failures the step falls back to a bounded rigid shift, which always fits because
    the original centers are in-bounds.  Returns ``(new_centers, Q, mode)``.
    """
    _, d = centers.shape
    box = float(ub - lb)
    for attempt in range(attempts):
        Q = _random_orthogonal(d, rng)
        rot = centers @ Q.T
        cmin = rot.min(axis=0)
        cmax = rot.max(axis=0)
        if np.all((cmax - cmin) <= box + 1e-9):
            t = _reflect_into_interval(np.zeros(d), lb - cmin, ub - cmax)
            return rot + t, Q, ("rotation" if attempt == 0 else "rotation_retry")

    # Fallback: bounded rigid shift (always fits since originals are in-bounds).
    cmin = centers.min(axis=0)
    cmax = centers.max(axis=0)
    raw = rng.uniform(-box, box, size=d)
    t = _reflect_into_interval(raw, lb - cmin, ub - cmax)
    return centers + t, np.eye(d), "shift_fallback"


def _apply_type2_spatial(inst: Dict[str, Any],
                         blocks: List[Dict[str, Any]],
                         rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Type 2 spatial policy: a common beta offset plus a per-block, bounded,
    distance-preserving rotation of the component centers about the origin.

    The rotation preserves pairwise center distances but not each basin's
    orientation relative to its position, so it deliberately alters landscape
    geometry; a common reflected shift keeps every center inside the admissible
    box.  It is reserved for the diverse (Type 2) mode.
    """
    P = inst["P"]
    details: List[Dict[str, Any]] = []

    for blk in blocks:
        start = int(blk["start"])
        end = int(blk["end"])
        idx = blk["dim_idx"]
        k_block = end - start

        rec = _block_record(blk, "rotation")
        rec.update(_apply_beta_offset(P, start, end, rng))

        centers_before = np.asarray(P["c"][start:end, :][:, idx], dtype=float).copy()
        moved, Q, mode = _bounded_center_move(centers_before, rng, CENTER_MIN, CENTER_MAX)
        for row, k in enumerate(range(start, end)):
            P["c"][k, idx] = moved[row]
        centers_after = np.asarray(P["c"][start:end, :][:, idx], dtype=float)

        dist_err = 0.0
        for i in range(k_block):
            for j in range(i + 1, k_block):
                dij_before = float(np.linalg.norm(centers_before[i] - centers_before[j]))
                dij_after = float(np.linalg.norm(centers_after[i] - centers_after[j]))
                dist_err = max(dist_err, abs(dij_after - dij_before))
        viol = 0.0
        if centers_after.size:
            viol = float(max(0.0, centers_after.max() - CENTER_MAX, CENTER_MIN - centers_after.min()))

        rec["spatial_mode"] = mode
        rec["center_rotation_matrix"] = Q.tolist()
        rec["center_pairwise_distance_max_abs_error"] = float(dist_err)
        rec["center_bound_max_violation"] = viol
        details.append(rec)

    return details


def _apply_type3_spatial(inst: Dict[str, Any],
                         blocks: List[Dict[str, Any]],
                         rng: np.random.Generator,
                         center_range: Tuple[float, float],
                         beta_range: Tuple[float, float],
                         beta_gap: float,
                         beta_opt_gap: float) -> List[Dict[str, Any]]:
    """Type 3 spatial policy: resample centers and basin depths per block.

    Every component center is drawn uniformly from ``center_range`` on its
    block-local dimensions.  Basin depths follow the shared-base rule used to
    sample the frozen PORTAL catalog: one base value per block is drawn from
    ``[beta_lo, beta_hi - gap]`` and per-component offsets from ``[0, gap]``,
    so within-block depth gaps never exceed ``beta_gap`` while the block's
    overall placement spans the full declared range.  Keeping the gap small
    relative to the minimum per-component rise scale (Delta) makes the draws
    compatible with the no-cover requirement.  This deliberately changes the
    multimodal arrangement and basin-depth structure, producing a different
    problem within the recipe's structural family; it is intended for
    large-scale dataset augmentation.
    """
    c_lo, c_hi = float(center_range[0]), float(center_range[1])
    b_lo, b_hi = float(beta_range[0]), float(beta_range[1])
    gap = float(beta_gap)
    if not (c_hi >= c_lo and b_hi >= b_lo):
        raise ValueError(
            f"Invalid ranges: center_range={center_range}, beta_range={beta_range}."
        )
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError(f"beta_gap must be finite and nonnegative; received {beta_gap!r}.")
    opt_gap = float(beta_opt_gap)
    if not np.isfinite(opt_gap) or opt_gap < 0.0:
        raise ValueError(
            f"beta_opt_gap must be finite and nonnegative; received {beta_opt_gap!r}.")

    span = b_hi - b_lo
    gap_eff = min(gap, span) if span > 0.0 else 0.0
    opt_gap_eff = min(opt_gap, gap_eff)

    P = inst["P"]
    details: List[Dict[str, Any]] = []

    for blk in blocks:
        start = int(blk["start"])
        end = int(blk["end"])
        idx = blk["dim_idx"]
        block_d = int(blk["d"])

        rec = _block_record(blk, "resample")
        base = float(rng.uniform(b_lo, b_hi - gap_eff))
        k_block = end - start
        # One randomly chosen component sits exactly at the base (the block
        # optimum); every other offset is confined to [opt_gap, gap], so no
        # decoy basin lies within beta_opt_gap of the optimum.
        offsets = opt_gap_eff + (gap_eff - opt_gap_eff) * rng.uniform(0.0, 1.0, size=k_block)
        best_local = int(rng.integers(k_block))
        offsets[best_local] = 0.0
        for row, k in enumerate(range(start, end)):
            P["c"][k, idx] = rng.uniform(c_lo, c_hi, size=block_d)
            P["beta"][k] = base + float(offsets[row])
        rec["center_range"] = [c_lo, c_hi]
        rec["beta_range"] = [b_lo, b_hi]
        rec["beta_base"] = base
        rec["beta_gap"] = gap_eff
        rec["beta_opt_gap"] = opt_gap_eff
        rec["best_component_one_based"] = int(start + best_local + 1)
        rec["betas_new"] = [float(P["beta"][k]) for k in range(start, end)]
        details.append(rec)

    return details


def _type3_failed_blocks(inst: Dict[str, Any],
                         blocks: List[Dict[str, Any]],
                         tol: float) -> List[Tuple[Dict[str, Any], List[Tuple[int, float, float, float]]]]:
    """Return Type 3 blocks containing at least one covered component center."""
    failed = []
    for blk in blocks:
        ok, offenders = validate_no_cover_block(
            inst,
            component_indices=range(int(blk["start"]), int(blk["end"])),
            dimension_indices=blk["dim_idx"],
            tol=tol,
        )
        if not ok:
            failed.append((blk, offenders))
    return failed


def _shuffle_signed_component_rows(minus_row: np.ndarray,
                                   plus_row: np.ndarray,
                                   rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Shuffle one signed pair row with derangement + direction swaps."""
    n = minus_row.size
    perm = _derangement(n, rng)
    m = minus_row[perm]
    p = plus_row[perm]
    swap = rng.integers(0, 2, size=n).astype(bool)
    out_minus = np.where(swap, p, m)
    out_plus = np.where(swap, m, p)
    meta = {"perm": perm.tolist(), "swap_mask": swap.astype(int).tolist()}
    return out_minus, out_plus, meta


def _apply_type2(inst: Dict[str, Any],
                 blocks: List[Dict[str, Any]],
                 rng: np.random.Generator) -> Dict[str, Any]:
    """Apply Type 2 shuffles per component and per active block dimensions."""
    P = inst["P"]
    details: Dict[str, Any] = {
        "components": {},
        "shuffle_counts": {
            "kappa": 0,
            "p": 0,
            "transforms": 0,
        },
    }

    def _shuffle_pair_for_component(k: int, idx: np.ndarray, minus_key: str, plus_key: str) -> Optional[Dict[str, Any]]:
        if minus_key not in P or plus_key not in P:
            return None
        minus_arr = np.asarray(P[minus_key], dtype=float)
        plus_arr = np.asarray(P[plus_key], dtype=float)
        if minus_arr.ndim != 2 or plus_arr.ndim != 2 or minus_arr.shape != plus_arr.shape:
            return None

        minus_row = np.asarray(minus_arr[k, idx], dtype=float)
        plus_row = np.asarray(plus_arr[k, idx], dtype=float)
        out_minus, out_plus, meta = _shuffle_signed_component_rows(minus_row, plus_row, rng)
        minus_arr[k, idx] = out_minus
        plus_arr[k, idx] = out_plus
        P[minus_key] = minus_arr
        P[plus_key] = plus_arr
        return meta

    # Assign RNG draws to unsuffixed blocks in canonical registry order, then
    # to independent occurrences in type/numeric occurrence order. This keeps
    # seeded landscapes independent of parameter-dictionary insertion order.
    type_order = {name: index for index, name in enumerate(_T_PER_DIM_FIELDS)}
    transform_blocks = sorted(
        iter_transform_blocks(P),
        key=lambda item: (
            item[0] != item[1],
            type_order[item[1]],
            int(item[0].partition('#')[2] or 1),
        ),
    )

    for blk in blocks:
        idx = blk["dim_idx"]
        for k in range(int(blk["start"]), int(blk["end"])):
            ck = details["components"].setdefault(str(int(k + 1)), {})

            kp_meta = _shuffle_pair_for_component(k, idx, "kappa_minus", "kappa_plus")
            if kp_meta is not None:
                ck["kappa"] = kp_meta
                details["shuffle_counts"]["kappa"] += 1

            p_meta = _shuffle_pair_for_component(k, idx, "p_minus", "p_plus")
            if p_meta is not None:
                ck["p"] = p_meta
                details["shuffle_counts"]["p"] += 1

            tf_details: Dict[str, Any] = {}
            for blk_name, base_name, T in transform_blocks:
                fields = _T_PER_DIM_FIELDS[base_name]

                plus_fields = [f for f in fields if f.endswith("_plus")]
                blk_meta: Dict[str, Any] = {}
                for plus_name in plus_fields:
                    minus_name = plus_name[:-5] + "_minus"
                    if minus_name not in fields:
                        continue
                    if plus_name not in T or minus_name not in T:
                        continue

                    plus_arr = np.asarray(T[plus_name], dtype=float)
                    minus_arr = np.asarray(T[minus_name], dtype=float)
                    if plus_arr.ndim != 2 or minus_arr.ndim != 2 or plus_arr.shape != minus_arr.shape:
                        continue

                    minus_row = np.asarray(minus_arr[k, idx], dtype=float)
                    plus_row = np.asarray(plus_arr[k, idx], dtype=float)
                    out_minus, out_plus, meta = _shuffle_signed_component_rows(minus_row, plus_row, rng)
                    minus_arr[k, idx] = out_minus
                    plus_arr[k, idx] = out_plus
                    T[minus_name] = minus_arr
                    T[plus_name] = plus_arr
                    blk_meta[f"{minus_name}|{plus_name}"] = meta

                if blk_meta:
                    tf_details[blk_name] = blk_meta

            if tf_details:
                ck["transforms"] = tf_details
                details["shuffle_counts"]["transforms"] += 1

    return details


def realize(anchor: Dict[str, Any],
            base_seed: int,
            sample_id: int,
            *,
            family_id: Optional[int] = None,
            recipe_index: Optional[int] = None,
            method: str = METHOD_CENTER_BETA,
            center_range: Optional[Tuple[float, float]] = None,
            beta_range: Optional[Tuple[float, float]] = None,
            beta_gap: Optional[float] = None,
            beta_opt_gap: Optional[float] = None,
            recipe_seed_key: Optional[int] = None,
            ensure_no_cover: bool = True,
            no_cover_max_attempts: int = NO_COVER_MAX_ATTEMPTS,
            no_cover_tolerance: float = NO_COVER_TOLERANCE,
            trace: bool = False) -> Dict[str, Any]:
    """Return a deep-copied realization of ``anchor``.

    Deterministic key: (base_seed, sample_id).
    ``center_range``/``beta_range`` apply only to method='augment' (Type 3)
    and default to the standard center/beta ranges. ``recipe_seed_key`` is
    optional for Type 3; when supplied, it identifies the source recipe in the
    augmentation RNG stream.  Type 3 rejects center/beta draws that cover any
    component center when ``ensure_no_cover`` is true.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unsupported realization method '{method}'. "
            f"Valid methods: {sorted(_VALID_METHODS)}"
        )
    if int(no_cover_max_attempts) < 1:
        raise ValueError("no_cover_max_attempts must be at least 1.")
    if not np.isfinite(no_cover_tolerance) or float(no_cover_tolerance) < 0.0:
        raise ValueError("no_cover_tolerance must be finite and nonnegative.")

    rng, seed_words = _rng_from_anchor(
        anchor, base_seed, sample_id, family_id, recipe_index, method, recipe_seed_key
    )

    inst = copy.deepcopy(anchor)
    # Realisations replace centres, rotations and parameters. Tensor caches
    # belong to the source values and must be rebuilt from the realised P.
    inst.pop('_torch_cache', None)
    inst.pop('_last_evaluation_backend', None)
    blocks = _normalize_schema_blocks(inst)

    suite = inst.setdefault("_suite", {})
    out_meta: Dict[str, Any] = {
        "base_seed": int(base_seed),
        "sample_id": int(sample_id),
        "method": str(method),
        "seed_key": seed_words,
        "trace_enabled": bool(trace),
    }
    if method == METHOD_AUGMENT:
        out_meta["recipe_seed_key"] = int(seed_words[1] | (seed_words[2] << 32))

    if method == METHOD_CENTER_BETA:
        # Type 1: rigid bounded shift + common beta offset (minimal perturbation).
        out_meta["type1"] = {"blocks": _apply_type1(inst, blocks, rng)}
    else:
        # Types 2 and 3 share the diversification machinery: global coordinate
        # permutation and block-weight reassignment (multi-block), a spatial
        # step, signed per-dimension shuffles, and rho recomputation.  They
        # differ only in the spatial step: Type 2 rotates the existing centers
        # (distance-preserving), whereas Type 3 independently resamples centers
        # and betas from user-supplied ranges (dataset augmentation).
        dim_perm_one_based: Optional[List[int]] = None
        if len(blocks) > 1:
            dim_perm_one_based = _apply_global_dimension_permutation(inst, blocks, rng)

        if method == METHOD_DIVERSE:
            spatial_details = _apply_type2_spatial(inst, blocks, rng)
        else:
            c_range = center_range if center_range is not None else (CENTER_MIN, CENTER_MAX)
            b_range = beta_range if beta_range is not None else (BETA_MIN, BETA_MAX)
            b_gap = float(beta_gap) if beta_gap is not None else DEFAULT_BETA_GAP
            b_opt = float(beta_opt_gap) if beta_opt_gap is not None else DEFAULT_BETA_OPT_GAP
            spatial_details = _apply_type3_spatial(
                inst, blocks, rng, c_range, b_range, b_gap, b_opt)

        shuffle_details = _apply_type2(inst, blocks, rng)

        form, rho_single, rho_plus, rho_minus = _precompute_scalings_blockwise(
            inst, blocks
        )
        inst["P"]["form"] = form
        inst["P"]["rho_single"] = rho_single
        inst["P"]["rho_plus"] = rho_plus
        inst["P"]["rho_minus"] = rho_minus

        no_cover_meta: Optional[Dict[str, Any]] = None
        if method == METHOD_AUGMENT:
            rounds = 1
            rejection_count = 0
            block_attempts = {int(blk["block_id"]): 1 for blk in blocks}
            block_rejections = {int(blk["block_id"]): 0 for blk in blocks}
            if ensure_no_cover:
                # Validate every Layer-2 block independently.  A rejection in
                # one block redraws only that block's centers and betas; the
                # Layer-3 weighting, coordinate layout, and evaluation are not
                # involved in this safeguard.
                failed = _type3_failed_blocks(
                    inst, blocks, float(no_cover_tolerance)
                )
                spatial_by_id = {
                    int(record["block_id"]): record for record in spatial_details
                }
                while failed and rounds < int(no_cover_max_attempts):
                    failed_blocks = [blk for blk, _ in failed]
                    for blk in failed_blocks:
                        block_id = int(blk["block_id"])
                        block_attempts[block_id] += 1
                        block_rejections[block_id] += 1
                        rejection_count += 1

                    updates = _apply_type3_spatial(
                        inst, failed_blocks, rng, c_range, b_range, b_gap, b_opt
                    )
                    for record in updates:
                        spatial_by_id[int(record["block_id"])] = record
                    spatial_details = [
                        spatial_by_id[int(blk["block_id"])] for blk in blocks
                    ]
                    rounds += 1
                    failed = _type3_failed_blocks(
                        inst, blocks, float(no_cover_tolerance)
                    )

                if failed:
                    block_labels = [int(blk["block_id"]) for blk, _ in failed]
                    indices = [
                        int(item[0]) + 1
                        for _, offenders in failed
                        for item in offenders
                    ][:8]
                    raise RuntimeError(
                        "Unable to produce a Type 3 augmentation with all "
                        "component centers exposed within every Layer-2 block "
                        f"after {int(no_cover_max_attempts)} proposal rounds. "
                        f"Affected block IDs are {block_labels}; covered component "
                        f"indices include {indices}. Adjust "
                        "center_range or beta_range, increase "
                        "no_cover_max_attempts, or explicitly set "
                        "ensure_no_cover=False when coverage is intentional."
                    )

            no_cover_meta = {
                "enforced": bool(ensure_no_cover),
                "validated": bool(ensure_no_cover),
                "tolerance": float(no_cover_tolerance),
                "attempts": int(rounds),
                "rejections": int(rejection_count),
                "blocks": [
                    {
                        "block_id": int(blk["block_id"]),
                        "attempts": int(block_attempts[int(blk["block_id"])]),
                        "rejections": int(block_rejections[int(blk["block_id"])]),
                    }
                    for blk in blocks
                ],
            }

        block_weight_shuffle: Optional[Dict[str, Any]] = None
        if len(blocks) > 1:
            # Use a tagged substream to separate weight reassignment from
            # center and parameter random streams.
            weight_rng = np.random.default_rng(
                np.random.SeedSequence([*seed_words, _BLOCK_WEIGHT_STREAM_TAG])
            )
            block_weight_shuffle = _shuffle_block_weights(blocks, weight_rng)

        if dim_perm_one_based is not None:
            out_meta["dimension_permutation_one_based"] = dim_perm_one_based
        key = "type2" if method == METHOD_DIVERSE else "type3"
        out_meta[key] = {"spatial": spatial_details, "shuffle": shuffle_details}
        if block_weight_shuffle is not None:
            out_meta[key]["block_weight_shuffle"] = block_weight_shuffle
        if no_cover_meta is not None:
            out_meta[key]["no_cover"] = no_cover_meta
        out_meta["rho_recomputed"] = True

    if trace:
        block_traces = []
        for blk in blocks:
            block_traces.append({
                "block_id": int(blk["block_id"]),
                "K": int(blk["K"]),
                "d": int(blk["d"]),
                "dim_indices_one_based": [int(i) + 1 for i in blk["dim_idx"].tolist()],
                "weight": float(blk["weight"]),
            })
        out_meta["blocks"] = block_traces

    suite["_realization"] = out_meta

    return inst
