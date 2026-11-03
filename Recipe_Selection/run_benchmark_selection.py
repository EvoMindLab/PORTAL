"""
PORTAL Fixed-Cardinality Benchmark Selection
============================================

Selects a fixed-cardinality subset from an MIS-filtered recipe pool using:

1. Constrained greedy facility location in joint ELA/Performance distance space.
2. Constraint-preserving local swap refinement on a coverage-and-spacing objective.

Distance normalization for selection:
  Pairwise Euclidean distances in each space are divided by the trimmed mean of
  all pairwise distances in that space (default trim fraction = 0.10, i.e. the
  central 80 % of pairs determine the scale).  This makes the trimmed-mean
  distance equal to 1 in both spaces so that equal weights (w_ELA = w_performance = 0.5)
  give genuinely equal average contribution from each representation.

  A plain mean (trim = 0) is equivalent; trimming is included to reduce the
  influence of the few very-far outlier pairs on the normalization scale.

The script is deterministic for a fixed input pool and parameter set.
"""

from __future__ import annotations

# Read-only compatibility for historical characterization artifacts.
try:
    from Recipe_Selection._terminology import read_characterization_csv
except ModuleNotFoundError as _terminology_error:
    if _terminology_error.name != 'Recipe_Selection':
        raise
    import sys as _source_sys
    from pathlib import Path as _SourcePath
    _source_sys.path.insert(0, str(_SourcePath(__file__).resolve().parents[1]))
    from Recipe_Selection._terminology import read_characterization_csv

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


FAMILY_ORDER = [
    item.strip()
    for item in os.environ.get(
        'PORTAL_FAMILY_ORDER', 'Family1,Family2,Family3,Family4'
    ).split(',')
    if item.strip()
]
_PORTAL_PAPER_FAMILIES = ['Family1', 'Family2', 'Family3', 'Family4']
if FAMILY_ORDER == _PORTAL_PAPER_FAMILIES:
    DEFAULT_FAMILY_LOWER_PCT = [12.0, 16.0, 20.0, 30.0]
    DEFAULT_FAMILY_UPPER_PCT = [16.0, 24.0, 30.0, 40.0]
else:
    DEFAULT_FAMILY_LOWER_PCT = [0.0] * len(FAMILY_ORDER)
    DEFAULT_FAMILY_UPPER_PCT = [100.0] * len(FAMILY_ORDER)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # project root (one level up from Recipe_Selection/)
PIPELINE_DIR = ROOT / 'pipeline_output'
NEAR_DUP_DIR = ROOT / 'near_dup_output'
OUT_DIR = ROOT / 'selection_output'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def default_pool_file() -> Path:
    preferred = NEAR_DUP_DIR / 'mis_pool_tau_0p05.csv'
    if preferred.exists():
        return preferred
    return NEAR_DUP_DIR / 'mis_pool_0.95.csv'


def parse_family_percentages(raw: str, label: str) -> dict[str, float]:
    values = [part.strip() for part in raw.split(',') if part.strip()]
    if len(values) != len(FAMILY_ORDER):
        raise ValueError(
            f'{label} must provide exactly {len(FAMILY_ORDER)} comma-separated percentages '
            f'in FAMILY_ORDER={FAMILY_ORDER}. Received: {raw!r}'
        )
    percentages = {family: float(value) for family, value in zip(FAMILY_ORDER, values)}
    for family, value in percentages.items():
        if not math.isfinite(value) or value < 0.0 or value > 100.0:
            raise ValueError(f'{label} percentage for {family} must lie in [0, 100]. Received: {value}')
    if label.endswith('family-lower') and sum(percentages.values()) > 100.0:
        raise ValueError(
            f'Sum of lower percentages ({sum(percentages.values()):.1f}%) exceeds 100%, '
            f'which would require more than k recipes in total.'
        )
    if label.endswith('family-upper') and sum(percentages.values()) < 100.0:
        raise ValueError(
            f'Sum of upper percentages ({sum(percentages.values()):.1f}%) is below 100%, '
            'so the bounds cannot contain a complete selection.'
        )
    return percentages


def derive_count_bounds(percentages: dict[str, float], k: int, mode: str) -> dict[str, int]:
    if mode not in {'lower', 'upper'}:
        raise ValueError(f'mode must be lower or upper. Received: {mode}')

    bounds: dict[str, int] = {}
    for family in FAMILY_ORDER:
        raw_count = k * percentages[family] / 100.0
        if mode == 'lower':
            bounds[family] = int(np.floor(raw_count + 1e-12))
        else:
            bounds[family] = int(np.ceil(raw_count - 1e-12))
    return bounds


def robust_mean_scale(distances: np.ndarray, trim_fraction: float) -> float:
    if not math.isfinite(trim_fraction) or not 0.0 <= trim_fraction < 0.5:
        raise ValueError(f'trim_fraction must be in [0, 0.5). Received: {trim_fraction}')
    if distances.size == 0:
        raise ValueError('Cannot compute a robust mean scale from an empty distance array.')

    sorted_dist = np.sort(distances)
    trim_n = int(np.floor(trim_fraction * sorted_dist.size))
    if trim_n == 0:
        scale = float(sorted_dist.mean())
    else:
        kept = sorted_dist[trim_n:-trim_n]
        if kept.size == 0:
            raise ValueError(
                f'trim_fraction={trim_fraction} trims away the entire distance array; '
                'choose a smaller value.'
            )
        scale = float(kept.mean())

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f'Robust mean scale must be positive and finite. Received: {scale}')
    return scale


def coverage_loss(distance_sq: np.ndarray, selected: list[int]) -> tuple[float, np.ndarray]:
    if not selected:
        return float('inf'), np.full(distance_sq.shape[0], np.inf)
    coverage = distance_sq[:, selected].min(axis=1)
    return float(coverage.sum()), coverage


def spacing_stats(distance_sq: np.ndarray, selected: list[int]) -> tuple[float, float]:
    if len(selected) <= 1:
        return 0.0, 0.0
    sub = distance_sq[np.ix_(selected, selected)].copy()
    np.fill_diagonal(sub, np.inf)
    nearest = sub.min(axis=1)
    upper = sub[np.triu_indices(len(selected), k=1)]
    return float(nearest.mean()), float(upper.min())


def composite_objective(
    distance_sq: np.ndarray,
    selected: list[int],
    spacing_lambda: float,
    spacing_mu: float,
) -> tuple[float, float, float, float]:
    loss, _ = coverage_loss(distance_sq, selected)
    mean_nnd, min_pair = spacing_stats(distance_sq, selected)
    score = -loss + spacing_lambda * mean_nnd + spacing_mu * min_pair
    return score, loss, mean_nnd, min_pair


def current_family_counts(selected: list[int], pool_families: list[str]) -> dict[str, int]:
    counts = {family: 0 for family in FAMILY_ORDER}
    for idx in selected:
        counts[pool_families[idx]] += 1
    return counts


def trial_family_counts(
    selected: list[int],
    pool_families: list[str],
    add_idx: int | None = None,
    remove_idx: int | None = None,
) -> dict[str, int]:
    counts = current_family_counts(selected, pool_families)
    if remove_idx is not None:
        counts[pool_families[remove_idx]] -= 1
    if add_idx is not None:
        counts[pool_families[add_idx]] += 1
    return counts


def selection_feasible(
    counts: dict[str, int],
    lower_bounds: dict[str, int],
    upper_bounds: dict[str, int],
    slots_left: int,
) -> bool:
    if any(counts[family] > upper_bounds[family] for family in FAMILY_ORDER):
        return False

    unmet_lower = sum(max(lower_bounds[family] - counts[family], 0) for family in FAMILY_ORDER)
    remaining_capacity = sum(max(upper_bounds[family] - counts[family], 0) for family in FAMILY_ORDER)
    return unmet_lower <= slots_left <= remaining_capacity


def final_counts_valid(
    counts: dict[str, int],
    lower_bounds: dict[str, int],
    upper_bounds: dict[str, int],
) -> bool:
    return all(lower_bounds[family] <= counts[family] <= upper_bounds[family] for family in FAMILY_ORDER)


def format_bounds(bounds: dict[str, int]) -> str:
    return ', '.join(f'{family}={bounds[family]}' for family in FAMILY_ORDER)


def format_percentages(percentages: dict[str, float]) -> str:
    return ', '.join(f'{family}={percentages[family]:.1f}%' for family in FAMILY_ORDER)


def align_cleaned_feature_tables(
    performance_df: pd.DataFrame,
    ela_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and align cleaned Performance/ELA tables by recipe_id."""
    meta_cols = ['recipe_id', 'family', 'family_id']
    for name, frame in [('Performance', performance_df), ('ELA', ela_df)]:
        missing = [col for col in meta_cols if col not in frame.columns]
        if missing:
            raise ValueError(f'{name} table is missing required columns: {missing}')
        if frame['recipe_id'].astype(str).duplicated().any():
            raise ValueError(f'{name} table contains duplicate recipe_id values.')

    performance = performance_df.copy()
    ela = ela_df.copy()
    performance['recipe_id'] = performance['recipe_id'].astype(str)
    ela['recipe_id'] = ela['recipe_id'].astype(str)

    performance_ids = performance['recipe_id'].tolist()
    if set(performance_ids) != set(ela['recipe_id']):
        missing_ela = sorted(set(performance_ids) - set(ela['recipe_id']))
        missing_performance = sorted(set(ela['recipe_id']) - set(performance_ids))
        raise ValueError(
            'Performance/ELA recipe sets differ. '
            f'Missing from ELA: {missing_ela[:5]}; missing from Performance: {missing_performance[:5]}.'
        )

    ela = ela.set_index('recipe_id', drop=False).loc[performance_ids].reset_index(drop=True)
    if not np.array_equal(
        performance['family'].astype(str).to_numpy(),
        ela['family'].astype(str).to_numpy(),
    ):
        raise ValueError('Performance/ELA family metadata disagree after recipe_id alignment.')
    if not np.array_equal(
        performance['family_id'].astype(str).to_numpy(),
        ela['family_id'].astype(str).to_numpy(),
    ):
        raise ValueError('Performance/ELA family_id metadata disagree after recipe_id alignment.')

    for name, frame in [('Performance', performance), ('ELA', ela)]:
        feature_cols = [col for col in frame.columns if col not in meta_cols]
        if not feature_cols:
            raise ValueError(f'{name} table contains no feature columns.')
        values = frame[feature_cols].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'{name} feature table contains non-finite values.')

    return performance, ela


def validate_distance_weights(w_ela: float, w_performance: float) -> None:
    if (
        not math.isfinite(w_ela)
        or not math.isfinite(w_performance)
        or w_ela < 0.0
        or w_performance < 0.0
        or (w_ela + w_performance) <= 0.0
    ):
        raise ValueError(
            f'Distance weights must be nonnegative with a positive sum; '
            f'received w_ela={w_ela}, w_performance={w_performance}.'
        )


def greedy_facility_location(
    distance_sq: np.ndarray,
    recipe_ids: list[str],
    pool_families: list[str],
    k: int,
    lower_bounds: dict[str, int],
    upper_bounds: dict[str, int],
    progress_label: str | None = None,
) -> list[int]:
    n = distance_sq.shape[0]
    if k > n:
        raise ValueError(f'Target size k={k} exceeds pool size n={n}.')

    selected: list[int] = []
    coverage = np.full(n, np.inf)

    while len(selected) < k:
        best_candidate = None
        best_gain = -np.inf
        best_coverage = None
        selected_set = set(selected)
        current_loss = float(coverage.sum())

        for candidate in range(n):
            if candidate in selected_set:
                continue

            counts = trial_family_counts(selected, pool_families, add_idx=candidate)
            slots_left = k - (len(selected) + 1)
            if not selection_feasible(counts, lower_bounds, upper_bounds, slots_left):
                continue

            candidate_coverage = np.minimum(coverage, distance_sq[:, candidate])
            if selected:
                gain = current_loss - float(candidate_coverage.sum())
            else:
                gain = float(candidate_coverage[np.isfinite(candidate_coverage)].sum())
            key = (gain, recipe_ids[candidate])

            if best_candidate is None or key > (best_gain, recipe_ids[best_candidate]):
                best_candidate = candidate
                best_gain = gain
                best_coverage = candidate_coverage

        if best_candidate is None:
            raise ValueError(
                'No feasible candidate remained during constrained greedy selection. '
                'Check the family bounds against the pool composition and target size.'
            )

        selected.append(best_candidate)
        coverage = best_coverage
        report_every = max(1, k // 20)
        if progress_label and (len(selected) == 1 or len(selected) % report_every == 0 or len(selected) == k):
            print(
                f'[{progress_label}] greedy coverage selection: {len(selected)}/{k} | '
                f'coverage loss={float(coverage.sum()):.6g}',
                flush=True,
            )

    return selected


def local_swap_refinement(
    distance_sq: np.ndarray,
    recipe_ids: list[str],
    pool_families: list[str],
    initial_selected: list[int],
    spacing_lambda: float,
    spacing_mu: float,
    lower_bounds: dict[str, int],
    upper_bounds: dict[str, int],
    max_passes: int = 10,
    progress_label: str | None = None,
) -> list[int]:
    selected = list(initial_selected)
    current_score, _, _, _ = composite_objective(distance_sq, selected, spacing_lambda, spacing_mu)

    for pass_index in range(max_passes):
        pass_started = time.time()
        improved = False
        selected_set = set(selected)
        outside = [idx for idx in range(distance_sq.shape[0]) if idx not in selected_set]
        best_trial = None
        best_score = current_score

        progress_every = max(1, len(selected) // 10)
        for pos, removed in enumerate(selected):
            for added in outside:
                counts = trial_family_counts(selected, pool_families, add_idx=added, remove_idx=removed)
                if not final_counts_valid(counts, lower_bounds, upper_bounds):
                    continue

                trial = list(selected)
                trial[pos] = added
                trial = sorted(trial, key=lambda idx: recipe_ids[idx])
                score, _, _, _ = composite_objective(distance_sq, trial, spacing_lambda, spacing_mu)
                if score > best_score + 1e-12:
                    best_score = score
                    best_trial = trial
            if progress_label and (
                (pos + 1) % progress_every == 0 or pos + 1 == len(selected)
            ):
                print(
                    f'[{progress_label}] local-swap pass {pass_index + 1}/{max_passes}: '
                    f'evaluated {pos + 1}/{len(selected)} selected positions | '
                    f'elapsed={time.time() - pass_started:.1f}s',
                    flush=True,
                )

        if best_trial is not None:
            selected = best_trial
            current_score = best_score
            improved = True

        if progress_label:
            outcome = 'improved' if improved else 'no improving swap'
            print(
                f'[{progress_label}] local-swap pass {pass_index + 1}/{max_passes}: '
                f'{outcome} | score={current_score:.6g} | '
                f'elapsed={time.time() - pass_started:.1f}s',
                flush=True,
            )

        if not improved:
            break

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description='Select a fixed-cardinality PORTAL benchmark subset.')
    parser.add_argument('--pool-file', default=str(default_pool_file()), help='CSV file containing the MIS pool.')
    parser.add_argument(
        '--performance-cleaned', '--isa-cleaned', dest='performance_cleaned',
        default=str(PIPELINE_DIR / 'performance_cleaned.csv'),
        help='Cleaned Performance feature table.',
    )
    parser.add_argument(
        '--ela-cleaned',
        default=str(PIPELINE_DIR / 'ela_cleaned.csv'),
        help='Cleaned ELA feature table.',
    )
    parser.add_argument('--k', type=int, default=50, help='Target subset size.')
    parser.add_argument('--w-ela', type=float, default=0.5, help='ELA weight in joint distance.')
    parser.add_argument('--w-performance', '--w-isa', dest='w_performance', type=float, default=0.5, help='Algorithm-performance weight in joint distance.')
    parser.add_argument(
        '--trim-fraction',
        type=float,
        default=0.10,
        help='Symmetric trim fraction used when computing the robust mean scale on all raw pairwise distances in each space.',
    )
    parser.add_argument(
        '--family-lower',
        default=','.join(str(value) for value in DEFAULT_FAMILY_LOWER_PCT),
        help=f'Lower percentage bounds in FAMILY_ORDER={FAMILY_ORDER}. Example: 12,16,20,30',
    )
    parser.add_argument(
        '--family-upper',
        default=','.join(str(value) for value in DEFAULT_FAMILY_UPPER_PCT),
        help=f'Upper percentage bounds in FAMILY_ORDER={FAMILY_ORDER}. Example: 16,24,30,40',
    )
    parser.add_argument('--spacing-lambda', type=float, default=1.0, help='Weight on mean nearest-neighbor spacing.')
    parser.add_argument('--spacing-mu', type=float, default=0.5, help='Weight on minimum pairwise spacing.')
    parser.add_argument(
        '--max-swap-passes',
        type=int,
        default=10,
        help='Maximum number of best-improvement local-swap passes; zero skips refinement.',
    )
    parser.add_argument('--out-dir', default=str(OUT_DIR), help='Directory for selection outputs.')
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError(f'k must be positive. Received: {args.k}')
    validate_distance_weights(args.w_ela, args.w_performance)
    if (
        not math.isfinite(args.spacing_lambda)
        or not math.isfinite(args.spacing_mu)
        or args.spacing_lambda < 0.0
        or args.spacing_mu < 0.0
    ):
        raise ValueError('spacing-lambda and spacing-mu must be nonnegative.')
    if args.max_swap_passes < 0:
        raise ValueError('max-swap-passes must be nonnegative.')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    performance_df = read_characterization_csv(args.performance_cleaned)
    ela_df = read_characterization_csv(args.ela_cleaned)
    pool_df = read_characterization_csv(args.pool_file)
    performance_df, ela_df = align_cleaned_feature_tables(performance_df, ela_df)

    if 'recipe_id' not in pool_df.columns:
        raise ValueError('Pool file must contain a recipe_id column.')
    pool_df = pool_df.copy()
    pool_df['recipe_id'] = pool_df['recipe_id'].astype(str)
    if pool_df['recipe_id'].duplicated().any():
        raise ValueError('Pool file contains duplicate recipe_id values.')

    meta_cols = ['recipe_id', 'family', 'family_id']
    recipe_ids = performance_df['recipe_id'].astype(str).tolist()
    recipe_to_idx = {recipe_id: idx for idx, recipe_id in enumerate(recipe_ids)}
    families = performance_df['family'].tolist()

    # Canonical ordering makes deterministic tie handling independent of the
    # incidental row order of an equivalent pool CSV.
    pool_recipe_ids = sorted(pool_df['recipe_id'].astype(str).tolist())
    unknown_pool_ids = sorted(set(pool_recipe_ids) - set(recipe_to_idx))
    if unknown_pool_ids:
        raise ValueError(f'Pool recipes are absent from cleaned feature tables: {unknown_pool_ids[:5]}')
    pool_indices = [recipe_to_idx[recipe_id] for recipe_id in pool_recipe_ids]
    pool_recipe_ids = [recipe_ids[idx] for idx in pool_indices]
    pool_families = [families[idx] for idx in pool_indices]
    unknown_families = sorted(set(pool_families) - set(FAMILY_ORDER))
    if unknown_families:
        raise ValueError(
            f'Pool contains families absent from FAMILY_ORDER: {unknown_families}'
        )
    if 'family' in pool_df.columns:
        pool_family_by_id = dict(
            zip(pool_df['recipe_id'], pool_df['family'].astype(str), strict=True)
        )
        mismatches = [
            recipe_id
            for recipe_id, family in zip(pool_recipe_ids, pool_families, strict=True)
            if pool_family_by_id[recipe_id] != str(family)
        ]
        if mismatches:
            raise ValueError(
                'Pool and cleaned feature tables disagree on family metadata for: '
                f'{mismatches[:5]}'
            )

    requested_lower_pct = parse_family_percentages(args.family_lower, 'family-lower')
    requested_upper_pct = parse_family_percentages(args.family_upper, 'family-upper')
    available_counts = {family: pool_families.count(family) for family in FAMILY_ORDER}

    for family in FAMILY_ORDER:
        if requested_lower_pct[family] > requested_upper_pct[family]:
            raise ValueError(
                f'Infeasible percentage bounds for {family}: lower={requested_lower_pct[family]:.1f}% exceeds '
                f'upper={requested_upper_pct[family]:.1f}%.'
            )

    lower_bounds = derive_count_bounds(requested_lower_pct, args.k, mode='lower')
    derived_upper = derive_count_bounds(requested_upper_pct, args.k, mode='upper')
    upper_bounds = {
        family: min(derived_upper[family], available_counts[family])
        for family in FAMILY_ORDER
    }

    for family in FAMILY_ORDER:
        if lower_bounds[family] > upper_bounds[family]:
            raise ValueError(
                f'Infeasible bounds for {family}: lower={lower_bounds[family]} exceeds '
                f'effective upper={upper_bounds[family]} given pool availability={available_counts[family]}.'
            )

    if sum(lower_bounds.values()) > args.k:
        raise ValueError(
            f'Sum of lower bounds ({sum(lower_bounds.values())}) exceeds target size k={args.k}.'
        )
    if sum(upper_bounds.values()) < args.k:
        raise ValueError(
            f'Sum of effective upper bounds ({sum(upper_bounds.values())}) is below target size k={args.k}.'
        )

    performance_x = performance_df.drop(columns=meta_cols).to_numpy()
    ela_x = ela_df.drop(columns=meta_cols).to_numpy()

    delta_ela = pdist(ela_x, metric='euclidean')
    delta_performance = pdist(performance_x, metric='euclidean')

    scale_ela = robust_mean_scale(delta_ela, args.trim_fraction)
    scale_performance = robust_mean_scale(delta_performance, args.trim_fraction)
    delta_ela = delta_ela / scale_ela
    delta_performance = delta_performance / scale_performance

    delta_ela_sq = squareform(delta_ela)
    delta_performance_sq = squareform(delta_performance)
    joint_sq_full = args.w_ela * delta_ela_sq + args.w_performance * delta_performance_sq
    joint_sq = joint_sq_full[np.ix_(pool_indices, pool_indices)]

    greedy_selected = greedy_facility_location(
        joint_sq,
        pool_recipe_ids,
        pool_families,
        args.k,
        lower_bounds,
        upper_bounds,
        progress_label=f'S{args.k}',
    )
    refined_selected = local_swap_refinement(
        joint_sq,
        pool_recipe_ids,
        pool_families,
        greedy_selected,
        args.spacing_lambda,
        args.spacing_mu,
        lower_bounds,
        upper_bounds,
        max_passes=args.max_swap_passes,
        progress_label=f'S{args.k}',
    )

    greedy_score, greedy_loss, greedy_mean_nnd, greedy_min_pair = composite_objective(
        joint_sq, greedy_selected, args.spacing_lambda, args.spacing_mu
    )
    final_score, final_loss, final_mean_nnd, final_min_pair = composite_objective(
        joint_sq, refined_selected, args.spacing_lambda, args.spacing_mu
    )

    selected_joint = joint_sq[np.ix_(refined_selected, refined_selected)].copy()
    np.fill_diagonal(selected_joint, np.inf)
    selected_nnd = selected_joint.min(axis=1) if len(refined_selected) > 1 else np.zeros(len(refined_selected))

    selection_rows = []
    for local_idx, selected_idx in enumerate(refined_selected):
        selection_rows.append(
            {
                'recipe_id': pool_recipe_ids[selected_idx],
                'family': pool_families[selected_idx],
                'nnd_joint_within_selection': float(selected_nnd[local_idx]),
            }
        )

    selection_df = pd.DataFrame(selection_rows).sort_values(['family', 'recipe_id'])
    selection_stem = f'benchmark{args.k}'
    selected_filename = f'{selection_stem}_selected.csv'
    report_filename = f'{selection_stem}_selection_report.md'
    selection_df.to_csv(out_dir / selected_filename, index=False)

    # ------------------------------------------------------------------
    # Extended statistics: coverage per pool recipe and NND within the selection
    # ------------------------------------------------------------------
    family_counts = selection_df['family'].value_counts().to_dict()

    # Full joint distance matrix (pool-only submatrix already computed as joint_sq)
    joint_sel_pool = joint_sq[np.ix_(refined_selected, list(range(len(pool_indices))))]
    coverage_per_pool = joint_sel_pool.min(axis=0)   # each pool recipe -> dist to nearest selected

    joint_sel_sel = joint_sq[np.ix_(refined_selected, refined_selected)].copy()
    np.fill_diagonal(joint_sel_sel, np.inf)
    nnd_within = (
        joint_sel_sel.min(axis=1)
        if len(refined_selected) > 1
        else np.zeros(len(refined_selected))
    )

    pool_fam_arr = np.array(pool_families)
    sel_fam_list = [pool_families[i] for i in refined_selected]

    def family_nnd_stats(family):
        idxs = [i for i, f in enumerate(sel_fam_list) if f == family]
        if not idxs:
            return None
        v = nnd_within[idxs]
        return int(len(idxs)), float(v.min()), float(v.mean()), float(v.max())

    def family_cov_stats(family):
        mask = pool_fam_arr == family
        if not mask.any():
            return None
        v = coverage_per_pool[mask]
        return int(mask.sum()), float(v.mean()), float(v.max())

    binding_labels = []
    for family in FAMILY_ORDER:
        selected_count = family_counts.get(family, 0)
        if selected_count == lower_bounds[family] == upper_bounds[family]:
            binding_labels.append(f'{family}={selected_count} (exact)')
        elif selected_count == lower_bounds[family]:
            binding_labels.append(f'{family}={selected_count} (lower)')
        elif selected_count == upper_bounds[family]:
            binding_labels.append(f'{family}={selected_count} (upper)')
    binding_text = ', '.join(binding_labels) if binding_labels else 'none'

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report_lines = [
        f'# PORTAL Fixed-Cardinality Selection Report (K={args.k})',
        '',
        '## 1. Configuration',
        '',
        '| Parameter | Value |',
        '|-----------|-------|',
        f'| Pool file | {Path(args.pool_file).name} |',
        f'| Pool size | {len(pool_indices)} |',
        f'| Target size K | {args.k} |',
        f'| Family lower percentages | {format_percentages(requested_lower_pct)} |',
        f'| Family upper percentages | {format_percentages(requested_upper_pct)} |',
        f'| Derived lower bounds | {format_bounds(lower_bounds)} |',
        f'| Derived upper bounds | {format_bounds(derived_upper)} |',
        f'| Effective upper bounds | {format_bounds(upper_bounds)} |',
        f'| ELA weight | {args.w_ela:.2f} |',
        f'| Performance weight | {args.w_performance:.2f} |',
        f'| Trim fraction | {args.trim_fraction:.0%} |',
        f'| ELA normalization scale | {scale_ela:.4f} |',
        f'| Performance normalization scale | {scale_performance:.4f} |',
        f'| Spacing lambda | {args.spacing_lambda:.2f} |',
        f'| Spacing mu | {args.spacing_mu:.2f} |',
        f'| Maximum local-swap passes | {args.max_swap_passes} |',
        '',
        '## 2. Algorithm Stages',
        '',
        '| Stage | Coverage loss | Mean NND | Min pair dist | Composite score |',
        '|-------|--------------|----------|---------------|-----------------|',
        f'| A: Greedy facility location | {greedy_loss:.4f} | {greedy_mean_nnd:.4f} | {greedy_min_pair:.4f} | {greedy_score:.4f} |',
        f'| B: Local swap refinement    | {final_loss:.4f} | {final_mean_nnd:.4f} | {final_min_pair:.4f} | {final_score:.4f} |',
        f'| Improvement | {greedy_loss - final_loss:.4f} | {final_mean_nnd - greedy_mean_nnd:+.4f} | {final_min_pair - greedy_min_pair:+.4f} | {final_score - greedy_score:+.4f} |',
        '',
        '## 3. Family Composition',
        '',
        '| Family | Pool | Lower | Upper | Selected | % pool | % suite | Binding? |',
        '|--------|------|-------|-------|----------|--------|-------|----------|',
    ]
    for family in FAMILY_ORDER:
        p = available_counts[family]
        s = family_counts.get(family, 0)
        lo = lower_bounds[family]
        hi = upper_bounds[family]
        binding = ('upper' if s == hi else ('lower' if s == lo else 'free'))
        report_lines.append(
            f'| {family} | {p} | {lo} | {hi} | {s} | {100*p/len(pool_indices):.1f}% | {100*s/args.k:.1f}% | {binding} |'
        )
    report_lines.append(
        f'| **Total** | **{len(pool_indices)}** | **{sum(lower_bounds.values())}** '
        f'| **{sum(upper_bounds.values())}** | **{sum(family_counts.values())}** | | | |'
    )

    report_lines += [
        '',
        '## 4. Coverage Analysis',
        '',
        'Coverage is measured as the distance from each pool recipe to its nearest selected recipe',
        f'(joint trimmed-mean-normalized distance, scales: ELA={scale_ela:.2f}, Performance={scale_performance:.2f}).',
        '',
        f'- **Total coverage loss C(S_K)**: {coverage_per_pool.sum():.4f}',
        f'- **Mean coverage distance**: {coverage_per_pool.mean():.4f}  (average pool recipe is this far from the nearest selected member)',
        f'- **Max coverage distance**: {coverage_per_pool.max():.4f}  (hardest-to-cover pool recipe)',
        '',
        '| Family | Pool recipes | Mean coverage | Max coverage |',
        '|--------|-------------|---------------|--------------|',
    ]
    for family in FAMILY_ORDER:
        stats = family_cov_stats(family)
        if stats:
            np_, mc, mx = stats
            report_lines.append(f'| {family} | {np_} | {mc:.4f} | {mx:.4f} |')

    report_lines += [
        '',
        'Family-level coverage values show how the available slots represent each category under',
        'the requested bounds; larger values indicate categories that are harder to cover tightly.',
        '',
        f'## 5. Spacing Within the Selected K={args.k} Suite',
        '',
        'Nearest-neighbor distance (NND) of each selected recipe to the next-nearest selected recipe.',
        '',
        f'- **Min NND**: {nnd_within.min():.4f}  (closest pair within the selected suite)',
        f'- **Mean NND**: {nnd_within.mean():.4f}',
        f'- **Median NND**: {float(np.median(nnd_within)):.4f}',
        f'- **Max NND**: {nnd_within.max():.4f}  (most isolated selected recipe)',
        '',
        '| Family | Selected | NND min | NND mean | NND max |',
        '|--------|----------|---------|----------|---------|',
    ]
    for family in FAMILY_ORDER:
        stats = family_nnd_stats(family)
        if stats:
            n_, mn, me, mx = stats
            report_lines.append(f'| {family} | {n_} | {mn:.3f} | {me:.3f} | {mx:.3f} |')

    report_lines += [
        '',
        'The family-level NND statistics describe how widely the selected representatives are spaced',
        'within each category; they are descriptive outcomes rather than additional constraints.',
        '',
        '## 6. Key Findings',
        '',
        f'1. **Selected cardinality**: {len(refined_selected)} recipes from a pool of {len(pool_indices)}.',
        f'2. **Binding category constraints**: {binding_text}.',
        f'3. **Coverage loss C(S_K)**: {coverage_per_pool.sum():.4f} '
        f'(mean assignment distance {coverage_per_pool.mean():.4f} per pool recipe).',
        f'4. **Within-suite spacing**: minimum pairwise distance = {nnd_within.min():.4f} '
        f'and mean NND = {nnd_within.mean():.4f} in joint selection units.',
        '5. **De-duplication inheritance**: because selection is restricted to the independent-set pool, '
        'the selected suite contains no pair confirmed as redundant by the preceding AND-rule stage. '
        'Selection-space distances are not compared numerically with the separately normalized de-duplication threshold.',
        '',
        '## 7. Output Files',
        '',
        '| File | Content |',
        '|------|---------|',
        f'| {selected_filename} | {args.k} selected recipes with family and within-suite NND |',
        f'| {report_filename} | This report |',
    ]

    (out_dir / report_filename).write_text('\n'.join(report_lines) + '\n', encoding='utf-8')

    print(f'=== PORTAL Fixed-Cardinality Selection (K={args.k}) ===')
    print(f'  Pool size: {len(pool_indices)}  |  Selected: {len(refined_selected)}')
    print(f'  Scales: ELA={scale_ela:.4f}  Performance={scale_performance:.4f}')
    print('  Family composition: ' + '  '.join(f'{f[-1]}={family_counts.get(f,0)}' for f in FAMILY_ORDER))
    print(f'  Coverage loss: {coverage_per_pool.sum():.4f}  |  Mean NND: {nnd_within.mean():.4f}  |  Min pair: {nnd_within.min():.4f}')
    print(f'  Saved: {selected_filename}, {report_filename}')


if __name__ == '__main__':
    main()
