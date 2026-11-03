"""
Create train/test selections from the deduplicated pool.

Pipeline:
1. Load the deduplicated pool (MIS output).
2. Construct disjoint train/test candidate pools by family.
3. Run constrained greedy+swap selection independently on each half,
   using independently configurable percentage-based family bounds and target
   sizes (train-k and test-k).

Design rationale -- split-then-select vs. select-then-split:
  We split the deduplicated pool before subset selection because the scientific
  objective is to construct two separately representative evaluation sets, not a
  single globally optimized set that is partitioned afterward.  Under the default
  protocol, every family is eligible on both sides, recipes are randomly divided
  within family, and no ELA/Performance matching is imposed across the split.  This yields
  comparable in-taxonomy candidate pools without deliberately pairing each test
  recipe with a nearby training recipe.  The selector is then run independently
  within each candidate pool.

  Separate train/test bounds also support controlled distribution shifts.  If a
  family's upper bound is zero on one side and positive on the other, all recipes
  from that family are assigned exclusively to the eligible side and are excluded
  from the other side's coverage objective.  Different nonzero bounds support
  different family mixtures while retaining shared families.  These options
  distinguish recipe-level generalization within a shared taxonomy from explicit
  family-holdout evaluation.

The joint ELA/Performance distance matrix is computed once over all recipes using the
same trimmed-mean normalization as run_benchmark_selection.py, ensuring
consistent distance units across both halves.

Family constraints are provided as percentages and converted to counts
separately for train-k and test-k, so each side can use a different target size
and family composition.
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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

try:  # Support both direct execution and ``python -m Recipe_Selection...``.
    from .run_benchmark_selection import (
        DEFAULT_FAMILY_LOWER_PCT,
        DEFAULT_FAMILY_UPPER_PCT,
        FAMILY_ORDER,
        align_cleaned_feature_tables,
        composite_objective,
        default_pool_file,
        derive_count_bounds,
        format_percentages,
        greedy_facility_location,
        local_swap_refinement,
        parse_family_percentages,
        robust_mean_scale,
        validate_distance_weights,
    )
except ImportError:  # Direct script execution places Recipe_Selection on sys.path.
    from run_benchmark_selection import (
        DEFAULT_FAMILY_LOWER_PCT,
        DEFAULT_FAMILY_UPPER_PCT,
        FAMILY_ORDER,
        align_cleaned_feature_tables,
        composite_objective,
        default_pool_file,
        derive_count_bounds,
        format_percentages,
        greedy_facility_location,
        local_swap_refinement,
        parse_family_percentages,
        robust_mean_scale,
        validate_distance_weights,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # project root (one level up from Recipe_Selection/)
PIPELINE_DIR = ROOT / 'pipeline_output'
DEFAULT_OUT_DIR = ROOT / 'train_test_output'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Construct disjoint candidate pools and select train/test subsets.'
    )
    parser.add_argument('--pool-file', default=str(default_pool_file()),
                        help='Deduplicated pool CSV (MIS output).')
    parser.add_argument('--performance-cleaned', '--isa-cleaned', dest='performance_cleaned',
                        default=str(PIPELINE_DIR / 'performance_cleaned.csv'),
                        help='Cleaned Performance feature table.')
    parser.add_argument('--ela-cleaned',
                        default=str(PIPELINE_DIR / 'ela_cleaned.csv'),
                        help='Cleaned ELA feature table.')
    parser.add_argument('--train-k', type=int, default=30,
                        help='Number of selected train instances.')
    parser.add_argument('--test-k', type=int, default=30,
                        help='Number of selected test instances.')
    parser.add_argument('--train-family-lower', default=None,
                        help=f'Train lower percentage bounds in FAMILY_ORDER={FAMILY_ORDER}.')
    parser.add_argument('--train-family-upper', default=None,
                        help=f'Train upper percentage bounds in FAMILY_ORDER={FAMILY_ORDER}.')
    parser.add_argument('--test-family-lower', default=None,
                        help=f'Test lower percentage bounds in FAMILY_ORDER={FAMILY_ORDER}.')
    parser.add_argument('--test-family-upper', default=None,
                        help=f'Test upper percentage bounds in FAMILY_ORDER={FAMILY_ORDER}.')
    parser.add_argument('--family-lower', default=None,
                        help='Optional shared lower percentages for backward compatibility; '
                             'side-specific values take precedence.')
    parser.add_argument('--family-upper', default=None,
                        help='Optional shared upper percentages for backward compatibility; '
                             'side-specific values take precedence.')
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Seed for the within-family candidate partition.')
    parser.add_argument('--w-ela', type=float, default=0.5,
                        help='ELA weight in joint distance.')
    parser.add_argument('--w-performance', '--w-isa', dest='w_performance', type=float, default=0.5,
                        help='Performance weight in joint distance.')
    parser.add_argument('--trim-fraction', type=float, default=0.10,
                        help='Trim fraction for robust mean distance scaling on all raw pairwise distances in each space.')
    parser.add_argument('--spacing-lambda', type=float, default=1.0,
                        help='Weight on mean nearest-neighbor spacing.')
    parser.add_argument('--spacing-mu', type=float, default=0.5,
                        help='Weight on minimum pairwise spacing.')
    parser.add_argument('--max-swap-passes', type=int, default=10,
                        help='Maximum local-swap passes per subset; zero skips refinement.')
    parser.add_argument('--out-dir', default=str(DEFAULT_OUT_DIR),
                        help='Directory for split and selection outputs.')
    return parser.parse_args()


def stratified_candidate_split(
    pool_df: pd.DataFrame,
    seed: int,
    train_k: int,
    test_k: int,
    train_lower_pct: dict[str, float],
    train_upper_pct: dict[str, float],
    test_lower_pct: dict[str, float],
    test_upper_pct: dict[str, float],
) -> tuple[list[str], list[str], list[str]]:
    """Construct disjoint candidate pools using family eligibility.

    A family with positive upper bounds on both sides is shuffled and divided
    as evenly as possible.  A family eligible on only one side is assigned in
    full to that side, which makes true family holdouts possible and prevents
    excluded recipes from entering the other side's coverage objective.  A
    family with zero upper bounds on both sides is omitted and reported.

    The default positive bounds for every family reproduce the original equal,
    family-stratified half split. If that allocation cannot support the
    requested side-specific cardinalities and percentage bounds, a small
    integer program changes the per-family allocation by the minimum amount
    needed for feasibility. The split is deterministic for a fixed seed and
    pool and does not use ELA/Performance geometry.
    """
    rng = np.random.default_rng(seed)
    if 'family' not in pool_df.columns:
        raise ValueError('Pool file must include a family column for stratified splitting.')

    family_to_ids: dict[str, list[str]] = {}
    for _, row in pool_df.iterrows():
        family_to_ids.setdefault(str(row['family']), []).append(str(row['recipe_id']))

    ordered_families = [f for f in FAMILY_ORDER if f in family_to_ids]
    ordered_families += sorted([f for f in family_to_ids if f not in FAMILY_ORDER])

    shared_families = [
        fam for fam in ordered_families
        if train_upper_pct.get(fam, 0.0) > 0.0 and test_upper_pct.get(fam, 0.0) > 0.0
    ]
    target_shared_train = sum(len(family_to_ids[fam]) for fam in shared_families) // 2
    floor_counts = {fam: len(family_to_ids[fam]) // 2 for fam in shared_families}
    odd_families = [fam for fam in shared_families if len(family_to_ids[fam]) % 2 == 1]
    extra_needed = target_shared_train - sum(floor_counts.values())

    if not (0 <= extra_needed <= len(odd_families)):
        raise ValueError('Could not construct a balanced stratified half split from the provided pool.')

    extra_train_families: set[str] = set()
    if extra_needed > 0:
        chosen = rng.choice(np.array(odd_families, dtype=object), size=extra_needed, replace=False)
        extra_train_families = set(chosen.tolist())

    initial_train_counts: dict[str, int] = {}
    for fam in ordered_families:
        n_family = len(family_to_ids[fam])
        train_allowed = train_upper_pct.get(fam, 0.0) > 0.0
        test_allowed = test_upper_pct.get(fam, 0.0) > 0.0
        if train_allowed and test_allowed:
            initial_train_counts[fam] = floor_counts[fam] + (
                1 if fam in extra_train_families else 0
            )
        elif train_allowed:
            initial_train_counts[fam] = n_family
        else:
            initial_train_counts[fam] = 0

    def side_is_feasible(
        counts: dict[str, int],
        k: int,
        lower_pct: dict[str, float],
        upper_pct: dict[str, float],
    ) -> bool:
        lower = derive_count_bounds(lower_pct, k, mode='lower')
        upper = derive_count_bounds(upper_pct, k, mode='upper')
        if sum(counts.values()) < k:
            return False
        if any(counts.get(family, 0) < lower[family] for family in FAMILY_ORDER):
            return False
        return sum(
            min(counts.get(family, 0), upper[family]) for family in FAMILY_ORDER
        ) >= k

    initial_test_counts = {
        fam: (
            len(family_to_ids[fam]) - initial_train_counts[fam]
            if test_upper_pct.get(fam, 0.0) > 0.0
            else 0
        )
        for fam in ordered_families
    }
    allocation = dict(initial_train_counts)
    balanced_feasible = side_is_feasible(
        initial_train_counts, train_k, train_lower_pct, train_upper_pct
    ) and side_is_feasible(
        initial_test_counts, test_k, test_lower_pct, test_upper_pct
    )

    if not balanced_feasible:
        try:
            import pulp
        except ImportError as exc:
            raise RuntimeError(
                'Adjusting the family-stratified candidate split requires PuLP/CBC.'
            ) from exc

        solver = pulp.PULP_CBC_CMD(msg=0)
        if not solver.available():
            raise RuntimeError(
                'PuLP is installed, but CBC is unavailable for candidate-split adjustment.'
            )

        problem = pulp.LpProblem('train_test_candidate_split', pulp.LpMinimize)
        train_count = {
            fam: pulp.LpVariable(
                f'n_train_{index}',
                lowBound=0,
                upBound=len(family_to_ids[fam]),
                cat='Integer',
            )
            for index, fam in enumerate(ordered_families)
        }
        deviations = {
            fam: pulp.LpVariable(f'dev_{index}', lowBound=0)
            for index, fam in enumerate(ordered_families)
        }
        train_capacity = {
            fam: pulp.LpVariable(f'train_capacity_{index}', lowBound=0)
            for index, fam in enumerate(ordered_families)
        }
        test_capacity = {
            fam: pulp.LpVariable(f'test_capacity_{index}', lowBound=0)
            for index, fam in enumerate(ordered_families)
        }

        train_lower = derive_count_bounds(train_lower_pct, train_k, mode='lower')
        train_upper = derive_count_bounds(train_upper_pct, train_k, mode='upper')
        test_lower = derive_count_bounds(test_lower_pct, test_k, mode='lower')
        test_upper = derive_count_bounds(test_upper_pct, test_k, mode='upper')
        train_candidate_terms = []
        test_candidate_terms = []

        for fam in ordered_families:
            n_family = len(family_to_ids[fam])
            train_allowed = train_upper_pct.get(fam, 0.0) > 0.0
            test_allowed = test_upper_pct.get(fam, 0.0) > 0.0

            if not train_allowed and not test_allowed:
                problem += train_count[fam] == 0
                problem += train_capacity[fam] == 0
                problem += test_capacity[fam] == 0
            elif train_allowed and not test_allowed:
                problem += train_count[fam] == n_family
                train_candidate_terms.append(train_count[fam])
                problem += train_count[fam] >= train_lower[fam]
                problem += train_capacity[fam] <= train_count[fam]
                problem += train_capacity[fam] <= train_upper[fam]
                problem += test_capacity[fam] == 0
            elif test_allowed and not train_allowed:
                problem += train_count[fam] == 0
                problem += train_capacity[fam] == 0
                test_count = n_family - train_count[fam]
                test_candidate_terms.append(test_count)
                problem += test_count >= test_lower[fam]
                problem += test_capacity[fam] <= test_count
                problem += test_capacity[fam] <= test_upper[fam]
            else:
                train_candidate_terms.append(train_count[fam])
                problem += train_count[fam] >= train_lower[fam]
                problem += train_capacity[fam] <= train_count[fam]
                problem += train_capacity[fam] <= train_upper[fam]
                test_count = n_family - train_count[fam]
                test_candidate_terms.append(test_count)
                problem += test_count >= test_lower[fam]
                problem += test_capacity[fam] <= test_count
                problem += test_capacity[fam] <= test_upper[fam]

            problem += deviations[fam] >= (
                train_count[fam] - initial_train_counts[fam]
            )
            problem += deviations[fam] >= (
                initial_train_counts[fam] - train_count[fam]
            )

        problem += pulp.lpSum(train_candidate_terms) >= train_k
        problem += pulp.lpSum(test_candidate_terms) >= test_k
        problem += pulp.lpSum(train_capacity.values()) >= train_k
        problem += pulp.lpSum(test_capacity.values()) >= test_k
        # The leading coefficient makes the total number of moved recipes the
        # primary objective. The bounded family-order term provides a stable
        # tie-break without changing that minimum.
        tie_bound = 1 + sum(
            (index + 1) * len(family_to_ids[fam])
            for index, fam in enumerate(ordered_families)
        )
        problem += tie_bound * pulp.lpSum(deviations.values()) + pulp.lpSum(
            (index + 1) * train_count[fam]
            for index, fam in enumerate(ordered_families)
        )
        status = problem.solve(solver)
        if status != pulp.LpStatusOptimal:
            raise ValueError(
                'No disjoint candidate split can satisfy the configured train/test '
                'sizes and family percentage bounds.'
            )
        allocation = {
            fam: int(round(float(pulp.value(train_count[fam]))))
            for fam in ordered_families
        }
        print(
            'Candidate split: adjusted the balanced per-family allocation for '
            'side-specific feasibility: '
            f'{initial_train_counts} -> {allocation}'
        )

    train_ids: list[str] = []
    test_ids: list[str] = []
    omitted_ids: list[str] = []

    for fam in ordered_families:
        # Sort before shuffling so a fixed seed depends on pool membership, not
        # on incidental CSV row order.
        fam_ids = sorted(family_to_ids[fam])
        rng.shuffle(fam_ids)
        train_allowed = train_upper_pct.get(fam, 0.0) > 0.0
        test_allowed = test_upper_pct.get(fam, 0.0) > 0.0
        if train_allowed and test_allowed:
            n_train_fam = allocation[fam]
            train_ids.extend(fam_ids[:n_train_fam])
            test_ids.extend(fam_ids[n_train_fam:])
        elif train_allowed:
            train_ids.extend(fam_ids)
        elif test_allowed:
            test_ids.extend(fam_ids)
        else:
            omitted_ids.extend(fam_ids)

    return train_ids, test_ids, omitted_ids


def build_joint_distance(
    performance_df: pd.DataFrame,
    ela_df: pd.DataFrame,
    w_ela: float,
    w_performance: float,
    trim_fraction: float,
) -> tuple[dict[str, int], list[str], np.ndarray, float, float]:
    """Compute the full joint ELA/Performance distance matrix over all recipes.

    Distances are normalized by the trimmed mean of all raw pairwise distances
    in each space (identical to run_benchmark_selection.py), so the two spaces
    contribute equally on average.  The matrix is computed once over all
    recipes and subsetted per candidate pool, ensuring consistent distance units
    across train and test.
    """
    validate_distance_weights(w_ela, w_performance)
    performance_df, ela_df = align_cleaned_feature_tables(performance_df, ela_df)
    meta_cols = ['recipe_id', 'family', 'family_id']
    recipe_ids = performance_df['recipe_id'].astype(str).tolist()
    recipe_to_idx = {rid: idx for idx, rid in enumerate(recipe_ids)}
    families = performance_df['family'].astype(str).tolist()

    performance_x = performance_df.drop(columns=meta_cols).to_numpy()
    ela_x = ela_df.drop(columns=meta_cols).to_numpy()

    delta_ela = pdist(ela_x, metric='euclidean')
    delta_performance = pdist(performance_x, metric='euclidean')
    scale_ela = robust_mean_scale(delta_ela, trim_fraction)
    scale_performance = robust_mean_scale(delta_performance, trim_fraction)
    delta_ela = delta_ela / scale_ela
    delta_performance = delta_performance / scale_performance

    joint_sq_full = w_ela * squareform(delta_ela) + w_performance * squareform(delta_performance)
    return recipe_to_idx, families, joint_sq_full, scale_ela, scale_performance


def derive_bounds_for_pool(
    k: int,
    lower_pct: dict[str, float],
    upper_pct: dict[str, float],
    pool_families: list[str],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    available_counts = {family: pool_families.count(family) for family in FAMILY_ORDER}

    lower_bounds = derive_count_bounds(lower_pct, k, mode='lower')
    derived_upper = derive_count_bounds(upper_pct, k, mode='upper')
    effective_upper = {
        family: min(derived_upper[family], available_counts[family])
        for family in FAMILY_ORDER
    }

    for family in FAMILY_ORDER:
        if lower_bounds[family] > effective_upper[family]:
            raise ValueError(
                f'Infeasible bounds for {family}: lower={lower_bounds[family]} exceeds '
                f'effective upper={effective_upper[family]} with availability={available_counts[family]}.'
            )

    if sum(lower_bounds.values()) > k:
        raise ValueError(f'Sum of lower bounds ({sum(lower_bounds.values())}) exceeds k={k}.')
    if sum(effective_upper.values()) < k:
        raise ValueError(
            f'Sum of effective upper bounds ({sum(effective_upper.values())}) is below k={k}.'
        )

    return lower_bounds, derived_upper, effective_upper, available_counts


def select_from_half(
    half_name: str,
    half_recipe_ids: list[str],
    recipe_to_idx: dict[str, int],
    families_full: list[str],
    joint_sq_full: np.ndarray,
    k: int,
    lower_pct: dict[str, float],
    upper_pct: dict[str, float],
    spacing_lambda: float,
    spacing_mu: float,
    max_swap_passes: int,
) -> dict[str, object]:
    if k <= 0:
        raise ValueError(f'{half_name}: k must be positive. Received {k}.')
    if k > len(half_recipe_ids):
        raise ValueError(
            f'{half_name}: k={k} exceeds candidate-pool size {len(half_recipe_ids)}.'
        )

    try:
        half_indices = [recipe_to_idx[rid] for rid in half_recipe_ids]
    except KeyError as exc:
        raise ValueError(
            f'{half_name}: recipe_id {exc.args[0]!r} not found in cleaned Performance/ELA tables.'
        ) from exc

    half_families = [families_full[idx] for idx in half_indices]
    joint_sq = joint_sq_full[np.ix_(half_indices, half_indices)]

    lower_bounds, derived_upper, effective_upper, available_counts = derive_bounds_for_pool(
        k, lower_pct, upper_pct, half_families
    )

    greedy_selected = greedy_facility_location(
        joint_sq,
        half_recipe_ids,
        half_families,
        k,
        lower_bounds,
        effective_upper,
        progress_label=half_name,
    )
    refined_selected = local_swap_refinement(
        joint_sq, half_recipe_ids, half_families, greedy_selected,
        spacing_lambda, spacing_mu, lower_bounds, effective_upper,
        max_passes=max_swap_passes,
        progress_label=half_name,
    )

    greedy_score, greedy_loss, greedy_mean_nnd, greedy_min_pair = composite_objective(
        joint_sq, greedy_selected, spacing_lambda, spacing_mu
    )
    final_score, final_loss, final_mean_nnd, final_min_pair = composite_objective(
        joint_sq, refined_selected, spacing_lambda, spacing_mu
    )

    # Coverage: distance from each half-pool recipe to its nearest selected recipe
    joint_sel_pool = joint_sq[np.ix_(refined_selected, list(range(len(half_recipe_ids))))]
    coverage_per_pool = joint_sel_pool.min(axis=0)

    # Spacing within selection
    joint_sel_sel = joint_sq[np.ix_(refined_selected, refined_selected)].copy()
    np.fill_diagonal(joint_sel_sel, np.inf)
    nnd_within = (
        joint_sel_sel.min(axis=1) if len(refined_selected) > 1
        else np.zeros(len(refined_selected))
    )

    sel_fam_list = [half_families[i] for i in refined_selected]

    # Global row indices into joint_sq_full for cross-split separation
    global_selected_indices = [half_indices[i] for i in refined_selected]

    selected_rows = []
    for local_idx, selected_idx in enumerate(refined_selected):
        selected_rows.append({
            'recipe_id': half_recipe_ids[selected_idx],
            'family': half_families[selected_idx],
            'nnd_joint_within_selection': float(nnd_within[local_idx]),
            'split': half_name,
        })
    selected_df = pd.DataFrame(selected_rows).sort_values(['family', 'recipe_id'])

    return {
        'half_name': half_name,
        'half_size': len(half_recipe_ids),
        'k': k,
        'available_counts': available_counts,
        'lower_bounds': lower_bounds,
        'derived_upper': derived_upper,
        'effective_upper': effective_upper,
        'greedy_loss': greedy_loss,
        'greedy_mean_nnd': greedy_mean_nnd,
        'greedy_min_pair': greedy_min_pair,
        'greedy_score': greedy_score,
        'final_loss': final_loss,
        'final_mean_nnd': final_mean_nnd,
        'final_min_pair': final_min_pair,
        'final_score': final_score,
        'selected_df': selected_df,
        'coverage_per_pool': coverage_per_pool,
        'nnd_within': nnd_within,
        'pool_families': half_families,
        'sel_fam_list': sel_fam_list,
        'global_selected_indices': global_selected_indices,
    }


def main() -> None:
    args = parse_args()

    if args.train_k <= 0 or args.test_k <= 0:
        raise ValueError(
            f'train-k and test-k must be positive; received '
            f'{args.train_k} and {args.test_k}.'
        )
    if args.split_seed < 0:
        raise ValueError(f'split-seed must be nonnegative; received {args.split_seed}.')
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

    default_lower = ','.join(str(v) for v in DEFAULT_FAMILY_LOWER_PCT)
    default_upper = ','.join(str(v) for v in DEFAULT_FAMILY_UPPER_PCT)
    train_lower_raw = args.train_family_lower or args.family_lower or default_lower
    train_upper_raw = args.train_family_upper or args.family_upper or default_upper
    test_lower_raw = args.test_family_lower or args.family_lower or default_lower
    test_upper_raw = args.test_family_upper or args.family_upper or default_upper

    train_lower_pct = parse_family_percentages(train_lower_raw, 'train-family-lower')
    train_upper_pct = parse_family_percentages(train_upper_raw, 'train-family-upper')
    test_lower_pct = parse_family_percentages(test_lower_raw, 'test-family-lower')
    test_upper_pct = parse_family_percentages(test_upper_raw, 'test-family-upper')

    for side, lower_pct, upper_pct in [
        ('train', train_lower_pct, train_upper_pct),
        ('test', test_lower_pct, test_upper_pct),
    ]:
        for family in FAMILY_ORDER:
            if lower_pct[family] > upper_pct[family]:
                raise ValueError(
                    f'Infeasible {side} percentage bounds for {family}: '
                    f'lower={lower_pct[family]:.1f}% > upper={upper_pct[family]:.1f}%.'
                )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    performance_df = read_characterization_csv(args.performance_cleaned)
    ela_df = read_characterization_csv(args.ela_cleaned)
    pool_df = read_characterization_csv(args.pool_file)
    if 'recipe_id' not in pool_df.columns:
        raise ValueError('Pool file must include recipe_id column.')
    if 'family' not in pool_df.columns:
        raise ValueError('Pool file must include family column.')

    pool_df = pool_df.copy()
    pool_df['recipe_id'] = pool_df['recipe_id'].astype(str)
    pool_df['family'] = pool_df['family'].astype(str)
    if pool_df['recipe_id'].duplicated().any():
        raise ValueError('Pool file contains duplicate recipe_id values.')
    unknown_families = sorted(set(pool_df['family']) - set(FAMILY_ORDER))
    if unknown_families:
        raise ValueError(
            f'Pool contains families absent from FAMILY_ORDER: {unknown_families}'
        )

    # Validate membership and family labels before splitting so stale pool
    # metadata cannot silently alter candidate eligibility.
    performance_df, ela_df = align_cleaned_feature_tables(performance_df, ela_df)
    cleaned_family = dict(
        zip(
            performance_df['recipe_id'].astype(str),
            performance_df['family'].astype(str),
            strict=True,
        )
    )
    missing_cleaned = sorted(set(pool_df['recipe_id']) - set(cleaned_family))
    if missing_cleaned:
        raise ValueError(
            f'Pool recipes are absent from cleaned feature tables: {missing_cleaned[:5]}'
        )
    family_mismatches = [
        recipe_id
        for recipe_id, family in zip(
            pool_df['recipe_id'], pool_df['family'], strict=True
        )
        if cleaned_family[recipe_id] != family
    ]
    if family_mismatches:
        raise ValueError(
            'Pool and cleaned feature tables disagree on family metadata for: '
            f'{family_mismatches[:5]}'
        )

    train_pool_ids, test_pool_ids, omitted_ids = stratified_candidate_split(
        pool_df,
        seed=args.split_seed,
        train_k=args.train_k,
        test_k=args.test_k,
        train_lower_pct=train_lower_pct,
        train_upper_pct=train_upper_pct,
        test_lower_pct=test_lower_pct,
        test_upper_pct=test_upper_pct,
    )

    train_pool_df = pool_df[pool_df['recipe_id'].isin(train_pool_ids)].copy()
    test_pool_df  = pool_df[pool_df['recipe_id'].isin(test_pool_ids)].copy()
    omitted_pool_df = pool_df[pool_df['recipe_id'].isin(omitted_ids)].copy()
    train_pool_df = train_pool_df.sort_values(['family', 'recipe_id'])
    test_pool_df  = test_pool_df.sort_values(['family', 'recipe_id'])
    omitted_pool_df = omitted_pool_df.sort_values(['family', 'recipe_id'])

    if args.train_k > len(train_pool_df):
        raise ValueError(
            f'train-k={args.train_k} exceeds train candidate-pool size {len(train_pool_df)}.'
        )
    if args.test_k > len(test_pool_df):
        raise ValueError(
            f'test-k={args.test_k} exceeds test candidate-pool size {len(test_pool_df)}.'
        )

    recipe_to_idx, families_full, joint_sq_full, scale_ela, scale_performance = build_joint_distance(
        performance_df, ela_df, args.w_ela, args.w_performance, args.trim_fraction,
    )

    train_result = select_from_half(
        half_name='train',
        half_recipe_ids=train_pool_df['recipe_id'].tolist(),
        recipe_to_idx=recipe_to_idx,
        families_full=families_full,
        joint_sq_full=joint_sq_full,
        k=args.train_k,
        lower_pct=train_lower_pct,
        upper_pct=train_upper_pct,
        spacing_lambda=args.spacing_lambda,
        spacing_mu=args.spacing_mu,
        max_swap_passes=args.max_swap_passes,
    )
    test_result = select_from_half(
        half_name='test',
        half_recipe_ids=test_pool_df['recipe_id'].tolist(),
        recipe_to_idx=recipe_to_idx,
        families_full=families_full,
        joint_sq_full=joint_sq_full,
        k=args.test_k,
        lower_pct=test_lower_pct,
        upper_pct=test_upper_pct,
        spacing_lambda=args.spacing_lambda,
        spacing_mu=args.spacing_mu,
        max_swap_passes=args.max_swap_passes,
    )

    # Disjointness: guaranteed by construction; verified explicitly
    train_ids_set = set(train_result['selected_df']['recipe_id'])
    test_ids_set  = set(test_result['selected_df']['recipe_id'])
    overlap = train_ids_set & test_ids_set
    assert not overlap, f'Train/test overlap detected: {overlap}'

    # Cross-split separation: joint distances between every train-selected and test-selected recipe
    cross_sq = joint_sq_full[np.ix_(
        train_result['global_selected_indices'],
        test_result['global_selected_indices'],
    )]
    cross_min  = float(cross_sq.min())
    cross_mean = float(cross_sq.mean())
    cross_max  = float(cross_sq.max())

    # Save CSVs
    train_pool_df.to_csv(out_dir / 'train_pool_half.csv', index=False)
    test_pool_df.to_csv(out_dir / 'test_pool_half.csv', index=False)
    omitted_path = out_dir / 'omitted_pool_recipes.csv'
    if len(omitted_pool_df) > 0:
        omitted_pool_df.to_csv(omitted_path, index=False)
    elif omitted_path.exists():
        omitted_path.unlink()

    train_selected_df = train_result['selected_df']
    test_selected_df  = test_result['selected_df']
    train_selected_df.to_csv(out_dir / 'train_selected.csv', index=False)
    test_selected_df.to_csv(out_dir / 'test_selected.csv', index=False)
    pd.concat([train_selected_df, test_selected_df], ignore_index=True).to_csv(
        out_dir / 'train_test_selected.csv', index=False
    )

    # ------------------------------------------------------------------
    # Report helpers
    # ------------------------------------------------------------------
    def family_cov_stats(family, pool_families, coverage_per_pool):
        idxs = [i for i, f in enumerate(pool_families) if f == family]
        if not idxs:
            return None
        v = coverage_per_pool[idxs]
        return int(len(idxs)), float(v.mean()), float(v.max())

    def family_nnd_stats(family, sel_fam_list, nnd_within):
        idxs = [i for i, f in enumerate(sel_fam_list) if f == family]
        if not idxs:
            return None
        v = nnd_within[idxs]
        return int(len(idxs)), float(v.min()), float(v.mean()), float(v.max())

    def write_half_section(lines: list[str], result: dict) -> None:
        half_name        = str(result['half_name'])
        k                = result['k']
        available_counts = result['available_counts']
        lower_bounds     = result['lower_bounds']
        derived_upper    = result['derived_upper']
        effective_upper  = result['effective_upper']
        selected_df      = result['selected_df']
        selected_counts  = selected_df['family'].value_counts().to_dict()
        coverage_per_pool = result['coverage_per_pool']
        nnd_within       = result['nnd_within']
        pool_families    = result['pool_families']
        sel_fam_list     = result['sel_fam_list']

        lines += [
            f'## {half_name.capitalize()} Selection',
            '',
            f'Candidate pool size: {result["half_size"]}  |  Target k: {k}',
            '',
            '### Bounds',
            '',
            '| Family | In candidate pool | Lower | Upper (derived) | Upper (effective) | Selected | Binding |',
            '|--------|--------------|-------|-----------------|-------------------|----------|---------|',
        ]
        for family in FAMILY_ORDER:
            s  = selected_counts.get(family, 0)
            lo = lower_bounds[family]
            hi = effective_upper[family]
            binding = 'upper' if s == hi else ('lower' if s == lo else 'free')
            lines.append(
                f'| {family} | {available_counts[family]} | {lo} | {derived_upper[family]} '
                f'| {hi} | {s} | {binding} |'
            )
        lines += [
            f'| **Total** | **{result["half_size"]}** | **{sum(lower_bounds.values())}** '
            f'| **{sum(derived_upper.values())}** | **{sum(effective_upper.values())}** '
            f'| **{k}** | |',
            '',
            '### Algorithm Stages',
            '',
            '| Stage | Coverage loss | Mean NND | Min pair dist | Composite score |',
            '|-------|--------------|----------|---------------|-----------------|',
            f'| A: Greedy | {result["greedy_loss"]:.4f} | {result["greedy_mean_nnd"]:.4f} '
            f'| {result["greedy_min_pair"]:.4f} | {result["greedy_score"]:.4f} |',
            f'| B: Swap   | {result["final_loss"]:.4f} | {result["final_mean_nnd"]:.4f} '
            f'| {result["final_min_pair"]:.4f} | {result["final_score"]:.4f} |',
            f'| Improvement | {result["greedy_loss"] - result["final_loss"]:.4f} '
            f'| {result["final_mean_nnd"] - result["greedy_mean_nnd"]:+.4f} '
            f'| {result["final_min_pair"] - result["greedy_min_pair"]:+.4f} '
            f'| {result["final_score"] - result["greedy_score"]:+.4f} |',
            '',
            '### Coverage Analysis',
            '',
            f'- **Total coverage loss C(S)**: {coverage_per_pool.sum():.4f}',
            f'- **Mean coverage distance**: {coverage_per_pool.mean():.4f}',
            f'- **Max coverage distance**: {coverage_per_pool.max():.4f}',
            '',
            '| Family | Pool recipes | Mean coverage | Max coverage |',
            '|--------|-------------|---------------|--------------|',
        ]
        for family in FAMILY_ORDER:
            stats = family_cov_stats(family, pool_families, coverage_per_pool)
            if stats:
                np_, mc, mx = stats
                lines.append(f'| {family} | {np_} | {mc:.4f} | {mx:.4f} |')

        lines += [
            '',
            '### Spacing Within Selection',
            '',
            f'- **Min NND**: {nnd_within.min():.4f}  (above zero -- no internal near-duplicates)',
            f'- **Mean NND**: {nnd_within.mean():.4f}',
            f'- **Median NND**: {float(np.median(nnd_within)):.4f}',
            f'- **Max NND**: {nnd_within.max():.4f}',
            '',
            '| Family | Selected | NND min | NND mean | NND max |',
            '|--------|----------|---------|----------|---------|',
        ]
        for family in FAMILY_ORDER:
            stats = family_nnd_stats(family, sel_fam_list, nnd_within)
            if stats:
                n_, mn, me, mx = stats
                lines.append(f'| {family} | {n_} | {mn:.3f} | {me:.3f} | {mx:.3f} |')
        lines.append('')

    # ------------------------------------------------------------------
    # Assemble report
    # ------------------------------------------------------------------
    train_pool_fc = train_pool_df['family'].value_counts().to_dict()
    test_pool_fc  = test_pool_df['family'].value_counts().to_dict()
    omitted_pool_fc = omitted_pool_df['family'].value_counts().to_dict()
    full_pool_fc = pool_df['family'].value_counts().to_dict()

    report_lines = [
        '# Train/Test Selection Report',
        '',
        '## Configuration',
        '',
        '| Parameter | Value |',
        '|-----------|-------|',
        f'| Pool file | {Path(args.pool_file).name} |',
        f'| Pool size | {len(pool_df)} |',
        f'| Split seed | {args.split_seed} |',
        f'| Train k | {args.train_k} |',
        f'| Train family lower percentages | {format_percentages(train_lower_pct)} |',
        f'| Train family upper percentages | {format_percentages(train_upper_pct)} |',
        f'| Test k | {args.test_k} |',
        f'| Test family lower percentages | {format_percentages(test_lower_pct)} |',
        f'| Test family upper percentages | {format_percentages(test_upper_pct)} |',
        f'| ELA weight | {args.w_ela:.2f} |',
        f'| Performance weight | {args.w_performance:.2f} |',
        f'| Trim fraction | {args.trim_fraction:.0%} |',
        f'| ELA normalization scale | {scale_ela:.4f} |',
        f'| Performance normalization scale | {scale_performance:.4f} |',
        f'| Spacing lambda | {args.spacing_lambda:.2f} |',
        f'| Spacing mu | {args.spacing_mu:.2f} |',
        f'| Maximum local-swap passes | {args.max_swap_passes} |',
        '',
        '## Pool Split Summary',
        '',
        '| Family | Full pool | Train candidates | Test candidates | Omitted |',
        '|--------|-----------|------------------|-----------------|---------|',
    ]
    for family in FAMILY_ORDER:
        n_train = train_pool_fc.get(family, 0)
        n_test  = test_pool_fc.get(family, 0)
        n_omitted = omitted_pool_fc.get(family, 0)
        report_lines.append(
            f'| {family} | {full_pool_fc.get(family, 0)} | {n_train} | {n_test} | {n_omitted} |'
        )
    report_lines += [
        f'| **Total** | **{len(pool_df)}** | **{len(train_pool_df)}** '
        f'| **{len(test_pool_df)}** | **{len(omitted_pool_df)}** |',
        '',
        'For a family eligible on both sides, recipes are shuffled independently (seed='
        f'{args.split_seed}) and divided as evenly as possible. If that balanced allocation '
        'cannot support the requested side-specific cardinalities and family bounds, the '
        'per-family candidate counts are changed by the minimum amount needed for feasibility. '
        'A family with a zero upper '
        'bound on one side is assigned entirely to the other eligible side; a family with '
        'zero upper bounds on both sides is omitted.',
        'The partition does not use ELA/Performance geometry. Train and test candidate pools are '
        'non-overlapping by construction.',
        'Under the default shared-family protocol, this preserves broad taxonomy composition '
        'without deliberately matching train and test recipes in feature space. Separate '
        'target sizes and family bounds can instead define different family mixtures or '
        'family-holdout protocols.',
        '',
    ]

    write_half_section(report_lines, train_result)
    write_half_section(report_lines, test_result)

    report_lines += [
        '## Cross-Split Separation',
        '',
        'Joint distances (trimmed-mean-normalized) between every train-selected recipe',
        'and every test-selected recipe.',
        '',
        f'- **Min cross-split distance**: {cross_min:.4f}',
        f'- **Mean cross-split distance**: {cross_mean:.4f}',
        f'- **Max cross-split distance**: {cross_max:.4f}',
        '',
        'Train and test selected sets are **disjoint by construction** (each candidate pool',
        'is non-overlapping), and this was verified programmatically: no recipe identifier',
        'appears in both sets. The minimum joint feature-space distance is '
        f'{cross_min:.4f}; it quantifies the closest cross-split pair under the configured',
        'representation rather than serving as the disjointness criterion.',
        '',
        '## Output Files',
        '',
        '| File | Content |',
        '|------|---------|',
        '| train_pool_half.csv | Train candidate pool (legacy filename retained) |',
        '| test_pool_half.csv | Test candidate pool (legacy filename retained) |',
        '| omitted_pool_recipes.csv | Recipes excluded from both sides, written only when nonempty |',
        '| train_selected.csv | Final selected train instances |',
        '| test_selected.csv | Final selected test instances |',
        '| train_test_selected.csv | Combined train+test selected instances with split label |',
        '| train_test_selection_report.md | This report |',
    ]

    (out_dir / 'train_test_selection_report.md').write_text(
        '\n'.join(report_lines) + '\n', encoding='utf-8'
    )

    print('=== Train/Test Selection from Deduplicated Pool ===')
    print(
        f'  Pool: {len(pool_df)}  |  Candidates: train={len(train_pool_df)} '
        f'test={len(test_pool_df)} omitted={len(omitted_pool_df)}'
    )
    print(f'  Selected: train={len(train_selected_df)}  test={len(test_selected_df)}')
    print(f'  Scales: ELA={scale_ela:.4f}  Performance={scale_performance:.4f}')
    print(f'  Cross-split separation: min={cross_min:.4f}  mean={cross_mean:.4f}  max={cross_max:.4f}')
    print(f'  Train/test overlap: {len(overlap)} recipes (expected 0)')
    print('  Saved: train_pool_half.csv, test_pool_half.csv, train_selected.csv,')
    print('         test_selected.csv, train_test_selected.csv, train_test_selection_report.md')


if __name__ == '__main__':
    main()
