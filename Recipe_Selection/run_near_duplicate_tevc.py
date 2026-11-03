"""
PORTAL TEVC-Aligned Near-Duplicate Detection
============================================

Implements the duplicate-removal protocol using P95-normalized distances:

    delta_ELA(i,j) = ||phi_i - phi_j||_2 / P95(d_ELA)
    delta_performance(i,j) = ||y_i - y_j||_2 / P95(d_performance)

A pair is confirmed as a near-duplicate only when:

    delta_ELA(i,j) <= tau  AND  delta_performance(i,j) <= tau

Outputs are written to near_dup_output/ with a tau-specific filename suffix.
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

import os
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

try:
    import pulp
except ImportError:  # Optional only when the explicit greedy backend is used.
    pulp = None

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # project root (one level up from Recipe_Selection/)


def _environment_float_list(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    values = [float(item.strip()) for item in raw.split(',') if item.strip()]
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f'{name} must contain positive finite comma-separated values.')
    return values


PIPELINE_DIR = Path(os.environ.get('PORTAL_PIPELINE_DIR', ROOT / 'pipeline_output')).resolve()
OUT_DIR = Path(os.environ.get('PORTAL_NEAR_DUP_OUT_DIR', ROOT / 'near_dup_output')).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = _environment_float_list(
    'PORTAL_DEDUP_THRESHOLDS', [0.03, 0.05, 0.08, 0.10]
)
PRIMARY_TAU = float(os.environ.get('PORTAL_PRIMARY_TAU', '0.05'))
if not math.isfinite(PRIMARY_TAU) or PRIMARY_TAU <= 0.0:
    raise ValueError('PORTAL_PRIMARY_TAU must be positive and finite.')
if not any(abs(value - PRIMARY_TAU) <= 1e-12 for value in THRESHOLDS):
    raise ValueError('PORTAL_PRIMARY_TAU must occur in PORTAL_DEDUP_THRESHOLDS.')
FAMILY_ORDER = [
    item.strip()
    for item in os.environ.get(
        'PORTAL_FAMILY_ORDER', 'Family1,Family2,Family3,Family4'
    ).split(',')
    if item.strip()
]
if not FAMILY_ORDER or len(FAMILY_ORDER) != len(set(FAMILY_ORDER)):
    raise ValueError('PORTAL_FAMILY_ORDER must contain unique family labels.')
_BASE_FAMILY_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]
FAMILY_COLORS = {
    family: _BASE_FAMILY_COLORS[index % len(_BASE_FAMILY_COLORS)]
    for index, family in enumerate(FAMILY_ORDER)
}
MIS_BACKEND = os.environ.get('PORTAL_MIS_BACKEND', 'auto').strip().casefold()
if MIS_BACKEND not in {'auto', 'cbc', 'greedy'}:
    raise ValueError('PORTAL_MIS_BACKEND must be auto, cbc, or greedy.')


def tau_token(value: float) -> str:
    return format(float(value), '.12g').replace('-', 'm').replace('.', 'p')


def condensed_to_pairs(flags: np.ndarray, n_nodes: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    k = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if flags[k]:
                pairs.append((i, j))
            k += 1
    return pairs


def solve_mis_ilp(
    n_nodes: int,
    edges: list[tuple[int, int]],
    recipe_ids: np.ndarray,
    nnd_joint: np.ndarray,
    nnd_performance: np.ndarray,
    nnd_ela: np.ndarray,
) -> list[int]:
    if pulp is None:
        raise RuntimeError('PuLP is required for the ILP MIS backend.')

    prob = pulp.LpProblem('MIS', pulp.LpMaximize)
    x = [pulp.LpVariable(f'x_{i}', cat='Binary') for i in range(n_nodes)]
    for i, j in edges:
        prob += x[i] + x[j] <= 1
    solver = pulp.PULP_CBC_CMD(msg=0)
    if not solver.available():
        raise RuntimeError('PuLP is installed but its CBC backend is unavailable.')

    # Stage 1 enforces the actual maximum-cardinality independent set. Keeping
    # tie-breaking out of this objective prevents any secondary preference from
    # trading away a retained recipe.
    cardinality = pulp.lpSum(x)
    prob += cardinality
    status = prob.solve(solver)
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(
            f'MIS integer program did not reach an optimal solution: '
            f'{pulp.LpStatus.get(status, status)}'
        )
    maximum_size = int(round(sum(float(pulp.value(variable)) for variable in x)))

    # Stage 2 chooses reproducibly among maximum-cardinality solutions. These
    # are the established PORTAL isolation preferences used for the released
    # de-duplicated pool. Cardinality is fixed explicitly above, so the
    # secondary coefficients cannot trade away a retained recipe.
    id_ranks = rankdata([str(recipe_id) for recipe_id in recipe_ids]) / max(n_nodes, 1)
    secondary = (
        1e-3 * nnd_joint
        + 1e-4 * nnd_performance
        + 1e-5 * nnd_ela
        - 1e-6 * id_ranks
    )
    prob += cardinality == maximum_size
    prob.setObjective(pulp.lpSum(float(secondary[i]) * x[i] for i in range(n_nodes)))
    status = prob.solve(solver)
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(
            'MIS tie-break integer program did not reach an optimal solution: '
            f'{pulp.LpStatus.get(status, status)}'
        )
    return [i for i in range(n_nodes) if pulp.value(x[i]) > 0.5]


def solve_mis_greedy(n_nodes: int, edges: list[tuple[int, int]]) -> list[int]:
    adjacency = {i: set() for i in range(n_nodes)}
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)

    remaining = set(range(n_nodes))
    selected: list[int] = []
    while remaining:
        node = min(remaining, key=lambda idx: len(adjacency[idx] & remaining))
        selected.append(node)
        remaining -= {node} | (adjacency[node] & remaining)
    return selected


def build_pairs_df(
    pairs: list[tuple[int, int]],
    recipe_ids: np.ndarray,
    families: np.ndarray,
    raw_ela: np.ndarray,
    raw_performance: np.ndarray,
    delta_ela: np.ndarray,
    delta_performance: np.ndarray,
) -> pd.DataFrame:
    ij_to_k: dict[tuple[int, int], int] = {}
    k = 0
    n_nodes = len(recipe_ids)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            ij_to_k[(i, j)] = k
            k += 1

    columns = [
        'recipe_i', 'family_i', 'recipe_j', 'family_j', 'same_family',
        'delta_ELA', 'delta_performance', 'd_ELA_raw', 'd_performance_raw',
    ]
    rows = []
    for i, j in pairs:
        idx = ij_to_k[(i, j)]
        rows.append(
            {
                'recipe_i': recipe_ids[i],
                'family_i': families[i],
                'recipe_j': recipe_ids[j],
                'family_j': families[j],
                'same_family': families[i] == families[j],
                'delta_ELA': float(delta_ela[idx]),
                'delta_performance': float(delta_performance[idx]),
                'd_ELA_raw': float(raw_ela[idx]),
                'd_performance_raw': float(raw_performance[idx]),
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(['delta_ELA', 'delta_performance'], ascending=[True, True])


def align_cleaned_tables(
    performance: pd.DataFrame,
    ela: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate recipe metadata and align ELA rows to Performance deterministically."""

    required = {'recipe_id', 'family', 'family_id'}
    for name, frame in (('Performance', performance), ('ELA', ela)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'{name} cleaned table is missing metadata columns: {sorted(missing)}')
        if frame['recipe_id'].astype(str).duplicated().any():
            raise ValueError(f'{name} cleaned table contains duplicate recipe_id values.')
    performance = performance.copy()
    ela = ela.copy()
    performance['recipe_id'] = performance['recipe_id'].astype(str)
    ela['recipe_id'] = ela['recipe_id'].astype(str)
    performance_ids = performance['recipe_id'].tolist()
    if set(performance_ids) != set(ela['recipe_id']):
        raise ValueError('Performance and ELA cleaned tables contain different recipe sets.')
    ela = ela.set_index('recipe_id', drop=False).loc[performance_ids].reset_index(drop=True)
    if not np.array_equal(
        performance['family'].astype(str).to_numpy(),
        ela['family'].astype(str).to_numpy(),
    ):
        raise ValueError('Performance and ELA family metadata disagree after recipe alignment.')
    return performance.reset_index(drop=True), ela


performance_df = read_characterization_csv(PIPELINE_DIR / 'performance_cleaned.csv')
ela_df = read_characterization_csv(PIPELINE_DIR / 'ela_cleaned.csv')
performance_df, ela_df = align_cleaned_tables(performance_df, ela_df)

meta_cols = ['recipe_id', 'family', 'family_id']
recipe_ids = performance_df['recipe_id'].to_numpy()
families = performance_df['family'].to_numpy()
n = len(recipe_ids)
if n < 2:
    raise ValueError('Near-duplicate detection requires at least two recipes.')
unknown_families = sorted(set(map(str, families)) - set(FAMILY_ORDER))
if unknown_families:
    raise ValueError(
        'Cleaned tables contain families absent from PORTAL_FAMILY_ORDER: '
        f'{unknown_families}'
    )

performance_x = performance_df.drop(columns=meta_cols).to_numpy()
ela_x = ela_df.drop(columns=meta_cols).to_numpy()

raw_ela = pdist(ela_x, metric='euclidean')
raw_performance = pdist(performance_x, metric='euclidean')
n_pairs = len(raw_ela)

p95_ela = np.percentile(raw_ela, 95)
p95_performance = np.percentile(raw_performance, 95)
if not np.isfinite(p95_ela) or p95_ela <= 0.0:
    raise ValueError(f'ELA P95 distance scale must be positive and finite; received {p95_ela}.')
if not np.isfinite(p95_performance) or p95_performance <= 0.0:
    raise ValueError(f'Performance P95 distance scale must be positive and finite; received {p95_performance}.')
delta_ela = raw_ela / p95_ela
delta_performance = raw_performance / p95_performance

delta_ela_sq = squareform(delta_ela)
delta_performance_sq = squareform(delta_performance)
np.fill_diagonal(delta_ela_sq, np.inf)
np.fill_diagonal(delta_performance_sq, np.inf)
delta_joint_sq = 0.5 * delta_ela_sq + 0.5 * delta_performance_sq

nnd_ela = delta_ela_sq.min(axis=1)
nnd_performance = delta_performance_sq.min(axis=1)
nnd_joint = delta_joint_sq.min(axis=1)

use_ilp = pulp is not None and MIS_BACKEND != 'greedy'
if pulp is None and MIS_BACKEND == 'cbc':
    raise RuntimeError(
        'CBC/PuLP is required by PORTAL_MIS_BACKEND=cbc but PuLP is not installed.'
    )

sensitivity_rows = []
mis_rows = []
mis_pools: dict[float, list[int]] = {}

for tau in THRESHOLDS:
    ela_flag = delta_ela <= tau
    performance_flag = delta_performance <= tau
    or_flag = ela_flag | performance_flag
    and_flag = ela_flag & performance_flag

    sensitivity_rows.append(
        {
            'tau': tau,
            'n_OR_pairs': int(or_flag.sum()),
            'n_ELA_only': int((ela_flag & ~performance_flag).sum()),
            'n_performance_only': int((performance_flag & ~ela_flag).sum()),
            'n_AND_pairs': int(and_flag.sum()),
            'pct_OR': float(or_flag.mean() * 100.0),
            'pct_AND': float(and_flag.mean() * 100.0),
        }
    )

    and_pairs = condensed_to_pairs(and_flag, n)
    print(
        f'De-duplication sensitivity: tau={tau:.6g} | '
        f'AND edges={len(and_pairs):,} | solving redundancy graph ...',
        flush=True,
    )
    if not and_pairs:
        retained = list(range(n))
        method = 'trivial'
    elif use_ilp:
        retained = solve_mis_ilp(n, and_pairs, recipe_ids, nnd_joint, nnd_performance, nnd_ela)
        method = 'ILP'
    else:
        retained = solve_mis_greedy(n, and_pairs)
        method = 'greedy'

    mis_pools[tau] = retained
    row = {
        'tau': tau,
        'n_AND_pairs': len(and_pairs),
        'pool_size': len(retained),
        'removed': n - len(retained),
        'method': method,
    }
    for family in FAMILY_ORDER:
        row[family] = int((families[retained] == family).sum())
    mis_rows.append(row)
    print(
        f'De-duplication sensitivity: tau={tau:.6g} | '
        f'pool={len(retained):,}/{n:,} | removed={n - len(retained):,} | method={method}',
        flush=True,
    )

sensitivity_df = pd.DataFrame(sensitivity_rows)
mis_df = pd.DataFrame(mis_rows)

sensitivity_df.to_csv(OUT_DIR / 'sensitivity_table_tau.csv', index=False)
mis_df.to_csv(OUT_DIR / 'mis_sweep_tau.csv', index=False)

primary_pairs_mask = (delta_ela <= PRIMARY_TAU) & (delta_performance <= PRIMARY_TAU)
primary_pairs = condensed_to_pairs(primary_pairs_mask, n)
primary_pool = mis_pools[PRIMARY_TAU]

primary_pairs_df = build_pairs_df(
    primary_pairs,
    recipe_ids,
    families,
    raw_ela,
    raw_performance,
    delta_ela,
    delta_performance,
)
primary_pairs_path = OUT_DIR / f'confirmed_pairs_tau_{tau_token(PRIMARY_TAU)}.csv'
primary_pairs_df.to_csv(primary_pairs_path, index=False)

primary_pool_df = pd.DataFrame(
    {
        'recipe_id': recipe_ids[primary_pool],
        'family': families[primary_pool],
        'nnd_joint': nnd_joint[primary_pool],
        'nnd_performance': nnd_performance[primary_pool],
        'nnd_ela': nnd_ela[primary_pool],
    }
).sort_values(['family', 'recipe_id'])
primary_pool_path = OUT_DIR / f'mis_pool_tau_{tau_token(PRIMARY_TAU)}.csv'
primary_pool_df.to_csv(primary_pool_path, index=False)

# Compare the four possible single-/dual-space graph rules at the primary
# threshold. Every rule is resolved with the same MIS formulation and
# deterministic tie-break, so differences are attributable to the edge rule.
primary_ela_mask = delta_ela <= PRIMARY_TAU
primary_performance_mask = delta_performance <= PRIMARY_TAU
rule_masks = {
    'ELA': primary_ela_mask,
    'Performance': primary_performance_mask,
    'OR': primary_ela_mask | primary_performance_mask,
    'AND': primary_ela_mask & primary_performance_mask,
}
rule_pools: dict[str, list[int]] = {'AND': primary_pool}
rule_comparison_rows = []
for rule_name, rule_mask in rule_masks.items():
    rule_pairs = condensed_to_pairs(rule_mask, n)
    if rule_name not in rule_pools:
        print(f'  Solving {rule_name}-rule MIS ({len(rule_pairs):,} edges) ...', flush=True)
        if not rule_pairs:
            rule_pools[rule_name] = list(range(n))
        elif use_ilp:
            rule_pools[rule_name] = solve_mis_ilp(
                n, rule_pairs, recipe_ids, nnd_joint, nnd_performance, nnd_ela
            )
        else:
            rule_pools[rule_name] = solve_mis_greedy(n, rule_pairs)

    retained = rule_pools[rule_name]
    row = {
        'rule': rule_name,
        'n_edges': len(rule_pairs),
        'retained_total': len(retained),
        'removed_total': n - len(retained),
        'method': 'ILP' if use_ilp and rule_pairs else ('greedy' if rule_pairs else 'trivial'),
    }
    for family in FAMILY_ORDER:
        retained_family = int((families[retained] == family).sum())
        original_family = int((families == family).sum())
        row[f'{family}_retained'] = retained_family
        row[f'{family}_removed'] = original_family - retained_family
    rule_comparison_rows.append(row)

rule_comparison_df = pd.DataFrame(rule_comparison_rows)
rule_comparison_path = OUT_DIR / f'rule_comparison_tau_{tau_token(PRIMARY_TAU)}.csv'
rule_comparison_df.to_csv(rule_comparison_path, index=False)

cross_family_pairs = int((~primary_pairs_df['same_family']).sum()) if not primary_pairs_df.empty else 0
within_family_pairs = int(primary_pairs_df['same_family'].sum()) if not primary_pairs_df.empty else 0

# ---------------------------------------------------------------------------
# Redundancy statistics (within/cross-family, degree, clusters, removal reasons)
# ---------------------------------------------------------------------------

retained_set = set(recipe_ids[primary_pool])
removed_set  = set(recipe_ids) - retained_set
family_of    = dict(zip(recipe_ids, families))

# Adjacency from AND pairs
adj: dict[str, set[str]] = defaultdict(set)
for _, row in primary_pairs_df.iterrows():
    adj[row['recipe_i']].add(row['recipe_j'])
    adj[row['recipe_j']].add(row['recipe_i'])

degrees = {r: len(adj[r]) for r in recipe_ids}

# Connected components
visited: set[str] = set()
components: list[list[str]] = []
for start in recipe_ids:
    if degrees[start] > 0 and start not in visited:
        cluster: list[str] = []
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            queue.extend(nb for nb in adj[node] if nb not in visited)
        components.append(cluster)

pure_within = sum(1 for c in components if len({family_of[r] for r in c}) == 1)
mixed       = len(components) - pure_within

# Cross-family pair matrix
cross_matrix = pd.DataFrame(0, index=FAMILY_ORDER, columns=FAMILY_ORDER)
for _, row in primary_pairs_df[~primary_pairs_df['same_family']].iterrows():
    cross_matrix.loc[row['family_i'], row['family_j']] += 1
    cross_matrix.loc[row['family_j'], row['family_i']] += 1

# Removal reasons
removal_records = []
for r in sorted(removed_set):
    partners       = list(adj[r])
    my_fam         = family_of[r]
    cross_partners = [p for p in partners if family_of[p] != my_fam]
    removal_records.append({
        'recipe_id'        : r,
        'family'           : my_fam,
        'degree'           : len(partners),
        'only_within'      : len(cross_partners) == 0,
        'has_cross'        : len(cross_partners) > 0,
        'cross_fam_set'    : frozenset(family_of[p] for p in partners if family_of[p] != my_fam),
    })
removal_df = pd.DataFrame(
    removal_records,
    columns=[
        'recipe_id', 'family', 'degree', 'only_within', 'has_cross',
        'cross_fam_set',
    ],
)
removal_df.to_csv(OUT_DIR / 'removal_reason_tevc.csv', index=False)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Figure 1 - delta_ELA vs delta_performance scatter (primary tau)
and_mask_full = (delta_ela <= PRIMARY_TAU) & (delta_performance <= PRIMARY_TAU)
or_only_mask  = ((delta_ela <= PRIMARY_TAU) | (delta_performance <= PRIMARY_TAU)) & ~and_mask_full

fig, ax = plt.subplots(figsize=(4.4, 4.0))
# Shade the lower-left intersection before drawing the pair markers.
ax.fill_between(
    [0, PRIMARY_TAU], [0, 0], [PRIMARY_TAU, PRIMARY_TAU],
    color='#fff0a6', alpha=0.65, zorder=0, label='AND region',
)
ax.scatter(delta_ela[or_only_mask], delta_performance[or_only_mask],
           s=3.5, alpha=0.28, color='#ff7f0e', linewidths=0,
           label=f'One-space only ({or_only_mask.sum():,})')
ax.scatter(delta_ela[and_mask_full], delta_performance[and_mask_full],
           s=6, alpha=0.72, color='#d62728', zorder=5, linewidths=0,
           label=f'AND-confirmed ({and_mask_full.sum():,})')
ax.axvline(PRIMARY_TAU, color='#1f77b4', lw=1.7, ls='--', zorder=6,
           label='ELA threshold')
ax.axhline(PRIMARY_TAU, color='#d62728', lw=1.7, ls='--', zorder=6,
           label='Performance threshold')
ax.set_xlabel(r'$\delta_{\mathrm{ELA}}$ (P95-normalized)', fontsize=10.5)
ax.set_ylabel(r'$\delta_{\mathrm{perf}}$ (P95-normalized)', fontsize=10.5)
# Keep a small linear neighborhood around zero and use logarithmic scaling
# elsewhere so the densely populated threshold intersection remains visible.
ax.set_xscale('symlog', linthresh=0.005, linscale=1.0)
ax.set_yscale('symlog', linthresh=0.005, linscale=1.0)
ax.set_xlim(0, min(delta_ela.max() * 1.05, 3.0))
ax.set_ylim(0, min(delta_performance.max() * 1.05, 3.0))
legend = ax.legend(
    fontsize=9, loc='lower right', framealpha=1.0, facecolor='white',
    edgecolor='#999999', borderpad=0.55, labelspacing=0.40,
    handlelength=1.8,
)
legend.set_zorder(100)
ax.tick_params(axis='both', which='major', labelsize=9)
ax.grid(alpha=0.16, which='both')
plt.tight_layout()
fig.savefig(OUT_DIR / 'tevc_distance_scatter.png', dpi=150, bbox_inches='tight')
fig.savefig(OUT_DIR / 'tevc_distance_scatter.pdf', bbox_inches='tight')
plt.close(fig)

# Figure 2 - sensitivity bar chart (pool size + pair counts)
merged_df = sensitivity_df.merge(mis_df[['tau', 'pool_size', 'removed']], on='tau')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

x = np.arange(len(THRESHOLDS))
labels = [f'tau={t:g}' for t in THRESHOLDS]

ax = axes[0]
ax.bar(x - 0.2, merged_df['n_AND_pairs'], 0.35, label='AND pairs', color='#d62728', alpha=0.85)
ax.bar(x + 0.2, merged_df['pool_size'],   0.35, label='Pool R size', color='#1f77b4', alpha=0.85)
for i, (r, p) in enumerate(zip(merged_df['n_AND_pairs'], merged_df['pool_size'])):
    ax.text(i - 0.2, r + 5, str(int(r)), ha='center', va='bottom', fontsize=9)
    ax.text(i + 0.2, p + 5, str(int(p)), ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Count')
ax.set_title('AND pairs and MIS pool size vs threshold', fontsize=10)
ax.legend(fontsize=9)

ax = axes[1]
n_family = max(len(FAMILY_ORDER), 1)
w = min(0.8 / n_family, 0.25)
for fi, fam in enumerate(FAMILY_ORDER):
    vals = [mis_df[mis_df['tau'] == t][fam].values[0] for t in THRESHOLDS]
    offset = (fi - (n_family - 1) / 2.0) * w
    ax.bar(x + offset, vals, w, label=fam, color=FAMILY_COLORS[fam], alpha=0.85)
orig = {f: int((families == f).sum()) for f in FAMILY_ORDER}
for fi, fam in enumerate(FAMILY_ORDER):
    ax.axhline(orig[fam], color=FAMILY_COLORS[fam], lw=1, ls=':', alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Recipes retained in pool R')
ax.set_title('Family composition of MIS pool vs threshold\n(dotted = original count)', fontsize=10)
ax.legend(fontsize=8)

fig.suptitle('TEVC-aligned near-duplicate sensitivity (P95-normalized distances)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / 'tevc_sensitivity.png', dpi=150, bbox_inches='tight')
fig.savefig(OUT_DIR / 'tevc_sensitivity.pdf', bbox_inches='tight')
plt.close(fig)

# Figure 3 - ELA vs Performance crowding (fraction of pairs within tau)
fig, ax = plt.subplots(figsize=(8, 5))
tau_min = max(0.001, min(THRESHOLDS) * 0.2)
tau_max = max(0.20, max(THRESHOLDS) * 1.25)
tau_range = np.linspace(tau_min, tau_max, 80)
ela_frac  = [(delta_ela <= t).mean() * 100 for t in tau_range]
performance_frac  = [(delta_performance <= t).mean() * 100 for t in tau_range]
ax.plot(tau_range, ela_frac, lw=2, color='#1f77b4', label='ELA')
ax.plot(tau_range, performance_frac, lw=2, color='#d62728', label='Performance')
for tau in THRESHOLDS:
    ax.axvline(tau, color='grey', lw=0.8, ls='--', alpha=0.6)
    ax.text(
        tau + 0.01 * tau_max,
        55,
        f'tau={tau:g}',
        fontsize=7,
        color='grey',
        rotation=90,
        va='top',
    )
ax.set_xlabel('tau threshold (P95-normalized distance)', fontsize=11)
ax.set_ylabel('% of all pairs within tau', fontsize=11)
ax.set_title('ELA vs Performance crowding: fraction of pairs within each threshold\n'
             'ELA and Performance contribute differently to the AND rule',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(tau_min, tau_max)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / 'tevc_crowding.png', dpi=150, bbox_inches='tight')
fig.savefig(OUT_DIR / 'tevc_crowding.pdf', bbox_inches='tight')
plt.close(fig)

print('  Saved figures: tevc_distance_scatter, tevc_sensitivity, and tevc_crowding (.png/.pdf)')

# ---------------------------------------------------------------------------
# Expanded report
# ---------------------------------------------------------------------------

ela_primary_count = int((delta_ela <= PRIMARY_TAU).sum())
performance_primary_count = int((delta_performance <= PRIMARY_TAU).sum())
ela_primary_pct = 100.0 * ela_primary_count / n_pairs
performance_primary_pct = 100.0 * performance_primary_count / n_pairs
if ela_primary_count > performance_primary_count:
    crowding_summary = (
        'ELA is more crowded than Performance at the primary normalized-distance threshold; '
        'Performance is therefore the more restrictive side of the AND rule for this data set.'
    )
elif performance_primary_count > ela_primary_count:
    crowding_summary = (
        'Performance is more crowded than ELA at the primary normalized-distance threshold; '
        'ELA is therefore the more restrictive side of the AND rule for this data set.'
    )
else:
    crowding_summary = (
        'ELA and Performance contain the same number of close pairs at the primary threshold, '
        'although their pair identities need not coincide.'
    )

mis_method = 'two-stage ILP via PuLP/CBC' if use_ilp else 'deterministic greedy heuristic'
family_rule_headers = ' | '.join(
    f'{family} retained (removed)' for family in FAMILY_ORDER
)
family_rule_divider = '|'.join(['----------------------:'] * len(FAMILY_ORDER))
family_sweep_headers = ' | '.join(FAMILY_ORDER)
family_sweep_divider = '|'.join(['----'] * len(FAMILY_ORDER))
family_matrix_header = '| | ' + ' | '.join(FAMILY_ORDER) + ' |'
family_matrix_divider = '|-|' + '|'.join(['---------'] * len(FAMILY_ORDER)) + '|'
threshold_text = ', '.join(f'{tau:g}' for tau in THRESHOLDS)

report_lines = [
    '# PORTAL TEVC-Aligned Near-Duplicate Detection',
    '',
    f'Generated for {n} recipes across {len(FAMILY_ORDER)} configured families.',
    f'Primary threshold: τ = {PRIMARY_TAU:g}  |  Rule: AND '
    '(both δ_ELA ≤ τ and δ_PERFORMANCE ≤ τ)',
    '',
    '---',
    '',
    '## 1. Methodology',
    '',
    f'ELA feature space: {ela_x.shape[1]} dimensions  |  '
    f'Performance feature space: {performance_x.shape[1]} dimensions',
    f'Total recipe pairs: {n_pairs:,}',
    '',
    'Euclidean distances are normalized by the 95th percentile (P95) of all '
    'pairwise distances in each representation:',
    '',
    f'    δ_ELA(i,j) = ||φ_i − φ_j||₂ / P95(d_ELA)   [P95 = {p95_ela:.6g}]',
    f'    δ_PERFORMANCE(i,j) = ||y_i − y_j||₂ / P95(d_performance)   [P95 = {p95_performance:.6g}]',
    '',
    'A pair is confirmed as a near-duplicate only when:',
    '',
    '    δ_ELA(i,j) ≤ τ  AND  δ_PERFORMANCE(i,j) ≤ τ',
    '',
    f'At the primary threshold, each representation must place the pair within '
    f'{100.0 * PRIMARY_TAU:g}% of its own P95 distance scale.',
    '',
    f'Redundancy-graph resolution: {mis_method}.',
    'When ILP is used, cardinality is optimized first. A second optimization '
    'then favors isolated recipes using joint, Performance, and ELA nearest-neighbor '
    'ranks, followed by a deterministic recipe-ID preference.',
    '',
    '---',
    '',
    '## 2. ELA and Performance Crowding',
    '',
    '| τ | ELA pairs within τ | Performance pairs within τ | Ratio ELA/Performance |',
    '|---|---------------------|---------------------|---------------|',
]

for tau in THRESHOLDS:
    n_ela_t = int((delta_ela <= tau).sum())
    n_performance_t = int((delta_performance <= tau).sum())
    ratio = n_ela_t / max(n_performance_t, 1)
    report_lines.append(
        f'| {tau:g} | {n_ela_t:,} ({100.0*n_ela_t/n_pairs:.1f}%) | '
        f'{n_performance_t:,} ({100.0*n_performance_t/n_pairs:.2f}%) | {ratio:.2f}× |'
    )

report_lines += [
    '',
    f'At τ={PRIMARY_TAU:g}, {ela_primary_count:,} ELA pairs '
    f'({ela_primary_pct:.1f}%) and {performance_primary_count:,} Performance pairs '
    f'({performance_primary_pct:.2f}%) lie within threshold.',
    crowding_summary,
    '',
    '### 2.1 Rule comparison at the primary threshold',
    '',
    f'| Rule | Graph edges | {family_rule_headers} | Total retained (removed) |',
    f'|------|------------:|{family_rule_divider}|-------------------------:|',
]

for _, row in rule_comparison_df.iterrows():
    family_cells = [
        f"{int(row[f'{family}_retained'])} ({int(row[f'{family}_removed'])})"
        for family in FAMILY_ORDER
    ]
    report_lines.append(
        f"| {row['rule']} | {int(row['n_edges']):,} | "
        + ' | '.join(family_cells)
        + f" | {int(row['retained_total'])} ({int(row['removed_total'])}) |"
    )

report_lines += [
    '',
    'ELA and Performance denote graphs constructed from one representation alone; OR '
    'takes the union of those edge sets, whereas AND takes their intersection.',
    'A single-space rule can discard recipes that differ in the other '
    'representation, and OR inherits both single-space risks. AND is the '
    'conservative rule because removal requires corroborating structural and '
    'behavioral proximity.',
    '',
    '---',
    '',
    '## 3. Threshold Sensitivity',
    '',
    f'| τ | OR pairs | ELA-only | Performance-only | AND pairs | Pool R | Removed | '
    f'{family_sweep_headers} |',
    f'|---|----------|----------|----------|-----------|--------|---------|'
    f'{family_sweep_divider}|',
]

mis_df_nomdup = mis_df.drop(columns=['n_AND_pairs'])
for _, row in sensitivity_df.merge(mis_df_nomdup, on='tau').iterrows():
    marker = '**' if abs(row['tau'] - PRIMARY_TAU) < 1e-12 else ''
    family_values = ' | '.join(str(int(row[family])) for family in FAMILY_ORDER)
    report_lines.append(
        f"| {marker}{row['tau']:g}{marker} | {int(row['n_OR_pairs']):,} | "
        f"{int(row['n_ELA_only']):,} | {int(row['n_performance_only']):,} | "
        f"{int(row['n_AND_pairs']):,} | {int(row['pool_size']):,} | "
        f"{int(row['removed']):,} | {family_values} |"
    )

report_lines += [
    '',
    'The bold row is the configured operating point. The sweep reports both '
    'edge counts and the resulting independent-pool cardinality, because a '
    'larger edge count does not translate one-for-one into removed recipes.',
    '',
    '---',
    '',
    f'## 4. Primary Result at τ = {PRIMARY_TAU:g}',
    '',
    f'Confirmed AND pairs: {len(primary_pairs):,}',
    f'  Within-family: {within_family_pairs:,} '
    f'({100.0*within_family_pairs/max(len(primary_pairs), 1):.1f}%)',
    f'  Cross-family: {cross_family_pairs:,} '
    f'({100.0*cross_family_pairs/max(len(primary_pairs), 1):.1f}%)',
    '',
    '| Family | Original | Retained | Removed | % Removed |',
    '|--------|----------|----------|---------|-----------|',
]

for family in FAMILY_ORDER:
    retained_count = int((primary_pool_df['family'] == family).sum())
    original_count = int((families == family).sum())
    removed_count = original_count - retained_count
    removed_pct = 100.0 * removed_count / original_count if original_count else 0.0
    report_lines.append(
        f'| {family} | {original_count} | {retained_count} | '
        f'{removed_count} | {removed_pct:.1f}% |'
    )

report_lines += [
    f'| **Total** | **{n}** | **{len(primary_pool)}** | '
    f'**{n-len(primary_pool)}** | **{100.0*(n-len(primary_pool))/n:.1f}%** |',
    '',
    'Cross-family AND-edge matrix (the diagonal is zero by definition):',
    '',
    family_matrix_header,
    family_matrix_divider,
]
for family_i in FAMILY_ORDER:
    row_values = ' | '.join(
        str(int(cross_matrix.loc[family_i, family_j]))
        for family_j in FAMILY_ORDER
    )
    report_lines.append(f'| {family_i} | {row_values} |')

report_lines += [
    '',
    '---',
    '',
    '## 5. Family-Level Removal Audit',
    '',
    '| Family | Incident AND edges | Removed | Within-only removals | '
    'Removals with cross-family partners | Cross partner families |',
    '|--------|-------------------:|--------:|---------------------:|'
    '------------------------------------:|------------------------|',
]

for family in FAMILY_ORDER:
    subset = removal_df[removal_df['family'] == family]
    incident_edges = int(
        (
            (primary_pairs_df['family_i'] == family)
            | (primary_pairs_df['family_j'] == family)
        ).sum()
    ) if not primary_pairs_df.empty else 0
    within_only = int(subset['only_within'].sum()) if not subset.empty else 0
    with_cross = int(subset['has_cross'].sum()) if not subset.empty else 0
    cross_family_counts: Counter[str] = Counter()
    for _, row in subset.iterrows():
        for partner_family in row['cross_fam_set']:
            cross_family_counts[partner_family] += 1
    cross_description = ', '.join(
        f'{name} ({count})' for name, count in sorted(cross_family_counts.items())
    ) or 'none'
    report_lines.append(
        f'| {family} | {incident_edges} | {len(subset)} | {within_only} | '
        f'{with_cross} | {cross_description} |'
    )

report_lines += [
    '',
    f'Redundancy graph: {len(components)} nontrivial connected components  |  '
    f'pure within-family: {pure_within}  |  mixed-family: {mixed}.',
    'Recipe-level partners, degrees, and removal classifications are retained '
    'in `removal_reason_tevc.csv` and the confirmed-pair table.',
    '',
    '---',
    '',
    '## 6. Figures and Output Files',
    '',
    '| File | Content |',
    '|------|---------|',
    '| tevc_distance_scatter.png / .pdf | P95-normalized ELA versus Performance '
    'distances on symmetric-log axes, with the AND region highlighted |',
    '| tevc_sensitivity.png / .pdf | Independent-pool size and family '
    'composition across thresholds |',
    '| tevc_crowding.png / .pdf | Fraction of ELA and Performance pairs within each threshold |',
    f'| sensitivity_table_tau.csv | OR/AND pair counts at τ in {{{threshold_text}}} |',
    '| mis_sweep_tau.csv | Independent-pool size and family counts at each threshold |',
    f'| {rule_comparison_path.name} | Family-wise ELA, Performance, OR, and AND outcomes '
    f'at τ={PRIMARY_TAU:g} |',
    f'| {primary_pairs_path.name} | {len(primary_pairs):,} confirmed AND pairs |',
    f'| {primary_pool_path.name} | {len(primary_pool):,} retained recipes |',
    '| removal_reason_tevc.csv | Recipe-level degree and within/cross-family removal audit |',
    '| near_dup_tevc_report.md | This report |',
    '',
    'These results quantify overlap under the configured ELA and Performance '
    'representations. They do not prove intrinsic mathematical equivalence. '
    'The threshold is a declared, sensitivity-audited operating point rather '
    'than a uniquely correct value.',
]

(OUT_DIR / 'near_dup_tevc_report.md').write_text(
    '\n'.join(report_lines) + '\n',
    encoding='utf-8',
)

print('=== PORTAL TEVC-Aligned Near-Duplicate Detection ===')
print(f'  Recipes: {n}  |  ELA dim: {ela_x.shape[1]}  |  Performance dim: {performance_x.shape[1]}')
print(f'  P95 scales: ELA={p95_ela:.6g}  Performance={p95_performance:.6g}')
print(
    f'  AND pairs (tau={PRIMARY_TAU:g}): {len(primary_pairs):,}  |  '
    f'Pool R: {len(primary_pool):,}  |  Removed: {n-len(primary_pool):,}'
)
print('  Saved: near_dup_tevc_report.md and three diagnostic figures (.png/.pdf)')
