"""
run_ela.py
==========
Compute ELA features for canonical unified PORTAL recipe JSONs.

Feature and sampling implementation: the adapted pflacco subset bundled in
this repository, by R. P. Prager and contributors (MIT; see pflacco/LICENSE).
Source: https://github.com/Reiyan/pflacco
Reference: R. P. Prager and H. Trautmann (2024), Pflacco: Feature-Based
Landscape Analysis of Continuous and Constrained Optimization Problems in
Python, Evolutionary Computation 32(3), 211-216.
https://doi.org/10.1162/evco_a_00341
ELA foundation: O. Mersmann et al. (2011), Exploratory Landscape Analysis,
GECCO, 829-836. https://doi.org/10.1145/2001576.2001690
PORTAL supplies the orchestration, canonical evaluator integration, and cache.

The input root may be flat or nested. Recipe identity, family membership,
dimension, and block count are read from each JSON rather than its path.

Output: ONE CSV with one row per anchor recipe → ela_portal_anchors.csv
    Columns: family, family_id, recipe_index, recipe_id, sample,
    instance_type, d, json_path, ela_failed_groups, ela_* features.
ELA features are kept RAW (no normalization, no feature removal). A group
that fails for a recipe leaves its features blank/NaN and its name is
recorded in the ela_failed_groups column.

Document alignment (Section 2 — Inputs and assumptions):
        - Fixed seed (ELA_SEED = 42) for all recipes.
      "The sampling design must be the same for every realization in the
      family so that feature differences reflect the generated landscapes
      rather than random variation in the ELA sampling design."
        - 1000 × d improved Latin-hypercube points per recipe
          (= 30000 points in the box [-100,100]^d for d=30).

Assumptions (not in document):
    - N_WORKERS = 90% of CPU cores, to allow concurrent daily work.

Usage:
    python run_ela.py
"""

import sys
import argparse
import os
import json
import time
import traceback
import multiprocessing
import threading
import queue as queue_module
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

_ELA_UNIT_SAMPLE = None

# ── Path setup ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # project root (one level up from Recipe_Selection/)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'portal_core'))
from portal_core.instance_format import decode_instance_file
from Recipe_Selection._execution import (
    initialize_numeric_worker,
    single_threaded_numerics,
)

try:
    from .suite_pipeline.discovery import discover_recipes, read_recipe_record
except ImportError:  # Direct execution from Recipe_Selection/.
    from suite_pipeline.discovery import discover_recipes, read_recipe_record

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
CONFIG = {
    # Recipe suite: the full 348-recipe PORTAL suite.
    'INSTANCES_ROOT': str(ROOT / 'Suite'),
    'OUTPUT_CSV':     str(ROOT / 'results' / 'ela_portal_anchors.csv'),
    # Auto-generated metadata note describing the exact ELA settings used.
    'METADATA_MD':    str(ROOT / 'results' / 'ela_metadata.md'),

    # 1000 × d = 30000 LHS points for d=30
    'ELA_SAMPLE_FACTOR': 1000,

    # Fixed seed for ALL realizations — document Section 2 and Section 11
    # "The ELA sampling design is fixed across all realizations in the family."
    'ELA_SEED': 42,

    # Use the torch CPU batch evaluator for ELA sample values by default.
    # Scalar feature callbacks still use the reference scalar wrapper.
    'EVALUATION_DEVICE': 'cpu',

    # ASSUMPTION (not in document): use 90% of CPU cores to allow daily work
    'N_WORKERS': max(1, int(multiprocessing.cpu_count() * 0.9)),

    # Runtime realization mode (default: Type 0 / anchor directly).
    'REALIZATION': {
        'type': 'none',
        'base_seed': 'train',
        'sample_id': 1,
    },

    # Log one aggregated heartbeat at roughly each 0.1% progress step.
    # ELA progress units are: sample generation, batch evaluation, and
    # the ten feature groups for each anchor recipe.
    'HEARTBEAT_PERCENT': 0.1,
    'HEARTBEAT_ALIVE_SECONDS': 30,
    'ELA_PROGRESS_UNITS_PER_ANCHOR': 12,
}
# ══════════════════════════════════════════════════════════════


def _get_instance_type(json_path):
    with open(json_path) as f:
        head = decode_instance_file(json.load(f))
    blocks = head.get('blocks')
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(
            f"Unified PORTAL schema required for ELA: {json_path} has no blocks array."
        )
    if isinstance(blocks, list) and len(blocks) > 1:
        return 'block_composite'
    return 'single'


def _load_instance(json_path, cfg):
    from portal_core.load_instance import load_instance
    from portal_core.realization import (
        realize,
        METHOD_CENTER_BETA,
        METHOD_DIVERSE,
        PORTAL_SEED_TRAIN,
        PORTAL_SEED_TEST,
    )

    def _resolve_base_seed(base_seed):
        if isinstance(base_seed, str):
            key = base_seed.strip().lower()
            if key == 'train':
                return int(PORTAL_SEED_TRAIN)
            if key == 'test':
                return int(PORTAL_SEED_TEST)
            return int(base_seed)
        return int(base_seed)

    def _map_type_to_method(rtype):
        mode = str(rtype).strip().lower()
        if mode == 'none':
            return None
        if mode == 'type1':
            return METHOD_CENTER_BETA
        if mode == 'type2':
            return METHOD_DIVERSE
        raise ValueError(f"Unsupported realization type: {rtype!r}")

    itype = _get_instance_type(json_path)
    with contextlib.redirect_stdout(io.StringIO()):
        instance = load_instance(json_path)

    realization_cfg = cfg.get('REALIZATION', {}) if isinstance(cfg, dict) else {}
    method = _map_type_to_method(realization_cfg.get('type', 'none'))
    if method is not None:
        base_seed = _resolve_base_seed(realization_cfg.get('base_seed', 'train'))
        sample_id = int(realization_cfg.get('sample_id', 1))
        instance = realize(instance, base_seed, sample_id, method=method)

    return instance, itype


def _evaluate(x, instance, itype):
    from portal_core.fitness import fitness
    return float(np.squeeze(fitness(x, instance)))


def _evaluate_batch(x, instance, itype, evaluation_device='numpy'):
    from portal_core.fitness import fitness_batch
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    values = fitness_batch(
        arr,
        instance,
        evaluation_device=evaluation_device,
    )
    return np.asarray(values, dtype=float).reshape(-1)


def _sample_cache_path(cfg, d, n_samples, seed):
    cache_dir = Path(cfg['OUTPUT_CSV']).parent / 'cache'
    return cache_dir / f"ela_lhs_unit_d{int(d)}_n{int(n_samples)}_seed{int(seed)}.npy"


def _dispersion_stats_cache_path(cfg, d, n_samples, seed, lb, ub):
    cache_dir = Path(cfg['OUTPUT_CSV']).parent / 'cache'
    return (
        cache_dir /
        f"ela_dispersion_full_stats_v2_d{int(d)}_n{int(n_samples)}_"
        f"seed{int(seed)}_bounds{float(lb):.17g}_{float(ub):.17g}.npz"
    )


def _build_or_load_unit_sample(cfg, d, n_samples, seed):
    """Create the fixed ELA LHS design once, then reuse it for all recipes."""
    from pflacco.sampling import create_initial_sample

    path = _sample_cache_path(cfg, d, n_samples, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        t0 = time.time()
        sample = np.load(path)
        print(
            f"Loaded cached ELA LHS sample: {path} "
            f"shape={sample.shape} in {_format_duration(time.time() - t0)}",
            flush=True,
        )
    else:
        print(
            "Building fixed improved-LHS ELA sample once "
            f"(d={d}, n={n_samples}, seed={seed}). "
            "This replaces generating the identical sample separately for "
            "each recipe.",
            flush=True,
        )
        t0 = time.time()
        df = create_initial_sample(
            d,
            n_samples,
            lower_bound=0.0,
            upper_bound=1.0,
            seed=seed,
        )
        sample = df.to_numpy(dtype=float)
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        with open(tmp_path, 'wb') as f:
            np.save(f, sample)
        os.replace(tmp_path, path)
        print(
            f"Cached fixed ELA LHS sample: {path} "
            f"shape={sample.shape} in {_format_duration(time.time() - t0)}",
            flush=True,
        )

    expected_shape = (int(n_samples), int(d))
    if sample.shape != expected_shape:
        raise ValueError(
            f"Cached ELA sample has shape {sample.shape}, expected {expected_shape}: {path}"
        )
    return np.asarray(sample, dtype=float)


def _build_or_load_dispersion_full_stats(cfg, unit_sample, d, n_samples, seed, lb, ub):
    from scipy.spatial.distance import pdist

    path = _dispersion_stats_cache_path(cfg, d, n_samples, seed, lb, ub)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        t0 = time.time()
        with np.load(path) as payload:
            stats = {
                'mean': float(payload['mean']),
                'median': float(payload['median']),
            }
        print(
            f"Loaded cached ELA dispersion full-distance stats: {path} "
            f"in {_format_duration(time.time() - t0)}",
            flush=True,
        )
        return stats

    print(
        "Building ELA dispersion full-distance stats once "
        f"(d={d}, n={n_samples}, bounds=[{lb}, {ub}]). "
        "This replaces recomputing full-sample pairwise distances for "
        "each recipe.",
        flush=True,
    )
    t0 = time.time()
    X = unit_sample * (float(ub) - float(lb)) + float(lb)
    distances = pdist(X, metric='euclidean')
    distances = distances[distances != 0]
    stats = {
        'mean': float(np.mean(distances)),
        'median': float(np.median(distances)),
    }
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'wb') as f:
        np.savez(f, **stats)
    os.replace(tmp_path, path)
    print(
        f"Cached ELA dispersion full-distance stats: {path} "
        f"mean={stats['mean']:.6g} median={stats['median']:.6g} "
        f"in {_format_duration(time.time() - t0)}",
        flush=True,
    )
    return stats


def _recipe_bounds(json_path):
    """Read the scalar search box used by this recipe's ELA evaluations."""
    with open(json_path, encoding='utf-8-sig') as handle:
        raw = decode_instance_file(json.load(handle))
    lower = float(raw['bounds_min'])
    upper = float(raw['bounds_max'])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"Invalid finite search bounds in {json_path}: {lower}, {upper}")
    return lower, upper


def _build_dispersion_stats_by_bounds(cfg, json_paths, unit_sample, d, n_samples, seed):
    """Reuse reference distances only among recipes with the same search box."""
    domains = sorted({_recipe_bounds(path) for path in json_paths})
    print(f"ELA dispersion reference domains: {len(domains)}", flush=True)
    return {
        bounds: _build_or_load_dispersion_full_stats(
            cfg, unit_sample, d, n_samples, seed, *bounds,
        )
        for bounds in domains
    }


def _dispersion_stats_for_recipe(cfg, lower, upper):
    by_bounds = cfg.get('ELA_DISPERSION_STATS_BY_BOUNDS')
    if by_bounds is None:
        # Direct worker callers without a cache calculate the exact distances
        # from their own sample rather than reuse an unqualified reference.
        return None
    bounds = (float(lower), float(upper))
    if bounds not in by_bounds:
        raise ValueError(f"No ELA dispersion reference statistics for bounds {bounds}.")
    return by_bounds[bounds]


def _get_unit_sample_from_cache(path, d, n_samples):
    global _ELA_UNIT_SAMPLE
    if _ELA_UNIT_SAMPLE is None:
        if path:
            _ELA_UNIT_SAMPLE = np.load(path)
        else:
            return None
    expected_shape = (int(n_samples), int(d))
    if _ELA_UNIT_SAMPLE.shape != expected_shape:
        raise ValueError(
            f"ELA unit sample has shape {_ELA_UNIT_SAMPLE.shape}, expected {expected_shape}"
        )
    return _ELA_UNIT_SAMPLE


def _emit_ela_progress(progress_queue, event):
    if progress_queue is None:
        return
    try:
        progress_queue.put(event)
    except Exception:
        pass


def _compute_ela(instance, itype, d, lb, ub, n_samples, seed,
                 evaluation_device='numpy', progress_label=None,
                 progress_queue=None, sample_cache_path=None,
                 dispersion_full_stats=None):
    """
    Compute ELA features using pflacco.

    Sampling design: fixed seed, 1000×d LHS points in [lb, ub]^d.
    Same design for every realization — document Section 2 requirement.

    Feature sets computed:
      ela_meta, ela_distribution, ela_level, pca, nbc, dispersion,
      information_content, fitness_distance_correlation,
      ela_conv, gradient_features

    Preserve pflacco group prefixes (e.g. 'ela_meta.lin_simple.adj_r2').
    Prepend 'ela_' only when absent to avoid duplicate prefixes.
    """
    import pflacco.classical_ela_features as clf
    import pflacco.misc_features as mf
    import warnings

    label = progress_label or 'unknown'
    unit_sample = _get_unit_sample_from_cache(sample_cache_path, d, n_samples)
    if unit_sample is None:
        from pflacco.sampling import create_initial_sample
        X = create_initial_sample(d, n_samples,
                                  lower_bound=lb, upper_bound=ub, seed=seed)
    else:
        lower = np.asarray(lb, dtype=float)
        upper = np.asarray(ub, dtype=float)
        X_arr = unit_sample * (upper - lower) + lower
        X = pd.DataFrame(X_arr, columns=[f'x{i}' for i in range(d)])
    _emit_ela_progress(
        progress_queue,
        {'kind': 'unit', 'label': label, 'step': 'sample'},
    )
    y = _evaluate_batch(
        X.to_numpy(dtype=float),
        instance,
        itype,
        evaluation_device=evaluation_device,
    )
    _emit_ela_progress(
        progress_queue,
        {'kind': 'unit', 'label': label, 'step': 'batch_eval'},
    )
    f = lambda x: _evaluate(np.asarray(x, dtype=float), instance, itype)

    feature_groups = [
        ('ela_meta', lambda: clf.calculate_ela_meta(X, y)),
        ('ela_distribution', lambda: clf.calculate_ela_distribution(X, y)),
        ('ela_level', lambda: clf.calculate_ela_level(X, y)),
        ('pca', lambda: clf.calculate_pca(X, y)),
        ('nbc', lambda: clf.calculate_nbc(X, y)),
        ('dispersion', lambda: clf.calculate_dispersion(
            X,
            y,
            dist_full_stats=dispersion_full_stats,
        )),
        ('information_content', lambda: clf.calculate_information_content(X, y, seed=seed)),
        ('fitness_distance', lambda: mf.calculate_fitness_distance_correlation(X, y)),
        ('ela_conv', lambda: clf.calculate_ela_conv(X, y, f, seed=seed)),
        ('gradient', lambda: mf.calculate_gradient_features(f, d, lb, ub, seed=seed)),
    ]

    features = {}
    failures = []
    total_groups = len(feature_groups)
    for group_idx, (name, fn) in enumerate(feature_groups, start=1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                features.update(fn())
            _emit_ela_progress(
                progress_queue,
                {
                    'kind': 'unit',
                    'label': label,
                    'step': name,
                    'group_index': group_idx,
                    'group_total': total_groups,
                },
            )
        except Exception:
            # Handoff requirement: keep failures explicit rather than
            # aborting the recipe. The failed group's features are simply
            # absent (-> NaN in the table); the group name is recorded.
            failures.append(name)
            _emit_ela_progress(
                progress_queue,
                {
                    'kind': 'unit',
                    'label': label,
                    'step': name,
                    'status': 'failed',
                    'group_index': group_idx,
                    'group_total': total_groups,
                },
            )

    # Only prepend 'ela_' if not already present
    feats = {(k if k.startswith('ela_') else f'ela_{k}'): v
             for k, v in features.items()}
    return feats, failures


def process_sample(args):
    """Worker: compute ELA for one anchor recipe. Returns one CSV row."""
    json_path, cfg = args
    progress_queue = cfg.get('_PROGRESS_QUEUE') if isinstance(cfg, dict) else None

    script_dir = Path(__file__).parent.parent  # project root
    for p in [str(script_dir), str(script_dir / 'portal_core')]:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        # Use the worker-local configuration carried in the task tuple.  On
        # Windows, multiprocessing uses spawn, so reading the module-level
        # CONFIG here would silently restore the default Type-0 realization.
        instance, itype = _load_instance(json_path, cfg)
        with open(json_path) as f:
            raw = decode_instance_file(json.load(f))
        suite = raw.get('_suite', {})
        record = read_recipe_record(json_path)
        d     = instance['d']
        lb    = float(instance['bounds_min'])
        ub    = float(instance['bounds_max'])
        family_id = int(record.family_id or 0)
        recipe_index = int(record.recipe_index or 0)
        recipe_id = record.recipe_id
        family = record.family

        n_samples = cfg['ELA_SAMPLE_FACTOR'] * d
        ela, failed_groups = _compute_ela(
            instance,
            itype,
            d,
            lb,
            ub,
            n_samples,
            cfg['ELA_SEED'],
            evaluation_device=cfg.get('EVALUATION_DEVICE', 'numpy'),
            progress_label=f"{family}/{recipe_id}",
            progress_queue=progress_queue,
            sample_cache_path=cfg.get('ELA_SAMPLE_CACHE_PATH'),
            dispersion_full_stats=_dispersion_stats_for_recipe(cfg, lb, ub),
        )

        sample = int(suite.get('sample', 1) or 1)

        row = {
            'json_path':         json_path,
            'family':            family,
            'family_id':         family_id,
            'recipe_index':      recipe_index,
            'recipe_id':         recipe_id,
            'sample':            sample,
            'instance_type':     itype,
            'd':                 d,
            'ela_failed_groups': ';'.join(failed_groups),
            **ela,
            'error':             None,
        }
        _emit_ela_progress(
            progress_queue,
            {'kind': 'anchor', 'status': 'ok', 'label': f"{family}/{recipe_id}"},
        )
        return row

    except Exception as e:
        print(f"  FAILED {json_path}: {e}", flush=True)
        traceback.print_exc()
        _emit_ela_progress(
            progress_queue,
            {'kind': 'anchor', 'status': 'error', 'label': str(json_path)},
        )
        return {'json_path': json_path, 'error': str(e)}


def collect_all_samples(instances_root):
    """Collect unified recipe JSONs using payload metadata, not folder names."""

    records = discover_recipes(instances_root, require_single_dimension=True)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.family] = counts.get(record.family, 0) + 1
    print(f"Found {len(counts)} families from unified JSON metadata:")
    for family in sorted(counts):
        print(f"  {family}: {counts[family]} recipes")
    return [str(record.path) for record in records], sorted(counts)


def _read_recipe_ids(path):
    """Read recipe identifiers from a one-column text or CSV file."""
    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Recipe-id file not found: {source}")
    values = []
    for line in source.read_text(encoding='utf-8-sig').splitlines():
        token = line.strip().split(',')[0].strip()
        if token and token.lower() not in {'recipe_id', 'recipe'}:
            values.append(Path(token).stem)
    if not values:
        raise ValueError(f"Recipe-id file is empty: {source}")
    if len(values) != len(set(values)):
        raise ValueError(f"Recipe-id file contains duplicates: {source}")
    return set(values)


def _filter_recipe_files(all_files, recipe_ids):
    if recipe_ids is None:
        return all_files
    records = [(path, read_recipe_record(path)) for path in all_files]
    selected = [path for path, record in records if record.recipe_id in recipe_ids]
    found = {record.recipe_id for _, record in records if record.recipe_id in recipe_ids}
    missing = sorted(recipe_ids - found)
    if missing:
        raise ValueError(f"Requested recipe IDs were not found: {missing}")
    return selected


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run PORTAL UF ELA extraction.')
    parser.add_argument(
        '--max-workers',
        type=_positive_int,
        default=None,
        help=(
            "Maximum number of realization-level ELA workers. "
            "Defaults to 90%% of available CPU cores."
        ),
    )
    parser.add_argument(
        '--evaluation-device',
        choices=('auto', 'numpy', 'cpu', 'cuda', 'mps'),
        default=CONFIG['EVALUATION_DEVICE'],
        help=(
            "PORTAL fitness backend for the initial ELA sample values. "
            "Use 'numpy' for deterministic NumPy vectorization, 'cuda' or "
            "'mps' for GPU-backed torch where available, or 'auto' to try "
            "torch devices before NumPy."
        ),
    )
    parser.add_argument(
        '--realization-type',
        choices=('none', 'type1', 'type2'),
        default=CONFIG['REALIZATION']['type'],
        help="Runtime realization mode for ELA extraction (default: none).",
    )
    parser.add_argument(
        '--realization-base-seed',
        default=str(CONFIG['REALIZATION']['base_seed']),
        help="Runtime realization base seed: train, test, or integer.",
    )
    parser.add_argument(
        '--realization-sample-id',
        type=_positive_int,
        default=int(CONFIG['REALIZATION']['sample_id']),
        help="Runtime realization sample id (used when type1/type2).",
    )
    parser.add_argument(
        '--instances-root',
        type=Path,
        default=Path(CONFIG['INSTANCES_ROOT']),
        help=(
            "Root recursively containing canonical unified recipe JSON files. "
            "Family and dimension are read from each JSON, not its path."
        ),
    )
    parser.add_argument(
        '--recipe-ids-file',
        type=Path,
        default=None,
        help="Optional one-column text/CSV file selecting recipe IDs.",
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=Path(CONFIG['OUTPUT_CSV']),
        help="ELA output CSV path.",
    )
    parser.add_argument(
        '--metadata-md',
        type=Path,
        default=Path(CONFIG['METADATA_MD']),
        help="ELA metadata output path.",
    )
    parser.add_argument(
        '--sample-factor',
        type=_positive_int,
        default=int(CONFIG['ELA_SAMPLE_FACTOR']),
        help="Number of ELA sample points per dimension.",
    )
    parser.add_argument(
        '--ela-seed',
        type=int,
        default=int(CONFIG['ELA_SEED']),
        help="Seed for the common improved-LHS ELA design.",
    )
    parser.add_argument(
        '--heartbeat-percent',
        type=_heartbeat_percent,
        default=float(CONFIG['HEARTBEAT_PERCENT']),
        help="Percentage interval between progress heartbeat messages.",
    )
    parser.add_argument(
        '--heartbeat-alive-seconds',
        type=_nonnegative_float,
        default=float(CONFIG['HEARTBEAT_ALIVE_SECONDS']),
        help="Emit an alive message after this many quiet seconds; zero disables it.",
    )
    return parser.parse_args(argv)


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _nonnegative_float(value):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _heartbeat_percent(value):
    value = _positive_float(value)
    if value > 100.0:
        raise argparse.ArgumentTypeError("must be <= 100")
    return value


def _resolve_max_workers(value, cpu_count=None, default_workers=None):
    n_cpu = multiprocessing.cpu_count() if cpu_count is None else int(cpu_count)
    if value is None:
        if default_workers is None:
            value = max(1, int(n_cpu * 0.9))
        else:
            value = int(default_workers)
    return max(1, min(int(value), n_cpu))


def _format_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _ela_completion_message(row, completed, total, start_time):
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else 0.0
    if row.get('error') is None:
        recipe = row.get('recipe_id') or Path(row.get('json_path', '?')).stem
        family = row.get('family', '?')
        failed = row.get('ela_failed_groups') or ''
        suffix = f" failed_groups={failed}" if failed else ''
        status = f"ELA {family}/{recipe} completed{suffix}"
    else:
        status = f"ELA {row.get('json_path', '?')} FAILED: {row.get('error', '?')}"
    return (
        f"[{int(completed)}/{int(total)}] {status} | "
        f"elapsed {_format_duration(elapsed)} | ETA {_format_duration(eta)}"
    )


def _ela_progress_listener(
    progress_queue,
    total_units,
    total_anchors,
    start_time,
    stop_event,
    heartbeat_percent,
    alive_seconds,
):
    total_units = max(1, int(total_units))
    total_anchors = max(1, int(total_anchors))
    heartbeat_percent = max(0.01, float(heartbeat_percent))
    completed_units = 0
    completed_anchors = 0
    failed_anchors = 0
    next_percent = heartbeat_percent
    last_status_time = time.time()

    print(
        "ELA heartbeat 0.0% | units 0/"
        f"{total_units:,} | recipes 0/{total_anchors:,}",
        flush=True,
    )

    while True:
        try:
            if stop_event.is_set():
                event = progress_queue.get_nowait()
            else:
                event = progress_queue.get(timeout=1.0)
        except queue_module.Empty:
            now = time.time()
            if (
                not stop_event.is_set()
                and alive_seconds > 0
                and now - last_status_time >= alive_seconds
            ):
                elapsed = now - start_time
                percent = 100.0 * completed_units / total_units
                rate = completed_units / elapsed if elapsed > 0 else 0.0
                eta = (total_units - completed_units) / rate if rate > 0 else 0.0
                print(
                    f"ELA heartbeat alive {percent:.1f}% | "
                    f"units {completed_units:,}/{total_units:,} | "
                    f"recipes {completed_anchors:,}/{total_anchors:,} | "
                    f"elapsed {_format_duration(elapsed)} | "
                    f"ETA {_format_duration(eta)}",
                    flush=True,
                )
                last_status_time = now
            if stop_event.is_set():
                break
            continue
        except Exception:
            if stop_event.is_set():
                break
            continue

        kind = event.get('kind')
        if kind == 'unit':
            completed_units += int(event.get('amount', 1))
        elif kind == 'anchor':
            completed_anchors += 1
            if event.get('status') == 'error':
                failed_anchors += 1
        else:
            continue

        percent = 100.0 * completed_units / total_units
        if completed_units >= total_units:
            percent = 100.0

        while percent + 1e-12 >= next_percent:
            elapsed = time.time() - start_time
            rate = completed_units / elapsed if elapsed > 0 else 0.0
            eta = (total_units - completed_units) / rate if rate > 0 else 0.0
            failure_part = f" | failed recipes {failed_anchors}" if failed_anchors else ""
            print(
                f"ELA heartbeat {next_percent:.1f}% | "
                f"units {completed_units:,}/{total_units:,} | "
                f"recipes {completed_anchors:,}/{total_anchors:,} | "
                f"elapsed {_format_duration(elapsed)} | "
                f"ETA {_format_duration(eta)}"
                f"{failure_part}",
                flush=True,
            )
            last_status_time = time.time()
            next_percent = round(next_percent + heartbeat_percent, 10)
            if next_percent > 100.0:
                break


def _write_metadata(cfg, n_anchors, n_features, failed_summary):
    """Write the ELA measurement settings and feature-failure summary."""
    text = f"""# ELA Feature Extraction — Metadata Note

Auto-generated by `run_ela.py`.

## Scope
- Recipe suite root: `{cfg['INSTANCES_ROOT']}`
- Recipes processed: {n_anchors}
- ELA feature columns produced: {n_features}

## Realization mode
- `REALIZATION = {{'type': '{cfg['REALIZATION']['type']}', 'base_seed': '{cfg['REALIZATION']['base_seed']}'}}`
- Type 0 (`none`) uses the anchor directly. Type 1 and Type 2 apply the
  configured deterministic realization before feature extraction.
- Realization sample identifier: {cfg['REALIZATION']['sample_id']}

## Sampling design (common unit design, mapped to each recipe's bounds)
- Sampler: improved Latin-hypercube sampling
  (`pflacco.sampling.create_initial_sample`).
- Sample size: {cfg['ELA_SAMPLE_FACTOR']} x d points
  (= {cfg['ELA_SAMPLE_FACTOR'] * 30} points for d=30).
- Domain / bounds: each recipe's declared [bounds_min, bounds_max]^d box.
- Dispersion reference distances are cached separately for each distinct box.
- Random seed policy: a single fixed seed (ELA_SEED = {cfg['ELA_SEED']})
  used for every recipe, so feature differences reflect the landscapes
  and not ELA sampling noise.
- Repetitions: 1 (one ELA sample per recipe).
- Evaluation backend: {cfg['EVALUATION_DEVICE']}

## Feature groups computed
- ela_meta, ela_distribution, ela_level, pca, nbc, dispersion,
  information_content, fitness_distance_correlation, ela_conv,
  gradient_features.
- Feature groups omitted: none — all ten groups are attempted for
  every recipe under the identical protocol.

## Missing / failed features
- Features are NOT normalized, standardized, or removed.
- If a feature group fails for a recipe, its features are left missing
  (blank / NaN) and the failed group names are recorded in the
  `ela_failed_groups` column of the output table.
- Failed-group summary: {failed_summary}

## Output
- ELA table: `{cfg['OUTPUT_CSV']}`
- One row per recipe; one column per ELA feature, plus
  `ela_failed_groups`.
"""
    path = Path(cfg['METADATA_MD'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def main(argv=None):
    args = _parse_args(argv)
    CONFIG['INSTANCES_ROOT'] = str(args.instances_root)
    CONFIG['OUTPUT_CSV'] = str(args.output_csv)
    CONFIG['METADATA_MD'] = str(args.metadata_md)
    CONFIG['EVALUATION_DEVICE'] = args.evaluation_device
    CONFIG['ELA_SAMPLE_FACTOR'] = int(args.sample_factor)
    CONFIG['ELA_SEED'] = int(args.ela_seed)
    CONFIG['HEARTBEAT_PERCENT'] = float(args.heartbeat_percent)
    CONFIG['HEARTBEAT_ALIVE_SECONDS'] = float(args.heartbeat_alive_seconds)
    CONFIG['REALIZATION']['type'] = args.realization_type
    CONFIG['REALIZATION']['base_seed'] = args.realization_base_seed
    CONFIG['REALIZATION']['sample_id'] = int(args.realization_sample_id)

    print("=" * 60)
    print("ELA Pipeline - PORTAL Recipes")
    print("=" * 60)

    all_files, recipes = collect_all_samples(CONFIG['INSTANCES_ROOT'])
    recipe_ids = _read_recipe_ids(args.recipe_ids_file)
    all_files = _filter_recipe_files(all_files, recipe_ids)
    print(f"\nTotal: {len(all_files)} recipes across {len(recipes)} families")

    n_cpu     = multiprocessing.cpu_count()
    n_workers = _resolve_max_workers(
        args.max_workers,
        n_cpu,
        default_workers=CONFIG['N_WORKERS'],
    )
    print(f"Using {n_workers} worker(s) on {n_cpu} available core(s)")
    print(f"ELA sampling: {CONFIG['ELA_SAMPLE_FACTOR']} x d points, seed={CONFIG['ELA_SEED']} (fixed for all recipes)")
    print(
        "Realization mode: "
        f"type={CONFIG['REALIZATION']['type']}, "
        f"base_seed={CONFIG['REALIZATION']['base_seed']}, "
        f"sample_id={CONFIG['REALIZATION']['sample_id']}"
    )
    print(f"Evaluation backend: {CONFIG['EVALUATION_DEVICE']}")
    print(f"Output: {CONFIG['OUTPUT_CSV']}\n")

    first_d = None
    if all_files:
        with open(all_files[0]) as f:
            first_raw = decode_instance_file(json.load(f))
            first_d = int(first_raw.get('d', 30))
    else:
        first_d = 30
    n_samples = CONFIG['ELA_SAMPLE_FACTOR'] * first_d
    global _ELA_UNIT_SAMPLE
    _ELA_UNIT_SAMPLE = _build_or_load_unit_sample(
        CONFIG,
        first_d,
        n_samples,
        CONFIG['ELA_SEED'],
    )
    dispersion_full_stats = _build_dispersion_stats_by_bounds(
        CONFIG,
        all_files,
        _ELA_UNIT_SAMPLE,
        first_d,
        n_samples,
        CONFIG['ELA_SEED'],
    )

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    run_cfg = dict(CONFIG)
    run_cfg['_PROGRESS_QUEUE'] = progress_queue
    run_cfg['ELA_SAMPLE_CACHE_PATH'] = str(
        _sample_cache_path(CONFIG, first_d, n_samples, CONFIG['ELA_SEED'])
    )
    run_cfg['ELA_DISPERSION_STATS_BY_BOUNDS'] = dispersion_full_stats
    args    = [(f, run_cfg) for f in all_files]
    t_start = time.time()
    stop_event = threading.Event()
    total_progress_units = (
        len(args) * int(CONFIG.get('ELA_PROGRESS_UNITS_PER_ANCHOR', 12))
    )
    progress_thread = threading.Thread(
        target=_ela_progress_listener,
        args=(
            progress_queue,
            total_progress_units,
            len(args),
            t_start,
            stop_event,
            CONFIG.get('HEARTBEAT_PERCENT', 0.1),
            CONFIG.get('HEARTBEAT_ALIVE_SECONDS', 30),
        ),
        daemon=True,
    )
    progress_thread.start()

    results = []
    try:
        with single_threaded_numerics():
            if n_workers == 1:
                for row in map(process_sample, args):
                    results.append(row)
            else:
                with multiprocessing.Pool(
                    processes=n_workers, initializer=initialize_numeric_worker,
                ) as pool:
                    for row in pool.imap_unordered(process_sample, args):
                        results.append(row)
    finally:
        stop_event.set()
        progress_thread.join(timeout=30)
        manager.shutdown()

    elapsed = time.time() - t_start

    good = [r for r in results if r.get('error') is None]
    bad  = [r for r in results if r.get('error') is not None]

    if bad:
        examples = ', '.join(f"{r.get('json_path', '?')}: {r.get('error', '?')}" for r in bad[:5])
        raise ValueError(f"ELA pipeline has failed recipes ({len(bad)}): {examples}")
    if len(good) != len(all_files):
        raise ValueError(
            f"ELA pipeline incomplete: expected {len(all_files)} rows, got {len(good)}."
        )

    df = pd.DataFrame(good)
    df = df.drop(columns=['error'], errors='ignore')

    meta_cols = [
        'family', 'family_id', 'recipe_index', 'recipe_id',
        'sample', 'instance_type', 'd', 'json_path', 'ela_failed_groups'
    ]
    ela_cols  = sorted([c for c in df.columns
                        if c.startswith('ela_') and c != 'ela_failed_groups'])
    ordered   = [c for c in meta_cols if c in df.columns] + ela_cols
    df = df[ordered]

    df = df.sort_values(['family_id', 'recipe_index', 'sample'])
    df = df.reset_index(drop=True)

    out_path = Path(CONFIG['OUTPUT_CSV'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    if 'ela_failed_groups' in df.columns:
        n_with_failures = int((df['ela_failed_groups'].fillna('') != '').sum())
    else:
        n_with_failures = 0
    failed_summary = (
        f"{n_with_failures} of {df.shape[0]} recipes had at least one "
        "failed feature group"
        if n_with_failures
        else "no feature-group failures"
    )
    meta_path = _write_metadata(CONFIG, df.shape[0], len(ela_cols), failed_summary)

    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} minutes")
    print(f"Recipes processed:      {len(good)}")
    if bad:
        print(f"Failed:                 {len(bad)}")
        for r in bad[:5]:
            print(f"  {r['json_path']}: {r['error']}")
    print(f"ELA features:           {len(ela_cols)}")
    print(f"Recipes w/ failed group:{n_with_failures}")
    print(f"CSV saved to:           {out_path}")
    print(f"Metadata note saved to: {meta_path}")
    print(f"CSV shape:              {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Expected:               {len(all_files)} rows")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main(sys.argv[1:])
