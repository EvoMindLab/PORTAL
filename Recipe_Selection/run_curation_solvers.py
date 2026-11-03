"""
run_curation_solvers.py
=======================
Performance raw data-collection stage for the PORTAL recipe-level study.

This script runs the agreed 7-algorithm solver portfolio on every anchor
recipe across Family1..Family4 and writes a RAW, run-level Performance table.

The default measurement protocol, configurable through the CLI, is:

- Default curation mode: Type 0 (REALIZATION['type'] = 'none'). The CLI also
  supports Type 1 and Type 2 for repeated-run realization studies; each run
  then receives its own deterministic realization sample identifier.
- 7 algorithms, all treated as stochastic, 11 independent runs each.
- Deterministic run seeds:  seed = BASE + 100 * algorithm_id + run_id,
  with algorithm_id in 1..7 and run_id in 1..11.
- For every run, the best-so-far value is captured at five fixed function
  evaluation (FE) checkpoints: 30k, 50k, 100k, 200k, 300k.
- The recorded quantity is the RAW absolute error  |f_best - f_star|.
  No log-transform, no normalization, no scaling, no clipping.
- No aggregation: one output row per (recipe, algorithm, run, checkpoint).

Output: results/performance_raw_portal_anchors.csv
    Columns: recipe_id, family, family_id, recipe_index, algorithm,
             algorithm_id, run_id, seed, checkpoint_fe, absolute_error,
             f_star, max_fe, realization_type, realization_base_seed

f_star is computed from the unified schema as:
    sum_j block_weight_j * min_k(beta_{j,k})  over block components.

Usage:
    python run_curation_solvers.py
    python run_curation_solvers.py --max-parallel-runs 12
"""

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

import sys
import os
import json
import argparse
import time
import traceback
import importlib
import multiprocessing
import contextlib
import io
import random
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # project root (one level up from Recipe_Selection/)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'algorithms'))
sys.path.insert(0, str(ROOT / 'portal_core'))
from portal_core.instance_format import decode_instance_file
from Recipe_Selection._execution import (
    cap_process_workers,
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

    # RAW run-level deliverable (one row per recipe/algorithm/run/checkpoint).
    'OUTPUT_CSV':     str(ROOT / 'results' / 'performance_raw_portal_anchors.csv'),
    # Resumable internal checkpoint (one wide row per recipe/algorithm/run).
    'CHECKPOINT_CSV': str(ROOT / 'results' / 'performance_raw_portal_anchors.partial.csv'),
    # Auto-generated metadata note describing the exact Performance settings used.
    'METADATA_MD':    str(ROOT / 'results' / 'performance_metadata.md'),

    # Evaluation budget: 10^4 x d  ->  300000 FE for d=30.
    'MAX_FE_PER_DIM': 10000,

    # Handoff: "run 11 independent runs" for each algorithm.
    'N_RUNS': 11,

    # Base value in the run seeding rule:
    #   seed = BASE_RUN_SEED + 100 * algorithm_id + run_id
    'BASE_RUN_SEED': 10000,

    # FE checkpoints at which the raw absolute error is recorded.
    'CHECKPOINTS_FE': [30000, 50000, 100000, 200000, 300000],

    # Resumable execution controls.
    'RESUME': True,
    'CHECKPOINT_EVERY': 25,
    'VERBOSE_LOADS': False,
    'ALGORITHM_PROTOCOL': 'batch_sync_v2',
    'HEARTBEAT_PERCENT': 0.1,
    'HEARTBEAT_ALIVE_SECONDS': 30.0,

    # Solver portfolio — the agreed 7 algorithms. Every algorithm in this
    # portfolio uses randomness (Powell and PatternSearch use random restarts
    # / random probes), so all 7 are treated as stochastic and seeded.
    'SOLVERS': [
        {'file': 'CMA_ES',        'family': 'cmaes',   'algorithm_id': 1},
        {'file': 'DE',            'family': 'de',      'algorithm_id': 2},
        {'file': 'LSHADE',        'family': 'de',      'algorithm_id': 3},
        {'file': 'PSO',           'family': 'pso',     'algorithm_id': 4},
        {'file': 'CLPSO',         'family': 'pso',     'algorithm_id': 5},
        {'file': 'PatternSearch', 'family': 'pattern', 'algorithm_id': 6},
        {'file': 'Powell',        'family': 'powell',  'algorithm_id': 7},
    ],

    # Realization mode — Type 0 only for the raw data-collection stage.
    'REALIZATION': {
        'type': 'none',
        'base_seed': 'train',
    },
}
# ══════════════════════════════════════════════════════════════


def _ae_cols(cfg):
    """Per-checkpoint absolute-error column names (wide checkpoint format)."""
    return [f'ae_{int(c)}' for c in cfg['CHECKPOINTS_FE']]


# Identity / validation columns stored in the resumable checkpoint file.
_RUN_COLS_BASE = [
    'family', 'family_id', 'recipe', 'recipe_index', 'sample',
    'solver', 'solver_family', 'algorithm_id',
    'replicate_id', 'run_number', 'seed',
    'f_star', 'f_best_final', 'fe_used',
]
_RUN_COLS_TAIL = [
    'runtime_seconds', 'n_runs', 'max_fe',
    'evaluation_device', 'algorithm_protocol',
    'realization_type', 'realization_base_seed',
]


def _run_cols(cfg):
    """Full ordered column list for the wide per-run checkpoint file."""
    return _RUN_COLS_BASE + _ae_cols(cfg) + _RUN_COLS_TAIL


def _derive_seed(base_seed, algorithm_id, run_number):
    """Deterministic run seed: BASE + 100 * algorithm_id + run_id."""
    return int(base_seed) + 100 * int(algorithm_id) + int(run_number)


def _load_algorithm(name, script_dir):
    """Load the algorithm class from the algorithms/ folder."""
    algorithms_dir = str(script_dir / 'algorithms')
    if algorithms_dir not in sys.path:
        sys.path.insert(0, algorithms_dir)
    module = importlib.import_module(name)
    from base_algorithm import BaseAlgorithm
    for attr in dir(module):
        cls = getattr(module, attr)
        if (isinstance(cls, type)
                and issubclass(cls, BaseAlgorithm)
                and cls is not BaseAlgorithm):
            return cls
    raise ValueError(f"No BaseAlgorithm subclass found in '{name}.py'.")


def _get_f_star(json_path):
    """
    Extract f_star (known optimum value) from the instance JSON.
    Unified schema: objective is the weighted sum of block objectives,
    where each block objective is the min over component baselines (beta).
    """
    with open(json_path) as f:
        raw = decode_instance_file(json.load(f))

    blocks = raw.get('blocks')
    if isinstance(blocks, list) and blocks:
        f_star = 0.0
        for block in blocks:
            comps = block.get('components', [])
            if not comps:
                raise KeyError(f"Missing components in block for {json_path}")
            min_beta = min(float(c.get('beta', 0.0)) for c in comps)
            weight = float(block.get('weight', 1.0))
            f_star += weight * min_beta
        return float(f_star)

    raise KeyError(
        "Unified schema required: expected 'blocks[].components[].beta' "
        f"in {json_path}"
    )


def _load_problem(
    json_path,
    max_fe,
    record_convergence=True,
    verbose_loads=True,
    evaluation_device=None,
    realization_type='none',
    realization_base_seed='train',
    realization_sample_id=1,
):
    """Load a PortalProblem, optionally suppressing verbose instance output."""
    from portal_problem import PortalProblem
    kwargs = {
        'max_fe': max_fe,
        # record_convergence=True is REQUIRED: it builds the full
        # (fe, best_f) curve from which the FE checkpoints are read.
        'record_convergence': record_convergence,
        'evaluation_device': evaluation_device,
        'realization_type': realization_type,
        'realization_base_seed': realization_base_seed,
        'realization_sample_id': realization_sample_id,
    }
    if verbose_loads:
        return PortalProblem(json_path, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return PortalProblem(json_path, **kwargs)


def _checkpoint_errors(convergence, fe_used, f_star, checkpoints):
    """
    Read the raw absolute error |f_best - f_star| at each FE checkpoint.

    ``convergence`` is the PortalProblem (fe, best_f) curve: one entry per
    function evaluation, with best_f the monotone best-so-far value. The
    value at checkpoint c is the best-so-far after min(c, fe_used) FEs.
    If a run ends before a checkpoint, the final best-so-far is carried.
    """
    out = {}
    if not convergence:
        return {int(c): float('nan') for c in checkpoints}
    n = len(convergence)
    fe_used = int(fe_used) if fe_used else n
    for c in checkpoints:
        c = int(c)
        idx = min(c, fe_used) - 1
        idx = max(0, min(idx, n - 1))
        best_f = float(convergence[idx][1])
        out[c] = abs(best_f - float(f_star))
    return out


def _read_realization_metadata(json_path):
    """Read the scheduling metadata needed for one recipe."""
    with open(json_path) as f:
        raw = decode_instance_file(json.load(f))
    suite = raw.get('_suite', {})
    record = read_recipe_record(json_path)
    family_id = int(record.family_id or 0)
    recipe_index = int(record.recipe_index or 0)
    recipe = record.recipe_id
    sample = int(suite.get('sample', 1) or 1)
    f_star = _get_f_star(json_path)
    return {
        'json_path': str(json_path),
        'suite_family': record.family,
        'family_id': family_id,
        'recipe_index': recipe_index,
        'recipe': recipe,
        'sample': sample,
        'realization_id': int(sample),
        'instance_d': int(raw['d']),
        'f_star': float(f_star),
    }


def _build_solver_tasks(all_files, cfg, evaluation_device='auto'):
    """Build one task per (recipe, solver, run replicate)."""
    checkpoints = [int(c) for c in cfg['CHECKPOINTS_FE']]
    n_runs = int(cfg['N_RUNS'])
    tasks = []
    for json_path in all_files:
        meta = _read_realization_metadata(json_path)
        max_fe = int(cfg.get('MAX_FE_PER_DIM', 10000) * meta['instance_d'])
        for solver_cfg in cfg['SOLVERS']:
            for replicate_id in range(n_runs):
                run_number = int(replicate_id) + 1
                seed = _derive_seed(
                    cfg.get('BASE_RUN_SEED', 10000),
                    solver_cfg['algorithm_id'],
                    run_number,
                )
                task = {
                    **meta,
                    'solver_cfg': solver_cfg,
                    'replicate_id': int(replicate_id),
                    'run_number': run_number,
                    # Repeated-run realizations are keyed by run number so
                    # Type 1/2 use a distinct concrete instance per run.
                    # Type 0 ignores this value.
                    'realization_id': run_number,
                    'n_runs': n_runs,
                    'seed': int(seed),
                    'max_fe': max_fe,
                    'checkpoints': checkpoints,
                    'verbose_loads': bool(cfg.get('VERBOSE_LOADS', True)),
                    'evaluation_device': str(evaluation_device),
                    'algorithm_protocol': str(cfg.get('ALGORITHM_PROTOCOL', 'batch_sync_v2')),
                    'realization_type': str(cfg.get('REALIZATION', {}).get('type', 'none')),
                    'realization_base_seed': str(cfg.get('REALIZATION', {}).get('base_seed', 'train')),
                }
                tasks.append(task)
    return tasks


def _task_key(task):
    return (
        str(task['recipe']),
        str(task['sample']),
        task['solver_cfg']['file'],
        int(task['replicate_id']),
    )


def _row_key(row):
    return (
        str(row['recipe']),
        str(row['sample']),
        row['solver'],
        int(row['replicate_id']),
    )


def _row_matches_task(row, task):
    """Confirm a checkpoint row was produced under the same task settings."""
    return (
        str(row.get('solver_family')) == str(task['solver_cfg']['family'])
        and int(row.get('algorithm_id')) == int(task['solver_cfg'].get('algorithm_id', -1))
        and int(row.get('n_runs')) == int(task['n_runs'])
        and int(row.get('replicate_id')) == int(task['replicate_id'])
        and int(row.get('seed')) == int(task['seed'])
        and int(row.get('max_fe')) == int(task['max_fe'])
        and str(row.get('evaluation_device', 'auto')) == str(task.get('evaluation_device', 'auto'))
        and str(row.get('algorithm_protocol', 'batch_sync_v2')) == str(
            task.get('algorithm_protocol', 'batch_sync_v2')
        )
        and str(row.get('realization_type', 'none')) == str(task.get('realization_type', 'none'))
        and str(row.get('realization_base_seed', 'train')) == str(task.get('realization_base_seed', 'train'))
    )


def _dedupe_solver_rows(rows):
    by_key = {}
    for row in rows:
        by_key[_row_key(row)] = row
    return list(by_key.values())


def process_solver_variant(task):
    """
    Worker: run one solver replicate on one recipe.

    Returns one wide row holding the raw absolute error at every FE
    checkpoint. Aggregation is intentionally NOT performed.
    """
    script_dir = Path(__file__).parent.parent  # project root
    for p in [str(script_dir),
              str(script_dir / 'algorithms'),
              str(script_dir / 'portal_core')]:
        if p not in sys.path:
            sys.path.insert(0, p)

    solver_cfg = task['solver_cfg']
    checkpoints = [int(c) for c in task['checkpoints']]
    try:
        algo_cls = _load_algorithm(solver_cfg['file'], script_dir)
        problem = _load_problem(
            task['json_path'],
            record_convergence=True,
            verbose_loads=task['verbose_loads'],
            max_fe=task['max_fe'],
            evaluation_device=task.get('evaluation_device', 'auto'),
            realization_type=task.get('realization_type', 'none'),
            realization_base_seed=task.get('realization_base_seed', 'train'),
            realization_sample_id=int(task.get('realization_id', 1)),
        )
        # Every algorithm in the portfolio is stochastic -> always seed all
        # RNG families used by the portfolio.  In particular, cma.purecma's
        # default normal generator is Python's ``random.normalvariate`` rather
        # than NumPy, so seeding NumPy alone does not make CMA-ES reproducible.
        np.random.seed(int(task['seed']))
        random.seed(int(task['seed']))
        problem.reset()
        algo = algo_cls()
        t_start = time.perf_counter()
        result = algo.run(problem)
        runtime_seconds = time.perf_counter() - t_start

        # Type 1/2 can change beta offsets and, for multi-block recipes,
        # reassign block weights. Compute the optimum from the concrete
        # realized instance rather than reusing the anchor optimum.
        from portal_core.optimum import global_optimum_value
        f_star = float(global_optimum_value(problem.instance))
        f_best_final = float(result['best_f'])
        fe_used = int(getattr(problem, 'fe_count', 0))
        cp_err = _checkpoint_errors(problem.convergence, fe_used, f_star, checkpoints)

        row = {
            'family': task['suite_family'],
            'family_id': int(task.get('family_id', 0)),
            'recipe': task['recipe'],
            'recipe_index': int(task.get('recipe_index', 0)),
            'sample': task['sample'],
            'solver': solver_cfg['file'],
            'solver_family': solver_cfg['family'],
            'algorithm_id': int(solver_cfg.get('algorithm_id', -1)),
            'replicate_id': int(task['replicate_id']),
            'run_number': int(task['run_number']),
            'seed': int(task['seed']),
            'f_star': f_star,
            'f_best_final': f_best_final,
            'fe_used': fe_used,
            'runtime_seconds': float(runtime_seconds),
            'n_runs': int(task['n_runs']),
            'max_fe': int(task['max_fe']),
            'evaluation_device': str(task.get('evaluation_device', 'auto')),
            'algorithm_protocol': str(task.get('algorithm_protocol', 'batch_sync_v2')),
            'realization_type': str(task.get('realization_type', 'none')),
            'realization_base_seed': str(task.get('realization_base_seed', 'train')),
            'error': None,
        }
        for c in checkpoints:
            row[f'ae_{c}'] = float(cp_err[c])
        return row

    except Exception as e:
        return {
            'json_path': task['json_path'],
            'family': task.get('suite_family', 'Family?'),
            'family_id': int(task.get('family_id', 0)),
            'recipe': task.get('recipe', ''),
            'recipe_index': int(task.get('recipe_index', 0)),
            'sample': task.get('sample', ''),
            'solver': solver_cfg.get('file', ''),
            'solver_family': solver_cfg.get('family', ''),
            'algorithm_id': int(solver_cfg.get('algorithm_id', -1)),
            'replicate_id': int(task.get('replicate_id', -1)),
            'run_number': int(task.get('run_number', 0)),
            'seed': int(task.get('seed', -1)),
            'evaluation_device': str(task.get('evaluation_device', 'auto')),
            'algorithm_protocol': str(task.get('algorithm_protocol', 'batch_sync_v2')),
            'realization_type': str(task.get('realization_type', 'none')),
            'realization_base_seed': str(task.get('realization_base_seed', 'train')),
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def validate_solver_portfolio(cfg):
    """Fail fast for missing solver modules or dependencies."""
    for solver_cfg in cfg['SOLVERS']:
        try:
            _load_algorithm(solver_cfg['file'], ROOT)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Cannot load solver {solver_cfg['file']!r}; "
                f"missing module dependency {e.name!r}."
            ) from e


def validate_output_paths(cfg):
    """Fail before expensive solver work if output dirs are not writable."""
    for key in ['OUTPUT_CSV', 'CHECKPOINT_CSV', 'METADATA_MD']:
        path = Path(cfg[key])
        parent = path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            probe = parent / f'.{path.name}.write-test-{os.getpid()}'
            with open(probe, 'w') as f:
                f.write('')
            probe.unlink()
        except OSError as e:
            raise PermissionError(
                f"Cannot write {key} under {parent}. "
                f"Fix directory permissions or set CONFIG[{key!r}] to a writable path."
            ) from e


def _read_solver_checkpoint(path, expected_tasks_by_key, cfg):
    """Load previously completed per-run checkpoint rows for resuming."""
    path = Path(path)
    if not path.exists():
        return []
    df = read_characterization_csv(path)
    ae_cols = _ae_cols(cfg)
    required = set(_run_cols(cfg))
    missing = required - set(df.columns)
    if missing:
        print(
            f"Checkpoint {path} is missing columns {sorted(missing)}; "
            "ignoring old checkpoint rows."
        )
        return []

    rows = []
    for row in df.to_dict('records'):
        key = _row_key(row)
        task = expected_tasks_by_key.get(key)
        complete = task is not None and all(pd.notna(row.get(c)) for c in ae_cols)
        if complete and _row_matches_task(row, task):
            row['replicate_id'] = int(row['replicate_id'])
            row['run_number'] = int(row['run_number'])
            row['family_id'] = int(row['family_id'])
            row['recipe_index'] = int(row['recipe_index'])
            row['algorithm_id'] = int(row['algorithm_id'])
            row['seed'] = int(row['seed'])
            row['f_star'] = float(row['f_star'])
            row['f_best_final'] = float(row['f_best_final'])
            row['fe_used'] = int(row['fe_used'])
            row['runtime_seconds'] = float(row['runtime_seconds'])
            row['n_runs'] = int(row['n_runs'])
            row['max_fe'] = int(row['max_fe'])
            for c in ae_cols:
                row[c] = float(row[c])
            row['evaluation_device'] = str(row['evaluation_device'])
            row['algorithm_protocol'] = str(row['algorithm_protocol'])
            row['realization_type'] = str(row['realization_type'])
            row['realization_base_seed'] = str(row['realization_base_seed'])
            rows.append(row)
    return _dedupe_solver_rows(rows)


def _write_solver_checkpoint(rows, path, cfg):
    """Persist completed per-run rows so the run can resume after a stop."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(_dedupe_solver_rows(rows))
    cols = _run_cols(cfg)
    df = df[[c for c in cols if c in df.columns]]
    df['_sample_num'] = pd.to_numeric(df['sample'], errors='coerce')
    df = df.sort_values(
        ['family_id', 'recipe_index', '_sample_num', 'algorithm_id', 'replicate_id']
    ).drop(columns=['_sample_num'])
    tmp_path = path.with_name(path.name + '.tmp')
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _format_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _print_progress(done, total, start_time):
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    print(
        f"Progress: {done:,}/{total:,} solver runs "
        f"({done / max(total, 1):.1%}) | "
        f"elapsed {_format_duration(elapsed)} | ETA {_format_duration(eta)}"
    )


def _next_heartbeat_percent(done, total, heartbeat_percent):
    heartbeat_percent = max(0.01, float(heartbeat_percent))
    current_percent = 100.0 * int(done) / max(int(total), 1)
    return (int(current_percent / heartbeat_percent) + 1) * heartbeat_percent


def _print_performance_heartbeat(
    target_percent,
    done,
    total,
    start_time,
    completed_before=0,
):
    elapsed = time.time() - start_time
    done_this_session = max(0, int(done) - int(completed_before))
    rate = done_this_session / elapsed if elapsed > 0 else 0.0
    eta = (int(total) - int(done)) / rate if rate > 0 else 0.0
    print(
        f"Performance heartbeat {float(target_percent):.1f}% | "
        f"solver runs {int(done):,}/{int(total):,} | "
        f"elapsed {_format_duration(elapsed)} | "
        f"ETA {_format_duration(eta)}",
        flush=True,
    )


def _run_completion_message(row, completed=None, total=None):
    problem = f"{row.get('recipe', '?')} sample {row.get('sample', '?')}"
    prefix = ''
    if completed is not None and total is not None:
        prefix = f"[{int(completed)}/{int(total)}] "
    return (
        prefix +
        f"{row.get('solver', '?')} algorithm, {problem} problem, "
        f"run #{int(row.get('run_number', 0))} completed, "
        f"running time {float(row.get('runtime_seconds', 0.0)):.2f} s"
    )


def _iter_completed_run_tasks(
    tasks,
    n_workers,
    *,
    alive_seconds=30.0,
    start_time=None,
):
    """EDOLAB-style streaming run scheduler."""
    total = len(tasks)
    if total == 0:
        return
    if n_workers == 1:
        with single_threaded_numerics():
            for completed, task in enumerate(tasks, start=1):
                yield process_solver_variant(task), completed, total
        return

    with single_threaded_numerics(), ProcessPoolExecutor(
        max_workers=n_workers, initializer=initialize_numeric_worker,
    ) as executor:
        task_iter = iter(tasks)
        pending = set()
        completed = 0

        def submit_next():
            try:
                task = next(task_iter)
            except StopIteration:
                return False
            pending.add(executor.submit(process_solver_variant, task))
            return True

        for _ in range(min(n_workers, total)):
            submit_next()

        while pending:
            timeout = float(alive_seconds) if float(alive_seconds) > 0.0 else None
            done, pending = wait(
                pending,
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                elapsed = time.time() - (start_time or time.time())
                print(
                    f"Performance workers alive | completed this session {completed:,}/{total:,} | "
                    f"active workers {len(pending):,} | elapsed {_format_duration(elapsed)}",
                    flush=True,
                )
                continue
            for future in done:
                completed += 1
                yield future.result(), completed, total
                while len(pending) < n_workers:
                    if not submit_next():
                        break


def _build_raw_table(rows, tasks, cfg):
    """
    Expand the wide per-run rows into the RAW long-format Performance table:
    one row per (recipe, algorithm, run, checkpoint). No aggregation.
    """
    rows = _dedupe_solver_rows(rows)
    expected_keys = {_task_key(task) for task in tasks}
    present_keys = {_row_key(row) for row in rows}
    missing = sorted(expected_keys - present_keys)
    if missing:
        examples = ', '.join(f"{r}/{s}/{a}/rep{rep}" for r, s, a, rep in missing[:5])
        raise ValueError(
            f"Performance pipeline incomplete: missing {len(missing)} solver run "
            f"result(s). Examples: {examples}"
        )

    checkpoints = [int(c) for c in cfg['CHECKPOINTS_FE']]
    long_rows = []
    for r in rows:
        if _row_key(r) not in expected_keys:
            continue
        for c in checkpoints:
            long_rows.append({
                'recipe_id':             r['recipe'],
                'family':                r['family'],
                'family_id':             int(r['family_id']),
                'recipe_index':          int(r['recipe_index']),
                'algorithm':             r['solver'],
                'algorithm_id':          int(r['algorithm_id']),
                'run_id':                int(r['run_number']),
                'seed':                  int(r['seed']),
                'checkpoint_fe':         int(c),
                'absolute_error':        float(r[f'ae_{c}']),
                'f_star':                float(r['f_star']),
                'max_fe':                int(r['max_fe']),
                'realization_type':      str(r['realization_type']),
                'realization_base_seed': str(r['realization_base_seed']),
            })

    df = pd.DataFrame(long_rows)
    df = df.sort_values(
        ['family_id', 'recipe_index', 'algorithm_id', 'run_id', 'checkpoint_fe']
    ).reset_index(drop=True)
    return df


def collect_all_realizations(instances_root):
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


def _write_metadata(cfg, n_anchors, n_tasks, evaluation_device):
    """Write the Performance measurement settings and raw-table contract."""
    checkpoints = ', '.join(f"{int(c):,}" for c in cfg['CHECKPOINTS_FE'])
    solvers = ', '.join(
        f"{s['file']} (algorithm_id={s['algorithm_id']})" for s in cfg['SOLVERS']
    )
    text = f"""# Performance Raw Data-Collection — Metadata Note

Auto-generated by `run_curation_solvers.py`.

## Scope
- Recipe suite root: `{cfg['INSTANCES_ROOT']}`
- Recipes processed: {n_anchors}
- Solver runs (recipes x algorithms x runs): {n_tasks}

## Realization mode
- `REALIZATION = {{'type': '{cfg['REALIZATION']['type']}', 'base_seed': '{cfg['REALIZATION']['base_seed']}'}}`
- Type 0 (`none`) uses the anchor directly. For Type 1/2, run number
  `1..N_RUNS` is also the realization sample identifier, yielding one
  deterministic concrete realization per repeated run.

## Solver portfolio ({len(cfg['SOLVERS'])} configured algorithms)
- {solvers}
- Powell and PatternSearch use random restarts / random probes, so all
  configured algorithms are seeded for repeated evaluation.

## Runs and seeding
- Independent runs per (recipe, algorithm): {cfg['N_RUNS']}
- Run seed rule: `seed = {cfg['BASE_RUN_SEED']} + 100 * algorithm_id + run_id`
  with `algorithm_id` in 1..7 and `run_id` in 1..{cfg['N_RUNS']}.

## Evaluation budget and checkpoints
- Budget: {cfg['MAX_FE_PER_DIM']:,} x d function evaluations
  (= {cfg['MAX_FE_PER_DIM'] * 30:,} FE for d=30).
- FE checkpoints recorded per run: {checkpoints}
- Evaluation backend: {evaluation_device}

## Recorded quantity
- Raw absolute error `|f_best - f_star|` at each checkpoint.
- `f_star` = sum_j block_weight_j * min_k(beta) over block components.
- NO log-transform, normalization, scaling, clipping, or aggregation.

## Output
- Raw table: `{cfg['OUTPUT_CSV']}`
- One row per (recipe, algorithm, run, checkpoint).
- Columns: recipe_id, family, family_id, recipe_index, algorithm,
  algorithm_id, run_id, seed, checkpoint_fe, absolute_error, f_star,
  max_fe, realization_type, realization_base_seed.
"""
    path = Path(cfg['METADATA_MD'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


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


def _csv_positive_ints(value):
    try:
        parsed = [int(item.strip()) for item in str(value).split(',') if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be comma-separated integers") from exc
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("must contain positive integers")
    if parsed != sorted(set(parsed)):
        raise argparse.ArgumentTypeError("must be unique and strictly increasing")
    return parsed


def _csv_names(value):
    parsed = [item.strip() for item in str(value).split(',') if item.strip()]
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("must be a non-empty comma-separated list without duplicates")
    return parsed


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the PORTAL Performance raw data-collection solver portfolio."
    )
    parser.add_argument(
        "--max-parallel-runs",
        type=_positive_int,
        default=None,
        help=(
            "Maximum number of solver runs to execute concurrently. "
            "Defaults to 90%% of available CPU cores."
        ),
    )
    parser.add_argument(
        "--evaluation-device",
        choices=("auto", "numpy", "cpu", "cuda", "mps"),
        default="cpu",
        help=(
            "PORTAL fitness backend. 'numpy' uses the NumPy path; "
            "'cpu', 'cuda', or 'mps' start from that torch device; "
            "'auto' tries cuda -> cpu -> mps -> NumPy. Default: cpu."
        ),
    )
    parser.add_argument(
        "--base-run-seed",
        type=int,
        default=int(CONFIG['BASE_RUN_SEED']),
        help="Base value in seed = Base + 100*algorithm_id + run_id.",
    )
    parser.add_argument(
        "--algorithms",
        type=_csv_names,
        default=[solver['file'] for solver in CONFIG['SOLVERS']],
        help="Comma-separated subset of the configured Performance algorithm portfolio.",
    )
    parser.add_argument(
        "--n-runs",
        type=_positive_int,
        default=int(CONFIG['N_RUNS']),
        help="Independent runs per recipe and algorithm.",
    )
    parser.add_argument(
        "--max-fe-per-dim",
        type=_positive_int,
        default=int(CONFIG['MAX_FE_PER_DIM']),
        help="Function-evaluation budget per problem dimension.",
    )
    parser.add_argument(
        "--checkpoints-fe",
        type=_csv_positive_ints,
        default=list(CONFIG['CHECKPOINTS_FE']),
        help="Comma-separated absolute FE checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=_positive_int,
        default=int(CONFIG['CHECKPOINT_EVERY']),
        help="Persist the resumable run table after this many completed runs.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG['RESUME']),
        help="Resume compatible completed runs from the checkpoint CSV.",
    )
    parser.add_argument(
        "--heartbeat-percent",
        type=_heartbeat_percent,
        default=float(CONFIG['HEARTBEAT_PERCENT']),
        help="Percentage interval between progress heartbeat messages.",
    )
    parser.add_argument(
        "--heartbeat-alive-seconds",
        type=_nonnegative_float,
        default=30.0,
        help="Emit an alive message after this many quiet seconds; zero disables it.",
    )
    parser.add_argument(
        "--realization-type",
        choices=("none", "type1", "type2"),
        default=str(CONFIG['REALIZATION']['type']),
        help="Runtime realization mode (default: none / Type 0).",
    )
    parser.add_argument(
        "--realization-base-seed",
        default=str(CONFIG['REALIZATION']['base_seed']),
        help="Runtime realization base seed: train, test, or integer.",
    )
    parser.add_argument(
        "--instances-root",
        type=Path,
        default=Path(CONFIG['INSTANCES_ROOT']),
        help=(
            "Root recursively containing canonical unified recipe JSON files. "
            "Family and dimension are read from each JSON, not its path."
        ),
    )
    parser.add_argument(
        "--recipe-ids-file",
        type=Path,
        default=None,
        help="Optional one-column text/CSV file selecting recipe IDs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(CONFIG['OUTPUT_CSV']),
        help="Raw Performance output CSV path.",
    )
    parser.add_argument(
        "--checkpoint-csv",
        type=Path,
        default=Path(CONFIG['CHECKPOINT_CSV']),
        help="Isolated resumable Performance checkpoint path.",
    )
    parser.add_argument(
        "--metadata-md",
        type=Path,
        default=Path(CONFIG['METADATA_MD']),
        help="Performance metadata output path.",
    )
    return parser.parse_args(argv)


def _resolve_max_parallel_runs(value, cpu_count=None):
    n_cpu = multiprocessing.cpu_count() if cpu_count is None else int(cpu_count)
    if value is None:
        value = max(1, int(n_cpu * 0.9))
    return cap_process_workers(value, n_cpu)


def main(argv=None):
    args = _parse_args(argv)
    CONFIG['INSTANCES_ROOT'] = str(args.instances_root)
    CONFIG['OUTPUT_CSV'] = str(args.output_csv)
    CONFIG['CHECKPOINT_CSV'] = str(args.checkpoint_csv)
    CONFIG['METADATA_MD'] = str(args.metadata_md)
    CONFIG['BASE_RUN_SEED'] = int(args.base_run_seed)
    available_solvers = {solver['file']: solver for solver in CONFIG['SOLVERS']}
    unknown_solvers = sorted(set(args.algorithms) - set(available_solvers))
    if unknown_solvers:
        raise ValueError(
            f"Unknown Performance algorithms {unknown_solvers}; available: {sorted(available_solvers)}"
        )
    CONFIG['SOLVERS'] = [available_solvers[name] for name in args.algorithms]
    CONFIG['N_RUNS'] = int(args.n_runs)
    CONFIG['MAX_FE_PER_DIM'] = int(args.max_fe_per_dim)
    CONFIG['CHECKPOINTS_FE'] = list(args.checkpoints_fe)
    CONFIG['CHECKPOINT_EVERY'] = int(args.checkpoint_every)
    CONFIG['RESUME'] = bool(args.resume)
    CONFIG['HEARTBEAT_PERCENT'] = float(args.heartbeat_percent)
    CONFIG['HEARTBEAT_ALIVE_SECONDS'] = float(args.heartbeat_alive_seconds)
    CONFIG['REALIZATION']['type'] = str(args.realization_type)
    CONFIG['REALIZATION']['base_seed'] = str(args.realization_base_seed)
    print("=" * 60)
    print("Performance Raw Data-Collection - PORTAL Recipe Suite")
    print("=" * 60)

    solver_names = [s['file'] for s in CONFIG['SOLVERS']]
    print(f"\nSolver portfolio (all stochastic, {CONFIG['N_RUNS']} runs each):")
    print(f"  {solver_names}")
    print(f"  Budget per run: {CONFIG['MAX_FE_PER_DIM']:,} x d FE")
    print(f"  FE checkpoints: {CONFIG['CHECKPOINTS_FE']}")
    print(f"  Evaluation backend: {args.evaluation_device}")
    print(
        "  Realization mode: "
        f"type={CONFIG['REALIZATION']['type']}, "
        f"base_seed={CONFIG['REALIZATION']['base_seed']}"
    )
    print(f"  Run seed base: {CONFIG['BASE_RUN_SEED']}")

    all_files, recipes = collect_all_realizations(CONFIG['INSTANCES_ROOT'])
    recipe_ids = _read_recipe_ids(args.recipe_ids_file)
    all_files = _filter_recipe_files(all_files, recipe_ids)
    total      = len(all_files)
    total_runs = total * CONFIG['N_RUNS'] * len(solver_names)
    print(f"\nTotal recipes:      {total}")
    print(f"Total solver runs:  {total_runs:,}")
    print(f"Output (raw table): {CONFIG['OUTPUT_CSV']}\n")

    validate_solver_portfolio(CONFIG)
    validate_output_paths(CONFIG)

    tasks = _build_solver_tasks(
        all_files,
        CONFIG,
        evaluation_device=args.evaluation_device,
    )
    expected_tasks_by_key = {_task_key(task): task for task in tasks}
    checkpoint_path = Path(CONFIG['CHECKPOINT_CSV'])
    checkpoint_rows = []
    if CONFIG.get('RESUME', True):
        checkpoint_rows = _read_solver_checkpoint(
            checkpoint_path, expected_tasks_by_key, CONFIG
        )

    completed_keys = {_row_key(row) for row in checkpoint_rows}
    pending_tasks = [task for task in tasks if _task_key(task) not in completed_keys]
    print(f"Solver run tasks: {len(tasks):,}")
    if checkpoint_rows:
        print(f"Resuming from checkpoint: {len(checkpoint_rows):,} completed")
    print(f"Pending solver runs: {len(pending_tasks):,}")
    print(f"Checkpoint: {checkpoint_path}\n")

    n_cpu     = multiprocessing.cpu_count()
    max_parallel_runs = _resolve_max_parallel_runs(args.max_parallel_runs, n_cpu)
    print(f"Using {max_parallel_runs} parallel run(s) on {n_cpu} available core(s)\n")

    t_start = time.time()

    new_rows = []
    bad = []
    checkpoint_every = max(1, int(CONFIG.get('CHECKPOINT_EVERY', 25)))
    completed_before = len(completed_keys)
    heartbeat_percent = float(CONFIG.get('HEARTBEAT_PERCENT', 0.1))
    next_heartbeat = _next_heartbeat_percent(
        completed_before,
        len(tasks),
        heartbeat_percent,
    )
    print(
        f"Performance heartbeat {100.0 * completed_before / max(len(tasks), 1):.1f}% | "
        f"solver runs {completed_before:,}/{len(tasks):,}",
        flush=True,
    )

    if pending_tasks:
        for result, idx, total_pending in _iter_completed_run_tasks(
            pending_tasks,
            max_parallel_runs,
            alive_seconds=CONFIG.get('HEARTBEAT_ALIVE_SECONDS', 30.0),
            start_time=t_start,
        ):
            if result.get('error') is None:
                new_rows.append(result)
                checkpoint_rows.append(result)
            else:
                bad.append(result)
                print(
                    f"Performance run FAILED: {result.get('json_path', '?')}::"
                    f"{result.get('solver', '?')}: {result.get('error', '?')}",
                    flush=True,
                )
            done_total = completed_before + idx
            current_percent = 100.0 * done_total / max(len(tasks), 1)
            while current_percent + 1e-12 >= next_heartbeat:
                _print_performance_heartbeat(
                    next_heartbeat,
                    done_total,
                    len(tasks),
                    t_start,
                    completed_before=completed_before,
                )
                next_heartbeat = round(next_heartbeat + heartbeat_percent, 10)
                if next_heartbeat > 100.0:
                    break
            if idx % checkpoint_every == 0 or idx == total_pending:
                _write_solver_checkpoint(checkpoint_rows, checkpoint_path, CONFIG)

    elapsed = time.time() - t_start

    if bad:
        examples = ', '.join(
            f"{r.get('json_path', '?')}::{r.get('solver', '?')}: "
            f"{r.get('error', '?')}"
            for r in bad[:5]
        )
        raise ValueError(f"Performance pipeline has failed solver runs ({len(bad)}): {examples}")

    solver_rows = _dedupe_solver_rows(checkpoint_rows + new_rows)
    df = _build_raw_table(solver_rows, tasks, CONFIG)

    out_path = Path(CONFIG['OUTPUT_CSV'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    meta_path = _write_metadata(CONFIG, total, len(tasks), args.evaluation_device)

    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} minutes")
    print(f"Solver runs processed this run: {len(new_rows)}")
    print(f"Solver runs in checkpoint:      {len(solver_rows)}")
    print(f"Raw Performance rows written:           {df.shape[0]:,}")
    print(f"  (= {total} recipes x {len(solver_names)} algorithms "
          f"x {CONFIG['N_RUNS']} runs x {len(CONFIG['CHECKPOINTS_FE'])} checkpoints)")
    print(f"Raw table saved to:             {out_path}")
    print(f"Metadata note saved to:         {meta_path}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
