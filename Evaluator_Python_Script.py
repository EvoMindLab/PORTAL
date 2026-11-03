"""
Author: Delaram Yazdani, Danial Yazdani, Mai Peng
Email: delaram.yazdani@yahoo.com
       danial.yazdani@gmail.com
       pengmai1998@gmail.com
Evaluator_Python_Script.py
Load and evaluate PORTAL benchmark instances using a sample optimizer.

This is the PORTAL2-local evaluator mirror. It keeps the same evaluator
workflow while using PORTAL2 self-contained modules in portal_core.
"""

import os
import re
import hashlib
import numpy as np
import matplotlib.pyplot as plt

from portal_core.load_instance import load_instance
from portal_core.fitness import fitness
from portal_core.realization import (
    realize,
    METHOD_CENTER_BETA,
    METHOD_DIVERSE,
    PORTAL_SEED_TRAIN,
    PORTAL_SEED_TEST,
)

# Configuration
SAMPLE_NAME = 'f01_r001.json'
SAMPLE_FOLDER = 'Instances'

RUNS = 5
MAX_EVALS = 10000 * 2
ACCEPTED_ERROR = 1e-8

DE_config = {
    'PopulationSize': 100,
    'F': 0.5,
    'Cr': 0.9,
    'variant': 'rand/1'
}

REALIZATION_NONE = 'none'
REALIZATION_TYPE1 = 'type1'
REALIZATION_TYPE2 = 'type2'

REALIZATION = {
    'type': REALIZATION_NONE,
    'base_seed': 'train',
}


def _resolve_instance_file(sample_name, sample_folder):
    candidate = sample_name
    if not str(candidate).lower().endswith('.json'):
        candidate = f"{candidate}.json"

    if os.path.exists(candidate):
        return candidate

    root = sample_folder or '.'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(root):
        root = os.path.normpath(root)
    if not os.path.exists(root):
        alt_root = os.path.join(script_dir, root)
        if os.path.exists(alt_root):
            root = alt_root
        else:
            raise FileNotFoundError(f"Sample folder not found: {root}")

    matches = []
    for walk_root, _, files in os.walk(root):
        if os.path.basename(candidate) in files:
            matches.append(os.path.join(walk_root, os.path.basename(candidate)))

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Sample JSON '{os.path.basename(candidate)}' not found under '{root}'."
        )

    msg = "\n  ".join(matches[:10])
    raise ValueError(
        f"Multiple sample files named '{os.path.basename(candidate)}' found:\n  {msg}\n"
        "Use a more specific sample name/path."
    )


def _schema_blocks_optimum(PORTAL):
    """Optimum for unified block schema: sum over blocks of weight * min(beta in block)."""
    schema_blocks = PORTAL.get('_schema_blocks')
    if not isinstance(schema_blocks, list) or len(schema_blocks) == 0:
        return float(np.min(np.asarray(PORTAL['P']['beta']).ravel()))

    betas = np.asarray(PORTAL['P']['beta']).ravel()
    K = int(PORTAL['K'])
    if betas.size != K:
        raise ValueError(f"Invalid P['beta'] length {betas.size}; expected K={K}.")

    total = 0.0
    cursor = 0
    for bi, block in enumerate(schema_blocks, start=1):
        k_block = int(block.get('K', 0))
        if k_block <= 0:
            raise ValueError(f"Invalid _schema_blocks[{bi - 1}].K={k_block}; expected positive integer.")
        if cursor + k_block > K:
            raise ValueError("Invalid _schema_blocks partition: cumulative block K exceeds total K.")

        weight = float(block.get('weight', 1.0))
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Invalid _schema_blocks[{bi - 1}].weight={weight}; expected finite > 0.")

        total += weight * float(np.min(betas[cursor:cursor + k_block]))
        cursor += k_block

    if cursor != K:
        raise ValueError(
            f"Invalid _schema_blocks partition: cumulative block K={cursor}, expected K={K}."
        )

    return float(total)


def _instance_optimum(PORTAL):
    return _schema_blocks_optimum(PORTAL)


def _resolve_realization_base_seed(base_seed):
    if isinstance(base_seed, str):
        key = base_seed.strip().lower()
        if key == 'train':
            return int(PORTAL_SEED_TRAIN)
        if key == 'test':
            return int(PORTAL_SEED_TEST)
        return int(base_seed)
    return int(base_seed)


def _derive_recipe_ids_from_sample_name(sample_name):
    base = os.path.basename(str(sample_name))
    stem = os.path.splitext(base)[0]

    m = re.search(r'f(\d+)_r(\d+)', stem, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))

    digest = hashlib.sha1(stem.encode('utf-8')).digest()
    family_id = int.from_bytes(digest[:4], byteorder='big', signed=False) % 1_000_000 + 1
    recipe_index = int.from_bytes(digest[4:8], byteorder='big', signed=False) % 1_000_000 + 1
    return family_id, recipe_index


def _map_realization_type_to_method(realization_type):
    rtype = str(realization_type).strip().lower()
    if rtype == REALIZATION_NONE:
        return None
    if rtype == REALIZATION_TYPE1:
        return METHOD_CENTER_BETA
    if rtype == REALIZATION_TYPE2:
        return METHOD_DIVERSE
    raise ValueError(
        f"Unsupported REALIZATION['type']={realization_type!r}. "
        f"Use {REALIZATION_NONE!r}, {REALIZATION_TYPE1!r}, or {REALIZATION_TYPE2!r}."
    )


def _apply_realization(instance, realization_config, run_index, sample_name):
    method = _map_realization_type_to_method(realization_config.get('type', REALIZATION_NONE))
    if method is None:
        return instance

    base_seed = _resolve_realization_base_seed(
        realization_config.get('base_seed', PORTAL_SEED_TRAIN)
    )
    sample_id = int(run_index) + 1
    _ = sample_name

    try:
        instance = realize(
            instance,
            base_seed,
            sample_id,
            method=method,
        )
    except ValueError as exc:
        raise ValueError(
            "Failed to apply realization in evaluator. "
            "Check REALIZATION['type'], REALIZATION['base_seed'], and SAMPLE_NAME. "
            f"Original error: {exc}"
        ) from exc

    return instance


def load_portal_instance(filename, max_evals, accepted_error,
                         realization_config=None, run_index=0,
                         sample_name=''):
    PORTAL = load_instance(filename)

    cfg = REALIZATION if realization_config is None else realization_config
    PORTAL = _apply_realization(PORTAL, cfg, run_index, sample_name)

    PORTAL['optimum_value'] = _instance_optimum(PORTAL)

    PORTAL['max_evals'] = max_evals
    PORTAL['accepted_error'] = accepted_error
    PORTAL['FE'] = 0
    PORTAL['FEhistory'] = np.full(max_evals, np.inf)
    PORTAL['BestFoundResult'] = np.inf
    PORTAL['AcceptanceReachPoint'] = np.inf

    return PORTAL


def initialize_population(NP, dim, bounds):
    lower, upper = bounds
    return lower + (upper - lower) * np.random.rand(NP, dim)


def mutation_rand_1(population, F):
    NP, dim = population.shape
    donors = np.zeros((NP, dim))

    for i in range(NP):
        candidates = list(set(range(NP)) - {i})
        r = np.random.choice(candidates, 3, replace=False)
        donors[i] = population[r[0]] + F * (population[r[1]] - population[r[2]])

    return donors


def binomial_crossover(population, donors, Cr):
    NP, dim = population.shape
    trials = population.copy()

    j_rand = np.random.randint(0, dim, NP)
    for i in range(NP):
        trials[i, j_rand[i]] = donors[i, j_rand[i]]

    crossover_mask = np.random.rand(NP, dim) < Cr
    trials[crossover_mask] = donors[crossover_mask]

    return trials


def boundary_correction(trials, population, bounds):
    lower, upper = bounds
    corrected = trials.copy()

    too_low = trials < lower
    too_high = trials > upper

    if np.any(too_low):
        midpoint_low = (lower + population) / 2
        corrected[too_low] = midpoint_low[too_low]

    if np.any(too_high):
        midpoint_high = (upper + population) / 2
        corrected[too_high] = midpoint_high[too_high]

    return corrected


def evaluate_population(X, PORTAL):
    n_samples = X.shape[0]
    result = np.full(n_samples, np.nan)

    for jj in range(n_samples):
        if PORTAL['FE'] >= PORTAL['max_evals']:
            return result, PORTAL

        x = X[jj]
        f_val = fitness(x, PORTAL)
        result[jj] = f_val

        PORTAL['FE'] += 1
        PORTAL['FEhistory'][PORTAL['FE'] - 1] = result[jj]

        if PORTAL['BestFoundResult'] > result[jj]:
            PORTAL['BestFoundResult'] = result[jj]

        if (abs(result[jj] - PORTAL['optimum_value']) < PORTAL['accepted_error'] and
            np.isinf(PORTAL['AcceptanceReachPoint'])):
            PORTAL['AcceptanceReachPoint'] = PORTAL['FE']

    return result, PORTAL


def differential_evolution(PORTAL, config):
    NP = config['PopulationSize']
    F = config['F']
    Cr = config['Cr']
    dim = PORTAL['d']
    bounds = (PORTAL['bounds_min'], PORTAL['bounds_max'])

    population = initialize_population(NP, dim, bounds)
    fitness_values, PORTAL = evaluate_population(population, PORTAL)

    best_idx = np.nanargmin(fitness_values)
    best_value = fitness_values[best_idx]
    best_solution = population[best_idx].copy()

    while PORTAL['FE'] < PORTAL['max_evals']:
        current_best_idx = np.nanargmin(fitness_values)
        current_best_val = fitness_values[current_best_idx]
        if current_best_val < best_value:
            best_value = current_best_val
            best_solution = population[current_best_idx].copy()

        donors = mutation_rand_1(population, F)
        trials = binomial_crossover(population, donors, Cr)
        trials = boundary_correction(trials, population, bounds)

        trial_fitness, PORTAL = evaluate_population(trials, PORTAL)

        improved = trial_fitness < fitness_values
        population[improved] = trials[improved]
        fitness_values[improved] = trial_fitness[improved]

        if PORTAL['FE'] >= PORTAL['max_evals']:
            break

    final_best_idx = np.nanargmin(fitness_values)
    final_best_val = fitness_values[final_best_idx]
    if final_best_val < best_value:
        best_value = final_best_val
        best_solution = population[final_best_idx].copy()

    convergence_history = PORTAL['FEhistory']

    return best_solution, best_value, PORTAL, convergence_history


def main():
    instance_file = _resolve_instance_file(SAMPLE_NAME, SAMPLE_FOLDER)

    print('PORTAL2 Instance Evaluator')
    print('=========================')
    print(f'Sample name: {SAMPLE_NAME}')
    print(f'Instance file: {instance_file}')
    print(f'Number of runs: {RUNS}')
    print(f'Max evaluations: {MAX_EVALS}\n')
    realization_type = str(REALIZATION.get('type', REALIZATION_NONE)).lower()
    if realization_type != REALIZATION_NONE:
        print('Realization mode enabled:')
        print(f"  type = {REALIZATION.get('type')}")
        print(f"  base_seed = {REALIZATION.get('base_seed')}")
        print('  sample_id policy = run index (1..RUNS)\n')
    else:
        print('Realization mode: none (anchor used directly)\n')

    Error = np.full(RUNS, np.nan)
    AcceptancePoints = np.full(RUNS, np.nan)
    FinalValues = np.full(RUNS, np.nan)
    ConvergenceHistory = []

    if not os.path.exists(instance_file):
        raise FileNotFoundError(f'Instance file not found: {instance_file}')

    for RunCounter in range(RUNS):
        print(f'Run {RunCounter + 1}/{RUNS}: ', end='')

        PORTAL = load_portal_instance(
            instance_file,
            MAX_EVALS,
            ACCEPTED_ERROR,
            realization_config=REALIZATION,
            run_index=RunCounter,
            sample_name=SAMPLE_NAME,
        )

        np.random.seed(None)

        best_solution, best_value, PORTAL, history = differential_evolution(PORTAL, DE_config)

        true_optimum = PORTAL['optimum_value']

        Error[RunCounter] = abs(best_value - true_optimum)
        AcceptancePoints[RunCounter] = PORTAL['AcceptanceReachPoint']
        FinalValues[RunCounter] = best_value
        ConvergenceHistory.append(np.minimum.accumulate(history))

        print(f'Best = {best_value:.6e}, Error = {Error[RunCounter]:.6e}', end='')
        if np.isfinite(PORTAL['AcceptanceReachPoint']):
            print(f', Acceptance at FE = {int(PORTAL["AcceptanceReachPoint"])}')
        else:
            print()

    print('\nSummary Results')
    print('===============')

    nonInfIndices = np.isfinite(AcceptancePoints)
    if np.any(nonInfIndices):
        nonInfValues = AcceptancePoints[nonInfIndices]
        print(f'Average FE to reach acceptance: {np.mean(nonInfValues):.2f} ({np.std(nonInfValues):.2f})')
        print(f'Acceptance ratio: {100 * np.sum(nonInfIndices) / len(AcceptancePoints):.1f}%')
    else:
        print('No runs reached acceptance threshold')

    print(f'Final error: {np.mean(Error):.6e} ({np.std(Error):.6e})')
    print(f'Best final value found: {np.min(FinalValues):.6f}')
    print(f'True optimum value: {true_optimum:.6f}')

    lens = [len(h) for h in ConvergenceHistory]
    minLen = min(lens)

    errData = np.full((RUNS, minLen), np.nan)
    for i in range(RUNS):
        hist = ConvergenceHistory[i][:minLen]
        errData[i, :] = np.abs(hist - true_optimum)

    avgErr = np.mean(errData, axis=0)

    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, minLen + 1), avgErr, 'k-', linewidth=2, label='Average Error')
    plt.axhline(y=ACCEPTED_ERROR, color='r', linestyle='--', linewidth=1.5,
                label='Acceptance Threshold')

    plt.xlabel('Function Evaluations')
    plt.ylabel('Error |f - f*|')
    plt.title(f'Average Error Convergence - {instance_file}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
