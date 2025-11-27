"""
Evaluator_Python_Script.py
Load and evaluate PORTAL benchmark instances using a sample optimizer.

This script demonstrates how to:
  - Load PORTAL benchmark instances from JSON files
  - Evaluate candidate solutions using the PORTAL fitness function
  - Apply a sample optimizer (Differential Evolution) for demonstration

Notes:
  - You can replace the sample DE implementation with any custom optimizer
  - The script is intended as a template for benchmarking experiments

Usage:
  1. Set the instance_file variable to your JSON instance path
  2. Optionally adjust optimizer parameters
  3. Run the script to evaluate or optimize the instance

Author: Delaram Yazdani, Danial Yazdani, Mai Peng
Email: delaram.yazdani@yahoo.com
       danial.yazdani@gmail.com
       pengmai1998@gmail.com
Last Edited: 2025-11-28
"""

import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from Utils.load_instance import load_instance
from Utils.fitness import fitness

# Configuration
instance_file = 'Instances/test_batch_1.json'  # Example: 'Instances/my_benchmark.json'
RUNS = 5  # Number of independent runs
MAX_EVALS = 10000 * 2  # Maximum function evaluations
ACCEPTED_ERROR = 1e-8  # Acceptance error threshold

# DE Algorithm parameters
DE_config = {
    'PopulationSize': 100,
    'F': 0.5,      # Scaling factor
    'Cr': 0.9,     # Crossover rate
    'variant': 'rand/1'  # DE variant
}

def load_portal_instance(filename, max_evals, accepted_error):
    """Load PORTAL instance from JSON and add evaluation tracking."""
    # Load PORTAL instance using the standard loader
    PORTAL = load_instance(filename)

    # Add evaluation tracking variables
    PORTAL['max_evals'] = max_evals
    PORTAL['accepted_error'] = accepted_error
    PORTAL['FE'] = 0
    PORTAL['FEhistory'] = np.full(max_evals, np.inf)
    PORTAL['BestFoundResult'] = np.inf
    PORTAL['AcceptanceReachPoint'] = np.inf

    # Compute true optimum (minimum beta value)
    PORTAL['optimum_value'] = np.min(PORTAL['P']['beta'])

    return PORTAL

def initialize_population(NP, dim, bounds):
    """Initialize population uniformly within bounds."""
    lower, upper = bounds
    return lower + (upper - lower) * np.random.rand(NP, dim)

def mutation_rand_1(population, F):
    """DE/rand/1 mutation strategy."""
    NP, dim = population.shape
    donors = np.zeros((NP, dim))

    for i in range(NP):
        # Select three distinct random individuals
        candidates = list(set(range(NP)) - {i})
        r = np.random.choice(candidates, 3, replace=False)

        # DE/rand/1 mutation
        donors[i] = population[r[0]] + F * (population[r[1]] - population[r[2]])

    return donors

def binomial_crossover(population, donors, Cr):
    """Binomial crossover."""
    NP, dim = population.shape
    trials = population.copy()

    # Ensure at least one parameter comes from donor
    j_rand = np.random.randint(0, dim, NP)
    for i in range(NP):
        trials[i, j_rand[i]] = donors[i, j_rand[i]]

    # Probabilistic crossover for other parameters
    crossover_mask = np.random.rand(NP, dim) < Cr
    trials[crossover_mask] = donors[crossover_mask]

    return trials

def boundary_correction(trials, population, bounds):
    """Midpoint boundary correction."""
    lower, upper = bounds
    corrected = trials.copy()

    # Find violations
    too_low = trials < lower
    too_high = trials > upper

    # Midpoint correction
    if np.any(too_low):
        midpoint_low = (lower + population) / 2
        corrected[too_low] = midpoint_low[too_low]

    if np.any(too_high):
        midpoint_high = (upper + population) / 2
        corrected[too_high] = midpoint_high[too_high]

    return corrected

def evaluate_population(X, PORTAL):
    """Evaluate population using PORTAL fitness function."""
    n_samples = X.shape[0]
    result = np.full(n_samples, np.nan)

    for jj in range(n_samples):
        # Extract current solution
        x = X[jj]

        # Evaluate using PORTAL fitness function
        f_val = fitness(x, PORTAL)
        result[jj] = f_val

        # Check evaluation limit
        if PORTAL['FE'] >= PORTAL['max_evals']:
            return result, PORTAL

        # Update tracking
        PORTAL['FE'] += 1
        PORTAL['FEhistory'][PORTAL['FE'] - 1] = result[jj]

        # Update best found
        if PORTAL['BestFoundResult'] > result[jj]:
            PORTAL['BestFoundResult'] = result[jj]

        # Check acceptance threshold
        if (abs(result[jj] - PORTAL['optimum_value']) < PORTAL['accepted_error'] and
            np.isinf(PORTAL['AcceptanceReachPoint'])):
            PORTAL['AcceptanceReachPoint'] = PORTAL['FE']

    return result, PORTAL

def differential_evolution(PORTAL, config):
    """Differential Evolution optimizer."""
    # Extract configuration
    NP = config['PopulationSize']
    F = config['F']
    Cr = config['Cr']
    dim = PORTAL['d']
    bounds = (PORTAL['bounds_min'], PORTAL['bounds_max'])

    # Initialize population
    population = initialize_population(NP, dim, bounds)

    # Evaluate initial population
    fitness_values, PORTAL = evaluate_population(population, PORTAL)

    # Find initial best
    best_idx = np.nanargmin(fitness_values)
    best_value = fitness_values[best_idx]
    best_solution = population[best_idx].copy()

    # Main DE loop
    while PORTAL['FE'] < PORTAL['max_evals']:
        # Store current best
        current_best_idx = np.nanargmin(fitness_values)
        current_best_val = fitness_values[current_best_idx]
        if current_best_val < best_value:
            best_value = current_best_val
            best_solution = population[current_best_idx].copy()

        # Generate donor vectors (mutation)
        donors = mutation_rand_1(population, F)

        # Create trial vectors (crossover)
        trials = binomial_crossover(population, donors, Cr)

        # Boundary correction
        trials = boundary_correction(trials, population, bounds)

        # Evaluate trial vectors
        trial_fitness, PORTAL = evaluate_population(trials, PORTAL)

        # Selection
        improved = trial_fitness < fitness_values
        population[improved] = trials[improved]
        fitness_values[improved] = trial_fitness[improved]

        # Check termination
        if PORTAL['FE'] >= PORTAL['max_evals']:
            break

    # Final best
    final_best_idx = np.nanargmin(fitness_values)
    final_best_val = fitness_values[final_best_idx]
    if final_best_val < best_value:
        best_value = final_best_val
        best_solution = population[final_best_idx].copy()

    # Return convergence history
    convergence_history = PORTAL['FEhistory']

    return best_solution, best_value, PORTAL, convergence_history

def main():
    """Main evaluation loop."""
    print('PORTAL Instance Evaluator')
    print('========================')
    print(f'Instance file: {instance_file}')
    print(f'Number of runs: {RUNS}')
    print(f'Max evaluations: {MAX_EVALS}\n')

    # Storage for results
    Error = np.full(RUNS, np.nan)
    AcceptancePoints = np.full(RUNS, np.nan)
    FinalValues = np.full(RUNS, np.nan)
    ConvergenceHistory = []

    # Check if instance file exists
    if not os.path.exists(instance_file):
        raise FileNotFoundError(f'Instance file not found: {instance_file}')

    # Run the optimizer multiple times
    for RunCounter in range(RUNS):
        print(f'Run {RunCounter + 1}/{RUNS}: ', end='')

        # Load PORTAL instance from JSON
        PORTAL = load_portal_instance(instance_file, MAX_EVALS, ACCEPTED_ERROR)

        # Set random seed for reproducibility within run
        np.random.seed(None)

        # Run DE optimizer
        best_solution, best_value, PORTAL, history = differential_evolution(PORTAL, DE_config)

        # Compute true optimum
        true_optimum = PORTAL['optimum_value']

        # Store results
        Error[RunCounter] = abs(best_value - true_optimum)
        AcceptancePoints[RunCounter] = PORTAL['AcceptanceReachPoint']
        FinalValues[RunCounter] = best_value
        ConvergenceHistory.append(np.minimum.accumulate(history))

        print(f'Best = {best_value:.6e}, Error = {Error[RunCounter]:.6e}', end='')
        if np.isfinite(PORTAL['AcceptanceReachPoint']):
            print(f', Acceptance at FE = {int(PORTAL["AcceptanceReachPoint"])}')
        else:
            print()

    # Display results
    print('\nSummary Results')
    print('===============')

    # Acceptance statistics
    nonInfIndices = np.isfinite(AcceptancePoints)
    if np.any(nonInfIndices):
        nonInfValues = AcceptancePoints[nonInfIndices]
        print(f'Average FE to reach acceptance: {np.mean(nonInfValues):.2f} ({np.std(nonInfValues):.2f})')
        print(f'Acceptance ratio: {100 * np.sum(nonInfIndices) / len(AcceptancePoints):.1f}%')
    else:
        print('No runs reached acceptance threshold')

    # Final error statistics
    print(f'Final error: {np.mean(Error):.6e} ({np.std(Error):.6e})')
    print(f'Best final value found: {np.min(FinalValues):.6f}')
    print(f'True optimum value: {true_optimum:.6f}')

    # Plot average error convergence with acceptance line
    # Truncate histories to common length
    lens = [len(h) for h in ConvergenceHistory]
    minLen = min(lens)

    # Build error matrix
    errData = np.full((RUNS, minLen), np.nan)
    for i in range(RUNS):
        hist = ConvergenceHistory[i][:minLen]
        errData[i, :] = np.abs(hist - true_optimum)

    # Average over runs
    avgErr = np.mean(errData, axis=0)

    # Plot average error curve
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
