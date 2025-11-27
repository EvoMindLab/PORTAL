% Evaluator_MATLAB_Script.m
% Load and evaluate PORTAL benchmark instances using a sample optimizer.
%
% This script demonstrates how to:
%   - Load PORTAL benchmark instances from JSON files
%   - Evaluate candidate solutions using the PORTAL fitness function
%   - Apply a sample optimizer (Differential Evolution) for demonstration
%
% Notes:
%   - You can replace the sample DE implementation with any custom optimizer
%   - The script is intended as a template for benchmarking experiments
%
% Usage:
%   1. Set the instance_file variable to your JSON instance path
%   2. Optionally adjust optimizer parameters
%   3. Run the script to evaluate or optimize the instance
%
% Author: Generated for PORTAL Framework
% Date: January 2025

clear; close all; clc;

%% Auto-include all subdirectories
script_path = fileparts(mfilename('fullpath'));
addpath(genpath(script_path));

%% Configuration
% Instance file to evaluate (change this to your JSON file)
instance_file = 'Instances/test_batch_1.json';  % Example: 'Instances/my_benchmark.json'

% Number of independent runs
RUNS = 5;

% Maximum function evaluations
MAX_EVALS = 10000 * 2;  % Adjust based on problem dimension

% Acceptance error threshold
ACCEPTED_ERROR = 1e-8;

% DE Algorithm parameters
DE_config = struct();
DE_config.PopulationSize = 100;
DE_config.F = 0.5;              % Scaling factor
DE_config.Cr = 0.9;             % Crossover rate
DE_config.variant = 'rand/1';   % DE variant

%% Main evaluation loop
fprintf('PORTAL Instance Evaluator\n');
fprintf('========================\n');
fprintf('Instance file: %s\n', instance_file);
fprintf('Number of runs: %d\n', RUNS);
fprintf('Max evaluations: %d\n\n', MAX_EVALS);

% Storage for results
Error = NaN(1, RUNS);
AcceptancePoints = NaN(1, RUNS);
FinalValues = NaN(1, RUNS);
ConvergenceHistory = cell(1, RUNS);

% Check if instance file exists
if ~exist(instance_file, 'file')
    error('Instance file not found: %s', instance_file);
end

% Run the optimizer multiple times
for RunCounter = 1:RUNS
    fprintf('Run %d/%d: ', RunCounter, RUNS);

    % Load PORTAL instance from JSON
    PORTAL = load_portal_instance(instance_file, MAX_EVALS, ACCEPTED_ERROR);

    % Set random seed for reproducibility within run
    rng('shuffle');

    % Run DE optimizer
    [best_solution, best_value, PORTAL, history] = ...
        differential_evolution(PORTAL, DE_config);

    % Compute true optimum (minimum of all component offsets beta)
    true_optimum = min(PORTAL.P.beta);

    % Store results
    Error(RunCounter) = abs(best_value - true_optimum);
    AcceptancePoints(RunCounter) = PORTAL.AcceptanceReachPoint;
    FinalValues(RunCounter) = best_value;
    ConvergenceHistory{RunCounter} = cummin(history);

    fprintf('Best = %.6e, Error = %.6e', best_value, Error(RunCounter));
    if isfinite(PORTAL.AcceptanceReachPoint)
        fprintf(', Acceptance at FE = %d', PORTAL.AcceptanceReachPoint);
    end
    fprintf('\n');
end

%% Display results
fprintf('\n');
fprintf('Summary Results\n');
fprintf('===============\n');

% Acceptance statistics
nonInfIndices = isfinite(AcceptancePoints);
if any(nonInfIndices)
    nonInfValues = AcceptancePoints(nonInfIndices);
    fprintf('Average FE to reach acceptance: %.2f (%.2f)\n', ...
        mean(nonInfValues), std(nonInfValues));
    fprintf('Acceptance ratio: %.1f%%\n', ...
        100 * sum(nonInfIndices) / length(AcceptancePoints));
else
    fprintf('No runs reached acceptance threshold\n');
end

% Final error statistics
fprintf('Final error: %.6e (%.6e)\n', mean(Error), std(Error));
fprintf('Best final value found: %.6f\n', min(FinalValues));
fprintf('True optimum value: %.6f\n', true_optimum);

%% Plot average error convergence with acceptance line
% Truncate histories to common length
lens = cellfun(@numel, ConvergenceHistory);
minLen = min(lens);

% Build error matrix: |f(t) - f*|
errData = NaN(RUNS, minLen);
for i = 1:RUNS
    hist = ConvergenceHistory{i}(1:minLen);
    errData(i, :) = abs(hist - true_optimum);
end

% Average over runs
avgErr = mean(errData, 1);

% Plot average error curve
figure('Name', 'Average DE Error Convergence on PORTAL Instance');
h1 = semilogy(1:minLen, avgErr, 'LineWidth', 2, 'Color', 'k', ...
              'DisplayName', 'Average Error');
hold on;

% Plot acceptance threshold as a red dotted line
h2 = yline(ACCEPTED_ERROR, 'r--', 'LineWidth', 1.5, ...
           'DisplayName', 'Acceptance Threshold');

hold off;

xlabel('Function Evaluations');
ylabel('Error |f - f^*|');
title(sprintf('Average Error Convergence - %s', instance_file), ...
      'Interpreter', 'none');

legend('Location', 'best');
grid on;

%% Function to load PORTAL instance from JSON
function PORTAL = load_portal_instance(filename, max_evals, accepted_error)
    % Load PORTAL instance using the standard loader
    PORTAL = load_instance(filename);

    % Add evaluation tracking variables
    PORTAL.max_evals = max_evals;
    PORTAL.accepted_error = accepted_error;
    PORTAL.FE = 0;
    PORTAL.FEhistory = inf(1, max_evals);
    PORTAL.BestFoundResult = inf;
    PORTAL.AcceptanceReachPoint = inf;

    % Compute true optimum (minimum beta value)
    PORTAL.optimum_value = min(PORTAL.P.beta);
end

%% Differential Evolution optimizer
function [best_solution, best_value, PORTAL, convergence_history] = ...
    differential_evolution(PORTAL, config)

    % Extract configuration
    NP = config.PopulationSize;
    F = config.F;
    Cr = config.Cr;
    dim = PORTAL.d;
    bounds = [PORTAL.bounds_min, PORTAL.bounds_max];

    % Initialize population
    population = initialize_population(NP, dim, bounds);

    % Evaluate initial population
    [fitness_values, PORTAL] = evaluate_population(population, PORTAL);

    % Find initial best
    [best_value, best_idx] = min(fitness_values);
    best_solution = population(best_idx, :);

    % Main DE loop
    while PORTAL.FE < PORTAL.max_evals
        % Store current best
        [current_best_val, current_best_idx] = min(fitness_values);
        if current_best_val < best_value
            best_value = current_best_val;
            best_solution = population(current_best_idx, :);
        end

        % Generate donor vectors (mutation)
        donors = mutation_rand_1(population, F);

        % Create trial vectors (crossover)
        trials = binomial_crossover(population, donors, Cr);

        % Boundary correction
        trials = boundary_correction(trials, population, bounds);

        % Evaluate trial vectors
        [trial_fitness, PORTAL] = evaluate_population(trials, PORTAL);

        % Selection
        improved = trial_fitness < fitness_values;
        population(improved, :) = trials(improved, :);
        fitness_values(improved) = trial_fitness(improved);

        % Check termination
        if PORTAL.FE >= PORTAL.max_evals
            break;
        end
    end

    % Final best
    [final_best_val, final_best_idx] = min(fitness_values);
    if final_best_val < best_value
        best_value = final_best_val;
        best_solution = population(final_best_idx, :);
    end

    % Update convergence history
    convergence_history = PORTAL.FEhistory;
end

%% Population initialization
function population = initialize_population(NP, dim, bounds)
    lower = bounds(1);
    upper = bounds(2);
    population = lower + (upper - lower) * rand(NP, dim);
end

%% Mutation: DE/rand/1
function donors = mutation_rand_1(population, F)
    [NP, dim] = size(population);
    donors = zeros(NP, dim);

    for i = 1:NP
        % Select three distinct random individuals
        candidates = setdiff(1:NP, i);
        r = candidates(randperm(length(candidates), 3));

        % DE/rand/1 mutation
        donors(i, :) = population(r(1), :) + ...
                       F * (population(r(2), :) - population(r(3), :));
    end
end

%% Binomial crossover
function trials = binomial_crossover(population, donors, Cr)
    [NP, dim] = size(population);
    trials = population;  % Start with target vectors

    % Ensure at least one parameter comes from donor
    j_rand = randi(dim, NP, 1);
    for i = 1:NP
        trials(i, j_rand(i)) = donors(i, j_rand(i));
    end

    % Probabilistic crossover for other parameters
    crossover_mask = rand(NP, dim) < Cr;
    trials(crossover_mask) = donors(crossover_mask);
end

%% Boundary correction
function corrected = boundary_correction(trials, population, bounds)
    lower = bounds(1);
    upper = bounds(2);
    corrected = trials;

    % Find violations
    too_low = trials < lower;
    too_high = trials > upper;

    % Midpoint correction
    if any(too_low(:))
        midpoint_low = (lower + population) / 2;
        corrected(too_low) = midpoint_low(too_low);
    end

    if any(too_high(:))
        midpoint_high = (upper + population) / 2;
        corrected(too_high) = midpoint_high(too_high);
    end
end

%% Evaluate population using PORTAL fitness function
function [result, PORTAL] = evaluate_population(X, PORTAL)
    [n_samples, ~] = size(X);
    result = NaN(n_samples, 1);

    for jj = 1:n_samples
        % Extract current solution
        x = X(jj, :)';

        % Evaluate using PORTAL fitness function
        f_val = fitness(x, PORTAL);
        result(jj) = f_val;

        % Check evaluation limit
        if PORTAL.FE >= PORTAL.max_evals
            return;
        end

        % Update tracking
        PORTAL.FE = PORTAL.FE + 1;
        PORTAL.FEhistory(PORTAL.FE) = result(jj);

        % Update best found
        if PORTAL.BestFoundResult > result(jj)
            PORTAL.BestFoundResult = result(jj);
        end

        % Check acceptance threshold
        if abs(result(jj) - PORTAL.optimum_value) < PORTAL.accepted_error && ...
           isinf(PORTAL.AcceptanceReachPoint)
            PORTAL.AcceptanceReachPoint = PORTAL.FE;
        end
    end
end
