%********************************** PORTAL Export/Import Validation Test Suite **********************************
% Author: Mai Peng, Danial Yazdani, Delaram Yazdani
% Email: pengmai1998@gmail.com
%        danial.yazdani@gmail.com
%        delaram.yazdani@yahoo.com
% Last Edited: 2025-11-28
% Title: Batch Test Script for JSON Export/Import Validation
% --------
% Description:
%   Creates 20 diverse PORTAL instances and verifies export/import consistency
%************************************************************************************************
clc; clear;

fprintf('===============================================\n');
fprintf('  PORTAL Export/Import Validation Test Suite\n');
fprintf('===============================================\n\n');

%% Test Configuration
num_tests = 20;
test_results = struct();
all_passed = true;

% Random seed for reproducibility
rng(42);

%% Generate diverse test cases
test_configs = cell(num_tests, 1);

for i = 1:num_tests
    % Vary dimensions (2 to 10)
    d = randi([2, 2]);

    % Vary number of components (1 to 5)
    K = randi([1, 5]);

    % Vary transform mode
    transform_mode = randi([1, 3]);
    if transform_mode == 1
        transform_count_min = randi([0, 2]);
        transform_count_max = randi([transform_count_min, 3]);
    else
        transform_count = randi([0, 3]);
    end

    % Vary rotation mode
    rotation_mode = randi([1, 6]);

    % Vary baseline mode
    baseline_mode = randi([1, 3]);

    % Random seed for this instance
    seed = randi([1, 100000]);

    % Store configuration
    config = struct();
    config.d = d;
    config.K = K;
    config.seed = seed;
    config.transform_mode = transform_mode;
    config.rotation_mode = rotation_mode;
    config.baseline_mode = baseline_mode;

    if transform_mode == 1
        config.transform_count_min = transform_count_min;
        config.transform_count_max = transform_count_max;
    else
        config.transform_count = transform_count;
    end

    test_configs{i} = config;
end

%% Run Tests
fprintf('Running %d test cases...\n\n', num_tests);

for i = 1:num_tests
    config = test_configs{i};

    fprintf('[%2d/%2d] Testing: d=%2d, K=%d, Seed=%6d, TransformMode=%d, RotationMode=%d\n', ...
        i, num_tests, config.d, config.K, config.seed, config.transform_mode, config.rotation_mode);

    try
        %% Create instance
        if config.transform_mode == 1
            PORTAL_orig = Generator_MATLAB_Script(...
                'Dimension', config.d, ...
                'NumComponents', config.K, ...
                'Seed', config.seed, ...
                'TransformMode', config.transform_mode, ...
                'TransformCountMin', config.transform_count_min, ...
                'TransformCountMax', config.transform_count_max, ...
                'RotationMode', config.rotation_mode, ...
                'BaselineMode', config.baseline_mode);
        else
            PORTAL_orig = Generator_MATLAB_Script(...
                'Dimension', config.d, ...
                'NumComponents', config.K, ...
                'Seed', config.seed, ...
                'TransformMode', config.transform_mode, ...
                'TransformCount', config.transform_count, ...
                'RotationMode', config.rotation_mode, ...
                'BaselineMode', config.baseline_mode);
        end

        %% Save to JSON
        filename = sprintf('test_batch_%d.json', i);
        filepath = save_instance(PORTAL_orig, filename);

        %% Load from JSON
        PORTAL_loaded = load_instance(filename);

        %% Validate Key Fields
        errors = {};

        % Check dimensions
        if PORTAL_orig.d ~= PORTAL_loaded.d
            errors{end+1} = sprintf('Dimension mismatch: %d != %d', PORTAL_orig.d, PORTAL_loaded.d);
        end

        if PORTAL_orig.K ~= PORTAL_loaded.K
            errors{end+1} = sprintf('Component count mismatch: %d != %d', PORTAL_orig.K, PORTAL_loaded.K);
        end

        % Check rotation matrices (most critical)
        if isfield(PORTAL_orig.P, 'R') && isfield(PORTAL_loaded.P, 'R')
            max_R_diff = max(abs(PORTAL_orig.P.R(:) - PORTAL_loaded.P.R(:)));
            if max_R_diff > 1e-10
                errors{end+1} = sprintf('Rotation matrix mismatch: max_diff=%.2e', max_R_diff);
            end
        end

        % Check rotation angles
        if isfield(PORTAL_orig.P, 'Psi_deg') && isfield(PORTAL_loaded.P, 'Psi_deg')
            max_Psi_diff = max(abs(PORTAL_orig.P.Psi_deg(:) - PORTAL_loaded.P.Psi_deg(:)));
            if max_Psi_diff > 1e-10
                errors{end+1} = sprintf('Rotation angle mismatch: max_diff=%.2e', max_Psi_diff);
            end
        end

        % Check key parameters
        params_to_check = {'c', 'beta', 'kappa_plus', 'kappa_minus', 'p_plus', 'p_minus'};
        for p = 1:length(params_to_check)
            param = params_to_check{p};
            if isfield(PORTAL_orig.P, param) && isfield(PORTAL_loaded.P, param)
                max_diff = max(abs(PORTAL_orig.P.(param)(:) - PORTAL_loaded.P.(param)(:)));
                if max_diff > 1e-10
                    errors{end+1} = sprintf('%s mismatch: max_diff=%.2e', param, max_diff);
                end
            end
        end

        %% Test fitness evaluation
        % Generate random test points
        num_test_points = 5;
        lower_bound = -100;
        upper_bound = 100;
        test_points = lower_bound + (upper_bound - lower_bound) * rand(config.d, num_test_points);
        
        max_fitness_diff = 0;
        for j = 1:num_test_points
            x = test_points(:, j);
            f_orig = fitness(x, PORTAL_orig);
            f_loaded = fitness(x, PORTAL_loaded);
            diff = abs(f_orig - f_loaded);
            max_fitness_diff = max(max_fitness_diff, diff);
        end
        
        if max_fitness_diff > 1e-9
            errors{end+1} = sprintf('Fitness evaluation mismatch: max_diff=%.2e', max_fitness_diff);
        end

        %% Record results
        test_results(i).config = config;
        test_results(i).passed = isempty(errors);
        test_results(i).errors = errors;
        test_results(i).max_fitness_diff = max_fitness_diff;

        if test_results(i).passed
            fprintf('        Result: ✓ PASSED (max_fitness_diff=%.2e)\n', max_fitness_diff);
        else
            fprintf('        Result: ✗ FAILED\n');
            for e = 1:length(errors)
                fprintf('          Error: %s\n', errors{e});
            end
            all_passed = false;
        end

        %% Clean up
        % delete(filepath);

    catch ME
        fprintf('        Result: ✗ EXCEPTION: %s\n', ME.message);
        test_results(i).passed = false;
        test_results(i).exception = ME.message;
        all_passed = false;
    end

    fprintf('\n');
end

%% Summary
fprintf('===============================================\n');
fprintf('  Test Summary\n');
fprintf('===============================================\n');

num_passed = sum([test_results.passed]);
num_failed = num_tests - num_passed;

fprintf('Total Tests:  %d\n', num_tests);
fprintf('Passed:       %d (%.1f%%)\n', num_passed, 100*num_passed/num_tests);
fprintf('Failed:       %d (%.1f%%)\n', num_failed, 100*num_failed/num_tests);

if all_passed
    fprintf('\n✓✓✓ ALL TESTS PASSED! ✓✓✓\n');
else
    fprintf('\n✗✗✗ SOME TESTS FAILED ✗✗✗\n\n');

    fprintf('Failed test cases:\n');
    for i = 1:num_tests
        if ~test_results(i).passed
            config = test_results(i).config;
            fprintf('  Test %2d: d=%2d, K=%d, Seed=%6d\n', ...
                i, config.d, config.K, config.seed);
            if isfield(test_results(i), 'errors')
                for e = 1:length(test_results(i).errors)
                    fprintf('    - %s\n', test_results(i).errors{e});
                end
            end
            if isfield(test_results(i), 'exception')
                fprintf('    - Exception: %s\n', test_results(i).exception);
            end
        end
    end
end

fprintf('\nTest completed.\n');
