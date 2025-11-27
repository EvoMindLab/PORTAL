"""
Author: Delaram Yazdani, Danial Yazdani, Mai Peng
Email: delaram.yazdani@yahoo.com
       danial.yazdani@gmail.com
       pengmai1998@gmail.com
Last Edited: 2025-11-28

Batch Test Script for JSON Export/Import Validation
Creates 20 diverse PORTAL instances and verifies export/import consistency
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Generator_Python_Script import benchmark_generator
from Utils.save_instance import save_instance
from Utils.load_instance import load_instance
from Utils.fitness import fitness


def test_export_import():
    """
    Main test function that creates diverse PORTAL instances,
    saves them to JSON, loads them back, and validates consistency.
    """
    print('=' * 60)
    print('  PORTAL Export/Import Validation Test Suite (Python)')
    print('=' * 60)
    print()

    # Test Configuration
    num_tests = 20
    test_results = []
    all_passed = True

    # Random seed for reproducibility
    np.random.seed(42)

    # Generate diverse test cases
    print(f'Generating {num_tests} diverse test configurations...\n')
    test_configs = []

    for i in range(num_tests):
        # Vary dimensions (2 to 10) - keeping d=2 like MATLAB version
        d = np.random.randint(2, 3)  # 2 to 2 (matching MATLAB)

        # Vary number of components (1 to 5)
        K = np.random.randint(1, 6)

        # Vary transform mode
        transform_mode = np.random.randint(1, 4)
        if transform_mode == 1:
            transform_count_min = np.random.randint(0, 3)
            transform_count_max = np.random.randint(transform_count_min, 4)
        else:
            transform_count = np.random.randint(0, 4)

        # Vary rotation mode
        rotation_mode = np.random.randint(1, 7)

        # Vary baseline mode
        baseline_mode = np.random.randint(1, 4)

        # Random seed for this instance
        seed = np.random.randint(1, 100001)

        # Store configuration
        config = {
            'd': d,
            'K': K,
            'seed': seed,
            'transform_mode': transform_mode,
            'rotation_mode': rotation_mode,
            'baseline_mode': baseline_mode
        }

        if transform_mode == 1:
            config['transform_count_min'] = transform_count_min
            config['transform_count_max'] = transform_count_max
        else:
            config['transform_count'] = transform_count

        test_configs.append(config)

    # Run Tests
    print(f'Running {num_tests} test cases...\n')

    for i, config in enumerate(test_configs, 1):
        print(f'[{i:2d}/{num_tests:2d}] Testing: d={config["d"]:2d}, K={config["K"]}, '
              f'Seed={config["seed"]:6d}, TransformMode={config["transform_mode"]}, '
              f'RotationMode={config["rotation_mode"]}')

        test_result = {'config': config, 'passed': False, 'errors': []}

        try:
            # Create instance
            kwargs = {
                'Dimension': config['d'],
                'NumComponents': config['K'],
                'Seed': config['seed'],
                'TransformMode': config['transform_mode'],
                'RotationMode': config['rotation_mode'],
                'BaselineMode': config['baseline_mode']
            }

            if config['transform_mode'] == 1:
                kwargs['TransformCountMin'] = config['transform_count_min']
                kwargs['TransformCountMax'] = config['transform_count_max']
            else:
                kwargs['TransformCount'] = config['transform_count']

            PORTAL_orig = benchmark_generator(**kwargs)

            # Save to JSON
            filename = f'test_batch_{i}.json'
            filepath = save_instance(PORTAL_orig, filename)

            # Load from JSON
            PORTAL_loaded = load_instance(filename)

            # Validate Key Fields
            errors = []

            # Check dimensions
            if PORTAL_orig['d'] != PORTAL_loaded['d']:
                errors.append(f'Dimension mismatch: {PORTAL_orig["d"]} != {PORTAL_loaded["d"]}')

            if PORTAL_orig['K'] != PORTAL_loaded['K']:
                errors.append(f'Component count mismatch: {PORTAL_orig["K"]} != {PORTAL_loaded["K"]}')

            # Check rotation matrices (most critical)
            if 'R' in PORTAL_orig['P'] and 'R' in PORTAL_loaded['P']:
                R_orig = np.array(PORTAL_orig['P']['R'])
                R_loaded = np.array(PORTAL_loaded['P']['R'])
                max_R_diff = np.max(np.abs(R_orig - R_loaded))
                if max_R_diff > 1e-10:
                    errors.append(f'Rotation matrix mismatch: max_diff={max_R_diff:.2e}')

            # Check rotation angles
            if 'Psi_deg' in PORTAL_orig['P'] and 'Psi_deg' in PORTAL_loaded['P']:
                Psi_orig = np.array(PORTAL_orig['P']['Psi_deg'])
                Psi_loaded = np.array(PORTAL_loaded['P']['Psi_deg'])
                max_Psi_diff = np.max(np.abs(Psi_orig - Psi_loaded))
                if max_Psi_diff > 1e-10:
                    errors.append(f'Rotation angle mismatch: max_diff={max_Psi_diff:.2e}')

            # Check key parameters
            params_to_check = ['c', 'beta', 'kappa_plus', 'kappa_minus', 'p_plus', 'p_minus']
            for param in params_to_check:
                if param in PORTAL_orig['P'] and param in PORTAL_loaded['P']:
                    orig_val = np.array(PORTAL_orig['P'][param])
                    loaded_val = np.array(PORTAL_loaded['P'][param])
                    max_diff = np.max(np.abs(orig_val - loaded_val))
                    if max_diff > 1e-10:
                        errors.append(f'{param} mismatch: max_diff={max_diff:.2e}')

            # Test fitness evaluation
            num_test_points = 5
            lower_bound = -100
            upper_bound = 100
            test_points = lower_bound + (upper_bound - lower_bound) * np.random.rand(config['d'], num_test_points)

            max_fitness_diff = 0.0
            for j in range(num_test_points):
                x = test_points[:, j]
                f_orig = fitness(x, PORTAL_orig)
                f_loaded = fitness(x, PORTAL_loaded)
                diff = abs(f_orig - f_loaded)
                max_fitness_diff = max(max_fitness_diff, diff)

            if max_fitness_diff > 1e-9:
                errors.append(f'Fitness evaluation mismatch: max_diff={max_fitness_diff:.2e}')

            # Record results
            test_result['passed'] = len(errors) == 0
            test_result['errors'] = errors
            test_result['max_fitness_diff'] = max_fitness_diff

            if test_result['passed']:
                print(f'        Result: ✓ PASSED (max_fitness_diff={max_fitness_diff:.2e})')
            else:
                print(f'        Result: ✗ FAILED')
                for error in errors:
                    print(f'          Error: {error}')
                all_passed = False

            # Clean up (optional - commented out to keep test files)
            # if os.path.exists(filepath):
            #     os.remove(filepath)

        except Exception as e:
            print(f'        Result: ✗ EXCEPTION: {str(e)}')
            test_result['passed'] = False
            test_result['exception'] = str(e)
            all_passed = False

        test_results.append(test_result)
        print()

    # Summary
    print('=' * 60)
    print('  Test Summary')
    print('=' * 60)

    num_passed = sum(1 for r in test_results if r['passed'])
    num_failed = num_tests - num_passed

    print(f'Total Tests:  {num_tests}')
    print(f'Passed:       {num_passed} ({100*num_passed/num_tests:.1f}%)')
    print(f'Failed:       {num_failed} ({100*num_failed/num_tests:.1f}%)')

    if all_passed:
        print('\n✓✓✓ ALL TESTS PASSED! ✓✓✓')
    else:
        print('\n✗✗✗ SOME TESTS FAILED ✗✗✗\n')
        print('Failed test cases:')
        for i, result in enumerate(test_results, 1):
            if not result['passed']:
                config = result['config']
                print(f'  Test {i:2d}: d={config["d"]:2d}, K={config["K"]}, Seed={config["seed"]:6d}')
                if 'errors' in result:
                    for error in result['errors']:
                        print(f'    - {error}')
                if 'exception' in result:
                    print(f'    - Exception: {result["exception"]}')

    print('\nTest completed.')
    
    return all_passed, test_results


if __name__ == '__main__':
    all_passed, results = test_export_import()
    sys.exit(0 if all_passed else 1)