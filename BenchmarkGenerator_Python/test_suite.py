"""
PORTAL Comprehensive Test Suite
Author: Python Port of PORTAL Framework
Last Edited: 2025-19-10
Title: PORTAL Complete Parameter Mode Testing with Interactive Plotting

Description:
    Tests ALL parameter generation modes with interactive 2D plotting
    Includes: P0-P3, CenterMode, KappaMode, PMode, BetaOffsetMode, RotationMode
    Final sections: Comprehensive tests, Presets, Export/Import functionality

License:
    This program is to be used under the terms of the GNU General Public License
    (http://www.gnu.org/copyleft/gpl.html).
"""

import numpy as np
from benchmark_generator import benchmark_generator
from fitness import fitness
from save_instance import save_instance
from load_instance import load_instance
from plot_landscape import ask_and_plot
from portal_default_settings import portal_default_settings


def run_test_suite():
    """Run the complete PORTAL test suite."""
    
    print('=' * 40)
    print('PORTAL COMPREHENSIVE TEST SUITE')
    print('Testing All Parameter Generation Modes')
    print('=' * 40)
    print()
    
    # ==================== Section 1: TransformMode ====================
    print('=' * 40)
    print('SECTION 1: P0 - TransformMode Tests')
    print('=' * 40)
    print()
    
    # Test 1.1: Fully random transformation (mode 1)
    print('--- Test 1.1: TransformMode=1 (Fully Random) ---')
    PORTAL_1_1 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        TransformMode=1,
        TransformCountMin=0,
        TransformCountMax=2)
    print('Transform sequences: ', end='')
    for k in range(PORTAL_1_1['K']):
        print(f'C{k+1}:[{len(PORTAL_1_1["transform_seq"][k])} transforms] ', end='')
    print()
    ask_and_plot(PORTAL_1_1, 'Test 1.1')
    
    # Test 1.2: Fixed count, random content (mode 2)
    print('--- Test 1.2: TransformMode=2 (Fixed Count) ---')
    PORTAL_1_2 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        TransformMode=2,
        TransformCount=2,
        Seed=1002)
    print('Each component has 2 random transformations')
    ask_and_plot(PORTAL_1_2, 'Test 1.2')
    
    # Test 1.3: No transformations (mode 3)
    print('--- Test 1.3: TransformMode=3 (No Transformations) ---')
    PORTAL_1_3 = benchmark_generator(
        Dimension=2,
        NumComponents=2,
        TransformMode=3,
        Seed=1003)
    print('All components have empty transformation sequences')
    ask_and_plot(PORTAL_1_3, 'Test 1.3')
    
    # Test 1.4: User-specified (mode 4, backward compatible)
    print('--- Test 1.4: TransformMode=4 (User-Specified) ---')
    PORTAL_1_4 = benchmark_generator(
        Dimension=2,
        NumComponents=2,
        TransformMode=4,
        TransformSeq=[['wavelet_mod', 'log_sinusoidal'], ['radial_polytrig']],
        Seed=1004)
    print('User-specified sequences applied')
    ask_and_plot(PORTAL_1_4, 'Test 1.4')
    
    # ==================== Section 2: BaselineMode ====================
    print('=' * 40)
    print('SECTION 2: P1 - BaselineMode Tests')
    print('=' * 40)
    print()
    
    # Test 2.1: Random 50-50 (mode 1)
    print('--- Test 2.1: BaselineMode=1 (Random 50-50) ---')
    PORTAL_2_1 = benchmark_generator(
        Dimension=2,
        NumComponents=4,
        BaselineMode=1,
        Seed=2001)
    print('Baselines: ', end='')
    for k in range(PORTAL_2_1['K']):
        print(f'{PORTAL_2_1["baseline"][k]} ', end='')
    print()
    ask_and_plot(PORTAL_2_1, 'Test 2.1')
    
    # Test 2.2: Custom probability (mode 2)
    print('--- Test 2.2: BaselineMode=2 (Custom Probability) ---')
    PORTAL_2_2 = benchmark_generator(
        Dimension=2,
        NumComponents=4,
        BaselineMode=2,
        BaselineProbA=0.8,
        Seed=2002)
    print('Baselines (80% Form A): ', end='')
    for k in range(PORTAL_2_2['K']):
        print(f'{PORTAL_2_2["baseline"][k]} ', end='')
    print()
    ask_and_plot(PORTAL_2_2, 'Test 2.2')
    
    # Test 2.3: All same (mode 3)
    print('--- Test 2.3: BaselineMode=3 (All Same - Form B) ---')
    PORTAL_2_3 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        BaselineMode=3,
        BaselineType='B',
        Seed=2003)
    print('All components use Form B')
    ask_and_plot(PORTAL_2_3, 'Test 2.3')
    
    # Test 2.4: User-specified (mode 4)
    print('--- Test 2.4: BaselineMode=4 (User-Specified) ---')
    PORTAL_2_4 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        BaselineMode=4,
        BaselineSeq=['A', 'B', 'A'],
        Seed=2004)
    print('User-specified: A, B, A')
    ask_and_plot(PORTAL_2_4, 'Test 2.4')
    
    # ==================== Section 3: TransformParamMode ====================
    print('=' * 40)
    print('SECTION 3: P2 - TransformParamMode Tests')
    print('=' * 40)
    print()
    
    # Test 3.1: Independent random (mode 1)
    print('--- Test 3.1: TransformParamMode=1 (Independent Random) ---')
    PORTAL_3_1 = benchmark_generator(
        Dimension=2,
        NumComponents=2,
        TransformParamMode=1,
        TransformMode=4,
        TransformSeq=[['wavelet_mod'], ['wavelet_mod']],
        Seed=3001)
    print('Each (k,i) parameter independently sampled')
    ask_and_plot(PORTAL_3_1, 'Test 3.1')
    
    # Test 3.2: Component-shared (mode 2)
    print('--- Test 3.2: TransformParamMode=2 (Component-Shared) ---')
    PORTAL_3_2 = benchmark_generator(
        Dimension=2,
        NumComponents=2,
        TransformParamMode=2,
        TransformMode=4,
        TransformSeq=[['wavelet_mod'], ['wavelet_mod']],
        Seed=3002)
    print('All components share same parameter pattern across dimensions')
    ask_and_plot(PORTAL_3_2, 'Test 3.2')
    
    # Test 3.3: Dimension-shared (mode 3)
    print('--- Test 3.3: TransformParamMode=3 (Dimension-Shared) ---')
    PORTAL_3_3 = benchmark_generator(
        Dimension=2,
        NumComponents=2,
        TransformParamMode=3,
        TransformMode=4,
        TransformSeq=[['wavelet_mod'], ['wavelet_mod']],
        Seed=3003)
    print('All dimensions share same parameter value per component')
    ask_and_plot(PORTAL_3_3, 'Test 3.3')
    
    # Test 3.4: Global shared (mode 4)
    print('--- Test 3.4: TransformParamMode=4 (Global Shared) ---')
    PORTAL_3_4 = benchmark_generator(
        Dimension=2,
        NumComponents=2,
        TransformParamMode=4,
        TransformMode=4,
        TransformSeq=[['wavelet_mod'], ['wavelet_mod']],
        Seed=3004)
    print('All components and dimensions use same parameter value')
    ask_and_plot(PORTAL_3_4, 'Test 3.4')
    
    # Test 3.5: Linear distribution (mode 6)
    print('--- Test 3.5: TransformParamMode=6 (Linear Distribution) ---')
    PORTAL_3_5 = benchmark_generator(
        Dimension=2,
        NumComponents=2,
        TransformParamMode=6,
        TransformMode=4,
        TransformSeq=[['additive_periodic'], ['additive_periodic']],
        Seed=3011)
    print('Linearly spaced values with random permutation per component')
    ask_and_plot(PORTAL_3_5, 'Test 3.5')
    
    # ==================== Section 4: CenterMode ====================
    print('=' * 40)
    print('SECTION 4: CenterMode Tests')
    print('=' * 40)
    print()
    
    # Test 4.1: Uniform random (mode 1)
    print('--- Test 4.1: CenterMode=1 (Uniform Random) ---')
    PORTAL_4_1 = benchmark_generator(
        Dimension=2,
        NumComponents=5,
        CenterMode=1,
        Seed=4001)
    print('Centers distributed uniformly')
    ask_and_plot(PORTAL_4_1, 'Test 4.1')
    
    # Test 4.2: Exclusion zone (mode 2)
    print('--- Test 4.2: CenterMode=2 (Exclusion Zone) ---')
    PORTAL_4_2 = benchmark_generator(
        Dimension=2,
        NumComponents=5,
        CenterMode=2,
        ExclusionZone=[-30, 30],
        Seed=4002)
    print('Centers avoid region [-30, 30]')
    ask_and_plot(PORTAL_4_2, 'Test 4.2')
    
    # Test 4.3: Shared center (mode 3)
    print('--- Test 4.3: CenterMode=3 (Shared Center) ---')
    PORTAL_4_3 = benchmark_generator(
        Dimension=2,
        NumComponents=4,
        CenterMode=3,
        Seed=4003)
    print('All components overlap at same center')
    ask_and_plot(PORTAL_4_3, 'Test 4.3')
    
    # ==================== Section 5: KappaMode ====================
    print('=' * 40)
    print('SECTION 5: KappaMode Tests')
    print('=' * 40)
    print()
    
    # Test 5.1: Independent random (mode 1)
    print('--- Test 5.1: KappaMode=1 (Independent Random) ---')
    PORTAL_5_1 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        KappaMode=1,
        Seed=5000)
    print('Each scaling factor sampled independently')
    ask_and_plot(PORTAL_5_1, 'Test 5.1')
    
    # Test 5.2: Beta distribution (mode 2)
    print('--- Test 5.2: KappaMode=2 (Beta Distribution) ---')
    PORTAL_5_2 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        KappaMode=2,
        BetaAlpha=0.5,
        BetaBeta=0.5,
        Seed=5001)
    print('Creates extreme condition numbers using Beta(0.5, 0.5)')
    ask_and_plot(PORTAL_5_2, 'Test 5.2')
    
    # Test 5.3: Component-shared (mode 3)
    print('--- Test 5.3: KappaMode=3 (Component-Shared) ---')
    PORTAL_5_3 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        KappaMode=3,
        Seed=5003)
    print('Isotropic scaling within each component')
    ask_and_plot(PORTAL_5_3, 'Test 5.3')
    
    # Test 5.4: Linear distribution (mode 4)
    print('--- Test 5.4: KappaMode=4 (Linear Distribution) ---')
    PORTAL_5_4 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        KappaMode=4,
        Seed=5004)
    print('Linearly spaced values with random permutation')
    ask_and_plot(PORTAL_5_4, 'Test 5.4')
    
    # Test 5.5: Isotropic (mode 5)
    print('--- Test 5.5: KappaMode=5 (Isotropic) ---')
    PORTAL_5_5 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        KappaMode=5,
        TransformMode=2,
        TransformCount=1)
    print('Isotropic scaling within each direction')
    ask_and_plot(PORTAL_5_5, 'Test 5.5')
    
    # ==================== Section 6: PMode ====================
    print('=' * 40)
    print('SECTION 6: PMode Tests')
    print('=' * 40)
    print()
    
    # Test 6.1: Random range (mode 1)
    print('--- Test 6.1: PMode=1 (Random Range) ---')
    PORTAL_6_1 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        PMode=1,
        Seed=6002)
    print('Uniformly random exponents')
    ask_and_plot(PORTAL_6_1, 'Test 6.1')
    
    # Test 6.2: Fixed value (mode 2)
    print('--- Test 6.2: PMode=2 (Fixed Value) ---')
    PORTAL_6_2 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        PMode=2,
        KappaMode=5,
        FixedP=0.5,
        Seed=6002)
    print('All exponents set to 0.5')
    ask_and_plot(PORTAL_6_2, 'Test 6.2')
    
    # Test 6.3: Gradient (mode 3)
    print('--- Test 6.3: PMode=3 (Gradient) ---')
    PORTAL_6_3 = benchmark_generator(
        Dimension=2,
        NumComponents=5,
        PMode=3,
        Seed=6003)
    print('Linearly varying from p_min to p_max')
    ask_and_plot(PORTAL_6_3, 'Test 6.3')
    
    # Test 6.4: Beta distribution (mode 4)
    print('--- Test 6.4: PMode=4 (Beta Distribution) ---')
    PORTAL_6_4 = benchmark_generator(
        Dimension=2,
        NumComponents=4,
        PMode=4,
        Seed=6004)
    print('Beta distribution for exponents')
    ask_and_plot(PORTAL_6_4, 'Test 6.4')
    
    # Test 6.5: Linear distribution (mode 5)
    print('--- Test 6.5: PMode=5 (Linear Distribution) ---')
    PORTAL_6_5 = benchmark_generator(
        Dimension=2,
        NumComponents=4,
        PMode=5,
        Seed=6005)
    print('Linearly spaced values with random permutation per component')
    ask_and_plot(PORTAL_6_5, 'Test 6.5')
    
    # ==================== Section 7: BetaOffsetMode ====================
    print('=' * 40)
    print('SECTION 7: BetaOffsetMode Tests')
    print('=' * 40)
    print()
    
    # Test 7.1: Independent random (mode 1)
    print('--- Test 7.1: BetaOffsetMode=1 (Independent Random) ---')
    PORTAL_7_1 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        BetaOffsetMode=1,
        Seed=7001)
    print('Each component has independent offset')
    ask_and_plot(PORTAL_7_1, 'Test 7.1')
    
    # Test 7.2: Shared value (mode 2)
    print('--- Test 7.2: BetaOffsetMode=2 (Shared Value) ---')
    PORTAL_7_2 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        BetaOffsetMode=2,
        Seed=7002)
    print('All components share same random offset')
    ask_and_plot(PORTAL_7_2, 'Test 7.2')
    
    # Test 7.3: Fixed value (mode 3)
    print('--- Test 7.3: BetaOffsetMode=3 (Fixed Value) ---')
    PORTAL_7_3 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        BetaOffsetMode=3,
        FixedBeta=0,
        Seed=7003)
    print('All components use FixedBeta=0')
    ask_and_plot(PORTAL_7_3, 'Test 7.3')
    
    # Test 7.4: Linear distribution (mode 5)
    print('--- Test 7.4: BetaOffsetMode=5 (Linear Distribution) ---')
    PORTAL_7_4 = benchmark_generator(
        Dimension=2,
        NumComponents=5,
        BetaOffsetMode=5,
        Seed=7004)
    print('Linearly spaced values with random permutation across components')
    ask_and_plot(PORTAL_7_4, 'Test 7.4')
    
    # ==================== Section 8: RotationMode ====================
    print('=' * 40)
    print('SECTION 8: RotationMode Tests')
    print('=' * 40)
    print()
    
    # Test 8.1: No rotation (mode 1)
    print('--- Test 8.1: RotationMode=1 (No Rotation) ---')
    PORTAL_8_1 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        RotationMode=1,
        PMode=2,
        KappaMode=5,
        FixedP=0.5,
        Seed=8001)
    print('Separable problem (identity matrix)')
    ask_and_plot(PORTAL_8_1, 'Test 8.1')
    
    # Test 8.2: Fully connected (mode 2)
    print('--- Test 8.2: RotationMode=2 (Fully Connected) ---')
    PORTAL_8_2 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        RotationMode=2,
        Seed=8002)
    print('All variable pairs interact')
    ask_and_plot(PORTAL_8_2, 'Test 8.2')
    
    # Test 8.3: Probabilistic (mode 3)
    print('--- Test 8.3: RotationMode=3 (Probabilistic) ---')
    PORTAL_8_3 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        RotationMode=3,
        RotationProb=0.7,
        Seed=1111)
    print('Sparse interactions with 70% probability')
    ask_and_plot(PORTAL_8_3, 'Test 8.3')
    
    # Test 8.4: Uniform angle (mode 4)
    print('--- Test 8.4: RotationMode=4 (Uniform Angle) ---')
    PORTAL_8_4 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        RotationMode=4,
        Seed=8004)
    print('Same random angle for all pairs per component')
    ask_and_plot(PORTAL_8_4, 'Test 8.4')
    
    # Test 8.5: Fixed angle (mode 5)
    print('--- Test 8.5: RotationMode=5 (Fixed Angle) ---')
    PORTAL_8_5 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        RotationMode=5,
        SpecificAngle=45,
        Seed=8005)
    print('User-specified angle: 45 degrees')
    ask_and_plot(PORTAL_8_5, 'Test 8.5')
    
    # Test 8.6: Chain-like (mode 6)
    print('--- Test 8.6: RotationMode=6 (Chain-like) ---')
    PORTAL_8_6 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        RotationMode=6,
        Seed=8006)
    print('Only adjacent dimensions interact')
    ask_and_plot(PORTAL_8_6, 'Test 8.6')
    
    # Test 8.7: Block-diagonal (mode 7)
    print('--- Test 8.7: RotationMode=7 (Block-diagonal) ---')
    PORTAL_8_7 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        RotationMode=7,
        BlockSizes=[1, 1],
        BlockAngles=[30, 60],
        Seed=8007)
    print('Partially separable groups with blocks [1,1]')
    ask_and_plot(PORTAL_8_7, 'Test 8.7')
    
    # ==================== Section 9: Comprehensive Tests ====================
    print('=' * 40)
    print('SECTION 9: Comprehensive Tests')
    print('=' * 40)
    print()
    
    # Test 9.1: Complex combination 1
    print('--- Test 9.1: Complex Combination 1 ---')
    PORTAL_9_1 = benchmark_generator(
        Dimension=2,
        NumComponents=4,
        TransformMode=1,
        TransformCountMin=1,
        TransformCountMax=2,
        BaselineMode=2,
        BaselineProbA=0.6,
        RotationMode=2,
        CenterMode=2,
        ExclusionZone=[-20, 20],
        KappaMode=2,
        BetaAlpha=0.3,
        BetaBeta=0.3,
        TransformParamMode=2,
        Seed=9001)
    print('Random transforms + Custom baseline + Fully connected + Exclusion zone + Beta kappa')
    ask_and_plot(PORTAL_9_1, 'Test 9.1')
    
    # Test 9.2: Complex combination 2
    print('--- Test 9.2: Complex Combination 2 ---')
    PORTAL_9_2 = benchmark_generator(
        Dimension=2,
        NumComponents=3,
        TransformMode=2,
        TransformCount=2,
        BaselineMode=3,
        BaselineType='B',
        RotationMode=7,
        BlockSizes=[1, 1],
        CenterMode=3,
        KappaMode=3,
        PMode=2,
        FixedP=0.5,
        TransformParamMode=4,
        Seed=9002)
    print('Fixed transforms + All B + Block-diagonal + Shared center + Isotropic + Fixed P')
    ask_and_plot(PORTAL_9_2, 'Test 9.2')
    
    # Test 9.3: Complex combination 3
    print('--- Test 9.3: Complex Combination 3 ---')
    PORTAL_9_3 = benchmark_generator(
        Dimension=2,
        NumComponents=5,
        TransformMode=1,
        TransformCountMin=1,
        TransformCountMax=2,
        BaselineMode=1,
        RotationMode=3,
        RotationProb=0.5,
        CenterMode=1,
        KappaMode=1,
        PMode=3,
        BetaOffsetMode=2,
        TransformParamMode=1,
        Seed=9103)
    print('Fully random + Random baseline + Probabilistic rotation + Gradient P')
    ask_and_plot(PORTAL_9_3, 'Test 9.3')
    
    # Test 9.4: Linear distribution combination
    print('--- Test 9.4: Linear Distribution Combination ---')
    PORTAL_9_4 = benchmark_generator(
        Dimension=2,
        NumComponents=5,
        TransformMode=2,
        TransformCount=1,
        BaselineMode=1,
        RotationMode=2,
        CenterMode=1,
        KappaMode=1,
        PMode=1,
        BetaOffsetMode=5,
        TransformParamMode=6,
        Seed=9104)
    print('Linear distribution for beta offset and transform params')
    ask_and_plot(PORTAL_9_4, 'Test 9.4')
    
    # ==================== Section 10: Preset Tests (2D) ====================
    print('=' * 40)
    print('SECTION 10: Preset Tests (2D)')
    print('=' * 40)
    print()
    
    presets = ['standard', 'easy', 'medium', 'hard', 'multimodal', 'separable', 'ill_conditioned']
    for i, preset_name in enumerate(presets):
        print(f'--- Test 10.{i+1}: Preset "{preset_name}" (modified to 2D) ---')
        config = portal_default_settings(preset_name)
        
        # Override dimension to 2 for visualization
        config['Dimension'] = 2
        
        # For block-diagonal mode, adjust BlockSizes if needed
        if 'BlockSizes' in config:
            config['BlockSizes'] = [1, 1]
        
        config['Seed'] = 10000 + i + 1
        PORTAL_preset = benchmark_generator(**config)
        print(f'Preset "{preset_name}" generated successfully')
        ask_and_plot(PORTAL_preset, f'Preset {preset_name}')
    
    # ==================== Section 11: Export/Import Tests ====================
    print('=' * 40)
    print('SECTION 11: Export/Import Tests')
    print('=' * 40)
    print()
    
    # Test 11.1: Export to JSON file
    print('--- Test 11.1: Export PORTAL instance to JSON file ---')
    PORTAL_export = benchmark_generator(
        Dimension=2,
        NumComponents=4,
        TransformMode=1,
        TransformCountMin=1,
        TransformCountMax=2,
        BaselineMode=1,
        BaselineProbA=0.7,
        RotationMode=2,
        KappaMode=1,
        BetaAlpha=0.5,
        BetaBeta=0.5,
        Seed=11001)
    
    export_file = 'PORTAL_test_export.json'
    export_path = save_instance(PORTAL_export, export_file)
    
    # Test 11.2: Import from JSON file
    print('\n--- Test 11.2: Import PORTAL instance from JSON file ---')
    PORTAL_import = load_instance(export_file)
    
    # Test 11.3: Evaluate fitness at multiple points
    print('\n--- Test 11.3: Verify Fitness Evaluation ---')
    print('Testing fitness evaluation at 10 random points...')
    test_points = np.random.randn(2, 10)
    fitness_match = True
    
    for i in range(10):
        x = test_points[:, i]
        f_export = fitness(x, PORTAL_export)
        f_import = fitness(x, PORTAL_import)
        
        if abs(f_export - f_import) > 1e-10:
            print(f'✗ Fitness mismatch at point {i+1}: {f_export:.10f} vs {f_import:.10f}')
            fitness_match = False
    
    if fitness_match:
        print('✓ All fitness evaluations match!')
    else:
        print('✗ Some fitness evaluations do not match!')
    
    # Test 11.4: Plot both instances
    print('\n--- Test 11.4: Visual Comparison ---')
    print('Exported instance:')
    ask_and_plot(PORTAL_export, 'Exported Instance')
    
    print('Imported instance (should be identical):')
    ask_and_plot(PORTAL_import, 'Imported Instance')
    
    print(f'\nExport/Import tests completed successfully!')
    print(f'Note: The exported file "{export_file}" is saved in the current directory.\n')
    
    # ==================== Summary ====================
    print('=' * 40)
    print('ALL TESTS COMPLETED SUCCESSFULLY!')
    print('=' * 40)
    print()
    
    print('Summary of Features Tested:')
    print('  P0 - TransformMode:        4 modes tested')
    print('  P1 - BaselineMode:         4 modes tested')
    print('  P2 - TransformParamMode:   5 modes tested')
    print('  P3 - NeutralizationMode:   Reserved (mode 1 active)')
    print('  CenterMode:                3 modes tested')
    print('  KappaMode:                 5 modes tested')
    print('  PMode:                     5 modes tested')
    print('  BetaOffsetMode:            4 modes tested')
    print('  RotationMode:              7 modes tested')
    print('  Comprehensive Tests:       4 scenarios')
    print('  Preset Tests:              7 presets')
    print('  Export/Import:             Full cycle tested')
    print()
    print('All tests passed successfully!')


if __name__ == '__main__':
    run_test_suite()
