"""
********************************** PORTAL Default Settings **********************************
Author: Delaram Yazdani, Danial Yazdani, Mai Peng
Email: delaram.yazdani@yahoo.com
       danial.yazdani@gmail.com
       pengmai1998@gmail.com
Last Edited: 2025-11-28
Title: Centralized Configuration Management for PORTAL Benchmarks

Description:
    Provides predefined configuration presets for PORTAL benchmark generation
    Inspired by GNBG's DefaultSettings architecture
    Supports different difficulty levels and testing scenarios

Usage:
    config = portal_default_settings('easy')
    portal = benchmark_generator(**config)

License:
    This program is to be used under the terms of the GNU General Public License
    (http://www.gnu.org/copyleft/gpl.html).
************************************************************************************************
"""

from typing import Dict, Any


def portal_default_settings(preset: str = 'standard') -> Dict[str, Any]:
    """
    Returns configuration dictionary for benchmark_generator.
    
    Available presets:
        'standard'        - Default behavior (mode=1 for all parameters)
        'easy'            - Simple landscape: separable, low conditioning
        'medium'          - Moderate difficulty: partial separability, medium conditioning
        'hard'            - Challenging: fully connected, high conditioning, exclusion zones
        'multimodal'      - Multiple overlapping components with shared centers
        'separable'       - Chain-like interactions for decomposition testing
        'ill_conditioned' - Extreme condition numbers using Beta distribution
    
    Args:
        preset: Name of the preset configuration
        
    Returns:
        Configuration dictionary for benchmark_generator
    """
    preset = preset.lower()
    
    if preset == 'standard':
        # Default behavior - mode=1 for most, user-specified transforms
        config = {
            'Dimension': 30,
            'NumComponents': 5,
            'TransformMode': 4,  # User-specified (backward compatible)
            'TransformSeq': [['additive_periodic']],
            'BaselineMode': 1,  # Random 50-50
            'RotationMode': 2,  # Fully connected
            'CenterMode': 1,  # Uniform random
            'KappaMode': 1,  # Independent random
            'PMode': 1,  # Random range
            'BetaOffsetMode': 1,  # Independent random
            'TransformParamMode': 1,  # Independent random
            'NeutralizationMode': 1  # Fixed (current)
        }
    
    elif preset == 'easy':
        # Simple landscape for algorithm development
        config = {
            'Dimension': 10,
            'NumComponents': 2,
            'TransformMode': 2,  # Fixed count
            'TransformCount': 1,  # 1 transformation per component
            'BaselineMode': 1,  # Random 50-50
            'RotationMode': 1,  # No rotation (separable)
            'CenterMode': 1,  # Uniform random
            'KappaMode': 3,  # Component-shared (isotropic)
            'PMode': 2,  # Fixed exponent
            'FixedP': 1.0,
            'BetaOffsetMode': 3,  # Fixed beta
            'FixedBeta': 0,
            'TransformParamMode': 1,  # Independent random
            'NeutralizationMode': 1  # Fixed
        }
    
    elif preset == 'medium':
        # Moderate difficulty with partial separability
        config = {
            'Dimension': 30,
            'NumComponents': 5,
            'TransformMode': 1,  # Fully random
            'TransformCountMin': 1,
            'TransformCountMax': 2,
            'BaselineMode': 1,  # Random 50-50
            'RotationMode': 7,  # Block-diagonal (partially separable)
            'BlockSizes': [10, 10, 10],
            'BlockAngles': [45, 90, 135],
            'CenterMode': 1,  # Uniform random
            'KappaMode': 1,  # Independent random
            'PMode': 1,  # Random range
            'BetaOffsetMode': 1,  # Independent random
            'TransformParamMode': 1,  # Independent random
            'NeutralizationMode': 1  # Fixed
        }
    
    elif preset == 'hard':
        # Challenging landscape with high conditioning
        config = {
            'Dimension': 50,
            'NumComponents': 10,
            'TransformMode': 1,  # Fully random
            'TransformCountMin': 2,
            'TransformCountMax': 3,
            'BaselineMode': 2,  # Custom probability
            'BaselineProbA': 0.7,  # 70% Form A
            'RotationMode': 2,  # Fully connected
            'CenterMode': 2,  # Exclusion zone
            'ExclusionZone': [-30, 30],
            'KappaMode': 2,  # Beta distribution
            'BetaAlpha': 0.2,
            'BetaBeta': 0.2,
            'PMode': 3,  # Gradient
            'BetaOffsetMode': 1,  # Independent random
            'TransformParamMode': 1,  # Independent random
            'NeutralizationMode': 1  # Fixed
        }
    
    elif preset == 'multimodal':
        # Multiple overlapping components
        config = {
            'Dimension': 20,
            'NumComponents': 8,
            'TransformMode': 2,  # Fixed count
            'TransformCount': 2,  # 2 transformations per component
            'BaselineMode': 3,  # All same
            'BaselineType': 'B',  # All use Form B
            'RotationMode': 2,  # Fully connected
            'CenterMode': 1,  # Independent random
            'KappaMode': 1,  # Independent random
            'PMode': 1,  # Random range
            'BetaOffsetMode': 2,  # Shared value
            'TransformParamMode': 2,  # Component-shared
            'NeutralizationMode': 1  # Fixed
        }
    
    elif preset == 'separable':
        # Chain-like structure for decomposition testing
        config = {
            'Dimension': 40,
            'NumComponents': 5,
            'TransformMode': 3,  # No transformations
            'BaselineMode': 3,  # All same
            'BaselineType': 'A',  # All use Form A (separable)
            'RotationMode': 6,  # Chain-like
            'CenterMode': 1,  # Uniform random
            'KappaMode': 4,  # Linear distribution
            'PMode': 2,  # Fixed exponent
            'FixedP': 0.5,
            'BetaOffsetMode': 1,  # Independent random
            'TransformParamMode': 3,  # Dimension-shared
            'NeutralizationMode': 1  # Fixed
        }
    
    elif preset == 'ill_conditioned':
        # Extreme condition numbers
        config = {
            'Dimension': 30,
            'NumComponents': 5,
            'TransformMode': 2,  # Fixed count
            'TransformCount': 1,  # 1 transformation per component
            'BaselineMode': 1,  # Random 50-50
            'RotationMode': 2,  # Fully connected
            'CenterMode': 1,  # Uniform random
            'KappaMode': 2,  # Beta distribution
            'BetaAlpha': 0.1,  # Very skewed distribution
            'BetaBeta': 0.1,
            'PMode': 1,  # Random range
            'BetaOffsetMode': 1,  # Independent random
            'TransformParamMode': 4,  # Global shared
            'NeutralizationMode': 1  # Fixed
        }
    
    else:
        raise ValueError(f"Unknown preset: {preset}. Available: standard, easy, medium, hard, multimodal, separable, ill_conditioned")
    
    return config