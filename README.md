# PORTAL:  Platform for Optimization Research, Testing, Analysis, and Learning

> A Controllable Landscape Generator for Continuous Optimization

**Git repository:** [EvoMindLab/PORTAL](https://github.com/EvoMindLab/PORTAL)

**The current version is `v1.0`**

## Author List

* **DANIAL YAZDANI**, danial.yazdani@gmail.com, School of Computing Technologies, RMIT University, Melbourne, Australia, 3000;
* **MAI PENG**, pengmai1998@gmail.com, School of Automation, China University of Geosciences, Wuhan, Hubei Key Laboratory of Advanced Control and Intelligent Automation for Complex Systems, and Engineering Research Center of Intelligent Technology for Geo-Exploration, Ministry of Education, China, 430074;
* **DELARAM YAZDANI**, delaram.yazdani@yahoo.com, Liverpool Logistics, Offshore and Marine (LOOM) Research Institute, Faculty of Engineering and Technology, Liverpool John Moores University, United Kingdom, L2 2ER;

## Get Started

```bash
# Clone the project
git clone https://github.com/EvoMindLab/PORTAL
cd PORTAL
```

## Architecture Overview

PORTAL provides both MATLAB and Python implementations with modular structure:

```
PORTAL/
├── MATLAB/                       # MATLAB implementation
│   ├── Generator_MATLAB_Script.m      # Benchmark instance generator
│   ├── Playground_MATLAB_Script.m     # 2D landscape visualization
│   ├── Evaluator_MATLAB_Script.m      # Sample evaluation script with DE
│   ├── Utils/                         # Utility functions
│   │   ├── fitness.m                  # Fitness evaluation
│   │   ├── load_instance.m            # JSON instance loader
│   │   ├── save_instance.m            # JSON instance saver
│   │   └── portal_default_settings.m  # Preset configurations
│   ├── Examples/                      # Example scripts
│   │   ├── test_suite.m               # Comprehensive test suite
│   │   └── test_export_import.m       # Export/import validation
│   └── Instances/                     # Stored benchmark instances
│
└── Python/                       # Python implementation
    ├── Generator_Python_Script.py     # Benchmark instance generator
    ├── Playground_Python_Script.py    # 2D landscape visualization
    ├── Evaluator_Python_Script.py     # Sample evaluation script with DE
    ├── Utils/                         # Utility functions
    │   ├── fitness.py                 # Fitness evaluation
    │   ├── load_instance.py           # JSON instance loader
    │   ├── save_instance.py           # JSON instance saver
    │   └── portal_default_settings.py # Preset configurations
    ├── Examples/                      # Example scripts
    │   ├── test_suite.py              # Comprehensive test suite
    │   └── test_export_import.py      # Export/import validation
    └── Instances/                     # Stored benchmark instances
```

## Running PORTAL

### Quick Start

#### MATLAB

```matlab
% Generate a 2D benchmark instance
PORTAL = Generator_MATLAB_Script('Dimension', 2, 'NumComponents', 3);

% Visualize the landscape
Playground_MATLAB_Script(PORTAL);

% Evaluate fitness at a point
x = randn(2, 1);
f = fitness(x, PORTAL);

% Save instance
save_instance(PORTAL, 'my_instance.json');

% Load instance
PORTAL_loaded = load_instance('my_instance.json');

% Run evaluator with DE optimizer
run Evaluator_MATLAB_Script.m
```

#### Python

```python
import numpy as np
from Generator_Python_Script import benchmark_generator
from Playground_Python_Script import plot_landscape
from Utils.fitness import fitness
from Utils.save_instance import save_instance
from Utils.load_instance import load_instance

# Generate a 2D benchmark instance
PORTAL = benchmark_generator(Dimension=2, NumComponents=3)

# Visualize the landscape
plot_landscape(PORTAL)

# Evaluate fitness at a point
x = np.random.randn(2)
f = fitness(x, PORTAL)

# Save instance
save_instance(PORTAL, 'my_instance.json')

# Load instance
PORTAL_loaded = load_instance('my_instance.json')

# Run evaluator with DE optimizer
python Evaluator_Python_Script.py
```

## Parameter Configuration Guide

### Core Parameters

| Parameter               | Default      | Description                     |
| ----------------------- | ------------ | ------------------------------- |
| **Dimension**     | 2            | Problem dimensionality          |
| **NumComponents** | 1            | Number of component functions   |
| **Seed**          | current time | Random seed for reproducibility |
| **BoundsMin**     | -100         | Lower bound of search space     |
| **BoundsMax**     | 100          | Upper bound of search space     |

### Transformation Modes

#### **BaselineMode** (Baseline Function Selection)

- **Mode 1** (default): Random 50-50 - Each component has 50% probability of Form A or Form B
- **Mode 2**: Custom probability - Use `BaselineProbA` to control probability of Form A
- **Mode 3**: All same - All components use same baseline (use `BaselineType`: 'A' or 'B')
- **Mode 4**: User-specified - Use `BaselineSeq` cell array with custom sequence

#### **TransformMode** (Transformation Sequence Generation)

- **Mode 1** (default): Fully random - Random count per component in [`TransformCountMin`, `TransformCountMax`], random transformations
- **Mode 2**: Fixed count - Each component has `TransformCount` random transformations
- **Mode 3**: No transformations - All components have empty transformation sequences
- **Mode 4**: User-specified - Use `TransformSeq` cell array (backward compatible)

**Related Parameters:**

- `TransformCountMin` (default: 0) - Minimum transform count for Mode 1
- `TransformCountMax` (default: 2) - Maximum transform count for Mode 1
- `TransformCount` (default: 1) - Fixed transform count for Mode 2

#### **TransformParamMode** (Transformation Parameter Distribution)

- **Mode 1** (default): Independent random - Each (k,i) parameter sampled independently (K×d matrix)
- **Mode 2**: Component-shared - All components share same pattern across dimensions (1×d replicated)
- **Mode 3**: Dimension-shared - All dimensions share same value per component (K×1 replicated)
- **Mode 4**: Global shared - All components and dimensions use same value (scalar)
- **Mode 5**: Beta distribution - Creates extreme parameter values (use `TransformParamBetaAlpha`, `TransformParamBetaBeta`)
- **Mode 6**: Linear distribution - Linearly spaced values with random permutation per component

### Landscape Configuration Modes

#### **CenterMode** (Component Center Distribution)

- **Mode 1** (default): Uniform random - Centers distributed uniformly in allowed range
- **Mode 2**: Exclusion zone - Centers avoid specified region (use `ExclusionZone`)
- **Mode 3**: Shared center - All components overlap at same center

#### **KappaMode** (Scaling Factor Distribution)

- **Mode 1** (default): Independent random - Each scaling factor sampled independently
- **Mode 2**: Beta distribution - Creates extreme condition numbers (use `BetaAlpha`, `BetaBeta`)
- **Mode 3**: Component-shared - Isotropic scaling within each component
- **Mode 4**: Linear distribution - Linearly spaced values with random permutation
- **Mode 5**: Isotropic (all directions) - Same scaling factor used in all directions

#### **PMode** (Exponent Distribution)

- **Mode 1** (default): Random range - Uniformly random exponents
- **Mode 2**: Fixed value - All exponents set to `FixedP`
- **Mode 3**: Gradient - Linearly varying from `p_min` to `p_max`
- **Mode 4**: Beta distribution - Creates extreme exponent values (use `PBetaAlpha`, `PBetaBeta`)
- **Mode 5**: Linear distribution - Linearly spaced values with random permutation per component (Form A only)

**Related Parameters:**

- `FixedP` (default: 2.0) - Fixed exponent value for Mode 2
- `p_min` (default: 0.5) - Minimum exponent for Mode 1 and Mode 3
- `p_max` (default: 2.5) - Maximum exponent for Mode 1 and Mode 3

#### **BetaOffsetMode** (Offset Value Distribution)

- **Mode 1** (default): Independent random - Each component has independent offset
- **Mode 2**: Shared value - All components share same random offset
- **Mode 3**: Fixed value - All components use `FixedBeta`
- **Mode 4**: Beta distribution - Creates extreme offset values (use `BetaOffsetBetaAlpha`, `BetaOffsetBetaBeta`)
- **Mode 5**: Linear distribution - Linearly spaced values with random permutation across components

**Related Parameters:**

- `FixedBeta` (default: 0) - Fixed offset value for Mode 3
- `beta_min` (default: -100) - Minimum offset
- `beta_max` (default: 100) - Maximum offset

#### **RotationMode** (Variable Interaction Pattern)

- **Mode 1** (default): No rotation - Separable problem (identity matrix)
- **Mode 2**: Fully connected - All variable pairs interact
- **Mode 3**: Probabilistic - Sparse interactions (use `RotationProb`)
- **Mode 4**: Uniform angle - Same angle for all pairs per component
- **Mode 5**: Fixed angle - User-specified angle for all pairs (use `SpecificAngle`)
- **Mode 6**: Chain-like - Only adjacent dimensions interact
- **Mode 7**: Block-diagonal - Partially separable groups (use `BlockSizes`, `BlockAngles`)

**Related Parameters:**

- `RotationProb` (default: 0.5) - Probability of interaction for Mode 3
- `SpecificAngle` (default: 45) - Fixed angle in degrees for Mode 5
- `BlockSizes` - Array of block sizes for Mode 7
- `BlockAngles` - Corresponding angles for each block in Mode 7

## Using Preset Configurations

PORTAL provides several preset configurations for common use cases:

### MATLAB

```matlab
% Load a preset configuration
config = portal_default_settings('easy');
PORTAL = Generator_MATLAB_Script(config{:});

% Available presets
% 'standard'       - Default behavior (mode=1 for all parameters)
% 'easy'           - Simple landscape: separable, low conditioning
% 'medium'         - Moderate difficulty: partial separability
% 'hard'           - Challenging: fully connected, high conditioning
% 'multimodal'     - Multiple overlapping components
% 'separable'      - Chain-like interactions
% 'ill_conditioned' - Extreme condition numbers
```

### Python

```python
from Utils.portal_default_settings import portal_default_settings

# Load a preset configuration
config = portal_default_settings('easy')
PORTAL = benchmark_generator(**config)
```

## Visualization (2D Only)

PORTAL includes landscape visualization tools for 2D instances:

### MATLAB

```matlab
% Generate 2D instance
PORTAL = Generator_MATLAB_Script('Dimension', 2, 'NumComponents', 3);

% Plot with default settings (resolution=250, show_contour=true)
Playground_MATLAB_Script(PORTAL);

% Custom settings
Playground_MATLAB_Script(PORTAL, 150, false);  % Lower resolution, no contour
```

### Python

```python
# Generate 2D instance
PORTAL = benchmark_generator(Dimension=2, NumComponents=3)

# Plot with default settings
plot_landscape(PORTAL)

# Custom settings
plot_landscape(PORTAL, resolution=150, show_contour=False)
```

## Evaluation and Optimization

The `Evaluator` scripts provide a template for using PORTAL with optimization algorithms. They include a sample Differential Evolution (DE) implementation.

### Configuration

Edit the configuration section in the evaluator script:

#### MATLAB (`Evaluator_MATLAB_Script.m`)

```matlab
%% Configuration
instance_file = 'Instances/my_instance.json';
RUNS = 5;                    % Number of independent runs
MAX_EVALS = 10000 * 2;       % Maximum function evaluations
ACCEPTED_ERROR = 1e-8;       % Acceptance threshold

% DE parameters
DE_config.PopulationSize = 100;
DE_config.F = 0.5;           % Scaling factor
DE_config.Cr = 0.9;          % Crossover rate
```

#### Python (`Evaluator_Python_Script.py`)

```python
# Configuration
instance_file = 'Instances/my_instance.json'
RUNS = 5
MAX_EVALS = 10000 * 2
ACCEPTED_ERROR = 1e-8

# DE parameters
DE_config = {
    'PopulationSize': 100,
    'F': 0.5,
    'Cr': 0.9
}
```

### Output

The evaluator provides:

- Per-run statistics (best value, error, acceptance point)
- Summary statistics (mean/std error, acceptance ratio)
- Convergence plot showing average error over function evaluations

## JSON Instance Format

PORTAL instances are saved in a structured JSON format that includes all parameters and metadata:

```json
{
  "d": 2,
  "K": 3,
  "Seed": 12345,
  "bounds_min": -100,
  "bounds_max": 100,
  "baseline": ["A", "B", "A"],
  "transform_seq": [["transformation1"], ["transformation2"], []],
  "P": {
    "c": [[x11, x12], [x21, x22], [x31, x32]],
    "beta": [offset1, offset2, offset3],
    "kappa_plus": [[...], [...], [...]],
    "p_plus": [[...], [...], [...]],
    "R": [[[rotation matrices]]]
  }
}
```

## System Requirements

### MATLAB

- **MATLAB R2016b+** for core functionality
- **MATLAB R2020a+** recommended for best compatibility

### Python

- **Python 3.7+**
- Required packages:
  ```bash
  pip install numpy matplotlib scipy
  ```

## License

This program is to be used under the terms of the [GNU General Public License](http://www.gnu.org/copyleft/gpl.html)

Copyright (c) 2025-present **Danial Yazdani, Mai Peng, Delaram Yazdani**

## For More Information

If you need assistance or have questions about PORTAL, please contact:

- Danial Yazdani: [danial.yazdani@gmail.com](mailto:danial.yazdani@gmail.com)
- Mai Peng: [pengmai1998@gmail.com](mailto:pengmai1998@gmail.com)
- Delaram Yazdani: [delaram.yazdani@yahoo.com](mailto:delaram.yazdani@yahoo.com)
