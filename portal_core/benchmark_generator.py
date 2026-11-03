"""
Author: Delaram Yazdani, Danial Yazdani, Mai Peng
Email: delaram.yazdani@yahoo.com
       danial.yazdani@gmail.com
       pengmai1998@gmail.com
Title: PORTAL Benchmark Instance Generator

Description:
    Generates random instances of PORTAL benchmark with configurable parameters
    Supports saving/loading instances to/from JSON files

License:
    MIT for PORTAL-owned code; see LICENSE and LICENSING.md at the repository root.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
import time

try:  # Package import (normal PORTAL use).
    from .validate_no_cover import validate_no_cover
except ImportError:  # Direct-script compatibility.
    from validate_no_cover import validate_no_cover  # type: ignore


def benchmark_generator(**kwargs) -> Dict[str, Any]:
    """
    Generate a PORTAL benchmark instance with configurable parameters.

    Args:
        **kwargs: Configuration parameters (see parameter list below)

    Returns:
        PORTAL benchmark instance dictionary

    Parameters:
        Seed: Random seed (default: current timestamp)
        Dimension: Problem dimension (default: 2)
        NumComponents: Number of components (default: 1)
        TransformMode: Transformation generation mode 1-4 (default: 1)
        TransformSeq: User-specified transform sequence (for mode 4)
        TransformCountMin: Min transform count (for mode 1)
        TransformCountMax: Max transform count (for mode 1)
        TransformCount: Fixed transform count (for mode 2)
        BaselineMode: Baseline selection mode 1-4 (default: 1)
        BaselineProbA: Probability for Form A (for mode 2)
        BaselineType: 'A' or 'B' (for mode 3)
        BaselineSeq: User-specified baseline sequence (for mode 4)
        RotationMode: Rotation mode 1-7 (default: 1, no rotation)
        RotationProb: Probability for mode 3
        SpecificAngle: Fixed angle for mode 5
        BlockSizes: Block sizes for mode 7
        BlockAngles: Block angles for mode 7
        CenterMode: Center generation mode 1-3 (default: 1)
        ExclusionZone: Exclusion zone for mode 2
        KappaMode: Scaling factor mode 1-5 (default: 1)
        BetaAlpha, BetaBeta: Beta distribution parameters for KappaMode=2
        PMode: Exponent mode 1-5 (default: 1)
        FixedP: Fixed value for PMode=2
        PBetaAlpha, PBetaBeta: Beta parameters for PMode=4
        BetaOffsetMode: Beta offset mode 1-5 (default: 1)
        FixedBeta: Fixed value for BetaOffsetMode=3
        BetaOffsetBetaAlpha, BetaOffsetBetaBeta: Beta parameters for mode 4
        BetaGap: Within-set offset span for BetaOffsetMode=6 (default: 20).
            Mode 6 draws one shared base in [beta_min, beta_max - gap]; one
            random component sits at the base and the others draw offsets in
            [BetaOptGap, gap].
        BetaOptGap: Minimum margin between the set optimum and every other
            component's beta for BetaOffsetMode=6 (default: 1).
        TransformParamMode: Transform parameter mode 1-6 (default: 1)
        TransformParamBetaAlpha, TransformParamBetaBeta: Beta parameters for mode 5
        DeltaMin: Lower bound of the per-component scale Delta (default: 50)
        DeltaMax: Upper bound of the per-component scale Delta (default: 200)
        RRef: Reference radius r_ref used by the neutralization (default: 100, fixed)
        LSSigma: Smoothing scale for the log-sinusoidal origin gate (default: 0.1)
        EnsureNoCover: Reject and redraw centers/betas until every component
            center is exposed in its Layer-2 block (default: True).
        NoCoverMaxAttempts: Maximum total center/beta draws (default: 1000).
        NoCoverTolerance: Numerical tolerance for the center-value check
            (default: 1e-6).
        WarnOutOfBounds: Whether `fitness` should warn when x is outside [bounds_min, bounds_max].
            Stored on the instance and read back by `fitness` (default: True).
    """
    # Parse input arguments with defaults
    seed = kwargs.get('Seed', int(time.time() * 1000) % (2**31))
    d = kwargs.get('Dimension', 2)
    K = kwargs.get('NumComponents', 1)

    # Layer-3 / out-of-bounds knobs (per-instance, do not affect parameter sampling)
    delta_min = float(kwargs.get('DeltaMin', 50.0))
    delta_max = float(kwargs.get('DeltaMax', 200.0))
    r_ref_val = float(kwargs.get('RRef', 100.0))
    center_min = kwargs.get('CenterMin', None)
    center_max = kwargs.get('CenterMax', None)
    beta_min = kwargs.get('BetaMin', None)
    beta_max = kwargs.get('BetaMax', None)
    ls_sigma = kwargs.get('LSSigma', 0.1)
    ensure_no_cover = bool(kwargs.get('EnsureNoCover', True))
    no_cover_max_attempts = int(kwargs.get('NoCoverMaxAttempts', 1000))
    no_cover_tolerance = float(kwargs.get('NoCoverTolerance', 1e-6))
    warn_oob = bool(kwargs.get('WarnOutOfBounds', True))

    if no_cover_max_attempts < 1:
        raise ValueError('NoCoverMaxAttempts must be at least 1.')
    if not np.isfinite(no_cover_tolerance) or no_cover_tolerance < 0.0:
        raise ValueError('NoCoverTolerance must be finite and nonnegative.')
    if (
        not np.isfinite(delta_min)
        or not np.isfinite(delta_max)
        or delta_min <= 0.0
        or delta_max < delta_min
    ):
        raise ValueError(
            'DeltaMin and DeltaMax must be finite and satisfy '
            '0 < DeltaMin <= DeltaMax.'
        )
    if not np.isfinite(r_ref_val) or r_ref_val <= 0.0:
        raise ValueError('RRef must be finite and greater than zero.')

    # Initialize structure
    portal = {}

    # Basic Configuration
    portal['Seed'] = seed
    portal['Rng'] = np.random.RandomState(seed)
    portal['d'] = d
    portal['K'] = K

    # Domain Bounds
    portal['bounds_min'] = -100
    portal['bounds_max'] = 100

    # Out-of-bounds policy (read by fitness)
    portal['warn_out_of_bounds'] = warn_oob

    # Get RNG for easier access
    rng = portal['Rng']

    # Parameter ranges (with user overrides for Delta and r_ref)
    ranges = _get_parameter_ranges(portal)
    if center_min is not None:
        ranges['c_min'] = float(center_min)
    if center_max is not None:
        ranges['c_max'] = float(center_max)
    if beta_min is not None:
        ranges['beta_min'] = float(beta_min)
    if beta_max is not None:
        ranges['beta_max'] = float(beta_max)
    ranges['Delta_min'] = float(delta_min)
    ranges['Delta_max'] = float(delta_max)
    ranges['r_ref_min'] = float(r_ref_val)
    ranges['r_ref_max'] = float(r_ref_val)
    portal['ranges'] = ranges

    # Baseline Selection
    baseline_mode = kwargs.get('BaselineMode', 1)
    baseline_prob_a = kwargs.get('BaselineProbA', 0.5)
    baseline_type = kwargs.get('BaselineType', 'A')
    baseline_seq = kwargs.get('BaselineSeq', [])

    portal['baseline'] = sample_baseline(rng, K, baseline_mode,
                                         baseline_prob_a, baseline_type, baseline_seq)

    # Transformation Sequence
    transform_mode = kwargs.get('TransformMode', 1)
    transform_count_min = kwargs.get('TransformCountMin', 0)
    transform_count_max = kwargs.get('TransformCountMax', 2)
    transform_count = kwargs.get('TransformCount', 1)
    transform_seq = kwargs.get('TransformSeq', [[]])

    portal['transform_seq'] = sample_transform_seq(rng, K, transform_mode,
                                                   transform_count_min, transform_count_max,
                                                   transform_count, transform_seq)

    # Rotation Mode
    rotation_mode = kwargs.get('RotationMode', 1)
    if rotation_mode < 1 or rotation_mode > 7:
        print(f"Warning: RotationMode must be 1-7. Using default mode 1.")
        rotation_mode = 1

    # Initialize parameters dictionary
    P = {}

    # Centers
    center_mode = kwargs.get('CenterMode', 1)
    exclusion_zone = kwargs.get('ExclusionZone', [-20, 20])
    P['c'] = sample_centers(rng, K, d, ranges, center_mode, exclusion_zone)

    # Beta offsets
    beta_offset_mode = kwargs.get('BetaOffsetMode', 1)
    fixed_beta = kwargs.get('FixedBeta', 0)
    beta_offset_alpha = kwargs.get('BetaOffsetBetaAlpha', 0.2)
    beta_offset_beta = kwargs.get('BetaOffsetBetaBeta', 0.2)
    beta_gap = float(kwargs.get('BetaGap', 20.0))
    beta_opt_gap = float(kwargs.get('BetaOptGap', 1.0))
    P['beta'] = sample_beta(rng, K, ranges, beta_offset_mode, fixed_beta,
                           beta_offset_alpha, beta_offset_beta,
                           beta_gap=beta_gap, beta_opt_gap=beta_opt_gap)

    # Kappa (scaling factors)
    kappa_mode = kwargs.get('KappaMode', 1)
    beta_alpha = kwargs.get('BetaAlpha', 0.2)
    beta_beta = kwargs.get('BetaBeta', 0.2)
    P['kappa_plus'] = sample_kappa(rng, K, d, ranges, kappa_mode, beta_alpha, beta_beta)
    P['kappa_minus'] = sample_kappa(rng, K, d, ranges, kappa_mode, beta_alpha, beta_beta)

    # Exponents
    p_mode = kwargs.get('PMode', 1)
    fixed_p = kwargs.get('FixedP', 0.6)
    p_beta_alpha = kwargs.get('PBetaAlpha', 0.5)
    p_beta_beta = kwargs.get('PBetaBeta', 0.5)
    P['p_single'] = sample_exponents(rng, K, 1, ranges, p_mode, fixed_p, p_beta_alpha, p_beta_beta)
    P['p_plus'] = sample_exponents(rng, K, d, ranges, p_mode, fixed_p, p_beta_alpha, p_beta_beta)
    P['p_minus'] = sample_exponents(rng, K, d, ranges, p_mode, fixed_p, p_beta_alpha, p_beta_beta)

    # Rotation angles and matrices
    rotation_prob = kwargs.get('RotationProb', 0.5)
    specific_angle = kwargs.get('SpecificAngle', 45)
    block_sizes = kwargs.get('BlockSizes', [])
    block_angles = kwargs.get('BlockAngles', [])

    P['Psi_deg'] = np.zeros((d, d, K))
    P['R'] = np.zeros((d, d, K))

    for k in range(K):
        Psi_deg_k = generate_rotation_angles(rng, d, ranges, rotation_mode,
                                             rotation_prob, specific_angle,
                                             block_sizes, block_angles)
        P['Psi_deg'][:, :, k] = Psi_deg_k
        P['R'][:, :, k] = givens_from_psi(np.deg2rad(Psi_deg_k))

    # Neutralization parameters
    P['neutralize_enable'] = True
    P['Delta'] = ranges['Delta_min'] + (ranges['Delta_max'] - ranges['Delta_min']) * rng.rand(K)
    P['r_ref'] = ranges['r_ref_min'] + (ranges['r_ref_max'] - ranges['r_ref_min']) * rng.rand(K)

    portal['P'] = P

    # Generate transformation parameters
    tpm = kwargs.get('TransformParamMode', 1)
    tpm_alpha = kwargs.get('TransformParamBetaAlpha', 0.2)
    tpm_beta = kwargs.get('TransformParamBetaBeta', 0.2)

    _generate_transform_parameters(portal, rng, K, d, ranges, tpm, tpm_alpha, tpm_beta,
                                   ls_sigma=ls_sigma)

    # Precompute scalings
    form, rho_single, rho_plus, rho_minus = precompute_scalings(portal)
    portal['P']['form'] = form
    portal['P']['rho_single'] = rho_single
    portal['P']['rho_plus'] = rho_plus
    portal['P']['rho_minus'] = rho_minus

    # The min-composition can hide a nominal component if another component
    # evaluates below beta_k at c_k.  Condition the stochastic center/beta
    # draw on every component center remaining exposed.  A finite attempt cap
    # prevents impossible user configurations (for example, coincident centers
    # with unequal fixed depths) from hanging indefinitely.
    no_cover_attempts = 1
    no_cover_rejections = 0
    no_cover_offenders = []
    if ensure_no_cover:
        no_cover_ok, no_cover_offenders = validate_no_cover(
            portal, tol=no_cover_tolerance
        )
        while not no_cover_ok and no_cover_attempts < no_cover_max_attempts:
            P['c'] = sample_centers(rng, K, d, ranges, center_mode, exclusion_zone)
            P['beta'] = sample_beta(
                rng, K, ranges, beta_offset_mode, fixed_beta,
                beta_offset_alpha, beta_offset_beta,
                beta_gap=beta_gap, beta_opt_gap=beta_opt_gap,
            )
            no_cover_attempts += 1
            no_cover_rejections += 1
            no_cover_ok, no_cover_offenders = validate_no_cover(
                portal, tol=no_cover_tolerance
            )

        if not no_cover_ok:
            indices = [int(item[0]) + 1 for item in no_cover_offenders[:8]]
            raise RuntimeError(
                'Unable to generate an uncovered component configuration after '
                f'{no_cover_max_attempts} center/beta draws. Covered component '
                f'indices include {indices}. Adjust the center/beta modes or ranges, '
                'increase NoCoverMaxAttempts, or explicitly set EnsureNoCover=False '
                'when covered components are intentional.'
            )

    # Optional generation metadata records resolved sampling ranges and modes
    # for reproducibility. The evaluation pipeline does not consume it.
    portal['_generation'] = {
        'schema_version': 1,
        'ranges': {k: float(v) for k, v in ranges.items()},
        'config': {
            'Seed': int(seed),
            'Dimension': int(d),
            'NumComponents': int(K),
            'BaselineMode': baseline_mode,
            'BaselineProbA': baseline_prob_a,
            'BaselineType': baseline_type,
            'BaselineSeq': list(baseline_seq) if baseline_seq else [],
            'TransformMode': transform_mode,
            'TransformCountMin': transform_count_min,
            'TransformCountMax': transform_count_max,
            'TransformCount': transform_count,
            'RotationMode': rotation_mode,
            'RotationProb': rotation_prob,
            'SpecificAngle': specific_angle,
            'BlockSizes': list(block_sizes) if block_sizes else [],
            'BlockAngles': list(block_angles) if block_angles else [],
            'CenterMode': center_mode,
            'ExclusionZone': list(exclusion_zone) if exclusion_zone else [],
            'KappaMode': kappa_mode,
            'BetaAlpha': beta_alpha,
            'BetaBeta': beta_beta,
            'PMode': p_mode,
            'FixedP': fixed_p,
            'PBetaAlpha': p_beta_alpha,
            'PBetaBeta': p_beta_beta,
            'BetaOffsetMode': beta_offset_mode,
            'FixedBeta': fixed_beta,
            'BetaOffsetBetaAlpha': beta_offset_alpha,
            'BetaOffsetBetaBeta': beta_offset_beta,
            'BetaGap': float(beta_gap),
            'BetaOptGap': float(beta_opt_gap),
            'TransformParamMode': tpm,
            'TransformParamBetaAlpha': tpm_alpha,
            'TransformParamBetaBeta': tpm_beta,
            'DeltaMin': float(delta_min),
            'DeltaMax': float(delta_max),
            'RRef': float(r_ref_val),
            'LSSigma': float(ls_sigma),
            'EnsureNoCover': bool(ensure_no_cover),
            'NoCoverMaxAttempts': int(no_cover_max_attempts),
            'NoCoverTolerance': float(no_cover_tolerance),
        },
        'no_cover': {
            'enforced': bool(ensure_no_cover),
            'tolerance': float(no_cover_tolerance),
            'attempts': int(no_cover_attempts),
            'rejections': int(no_cover_rejections),
            'validated': bool(ensure_no_cover),
        },
    }

    print(f"PORTAL Benchmark Instance Generated:")
    print(f"  Dimension: {portal['d']}")
    print(f"  Components: {portal['K']}")
    print(f"  Seed: {portal['Seed']}")
    print(f"  Rotation Mode: {rotation_mode}")
    print(f"  Baselines: {' '.join(portal['baseline'])}")
    if ensure_no_cover and no_cover_rejections:
        print(f"  No-cover redraws: {no_cover_rejections}")

    return portal


def _get_parameter_ranges(portal: Dict[str, Any]) -> Dict[str, float]:
    """Get parameter ranges."""
    ranges = {}

    # Component-Level Parameters
    ranges['c_min'] = portal['bounds_min'] + 10
    ranges['c_max'] = portal['bounds_max'] - 10
    ranges['beta_min'] = 0
    ranges['beta_max'] = 10
    ranges['p_min'] = 0.2
    ranges['p_max'] = 1.2
    ranges['kappa_min'] = 1
    ranges['kappa_max'] = 100
    ranges['theta_min'] = -180
    ranges['theta_max'] = 180

    # Neutralization Parameters (defaults; may be overridden via kwargs in benchmark_generator)
    ranges['Delta_min'] = 50
    ranges['Delta_max'] = 200
    ranges['r_ref_min'] = 100
    ranges['r_ref_max'] = 100

    # Additive Periodic Transform
    ranges['AP_mu_min'] = 0.1
    ranges['AP_mu_max'] = 0.7
    ranges['AP_gamma_min'] = 0.002
    ranges['AP_gamma_max'] = 0.2
    ranges['AP_omega_min'] = 0.05
    ranges['AP_omega_max'] = 1.0

    # Log-Sinusoidal Transform
    ranges['LS_mu_min'] = 0.05
    ranges['LS_mu_max'] = 0.50
    ranges['LS_omega_min'] = 5
    ranges['LS_omega_max'] = 50

    # Wavelet-Inspired Transform (envelope length L is derived as eta/omega; not sampled)
    ranges['WM_mu_min'] = 10
    ranges['WM_mu_max'] = 50
    ranges['WM_omega_min'] = 0.3
    ranges['WM_omega_max'] = 1.0
    ranges['WM_eta_min'] = 10
    ranges['WM_eta_max'] = 24

    # Tensor Interference Transform
    ranges['TI_mu0_min'] = 10
    ranges['TI_mu0_max'] = 20
    ranges['TI_omega_min'] = 0.1
    ranges['TI_omega_max'] = 0.7

    # Radial Polytrig Transform
    ranges['RP_mu_min'] = 0.4
    ranges['RP_mu_max'] = 2.0
    ranges['RP_p_min'] = 0.4
    ranges['RP_p_max'] = 0.7
    ranges['RP_q_min'] = 0.4
    ranges['RP_q_max'] = 1.2
    ranges['RP_omega_min'] = 0.1
    ranges['RP_omega_max'] = 10

    return ranges


def sample_baseline(rng: np.random.RandomState, K: int, mode: int,
                   prob_a: float = 0.5, baseline_type: str = 'A',
                   user_seq: List[str] = []) -> List[str]:
    """Sample baseline (Form A or B) with different strategies."""
    if mode == 1:  # Random 50-50
        return ['A' if rng.rand() < 0.5 else 'B' for _ in range(K)]
    elif mode == 2:  # Custom probability
        return ['A' if rng.rand() < prob_a else 'B' for _ in range(K)]
    elif mode == 3:  # All same
        return [baseline_type] * K
    elif mode == 4:  # User-specified
        if len(user_seq) != K:
            raise ValueError(f"User-specified baseline_seq must have K={K} elements")
        return user_seq
    else:
        raise ValueError(f"Invalid baseline_mode: {mode}. Valid modes are 1-4.")


def sample_transform_seq(rng: np.random.RandomState, K: int, mode: int,
                         count_min: int = 0, count_max: int = 1,
                         count: Union[int, List[int]] = 1,
                         user_seq: List[List[str]] = [[]]) -> List[List[str]]:
    """Sample transformation sequences with different strategies."""
    available_transforms = ['additive_periodic', 'log_sinusoidal', 'wavelet_mod',
                           'tensor_interference', 'radial_polytrig']

    if mode == 1:  # Fully random
        transform_seq = []
        for k in range(K):
            cnt = rng.randint(count_min, count_max + 1)
            if cnt == 0:
                transform_seq.append([])
            else:
                indices = rng.randint(0, len(available_transforms), cnt)
                transform_seq.append([available_transforms[i] for i in indices])
        return transform_seq

    elif mode == 2:  # Fixed count
        counts = [count] * K if isinstance(count, int) else count
        if len(counts) != K:
            raise ValueError(f"count must be scalar or K-element array (K={K})")

        transform_seq = []
        for k in range(K):
            if counts[k] == 0:
                transform_seq.append([])
            else:
                indices = rng.randint(0, len(available_transforms), counts[k])
                transform_seq.append([available_transforms[i] for i in indices])
        return transform_seq

    elif mode == 3:  # No transformations
        return [[] for _ in range(K)]

    elif mode == 4:  # User-specified
        if len(user_seq) == 1 and K > 1:
            return user_seq * K
        return user_seq

    else:
        raise ValueError(f"Invalid transform_mode: {mode}. Valid modes are 1-4.")


def sample_centers(rng: np.random.RandomState, K: int, d: int,
                  ranges: Dict[str, float], mode: int,
                  exclusion_zone: List[float] = [-20, 20]) -> np.ndarray:
    """Sample component center positions with different strategies."""
    if mode == 1:  # Uniform random
        return ranges['c_min'] + (ranges['c_max'] - ranges['c_min']) * rng.rand(K, d)

    elif mode == 2:  # With exclusion zone
        # Generate candidates in lower or upper ranges
        lower_range = ranges['c_min'] + \
                     (exclusion_zone[0] - ranges['c_min']) * rng.rand(K, d)
        upper_range = exclusion_zone[1] + \
                     (ranges['c_max'] - exclusion_zone[1]) * rng.rand(K, d)
        # Randomly select from lower or upper range
        selector = rng.randint(0, 2, (K, d))
        return selector * lower_range + (1 - selector) * upper_range

    elif mode == 3:  # Shared center
        center = ranges['c_min'] + (ranges['c_max'] - ranges['c_min']) * rng.rand(1, d)
        return np.tile(center, (K, 1))

    else:
        raise ValueError(f"Invalid center_mode: {mode}. Valid modes are 1-3.")


def sample_kappa(rng: np.random.RandomState, K: int, d: int,
                ranges: Dict[str, float], mode: int,
                beta_alpha: float = 0.2, beta_beta: float = 0.2) -> np.ndarray:
    """Sample scaling factors with different strategies."""
    if mode == 1:  # Independent random
        return ranges['kappa_min'] + (ranges['kappa_max'] - ranges['kappa_min']) * rng.rand(K, d)

    elif mode == 2:  # Beta distribution
        beta_vals = rng.beta(beta_alpha, beta_beta, (K, d))
        return ranges['kappa_min'] + (ranges['kappa_max'] - ranges['kappa_min']) * beta_vals

    elif mode == 3:  # Component-shared (isotropic)
        scales = ranges['kappa_min'] + (ranges['kappa_max'] - ranges['kappa_min']) * rng.rand(K, 1)
        return np.tile(scales, (1, d))

    elif mode == 4:  # Linear distribution with permutation
        kappa = np.zeros((K, d))
        kappa_vals = np.linspace(ranges['kappa_min'], ranges['kappa_max'], d)
        for k in range(K):
            kappa[k, :] = kappa_vals[rng.permutation(d)]
        return kappa

    elif mode == 5:  # Isotropic (all directions)
        return np.ones((K, d))

    else:
        raise ValueError(f"Invalid kappa_mode: {mode}. Valid modes are 1-5.")


def sample_exponents(rng: np.random.RandomState, K: int, d: int,
                    ranges: Dict[str, float], mode: int,
                    fixed_p: float = 0.8,
                    p_beta_alpha: float = 0.5, p_beta_beta: float = 0.5) -> np.ndarray:
    """Sample exponent values with different strategies."""
    if mode == 1:  # Random range
        return ranges['p_min'] + (ranges['p_max'] - ranges['p_min']) * rng.rand(K, d)

    elif mode == 2:  # Fixed value
        return fixed_p * np.ones((K, d))

    elif mode == 3:  # Gradient
        gradient = np.linspace(ranges['p_min'], ranges['p_max'], d)
        return np.tile(gradient, (K, 1))

    elif mode == 4:  # Beta distribution
        beta_vals = rng.beta(p_beta_alpha, p_beta_beta, (K, d))
        return ranges['p_min'] + (ranges['p_max'] - ranges['p_min']) * beta_vals

    elif mode == 5:  # Linear distribution with permutation
        if d != 1:
            p_vals = np.zeros((K, d))
            p_vals_linear = np.linspace(ranges['p_min'], ranges['p_max'], d)
            for k in range(K):
                p_vals[k, :] = p_vals_linear[rng.permutation(d)]
            return p_vals
        else:
            return ranges['p_min'] + (ranges['p_max'] - ranges['p_min']) * rng.rand(K, d)

    else:
        raise ValueError(f"Invalid p_mode: {mode}. Valid modes are 1-5.")


def sample_beta(rng: np.random.RandomState, K: int, ranges: Dict[str, float],
               mode: int, fixed_beta: float = 50,
               beta_alpha: float = 0.2, beta_beta_param: float = 0.2,
               beta_gap: float = 20.0,
               beta_opt_gap: float = 1.0) -> np.ndarray:
    """Sample offset values with different strategies."""
    if mode == 1:  # Independent random
        return ranges['beta_min'] + (ranges['beta_max'] - ranges['beta_min']) * rng.rand(K)

    elif mode == 2:  # Shared random value
        val = ranges['beta_min'] + (ranges['beta_max'] - ranges['beta_min']) * rng.rand()
        return np.full(K, val)

    elif mode == 3:  # Fixed value
        return np.full(K, fixed_beta)

    elif mode == 4:  # Beta distribution
        beta_vals = rng.beta(beta_alpha, beta_beta_param, K)
        return ranges['beta_min'] + (ranges['beta_max'] - ranges['beta_min']) * beta_vals

    elif mode == 5:  # Linear distribution with permutation
        beta_vals_linear = np.linspace(ranges['beta_min'], ranges['beta_max'], K)
        return beta_vals_linear[rng.permutation(K)]

    elif mode == 6:  # Shared random base + bounded, optimum-protected offsets
        # One base value places the whole component set inside
        # [beta_min, beta_max]; one randomly chosen component sits exactly at
        # the base (the set optimum) and every other component's offset is
        # confined to [beta_opt_gap, beta_gap], so no decoy lies within
        # beta_opt_gap of the optimum.  Keeping the offset span small relative
        # to the minimum per-component rise scale (Delta) makes basin-depth
        # gaps compatible with the no-cover condition; both bounds mirror the
        # frozen PORTAL catalog (observed span 20 and observed optimum margin
        # >= 1.18 against Delta >= 50).
        if not np.isfinite(beta_gap) or beta_gap < 0.0:
            raise ValueError(f"beta_gap must be finite and nonnegative; received {beta_gap!r}.")
        if not np.isfinite(beta_opt_gap) or beta_opt_gap < 0.0:
            raise ValueError(
                f"beta_opt_gap must be finite and nonnegative; received {beta_opt_gap!r}.")
        span = float(ranges['beta_max'] - ranges['beta_min'])
        gap_eff = min(float(beta_gap), span) if span > 0.0 else 0.0
        opt_gap_eff = min(float(beta_opt_gap), gap_eff)
        base = ranges['beta_min'] + (span - gap_eff) * rng.rand()
        betas = base + opt_gap_eff + (gap_eff - opt_gap_eff) * rng.rand(K)
        best = int(rng.randint(K))
        betas[best] = base
        return betas

    else:
        raise ValueError(f"Invalid beta_mode: {mode}. Valid modes are 1-6.")


def generate_rotation_angles(rng: np.random.RandomState, d: int,
                            ranges: Dict[str, float], mode: int,
                            prob: float = 0.5, specific_angle: float = 45,
                            block_sizes: List[int] = [],
                            block_angles: List[float] = [],
                            randomize_angles: bool = False) -> np.ndarray:
    """Generate rotation angles based on mode."""
    Psi_deg = np.zeros((d, d))

    if mode == 1:  # No rotation
        return Psi_deg

    elif mode == 2:  # Fully connected
        for u in range(d - 1):
            for v in range(u + 1, d):
                Psi_deg[u, v] = ranges['theta_min'] + \
                               (ranges['theta_max'] - ranges['theta_min']) * rng.rand()
        return Psi_deg

    elif mode == 3:  # Probabilistic
        for u in range(d - 1):
            for v in range(u + 1, d):
                if rng.rand() < prob:
                    Psi_deg[u, v] = ranges['theta_min'] + \
                                   (ranges['theta_max'] - ranges['theta_min']) * rng.rand()
        return Psi_deg

    elif mode == 4:  # Uniform angle
        angle = ranges['theta_min'] + (ranges['theta_max'] - ranges['theta_min']) * rng.rand()
        for u in range(d - 1):
            for v in range(u + 1, d):
                Psi_deg[u, v] = angle
        return Psi_deg

    elif mode == 5:  # Fixed angle
        for u in range(d - 1):
            for v in range(u + 1, d):
                Psi_deg[u, v] = specific_angle
        return Psi_deg

    elif mode == 6:  # Chain-like
        for u in range(d - 1):
            Psi_deg[u, u + 1] = ranges['theta_min'] + \
                               (ranges['theta_max'] - ranges['theta_min']) * rng.rand()
        return Psi_deg

    elif mode == 7:  # Block-diagonal
        if len(block_sizes) == 0:
            n_blocks = 3 if d >= 30 else 2
            base_size = d // n_blocks
            remainder = d % n_blocks
            block_sizes = [base_size] * n_blocks
            for b in range(remainder):
                block_sizes[b] += 1

        all_dims = rng.permutation(d)
        groups, start = [], 0
        for size in block_sizes:
            groups.append(list(all_dims[start:start + size]))
            start += size

        if randomize_angles:
            angles = None
        elif len(block_angles) == 0 or len(block_angles) != len(groups):
            default_angles = [45, 135, 22.5, 112.5]
            angles = [default_angles[b % len(default_angles)]
                      for b in range(len(groups))]
        else:
            angles = list(block_angles)

        for b, group in enumerate(groups):
            gd = list(group)
            for i in range(len(gd)):
                for j in range(i + 1, len(gd)):
                    u, v = gd[i], gd[j]
                    if u > v:
                        u, v = v, u
                    if randomize_angles:
                        Psi_deg[u, v] = ranges['theta_min'] + \
                                        (ranges['theta_max'] - ranges['theta_min']) * rng.rand()
                    else:
                        Psi_deg[u, v] = angles[b]
        return Psi_deg

    else:
        print(f"Warning: Unknown rotation mode {mode}, using identity matrix")
        return Psi_deg


def givens_from_psi(Psi: np.ndarray) -> np.ndarray:
    """Build orthogonal R from upper-triangular angle matrix Psi via Givens rotations."""
    d = Psi.shape[0]
    R = np.eye(d)
    for u in range(d - 1):
        for v in range(u + 1, d):
            psi = Psi[u, v]
            if not np.isfinite(psi) or abs(psi) < 1e-15:
                continue
            G = np.eye(d)
            c, s = np.cos(psi), np.sin(psi)
            G[u, u] = c
            G[u, v] = -s
            G[v, u] = s
            G[v, v] = c
            R = R @ G
    return R


def sample_transform_param(rng: np.random.RandomState, K: int, d: int,
                           range_min: float, range_max: float, mode: int,
                           beta_alpha: float = 0.2, beta_beta: float = 0.2) -> np.ndarray:
    """Sample transformation parameter matrix with different strategies."""
    if mode == 1:  # Independent random
        return range_min + (range_max - range_min) * rng.rand(K, d)

    elif mode == 2:  # Component-shared
        row = range_min + (range_max - range_min) * rng.rand(1, d)
        return np.tile(row, (K, 1))

    elif mode == 3:  # Dimension-shared
        col = range_min + (range_max - range_min) * rng.rand(K, 1)
        return np.tile(col, (1, d))

    elif mode == 4:  # Global shared
        val = range_min + (range_max - range_min) * rng.rand()
        return np.full((K, d), val)

    elif mode == 5:  # Beta distribution
        beta_vals = rng.beta(beta_alpha, beta_beta, (K, d))
        return range_min + (range_max - range_min) * beta_vals

    elif mode == 6:  # Linear distribution with permutation
        param_matrix = np.zeros((K, d))
        param_vals_linear = np.linspace(range_min, range_max, d)
        for k in range(K):
            param_matrix[k, :] = param_vals_linear[rng.permutation(d)]
        return param_matrix

    else:
        raise ValueError(f"Invalid transform_param_mode: {mode}. Valid modes are 1-6.")


def _generate_transform_parameters(portal: Dict[str, Any], rng: np.random.RandomState,
                                   K: int, d: int, ranges: Dict[str, float],
                                   tpm: int, tpm_alpha: float, tpm_beta: float,
                                   ls_sigma: float = 0.1):
    """Generate all transformation parameters."""
    # Additive Periodic
    T_AP = {}
    T_AP['mu_plus'] = sample_transform_param(rng, K, d, ranges['AP_mu_min'], ranges['AP_mu_max'], tpm, tpm_alpha, tpm_beta)
    T_AP['mu_minus'] = sample_transform_param(rng, K, d, ranges['AP_mu_min'], ranges['AP_mu_max'], tpm, tpm_alpha, tpm_beta)
    T_AP['gamma_plus'] = sample_transform_param(rng, K, d, ranges['AP_gamma_min'], ranges['AP_gamma_max'], tpm, tpm_alpha, tpm_beta)
    T_AP['gamma_minus'] = sample_transform_param(rng, K, d, ranges['AP_gamma_min'], ranges['AP_gamma_max'], tpm, tpm_alpha, tpm_beta)
    T_AP['omega_plus'] = sample_transform_param(rng, K, d, ranges['AP_omega_min'], ranges['AP_omega_max'], tpm, tpm_alpha, tpm_beta)
    T_AP['omega_minus'] = sample_transform_param(rng, K, d, ranges['AP_omega_min'], ranges['AP_omega_max'], tpm, tpm_alpha, tpm_beta)
    portal['P']['T_additive_periodic'] = T_AP

    # Log-Sinusoidal
    T_LS = {}
    T_LS['mu_plus'] = sample_transform_param(rng, K, d, ranges['LS_mu_min'], ranges['LS_mu_max'], tpm, tpm_alpha, tpm_beta)
    T_LS['mu_minus'] = sample_transform_param(rng, K, d, ranges['LS_mu_min'], ranges['LS_mu_max'], tpm, tpm_alpha, tpm_beta)
    T_LS['omega1_plus'] = sample_transform_param(rng, K, d, ranges['LS_omega_min'], ranges['LS_omega_max'], tpm, tpm_alpha, tpm_beta)
    T_LS['omega2_plus'] = sample_transform_param(rng, K, d, ranges['LS_omega_min'], ranges['LS_omega_max'], tpm, tpm_alpha, tpm_beta)
    T_LS['omega1_minus'] = sample_transform_param(rng, K, d, ranges['LS_omega_min'], ranges['LS_omega_max'], tpm, tpm_alpha, tpm_beta)
    T_LS['omega2_minus'] = sample_transform_param(rng, K, d, ranges['LS_omega_min'], ranges['LS_omega_max'], tpm, tpm_alpha, tpm_beta)
    # Smoothing scale used by the origin gate g(a) = 1 - exp(-a^2 / sigma^2).
    # Stored as a per-component vector so it can be varied if desired; default constant.
    T_LS['sigma'] = float(ls_sigma) * np.ones(K)
    portal['P']['T_log_sinusoidal'] = T_LS

    # Wavelet-Inspired (envelope length L derived as eta/omega in fitness; not stored)
    T_WM = {}
    T_WM['mu_plus'] = sample_transform_param(rng, K, d, ranges['WM_mu_min'], ranges['WM_mu_max'], tpm, tpm_alpha, tpm_beta)
    T_WM['mu_minus'] = sample_transform_param(rng, K, d, ranges['WM_mu_min'], ranges['WM_mu_max'], tpm, tpm_alpha, tpm_beta)
    T_WM['omega_plus'] = sample_transform_param(rng, K, d, ranges['WM_omega_min'], ranges['WM_omega_max'], tpm, tpm_alpha, tpm_beta)
    T_WM['omega_minus'] = sample_transform_param(rng, K, d, ranges['WM_omega_min'], ranges['WM_omega_max'], tpm, tpm_alpha, tpm_beta)
    T_WM['eta'] = ranges['WM_eta_min'] + (ranges['WM_eta_max'] - ranges['WM_eta_min']) * rng.rand(K)
    T_WM['eps'] = 1e-12
    portal['P']['T_wavelet_mod'] = T_WM

    # Tensor Interference
    T_TI = {}
    T_TI['mu_plus'] = sample_transform_param(rng, K, d, ranges['TI_mu0_min'], ranges['TI_mu0_max'], tpm, tpm_alpha, tpm_beta)
    T_TI['mu_minus'] = sample_transform_param(rng, K, d, ranges['TI_mu0_min'], ranges['TI_mu0_max'], tpm, tpm_alpha, tpm_beta)
    T_TI['omega_plus'] = sample_transform_param(rng, K, d, ranges['TI_omega_min'], ranges['TI_omega_max'], tpm, tpm_alpha, tpm_beta)
    T_TI['omega_minus'] = sample_transform_param(rng, K, d, ranges['TI_omega_min'], ranges['TI_omega_max'], tpm, tpm_alpha, tpm_beta)
    portal['P']['T_tensor_interference'] = T_TI

    # Radial Polytrig
    T_RP = {}
    T_RP['mu'] = ranges['RP_mu_min'] + (ranges['RP_mu_max'] - ranges['RP_mu_min']) * rng.rand(K)
    T_RP['p'] = ranges['RP_p_min'] + (ranges['RP_p_max'] - ranges['RP_p_min']) * rng.rand(K)
    T_RP['q'] = ranges['RP_q_min'] + (ranges['RP_q_max'] - ranges['RP_q_min']) * rng.rand(K)
    T_RP['omega'] = ranges['RP_omega_min'] + (ranges['RP_omega_max'] - ranges['RP_omega_min']) * rng.rand(K)
    T_RP['eps'] = 1e-12
    portal['P']['T_radial_polytrig'] = T_RP


def precompute_scalings(portal: Dict[str, Any]):
    """Precompute scaling factors based on baseline forms."""
    K = portal['K']
    d = portal['d']
    P = portal['P']

    form = np.zeros(K, dtype=int)
    rho_single = np.full(K, np.nan)
    rho_plus = np.full((K, d), np.nan)
    rho_minus = np.full((K, d), np.nan)

    def _p_single_at(k: int) -> float:
        arr = np.asarray(P['p_single'])
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1:
            return float(arr[k])
        return float(arr[k, 0])

    def _neutralize_at(k: int) -> bool:
        enabled = np.asarray(P.get('neutralize_enable', True), dtype=bool).reshape(-1)
        if enabled.size == 0:
            return True
        if enabled.size == 1:
            return bool(enabled[0])
        if enabled.size != K:
            raise ValueError(
                "neutralize_enable must be scalar or have one value per component."
            )
        return bool(enabled[k])

    for k in range(K):
        # Determine form based on baseline
        if portal['baseline'][k] == 'A':
            form[k] = 1
        else:
            form[k] = 2

        # Validate the mathematical inputs even when neutralisation is disabled.
        # Clamping an invalid exponent in a denominator would silently evaluate
        # a different function and can conceal invalid GUI/imported settings.
        positive_inputs = {
            'kappa_plus': np.asarray(P['kappa_plus'])[k, :],
            'kappa_minus': np.asarray(P['kappa_minus'])[k, :],
        }
        if form[k] == 1:
            positive_inputs.update({
                'p_plus': np.asarray(P['p_plus'])[k, :],
                'p_minus': np.asarray(P['p_minus'])[k, :],
            })
        else:
            positive_inputs['p_single'] = np.asarray(_p_single_at(k))
        for name, values in positive_inputs.items():
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(
                    f"{name} must contain finite values > 0 for component {k + 1}."
                )

        if _neutralize_at(k):
            delta_k = float(np.asarray(P['Delta']).reshape(-1)[k])
            r_ref_k = float(np.asarray(P['r_ref']).reshape(-1)[k])
            if not np.isfinite(delta_k) or delta_k <= 0.0:
                raise ValueError(
                    f"Enabled neutralization requires finite Delta > 0 for "
                    f"component {k + 1}; received {delta_k!r}."
                )
            if not np.isfinite(r_ref_k) or r_ref_k <= 0.0:
                raise ValueError(
                    f"Enabled neutralization requires finite r_ref > 0 for "
                    f"component {k + 1}; received {r_ref_k!r}."
                )
            try:
                with np.errstate(over='raise', divide='raise', invalid='raise'):
                    if form[k] == 1:  # Form A
                        bar_kappa_ki = (P['kappa_plus'][k, :] + P['kappa_minus'][k, :]) / 2
                        bar_kappa_k = np.mean(bar_kappa_ki)
                        for i in range(d):
                            rho_plus[k, i] = (delta_k / d) / (bar_kappa_k * r_ref_k ** (2 * P['p_plus'][k, i]))
                            rho_minus[k, i] = (delta_k / d) / (bar_kappa_k * r_ref_k ** (2 * P['p_minus'][k, i]))
                    else:  # Form B
                        bar_kappa_ki = (P['kappa_plus'][k, :] + P['kappa_minus'][k, :]) / 2
                        p = _p_single_at(k)
                        sum_term = np.sum(bar_kappa_ki ** (1 / max(p, np.finfo(float).eps)))
                        rho_single[k] = delta_k / ((r_ref_k ** 2 * sum_term) ** max(p, np.finfo(float).eps))
            except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
                raise ValueError(
                    f"Cannot represent scaling factors for component {k + 1}; "
                    "choose conditioning, exponents and reference controls "
                    "that do not overflow or divide by zero."
                ) from exc
        else:
            if form[k] == 1:
                rho_plus[k, :] = 1
                rho_minus[k, :] = 1
            else:
                rho_single[k] = 1
        active_scales = (np.concatenate((rho_plus[k], rho_minus[k]))
                         if form[k] == 1 else np.asarray(rho_single[k]))
        if not np.all(np.isfinite(active_scales)) or np.any(active_scales <= 0):
            raise ValueError(
                f"Computed scaling factors for component {k + 1} must be "
                "finite and greater than zero; these settings cannot be represented."
            )

    return form, rho_single, rho_plus, rho_minus
