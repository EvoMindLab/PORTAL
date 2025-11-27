"""
Author: Delaram Yazdani, Danial Yazdani, Mai Peng
Email: delaram.yazdani@yahoo.com
       danial.yazdani@gmail.com
       pengmai1998@gmail.com
Last Edited: 2025-11-28
Title: PORTAL Fitness Evaluation

Description:
    Evaluates the fitness of a solution vector using the PORTAL benchmark

Inputs:
    x: Solution vector (d×1 or 1D array)
    portal_instance: PORTAL benchmark instance dictionary

Output:
    f: Fitness value (scalar)
"""

import numpy as np
from typing import Dict, Any, List, Union


def fitness(x: np.ndarray, portal_instance: Dict[str, Any]) -> float:
    """
    Evaluate the fitness of a solution using the PORTAL benchmark.
    
    Args:
        x: Solution vector (d×1 or 1D array)
        portal_instance: PORTAL benchmark instance dictionary
        
    Returns:
        Fitness value (scalar)
    """
    # Ensure x is a column vector
    x = np.atleast_1d(x).flatten()
    
    # Check dimension
    if len(x) != portal_instance['d']:
        raise ValueError(f"Solution dimension mismatch. Expected {portal_instance['d']}, got {len(x)}")
    
    # Check bounds
    bounds_min = portal_instance['bounds_min']
    bounds_max = portal_instance['bounds_max']
    if np.any(x < bounds_min) or np.any(x > bounds_max):
        print("Warning: Solution out of bounds. Clamping to feasible region.")
        x = np.clip(x, bounds_min, bounds_max)
    
    # Evaluate all components and take minimum
    K = portal_instance['K']
    vals = np.zeros(K)
    
    for k in range(K):
        seq_k = seq_for_component(portal_instance['transform_seq'], k, K)
        vals[k] = phi_component(x, portal_instance['P'], portal_instance, k, seq_k)
    
    return np.min(vals)


def phi_component(x: np.ndarray, P: Dict[str, Any], portal: Dict[str, Any], 
                  k: int, seq_k: List[str]) -> float:
    """
    Evaluate a single component.
    
    Args:
        x: Solution vector
        P: Parameters dictionary
        portal: PORTAL instance
        k: Component index
        seq_k: Transform sequence for this component
        
    Returns:
        Component fitness value
    """
    d = portal['d']
    ck = P['c'][k, :].reshape(-1, 1)
    
    # Internal coordinate transformation
    x_col = x.reshape(-1, 1)
    a = P['R'][:, :, k] @ (x_col - ck)
    
    # Apply transformation sequence
    z = apply_transform_seq(a, P, seq_k, k)
    
    # Compute fitness based on form
    form = P['form'][k]
    
    if form == 1:  # Form A
        acc = 0.0
        for i in range(d):
            if z[i, 0] >= 0:
                h = P['kappa_plus'][k, i]
                p_i = P['p_plus'][k, i]
                rho_i = P['rho_plus'][k, i]
            else:
                h = P['kappa_minus'][k, i]
                p_i = P['p_minus'][k, i]
                rho_i = P['rho_minus'][k, i]
            acc += rho_i * h * (abs(z[i, 0]) ** (2 * p_i))
        return P['beta'][k] + acc
    
    else:  # Form B (form == 2)
        qsum = 0.0
        p_single = P['p_single'][k]
        for i in range(d):
            if z[i, 0] >= 0:
                h = P['kappa_plus'][k, i]
            else:
                h = P['kappa_minus'][k, i]
            qsum += h ** (1 / max(p_single, np.finfo(float).eps)) * (abs(z[i, 0]) ** 2)
        return P['beta'][k] + P['rho_single'][k] * (qsum ** p_single)


def apply_transform_seq(a: np.ndarray, P: Dict[str, Any], seq: List[str], k: int) -> np.ndarray:
    """
    Apply a sequence of transformations.
    
    Args:
        a: Input vector
        P: Parameters dictionary
        seq: List of transform names
        k: Component index
        
    Returns:
        Transformed vector
    """
    z = a.copy()
    for transform_name in seq:
        z = apply_transform_once(z, P, transform_name, k)
    return z


def apply_transform_once(a: np.ndarray, P: Dict[str, Any], name: str, k: int) -> np.ndarray:
    """
    Apply a single transformation.
    
    Args:
        a: Input vector (d×1)
        P: Parameters dictionary
        name: Transform name
        k: Component index
        
    Returns:
        Transformed vector (d×1)
    """
    d = a.shape[0]
    z = a.copy()
    epslog = 1e-12
    
    name_lower = name.lower()
    
    if name_lower == 'none':
        return z
    
    elif name_lower == 'additive_periodic':
        for i in range(d):
            ai = a[i, 0]
            if ai >= 0:
                mu = P['T_additive_periodic']['mu_plus'][k, i]
                gam = P['T_additive_periodic']['gamma_plus'][k, i]
                om = P['T_additive_periodic']['omega_plus'][k, i]
            else:
                mu = P['T_additive_periodic']['mu_minus'][k, i]
                gam = P['T_additive_periodic']['gamma_minus'][k, i]
                om = P['T_additive_periodic']['omega_minus'][k, i]
            g = ai * (1 - np.exp(-gam * abs(ai))) * np.sin(om * abs(ai))
            z[i, 0] = ai + mu * g
    
    elif name_lower == 'log_sinusoidal':
        for i in range(d):
            ai = a[i, 0]
            if ai == 0:
                z[i, 0] = 0
                continue
            
            abs_ai = abs(ai)
            L = np.log(abs_ai + epslog)
            if ai > 0:
                mu = P['T_log_sinusoidal']['mu_plus'][k, i]
                o1 = P['T_log_sinusoidal']['omega1_plus'][k, i]
                o2 = P['T_log_sinusoidal']['omega2_plus'][k, i]
            else:
                mu = P['T_log_sinusoidal']['mu_minus'][k, i]
                o1 = P['T_log_sinusoidal']['omega1_minus'][k, i]
                o2 = P['T_log_sinusoidal']['omega2_minus'][k, i]
            modulation = mu * (np.sin(o1 * L) + np.sin(o2 * L))
            z[i, 0] = np.sign(ai) * np.exp(L + modulation)
    
    elif name_lower == 'wavelet_mod':
        eta = P['T_wavelet_mod']['eta'][k]
        epss = P['T_wavelet_mod']['eps']
        for i in range(d):
            ai = a[i, 0]
            t = abs(ai)
            if t < epss:
                z[i, 0] = ai
                continue
            
            if ai >= 0:
                mu = P['T_wavelet_mod']['mu_plus'][k, i]
                omg = P['T_wavelet_mod']['omega_plus'][k, i]
                L = P['T_wavelet_mod']['L_plus'][k, i]
                if L is None or L <= 0:
                    L = eta / max(omg, epss)
            else:
                mu = P['T_wavelet_mod']['mu_minus'][k, i]
                omg = P['T_wavelet_mod']['omega_minus'][k, i]
                L = P['T_wavelet_mod']['L_minus'][k, i]
                if L is None or L <= 0:
                    L = eta / max(omg, epss)
            
            if omg <= 0 or mu == 0:
                z[i, 0] = ai
                continue
            
            env = (t / L) ** 2 * np.exp(-(t / L) ** 2)
            phi = env * np.sin(omg * t - np.pi / 2)
            z[i, 0] = ai + mu * phi * (ai / (t + epss))
    
    elif name_lower == 'tensor_interference':
        for i in range(d):
            ai = a[i, 0]
            if ai >= 0:
                mu_i = P['T_tensor_interference']['mu_plus'][k, i]
                om_i = P['T_tensor_interference']['omega_plus'][k, i]
            else:
                mu_i = P['T_tensor_interference']['mu_minus'][k, i]
                om_i = P['T_tensor_interference']['omega_minus'][k, i]
            
            if mu_i == 0 or om_i <= 0:
                z[i, 0] = ai
                continue
            
            Phi_even = 1.0
            for j in range(d):
                if j == i:
                    continue
                aj = a[j, 0]
                if aj >= 0:
                    om_j = P['T_tensor_interference']['omega_plus'][k, j]
                else:
                    om_j = P['T_tensor_interference']['omega_minus'][k, j]
                Phi_even *= np.sin(om_j * aj) ** 2
            
            scaled_mu = mu_i * (2 ** (d - 1))
            z[i, 0] = ai + scaled_mu * np.sin(om_i * ai) * Phi_even
    
    elif name_lower == 'radial_polytrig':
        mu = P['T_radial_polytrig']['mu'][k]
        p = P['T_radial_polytrig']['p'][k]
        q = P['T_radial_polytrig']['q'][k]
        om = P['T_radial_polytrig']['omega'][k]
        epss = P['T_radial_polytrig']['eps']
        r = np.linalg.norm(a)
        if r < epss:
            return a
        phi = r ** p * np.sin(om * r ** q)
        z = a + mu * phi * (a / (r + epss))
    
    return z


def seq_for_component(seq_cfg: List[List[str]], k: int, K: int) -> List[str]:
    """
    Get transformation sequence for a component.
    
    Args:
        seq_cfg: List of transformation sequences (one per component)
        k: Component index
        K: Total number of components
        
    Returns:
        List of transform names for this component
    """
    if not isinstance(seq_cfg, list) or len(seq_cfg) != K:
        raise ValueError(f"transform_seq must be a list of length K={K}; each entry a list of transform names.")
    
    item = seq_cfg[k]
    if item is None or len(item) == 0:
        return []
    elif isinstance(item, list):
        return [str(t) for t in item]
    elif isinstance(item, str):
        return [item]
    else:
        raise ValueError(f"transform_seq[{k}] must be list, string, or empty.")