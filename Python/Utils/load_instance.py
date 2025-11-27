"""
********************************** PORTAL Load Instance Function **********************************
Author: Delaram Yazdani, Danial Yazdani, Mai Peng
Email: delaram.yazdani@yahoo.com
       danial.yazdani@gmail.com
       pengmai1998@gmail.com
Last Edited: 2025-11-28
Title: Load PORTAL Instance from JSON

Description:
    Loads a PORTAL benchmark instance from a JSON file

Inputs:
    filename: Filename or full path to the JSON file

Output:
    portal_instance: PORTAL benchmark instance dictionary
************************************************************************************************
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import numpy as np


def load_instance(filename: str) -> Dict[str, Any]:
    """
    Load a PORTAL benchmark instance from a JSON file.
    
    Args:
        filename: Filename or full path to the JSON file
        
    Returns:
        PORTAL benchmark instance dictionary
    """
    # Determine file path
    file_path = Path(filename)
    if not file_path.exists():
        # Try in Instances directory
        script_dir = Path(__file__).parent
        instances_dir = script_dir / 'Instances'
        file_path = instances_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Instance file not found: {filename}")
    
    # Load from JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as fid:
            portal = json.load(fid)
        print(f"Instance loaded successfully from:\n  {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load instance: {str(e)}")
    
    # Convert lists back to numpy arrays
    portal = _convert_lists_to_arrays(portal)
    
    # Reconstruct RNG (stored as Seed in JSON)
    if 'Seed' in portal:
        portal['Rng'] = np.random.RandomState(portal['Seed'])
    else:
        print("Warning: Seed not found in loaded instance. Creating new RNG.")
        portal['Rng'] = np.random.RandomState(1234)
    
    # Reconstruct rotation angles from Psi_deg_cell
    if 'P' in portal and 'Psi_deg_cell' in portal['P']:
        Psi_deg_cell = portal['P']['Psi_deg_cell']
        K = portal['K']
        d = portal['d']
        
        # Convert to 3D array
        Psi_deg_temp = np.zeros((d, d, K))
        for k in range(K):
            Psi_deg_temp[:, :, k] = np.array(Psi_deg_cell[k])
        
        portal['P']['Psi_deg'] = Psi_deg_temp
        del portal['P']['Psi_deg_cell']
        
        # Compute rotation matrices R from Psi_deg
        R_temp = np.zeros((d, d, K))
        for k in range(K):
            Psi_rad = np.deg2rad(Psi_deg_temp[:, :, k])
            R_temp[:, :, k] = givens_from_psi(Psi_rad)
        portal['P']['R'] = R_temp
    
    # Fix all parameter dimensions (especially for K=1 case)
    portal['P'] = fix_parameter_dimensions(portal['P'], portal['K'], portal['d'])
    
    # Display instance info
    print(f"Loaded PORTAL Instance:")
    print(f"  Dimension: {portal['d']}")
    print(f"  Components: {portal['K']}")
    print(f"  Seed: {portal['Seed']}")
    
    if 'baseline' in portal:
        print(f"  Baselines: {' '.join(portal['baseline'])}")
    
    if 'transform_seq' in portal:
        print(f"  Transformations: ", end='')
        for k in range(portal['K']):
            transforms = portal['transform_seq'][k]
            if isinstance(transforms, list):
                print("{", end='')
                print("→".join([f"'{t}'" for t in transforms]), end='')
                print("}", end='')
        print()
    
    return portal


def givens_from_psi(Psi: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix R from upper-triangular angle matrix Psi
    using Givens rotations.
    
    Args:
        Psi: d×d upper-triangular matrix of rotation angles (radians)
        
    Returns:
        R: d×d orthogonal rotation matrix
    """
    d = Psi.shape[0]
    R = np.eye(d)
    
    for u in range(d - 1):
        for v in range(u + 1, d):
            psi = Psi[u, v]
            if not np.isfinite(psi) or abs(psi) < 1e-15:
                continue
            
            G = np.eye(d)
            c = np.cos(psi)
            s = np.sin(psi)
            G[u, u] = c
            G[u, v] = -s
            G[v, u] = s
            G[v, v] = c
            R = R @ G
    
    return R


def fix_parameter_dimensions(P: Dict[str, Any], K: int, d: int) -> Dict[str, Any]:
    """
    Fix parameter dimensions after JSON load.
    When K=1, JSON may compress dimensions. This function ensures
    all parameters have the correct shape.
    
    Args:
        P: Parameter dictionary
        K: Number of components
        d: Dimension
        
    Returns:
        Fixed parameter dictionary
    """
    # Fix K×d matrices (should be K rows, d columns)
    kd_fields = ['c', 'kappa_plus', 'kappa_minus', 'p_plus', 'p_minus', 
                 'rho_plus', 'rho_minus']
    for field in kd_fields:
        if field in P and P[field] is not None:
            P[field] = ensure_size(P[field], K, d)
    
    # Fix K×1 vectors (should be K elements)
    k1_fields = ['beta', 'Delta', 'r_ref', 'rho_single', 'form']
    for field in k1_fields:
        if field in P and P[field] is not None:
            arr = ensure_size(P[field], K, 1)
            P[field] = arr.ravel() if arr.size > 0 else arr
    
    # p_single is special: MATLAB stores as 1D array, but internally we use K×1
    if 'p_single' in P and P['p_single'] is not None:
        p_single = np.array(P['p_single'])
        # If it's already a simple K-element array, reshape to K×1 for internal use
        if p_single.ndim == 1 and len(p_single) == K:
            P['p_single'] = p_single.reshape(K, 1)
        else:
            # Otherwise ensure proper sizing
            P['p_single'] = ensure_size(P['p_single'], K, 1)
    
    # Fix 3D arrays: Psi_deg and R (d×d×K)
    if 'Psi_deg' in P:
        P['Psi_deg'] = ensure_size_3d(P['Psi_deg'], d, d, K)
    
    if 'R' in P:
        P['R'] = ensure_size_3d(P['R'], d, d, K)
    
    # Fix transform parameters
    if 'T_additive_periodic' in P:
        T = P['T_additive_periodic']
        T['mu_plus'] = ensure_size(T['mu_plus'], K, d)
        T['mu_minus'] = ensure_size(T['mu_minus'], K, d)
        T['gamma_plus'] = ensure_size(T['gamma_plus'], K, d)
        T['gamma_minus'] = ensure_size(T['gamma_minus'], K, d)
        T['omega_plus'] = ensure_size(T['omega_plus'], K, d)
        T['omega_minus'] = ensure_size(T['omega_minus'], K, d)
        P['T_additive_periodic'] = T
    
    if 'T_log_sinusoidal' in P:
        T = P['T_log_sinusoidal']
        T['mu_plus'] = ensure_size(T['mu_plus'], K, d)
        T['mu_minus'] = ensure_size(T['mu_minus'], K, d)
        T['omega1_plus'] = ensure_size(T['omega1_plus'], K, d)
        T['omega2_plus'] = ensure_size(T['omega2_plus'], K, d)
        T['omega1_minus'] = ensure_size(T['omega1_minus'], K, d)
        T['omega2_minus'] = ensure_size(T['omega2_minus'], K, d)
        P['T_log_sinusoidal'] = T
    
    if 'T_wavelet_mod' in P:
        T = P['T_wavelet_mod']
        T['mu_plus'] = ensure_size(T['mu_plus'], K, d)
        T['mu_minus'] = ensure_size(T['mu_minus'], K, d)
        T['omega_plus'] = ensure_size(T['omega_plus'], K, d)
        T['omega_minus'] = ensure_size(T['omega_minus'], K, d)
        T['L_plus'] = ensure_size(T['L_plus'], K, d)
        T['L_minus'] = ensure_size(T['L_minus'], K, d)
        arr = ensure_size(T['eta'], K, 1)
        T['eta'] = arr.ravel() if arr.size > 0 else arr
        P['T_wavelet_mod'] = T
    
    if 'T_tensor_interference' in P:
        T = P['T_tensor_interference']
        T['mu_plus'] = ensure_size(T['mu_plus'], K, d)
        T['mu_minus'] = ensure_size(T['mu_minus'], K, d)
        T['omega_plus'] = ensure_size(T['omega_plus'], K, d)
        T['omega_minus'] = ensure_size(T['omega_minus'], K, d)
        P['T_tensor_interference'] = T
    
    if 'T_radial_polytrig' in P:
        T = P['T_radial_polytrig']
        for field in ['mu', 'p', 'q', 'omega']:
            arr = ensure_size(T[field], K, 1)
            T[field] = arr.ravel() if arr.size > 0 else arr
        P['T_radial_polytrig'] = T
    
    return P


def ensure_size(A: Any, rows: int, cols: int) -> np.ndarray:
    """
    Ensure matrix A has size [rows × cols].
    Handles cases where JSON compression changed dimensions.
    
    Args:
        A: Input array (can be list, scalar, or array)
        rows: Target number of rows
        cols: Target number of columns
        
    Returns:
        Array with correct shape
    """
    A = np.atleast_1d(np.array(A, dtype=float))
    
    if A.size == 0:
        return np.zeros((rows, cols))
    
    current_shape = A.shape
    
    # If already correct size
    if current_shape == (rows, cols):
        return A
    
    # If scalar and should be rows×cols
    if A.size == 1:
        return np.full((rows, cols), A.item())
    
    # If row vector when should be column vector
    if current_shape == (1, rows) and cols == 1:
        return A.T
    
    # If column vector when should be row vector
    if current_shape == (rows, 1) and cols > 1:
        return np.tile(A, (1, cols))
    
    # If dimensions are swapped
    if current_shape == (cols, rows):
        return A.T
    
    # If it's a row vector [1×cols] but should be [rows×cols] with rows=1
    if current_shape == (1, cols) and rows == 1:
        return A
    
    # If it's a 1D array of correct length for either dimension
    if A.ndim == 1:
        if A.size == rows * cols:
            return A.reshape(rows, cols)
        elif A.size == rows and cols == 1:
            return A.reshape(rows, 1)
        elif A.size == cols and rows == 1:
            return A.reshape(1, cols)
    
    # If total elements match, try to reshape
    if A.size == rows * cols:
        return A.reshape(rows, cols)
    
    # If we get here, something unexpected happened
    print(f"Warning: Cannot reshape array from {current_shape} to [{rows}×{cols}]")
    return A


def ensure_size_3d(A: Any, dim1: int, dim2: int, dim3: int) -> np.ndarray:
    """
    Ensure 3D array A has size [dim1 × dim2 × dim3].
    Handles cases where JSON compression changed dimensions for K=1.
    
    Args:
        A: Input array
        dim1, dim2, dim3: Target dimensions
        
    Returns:
        Reshaped array with correct dimensions
    """
    A = np.array(A, dtype=float)
    
    if A.size == 0:
        return np.zeros((dim1, dim2, dim3))
    
    current_shape = A.shape
    
    # If already correct size
    if current_shape == (dim1, dim2, dim3):
        return A
    
    # Case 1: K=1 and array was compressed from [d×d×1] to [d×d]
    if A.ndim == 2 and dim3 == 1 and current_shape == (dim1, dim2):
        return A.reshape(dim1, dim2, 1)
    
    # Case 2: Array is already 3D but dimensions match by element count
    if A.size == dim1 * dim2 * dim3:
        return A.reshape(dim1, dim2, dim3)
    
    # If we get here, something unexpected happened
    print(f"Warning: Cannot reshape 3D array from {current_shape} to [{dim1}×{dim2}×{dim3}]")
    return A


def _convert_lists_to_arrays(obj: Any) -> Any:
    """
    Recursively convert lists to numpy arrays where appropriate.
    
    Args:
        obj: Object to convert (can be dict, list, etc.)
        
    Returns:
        Converted object
    """
    if isinstance(obj, dict):
        # Don't convert certain fields that should remain as lists
        result = {}
        for key, value in obj.items():
            if key in ['baseline', 'transform_seq']:
                result[key] = value  # Keep as list
            else:
                result[key] = _convert_lists_to_arrays(value)
        return result
    elif isinstance(obj, list):
        # Check if it's a numeric list
        try:
            # Try to convert to array
            arr = np.array(obj)
            if arr.dtype.kind in ['i', 'f']:  # integer or float
                return arr
            else:
                return obj
        except (ValueError, TypeError):
            return obj
    else:
        return obj