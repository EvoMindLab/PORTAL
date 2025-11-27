"""
********************************** PORTAL Save Instance Function **********************************
Author: Delaram Yazdani, Danial Yazdani, Mai Peng
Email: delaram.yazdani@yahoo.com
       danial.yazdani@gmail.com
       pengmai1998@gmail.com
Last Edited: 2025-11-28
Title: Save PORTAL Instance to JSON

Description:
    Saves a PORTAL benchmark instance to a JSON file (default: BenchmarkGenerator_Python/Instances,
    override with output_dir)

Inputs:
    portal_instance: PORTAL benchmark instance dictionary
    filename: (optional) Filename for the JSON file (default: auto-generated)
    output_dir: (optional) Custom directory for saving instances

Output:
    filepath: Full path to the saved file
************************************************************************************************
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import copy


def save_instance(
    portal_instance: Dict[str, Any],
    filename: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Save a PORTAL benchmark instance to a JSON file.
    
    Args:
        portal_instance: PORTAL benchmark instance dictionary
        filename: Optional filename for the JSON file
        
    Returns:
        Full path to the saved file
    """
    # Determine filename
    if filename is None or filename == "":
        filename = f"PORTAL_d{portal_instance['d']}_K{portal_instance['K']}_seed{portal_instance['Seed']}.json"
    
    # Ensure .json extension
    if not filename.endswith('.json'):
        filename = Path(filename).stem + '.json'
    
    # Determine save path
    script_dir = Path(__file__).parent
    instances_dir = Path(output_dir) if output_dir else script_dir / 'Instances'
    
    # Create Instances directory if it doesn't exist
    instances_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = instances_dir / filename
    
    # Convert PORTAL structure to JSON-compatible format
    # Use deep copy to avoid modifying the original instance
    portal_save = copy.deepcopy(portal_instance)
    
    # Remove RNG object (not JSON serializable)
    if 'Rng' in portal_save:
        del portal_save['Rng']
    
    # Convert Psi_deg (rotation angles) to list format for JSON
    if 'P' in portal_save and 'Psi_deg' in portal_save['P']:
        K = portal_instance['K']
        Psi_deg_temp = portal_save['P']['Psi_deg']
        Psi_deg_cell = []
        for k in range(K):
            Psi_deg_cell.append(Psi_deg_temp[:, :, k].tolist())
        portal_save['P']['Psi_deg_cell'] = Psi_deg_cell
        del portal_save['P']['Psi_deg']
    
    # Remove R matrices (rotation matrices) - will be computed from Psi_deg when loaded
    if 'P' in portal_save and 'R' in portal_save['P']:
        del portal_save['P']['R']
    
    # Convert all numpy arrays to lists for JSON serialization
    portal_save = _convert_arrays_to_lists(portal_save)
    
    # Fix p_single format: flatten from [[val1], [val2], ...] to [val1, val2, ...]
    # MATLAB stores p_single as a simple K-element vector, not K×1 nested arrays
    if 'P' in portal_save and 'p_single' in portal_save['P']:
        p_single = portal_save['P']['p_single']
        if isinstance(p_single, list) and len(p_single) > 0:
            # Check if it's nested (e.g., [[0.5], [0.8], ...])
            if isinstance(p_single[0], list) and len(p_single[0]) == 1:
                # Flatten: [[val1], [val2]] -> [val1, val2]
                portal_save['P']['p_single'] = [item[0] for item in p_single]
    
    # Reorder JSON to match MATLAB structure
    portal_save = _reorder_for_matlab_compatibility(portal_save)
    
    # Save to JSON
    try:
        with open(filepath, 'w', encoding='utf-8') as fid:
            json.dump(portal_save, fid, indent=2, ensure_ascii=False)
        print(f"Instance saved successfully to:\n  {filepath}")
        return str(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to save instance: {str(e)}")


def _convert_arrays_to_lists(obj: Any) -> Any:
    """
    Recursively convert numpy arrays to lists for JSON serialization.
    
    Args:
        obj: Object to convert (can be dict, list, array, etc.)
        
    Returns:
        Converted object with arrays as lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_arrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_arrays_to_lists(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def _reorder_for_matlab_compatibility(portal_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reorder dictionary keys to match MATLAB's JSON output structure.
    MATLAB's jsonencode produces a specific field order that we replicate here.
    
    Args:
        portal_dict: PORTAL instance dictionary
        
    Returns:
        Reordered dictionary matching MATLAB structure
    """
    # Top-level field order (MATLAB jsonencode order)
    top_order = ['Seed', 'd', 'K', 'bounds_min', 'bounds_max', 'baseline', 'transform_seq', 'ranges', 'P']
    
    # Reorder top-level fields
    ordered = {}
    for key in top_order:
        if key in portal_dict:
            ordered[key] = portal_dict[key]
    
    # Add any remaining fields not in the predefined order
    for key in portal_dict:
        if key not in ordered:
            ordered[key] = portal_dict[key]
    
    # Reorder P subfields to match MATLAB structure
    if 'P' in ordered and isinstance(ordered['P'], dict):
        p_order = [
            'c', 'beta', 'kappa_plus', 'kappa_minus', 'p_single', 'p_plus', 'p_minus',
            'neutralize_enable', 'Delta', 'r_ref',
            'T_additive_periodic', 'T_log_sinusoidal', 'T_wavelet_mod', 
            'T_tensor_interference', 'T_radial_polytrig',
            'form', 'rho_single', 'rho_plus', 'rho_minus', 'Psi_deg_cell'
        ]
        
        p_ordered = {}
        for key in p_order:
            if key in ordered['P']:
                p_ordered[key] = ordered['P'][key]
        
        # Add any remaining P fields
        for key in ordered['P']:
            if key not in p_ordered:
                p_ordered[key] = ordered['P'][key]
        
        ordered['P'] = p_ordered
    
    return ordered
