"""
Author: Delaram Yazdani, Danial Yazdani, Mai Peng
Email: delaram.yazdani@yahoo.com
       danial.yazdani@gmail.com
       pengmai1998@gmail.com
Last Edited: 2025-11-28
Title: PORTAL Landscape Visualization

Description:
    Visualizes the PORTAL benchmark landscape with surface, contour, and combined plots

Inputs:
    portal_instance: PORTAL benchmark instance dictionary
    resolution: (optional) Grid resolution for plotting (default: 250)
    show_contour: (optional) Show contour lines on surface plot (default: True)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, Optional
from Utils.fitness import fitness


def plot_landscape(portal_instance: Dict[str, Any], 
                   resolution: int = 250, 
                   show_contour: bool = True) -> None:
    """
    Visualize the PORTAL benchmark landscape.
    
    Args:
        portal_instance: PORTAL benchmark instance dictionary
        resolution: Grid resolution for plotting (default: 250)
        show_contour: Show contour lines on surface plot (default: True)
    """
    # Check dimension
    if portal_instance['d'] != 2:
        print(f"Warning: Visualization requires d=2. Current dimension is {portal_instance['d']}. Skipping plot.")
        return
    
    # Grid evaluation
    xb = [portal_instance['bounds_min'], portal_instance['bounds_max']]
    x1_vec = np.linspace(xb[0], xb[1], resolution)
    x2_vec = np.linspace(xb[0], xb[1], resolution)
    x1, x2 = np.meshgrid(x1_vec, x2_vec)
    F = np.zeros((resolution, resolution))
    
    print(f'Evaluating landscape on {resolution}x{resolution} grid...')
    for i in range(resolution):
        for j in range(resolution):
            x = np.array([x1[i, j], x2[i, j]])
            F[i, j] = fitness(x, portal_instance)
    
    # Replace grid points closest to component centers with exact values
    P = portal_instance['P']
    K = portal_instance['K']
    for k in range(K):
        dist_sq = (x1 - P['c'][k, 0])**2 + (x2 - P['c'][k, 1])**2
        min_idx = np.argmin(dist_sq)
        min_i, min_j = np.unravel_index(min_idx, dist_sq.shape)
        x1[min_i, min_j] = P['c'][k, 0]
        x2[min_i, min_j] = P['c'][k, 1]
        F[min_i, min_j] = P['beta'][k]
    
    # Find min/max for contour placement
    minF = np.min(F)
    maxF = np.max(F)
    
    print(f'Fitness range: [{minF:.4f}, {maxF:.4f}]')
    print('Generating plots...')
    
    # Figure positioning
    seed = portal_instance['Seed']
    
    # ---- SURFACE PLOT ----
    fig1 = plt.figure(figsize=(10, 9))
    fig1.canvas.manager.set_window_title(f'PORTAL Surface (K={K}, Seed={seed})')
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Plot surface
    surf1 = ax1.plot_surface(x1, x2, F, cmap=cm.viridis, alpha=0.8, 
                             linewidth=0, antialiased=True, edgecolor='none')
    
    if show_contour:
        # Add contour lines
        ax1.contour(x1, x2, F, levels=40, colors='k', linewidths=0.6, alpha=0.6)
    
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_zlabel('f(x₁, x₂)')
    ax1.set_title(f'PORTAL Surface (K={K}, Seed={seed})')
    ax1.view_init(elev=30, azim=240)
    ax1.grid(True)
    
    # Set tight axis limits to remove gaps
    ax1.set_xlim([xb[0], xb[1]])
    ax1.set_ylim([xb[0], xb[1]])
    ax1.set_zlim([minF, maxF])
    
    # Make axes equal aspect
    ax1.set_box_aspect([1, 1, 1])
    
    # ---- CONTOUR PLOT WITH COLORBAR ----
    fig2 = plt.figure(figsize=(10, 9))
    fig2.canvas.manager.set_window_title(f'PORTAL Contour (K={K}, Seed={seed})')
    ax2 = fig2.add_subplot(111)
    
    # Plot contours
    contour2 = ax2.contour(x1, x2, F, levels=40, linewidths=1)
    
    # Mark component centers
    for k in range(K):
        ax2.plot(P['c'][k, 0], P['c'][k, 1], 'r*', markersize=10, 
                markeredgewidth=2, label=f'Center {k+1}' if k == 0 else '')
    
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title(f'PORTAL Contour (K={K}, Seed={seed})')
    
    # Add colorbar
    cbar2 = plt.colorbar(contour2, ax=ax2)
    cbar2.set_label('f(x₁, x₂)')
    
    # ---- COMBINED SURFACE + BASE CONTOUR ----
    fig3 = plt.figure(figsize=(10, 9))
    fig3.canvas.manager.set_window_title(f'PORTAL Combined (K={K}, Seed={seed})')
    ax3 = fig3.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf3 = ax3.plot_surface(x1, x2, F, cmap=cm.viridis, alpha=0.8, 
                             linewidth=0, antialiased=True, edgecolor='none')
    
    # Add colored contours at the base
    nLevels = 40
    levelList = np.linspace(minF, maxF, nLevels)
    colorMap = cm.viridis(np.linspace(0, 1, nLevels))
    zBase = minF - 0.01 * (maxF - minF)
    
    # Plot contours at base
    for i, level in enumerate(levelList):
        contour_lines = ax3.contour(x1, x2, F, levels=[level], 
                                    colors=[colorMap[i]], linewidths=1,
                                    offset=zBase, zdir='z')
    
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.set_zlabel('f(x₁, x₂)')
    ax3.set_title(f'PORTAL Combined (K={K}, Seed={seed})')
    ax3.view_init(elev=30, azim=240)
    ax3.grid(True)
    
    # Set tight axis limits to remove gaps
    ax3.set_xlim([xb[0], xb[1]])
    ax3.set_ylim([xb[0], xb[1]])
    ax3.set_zlim([zBase, maxF])
    
    # Make axes equal aspect
    ax3.set_box_aspect([1, 1, 1])
    
    print('Plotting complete!')
    plt.show()


def ask_and_plot(portal_instance: Dict[str, Any], test_name: str) -> None:
    """
    Helper function for interactive plotting.
    
    Args:
        portal_instance: PORTAL benchmark instance dictionary
        test_name: Name of the test
    """
    print(f'\n[{test_name}] completed.')
    response = input('Plot this instance? (y/n): ').strip().lower()
    if response == 'y':
        plot_landscape(portal_instance, 200, True)
        print('Plotting complete. Close figures to continue.')
    print()