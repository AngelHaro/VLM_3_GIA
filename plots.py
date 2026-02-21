"""
Vortex Lattice Method (VLM) - Plotting Module

This module provides plotting utilities for visualizing VLM aerodynamic results.
All plots follow a clean, technical style suitable for engineering reports.

Functions:
    plot_global_coefficients: Global aerodynamic coefficients (2x2 subplot layout)
    plot_spanwise_all: All spanwise distributions (2x2 subplot layout)
    plot_all: Master function that displays all plots
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_global_coefficients(results):
    """
    Plot global aerodynamic coefficients in a 2x2 subplot layout.
    
    Creates a figure with four subplots showing:
    - Top-left: Lift curve (CL vs alpha)
    - Top-right: Induced drag (CDi vs alpha)
    - Bottom-left: Pitching moment (CMy_ac vs alpha)
    - Bottom-right: Aerodynamic polar (CD vs CL)
    
    Parameters
    ----------
    results : dict
        Results dictionary from AlphaSweep.run()
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("Global Aerodynamic Coefficients", fontsize=13, fontweight="bold")
    
    alpha_deg = results["alpha_deg"]
    CL = results["CL"]
    CD = results["CD"]
    CDi = results["CDi"]
    CMy_origin = results["CMy_origin"]
    CMy_ac = results["CMy_ac"]

    alpha_rad = np.deg2rad(alpha_deg)
    cl_fit = np.polyfit(alpha_rad, CL, 1)
    cl_slope = cl_fit[0]
    cl0 = cl_fit[1]
    cm_origin_slope = np.polyfit(alpha_rad, CMy_origin, 1)[0]
    cmy_ac_value = float(np.mean(CMy_ac))
    
    # ===== Top-left: Lift curve (CL vs alpha) =====
    ax = axes[0, 0]
    ax.plot(alpha_deg, CL, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Angle of Attack (α) [deg]", fontsize=7)
    ax.set_ylabel("Lift Coefficient (CL) [-]", fontsize=7)
    ax.set_title("Lift Curve", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.95,
        f"dCL/dα = {cl_slope:.4f} [1/rad]\nCL0 = {cl0:.4f}",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
    )
    
    # ===== Top-right: Induced drag (CDi vs alpha) =====
    ax = axes[0, 1]
    ax.plot(alpha_deg, CDi, "s-", linewidth=2, markersize=6, color="tab:orange")
    
    # Find minimum CDi and corresponding alpha
    idx_min = np.argmin(CDi)
    alpha_min = alpha_deg[idx_min]
    CDi_min = CDi[idx_min]
    
    # Draw vertical and horizontal dashed lines at minimum
    ax.axvline(x=alpha_min, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(y=CDi_min, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    
    # Annotate minimum point
    ax.plot(alpha_min, CDi_min, "r*", markersize=12)
    ax.annotate(f"α={alpha_min:.1f}°\nCDi={CDi_min:.4f}",
                xy=(alpha_min, CDi_min),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax.set_xlabel("Angle of Attack (α) [deg]", fontsize=7)
    ax.set_ylabel("Induced Drag Coefficient (CDi) [-]", fontsize=7)
    ax.set_title("Induced Drag", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # ===== Bottom-left: Pitching moment (About Origin vs About AC) =====
    ax = axes[1, 0]
    ax.plot(alpha_deg, CMy_origin, "o-", linewidth=2, markersize=6, color="tab:blue", label="About Origin")
    ax.plot(alpha_deg, CMy_ac, "^-", linewidth=2, markersize=6, color="tab:green", label="About Aerodynamic Center")
    ax.set_xlabel("Angle of Attack (α) [deg]", fontsize=7)
    ax.set_ylabel("Pitching Moment Coefficient (CMy) [-]", fontsize=7)
    ax.set_title("Pitching Moment", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.text(
        0.02,
        0.95,
        f"dCMy/dα = {cm_origin_slope:.4f} [1/rad]",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
    )
    ax.annotate(
        f"CMy_ac = {cmy_ac_value:.4f}",
        xy=(alpha_deg[-1], cmy_ac_value),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=7,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
    )
    ax.legend(loc="best", fontsize=8)
    
    # ===== Bottom-right: Aerodynamic polar (CD vs CL) =====
    ax = axes[1, 1]
    ax.plot(CD, CL, "d-", linewidth=2, markersize=6, color="tab:red")
    
    # Find minimum CD and corresponding CL
    idx_min_cd = np.argmin(CD)
    CD_min = CD[idx_min_cd]
    CL_at_min_cd = CL[idx_min_cd]
    
    # Draw horizontal dashed line at minimum CD
    ax.axhline(y=CL_at_min_cd, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(x=CD_min, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    
    # Mark minimum point
    ax.plot(CD_min, CL_at_min_cd, "r*", markersize=12)
    ax.annotate(f"CD={CD_min:.4f}\nCL={CL_at_min_cd:.3f}",
                xy=(CD_min, CL_at_min_cd),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax.set_xlabel("Drag Coefficient (CD) [-]", fontsize=7)
    ax.set_ylabel("Lift Coefficient (CL) [-]", fontsize=7)
    ax.set_title("Aerodynamic Polar", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Label data points with alpha values on polar
    for i, alpha in enumerate(alpha_deg):
        ax.annotate(f"{alpha:.0f}°",
            xy=(CD[i], CL[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8)


def plot_spanwise_cl(results):
    """
    Plot spanwise lift coefficient distribution for all angles of attack.
    
    Creates a single plot showing Cl(η) curves for each angle of attack.
    Spanwise coordinate is normalized as: η = y / max(|y|)
    
    Excludes the first and last control points (at wing tips) to avoid 
    artificial data at η = ±1.
    
    Parameters
    ----------
    results : dict
        Results dictionary from AlphaSweep.run()
    """
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    
    alpha_deg = results["alpha_deg"]
    spanwise_list = results["spanwise"]
    
    # Plot Cl distribution for each angle of attack
    for i_alpha, alpha in enumerate(alpha_deg):
        y = spanwise_list[i_alpha]["y"]
        cl = spanwise_list[i_alpha]["cl"]
        
        # Normalize spanwise coordinate
        y_max = np.max(np.abs(y))
        if y_max > 1e-10:
            eta = y / y_max
        else:
            eta = y
        
        # Exclude first and last points (wing tips)
        eta_plot = eta[1:-1]
        cl_plot = cl[1:-1]
        
        ax.plot(eta_plot, cl_plot, "o-", linewidth=2, markersize=7, 
               label=f"α = {alpha:.1f}°")
    
    ax.set_xlabel("Spanwise Coordinate (η = 2y/b) [-]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Lift Coefficient (Cl) [-]", fontsize=12, fontweight="bold")
    ax.set_title("Spanwise Lift Coefficient Distribution", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11, ncol=1)
    ax.set_xlim([-1, 1])
    
    return fig


def plot_spanwise_decomposition(results):
    """
    Plot spanwise load decomposition: basic (clb), additional (cla), and combined.
    
    Creates a single plot showing three spanwise distributions:
    - Clb(η): Basic (zero-lift) load distribution
    - Cla(η): Additional load distribution per unit lift
    - Cl(η) at CL = 1: Combined load at unit lift coefficient
    
    Spanwise coordinate is normalized as: η = y / max(|y|)
    
    Excludes the first and last control points (at wing tips) to avoid 
    artificial data at η = ±1.
    
    Parameters
    ----------
    results : dict
        Results dictionary from AlphaSweep.run()
    """
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    
    cl_basic = results["cl_basic"]
    y = cl_basic["y"]
    clb = cl_basic["clb"]
    cla = cl_basic["cla"]
    cl_CL1 = cl_basic["cl_CL1"]
    
    # Normalize spanwise coordinate
    y_max = np.max(np.abs(y))
    if y_max > 1e-10:
        eta = y / y_max
    else:
        eta = y
    
    # Exclude first and last points (wing tips)
    eta_plot = eta[1:-1]
    clb_plot = clb[1:-1]
    cla_plot = cla[1:-1]
    cl_CL1_plot = cl_CL1[1:-1]
    
    # Plot basic distribution (dashed line)
    ax.plot(eta_plot, clb_plot, "o--", linewidth=2, markersize=7, 
           label="Clb(η) - Basic", color="tab:blue")
    
    # Plot additional distribution (dashed line)
    ax.plot(eta_plot, cla_plot, "s--", linewidth=2, markersize=7, 
           label="Cla(η) - Additional", color="tab:orange")
    
    # Plot combined distribution at CL = 1 (solid line)
    ax.plot(eta_plot, cl_CL1_plot, "^-", linewidth=2.5, markersize=8, 
           label="Cl(η) at CL = 1.0", color="tab:green")
    
    ax.set_xlabel("Spanwise Coordinate (η = 2y/b) [-]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Lift Coefficient (Cl) [-]", fontsize=12, fontweight="bold")
    ax.set_title("Spanwise Load Decomposition", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlim([-1, 1])
    
    return fig


def plot_spanwise_all(results):
    """
    Plot all spanwise aerodynamic distributions in a 2x2 subplot layout.
    
    Creates a figure with four subplots showing:
    - Top-left: Spanwise sectional lift coefficient Cl(y)
    - Top-right: Spanwise sectional pitching moment coefficient CMy(y) about origin
    - Bottom-left: Spanwise induced velocity w_induced(y)
    - Bottom-right: Spanwise load decomposition (clb, cla, cl_CL1)
    
    Excludes the first and last control points (at wing tips) to avoid 
    artificial data at η = ±1.
    
    Parameters
    ----------
    results : dict
        Results dictionary from AlphaSweep.run()
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Spanwise Aerodynamic Distributions", fontsize=13, fontweight="bold")
    
    alpha_deg = results["alpha_deg"]
    spanwise_list = results["spanwise"]
    cl_basic = results["cl_basic"]
    
    # ===== Top-left: Spanwise sectional CL =====
    ax = axes[0, 0]
    for i_alpha, alpha in enumerate(alpha_deg):
        y = spanwise_list[i_alpha]["y"]
        cl = spanwise_list[i_alpha]["cl"]
        
        # Normalize spanwise coordinate
        y_max = np.max(np.abs(y))
        if y_max > 1e-10:
            eta = y / y_max
        else:
            eta = y
        
        # Exclude first and last points (wing tips)
        eta_plot = eta[1:-1]
        cl_plot = cl[1:-1]
        
        ax.plot(eta_plot, cl_plot, "o-", linewidth=2, markersize=6, 
               label=f"α = {alpha:.1f}°")
    
    ax.set_xlabel("Spanwise Coordinate (η = 2y/b) [-]", fontsize=7)
    ax.set_ylabel("Lift Coefficient (Cl) [-]", fontsize=7)
    ax.set_title("Spanwise sectional CL", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    
    # ===== Top-right: Spanwise sectional CMy (about origin) =====
    ax = axes[0, 1]
    for i_alpha, alpha in enumerate(alpha_deg):
        y = spanwise_list[i_alpha]["y"]
        cm = spanwise_list[i_alpha]["cm_origin"]
        
        # Normalize spanwise coordinate
        y_max = np.max(np.abs(y))
        if y_max > 1e-10:
            eta = y / y_max
        else:
            eta = y
        
        # Exclude first and last points (wing tips)
        eta_plot = eta[1:-1]
        cm_plot = cm[1:-1]
        
        ax.plot(eta_plot, cm_plot, "s-", linewidth=2, markersize=6, 
               label=f"α = {alpha:.1f}°")
    
    ax.set_xlabel("Spanwise Coordinate (η = 2y/b) [-]", fontsize=7)
    ax.set_ylabel("Pitching Moment Coefficient (CMy) [-]", fontsize=7)
    ax.set_title("Spanwise sectional CMy (about origin)", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
    
    # ===== Bottom-left: Spanwise induced velocity =====
    ax = axes[1, 0]
    right_tip_max = None
    for i_alpha, alpha in enumerate(alpha_deg):
        y = spanwise_list[i_alpha]["y"]
        w = spanwise_list[i_alpha]["w_induced"]
        
        # Normalize spanwise coordinate
        y_max = np.max(np.abs(y))
        if y_max > 1e-10:
            eta = y / y_max
        else:
            eta = y
        
        # Exclude first and last points (wing tips)
        eta_plot = eta[1:-1]
        w_plot = w[1:-1]
        
        ax.plot(eta_plot, w_plot, "^-", linewidth=2, markersize=6, 
        label=f"α = {alpha:.1f}°")
        if w_plot.size:
            right_tip_value = float(w_plot[-1])
            if right_tip_max is None or right_tip_value > right_tip_max:
                right_tip_max = right_tip_value
    
    ax.set_xlabel("Spanwise Coordinate (η = 2y/b) [-]", fontsize=7)
    ax.set_ylabel("Induced Velocity (w) [m/s]", fontsize=7)
    ax.set_title("Spanwise induced velocity", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    if right_tip_max is not None:
        current_bottom = ax.get_ylim()[0]
        ax.set_ylim(bottom=current_bottom, top=right_tip_max)
    
    # ===== Bottom-right: Spanwise load decomposition =====
    ax = axes[1, 1]
    y = cl_basic["y"]
    clb = cl_basic["clb"]
    cla = cl_basic["cla"]
    cl_CL1 = cl_basic["cl_CL1"]
    
    # Normalize spanwise coordinate
    y_max = np.max(np.abs(y))
    if y_max > 1e-10:
        eta = y / y_max
    else:
        eta = y
    
    # Exclude first and last points (wing tips)
    eta_plot = eta[1:-1]
    clb_plot = clb[1:-1]
    cla_plot = cla[1:-1]
    cl_CL1_plot = cl_CL1[1:-1]
    
    # Plot basic distribution (dashed line)
    ax.plot(eta_plot, clb_plot, "o--", linewidth=2, markersize=6, 
           label="Clb(η) - Basic", color="tab:blue")
    
    # Plot additional distribution (dashed line)
    ax.plot(eta_plot, cla_plot, "s--", linewidth=2, markersize=6, 
           label="Cla(η) - Additional", color="tab:orange")
    
    # Plot combined distribution at CL = 1 (solid line)
    ax.plot(eta_plot, cl_CL1_plot, "^-", linewidth=2.5, markersize=7, 
           label="Cl(η) at CL = 1.0", color="tab:green")
    
    ax.set_xlabel("Spanwise Coordinate (η = 2y/b) [-]", fontsize=7)
    ax.set_ylabel("Lift Coefficient (Cl) [-]", fontsize=7)
    ax.set_title("Spanwise load decomposition", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
    
    return fig


def plot_all(results):
    """
    Display all aerodynamic plots in separate figure windows.
    
    Opens and displays the following plots:
    1. Global aerodynamic coefficients (2x2 layout)
    2. Spanwise aerodynamic distributions (2x2 layout)
    
    Parameters
    ----------
    results : dict
        Results dictionary from AlphaSweep.run()
    
    Notes
    -----
    This function calls plt.show() at the end to display all figures.
    Plots are not saved to disk; they are displayed interactively.
    """
    # Create all plots
    plot_global_coefficients(results)
    plot_spanwise_all(results)
    
    # Display all figures
    plt.show()
