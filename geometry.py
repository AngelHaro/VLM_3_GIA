"""
Vortex Lattice Method (VLM) - Geometry Module

This module defines the wing geometric parameters and derived properties.
It provides a clean interface for basic planform calculations used by the mesh.

No aerodynamic singularities or solver logic are defined here.
This is purely for geometric representation of the wing.
"""

import math
import matplotlib.pyplot as plt
import numpy as np


class Wing:
    
    def __init__(self, wing_area, taper_ratio, aspect_ratio, sweep_angle, twist):
        """Initialize the Wing class with basic parameters."""
        self.wing_area = wing_area
        self.taper_ratio = taper_ratio
        self.aspect_ratio = aspect_ratio
        self.sweep_angle = sweep_angle
        self.twist = twist
    
    @property
    def wing_span(self):
        return math.sqrt(self.aspect_ratio * self.wing_area)
    
    @property
    def mean_chord(self):
        return math.sqrt(self.wing_area / self.aspect_ratio)
    
    @property
    def root_chord(self):
        return 2 * self.mean_chord / (1 + self.taper_ratio)
    
    @property
    def tip_chord(self):
        return self.taper_ratio * self.root_chord

    @property
    def mean_aerodynamic_chord(self):
        """Calculate the Mean Aerodynamic Chord (MAC) for the trapezoidal wing."""
        return (2/3) * self.root_chord * (1 + self.taper_ratio + self.taper_ratio**2) / (1 + self.taper_ratio)
    
    @property
    def le_sweep_angle(self):
        """Calculate the leading edge sweep angle in radians."""
        return np.arctan(np.tan(np.radians(self.sweep_angle))+2*self.root_chord/self.wing_span*(1-self.taper_ratio*np.cos(np.radians(self.twist)))*(0.25-0))

    @property
    def y_mac(self):
        """Calculate the spanwise position of the MAC from the wing root."""
        return (self.wing_span / 6) * (1 + 2 * self.taper_ratio) / (1 + self.taper_ratio)
    
    @property
    def x_le_mac(self):
        """Calculate the leading edge x-position of the MAC."""
        return self.y_mac * np.tan(self.le_sweep_angle)
    
    @property
    def x_ac(self):
        """Calculate the aerodynamic center position from the root leading edge."""
        return self.x_le_mac + 0.25 * self.mean_aerodynamic_chord

    def chord_at_y(self, y):
        semi_span = self.wing_span / 2
        return self.root_chord + (self.tip_chord - self.root_chord) * (y / semi_span)
    
    def x_le_at_y(self, y):
        return y * np.tan(self.le_sweep_angle)

    def __str__(self):
        """Return a formatted string with wing parameters and calculated values."""
        return (f"Wing Parameters:\n"
                f"  Wing Area: {self.wing_area} m²\n"
                f"  Aspect Ratio: {self.aspect_ratio}\n"
                f"  Taper Ratio: {self.taper_ratio}\n"
                f"  Sweep Angle: {self.sweep_angle}°\n"
                f"  Twist: {self.twist}°\n"
                f"\nCalculated Properties:\n"
                f"  Wing Span: {self.wing_span:.4f} m\n"
                f"  Mean Chord: {self.mean_chord:.4f} m\n"
                f"  Root Chord: {self.root_chord:.4f} m\n"
                f"  Tip Chord: {self.tip_chord:.4f} m\n"
                f"  Mean Aerodynamic Chord (MAC): {self.mean_aerodynamic_chord:.4f} m\n"
                f"  Leading Edge Sweep Angle: {np.rad2deg(self.le_sweep_angle):.2f}°\n"
                f"  Spanwise Position of MAC: {self.y_mac:.4f} m\n"
                f"  Leading Edge x-Position of MAC: {self.x_le_mac:.4f} m\n"
                f"  Aerodynamic Center x-Position: {self.x_ac:.4f} m")