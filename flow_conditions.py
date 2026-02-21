"""
Freestream flow conditions module for Vortex Lattice Method (VLM) solver.

This module defines the aerodynamic flow conditions relative to the aircraft,
including velocity magnitude, flow angles, and atmospheric properties.

Classes:
    FlowCondition: Represents uniform freestream flow conditions
"""

import numpy as np
from atmosphere import ISA


class FlowCondition:
    """
    Freestream flow condition for aerodynamic analysis.
    
    Defines a uniform freestream flow characterized by velocity magnitude,
    angle of attack, sideslip angle, and atmospheric properties at a given altitude.
    
    The freestream velocity vector is computed as:
        V_inf = V * [cos(alpha)*cos(beta), sin(beta), sin(alpha)*cos(beta)]
    
    Attributes:
        V (float): Freestream speed [m/s]
        alpha (float): Angle of attack [rad]
        beta (float): Sideslip angle [rad]
        altitude (float): Flight altitude [m]
        rho (float): Air density [kg/m^3]
        T (float): Air temperature [K]
        p (float): Air pressure [Pa]
        a (float): Speed of sound [m/s]
        q_inf (float): Dynamic pressure [Pa]
        M (float): Mach number [-]
    """
    
    def __init__(self, V, alpha=0.0, altitude=0.0):
        """
        Initialize freestream flow conditions.
        
        Parameters
        ----------
        V : float
            Freestream speed [m/s]
        alpha : float, optional
            Angle of attack [rad]. Default is 0.0
        altitude : float, optional
            Flight altitude [m]. Default is 0.0 (sea level)
        """
        self.V = float(V)
        self.alpha = float(alpha)
        self.altitude = float(altitude)
        
        # Get atmospheric properties from ISA model
        atm = ISA()
        self.T, self.p, self.rho = atm.properties(self.altitude)
        
        # Compute speed of sound
        gamma = 1.4  # Ratio of specific heats for air
        R = 287.0    # Gas constant [J/(kg·K)]
        self.a = np.sqrt(gamma * R * self.T)
        
        # Compute dynamic pressure
        self.q_inf = 0.5 * self.rho * self.V**2
        
        # Compute Mach number
        self.M = self.V / self.a
    
    def freestream_vector(self):
        """
        Compute the freestream velocity vector.
        
        The velocity vector is defined in the body-fixed coordinate system
        using the freestream speed, angle of attack, and sideslip angle.
        
        Returns
        -------
        numpy.ndarray
            Freestream velocity vector [m/s], shape (3,)
            Components: [Vx, Vy, Vz]
        """
        cos_alpha = np.cos(self.alpha)
        sin_alpha = np.sin(self.alpha)
        
        V_inf = self.V * np.array([
            cos_alpha,
            0.0,
            sin_alpha
        ])
        
        return V_inf