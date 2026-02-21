"""
International Standard Atmosphere (ISA) module.

This module provides a Python implementation of the ISA model up to 20,000 m
altitude using a two-layer model (troposphere and lower stratosphere).

Classes:
    ISA: International Standard Atmosphere model
"""

import numpy as np


class ISA:
    """
    International Standard Atmosphere (ISA) model.
    
    Provides temperature, pressure, and density as functions of geometric
    altitude using the ISA standard model with two atmospheric layers:
    - Troposphere: 0 to 11,000 m (linear temperature gradient)
    - Lower Stratosphere: 11,000 to 20,000 m (isothermal)
    
    All values are in SI units.
    """
    
    # ISA constants
    T0 = 288.15        # Sea level temperature [K]
    p0 = 101325.0      # Sea level pressure [Pa]
    rho0 = 1.225       # Sea level density [kg/m^3]
    L0 = -6.5e-3       # Temperature lapse rate [K/m]
    R = 287.0          # Gas constant for air [J/(kg·K)]
    g0 = 9.80665       # Gravity acceleration [m/s^2]
    
    # Layer boundaries
    z11 = 11000.0      # Tropopause altitude [m]
    z_max = 20000.0    # Maximum valid altitude [m]
    
    def __init__(self):
        """
        Initialize the ISA model.
        
        Pre-computes the atmospheric properties at the tropopause (11 km)
        to ensure continuity between layers.
        """
        # Compute properties at z = 11,000 m using troposphere equations
        self.T11 = self.T0 + self.L0 * self.z11
        self.p11 = self.p0 * (1 + (self.L0 * self.z11) / self.T0) ** (-self.g0 / (self.R * self.L0))
        self.rho11 = self.p11 / (self.R * self.T11)
    
    def properties(self, z):
        """
        Compute atmospheric properties at a given altitude.
        
        Parameters
        ----------
        z : float
            Geometric altitude [m]
        
        Returns
        -------
        T : float
            Temperature [K]
        p : float
            Pressure [Pa]
        rho : float
            Density [kg/m^3]
        
        Raises
        ------
        ValueError
            If altitude is above 20,000 m
        """
        # Handle negative altitudes
        if z < 0:
            z = 0.0
        
        # Check maximum altitude
        if z > self.z_max:
            raise ValueError(f"Altitude {z} m exceeds maximum valid altitude of {self.z_max} m")
        
        # Select atmospheric layer and compute properties
        if z <= self.z11:
            # Troposphere (0 to 11,000 m)
            T = self.T0 + self.L0 * z
            p = self.p0 * (1 + (self.L0 * z) / self.T0) ** (-self.g0 / (self.R * self.L0))
            rho = p / (self.R * T)
        else:
            # Lower Stratosphere (11,000 to 20,000 m) - isothermal
            T = self.T11
            p = self.p11 * np.exp(-self.g0 / (self.R * self.T11) * (z - self.z11))
            rho = self.rho11 * np.exp(-self.g0 / (self.R * self.T11) * (z - self.z11))
        
        return T, p, rho