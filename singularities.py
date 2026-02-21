"""
Vortex singularities module for Vortex Lattice Method (VLM) solver.

This module defines vortex singularity elements used in aerodynamic calculations.
It provides implementations of the Biot-Savart law for vortex-induced velocities.

Classes:
    VortexSegment: A straight vortex filament between two points
    HorseshoeVortex: A classical horseshoe vortex with bound and wake segments
"""

import numpy as np


class VortexSegment:
    """
    Represents a straight vortex filament between two points.
    
    The segment induces velocity in the surrounding flow according to the
    Biot-Savart law for a finite vortex filament.
    
    Attributes:
        A (numpy.ndarray): Starting point of the segment (3D vector)
        B (numpy.ndarray): Ending point of the segment (3D vector)
        gamma (float): Circulation strength (default = 1.0)
    """
    
    def __init__(self, A, B, gamma=1.0):
        """
        Initialize a vortex segment.
        
        Args:
            A (array-like): Starting point of the segment (3D)
            B (array-like): Ending point of the segment (3D)
            gamma (float, optional): Circulation strength. Defaults to 1.0.
        """
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.gamma = float(gamma)
    
    def induced_velocity(self, P):
        """
        Compute the velocity induced at point P by this vortex segment.
        
        Uses the Biot-Savart law for a straight, finite vortex filament:
        
            v = (gamma / 4π) * (r1 × r2) / |r1 × r2|² * r0 · (r1_hat - r2_hat)
        
        where:
            r0 = B - A (segment vector)
            r1 = P - A (vector from start to evaluation point)
            r2 = P - B (vector from end to evaluation point)
        
        Args:
            P (array-like): Evaluation point (3D)
        
        Returns:
            numpy.ndarray: Induced velocity vector at P (3D)
        """
        P = np.asarray(P, dtype=float)
        
        # Vectors from segment endpoints to evaluation point
        r1 = P - self.A
        r2 = P - self.B
        
        # Segment vector
        r0 = self.B - self.A
        
        # Compute magnitudes
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)
        r0_mag = np.linalg.norm(r0)
        
        # Tolerance for singularity detection
        epsilon = 1e-10
        
        # Check if point is too close to segment endpoints
        if r1_mag < epsilon or r2_mag < epsilon:
            return np.zeros(3)
        
        # Check if segment is degenerate (A and B are the same)
        if r0_mag < epsilon:
            return np.zeros(3)
        
        # Cross product r1 × r2
        cross_product = np.cross(r1, r2)
        cross_mag_sq = np.dot(cross_product, cross_product)
        
        # Check if point is on the vortex line (cross product is zero)
        if cross_mag_sq < epsilon:
            return np.zeros(3)
        
        # Unit vectors
        r1_hat = r1 / r1_mag
        r2_hat = r2 / r2_mag
        
        # Biot-Savart law for finite segment
        # v = (gamma / 4π) * (r1 × r2) / |r1 × r2|² * r0 · (r1_hat - r2_hat)
        coefficient = self.gamma / (4.0 * np.pi * cross_mag_sq)
        geometric_factor = np.dot(r0, r1_hat - r2_hat)
        
        velocity = coefficient * geometric_factor * cross_product
        
        return velocity


class HorseshoeVortex:
    """
    Represents a classical horseshoe vortex with one bound segment and two wake segments.
    
    The horseshoe vortex consists of:
    - A bound vortex segment from A to B
    - A left wake segment extending from A in the wake direction
    - A right wake segment extending from B in the wake direction
    
    Attributes:
        A (numpy.ndarray): Left end of the bound vortex (3D vector)
        B (numpy.ndarray): Right end of the bound vortex (3D vector)
        wake_direction (numpy.ndarray): Direction of the wake (3D vector)
        wake_length (float): Length of the wake segments
        gamma (float): Circulation strength (default = 1.0)
        bound_segment (VortexSegment): The bound vortex segment
        left_wake_segment (VortexSegment): The left wake segment
        right_wake_segment (VortexSegment): The right wake segment
    """
    
    def __init__(self, A, B, wake_direction, wake_length, gamma=1.0):
        """
        Initialize a horseshoe vortex.
        
        Args:
            A (array-like): Left end of the bound vortex (3D)
            B (array-like): Right end of the bound vortex (3D)
            wake_direction (array-like): Direction of the wake (3D)
            wake_length (float): Length of the wake segments
            gamma (float, optional): Circulation strength. Defaults to 1.0.
        """
        # Store basic attributes
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.wake_direction = np.asarray(wake_direction, dtype=float)
        self.wake_length = float(wake_length)
        self.gamma = float(gamma)
        
        # Normalize wake direction
        wake_direction_mag = np.linalg.norm(self.wake_direction)
        if wake_direction_mag > 1e-10:
            wake_direction_unit = self.wake_direction / wake_direction_mag
        else:
            # Default to x-direction if wake direction is zero
            wake_direction_unit = np.array([1.0, 0.0, 0.0])
        
        # Compute wake endpoints
        A_wake = self.A + self.wake_length * wake_direction_unit
        B_wake = self.B + self.wake_length * wake_direction_unit
        
        # Create the three vortex segments (physically consistent horseshoe)
        self.left_wake_segment = VortexSegment(A_wake, self.A, self.gamma)
        self.bound_segment = VortexSegment(self.A, self.B, self.gamma)
        self.right_wake_segment = VortexSegment(self.B, B_wake, self.gamma)

    
    def induced_velocity(self, P):
        """
        Compute the velocity induced at point P by the horseshoe vortex.
        
        The total induced velocity is the vector sum of the velocities induced
        by the bound segment and the two wake segments.
        
        Args:
            P (array-like): Evaluation point (3D)
        
        Returns:
            numpy.ndarray: Induced velocity vector at P (3D)
        """
        P = np.asarray(P, dtype=float)
        
        # Sum contributions from all three segments
        v_bound = self.bound_segment.induced_velocity(P)
        v_left_wake = self.left_wake_segment.induced_velocity(P)
        v_right_wake = self.right_wake_segment.induced_velocity(P)
        
        total_velocity = v_bound + v_left_wake + v_right_wake
        
        return total_velocity
