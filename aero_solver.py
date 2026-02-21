"""
Vortex Lattice Method (VLM) aerodynamic solver module.

This module builds the aerodynamic influence matrix and right-hand side vector
for the VLM linear system. 

Classes:
    VLMAeroSolver: Builds the VLM aerodynamic system A*gamma = b
"""

import numpy as np
from singularities import HorseshoeVortex


class VLMAeroSolver:
    """
    Vortex Lattice Method aerodynamic solver.
    
    Constructs the aerodynamic influence matrix and right-hand side vector
    for a lifting surface using the horseshoe vortex model.
    
    The aerodynamic system is:
        A * gamma = b
    
    Where:
        A[i,j] = normal component of velocity induced at control point i
                 by horseshoe vortex j
        b[i]   = -V_inf · n_i (freestream boundary condition)
        gamma  = vector of unknown circulation strengths
    
    Attributes:
        mesh: WingMesh object containing panel geometry
        flow: FlowCondition object defining freestream conditions
        panels (list): List of Panel objects from the mesh
        num_panels (int): Total number of panels
        vortices (list): List of HorseshoeVortex objects
    """
    
    def __init__(self, mesh, flow_conditions):
        """
        Initialize the VLM aerodynamic solver.
        
        Creates horseshoe vortices for each panel with:
        - Bound vortex at the quarter-chord line
        - Wake aligned with freestream direction
        - Wake length equivalent to infinite wake
        
        Parameters
        ----------
        mesh : WingMesh
            Mesh object containing panel geometry
        flow_conditions : FlowCondition
            Flow conditions object with freestream properties
        """
        # Store mesh and flow conditions
        self.mesh = mesh
        self.flow = flow_conditions
        
        # Extract panels
        self.panels = mesh.panels
        self.num_panels = len(self.panels)
        
        # Compute wake parameters
        wake_direction = self.flow.freestream_vector()
        wake_length = 500 * mesh.wing.wing_span
        
        # Build horseshoe vortices for each panel
        self.vortices = []
        
        for panel in self.panels:
            # Get panel nodes
            P1, P2, P3, P4 = panel.nodes
            
            # Compute bound vortex endpoints at quarter-chord
            A = P1 + 0.25 * (P4 - P1)  # Left end
            B = P2 + 0.25 * (P3 - P2)  # Right end
            
            # Create horseshoe vortex with unit circulation
            vortex = HorseshoeVortex(
                A=A,
                B=B,
                wake_direction=wake_direction,
                wake_length=wake_length,
                gamma=1.0
            )
            
            self.vortices.append(vortex)
    
    def build_influence_matrix(self):
        """
        Build the aerodynamic influence matrix.
        
        Computes the influence coefficient matrix A where A[i,j] represents
        the normal component of velocity induced at control point i by
        horseshoe vortex j (with unit circulation).
        
        Returns
        -------
        numpy.ndarray
            Influence matrix A of shape (N, N) where N is the number of panels
        """
        N = self.num_panels
        A = np.zeros((N, N))
        
        # Loop over control points (rows)
        for i in range(N):
            # Get control point and normal of panel i
            P_i = self.panels[i].get_three_quarter_chord()
            n_i = self.panels[i].normal
            
            # Loop over horseshoe vortices (columns)
            for j in range(N):
                # Get induced velocity at P_i due to vortex j
                V_ij = self.vortices[j].induced_velocity(P_i)
                
                # Project onto normal direction
                A[i, j] = np.dot(V_ij, n_i)
        
        return A
    
    def build_rhs_vector(self):
        """
        Build the right-hand side vector for the VLM system.
        
        Computes the RHS vector b where b[i] represents the negative of
        the freestream velocity component normal to panel i. This enforces
        the impermeability boundary condition (no flow through the surface).
        
        Returns
        -------
        numpy.ndarray
            Right-hand side vector b of shape (N,) where N is the number of panels
        """
        N = self.num_panels
        b = np.zeros(N)
        
        # Get freestream velocity vector
        V_inf = self.flow.freestream_vector()
        
        # Loop over panels
        for i in range(N):
            # Get panel normal
            n_i = self.panels[i].normal
            n_i[1] = 0.0  # Ensure normal is in x-z plane (for 2D flow)
            
            # Compute RHS: negative of freestream normal component
            b[i] = -np.dot(V_inf, n_i)/np.linalg.norm(V_inf)/np.linalg.norm(n_i)
        
        return b
