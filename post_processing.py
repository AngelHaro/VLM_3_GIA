"""
Vortex Lattice Method (VLM) - Post-Processing Module (Lifting-Line Approach)

This module performs aerodynamic post-processing for VLM solutions using a 
strict lifting-line discretization approach. 

Key features:
    - Lifting-line style sectional coefficients
    - Trefftz-plane induced drag formulation
    - Circulatory moment calculations
    - Efficient alpha-sweep with single influence matrix

Classes:
    VLMPostProcessor: Post-processor for a single VLM solution
    AlphaSweep: Automated angle-of-attack sweep analysis
"""

import numpy as np


class VLMPostProcessor:
    """
    Post-processor for a single VLM solution using lifting-line formulation.
    
    All aerodynamic quantities are derived from spanwise circulation Gamma(y).
    No panel forces or 3D cross products are used.
    
    Assumes:
        - n_chord = 1 (single chordwise panel per station)
        - V = 1.0 (normalized freestream)
        - No sideslip (beta = 0)
    """

    def __init__(self, wing, mesh, flow, gamma):
        """
        Initialize post-processor.
        
        Parameters
        ----------
        wing : Wing
            Wing geometry object
        mesh : WingMesh
            Mesh object with panel definitions
        flow : FlowCondition
            Flow conditions (V, alpha, altitude)
        gamma : ndarray
            Circulation distribution [1D array of length n_panels]
        """
        self.wing = wing
        self.mesh = mesh
        self.flow = flow
        self.gamma = np.array(gamma).flatten()

        self.spanwise_data = None
        self.CDi = None

    # ==================================================================
    # MAIN
    # ==================================================================

    def compute(self):
        """
        Compute all aerodynamic coefficients and spanwise distributions.
        
        Returns
        -------
        dict
            Dictionary containing:
            - CL: Total lift coefficient
            - CD: Total drag coefficient (= CDi)
            - CDi: Induced drag coefficient
            - CMy_origin: Pitching moment about origin
            - CMy_ac: Pitching moment about aerodynamic center
            - spanwise: Dictionary with spanwise distributions
        """
        self._compute_spanwise()
        self._compute_trefftz()
        
        # Global coefficients
        CL = self._compute_CL()
        CDi = self._compute_trefftz() 
        
        # Moments
        CMy_origin, CMy_ac = self._compute_moments(CL)
        
        return {
            "CL": CL,
            "CD": CDi,
            "CDi": CDi,
            "CMy_origin": CMy_origin,
            "CMy_ac": CMy_ac,
            "spanwise": self.spanwise_data
        }

    # ==================================================================
    # SPANWISE COMPUTATION
    # ==================================================================

    def _compute_spanwise(self):
        """
        Compute sectional lift and moment coefficients from circulation.
        
        For lifting-line approach:
            cl_j = 2 * Gamma_j / c_j
            cm_origin_j = sectional moment coefficient
        """
        n_span = self.mesh.n_span
        n_chord = self.mesh.n_chord

        y_list = []
        cl_list = []
        gamma_list = []
        cm_origin_list = []

        idx = 0

        for i in range(n_span):

            # Accumulate values for this spanwise station
            Gamma_section = 0.0
            chord_section = 0.0
            y_mean = 0.0
            x_quarter_mean = 0.0

            for j in range(n_chord):
                panel = self.mesh.panels[idx]

                Gamma_section += self.gamma[idx]
                chord_section += np.linalg.norm(panel.chord_vector)
                y_mean += panel.get_center()[1]
                
                # Quarter-chord x-position for moment (use panel's 1/4-chord point)
                quarter_chord_point = panel.get_quarter_chord()
                x_quarter_mean += quarter_chord_point[0]

                idx += 1

            # Average over chordwise panels
            chord_section /= n_chord
            y_mean /= n_chord
            x_quarter_mean /= n_chord

            # Sectional lift coefficient: cl = 2 * Gamma / c
            cl = 2.0 * Gamma_section / chord_section

            # Sectional moment coefficient about leading edge (lifting-line style)
            # cm_le = -cl * (x_25 / c)  where x_25 is quarter-chord relative to LE
            cm_le = -cl * (x_quarter_mean / chord_section)

            y_list.append(y_mean)
            cl_list.append(cl)
            gamma_list.append(Gamma_section)
            cm_origin_list.append(cm_le)

        self.spanwise_data = {
            "y": np.array(y_list),
            "cl": np.array(cl_list),
            "gamma": np.array(gamma_list),
            "cm_origin": np.array(cm_origin_list),
            "w_induced": np.zeros(n_span)  # Will be filled in _compute_trefftz
        }

    # ==================================================================
    # TREFFTZ PLANE / INDUCED DRAG
    # ==================================================================

    def _compute_trefftz(self):
        """
        Compute induced velocity and drag using Trefftz-plane formulation.
        
        Based on lifting-line theory:
            w_j = sum_i ( DeltaGamma_i / (2π (y_j - y_i)) )
            cdi_j = -cl_j * w_j / 2
            CDi = sum( cdi_j * S_j ) / S
        """
        y = self.spanwise_data["y"]
        Gamma = self.spanwise_data["gamma"]
        cl = self.spanwise_data["cl"]

        S = self.wing.wing_area
        N = len(y)

        # -----------------------------------------------------------
        # 1) Extend Gamma to enforce zero at tips
        # -----------------------------------------------------------
        Gamma_ext = np.zeros(N + 2)
        Gamma_ext[1:-1] = Gamma

        # -----------------------------------------------------------
        # 2) Build DeltaGamma correctly
        # -----------------------------------------------------------
        DeltaGamma = np.diff(Gamma_ext)

        # -----------------------------------------------------------
        # 3) Construct extended y array (include tips)
        # -----------------------------------------------------------
        y_ext = np.zeros(N + 2)
        y_ext[1:-1] = y
        y_ext[0] = y[0] - (y[1] - y[0])
        y_ext[-1] = y[-1] + (y[-1] - y[-2])

        # -----------------------------------------------------------
        # 4) Compute induced velocity
        # -----------------------------------------------------------
        w_inducido = np.zeros(N)

        for j in range(N):
            for i in range(N + 1):
                dy = y[j] - y_ext[i]
                if abs(dy) > 1e-12:
                    w_inducido[j] += DeltaGamma[i] / (2.0 * np.pi * dy)
                else:
                    # Handle singularity (self-induced velocity at station)
                    w_inducido[j] += 0.0  # No contribution from self-induced velocity

        # ---------------------------------------------------------------
        # 5) Compute local induced drag coefficient
        # ---------------------------------------------------------------
        cdi_local = -cl * w_inducido / 2.0

        # ---------------------------------------------------------------
        # 6) Integrate over surface to get total CDi
        # ---------------------------------------------------------------
        n_span = self.mesh.n_span
        n_chord = self.mesh.n_chord

        idx = 0
        area_sections = np.zeros(N)

        for i in range(n_span):
            area_section = 0.0
            for j in range(n_chord):
                area_section += self.mesh.panels[idx].area
                idx += 1
            area_sections[i] = area_section

        # Weighted integration
        CDi = np.sum(cdi_local * area_sections) / S

        # Store induced velocity
        self.spanwise_data["w_induced"] = w_inducido

        return CDi

    # ==================================================================
    # GLOBAL COEFFICIENTS
    # ==================================================================

    def _compute_CL(self):
        """
        Compute total lift coefficient from circulation distribution.
        
        Using lifting-line formula:
            CL = (2 / S) * integral( Gamma(y) dy )
        
        For discrete: CL = (2 / S) * sum( Gamma_j * dy_j )
        """
        y = self.spanwise_data["y"]
        Gamma = self.spanwise_data["gamma"]
        S = self.wing.wing_area

        N = len(y)

        # Compute dy between consecutive stations using cell-center integration
        # This ensures correct spanwise distribution and natural decay at wing tips
        dy = np.zeros(N)
        if N == 1:
            dy[0] = 2.0 * np.abs(y[0])  # Full span if only one station
        else:
            # First station: integrate from y[0] to midpoint between y[0] and y[1]
            dy[0] = np.abs(y[1] - y[0]) / 2.0
            
            # Internal stations: integrate from midpoint to midpoint
            for i in range(1, N - 1):
                dy[i] = np.abs(y[i+1] - y[i-1]) / 2.0
            
            # Last station: integrate from midpoint between y[N-2] and y[N-1] to y[N-1]
            dy[N-1] = np.abs(y[N-1] - y[N-2]) / 2.0

        # CL = 2 * sum(Gamma * dy) / S
        CL = 2.0 * np.sum(Gamma * dy) / S

        return CL

    # ==================================================================
    # MOMENTS
    # ==================================================================

    def _compute_moments(self, CL):
        """
        Compute pitching moment about origin and aerodynamic center.
        """

        y = self.spanwise_data["y"]
        cl = self.spanwise_data["cl"]

        S = self.wing.wing_area
        MAC = self.wing.mean_aerodynamic_chord
        x_ac = self.wing.x_ac

        N = len(y)

        n_span = self.mesh.n_span
        n_chord = self.mesh.n_chord

        idx = 0
        CM_origin_sum = 0.0

        for i in range(n_span):

            panel = self.mesh.panels[idx]

            # Local chord
            c = np.linalg.norm(panel.chord_vector)

            # Global quarter-chord x position
            x_qc = panel.get_quarter_chord()[0]

            # Panel area
            S_panel = panel.area

            # Local sectional lift coefficient
            cl_local = cl[i]

            # Local moment coefficient relative to origin
            cm_local = -cl_local * (x_qc / c)

            # Add contribution weighted by panel area
            CM_origin_sum += cm_local * c * S_panel

            idx += n_chord

        # Final coefficient
        CMy_origin = CM_origin_sum / (S * MAC)

        # Moment about aerodynamic center
        CMy_ac = CMy_origin - CL * (x_ac / MAC)

        return CMy_origin, CMy_ac





class AlphaSweep:
    """
    Perform angle-of-attack sweep using lifting-line post-processing.
    
    Builds the influence matrix A ONCE, then reuses it for all angles.
    Only rebuilds RHS vector b for each alpha value.
    """

    def __init__(self,
                 wing,
                 mesh,
                 base_flow,
                 aero_solver_class,
                 linear_solver,
                 alphas_deg):
        """
        Initialize alpha sweep.
        
        Parameters
        ----------
        wing : Wing
            Wing geometry
        mesh : WingMesh
            Mesh object
        base_flow : FlowCondition
            Base flow condition (used for rho, V)
        aero_solver_class : class
            Aerodynamic solver class (VLMAeroSolver)
        linear_solver : LinearSystemSolver
            Linear system solver
        alphas_deg : array-like
            Array of angles of attack in degrees
        """
        self.wing = wing
        self.mesh = mesh
        self.base_flow = base_flow
        self.aero_solver_class = aero_solver_class
        self.linear_solver = linear_solver
        self.alphas_deg = np.array(alphas_deg)

        self.results = None

    # ==================================================================
    # MAIN
    # ==================================================================

    def run(self):
        """
        Execute angle-of-attack sweep.
        
        - Build A once at base_flow condition
        - For each alpha: rebuild b, solve for Gamma, post-process
        - Fit lift curve (CL vs alpha)
        
        Returns
        -------
        dict
            Comprehensive results dictionary
        """
        from flow_conditions import FlowCondition
        from post_processing import VLMPostProcessor

        n_alpha = len(self.alphas_deg)

        CL = np.zeros(n_alpha)
        CDi = np.zeros(n_alpha)
        CMy_origin = np.zeros(n_alpha)
        CMy_ac = np.zeros(n_alpha)

        spanwise_list = []

        # ==============================================================
        # BUILD INFLUENCE MATRIX ONCE (does not depend on alpha)
        # ==============================================================
        aero_solver_base = self.aero_solver_class(self.mesh, self.base_flow)
        A = aero_solver_base.build_influence_matrix()

        # ==============================================================
        # LOOP OVER ANGLES OF ATTACK
        # ==============================================================
        for i_alpha, alpha_deg in enumerate(self.alphas_deg):

            alpha_rad = np.deg2rad(alpha_deg)

            # Update flow condition with new alpha
            flow = FlowCondition(
                V=self.base_flow.V,
                alpha=alpha_rad,
                altitude=self.base_flow.altitude
            )

            # IMPORTANT: Create solver with new flow (for RHS computation)
            aero_solver = self.aero_solver_class(self.mesh, flow)

            # Rebuild RHS vector for this alpha
            b = aero_solver.build_rhs_vector()

            # Solve linear system with REUSED influence matrix
            gamma, _ = self.linear_solver.solve(A, b)

            # Post-process solution
            processor = VLMPostProcessor(
                self.wing, self.mesh, flow, gamma
            )

            res = processor.compute()

            CL[i_alpha] = res["CL"]
            CDi[i_alpha] = res["CDi"]
            CMy_origin[i_alpha] = res["CMy_origin"]

            spanwise_list.append(res["spanwise"])

        # ==============================================================
        # FIT LIFT CURVE
        # ==============================================================
        alphas_rad = np.deg2rad(self.alphas_deg)

        # Fit: CL = CL_alpha * alpha + CL0
        coeffs = np.polyfit(alphas_rad, CL, 1)

        CL_alpha = coeffs[0]
        CL0 = coeffs[1]

        # Fit moment about origin
        coeffs_CM = np.polyfit(alphas_rad, CMy_origin, 1)
        CM_alpha = coeffs_CM[0]

        MAC = self.wing.mean_aerodynamic_chord

        if abs(CL_alpha) > 1e-12:
            x_ac_effective = -CM_alpha / CL_alpha * MAC
        else:
            x_ac_effective = 0.0

        # Recompute moment about aerodynamic center
        CMy_ac_corrected = CMy_origin + CL * (x_ac_effective / MAC)

        if abs(CL_alpha) > 1e-12:
            alpha_CL0_rad = -CL0 / CL_alpha
            alpha_CL0_deg = np.rad2deg(alpha_CL0_rad)
        else:
            alpha_CL0_deg = 0.0

        # ==============================================================
        # SPANWISE LOAD DECOMPOSITION (Lifting-Line Linear Regression)
        # ==============================================================
        # For each spanwise station, fit: Cl_j = Cla_j * CL + Clb_j
        # This decomposition is valid for linear aerodynamics (VLM)
        
        if len(spanwise_list) > 0:
            y_reference = spanwise_list[0]["y"]
            n_stations = len(y_reference)
            
            # Check if CL has sufficient variance for regression
            CL_variance = np.var(CL)
            
            if CL_variance > 1e-12:
                # Initialize decomposition arrays
                Cla = np.zeros(n_stations)
                Clb = np.zeros(n_stations)
                
                # For each spanwise station, perform linear regression
                for j in range(n_stations):
                    # Extract sectional cl values at this station for all alphas
                    cl_j = np.array([spanwise_list[i]["cl"][j] for i in range(n_alpha)])
                    
                    # Fit linear model: cl_j = a_j * CL + b_j
                    coeffs = np.polyfit(CL, cl_j, 1)
                    
                    Cla[j] = coeffs[0]  # slope: additional load per unit CL
                    Clb[j] = coeffs[1]  # intercept: basic load at CL=0
                
                # Compute combined distribution at CL = 1
                Cl_CL1 = Clb + Cla * 1.0
                
            else:
                # If CL is constant, cannot decompose
                Cla = np.zeros(n_stations)
                Clb = np.zeros(n_stations)
                Cl_CL1 = np.zeros(n_stations)
            
            cl_basic = {
                "y": y_reference,
                "clb": Clb,
                "cla": Cla,
                "cl_CL1": Cl_CL1
            }
        else:
            # No spanwise data available
            cl_basic = {
                "y": np.array([]),
                "clb": np.array([]),
                "cla": np.array([]),
                "cl_CL1": np.array([])
            }

        # ==============================================================
        # RETURN RESULTS
        # ==============================================================
        self.results = {
            "alpha_deg": self.alphas_deg,
            "CL": CL,
            "CD": CDi,
            "CDi": CDi,
            "CMy_origin": CMy_origin,
            "CMy_ac": CMy_ac_corrected,
            "CL_alpha": CL_alpha,
            "CL0": CL0,
            "alpha_CL0_deg": alpha_CL0_deg,
            "spanwise": spanwise_list,
            "cl_basic": cl_basic
        }

        return self.results, x_ac_effective
