import numpy as np
from geometry import Wing
from flow_conditions import FlowCondition
from mesh import WingMesh
from aero_solver import VLMAeroSolver
from numerical_solver import LinearSystemSolver
from post_processing import AlphaSweep, VLMPostProcessor
from plots import plot_all

def main():
    # === CREATE WING GEOMETRY ===
    wing = Wing(
        wing_area=1,      # m² - reference area
        taper_ratio=0.5,      # tip to root chord ratio
        aspect_ratio=15,     # b²/S
        sweep_angle=15,       # degrees
        twist=-3               # degrees
    )
    
    # Print wing parameters
    print(wing)
    print("\n" + "="*50 + "\n")
    
    # === CREATE WING MESH ===
    n_span = 30  # Number of spanwise panels
    n_chord = 1      # Chordwise panels
    
    mesh = WingMesh(wing, n_span, n_chord, spanwise_spacing="cosine")

    # Print mesh information
    mesh.print_mesh_info()

    print("\n" + "="*50 + "\n")
    
    # === DEFINE FLOW CONDITIONS ===
    flow = FlowCondition(
        V=1.0,              # m/s - freestream velocity
        alpha=np.deg2rad(7.0),  # rad - angle of attack
        altitude=0.0         # m - sea level
    )
    
    print("Flow conditions:")
    print(f"  Velocity: {flow.V} m/s")
    print(f"  Angle of attack: {np.rad2deg(flow.alpha):.2f} deg")
    print(f"  Altitude: {flow.altitude} m")
    print(f"  Density: {flow.rho:.4f} kg/m³")
    
    print("\n" + "="*50 + "\n")

        # === CREATE AERODYNAMIC SOLVER ===
    aero_solver = VLMAeroSolver(mesh, flow)
    
    # === BUILD LINEAR SYSTEM ===
    print("Building aerodynamic influence matrix...")
    A = aero_solver.build_influence_matrix()
    b = aero_solver.build_rhs_vector()
    print("\n" + "="*50 + "\n")
    
    # === SOLVE LINEAR SYSTEM ===
    print("Solving for circulation distribution...")
    num_solver = LinearSystemSolver(method="direct", check_condition=True)
    """
    Method supported options:
    - "direct": Uses numpy.linalg.solve for direct solution (suitable for small to medium systems)
    - "lstsq": Uses numpy.linalg.lstsq (least-squares for ill-conditioned systems)
    
    check_condition: If True, computes and prints the condition number of A to assess numerical stability. High condition numbers may indicate potential issues with the solution accuracy.
    """
    gamma, solver_info = num_solver.solve(A, b)
    
    print("\nSolver diagnostics:")
    for key, value in solver_info.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.6e}")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    # Store solution
    aero_solver.gamma = gamma
    
    print(f"\nCirculation vector computed:")
    print(f"  Gamma shape: ({gamma.shape[0]}, 1)")
    print(f"  Gamma: {gamma.flatten()}")  # Print as 1D array for readability
    
    print("\n" + "="*50 + "\n")

    # === AERODYNAMIC SWEEP AND POST-PROCESSING ===
    # Perform an angle-of-attack sweep to analyze aerodynamic performance
    # across a range of flight conditions (from negative to positive angles)
    
    alphas_deg = np.linspace(-8, 8, 17)  # Angles of attack [deg]
    
    print("Starting angle-of-attack sweep...")
    print(f"  Sweep range: {alphas_deg.min():.1f}° to {alphas_deg.max():.1f}°")
    print(f"  Number of points: {len(alphas_deg)}")
    print()
    
    # Create and run the alpha sweep using the existing solver
    sweep = AlphaSweep(
        wing=wing,
        mesh=mesh,
        base_flow=flow,
        aero_solver_class=VLMAeroSolver,
        linear_solver=num_solver,
        alphas_deg=alphas_deg
    )
    
    # Execute the sweep (solves VLM at each angle, performs lift-curve fitting)
    results = sweep.run()[0]
    
    print("Alpha sweep completed successfully!\n")
    
    # === AERODYNAMIC SUMMARY ===
    print("="*50)
    print("AERODYNAMIC ANALYSIS SUMMARY")
    print("="*50)
    print()
    
    print("Lift Curve Properties:")
    print(f"  CL_alpha (lift curve slope): {results['CL_alpha']:.6f} [1/rad]")
    print(f"  CL_alpha (in deg^-1):        {np.rad2deg(results['CL_alpha']):.6f} [1/deg]")
    print(f"  CL0 (zero-alpha lift):       {results['CL0']:.6f} [-]")
    print(f"  α_CL0 (zero-lift angle):     {results['alpha_CL0_deg']:.3f} [deg]")
    print(f"x_ac_effective (aerodynamic center): {sweep.run()[1]:.4f} m from root leading edge")
    print()
    
    print("Aerodynamic Coefficients at Sweep Points:")
    print()
    print("  α [deg]    CL       CDi      CMy_ac")
    print("  " + "-"*48)
    for i, alpha in enumerate(results["alpha_deg"]):
        CL = results["CL"][i]
        CDi = results["CDi"][i]
        CMy = results["CMy_ac"][i]
        print(f"  {alpha:6.1f}    {CL:7.4f}  {CDi:7.4f}  {CMy:7.4f}")
    print()
    
    print("="*50)
    print()

    # === VISUALIZE MESH ===
    mesh.plot_mesh(show_nodes=True)
    
    print("\n" + "="*50 + "\n")

    # Generate and display all plots
    print("Generating aerodynamic plots...")
    plot_all(results)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()