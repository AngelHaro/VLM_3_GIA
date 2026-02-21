"""
Vortex Lattice Method (VLM) - Geometric Mesh Module

This module handles the geometric discretization of a wing into panels.
It creates a structured grid of nodes and builds panel elements.

No aerodynamic singularities or control points are defined here.
This is purely for geometric representation of the wing surface.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Panel:
    """
    Represents a single quadrilateral panel of the wing mesh.
    
    A panel consists of 4 nodes defining its corners.
    Geometric properties (area, normal, etc.) are computed internally.
    """
    
    def __init__(self, nodes):
        """
        Initialize a panel from 4 node coordinates.
        
        Parameters:
        -----------
        nodes : array of shape (4, 3)
            The 4 corner nodes [P1, P2, P3, P4] in order:
            P1 ---- P2   (leading edge)
             |      |
             |      |
            P4 ---- P3   (trailing edge)
        """
        self.nodes = np.array(nodes)  # (4, 3) array
        self.P1, self.P2, self.P3, self.P4 = self.nodes
        
        # Compute geometric properties
        self._compute_area()
        self._compute_normal()
        self._compute_chord_span()
    
    def _compute_area(self):
        """Compute the area of the quadrilateral panel using two triangles."""
        # Triangle 1: P1, P2, P3
        v1 = self.P2 - self.P1
        v2 = self.P3 - self.P1
        area1 = 0.5 * np.linalg.norm(np.cross(v1, v2))

        # Triangle 2: P1, P3, P4
        v3 = self.P3 - self.P1
        v4 = self.P4 - self.P1
        area2 = 0.5 * np.linalg.norm(np.cross(v3, v4))

        self.area = area1 + area2

    
    def _compute_normal(self):
        """Compute the unit normal vector of the panel."""
        le_mid = 0.5 * (self.P1 + self.P2)
        te_mid = 0.5 * (self.P4 + self.P3)
        threequarter_left = self.P1 + 0.75 * (self.P4 - self.P1)
        threequarter_right = self.P2 + 0.75 * (self.P3 - self.P2)

        # Normal = (P4 - P1) × (P2 - P1)
        v1 = threequarter_right - threequarter_left
        v2 = te_mid - le_mid
        normal = np.cross(v2, v1)

        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            self.normal = normal / norm
        else:
            self.normal = np.array([0.0, 0.0, 1.0])  # Default if degenerate
    
    def _compute_chord_span(self):
        """Compute local chord and span vectors."""

        # Midpoint of leading edge
        le_mid = 0.5 * (self.P1 + self.P2)

        # Midpoint of trailing edge
        te_mid = 0.5 * (self.P4 + self.P3)

        # Chord vector: from leading edge to trailing edge
        self.chord_vector = te_mid - le_mid

        # Span vector: from left side to right side
        left_mid = 0.5 * (self.P1 + self.P4)
        right_mid = 0.5 * (self.P2 + self.P3)

        self.span_vector = right_mid - left_mid

        # Quarter-chord points on left and right edges
        quarter_left = self.P1 + 0.25 * (self.P4 - self.P1)
        quarter_right = self.P2 + 0.25 * (self.P3 - self.P2)

        # Bound vortex vector (spanwise at 1/4 chord)
        self.bound_vector = quarter_right - quarter_left


    
    def get_center(self):
        """Return the center point of the panel."""
        return np.mean(self.nodes, axis=0)
    
    def get_quarter_chord(self):
        """Return the quarter-chord point (25% of the chord from the leading edge)."""
        le_point = 0.5 * (self.P1 + self.P2)
        te_point = 0.5 * (self.P4 + self.P3)
        return le_point + 0.25 * (te_point - le_point)
    
    def get_three_quarter_chord(self):
        """Return the three-quarter-chord point (75% of the chord from the leading edge)."""
        le_point = 0.5 * (self.P1 + self.P2)
        te_point = 0.5 * (self.P4 + self.P3)
        return le_point + 0.75 * (te_point - le_point)


class WingMesh:

    def __init__(self, wing, n_span, n_chord,
                 spanwise_spacing="cosine"):

        self.wing = wing
        self.n_span = n_span
        self.n_chord = n_chord
        self.spanwise_spacing = spanwise_spacing

        self.nodes = None
        self.panels = []

        self._generate_nodes()
        self._generate_panels()

    def _generate_nodes(self):

        n_nodes_span = self.n_span + 1
        n_nodes_chord = self.n_chord + 1

        self.nodes = np.zeros((n_nodes_span,
                               n_nodes_chord, 3))

        # Spanwise distribution (FULL wing)
        if self.spanwise_spacing == "cosine":
            theta = np.linspace(0, np.pi, n_nodes_span)
            eta = np.cos(theta)
        else:
            raise ValueError("Invalid spanwise_spacing")

        y_dist = (self.wing.wing_span / 2) * eta

        for i, y in enumerate(y_dist):

            c = self.wing.chord_at_y(abs(y))
            x_le = self.wing.x_le_at_y(abs(y))

            twist = -np.radians(self.wing.twist * (abs(y) / (self.wing.wing_span / 2)))

            x_qc = x_le + 0.25 * c
            z_qc = 0.0

            for j in range(n_nodes_chord):

                xi = j / self.n_chord
                x = x_le + xi * c
                z = 0.0

                dx = x - x_qc
                dz = z - z_qc

                x_rot = x_qc + dx * np.cos(twist)
                z_rot = z_qc + dx * np.sin(twist)

                self.nodes[i, j] = [x_rot, y, z_rot]

    def _generate_panels(self):

        self.panels = []

        for i in range(self.n_span):
            for j in range(self.n_chord):

                P1 = self.nodes[i, j]
                P2 = self.nodes[i+1, j]
                P3 = self.nodes[i+1, j+1]
                P4 = self.nodes[i, j+1]

                self.panels.append(Panel([P1, P2, P3, P4]))

    def get_num_nodes(self):
        """Return total number of nodes in the FULL wing."""
        return (self.n_span + 1) * (self.n_chord + 1)

    def get_num_panels(self):
        return len(self.panels)

    def plot_mesh(self, title=None, show_nodes=True):

        if title is None:
            title = f"Wing Mesh - VLM Geometric Discretization\n({self.get_num_panels()} Panels: {self.n_span} span × {self.n_chord} chord)"

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Draw panels as semi-transparent polygons with edge colors
        panel_verts = [p.nodes for p in self.panels]
        poly = Poly3DCollection(panel_verts, 
                               alpha=0.25,
                               facecolor='cyan',
                               edgecolor='darkblue',
                               linewidth=1.2)
        ax.add_collection3d(poly)

        # Plot nodes
        nodes_flat = self.nodes.reshape(-1, 3)
        ax.scatter(nodes_flat[:, 0],
                   nodes_flat[:, 1],
                   nodes_flat[:, 2],
                   c='red', s=20, marker='o', 
                   edgecolors='darkred', linewidth=0.5, zorder=5)

        # Set equal aspect ratio for all axes
        x_data = nodes_flat[:, 0]
        y_data = nodes_flat[:, 1]
        z_data = nodes_flat[:, 2]
        
        max_range = np.array([x_data.max() - x_data.min(),
                              y_data.max() - y_data.min(),
                              z_data.max() - z_data.min()]).max() / 2.0

        mid_x = (x_data.max() + x_data.min()) * 0.5
        mid_y = (y_data.max() + y_data.min()) * 0.5
        mid_z = (z_data.max() + z_data.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Labels and formatting
        ax.set_xlabel('x - Streamwise (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('y - Spanwise (m)', fontsize=11, fontweight='bold')
        ax.set_zlabel('z - Vertical (m)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        plt.show()

    def print_mesh_info(self):
            """Print mesh information to console."""
            print("\n" + "="*60)
            print("WING MESH INFORMATION")
            print("="*60)
            print(f"Spanwise panels: {self.n_span}")
            print(f"Chordwise panels: {self.n_chord}")
            print(f"Spanwise spacing: {self.spanwise_spacing}")
            print(f"Total nodes: {self.get_num_nodes()}")
            print(f"Total panels: {self.get_num_panels()}")
            # print(f"Quarter-chord points of each panel:")
            # for i, panel in enumerate(self.panels):
                # qc = panel.get_quarter_chord()
                # print(f"  Panel {i+1}: {qc}")
            print("="*60 + "\n")
