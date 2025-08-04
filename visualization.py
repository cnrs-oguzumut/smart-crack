"""
Visualization module for 2D phase-field simulation results.
Handles plotting of displacement, damage, and residual fields.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri

class SimulationVisualizer:
    """Handles visualization of simulation results."""
    
    def __init__(self, x, y, connectivity):
        """Initialize with mesh data."""
        self.x = x
        self.y = y
        self.connectivity = connectivity
        self.tri_obj = plt.matplotlib.tri.Triangulation(x, y, connectivity)
    
    def plot_results_2d(self, u, v, d, damage_residual=None, step_number=None, 
                       model_type='AT1', save_png=False, gp_coords=None, gp_residuals=None):
        """Plot the current solution fields including damage residual."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'2D {model_type} Solver - Step {step_number}', 
                    fontsize=16, fontweight='bold')
        
        # Film displacement magnitude
        U_mag = np.sqrt(u[:, 0]**2 + u[:, 1]**2)
        im1 = axes[0, 0].tricontourf(self.tri_obj, U_mag, levels=20, cmap='viridis')
        axes[0, 0].set_title(f'Film Displacement |u| (max={np.max(U_mag):.4f})')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Substrate displacement magnitude
        V_mag = np.sqrt(v[:, 0]**2 + v[:, 1]**2)
        im2 = axes[0, 1].tricontourf(self.tri_obj, V_mag, levels=20, cmap='plasma')
        axes[0, 1].set_title(f'Substrate Displacement |v| (max={np.max(V_mag):.4f})')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Damage field
        im3 = axes[0, 2].tricontourf(self.tri_obj, d, levels=20, cmap='Reds', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Damage d (max={np.max(d):.4f})')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Displacement difference (film - substrate)
        diff_x = u[:, 0] - v[:, 0]
        diff_y = u[:, 1] - v[:, 1]
        diff_mag = np.sqrt(diff_x**2 + diff_y**2)
        im4 = axes[1, 0].tricontourf(self.tri_obj, diff_mag, levels=20, cmap='coolwarm')
        axes[1, 0].set_title(f'Displacement Difference |u-v| (max={np.max(diff_mag):.4f})')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # Damage residual plotting
        self._plot_damage_residual(axes[1, 1], damage_residual, gp_coords, gp_residuals)
        
        # Radial profiles
        self._plot_radial_profiles(axes[1, 2], u, v, damage_residual, gp_coords, gp_residuals)
        
        # Add mesh overlay to contour plots
        for i in range(2):
            for j in range(3):
                if j < 2 or i == 0:  # Skip the last subplot which is a line plot
                    axes[i, j].triplot(self.tri_obj, 'k-', lw=0.2, alpha=0.3)
                    axes[i, j].set_xlabel('x')
                    axes[i, j].set_ylabel('y')
                    axes[i, j].set_aspect('equal')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_png:
            filename = f'{model_type.lower()}_2d_solver_step_{step_number:03d}.png'
            plt.savefig(filename, dpi=120, bbox_inches='tight')
            print(f"    Saved plot: {filename}")
        
        plt.close(fig)
    
    def _plot_damage_residual(self, ax, damage_residual, gp_coords=None, gp_residuals=None):
        """Plot damage residual field."""
        try:
            if gp_coords is not None and gp_residuals is not None:
                # Plot Gauss points
                residual_min = np.min(gp_residuals)
                residual_max = np.max(gp_residuals)
                residual_abs_max = np.max(np.abs(gp_residuals))
                
                print(f"Damage residual (Gauss pts): min={residual_min:.2e}, max={residual_max:.2e}")
                
                if residual_abs_max > 1e-12:
                    scatter = ax.scatter(gp_coords[:, 0], gp_coords[:, 1], 
                                        c=gp_residuals, cmap='RdBu_r', s=.1, alpha=0.5)
                    ax.set_title(f'Damage Residual at Gauss Points\n(min={residual_min:.2e}, max={residual_max:.2e})')
                    plt.colorbar(scatter, ax=ax)
                else:
                    ax.scatter(gp_coords[:, 0], gp_coords[:, 1], c='gray', s=.1, alpha=0.5)
                    ax.set_title(f'Damage Residual at Gauss Points (max={residual_abs_max:.2e})')
            
            elif damage_residual is not None:
                # Plot nodal values
                residual_min = np.min(damage_residual)
                residual_max = np.max(damage_residual)
                residual_abs_max = np.max(np.abs(damage_residual))
                
                print(f"Damage residual (nodal): min={residual_min:.2e}, max={residual_max:.2e}")
                
                if residual_abs_max > 1e-12:
                    # Use asymmetric color scale based on actual data range
                    if residual_min < 0 and residual_max > 0:
                        vmax = max(abs(residual_min), abs(residual_max))
                        levels = np.linspace(-vmax, vmax, 21)
                    else:
                        levels = np.linspace(residual_min, residual_max, 21)
                    
                    im5 = ax.tricontourf(self.tri_obj, damage_residual,
                                        levels=levels, cmap='RdBu_r', extend='both')
                    ax.set_title(f'Damage Residual ∂E/∂d\n(min={residual_min:.2e}, max={residual_max:.2e})')
                    plt.colorbar(im5, ax=ax)
                else:
                    im5 = ax.tricontourf(self.tri_obj, damage_residual, levels=20, cmap='RdBu_r')
                    ax.set_title(f'Damage Residual ∂E/∂d (max={residual_abs_max:.2e})')
                    plt.colorbar(im5, ax=ax)
            else:
                ax.set_title('Damage Residual (No Data)')
                
        except Exception as e:
            ax.set_title('Damage Residual (Error)')
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', 
                   transform=ax.transAxes)
    
    def _plot_radial_profiles(self, ax, u, v, damage_residual=None, gp_coords=None, gp_residuals=None):
        """Plot radial profiles of displacement and residual."""
        r = np.sqrt(self.x**2 + self.y**2)
        r[r < 1e-12] = 1e-12  # Avoid division by zero
        
        # Displacement profiles
        ur_film = (u[:, 0] * self.x + u[:, 1] * self.y) / r
        ur_substrate = (v[:, 0] * self.x + v[:, 1] * self.y) / r
        
        try:
            # Create subplot with twin y-axes
            ax2 = ax.twinx()
            
            # Plot displacements on left y-axis
            line1 = ax.plot(r, ur_film, 'b.', alpha=0.6, label='Film ur')
            line2 = ax.plot(r, ur_substrate, 'r.', alpha=0.6, label='Substrate ur')
            ax.set_xlabel('Distance from center')
            ax.set_ylabel('Radial displacement', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            
            # Plot residual on right y-axis if available
            if gp_coords is not None and gp_residuals is not None:
                r_gp = np.sqrt(gp_coords[:, 0]**2 + gp_coords[:, 1]**2)
                line3 = ax2.plot(r_gp, gp_residuals, 'g.', alpha=0.6, label='Damage Residual (GP)')
                residual_label = 'Damage Residual (GP)'
            elif damage_residual is not None:
                line3 = ax2.plot(r, damage_residual, 'g.', alpha=0.6, label='Damage Residual (Nodal)')
                residual_label = 'Damage Residual (Nodal)'
            else:
                line3 = []
            
            if line3:
                ax2.set_ylabel('Damage Residual', color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                
                # Combined legend
                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='upper left')
            else:
                ax.legend()
            
            ax.set_title('Radial Profiles: Displacement & Residual')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            # Fallback: just plot displacements
            ax.plot(r, ur_film, 'b.', alpha=0.6, label='Film ur')
            ax.plot(r, ur_substrate, 'r.', alpha=0.6, label='Substrate ur')
            ax.set_title('Radial Displacement vs Distance')
            ax.set_xlabel('Distance from center')
            ax.set_ylabel('Radial displacement')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def plot_mesh(self, boundary_nodes, radius, save_png=True):
        """Plot the mesh with boundary nodes highlighted."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Plot triangulation
        ax.triplot(self.x, self.y, self.connectivity, 'b-', lw=0.5, alpha=0.7)
        
        # Plot all nodes
        ax.scatter(self.x, self.y, c='red', s=10, alpha=0.6, label='Interior Nodes')
        
        # Highlight boundary nodes
        ax.scatter(self.x[boundary_nodes], self.y[boundary_nodes], 
                   c='green', s=40, label='Boundary Nodes (Radial Tension)', zorder=5)
        
        # Draw circle boundary
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = radius * np.cos(theta)
        circle_y = radius * np.sin(theta)
        ax.plot(circle_x, circle_y, 'k--', lw=2, alpha=0.8, label='Domain Boundary')
        
        ax.set_title(f'Circular Domain Mesh\n{len(self.connectivity)} elements, {len(self.x)} nodes')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_png:
            plt.savefig('circular_mesh.png', dpi=120, bbox_inches='tight')
            print("✓ Saved mesh plot: circular_mesh.png")
        
        plt.show()