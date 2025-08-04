"""
Boundary conditions module for radial loading problems.
Handles displacement and constraint enforcement.
"""

import numpy as np
from config import PETSC_AVAILABLE

if PETSC_AVAILABLE:
    from petsc4py import PETSc

class RadialBoundaryConditions:
    """Handles radial displacement boundary conditions."""
    
    def __init__(self, mesh_coords, boundary_nodes):
        """Initialize with mesh coordinates and boundary nodes."""
        self.x = mesh_coords[:, 0]
        self.y = mesh_coords[:, 1]
        self.boundary_nodes = boundary_nodes
        self.applied_displacement = 0.0
    
    def set_applied_displacement(self, displacement):
        """Set the applied radial displacement."""
        self.applied_displacement = displacement
    
    def apply_displacement_bcs(self, u, v, R_u, R_v):
        """Apply displacement boundary conditions to residual."""
        if self.applied_displacement == 0.0:
            return
        
        for node_idx in self.boundary_nodes:
            x_node, y_node = self.x[node_idx], self.y[node_idx]
            norm = np.sqrt(x_node**2 + y_node**2)
            
            if norm < 1e-9:
                continue
            
            # Unit normal vector pointing radially outward
            nx, ny = x_node / norm, y_node / norm
            
            # Target displacements (radial expansion)
            u_target_x = self.applied_displacement * nx
            u_target_y = self.applied_displacement * ny
            


            # Apply boundary conditions to substrate (v)
            R_v[node_idx, 0] = v[node_idx, 0] - u_target_x
            R_v[node_idx, 1] = v[node_idx, 1] - u_target_y


    def apply_displacement_jacobian_bcs(self, J_petsc, n_nodes):
        """Apply boundary conditions to displacement Jacobian."""
        # Substrate DOFs (vx, vy) - film DOFs are commented out
        bc_dofs = []
        for node in self.boundary_nodes:
            bc_dofs.extend([2 * n_nodes + 2 * node, 2 * n_nodes + 2 * node + 1])
        
        bc_rows = np.array(bc_dofs, dtype=PETSc.IntType)
        J_petsc.zeroRows(bc_rows, diag=1.0)


    # import numpy as np
    # from petsc4py import PETSc

    # def apply_displacement_bcs(self, u, v, R_u, R_v):
    #     """Apply displacement boundary conditions to residual for rectangular domain."""
    #     if self.applied_displacement == 0.0:
    #         return
        
    #     # Get domain bounds
    #     x_min, x_max = np.min(self.x), np.max(self.x)
    #     y_min, y_max = np.min(self.y), np.max(self.y)
        
    #     # Tolerance for identifying boundary nodes
    #     tolerance = (x_max - x_min) * 0.01  # 1% of domain width
        
    #     for node_idx in self.boundary_nodes:
    #         x_node, y_node = self.x[node_idx], self.y[node_idx]
            
    #         # Identify which boundary this node belongs to
    #         is_left = abs(x_node - x_min) < tolerance
    #         is_right = abs(x_node - x_max) < tolerance
            
    #         # Apply boundary conditions based on boundary type
    #         if is_left:
    #             # Fix left boundary COMPLETELY (both x and y displacement = 0)
    #             u_target_x = 0.0  # Fixed at zero
    #             u_target_y = 0.0  # Fixed at zero
                
    #             R_v[node_idx, 0] = v[node_idx, 0] - u_target_x  # u_x = 0
    #             R_v[node_idx, 1] = v[node_idx, 1] - u_target_y  # u_y = 0
                
    #         elif is_right:
    #             # Pull right boundary in x direction
    #             u_target_x = self.applied_displacement  # Pull right (positive x)
    #             u_target_y = 0.0  # No y displacement (or could be free)
                
    #             # Constrain x-displacement to pull right
    #             R_v[node_idx, 0] = v[node_idx, 0] - u_target_x
    #             # Option 1: Also constrain y displacement to zero
    #             R_v[node_idx, 1] = v[node_idx, 1] - u_target_y
    #             # Option 2: Leave y free (comment out the line above)


    # def apply_displacement_jacobian_bcs(self, J_petsc, n_nodes):
    #     """Apply boundary conditions to displacement Jacobian for rectangular domain."""
    #     if self.applied_displacement == 0.0:
    #         return
        
    #     # Get domain bounds
    #     x_min, x_max = np.min(self.x), np.max(self.x)
    #     y_min, y_max = np.min(self.y), np.max(self.y)
    #     tolerance = (x_max - x_min) * 0.01
        
    #     # Collect DOFs that need boundary conditions
    #     bc_dofs = []
        
    #     for node_idx in self.boundary_nodes:
    #         x_node, y_node = self.x[node_idx], self.y[node_idx]
            
    #         # Identify boundary type
    #         is_left = abs(x_node - x_min) < tolerance
    #         is_right = abs(x_node - x_max) < tolerance
            
    #         if is_left:
    #             # Fix left boundary: constrain both x and y DOFs
    #             # Substrate DOFs: vx at (2 * n_nodes + 2 * node), vy at (2 * n_nodes + 2 * node + 1)
    #             bc_dofs.extend([2 * n_nodes + 2 * node_idx,       # vx DOF
    #                         2 * n_nodes + 2 * node_idx + 1])   # vy DOF
                
    #         elif is_right:
    #             # Pull right boundary: constrain x DOF, and optionally y DOF
    #             bc_dofs.append(2 * n_nodes + 2 * node_idx)        # vx DOF
    #             # Option 1: Also constrain y DOF
    #             bc_dofs.append(2 * n_nodes + 2 * node_idx + 1)    # vy DOF
    #             # Option 2: Leave y free (comment out the line above)
        
    #     # Apply boundary conditions to Jacobian
    #     bc_rows = np.array(bc_dofs, dtype=PETSc.IntType)
    #     J_petsc.zeroRows(bc_rows, diag=1.0)








class DamageBoundaryConditions:
    """Handles damage boundary conditions and constraints."""
    
    def __init__(self, irreversibility_threshold=0.0):
        """Initialize damage boundary conditions."""
        self.irreversibility_threshold = irreversibility_threshold
    
    def get_effective_lower_bounds(self, d_old):
        """Get effective lower bounds for damage with conditional irreversibility."""
        if self.irreversibility_threshold < 1.0:
            mask = d_old > self.irreversibility_threshold
            bounds = np.zeros_like(d_old)
            bounds[mask] = d_old[mask]
            return bounds
        else:
            return d_old.copy()
    
    def get_effective_upper_bounds(self, d):
        """Upper bounds are always 1."""
        return np.ones_like(d)
    
    def project_damage_bounds(self, d_new, d_old):
        """Project damage to satisfy bounds."""
        lower = self.get_effective_lower_bounds(d_old)
        upper = self.get_effective_upper_bounds(d_new)
        return np.minimum(np.maximum(d_new, lower), upper)
    
    def constraint_violation(self, d, d_old):
        """Compute constraint violation."""
        lower = self.get_effective_lower_bounds(d_old)
        upper = self.get_effective_upper_bounds(d)
        lower_viol = np.linalg.norm(np.maximum(0, lower - d))
        upper_viol = np.linalg.norm(np.maximum(0, d - upper))
        return lower_viol + upper_viol
    
    def compute_active_set_residual(self, physical_residual, d, d_old):
        """Compute residual using active-set method."""
        tol = 1e-12
        effective_lower = self.get_effective_lower_bounds(d_old)
        
        # Active constraints: d at lower bound AND positive residual
        constraint_active = (d <= effective_lower + tol) & (physical_residual > tol)
        
        # Modify residual for active constraints
        R_d_solver = physical_residual.copy()
        R_d_solver[constraint_active] = d[constraint_active] - effective_lower[constraint_active]
        
        return R_d_solver, constraint_active