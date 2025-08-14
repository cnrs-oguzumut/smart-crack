"""
Boundary conditions using deformation gradient approach for substrate displacement.
Uses Y - X = F·X - X = X·(F - I) = u on boundary nodes.
"""

import numpy as np
from config import PETSC_AVAILABLE

if PETSC_AVAILABLE:
    from petsc4py import PETSc


class RadialBoundaryConditions:
    """Handles displacement boundary conditions using deformation gradient approach."""
    
    def __init__(self, mesh_coords, boundary_nodes, loading_mode='uniaxial', biaxiality_ratio=1.0):
        """
        Initialize with mesh coordinates and boundary nodes.
        
        Parameters:
        - mesh_coords: Node coordinates (reference configuration X)
        - boundary_nodes: Indices of boundary nodes
        - loading_mode: 'uniaxial', 'biaxial', or 'circular'
        - biaxiality_ratio: Ratio for biaxial loading (strain_y / strain_x)
        """
        self.x = mesh_coords[:, 0]
        self.y = mesh_coords[:, 1]
        self.boundary_nodes = boundary_nodes
        self.loading_mode = loading_mode
        self.biaxiality_ratio = biaxiality_ratio
        
        # Applied strain parameter (will be set during simulation)
        self.applied_strain = 0.0
        
        # Determine domain type
        x_range = np.max(self.x) - np.min(self.x)
        y_range = np.max(self.y) - np.min(self.y)
        
        if abs(x_range - y_range) / max(x_range, y_range) < 0.5:
            self.domain_type = 'rectangular'
        else:
            self.domain_type = 'circular'
        
        print(f"Deformation gradient BC: {self.domain_type} domain detected")
        print(f"Loading mode: {self.loading_mode}")
        if self.loading_mode == 'biaxial':
            print(f"Biaxiality ratio: {self.biaxiality_ratio}")
        
        # Initialize deformation gradient
        self._update_deformation_gradient()
    
    def set_applied_displacement(self, displacement):
        """Set applied displacement and convert to strain for deformation gradient."""
        if self.domain_type == 'rectangular':
            domain_size = max(np.max(self.x) - np.min(self.x), np.max(self.y) - np.min(self.y))
            self.applied_strain = displacement / domain_size if domain_size > 0 else displacement
        else:
            self.applied_strain = displacement
        
        self._update_deformation_gradient()
        # print(f"Applied strain: {self.applied_strain:.6f}")
        
        # # ADD OPTION 2 DEBUG HERE:
        # print(f"DEBUG - Applied displacement: {displacement}")
        # print(f"DEBUG - Domain size: {max(np.max(self.x) - np.min(self.x), np.max(self.y) - np.min(self.y))}")
        # print(f"DEBUG - Loading mode: {self.loading_mode}")
        # print(f"DEBUG - Biaxiality ratio: {self.biaxiality_ratio}")
        # print(f"DEBUG - F matrix:\n{self.F}")
        # print(f"DEBUG - F-I matrix:\n{self.F_minus_I}")
        
        # # Test a sample point to see displacement direction
        # test_x, test_y = 32.0, 19.2  # A positive coordinate
        # u_test_x, u_test_y = self._compute_displacement_from_deformation_gradient(test_x, test_y)
        # print(f"DEBUG - Test point ({test_x}, {test_y}) → displacement: ({u_test_x:.6f}, {u_test_y:.6f})")



    def set_applied_strain(self, strain):
        """Directly set the applied strain."""
        self.applied_strain = strain
        self._update_deformation_gradient()
        print(f"Applied strain: {self.applied_strain:.6f}")
    
    def _update_deformation_gradient(self):
        """Update strain tensor (stored as F) based on loading mode and applied strain for linear elasticity."""
        if self.loading_mode == 'uniaxial':
            # Uniaxial loading: εxx = applied_strain, εyy = 0 (plane stress)
            self.F = np.array([
                [self.applied_strain, 0.0],
                [0.0, 0.0]  # Plane stress (free in y-direction)
            ])
            print(f"Uniaxial strain: εxx = {self.applied_strain:.6f}, εyy = 0")
            
        elif self.loading_mode == 'biaxial':
            # Biaxial loading: εxx = strain_x, εyy = strain_y
            print("self.applied_strain",self.applied_strain)
            strain_x = self.applied_strain
            strain_y = self.applied_strain * self.biaxiality_ratio
            self.F = np.array([
                [strain_x, 0.0],
                [0.0, strain_y]
            ])
            print(f"Biaxial strain: εxx = {strain_x:.6f}, εyy = {strain_y:.6f}")
            
        elif self.loading_mode == 'shear':
            # Pure shear loading: εxy = γ/2 (engineering shear strain γ)
            shear_strain = self.applied_strain / 2.0  # Convert engineering to tensor shear
            self.F = np.array([
                [0.0, shear_strain],
                [shear_strain, 0.0]  # Symmetric tensor
            ])
            print(f"Shear strain: γ = {self.applied_strain:.6f}, εxy = {shear_strain:.6f}")
            
        elif self.loading_mode == 'circular':
            # Isotropic loading: εxx = εyy = applied_strain
            self.F = np.array([
                [self.applied_strain, 0.0],
                [0.0, self.applied_strain]
            ])
            print(f"Isotropic strain: ε = {self.applied_strain:.6f}")
            
        else:
            # Default to zero strain
            self.F = np.zeros((2, 2))
            print("Default strain: Zero tensor")
            
        # Store strain tensor directly (no F-I needed for linear elasticity)
        self.F_minus_I = self.F.copy()

    def _compute_displacement_from_deformation_gradient(self, x_coord, y_coord):
        """
        Compute displacement from strain tensor for linear elasticity.
        u = ε · X where X = [x_coord, y_coord]
        
        Parameters:
        - x_coord, y_coord: Reference coordinates
        Returns:
        - u_x, u_y: Displacement components
        """
        X = np.array([x_coord, y_coord])
        u = np.dot(self.F_minus_I, X)  # F_minus_I now contains strain tensor
        return u[0], u[1]
    
    def apply_displacement_bcs(self, u, v, R_u, R_v):
        """Apply displacement boundary conditions to substrate using deformation gradient."""
        if abs(self.applied_strain) < 1e-12:
            return
        # print(f"Film boundary displacements before BC:")
        # print(f"  u[boundary[0:3]] = {u[self.boundary_nodes[:3]]}")
        
        # Apply to substrate displacement field (v)
        for node_idx in self.boundary_nodes:
            x_coord = self.x[node_idx]
            y_coord = self.y[node_idx]
            
            # Calculate target displacement using deformation gradient
            u_target_x, u_target_y = self._compute_displacement_from_deformation_gradient(x_coord, y_coord)
            
            # Apply boundary condition to substrate
            R_v[node_idx, 0] = v[node_idx, 0] - u_target_x
            R_v[node_idx, 1] = v[node_idx, 1] - u_target_y
    
            # # Apply boundary condition to film
            # R_u[node_idx, 0] = (u[node_idx, 0] - u_target_x)
            # R_u[node_idx, 1] = (u[node_idx, 1] - u_target_y
        # print(f"Film boundary residuals after BC:")
        # print(f"  R_u[boundary[0:3]] = {R_u[self.boundary_nodes[:3]]}")


    def apply_displacement_jacobian_bcs(self, J_petsc, n_nodes):
        """Apply boundary conditions to displacement Jacobian."""
        if abs(self.applied_strain) < 1e-12:
            return
        
        if PETSC_AVAILABLE:
            bc_dofs = []

            for node_idx in self.boundary_nodes:
                bc_dofs.extend([
                    # Film DOFs (u) - indices 0 to 2*n_nodes-1
                    # 2 * node_idx,     # ux DOF 
                    # 2 * node_idx + 1 # uy DOF
                    
                    # Substrate DOFs (v) - indices 2*n_nodes to 4*n_nodes-1  
                    2 * n_nodes + 2 * node_idx,     # vx DOF
                    2 * n_nodes + 2 * node_idx + 1  # vy DOF
                ])


            # # Apply to substrate DOFs (v) - these are at indices 2*n_nodes + 2*node_idx
            # for node_idx in self.boundary_nodes:
            #     bc_dofs.extend([
            #         2 * n_nodes + 2 * node_idx,       # vx DOF
            #         2 * n_nodes + 2 * node_idx + 1    # vy DOF
            #     ])

            # # Apply to substrate DOFs (v) - these are at indices 2*n_nodes + 2*node_idx
            # # for node_idx in self.boundary_nodes:
            # #     bc_dofs.extend([
            # #         0 * n_nodes + 2 * node_idx,       # ux DOF
            # #         0 * n_nodes + 2 * node_idx + 1    # uy DOF
            # #     ])



            if bc_dofs:
                bc_rows = np.array(bc_dofs, dtype=PETSc.IntType)
                J_petsc.zeroRows(bc_rows, diag=1.0)
    
    def get_loading_description(self):
        """Get a description of the current loading condition."""
        if self.loading_mode == 'uniaxial':
            return f"Uniaxial deformation gradient (ε_x = {self.applied_strain:.4f}, ε_y = 0)"
        elif self.loading_mode == 'biaxial':
            strain_y = self.applied_strain * self.biaxiality_ratio
            return f"Biaxial deformation gradient (ε_x = {self.applied_strain:.4f}, ε_y = {strain_y:.4f})"
        elif self.loading_mode == 'circular':
            return f"Radial deformation gradient (ε_r = {self.applied_strain:.4f})"
        else:
            return f"Unknown loading mode: {self.loading_mode}"
    
    def get_current_deformation_gradient(self):
        """Return current deformation gradient tensor."""
        return self.F.copy()
    
    def get_strain_tensor(self):
        """Return the strain tensor (small strain approximation: ε = (F - I))."""
        return self.F_minus_I.copy()
    
    def get_boundary_displacements(self):
        """Get computed displacements for all boundary nodes."""
        displacements = []
        for node_idx in self.boundary_nodes:
            x_coord = self.x[node_idx]
            y_coord = self.y[node_idx]
            u_x, u_y = self._compute_displacement_from_deformation_gradient(x_coord, y_coord)
            displacements.append([node_idx, x_coord, y_coord, u_x, u_y])
        return np.array(displacements)
    
    def get_boundary_info(self):
        """Get information about boundary setup for debugging."""
        info = {
            'domain_type': self.domain_type,
            'loading_mode': self.loading_mode,
            'total_boundary_nodes': len(self.boundary_nodes),
            'applied_strain': self.applied_strain,
            'loading_description': self.get_loading_description(),
            'deformation_gradient': self.F.tolist(),
            'strain_tensor': self.F_minus_I.tolist()
        }
        
        if self.loading_mode == 'biaxial':
            info['biaxiality_ratio'] = self.biaxiality_ratio
            info['strain_y'] = self.applied_strain * self.biaxiality_ratio
        
        return info
    
    def set_biaxiality_ratio(self, ratio):
        """Set the biaxiality ratio and update deformation gradient."""
        self.biaxiality_ratio = ratio
        self._update_deformation_gradient()
        print(f"Updated biaxiality ratio: {ratio}")
    
    def set_applied_displacement_biaxial(self, strain_x, strain_y):
        """Set different strains for x and y directions."""
        self.applied_strain = strain_x
        if abs(strain_x) > 1e-12:
            self.biaxiality_ratio = strain_y / strain_x
        self._update_deformation_gradient()
        print(f"Set biaxial strains: ε_x = {strain_x:.6f}, ε_y = {strain_y:.6f}")
        print(f"Biaxiality ratio: {self.biaxiality_ratio:.6f}")


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


# Test function
def test_deformation_gradient_bc():
    """Test deformation gradient boundary conditions."""
    print("=" * 80)
    print("TESTING DEFORMATION GRADIENT BOUNDARY CONDITIONS")
    print("=" * 80)
    
    # Create test mesh
    x = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    y = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    mesh_coords = np.column_stack((x, y))
    boundary_nodes = np.array([0, 1, 2, 3])
    
    # Test different loading modes
    test_cases = [
        ('uniaxial', 1.0),
        ('biaxial', 1.0),  # Equibiaxial
        ('biaxial', 0.5),  # Plane strain-like
        ('circular', 1.0)
    ]
    
    for loading_mode, biaxiality_ratio in test_cases:
        print(f"\n--- Testing {loading_mode} loading (ratio={biaxiality_ratio}) ---")
        
        bc = RadialBoundaryConditions(
            mesh_coords, boundary_nodes,
            loading_mode=loading_mode,
            biaxiality_ratio=biaxiality_ratio
        )
        
        # Set applied strain
        bc.set_applied_strain(0.1)
        
        print("\nBoundary condition info:")
        info = bc.get_boundary_info()
        for key, value in info.items():
            if isinstance(value, (list, np.ndarray)):
                print(f"  {key}: {np.array(value)}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nBoundary node displacements:")
        displacements = bc.get_boundary_displacements()
        for i, (node_idx, x_coord, y_coord, u_x, u_y) in enumerate(displacements):
            print(f"  Node {int(node_idx)}: X=({x_coord:5.1f},{y_coord:5.1f}) → u=({u_x:8.4f},{u_y:8.4f})")
    
    print(f"\n✅ Deformation gradient tests completed!")


if __name__ == "__main__":
    test_deformation_gradient_bc()