"""
Finite element utilities for triangular elements.
Contains shape functions, Gauss integration, and element matrices.
"""

import numpy as np

class TriangularElement:
    """Utilities for linear triangular finite elements."""
    
    @staticmethod
    def shape_functions(xi, eta):
        """Linear triangular shape functions and derivatives."""
        zeta = 1.0 - xi - eta
        N = np.array([zeta, xi, eta])
        dN_dxi = np.array([-1., 1., 0.])
        dN_deta = np.array([-1., 0., 1.])
        return N, dN_dxi, dN_deta
    
    @staticmethod
    def gauss_points():
        """Gauss points and weights for triangular elements."""
        # xi = np.array([1/6, 2/3, 1/6])
        # eta = np.array([1/6, 1/6, 2/3])
        # w = np.array([1/6, 1/6, 1/6])
        xi = np.array([1/3])
        eta = np.array([1/3]) 
        w = np.array([1/2])  # Weight = 1/2 (area of reference triangle)

        return xi, eta, w
    
    @staticmethod
    def compute_jacobian(nodes_coords, xi, eta):
        """Compute Jacobian matrix and derivatives."""
        _, dN_dxi, dN_deta = TriangularElement.shape_functions(xi, eta)
        x_e, y_e = nodes_coords[:, 0], nodes_coords[:, 1]
        
        J = np.array([
            [np.dot(dN_dxi, x_e), np.dot(dN_dxi, y_e)],   # dx/dξ, dy/dξ
            [np.dot(dN_deta, x_e), np.dot(dN_deta, y_e)]  # dx/dη, dy/dη
        ])        
        detJ = np.linalg.det(J)
        if detJ <= 1e-12:
            raise ValueError(f"Negative/zero Jacobian: {detJ}")
        
        J_inv = np.linalg.inv(J)
        dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
        
        return detJ, dN_dx, dN_dy
    
    @staticmethod
    def strain_displacement_matrix(dN_dx, dN_dy):
        """Construct B-matrix (strain-displacement matrix)."""
        B = np.zeros((3, 6))
        for i in range(3):
            B[0, 2*i] = dN_dx[i]      # εxx
            B[1, 2*i+1] = dN_dy[i]    # εyy
            B[2, 2*i] = dN_dy[i]      # γxy
            B[2, 2*i+1] = dN_dx[i]    # γxy
        return B

class ElementAssembler:
    """Assembles element contributions to global system."""
    
    def __init__(self, mesh_coords, connectivity):
        """Initialize with mesh data."""
        self.x = mesh_coords[:, 0]
        self.y = mesh_coords[:, 1]
        self.connectivity = connectivity
        self.element = TriangularElement()
    
    def get_element_coordinates(self, element_id):
        """Get coordinates of element nodes."""
        nodes = self.connectivity[element_id]
        coords = np.column_stack((self.x[nodes], self.y[nodes]))
        return coords
    
    def integrate_over_element(self, element_id, integrand_function):
        """Integrate a function over an element using Gauss quadrature."""
        nodes = self.connectivity[element_id]
        coords = self.get_element_coordinates(element_id)
        
        xi_gp, eta_gp, w_gp = self.element.gauss_points()
        integral = 0.0
        
        for gp in range(len(xi_gp)):
            N, _, _ = self.element.shape_functions(xi_gp[gp], eta_gp[gp])
            detJ, dN_dx, dN_dy = self.element.compute_jacobian(coords, xi_gp[gp], eta_gp[gp])
            
            # Call the integrand function with current Gauss point data
            contribution = integrand_function(N, dN_dx, dN_dy, detJ, nodes, gp)
            integral += contribution * w_gp[gp]
        
        return integral