"""
Material models and damage functions for phase-field fracture.
Direct classes without unnecessary inheritance.
"""

import numpy as np

class FilmMaterial:
    """Film material with crystal elasticity and damage coupling."""
    
    def __init__(self, E=1.0, nu=0.3, hf=1.0, hs=1.0, chi=1.0):
        # Note: E, nu are ignored - we use Nickel crystal constants
        self.hf = hf
        self.hs = hs  # You might not need this for crystal
        self.chi = chi
        self.D = self._compute_stiffness_matrix_crystal(0.0)  # Default 0° orientation

    def degradation_function(self, damage):
        """Damage degradation function g(d)."""
        return (1 - damage)**2  # Standard quadratic degradation
        
    def _compute_stiffness_matrix_crystal(self, crystal_orientation_degrees):
        """Compute rotated crystal stiffness matrix for Nickel."""
        
        # Cubic stiffness constants for Nickel (GPa)
        C11, C12, C44 = 246.5/124.7, 147.3/124.7, 124.7/124.7
        
        # 6x6 cubic stiffness in Voigt notation
        C_voigt = np.array([
            [C11, C12, C12, 0, 0, 0],
            [C12, C11, C12, 0, 0, 0],
            [C12, C12, C11, 0, 0, 0],
            [0, 0, 0, C44, 0, 0],
            [0, 0, 0, 0, C44, 0],
            [0, 0, 0, 0, 0, C44]
        ])
        
        # Convert Voigt to 4th-order tensor
        C_tensor = np.zeros((3, 3, 3, 3))
        voigt_map = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]
        
        for I, (i,j) in enumerate(voigt_map):
            for J, (k,l) in enumerate(voigt_map):
                C_tensor[i,j,k,l] = C_voigt[I,J]
                # Apply tensor symmetries
                C_tensor[j,i,k,l] = C_voigt[I,J]
                C_tensor[i,j,l,k] = C_voigt[I,J] 
                C_tensor[j,i,l,k] = C_voigt[I,J]
        
        # Rotation matrix (z-axis rotation)
        theta = np.radians(crystal_orientation_degrees)
        Q = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Rotate tensor
        C_rot_tensor = np.einsum('ia,jb,kc,ld,abcd->ijkl', Q, Q, Q, Q, C_tensor)
        
        # Convert back to Voigt notation
        C_rot_voigt = np.zeros((6, 6))
        for I, (i,j) in enumerate(voigt_map):
            for J, (k,l) in enumerate(voigt_map):
                C_rot_voigt[I,J] = C_rot_tensor[i,j,k,l]
        
        # Plane stress reduction
        in_plane_idx = [0, 1, 5]  # xx, yy, xy
        out_plane_idx = [2, 3, 4]  # zz, yz, xz
        
        C11_block = C_rot_voigt[np.ix_(in_plane_idx, in_plane_idx)]
        C12_block = C_rot_voigt[np.ix_(in_plane_idx, out_plane_idx)]
        C22_block = C_rot_voigt[np.ix_(out_plane_idx, out_plane_idx)]
        
        D_plane_stress = C11_block - C12_block @ np.linalg.inv(C22_block) @ C12_block.T
        
        return self.hf * self.chi * D_plane_stress
        
    def compute_stress(self, strain, damage, crystal_angle=0.0):
        """Compute stress with damage degradation and crystal orientation."""
        if crystal_angle != 0.0:
            D_crystal = self._compute_stiffness_matrix_crystal(crystal_angle)
        else:
            D_crystal = self.D  # Use precomputed for 0°
            
        g_d = self.degradation_function(damage)
        return g_d * np.dot(D_crystal, strain)

class SubstrateMaterial:
    """Substrate material with isotropic elasticity (no damage)."""
    
    def __init__(self, E=1.0, nu=0.3, hf=1.0, hs=1.0, chi=1.0):
        self.E = E
        self.nu = nu
        self.hf = hf
        self.hs = hs
        self.chi = chi
        self.D = self._compute_isotropic_stiffness()

    def _compute_isotropic_stiffness(self):
        """Compute plane stress stiffness matrix for isotropic material."""
        factor = self.chi * self.hs * self.E / (1 - self.nu**2)
        D = factor * np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1 - self.nu) / 2]
        ])
        return D
    
    def compute_stress(self, strain):
        """Compute stress without damage."""
        return np.dot(self.D, strain)


class DamageModel:
    """Damage model for phase-field fracture."""
    
    def __init__(self, model_type='AT1', length_scale=0.013):
        self.model_type = model_type.upper()
        self.l = length_scale
        self.psi_c = 0.5 if model_type == 'AT1' else 0.0
    
    def damage_potential_derivative(self, d):
        """First derivative of damage potential w'(d)."""
        if self.model_type == 'AT1':
            return np.ones_like(d)
        else:  # AT2
            return 2.0 * d
    
    def damage_potential_second_derivative(self, d):
        """Second derivative of damage potential w''(d)."""
        if self.model_type == 'AT1':
            return np.full_like(d, 1e-8)  # Small regularization
        else:  # AT2
            return 2.0 * np.ones_like(d)
    
    def compute_driving_force(self, elastic_energy, damage):
        """Compute damage driving force."""
        return 2.0 * (1 - damage) * elastic_energy


class ElasticEnergyCalculator:
    """Calculates elastic energy for damage driving force."""
    
    @staticmethod
    def compute_positive_energy(strain, stiffness_matrix):
        """Compute positive part of elastic energy density."""
        elastic_energy_density = 0.5 * np.dot(strain, np.dot(stiffness_matrix, strain))
        return elastic_energy_density
    
    @staticmethod
    def compute_strain_from_displacement(u_element, B_matrix):
        """Compute strain from displacement using B-matrix."""
        return np.dot(B_matrix, u_element.flatten())