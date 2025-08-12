"""
Material models and damage functions for phase-field fracture.
Contains constitutive relations and damage evolution laws.
"""

import numpy as np

class MaterialModel:
    """Base class for material models."""
    
    def __init__(self, E, nu):
        """Initialize material with Young's modulus and Poisson's ratio."""
        self.E = E
        self.nu = nu
        self.D = self._compute_stiffness_matrix()
    
    def _compute_stiffness_matrix_isotropic(self):
        """Compute plane stress stiffness matrix for isotropic material."""
        factor = self.E / (1 - self.nu**2)
        D =factor * np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1 - self.nu) / 2]
        ])
        return D

    def _compute_stiffness_matrix_crystal(self):
        """Nickel single crystal in plane stress (FCC structure) - corrected."""
        # 3D elastic constants for Nickel
        # C11 = 1  # 246.5e9   # Stiffness along <100>
        # C12 = 147.3e9 / 246.5e9    # Cross-coupling
        # C44 = 124.7e9 / 246.5e9   # Shear stiffness <100>{001}
        
        # # Plane stress reduction
        # C11_eff = 3.5#C11 - C12**2/C11
        # C12_eff = 1.5 #C12 - C12**2/C11  
        # C44_eff = 1.  #C44
        
        # D = 0.25*769*np.array([
        #     [C11_eff, C12_eff, 0],
        #     [C12_eff, C11_eff, 0],
        #     [0, 0, C44_eff]
        # ])

        factor = self.E / (1 - self.nu**2)
        D = factor * np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1 - self.nu) / 2]
        ])


        return D

    # Default implementation (will be overridden by subclasses)
    def _compute_stiffness_matrix(self):
        """Default stiffness matrix - isotropic."""
        return self._compute_stiffness_matrix_isotropic()


class FilmMaterial(MaterialModel):
    """Film material with damage coupling - uses single crystal elasticity."""
    
    def __init__(self, E=1.0, nu=0.3):
        super().__init__(E, nu)
    
    def _compute_stiffness_matrix(self):
        """Override to use crystal elasticity for film."""
        return self._compute_stiffness_matrix_crystal()
    
    def compute_stress(self, strain, damage):
        """Compute stress with damage degradation."""
        g_d = self.degradation_function(damage)
        return g_d * np.dot(self.D, strain)
    
    @staticmethod
    def degradation_function(d):
        """Degradation function g(d) = (1-d)^2."""
        return (1 - d)**2 + 1e-6


class SubstrateMaterial(MaterialModel):
    """Substrate material without damage - uses isotropic elasticity."""
    
    def __init__(self, E=0.5, nu=0.3):
        super().__init__(E, nu)
    
    def _compute_stiffness_matrix(self):
        """Override to use isotropic elasticity for substrate."""
        return self._compute_stiffness_matrix_isotropic()
    
    def compute_stress(self, strain):
        """Compute stress without damage."""
        return np.dot(self.D, strain)


class DamageModel:
    """Base class for damage models."""
    
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
        #return max(0, elastic_energy_density)
        return elastic_energy_density
    
    @staticmethod
    def compute_strain_from_displacement(u_element, B_matrix):
        """Compute strain from displacement using B-matrix."""
        return np.dot(B_matrix, u_element.flatten())