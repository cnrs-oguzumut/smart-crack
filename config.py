"""
Configuration file for 2D Phase-Field Fracture simulation.
Contains all simulation parameters and constants.
"""

import numpy as np

class SimulationConfig:
    """Configuration class for simulation parameters."""
    
    # Mesh parameters
    RADIUS = 1.0
    MESH_RESOLUTION = 0.03
    
    # Material properties
    E_FILM = 1.0
    E_SUBSTRATE = 0.5
    NU_FILM = 0.3
    NU_SUBSTRATE = 0.3
    
    # Phase-field parameters
    LENGTH_SCALE = 0.013
    COUPLING_PARAMETER = 0.34
    SUBSTRATE_STIFFNESS = 0.5
    PSI_C_AT1 = 0.5
    PSI_C_AT2 = 0.0
    
    # Solver parameters
    MAX_NEWTON_ITER = 1000
    NEWTON_RTOL = 1e-6
    NEWTON_ATOL = 1e-6
    MAX_ALT_ITER = 200
    ALT_TOL = 1e-8
    
    # Simulation parameters
    N_STEPS = 80
    MAX_DISPLACEMENT = 1.5
    PLOT_FREQUENCY = 5
    
    # Irreversibility
    IRREVERSIBILITY_THRESHOLD = 0.0
    
    # Files
    STATE_FILE = "simulation_state.npz"
    
    # Solver options
    USE_VI_SOLVER = True
    RESTART_SIMULATION = False

class PETScConfig:
    """PETSc-specific configuration."""
    
    PETSC_OPTIONS = [
        '-snes_type', 'newtonls', 
        '-snes_linesearch_type', 'bt',
        '-ksp_type', 'preonly', 
        '-pc_type', 'lu', 
        '-pc_factor_mat_solver_type', 'mumps'
    ]
    
    @staticmethod
    def get_petsc_options(config):
        """Get PETSc options with runtime values."""
        return PETScConfig.PETSC_OPTIONS + [
            '-snes_rtol', str(config.NEWTON_RTOL), 
            '-snes_atol', str(config.NEWTON_ATOL),
            '-snes_max_it', str(config.MAX_NEWTON_ITER)
        ]

# Check PETSc availability
try:
    from petsc4py import PETSc
    PETSC_AVAILABLE = True
    print("✓ PETSc4py successfully imported")
except ImportError:
    PETSC_AVAILABLE = False
    print("⚠ PETSc4py not available")