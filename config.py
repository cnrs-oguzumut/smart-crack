"""
Configuration file for 2D Phase-Field Fracture simulation.
Contains all simulation parameters and constants.
"""

import numpy as np

class SimulationConfig:
    """Configuration class for simulation parameters."""
    
    # Mesh parameters
    RADIUS = 150
    
    
    # Material properties
    E_FILM = 1.0
    E_SUBSTRATE = 0.0637
    NU_FILM = 0.3
    NU_SUBSTRATE = 0.3
    HF=0.25
    HS=25.
    CHI=769.
    GC=1.
    AT1=1.
    AT2=2.
    
    # Pha   se-field parameters
    LENGTH_SCALE = 3.
    MESH_RESOLUTION = LENGTH_SCALE/3
    COUPLING_PARAMETER = 1.75
    SUBSTRATE_STIFFNESS = 1
    PSI_C_AT1 = 0.5
    PSI_C_AT2 = 0.0
    
    # Solver parameters
    MAX_NEWTON_ITER = 1000
    NEWTON_RTOL = 1e-5
    NEWTON_ATOL = 1e-5
    MAX_ALT_ITER = 500
    ALT_TOL = 1e-5
    
    # Simulation parameters
    N_STEPS = 100
    MAX_DISPLACEMENT = 3.
    PLOT_FREQUENCY = 1
    LOADING_MODE = "biaxial"
    BIAXIALITY_RATIO=.95

    
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