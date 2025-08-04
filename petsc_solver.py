"""
PETSc solver interface for displacement and damage problems.
Handles both VI (variational inequality) and active-set methods.
"""

import numpy as np
from config import PETSC_AVAILABLE, PETScConfig

if PETSC_AVAILABLE:
    from petsc4py import PETSc

class PETScSolverBase:
    """Base class for PETSc solvers."""
    
    def __init__(self, config):
        """Initialize PETSc solver."""
        self.config = config
        self._init_petsc()
    
    def _init_petsc(self):
        """Initialize PETSc system."""
        if not PETSC_AVAILABLE:
            raise RuntimeError("PETSc not available")
        
        if not PETSc.Sys.isInitialized():
            petsc_options = PETScConfig.get_petsc_options(self.config)
            PETSc.Sys.Initialize(petsc_options)
    
    def create_vector(self, size):
        """Create PETSc vector."""
        return PETSc.Vec().createSeq(size)
    
    def create_matrix(self, size, nnz=60):
        """Create PETSc matrix."""
        mat = PETSc.Mat().createAIJ([size, size], nnz=nnz)
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        return mat

class DisplacementSolver(PETScSolverBase):
    """PETSc solver for displacement problems."""
    
    def __init__(self, config, n_nodes):
        """Initialize displacement solver."""
        super().__init__(config)
        self.n_nodes = n_nodes
        self.dof_size = 4 * n_nodes  # u_x, u_y, v_x, v_y
        
        # Create PETSc objects
        self.snes = PETSc.SNES().create()
        self.solution = self.create_vector(self.dof_size)
        self.jacobian = self.create_matrix(self.dof_size, nnz=60)
        
        # Set up SNES
        self.snes.setFunction(self._residual_callback, self.solution.copy())
        self.snes.setJacobian(self._jacobian_callback, self.jacobian)
        self.snes.setFromOptions()
        
        # Callback functions (to be set by parent class)
        self.residual_function = None
        self.jacobian_function = None
    
    def _residual_callback(self, snes, x_petsc, f_petsc):
        """PETSc callback for residual evaluation."""
        if self.residual_function:
            self.residual_function(snes, x_petsc, f_petsc)
    
    def _jacobian_callback(self, snes, x_petsc, J_petsc, P_petsc):
        """PETSc callback for Jacobian evaluation."""
        if self.jacobian_function:
            self.jacobian_function(snes, x_petsc, J_petsc, P_petsc)
    
    def solve(self, initial_guess):
        """Solve displacement problem."""
        self.solution.setArray(initial_guess)
        self.snes.solve(None, self.solution)
        
        reason = self.snes.getConvergedReason()
        iterations = self.snes.getIterationNumber()
        
        if reason > 0:
            solution = self.solution.getArray().copy()
            return True, solution, iterations
        else:
            return False, None, iterations

class DamageSolver(PETScSolverBase):
    """PETSc solver for damage problems with VI capability."""
    
    def __init__(self, config, n_nodes, use_vi_solver=True):
        """Initialize damage solver."""
        super().__init__(config)
        self.n_nodes = n_nodes
        self.use_vi_solver = use_vi_solver
        
        # Create PETSc objects
        self.snes = PETSc.SNES().create()
        self.solution = self.create_vector(n_nodes)
        self.jacobian = self.create_matrix(n_nodes, nnz=30)
        
        if use_vi_solver:
            self._setup_vi_solver()
        else:
            self._setup_active_set_solver()
        
        # Callback functions (to be set by parent class)
        self.residual_function = None
        self.jacobian_function = None
    
    def _setup_vi_solver(self):
        """Set up VI (variational inequality) solver."""
        self.snes.setType('vinewtonrsls')
        
        # Create bounds vectors
        self.lower_bound = self.create_vector(self.n_nodes)
        self.upper_bound = self.create_vector(self.n_nodes)
        
        # Set function first, then bounds
        self.snes.setFunction(self._residual_callback, self.solution.copy())
        self.snes.setJacobian(self._jacobian_callback, self.jacobian)
        self.snes.setVariableBounds(self.lower_bound, self.upper_bound)
        
        print("✓ VI solver configured for damage")
    
    def _setup_active_set_solver(self):
        """Set up active-set solver."""
        self.snes.setFunction(self._residual_callback, self.solution.copy())
        self.snes.setJacobian(self._jacobian_callback, self.jacobian)
        print("✓ Active-set solver configured for damage")
    
    def _residual_callback(self, snes, x_petsc, f_petsc):
        """PETSc callback for residual evaluation."""
        if self.residual_function:
            self.residual_function(snes, x_petsc, f_petsc)
    
    def _jacobian_callback(self, snes, x_petsc, J_petsc, P_petsc):
        """PETSc callback for Jacobian evaluation."""
        if self.jacobian_function:
            self.jacobian_function(snes, x_petsc, J_petsc, P_petsc)
    
    def update_bounds(self, lower_bounds, upper_bounds):
        """Update VI bounds."""
        if self.use_vi_solver:
            self.lower_bound.setArray(lower_bounds)
            self.upper_bound.setArray(upper_bounds)
    
    def solve(self, initial_guess, lower_bounds=None, upper_bounds=None):
        """Solve damage problem."""
        if self.use_vi_solver and lower_bounds is not None:
            self.update_bounds(lower_bounds, upper_bounds)
        
        self.solution.setArray(initial_guess)
        self.snes.solve(None, self.solution)
        
        reason = self.snes.getConvergedReason()
        iterations = self.snes.getIterationNumber()
        
        if reason > 0:
            solution = self.solution.getArray().copy()
            return True, solution, iterations
        else:
            return False, None, iterations
    
    def cleanup(self):
        """Clean up PETSc objects."""
        objects_to_cleanup = [self.snes, self.solution, self.jacobian]
        
        if hasattr(self, 'lower_bound'):
            objects_to_cleanup.extend([self.lower_bound, self.upper_bound])
        
        for obj in objects_to_cleanup:
            if obj is not None:
                try:
                    obj.destroy()
                except:
                    pass