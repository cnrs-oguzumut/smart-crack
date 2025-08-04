"""
Main solver class for 2D AT1/AT2 Phase-Field fracture simulation.
Integrates all components for film-substrate system with radial loading.
"""

import numpy as np
import time
import warnings
from config import SimulationConfig, PETSC_AVAILABLE
from mesh_generator import ProfessionalCircularMeshGenerator as CircularMeshGenerator
from material_models import FilmMaterial, SubstrateMaterial, DamageModel, ElasticEnergyCalculator
from finite_element import TriangularElement, ElementAssembler
from petsc_solver import DisplacementSolver, DamageSolver
from boundary_conditions import RadialBoundaryConditions, DamageBoundaryConditions
from visualization import SimulationVisualizer
from state_manager import SimulationStateManager

try:
    from vtk_visualizer import VTKVisualizer
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    print("⚠ VTK not available - skipping VTK visualization")

if PETSC_AVAILABLE:
    from petsc4py import PETSc

warnings.filterwarnings('ignore')

class AT1_2D_ActiveSet_Solver:
    """
    2D AT1/AT2 Phase-Field solver for a circular domain under radial tension.
    This version uses VI solver by default with active-set logic.
    """
    
    def __init__(self, radius=1.0, mesh_resolution=0.05, model_type='AT1', 
                 irreversibility_threshold=0.9, use_vi_solver=True, config=None):
        """Initialize the 2D solver with VI solver by default."""
        print(f"\n=== Initializing 2D {model_type} Solver ===")
        
        # Use provided config or default
        self.config = config or SimulationConfig()
        
        self.model_type = model_type.upper()
        self.radius = radius
        self.mesh_resolution = mesh_resolution  # FIXED: Store the actual mesh resolution
        self.use_vi_solver = use_vi_solver
        self.load_step = 0
        
        # Initialize mesh
        self._setup_mesh(mesh_resolution)

        # Initialize VTK visualizer if available
        if VTK_AVAILABLE:
            try:
                self.vtk_visualizer = VTKVisualizer(self.x, self.y, self.connectivity)
                print("✓ VTK visualizer initialized")
            except Exception as e:
                print(f"⚠ VTK visualizer failed to initialize: {e}")
                self.vtk_visualizer = None
        else:
            self.vtk_visualizer = None
        
        # Initialize materials and damage model
        self._setup_materials()
        
        # Initialize fields
        self.u = np.zeros((self.n_nodes, 2))  # Film displacement
        self.v = np.zeros((self.n_nodes, 2))  # Substrate displacement
        self.d = np.zeros(self.n_nodes)       # Damage
        self.d_old = np.zeros(self.n_nodes)   # Previous damage
        
        # Initialize boundary conditions
        mesh_coords = np.column_stack((self.x, self.y))
        self.bc_displacement = RadialBoundaryConditions(mesh_coords, self.boundary_nodes)
        self.bc_damage = DamageBoundaryConditions(irreversibility_threshold)
        
        # Initialize visualization and state management
        self.visualizer = SimulationVisualizer(self.x, self.y, self.connectivity)
        self.state_manager = SimulationStateManager()
        
        # Initialize PETSc solvers
        self._init_solvers()
        
        solver_type = "VI Solver" if self.use_vi_solver else "Active-Set"
        print(f"✓ Model: {self.model_type} Phase Field with {solver_type}")
    
    def _setup_mesh(self, resolution):
        """Setup circular mesh using mesh generator."""
        mesh_gen = CircularMeshGenerator(self.radius, resolution)
        (self.x, self.y, self.connectivity, self.boundary_nodes, 
         self.n_nodes, self.n_elem) = mesh_gen.generate_mesh()
    
    def _setup_materials(self):
        """Setup material models."""
        self.film_material = FilmMaterial(E=self.config.E_FILM, nu=self.config.NU_FILM)
        self.substrate_material = SubstrateMaterial(E=self.config.E_SUBSTRATE, nu=self.config.NU_SUBSTRATE)
        self.damage_model = DamageModel(self.model_type, self.config.LENGTH_SCALE)
        self.energy_calc = ElasticEnergyCalculator()
        
        # Store parameters for easy access
        self.l = self.config.LENGTH_SCALE
        self.Lambda = self.config.COUPLING_PARAMETER
        self.rho = self.config.SUBSTRATE_STIFFNESS
    
    def _init_solvers(self):
        """Initialize PETSc solvers."""
        if not PETSC_AVAILABLE:
            raise RuntimeError("PETSc not available")
        
        print("DEBUG: Initializing solvers...")
        
        # Displacement solver
        self.disp_solver = DisplacementSolver(self.config, self.n_nodes)
        print("DEBUG: DisplacementSolver created")
        
        # Set displacement callbacks
        self.disp_solver.residual_function = self._displacement_residual
        self.disp_solver.jacobian_function = self._displacement_jacobian
        print("DEBUG: Displacement callbacks assigned")
        
        # Damage solver
        self.damage_solver = DamageSolver(self.config, self.n_nodes, self.use_vi_solver)
        print("DEBUG: DamageSolver created")
        
        # Set damage callbacks based on solver type
        if self.use_vi_solver:
            print("DEBUG: Setting VI solver callbacks")
            self.damage_solver.residual_function = self._damage_residual_vi
            self.damage_solver.jacobian_function = self._damage_jacobian_vi
        else:
            print("DEBUG: Setting active-set solver callbacks")
            self.damage_solver.residual_function = self._damage_residual_active_set
            self.damage_solver.jacobian_function = self._damage_jacobian_active_set
        
        print("DEBUG: Damage callbacks assigned")
        
        # Verify callbacks are set
        print(f"DEBUG: disp_solver.residual_function is None: {self.disp_solver.residual_function is None}")
        print(f"DEBUG: damage_solver.residual_function is None: {self.damage_solver.residual_function is None}")
        
        print(f"✓ PETSc SNES solvers initialized successfully.")

    @classmethod
    def from_saved_state(cls, filename, config_override=None):
        """Create solver instance from complete saved state."""
        print(f"\n=== Creating Solver from Saved State ===")
        
        # Load complete state
        from state_manager import SimulationStateManager
        state_manager = SimulationStateManager()
        state = state_manager.load_complete_state(filename)
        
        if state is None:
            print("❌ Failed to load state. Cannot create solver.")
            return None
        
        # Create config from saved state
        from config import SimulationConfig
        config = SimulationConfig()
        
        # Update config with saved values
        config.E_FILM = state['E_FILM']
        config.E_SUBSTRATE = state['E_SUBSTRATE']
        config.NU_FILM = state['NU_FILM']
        config.NU_SUBSTRATE = state['NU_SUBSTRATE']
        config.LENGTH_SCALE = state['LENGTH_SCALE']
        config.COUPLING_PARAMETER = state['COUPLING_PARAMETER']
        config.SUBSTRATE_STIFFNESS = state['SUBSTRATE_STIFFNESS']
        config.MAX_NEWTON_ITER = state['MAX_NEWTON_ITER']
        config.NEWTON_RTOL = state['NEWTON_RTOL']
        config.NEWTON_ATOL = state['NEWTON_ATOL']
        config.MAX_ALT_ITER = state['MAX_ALT_ITER']
        config.ALT_TOL = state['ALT_TOL']
        config.N_STEPS = state['N_STEPS']
        config.MAX_DISPLACEMENT = state['MAX_DISPLACEMENT']
        config.PLOT_FREQUENCY = state['PLOT_FREQUENCY']
        
        # Apply overrides if provided
        if config_override:
            for key, value in config_override.items():
                if hasattr(config, key):
                    old_value = getattr(config, key)
                    setattr(config, key, value)
                    print(f"      🔄 Override: {key} = {old_value} → {value}")
                else:
                    print(f"      ⚠️ Unknown config parameter: {key}")
        
        # Create solver instance (using dummy mesh parameters - will be replaced)
        solver = cls.__new__(cls)  # Create without calling __init__
        
        # Set basic attributes
        solver.config = config
        solver.model_type = state['model_type']
        solver.radius = state['radius']
        solver.mesh_resolution = state['mesh_resolution']
        solver.use_vi_solver = state['use_vi_solver']
        solver.load_step = state['load_step']
        
        # Set mesh data from saved state
        solver.x = state['x']
        solver.y = state['y']
        solver.connectivity = state['connectivity']
        solver.boundary_nodes = state['boundary_nodes']
        solver.n_nodes = state['n_nodes']
        solver.n_elem = state['n_elements']
        
        # Initialize materials and damage model
        solver._setup_materials()
        
        # Initialize boundary conditions with loaded mesh
        mesh_coords = np.column_stack((solver.x, solver.y))
        solver.bc_displacement = RadialBoundaryConditions(mesh_coords, solver.boundary_nodes)
        solver.bc_displacement.set_applied_displacement(state['applied_displacement'])
        
        from boundary_conditions import DamageBoundaryConditions
        solver.bc_damage = DamageBoundaryConditions(state['irreversibility_threshold'])
        
        # Initialize visualization
        solver.visualizer = SimulationVisualizer(solver.x, solver.y, solver.connectivity)
        solver.state_manager = SimulationStateManager()
        
        # Initialize VTK visualizer if available
        if VTK_AVAILABLE:
            try:
                solver.vtk_visualizer = VTKVisualizer(solver.x, solver.y, solver.connectivity)
                print("✓ VTK visualizer initialized")
            except Exception as e:
                print(f"⚠ VTK visualizer failed: {e}")
                solver.vtk_visualizer = None
        else:
            solver.vtk_visualizer = None
        
        # Initialize PETSc solvers
        solver._init_solvers()
        
        # Load solution fields
        solver.u = state['u'].copy()
        solver.v = state['v'].copy()
        solver.d = state['d'].copy()
        solver.d_old = state['d_old'].copy()
        
        print(f"✅ Solver created from saved state:")
        print(f"   Model: {state['model_type']}, Step: {state['load_step']}")
        print(f"   Mesh: {state['n_nodes']} nodes, {state['n_elements']} elements")
        print(f"   Resolution: {state['mesh_resolution']:.4f}")
        print(f"   Max damage: {np.max(state['d']):.4f}")
        
        return solver
    
    def solve_load_step(self, applied_displacement):
        """Solve one load step using alternating minimization."""
        self.load_step += 1
        self.bc_displacement.set_applied_displacement(applied_displacement)
        
        print(f"\n--- Load Step {self.load_step}: Applied Radial Displacement = {applied_displacement:.6f} ---")
        
        u_prev, v_prev, d_prev = self.u.copy(), self.v.copy(), self.d.copy()
        
        for alt_iter in range(1, self.config.MAX_ALT_ITER + 1):
            # Solve displacement
            success_disp, disp_iter = self._solve_displacement_step()
            if not success_disp:
                print("  Displacement solve FAILED")
                return False
            
            # Solve damage
            success_damage, damage_iter = self._solve_damage_step()
            if not success_damage:
                print("  Damage solve FAILED")
                return False
            
            # Check convergence
            u_change = np.linalg.norm(self.u - u_prev) / (np.linalg.norm(u_prev) + 1e-12)
            v_change = np.linalg.norm(self.v - v_prev) / (np.linalg.norm(v_prev) + 1e-12)
            d_change = np.linalg.norm(self.d - d_prev) / (np.linalg.norm(d_prev) + 1e-12)
            total_change = u_change + v_change + d_change
            constraint_viol = self.bc_damage.constraint_violation(self.d, self.d_old)
            
            print(f"  Alt iter {alt_iter}: Disp iters={disp_iter}, Dmg iters={damage_iter}, "
                  f"Change={total_change:.2e}, Constraint viol={constraint_viol:.2e}")
            
            if total_change < self.config.ALT_TOL and constraint_viol < 1e-10:
                print(f"  Converged in {alt_iter} alternating iterations.")
                break
            
            u_prev, v_prev, d_prev = self.u.copy(), self.v.copy(), self.d.copy()
        else:
            print(f"  Warning: Max alternating iterations ({self.config.MAX_ALT_ITER}) reached.")
        
        self.d_old = self.d.copy()
        return True
    
    def _solve_displacement_step(self):
        """Solve displacement subproblem."""
        initial_guess = np.concatenate([self.u.flatten(), self.v.flatten()])
        success, solution, iterations = self.disp_solver.solve(initial_guess)
        
        if success:
            n = self.n_nodes
            self.u = solution[:2*n].reshape((n, 2)).copy()
            self.v = solution[2*n:].reshape((n, 2)).copy()
        
        return success, iterations
    
    def _solve_damage_step(self):
        """Solve damage subproblem."""
        lower_bounds = self.bc_damage.get_effective_lower_bounds(self.d_old)
        upper_bounds = self.bc_damage.get_effective_upper_bounds(self.d)
        
        success, solution, iterations = self.damage_solver.solve(
            self.d, lower_bounds, upper_bounds
        )
        
        if success:
            if not self.use_vi_solver:
                solution = self.bc_damage.project_damage_bounds(solution, self.d_old)
            self.d = solution
        else:
            # Fix: Use current self.d as fallback base, not the returned solution
            print(f" Damage solver failed, trying fallback projection")
            
            # Try projecting the current state (what solver was working from)
            d_fallback = self.d.copy()  # Start from current damage field
            d_fallback = self.bc_damage.project_damage_bounds(d_fallback, self.d_old)
            
            constraint_viol = self.bc_damage.constraint_violation(d_fallback, self.d_old)
            if constraint_viol < 1e-6:
                print(f" Using fallback projection with acceptable constraint violation")
                self.d = d_fallback
                success = True
            else:
                # If that fails, try the solver's partial solution (if it exists)
                if solution is not None:
                    solution = self.bc_damage.project_damage_bounds(solution, self.d_old)
                    constraint_viol = self.bc_damage.constraint_violation(solution, self.d_old)
                    if constraint_viol < 1e-6:
                        print(f" Using solver's partial solution with projection")
                        self.d = solution
                        success = True
        
        return success, iterations    
    
    def compute_damage_residual(self, u, d, collect_gauss_points=False):
        """Compute the damage residual with optional Gauss point collection."""
        R_d = np.zeros(self.n_nodes)
        
        # For Gauss point plotting
        if collect_gauss_points:
            gp_coords = []
            gp_residuals = []
        
        element = TriangularElement()
        xi_gp, eta_gp, w_gp = element.gauss_points()
        
        for e in range(self.n_elem):
            nodes = self.connectivity[e]
            u_e, d_e = u[nodes], d[nodes]
            coords = np.column_stack((self.x[nodes], self.y[nodes]))
            R_ed = np.zeros(3)
            
            for gp in range(len(xi_gp)):
                N, _, _ = element.shape_functions(xi_gp[gp], eta_gp[gp])
                detJ, dN_dx, dN_dy = element.compute_jacobian(coords, xi_gp[gp], eta_gp[gp])
                
                # Strain-displacement matrix
                B = element.strain_displacement_matrix(dN_dx, dN_dy)
                
                # Film strain and positive elastic energy
                strain = self.energy_calc.compute_strain_from_displacement(u_e, B)
                elastic_energy_density = self.energy_calc.compute_positive_energy(
                    strain, self.film_material.D
                )
                psi_pos = elastic_energy_density
                
                # Damage field and gradient
                d_gp = np.dot(N, d_e)
                d_grad = np.array([np.dot(dN_dx, d_e), np.dot(dN_dy, d_e)])
                
                # Damage residual terms
                w_prime = self.damage_model.damage_potential_derivative(d_gp)
                
                # Regularization term
                regularization = w_prime * N
                
                # Gradient term
                gradient_term = self.l**2 * (d_grad[0] * dN_dx + d_grad[1] * dN_dy)
                
                # Driving force term
                driving_term = -self.damage_model.compute_driving_force(psi_pos, d_gp) * N
                
                # For Gauss point collection
                if collect_gauss_points:
                    # Global coordinates of Gauss point
                    x_gp = np.dot(N, self.x[nodes])
                    y_gp = np.dot(N, self.y[nodes])
                    gp_coords.append([x_gp, y_gp])
                    
                    # Total residual at this Gauss point (scalar value)
                    total_residual_gp = w_prime - 2.0 * (1 - d_gp) * psi_pos
                    gp_residuals.append(total_residual_gp)
                
                R_ed += (regularization + gradient_term + driving_term) * detJ * w_gp[gp]
            
            R_d[nodes] += R_ed
        
        if collect_gauss_points:
            return R_d, np.array(gp_coords), np.array(gp_residuals)
        else:
            return R_d
    
    def _displacement_residual(self, snes, x, f):
        """PETSc callback for displacement residual with film-substrate coupling."""
        uv = x.getArray(readonly=True)
        n = self.n_nodes
        u = uv[:2*n].reshape((n, 2))      # Film displacement
        v = uv[2*n:].reshape((n, 2))      # Substrate displacement
        
        R_u = np.zeros((n, 2))  # Film residual
        R_v = np.zeros((n, 2))  # Substrate residual
        
        element = TriangularElement()
        xi_gp, eta_gp, w_gp = element.gauss_points()
        
        for e in range(self.n_elem):
            nodes = self.connectivity[e]
            u_e, v_e, d_e = u[nodes], v[nodes], self.d[nodes]
            coords = np.column_stack((self.x[nodes], self.y[nodes]))
            R_eu, R_ev = np.zeros((3, 2)), np.zeros((3, 2))
            
            for gp in range(len(xi_gp)):
                N, _, _ = element.shape_functions(xi_gp[gp], eta_gp[gp])
                detJ, dN_dx, dN_dy = element.compute_jacobian(coords, xi_gp[gp], eta_gp[gp])
                
                # Interpolate fields at Gauss point
                d_gp = np.dot(N, d_e)
                u_gp = np.dot(N, u_e)  # Shape: (2,)
                v_gp = np.dot(N, v_e)  # Shape: (2,)
                
                # Strain-displacement matrix
                B = element.strain_displacement_matrix(dN_dx, dN_dy)
                
                # Strains
                strain_u = self.energy_calc.compute_strain_from_displacement(u_e, B)
                strain_v = self.energy_calc.compute_strain_from_displacement(v_e, B)
                
                # Stresses
                stress_u = self.film_material.compute_stress(strain_u, d_gp)
                stress_v = self.substrate_material.compute_stress(strain_v)
                
                # Internal forces
                f_int_u = np.dot(B.T, stress_u) * detJ * w_gp[gp]
                f_int_v = np.dot(B.T, stress_v) * detJ * w_gp[gp]
                
                # Coupling forces (film-substrate interaction)
                coupling_factor = 1.0 / self.Lambda**2
                f_coupling = np.zeros(6)
                for i in range(3):
                    # Coupling force = (u - v) / Lambda^2
                    f_coupling[2*i] = coupling_factor * (u_gp[0] - v_gp[0]) * N[i] * detJ * w_gp[gp]
                    f_coupling[2*i+1] = coupling_factor * (u_gp[1] - v_gp[1]) * N[i] * detJ * w_gp[gp]
                
                # Assemble element residuals
                R_eu += (f_int_u + f_coupling).reshape(3, 2)
                R_ev += (f_int_v - f_coupling).reshape(3, 2)
            
            R_u[nodes] += R_eu
            R_v[nodes] += R_ev
        
        # Apply boundary conditions
        self.bc_displacement.apply_displacement_bcs(u, v, R_u, R_v)
        
        f.setArray(np.concatenate([R_u.flatten(), R_v.flatten()]))
    
    def _displacement_jacobian(self, snes, x, J, P):
        """PETSc callback for displacement Jacobian with film-substrate coupling."""
        J.zeroEntries()
        n = self.n_nodes
        
        element = TriangularElement()
        xi_gp, eta_gp, w_gp = element.gauss_points()
        
        for e in range(self.n_elem):
            nodes = self.connectivity[e]
            d_e = self.d[nodes]
            coords = np.column_stack((self.x[nodes], self.y[nodes]))
            
            K_uu = np.zeros((6, 6))  # Film-film coupling
            K_uv = np.zeros((6, 6))  # Film-substrate coupling
            K_vu = np.zeros((6, 6))  # Substrate-film coupling
            K_vv = np.zeros((6, 6))  # Substrate-substrate coupling
            
            for gp in range(len(xi_gp)):
                N, _, _ = element.shape_functions(xi_gp[gp], eta_gp[gp])
                detJ, dN_dx, dN_dy = element.compute_jacobian(coords, xi_gp[gp], eta_gp[gp])
                
                d_gp = np.dot(N, d_e)
                g_d = self.film_material.degradation_function(d_gp)
                
                # Strain-displacement matrix
                B = element.strain_displacement_matrix(dN_dx, dN_dy)
                
                # Stiffness contributions
                K_film = g_d * np.dot(B.T, np.dot(self.film_material.D, B)) * detJ * w_gp[gp]
                K_substrate = np.dot(B.T, np.dot(self.substrate_material.D, B)) * detJ * w_gp[gp]
                
                # Coupling stiffness
                coupling_factor = 1.0 / self.Lambda**2
                K_coupling = np.zeros((6, 6))
                for i in range(3):
                    for j in range(3):
                        c = coupling_factor * N[i] * N[j] * detJ * w_gp[gp]
                        K_coupling[2*i, 2*j] = c
                        K_coupling[2*i+1, 2*j+1] = c
                
                # Assemble element stiffness matrices
                K_uu += K_film + K_coupling
                K_uv -= K_coupling
                K_vu -= K_coupling
                K_vv += K_substrate + K_coupling
            
            # Global assembly
            dof_u = np.array([2*node+i for node in nodes for i in range(2)], dtype=PETSc.IntType)
            dof_v = np.array([2*n+2*node+i for node in nodes for i in range(2)], dtype=PETSc.IntType)
            
            J.setValues(dof_u, dof_u, K_uu, addv=True)
            J.setValues(dof_u, dof_v, K_uv, addv=True)
            J.setValues(dof_v, dof_u, K_vu, addv=True)
            J.setValues(dof_v, dof_v, K_vv, addv=True)
        
        J.assemblyBegin()
        J.assemblyEnd()
        
        # Apply boundary conditions
        self.bc_displacement.apply_displacement_jacobian_bcs(J, n)
        
        if J != P:
            P.assemblyBegin()
            P.assemblyEnd()
    
    def _damage_residual_vi(self, snes, x_petsc, f_petsc):
        """PETSc callback for damage residual using VI solver WITH active-set logic."""
        d = x_petsc.getArray(readonly=True)
        
        # Compute physical residual
        R_d_physical = self.compute_damage_residual(self.u, d)
        
        # Apply active-set logic
        R_d_solver, constraint_active = self.bc_damage.compute_active_set_residual(
            R_d_physical, d, self.d_old
        )
        
        # Debug output
        n_active = np.sum(constraint_active)
        # if n_active > 0:
        #     print(f"      VI + Active-set: {n_active} constraints active")
        
        f_petsc.setArray(R_d_solver)
    
    def _damage_jacobian_vi(self, snes, x_petsc, J_petsc, P_petsc):
        """PETSc callback for damage Jacobian using VI solver WITH active-set logic."""
        d = x_petsc.getArray(readonly=True)
        J_petsc.zeroEntries()
        
        # Assemble physical Jacobian
        self._assemble_physical_jacobian(d, J_petsc)
        
        # Apply active-set constraints to Jacobian
        R_d_physical = self.compute_damage_residual(self.u, d)
        _, constraint_active = self.bc_damage.compute_active_set_residual(
            R_d_physical, d, self.d_old
        )
        constraint_nodes = np.where(constraint_active)[0].astype(PETSc.IntType)
        if len(constraint_nodes) > 0:
            J_petsc.zeroRows(constraint_nodes, diag=1.0)
        
        J_petsc.assemblyBegin()
        J_petsc.assemblyEnd()
        
        if J_petsc != P_petsc:
            P_petsc.assemblyBegin()
            P_petsc.assemblyEnd()
    
    def _assemble_physical_jacobian(self, d, J_petsc):
        """Assemble the physical damage Jacobian matrix."""
        element = TriangularElement()
        xi_gp, eta_gp, w_gp = element.gauss_points()
        
        for e in range(self.n_elem):
            nodes = self.connectivity[e]
            u_e, d_e = self.u[nodes], d[nodes]
            coords = np.column_stack((self.x[nodes], self.y[nodes]))
            K_dd = np.zeros((3, 3))
            
            for gp in range(len(xi_gp)):
                N, _, _ = element.shape_functions(xi_gp[gp], eta_gp[gp])
                detJ, dN_dx, dN_dy = element.compute_jacobian(coords, xi_gp[gp], eta_gp[gp])
                
                # Strain calculation
                B = element.strain_displacement_matrix(dN_dx, dN_dy)
                strain = self.energy_calc.compute_strain_from_displacement(u_e, B)
                elastic_energy_density = self.energy_calc.compute_positive_energy(
                    strain, self.film_material.D
                )
                psi_pos = elastic_energy_density
                
                d_gp = np.dot(N, d_e)
                
                # Jacobian terms
                w_double_prime = self.damage_model.damage_potential_second_derivative(d_gp)
                
                # Regularization contribution
                regularization_contrib = w_double_prime * np.outer(N, N)
                
                # Gradient contribution
                gradient_contrib = self.l**2 * (np.outer(dN_dx, dN_dx) + np.outer(dN_dy, dN_dy))
                
                # Driving force contribution
                driving_contrib = 2.0 * psi_pos * np.outer(N, N)
                
                K_dd += (regularization_contrib + gradient_contrib + driving_contrib) * detJ * w_gp[gp]
            
            nodes_petsc = np.array(nodes, dtype=PETSc.IntType)
            J_petsc.setValues(nodes_petsc, nodes_petsc, K_dd, addv=True)
        
        J_petsc.assemblyBegin()
        J_petsc.assemblyEnd()
    
    def run_simulation(self, n_steps=40, max_displacement=0.02, plot_every=5, 
                      state_file="simulation_state.npz"):
        """Run a simulation with restart capability and VTK output."""
        print(f"\n=== Running 2D Radial Tension Simulation ===")
        
        start_step = self.load_step
        if start_step > 0:
            print(f"      Resuming simulation from step {start_step + 1}...")
        
        start_time = time.time()
        successful_steps = 0
        
        # Main simulation loop
        for step in range(start_step, n_steps):
            applied_disp = max_displacement * (step + 1) / n_steps
            success = self.solve_load_step(applied_disp)
            
            if not success:
                print(f"❌ Solver failed at step {step+1}! Halting simulation.")
                self.save_state(state_file)
                break

            successful_steps += 1
            max_damage = np.max(self.d)
            max_film_disp = np.max(np.sqrt(self.u[:, 0]**2 + self.u[:, 1]**2))
            max_substrate_disp = np.max(np.sqrt(self.v[:, 0]**2 + self.v[:, 1]**2))
            
            print(f"  Step {step+1} successful. Max damage: {max_damage:.4f}, "
                  f"Max film disp: {max_film_disp:.4f}, Max substrate disp: {max_substrate_disp:.4f}")
            
            # Compute and print damage residual info
            damage_residual = self.compute_damage_residual(self.u, self.d)
            residual_max = np.max(np.abs(damage_residual))
            residual_min = np.min(damage_residual)
            residual_max_pos = np.max(damage_residual)
            print(f"Damage residual: min={residual_min:.2e}, max={residual_max_pos:.2e}")
            
            # Plotting and saving state
            if ((step + 1) % plot_every == 0 or (step + 1) == n_steps or 
                (max_damage > 0.001 and np.max(self.d_old) < 0.001)):
                
                # Existing matplotlib plotting
                self.plot_results_2d(step_number=step + 1, save_png=True)
                
                # VTK output for ParaView visualization
                if self.vtk_visualizer is not None:
                    try:
                        self.vtk_visualizer.save_vtk_file(
                            self.u, self.v, self.d, self.film_material, 
                            step_number=step + 1
                        )
                    except Exception as e:
                        print(f"    ⚠ VTK output failed: {e}")
                
                self.save_state(state_file)
            
            # Check for high damage
            if max_damage > 0.95:
                print("\n⚠ High damage detected. Stopping simulation.")
                self.plot_results_2d(step_number=step + 1, save_png=True)
                
                # Final VTK output
                if self.vtk_visualizer is not None:
                    try:
                        self.vtk_visualizer.save_vtk_file(
                            self.u, self.v, self.d, self.film_material, 
                            step_number=step + 1
                        )
                    except Exception as e:
                        print(f"    ⚠ Final VTK output failed: {e}")
                
                self.save_state(state_file)
                #break
        
        end_time = time.time()
        print(f"\n=== 2D Simulation Finished ===")
        print(f"✓ Completed {successful_steps}/{n_steps} steps in {end_time-start_time:.2f}s")
        
        # VTK output summary
        if self.vtk_visualizer is not None:
            print(f"✓ VTK files saved in 'vtk_output/' directory")
            print(f"  → Open 'vtk_output/phase_field_series.pvd' in ParaView for visualization")
        
        return successful_steps > 0
    
    def plot_results_2d(self, step_number=None, save_png=False):
        """Plot the current solution fields including damage residual."""
        damage_residual = self.compute_damage_residual(self.u, self.d)
        self.visualizer.plot_results_2d(
            self.u, self.v, self.d, damage_residual, step_number, 
            self.model_type, save_png)
    
    def plot_mesh(self, save_png=True):
        """Plot the mesh with boundary nodes highlighted."""
        self.visualizer.plot_mesh(self.boundary_nodes, self.radius, save_png)
    
    def save_state(self, filename="simulation_state.npz"):
        """Save complete simulation state including mesh and all parameters."""
        return self.state_manager.save_complete_state(filename, self)
    
    
    def cleanup(self):
        """Clean up PETSc resources."""
        if PETSC_AVAILABLE:
            print("\nCleaning up PETSc objects...")
            if hasattr(self, 'damage_solver'):
                self.damage_solver.cleanup()
            print("✓ PETSc cleanup complete.")
        if len(constraint_nodes) > 0:
            J_petsc.zeroRows(constraint_nodes, diag=1.0)
        
        J_petsc.assemblyBegin()
        J_petsc.assemblyEnd()
        
        if J_petsc != P_petsc:
            P_petsc.assemblyBegin()
            P_petsc.assemblyEnd()
    
    def _damage_residual_active_set(self, snes, x_petsc, f_petsc):
        """PETSc callback for damage residual using active-set method."""
        d = x_petsc.getArray(readonly=True)
        
        # Compute physical residual
        R_d_physical = self.compute_damage_residual(self.u, d)
        
        # Apply active-set logic
        R_d_solver, constraint_active = self.bc_damage.compute_active_set_residual(
            R_d_physical, d, self.d_old
        )
        
        # Debug output
        n_active = np.sum(constraint_active)
        if n_active > 0:
            print(f"      Active-set: {n_active} constraints active")
        
        f_petsc.setArray(R_d_solver)
    
    def _damage_jacobian_active_set(self, snes, x_petsc, J_petsc, P_petsc):
        """PETSc callback for damage Jacobian using the manual active-set method."""
        d = x_petsc.getArray(readonly=True)
        J_petsc.zeroEntries()
        
        # Assemble physical Jacobian
        self._assemble_physical_jacobian(d, J_petsc)
        
        # Apply active-set constraints to Jacobian
        R_d_physical = self.compute_damage_residual(self.u, d)
        _, constraint_active = self.bc_damage.compute_active_set_residual(
            R_d_physical, d, self.d_old
        )
        constraint_nodes = np.where(constraint_active)[0].astype(PETSc.IntType)
        if len(constraint_nodes) > 0:
            J_petsc.zeroRows(constraint_nodes, diag=1.0)
        
        J_petsc.assemblyBegin()
        J_petsc.assemblyEnd()
        
        if J_petsc != P_petsc:
            P_petsc.assemblyBegin()
            P_petsc.assemblyEnd()