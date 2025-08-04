2D Phase-Field Fracture Simulation

A modular Python implementation of 2D phase-field fracture mechanics for film-substrate systems under radial loading. This code simulates crack propagation using AT1/AT2 models with PETSc-based solvers.
Project Structure

smart-crack/
├── LICENSE                    # ✅ Keep this (already exists)
├── README.md                  # ✅ Update this with project description
├── requirements.txt           # ✅ Add this (Python dependencies)
├── main.py                    # ✅ Your main script
├── config.py                  # ✅ Configuration
├── solver.py                  # ✅ Main solver class
├── boundary_conditions.py     # ✅ Boundary conditions
├── material_models.py         # ✅ Material and damage models
├── mesh_generator.py          # ✅ Mesh generation
├── finite_element.py          # ✅ FE utilities
├── petsc_solver.py            # ✅ PETSc interface
├── state_manager.py           # ✅ State saving/loading
├── visualization.py           # ✅ Matplotlib plots
├── vtk_visualizer.py          # ✅ VTK/ParaView output
├── docs/                      # 📁 Documentation folder
│   ├── installation.md
│   ├── usage.md
│   └── examples.md
├── examples/                  # 📁 Example scripts
│   ├── basic_simulation.py
│   └── custom_materials.py
└── tests/                     # 📁 Future tests
    └── test_solver.py

Features

    Phase-Field Models: Support for both AT1 and AT2 damage models
    Solver Options: VI (Variational Inequality) and Active-Set methods
    Film-Substrate Coupling: Handles two-layer systems with different material properties
    Radial Loading: Specialized boundary conditions for circular domains
    State Management: Save/restart simulation capability
    Visualization: Comprehensive plotting of displacement, damage, and residual fields
    Modular Design: Easy to extend and modify individual components

Dependencies

Required packages:

pip install numpy scipy matplotlib petsc4py

Note: PETSc installation may require additional system dependencies. See PETSc installation guide.
Quick Start

Basic simulation:

python main.py

Custom configuration:

from config import SimulationConfig
from solver import AT1_2D_ActiveSet_Solver

config = SimulationConfig()
config.MESH_RESOLUTION = 0.02  # Finer mesh
config.N_STEPS = 50

solver = AT1_2D_ActiveSet_Solver(
    model_type='AT2',
    use_vi_solver=True,
    config=config
)

solver.run_simulation()

Configuration

Key parameters in config.py:
Mesh Parameters

    RADIUS: Domain radius (default: 1.0)
    MESH_RESOLUTION: Element size (default: 0.03)

Material Properties

    E_FILM, E_SUBSTRATE: Young's moduli
    NU_FILM, NU_SUBSTRATE: Poisson's ratios

Phase-Field Parameters

    LENGTH_SCALE: Regularization length scale (default: 0.013)
    COUPLING_PARAMETER: Film-substrate coupling (default: 0.1)

Simulation Parameters

    N_STEPS: Number of load steps (default: 46)
    MAX_DISPLACEMENT: Maximum applied displacement (default: 1.3)
    USE_VI_SOLVER: Use VI solver vs. Active-Set (default: True)

Model Types
AT1 Model

    Linear damage potential: w(d) = d
    Suitable for brittle fracture
    Critical energy threshold: ψc = 0.5

AT2 Model

    Quadratic damage potential: w(d) = d²
    More physically motivated
    Critical energy threshold: ψc = 0.0

Solver Methods
VI (Variational Inequality) Solver

    Uses PETSc's vinewtonrsls
    Automatically enforces damage constraints
    More robust for constraint handling

Active-Set Method

    Manual constraint enforcement
    Modifies residual and Jacobian for active constraints
    Fallback option when VI solver unavailable

Output Files

The simulation generates several output files:

    PNG plots: Solution fields at specified intervals
        at1_2d_solver_step_XXX.png: Displacement, damage, and residual fields
        circular_mesh.png: Mesh visualization

    State files: For simulation restart
        simulation_state.npz: Current simulation state

Visualization

The plotting system shows:

    Film displacement magnitude |u|
    Substrate displacement magnitude |v|
    Damage field d ∈ [0,1]
    Displacement difference |u-v|
    Damage residual ∂E/∂d (driving force)
    Radial profiles of displacement and residual

Restart Capability

To restart a simulation:

config.RESTART_SIMULATION = True
config.STATE_FILE = "simulation_state.npz"

The solver will automatically load the previous state and continue from the last completed step.
Extending the Code
Adding New Material Models

Create new material class in material_models.py:

class CustomMaterial(MaterialModel):
    def compute_stress(self, strain, damage=None):
        # Custom constitutive law
        pass

Update solver to use new material:

self.custom_material = CustomMaterial(E=2.0, nu=0.25)

Custom Boundary Conditions
Extend boundary_conditions.py:

class CustomBoundaryConditions:
    def apply_custom_bcs(self, ...):
        # Custom BC implementation
        pass

New Visualization Options
Add methods to visualization.py:

def plot_custom_field(self, field_data, title):
    # Custom plotting logic
    pass

Troubleshooting
PETSc Import Errors

    Ensure PETSc is properly installed: python -c "from petsc4py import PETSc"
    Check system dependencies (MPI, BLAS, LAPACK)

Memory Issues

    Reduce mesh resolution: increase MESH_RESOLUTION value
    Decrease number of steps: reduce N_STEPS

Convergence Problems

    Adjust solver tolerances in config.py
    Try different solver types (VI vs. Active-Set)
    Reduce load step size: decrease MAX_DISPLACEMENT

Visualization Issues

    Ensure matplotlib backend supports display: matplotlib.use('Agg') for headless
    Check file permissions for PNG output

References

    Miehe, C., Welschinger, F., & Hofacker, M. (2010). Thermodynamically consistent phase‐field models of fracture. Computer Methods in Applied Mechanics and Engineering, 199(45-48), 2765-2778.

    Bourdin, B., Francfort, G. A., & Marigo, J. J. (2000). Numerical experiments in revisited brittle fracture. Journal of the Mechanics and Physics of Solids, 48(4), 797-826.

    Borden, M. J., Verhoosel, C. V., Scott, M. A., Hughes, T. J., & Landis, C. M. (2012). A phase-field description of dynamic brittle fracture. Computer Methods in Applied Mechanics and Engineering, 217, 77-95.

License

This project is released under the MIT License. See LICENSE file for details.
Contributing

Contributions are welcome! Please:

    Fork the repository
    Create a feature branch
    Add tests for new functionality
    Submit a pull request

For bugs and feature requests, please open an issue with:

    Problem description
    Minimal reproducible example
    System information (OS, Python version, PETSc version)


<img width="2149" height="1419" alt="at1_2d_solver_step_046" src="https://github.com/user-attachments/assets/e09a497b-d192-4dd4-8da6-1347824107bc" />
<img width="2149" height="1419" alt="at1_2d_solver_step_056" src="https://github.com/user-attachments/assets/7e33af23-2ab6-4d6c-aee9-0f0ad5057ed8" />

