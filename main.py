"""
Updated main.py that uses the new configuration system and improved state management.
Supports all new config parameters and proper restart handling.
"""

import os
from config import SimulationConfig, PETScConfig, PETSC_AVAILABLE
from solver import AT1_2D_ActiveSet_Solver
from state_manager import SimulationStateManager

def print_simulation_info(config, model_type):
    """Print simulation configuration information."""
    print("🚀 2D Phase-Field Fracture: Film-Substrate System with Radial Loading")
    print("=" * 80)
    print(f"📋 Configuration Summary:")
    print(f"   Model Type: {model_type}")
    print(f"   Mesh Resolution: {config.MESH_RESOLUTION:.4f}")
    print(f"   Radius: {config.RADIUS}")
    print(f"   Steps: {config.N_STEPS}")
    print(f"   Max Displacement: {config.MAX_DISPLACEMENT}")
    print(f"   Length Scale: {config.LENGTH_SCALE}")
    print(f"   Material Film E: {config.E_FILM}, Substrate E: {config.E_SUBSTRATE}")
    print(f"   New Parameters: HF={config.HF}, HS={config.HS}, CHI={config.CHI}")
    print(f"   Phase-field: GC={config.GC}, AT1={config.AT1}, AT2={config.AT2}")
    print(f"   Critical energies: PSI_C_AT1={config.PSI_C_AT1}, PSI_C_AT2={config.PSI_C_AT2}")
    print(f"   VI Solver: {config.USE_VI_SOLVER}")
    print(f"   PETSc Available: {PETSC_AVAILABLE}")
    print(f"   State File: {config.STATE_FILE}")
    print("=" * 80)

def validate_config(config):
    """Validate configuration parameters."""
    issues = []
    
    # Check mesh resolution vs length scale
    if config.MESH_RESOLUTION > config.LENGTH_SCALE / 2:
        issues.append(f"Mesh resolution ({config.MESH_RESOLUTION}) is too coarse for length scale ({config.LENGTH_SCALE})")
        issues.append(f"Recommended: mesh_resolution ≤ {config.LENGTH_SCALE/3:.4f}")
    
    # Check material parameters
    if config.E_FILM <= 0 or config.E_SUBSTRATE <= 0:
        issues.append("Young's moduli must be positive")
    
    if not (0 <= config.NU_FILM < 0.5) or not (0 <= config.NU_SUBSTRATE < 0.5):
        issues.append("Poisson's ratios must be in [0, 0.5)")
    
    # Check new parameters
    if config.HF <= 0 or config.HS <= 0:
        issues.append("Film and substrate thicknesses (HF, HS) must be positive")
    
    if config.CHI <= 0:
        issues.append("CHI parameter must be positive")
    
    if config.GC <= 0:
        issues.append("Critical energy release rate (GC) must be positive")
    
    # Check phase-field parameters
    if config.LENGTH_SCALE <= 0:
        issues.append("Length scale must be positive")
    
    # if not (0 < config.COUPLING_PARAMETER <= 1):
    #     issues.append("Coupling parameter should be in (0, 1]")
    
    if issues:
        print("⚠️  Configuration Issues Detected:")
        for issue in issues:
            print(f"   • {issue}")
        print()
        
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            return False
    
    return True

def setup_simulation_config():
    """Setup and customize simulation configuration."""
    config = SimulationConfig()
    
    # Simulation mode selection
    print("\n🎛️  Simulation Configuration:")
    print("1. Quick test (low resolution, few steps)")
    print("2. Standard simulation (parameters in config.py)")
    print("3. High resolution (fine mesh, many steps)")
    print("4. Custom configuration")
    
    mode = input("Select mode (1-4) [2]: ").strip() or "2"
    
    if mode == "1":
        # Quick test
        config.MESH_RESOLUTION = config.LENGTH_SCALE / 2
        config.N_STEPS = 20
        config.MAX_DISPLACEMENT = 1.0
        config.PLOT_FREQUENCY = 5
        print("   📦 Quick test mode selected")
        
    elif mode == "3":
        # High resolution
        config.MESH_RESOLUTION = config.LENGTH_SCALE / 5
        config.N_STEPS = 200
        config.MAX_DISPLACEMENT = 5.0
        config.PLOT_FREQUENCY = 10
        print("   🔬 High resolution mode selected")
        
    elif mode == "4":
        # Custom configuration
        print("\n   Custom Configuration:")
        try:
            config.MESH_RESOLUTION = float(input(f"   Mesh resolution [{config.MESH_RESOLUTION}]: ") or config.MESH_RESOLUTION)
            config.N_STEPS = int(input(f"   Number of steps [{config.N_STEPS}]: ") or config.N_STEPS)
            config.MAX_DISPLACEMENT = float(input(f"   Max displacement [{config.MAX_DISPLACEMENT}]: ") or config.MAX_DISPLACEMENT)
            config.PLOT_FREQUENCY = int(input(f"   Plot frequency [{config.PLOT_FREQUENCY}]: ") or config.PLOT_FREQUENCY)
            config.LENGTH_SCALE = float(input(f"   Length scale [{config.LENGTH_SCALE}]: ") or config.LENGTH_SCALE)
        except ValueError:
            print("   ⚠️  Invalid input, using defaults")
        print("   ⚙️  Custom configuration applied")
    else:
        # Standard mode (default)
        print("   📋 Standard mode selected")
    
    # Additional options
    config.USE_VI_SOLVER = input(f"Use VI solver? (Y/n) [{config.USE_VI_SOLVER}]: ").strip().lower() not in ['n', 'no']
    config.IRREVERSIBILITY_THRESHOLD = 0.0  # Keep strict irreversibility
    
    return config

def main():
    """Main function with comprehensive restart handling and new config support."""
    
    # Model selection
    MODEL_TO_TEST = 'AT1'  # Options: 'AT1', 'AT2'
    
    # Setup configuration
    config = setup_simulation_config()
    
    # Print configuration info
    print_simulation_info(config, MODEL_TO_TEST)
    
    # Validate configuration
    if not validate_config(config):
        print("❌ Simulation aborted due to configuration issues.")
        return
    
    # Initialize state manager
    state_manager = SimulationStateManager()
    
    # Check for existing state files
    if not config.RESTART_SIMULATION:
        available_states = state_manager.list_available_states()
        if available_states:
            print(f"\n📁 Found {len(available_states)} existing state file(s)")
            restart_choice = input("Restart from existing state? (y/N): ").strip().lower()
            if restart_choice == 'y':
                config.RESTART_SIMULATION = True
                if len(available_states) > 1:
                    try:
                        choice = int(input(f"Select file (1-{len(available_states)}): ")) - 1
                        config.STATE_FILE = available_states[choice]
                    except (ValueError, IndexError):
                        print("Invalid selection, using first file")
                        config.STATE_FILE = available_states[0]
                else:
                    config.STATE_FILE = available_states[0]
                print(f"   Selected: {config.STATE_FILE}")
    
    solver = None
    success = False
    
    try:
        # RESTART HANDLING with new state manager
        if config.RESTART_SIMULATION and os.path.exists(config.STATE_FILE):
            print(f"\n🔄 RESTARTING from {config.STATE_FILE}")
            
            # Load complete state using new state manager
            state_data = state_manager.load_complete_state(config.STATE_FILE)
            
            if state_data is not None:
                print("   🔧 Reconstructing solver from complete state...")
                
                # Create config from saved state (preserves exact simulation parameters)
                saved_config = state_manager.create_config_from_state(state_data)
                
                # Allow override of certain parameters for continuation
                override_params = {
                    'N_STEPS': config.N_STEPS,
                    'MAX_DISPLACEMENT': config.MAX_DISPLACEMENT,
                    'PLOT_FREQUENCY': config.PLOT_FREQUENCY,
                    'USE_VI_SOLVER': config.USE_VI_SOLVER
                }
                
                # Apply overrides
                for param, value in override_params.items():
                    setattr(saved_config, param, value)
                    print(f"   📝 Override: {param} = {value}")
                
                # Use the from_saved_state class method
                solver = AT1_2D_ActiveSet_Solver.from_saved_state(
                    config.STATE_FILE,
                    config_override=override_params
                )
                
                if solver is None:
                    raise Exception("Failed to create solver from complete state")
                
                print(f"   ✅ Solver successfully restored from step {state_data['load_step']}")
                
            else:
                raise Exception("Failed to load state data")
                
        else:
            print(f"\n🆕 CREATING new solver")
            # Create new solver with current config
            solver = AT1_2D_ActiveSet_Solver(
                radius=config.RADIUS,
                mesh_resolution=config.MESH_RESOLUTION,
                model_type=MODEL_TO_TEST,
                irreversibility_threshold=config.IRREVERSIBILITY_THRESHOLD,
                use_vi_solver=config.USE_VI_SOLVER,
                config=config
            )
            
            # Show mesh for new solver
            solver.plot_mesh()
        
        # RUN SIMULATION
        print(f"\n🎯 Starting SIMULATION...")
        print(f"🔧 Running full simulation with solving")
        
        success = solver.run_simulation(
            n_steps=config.N_STEPS,
            max_displacement=config.MAX_DISPLACEMENT,
            plot_every=config.PLOT_FREQUENCY,
            state_file=config.STATE_FILE
        )
        
        # RESULTS
        if success:
            print(f"\n🎉 2D {MODEL_TO_TEST} SIMULATION: SUCCESS!")
            print(f"   💾 Final state saved to: {config.STATE_FILE}")
            print(f"   📈 Simulation data and plots generated")
        else:
            print(f"\n⚠️  2D {MODEL_TO_TEST} SIMULATION: COMPLETED WITH ISSUES")
            
    except Exception as e:
        print(f"\n💥 Error during simulation: {e}")
        
        # FALLBACK: Try creating new solver if restart failed
        if "restart" in str(e).lower() or "state" in str(e).lower():
            print("\n🔄 Restart failed, attempting to create new solver...")
            try:
                # Backup problematic state file
                if os.path.exists(config.STATE_FILE):
                    backup_name = config.STATE_FILE.replace('.npz', '_corrupted_backup.npz')
                    os.rename(config.STATE_FILE, backup_name)
                    print(f"   📋 Backed up problematic state to: {backup_name}")
                
                # Create fresh solver
                solver = AT1_2D_ActiveSet_Solver(
                    radius=config.RADIUS,
                    mesh_resolution=config.MESH_RESOLUTION,
                    model_type=MODEL_TO_TEST,
                    irreversibility_threshold=config.IRREVERSIBILITY_THRESHOLD,
                    use_vi_solver=config.USE_VI_SOLVER,
                    config=config
                )
                
                solver.plot_mesh()
                
                # Run simulation with fresh solver
                success = solver.run_simulation(
                    n_steps=config.N_STEPS,
                    max_displacement=config.MAX_DISPLACEMENT,
                    plot_every=config.PLOT_FREQUENCY,
                    state_file=config.STATE_FILE
                )
                
                if success:
                    print(f"\n🎉 SUCCESS after restart failure recovery!")
                    
            except Exception as e2:
                print(f"💥 Failed to create new solver: {e2}")
                import traceback
                traceback.print_exc()
        else:
            import traceback
            traceback.print_exc()
            
    finally:
        # CLEANUP
        if solver:
            solver.cleanup()
            print("🧹 Solver cleanup completed")
    
    print("\n" + "=" * 80)
    print(f"🏁 2D {MODEL_TO_TEST} simulation session completed.")
    
    if success:
        print("✅ All operations completed successfully!")
    else:
        print("⚠️  Session completed with issues - check output above")

if __name__ == "__main__":
    main()