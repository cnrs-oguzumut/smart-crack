"""
Fixed main.py that uses the proper restart method to avoid mesh mismatch.
"""

import os
from config import SimulationConfig
from solver import AT1_2D_ActiveSet_Solver

def main():
    """Main function with proper restart handling."""
    print("🚀 2D Phase-Field Fracture: Film-Substrate System with Radial Loading")
    print("=" * 80)
    
    # Create configuration
    config = SimulationConfig()
    
    # Override default parameters if needed
    config.MESH_RESOLUTION = 0.06  # This will be ignored if restarting
    config.N_STEPS = 46
    config.MAX_DISPLACEMENT = 1.3
    config.PLOT_FREQUENCY = 5
    config.IRREVERSIBILITY_THRESHOLD = 0.0
    config.USE_VI_SOLVER = True
    config.RESTART_SIMULATION = True  # Set this for restart
    
    MODEL_TO_TEST = 'AT1'
    
    solver = None
    try:
        # PROPER RESTART CHECK
        if config.RESTART_SIMULATION and os.path.exists(config.STATE_FILE):
            print(f"🔄 RESTARTING from {config.STATE_FILE}")
            
            # Use the PROPER restart method that loads the exact saved mesh
            solver = AT1_2D_ActiveSet_Solver.from_saved_state(
                config.STATE_FILE,
                config_override={
                    'N_STEPS': config.N_STEPS,           # Continue with more steps
                    'MAX_DISPLACEMENT': config.MAX_DISPLACEMENT,  # Can change target
                    'PLOT_FREQUENCY': config.PLOT_FREQUENCY      # Can change frequency
                }
            )
            
            if solver is None:
                print("❌ Restart failed, creating new solver...")
                # Delete the problematic state file and start fresh
                try:
                    os.rename(config.STATE_FILE, config.STATE_FILE + ".backup")
                    print(f"   Backed up old state file to {config.STATE_FILE}.backup")
                except:
                    pass
                raise Exception("Restart failed - will create new solver")
                
        else:
            print(f"🆕 CREATING new solver")
            # Create new solver normally
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
        
        # Run simulation (continues from saved step if restarted)
        success = solver.run_simulation(
            n_steps=config.N_STEPS,
            max_displacement=config.MAX_DISPLACEMENT,
            plot_every=config.PLOT_FREQUENCY,
            state_file=config.STATE_FILE
        )
        
        if success:
            print(f"\n🎉 2D {MODEL_TO_TEST} Simulation: SUCCESS!")
        else:
            print(f"\n❌ 2D {MODEL_TO_TEST} Simulation: FAILED OR HALTED.")
            
    except Exception as e:
        print(f"\n💥 Unhandled error during simulation: {e}")
        
        # If restart failed, try creating new solver
        if "Restart failed" in str(e):
            print("\n🔄 Attempting to create new solver after restart failure...")
            try:
                solver = AT1_2D_ActiveSet_Solver(
                    radius=config.RADIUS,
                    mesh_resolution=config.MESH_RESOLUTION,
                    model_type=MODEL_TO_TEST,
                    irreversibility_threshold=config.IRREVERSIBILITY_THRESHOLD,
                    use_vi_solver=config.USE_VI_SOLVER,
                    config=config
                )
                solver.plot_mesh()
                
                success = solver.run_simulation(
                    n_steps=config.N_STEPS,
                    max_displacement=config.MAX_DISPLACEMENT,
                    plot_every=config.PLOT_FREQUENCY,
                    state_file=config.STATE_FILE
                )
                
                if success:
                    print(f"\n🎉 2D {MODEL_TO_TEST} Simulation: SUCCESS (after restart failure)!")
            except Exception as e2:
                print(f"\n💥 Failed to create new solver: {e2}")
                import traceback
                traceback.print_exc()
        else:
            import traceback
            traceback.print_exc()
    finally:
        if solver:
            solver.cleanup()
        
    print("\n" + "=" * 80)
    print("2D simulation completed.")

if __name__ == "__main__":
    main()