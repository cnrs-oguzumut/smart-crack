"""
Complete state management module for simulation restart and checkpointing.
Handles saving and loading of COMPLETE simulation state including mesh.
Updated to include all new configuration parameters.
"""

import numpy as np
import os
from datetime import datetime

class SimulationStateManager:
    """Complete state manager - saves everything needed for restart."""
    
    def __init__(self):
        """Initialize state manager."""
        pass
    
    def save_complete_state(self, filename, solver):
        """
        Save COMPLETE simulation state - everything needed for restart.
        
        Parameters:
        -----------
        solver : AT1_2D_ActiveSet_Solver
            The solver object containing all simulation data
        """
        try:
            print(f"      💾 Saving complete simulation state...")
            
            # Core simulation data
            save_data = {
                # Solution fields
                'u': solver.u,
                'v': solver.v,
                'd': solver.d,
                'd_old': solver.d_old,
                'load_step': np.array([solver.load_step]),
                
                # Complete mesh data
                'mesh_x': solver.x,
                'mesh_y': solver.y,
                'mesh_connectivity': solver.connectivity,
                'mesh_boundary_nodes': solver.boundary_nodes,
                'mesh_n_nodes': np.array([solver.n_nodes]),
                'mesh_n_elements': np.array([solver.n_elem]),
                
                # Configuration parameters - UPDATED
                'config_radius': np.array([getattr(solver, 'radius', solver.config.RADIUS)]),
                'config_mesh_resolution': np.array([getattr(solver, 'mesh_resolution', solver.config.MESH_RESOLUTION)]),
                'config_model_type': solver.model_type.encode('utf-8'),  # String to bytes
                'config_use_vi_solver': np.array([getattr(solver, 'use_vi_solver', solver.config.USE_VI_SOLVER)]),
                'config_restart_simulation': np.array([solver.config.RESTART_SIMULATION]),
                'config_loading_mode': solver.config.LOADING_MODE.encode('utf-8') if hasattr(solver.config, 'LOADING_MODE') else b'radial',
                'config_biaxiality_ratio': np.array([getattr(solver.config, 'BIAXIALITY_RATIO', 1.0)]),
                
                # Boundary condition parameters
                'bc_applied_displacement': np.array([getattr(solver.bc_displacement, 'applied_displacement', 0.0)]),
                'bc_irreversibility_threshold': np.array([getattr(solver.bc_damage, 'irreversibility_threshold', solver.config.IRREVERSIBILITY_THRESHOLD)]),
                
                # Material parameters - UPDATED with new properties
                'material_E_film': np.array([solver.config.E_FILM]),
                'material_E_substrate': np.array([solver.config.E_SUBSTRATE]),
                'material_nu_film': np.array([solver.config.NU_FILM]),
                'material_nu_substrate': np.array([solver.config.NU_SUBSTRATE]),
                'material_HF': np.array([solver.config.HF]),
                'material_HS': np.array([solver.config.HS]),
                'material_CHI': np.array([solver.config.CHI]),
                'material_GC': np.array([solver.config.GC]),
                'material_AT1': np.array([solver.config.AT1]),
                'material_AT2': np.array([solver.config.AT2]),
                'material_length_scale': np.array([solver.config.LENGTH_SCALE]),
                'material_coupling_param': np.array([solver.config.COUPLING_PARAMETER]),
                'material_substrate_stiffness': np.array([solver.config.SUBSTRATE_STIFFNESS]),
                'material_PSI_C_AT1': np.array([solver.config.PSI_C_AT1]),
                'material_PSI_C_AT2': np.array([solver.config.PSI_C_AT2]),
                
                # Solver parameters
                'solver_max_newton_iter': np.array([solver.config.MAX_NEWTON_ITER]),
                'solver_newton_rtol': np.array([solver.config.NEWTON_RTOL]),
                'solver_newton_atol': np.array([solver.config.NEWTON_ATOL]),
                'solver_max_alt_iter': np.array([solver.config.MAX_ALT_ITER]),
                'solver_alt_tol': np.array([solver.config.ALT_TOL]),
                
                # Simulation parameters
                'sim_n_steps': np.array([solver.config.N_STEPS]),
                'sim_max_displacement': np.array([solver.config.MAX_DISPLACEMENT]),
                'sim_plot_frequency': np.array([solver.config.PLOT_FREQUENCY]),
                
                # File configuration
                'config_state_file': solver.config.STATE_FILE.encode('utf-8'),
                
                # Metadata
                'save_timestamp': datetime.now().isoformat().encode('utf-8'),
                'version': np.array([2.0])  # Updated version for new config format
            }
            
            # Save to file
            np.savez_compressed(filename, **save_data)  # Use compression
            
            # Print summary
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"      ✅ Complete state saved to {filename}")
            print(f"         File size: {file_size:.2f} MB")
            print(f"         Contains: mesh ({solver.n_nodes} nodes), solution fields, all parameters")
            print(f"         Version: 2.0 (updated config format)")
            
            return True
            
        except Exception as e:
            print(f"      ❌ ERROR: Failed to save complete state to {filename}")
            print(f"         Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_complete_state(self, filename):
        """
        Load COMPLETE simulation state - everything needed for restart.
        
        Returns:
        --------
        dict: Complete state data ready for solver reconstruction
        """
        try:
            if not os.path.exists(filename):
                print(f"      ⚠️ WARNING: Restart file not found: {filename}")
                return None
            
            print(f"      📖 Loading complete simulation state from {filename}...")
            
            data = np.load(filename, allow_pickle=True)
            
            # Check version for backward compatibility
            version = float(data['version'][0]) if 'version' in data else 1.0
            
            # Extract all data
            state = {
                # Solution fields
                'u': data['u'],
                'v': data['v'],
                'd': data['d'],
                'd_old': data['d_old'],
                'load_step': int(data['load_step'][0]),
                
                # Complete mesh data
                'x': data['mesh_x'],
                'y': data['mesh_y'],
                'connectivity': data['mesh_connectivity'],
                'boundary_nodes': data['mesh_boundary_nodes'],
                'n_nodes': int(data['mesh_n_nodes'][0]),
                'n_elements': int(data['mesh_n_elements'][0]),
                
                # Configuration parameters
                'radius': float(data['config_radius'][0]),
                'mesh_resolution': float(data['config_mesh_resolution'][0]),
                'model_type': data['config_model_type'].tobytes().decode('utf-8'),
                'use_vi_solver': bool(data['config_use_vi_solver'][0]),
                'restart_simulation': bool(data['config_restart_simulation'][0]) if version >= 2.0 else True,
                'loading_mode': data['config_loading_mode'].tobytes().decode('utf-8') if 'config_loading_mode' in data else 'radial',
                'biaxiality_ratio': float(data['config_biaxiality_ratio'][0]) if 'config_biaxiality_ratio' in data else 1.0,
                
                # Boundary condition parameters
                'applied_displacement': float(data['bc_applied_displacement'][0]),
                'irreversibility_threshold': float(data['bc_irreversibility_threshold'][0]),
                
                # Material parameters - Basic (always present)
                'E_FILM': float(data['material_E_film'][0]),
                'E_SUBSTRATE': float(data['material_E_substrate'][0]),
                'NU_FILM': float(data['material_nu_film'][0]),
                'NU_SUBSTRATE': float(data['material_nu_substrate'][0]),
                'LENGTH_SCALE': float(data['material_length_scale'][0]),
                'COUPLING_PARAMETER': float(data['material_coupling_param'][0]),
                'SUBSTRATE_STIFFNESS': float(data['material_substrate_stiffness'][0]),
                
                # Solver parameters
                'MAX_NEWTON_ITER': int(data['solver_max_newton_iter'][0]),
                'NEWTON_RTOL': float(data['solver_newton_rtol'][0]),
                'NEWTON_ATOL': float(data['solver_newton_atol'][0]),
                'MAX_ALT_ITER': int(data['solver_max_alt_iter'][0]),
                'ALT_TOL': float(data['solver_alt_tol'][0]),
                
                # Simulation parameters
                'N_STEPS': int(data['sim_n_steps'][0]),
                'MAX_DISPLACEMENT': float(data['sim_max_displacement'][0]),
                'PLOT_FREQUENCY': int(data['sim_plot_frequency'][0]),
                
                # Metadata
                'save_timestamp': data['save_timestamp'].tobytes().decode('utf-8') if 'save_timestamp' in data else 'unknown',
                'version': version
            }
            
            # Add new material parameters if available (version 2.0+)
            if version >= 2.0:
                state.update({
                    'HF': float(data['material_HF'][0]),
                    'HS': float(data['material_HS'][0]),
                    'CHI': float(data['material_CHI'][0]),
                    'GC': float(data['material_GC'][0]),
                    'AT1': float(data['material_AT1'][0]),
                    'AT2': float(data['material_AT2'][0]),
                    'PSI_C_AT1': float(data['material_PSI_C_AT1'][0]),
                    'PSI_C_AT2': float(data['material_PSI_C_AT2'][0]),
                    'STATE_FILE': data['config_state_file'].tobytes().decode('utf-8')
                })
            else:
                # Provide defaults for backward compatibility
                print(f"      📄 Loading version {version} file - using defaults for new parameters")
                state.update({
                    'HF': 0.25,
                    'HS': 25,
                    'CHI': 769,
                    'GC': 1,
                    'AT1': 1,
                    'AT2': 2,
                    'PSI_C_AT1': 0.5,
                    'PSI_C_AT2': 0.0,
                    'STATE_FILE': "simulation_state.npz"
                })
            
            # Print summary
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"      ✅ Complete state loaded successfully!")
            print(f"         File size: {file_size:.2f} MB")
            print(f"         Version: {version}")
            print(f"         Saved on: {state['save_timestamp']}")
            print(f"         Model: {state['model_type']}, Step: {state['load_step']}")
            print(f"         Mesh: {state['n_nodes']} nodes, {state['n_elements']} elements")
            print(f"         Resolution: {state['mesh_resolution']:.4f}")
            print(f"         Max damage: {np.max(state['d']):.4f}")
            
            return state
            
        except Exception as e:
            print(f"      ❌ ERROR: Failed to load complete state from {filename}")
            print(f"         Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_config_from_state(self, state):
        """
        Create a new SimulationConfig object from loaded state data.
        
        Parameters:
        -----------
        state : dict
            State data from load_complete_state()
            
        Returns:
        --------
        SimulationConfig: New config object with state parameters
        """
        try:
            # Import the config class
            from config import SimulationConfig
            
            # Create new config instance
            config = SimulationConfig()
            
            # Update all parameters from state
            config.RADIUS = state['radius']
            config.MESH_RESOLUTION = state['mesh_resolution']
            config.E_FILM = state['E_FILM']
            config.E_SUBSTRATE = state['E_SUBSTRATE']
            config.NU_FILM = state['NU_FILM']
            config.NU_SUBSTRATE = state['NU_SUBSTRATE']
            config.HF = state['HF']
            config.HS = state['HS']
            config.CHI = state['CHI']
            config.GC = state['GC']
            config.AT1 = state['AT1']
            config.AT2 = state['AT2']
            config.LENGTH_SCALE = state['LENGTH_SCALE']
            config.COUPLING_PARAMETER = state['COUPLING_PARAMETER']
            config.SUBSTRATE_STIFFNESS = state['SUBSTRATE_STIFFNESS']
            config.PSI_C_AT1 = state['PSI_C_AT1']
            config.PSI_C_AT2 = state['PSI_C_AT2']
            config.MAX_NEWTON_ITER = state['MAX_NEWTON_ITER']
            config.NEWTON_RTOL = state['NEWTON_RTOL']
            config.NEWTON_ATOL = state['NEWTON_ATOL']
            config.MAX_ALT_ITER = state['MAX_ALT_ITER']
            config.ALT_TOL = state['ALT_TOL']
            config.N_STEPS = state['N_STEPS']
            config.MAX_DISPLACEMENT = state['MAX_DISPLACEMENT']
            config.PLOT_FREQUENCY = state['PLOT_FREQUENCY']
            config.IRREVERSIBILITY_THRESHOLD = state['irreversibility_threshold']
            config.STATE_FILE = state['STATE_FILE']
            config.USE_VI_SOLVER = state['use_vi_solver']
            config.RESTART_SIMULATION = state['restart_simulation']
            config.LOADING_MODE = state['loading_mode']
            config.BIAXIALITY_RATIO = state['biaxiality_ratio']
            
            print(f"      ⚙️ Configuration reconstructed from state (version {state['version']})")
            
            return config
            
        except Exception as e:
            print(f"      ❌ ERROR: Failed to create config from state: {e}")
            return None
    
    # LEGACY METHODS - Keep for backward compatibility with existing code
    def save_state(self, filename, u, v, d, d_old, load_step):
        """Legacy method - saves only basic fields (MESH MISMATCH RISK!)."""
        print("⚠️ WARNING: Using legacy save_state method. No mesh info saved!")
        print("   Use save_complete_state() for full restart capability.")
        try:
            np.savez(
                filename,
                u=u,
                v=v,
                d=d,
                d_old=d_old,
                load_step=np.array([load_step])
            )
            print(f"      💾 Legacy state saved to {filename}")
            return True
        except Exception as e:
            print(f"      ❌ ERROR: Failed to save legacy state to {filename}. Error: {e}")
            return False
    
    def load_state(self, filename, u_shape, v_shape, d_shape):
        """Legacy method - loads only basic fields (MESH MISMATCH RISK!)."""
        print("⚠️ WARNING: Using legacy load_state method.")
        print("   This may cause mesh mismatch errors!")
        try:
            if not os.path.exists(filename):
                print(f"      ⚠️ WARNING: Restart file not found: {filename}. Starting from scratch.")
                return None, None, None, None, 0
            
            data = np.load(filename)
            
            # Check if this is a complete state file
            if 'mesh_x' in data:
                print("      📖 Detected complete state file - loading with compatibility check...")
                complete_state = self.load_complete_state(filename)
                if complete_state:
                    # Check shape compatibility
                    if (complete_state['u'].shape == u_shape and 
                        complete_state['v'].shape == v_shape and 
                        complete_state['d'].shape == d_shape):
                        print("      ✅ Shape compatibility confirmed")
                        return (complete_state['u'], complete_state['v'], 
                               complete_state['d'], complete_state['d_old'], 
                               complete_state['load_step'])
                    else:
                        print(f"      ❌ ERROR: Shape mismatch detected")
                        print(f"         Expected u: {u_shape}, got: {complete_state['u'].shape}")
                        print(f"         Expected d: {d_shape}, got: {complete_state['d'].shape}")
                        return None, None, None, None, 0
            
            # Legacy file format
            # Check shape compatibility
            if (data['u'].shape != u_shape or 
                data['v'].shape != v_shape or 
                data['d'].shape != d_shape):
                print(f"      ❌ ERROR: Mesh mismatch. Cannot load state from {filename}.")
                print(f"         Expected u: {u_shape}, got: {data['u'].shape}")
                print(f"         Expected d: {d_shape}, got: {data['d'].shape}")
                print(f"         💡 TIP: Use complete state saving to avoid this issue")
                return None, None, None, None, 0
            
            u = data['u']
            v = data['v']
            d = data['d']
            d_old = data['d_old']
            load_step = int(data['load_step'][0])
            
            print(f"      ✅ Legacy state successfully loaded from {filename} at step {load_step}.")
            return u, v, d, d_old, load_step
            
        except Exception as e:
            print(f"      ❌ ERROR: Failed to load state from {filename}. Error: {e}")
            return None, None, None, None, 0
    
    def create_backup(self, filename):
        """Create a backup of the current state file."""
        if os.path.exists(filename):
            backup_filename = filename.replace('.npz', '_backup.npz')
            try:
                import shutil
                shutil.copy2(filename, backup_filename)
                print(f"      📋 Backup created: {backup_filename}")
                return True
            except Exception as e:
                print(f"      ⚠️ WARNING: Failed to create backup: {e}")
                return False
        return False
    
    def list_available_states(self, directory="."):
        """List available state files in directory."""
        state_files = []
        try:
            for file in os.listdir(directory):
                if file.endswith('.npz') and 'state' in file.lower():
                    state_files.append(file)
            
            if state_files:
                print(f"Available state files in {directory}:")
                for i, file in enumerate(state_files, 1):
                    info = self.get_state_info(file)
                    if info:
                        version = info.get('version', 1.0)
                        file_type = f"v{version:.1f}" if 'mesh_resolution' in info else "Legacy"
                        print(f"  {i}. {file} [{file_type}] - Step {info.get('load_step', '?')}, "
                              f"Damage: {info.get('max_damage', 0):.3f}")
                    else:
                        print(f"  {i}. {file} [Unknown format]")
            else:
                print(f"No state files found in {directory}")
                
        except Exception as e:
            print(f"Error listing state files: {e}")
        
        return state_files
    
    def get_state_info(self, filename):
        """Get information about a state file without loading it."""
        try:
            if not os.path.exists(filename):
                return None
            
            data = np.load(filename, allow_pickle=True)
            info = {
                'load_step': int(data['load_step'][0]),
                'u_shape': data['u'].shape,
                'v_shape': data['v'].shape,
                'd_shape': data['d'].shape,
                'max_damage': np.max(data['d']),
                'max_displacement': np.max(np.sqrt(data['u'][:, 0]**2 + data['u'][:, 1]**2))
            }
            
            # Add complete state info if available
            if 'mesh_n_nodes' in data:
                info['n_nodes'] = int(data['mesh_n_nodes'][0])
                info['n_elements'] = int(data['mesh_n_elements'][0])
                info['mesh_resolution'] = float(data['config_mesh_resolution'][0])
                info['model_type'] = data['config_model_type'].tobytes().decode('utf-8')
                info['save_timestamp'] = data['save_timestamp'].tobytes().decode('utf-8') if 'save_timestamp' in data else 'unknown'
                info['version'] = float(data['version'][0]) if 'version' in data else 1.0
            
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            info['file_size_mb'] = file_size
            
            return info
            
        except Exception as e:
            print(f"Error reading state file info: {e}")
            return None