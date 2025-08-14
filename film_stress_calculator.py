import numpy as np
import csv
from finite_element import TriangularElement

class FilmStressCalculator:
    """
    Independent class to calculate volume-averaged stress in the film only.
    Uses reference coordinates for proper engineering stress calculation.
    """
    
    def __init__(self, output_filename='film_stress.csv'):
        """
        Initialize film stress calculator.
        
        Parameters:
        -----------
        output_filename : str
            CSV file to write stress data
        """
        self.output_filename = output_filename
        self.data_history = []
        self._write_header()
    
    def _write_header(self):
        """Write CSV header."""
        header = [
            'load_step', 
            'applied_displacement',
            'film_stress_11',     # σ11 in film
            'film_stress_22',     # σ22 in film  
            'film_stress_12',     # σ12 in film
            'film_strain_11',     # ε11 in film
            'film_strain_22',     # ε22 in film
            'film_strain_12',     # γ12 in film
            'max_damage',
            'avg_damage',
            'film_volume'         # Total film volume (reference)
        ]
        
        with open(self.output_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        
        print(f"✓ Film stress file initialized: {self.output_filename}")
    
    def calculate_film_stress(self, solver):
        """
        Calculate volume-averaged stress and strain in the film using reference coordinates.
        
        Parameters:
        -----------
        solver : AT1_2D_ActiveSet_Solver
            The phase field solver instance
            
        Returns:
        --------
        film_stress : ndarray (3,)
            Volume-averaged stress [σ11, σ22, σ12] in film
        film_strain : ndarray (3,)
            Volume-averaged strain [ε11, ε22, γ12] in film
        max_damage : float
            Maximum damage in domain
        avg_damage : float
            Average damage in domain
        film_volume : float
            Total film volume (reference)
        """
        
        # Initialize accumulators
        total_film_volume = 0.0
        volume_weighted_stress = np.zeros(3)  # [σ11, σ22, σ12]
        volume_weighted_strain = np.zeros(3)  # [ε11, ε22, γ12]
        
        # Get finite element and quadrature
        element = TriangularElement()
        xi_gp, eta_gp, w_gp = element.gauss_points()
        

        
        # Loop over all elements
        for e in range(solver.n_elem):
            nodes = solver.connectivity[e]
            u_e = solver.u[nodes]  # Film displacement at element nodes
            v_e = solver.v[nodes]  # Film displacement at element nodes
            d_e = solver.d[nodes]  # Damage at element nodes
            # USE REFERENCE COORDINATES (undeformed mesh)
            coords_ref = np.column_stack((solver.x[nodes], solver.y[nodes]))
            # print(f"  DEBUG Element {e}:")
            # print(f"    Nodes: {nodes}")
            # print(f"    Reference coords: {coords_ref}")
            # print(f"    Displacements: {u_e}")
            # print(f"    Max displacement magnitude: {np.max(np.sqrt(u_e[:,0]**2 + u_e[:,1]**2)):.6f}")
                
            # Element volume and stress/strain contributions
            element_volume = 0.0
            element_stress_contribution = np.zeros(3)
            element_strain_contribution = np.zeros(3)
            
            # Gauss point integration
            for gp in range(len(xi_gp)):
                # Shape functions and derivatives w.r.t. REFERENCE coordinates
                N, _, _ = element.shape_functions(xi_gp[gp], eta_gp[gp])
                detJ, dN_dx, dN_dy = element.compute_jacobian(coords_ref, xi_gp[gp], eta_gp[gp])
                
                # Integration weight (REFERENCE volume)
                weight = detJ * w_gp[gp]
                element_volume += weight
                
                # Strain-displacement matrix (w.r.t. reference coordinates)
                B = element.strain_displacement_matrix(dN_dx, dN_dy)
                # print(f"    B matrix shape: {B.shape}")
                # print(f"    B matrix:")
                # print(f"      {B}")
                
                # Compute strain at Gauss point
                strain_gp = solver.energy_calc.compute_strain_from_displacement(u_e, B)
                
                # Compute stress at Gauss point (in film only)
                d_gp = np.dot(N, d_e)  # Damage at Gauss point
                # print(f"    Strain at GP: ε11={strain_gp[0]:.6f}, ε22={strain_gp[1]:.6f}, γ12={strain_gp[2]:.6f}")

                # print(f"    Material D matrix diagonal: {np.diag(solver.film_material.D)}")
                
                stress_gp = solver.film_material.compute_stress(strain_gp, d_e[0])
                #stress_gp = solver.substrate_material.compute_stress(strain_gp)

                # print(f"    Stress at GP: σ11={stress_gp[0]:.6f}, σ22={stress_gp[1]:.6f}, σ12={stress_gp[2]:.6f}")                
                # Accumulate contributions
                element_stress_contribution += weight * stress_gp
                element_strain_contribution += strain_gp
            
            # Add element contributions to global totals
            total_film_volume += element_volume
            volume_weighted_stress += element_stress_contribution
            volume_weighted_strain += element_strain_contribution
        
        # Volume-average the stress and strain
        if total_film_volume > 1e-12:
            film_stress = volume_weighted_stress / total_film_volume
            film_strain = volume_weighted_strain 
        else:
            film_stress = np.zeros(3)
            film_strain = np.zeros(3)
        
        # Calculate damage statistics
        max_damage = np.max(solver.d)
        avg_damage = np.mean(solver.d)


        # print(f"    Strain at GP: ε11={strain_gp[0]:.6f}, ε22={strain_gp[1]:.6f}, γ12={strain_gp[2]:.6f}")

        all_coords = np.column_stack((solver.x, solver.y))
        all_disps = solver.u
        radial_disps = []
        # print(f"  Film strain: {film_strain[0]:.6f}")
        # print(f"  Film strain: {film_strain[1]:.6f}")
        # print(f"  Film strain: {film_strain[2]:.6f}")


        # for i in range(len(solver.boundary_nodes)):
        #     idx = solver.boundary_nodes[i]
        #     x, y = all_coords[idx]
        #     ux, uy = all_disps[idx]
        #     r = np.sqrt(x**2 + y**2)
        #     ur = (x*ux + y*uy) / r
        #     radial_disps.append(ur)

        # print(f"  Average boundary radial displacement: {np.mean(radial_disps):.6f}")
        # print(f"  Min/Max boundary radial displacement: {np.min(radial_disps):.6f} / {np.max(radial_disps):.6f}")        
        return film_stress, film_strain, max_damage, avg_damage, total_film_volume
    
    def write_data(self, solver, load_step, applied_displacement):
        """
        Calculate and write film stress data to CSV file.
        
        Parameters:
        -----------
        solver : AT1_2D_ActiveSet_Solver
            The phase field solver
        load_step : int
            Current load step number
        applied_displacement : float
            Applied boundary displacement
        """
        
        # Calculate film stress and strain
        film_stress, film_strain, max_damage, avg_damage, film_volume = self.calculate_film_stress(solver)
        
        # Prepare data row
        data_row = [
            load_step,
            applied_displacement,
            film_stress[0],    # σ11
            film_stress[1],    # σ22
            film_stress[2],    # σ12
            film_strain[0],    # ε11
            film_strain[1],    # ε22
            film_strain[2],    # γ12
            max_damage,
            avg_damage,
            film_volume
        ]
        
        # Store in memory
        self.data_history.append(data_row)
        
        # Write to CSV file
        with open(self.output_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data_row)
        
        # Print summary
        # print(f"  Film stress: σ11={film_stress[0]:.3e}, σ22={film_stress[1]:.3e}, "
        #       f"σ12={film_stress[2]:.3e}, max_dmg={max_damage:.4f}")
        
        return film_stress, film_strain
    
    def save_numpy_data(self, filename='film_stress_data.npz'):
        """Save data in numpy format."""
        if not self.data_history:
            print("No data to save!")
            return
        
        data_array = np.array(self.data_history)
        
        np.savez(filename,
                load_steps=data_array[:, 0],
                applied_displacement=data_array[:, 1],
                film_stress_11=data_array[:, 2],
                film_stress_22=data_array[:, 3],
                film_stress_12=data_array[:, 4],
                film_strain_11=data_array[:, 5],
                film_strain_22=data_array[:, 6],
                film_strain_12=data_array[:, 7],
                max_damage=data_array[:, 8],
                avg_damage=data_array[:, 9],
                film_volume=data_array[:, 10])
        
        print(f"✓ Film stress numpy data saved: {filename}")


# Standalone function (no class needed)
def calculate_film_volume_averaged_stress(solver):
    """
    Standalone function to calculate volume-averaged stress in film using reference coordinates.
    
    Parameters:
    -----------
    solver : AT1_2D_ActiveSet_Solver
        The phase field solver instance
        
    Returns:
    --------
    dict : Dictionary with stress/strain components and statistics
    """
    
    # Initialize
    total_volume = 0.0
    weighted_stress = np.zeros(3)
    weighted_strain = np.zeros(3)
    
    element = TriangularElement()
    xi_gp, eta_gp, w_gp = element.gauss_points()
    
    # Integrate over all elements
    for e in range(solver.n_elem):
        nodes = solver.connectivity[e]
        u_e = solver.u[nodes]  # Film displacement
        d_e = solver.d[nodes]  # Damage
        coords_ref = np.column_stack((solver.x[nodes], solver.y[nodes]))  # Reference coordinates
        
        for gp in range(len(xi_gp)):
            N, _, _ = element.shape_functions(xi_gp[gp], eta_gp[gp])
            detJ, dN_dx, dN_dy = element.compute_jacobian(coords_ref, xi_gp[gp], eta_gp[gp])
            weight = detJ * w_gp[gp]
            
            # Strain and stress at Gauss point
            B = element.strain_displacement_matrix(dN_dx, dN_dy)
            strain_gp = solver.energy_calc.compute_strain_from_displacement(u_e, B)
            d_gp = np.dot(N, d_e)
            stress_gp = solver.film_material.compute_stress(strain_gp, d_gp)
            
            # Accumulate
            total_volume += weight
            weighted_stress += weight * stress_gp
            weighted_strain += weight * strain_gp
    
    # Volume average
    if total_volume > 1e-12:
        avg_stress = weighted_stress / total_volume
        avg_strain = weighted_strain / total_volume
    else:
        avg_stress = np.zeros(3)
        avg_strain = np.zeros(3)
    
    return {
        'stress_11': avg_stress[0],
        'stress_22': avg_stress[1], 
        'stress_12': avg_stress[2],
        'strain_11': avg_strain[0],
        'strain_22': avg_strain[1],
        'strain_12': avg_strain[2],
        'max_damage': np.max(solver.d),
        'avg_damage': np.mean(solver.d),
        'volume': total_volume
    }


# Simple usage examples:

def example_usage_with_class():
    """Example: Using the FilmStressCalculator class"""
    
    # Create calculator
    film_calc = FilmStressCalculator('my_film_stress.csv')
    
    # In your simulation loop:
    # for step in range(n_steps):
    #     applied_disp = max_displacement * (step + 1) / n_steps
    #     success = solver.solve_load_step(applied_disp)
    #     
    #     if success:
    #         film_calc.write_data(solver, step + 1, applied_disp)
    
    # At the end:
    # film_calc.save_numpy_data()
    
    print("Usage: film_calc.write_data(solver, load_step, applied_displacement)")


def example_usage_standalone():
    """Example: Using the standalone function"""
    
    # In your simulation loop:
    # for step in range(n_steps):
    #     applied_disp = max_displacement * (step + 1) / n_steps  
    #     success = solver.solve_load_step(applied_disp)
    #     
    #     if success:
    #         stress_data = calculate_film_volume_averaged_stress(solver)
    #         print(f"Step {step}: σ11={stress_data['stress_11']:.3e}")
    
    print("Usage: stress_data = calculate_film_volume_averaged_stress(solver)")


# Integration with your solver (add this method to your solver class):
def add_to_solver_class():
    """Add this method to your AT1_2D_ActiveSet_Solver class"""
    
    def get_film_stress(self):
        """Get current volume-averaged film stress."""
        return calculate_film_volume_averaged_stress(self)
    
    def setup_film_stress_output(self, filename='film_stress.csv'):
        """Setup automatic film stress output."""
        self.film_stress_calc = FilmStressCalculator(filename)
    
    def write_film_stress_data(self, applied_displacement):
        """Write film stress data (call after each load step)."""
        if hasattr(self, 'film_stress_calc'):
            self.film_stress_calc.write_data(self, self.load_step, applied_displacement)


if __name__ == "__main__":
    print("Film Stress Calculator")
    print("======================")
    print()
    print("Class usage:")
    print("  calc = FilmStressCalculator('output.csv')")
    print("  calc.write_data(solver, step, displacement)")
    print()
    print("Function usage:")
    print("  data = calculate_film_volume_averaged_stress(solver)")
    print("  print(data['stress_11'], data['stress_22'], data['stress_12'])")
    print()
    print("This version uses REFERENCE coordinates for proper engineering stress calculation.")
    print("Film volume should remain constant, and stress should increase with applied load.")