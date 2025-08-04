"""
VTK visualization module for 2D phase-field simulation results.
Handles real-time visualization of displacement, stress, and damage fields.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import os

class VTKVisualizer:
    """VTK-based visualizer for phase-field simulation results."""
    
    def __init__(self, x, y, connectivity):
        """Initialize VTK visualizer with mesh data."""
        self.x = x
        self.y = y
        self.connectivity = connectivity
        self.n_nodes = len(x)
        self.n_elements = len(connectivity)
        
        # Create VTK mesh objects
        self.points = vtk.vtkPoints()
        self.cells = vtk.vtkCellArray()
        self.ugrid = vtk.vtkUnstructuredGrid()
        
        self._setup_mesh()
        print(f"✓ VTK visualizer initialized: {self.n_nodes} nodes, {self.n_elements} elements")
    
    def _setup_mesh(self):
        """Set up VTK mesh structure."""
        # Add points (nodes) - extend 2D to 3D
        for i in range(self.n_nodes):
            self.points.InsertNextPoint(self.x[i], self.y[i], 0.0)
        
        # Add cells (triangular elements)
        for element in self.connectivity:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, element[0])
            triangle.GetPointIds().SetId(1, element[1])
            triangle.GetPointIds().SetId(2, element[2])
            self.cells.InsertNextCell(triangle)
        
        # Set up unstructured grid
        self.ugrid.SetPoints(self.points)
        self.ugrid.SetCells(vtk.VTK_TRIANGLE, self.cells)
    
    def compute_stress_field(self, u, d, film_material):
        """Compute stress field for visualization."""
        from finite_element import TriangularElement
        
        stress_field = np.zeros((self.n_nodes, 3))  # [σxx, σyy, σxy]
        node_count = np.zeros(self.n_nodes)  # For averaging
        
        element = TriangularElement()
        xi_gp, eta_gp, w_gp = element.gauss_points()
        
        for e in range(self.n_elements):
            nodes = self.connectivity[e]
            u_e, d_e = u[nodes], d[nodes]
            coords = np.column_stack((self.x[nodes], self.y[nodes]))
            
            # Compute stress at element center
            xi_center, eta_center = 1/3, 1/3
            N, _, _ = element.shape_functions(xi_center, eta_center)
            detJ, dN_dx, dN_dy = element.compute_jacobian(coords, xi_center, eta_center)
            
            # Strain-displacement matrix
            B = element.strain_displacement_matrix(dN_dx, dN_dy)
            
            # Compute strain and damage at center
            from material_models import ElasticEnergyCalculator
            energy_calc = ElasticEnergyCalculator()
            strain = energy_calc.compute_strain_from_displacement(u_e, B)
            d_center = np.dot(N, d_e)
            
            # Compute stress
            stress = film_material.compute_stress(strain, d_center)
            
            # Distribute to nodes (simple averaging)
            for i, node in enumerate(nodes):
                stress_field[node] += stress
                node_count[node] += 1
        
        # Average stress at nodes
        for i in range(self.n_nodes):
            if node_count[i] > 0:
                stress_field[i] /= node_count[i]
        
        return stress_field
    
    def save_vtk_file(self, u, v, d, film_material, step_number, output_dir="vtk_output"):
        """Save VTK file with current solution fields."""
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create a copy of the grid for this timestep
        ugrid_copy = vtk.vtkUnstructuredGrid()
        ugrid_copy.DeepCopy(self.ugrid)
        
        # Update node positions with current displacement (film only)
        points_displaced = vtk.vtkPoints()
        for i in range(self.n_nodes):
            # Displaced position (amplify for visualization if needed)
            amplification = 1.0  # Set to >1 to amplify displacement for visualization
            x_new = self.x[i] + amplification * u[i, 0]
            y_new = self.y[i] + amplification * u[i, 1]
            points_displaced.InsertNextPoint(x_new, y_new, 0.0)
        
        ugrid_copy.SetPoints(points_displaced)
        
        # Add film displacement field
        u_magnitude = np.sqrt(u[:, 0]**2 + u[:, 1]**2)
        u_mag_vtk = numpy_to_vtk(u_magnitude, deep=True, array_type=vtk.VTK_FLOAT)
        u_mag_vtk.SetName("Film_Displacement_Magnitude")
        ugrid_copy.GetPointData().AddArray(u_mag_vtk)
        
        # Add film displacement components
        u_x_vtk = numpy_to_vtk(u[:, 0], deep=True, array_type=vtk.VTK_FLOAT)
        u_x_vtk.SetName("Film_Displacement_X")
        ugrid_copy.GetPointData().AddArray(u_x_vtk)
        
        u_y_vtk = numpy_to_vtk(u[:, 1], deep=True, array_type=vtk.VTK_FLOAT)
        u_y_vtk.SetName("Film_Displacement_Y")
        ugrid_copy.GetPointData().AddArray(u_y_vtk)
        
        # Add damage field
        d_vtk = numpy_to_vtk(d, deep=True, array_type=vtk.VTK_FLOAT)
        d_vtk.SetName("Damage")
        ugrid_copy.GetPointData().AddArray(d_vtk)
        ugrid_copy.GetPointData().SetActiveScalars("Damage")  # Set as default for coloring
        
        # Compute and add stress field
        try:
            stress_field = self.compute_stress_field(u, d, film_material)
            
            # Stress magnitude
            stress_magnitude = np.sqrt(stress_field[:, 0]**2 + stress_field[:, 1]**2 + 
                                     0.5 * stress_field[:, 2]**2)  # Von Mises-like
            stress_mag_vtk = numpy_to_vtk(stress_magnitude, deep=True, array_type=vtk.VTK_FLOAT)
            stress_mag_vtk.SetName("Stress_Magnitude")
            ugrid_copy.GetPointData().AddArray(stress_mag_vtk)
            
            # Stress components
            stress_xx_vtk = numpy_to_vtk(stress_field[:, 0], deep=True, array_type=vtk.VTK_FLOAT)
            stress_xx_vtk.SetName("Stress_XX")
            ugrid_copy.GetPointData().AddArray(stress_xx_vtk)
            
            stress_yy_vtk = numpy_to_vtk(stress_field[:, 1], deep=True, array_type=vtk.VTK_FLOAT)
            stress_yy_vtk.SetName("Stress_YY")
            ugrid_copy.GetPointData().AddArray(stress_yy_vtk)
            
            stress_xy_vtk = numpy_to_vtk(stress_field[:, 2], deep=True, array_type=vtk.VTK_FLOAT)
            stress_xy_vtk.SetName("Stress_XY")
            ugrid_copy.GetPointData().AddArray(stress_xy_vtk)
            
        except Exception as e:
            print(f"Warning: Could not compute stress field: {e}")
        
        # Add substrate displacement for reference (but don't visualize)
        v_magnitude = np.sqrt(v[:, 0]**2 + v[:, 1]**2)
        v_mag_vtk = numpy_to_vtk(v_magnitude, deep=True, array_type=vtk.VTK_FLOAT)
        v_mag_vtk.SetName("Substrate_Displacement_Magnitude")
        ugrid_copy.GetPointData().AddArray(v_mag_vtk)
        
        # Write VTK file
        filename = os.path.join(output_dir, f"phase_field_step_{step_number:04d}.vtu")
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(ugrid_copy)
        writer.Write()
        
        print(f"    ✓ VTK file saved: {filename}")
        
        # Also save a summary file for the series
        self._update_pvd_file(step_number, output_dir)
        
        return filename
    
    def _update_pvd_file(self, step_number, output_dir):
        """Update PVD file for time series visualization in ParaView."""
        pvd_filename = os.path.join(output_dir, "phase_field_series.pvd")
        vtu_filename = f"phase_field_step_{step_number:04d}.vtu"
        
        # Read existing PVD or create new one
        if os.path.exists(pvd_filename):
            # Read and append
            with open(pvd_filename, 'r') as f:
                lines = f.readlines()
            
            # Remove closing tags
            if lines and '</Collection>' in lines[-1]:
                lines = lines[:-2]  # Remove last two lines
        else:
            # Create new PVD file
            lines = [
                '<?xml version="1.0"?>\\n',
                '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\\n',
                '  <Collection>\\n'
            ]
        
        # Add new timestep
        lines.append(f'    <DataSet timestep="{step_number}" group="" part="0" file="{vtu_filename}"/>\\n')
        
        # Add closing tags
        lines.extend([
            '  </Collection>\\n',
            '</VTKFile>\\n'
        ])
        
        # Write updated PVD
        with open(pvd_filename, 'w') as f:
            f.writelines(lines)
    
    def create_preview_image(self, u, d, step_number, output_dir="vtk_output"):
        """Create a preview image using VTK rendering (optional)."""
        try:
            # Create a copy with current data
            ugrid_preview = vtk.vtkUnstructuredGrid()
            ugrid_preview.DeepCopy(self.ugrid)
            
            # Add damage field
            d_vtk = numpy_to_vtk(d, deep=True, array_type=vtk.VTK_FLOAT)
            d_vtk.SetName("Damage")
            ugrid_preview.GetPointData().SetScalars(d_vtk)
            
            # Create mapper
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(ugrid_preview)
            mapper.SetScalarRange(0, 1)  # Damage range
            
            # Create actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Create renderer
            renderer = vtk.vtkRenderer()
            renderer.AddActor(actor)
            renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background
            
            # Create render window
            render_window = vtk.vtkRenderWindow()
            render_window.AddRenderer(renderer)
            render_window.SetSize(800, 800)
            render_window.SetOffScreenRendering(1)  # No window
            
            # Render
            render_window.Render()
            
            # Save image
            w2if = vtk.vtkWindowToImageFilter()
            w2if.SetInput(render_window)
            w2if.Update()
            
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(os.path.join(output_dir, f"preview_step_{step_number:04d}.png"))
            writer.SetInputConnection(w2if.GetOutputPort())
            writer.Write()
            
            print(f"    ✓ Preview image saved")
            
        except Exception as e:
            print(f"    Warning: Could not create preview image: {e}")


# Integration with the main solver
def add_vtk_to_solver():
    """
    Code to add to your main solver class (solver.py).
    Add this to the __init__ method after mesh setup:
    """
    code_to_add = '''
    # Add this to AT1_2D_ActiveSet_Solver.__init__ after mesh setup
    self.vtk_visualizer = VTKVisualizer(self.x, self.y, self.connectivity)
    
    # Add this to your solve_load_step method where you do plotting
    # (replace or add alongside your existing plotting)
    if ((step + 1) % plot_every == 0 or (step + 1) == n_steps or 
        (max_damage > 0.001 and np.max(self.d_old) < 0.001)):
        
        # Existing matplotlib plotting
        self.plot_results_2d(step_number=step + 1, save_png=True)
        
        # NEW: VTK output
        self.vtk_visualizer.save_vtk_file(
            self.u, self.v, self.d, self.film_material, 
            step_number=step + 1
        )
        
        self.save_state(state_file)
    '''
    
    return code_to_add

if __name__ == "__main__":
    print("VTK Visualizer module")
    print("Add this to your solver to enable VTK output:")
    print(add_vtk_to_solver())