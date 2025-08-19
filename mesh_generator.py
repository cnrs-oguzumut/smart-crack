"""
Professional mesh generator using Triangle library.
Drop-in replacement for the old mesh_generator.py

Install: pip install triangle

This maintains the same interface as the old BaseMeshGenerator
so your solver.py won't need any changes.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time


def print_progress(percentage, step_name=""):
    """Print progress bar with percentage."""
    bar_length = 40
    filled_length = int(bar_length * percentage // 100)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    if step_name:
        print(f"\r🔄 {step_name}: |{bar}| {percentage:.1f}% ", end='', flush=True)
    else:
        print(f"\r|{bar}| {percentage:.1f}% ", end='', flush=True)
    
    if percentage >= 100:
        print("✅ Complete!")


class ProfessionalCircularMeshGenerator:
    """Professional circular mesh generator using Triangle library."""
    
    def __init__(self, radius=1.0, target_edge_length=0.05, boundary_layers=3):
        """
        Initialize with same interface as old generator.
        
        Parameters:
        - radius: Domain radius
        - target_edge_length: Target edge length (converted to area constraint)
        - boundary_layers: Not used in Triangle, but kept for compatibility
        """
        try:
            import triangle
            self.triangle = triangle
        except ImportError:
            print("⚠️  Triangle library not found!")
            print("   Install with: pip install triangle")
            print("   Falling back to simple mesh generator...")
            self.triangle = None
        
        self.radius = radius
        self.target_edge_length = target_edge_length
        self.boundary_layers = boundary_layers  # Kept for compatibility
        
        # Convert edge length to area constraint for Triangle
        self.max_area = (target_edge_length ** 2) * 0.5  # Approximately equilateral triangles
        self.min_angle = 25  # Minimum angle constraint
    
    def generate_mesh(self):
        """Generate mesh with same interface as old generator."""
        if self.triangle is None:
            return self._fallback_simple_mesh()
        
        print("🚀 Starting Triangle mesh generation...")
        start_time = time.time()
        
        print_progress(10, "Setting up boundary")
        boundary_dict = self._create_circle_boundary()
        
        print_progress(40, "Triangulating with quality constraints")
        
        # Triangle options:
        # 'p' = planar straight line graph (respects boundary)
        # 'q' = quality mesh with minimum angle
        # 'a' = maximum area constraint  
        # 'D' = Delaunay triangulation
        options = f'pq{self.min_angle}a{self.max_area}D'
        
        try:
            mesh = self.triangle.triangulate(boundary_dict, options)
        except Exception as e:
            print(f"Triangle failed: {e}")
            print("Falling back to simple mesh...")
            return self._fallback_simple_mesh()
        
        print_progress(80, "Processing results")
        
        # Extract results
        vertices = mesh['vertices']
        triangles = mesh['triangles']
        
        # Detect boundary nodes
        boundary_nodes = self._detect_boundary_nodes(triangles, vertices)
        
        print_progress(100, "Finalizing")
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds")
        
        x_coords, y_coords = vertices[:, 0], vertices[:, 1]
        n_nodes, n_elem = len(vertices), len(triangles)
        
        # Calculate quality metrics
        quality_metrics = self._compute_quality_metrics(vertices, triangles)
        
        self._print_mesh_info(n_nodes, n_elem, boundary_nodes, quality_metrics)
        
        return x_coords, y_coords, triangles, boundary_nodes, n_nodes, n_elem
    
    def _create_circle_boundary(self):
        """Create circle boundary for Triangle."""
        # Calculate number of boundary points based on target edge length
        circumference = 2 * np.pi * self.radius
        n_boundary = max(int(circumference / self.target_edge_length), 32)
        
        theta = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
        
        vertices = np.column_stack([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta)
        ])
        
        # Create segments (edges) connecting consecutive boundary points
        segments = np.column_stack([
            np.arange(n_boundary),
            (np.arange(n_boundary) + 1) % n_boundary
        ])
        
        return {'vertices': vertices, 'segments': segments}
    
    def _detect_boundary_nodes(self, triangles, vertices):
        """Detect boundary nodes from triangulation."""
        edge_counter = Counter()
        for tri in triangles:
            edges = [(tri[i], tri[(i+1)%3]) for i in range(3)]
            for a, b in edges:
                edge_counter[tuple(sorted((a, b)))] += 1
        
        # Boundary edges appear only once
        boundary_nodes = {node for (a, b), count in edge_counter.items() 
                         if count == 1 for node in (a, b)}
        return np.array(sorted(boundary_nodes))
    
    def _compute_quality_metrics(self, vertices, triangles):
        """Compute mesh quality metrics."""
        angles, aspect_ratios, qualities = [], [], []
        
        for tri in triangles:
            triangle_points = vertices[tri]
            
            # Triangle angles
            tri_angles = self._triangle_angles(triangle_points)
            angles.extend(tri_angles)
            
            # Aspect ratio
            edge_lengths = [np.linalg.norm(triangle_points[(i+1)%3] - triangle_points[i]) 
                          for i in range(3)]
            aspect_ratios.append(max(edge_lengths) / min(edge_lengths))
            
            # Quality metric
            qualities.append(self._triangle_quality(triangle_points))
        
        return {
            'min_angle': np.min(angles),
            'max_angle': np.max(angles),
            'avg_aspect_ratio': np.mean(aspect_ratios),
            'avg_quality': np.mean(qualities)
        }
    
    def _triangle_angles(self, triangle_points):
        """Compute triangle angles in degrees."""
        p1, p2, p3 = triangle_points
        
        v1, v2, v3 = p2 - p1, p3 - p1, p3 - p2
        
        def safe_angle(va, vb):
            cos_angle = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle)
        
        angle1 = safe_angle(v1, v2)
        angle2 = safe_angle(-v1, v3)
        angle3 = np.pi - angle1 - angle2
        
        return [np.degrees(angle) for angle in [angle1, angle2, angle3]]
    
    def _triangle_quality(self, triangle_points):
        """Compute triangle quality metric (0=poor, 1=perfect)."""
        p1, p2, p3 = triangle_points
        
        # Edge lengths
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        
        # Area using cross product
        area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
        
        if area < 1e-12:
            return 0.0
        
        # Quality metric: ratio of inscribed to circumscribed circle radius
        perimeter = a + b + c
        return 4 * np.sqrt(3) * area / (perimeter**2)
    
    def _print_mesh_info(self, n_nodes, n_elem, boundary_nodes, quality_metrics):
        """Print mesh generation summary."""
        print(f"✅ Professional Triangle Mesh Generated:")
        print(f"   📊 Nodes: {n_nodes}, Elements: {n_elem}, Boundary nodes: {len(boundary_nodes)}")
        print(f"   📐 Min angle: {quality_metrics['min_angle']:.1f}° (target: {self.min_angle}°)")
        print(f"   📐 Max angle: {quality_metrics['max_angle']:.1f}°")
        print(f"   📏 Aspect ratio (avg): {quality_metrics['avg_aspect_ratio']:.2f}")
        print(f"   ⭐ Quality metric (avg): {quality_metrics['avg_quality']:.3f}")
        print(f"   🎯 Target edge length: {self.target_edge_length}")
    
    def plot_mesh(self, x, y, connectivity, boundary_nodes, save_png=True):
        """Plot mesh with quality visualization."""
        print("🎨 Generating mesh visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left plot: Basic mesh
        ax1.triplot(x, y, connectivity, 'b-', lw=0.3, alpha=0.7)
        ax1.scatter(x, y, c='lightblue', s=8, alpha=0.6, label='Interior Nodes')
        ax1.scatter(x[boundary_nodes], y[boundary_nodes], 
                   c='red', s=25, label='Boundary Nodes', zorder=5)
        
        # Plot boundary circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = self.radius * np.cos(theta)
        circle_y = self.radius * np.sin(theta)
        ax1.plot(circle_x, circle_y, 'k--', lw=2, alpha=0.8, label='Domain Boundary')
        
        ax1.set_title(f'Triangle Mesh\n{len(connectivity)} elements, {len(x)} nodes')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Quality visualization
        points = np.column_stack([x, y])
        qualities = [self._triangle_quality(points[tri]) for tri in connectivity]
        
        im = ax2.tripcolor(x, y, connectivity, qualities, shading='flat', 
                          cmap='RdYlGn', vmin=0, vmax=1)
        ax2.triplot(x, y, connectivity, 'k-', lw=0.1, alpha=0.3)
        
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Triangle Quality (0=poor, 1=perfect)')
        
        # Plot boundary circle
        ax2.plot(circle_x, circle_y, 'k--', lw=2, alpha=0.8)
        
        ax2.set_title(f'Quality: Avg={np.mean(qualities):.3f}, Min={np.min(qualities):.3f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_png:
            plt.savefig('triangle_circular_mesh.png', dpi=150, bbox_inches='tight')
            print("✅ Saved mesh plot: triangle_circular_mesh.png")
        
        plt.show()
    
    def _fallback_simple_mesh(self):
        """Fallback to simple mesh if Triangle is not available."""
        print("🔄 Using fallback simple mesh generator...")
        
        # Simple grid-based mesh
        margin = self.target_edge_length
        x_range = np.arange(-self.radius - margin, self.radius + margin, self.target_edge_length)
        y_range = np.arange(-self.radius - margin, self.radius + margin, self.target_edge_length)
        X, Y = np.meshgrid(x_range, y_range)
        grid_points = np.vstack((X.ravel(), Y.ravel())).T

        # Keep points inside circle
        r2 = np.sum(grid_points**2, axis=1)
        points = grid_points[r2 < (self.radius * 1.02)**2]

        # Simple Delaunay triangulation
        from scipy.spatial import Delaunay
        delaunay = Delaunay(points)
        
        # Filter triangles by centroid
        is_inside = lambda tri: np.linalg.norm(points[tri].mean(axis=0)) < self.radius * 0.99
        simplices = np.array([tri for tri in delaunay.simplices if is_inside(tri)])

        # Remap indices
        used_nodes = np.unique(simplices)
        new_index = -np.ones(len(points), dtype=int)
        new_index[used_nodes] = np.arange(len(used_nodes))
        
        final_points = points[used_nodes]
        connectivity = new_index[simplices]
        boundary_nodes = self._detect_boundary_nodes(connectivity, final_points)
        
        print(f"✅ Fallback mesh: {len(final_points)} nodes, {len(connectivity)} triangles")
        
        return final_points[:, 0], final_points[:, 1], connectivity, boundary_nodes, len(final_points), len(connectivity)


class ProfessionalRectangularMeshGenerator:
    """Professional rectangular mesh generator using Triangle library."""
    
    def __init__(self, radius=1.0, target_edge_length=0.05, boundary_layers=3):
        """
        Initialize rectangular mesh generator.
        Maps circular parameters to rectangular domain for compatibility.
        """
        try:
            import triangle
            self.triangle = triangle
        except ImportError:
            print("⚠️  Triangle library not found!")
            print("   Install with: pip install triangle")
            self.triangle = None
        
        self.radius = radius
        self.target_edge_length = target_edge_length
        self.boundary_layers = boundary_layers
        
        # Map to rectangular domain
        self.width = 1.*radius
        self.height = .9*radius
        self.x_min, self.x_max = -self.width/2, self.width/2
        self.y_min, self.y_max = -self.height/2, self.height/2
        
        # Convert edge length to area constraint
        self.max_area = (target_edge_length ** 2) * 0.5
        self.min_angle = 25
    
    def generate_mesh(self):
        """Generate rectangular mesh."""
        if self.triangle is None:
            return self._fallback_simple_rectangular_mesh()
        
        print("🚀 Starting Triangle rectangular mesh generation...")
        start_time = time.time()
        
        print_progress(10, "Setting up rectangular boundary")
        boundary_dict = self._create_rectangle_boundary()
        
        print_progress(40, "Triangulating")
        options = f'pq{self.min_angle}a{self.max_area}D'
        
        try:
            mesh = self.triangle.triangulate(boundary_dict, options)
        except Exception as e:
            print(f"Triangle failed: {e}")
            return self._fallback_simple_rectangular_mesh()
        
        print_progress(80, "Processing results")
        
        vertices = mesh['vertices']
        triangles = mesh['triangles']
        boundary_nodes = self._detect_boundary_nodes(triangles, vertices)
        
        print_progress(100, "Finalizing")
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds")
        
        x_coords, y_coords = vertices[:, 0], vertices[:, 1]
        n_nodes, n_elem = len(vertices), len(triangles)
        
        quality_metrics = self._compute_quality_metrics(vertices, triangles)
        self._print_mesh_info(n_nodes, n_elem, boundary_nodes, quality_metrics)
        
        return x_coords, y_coords, triangles, boundary_nodes, n_nodes, n_elem
    
    def _create_rectangle_boundary(self):
        """Create rectangle boundary for Triangle."""
        # Calculate boundary point density
        perimeter = 2 * (self.width + self.height)
        n_boundary_total = max(int(perimeter / self.target_edge_length), 40)
        
        n_x = max(int(n_boundary_total * self.width / perimeter), 4)
        n_y = max(int(n_boundary_total * self.height / perimeter), 4)
        
        # Create boundary points
        vertices = []
        segments = []
        current_idx = 0
        
        # Bottom edge
        x_bottom = np.linspace(self.x_min, self.x_max, n_x)
        y_bottom = np.full(n_x, self.y_min)
        bottom_vertices = np.column_stack([x_bottom, y_bottom])
        vertices.append(bottom_vertices)
        
        # Bottom segments
        for i in range(n_x - 1):
            segments.append([current_idx + i, current_idx + i + 1])
        current_idx += n_x
        
        # Right edge (excluding corner)
        y_right = np.linspace(self.y_min, self.y_max, n_y)[1:]
        x_right = np.full(n_y - 1, self.x_max)
        right_vertices = np.column_stack([x_right, y_right])
        vertices.append(right_vertices)
        
        # Right segments
        segments.append([current_idx - 1, current_idx])  # Connect to bottom
        for i in range(n_y - 2):
            segments.append([current_idx + i, current_idx + i + 1])
        current_idx += n_y - 1
        
        # Top edge (excluding corner)
        x_top = np.linspace(self.x_max, self.x_min, n_x)[1:]
        y_top = np.full(n_x - 1, self.y_max)
        top_vertices = np.column_stack([x_top, y_top])
        vertices.append(top_vertices)
        
        # Top segments
        segments.append([current_idx - 1, current_idx])  # Connect to right
        for i in range(n_x - 2):
            segments.append([current_idx + i, current_idx + i + 1])
        current_idx += n_x - 1
        
        # Left edge (excluding corners)
        y_left = np.linspace(self.y_max, self.y_min, n_y)[1:-1]
        x_left = np.full(len(y_left), self.x_min)
        left_vertices = np.column_stack([x_left, y_left])
        vertices.append(left_vertices)
        
        # Left segments
        segments.append([current_idx - 1, current_idx])  # Connect to top
        for i in range(len(y_left) - 1):
            segments.append([current_idx + i, current_idx + i + 1])
        segments.append([current_idx + len(y_left) - 1, 0])  # Connect back to start
        
        all_vertices = np.vstack(vertices)
        
        return {'vertices': all_vertices, 'segments': np.array(segments)}
    
    def _detect_boundary_nodes(self, triangles, vertices):
        """Detect all boundary nodes (left, right, top, bottom edges)."""
        from collections import Counter
        
        edge_counter = Counter()
        for tri in triangles:
            edges = [(tri[i], tri[(i+1)%3]) for i in range(3)]
            for a, b in edges:
                edge_counter[tuple(sorted((a, b)))] += 1
        
        all_boundary_nodes = {node for (a, b), count in edge_counter.items()
                            if count == 1 for node in (a, b)}
        
        # Filter for all boundary nodes (left, right, top, bottom)
        tolerance = self.target_edge_length * 0.2
        
        left_nodes = [n for n in all_boundary_nodes
                    if abs(vertices[n, 0] - self.x_min) < tolerance]
        right_nodes = [n for n in all_boundary_nodes
                    if abs(vertices[n, 0] - self.x_max) < tolerance]
        bottom_nodes = [n for n in all_boundary_nodes
                        if abs(vertices[n, 1] - self.y_min) < tolerance]
        top_nodes = [n for n in all_boundary_nodes
                    if abs(vertices[n, 1] - self.y_max) < tolerance]
        
        return np.array(sorted(left_nodes + right_nodes + bottom_nodes + top_nodes)) 
        #return np.array(sorted(all_boundary_nodes))  

        #return np.array(sorted(left_nodes + right_nodes ))    

    # Include the same helper methods as circular generator
    def _compute_quality_metrics(self, vertices, triangles):
        """Same as circular generator."""
        angles, aspect_ratios, qualities = [], [], []
        
        for tri in triangles:
            triangle_points = vertices[tri]
            
            # Triangle angles
            tri_angles = self._triangle_angles(triangle_points)
            angles.extend(tri_angles)
            
            # Aspect ratio
            edge_lengths = [np.linalg.norm(triangle_points[(i+1)%3] - triangle_points[i]) 
                          for i in range(3)]
            aspect_ratios.append(max(edge_lengths) / min(edge_lengths))
            
            # Quality metric
            qualities.append(self._triangle_quality(triangle_points))
        
        return {
            'min_angle': np.min(angles),
            'max_angle': np.max(angles),
            'avg_aspect_ratio': np.mean(aspect_ratios),
            'avg_quality': np.mean(qualities)
        }
    
    def _triangle_angles(self, triangle_points):
        """Same as circular generator."""
        p1, p2, p3 = triangle_points
        
        v1, v2, v3 = p2 - p1, p3 - p1, p3 - p2
        
        def safe_angle(va, vb):
            cos_angle = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle)
        
        angle1 = safe_angle(v1, v2)
        angle2 = safe_angle(-v1, v3)
        angle3 = np.pi - angle1 - angle2
        
        return [np.degrees(angle) for angle in [angle1, angle2, angle3]]
    
    def _triangle_quality(self, triangle_points):
        """Same as circular generator."""
        p1, p2, p3 = triangle_points
        
        # Edge lengths
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        
        # Area using cross product
        area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
        
        if area < 1e-12:
            return 0.0
        
        # Quality metric: ratio of inscribed to circumscribed circle radius
        perimeter = a + b + c
        return 4 * np.sqrt(3) * area / (perimeter**2)
    
    def _print_mesh_info(self, n_nodes, n_elem, boundary_nodes, quality_metrics):
        """Print rectangular mesh info."""
        print(f"✅ Professional Rectangular Triangle Mesh Generated:")
        print(f"   📊 Nodes: {n_nodes}, Elements: {n_elem}, Boundary nodes: {len(boundary_nodes)}")
        print(f"   📐 Min angle: {quality_metrics['min_angle']:.1f}° (target: {self.min_angle}°)")
        print(f"   📐 Max angle: {quality_metrics['max_angle']:.1f}°")
        print(f"   📏 Aspect ratio (avg): {quality_metrics['avg_aspect_ratio']:.2f}")
        print(f"   ⭐ Quality metric (avg): {quality_metrics['avg_quality']:.3f}")
        print(f"   🎯 Target edge length: {self.target_edge_length}")
        print(f"   📦 Domain: {self.width} × {self.height}")
    
    def plot_mesh(self, x, y, connectivity, boundary_nodes, save_png=True):
        """Plot rectangular mesh."""
        print("🎨 Generating rectangular mesh visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left plot: Basic mesh
        ax1.triplot(x, y, connectivity, 'b-', lw=0.3, alpha=0.7)
        ax1.scatter(x, y, c='lightblue', s=8, alpha=0.6, label='Interior Nodes')
        
        # Detect different boundary types for coloring
        tolerance = self.target_edge_length * 0.2
        left_mask = np.abs(x[boundary_nodes] - self.x_min) < tolerance
        right_mask = np.abs(x[boundary_nodes] - self.x_max) < tolerance
        
        if np.any(left_mask):
            ax1.scatter(x[boundary_nodes][left_mask], y[boundary_nodes][left_mask], 
                       c='red', s=40, label='Left BC', zorder=5, marker='s')
        if np.any(right_mask):
            ax1.scatter(x[boundary_nodes][right_mask], y[boundary_nodes][right_mask], 
                       c='darkred', s=40, label='Right BC', zorder=5, marker='s')
        
        # Plot rectangular boundary
        rect_x = [self.x_min, self.x_max, self.x_max, self.x_min, self.x_min]
        rect_y = [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min]
        ax1.plot(rect_x, rect_y, 'k--', lw=2, alpha=0.8, label='Domain Boundary')
        
        ax1.set_title(f'Rectangular Triangle Mesh\n{len(connectivity)} elements, {len(x)} nodes')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Quality visualization
        points = np.column_stack([x, y])
        qualities = [self._triangle_quality(points[tri]) for tri in connectivity]
        
        im = ax2.tripcolor(x, y, connectivity, qualities, shading='flat', 
                          cmap='RdYlGn', vmin=0, vmax=1)
        ax2.triplot(x, y, connectivity, 'k-', lw=0.1, alpha=0.3)
        
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Triangle Quality (0=poor, 1=perfect)')
        
        ax2.plot(rect_x, rect_y, 'k--', lw=2, alpha=0.8)
        
        ax2.set_title(f'Quality: Avg={np.mean(qualities):.3f}, Min={np.min(qualities):.3f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_png:
            plt.savefig('triangle_rectangular_mesh.png', dpi=150, bbox_inches='tight')
            print("✅ Saved mesh plot: triangle_rectangular_mesh.png")
        
        plt.show()
    
    def _fallback_simple_rectangular_mesh(self):
        """Fallback rectangular mesh if Triangle unavailable."""
        print("🔄 Using fallback rectangular mesh...")
        
        # Simple grid approach
        x_range = np.arange(self.x_min, self.x_max + self.target_edge_length/2, self.target_edge_length)
        y_range = np.arange(self.y_min, self.y_max + self.target_edge_length/2, self.target_edge_length)
        X, Y = np.meshgrid(x_range, y_range)
        points = np.vstack((X.ravel(), Y.ravel())).T
        
        # Simple Delaunay
        from scipy.spatial import Delaunay
        delaunay = Delaunay(points)
        
        # Filter triangles
        is_inside = lambda tri: (self.x_min < points[tri].mean(axis=0)[0] < self.x_max and
                                self.y_min < points[tri].mean(axis=0)[1] < self.y_max)
        simplices = np.array([tri for tri in delaunay.simplices if is_inside(tri)])
        
        # Remap
        used_nodes = np.unique(simplices)
        new_index = -np.ones(len(points), dtype=int)
        new_index[used_nodes] = np.arange(len(used_nodes))
        
        final_points = points[used_nodes]
        connectivity = new_index[simplices]
        boundary_nodes = self._detect_boundary_nodes(connectivity, final_points)
        
        print(f"✅ Fallback rectangular mesh: {len(final_points)} nodes, {len(connectivity)} triangles")
        
        return final_points[:, 0], final_points[:, 1], connectivity, boundary_nodes, len(final_points), len(connectivity)


# Simple mesh generator (unchanged for backward compatibility)
class CircularMeshGenerator:
    """Simple circular mesh generator with progress tracking (fallback option)."""
    
    def __init__(self, radius=1.0, resolution=0.05):
        self.radius = radius
        self.resolution = resolution
        
    def generate_mesh(self):
        """Generate basic triangular mesh in circle with progress tracking."""
        print("🚀 Starting simple mesh generation...")
        start_time = time.time()
        
        # Create uniform grid
        print_progress(10, "Creating grid points")
        margin = self.resolution
        x = np.arange(-self.radius - margin, self.radius + margin, self.resolution)
        y = np.arange(-self.radius - margin, self.radius + margin, self.resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.vstack((X.ravel(), Y.ravel())).T

        # Keep points inside circle
        print_progress(30, "Filtering interior points")
        r2 = np.sum(grid_points**2, axis=1)
        points = grid_points[r2 < (self.radius * 1.05)**2]

        # Delaunay triangulation and filtering
        print_progress(50, "Creating triangulation")
        from scipy.spatial import Delaunay
        delaunay = Delaunay(points)
        
        print_progress(70, "Filtering triangles")
        is_inside = lambda tri: np.linalg.norm(points[tri].mean(axis=0)) < self.radius * 0.999
        simplices = np.array([tri for tri in delaunay.simplices if is_inside(tri)])

        # Filter by area
        print_progress(85, "Quality filtering")
        areas = np.array([0.5 * abs(points[tri[0], 0] * (points[tri[1], 1] - points[tri[2], 1]) +
                                   points[tri[1], 0] * (points[tri[2], 1] - points[tri[0], 1]) +
                                   points[tri[2], 0] * (points[tri[0], 1] - points[tri[1], 1]))
                         for tri in simplices])
        
        median_area = np.median(areas)
        tolerance = 0.2 * median_area
        mask = (areas > median_area - tolerance) & (areas < median_area + tolerance)
        simplices = simplices[mask]

        # Remap indices
        print_progress(95, "Finalizing")
        used_nodes = np.unique(simplices)
        new_index = -np.ones(len(points), dtype=int)
        new_index[used_nodes] = np.arange(len(used_nodes))
        
        final_points = points[used_nodes]
        connectivity = new_index[simplices]
        boundary_nodes = self._detect_boundary_nodes(connectivity)
        
        print_progress(100, "Complete")
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds")
        print(f"✅ Mesh: {len(final_points)} nodes, {len(connectivity)} triangles, "
              f"{len(boundary_nodes)} boundary nodes.")
              
        return final_points[:, 0], final_points[:, 1], connectivity, boundary_nodes, len(final_points), len(connectivity)
    
    def _detect_boundary_nodes(self, connectivity):
        """Detect boundary nodes."""
        edge_counter = Counter()
        for tri in connectivity:
            for i in range(3):
                edge = tuple(sorted((tri[i], tri[(i+1)%3])))
                edge_counter[edge] += 1
        
        boundary_nodes = {node for (a, b), count in edge_counter.items() 
                         if count == 1 for node in (a, b)}
        return np.array(sorted(boundary_nodes))
    
    def plot_mesh(self, x, y, connectivity, boundary_nodes, save_png=True):
        """Plot basic mesh."""
        print("🎨 Creating visualization...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        ax.triplot(x, y, connectivity, 'b-', lw=0.5, alpha=0.7)
        ax.scatter(x, y, c='red', s=10, alpha=0.6, label='Interior Nodes')
        ax.scatter(x[boundary_nodes], y[boundary_nodes], 
                  c='green', s=40, label='Boundary Nodes', zorder=5)
        
        # Circle boundary
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(self.radius * np.cos(theta), self.radius * np.sin(theta), 
               'k--', lw=2, alpha=0.8, label='Domain Boundary')
        
        ax.set_title(f'Simple Circular Mesh\n{len(connectivity)} elements, {len(x)} nodes')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_png:
            plt.savefig('simple_circular_mesh.png', dpi=120, bbox_inches='tight')
            print("✅ Saved mesh plot: simple_circular_mesh.png")
        
        plt.show()


class VoronoiGrainGenerator:
    """Generate Voronoi grain structure for existing mesh."""
    
    def __init__(self, n_grains=10, seed=42):
        """
        Initialize grain generator.
        
        Parameters:
        -----------
        n_grains : int
            Number of grain seeds to generate
        seed : int
            Random seed for reproducibility
        """
        self.n_grains = n_grains
        self.seed = seed
        self.grain_seeds = None
        self.element_grain_ids = None
        self.node_grain_ids = None  # Track grain assignment for each node
        
    def generate_grains(self, x_coords, y_coords, triangles):
        """
        Generate Voronoi grain structure for given mesh.
        
        Parameters:
        -----------
        x_coords : array
            Node x-coordinates
        y_coords : array  
            Node y-coordinates
        triangles : array
            Element connectivity (n_elem x 3)
            
        Returns:
        --------
        element_grain_ids : array
            Grain ID for each element
        grain_seeds : array
            Grain seed coordinates (n_grains x 2)
        element_angles : array
            Crystal orientation angle in degrees for each element
        grain_boundary_flags : array
            Flag indicating element type:
            0 = grain interior
            1 = grain boundary (2 grains meet)
            2 = triple junction (3+ grains meet)
        """
        
        print(f"🔬 Generating {self.n_grains} Voronoi grains...")
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        # Get domain bounds from mesh
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        width = x_max - x_min
        height = y_max - y_min
        
        print(f"   Domain: {width:.1f} x {height:.1f}")
        
        # Generate grain seeds within domain
        seed_x = np.random.uniform(x_min, x_max, self.n_grains)
        seed_y = np.random.uniform(y_min, y_max, self.n_grains)
        self.grain_seeds = np.column_stack((seed_x, seed_y))
        
        # Generate random crystal orientations for each grain (0-90 degrees)
        self.grain_orientations = np.random.uniform(0, 90, self.n_grains)
        
        # Add boundary seeds for better grain shapes at edges
        margin = max(width, height) * 0.2
        n_boundary = self.n_grains // 3
        
        boundary_x = np.random.uniform(x_min - margin, x_max + margin, n_boundary)
        boundary_y = np.random.uniform(y_min - margin, y_max + margin, n_boundary)
        boundary_seeds = np.column_stack((boundary_x, boundary_y))
        
        # Combine interior and boundary seeds
        all_seeds = np.vstack((self.grain_seeds, boundary_seeds))
        
        print(f"   Total seeds: {len(all_seeds)} ({self.n_grains} interior + {n_boundary} boundary)")
        
        # First assign grain IDs to nodes based on Voronoi assignment
        vertices = np.column_stack((x_coords, y_coords))
        n_nodes = len(vertices)
        n_elements = len(triangles)
        
        print("   Computing node-grain assignments...")
        self.node_grain_ids = np.zeros(n_nodes, dtype=int)
        
        for node_id, node_pos in enumerate(vertices):
            # Find nearest grain seed (only consider interior seeds for assignment)
            distances = np.linalg.norm(self.grain_seeds - node_pos, axis=1)
            nearest_grain = np.argmin(distances)
            self.node_grain_ids[node_id] = nearest_grain
        
        # Now assign element grain IDs and detect grain boundaries and triple junctions
        self.element_grain_ids = np.zeros(n_elements, dtype=int)
        self.element_angles = np.zeros(n_elements)
        self.grain_boundary_flags = np.zeros(n_elements, dtype=int)
        
        print("   Computing element-grain assignments, grain boundaries, and triple junctions...")
        
        n_boundary_elements = 0
        n_triple_junction_elements = 0
        
        for elem_id, triangle in enumerate(triangles):
            # Get grain IDs of the three nodes of this element
            node_grains = self.node_grain_ids[triangle]
            
            # Check if all nodes belong to the same grain
            unique_grains = np.unique(node_grains)
            n_unique_grains = len(unique_grains)
            
            if n_unique_grains == 1:
                # All nodes belong to same grain - interior element
                grain_id = unique_grains[0]
                self.element_grain_ids[elem_id] = grain_id
                self.element_angles[elem_id] = self.grain_orientations[grain_id]
                self.grain_boundary_flags[elem_id] = 0  # Interior
                
            elif n_unique_grains == 2:
                # Two grains meet - grain boundary element
                n_boundary_elements += 1
                
                # Assign to the most frequent grain among nodes
                grain_counts = np.bincount(node_grains, minlength=self.n_grains)
                dominant_grain = np.argmax(grain_counts)
                self.element_grain_ids[elem_id] = dominant_grain
                
                # Calculate misorientation angle between the two grains
                grain1, grain2 = unique_grains[0], unique_grains[1]
                angle1 = self.grain_orientations[grain1]
                angle2 = self.grain_orientations[grain2]
                
                # Calculate misorientation angle (smallest angle between orientations)
                misorientation = abs(angle1 - angle2)
                misorientation = min(misorientation, 180 - misorientation)
                self.element_angles[elem_id] = misorientation
                
                # Flag as grain boundary
                self.grain_boundary_flags[elem_id] = 1  # Grain boundary
                
            else:  # n_unique_grains >= 3
                # Three or more grains meet - triple junction (or higher order junction)
                n_triple_junction_elements += 1
                
                # Assign to the most frequent grain among nodes
                grain_counts = np.bincount(node_grains, minlength=self.n_grains)
                dominant_grain = np.argmax(grain_counts)
                self.element_grain_ids[elem_id] = dominant_grain
                
                # For triple junctions, calculate average misorientation
                involved_orientations = self.grain_orientations[unique_grains]
                
                # Calculate pairwise misorientations and take average
                misorientations = []
                for i in range(len(unique_grains)):
                    for j in range(i+1, len(unique_grains)):
                        angle1 = self.grain_orientations[unique_grains[i]]
                        angle2 = self.grain_orientations[unique_grains[j]]
                        misorientation = abs(angle1 - angle2)
                        misorientation = min(misorientation, 180 - misorientation)
                        misorientations.append(misorientation)
                
                self.element_angles[elem_id] = np.mean(misorientations) if misorientations else 0
                
                # Flag as triple junction
                self.grain_boundary_flags[elem_id] = 2  # Triple junction
        
        # Verify all grains are used
        used_grains = np.unique(self.element_grain_ids)
        print(f"   ✅ Generated {len(used_grains)} grains covering {n_elements} elements")
        print(f"   🔄 Crystal orientations: {np.min(self.grain_orientations):.1f}° to {np.max(self.grain_orientations):.1f}°")
        print(f"   🎯 Grain boundary elements: {n_boundary_elements} ({n_boundary_elements/n_elements*100:.1f}%)")
        print(f"   🔺 Triple junction elements: {n_triple_junction_elements} ({n_triple_junction_elements/n_elements*100:.1f}%)")
        
        return self.element_grain_ids, self.grain_seeds, self.element_angles, self.grain_boundary_flags
    
    def get_grain_info(self):
        """Return grain structure information."""
        if self.element_grain_ids is None:
            return None
            
        n_grains_used = len(np.unique(self.element_grain_ids))
        elements_per_grain = len(self.element_grain_ids) / n_grains_used
        n_interior_elements = np.sum(self.grain_boundary_flags == 0) if self.grain_boundary_flags is not None else 0
        n_boundary_elements = np.sum(self.grain_boundary_flags == 1) if self.grain_boundary_flags is not None else 0
        n_triple_junction_elements = np.sum(self.grain_boundary_flags == 2) if self.grain_boundary_flags is not None else 0
        
        return {
            'n_grains_generated': self.n_grains,
            'n_grains_used': n_grains_used,
            'elements_per_grain': elements_per_grain,
            'total_elements': len(self.element_grain_ids),
            'interior_elements': n_interior_elements,
            'boundary_elements': n_boundary_elements,
            'triple_junction_elements': n_triple_junction_elements,
            'interior_percentage': n_interior_elements / len(self.element_grain_ids) * 100,
            'boundary_percentage': n_boundary_elements / len(self.element_grain_ids) * 100,
            'triple_junction_percentage': n_triple_junction_elements / len(self.element_grain_ids) * 100
        }
    
    def plot_grain_structure(self, x_coords, y_coords, triangles, save_fig=True, show_boundaries=True):
        """Visualize the generated grain structure."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        if self.element_grain_ids is None:
            print("No grain structure generated yet!")
            return
            
        if show_boundaries and self.grain_boundary_flags is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        vertices = np.column_stack((x_coords, y_coords))
        
        # Plot 1: Grain structure
        for elem_id, triangle in enumerate(triangles):
            element_nodes = vertices[triangle]
            grain_id = self.element_grain_ids[elem_id]
            
            # Color by grain ID
            color = plt.cm.tab20(grain_id % 20)
            triangle_patch = patches.Polygon(element_nodes, 
                                           facecolor=color, 
                                           edgecolor='black', 
                                           linewidth=0.05, 
                                           alpha=0.8)
            ax1.add_patch(triangle_patch)
        
        # Plot grain seeds
        ax1.scatter(self.grain_seeds[:, 0], self.grain_seeds[:, 1], 
                  c='red', s=100, marker='x', linewidths=3, 
                  label='Grain Seeds', zorder=10)
        
        # Add grain numbers
        for i, seed in enumerate(self.grain_seeds):
            ax1.annotate(f'{i}', (seed[0], seed[1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        ax1.set_xlim(np.min(x_coords), np.max(x_coords))
        ax1.set_ylim(np.min(y_coords), np.max(y_coords))
        ax1.set_aspect('equal')
        ax1.set_title(f'Voronoi Grain Structure ({self.n_grains} grains)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Grain boundaries and triple junctions (if requested and available)
        if show_boundaries and self.grain_boundary_flags is not None:
            for elem_id, triangle in enumerate(triangles):
                element_nodes = vertices[triangle]
                flag = self.grain_boundary_flags[elem_id]
                
                if flag == 0:
                    # Interior elements in light gray
                    color = 'lightgray'
                    alpha = 0.3
                    linewidth = 0.05
                elif flag == 1:
                    # Grain boundary elements in red
                    color = 'red'
                    alpha = 0.8
                    linewidth = 0.2
                else:  # flag == 2
                    # Triple junction elements in blue
                    color = 'blue'
                    alpha = 0.9
                    linewidth = 0.3
                
                triangle_patch = patches.Polygon(element_nodes, 
                                               facecolor=color, 
                                               edgecolor='black', 
                                               linewidth=linewidth, 
                                               alpha=alpha)
                ax2.add_patch(triangle_patch)
            
            ax2.set_xlim(np.min(x_coords), np.max(x_coords))
            ax2.set_ylim(np.min(y_coords), np.max(y_coords))
            ax2.set_aspect('equal')
            ax2.set_title(f'Grain Boundaries & Triple Junctions\n(Gray=Interior, Red=GB, Blue=TJ)')
            ax2.grid(True, alpha=0.3)
        
        if save_fig:
            plt.savefig('voronoi_grains_with_triple_junctions.png', dpi=300, bbox_inches='tight')
            print("   💾 Grain structure saved to 'voronoi_grains_with_triple_junctions.png'")
        
        #plt.show()
        
        # Print statistics
        info = self.get_grain_info()
        if info:
            print(f"\n📊 Grain Statistics:")
            print(f"   Grains generated: {info['n_grains_generated']}")
            print(f"   Grains used: {info['n_grains_used']}")
            print(f"   Elements per grain: {info['elements_per_grain']:.1f}")
            print(f"   Total elements: {info['total_elements']}")
            print(f"   Interior elements: {info['interior_elements']} ({info['interior_percentage']:.1f}%)")
            print(f"   Boundary elements: {info['boundary_elements']} ({info['boundary_percentage']:.1f}%)")
            print(f"   Triple junction elements: {info['triple_junction_elements']} ({info['triple_junction_percentage']:.1f}%)")
    
    def get_grain_boundary_elements(self):
        """Return indices of elements that are on grain boundaries."""
        if self.grain_boundary_flags is None:
            return None
        return np.where(self.grain_boundary_flags == 1)[0]
    
    def get_triple_junction_elements(self):
        """Return indices of elements that are at triple junctions."""
        if self.grain_boundary_flags is None:
            return None
        return np.where(self.grain_boundary_flags == 2)[0]
    
    def get_interior_elements(self):
        """Return indices of elements that are in grain interiors."""
        if self.grain_boundary_flags is None:
            return None
        return np.where(self.grain_boundary_flags == 0)[0]
    
    def get_element_type_counts(self):
        """Return counts of each element type."""
        if self.grain_boundary_flags is None:
            return None
            
        unique, counts = np.unique(self.grain_boundary_flags, return_counts=True)
        result = {}
        for flag, count in zip(unique, counts):
            if flag == 0:
                result['interior'] = count
            elif flag == 1:
                result['grain_boundary'] = count
            elif flag == 2:
                result['triple_junction'] = count
        
        return result
    
    def get_misorientation_stats(self):
        """Return statistics about grain boundary and triple junction misorientations."""
        if self.grain_boundary_flags is None or self.element_angles is None:
            return None
            
        # Separate statistics for grain boundaries and triple junctions
        gb_elements = self.grain_boundary_flags == 1
        tj_elements = self.grain_boundary_flags == 2
        
        gb_angles = self.element_angles[gb_elements]
        tj_angles = self.element_angles[tj_elements]
        
        stats = {}
        
        if len(gb_angles) > 0:
            stats['grain_boundary'] = {
                'mean_misorientation': np.mean(gb_angles),
                'std_misorientation': np.std(gb_angles),
                'min_misorientation': np.min(gb_angles),
                'max_misorientation': np.max(gb_angles),
                'n_elements': len(gb_angles)
            }
        
        if len(tj_angles) > 0:
            stats['triple_junction'] = {
                'mean_misorientation': np.mean(tj_angles),
                'std_misorientation': np.std(tj_angles),
                'min_misorientation': np.min(tj_angles),
                'max_misorientation': np.max(tj_angles),
                'n_elements': len(tj_angles)
            }
        
        return stats



import numpy as np
from scipy.spatial.distance import cdist

class GcDistribution:
    """
    Generate random Gc distribution for elements with optional microstructure effects.
    Now supports both uncorrelated and spatially correlated noise.
    """
    
    def __init__(self, mean_gc=1.0, std_gc=0.1, seed=42, 
                 gb_factor=0.5, tj_factor=0.8, apply_microstructure=True,
                 noise_type='uncorrelated', correlation_length=3.0):
        """
        Initialize Gc distribution generator.
        
        Parameters:
        -----------
        mean_gc : float
            Mean value for Gc distribution
        std_gc : float
            Standard deviation for Gc distribution
        seed : int
            Random seed for reproducibility
        gb_factor : float
            Multiplicative factor for grain boundaries
        tj_factor : float
            Multiplicative factor for triple junctions
        apply_microstructure : bool
            Whether to apply microstructure-based modifications
        noise_type : str
            Type of noise to generate. Options:
            - 'uncorrelated': Original random normal distribution
            - 'correlated': Spatially correlated noise (requires coordinates)
            - 'perlin': Perlin-like smooth noise
            - 'fractal': Fractal/self-similar noise
            - 'gradient': Linear gradient with fluctuations
            - 'clusters': Clustered defect regions
            - 'bands': Banded/layered structure
            - 'auto': Automatically choose based on available inputs
        correlation_length : float
            Spatial correlation length for correlated noise (in mesh units)
        """
        self.mean_gc = mean_gc
        self.std_gc = std_gc
        self.seed = seed
        self.gb_factor = gb_factor
        self.tj_factor = tj_factor
        self.apply_microstructure = apply_microstructure
        self.noise_type = noise_type
        self.correlation_length = correlation_length
        
        # Validate noise type
        valid_types = ['uncorrelated', 'correlated', 'perlin', 'fractal', 'gradient', 'clusters', 'bands', 'auto']
        if noise_type not in valid_types:
            raise ValueError(f"noise_type must be one of {valid_types}, got '{noise_type}'")
    
    def set_coordinates(self, coordinates):
        """
        Set element coordinates for correlated noise generation.
        Call this before generate_element_gc() if you want correlated noise.
        
        Parameters:
        -----------
        coordinates : array, shape (n_elements, 2)
            Element centroid coordinates
        """
        self.coordinates = coordinates
        print(f"✅ Coordinates set for {len(coordinates)} elements")
    
    def generate_uncorrelated_gc(self, n_elements):
        """
        Generate uncorrelated (original) Gc distribution.
        
        Parameters:
        -----------
        n_elements : int
            Number of elements
            
        Returns:
        --------
        element_gc : array
            Gc values for each element
        """
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        # Generate base random distribution (original method)
        element_gc = np.random.normal(self.mean_gc, self.std_gc, n_elements)
        
        # Ensure positive values
        element_gc = np.maximum(element_gc, 0.01)  # Minimum Gc of 0.01
        
        return element_gc
    
    def generate_correlated_gc(self, coordinates):
        """
        Generate spatially correlated Gc distribution.
        
        Parameters:
        -----------
        coordinates : array, shape (n_elements, 2)
            Element centroid coordinates
            
        Returns:
        --------
        element_gc : array
            Spatially correlated Gc values
        """
        np.random.seed(self.seed)
        n_elements = len(coordinates)
        
        # Generate uncorrelated random field
        random_field = np.random.normal(0, 1, n_elements)
        
        # Create spatial correlation matrix
        distances = cdist(coordinates, coordinates)
        correlation_matrix = np.exp(-0.5 * (distances / self.correlation_length)**2)
        
        try:
            # Apply correlation using Cholesky decomposition
            L = np.linalg.cholesky(correlation_matrix + 1e-6 * np.eye(n_elements))
            correlated_field = L @ random_field
        except np.linalg.LinAlgError:
            print("⚠️  Cholesky decomposition failed, using eigendecomposition fallback")
            # Fallback to eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 1e-6)
            correlated_field = eigenvecs @ (np.sqrt(eigenvals) * (eigenvecs.T @ random_field))
        
        # Scale and shift to desired mean and std
        correlated_field = correlated_field / np.std(correlated_field) * self.std_gc
        element_gc = self.mean_gc + correlated_field
        
        # Ensure positive values
        element_gc = np.maximum(element_gc, 0.01)
        
        return element_gc
    
    def generate_perlin_gc(self, coordinates):
        """
        Generate Perlin-like noise for smooth, natural-looking variations.
        Great for modeling smooth processing variations.
        
        Parameters:
        -----------
        coordinates : array, shape (n_elements, 2)
            Element centroid coordinates
            
        Returns:
        --------
        element_gc : array
            Perlin noise-based Gc values
        """
        np.random.seed(self.seed)
        n_elements = len(coordinates)
        
        # Normalize coordinates to [0, 1] range
        coord_min = np.min(coordinates, axis=0)
        coord_max = np.max(coordinates, axis=0)
        coord_range = coord_max - coord_min
        coords_norm = (coordinates - coord_min) / (coord_range + 1e-10)
        
        # Generate multiple octaves of noise
        noise = np.zeros(n_elements)
        amplitude = 1.0
        frequency = 1.0 / self.correlation_length
        
        # Multiple frequency components (octaves)
        for octave in range(4):
            # Random phases for this octave
            phase_x = np.random.uniform(0, 2*np.pi)
            phase_y = np.random.uniform(0, 2*np.pi)
            
            # Sinusoidal noise at this frequency
            noise_x = np.sin(2*np.pi * frequency * coords_norm[:, 0] + phase_x)
            noise_y = np.cos(2*np.pi * frequency * coords_norm[:, 1] + phase_y)
            
            # Combine with cross terms for more complexity
            octave_noise = (noise_x + noise_y + 
                          0.5 * np.sin(2*np.pi * frequency * 
                                     (coords_norm[:, 0] + coords_norm[:, 1]) + phase_x))
            
            noise += amplitude * octave_noise
            
            # Prepare for next octave
            amplitude *= 0.5  # Reduce amplitude
            frequency *= 2.0  # Increase frequency
        
        # Normalize and scale
        noise = noise / np.std(noise) * self.std_gc
        element_gc = self.mean_gc + noise
        
        # Ensure positive values
        element_gc = np.maximum(element_gc, 0.01)
        
        return element_gc
    
    def generate_fractal_gc(self, coordinates, hurst_exponent=0.7):
        """
        Generate fractal noise with power-law correlations.
        Models self-similar material structures.
        
        Parameters:
        -----------
        coordinates : array, shape (n_elements, 2)
            Element centroid coordinates
        hurst_exponent : float (0 < H < 1)
            Controls roughness: H≈0.2 (very rough), H≈0.8 (smooth)
            
        Returns:
        --------
        element_gc : array
            Fractal noise-based Gc values
        """
        np.random.seed(self.seed)
        n_elements = len(coordinates)
        
        # Simple fractal approximation using distance-based weighting
        noise = np.zeros(n_elements)
        
        # Generate random "source points" for fractal features
        n_sources = max(5, int(np.sqrt(n_elements) / 3))
        coord_min = np.min(coordinates, axis=0)
        coord_max = np.max(coordinates, axis=0)
        sources = np.random.uniform(coord_min, coord_max, size=(n_sources, 2))
        source_strengths = np.random.normal(0, 1, n_sources)
        
        for i, coord in enumerate(coordinates):
            fractal_value = 0
            for j, source in enumerate(sources):
                distance = np.linalg.norm(coord - source) + 1e-6
                # Power-law decay with Hurst exponent
                contribution = source_strengths[j] * (distance ** (-hurst_exponent))
                fractal_value += contribution
            
            noise[i] = fractal_value
        
        # Normalize and scale
        noise = noise / np.std(noise) * self.std_gc
        element_gc = self.mean_gc + noise
        
        # Ensure positive values
        element_gc = np.maximum(element_gc, 0.01)
        
        return element_gc
    
    def generate_gradient_gc(self, coordinates, gradient_direction='random', 
                           gradient_strength=2.0):
        """
        Generate linear/thermal gradient with local fluctuations.
        Models effects from processing gradients.
        
        Parameters:
        -----------
        coordinates : array, shape (n_elements, 2)
            Element centroid coordinates
        gradient_direction : str or float
            'random', 'x', 'y', 'radial', or angle in radians
        gradient_strength : float
            Strength of the gradient effect
            
        Returns:
        --------
        element_gc : array
            Gradient-based Gc values
        """
        np.random.seed(self.seed)
        n_elements = len(coordinates)
        
        # Normalize coordinates
        coord_min = np.min(coordinates, axis=0)
        coord_max = np.max(coordinates, axis=0)
        coord_range = coord_max - coord_min
        coords_norm = (coordinates - coord_min) / (coord_range + 1e-10)
        
        # Determine gradient direction
        if gradient_direction == 'random':
            angle = np.random.uniform(0, 2*np.pi)
        elif gradient_direction == 'x':
            angle = 0
        elif gradient_direction == 'y':
            angle = np.pi/2
        elif gradient_direction == 'radial':
            # Radial gradient from center
            center = np.mean(coordinates, axis=0)
            radial_dist = np.linalg.norm(coordinates - center, axis=1)
            radial_dist_norm = radial_dist / np.max(radial_dist)
            
            gradient_noise = gradient_strength * self.std_gc * radial_dist_norm
            local_fluctuations = np.random.normal(0, self.std_gc * 0.3, n_elements)
            
            element_gc = self.mean_gc + gradient_noise + local_fluctuations
            element_gc = np.maximum(element_gc, 0.01)
            return element_gc
        else:
            angle = float(gradient_direction)
        
        # Linear gradient
        gradient_vector = np.array([np.cos(angle), np.sin(angle)])
        projected_coords = coords_norm @ gradient_vector
        
        # Apply gradient with local fluctuations
        gradient_noise = gradient_strength * self.std_gc * projected_coords
        local_fluctuations = np.random.normal(0, self.std_gc * 0.5, n_elements)
        
        element_gc = self.mean_gc + gradient_noise + local_fluctuations
        
        # Ensure positive values
        element_gc = np.maximum(element_gc, 0.01)
        
        return element_gc
    
    def generate_clusters_gc(self, coordinates, n_clusters=8, cluster_strength=3.0):
        """
        Generate clustered defects/weak regions.
        Models localized material weakening from defects.
        
        Parameters:
        -----------
        coordinates : array, shape (n_elements, 2)
            Element centroid coordinates
        n_clusters : int
            Number of defect clusters
        cluster_strength : float
            Strength of clustering effect
            
        Returns:
        --------
        element_gc : array
            Cluster-based Gc values
        """
        np.random.seed(self.seed)
        n_elements = len(coordinates)
        
        # Generate cluster centers
        coord_min = np.min(coordinates, axis=0)
        coord_max = np.max(coordinates, axis=0)
        cluster_centers = np.random.uniform(coord_min, coord_max, size=(n_clusters, 2))
        
        # Random cluster properties
        cluster_strengths = np.random.normal(0, cluster_strength * self.std_gc, n_clusters)
        cluster_sizes = np.random.uniform(self.correlation_length * 0.5, 
                                        self.correlation_length * 2.0, n_clusters)
        
        # Calculate influence of each cluster
        noise = np.zeros(n_elements)
        for i, coord in enumerate(coordinates):
            total_influence = 0
            for j in range(n_clusters):
                distance = np.linalg.norm(coord - cluster_centers[j])
                # Gaussian influence
                influence = cluster_strengths[j] * np.exp(-0.5 * (distance / cluster_sizes[j])**2)
                total_influence += influence
            
            noise[i] = total_influence
        
        # Add background noise
        background_noise = np.random.normal(0, self.std_gc * 0.5, n_elements)
        
        element_gc = self.mean_gc + noise + background_noise
        
        # Ensure positive values
        element_gc = np.maximum(element_gc, 0.01)
        
        return element_gc
    
    def generate_bands_gc(self, coordinates, band_direction='random', 
                         band_spacing=None, band_contrast=2.0):
        """
        Generate banded/layered structure variations.
        Models effects from layered deposition or rolling.
        
        Parameters:
        -----------
        coordinates : array, shape (n_elements, 2)
            Element centroid coordinates
        band_direction : str or float
            'random', 'x', 'y', or angle in radians
        band_spacing : float, optional
            Spacing between bands. If None, uses correlation_length
        band_contrast : float
            Contrast between bands
            
        Returns:
        --------
        element_gc : array
            Band-based Gc values
        """
        np.random.seed(self.seed)
        n_elements = len(coordinates)
        
        # Determine band direction
        if band_direction == 'random':
            angle = np.random.uniform(0, 2*np.pi)
        elif band_direction == 'x':
            angle = 0
        elif band_direction == 'y':
            angle = np.pi/2
        else:
            angle = float(band_direction)
        
        # Band spacing
        if band_spacing is None:
            band_spacing = self.correlation_length * 2.0
        
        # Normalize coordinates
        coord_min = np.min(coordinates, axis=0)
        coord_max = np.max(coordinates, axis=0)
        coord_range = coord_max - coord_min
        coords_norm = (coordinates - coord_min) / (coord_range + 1e-10)
        
        # Project coordinates onto band direction
        band_vector = np.array([np.cos(angle), np.sin(angle)])
        projected_coords = coords_norm @ band_vector
        
        # Create banded pattern using sine wave
        domain_size = np.max(projected_coords) - np.min(projected_coords)
        frequency = domain_size / band_spacing
        
        band_pattern = np.sin(2 * np.pi * frequency * projected_coords)
        
        # Add some randomness to band strength
        random_phase = np.random.uniform(0, 2*np.pi)
        band_pattern += 0.3 * np.sin(2 * np.pi * frequency * projected_coords + random_phase)
        
        # Scale and add noise
        band_noise = band_contrast * self.std_gc * band_pattern
        local_noise = np.random.normal(0, self.std_gc * 0.3, n_elements)
        
        element_gc = self.mean_gc + band_noise + local_noise
        
        # Ensure positive values
        element_gc = np.maximum(element_gc, 0.01)
        
        return element_gc
    
    def generate_element_gc(self, n_elements, grain_boundary_flags=None):
        """
        Generate Gc values for all elements using the specified noise type.
        
        Parameters:
        -----------
        n_elements : int
            Number of elements in the mesh
        grain_boundary_flags : array, optional
            Element flags: 0=interior, 1=grain_boundary, 2=triple_junction
            
        Returns:
        --------
        element_gc : array
            Gc value for each element
        """
        
        # Check if coordinates are available for correlated noise
        coordinates = getattr(self, 'coordinates', None)
        
        # Determine which noise generation method to use
        if self.noise_type == 'auto':
            # Auto-select based on available inputs
            if coordinates is not None:
                actual_noise_type = 'correlated'
                print("🔄 Auto-selected correlated noise (coordinates available)")
            else:
                actual_noise_type = 'uncorrelated'
                print("🔄 Auto-selected uncorrelated noise (no coordinates)")
        else:
            actual_noise_type = self.noise_type
        
        # Generate base Gc distribution based on selected type
        if actual_noise_type == 'uncorrelated':
            element_gc = self.generate_uncorrelated_gc(n_elements)
            print(f"✅ Generated uncorrelated Gc distribution")
            
        elif actual_noise_type == 'correlated':
            if coordinates is None:
                print("⚠️  Correlated noise requires coordinates. Use set_coordinates() first or falling back to uncorrelated.")
                element_gc = self.generate_uncorrelated_gc(n_elements)
            else:
                element_gc = self.generate_correlated_gc(coordinates)
                print(f"✅ Generated correlated Gc distribution (correlation_length={self.correlation_length})")
        
        elif actual_noise_type == 'perlin':
            if coordinates is None:
                print("⚠️  Perlin noise requires coordinates. Falling back to uncorrelated.")
                element_gc = self.generate_uncorrelated_gc(n_elements)
            else:
                element_gc = self.generate_perlin_gc(coordinates)
                print(f"✅ Generated Perlin noise Gc distribution")
        
        elif actual_noise_type == 'fractal':
            if coordinates is None:
                print("⚠️  Fractal noise requires coordinates. Falling back to uncorrelated.")
                element_gc = self.generate_uncorrelated_gc(n_elements)
            else:
                element_gc = self.generate_fractal_gc(coordinates)
                print(f"✅ Generated fractal noise Gc distribution")
        
        elif actual_noise_type == 'gradient':
            if coordinates is None:
                print("⚠️  Gradient noise requires coordinates. Falling back to uncorrelated.")
                element_gc = self.generate_uncorrelated_gc(n_elements)
            else:
                element_gc = self.generate_gradient_gc(coordinates)
                print(f"✅ Generated gradient Gc distribution")
        
        elif actual_noise_type == 'clusters':
            if coordinates is None:
                print("⚠️  Cluster noise requires coordinates. Falling back to uncorrelated.")
                element_gc = self.generate_uncorrelated_gc(n_elements)
            else:
                element_gc = self.generate_clusters_gc(coordinates)
                print(f"✅ Generated cluster-based Gc distribution")
        
        elif actual_noise_type == 'bands':
            if coordinates is None:
                print("⚠️  Band noise requires coordinates. Falling back to uncorrelated.")
                element_gc = self.generate_uncorrelated_gc(n_elements)
            else:
                element_gc = self.generate_bands_gc(coordinates)
                print(f"✅ Generated banded Gc distribution")
        
        # Apply microstructure effects if requested
        if self.apply_microstructure and grain_boundary_flags is not None:
            for e in range(n_elements):
                flag = grain_boundary_flags[e]
                
                if flag == 1:  # Grain boundary
                    element_gc[e] *= self.gb_factor
                elif flag == 2:  # Triple junction
                    element_gc[e] *= self.tj_factor
                # flag == 0 (interior) remains unchanged
        
        return element_gc
    
    def get_statistics(self, element_gc, grain_boundary_flags=None):
        """
        Get statistics about the generated Gc distribution.
        
        Parameters:
        -----------
        element_gc : array
            Generated Gc values
        grain_boundary_flags : array, optional
            Element flags for detailed statistics
            
        Returns:
        --------
        stats : dict
            Statistics dictionary
        """
        stats = {
            'mean_gc': np.mean(element_gc),
            'std_gc': np.std(element_gc),
            'min_gc': np.min(element_gc),
            'max_gc': np.max(element_gc),
            'total_elements': len(element_gc),
            'noise_type': self.noise_type,
            'correlation_length': self.correlation_length if self.noise_type in ['correlated', 'auto'] else None
        }
        
        if grain_boundary_flags is not None:
            # Separate statistics by element type
            interior_mask = grain_boundary_flags == 0
            gb_mask = grain_boundary_flags == 1
            tj_mask = grain_boundary_flags == 2
            
            if np.any(interior_mask):
                stats['interior_gc_mean'] = np.mean(element_gc[interior_mask])
                stats['interior_gc_std'] = np.std(element_gc[interior_mask])
                
            if np.any(gb_mask):
                stats['gb_gc_mean'] = np.mean(element_gc[gb_mask])
                stats['gb_gc_std'] = np.std(element_gc[gb_mask])
                
            if np.any(tj_mask):
                stats['tj_gc_mean'] = np.mean(element_gc[tj_mask])
                stats['tj_gc_std'] = np.std(element_gc[tj_mask])
        
        return stats
    
    def plot_distribution(self, element_gc, grain_boundary_flags=None, coordinates=None, save_fig=True, output_folder="plots"):
        """
        Plot histogram of Gc distribution with spatial visualization if coordinates provided.
        
        Parameters:
        -----------
        element_gc : array
            Generated Gc values
        grain_boundary_flags : array, optional
            Element flags for color coding
        coordinates : array, optional
            Element coordinates for spatial plot
        save_fig : bool
            Whether to save the figure
        output_folder : str
            Folder to save the plot
        """
        import matplotlib.pyplot as plt
        import os
        
        # Create output folder if it doesn't exist
        if save_fig and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Determine subplot layout
        if coordinates is not None:
            fig = plt.figure(figsize=(20, 6))
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax3 = None
        
        # Overall distribution histogram
        ax1.hist(element_gc, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(element_gc), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(element_gc):.3f}')
        ax1.axvline(np.mean(element_gc) + np.std(element_gc), color='orange', 
                   linestyle='--', alpha=0.7, label=f'±1σ')
        ax1.axvline(np.mean(element_gc) - np.std(element_gc), color='orange', 
                   linestyle='--', alpha=0.7)
        ax1.set_xlabel('Gc')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Gc Distribution ({self.noise_type} noise)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribution by element type
        if grain_boundary_flags is not None:
            interior_gc = element_gc[grain_boundary_flags == 0]
            gb_gc = element_gc[grain_boundary_flags == 1]
            tj_gc = element_gc[grain_boundary_flags == 2]
            
            if len(interior_gc) > 0:
                ax2.hist(interior_gc, bins=30, alpha=0.7, color='green', 
                        label=f'Interior (n={len(interior_gc)})', edgecolor='black')
            if len(gb_gc) > 0:
                ax2.hist(gb_gc, bins=30, alpha=0.7, color='red', 
                        label=f'Grain Boundary (n={len(gb_gc)})', edgecolor='black')
            if len(tj_gc) > 0:
                ax2.hist(tj_gc, bins=30, alpha=0.7, color='blue', 
                        label=f'Triple Junction (n={len(tj_gc)})', edgecolor='black')
                
            ax2.set_xlabel('Gc')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Gc Distribution by Element Type')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.axis('off')
        
        # Spatial distribution plot
        if coordinates is not None and ax3 is not None:
            scatter = ax3.scatter(coordinates[:, 0], coordinates[:, 1], 
                                c=element_gc, s=2, cmap='viridis')
            ax3.set_xlabel('X coordinate')
            ax3.set_ylabel('Y coordinate')
            ax3.set_title('Spatial Gc Distribution')
            ax3.set_aspect('equal')
            plt.colorbar(scatter, ax=ax3, shrink=0.8, label='Gc')
        
        plt.tight_layout()
        
        if save_fig:
            filename = f'gc_distribution_{self.noise_type}.png'
            filepath = os.path.join(output_folder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   💾 Gc distribution saved to '{filepath}'")
        
        return fig


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def demonstrate_noise_types():
    """
    Demonstrate different noise types with the same parameters.
    """
    # Create example mesh coordinates
    nx, ny = 50, 30
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 6, ny)
    X, Y = np.meshgrid(x, y)
    coordinates = np.column_stack([X.ravel(), Y.ravel()])
    n_elements = len(coordinates)
    
    # Create some grain boundary flags
    grain_boundary_flags = np.zeros(n_elements, dtype=int)
    gb_mask = ((coordinates[:, 0] % 3) < 0.2) | ((coordinates[:, 1] % 2) < 0.2)
    grain_boundary_flags[gb_mask] = 1
    
    # Common parameters
    common_params = {
        'mean_gc': 1.0,
        'std_gc': 0.1,
        'seed': 42,
        'gb_factor': 0.7,
        'tj_factor': 0.5,
        'apply_microstructure': True
    }
    
    print("🔬 Demonstrating different noise types")
    print("=" * 60)
    
    # Test each noise type
    noise_types = ['uncorrelated', 'correlated', 'perlin', 'fractal', 'gradient', 'clusters', 'bands', 'auto']
    
    for noise_type in noise_types:
        print(f"\n📊 Testing {noise_type} noise...")
        
        # Initialize generator
        gc_gen = GcDistribution(
            noise_type=noise_type,
            correlation_length=2.0,
            **common_params
        )
        
        # Set coordinates for spatial noise types
        if noise_type != 'uncorrelated':
            gc_gen.set_coordinates(coordinates)
        
        # Generate Gc values
        element_gc = gc_gen.generate_element_gc(n_elements, grain_boundary_flags)
        
        # Get statistics
        stats = gc_gen.get_statistics(element_gc, grain_boundary_flags)
        print(f"   Mean: {stats['mean_gc']:.3f}, Std: {stats['std_gc']:.3f}")
        print(f"   Range: {stats['min_gc']:.3f} - {stats['max_gc']:.3f}")


if __name__ == "__main__":
    demonstrate_noise_types()