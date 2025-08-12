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
        self.height = radius
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


# Example usage and testing
if __name__ == "__main__":
    print("=== TESTING TRIANGLE-BASED MESH GENERATORS ===")
    
    # Test circular mesh
    print("\n=== Professional Circular Mesh (Triangle) ===")
    try:
        circular_gen = ProfessionalCircularMeshGenerator(radius=1.0, target_edge_length=0.08)
        x, y, conn, boundary, n_nodes, n_elem = circular_gen.generate_mesh()
        circular_gen.plot_mesh(x, y, conn, boundary)
    except Exception as e:
        print(f"Circular mesh error: {e}")
    
    # Test rectangular mesh
    print("\n=== Professional Rectangular Mesh (Triangle) ===")
    try:
        rect_gen = ProfessionalRectangularMeshGenerator(radius=1.0, target_edge_length=0.08)
        x, y, conn, boundary, n_nodes, n_elem = rect_gen.generate_mesh()
        rect_gen.plot_mesh(x, y, conn, boundary)
    except Exception as e:
        print(f"Rectangular mesh error: {e}")
    
    # Test simple fallback
    print("\n=== Simple Circular Mesh (Fallback) ===")
    try:
        simple_gen = CircularMeshGenerator(radius=1.0, resolution=0.08)
        x, y, conn, boundary, n_nodes, n_elem = simple_gen.generate_mesh()
        simple_gen.plot_mesh(x, y, conn, boundary)
    except Exception as e:
        print(f"Simple mesh error: {e}")
    
    print("\n=== INSTALLATION INSTRUCTIONS ===")
    print("For best results, install Triangle:")
    print("pip install triangle")
    print("\nAlternative mesh libraries:")
    print("pip install gmsh")
    print("pip install pygmsh")
    print("pip install dmsh")