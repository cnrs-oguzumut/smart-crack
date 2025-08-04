"""
Simplified mesh generation module with shared base class to avoid repetition.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from collections import Counter


class BaseMeshGenerator:
    """Base class with shared mesh generation functionality."""
    
    def __init__(self, radius=1.0, target_edge_length=0.05, boundary_layers=3):
        """Initialize base mesh generator."""
        self.radius = radius
        self.target_edge_length = target_edge_length
        self.boundary_layers = boundary_layers
    
    def generate_mesh(self):
        """Generate mesh using template method pattern."""
        # Step 1: Generate boundary points (domain-specific)
        boundary_points = self._generate_boundary_points()
        
        # Step 2: Generate interior points using Poisson disk sampling
        interior_points = self._generate_interior_points_poisson(boundary_points)
        
        # Step 3: Combine all points
        all_points = np.vstack([boundary_points, interior_points])
        
        # Step 4: Create Delaunay triangulation
        delaunay = Delaunay(all_points)
        
        # Step 5: Filter triangles and improve quality
        filtered_simplices = self._filter_and_improve_triangles(all_points, delaunay.simplices)
        
        # Step 6: Apply mesh smoothing
        smoothed_points = self._apply_laplacian_smoothing(all_points, filtered_simplices)
        
        # Step 7: Final quality check and cleanup
        final_simplices = self._final_quality_filter(smoothed_points, filtered_simplices)
        
        # Step 8: Remove unused nodes and remap
        used_nodes = np.unique(final_simplices)
        new_index = -np.ones(len(smoothed_points), dtype=int)
        new_index[used_nodes] = np.arange(len(used_nodes))
        remapped_simplices = new_index[final_simplices]
        
        final_points = smoothed_points[used_nodes]
        x_coords, y_coords = final_points[:, 0], final_points[:, 1]
        connectivity = remapped_simplices
        
        # Detect boundary nodes (domain-specific)
        boundary_nodes = self._get_boundary_nodes_for_bc(connectivity, final_points)
        
        n_nodes, n_elem = len(x_coords), len(connectivity)
        
        # Quality metrics
        quality_metrics = self._compute_quality_metrics(final_points, connectivity)
        
        self._print_mesh_info(n_nodes, n_elem, boundary_nodes, quality_metrics)
        
        return x_coords, y_coords, connectivity, boundary_nodes, n_nodes, n_elem
    
    def _generate_interior_points_poisson(self, boundary_points):
        """Generate interior points using Poisson disk sampling."""
        min_distance = self.target_edge_length * 0.9
        max_attempts = 30
        
        # Start with center point
        points = [np.array([0.0, 0.0])]
        active_list = [0]
        
        while active_list:
            active_idx = np.random.choice(len(active_list))
            active_point_idx = active_list[active_idx]
            active_point = points[active_point_idx]
            
            found_valid = False
            
            for _ in range(max_attempts):
                angle = np.random.uniform(0, 2*np.pi)
                distance = np.random.uniform(min_distance, 2*min_distance)
                new_point = active_point + distance * np.array([np.cos(angle), np.sin(angle)])
                
                # Check if point is inside domain (domain-specific)
                if not self._is_point_inside_domain(new_point):
                    continue
                
                # Check minimum distance to existing points
                if len(points) > 0:
                    distances = cdist([new_point], points)[0]
                    if np.min(distances) < min_distance:
                        continue
                
                points.append(new_point)
                active_list.append(len(points) - 1)
                found_valid = True
                break
            
            if not found_valid:
                active_list.remove(active_point_idx)
        
        return np.array(points[1:])
    
    def _filter_and_improve_triangles(self, points, simplices):
        """Filter triangles based on quality metrics."""
        valid_triangles = []
        
        for tri in simplices:
            centroid = np.mean(points[tri], axis=0)
            if not self._is_centroid_inside_domain(centroid):
                continue
            
            quality = self._triangle_quality(points[tri])
            if quality > 0.3:
                valid_triangles.append(tri)
        
        return np.array(valid_triangles)
    
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
        
        # Quality metric (ratio of inscribed to circumscribed circle radius)
        perimeter = a + b + c
        circumradius = (a * b * c) / (4 * area)
        inradius = area / (0.5 * perimeter)
        
        return 2 * inradius / circumradius
    
    def _apply_laplacian_smoothing(self, points, simplices, iterations=3):
        """Apply Laplacian smoothing to improve mesh quality."""
        smoothed_points = points.copy()
        
        # Build node-to-node connectivity
        node_neighbors = [set() for _ in range(len(points))]
        for tri in simplices:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        node_neighbors[tri[i]].add(tri[j])
        
        # Identify boundary nodes (domain-specific)
        boundary_mask = self._get_boundary_mask(points)
        
        for iteration in range(iterations):
            new_points = smoothed_points.copy()
            
            for i, neighbors in enumerate(node_neighbors):
                if boundary_mask[i] or len(neighbors) == 0:
                    continue
                
                neighbor_coords = smoothed_points[list(neighbors)]
                avg_pos = np.mean(neighbor_coords, axis=0)
                
                damping = 0.5
                new_points[i] = smoothed_points[i] + damping * (avg_pos - smoothed_points[i])
                
                # Ensure point stays inside domain
                if not self._is_point_inside_domain(new_points[i]):
                    new_points[i] = smoothed_points[i]
            
            smoothed_points = new_points
        
        return smoothed_points
    
    def _final_quality_filter(self, points, simplices):
        """Final quality filter."""
        return np.array([tri for tri in simplices 
                        if self._triangle_quality(points[tri]) > 0.2])
    
    def _compute_quality_metrics(self, points, connectivity):
        """Compute comprehensive mesh quality metrics."""
        angles, aspect_ratios, qualities = [], [], []
        
        for tri in connectivity:
            triangle_points = points[tri]
            
            # Angles
            tri_angles = self._triangle_angles(triangle_points)
            angles.extend(tri_angles)
            
            # Aspect ratio
            edge_lengths = [np.linalg.norm(triangle_points[(i+1)%3] - triangle_points[i]) 
                          for i in range(3)]
            aspect_ratios.append(max(edge_lengths) / min(edge_lengths))
            
            # Quality
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
        
        angle1 = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
        angle2 = np.arccos(np.clip(np.dot(-v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)), -1, 1))
        angle3 = np.pi - angle1 - angle2
        
        return [np.degrees(angle) for angle in [angle1, angle2, angle3]]
    
    def plot_mesh(self, x, y, connectivity, boundary_nodes, save_png=True):
        """Plot mesh with quality visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left plot: Basic mesh
        ax1.triplot(x, y, connectivity, 'b-', lw=0.3, alpha=0.7)
        ax1.scatter(x, y, c='lightblue', s=8, alpha=0.6, label='Interior Nodes')
        
        # Plot boundary nodes (domain-specific coloring)
        self._plot_boundary_nodes(ax1, x, y, boundary_nodes)
        self._plot_domain_boundary(ax1)
        
        ax1.set_title(f'Professional Mesh\n{len(connectivity)} elements, {len(x)} nodes')
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
        
        self._plot_domain_boundary(ax2)
        ax2.set_title(f'Quality Visualization\nAvg Quality: {np.mean(qualities):.3f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_png:
            plt.savefig('professional_circular_mesh.png', dpi=150, bbox_inches='tight')
            print("✓ Saved mesh plot: professional_circular_mesh.png")
        
        plt.show()
    
    def _print_mesh_info(self, n_nodes, n_elem, boundary_nodes, quality_metrics):
        """Print mesh generation summary."""
        print(f"✅ Professional Mesh Generated:")
        print(f"   Nodes: {n_nodes}, Elements: {n_elem}, Boundary nodes: {len(boundary_nodes)}")
        print(f"   Min angle: {quality_metrics['min_angle']:.1f}°")
        print(f"   Max angle: {quality_metrics['max_angle']:.1f}°")
        print(f"   Aspect ratio (avg): {quality_metrics['avg_aspect_ratio']:.2f}")
        print(f"   Quality metric (avg): {quality_metrics['avg_quality']:.3f}")
    
    # Abstract methods - must be implemented by subclasses
    def _generate_boundary_points(self):
        """Generate boundary points (domain-specific)."""
        raise NotImplementedError
    
    def _is_point_inside_domain(self, point):
        """Check if point is inside domain (domain-specific)."""
        raise NotImplementedError
    
    def _is_centroid_inside_domain(self, centroid):
        """Check if triangle centroid is inside domain (domain-specific)."""
        raise NotImplementedError
    
    def _get_boundary_mask(self, points):
        """Get mask for boundary nodes (domain-specific)."""
        raise NotImplementedError
    
    def _get_boundary_nodes_for_bc(self, connectivity, points):
        """Get boundary nodes for BC application (domain-specific)."""
        raise NotImplementedError
    
    def _plot_boundary_nodes(self, ax, x, y, boundary_nodes):
        """Plot boundary nodes (domain-specific)."""
        raise NotImplementedError
    
    def _plot_domain_boundary(self, ax):
        """Plot domain boundary (domain-specific)."""
        raise NotImplementedError


class ProfessionalCircularMeshGenerator(BaseMeshGenerator):
    """Professional circular mesh generator."""
    
    def _generate_boundary_points(self):
        """Generate boundary points for circular domain."""
        circumference = 2 * np.pi * self.radius
        n_boundary = max(int(circumference / self.target_edge_length), 20)
        theta = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
        
        boundary_points = []
        for layer in range(self.boundary_layers):
            r = self.radius * (1.0 - layer * 0.02)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            boundary_points.append(np.column_stack([x, y]))
        
        return np.vstack(boundary_points)
    
    def _is_point_inside_domain(self, point):
        """Check if point is inside circle."""
        return np.linalg.norm(point) < self.radius * 0.95
    
    def _is_centroid_inside_domain(self, centroid):
        """Check if triangle centroid is inside circle."""
        return np.linalg.norm(centroid) < self.radius * 0.98
    
    def _get_boundary_mask(self, points):
        """Get boundary mask for circular domain."""
        return np.linalg.norm(points, axis=1) > self.radius * 0.97
    
    def _get_boundary_nodes_for_bc(self, connectivity, points):
        """Get all boundary nodes for circular domain."""
        edge_counter = Counter()
        for tri in connectivity:
            edges = [(tri[i], tri[(i+1)%3]) for i in range(3)]
            for a, b in edges:
                edge = tuple(sorted((a, b)))
                edge_counter[edge] += 1
        
        boundary_nodes_edges = set()
        for (a, b), count in edge_counter.items():
            if count == 1:
                boundary_nodes_edges.update((a, b))
        
        # Also use geometric detection
        distances_to_boundary = np.abs(np.linalg.norm(points, axis=1) - self.radius)
        geometric_boundary = np.where(distances_to_boundary < self.target_edge_length * 0.5)[0]
        
        boundary_nodes = boundary_nodes_edges.union(set(geometric_boundary))
        return np.array(sorted(boundary_nodes))
    
    def _plot_boundary_nodes(self, ax, x, y, boundary_nodes):
        """Plot boundary nodes for circular domain."""
        ax.scatter(x[boundary_nodes], y[boundary_nodes], 
                  c='green', s=25, label='Boundary Nodes', zorder=5)
    
    def _plot_domain_boundary(self, ax):
        """Plot circular domain boundary."""
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = self.radius * np.cos(theta)
        circle_y = self.radius * np.sin(theta)
        ax.plot(circle_x, circle_y, 'k--', lw=2, alpha=0.8, label='Domain Boundary')


class ProfessionalRectangularMeshGenerator(BaseMeshGenerator):
    """Professional rectangular mesh generator with left/right BC focus."""
    
    def __init__(self, radius=1.0, target_edge_length=0.05, boundary_layers=3):
        """Initialize rectangular mesh generator."""
        super().__init__(radius, target_edge_length, boundary_layers)
        # Map circular parameters to rectangular domain
        self.width = 2.0 * radius
        self.height = radius
        self.x_min, self.x_max = -self.width/2, self.width/2
        self.y_min, self.y_max = -self.height/2, self.height/2
    
    def _generate_boundary_points(self):
        """Generate boundary points for rectangular domain."""
        perimeter = 2 * (self.width + self.height)
        n_boundary_total = max(int(perimeter / self.target_edge_length), 40)
        
        n_x = max(int(n_boundary_total * self.width / perimeter), 10)
        n_y = max(int(n_boundary_total * self.height / perimeter), 6)
        
        boundary_points = []
        for layer in range(self.boundary_layers):
            offset = layer * self.target_edge_length * 0.3
            
            # Bottom and top edges
            x_bottom = np.linspace(self.x_min + offset, self.x_max - offset, n_x)
            boundary_points.extend([
                np.column_stack([x_bottom, np.full(n_x, self.y_min + offset)]),
                np.column_stack([x_bottom, np.full(n_x, self.y_max - offset)])
            ])
            
            # Left and right edges (excluding corners)
            y_sides = np.linspace(self.y_min + offset, self.y_max - offset, n_y)[1:-1]
            boundary_points.extend([
                np.column_stack([np.full(len(y_sides), self.x_min + offset), y_sides]),
                np.column_stack([np.full(len(y_sides), self.x_max - offset), y_sides])
            ])
        
        return np.vstack(boundary_points)
    
    def _is_point_inside_domain(self, point):
        """Check if point is inside rectangle."""
        margin = self.target_edge_length * 0.5
        return (self.x_min + margin < point[0] < self.x_max - margin and
                self.y_min + margin < point[1] < self.y_max - margin)
    
    def _is_centroid_inside_domain(self, centroid):
        """Check if triangle centroid is inside rectangle."""
        return (self.x_min < centroid[0] < self.x_max and
                self.y_min < centroid[1] < self.y_max)
    
    def _get_boundary_mask(self, points):
        """Get boundary mask for rectangular domain."""
        tolerance = self.target_edge_length * 0.1
        return (
            (np.abs(points[:, 0] - self.x_min) < tolerance) |
            (np.abs(points[:, 0] - self.x_max) < tolerance) |
            (np.abs(points[:, 1] - self.y_min) < tolerance) |
            (np.abs(points[:, 1] - self.y_max) < tolerance)
        )
    
    def _get_boundary_nodes_for_bc(self, connectivity, points):
        """Get left and right boundary nodes for BC application."""
        # Edge-based detection
        edge_counter = Counter()
        for tri in connectivity:
            edges = [(tri[i], tri[(i+1)%3]) for i in range(3)]
            for a, b in edges:
                edge_counter[tuple(sorted((a, b)))] += 1
        
        boundary_nodes_edges = {node for (a, b), count in edge_counter.items() 
                               if count == 1 for node in (a, b)}
        
        # Geometric detection for left and right boundaries
        tolerance = self.target_edge_length * 0.2
        left_nodes = [n for n in np.where(np.abs(points[:, 0] - self.x_min) < tolerance)[0] 
                     if n in boundary_nodes_edges]
        right_nodes = [n for n in np.where(np.abs(points[:, 0] - self.x_max) < tolerance)[0] 
                      if n in boundary_nodes_edges]
        
        return np.concatenate([left_nodes, right_nodes])
    
    def _plot_boundary_nodes(self, ax, x, y, boundary_nodes):
        """Plot boundary nodes for rectangular domain."""
        # Detect different boundary types for coloring
        tolerance = self.target_edge_length * 0.2
        left_mask = np.abs(x[boundary_nodes] - self.x_min) < tolerance
        right_mask = np.abs(x[boundary_nodes] - self.x_max) < tolerance
        top_mask = np.abs(y[boundary_nodes] - self.y_max) < tolerance
        bottom_mask = np.abs(y[boundary_nodes] - self.y_min) < tolerance
        
        if np.any(left_mask):
            ax.scatter(x[boundary_nodes][left_mask], y[boundary_nodes][left_mask], 
                      c='red', s=40, label='Left BC', zorder=5, marker='s')
        if np.any(right_mask):
            ax.scatter(x[boundary_nodes][right_mask], y[boundary_nodes][right_mask], 
                      c='darkred', s=40, label='Right BC', zorder=5, marker='s')
        if np.any(top_mask):
            ax.scatter(x[boundary_nodes][top_mask], y[boundary_nodes][top_mask], 
                      c='green', s=25, label='Top', zorder=4, marker='^')
        if np.any(bottom_mask):
            ax.scatter(x[boundary_nodes][bottom_mask], y[boundary_nodes][bottom_mask], 
                      c='orange', s=25, label='Bottom', zorder=4, marker='v')
    
    def _plot_domain_boundary(self, ax):
        """Plot rectangular domain boundary."""
        rect_x = [self.x_min, self.x_max, self.x_max, self.x_min, self.x_min]
        rect_y = [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min]
        ax.plot(rect_x, rect_y, 'k--', lw=2, alpha=0.8, label='Domain Boundary')


# Simple mesh generator for basic use cases
class CircularMeshGenerator:
    """Simple circular mesh generator."""
    
    def __init__(self, radius=1.0, resolution=0.05):
        self.radius = radius
        self.resolution = resolution
        
    def generate_mesh(self):
        """Generate basic triangular mesh in circle."""
        # Create uniform grid
        margin = self.resolution
        x = np.arange(-self.radius - margin, self.radius + margin, self.resolution)
        y = np.arange(-self.radius - margin, self.radius + margin, self.resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.vstack((X.ravel(), Y.ravel())).T

        # Keep points inside circle
        r2 = np.sum(grid_points**2, axis=1)
        points = grid_points[r2 < (self.radius * 1.05)**2]

        # Delaunay triangulation and filtering
        delaunay = Delaunay(points)
        is_inside = lambda tri: np.linalg.norm(points[tri].mean(axis=0)) < self.radius * 0.999
        simplices = np.array([tri for tri in delaunay.simplices if is_inside(tri)])

        # Filter by area
        areas = np.array([0.5 * abs(points[tri[0], 0] * (points[tri[1], 1] - points[tri[2], 1]) +
                                   points[tri[1], 0] * (points[tri[2], 1] - points[tri[0], 1]) +
                                   points[tri[2], 0] * (points[tri[0], 1] - points[tri[1], 1]))
                         for tri in simplices])
        
        median_area = np.median(areas)
        tolerance = 0.2 * median_area
        mask = (areas > median_area - tolerance) & (areas < median_area + tolerance)
        simplices = simplices[mask]

        # Remap indices
        used_nodes = np.unique(simplices)
        new_index = -np.ones(len(points), dtype=int)
        new_index[used_nodes] = np.arange(len(used_nodes))
        
        final_points = points[used_nodes]
        connectivity = new_index[simplices]
        boundary_nodes = self._detect_boundary_nodes(connectivity)
        
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
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        ax.triplot(x, y, connectivity, 'b-', lw=0.5, alpha=0.7)
        ax.scatter(x, y, c='red', s=10, alpha=0.6, label='Interior Nodes')
        ax.scatter(x[boundary_nodes], y[boundary_nodes], 
                  c='green', s=40, label='Boundary Nodes', zorder=5)
        
        # Circle boundary
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(self.radius * np.cos(theta), self.radius * np.sin(theta), 
               'k--', lw=2, alpha=0.8, label='Domain Boundary')
        
        ax.set_title(f'Circular Mesh\n{len(connectivity)} elements, {len(x)} nodes')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_png:
            plt.savefig('circular_mesh.png', dpi=120, bbox_inches='tight')
            print("✓ Saved mesh plot: circular_mesh.png")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Professional circular mesh
    print("=== Professional Circular Mesh ===")
    circular_gen = ProfessionalCircularMeshGenerator(radius=1.0, target_edge_length=0.08)
    x, y, conn, boundary, n_nodes, n_elem = circular_gen.generate_mesh()
    circular_gen.plot_mesh(x, y, conn, boundary)
    
    # Professional rectangular mesh (drop-in replacement)
    print("\n=== Professional Rectangular Mesh ===")
    rect_gen = ProfessionalRectangularMeshGenerator(radius=1.0, target_edge_length=0.08)
    x, y, conn, boundary, n_nodes, n_elem = rect_gen.generate_mesh()
    rect_gen.plot_mesh(x, y, conn, boundary)