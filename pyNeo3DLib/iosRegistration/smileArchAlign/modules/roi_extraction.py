"""
SmileArch ROI Extraction Module
===============================

Extracts the region of interest (tooth surface area) from the aligned smileArch model.
This region should match the reference smile_arch_half.stl.

Extraction Strategies:
1. Z-coordinate based: Extract top N% of Z coordinates
2. Normal vector based: Extract faces with +Z facing normals
3. Hybrid: Combine both criteria
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, cKDTree
from pathlib import Path


# ============================================================================
# ROI Extraction Functions
# ============================================================================

def extract_by_z_threshold(mesh: trimesh.Trimesh, percentile: float = 70) -> trimesh.Trimesh:
    """
    Extract faces in the upper Z region.
    
    Args:
        mesh: Aligned input mesh
        percentile: Z percentile threshold (e.g., 70 means top 30%)
        
    Returns:
        Submesh containing only the upper region faces
    """
    # Get face centroids Z values
    face_centroids = mesh.triangles_center
    z_values = face_centroids[:, 2]
    
    # Calculate threshold
    z_threshold = np.percentile(z_values, percentile)
    
    # Select faces above threshold
    mask = z_values >= z_threshold
    selected_faces = mesh.faces[mask]
    
    print(f"  Z threshold: {z_threshold:.2f} (percentile: {percentile})")
    print(f"  Selected faces: {np.sum(mask):,} / {len(mesh.faces):,} ({100*np.sum(mask)/len(mesh.faces):.1f}%)")
    
    # Create submesh
    if len(selected_faces) == 0:
        print("  Warning: No faces selected!")
        return mesh.copy()
    
    submesh = mesh.submesh([np.where(mask)[0]], append=True)
    return submesh


def extract_by_normal_direction(mesh: trimesh.Trimesh, 
                                 direction: np.ndarray = None, 
                                 threshold: float = 0.3) -> trimesh.Trimesh:
    """
    Extract faces with normals pointing in a specific direction.
    
    Args:
        mesh: Input mesh
        direction: Target normal direction (default: +Z)
        threshold: Minimum dot product with direction (0-1)
        
    Returns:
        Submesh containing only faces with matching normals
    """
    if direction is None:
        direction = np.array([0, 0, 1])  # +Z direction
    
    direction = direction / np.linalg.norm(direction)
    
    # Get face normals
    face_normals = mesh.face_normals
    
    # Calculate dot product with target direction
    dot_products = np.dot(face_normals, direction)
    
    # Select faces with normals aligned to direction
    mask = dot_products >= threshold
    
    print(f"  Normal direction: {direction}")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Selected faces: {np.sum(mask):,} / {len(mesh.faces):,} ({100*np.sum(mask)/len(mesh.faces):.1f}%)")
    
    if np.sum(mask) == 0:
        print("  Warning: No faces selected!")
        return mesh.copy()
    
    submesh = mesh.submesh([np.where(mask)[0]], append=True)
    return submesh


def extract_arch_region(mesh: trimesh.Trimesh, 
                        z_percentile: float = 60,
                        normal_threshold: float = 0.2) -> trimesh.Trimesh:
    """
    Extract arch-shaped tooth surface region using combined criteria.
    
    This is the recommended method for smileArch -> smile_arch_half matching.
    
    Args:
        mesh: Aligned input mesh
        z_percentile: Z coordinate percentile (lower = more faces)
        normal_threshold: Normal direction threshold (lower = more faces)
        
    Returns:
        Submesh containing the arch region
    """
    print("\n  [Hybrid ROI Extraction]")
    
    # Get face data
    face_centroids = mesh.triangles_center
    face_normals = mesh.face_normals
    
    z_values = face_centroids[:, 2]
    z_threshold = np.percentile(z_values, z_percentile)
    
    # +Z normal criterion
    z_normal_dot = face_normals[:, 2]
    
    # Combined mask: high Z AND +Z facing normal
    z_mask = z_values >= z_threshold
    normal_mask = z_normal_dot >= normal_threshold
    combined_mask = z_mask & normal_mask
    
    print(f"  Z threshold: {z_threshold:.2f} (percentile {z_percentile})")
    print(f"  Normal threshold: {normal_threshold:.2f}")
    print(f"  Z mask: {np.sum(z_mask):,} faces")
    print(f"  Normal mask: {np.sum(normal_mask):,} faces")
    print(f"  Combined: {np.sum(combined_mask):,} faces ({100*np.sum(combined_mask)/len(mesh.faces):.1f}%)")
    
    if np.sum(combined_mask) == 0:
        print("  Warning: Combined mask empty, falling back to Z-only")
        combined_mask = z_mask
    
    submesh = mesh.submesh([np.where(combined_mask)[0]], append=True)
    return submesh


def extract_by_front_point(mesh: trimesh.Trimesh, 
                           z_offset: float = 0.0,
                           normal_threshold: float = 0.0,
                           remove_narrow_noise: bool = True,
                           min_width_ratio: float = 0.2) -> trimesh.Trimesh:
    """
    Extract ROI based on the front point (max Y) Z-coordinate.
    
    Algorithm:
    1. Find the point with maximum Y value (front/anterior tip)
    2. Use that point's Z value as the cutting threshold
    3. Keep all faces with centroid Z >= front_point_z + offset
    4. Optionally filter by +Z normal direction
    5. Remove narrow noise in +Y direction (impression material artifacts)
    
    This is more robust than percentile-based methods because it's based on
    the actual geometry of the aligned arch.
    
    Args:
        mesh: Aligned input mesh (must be properly aligned with +Y = front)
        z_offset: Offset to add to the front point Z (negative = include more)
        normal_threshold: Minimum Z-component of normal (0 = disabled)
        remove_narrow_noise: Whether to filter out narrow elongated noise
        min_width_ratio: Minimum X-width ratio compared to max width (0.2 = 20%)
        
    Returns:
        Submesh containing the arch surface region
    """
    print("\n  [Front Point Based ROI Extraction]")
    
    vertices = mesh.vertices
    
    # Find the front-most point (max Y)
    front_idx = np.argmax(vertices[:, 1])
    front_point = vertices[front_idx]
    front_y = front_point[1]
    front_z = front_point[2]
    
    print(f"  Front point: Y={front_y:.2f}, Z={front_z:.2f}")
    print(f"  Z offset: {z_offset:.2f}")
    
    z_threshold = front_z + z_offset
    print(f"  Z threshold: {z_threshold:.2f}")
    
    # Get face data
    face_centroids = mesh.triangles_center
    z_values = face_centroids[:, 2]
    y_values = face_centroids[:, 1]
    x_values = face_centroids[:, 0]
    
    # Primary mask: faces above threshold
    z_mask = z_values >= z_threshold
    print(f"  Z mask: {np.sum(z_mask):,} faces")
    
    # Optional: normal direction filter
    combined_mask = z_mask.copy()
    if normal_threshold > 0:
        face_normals = mesh.face_normals
        normal_mask = face_normals[:, 2] >= normal_threshold
        combined_mask = combined_mask & normal_mask
        print(f"  Normal threshold: {normal_threshold:.2f}")
        print(f"  Normal mask: {np.sum(normal_mask):,} faces")
    
    # Remove narrow elongated noise
    if remove_narrow_noise and np.sum(combined_mask) > 0:
        print(f"\n  [Narrow Noise Filter]")
        
        # Get selected face data
        selected_indices = np.where(combined_mask)[0]
        selected_y = y_values[combined_mask]
        selected_x = x_values[combined_mask]
        y_min, y_max = np.min(selected_y), np.max(selected_y)
        
        # Divide into Y bins and calculate X width
        n_bins = 10
        bin_widths = []
        for i in range(n_bins):
            y_low = y_min + (y_max - y_min) * i / n_bins
            y_high = y_min + (y_max - y_min) * (i + 1) / n_bins
            bin_mask = (selected_y >= y_low) & (selected_y < y_high)
            if np.sum(bin_mask) > 1:
                bin_x = selected_x[bin_mask]
                width = np.max(bin_x) - np.min(bin_x)
            else:
                width = 0
            bin_widths.append(width)
        
        max_width = max(bin_widths) if bin_widths else 1
        print(f"  Y-bin widths: {[f'{w:.1f}' for w in bin_widths]}")
        print(f"  Max width: {max_width:.2f}, Min ratio: {min_width_ratio:.2f}")
        
        # Find narrow bins at both ends
        min_threshold = max_width * min_width_ratio
        
        # Cut from +Y end (check backwards)
        cutoff_y_high = y_max
        for i in range(n_bins - 1, -1, -1):
            if bin_widths[i] < min_threshold:
                cutoff_y_high = y_min + (y_max - y_min) * (i + 1) / n_bins
                print(f"  High-Y cutoff at Y={cutoff_y_high:.2f} (bin {i+1}, width={bin_widths[i]:.1f})")
            else:
                break
        
        # Cut from -Y end (check forwards)  
        cutoff_y_low = y_min
        for i in range(n_bins):
            if bin_widths[i] < min_threshold:
                cutoff_y_low = y_min + (y_max - y_min) * (i + 1) / n_bins
                print(f"  Low-Y cutoff at Y={cutoff_y_low:.2f} (bin {i+1}, width={bin_widths[i]:.1f})")
            else:
                break
        
        # Apply both cutoffs
        if cutoff_y_low > y_min or cutoff_y_high < y_max:
            y_filter_mask = (y_values >= cutoff_y_low) & (y_values <= cutoff_y_high)
            before_count = np.sum(combined_mask)
            combined_mask = combined_mask & y_filter_mask
            after_count = np.sum(combined_mask)
            print(f"  Removed {before_count - after_count} narrow noise faces")
    
    # Density-based filtering to remove isolated outliers
    if np.sum(combined_mask) > 100:
        print(f"\n  [Density Filter]")
        selected_centroids = face_centroids[combined_mask]
        
        # Build KD-tree and find neighbors within radius
        tree = cKDTree(selected_centroids)
        
        # Estimate appropriate radius based on model size
        bbox_size = np.max(selected_centroids, axis=0) - np.min(selected_centroids, axis=0)
        radius = np.mean(bbox_size) * 0.05  # 5% of average bbox size
        
        # Count neighbors for each point
        neighbor_counts = np.array([len(tree.query_ball_point(pt, radius)) for pt in selected_centroids])
        
        # Threshold: keep only dense regions
        min_neighbors = max(3, int(np.median(neighbor_counts) / 4))
        print(f"  Radius: {radius:.2f}, Min neighbors: {min_neighbors}")
        
        # Find high-density faces (main arch region)
        high_density_mask = neighbor_counts >= min_neighbors
        dense_points_2d = selected_centroids[high_density_mask][:, :2]  # XY only
        
        if len(dense_points_2d) > 10:
            # Compute convex hull of dense region in XY plane
            try:
                hull = ConvexHull(dense_points_2d)
                hull_path = dense_points_2d[hull.vertices]
                
                # Check which faces are inside the convex hull
                all_centroids_2d = face_centroids[combined_mask][:, :2]
                inside_mask = _points_in_convex_hull(all_centroids_2d, hull_path)
                
                selected_indices = np.where(combined_mask)[0]
                outside_indices = selected_indices[~inside_mask]
                
                before_count = np.sum(combined_mask)
                combined_mask[outside_indices] = False
                after_count = np.sum(combined_mask)
                print(f"  Dense cluster: {len(dense_points_2d):,} faces")
                print(f"  Convex hull vertices: {len(hull.vertices)}")
                print(f"  Removed {before_count - after_count} faces outside hull")
            except Exception as e:
                print(f"  Convex hull failed: {e}")
    
    print(f"  Final: {np.sum(combined_mask):,} faces ({100*np.sum(combined_mask)/len(mesh.faces):.1f}%)")
    
    if np.sum(combined_mask) == 0:
        print("  Warning: No faces selected!")
        return mesh.copy()
    
    submesh = mesh.submesh([np.where(combined_mask)[0]], append=True)
    return submesh


def _points_in_convex_hull(points: np.ndarray, hull_vertices: np.ndarray) -> np.ndarray:
    """
    Check if 2D points are inside a convex hull.
    Uses cross product method for efficiency.
    """
    n = len(hull_vertices)
    inside = np.ones(len(points), dtype=bool)
    
    for i in range(n):
        p1 = hull_vertices[i]
        p2 = hull_vertices[(i + 1) % n]
        
        # Edge vector
        edge = p2 - p1
        
        # Vector from edge start to each point
        to_points = points - p1
        
        # Cross product (positive = left of edge, negative = right)
        cross = edge[0] * to_points[:, 1] - edge[1] * to_points[:, 0]
        
        # Points should be on the same side for all edges
        inside = inside & (cross >= -1e-6)  # Small tolerance
    
    return inside


def filter_by_reference(roi_mesh: trimesh.Trimesh, 
                        reference_mesh: trimesh.Trimesh,
                        margin: float = 5.0) -> trimesh.Trimesh:
    """
    Filter ROI by reference model's XY convex hull.
    
    Keeps only faces that fall within the reference's XY convex hull
    (with optional margin expansion).
    
    Args:
        roi_mesh: Extracted ROI mesh
        reference_mesh: Reference mesh (smile_arch_half.stl)
        margin: Margin to expand the reference hull (in mm)
        
    Returns:
        Filtered mesh
    """
    print("\n  [Reference-Based Filter]")
    
    # Get reference XY convex hull
    ref_vertices = reference_mesh.vertices
    ref_2d = ref_vertices[:, :2]
    
    try:
        hull = ConvexHull(ref_2d)
        hull_points = ref_2d[hull.vertices]
        
        # Expand hull by margin
        if margin > 0:
            # Calculate centroid
            centroid = np.mean(hull_points, axis=0)
            # Expand each hull point outward
            directions = hull_points - centroid
            distances = np.linalg.norm(directions, axis=1, keepdims=True)
            unit_dirs = directions / (distances + 1e-6)
            hull_points = hull_points + unit_dirs * margin
        
        print(f"  Reference hull: {len(hull.vertices)} vertices")
        print(f"  Margin: {margin:.1f}mm")
        
        # Get ROI face centroids
        face_centroids = roi_mesh.triangles_center
        centroids_2d = face_centroids[:, :2]
        
        # Check which faces are inside the hull
        inside_mask = _points_in_convex_hull(centroids_2d, hull_points)
        
        inside_count = np.sum(inside_mask)
        total_count = len(inside_mask)
        print(f"  Inside hull: {inside_count:,} / {total_count:,} faces ({100*inside_count/total_count:.1f}%)")
        
        if inside_count == 0:
            print("  Warning: No faces inside hull!")
            return roi_mesh.copy()
        
        # Create filtered submesh
        filtered_mesh = roi_mesh.submesh([np.where(inside_mask)[0]], append=True)
        return filtered_mesh
        
    except Exception as e:
        print(f"  Error: {e}")
        return roi_mesh.copy()


def extract_roi(mesh: trimesh.Trimesh, 
                method: str = 'front_point',
                visualize: bool = True,
                output_dir: str = None,
                **kwargs) -> tuple:
    """
    Main ROI extraction function.
    
    Args:
        mesh: Aligned input mesh
        method: 'z_threshold', 'normal', 'hybrid', or 'front_point' (default)
        visualize: Whether to visualize results
        output_dir: Path to save visualizations
        **kwargs: Method-specific parameters
        
    Returns:
        (roi_mesh, extraction_info)
    """
    extraction_info = {
        'method': method,
        'original_faces': len(mesh.faces),
        'roi_faces': 0,
        'parameters': kwargs
    }
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    print("\n" + "=" * 50)
    print("[ROI Extraction]")
    print("=" * 50)
    print(f"  Method: {method}")
    
    if method == 'z_threshold':
        percentile = kwargs.get('percentile', 70)
        roi_mesh = extract_by_z_threshold(mesh, percentile=percentile)
        extraction_info['parameters']['percentile'] = percentile
        
    elif method == 'normal':
        direction = kwargs.get('direction', np.array([0, 0, 1]))
        threshold = kwargs.get('threshold', 0.3)
        roi_mesh = extract_by_normal_direction(mesh, direction=direction, threshold=threshold)
        extraction_info['parameters']['threshold'] = threshold
        
    elif method == 'hybrid':
        z_percentile = kwargs.get('z_percentile', 60)
        normal_threshold = kwargs.get('normal_threshold', 0.2)
        roi_mesh = extract_arch_region(mesh, z_percentile=z_percentile, normal_threshold=normal_threshold)
        extraction_info['parameters']['z_percentile'] = z_percentile
        extraction_info['parameters']['normal_threshold'] = normal_threshold
    
    elif method == 'front_point':
        z_offset = kwargs.get('z_offset', 0.0)
        normal_threshold = kwargs.get('normal_threshold', 0.0)
        roi_mesh = extract_by_front_point(mesh, z_offset=z_offset, normal_threshold=normal_threshold)
        extraction_info['parameters']['z_offset'] = z_offset
        extraction_info['parameters']['normal_threshold'] = normal_threshold
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    extraction_info['roi_faces'] = len(roi_mesh.faces)
    extraction_info['extraction_ratio'] = len(roi_mesh.faces) / len(mesh.faces)
    
    print(f"\n  Result: {extraction_info['roi_faces']:,} faces extracted ({100*extraction_info['extraction_ratio']:.1f}%)")
    
    if visualize:
        save_path = str(output_path / "roi_extraction.png") if output_path else None
        visualize_roi_extraction(mesh, roi_mesh, extraction_info, save_path=save_path)
    
    return roi_mesh, extraction_info


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_roi_extraction(original_mesh: trimesh.Trimesh, 
                             roi_mesh: trimesh.Trimesh,
                             info: dict,
                             save_path: str = None):
    """Visualize ROI extraction result."""
    fig = plt.figure(figsize=(15, 10))
    method = info.get('method', 'unknown')
    ratio = info.get('extraction_ratio', 0) * 100
    fig.suptitle(f"ROI Extraction ({method}) - {ratio:.1f}% extracted", fontsize=14, fontweight='bold')
    
    # Sample vertices for visualization
    def sample_vertices(mesh, n=5000):
        v = mesh.vertices
        if len(v) > n:
            idx = np.random.choice(len(v), n, replace=False)
            return v[idx]
        return v
    
    orig_v = sample_vertices(original_mesh)
    roi_v = sample_vertices(roi_mesh)
    
    # Row 1: Original mesh views
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    z_colors = plt.cm.viridis((orig_v[:, 2] - orig_v[:, 2].min()) / (orig_v[:, 2].max() - orig_v[:, 2].min() + 1e-6))
    ax1.scatter(orig_v[:, 0], orig_v[:, 1], orig_v[:, 2], s=0.5, alpha=0.5, c=z_colors)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title(f"Original 3D\n({info['original_faces']:,} faces)")
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(orig_v[:, 0], orig_v[:, 1], s=0.3, alpha=0.3, c='gray')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    ax2.set_title("Original Top View (XY)")
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(orig_v[:, 1], orig_v[:, 2], s=0.3, alpha=0.3, c='gray')
    ax3.set_xlabel('Y'); ax3.set_ylabel('Z')
    ax3.set_title("Original Side View (YZ)")
    ax3.set_aspect('equal'); ax3.grid(True, alpha=0.3)
    
    # Row 2: ROI mesh views
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    z_colors_roi = plt.cm.plasma((roi_v[:, 2] - roi_v[:, 2].min()) / (roi_v[:, 2].max() - roi_v[:, 2].min() + 1e-6))
    ax4.scatter(roi_v[:, 0], roi_v[:, 1], roi_v[:, 2], s=0.5, alpha=0.5, c=z_colors_roi)
    ax4.set_xlabel('X'); ax4.set_ylabel('Y'); ax4.set_zlabel('Z')
    ax4.set_title(f"ROI 3D\n({info['roi_faces']:,} faces)")
    
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(roi_v[:, 0], roi_v[:, 1], s=0.5, alpha=0.5, c='coral')
    ax5.set_xlabel('X'); ax5.set_ylabel('Y')
    ax5.set_title("ROI Top View (XY)")
    ax5.set_aspect('equal'); ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(roi_v[:, 1], roi_v[:, 2], s=0.5, alpha=0.5, c='coral')
    ax6.set_xlabel('Y'); ax6.set_ylabel('Z')
    ax6.set_title("ROI Side View (YZ)")
    ax6.set_aspect('equal'); ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f">>> Saved: {save_path}")
    plt.show()


def visualize_roi_vs_reference(roi_mesh: trimesh.Trimesh,
                               reference_mesh: trimesh.Trimesh,
                               save_path: str = None):
    """Visualize ROI compared to reference (smile_arch_half)."""
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("ROI vs Reference Comparison", fontsize=14, fontweight='bold')
    
    def sample_vertices(mesh, n=5000):
        v = mesh.vertices
        if len(v) > n:
            idx = np.random.choice(len(v), n, replace=False)
            return v[idx]
        return v
    
    roi_v = sample_vertices(roi_mesh)
    ref_v = sample_vertices(reference_mesh)
    
    # 3D overlay
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(ref_v[:, 0], ref_v[:, 1], ref_v[:, 2], s=0.5, alpha=0.3, c='blue', label='Reference')
    ax1.scatter(roi_v[:, 0], roi_v[:, 1], roi_v[:, 2], s=0.5, alpha=0.3, c='red', label='ROI')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title("3D Overlay")
    ax1.legend()
    
    # Top view overlay
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(ref_v[:, 0], ref_v[:, 1], s=0.5, alpha=0.3, c='blue', label='Reference')
    ax2.scatter(roi_v[:, 0], roi_v[:, 1], s=0.5, alpha=0.3, c='red', label='ROI')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    ax2.set_title("Top View (XY)")
    ax2.set_aspect('equal'); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    # Side view overlay
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(ref_v[:, 1], ref_v[:, 2], s=0.5, alpha=0.3, c='blue', label='Reference')
    ax3.scatter(roi_v[:, 1], roi_v[:, 2], s=0.5, alpha=0.3, c='red', label='ROI')
    ax3.set_xlabel('Y'); ax3.set_ylabel('Z')
    ax3.set_title("Side View (YZ)")
    ax3.set_aspect('equal'); ax3.legend(); ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f">>> Saved: {save_path}")
    plt.show()


# ============================================================================
# Test Execution
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from axis_alignment import load_mesh, align_model
    
    parser = argparse.ArgumentParser(description='SmileArch ROI Extraction Test')
    parser.add_argument('--sample', type=int, default=1, help='Sample number (1, 3, 4)')
    parser.add_argument('--method', type=str, default='front_point', 
                        choices=['z_threshold', 'normal', 'hybrid', 'front_point'],
                        help='Extraction method')
    parser.add_argument('--no-visualize', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent.parent.parent / "3dmodel"
    sample_path = base_path / f"sample_{args.sample}" / "smileArch.stl"
    ref_path = base_path / "smile_arch_half.stl"
    output_dir = Path(__file__).parent.parent / "output" / f"sample_{args.sample}" / "visualization"
    
    if not sample_path.exists():
        print(f"Error: {sample_path} not found.")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"SmileArch ROI Extraction Test - Sample {args.sample}")
    print(f"{'='*60}")
    
    # Load and align
    mesh_original, mesh_ds = load_mesh(str(sample_path), downsample_target=10000)
    aligned_mesh, align_info = align_model(mesh_ds, visualize=False)
    
    # Extract ROI
    roi_mesh, roi_info = extract_roi(
        aligned_mesh,
        method=args.method,
        visualize=not args.no_visualize,
        output_dir=str(output_dir)
    )
    
    # Compare with reference if available
    if ref_path.exists() and not args.no_visualize:
        print(f"\n  Loading reference: {ref_path}")
        ref_mesh = trimesh.load(str(ref_path))
        
        # Note: Reference-based filter disabled - noise will be handled by Robust ICP
        # roi_filtered = filter_by_reference(roi_mesh, ref_mesh, margin=5.0)
        
        save_path = str(output_dir / "roi_vs_reference.png")
        visualize_roi_vs_reference(roi_mesh, ref_mesh, save_path=save_path)
    
    # Result
    print(f"\n{'='*60}")
    print("ROI Extraction Complete!")
    print(f"{'='*60}")
    print(f"  Method: {roi_info['method']}")
    print(f"  Original faces: {roi_info['original_faces']:,}")
    print(f"  ROI faces: {roi_info['roi_faces']:,}")
    print(f"  Extraction ratio: {100*roi_info['extraction_ratio']:.1f}%")
