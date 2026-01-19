"""
SmileArch Coarse Alignment Module
=================================

Performs initial coarse alignment between ROI (extracted from smileArch)
and reference model (smile_arch_half.stl).

Alignment Steps:
1. Center alignment: Match centroids
2. Scale adjustment (if needed)
3. Bounding box based position refinement
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from pathlib import Path


# ============================================================================
# Coarse Alignment Functions
# ============================================================================

def align_centroids(source_mesh: trimesh.Trimesh, 
                    target_mesh: trimesh.Trimesh,
                    exclude_y: bool = False) -> tuple:
    """
    Align source mesh centroid to target mesh centroid.
    
    Args:
        source_mesh: Source mesh to be transformed
        target_mesh: Target/reference mesh (fixed)
        exclude_y: If True, only align X and Z (Y handled by Y-sliding)
        
    Returns:
        (aligned_mesh, translation_vector)
    """
    source_centroid = source_mesh.centroid
    target_centroid = target_mesh.centroid
    
    translation = target_centroid - source_centroid
    
    if exclude_y:
        translation[1] = 0  # Don't translate in Y direction
    
    aligned_mesh = source_mesh.copy()
    aligned_mesh.apply_translation(translation)
    
    print(f"  Source centroid: [{source_centroid[0]:.2f}, {source_centroid[1]:.2f}, {source_centroid[2]:.2f}]")
    print(f"  Target centroid: [{target_centroid[0]:.2f}, {target_centroid[1]:.2f}, {target_centroid[2]:.2f}]")
    print(f"  Translation: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]")
    if exclude_y:
        print(f"  (Y excluded - handled by Y-sliding)")
    
    return aligned_mesh, translation


def compute_scale_factor(source_mesh: trimesh.Trimesh, 
                         target_mesh: trimesh.Trimesh,
                         method: str = 'bbox') -> float:
    """
    Compute scale factor to match source to target size.
    
    Args:
        source_mesh: Source mesh
        target_mesh: Target mesh
        method: 'bbox' for bounding box, 'pca' for PCA-based
        
    Returns:
        Scale factor (source * scale = target size)
    """
    if method == 'bbox':
        source_extents = source_mesh.bounds[1] - source_mesh.bounds[0]
        target_extents = target_mesh.bounds[1] - target_mesh.bounds[0]
        
        # Use average of XY extents (ignore Z as it may differ significantly)
        source_size = np.mean(source_extents[:2])
        target_size = np.mean(target_extents[:2])
        
        scale = target_size / source_size if source_size > 0 else 1.0
        
        print(f"  Source XY size: {source_size:.2f}")
        print(f"  Target XY size: {target_size:.2f}")
        print(f"  Scale factor: {scale:.4f}")
        
        return scale
    else:
        # PCA-based method (future enhancement)
        return 1.0


def apply_scale(mesh: trimesh.Trimesh, scale: float, 
                center: np.ndarray = None) -> trimesh.Trimesh:
    """
    Apply uniform scaling to mesh around a center point.
    
    Args:
        mesh: Input mesh
        scale: Scale factor
        center: Center of scaling (default: mesh centroid)
        
    Returns:
        Scaled mesh
    """
    if center is None:
        center = mesh.centroid
    
    scaled_mesh = mesh.copy()
    
    # Translate to origin, scale, translate back
    scaled_mesh.apply_translation(-center)
    scaled_mesh.apply_scale(scale)
    scaled_mesh.apply_translation(center)
    
    return scaled_mesh


def refine_z_alignment(source_mesh: trimesh.Trimesh,
                       target_mesh: trimesh.Trimesh) -> tuple:
    """
    Refine Z-axis alignment based on bounding box.
    
    Aligns the bottom surfaces (min Z) of both meshes.
    The base of dental arch is more consistent than the top.
    
    Args:
        source_mesh: Source mesh (already centroid aligned)
        target_mesh: Target mesh
        
    Returns:
        (aligned_mesh, z_offset)
    """
    source_z_min = np.min(source_mesh.vertices[:, 2])
    target_z_min = np.min(target_mesh.vertices[:, 2])
    
    z_offset = target_z_min - source_z_min
    
    aligned_mesh = source_mesh.copy()
    aligned_mesh.apply_translation([0, 0, z_offset])
    
    print(f"  Source Z min: {source_z_min:.2f}")
    print(f"  Target Z min: {target_z_min:.2f}")
    print(f"  Z offset: {z_offset:.2f}")
    
    return aligned_mesh, z_offset


def compute_proximity_score(source_mesh: trimesh.Trimesh,
                            target_mesh: trimesh.Trimesh,
                            threshold: float = 2.0,
                            max_points: int = 2000) -> float:
    """
    Compute proximity score: ratio of source points near target surface.
    
    Uses KD-tree for fast nearest neighbor search.
    
    Args:
        source_mesh: Source mesh
        target_mesh: Target mesh
        threshold: Distance threshold (mm) to consider "near"
        max_points: Maximum points to sample for speed
        
    Returns:
        Proximity score (0-1): ratio of nearby points
    """
    from scipy.spatial import cKDTree
    
    # Sample points from source mesh
    src_pts = source_mesh.vertices
    if len(src_pts) > max_points:
        idx = np.random.choice(len(src_pts), max_points, replace=False)
        src_pts = src_pts[idx]
    
    # Build KD-tree from target vertices (fast)
    tgt_pts = target_mesh.vertices
    tree = cKDTree(tgt_pts)
    
    # Find nearest neighbor distances
    distances, _ = tree.query(src_pts, k=1)
    
    # Count points within threshold distance
    nearby_count = np.sum(distances < threshold)
    proximity_score = nearby_count / len(src_pts)
    
    return proximity_score


def y_sliding_alignment(source_mesh: trimesh.Trimesh,
                        target_mesh: trimesh.Trimesh,
                        step_size: float = 1.0,
                        start_gap: float = 5.0,
                        max_distance: float = 60.0) -> tuple:
    """
    Find Y position by sliding from back to front.
    
    Places source maxY behind target minY (with gap), then slides in +Y direction
    until overlap increases sharply.
    
    Args:
        source_mesh: Source mesh (already Z-aligned)
        target_mesh: Target mesh
        step_size: Step size for Y movement (mm)
        start_gap: Gap between source maxY and target minY at start
        max_distance: Maximum distance to scan in +Y direction (mm)
        
    Returns:
        (aligned_mesh, y_offset, overlap_history)
    """
    print(f"\n[Y-Sliding Alignment]")
    
    # Calculate initial offset: place source maxY at target_minY - start_gap
    source_max_y = np.max(source_mesh.vertices[:, 1])
    target_min_y = np.min(target_mesh.vertices[:, 1])
    target_max_y = np.max(target_mesh.vertices[:, 1])
    
    # Start position: source maxY = target minY - start_gap (completely separated)
    start_y = target_min_y - start_gap
    initial_offset = start_y - source_max_y
    
    print(f"  Source maxY: {source_max_y:.1f}mm")
    print(f"  Target Y range: [{target_min_y:.1f}, {target_max_y:.1f}]mm")
    print(f"  Start Y (source maxY at): {start_y:.1f}mm (target minY - {start_gap}mm)")
    print(f"  Initial Y translation: {initial_offset:.1f}mm")
    print(f"  Step size: {step_size:.1f}mm, Max distance: {max_distance:.0f}mm")
    print(f"  Direction: +Y (towards target)")
    
    overlap_history = []
    max_steps = int(max_distance / step_size)
    target_y_offset = initial_offset
    prev_score = None
    cliff_found = False
    increases = []  # Track score increase history
    
    # Track if we've started getting significant proximity
    min_proximity_reached = False
    first_peak_found = False
    
    # Scan from back to front (+Y direction)
    for step in range(max_steps + 1):
        y_offset = initial_offset + step * step_size  # Move in +Y direction
        
        # Create mesh at current offset
        test_mesh = source_mesh.copy()
        test_mesh.apply_translation([0, y_offset, 0])
        
        score = compute_proximity_score(test_mesh, target_mesh, threshold=2.0)
        overlap_history.append((y_offset, score))
        
        # Track when we first get significant proximity
        if score > 0.05:  # 5% threshold
            min_proximity_reached = True
        
        # Calculate increase from previous step
        if prev_score is not None:
            increase = score - prev_score
            increases.append(increase)
            
            print(f"  Y={y_offset:.1f}mm: Proximity={score*100:.1f}% (delta={increase*100:.1f}%)")
            
            # Check for FIRST peak: only after we've reached minimum proximity
            # and the score starts dropping
            if min_proximity_reached and len(increases) >= 2:
                # Peak = score is dropping after it was increasing
                if increase < -0.005:  # -0.5% drop
                    target_y_offset = y_offset - step_size  # Position BEFORE the drop
                    first_peak_found = True
                    print(f"  >>> FIRST PEAK DETECTED!")
                    print(f"     Proximity was rising, now dropping by {-increase*100:.1f}%")
                    print(f"     Using FIRST peak position: Y offset = {target_y_offset:.1f}mm")
                    break
        else:
            print(f"  Y={y_offset:.1f}mm: Proximity={score*100:.1f}%")
        
        prev_score = score
    
    if not first_peak_found:
        # Find the FIRST significant peak in the history
        scores = [h[1] for h in overlap_history]
        for i in range(1, len(scores) - 1):
            if scores[i] > 0.05 and scores[i] > scores[i-1] and scores[i] > scores[i+1]:
                target_y_offset = overlap_history[i][0]
                print(f"  No early peak detected, using first local max: Y offset = {target_y_offset:.1f}mm")
                break
        else:
            # Fallback to max proximity
            max_score_idx = np.argmax(scores)
            target_y_offset = overlap_history[max_score_idx][0]
            print(f"  No peak detected, using max proximity position: Y offset = {target_y_offset:.1f}mm")
    
    # Apply target offset
    result_mesh = source_mesh.copy()
    result_mesh.apply_translation([0, target_y_offset, 0])
    
    final_score = compute_proximity_score(result_mesh, target_mesh, threshold=2.0)
    print(f"\n  >>> Final Y offset: {target_y_offset:.1f}mm")
    print(f"  >>> Final proximity: {final_score*100:.1f}%")
    
    return result_mesh, target_y_offset, overlap_history


def coarse_align(source_mesh: trimesh.Trimesh,
                 target_mesh: trimesh.Trimesh,
                 scale_adjust: bool = False,
                 z_refine: bool = True,
                 visualize: bool = True,
                 output_dir: str = None) -> tuple:
    """
    Full coarse alignment pipeline.
    
    Args:
        source_mesh: Source ROI mesh (extracted from smileArch)
        target_mesh: Target reference mesh (smile_arch_half.stl)
        scale_adjust: Whether to adjust scale
        z_refine: Whether to refine Z alignment
        visualize: Whether to visualize results
        output_dir: Path to save visualizations
        
    Returns:
        (aligned_mesh, alignment_info)
    """
    alignment_info = {
        'translation': np.zeros(3),
        'scale': 1.0,
        'z_offset': 0.0
    }
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    print("\n" + "=" * 50)
    print("[Coarse Alignment]")
    print("=" * 50)
    
    # Step 1: Centroid alignment (XZ only, Y handled by Y-sliding)
    print("\n[Step 1] Centroid Alignment (XZ only)")
    aligned, translation = align_centroids(source_mesh, target_mesh, exclude_y=True)
    alignment_info['translation'] = translation
    
    # Step 2: Scale adjustment (optional)
    if scale_adjust:
        print("\n[Step 2] Scale Adjustment")
        scale = compute_scale_factor(aligned, target_mesh)
        if 0.8 < scale < 1.2:  # Only apply if reasonable
            aligned = apply_scale(aligned, scale, target_mesh.centroid)
            alignment_info['scale'] = scale
        else:
            print(f"  Scale {scale:.2f} out of reasonable range, skipping")
    
    # Step 3: Z refinement (optional)
    if z_refine:
        print("\n[Step 3] Z-Axis Refinement (Bottom Alignment)")
        aligned, z_offset = refine_z_alignment(aligned, target_mesh)
        alignment_info['z_offset'] = z_offset
    
    # Visualize Y-sliding start position
    start_gap = 5.0
    if visualize and output_path:
        # Calculate start position: source maxY at target minY - start_gap
        source_max_y = np.max(aligned.vertices[:, 1])
        target_min_y = np.min(target_mesh.vertices[:, 1])
        start_y = target_min_y - start_gap
        initial_offset = start_y - source_max_y
        
        print(f"\n[Visualizing Y-sliding start position (source maxY at {start_y:.1f}mm)...]")
        
        # Create a temporary mesh at the start position
        temp_mesh = aligned.copy()
        temp_mesh.apply_translation([0, initial_offset, 0])
        visualize_before_y_sliding(temp_mesh, target_mesh, 
                                   save_path=str(output_path / "y_sliding_start.png"))
    
    # Step 4: Y-sliding alignment
    print("\n[Step 4] Y-Sliding Alignment")
    aligned, y_offset, overlap_history = y_sliding_alignment(
        aligned, target_mesh, 
        step_size=1.0, 
        start_gap=start_gap,
        max_distance=60.0
    )
    alignment_info['y_offset'] = y_offset
    alignment_info['overlap_history'] = overlap_history
    
    # Compute alignment quality metrics
    print("\n[Alignment Quality]")
    source_verts = aligned.vertices
    target_verts = target_mesh.vertices
    
    # Simple distance check (centroid to centroid)
    centroid_dist = np.linalg.norm(aligned.centroid - target_mesh.centroid)
    print(f"  Centroid distance: {centroid_dist:.4f}")
    
    # Bounding box overlap
    src_bounds = aligned.bounds
    tgt_bounds = target_mesh.bounds
    
    overlap_min = np.maximum(src_bounds[0], tgt_bounds[0])
    overlap_max = np.minimum(src_bounds[1], tgt_bounds[1])
    overlap_size = np.maximum(overlap_max - overlap_min, 0)
    overlap_volume = np.prod(overlap_size) if np.all(overlap_size > 0) else 0
    
    tgt_volume = np.prod(tgt_bounds[1] - tgt_bounds[0])
    overlap_ratio = overlap_volume / tgt_volume if tgt_volume > 0 else 0
    
    print(f"  Bounding box overlap: {100*overlap_ratio:.1f}%")
    alignment_info['overlap_ratio'] = overlap_ratio
    
    if visualize:
        save_path = str(output_path / "coarse_alignment.png") if output_path else None
        visualize_coarse_alignment(aligned, target_mesh, alignment_info, save_path=save_path)
    
    return aligned, alignment_info


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_before_y_sliding(source_mesh: trimesh.Trimesh,
                               target_mesh: trimesh.Trimesh,
                               save_path: str = None):
    """Visualize state before Y-sliding to debug alignment issues."""
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Before Y-Sliding (after XZ Centroid + Z Alignment)", fontsize=14, fontweight='bold')
    
    def sample_vertices(mesh, n=5000):
        v = mesh.vertices
        if len(v) > n:
            idx = np.random.choice(len(v), n, replace=False)
            return v[idx]
        return v
    
    src_v = sample_vertices(source_mesh)
    tgt_v = sample_vertices(target_mesh)
    
    # Print Y ranges
    src_y_range = [np.min(src_v[:, 1]), np.max(src_v[:, 1])]
    tgt_y_range = [np.min(tgt_v[:, 1]), np.max(tgt_v[:, 1])]
    print(f"  Source Y range: [{src_y_range[0]:.1f}, {src_y_range[1]:.1f}]mm")
    print(f"  Target Y range: [{tgt_y_range[0]:.1f}, {tgt_y_range[1]:.1f}]mm")
    
    # Top view (XY)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(tgt_v[:, 0], tgt_v[:, 1], s=0.5, alpha=0.3, c='blue', label='Reference')
    ax1.scatter(src_v[:, 0], src_v[:, 1], s=0.5, alpha=0.3, c='red', label='Source (before Y-slide)')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y')
    ax1.set_title("Top View (XY)")
    ax1.set_aspect('equal'); ax1.legend(); ax1.grid(True, alpha=0.3)
    
    # Side view (YZ)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(tgt_v[:, 1], tgt_v[:, 2], s=0.5, alpha=0.3, c='blue', label='Reference')
    ax2.scatter(src_v[:, 1], src_v[:, 2], s=0.5, alpha=0.3, c='red', label='Source')
    ax2.set_xlabel('Y'); ax2.set_ylabel('Z')
    ax2.set_title("Side View (YZ)")
    ax2.set_aspect('equal'); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    # Y position comparison
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.barh(['Target', 'Source'], [tgt_y_range[1] - tgt_y_range[0], src_y_range[1] - src_y_range[0]])
    ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Y extent (mm)')
    ax3.set_title(f"Y Ranges\nTarget: [{tgt_y_range[0]:.0f}, {tgt_y_range[1]:.0f}]\nSource: [{src_y_range[0]:.0f}, {src_y_range[1]:.0f}]")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f">>> Saved: {save_path}")
    plt.show()

def visualize_coarse_alignment(source_mesh: trimesh.Trimesh,
                               target_mesh: trimesh.Trimesh,
                               info: dict,
                               save_path: str = None):
    """Visualize coarse alignment result."""
    fig = plt.figure(figsize=(15, 5))
    overlap = info.get('overlap_ratio', 0) * 100
    fig.suptitle(f"Coarse Alignment (Overlap: {overlap:.1f}%)", fontsize=14, fontweight='bold')
    
    def sample_vertices(mesh, n=5000):
        v = mesh.vertices
        if len(v) > n:
            idx = np.random.choice(len(v), n, replace=False)
            return v[idx]
        return v
    
    src_v = sample_vertices(source_mesh)
    tgt_v = sample_vertices(target_mesh)
    
    # 3D overlay
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(tgt_v[:, 0], tgt_v[:, 1], tgt_v[:, 2], s=0.5, alpha=0.3, c='blue', label='Reference')
    ax1.scatter(src_v[:, 0], src_v[:, 1], src_v[:, 2], s=0.5, alpha=0.3, c='red', label='Aligned Source')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title("3D Overlay")
    ax1.legend()
    
    # Top view
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(tgt_v[:, 0], tgt_v[:, 1], s=0.5, alpha=0.3, c='blue', label='Reference')
    ax2.scatter(src_v[:, 0], src_v[:, 1], s=0.5, alpha=0.3, c='red', label='Aligned')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    ax2.set_title("Top View (XY)")
    ax2.set_aspect('equal'); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    # Side view
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(tgt_v[:, 1], tgt_v[:, 2], s=0.5, alpha=0.3, c='blue', label='Reference')
    ax3.scatter(src_v[:, 1], src_v[:, 2], s=0.5, alpha=0.3, c='red', label='Aligned')
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
    from roi_extraction import extract_roi
    
    parser = argparse.ArgumentParser(description='SmileArch Coarse Alignment Test')
    parser.add_argument('--sample', type=int, default=1, help='Sample number (1, 3, 4)')
    parser.add_argument('--no-visualize', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent.parent.parent / "3dmodel"
    sample_path = base_path / f"sample_{args.sample}" / "smileArch.stl"
    ref_path = base_path / "smile_arch_half.stl"
    output_dir = Path(__file__).parent.parent / "output" / f"sample_{args.sample}" / "visualization"
    
    if not sample_path.exists():
        print(f"Error: {sample_path} not found.")
        exit(1)
    
    if not ref_path.exists():
        print(f"Error: {ref_path} not found.")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"SmileArch Coarse Alignment Test - Sample {args.sample}")
    print(f"{'='*60}")
    
    # Step 1: Load and align axis
    print("\n[Loading Source Model]")
    mesh_original, mesh_ds = load_mesh(str(sample_path), downsample_target=10000)
    aligned_mesh, align_info = align_model(mesh_ds, visualize=False)
    
    # Step 2: Extract ROI
    print("\n[Extracting ROI]")
    roi_mesh, roi_info = extract_roi(aligned_mesh, method='front_point', visualize=False)
    print(f"  ROI faces: {len(roi_mesh.faces):,}")
    
    # Step 3: Load reference
    print("\n[Loading Reference Model]")
    ref_mesh = trimesh.load(str(ref_path))
    print(f"  Reference faces: {len(ref_mesh.faces):,}")
    
    # Step 4: Coarse alignment
    coarse_aligned, coarse_info = coarse_align(
        roi_mesh,
        ref_mesh,
        scale_adjust=False,
        z_refine=True,
        visualize=not args.no_visualize,
        output_dir=str(output_dir)
    )
    
    # Result
    print(f"\n{'='*60}")
    print("Coarse Alignment Complete!")
    print(f"{'='*60}")
    print(f"  Translation: {coarse_info['translation']}")
    print(f"  Z offset: {coarse_info['z_offset']:.2f}")
    print(f"  Overlap ratio: {100*coarse_info['overlap_ratio']:.1f}%")
