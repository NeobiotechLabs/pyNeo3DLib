"""
Fine Alignment Module - Welsch ICP
==================================

Uses robust ICP with Welsch loss function for fine alignment.
Welsch loss effectively down-weights outliers based on distance.

Usage:
    from smileArchAlign.modules.fine_alignment import welsch_icp
    aligned, info = welsch_icp(source_mesh, target_mesh)
"""

import numpy as np
import trimesh
import open3d as o3d
from typing import Tuple, Dict, Optional


def filter_icp_faces(source_mesh: trimesh.Trimesh,
                     target_mesh: trimesh.Trimesh,
                     distance_threshold: float = 2.0,
                     y_visible: bool = True) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Filter source AND target mesh faces for ICP.
    
    Ensures both meshes face each other:
    - Source: faces with +Y-ish normals (outer surface), within Z/X range
    - Target: faces with -Y normals (facing towards smileArch)
    
    Args:
        source_mesh: Source mesh (smileArch) to filter
        target_mesh: Target mesh (reference) to filter
        distance_threshold: Max distance to target (mm)
        y_visible: If True, filter by normal direction
        
    Returns:
        (filtered_source, filtered_target) meshes
    """
    from scipy.spatial import cKDTree
    
    print(f"\n  [ICP Face Filtering]")
    
    # === Filter SOURCE (smileArch) ===
    print(f"    Source original: {len(source_mesh.faces):,} faces")
    
    face_centroids = source_mesh.triangles_center
    
    # Step 1: X range filter (-30 to 30mm) - full arch width
    x_mask = (face_centroids[:, 0] >= -20) & (face_centroids[:, 0] <= 20)
    print(f"    Source X in [-30, 30]mm: {x_mask.sum():,} faces")
    
    # Step 2: Z range filter (0 to 15mm) - outer surface area
    z_mask = (face_centroids[:, 2] >= 0) & (face_centroids[:, 2] <= 10)
    print(f"    Source Z in [0, 15]mm: {z_mask.sum():,} faces")
    
    # Step 3: +Y-ish normals (relaxed condition)
    if y_visible:
        src_y_mask = source_mesh.face_normals[:, 1] > -0.5  # Relaxed from > 0
        print(f"    Source +Y-ish visible (>-0.5): {src_y_mask.sum():,} faces")
    else:
        src_y_mask = np.ones(len(source_mesh.faces), dtype=bool)
    
    # Step 4: Distance to reference
    tree = cKDTree(target_mesh.vertices)
    distances, _ = tree.query(face_centroids, k=1)
    nearby_mask = distances < distance_threshold
    print(f"    Source within {distance_threshold}mm: {nearby_mask.sum():,} faces")
    
    # Combine all conditions
    src_final_mask = x_mask & z_mask & src_y_mask & nearby_mask
    print(f"    Source filtered (all conditions): {src_final_mask.sum():,} faces")
    
    # === Filter TARGET (reference) ===
    print(f"    Target original: {len(target_mesh.faces):,} faces")
    
    if y_visible:
        # Target faces with -Y normals (facing smileArch)
        tgt_y_mask = target_mesh.face_normals[:, 1] < 0
        print(f"    Target -Y visible: {tgt_y_mask.sum():,} faces")
        tgt_final_mask = tgt_y_mask
    else:
        tgt_final_mask = np.ones(len(target_mesh.faces), dtype=bool)
    
    # === Create filtered meshes ===
    def extract_submesh(mesh, mask):
        if mask.sum() == 0:
            return mesh.copy()
        filtered_faces = mesh.faces[mask]
        unique_verts = np.unique(filtered_faces.flatten())
        vert_mapping = {old: new for new, old in enumerate(unique_verts)}
        new_vertices = mesh.vertices[unique_verts]
        new_faces = np.array([[vert_mapping[v] for v in face] for face in filtered_faces])
        return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    
    filtered_source = extract_submesh(source_mesh, src_final_mask)
    filtered_target = extract_submesh(target_mesh, tgt_final_mask)
    
    print(f"    Final source: {len(filtered_source.faces):,} faces")
    print(f"    Final target: {len(filtered_target.faces):,} faces")
    
    return filtered_source, filtered_target


def trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.PointCloud:
    """Convert trimesh to Open3D point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    
    # Compute normals if not present
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        pcd.normals = o3d.utility.Vector3dVector(np.array(mesh.vertex_normals).copy())
    else:
        pcd.estimate_normals()
    
    return pcd


def mesh_to_o3d_mesh(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Convert trimesh to Open3D mesh."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def raycast_region_growing_icp(source_mesh: trimesh.Trimesh,
                                target_mesh: trimesh.Trimesh,
                                x_range: tuple = (-10, 10),
                                z_range: tuple = (5, 10),
                                normal_threshold: float = 0.7,  # cos(45°)
                                max_icp_distance: float = 2.0,
                                verbose: bool = True) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Box Region Selection + Region Growing + ICP alignment.
    
    1. Select faces within XZ bounding box (+Y visible only)
    2. Region growing from selected faces based on normal consistency
    3. Run ICP on extracted regions
    
    Args:
        source_mesh: Source mesh (smileArch)
        target_mesh: Target mesh (reference)
        x_range: X range for selection (mm)
        z_range: Z range for selection (mm)
        normal_threshold: Cosine threshold for region growing
        max_icp_distance: Max ICP correspondence distance
        
    Returns:
        (aligned_mesh, info_dict)
    """
    if verbose:
        print("\n" + "="*60)
        print("[Box Region + Region Growing + ICP]")
        print("="*60)
        print(f"  Selection area: X∈[{x_range[0]}, {x_range[1]}], Z∈[{z_range[0]}, {z_range[1]}]mm")
        print(f"  Normal threshold: {normal_threshold} (cos)")
    
    # Step 1: Select faces within XZ box AND +Y-facing (outer surface)
    def select_box_faces(mesh, x_range, z_range):
        """Select faces within XZ bounding box that are +Y visible."""
        centroids = mesh.triangles_center
        normals = mesh.face_normals
        
        # XZ range filter
        x_mask = (centroids[:, 0] >= x_range[0]) & (centroids[:, 0] <= x_range[1])
        z_mask = (centroids[:, 2] >= z_range[0]) & (centroids[:, 2] <= z_range[1])
        
        # +Y visible (outer surface facing +Y)
        y_mask = normals[:, 1] > 0
        
        # Combine all conditions
        mask = x_mask & z_mask & y_mask
        return set(np.where(mask)[0])
    
    # For target, we want -Y facing (facing towards source)
    def select_box_faces_target(mesh, x_range, z_range):
        """Select target faces within XZ box that are -Y visible."""
        centroids = mesh.triangles_center
        normals = mesh.face_normals
        
        x_mask = (centroids[:, 0] >= x_range[0]) & (centroids[:, 0] <= x_range[1])
        z_mask = (centroids[:, 2] >= z_range[0]) & (centroids[:, 2] <= z_range[1])
        y_mask = normals[:, 1] < 0  # -Y facing
        
        mask = x_mask & z_mask & y_mask
        return set(np.where(mask)[0])
    
    src_seed_faces = select_box_faces(source_mesh, x_range, z_range)
    tgt_seed_faces = select_box_faces_target(target_mesh, x_range, z_range)
    
    if verbose:
        print(f"  Source seed faces (in box): {len(src_seed_faces)}")
        print(f"  Target seed faces (in box): {len(tgt_seed_faces)}")
    
    # Step 3: Region growing from hit faces
    def region_growing(mesh, seed_faces, normal_thresh):
        """Grow region from seed faces based on normal consistency."""
        from scipy.sparse import lil_matrix
        
        # Build face adjacency
        n_faces = len(mesh.faces)
        adjacency = lil_matrix((n_faces, n_faces), dtype=bool)
        
        # Create edge to face mapping
        edge_to_face = {}
        for fi, face in enumerate(mesh.faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1)%3]]))
                if edge not in edge_to_face:
                    edge_to_face[edge] = []
                edge_to_face[edge].append(fi)
        
        # Fill adjacency from edge sharing
        for edge, faces in edge_to_face.items():
            for i, f1 in enumerate(faces):
                for f2 in faces[i+1:]:
                    adjacency[f1, f2] = True
                    adjacency[f2, f1] = True
        
        # Get face normals
        normals = mesh.face_normals
        
        # Region growing with max expansion limit
        MAX_EXPANSION = 5  # Max BFS depth/iterations
        MAX_FACES = 500    # Max total faces
        
        selected = set(seed_faces)
        current_level = list(seed_faces)
        
        for level in range(MAX_EXPANSION):
            if len(selected) >= MAX_FACES:
                break
            
            next_level = []
            for current in current_level:
                current_normal = normals[current]
                neighbors = adjacency[current].nonzero()[1]
                
                for neighbor in neighbors:
                    if neighbor not in selected and len(selected) < MAX_FACES:
                        neighbor_normal = normals[neighbor]
                        cos_sim = np.dot(current_normal, neighbor_normal)
                        
                        if cos_sim >= normal_thresh:
                            selected.add(neighbor)
                            next_level.append(neighbor)
            
            current_level = next_level
            if not current_level:
                break
        
        return selected
    
    if len(src_seed_faces) > 0:
        src_region = region_growing(source_mesh, src_seed_faces, normal_threshold)
    else:
        src_region = set(range(len(source_mesh.faces)))
    
    if len(tgt_seed_faces) > 0:
        tgt_region = region_growing(target_mesh, tgt_seed_faces, normal_threshold)
    else:
        tgt_region = set(range(len(target_mesh.faces)))
    
    if verbose:
        print(f"  Source region after growing: {len(src_region)} faces")
        print(f"  Target region after growing: {len(tgt_region)} faces")
    
    # Step 4: Extract submeshes
    def extract_submesh(mesh, face_indices):
        face_mask = np.zeros(len(mesh.faces), dtype=bool)
        face_mask[list(face_indices)] = True
        
        filtered_faces = mesh.faces[face_mask]
        unique_verts = np.unique(filtered_faces.flatten())
        vert_mapping = {old: new for new, old in enumerate(unique_verts)}
        new_vertices = mesh.vertices[unique_verts]
        new_faces = np.array([[vert_mapping[v] for v in face] for face in filtered_faces])
        return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    
    src_submesh = extract_submesh(source_mesh, src_region)
    tgt_submesh = extract_submesh(target_mesh, tgt_region)
    
    # Step 5: ICP on extracted regions
    if verbose:
        print(f"\n  [Running ICP on extracted regions...]")
    
    source_pcd = trimesh_to_o3d(src_submesh)
    target_pcd = trimesh_to_o3d(tgt_submesh)
    
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance=max_icp_distance,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    if verbose:
        print(f"  ICP Fitness: {icp_result.fitness:.4f}")
        print(f"  ICP RMSE: {icp_result.inlier_rmse:.4f}mm")
    
    # Apply transform to original source mesh
    aligned = source_mesh.copy()
    aligned.apply_transform(icp_result.transformation)
    
    info = {
        'src_seed_faces': len(src_seed_faces),
        'tgt_seed_faces': len(tgt_seed_faces),
        'src_region': len(src_region),
        'tgt_region': len(tgt_region),
        'icp_fitness': icp_result.fitness,
        'final_rmse': icp_result.inlier_rmse,
        'final_transformation': icp_result.transformation
    }
    
    return aligned, info


def contour_based_alignment(source_mesh: trimesh.Trimesh,
                            target_mesh: trimesh.Trimesh,
                            x_range: tuple = (-3, 3),
                            z_range: tuple = (-3, 3),
                            rz_range: tuple = (-5, 5),  # Z-axis rotation (yaw)
                            step_trans: float = 0.5,
                            step_rot: float = 1.0,
                            resolution: float = 0.2,  # mm per pixel
                            verbose: bool = True) -> Tuple[trimesh.Trimesh, Dict]:
    """
    2D Contour-based alignment using XZ plane projection.
    
    Projects meshes to XZ plane (view from +Y), extracts contours,
    and finds best alignment by maximizing contour overlap (IOU).
    
    Args:
        source_mesh: Source mesh
        target_mesh: Target mesh
        x_range: X translation range (mm)
        z_range: Z translation range (mm)
        rz_range: Z-axis rotation range (degrees)
        step_trans: Translation step (mm)
        step_rot: Rotation step (degrees)
        resolution: mm per pixel for rasterization
        
    Returns:
        (aligned_mesh, info_dict)
    """
    from itertools import product
    
    if verbose:
        print("\n" + "="*60)
        print("[2D Contour-Based Alignment]")
        print("="*60)
        print(f"  Projection: XZ plane (view from +Y)")
        print(f"  X translation: {x_range[0]} to {x_range[1]}mm")
        print(f"  Z translation: {z_range[0]} to {z_range[1]}mm")
        print(f"  Z rotation: {rz_range[0]} to {rz_range[1]}°")
        print(f"  Resolution: {resolution}mm/pixel")
    
    # Get outer surface points
    # Source: +Y facing (outer surface of smileArch)
    # Target: -Y facing (inner surface facing smileArch)
    src_mask = source_mesh.face_normals[:, 1] > 0
    tgt_mask = target_mesh.face_normals[:, 1] < 0  # Fixed: was > 0, should be < 0
    
    if src_mask.sum() == 0:
        src_verts = source_mesh.vertices
    else:
        src_verts = source_mesh.vertices[np.unique(source_mesh.faces[src_mask].flatten())]
    
    if tgt_mask.sum() == 0:
        tgt_verts = target_mesh.vertices
    else:
        tgt_verts = target_mesh.vertices[np.unique(target_mesh.faces[tgt_mask].flatten())]
    
    if verbose:
        print(f"  Source outer points: {len(src_verts)}")
        print(f"  Target inner points: {len(tgt_verts)}")
    
    # Project to XZ plane (take X and Z coordinates)
    src_xz = src_verts[:, [0, 2]]  # X, Z
    tgt_xz = tgt_verts[:, [0, 2]]  # X, Z
    
    # Compute bounds for rasterization
    all_pts = np.vstack([src_xz, tgt_xz])
    x_min, z_min = np.min(all_pts, axis=0) - 5
    x_max, z_max = np.max(all_pts, axis=0) + 5
    
    # Create image coordinates
    width = int((x_max - x_min) / resolution) + 1
    height = int((z_max - z_min) / resolution) + 1
    
    def points_to_image(pts_xz, offset_x=0, offset_z=0, angle=0):
        """Convert XZ points to binary image after transform."""
        pts = pts_xz.copy()
        
        # Rotation around Z-axis (in XZ plane, this is rotation around origin)
        if angle != 0:
            centroid = np.mean(pts, axis=0)
            pts = pts - centroid
            rad = np.radians(angle)
            c, s = np.cos(rad), np.sin(rad)
            R = np.array([[c, -s], [s, c]])
            pts = pts @ R.T
            pts = pts + centroid
        
        # Translation
        pts[:, 0] += offset_x
        pts[:, 1] += offset_z
        
        # Rasterize to image
        img = np.zeros((height, width), dtype=np.uint8)
        px = ((pts[:, 0] - x_min) / resolution).astype(int)
        pz = ((pts[:, 1] - z_min) / resolution).astype(int)
        
        valid = (px >= 0) & (px < width) & (pz >= 0) & (pz < height)
        img[pz[valid], px[valid]] = 1
        
        return img
    
    # Create target image (fixed)
    tgt_img = points_to_image(tgt_xz)
    
    # Dilate target for overlap computation
    from scipy.ndimage import binary_dilation
    tgt_dilated = binary_dilation(tgt_img, iterations=3)
    
    # Grid search
    x_offsets = np.arange(x_range[0], x_range[1] + step_trans, step_trans)
    z_offsets = np.arange(z_range[0], z_range[1] + step_trans, step_trans)
    rz_angles = np.arange(rz_range[0], rz_range[1] + step_rot, step_rot)
    
    total = len(x_offsets) * len(z_offsets) * len(rz_angles)
    if verbose:
        print(f"  Total combinations: {total}")
    
    best_score = 0
    best_params = {'dx': 0, 'dz': 0, 'rz': 0}
    
    for dx, dz, rz in product(x_offsets, z_offsets, rz_angles):
        src_img = points_to_image(src_xz, dx, dz, rz)
        
        # Compute overlap (intersection)
        intersection = np.sum(src_img & tgt_dilated)
        union = np.sum(src_img | tgt_img)
        
        if union > 0:
            score = intersection / np.sum(src_img) if np.sum(src_img) > 0 else 0
        else:
            score = 0
        
        if score > best_score:
            best_score = score
            best_params = {'dx': dx, 'dz': dz, 'rz': rz}
    
    if verbose:
        print(f"\n  Best params: dx={best_params['dx']:.1f}mm, dz={best_params['dz']:.1f}mm, rz={best_params['rz']:.1f}°")
        print(f"  Best overlap score: {best_score*100:.1f}%")
    
    # Apply best transform to mesh
    aligned = source_mesh.copy()
    verts = aligned.vertices.copy()
    centroid = np.mean(verts, axis=0)
    
    # Rotation around Z-axis
    rad = np.radians(best_params['rz'])
    c, s = np.cos(rad), np.sin(rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    new_verts = (Rz @ (verts - centroid).T).T + centroid
    new_verts[:, 0] += best_params['dx']
    new_verts[:, 2] += best_params['dz']
    aligned.vertices = new_verts
    
    # Build transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = Rz
    transform[0, 3] = best_params['dx']
    transform[2, 3] = best_params['dz']
    
    info = {
        'params': best_params,
        'score': best_score,
        'final_transformation': transform
    }
    
    return aligned, info


def grid_search_icp(source_mesh: trimesh.Trimesh,
                    target_mesh: trimesh.Trimesh,
                    x_range: tuple = (-2, 2),  # mm
                    z_range: tuple = (-2, 2),  # mm
                    rx_range: tuple = (-3, 3),  # degrees (rotation around X)
                    rz_range: tuple = (-3, 3),  # degrees (rotation around Z)
                    step_trans: float = 1.0,  # mm
                    step_rot: float = 1.0,  # degrees
                    verbose: bool = True) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Grid Search + ICP: Find best initial pose then apply ICP.
    
    Args:
        source_mesh: Source mesh
        target_mesh: Target mesh
        x_range: X translation range (min, max) in mm
        z_range: Z translation range (min, max) in mm
        rx_range: Rotation around X range (min, max) in degrees
        rz_range: Rotation around Z range (min, max) in degrees
        step_trans: Translation step size in mm
        step_rot: Rotation step size in degrees
        
    Returns:
        (best_aligned_mesh, info_dict)
    """
    from scipy.spatial import cKDTree
    from itertools import product
    
    if verbose:
        print("\n" + "="*60)
        print("[Grid Search + Global ICP]")
        print("="*60)
        print(f"  X translation: {x_range[0]} to {x_range[1]}mm, step={step_trans}mm")
        print(f"  Z translation: {z_range[0]} to {z_range[1]}mm, step={step_trans}mm")
        print(f"  X rotation: {rx_range[0]} to {rx_range[1]}°, step={step_rot}°")
        print(f"  Z rotation: {rz_range[0]} to {rz_range[1]}°, step={step_rot}°")
    
    # Generate grid
    x_offsets = np.arange(x_range[0], x_range[1] + step_trans, step_trans)
    z_offsets = np.arange(z_range[0], z_range[1] + step_trans, step_trans)
    rx_angles = np.arange(rx_range[0], rx_range[1] + step_rot, step_rot)
    rz_angles = np.arange(rz_range[0], rz_range[1] + step_rot, step_rot)
    
    total_combinations = len(x_offsets) * len(z_offsets) * len(rx_angles) * len(rz_angles)
    if verbose:
        print(f"  Total combinations: {total_combinations}")
    
    # Build KD-tree for target
    target_tree = cKDTree(target_mesh.vertices)
    source_verts = source_mesh.vertices.copy()
    centroid = np.mean(source_verts, axis=0)
    
    best_score = float('inf')
    best_transform = np.eye(4)
    best_params = {}
    
    # Grid search
    for i, (dx, dz, rx_deg, rz_deg) in enumerate(product(x_offsets, z_offsets, rx_angles, rz_angles)):
        # Build transformation matrix
        rx = np.radians(rx_deg)
        rz = np.radians(rz_deg)
        
        cx, sx = np.cos(rx), np.sin(rx)
        cz, sz = np.cos(rz), np.sin(rz)
        
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = Rz @ Rx
        
        # Apply rotation around centroid, then translation
        transformed = (R @ (source_verts - centroid).T).T + centroid
        transformed[:, 0] += dx
        transformed[:, 2] += dz
        
        # Compute mean distance to target
        distances, _ = target_tree.query(transformed, k=1)
        score = np.mean(distances)
        
        if score < best_score:
            best_score = score
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = [dx, 0, dz]
            best_transform = transform
            best_params = {'dx': dx, 'dz': dz, 'rx': rx_deg, 'rz': rz_deg}
    
    if verbose:
        print(f"\n  Best params: dx={best_params['dx']:.1f}mm, dz={best_params['dz']:.1f}mm, "
              f"rx={best_params['rx']:.1f}°, rz={best_params['rz']:.1f}°")
        print(f"  Best mean distance: {best_score:.3f}mm")
    
    # Apply best transform
    best_mesh = source_mesh.copy()
    verts = best_mesh.vertices.copy()
    centroid = np.mean(verts, axis=0)
    
    rx = np.radians(best_params['rx'])
    rz = np.radians(best_params['rz'])
    cx, sx = np.cos(rx), np.sin(rx)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Rx
    
    new_verts = (R @ (verts - centroid).T).T + centroid
    new_verts[:, 0] += best_params['dx']
    new_verts[:, 2] += best_params['dz']
    best_mesh.vertices = new_verts
    
    # Now apply standard ICP on the best position
    if verbose:
        print("\n  [Refining with ICP...]")
    
    # Use Open3D ICP for refinement
    source_pcd = trimesh_to_o3d(best_mesh)
    target_pcd = trimesh_to_o3d(target_mesh)
    
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance=2.0,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    # Apply ICP result
    best_mesh.apply_transform(icp_result.transformation)
    
    # Combine transforms
    final_transform = icp_result.transformation @ best_transform
    
    if verbose:
        print(f"  ICP Fitness: {icp_result.fitness:.4f}")
        print(f"  ICP RMSE: {icp_result.inlier_rmse:.4f}mm")
    
    info = {
        'grid_params': best_params,
        'grid_score': best_score,
        'icp_fitness': icp_result.fitness,
        'final_rmse': icp_result.inlier_rmse,
        'final_transformation': final_transform
    }
    
    return best_mesh, info


def surface_contact_icp(source_mesh: trimesh.Trimesh,
                         target_mesh: trimesh.Trimesh,
                         distance_stages: list = [3.0, 2.0, 1.0, 0.5],
                         iterations_per_stage: int = 50,
                         convergence_threshold: float = 1e-6,
                         verbose: bool = True) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Custom ICP with surface contact constraint.
    
    Ensures source mesh stays on +Y side of target mesh (surfaces touch, don't penetrate).
    Uses correspondence filtering: only pairs where source_y > target_y are used.
    
    Args:
        source_mesh: Source mesh (smileArch)
        target_mesh: Target mesh (reference)
        max_distance: Max correspondence distance
        max_iterations: Maximum iterations
        convergence_threshold: Convergence threshold for RMSE change
        verbose: Print progress
        
    Returns:
        (aligned_mesh, info_dict)
    """
    from scipy.spatial import cKDTree
    
    if verbose:
        print("\n" + "="*60)
        print("[Surface Contact ICP - Multi-Stage]")
        print("="*60)
        print(f"  Source points: {len(source_mesh.vertices):,}")
        print(f"  Target points: {len(target_mesh.vertices):,}")
        print(f"  Distance stages: {distance_stages}mm")
        print(f"  Iterations per stage: {iterations_per_stage}")
        print(f"  Constraint: source must be on +Y side of target")
    
    # Work with point clouds (vertices)
    source_pts = source_mesh.vertices.copy()
    target_pts = target_mesh.vertices.copy()
    
    # Get target Y range
    target_y_min = np.min(target_pts[:, 1])
    target_y_max = np.max(target_pts[:, 1])
    
    if verbose:
        print(f"  Target Y range: [{target_y_min:.1f}, {target_y_max:.1f}]mm")
    
    # Build KD-tree for target
    target_tree = cKDTree(target_pts)
    
    # Initialize transformation
    total_transform = np.eye(4)
    total_iterations = 0
    final_rmse = float('inf')
    
    info = {
        'iterations': 0,
        'final_rmse': 0,
        'final_transformation': np.eye(4),
        'convergence': False,
        'stages': []
    }
    
    # Multi-stage ICP - progressively reduce max_distance
    for stage, max_distance in enumerate(distance_stages):
        if verbose:
            print(f"\n  [Stage {stage+1}] max_distance = {max_distance}mm")
        
        prev_rmse = float('inf')
        stage_converged = False
        
        for iteration in range(iterations_per_stage):
            # Find correspondences
            distances, indices = target_tree.query(source_pts, k=1)
            
            # Filter correspondences:
            # 1. Within max distance
            # 2. Source in target Y range
            # 3. Source Y >= Target Y (surface contact)
            valid_mask = distances < max_distance
            y_range_mask = source_pts[:, 1] >= (target_y_min - 3.0)
            valid_mask = valid_mask & y_range_mask
            
            if valid_mask.sum() > 0:
                src_y = source_pts[valid_mask, 1]
                tgt_y = target_pts[indices[valid_mask], 1]
                contact_mask = src_y >= (tgt_y - 0.3)  # Tighter tolerance
                valid_indices = np.where(valid_mask)[0][contact_mask]
                
                if len(valid_indices) < 10:
                    valid_indices = np.where(valid_mask)[0]
            else:
                break
            
            if len(valid_indices) < 10:
                break
            
            # Get matched points and distances
            src_matched = source_pts[valid_indices]
            tgt_matched = target_pts[indices[valid_indices]]
            matched_distances = distances[valid_indices]
            
            # Welsch weights (close points have more weight)
            k = max_distance / 2
            weights = np.exp(-(matched_distances ** 2) / (k ** 2))
            weights = weights / np.sum(weights)
            
            # Weighted RMSE
            rmse = np.sqrt(np.sum(weights * np.sum((src_matched - tgt_matched)**2, axis=1)))
            
            # Check convergence
            rmse_change = abs(prev_rmse - rmse)
            if rmse_change < convergence_threshold and iteration > 3:
                stage_converged = True
                break
            prev_rmse = rmse
            
            # Weighted SVD for rigid transform
            src_centroid = np.sum(weights[:, np.newaxis] * src_matched, axis=0)
            tgt_centroid = np.sum(weights[:, np.newaxis] * tgt_matched, axis=0)
            
            src_centered = src_matched - src_centroid
            tgt_centered = tgt_matched - tgt_centroid
            
            H = (weights[:, np.newaxis] * src_centered).T @ tgt_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Limit rotation to ±10 degrees per axis
            MAX_ANGLE = np.radians(10)
            
            # Extract Euler angles (XYZ order)
            sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
            if sy > 1e-6:
                rx = np.arctan2(R[2,1], R[2,2])
                ry = np.arctan2(-R[2,0], sy)
                rz = np.arctan2(R[1,0], R[0,0])
            else:
                rx = np.arctan2(-R[1,2], R[1,1])
                ry = np.arctan2(-R[2,0], sy)
                rz = 0
            
            # Clamp angles
            rx = np.clip(rx, -MAX_ANGLE, MAX_ANGLE)
            ry = np.clip(ry, -MAX_ANGLE, MAX_ANGLE)
            rz = np.clip(rz, -MAX_ANGLE, MAX_ANGLE)
            
            # Reconstruct rotation matrix from clamped Euler angles
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            
            t = tgt_centroid - R @ src_centroid
            
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = t
            
            source_pts = (R @ source_pts.T).T + t
            total_transform = transform @ total_transform
            total_iterations += 1
        
        final_rmse = rmse
        
        if verbose:
            status = "Converged" if stage_converged else "Max iter"
            print(f"    {status}: {iteration+1} iters, RMSE={rmse:.4f}mm, Pairs={len(valid_indices)}")
        
        info['stages'].append({
            'max_distance': max_distance,
            'iterations': iteration + 1,
            'rmse': rmse,
            'converged': stage_converged
        })
    
    info['iterations'] = total_iterations
    info['final_rmse'] = final_rmse
    info['final_transformation'] = total_transform
    info['convergence'] = stage_converged
    
    if verbose:
        print(f"\n  >>> Final: {total_iterations} total iterations, RMSE={final_rmse:.4f}mm")
    
    # Apply transformation to original mesh
    aligned_mesh = source_mesh.copy()
    aligned_mesh.apply_transform(total_transform)
    
    return aligned_mesh, info


def welsch_icp(source_mesh: trimesh.Trimesh,
               target_mesh: trimesh.Trimesh,
               max_distance: float = 5.0,
               max_iterations: int = 100,
               k_values: list = [1.0, 0.5, 0.2],
               verbose: bool = True) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Robust ICP using Welsch loss function.
    
    Welsch loss: rho(r) = 1 - exp(-r^2 / (2*k^2))
    - Small residuals: nearly quadratic (like L2)
    - Large residuals: bounded (outliers are down-weighted)
    
    Uses multi-stage refinement with decreasing k values for
    coarse-to-fine registration.
    
    Args:
        source_mesh: Source mesh to align
        target_mesh: Target mesh (reference)
        max_distance: Maximum correspondence distance
        max_iterations: Max iterations per stage
        k_values: Welsch kernel parameter for each stage (decreasing)
        verbose: Print progress
        
    Returns:
        (aligned_mesh, info_dict)
    """
    if verbose:
        print("\n" + "="*60)
        print("[Welsch ICP Fine Alignment]")
        print("="*60)
    
    # Convert to Open3D point clouds
    source_pcd = trimesh_to_o3d(source_mesh)
    target_pcd = trimesh_to_o3d(target_mesh)
    
    if verbose:
        print(f"  Source points: {len(source_pcd.points):,}")
        print(f"  Target points: {len(target_pcd.points):,}")
        print(f"  Max correspondence distance: {max_distance}mm")
        print(f"  Welsch k values: {k_values}")
    
    # Initial transformation (identity)
    transformation = np.eye(4)
    
    info = {
        'stages': [],
        'final_fitness': 0,
        'final_rmse': 0,
        'final_transformation': None
    }
    
    # Multi-stage ICP with decreasing k
    for stage, k in enumerate(k_values, 1):
        if verbose:
            print(f"\n  [Stage {stage}] Welsch k = {k}")
        
        # Create Robust Kernel with Welsch/Leclerc loss
        # Note: Open3D uses TukeyLoss, GMLoss, or L2Loss
        # We'll use TukeyLoss as approximation (also robust)
        loss = o3d.pipelines.registration.TukeyLoss(k=k * max_distance)
        
        # Point-to-plane ICP with robust kernel
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, 
            target_pcd,
            max_correspondence_distance=max_distance,
            init=transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7,
                relative_rmse=1e-7,
                max_iteration=max_iterations
            )
        )
        
        transformation = result.transformation
        
        # Apply transformation to source
        source_pcd.transform(result.transformation)
        
        stage_info = {
            'k': k,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse
        }
        info['stages'].append(stage_info)
        
        if verbose:
            print(f"      Fitness: {result.fitness:.4f}")
            print(f"      Inlier RMSE: {result.inlier_rmse:.4f}mm")
        
        # Reduce max distance for finer stages
        max_distance = max(max_distance * 0.7, 1.0)
    
    # Final transformation
    info['final_fitness'] = result.fitness
    info['final_rmse'] = result.inlier_rmse
    info['final_transformation'] = transformation
    
    # Apply transformation to original trimesh
    aligned_mesh = source_mesh.copy()
    aligned_mesh.apply_transform(transformation)
    
    if verbose:
        print(f"\n  >>> Final fitness: {info['final_fitness']:.4f}")
        print(f"  >>> Final RMSE: {info['final_rmse']:.4f}mm")
    
    return aligned_mesh, info


def compute_fpfh_features(pcd: o3d.geometry.PointCloud, 
                          voxel_size: float = 1.0) -> o3d.pipelines.registration.Feature:
    """
    Compute FPFH (Fast Point Feature Histograms) features for point cloud.
    
    Args:
        pcd: Point cloud with normals
        voxel_size: Voxel size for feature computation
        
    Returns:
        FPFH feature descriptors
    """
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    
    # Estimate normals if not present
    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
    
    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    
    return fpfh


def fpfh_ransac_registration(source_mesh: trimesh.Trimesh,
                              target_mesh: trimesh.Trimesh,
                              voxel_size: float = 1.0,
                              distance_threshold: float = 2.0,
                              verbose: bool = True) -> Tuple[trimesh.Trimesh, Dict]:
    """
    FPFH feature-based RANSAC registration.
    
    Uses FPFH descriptors to find feature correspondences,
    then RANSAC to find robust initial alignment.
    
    Args:
        source_mesh: Source mesh to align
        target_mesh: Target mesh (reference)
        voxel_size: Voxel size for downsampling and feature computation
        distance_threshold: RANSAC inlier distance threshold
        verbose: Print progress
        
    Returns:
        (aligned_mesh, info_dict)
    """
    if verbose:
        print("\n" + "="*60)
        print("[FPFH + RANSAC Feature-Based Registration]")
        print("="*60)
    
    # Convert to Open3D point clouds
    source_pcd = trimesh_to_o3d(source_mesh)
    target_pcd = trimesh_to_o3d(target_mesh)
    
    # Downsample for faster feature computation
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    
    if verbose:
        print(f"  Original: source={len(source_pcd.points)}, target={len(target_pcd.points)}")
        print(f"  Downsampled: source={len(source_down.points)}, target={len(target_down.points)}")
        print(f"  Voxel size: {voxel_size}mm")
    
    # Estimate normals
    radius_normal = voxel_size * 2
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    
    # Compute FPFH features
    if verbose:
        print(f"\n  [Computing FPFH features...]")
    
    source_fpfh = compute_fpfh_features(source_down, voxel_size)
    target_fpfh = compute_fpfh_features(target_down, voxel_size)
    
    if verbose:
        print(f"    Source features: {source_fpfh.data.shape}")
        print(f"    Target features: {target_fpfh.data.shape}")
    
    # RANSAC registration based on feature matching
    if verbose:
        print(f"\n  [RANSAC registration...]")
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    
    info = {
        'fitness': result.fitness,
        'inlier_rmse': result.inlier_rmse,
        'correspondence_set_size': len(result.correspondence_set),
        'transformation': result.transformation
    }
    
    if verbose:
        print(f"    Fitness: {result.fitness:.4f}")
        print(f"    Inlier RMSE: {result.inlier_rmse:.4f}mm")
        print(f"    Correspondences: {len(result.correspondence_set)}")
    
    # Apply transformation to original mesh
    aligned_mesh = source_mesh.copy()
    aligned_mesh.apply_transform(result.transformation)
    
    if verbose:
        print(f"\n  >>> RANSAC complete")
    
    return aligned_mesh, info


def fpfh_ransac_then_icp(source_mesh: trimesh.Trimesh,
                          target_mesh: trimesh.Trimesh,
                          voxel_size: float = 1.0,
                          icp_distance: float = 3.0,
                          k_values: list = [1.0, 0.5, 0.2],
                          verbose: bool = True) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Full pipeline: FPFH RANSAC for initial alignment, then Welsch ICP for refinement.
    
    Args:
        source_mesh: Source mesh to align
        target_mesh: Target mesh (reference)
        voxel_size: Voxel size for FPFH
        icp_distance: Max correspondence distance for ICP
        k_values: Welsch kernel parameters for ICP
        verbose: Print progress
        
    Returns:
        (aligned_mesh, info_dict)
    """
    # Step 1: FPFH + RANSAC
    ransac_aligned, ransac_info = fpfh_ransac_registration(
        source_mesh, target_mesh,
        voxel_size=voxel_size,
        distance_threshold=voxel_size * 2,
        verbose=verbose
    )
    
    # Step 2: Welsch ICP for refinement
    icp_aligned, icp_info = welsch_icp(
        ransac_aligned, target_mesh,
        max_distance=icp_distance,
        k_values=k_values,
        verbose=verbose
    )
    
    # Combine transformations
    combined_transform = icp_info['final_transformation'] @ ransac_info['transformation']
    
    info = {
        'ransac': ransac_info,
        'icp': icp_info,
        'final_transformation': combined_transform,
        'final_fitness': icp_info['final_fitness'],
        'final_rmse': icp_info['final_rmse']
    }
    
    return icp_aligned, info


def welsch_icp_with_y_search(source_mesh: trimesh.Trimesh,
                              target_mesh: trimesh.Trimesh,
                              y_offsets: list = [-2, -1, 0, 1, 2],
                              max_distance: float = 5.0,
                              max_iterations: int = 100,
                              k_values: list = [1.0, 0.5, 0.2],
                              verbose: bool = True) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Welsch ICP with Y-offset grid search.
    
    Tries multiple Y offsets and picks the best result.
    
    Args:
        source_mesh: Source mesh to align
        target_mesh: Target mesh (reference)
        y_offsets: List of Y offsets to try (mm)
        max_distance: Maximum correspondence distance
        max_iterations: Max iterations per stage
        k_values: Welsch kernel parameter for each stage
        verbose: Print progress
        
    Returns:
        (aligned_mesh, info_dict) with best result
    """
    if verbose:
        print("\n" + "="*60)
        print("[Welsch ICP with Y-Offset Grid Search]")
        print("="*60)
        print(f"  Y offsets to try: {y_offsets}mm")
    
    best_result = None
    best_rmse = float('inf')
    best_offset = 0
    results_summary = []
    
    for offset in y_offsets:
        if verbose:
            print(f"\n{'- '*30}")
            print(f"  Testing Y offset: {offset:+.1f}mm")
        
        # Apply Y offset
        test_mesh = source_mesh.copy()
        test_mesh.apply_translation([0, offset, 0])
        
        # Run ICP
        aligned, info = welsch_icp(
            test_mesh, target_mesh,
            max_distance=max_distance,
            max_iterations=max_iterations,
            k_values=k_values,
            verbose=False
        )
        
        # Compute error
        error = compute_alignment_error(aligned, target_mesh)
        
        results_summary.append({
            'offset': offset,
            'fitness': info['final_fitness'],
            'rmse': info['final_rmse'],
            'mean_error': error['mean'],
            'within_2mm': error['within_2mm']
        })
        
        if verbose:
            print(f"    Fitness: {info['final_fitness']:.4f}, RMSE: {info['final_rmse']:.4f}mm")
            print(f"    Mean error: {error['mean']:.2f}mm, Within 2mm: {error['within_2mm']*100:.1f}%")
        
        # Track best (lowest mean error)
        if error['mean'] < best_rmse:
            best_rmse = error['mean']
            best_result = (aligned, info)
            best_offset = offset
            # Update transformation with Y offset included
            y_transform = np.eye(4)
            y_transform[1, 3] = offset
            info['y_offset_applied'] = offset
            info['final_transformation'] = info['final_transformation'] @ y_transform
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  >>> Best Y offset: {best_offset:+.1f}mm")
        print(f"  >>> Best mean error: {best_rmse:.2f}mm")
        print(f"{'='*60}")
    
    best_info = best_result[1]
    best_info['all_results'] = results_summary
    best_info['best_y_offset'] = best_offset
    
    return best_result[0], best_info


def point_to_plane_icp(source_mesh: trimesh.Trimesh,
                       target_mesh: trimesh.Trimesh,
                       max_distance: float = 3.0,
                       max_iterations: int = 50,
                       verbose: bool = True) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Standard Point-to-Plane ICP (no robust kernel).
    
    Faster but less robust to outliers.
    """
    if verbose:
        print("\n[Point-to-Plane ICP]")
    
    source_pcd = trimesh_to_o3d(source_mesh)
    target_pcd = trimesh_to_o3d(target_mesh)
    
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, 
        target_pcd,
        max_correspondence_distance=max_distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=max_iterations
        )
    )
    
    aligned_mesh = source_mesh.copy()
    aligned_mesh.apply_transform(result.transformation)
    
    info = {
        'fitness': result.fitness,
        'inlier_rmse': result.inlier_rmse,
        'transformation': result.transformation
    }
    
    if verbose:
        print(f"  Fitness: {result.fitness:.4f}")
        print(f"  Inlier RMSE: {result.inlier_rmse:.4f}mm")
    
    return aligned_mesh, info


def compute_alignment_error(source_mesh: trimesh.Trimesh,
                           target_mesh: trimesh.Trimesh,
                           max_points: int = 5000) -> Dict:
    """
    Compute alignment error metrics.
    
    Returns:
        Dict with mean, std, median, max distances
    """
    from scipy.spatial import cKDTree
    
    src_pts = source_mesh.vertices
    if len(src_pts) > max_points:
        idx = np.random.choice(len(src_pts), max_points, replace=False)
        src_pts = src_pts[idx]
    
    tree = cKDTree(target_mesh.vertices)
    distances, _ = tree.query(src_pts, k=1)
    
    return {
        'mean': np.mean(distances),
        'std': np.std(distances),
        'median': np.median(distances),
        'max': np.max(distances),
        'min': np.min(distances),
        'within_1mm': np.sum(distances < 1.0) / len(distances),
        'within_2mm': np.sum(distances < 2.0) / len(distances)
    }


# ============================================================================
# Test / Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import matplotlib.pyplot as plt
    
    # Import from parent modules
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from smileArchAlign.modules.axis_alignment import align_model, load_mesh
    from smileArchAlign.modules.roi_extraction import extract_roi
    from smileArchAlign.modules.coarse_alignment import coarse_align
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1)
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent.parent.parent / "3dmodel"
    sample_dir = base_path / f"sample_{args.sample}"
    output_dir = Path(__file__).parent.parent / "output" / f"sample_{args.sample}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"Welsch ICP Test - Sample {args.sample}")
    print("="*60)
    
    # Load source - keep both original and downsampled
    source_path = sample_dir / "smileArch.stl"
    original_mesh, source_mesh = load_mesh(str(source_path), downsample_target=10000)
    
    # Track cumulative transformation
    cumulative_transform = np.eye(4)
    
    # Step 1: Axis alignment (on downsampled mesh)
    print("\n[Step 1] Axis Alignment")
    aligned_mesh, axis_info = align_model(source_mesh, output_dir=str(output_dir / "visualization"))
    
    # Compute transformation from source_mesh to aligned_mesh
    # Using centroid and comparing vertices (rigid transform estimation)
    # Apply same transformation to original_mesh
    src_centroid = source_mesh.centroid
    aligned_centroid = aligned_mesh.centroid
    
    # For rotation: use SVD-based Procrustes on a sample of corresponding points
    # Since vertices are in same order, we can use them directly
    n_sample = min(1000, len(source_mesh.vertices))
    idx = np.linspace(0, len(source_mesh.vertices) - 1, n_sample, dtype=int)
    
    src_pts = source_mesh.vertices[idx] - src_centroid
    aligned_pts = aligned_mesh.vertices[idx] - aligned_centroid
    
    # SVD for rotation
    H = src_pts.T @ aligned_pts
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Translation
    t = aligned_centroid - R @ src_centroid
    
    # Build 4x4 transformation matrix
    axis_transform = np.eye(4)
    axis_transform[:3, :3] = R
    axis_transform[:3, 3] = t
    
    # Apply to original mesh
    original_aligned = original_mesh.copy()
    original_aligned.apply_transform(axis_transform)
    print(f"  Applied axis transform to original mesh")
    
    # Step 2: ROI extraction
    print("\n[Step 2] ROI Extraction")
    roi_mesh, _ = extract_roi(aligned_mesh, output_dir=str(output_dir / "visualization"))
    
    # Load reference
    ref_path = base_path / "smile_arch_half.stl"
    ref_mesh = trimesh.load(str(ref_path))
    print(f"\n[Reference] {len(ref_mesh.faces)} faces")
    
    # Step 3: Coarse alignment
    print("\n[Step 3] Coarse Alignment")
    coarse_mesh, coarse_info = coarse_align(
        roi_mesh, ref_mesh, 
        output_dir=str(output_dir / "visualization")
    )
    
    # Build coarse translation matrix
    coarse_translation = np.eye(4)
    if 'translation' in coarse_info:
        coarse_translation[:3, 3] = coarse_info['translation']
    if 'z_offset' in coarse_info:
        coarse_translation[2, 3] += coarse_info['z_offset']
    if 'y_offset' in coarse_info:
        coarse_translation[1, 3] += coarse_info['y_offset']
    
    # Step 4: Check for outliers (+Y direction) and selective ICP
    print("\n[Step 4] Inverse ICP (Reference → Source)")
    print("="*60)
    
    # Check if source has parts far away in +Y direction
    from scipy.spatial import cKDTree
    
    ref_tree = cKDTree(ref_mesh.vertices)
    distances, _ = ref_tree.query(coarse_mesh.vertices, k=1)
    
    # Find vertices that are >10mm away AND in +Y direction relative to reference
    ref_max_y = ref_mesh.vertices[:, 1].max()
    outlier_mask = (coarse_mesh.vertices[:, 1] > ref_max_y + 10)  # +10mm beyond reference
    outlier_ratio = outlier_mask.sum() / len(coarse_mesh.vertices)
    
    print(f"  Outlier check: {outlier_mask.sum()} vertices >10mm in +Y direction ({outlier_ratio*100:.1f}%)")
    
    # Prepare meshes for ICP
    source_for_icp = coarse_mesh
    ref_for_icp = ref_mesh
    
    if outlier_ratio > 0.005:  # More than 0.5% outliers
        print(f"  >>> Outliers detected! Using XZ 20-80% region only")
        
        # Filter reference to Z 20-80%
        ref_z_min, ref_z_max = ref_mesh.vertices[:, 2].min(), ref_mesh.vertices[:, 2].max()
        z_range = ref_z_max - ref_z_min
        z_low = ref_z_min + z_range * 0.2
        z_high = ref_z_min + z_range * 0.8
        
        # Filter reference to X 20-80%
        ref_x_min, ref_x_max = ref_mesh.vertices[:, 0].min(), ref_mesh.vertices[:, 0].max()
        x_range = ref_x_max - ref_x_min
        x_low = ref_x_min + x_range * 0.2
        x_high = ref_x_min + x_range * 0.8
        
        # Filter source faces in XZ range
        src_centroids = coarse_mesh.triangles_center
        src_z_mask = (src_centroids[:, 2] >= z_low) & (src_centroids[:, 2] <= z_high)
        src_x_mask = (src_centroids[:, 0] >= x_low) & (src_centroids[:, 0] <= x_high)
        src_mask = src_z_mask & src_x_mask
        
        # Filter reference faces in XZ range
        ref_centroids = ref_mesh.triangles_center
        ref_z_mask = (ref_centroids[:, 2] >= z_low) & (ref_centroids[:, 2] <= z_high)
        ref_x_mask = (ref_centroids[:, 0] >= x_low) & (ref_centroids[:, 0] <= x_high)
        ref_mask = ref_z_mask & ref_x_mask
        
        print(f"  X range: [{x_low:.1f}, {x_high:.1f}]mm")
        print(f"  Z range: [{z_low:.1f}, {z_high:.1f}]mm")
        print(f"  Source faces in range: {src_mask.sum()} / {len(coarse_mesh.faces)}")
        print(f"  Reference faces in range: {ref_mask.sum()} / {len(ref_mesh.faces)}")
        
        # Extract filtered submeshes
        def extract_faces_by_mask(mesh, mask):
            filtered_faces = mesh.faces[mask]
            if len(filtered_faces) == 0:
                return mesh  # Return original if no faces match
            unique_verts = np.unique(filtered_faces.flatten())
            vert_mapping = {old: new for new, old in enumerate(unique_verts)}
            new_vertices = mesh.vertices[unique_verts]
            new_faces = np.array([[vert_mapping[v] for v in face] for face in filtered_faces])
            return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        
        source_for_icp = extract_faces_by_mask(coarse_mesh, src_mask)
        ref_for_icp = extract_faces_by_mask(ref_mesh, ref_mask)
    
    # Convert to point clouds
    source_pcd = trimesh_to_o3d(source_for_icp)  
    target_pcd = trimesh_to_o3d(ref_for_icp)
    
    # ICP: reference → source (swap source and target)
    icp_result = o3d.pipelines.registration.registration_icp(
        target_pcd, source_pcd,  # Reference aligns to Source
        max_correspondence_distance=3.0,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    print(f"  ICP Fitness: {icp_result.fitness:.4f}")
    print(f"  ICP RMSE: {icp_result.inlier_rmse:.4f}mm")
    
    # Invert the transformation
    icp_transform = np.linalg.inv(icp_result.transformation)
    print(f"  Applied inverse transform to source")
    
    # Apply ICP to coarse mesh for visualization
    fine_mesh = coarse_mesh.copy()
    fine_mesh.apply_transform(icp_transform)
    
    icp_info = {
        'final_transformation': icp_transform,
        'icp_fitness': icp_result.fitness,
        'final_rmse': icp_result.inlier_rmse,
        'outlier_detected': outlier_ratio > 0.1
    }
    
    # Apply coarse + ICP transform to original aligned mesh
    full_transform = icp_info['final_transformation'] @ coarse_translation
    original_final = original_aligned.copy()
    original_final.apply_transform(full_transform)
    
    # Compute error metrics
    print("\n[Alignment Error Analysis]")
    error = compute_alignment_error(fine_mesh, ref_mesh)
    print(f"  Mean distance: {error['mean']:.3f}mm")
    print(f"  Std: {error['std']:.3f}mm")
    print(f"  Median: {error['median']:.3f}mm")
    print(f"  Max: {error['max']:.3f}mm")
    print(f"  Within 1mm: {error['within_1mm']*100:.1f}%")
    print(f"  Within 2mm: {error['within_2mm']*100:.1f}%")
    
    # Visualize
    fig = plt.figure(figsize=(15, 5))
    
    def sample_verts(mesh, n=5000):
        v = mesh.vertices
        if len(v) > n:
            idx = np.random.choice(len(v), n, replace=False)
            return v[idx]
        return v
    
    src_v = sample_verts(fine_mesh)
    ref_v = sample_verts(ref_mesh)
    
    # Top view
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(ref_v[:, 0], ref_v[:, 1], s=0.5, alpha=0.3, c='blue', label='Reference')
    ax1.scatter(src_v[:, 0], src_v[:, 1], s=0.5, alpha=0.3, c='red', label='Aligned')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y')
    ax1.set_title("Top View (XY)")
    ax1.set_aspect('equal'); ax1.legend()
    
    # Side view
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(ref_v[:, 1], ref_v[:, 2], s=0.5, alpha=0.3, c='blue', label='Reference')
    ax2.scatter(src_v[:, 1], src_v[:, 2], s=0.5, alpha=0.3, c='red', label='Aligned')
    ax2.set_xlabel('Y'); ax2.set_ylabel('Z')
    ax2.set_title("Side View (YZ)")
    ax2.set_aspect('equal'); ax2.legend()
    
    # Front view
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(ref_v[:, 0], ref_v[:, 2], s=0.5, alpha=0.3, c='blue', label='Reference')
    ax3.scatter(src_v[:, 0], src_v[:, 2], s=0.5, alpha=0.3, c='red', label='Aligned')
    ax3.set_xlabel('X'); ax3.set_ylabel('Z')
    ax3.set_title("Front View (XZ)")
    ax3.set_aspect('equal'); ax3.legend()
    
    fig.suptitle(f"Welsch ICP Result (RMSE: {error['mean']:.2f}mm)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / "visualization" / "welsch_icp_result.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"\n>>> Saved: {save_path}")
    plt.show()
    
    # Save aligned ORIGINAL full mesh (not ROI)
    out_mesh_path = output_dir / "smileArch_aligned.stl"
    original_final.export(str(out_mesh_path))
    print(f">>> Saved FULL mesh: {out_mesh_path}")
    print(f"    Vertices: {len(original_final.vertices):,}, Faces: {len(original_final.faces):,}")
    
    # Also save final transformation matrix
    transform_path = output_dir / "final_transform.npy"
    np.save(str(transform_path), full_transform)
    print(f">>> Saved transform: {transform_path}")
    
    print("\n" + "="*60)
    print("Fine Alignment Complete!")
    print("="*60)
