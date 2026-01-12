"""
IOS Upper/Lower Registration Module
====================================

상악(Upper)과 하악(Lower)을 SmileArch와 함께 정합합니다.

접근 방식 (수학적):
1. T1_arch, T1_upper, T1_lower = align_dental_set_v4 알고리즘으로 각각 정규화
   → 이 상태에서 3개 메시는 서로 정합된 관계 유지
2. T2 = ios_laminate_result (SmileArch의 최종 목표 위치)
3. T3 = T2 × T1_arch⁻¹ (정규화된 SmileArch → 최종 위치)
4. 최종 상악 = T3 × T1_upper, 최종 하악 = T3 × T1_lower

Author: Antigravity Assistant
Date: 2026-01-09
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree, ConvexHull
from scipy.optimize import minimize_scalar

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from pyNeo3DLib.fileLoader.mesh import Mesh


# ============================================================
# align_dental_set_v4.py 알고리즘 (전체 세트 정렬)
# ============================================================

def get_2d_convex_hull_points(vertices: np.ndarray, num_points: int = 200) -> np.ndarray:
    """+Z 방향에서 본 2D Convex Hull 외곽선 점들을 추출합니다."""
    points_2d = vertices[:, :2]
    hull = ConvexHull(points_2d)
    hull_vertices = points_2d[hull.vertices]
    
    hull_closed = np.vstack([hull_vertices, hull_vertices[0]])
    distances = np.sqrt(np.sum(np.diff(hull_closed, axis=0) ** 2, axis=1))
    cumulative = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative[-1]
    
    sample_distances = np.linspace(0, total_length, num_points, endpoint=False)
    sampled_points = []
    
    for d in sample_distances:
        idx = np.searchsorted(cumulative, d) - 1
        idx = max(0, min(idx, len(hull_vertices) - 1))
        next_idx = (idx + 1) % len(hull_vertices)
        
        segment_start = cumulative[idx]
        segment_length = distances[idx]
        t = (d - segment_start) / segment_length if segment_length > 0 else 0
        
        point = hull_vertices[idx] * (1 - t) + hull_vertices[next_idx] * t
        sampled_points.append(point)
    
    return np.array(sampled_points)


def compute_symmetry_score(hull_points_2d: np.ndarray, angle_degrees: float) -> float:
    """2D Convex Hull 기반 반사 대칭 점수를 계산합니다."""
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated_points = hull_points_2d @ rotation_2d.T
    reflected_points = rotated_points.copy()
    reflected_points[:, 0] = -reflected_points[:, 0]
    
    tree = cKDTree(rotated_points)
    distances, _ = tree.query(reflected_points, k=1)
    
    return np.sqrt(np.mean(distances ** 2))


def find_symmetry_axis(vertices: np.ndarray) -> Tuple[float, np.ndarray]:
    """대칭축 각도를 찾습니다."""
    hull_points_2d = get_2d_convex_hull_points(vertices, num_points=200)
    
    angles = np.arange(0, 180, 10)
    scores = [compute_symmetry_score(hull_points_2d, a) for a in angles]
    
    best_coarse_angle = angles[np.argmin(scores)]
    
    search_min = max(0, best_coarse_angle - 10)
    search_max = min(180, best_coarse_angle + 10)
    
    result = minimize_scalar(
        lambda angle: compute_symmetry_score(hull_points_2d, angle),
        bounds=(search_min, search_max),
        method='bounded',
        options={'xatol': 0.1}
    )
    
    return result.x, hull_points_2d


def compute_average_width(hull_points: np.ndarray, y_positive: bool = True) -> float:
    """상/하반부의 평균 폭을 계산합니다."""
    y_values = hull_points[:, 1]
    y_median = np.median(y_values)
    
    mask = y_values > y_median if y_positive else y_values < y_median
    selected_points = hull_points[mask]
    
    if len(selected_points) < 2:
        return 0.0
    
    y_min, y_max = np.min(selected_points[:, 1]), np.max(selected_points[:, 1])
    n_bins = 5
    widths = []
    
    for i in range(n_bins):
        y_low = y_min + (y_max - y_min) * i / n_bins
        y_high = y_min + (y_max - y_min) * (i + 1) / n_bins
        bin_mask = (selected_points[:, 1] >= y_low) & (selected_points[:, 1] < y_high)
        bin_points = selected_points[bin_mask]
        
        if len(bin_points) >= 2:
            width = np.max(bin_points[:, 0]) - np.min(bin_points[:, 0])
            widths.append(width)
    
    return np.mean(widths) if widths else 0.0


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """면 법선 벡터를 계산합니다."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return normals / norms


def compute_z_curvature(vertices: np.ndarray, z_positive: bool = True) -> float:
    """메쉬의 +Z 또는 -Z 영역의 볼록함을 측정합니다."""
    y_range = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    y_center = np.median(vertices[:, 1])
    y_tolerance = y_range * 0.2
    
    center_mask = np.abs(vertices[:, 1] - y_center) < y_tolerance
    center_vertices = vertices[center_mask]
    
    if len(center_vertices) < 10:
        return 0.0
    
    z_median = np.median(center_vertices[:, 2])
    
    if z_positive:
        z_mask = center_vertices[:, 2] > z_median
    else:
        z_mask = center_vertices[:, 2] < z_median
    
    selected_vertices = center_vertices[z_mask]
    
    if len(selected_vertices) < 5:
        return 0.0
    
    z_std = np.std(selected_vertices[:, 2])
    z_extent = np.abs(np.max(selected_vertices[:, 2]) - z_median) if z_positive else np.abs(z_median - np.min(selected_vertices[:, 2]))
    
    return z_std * z_extent


def compute_alignment_transform(
    vertices: np.ndarray, 
    faces: np.ndarray, 
    model_type: str,
    verbose: bool = False
) -> np.ndarray:
    """OBB 기반으로 메시를 정규화하는 변환 행렬을 계산합니다."""
    if not HAS_TRIMESH:
        return np.eye(4)
    
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy())
    total_transform = np.eye(4)
    
    # 1단계: OBB 기반 Z축 정렬
    obb = mesh.bounding_box_oriented
    obb_extents = obb.primitive.extents
    obb_transform = obb.primitive.transform
    
    shortest_axis = np.argmin(obb_extents)
    obb_rotation = obb_transform[:3, :3]
    obb_axes = obb_rotation.T
    shortest_axis_direction = obb_axes[shortest_axis]
    
    target_direction = np.array([0, 0, 1])
    v = np.cross(shortest_axis_direction, target_direction)
    c = np.dot(shortest_axis_direction, target_direction)
    
    if np.linalg.norm(v) < 1e-6:
        rotation_matrix = np.eye(3) if c > 0 else Rotation.from_euler('x', 180, degrees=True).as_matrix()
    else:
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    total_transform = transform @ total_transform
    mesh.apply_transform(transform)
    
    # 중심 이동
    center = mesh.centroid
    translation = np.eye(4)
    translation[:3, 3] = -center
    total_transform = translation @ total_transform
    mesh.apply_translation(-center)
    
    # 2단계: 모델 타입별 Z축 방향 조정
    face_normals = compute_face_normals(mesh.vertices, mesh.faces)
    avg_normal_z = np.mean(face_normals[:, 2])
    need_flip = False
    
    if model_type == 'upper' and avg_normal_z > 0:
        need_flip = True
    elif model_type == 'lower' and avg_normal_z < 0:
        need_flip = True
    elif model_type == 'smileArch':
        curvature_positive = compute_z_curvature(mesh.vertices, z_positive=True)
        curvature_negative = compute_z_curvature(mesh.vertices, z_positive=False)
        if curvature_negative > curvature_positive:
            need_flip = True
    
    if need_flip:
        flip_rotation = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = flip_rotation
        total_transform = transform @ total_transform
        mesh.apply_transform(transform)
    
    # 3단계: 대칭축 정렬
    optimal_angle, hull_points_2d = find_symmetry_axis(mesh.vertices)
    if verbose:
        print(f"    대칭축 각도: {optimal_angle:.2f}°")
    
    rotation = Rotation.from_euler('z', optimal_angle, degrees=True).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    total_transform = transform @ total_transform
    mesh.apply_transform(transform)
    
    # hull 점도 회전 적용
    angle_rad = np.radians(optimal_angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    aligned_hull = hull_points_2d @ rotation_2d.T
    
    # 4단계: Y축 방향 정렬 (앞쪽이 +Y)
    width_positive = compute_average_width(aligned_hull, y_positive=True)
    width_negative = compute_average_width(aligned_hull, y_positive=False)
    
    if width_positive > width_negative:
        if verbose:
            print(f"    Y축 방향 180° 회전 적용")
        rotation = Rotation.from_euler('z', 180, degrees=True).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        total_transform = transform @ total_transform
    
    return total_transform


def close_gap_to_contact(
    source_vertices: np.ndarray, 
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    is_upper: bool,
    verbose: bool = False
) -> Optional[float]:
    """Source 메쉬를 Target 방향으로 이동시킬 거리를 계산합니다."""
    if not HAS_TRIMESH:
        return None
    
    target_mesh = trimesh.Trimesh(vertices=target_vertices, faces=target_faces)
    
    direction = np.array([0, 0, -1]) if is_upper else np.array([0, 0, 1])
    safety_lift = 50.0
    lift_vec = -direction * safety_lift
    
    t_bounds = target_mesh.bounds
    mask_x = (source_vertices[:, 0] >= t_bounds[0][0]) & (source_vertices[:, 0] <= t_bounds[1][0])
    mask_y = (source_vertices[:, 1] >= t_bounds[0][1]) & (source_vertices[:, 1] <= t_bounds[1][1])
    mask = mask_x & mask_y
    
    candidates = source_vertices[mask]
    
    if len(candidates) == 0:
        if verbose:
            print(f"    {'상악' if is_upper else '하악'}: 겹치는 XY 영역이 없어 스킵합니다.")
        return None
    
    if len(candidates) > 5000:
        idx = np.random.choice(len(candidates), 5000, replace=False)
        origins = candidates[idx]
    else:
        origins = candidates
    
    origins_lifted = origins + lift_vec
    vectors = np.tile(direction, (len(origins_lifted), 1))
    
    locations, index_ray, _ = target_mesh.ray.intersects_location(
        ray_origins=origins_lifted,
        ray_directions=vectors,
        multiple_hits=False
    )
    
    if len(locations) == 0:
        if verbose:
            print(f"    {'상악' if is_upper else '하악'}: 충돌 지점을 찾을 수 없습니다.")
        return None
    
    matched_origins = origins_lifted[index_ray]
    dists = np.linalg.norm(matched_origins - locations, axis=1)
    real_gaps = dists - safety_lift
    
    contact_gap = np.percentile(real_gaps, 1)
    final_move = contact_gap - 0.1
    
    if verbose:
        print(f"    {'상악' if is_upper else '하악'}: {final_move:+.2f}mm 이동 (Contact)")
    
    return final_move


def welsch_weight(residuals: np.ndarray, sigma: float) -> np.ndarray:
    """Welsch (Leclerc) 가중치 함수"""
    return np.exp(-residuals**2 / (2 * sigma**2))


def run_welsch_robust_icp(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    initial_transform: np.ndarray,
    max_iterations: int = 30,
    sigma: float = 3.0,
    tolerance: float = 0.001,
    verbose: bool = True
) -> np.ndarray:
    """Welsch 함수 기반 Robust ICP (원본 align_dental_set_v4.py에서 가져옴)"""
    target_tree = cKDTree(target_pts)
    current_pts = source_pts.copy()
    accumulated_transform = initial_transform.copy()
    
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        distances, indices = target_tree.query(current_pts, k=1)
        correspondences = target_pts[indices]
        
        weights = welsch_weight(distances, sigma)
        inlier_mask = weights > 0.1
        n_inliers = np.sum(inlier_mask)
        
        if n_inliers < 10:
            if verbose:
                print(f"      경고: 너무 적은 inlier ({n_inliers}개)")
            break
        
        weighted_error = np.sum(weights * distances) / np.sum(weights)
        
        if abs(prev_error - weighted_error) < tolerance:
            if verbose:
                print(f"      수렴 (iteration {iteration+1})")
            break
        prev_error = weighted_error
        
        w = weights[inlier_mask]
        src = current_pts[inlier_mask]
        tgt = correspondences[inlier_mask]
        
        w_sum = np.sum(w)
        src_centroid = np.sum(w[:, np.newaxis] * src, axis=0) / w_sum
        tgt_centroid = np.sum(w[:, np.newaxis] * tgt, axis=0) / w_sum
        
        src_centered = src - src_centroid
        tgt_centered = tgt - tgt_centroid
        
        H = (w[:, np.newaxis] * src_centered).T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = tgt_centroid - R @ src_centroid
        
        T_iter = np.eye(4)
        T_iter[:3, :3] = R
        T_iter[:3, 3] = t
        
        current_pts = (R @ current_pts.T).T + t
        accumulated_transform = T_iter @ accumulated_transform
    
    # 최종 통계
    final_distances, _ = target_tree.query(current_pts, k=1)
    final_weights = welsch_weight(final_distances, sigma)
    inlier_ratio = np.sum(final_weights > 0.5) / len(final_weights)
    weighted_rmse = np.sqrt(np.sum(final_weights * final_distances**2) / np.sum(final_weights))
    
    if verbose:
        print(f"      Welsch ICP: inlier={inlier_ratio*100:.1f}%, RMSE={weighted_rmse:.2f} (σ={sigma}mm)")
    
    return accumulated_transform


def run_rotation_limited_icp(
    source_mesh: "trimesh.Trimesh",
    target_mesh: "trimesh.Trimesh",
    max_rotation: float = 10.0,
    max_translation: float = 40.0,
    is_upper: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    Y-슬라이딩 + 회전 제한 ICP (원본 align_dental_set_v4.py에서 가져옴)
    
    +Y 방향에서 시작해서 -Y 방향으로 슬라이딩하며 최적 위치 탐색.
    """
    start_time = time.time()
    
    # Target: 법선 방향으로 올바른 면 선택
    target_normals = target_mesh.face_normals
    target_centroids = target_mesh.triangles_center
    
    if is_upper:
        face_mask = target_normals[:, 2] > 0.5  # +Z 법선 = 상악이 들어갈 자리
    else:
        face_mask = target_normals[:, 2] < -0.5  # -Z 법선 = 하악이 들어갈 자리
    
    target_pts = target_centroids[face_mask]
    
    # 소스 샘플링
    source_pts = np.array(source_mesh.vertices)
    n_samples = 5000
    
    if len(source_pts) > n_samples:
        idx = np.random.choice(len(source_pts), n_samples, replace=False)
        source_pts = source_pts[idx]
    if len(target_pts) > n_samples:
        idx = np.random.choice(len(target_pts), n_samples, replace=False)
        target_pts = target_pts[idx]
    
    # 교합면 영역만 선택 (source)
    if is_upper:
        z_threshold = np.percentile(source_pts[:, 2], 25)  # 상악 아래쪽
        source_pts = source_pts[source_pts[:, 2] <= z_threshold]
    else:
        z_threshold = np.percentile(source_pts[:, 2], 85)  # 하악 위쪽
        source_pts = source_pts[source_pts[:, 2] >= z_threshold]
    
    if verbose:
        print(f"    샘플: source={len(source_pts)}, target={len(target_pts)}")
    
    target_tree = cKDTree(target_pts)
    source_center = np.mean(source_pts, axis=0)
    
    best_params = {'rx': 0, 'ry': 0, 'rz': 0, 'ty': 0, 'tz': 0}
    best_score = float('inf')
    
    y_scores = {}
    
    if is_upper:
        # 상악: 단순 거리 기반 슬라이딩
        if verbose:
            print(f"    1단계: Y-슬라이딩 탐색 (+5 → -40mm)...")
        
        for ty in np.linspace(5, -40, 19):  # +5 → -40mm, 2.5mm step
            best_z_score = float('inf')
            
            for tz in np.linspace(-10, 10, 11):
                test_pts = source_pts.copy()
                test_pts[:, 1] += ty
                test_pts[:, 2] += tz
                
                distances, _ = target_tree.query(test_pts, k=1)
                score = np.mean(distances)
                
                if score < best_z_score:
                    best_z_score = score
                
                if score < best_score:
                    best_score = score
                    best_params['ty'] = ty
                    best_params['tz'] = tz
            
            y_scores[ty] = best_z_score
    else:
        # 하악: Welsch ICP 기반 슬라이딩 (2.5mm 스텝)
        if verbose:
            print(f"    1단계: Y-슬라이딩 + ICP 탐색 (+5 → -25mm, 2.5mm step)...")
        
        for ty in np.linspace(5, -20, 13):  # +5 → -25mm, 2.5mm step (30mm 범위)
            best_z_score = float('inf')
            
            for tz in np.linspace(-10, 10, 5):
                test_pts = source_pts.copy()
                test_pts[:, 1] += ty
                test_pts[:, 2] += tz
                
                # Welsch ICP 실행 (조용히)
                T_offset = np.eye(4)
                T_offset[1, 3] = ty
                T_offset[2, 3] = tz
                
                T_test = run_welsch_robust_icp(
                    test_pts, target_pts,
                    T_offset, max_iterations=10, verbose=False
                )
                
                # ICP 후 거리 계산
                transformed = (T_test @ np.hstack([test_pts, np.ones((len(test_pts), 1))]).T).T[:, :3]
                dists, _ = target_tree.query(transformed, k=1)
                weights = welsch_weight(dists, sigma=3.0)
                score = np.sqrt(np.sum(weights * dists**2) / np.sum(weights))
                
                if score < best_z_score:
                    best_z_score = score
                
                if score < best_score:
                    best_score = score
                    best_params['ty'] = ty
                    best_params['tz'] = tz
            
            y_scores[ty] = best_z_score
    
    if verbose:
        print(f"      → Best Y={best_params['ty']:.1f}mm, Z={best_params['tz']:.1f}mm, score={best_score:.2f}")
    
    # 2단계: 회전 탐색
    if verbose:
        print(f"    2단계: 회전 탐색 (±{max_rotation}°)...")
    
    # 원본은 linspace(..,.., 1)로 회전 탐색을 스킵함. 여기서는 간단히 탐색
    for rx in np.linspace(-max_rotation, max_rotation, 5):
        for ry in np.linspace(-max_rotation, max_rotation, 5):
            for rz in np.linspace(-max_rotation, max_rotation, 5):
                R = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
                
                pts_centered = source_pts - source_center
                pts_rotated = pts_centered @ R.T + source_center
                pts_rotated[:, 1] += best_params['ty']
                pts_rotated[:, 2] += best_params['tz']
                
                distances, _ = target_tree.query(pts_rotated, k=1)
                score = np.mean(distances)
                
                if score < best_score:
                    best_score = score
                    best_params['rx'] = rx
                    best_params['ry'] = ry
                    best_params['rz'] = rz
    
    if verbose:
        print(f"      R=[{best_params['rx']:.1f}°, {best_params['ry']:.1f}°, {best_params['rz']:.1f}°]")
        print(f"      T=[0, {best_params['ty']:.1f}, {best_params['tz']:.1f}]mm, score={best_score:.2f}")
    
    # 변환 행렬 구성
    R = Rotation.from_euler('xyz', [best_params['rx'], best_params['ry'], best_params['rz']], degrees=True).as_matrix()
    
    T1 = np.eye(4)
    T1[:3, 3] = -source_center
    
    R_mat = np.eye(4)
    R_mat[:3, :3] = R
    
    T2 = np.eye(4)
    T2[:3, 3] = source_center
    
    T_offset = np.eye(4)
    T_offset[:3, 3] = [0, best_params['ty'], best_params['tz']]
    
    grid_transform = T_offset @ T2 @ R_mat @ T1
    
    # 3단계: 최종 ICP 정밀 정합
    source_transformed = (grid_transform @ np.hstack([source_pts, np.ones((len(source_pts), 1))]).T).T[:, :3]
    
    if is_upper and HAS_OPEN3D:
        # 상악: Open3D ICP → Welsch ICP
        if verbose:
            print(f"    3단계: 최종 ICP 정밀 정합...")
        
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_transformed)
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_pts)
        
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 3.0,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        if verbose:
            print(f"      ICP Fitness: {icp_result.fitness:.4f}, RMSE: {icp_result.inlier_rmse:.2f}")
        
        icp_transform = icp_result.transformation @ grid_transform
        
        # Welsch ICP 정밀 정합
        if verbose:
            print(f"    4단계: Welsch Robust ICP 정밀 정합...")
        
        source_after_icp = (icp_transform @ np.hstack([source_pts, np.ones((len(source_pts), 1))]).T).T[:, :3]
        final_transform = run_welsch_robust_icp(
            source_after_icp, target_pts,
            icp_transform, verbose=verbose
        )
    else:
        # 하악: Welsch ICP만
        if verbose:
            print(f"    3단계: Welsch Robust ICP 정밀 정합...")
        
        final_transform = run_welsch_robust_icp(
            source_transformed, target_pts,
            grid_transform, verbose=verbose
        )
    
    if verbose:
        print(f"    완료 ({time.time() - start_time:.2f}s)")
    
    return final_transform


def align_dental_set(
    upper_vertices: np.ndarray,
    upper_faces: np.ndarray,
    lower_vertices: np.ndarray,
    lower_faces: np.ndarray,
    arch_vertices: np.ndarray,
    arch_faces: np.ndarray,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    align_dental_set_v4.py의 전체 정렬 알고리즘
    """
    if not HAS_TRIMESH:
        return {'upper': np.eye(4), 'lower': np.eye(4), 'smileArch': np.eye(4)}
    
    def apply_transform(vertices, transform):
        ones = np.ones((len(vertices), 1))
        vertices_homo = np.hstack([vertices, ones])
        transformed = (transform @ vertices_homo.T).T
        return transformed[:, :3]
    
    # Step 1: 각 메시 개별 정규화
    if verbose:
        print("  [SmileArch] OBB 정규화...")
    T1_arch = compute_alignment_transform(arch_vertices, arch_faces, 'smileArch', verbose=verbose)
    
    if verbose:
        print("  [Upper] OBB 정규화...")
    T1_upper = compute_alignment_transform(upper_vertices, upper_faces, 'upper', verbose=verbose)
    
    if verbose:
        print("  [Lower] OBB 정규화...")
    T1_lower = compute_alignment_transform(lower_vertices, lower_faces, 'lower', verbose=verbose)
    
    # 정규화된 메시들 생성
    aligned_arch = apply_transform(arch_vertices, T1_arch)
    aligned_upper = apply_transform(upper_vertices, T1_upper)
    aligned_lower = apply_transform(lower_vertices, T1_lower)
    
    # Step 2: 상대적 위치 조정 (Z축 스태킹)
    arch_bounds = np.array([aligned_arch.min(axis=0), aligned_arch.max(axis=0)])
    upper_bounds = np.array([aligned_upper.min(axis=0), aligned_upper.max(axis=0)])
    lower_bounds = np.array([aligned_lower.min(axis=0), aligned_lower.max(axis=0)])
    
    z_offset_upper = arch_bounds[1][2] - upper_bounds[0][2]
    T_upper_stack = np.eye(4)
    T_upper_stack[2, 3] = z_offset_upper
    T1_upper = T_upper_stack @ T1_upper
    aligned_upper = apply_transform(upper_vertices, T1_upper)
    
    z_offset_lower = arch_bounds[0][2] - lower_bounds[1][2]
    T_lower_stack = np.eye(4)
    T_lower_stack[2, 3] = z_offset_lower
    T1_lower = T_lower_stack @ T1_lower
    aligned_lower = apply_transform(lower_vertices, T1_lower)
    
    if verbose:
        print(f"  상악 Z 오프셋: {z_offset_upper:.2f}mm")
        print(f"  하악 Z 오프셋: {z_offset_lower:.2f}mm")
    
    # Step 3: 갭 제거 (Close Gap)
    if verbose:
        print("  [접촉 이동] 갭 제거...")
    
    gap_upper = close_gap_to_contact(aligned_upper, aligned_arch, arch_faces, is_upper=True, verbose=verbose)
    if gap_upper is not None:
        T_gap_up = np.eye(4)
        T_gap_up[2, 3] = -gap_upper
        T1_upper = T_gap_up @ T1_upper
        aligned_upper = apply_transform(upper_vertices, T1_upper)
    
    gap_lower = close_gap_to_contact(aligned_lower, aligned_arch, arch_faces, is_upper=False, verbose=verbose)
    if gap_lower is not None:
        T_gap_low = np.eye(4)
        T_gap_low[2, 3] = gap_lower
        T1_lower = T_gap_low @ T1_lower
        aligned_lower = apply_transform(lower_vertices, T1_lower)
    
    # Step 4: Y-슬라이딩 ICP
    if verbose:
        print("  [Y-슬라이딩 ICP] 상악/하악 정밀 정합...")
        print("    - +Y(앞)에서 시작 → -Y(뒤)로 슬라이딩")
        print("    - 각 축 ±10° 회전 제한")
    
    # +Y 방향으로 초기 이동
    arch_y_max = np.max(aligned_arch[:, 1])
    upper_y_max = np.max(aligned_upper[:, 1])
    lower_y_max = np.max(aligned_lower[:, 1])
    
    y_forward_upper = arch_y_max - upper_y_max + 5.0
    y_forward_lower = arch_y_max - lower_y_max + 5.0
    
    if verbose:
        print(f"\n    [초기 Y 이동] 상악: +{y_forward_upper:.1f}mm, 하악: +{y_forward_lower:.1f}mm")
    
    T_forward_up = np.eye(4)
    T_forward_up[1, 3] = y_forward_upper
    T1_upper = T_forward_up @ T1_upper
    aligned_upper = apply_transform(upper_vertices, T1_upper)
    
    T_forward_low = np.eye(4)
    T_forward_low[1, 3] = y_forward_lower
    T1_lower = T_forward_low @ T1_lower
    aligned_lower = apply_transform(lower_vertices, T1_lower)
    
    # trimesh 메시 생성
    arch_mesh = trimesh.Trimesh(vertices=aligned_arch, faces=arch_faces)
    upper_mesh = trimesh.Trimesh(vertices=aligned_upper, faces=upper_faces)
    lower_mesh = trimesh.Trimesh(vertices=aligned_lower, faces=lower_faces)
    
    # 상악 ICP
    if verbose:
        print("\n    [상악] +Y에서 -Y로 슬라이딩...")
    T_icp_up = run_rotation_limited_icp(
        upper_mesh, arch_mesh,
        max_rotation=10.0, max_translation=40.0,
        is_upper=True, verbose=verbose
    )
    T1_upper = T_icp_up @ T1_upper
    
    # 하악 ICP
    if verbose:
        print("\n    [하악] +Y에서 -Y로 슬라이딩...")
    T_icp_low = run_rotation_limited_icp(
        lower_mesh, arch_mesh,
        max_rotation=20.0, max_translation=40.0,
        is_upper=False, verbose=verbose
    )
    T1_lower = T_icp_low @ T1_lower
    
    return {'upper': T1_upper, 'lower': T1_lower, 'smileArch': T1_arch}


# ============================================================
# IOS Upper/Lower Registration 클래스
# ============================================================

class IOSUpperLowerRegistration:
    """
    상악과 하악을 SmileArch에 정합하는 클래스.
    
    수학적 접근:
    - T1 = align_dental_set_v4 알고리즘으로 정규화
    - T2 = ios_laminate_result (SmileArch 목표 위치)
    - T3 = T2 × T1_arch⁻¹ (정규화 → 최종 위치)
    - 최종 상악 = T3 × T1_upper
    - 최종 하악 = T3 × T1_lower
    """
    
    IDENTITY_MATRIX = np.eye(4)
    
    def __init__(
        self,
        upper_mesh: Mesh,
        lower_mesh: Mesh,
        smile_arch_mesh: Mesh,
        ios_laminate_result: np.ndarray,
        verbose: bool = True
    ):
        self.upper_mesh = upper_mesh
        self.lower_mesh = lower_mesh
        self.smile_arch_mesh = smile_arch_mesh
        self.ios_laminate_result = ios_laminate_result
        self.verbose = verbose
    
    def _apply_transform(self, vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
        ones = np.ones((len(vertices), 1))
        vertices_homo = np.hstack([vertices, ones])
        transformed = (transform @ vertices_homo.T).T
        return transformed[:, :3]
    
    async def compute_transformations(self, visualize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """상악과 하악의 변환 행렬을 계산합니다."""
        try:
            if self.verbose:
                print("\n" + "=" * 60)
                print("[IOS Upper/Lower Registration]")
                print("=" * 60)
            
            # Step 1: align_dental_set_v4 알고리즘으로 정규화 (T1)
            if self.verbose:
                print("\n[Step 1] align_dental_set 알고리즘 실행...")
            
            T1 = align_dental_set(
                self.upper_mesh.vertices, self.upper_mesh.faces,
                self.lower_mesh.vertices, self.lower_mesh.faces,
                self.smile_arch_mesh.vertices, self.smile_arch_mesh.faces,
                verbose=self.verbose
            )
            
            T1_arch = T1['smileArch']
            T1_upper = T1['upper']
            T1_lower = T1['lower']
            
            if self.verbose:
                print("\n[Step 2] 변환 행렬 계산...")
                print(f"  T2 (ios_laminate_result): SmileArch 최종 위치")
            
            # Step 2: T3 = T2 × T1_arch⁻¹
            T2 = self.ios_laminate_result
            T1_arch_inv = np.linalg.inv(T1_arch)
            T3 = T2 @ T1_arch_inv
            
            if self.verbose:
                print(f"  T3 = T2 × T1_arch⁻¹ 계산 완료")
            
            # Step 3: 최종 변환 계산
            upper_final = T3 @ T1_upper
            lower_final = T3 @ T1_lower
            
            if self.verbose:
                print(f"  최종 상악 = T3 × T1_upper")
                print(f"  최종 하악 = T3 × T1_lower")
                print("\n✅ 정합 완료")
            
            if visualize and HAS_TRIMESH:
                self._visualize_result(T2, upper_final, lower_final)
            
            return upper_final, lower_final
            
        except Exception as e:
            print(f"[ERROR] Registration failed: {e}")
            import traceback
            traceback.print_exc()
            return self.IDENTITY_MATRIX.copy(), self.IDENTITY_MATRIX.copy()
    
    def _visualize_result(self, arch_transform, upper_transform, lower_transform):
        """정합 결과를 시각화합니다."""
        if not HAS_TRIMESH:
            return
        
        print("\n[시각화] 3D 뷰어 열기...")
        print("  - 초록: SmileArch")
        print("  - 빨강: Upper (상악)")
        print("  - 파랑: Lower (하악)")
        
        scene = trimesh.Scene()
        
        arch_vertices = self._apply_transform(self.smile_arch_mesh.vertices, arch_transform)
        arch_mesh = trimesh.Trimesh(vertices=arch_vertices, faces=self.smile_arch_mesh.faces)
        arch_mesh.visual.face_colors = [100, 255, 100, 180]
        scene.add_geometry(arch_mesh, node_name='smileArch')
        
        upper_vertices = self._apply_transform(self.upper_mesh.vertices, upper_transform)
        upper_mesh = trimesh.Trimesh(vertices=upper_vertices, faces=self.upper_mesh.faces)
        upper_mesh.visual.face_colors = [255, 100, 100, 180]
        scene.add_geometry(upper_mesh, node_name='upper')
        
        lower_vertices = self._apply_transform(self.lower_mesh.vertices, lower_transform)
        lower_mesh = trimesh.Trimesh(vertices=lower_vertices, faces=self.lower_mesh.faces)
        lower_mesh.visual.face_colors = [100, 100, 255, 180]
        scene.add_geometry(lower_mesh, node_name='lower')
        
        scene.show(flags={'cull': False})


async def compute_ios_transformations(
    ios_upper_mesh: Mesh,
    ios_lower_mesh: Mesh,
    smile_arch_mesh: Mesh,
    ios_laminate_result: np.ndarray,
    visualize: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """상악과 하악을 SmileArch에 정합합니다."""
    registration = IOSUpperLowerRegistration(
        ios_upper_mesh, ios_lower_mesh, smile_arch_mesh, ios_laminate_result
    )
    return await registration.compute_transformations(visualize=visualize)
