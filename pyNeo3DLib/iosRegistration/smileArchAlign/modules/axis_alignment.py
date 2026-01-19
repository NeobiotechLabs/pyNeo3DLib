"""
SmileArch 축 정렬 모듈
======================

smileArch.stl 모델의 축을 정렬하는 함수들을 제공합니다.
align_single_model.py의 노이즈 강건 버전을 기반으로 합니다.

정렬 순서:
1. Z축 정렬: OBB 기반 최단축 → Z축, +Z가 볼록면
2. 대칭축 정렬: 멀티레벨 컨투어 기반 강건한 대칭축 탐지
3. Y방향 정렬: 좁은 부분(전치부)이 +Y를 향하도록
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree, ConvexHull
from pathlib import Path
import copy


# ============================================================================
# 메쉬 유틸리티 함수
# ============================================================================

def downsample_mesh(mesh: trimesh.Trimesh, target_faces: int = 50000) -> trimesh.Trimesh:
    """메쉬 다운샘플링 (정렬 계산 성능 최적화)"""
    if len(mesh.faces) <= target_faces:
        return mesh.copy()
    
    simplified = mesh.copy()
    face_sample_indices = np.random.choice(len(mesh.faces), min(target_faces, len(mesh.faces)), replace=False)
    simplified.faces = mesh.faces[face_sample_indices]
    simplified.remove_unreferenced_vertices()
    return simplified


def load_mesh(file_path: str, downsample_target: int = 5000) -> tuple:
    """메쉬 로드 및 다운샘플링 버전 생성"""
    mesh = trimesh.load(file_path)
    print(f"로드: {file_path}")
    print(f"  정점: {len(mesh.vertices):,}, 면: {len(mesh.faces):,}")
    
    if len(mesh.faces) > downsample_target:
        mesh_ds = downsample_mesh(mesh, downsample_target)
        print(f"  → 다운샘플: {len(mesh_ds.faces):,}")
    else:
        mesh_ds = mesh.copy()
    
    return mesh, mesh_ds


# ============================================================================
# Z축 정렬 함수
# ============================================================================

def compute_z_curvature(mesh: trimesh.Trimesh, z_positive: bool = True) -> float:
    """메쉬의 +Z 또는 -Z 영역의 볼록함(곡률) 측정"""
    vertices = mesh.vertices
    y_range = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    y_center = np.median(vertices[:, 1])
    center_mask = np.abs(vertices[:, 1] - y_center) < y_range * 0.2
    center_vertices = vertices[center_mask]
    
    if len(center_vertices) < 10:
        return 0.0
    
    z_median = np.median(center_vertices[:, 2])
    z_mask = center_vertices[:, 2] > z_median if z_positive else center_vertices[:, 2] < z_median
    selected = center_vertices[z_mask]
    
    if len(selected) < 5:
        return 0.0
    
    z_std = np.std(selected[:, 2])
    z_extent = np.abs(np.max(selected[:, 2]) - z_median) if z_positive else np.abs(z_median - np.min(selected[:, 2]))
    return z_std * z_extent


def align_shortest_axis_to_z(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """OBB 기반 최단축을 Z축으로 정렬 (+Z가 볼록면)"""
    obb = mesh.bounding_box_oriented
    obb_extents = obb.primitive.extents
    obb_transform = obb.primitive.transform
    
    print(f"  OBB 크기: X={obb_extents[0]:.2f}, Y={obb_extents[1]:.2f}, Z={obb_extents[2]:.2f}")
    
    shortest_axis = np.argmin(obb_extents)
    print(f"  최단축: {['X', 'Y', 'Z'][shortest_axis]} (크기: {obb_extents[shortest_axis]:.2f})")
    
    obb_rotation = obb_transform[:3, :3]
    shortest_axis_direction = obb_rotation.T[shortest_axis]
    target_direction = np.array([0, 0, 1])
    
    v = np.cross(shortest_axis_direction, target_direction)
    c = np.dot(shortest_axis_direction, target_direction)
    
    if np.linalg.norm(v) < 1e-6:
        rotation_matrix = np.eye(3) if c > 0 else Rotation.from_euler('x', 180, degrees=True).as_matrix()
    else:
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    
    aligned_mesh = mesh.copy()
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    aligned_mesh.apply_transform(transform)
    aligned_mesh.apply_translation(-aligned_mesh.centroid)
    
    curvature_pos = compute_z_curvature(aligned_mesh, z_positive=True)
    curvature_neg = compute_z_curvature(aligned_mesh, z_positive=False)
    print(f"  +Z 볼록도: {curvature_pos:.4f}, -Z 볼록도: {curvature_neg:.4f}")
    
    if curvature_neg > curvature_pos:
        print("  → 180도 회전 (+Z가 볼록면이 되도록)")
        flip = np.eye(4)
        flip[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        aligned_mesh.apply_transform(flip)
    
    return aligned_mesh


# ============================================================================
# 대칭축 정렬 함수
# ============================================================================

def get_2d_convex_hull_points(mesh: trimesh.Trimesh, num_points: int = 100) -> np.ndarray:
    """XY 평면 투영의 2D Convex Hull 점 추출"""
    points_2d = mesh.vertices[:, :2]
    hull = ConvexHull(points_2d)
    hull_vertices = points_2d[hull.vertices]
    
    hull_closed = np.vstack([hull_vertices, hull_vertices[0]])
    distances = np.sqrt(np.sum(np.diff(hull_closed, axis=0) ** 2, axis=1))
    cumulative = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative[-1]
    
    sample_distances = np.linspace(0, total_length, num_points, endpoint=False)
    sampled_points = []
    
    for d in sample_distances:
        idx = max(0, min(np.searchsorted(cumulative, d) - 1, len(hull_vertices) - 1))
        next_idx = (idx + 1) % len(hull_vertices)
        t = (d - cumulative[idx]) / distances[idx] if distances[idx] > 0 else 0
        point = hull_vertices[idx] * (1 - t) + hull_vertices[next_idx] * t
        sampled_points.append(point)
    
    return np.array(sampled_points)


def compute_symmetry_score_from_points(points_2d, angle_degrees):
    """2D 점들로부터 대칭 점수 계산 (낮을수록 대칭)"""
    if len(points_2d) < 3:
        return float('inf')
    
    angle_rad = np.radians(angle_degrees)
    rotation_2d = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], 
                            [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated = points_2d @ rotation_2d.T
    reflected = rotated.copy()
    reflected[:, 0] = -reflected[:, 0]
    
    tree = cKDTree(rotated)
    distances, _ = tree.query(reflected, k=1)
    return np.sqrt(np.mean(distances ** 2))


def find_symmetry_axis_from_points(points_2d, coarse_step=10.0):
    """2D 점들에서 대칭축 각도 탐색"""
    if len(points_2d) < 10:
        return None, float('inf')
    
    angles = np.arange(0, 180, coarse_step)
    scores = [compute_symmetry_score_from_points(points_2d, a) for a in angles]
    best_coarse = angles[np.argmin(scores)]
    
    result = minimize_scalar(
        lambda a: compute_symmetry_score_from_points(points_2d, a),
        bounds=(max(0, best_coarse - coarse_step), min(180, best_coarse + coarse_step)),
        method='bounded', options={'xatol': 0.1}
    )
    return result.x, result.fun


def extract_z_level_contours(mesh: trimesh.Trimesh, n_levels: int = 10) -> list:
    """여러 Z레벨에서 컨투어 추출"""
    print(f"\n  [Z-Level 컨투어] {n_levels} 레벨")
    
    vertices = mesh.vertices
    z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])
    z_range = z_max - z_min
    
    contours = []
    for i in range(n_levels):
        z_low = z_min + z_range * i / n_levels
        z_high = z_min + z_range * (i + 1) / n_levels
        z_mid = (z_low + z_high) / 2
        
        level_mask = (vertices[:, 2] >= z_low) & (vertices[:, 2] < z_high)
        level_vertices = vertices[level_mask]
        
        if len(level_vertices) < 10:
            continue
        
        try:
            hull = ConvexHull(level_vertices[:, :2])
            hull_points = level_vertices[:, :2][hull.vertices]
            print(f"     레벨 {i+1}: Z={z_mid:.1f} - {len(hull_points)} pts")
            contours.append((z_mid, hull_points))
        except:
            pass
    
    return contours


def find_robust_symmetry_axis(contours: list, outlier_threshold: float = 2.0) -> tuple:
    """멀티레벨 컨투어에서 강건한 대칭축 각도 탐색"""
    print("\n  [멀티레벨 대칭 분석]")
    
    angles, scores = [], []
    for i, (z_height, contour_points) in enumerate(contours):
        angle, score = find_symmetry_axis_from_points(contour_points)
        if angle is not None:
            angles.append(angle)
            scores.append(score)
            print(f"     레벨 {i+1} (Z={z_height:.1f}): 각도={angle:.1f}°, 점수={score:.4f}")
    
    if len(angles) < 3:
        return (np.median(angles), angles, []) if angles else (None, [], [])
    
    angles = np.array(angles)
    scores = np.array(scores)
    angles_normalized = angles % 180
    
    median_angle = np.median(angles_normalized)
    mad = np.median(np.abs(angles_normalized - median_angle))
    
    if mad > 0:
        z_scores = np.abs(angles_normalized - median_angle) / mad
        outlier_mask = z_scores > outlier_threshold
    else:
        outlier_mask = np.zeros(len(angles), dtype=bool)
    
    outlier_indices = np.where(outlier_mask)[0].tolist()
    valid_mask = ~outlier_mask
    
    print(f"\n     결과: {len(angles)}개 레벨, {len(outlier_indices)}개 이상치, {np.sum(valid_mask)}개 유효")
    
    if np.sum(valid_mask) == 0:
        robust_angle = angles[np.argmin(scores)]
    else:
        valid_angles = angles_normalized[valid_mask]
        valid_scores = scores[valid_mask]
        weights = 1.0 / (valid_scores + 1e-6)
        robust_angle = np.average(valid_angles, weights=weights)
    
    print(f"     → 강건한 각도: {robust_angle:.1f}°")
    return robust_angle, angles_normalized.tolist(), outlier_indices


def align_symmetry_to_x(mesh: trimesh.Trimesh) -> tuple:
    """대칭축을 X축에 정렬 (기본 방식)"""
    hull_points_2d = get_2d_convex_hull_points(mesh, num_points=200)
    optimal_angle, _ = find_symmetry_axis_from_points(hull_points_2d)
    
    rotation = Rotation.from_euler('z', optimal_angle, degrees=True).as_matrix()
    aligned_mesh = mesh.copy()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    aligned_mesh.apply_transform(transform)
    
    angle_rad = np.radians(optimal_angle)
    rotation_2d = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], 
                            [np.sin(angle_rad), np.cos(angle_rad)]])
    aligned_hull = hull_points_2d @ rotation_2d.T
    
    return aligned_mesh, optimal_angle, aligned_hull


def apply_rotation(mesh: trimesh.Trimesh, angle_degrees: float, axis: str = 'z') -> trimesh.Trimesh:
    """메쉬에 회전 적용"""
    rotation = Rotation.from_euler(axis, angle_degrees, degrees=True).as_matrix()
    aligned_mesh = mesh.copy()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    aligned_mesh.apply_transform(transform)
    return aligned_mesh


# ============================================================================
# Y방향 정렬 함수
# ============================================================================

def compute_average_width(hull_points: np.ndarray, y_positive: bool = True) -> float:
    """Hull의 +Y 또는 -Y 영역의 평균 X축 폭 측정"""
    y_median = np.median(hull_points[:, 1])
    mask = hull_points[:, 1] > y_median if y_positive else hull_points[:, 1] < y_median
    selected = hull_points[mask]
    
    if len(selected) < 2:
        return 0.0
    
    y_min, y_max = np.min(selected[:, 1]), np.max(selected[:, 1])
    widths = []
    for i in range(5):
        y_low = y_min + (y_max - y_min) * i / 5
        y_high = y_min + (y_max - y_min) * (i + 1) / 5
        bin_pts = selected[(selected[:, 1] >= y_low) & (selected[:, 1] < y_high)]
        if len(bin_pts) >= 2:
            widths.append(np.max(bin_pts[:, 0]) - np.min(bin_pts[:, 0]))
    
    return np.mean(widths) if widths else 0.0


def align_y_direction(mesh: trimesh.Trimesh, hull_points: np.ndarray) -> tuple:
    """좁은 부분(전치부)이 +Y를 향하도록 정렬"""
    width_pos = compute_average_width(hull_points, y_positive=True)
    width_neg = compute_average_width(hull_points, y_positive=False)
    print(f"  +Y 폭: {width_pos:.2f}, -Y 폭: {width_neg:.2f}")
    
    if width_pos > width_neg:
        print("  → 180도 회전 (+Y가 더 넓음, 좁아야 함)")
        rotation = Rotation.from_euler('z', 180, degrees=True).as_matrix()
        aligned_mesh = mesh.copy()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        aligned_mesh.apply_transform(transform)
        return aligned_mesh, True, -hull_points
    else:
        print("  → OK (+Y가 더 좁음 = 전면)")
        return mesh.copy(), False, hull_points.copy()


def check_arch_shape(hull_points: np.ndarray, threshold: float = 0.3) -> tuple:
    """아치 형태 검증 (노이즈 검출)"""
    print("\n  [아치 형태 확인]")
    
    y_values = hull_points[:, 1]
    y_min, y_max = np.min(y_values), np.max(y_values)
    y_range = y_max - y_min
    
    n_sections = 5
    section_widths = []
    
    for i in range(n_sections):
        y_low = y_min + y_range * i / n_sections
        y_high = y_min + y_range * (i + 1) / n_sections
        section_mask = (hull_points[:, 1] >= y_low) & (hull_points[:, 1] < y_high)
        section_points = hull_points[section_mask]
        width = np.max(section_points[:, 0]) - np.min(section_points[:, 0]) if len(section_points) >= 2 else 0
        section_widths.append(width)
    
    print(f"     섹션별 폭: {[f'{w:.1f}' for w in section_widths]}")
    
    front_width = section_widths[-1]
    back_width = section_widths[0]
    width_ratio = front_width / back_width if back_width > 0 else 1.0
    print(f"     전면/후면 비율: {width_ratio:.3f}")
    
    is_arch = width_ratio < (1.0 - threshold)
    is_smooth = np.max(np.abs(np.diff(section_widths))) / np.mean(section_widths) < 0.5 if np.mean(section_widths) > 0 else True
    
    left_count = np.sum(hull_points[:, 0] < 0)
    right_count = np.sum(hull_points[:, 0] >= 0)
    is_symmetric = min(left_count, right_count) / max(left_count, right_count) > 0.7 if max(left_count, right_count) > 0 else True
    
    if is_arch and is_smooth and is_symmetric:
        return True, "정상 아치 형태", section_widths
    else:
        issues = []
        if not is_arch: issues.append("폭 비율 이상")
        if not is_smooth: issues.append("급격한 폭 변화")
        if not is_symmetric: issues.append("비대칭")
        return False, f"노이즈 감지: {', '.join(issues)}", section_widths


# ============================================================================
# 메인 정렬 함수
# ============================================================================

def align_model(mesh: trimesh.Trimesh, visualize: bool = True, output_dir: str = None) -> tuple:
    """
    전체 축 정렬 파이프라인 (노이즈 강건 버전)
    
    Args:
        mesh: 입력 메쉬
        visualize: 시각화 여부
        output_dir: 시각화 저장 경로
        
    Returns:
        (정렬된 메쉬, 정렬 정보 딕셔너리)
    """
    alignment_info = {
        'z_aligned': False, 'symmetry_aligned': False, 'noise_detected': False,
        'robust_alignment': False, 'y_aligned': False, 'final_symmetry_angle': 0.0, 'outliers': []
    }
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    # Step 1: Z축 정렬
    print("\n" + "=" * 50)
    print("[Step 1] Z축 정렬")
    print("=" * 50)
    z_aligned = align_shortest_axis_to_z(mesh)
    alignment_info['z_aligned'] = True
    
    if visualize:
        save_path = str(output_path / "step1_z_aligned.png") if output_path else None
        visualize_z_aligned(z_aligned, save_path=save_path)
    
    # Step 2: 대칭축 정렬
    print("\n" + "=" * 50)
    print("[Step 2] 대칭축 정렬")
    print("=" * 50)
    sym_aligned, sym_angle, hull_points = align_symmetry_to_x(z_aligned)
    alignment_info['symmetry_aligned'] = True
    alignment_info['initial_symmetry_angle'] = sym_angle
    
    # Z-level 컨투어 추출
    contours = extract_z_level_contours(z_aligned, n_levels=10)
    
    if visualize:
        save_path = str(output_path / "step2_symmetry_aligned.png") if output_path else None
        visualize_symmetry_aligned(sym_aligned, hull_points, sym_angle, save_path=save_path)
    
    # Step 3: 아치 형태 확인 (노이즈 검출)
    print("\n" + "=" * 50)
    print("[Step 3] 아치 형태 확인 (노이즈 검출)")
    print("=" * 50)
    is_arch, message, section_widths = check_arch_shape(hull_points)
    print(f"\n  결과: {message}")
    
    if visualize:
        save_path = str(output_path / "step3_arch_check.png") if output_path else None
        visualize_arch_check(hull_points, section_widths, is_arch, save_path=save_path)
    
    if is_arch:
        print("\n  OK - 정상 아치 형태, 추가 처리 불필요")
        final_aligned = sym_aligned
        alignment_info['final_symmetry_angle'] = sym_angle
    else:
        print("\n  노이즈 감지 - Z-level 컨투어로 재계산")
        alignment_info['noise_detected'] = True
        
        # Step 4: 강건한 대칭축 계산
        print("\n" + "=" * 50)
        print("[Step 4] 강건한 대칭축 재계산")
        print("=" * 50)
        
        if visualize and contours:
            save_path = str(output_path / "step4a_z_contours.png") if output_path else None
            visualize_z_level_contours(z_aligned, contours, save_path=save_path)
        
        if contours:
            robust_angle, all_angles, outliers = find_robust_symmetry_axis(contours)
            
            if visualize:
                save_path = str(output_path / "step4b_robust_symmetry.png") if output_path else None
                visualize_robust_symmetry(contours, all_angles, outliers, robust_angle, save_path=save_path)
            
            if robust_angle is not None:
                print(f"\n  최종 강건 각도: {robust_angle:.1f}°")
                final_aligned = apply_rotation(z_aligned, robust_angle, 'z')
                alignment_info['robust_alignment'] = True
                alignment_info['final_symmetry_angle'] = robust_angle
                alignment_info['all_level_angles'] = all_angles
                alignment_info['outliers'] = outliers
                hull_points = get_2d_convex_hull_points(final_aligned)
            else:
                final_aligned = sym_aligned
                alignment_info['final_symmetry_angle'] = sym_angle
        else:
            final_aligned = sym_aligned
            alignment_info['final_symmetry_angle'] = sym_angle
    
    # Step 5: Y방향 정렬
    print("\n" + "=" * 50)
    print("[Step 5] Y방향 정렬 (전면 → +Y)")
    print("=" * 50)
    final_mesh, y_rotated, final_hull = align_y_direction(final_aligned, hull_points)
    alignment_info['y_aligned'] = True
    alignment_info['y_rotated_180'] = y_rotated
    
    if visualize:
        save_path = str(output_path / "step5_final_result.png") if output_path else None
        visualize_final_result(final_mesh, save_path=save_path)
    
    return final_mesh, alignment_info


# ============================================================================
# 시각화 함수
# ============================================================================

def visualize_z_aligned(mesh: trimesh.Trimesh, save_path: str = None):
    """Visualize Z-axis alignment result."""
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Step 1: Z-Axis Aligned (Shortest -> Z, +Z Convex)", fontsize=14, fontweight='bold')
    
    vertices = mesh.vertices
    if len(vertices) > 5000:
        sample_idx = np.random.choice(len(vertices), 5000, replace=False)
        vertices = vertices[sample_idx]
    
    extents = mesh.bounds[1] - mesh.bounds[0]
    max_range = np.max(extents) / 2 * 1.1
    
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    z_colors = plt.cm.viridis((vertices[:, 2] - vertices[:, 2].min()) / (vertices[:, 2].max() - vertices[:, 2].min() + 1e-6))
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.5, alpha=0.5, c=z_colors)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z (Height)')
    ax1.set_title(f"3D View (Z color)\nZ={extents[2]:.1f} (shortest)")
    
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(vertices[:, 0], vertices[:, 1], s=0.3, alpha=0.3, c='coral')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    ax2.set_title("XY Plane (Top View)")
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(vertices[:, 1], vertices[:, 2], s=0.3, alpha=0.3, c='coral')
    ax3.set_xlabel('Y'); ax3.set_ylabel('Z')
    ax3.set_title("YZ Plane (Side View)\n+Z should be convex")
    ax3.set_aspect('equal'); ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f">>> Saved: {save_path}")
    plt.show()


def visualize_symmetry_aligned(mesh: trimesh.Trimesh, hull_points: np.ndarray, 
                                sym_angle: float, save_path: str = None):
    """Visualize symmetry axis alignment result."""
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"Step 2: Symmetry Aligned ({sym_angle:.1f} deg rotation)", fontsize=14, fontweight='bold')
    
    vertices = mesh.vertices
    if len(vertices) > 5000:
        sample_idx = np.random.choice(len(vertices), 5000, replace=False)
        vertices = vertices[sample_idx]
    
    extents = mesh.bounds[1] - mesh.bounds[0]
    max_range = np.max(extents) / 2 * 1.1
    
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    colors = np.where(vertices[:, 0] >= 0, 'royalblue', 'coral')
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.5, alpha=0.5, c=colors)
    ax1.set_xlabel('X (L/R)'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title("3D View (blue=+X, red=-X)")
    
    ax2 = fig.add_subplot(1, 3, 2)
    colors_2d = np.where(vertices[:, 0] >= 0, 'royalblue', 'coral')
    ax2.scatter(vertices[:, 0], vertices[:, 1], s=0.3, alpha=0.3, c=colors_2d)
    hull_closed = np.vstack([hull_points, hull_points[0]])
    ax2.plot(hull_closed[:, 0], hull_closed[:, 1], 'g-', linewidth=2, label='Convex Hull')
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=2, label='Symmetry Axis')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    ax2.set_title("XY View + Convex Hull")
    ax2.set_aspect('equal'); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(vertices[:, 0], vertices[:, 1], s=0.3, alpha=0.2, c='gray')
    ax3.scatter(-vertices[:, 0], vertices[:, 1], s=0.3, alpha=0.2, c='lightblue', label='Reflected')
    ax3.axvline(x=0, color='red', linestyle='-', linewidth=2)
    ax3.set_xlabel('X'); ax3.set_ylabel('Y')
    ax3.set_title("Symmetry Check\n(gray=original, blue=reflected)")
    ax3.set_aspect('equal'); ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f">>> Saved: {save_path}")
    plt.show()


def visualize_arch_check(hull_points: np.ndarray, section_widths: list, 
                         is_arch: bool, save_path: str = None):
    """Visualize arch shape check."""
    fig = plt.figure(figsize=(12, 5))
    status = "PASS - Normal Arch" if is_arch else "FAIL - Noise Detected"
    fig.suptitle(f"Step 3: Arch Shape Check - {status}", fontsize=14, fontweight='bold')
    
    ax1 = fig.add_subplot(1, 2, 1)
    hull_closed = np.vstack([hull_points, hull_points[0]])
    ax1.plot(hull_closed[:, 0], hull_closed[:, 1], 'b-', linewidth=2)
    ax1.fill(hull_closed[:, 0], hull_closed[:, 1], alpha=0.2, color='blue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y')
    ax1.set_title("Hull + Section Division")
    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(1, 2, 2)
    y_values = hull_points[:, 1]
    y_min, y_max = np.min(y_values), np.max(y_values)
    n_sections = len(section_widths)
    y_centers = [y_min + (y_max - y_min) * (i + 0.5) / n_sections for i in range(n_sections)]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_sections))
    ax2.barh(y_centers, section_widths, height=(y_max - y_min) / n_sections * 0.8, 
             color=colors, edgecolor='black')
    ax2.set_xlabel('Width'); ax2.set_ylabel('Y Position')
    ax2.set_title("Width Profile\n(should narrow toward +Y)")
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f">>> Saved: {save_path}")
    plt.show()


def visualize_z_level_contours(mesh: trimesh.Trimesh, contours: list, save_path: str = None):
    """Visualize Z-level contours."""
    n_contours = len(contours)
    n_cols = min(5, n_contours)
    n_rows = (n_contours + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle("Step 4a: Z-Level Contours Extraction", fontsize=14, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_contours))
    
    for i, (z_height, contour_points) in enumerate(contours):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        hull_closed = np.vstack([contour_points, contour_points[0]])
        ax.plot(hull_closed[:, 0], hull_closed[:, 1], '-', linewidth=2, color=colors[i])
        ax.fill(hull_closed[:, 0], hull_closed[:, 1], alpha=0.3, color=colors[i])
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f"Level {i+1}: Z={z_height:.1f}\n({len(contour_points)} pts)", fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f">>> Saved: {save_path}")
    plt.show()


def visualize_robust_symmetry(contours: list, angles: list, outlier_indices: list, 
                               robust_angle: float, save_path: str = None):
    """Visualize robust symmetry axis detection."""
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("Step 4b: Robust Symmetry Axis Detection", fontsize=14, fontweight='bold')
    
    ax1 = fig.add_subplot(1, 2, 1)
    x_positions = np.arange(len(angles))
    colors = ['red' if i in outlier_indices else 'green' for i in range(len(angles))]
    ax1.bar(x_positions, angles, color=colors, edgecolor='black', alpha=0.7)
    ax1.axhline(y=robust_angle, color='blue', linestyle='--', linewidth=2, label=f'Robust angle: {robust_angle:.1f} deg')
    ax1.set_xlabel('Z-Level Index'); ax1.set_ylabel('Symmetry Angle (deg)')
    ax1.set_title("Symmetry Angles by Level\n(red=outlier, green=valid)")
    ax1.legend(); ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = fig.add_subplot(1, 2, 2)
    colors_contour = plt.cm.viridis(np.linspace(0.1, 0.9, len(contours)))
    
    for i, (z_height, contour_points) in enumerate(contours):
        hull_closed = np.vstack([contour_points, contour_points[0]])
        style = '--' if i in outlier_indices else '-'
        alpha = 0.3 if i in outlier_indices else 0.8
        ax2.plot(hull_closed[:, 0], hull_closed[:, 1], style, linewidth=1.5, 
                 color=colors_contour[i], alpha=alpha)
    
    ax2.axvline(x=0, color='red', linestyle='-', linewidth=2, label='Symmetry Axis')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    ax2.set_title("All Contours Overlay\n(dashed=outlier)")
    ax2.set_aspect('equal'); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f">>> Saved: {save_path}")
    plt.show()


def visualize_final_result(mesh: trimesh.Trimesh, save_path: str = None):
    """Visualize final alignment result."""
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Final Result: Aligned Model", fontsize=14, fontweight='bold')
    
    vertices = mesh.vertices
    if len(vertices) > 5000:
        sample_idx = np.random.choice(len(vertices), 5000, replace=False)
        vertices = vertices[sample_idx]
    
    extents = mesh.bounds[1] - mesh.bounds[0]
    max_range = np.max(extents) / 2 * 1.1
    
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    colors = np.where(vertices[:, 0] >= 0, 'royalblue', 'coral')
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.5, alpha=0.5, c=colors)
    ax1.set_xlabel('X (Symmetry)'); ax1.set_ylabel('Y (Front +)'); ax1.set_zlabel('Z (Height)')
    ax1.set_title("3D View")
    
    axis_len = max_range * 0.5
    ax1.quiver(0, 0, 0, axis_len, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2)
    ax1.quiver(0, 0, 0, 0, axis_len, 0, color='g', arrow_length_ratio=0.1, linewidth=2)
    ax1.quiver(0, 0, 0, 0, 0, axis_len, color='b', arrow_length_ratio=0.1, linewidth=2)
    
    ax2 = fig.add_subplot(1, 3, 2)
    colors_2d = np.where(vertices[:, 0] >= 0, 'royalblue', 'coral')
    ax2.scatter(vertices[:, 0], vertices[:, 1], s=0.3, alpha=0.3, c=colors_2d)
    
    hull_points = get_2d_convex_hull_points(mesh)
    hull_closed = np.vstack([hull_points, hull_points[0]])
    ax2.plot(hull_closed[:, 0], hull_closed[:, 1], 'g-', linewidth=2)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=2)
    ax2.arrow(0, np.min(vertices[:, 1]) * 0.8, 0, np.max(vertices[:, 1]) * 0.3, 
              head_width=2, head_length=2, fc='green', ec='green')
    ax2.text(3, np.max(vertices[:, 1]) * 0.5, '+Y (Front)', fontsize=10, color='green')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_title("Top View (XY)")
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(vertices[:, 1], vertices[:, 2], s=0.3, alpha=0.3, c='coral')
    ax3.set_xlabel('Y'); ax3.set_ylabel('Z'); ax3.set_title("Side View (YZ)\n+Z is convex")
    ax3.set_aspect('equal'); ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f">>> Saved: {save_path}")
    plt.show()


# ============================================================================
# 테스트 실행
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SmileArch 축 정렬 테스트')
    parser.add_argument('--sample', type=int, default=1, help='샘플 번호 (1, 3, 4)')
    parser.add_argument('--no-visualize', action='store_true', help='시각화 비활성화')
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent.parent.parent / "3dmodel"
    sample_path = base_path / f"sample_{args.sample}" / "smileArch.stl"
    output_dir = Path(__file__).parent.parent / "output" / f"sample_{args.sample}" / "visualization"
    
    if not sample_path.exists():
        print(f"오류: {sample_path} 파일을 찾을 수 없습니다.")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"SmileArch 축 정렬 테스트 - Sample {args.sample}")
    print(f"{'='*60}")
    
    # 로드
    mesh_original, mesh_ds = load_mesh(str(sample_path), downsample_target=5000)
    
    # 정렬
    aligned_mesh, info = align_model(
        mesh_ds, 
        visualize=not args.no_visualize, 
        output_dir=str(output_dir)
    )
    
    # 결과 출력
    print(f"\n{'='*60}")
    print("정렬 완료!")
    print(f"{'='*60}")
    print(f"  Z축 정렬: {info['z_aligned']}")
    print(f"  대칭축 정렬: {info['symmetry_aligned']}")
    print(f"  노이즈 감지: {info['noise_detected']}")
    print(f"  강건 정렬: {info['robust_alignment']}")
    print(f"  Y방향 정렬: {info['y_aligned']}")
    print(f"  최종 대칭 각도: {info['final_symmetry_angle']:.1f}°")
