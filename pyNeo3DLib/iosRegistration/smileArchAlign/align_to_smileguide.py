"""
SmileArch → SmileGuide 정합 라이브러리
======================================

Source 메쉬(smileArch)를 SmileGuide 메쉬에 정합하는 변환 행렬을 계산합니다.

사용 예시:
    from smileArchAlign import align_to_smileguide
    
    transform = align_to_smileguide("smileArch.stl", "smileguide.stl")
    
    source = trimesh.load("smileArch.stl")
    source.apply_transform(transform)  # 이제 SmileGuide에 정합됨
"""

import numpy as np
import trimesh
import open3d as o3d
from pathlib import Path
from typing import Dict, Optional

# 내부 모듈 임포트
from .modules.axis_alignment import align_model, load_mesh
from .modules.roi_extraction import extract_roi
from .modules.coarse_alignment import coarse_align
from .modules.fine_alignment import trimesh_to_o3d, compute_alignment_error


def _compute_rigid_transform(source_pts: np.ndarray, 
                             aligned_pts: np.ndarray,
                             source_centroid: np.ndarray = None,
                             aligned_centroid: np.ndarray = None) -> np.ndarray:
    """
    두 점 집합 사이의 강체 변환 행렬을 계산합니다.
    
    SVD 기반 Procrustes 분석을 사용합니다.
    
    Args:
        source_pts: 원본 점 배열 (N x 3)
        aligned_pts: 정렬된 점 배열 (N x 3)
        source_centroid: 원본 중심점 (없으면 계산)
        aligned_centroid: 정렬된 중심점 (없으면 계산)
        
    Returns:
        4x4 변환 행렬
    """
    if source_centroid is None:
        source_centroid = np.mean(source_pts, axis=0)
    if aligned_centroid is None:
        aligned_centroid = np.mean(aligned_pts, axis=0)
    
    # 중심점 기준으로 정규화
    src_centered = source_pts - source_centroid
    aligned_centered = aligned_pts - aligned_centroid
    
    # SVD로 회전 행렬 계산
    H = src_centered.T @ aligned_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 반사 행렬 방지 (det = 1 보장)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 이동 벡터 계산
    t = aligned_centroid - R @ source_centroid
    
    # 4x4 변환 행렬 구성
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    
    return transform


def _extract_faces_by_mask(mesh: trimesh.Trimesh, mask: np.ndarray) -> trimesh.Trimesh:
    """마스크를 사용하여 메쉬에서 부분 메쉬를 추출합니다."""
    filtered_faces = mesh.faces[mask]
    if len(filtered_faces) == 0:
        return mesh.copy()
    
    unique_verts = np.unique(filtered_faces.flatten())
    vert_mapping = {old: new for new, old in enumerate(unique_verts)}
    new_vertices = mesh.vertices[unique_verts]
    new_faces = np.array([[vert_mapping[v] for v in face] for face in filtered_faces])
    
    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)


def align_to_smileguide(
    source_path: str,
    reference_path: str,
    visualize: bool = False,
    output_dir: str = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Source 메쉬를 SmileGuide 메쉬에 정합하는 변환 행렬을 계산합니다.
    
    내부적으로 다음 파이프라인을 수행합니다:
    1. 축 정렬 (OBB 기반 Z축 정렬, 대칭축 X축 정렬)
    2. ROI 추출 (치아 표면 영역)
    3. Coarse 정합 (Centroid 정렬, Y-sliding)
    4. Fine 정합 (Inverse ICP)
    
    Args:
        source_path: Source 메쉬 파일 경로 (smileArch.stl)
        reference_path: SmileGuide 메쉬 파일 경로
        visualize: 시각화 여부 (True면 각 단계 시각화 저장)
        output_dir: 시각화 저장 경로 (visualize=True일 때 필요)
        verbose: 진행 상황 출력 여부
        
    Returns:
        4x4 변환 행렬 (numpy.ndarray)
        - source.apply_transform(matrix) 하면 SmileGuide에 정합됨
    """
    from scipy.spatial import cKDTree
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("SmileArch → SmileGuide 정합")
        print("=" * 60)
        print(f"  Source: {source_path}")
        print(f"  Reference: {reference_path}")
    
    # =========================================================================
    # 메쉬 로드
    # =========================================================================
    original_mesh, source_mesh = load_mesh(str(source_path), downsample_target=10000)
    ref_mesh = trimesh.load(str(reference_path))
    
    if verbose:
        print(f"\n[Load] Source: {len(source_mesh.vertices):,} verts (downsampled)")
        print(f"[Load] Original: {len(original_mesh.vertices):,} verts")
        print(f"[Load] Reference: {len(ref_mesh.vertices):,} verts")
    
    # 누적 변환 행렬 초기화 (최종 결과)
    cumulative_transform = np.eye(4)
    
    # =========================================================================
    # Step 1: 축 정렬
    # =========================================================================
    if verbose:
        print("\n[Step 1] 축 정렬 (Axis Alignment)")
    
    vis_dir = str(output_dir / "visualization") if (visualize and output_dir) else None
    aligned_mesh, axis_info = align_model(source_mesh, 
                                          visualize=visualize, 
                                          output_dir=vis_dir)
    
    # 축 정렬 변환 행렬 계산 (SVD 기반)
    n_sample = min(1000, len(source_mesh.vertices))
    idx = np.linspace(0, len(source_mesh.vertices) - 1, n_sample, dtype=int)
    
    axis_transform = _compute_rigid_transform(
        source_pts=source_mesh.vertices[idx],
        aligned_pts=aligned_mesh.vertices[idx],
        source_centroid=source_mesh.centroid,
        aligned_centroid=aligned_mesh.centroid
    )
    
    cumulative_transform = axis_transform @ cumulative_transform
    
    if verbose:
        print(f"  축 정렬 완료")
    
    # =========================================================================
    # Step 2: ROI 추출
    # =========================================================================
    if verbose:
        print("\n[Step 2] ROI 추출 (ROI Extraction)")
    
    roi_mesh, roi_info = extract_roi(aligned_mesh, 
                                     visualize=visualize,
                                     output_dir=vis_dir)
    
    if verbose:
        print(f"  ROI: {len(roi_mesh.vertices):,} verts, {len(roi_mesh.faces):,} faces")
    
    # =========================================================================
    # Step 3: Coarse 정합
    # =========================================================================
    if verbose:
        print("\n[Step 3] Coarse 정합")
    
    coarse_mesh, coarse_info = coarse_align(
        roi_mesh, ref_mesh,
        visualize=visualize,
        output_dir=vis_dir
    )
    
    # Coarse 정합 변환 행렬 구성
    coarse_transform = np.eye(4)
    if 'translation' in coarse_info:
        coarse_transform[:3, 3] = coarse_info['translation']
    if 'z_offset' in coarse_info:
        coarse_transform[2, 3] += coarse_info['z_offset']
    if 'y_offset' in coarse_info:
        coarse_transform[1, 3] += coarse_info['y_offset']
    
    if verbose:
        print(f"  Coarse 정합 완료")
        print(f"  Y offset: {coarse_info.get('y_offset', 0):.2f}mm")
    
    # =========================================================================
    # Step 4: Fine 정합 (Inverse ICP)
    # =========================================================================
    if verbose:
        print("\n[Step 4] Fine 정합 (Inverse ICP)")
    
    # Outlier 검사
    ref_tree = cKDTree(ref_mesh.vertices)
    distances, _ = ref_tree.query(coarse_mesh.vertices, k=1)
    
    ref_max_y = ref_mesh.vertices[:, 1].max()
    outlier_mask = (coarse_mesh.vertices[:, 1] > ref_max_y + 10)
    outlier_ratio = outlier_mask.sum() / len(coarse_mesh.vertices)
    
    if verbose:
        print(f"  Outlier 비율: {outlier_ratio*100:.1f}%")
    
    # ICP용 메쉬 준비
    source_for_icp = coarse_mesh
    ref_for_icp = ref_mesh
    
    # Outlier가 많으면 XZ 영역 필터링
    if outlier_ratio > 0.005:
        if verbose:
            print(f"  > Outliers 감지! XZ 20-80% 영역만 사용")
        
        # Reference의 Z, X 범위 계산
        ref_z_min, ref_z_max = ref_mesh.vertices[:, 2].min(), ref_mesh.vertices[:, 2].max()
        z_range = ref_z_max - ref_z_min
        z_low, z_high = ref_z_min + z_range * 0.2, ref_z_min + z_range * 0.8
        
        ref_x_min, ref_x_max = ref_mesh.vertices[:, 0].min(), ref_mesh.vertices[:, 0].max()
        x_range = ref_x_max - ref_x_min
        x_low, x_high = ref_x_min + x_range * 0.2, ref_x_min + x_range * 0.8
        
        # Source 필터링
        src_centroids = coarse_mesh.triangles_center
        src_mask = ((src_centroids[:, 2] >= z_low) & (src_centroids[:, 2] <= z_high) &
                    (src_centroids[:, 0] >= x_low) & (src_centroids[:, 0] <= x_high))
        
        # Reference 필터링
        ref_centroids = ref_mesh.triangles_center
        ref_mask = ((ref_centroids[:, 2] >= z_low) & (ref_centroids[:, 2] <= z_high) &
                    (ref_centroids[:, 0] >= x_low) & (ref_centroids[:, 0] <= x_high))
        
        source_for_icp = _extract_faces_by_mask(coarse_mesh, src_mask)
        ref_for_icp = _extract_faces_by_mask(ref_mesh, ref_mask)
    
    # Point cloud로 변환
    source_pcd = trimesh_to_o3d(source_for_icp)
    target_pcd = trimesh_to_o3d(ref_for_icp)
    
    # Inverse ICP: Reference → Source 정합 후 역변환
    icp_result = o3d.pipelines.registration.registration_icp(
        target_pcd, source_pcd,  # Reference → Source
        max_correspondence_distance=3.0,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    # 역변환
    icp_inverse_transform = np.linalg.inv(icp_result.transformation)
    
    if verbose:
        print(f"  ICP Fitness: {icp_result.fitness:.4f}")
        print(f"  ICP RMSE: {icp_result.inlier_rmse:.4f}mm")
    
    # =========================================================================
    # 최종 변환 행렬 조합
    # =========================================================================
    # 변환 순서: axis_transform → coarse_transform → icp_inverse_transform
    # 실제 적용 순서: source → axis_aligned → coarse_aligned → fine_aligned
    
    final_transform = icp_inverse_transform @ coarse_transform @ cumulative_transform
    
    if verbose:
        print("\n" + "=" * 60)
        print("정합 완료!")
        print("=" * 60)
    
    # =========================================================================
    # 검증 (선택적)
    # =========================================================================
    if verbose:
        # 검증을 위해 원본 메쉬에 변환 적용
        test_mesh = original_mesh.copy()
        test_mesh.apply_transform(final_transform)
        
        error = compute_alignment_error(test_mesh, ref_mesh)
        print(f"\n[검증] 정합 오차:")
        print(f"  Mean distance: {error['mean']:.3f}mm")
        print(f"  Within 1mm: {error['within_1mm']*100:.1f}%")
        print(f"  Within 2mm: {error['within_2mm']*100:.1f}%")
    
    # =========================================================================
    # 시각화 저장 (선택적)
    # =========================================================================
    if visualize and output_dir:
        import matplotlib.pyplot as plt
        
        test_mesh = original_mesh.copy()
        test_mesh.apply_transform(final_transform)
        
        fig = plt.figure(figsize=(15, 5))
        
        def sample_verts(mesh, n=5000):
            v = mesh.vertices
            if len(v) > n:
                idx = np.random.choice(len(v), n, replace=False)
                return v[idx]
            return v
        
        src_v = sample_verts(test_mesh)
        ref_v = sample_verts(ref_mesh)
        
        # Top view (XY)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.scatter(ref_v[:, 0], ref_v[:, 1], s=0.5, alpha=0.3, c='blue', label='SmileGuide')
        ax1.scatter(src_v[:, 0], src_v[:, 1], s=0.5, alpha=0.3, c='red', label='Aligned')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y')
        ax1.set_title("Top View (XY)")
        ax1.set_aspect('equal'); ax1.legend()
        
        # Side view (YZ)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.scatter(ref_v[:, 1], ref_v[:, 2], s=0.5, alpha=0.3, c='blue', label='SmileGuide')
        ax2.scatter(src_v[:, 1], src_v[:, 2], s=0.5, alpha=0.3, c='red', label='Aligned')
        ax2.set_xlabel('Y'); ax2.set_ylabel('Z')
        ax2.set_title("Side View (YZ)")
        ax2.set_aspect('equal'); ax2.legend()
        
        # Front view (XZ)
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(ref_v[:, 0], ref_v[:, 2], s=0.5, alpha=0.3, c='blue', label='SmileGuide')
        ax3.scatter(src_v[:, 0], src_v[:, 2], s=0.5, alpha=0.3, c='red', label='Aligned')
        ax3.set_xlabel('X'); ax3.set_ylabel('Z')
        ax3.set_title("Front View (XZ)")
        ax3.set_aspect('equal'); ax3.legend()
        
        fig.suptitle("SmileArch → SmileGuide 정합 결과", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / "visualization" / "final_alignment_result.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"\n>>> 시각화 저장: {save_path}")
        
        # 변환 행렬도 저장
        np.save(str(output_dir / "final_transform.npy"), final_transform)
        if verbose:
            print(f">>> 변환 행렬 저장: {output_dir / 'final_transform.npy'}")
    
    return final_transform


# =============================================================================
# CLI 지원
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SmileArch를 SmileGuide에 정합하는 변환 행렬을 계산합니다."
    )
    parser.add_argument("--source", "-s", required=True, 
                        help="Source 메쉬 경로 (smileArch.stl)")
    parser.add_argument("--reference", "-r", required=True,
                        help="Reference/SmileGuide 메쉬 경로")
    parser.add_argument("--output", "-o", default=None,
                        help="출력 디렉토리 (기본: source 파일 위치)")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="시각화 저장 여부")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="진행 상황 출력 끄기")
    
    args = parser.parse_args()
    
    transform = align_to_smileguide(
        source_path=args.source,
        reference_path=args.reference,
        visualize=args.visualize,
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    print("\n변환 행렬:")
    print(transform)
