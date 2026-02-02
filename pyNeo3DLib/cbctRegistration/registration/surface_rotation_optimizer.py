"""
SDF 기반 표면 정제 및 Z축 회전 최적화 모듈

이 모듈은 CBCT와 FaceScan 간의 최적 정합을 위해
Z축 회전 탐색을 수행하고 SDF(Signed Distance Function)를 
사용하여 표면 근처 포인트만 필터링합니다.
"""

import time
from typing import Tuple, List, Optional
import numpy as np
import open3d as o3d

from ..utils import apply_transform, compute_translation_matrix
from ..visualization import AlignmentVisualizer
from ..processing import GeometryProcessor


class SurfaceRotationOptimizer:
    """
    SDF 기반 표면 정제 및 Z축 회전 최적화 클래스
    
    주요 기능:
    - Z축 회전 각도 탐색을 통한 최적 정합 찾기
    - SDF를 사용한 표면 근처 포인트 필터링
    - 다운샘플링을 통한 성능 최적화
    - RMSE 기반 정합 품질 평가
    """
    
    def __init__(
        self,
        visualizer: Optional[AlignmentVisualizer] = None,
        geometry_processor: Optional[GeometryProcessor] = None
    ):
        """
        Args:
            visualizer: 시각화 도구 (None이면 생성)
            geometry_processor: 기하학 처리 도구 (None이면 생성)
        """
        self.visualizer = visualizer or AlignmentVisualizer()
        self.geometry_processor = geometry_processor or GeometryProcessor()
    
    def optimize_rotation(
        self,
        pcd_cbct_full: o3d.geometry.PointCloud,
        facescan_mesh: o3d.geometry.TriangleMesh,
        facescan_nose_point: np.ndarray,
        distance_threshold: float = 5.0,
        rotation_range: Tuple[float, float] = (-15, 15),
        rotation_step: float = 1.0,
        downsample_voxel_size: float = 2.0,
        visualize: bool = False,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float, float]:
        """
        SDF 기반 표면 필터링 + Z축 회전 탐색으로 최적 정합 찾기
        
        최적화 전략:
        - CBCT 다운샘플링으로 포인트 수 감소 (속도 향상 5~10배)
        - RaycastingScene 재사용으로 SDF 계산 최적화
        
        프로세스:
        1. CBCT 다운샘플링
        2. 각 회전 각도마다:
           - 다운샘플링된 CBCT를 FaceScan 코 중심 기준으로 Z축 회전
           - 회전된 CBCT에서 SDF로 FaceScan 표면 근처 포인트만 추출
           - 추출된 포인트와 FaceScan 표면 간 RMSE 계산
        3. 최소 RMSE를 가진 회전 각도 선택
        4. 최적 회전 변환 행렬 반환
        
        Args:
            pcd_cbct_full: CBCT 전체 포인트 클라우드
            facescan_mesh: FaceScan 메쉬
            facescan_nose_point: FaceScan 코 중심점 (회전 중심)
            distance_threshold: 표면 필터링 거리 임계값 (mm)
            rotation_range: Z축 회전 탐색 범위 (도 단위)
            rotation_step: 회전 탐색 간격 (도 단위)
            downsample_voxel_size: 다운샘플링 복셀 크기 (mm, 0이면 다운샘플링 안 함)
            visualize: 시각화 여부
            verbose: 상세 출력 여부
        
        Returns:
            Tuple[transform_matrix, best_angle, best_rmse]:
                - transform_matrix: 최적 회전 변환 행렬 (4x4)
                - best_angle: 최적 회전 각도 (도)
                - best_rmse: 최소 RMSE 값
        """
        start_time = time.time()
        
        if verbose:
            self._print_header(
                rotation_range, rotation_step, 
                distance_threshold, downsample_voxel_size,
                facescan_nose_point
            )
        
        # 1단계: CBCT 다운샘플링
        pcd_cbct_working = self._downsample_cbct(
            pcd_cbct_full, downsample_voxel_size, verbose
        )
        
        # 2단계: FaceScan 표면 샘플링 및 RaycastingScene 생성
        pcd_facescan_surface, scene = self._prepare_facescan_surface(
            facescan_mesh, verbose
        )
        
        # 3단계: 회전 탐색
        best_transform, best_angle, best_rmse, rmse_history = self._search_rotation(
            pcd_cbct_working, pcd_facescan_surface, scene,
            facescan_nose_point, distance_threshold,
            rotation_range, rotation_step,
            start_time, verbose
        )
        
        total_time = time.time() - start_time
        
        if verbose:
            self._print_results(
                best_angle, best_rmse, len(rmse_history), total_time
            )
        
        return best_transform, best_angle, best_rmse
    
    def _downsample_cbct(
        self,
        pcd_cbct_full: o3d.geometry.PointCloud,
        downsample_voxel_size: float,
        verbose: bool
    ) -> o3d.geometry.PointCloud:
        """CBCT 포인트 클라우드 다운샘플링"""
        if downsample_voxel_size > 0:
            if verbose:
                print(f"\n[1단계] CBCT 다운샘플링")
                print(f"  원본 포인트 수: {len(pcd_cbct_full.points):,}")
            
            pcd_cbct_working = pcd_cbct_full.voxel_down_sample(
                voxel_size=downsample_voxel_size
            )
            
            if verbose:
                reduction_rate = (1 - len(pcd_cbct_working.points) / len(pcd_cbct_full.points)) * 100
                print(f"  다운샘플링 후: {len(pcd_cbct_working.points):,}")
                print(f"  감소율: {reduction_rate:.1f}%")
        else:
            pcd_cbct_working = pcd_cbct_full
            if verbose:
                print(f"\n[1단계] 다운샘플링 스킵 (원본 사용)")
                print(f"  포인트 수: {len(pcd_cbct_working.points):,}")
        
        return pcd_cbct_working
    
    def _prepare_facescan_surface(
        self,
        facescan_mesh: o3d.geometry.TriangleMesh,
        verbose: bool
    ) -> Tuple[o3d.geometry.PointCloud, o3d.t.geometry.RaycastingScene]:
        """FaceScan 표면 샘플링 및 RaycastingScene 생성"""
        pcd_facescan_surface = facescan_mesh.sample_points_uniformly(
            number_of_points=50000
        )
        
        if verbose:
            print(f"\n[2단계] FaceScan 표면 샘플링 및 RaycastingScene 생성")
            print(f"  샘플링된 포인트 수: {len(pcd_facescan_surface.points):,}")
        
        # RaycastingScene 생성 (1회만) - 속도 최적화의 핵심!
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(facescan_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        
        if verbose:
            print(f"  RaycastingScene 생성 완료 (재사용)")
        
        return pcd_facescan_surface, scene
    
    def _search_rotation(
        self,
        pcd_cbct_working: o3d.geometry.PointCloud,
        pcd_facescan_surface: o3d.geometry.PointCloud,
        scene: o3d.t.geometry.RaycastingScene,
        facescan_nose_point: np.ndarray,
        distance_threshold: float,
        rotation_range: Tuple[float, float],
        rotation_step: float,
        start_time: float,
        verbose: bool
    ) -> Tuple[np.ndarray, float, float, List[Tuple[float, float]]]:
        """Z축 회전 각도 탐색"""
        best_rmse = float('inf')
        best_angle = 0.0
        best_transform = np.eye(4)
        rmse_history = []
        
        angles = np.arange(rotation_range[0], rotation_range[1] + rotation_step, rotation_step)
        
        if verbose:
            print(f"\n[3단계] 회전 탐색 시작")
            print(f"  총 {len(angles)}개 각도 테스트")
        
        for i, angle in enumerate(angles):
            # 회전 변환 행렬 생성
            transform = self._create_rotation_transform(angle, facescan_nose_point)
            
            # CBCT 회전
            pcd_cbct_rotated = apply_transform(pcd_cbct_working, transform)
            
            # 표면 근처 포인트 필터링 및 RMSE 계산
            rmse = self._compute_surface_rmse(
                pcd_cbct_rotated, pcd_facescan_surface, scene,
                distance_threshold
            )
            
            rmse_history.append((angle, rmse))
            
            # 최적값 업데이트
            if rmse < best_rmse:
                best_rmse = rmse
                best_angle = angle
                best_transform = transform.copy()
            
            if verbose and (i + 1) % 5 == 0:
                elapsed = time.time() - start_time
                print(f"  진행: {i+1}/{len(angles)} | "
                      f"현재 각도: {angle:+.1f}° | RMSE: {rmse:.3f} mm | "
                      f"최적: {best_angle:+.1f}° ({best_rmse:.3f} mm) | "
                      f"경과: {elapsed:.1f}초")
        
        return best_transform, best_angle, best_rmse, rmse_history
    
    def _create_rotation_transform(
        self,
        angle: float,
        rotation_center: np.ndarray
    ) -> np.ndarray:
        """Z축 회전 변환 행렬 생성 (회전 중심 기준)"""
        angle_rad = np.radians(angle)
        
        # 회전 행렬 (Z축)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 회전 중심으로 이동 -> 회전 -> 원위치
        translate_to_origin = compute_translation_matrix(-rotation_center)
        translate_back = compute_translation_matrix(rotation_center)
        
        # 최종 변환: 원위치 @ 회전 @ 중심이동
        return translate_back @ rotation_matrix @ translate_to_origin
    
    def _compute_surface_rmse(
        self,
        pcd_cbct_rotated: o3d.geometry.PointCloud,
        pcd_facescan_surface: o3d.geometry.PointCloud,
        scene: o3d.t.geometry.RaycastingScene,
        distance_threshold: float
    ) -> float:
        """표면 근처 포인트 필터링 후 RMSE 계산"""
        # SDF로 표면 근처 포인트만 필터링
        points = np.asarray(pcd_cbct_rotated.points)
        query_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        signed_distances = scene.compute_signed_distance(query_points).numpy()
        distances = np.abs(signed_distances)
        
        mask = distances < distance_threshold
        filtered_points = points[mask]
        
        # 포인트가 너무 적으면 무한대 반환
        if len(filtered_points) < 100:
            return float('inf')
        
        pcd_cbct_surface = o3d.geometry.PointCloud()
        pcd_cbct_surface.points = o3d.utility.Vector3dVector(filtered_points)
        
        # KDTree로 최근접 거리 계산
        tree = o3d.geometry.KDTreeFlann(pcd_facescan_surface)
        distances = np.zeros(len(filtered_points))
        
        for j, point in enumerate(filtered_points):
            [_, _, dist] = tree.search_knn_vector_3d(point, 1)
            distances[j] = np.sqrt(dist[0])
        
        return np.sqrt(np.mean(distances ** 2))
    
    def _visualize_results(
        self,
        pcd_cbct_full: o3d.geometry.PointCloud,
        facescan_mesh: o3d.geometry.TriangleMesh,
        pcd_facescan_surface: o3d.geometry.PointCloud,
        best_transform: np.ndarray,
        best_angle: float,
        best_rmse: float,
        distance_threshold: float,
        rmse_history: List[Tuple[float, float]]
    ):
        """결과 시각화"""
        # RMSE 그래프
        if len(rmse_history) > 0:
            self._plot_rmse_history(rmse_history, best_angle)
        
        # 최적 회전 결과 3D 시각화
        pcd_cbct_best = apply_transform(pcd_cbct_full, best_transform)
        pcd_cbct_best_surface = self.geometry_processor.filter_points_near_surface(
            pcd_cbct_best,
            facescan_mesh,
            distance_threshold=distance_threshold,
            verbose=False
        )
        
        self.visualizer.visualize_alignment(
            pcd_cbct_best_surface, pcd_facescan_surface,
            f"최적 회전 결과 (빨강:CBCT, 초록:FaceScan) - 각도: {best_angle:.1f}°, RMSE: {best_rmse:.3f}mm"
        )
    
    def _plot_rmse_history(
        self,
        rmse_history: List[Tuple[float, float]],
        best_angle: float
    ):
        """RMSE 히스토리 그래프 출력"""
        import matplotlib.pyplot as plt
        
        angles_list = [h[0] for h in rmse_history]
        rmse_list = [h[1] if h[1] != float('inf') else None for h in rmse_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(angles_list, rmse_list, 'b-', linewidth=2, label='RMSE')
        plt.axvline(x=best_angle, color='r', linestyle='--', linewidth=2, 
                   label=f'최적 각도: {best_angle:.1f}°')
        plt.xlabel('회전 각도 (도)', fontsize=12)
        plt.ylabel('RMSE (mm)', fontsize=12)
        plt.title('Z축 회전 각도별 RMSE', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
    
    def _print_header(
        self,
        rotation_range: Tuple[float, float],
        rotation_step: float,
        distance_threshold: float,
        downsample_voxel_size: float,
        facescan_nose_point: np.ndarray
    ):
        """헤더 출력"""
        print("\n[Step 9] SDF 기반 표면 정제 (Z축 회전 탐색) - 다운샘플링 최적화")
        print("=" * 60)
        print(f"  회전 범위: {rotation_range[0]}° ~ {rotation_range[1]}°")
        print(f"  회전 간격: {rotation_step}°")
        print(f"  표면 필터링 임계값: {distance_threshold} mm")
        print(f"  다운샘플링 복셀 크기: {downsample_voxel_size} mm")
        print(f"  회전 중심: {facescan_nose_point}")
    
    def _print_results(
        self,
        best_angle: float,
        best_rmse: float,
        num_angles: int,
        total_time: float
    ):
        """결과 출력"""
        print(f"\n[회전 탐색 완료]")
        print(f"  최적 회전 각도: {best_angle:+.1f}°")
        print(f"  최소 RMSE: {best_rmse:.3f} mm")
        print(f"  탐색한 각도 수: {num_angles}")
        print(f"  총 소요 시간: {total_time:.2f}초")
        print(f"  각도당 평균 시간: {total_time / num_angles:.3f}초")

