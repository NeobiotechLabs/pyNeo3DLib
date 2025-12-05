"""
ICP(Iterative Closest Point) 정합 관련 기능을 담당하는 모듈
"""
import open3d as o3d
import numpy as np
from typing import Tuple
from .constants import (
    ICPConfig,
    MultiScaleICPConfig
)


class ICPRegistration:
    """ICP 정합을 수행하는 클래스"""
    
    def __init__(self, 
                 max_corr_dist_p2plane: float = ICPConfig.MAX_CORRESPONDENCE_DISTANCE_P2PLANE,
                 max_iterations: int = ICPConfig.MAX_ITERATIONS,
                 relative_fitness_p2plane: float = ICPConfig.RELATIVE_FITNESS_P2PLANE,
                 relative_rmse_p2plane: float = ICPConfig.RELATIVE_RMSE_P2PLANE,
                 normal_radius_multiplier: float = ICPConfig.NORMAL_ESTIMATION_RADIUS_MULTIPLIER,
                 normal_max_nn: int = ICPConfig.NORMAL_ESTIMATION_MAX_NN):
        """
        Args:
            max_corr_dist_p2plane: Point-to-Plane ICP 최대 대응 거리
            max_iterations: ICP 최대 반복 횟수
            relative_fitness_p2plane: ICP 정합의 상대적인 적합도 변화량 임계값 (Point-to-Plane)
            relative_rmse_p2plane: ICP 정합의 상대적인 RMSE 변화량 임계값 (Point-to-Plane)
            normal_radius_multiplier: 법선 추정 반경 배율
            normal_max_nn: 법선 추정 최대 이웃 개수
        """
        self.max_corr_dist_p2plane = max_corr_dist_p2plane
        self.max_iterations = max_iterations
        self.relative_fitness_p2plane = relative_fitness_p2plane
        self.relative_rmse_p2plane = relative_rmse_p2plane
        self.normal_radius_multiplier = normal_radius_multiplier
        self.normal_max_nn = normal_max_nn

    
    def run_point_to_plane(self,
                          source_pcd: o3d.geometry.PointCloud,
                          target_pcd: o3d.geometry.PointCloud,
                          init_transformation: np.ndarray = None,
                          max_corr_dist: float = None,
                          max_iter: int = None) -> o3d.pipelines.registration.RegistrationResult:
        """
        Point-to-Plane ICP 정합 수행
        
        Args:
            source_pcd: 변환하려는 포인트 클라우드
            target_pcd: 타겟 포인트 클라우드
            init_transformation: 초기 변환 행렬 (기본값: 항등행렬)
            max_corr_dist: ICP 최대 대응 거리 (기본값: 설정된 값 사용)
            max_iter: ICP 최대 반복 횟수 (기본값: 설정된 값 사용)
            
        Returns:
            o3d.pipelines.registration.RegistrationResult: ICP 결과 객체
        """
        if init_transformation is None:
            init_transformation = np.eye(4)
        if max_corr_dist is None:
            max_corr_dist = self.max_corr_dist_p2plane
        if max_iter is None:
            max_iter = self.max_iterations
        
        # 법선 추정
        normal_radius = max_corr_dist * self.normal_radius_multiplier
        source_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, 
                max_nn=self.normal_max_nn
            )
        )
        target_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, 
                max_nn=self.normal_max_nn
            )
        )
        
        # Point-to-Plane ICP 실행
        reg = o3d.pipelines.registration.registration_icp(
            source_pcd, 
            target_pcd,
            max_corr_dist,
            init_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=self.relative_fitness_p2plane,relative_rmse=self.relative_rmse_p2plane,max_iteration=max_iter)
        )
        return reg
    
    def run_multi_scale_with_rmse(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        init_transformation: np.ndarray | None = None,
        target_rmse: float = MultiScaleICPConfig.TARGET_RMSE,
        rmse_tol: float = MultiScaleICPConfig.RMSE_TOLERANCE,
        verbose: bool = True
    ) -> o3d.pipelines.registration.RegistrationResult:

        """
        RMS 기반 Adaptive Multi-Scale Point-to-Plane ICP
        ------------------------------------------------
        coarse → mid → fine 단계로 진행하면서,
        각 단계마다 inlier_rmse를 확인하여:
        1) target_rmse 이하 도달하면 종료
        2) rmse 개선이 거의 없으면 종료
        """

        # 기본 초기 변환
        if init_transformation is None:
            init_transformation = np.eye(4)

        # (max_correspondence_dist, max_iteration)
        scales = MultiScaleICPConfig.SCALES
        if not scales:
            raise ValueError("MultiScaleICPConfig.SCALES cannot be empty for multi-scale ICP.")

        current_T = init_transformation
        prev_rmse = None
        last_result = None

        for idx, (dist, it) in enumerate(scales):

            if verbose:
                print(f"\n[Scale {idx+1}] dist={dist}, iter={it}")

            # 기존 run_point_to_plane() 활용
            result = self.run_point_to_plane(
                source_pcd=source_pcd,
                target_pcd=target_pcd,
                init_transformation=current_T,
                max_corr_dist=dist,
                max_iter=it
            )

            # update transform
            current_T = result.transformation
            last_result = result

            rmse = result.inlier_rmse
            fitness = result.fitness

            if verbose:
                print(f"  → RMS: {rmse:.5f}, fitness: {fitness:.4f}")

            # 1) 목표 RMS 이하이면 종료
            if rmse <= target_rmse:
                if verbose:
                    print(f"  → STOP: RMS {rmse:.5f} <= target {target_rmse}")
                break

            # 2) 이전 단계와 비교해 RMS 개선 거의 없으면 종료
            if prev_rmse is not None:
                if abs(prev_rmse - rmse) < rmse_tol:
                    if verbose:
                        print(f"  → STOP: RMS improvement < tolerance ({rmse_tol})")
                    break

            prev_rmse = rmse

        return last_result
