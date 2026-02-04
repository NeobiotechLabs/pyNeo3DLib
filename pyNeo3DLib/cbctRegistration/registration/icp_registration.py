"""
ICP 정합 모듈
"""
import numpy as np
import open3d as o3d
import copy
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ICPResult:
    """ICP 정합 결과"""
    transformation: np.ndarray  # 4x4 변환행렬
    fitness: float  # 정합 품질 (0~1)
    inlier_rmse: float  # RMSE 값 (mm)
    method: str  # 사용된 방법


class ICPRegistration:
    """ICP 정합을 담당하는 클래스"""
    
    def __init__(
        self,
        max_correspondence_distance: float = 1.0,
        max_iteration: int = 100,
        relative_fitness: float = 1e-6,
        relative_rmse: float = 1e-6,
        normal_search_radius: float = 2.0,
        normal_max_nn: int = 30
    ):
        """
        Args:
            max_correspondence_distance: ICP 최대 대응점 거리 (mm)
            max_iteration: ICP 최대 반복 횟수
            relative_fitness: 상대 fitness 수렴 기준
            relative_rmse: 상대 RMSE 수렴 기준
            normal_search_radius: 법선 추정 반경
            normal_max_nn: 법선 추정 최대 이웃 개수
        """
        self.max_correspondence_distance = max_correspondence_distance
        self.max_iteration = max_iteration
        self.relative_fitness = relative_fitness
        self.relative_rmse = relative_rmse
        self.normal_search_radius = normal_search_radius
        self.normal_max_nn = normal_max_nn
    
    def _point_to_point_icp(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        verbose: bool = True
    ) -> o3d.pipelines.registration.RegistrationResult:
        """
        Point-to-Point ICP 실행
        
        Args:
            source_pcd: 소스 포인트 클라우드
            target_pcd: 타겟 포인트 클라우드
            verbose: 상세 출력 여부
        
        Returns:
            o3d.pipelines.registration.RegistrationResult: ICP 결과
        """
        if verbose:
            print("\n[방법 1] Point-to-Point ICP")
        
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            max_correspondence_distance=self.max_correspondence_distance,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iteration,
                relative_fitness=self.relative_fitness,
                relative_rmse=self.relative_rmse
            )
        )
        
        if verbose:
            print(f"  Fitness: {icp_result.fitness:.6f}")
            print(f"  RMSE: {icp_result.inlier_rmse:.6f} mm")
        
        return icp_result
    
    def _point_to_plane_icp(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        verbose: bool = True
    ) -> o3d.pipelines.registration.RegistrationResult:
        """
        Point-to-Plane ICP 실행 (더 정확하지만 법선 필요)
        
        Args:
            source_pcd: 소스 포인트 클라우드
            target_pcd: 타겟 포인트 클라우드
            verbose: 상세 출력 여부
        
        Returns:
            o3d.pipelines.registration.RegistrationResult: ICP 결과
        """
        if verbose:
            print("\n[방법 2] Point-to-Plane ICP")
        
        # 타겟 포인트 클라우드에 법선 추정
        target_pcd_with_normals = copy.deepcopy(target_pcd)
        target_pcd_with_normals.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_search_radius,
                max_nn=self.normal_max_nn
            )
        )
        
        # Tukey loss를 사용한 robust ICP
        # k 값이 작을수록 outlier에 강건함 (일반적으로 k=0.5~1.0 사용)
        tukey_loss = o3d.pipelines.registration.TukeyLoss(k=1)
        
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd_with_normals,
            max_correspondence_distance=self.max_correspondence_distance,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(tukey_loss),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iteration,
                relative_fitness=self.relative_fitness,
                relative_rmse=self.relative_rmse
            )
        )
        
        if verbose:
            print(f"  Fitness: {icp_result.fitness:.6f}")
            print(f"  RMSE: {icp_result.inlier_rmse:.6f} mm")
        
        return icp_result
    
    def register(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        verbose: bool = True
    ) -> ICPResult:
        """
        Point-to-Plane ICP를 사용하여 정합 수행
        
        Args:
            source_pcd: 소스 포인트 클라우드 (정합될 대상, CBCT)
            target_pcd: 타겟 포인트 클라우드 (정합 목표, FaceScan)
            verbose: 상세 출력 여부
        
        Returns:
            ICPResult: ICP 정합 결과
        """
        if verbose:
            print("\n[ICP 정밀 정합 실행]")
            print("-" * 40)
        
        # Point-to-Plane ICP
        icp_p2plane = self._point_to_plane_icp(source_pcd, target_pcd, verbose)
        
        if verbose:
            print(f"최종 Fitness: {icp_p2plane.fitness:.6f}")
            print(f"최종 RMSE: {icp_p2plane.inlier_rmse:.6f} mm")
            print(f"\n최종 변환행렬:\n{icp_p2plane.transformation}")
        
        return ICPResult(
            transformation=icp_p2plane.transformation,
            fitness=icp_p2plane.fitness,
            inlier_rmse=icp_p2plane.inlier_rmse,
            method="Point-to-Plane"
        )


