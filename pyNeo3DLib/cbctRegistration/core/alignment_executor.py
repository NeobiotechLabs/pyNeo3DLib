"""
정합 실행 모듈

CBCT와 FaceScan 간의 정합 로직을 담당합니다.

단일 책임: 포인트 클라우드 간 정합 수행
"""
import numpy as np
import open3d as o3d
from typing import Tuple, Optional

from ..config import AlignmentConfig
from ..utils import apply_transform, compute_translation_matrix
from ..types import AlignmentStepResult, ICPAlignmentResult
from ..registration import ICPRegistration, ICPResult


class AlignmentExecutor:
    """
    정합 실행 클래스
    
    담당 기능:
    - 초기 정렬 (중심점 기반)
    - ICP 정밀 정합 (Z축 그리드 탐색 포함)
    """
    
    def __init__(self, config: Optional[AlignmentConfig] = None):
        """
        Args:
            config: 정합 설정 (None일 경우 기본값 사용)
        """
        self.config = config if config is not None else AlignmentConfig()
        
        self.icp_registration = ICPRegistration(
            max_correspondence_distance=self.config.icp.max_correspondence_distance,
            max_iteration=self.config.icp.max_iteration,
            relative_fitness=self.config.icp.relative_fitness,
            relative_rmse=self.config.icp.relative_rmse,
            normal_search_radius=self.config.icp.normal_search_radius,
            normal_max_nn=self.config.icp.normal_max_nn
        )
    
    def compute_initial_alignment(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        verbose: bool = True
    ) -> AlignmentStepResult:
        """
        소스를 타겟에 중심점 기반으로 초기 정렬
        
        Args:
            source_pcd: 소스 포인트 클라우드 (이동 대상, CBCT)
            target_pcd: 타겟 포인트 클라우드 (목표 위치, FaceScan)
            verbose: 상세 출력 여부
        
        Returns:
            AlignmentStepResult: 정렬 결과
        """
        if verbose:
            print("\n[초기 정렬] (중심점 기반)")
            print("-" * 50)
        
        # 중심점 계산
        source_center = source_pcd.get_center()
        target_center = target_pcd.get_center()
        translation_vector = target_center - source_center
        
        # 변환 행렬 생성
        transform_matrix = compute_translation_matrix(translation_vector)
        
        # 변환 적용
        aligned_pcd = apply_transform(source_pcd, transform_matrix)
        
        if verbose:
            print(f"소스 중심: {source_center}")
            print(f"타겟 중심: {target_center}")
            print(f"이동 벡터: {translation_vector}")
            print(f"이동 거리: {np.linalg.norm(translation_vector):.2f} mm")
            print(f"\n변환 행렬:")
            print(transform_matrix)
        
        return AlignmentStepResult(
            aligned_pcd=aligned_pcd,
            transform_matrix=transform_matrix
        )
    
    def perform_icp_registration(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        z_search_range: Tuple[int, int] = (-10, 10),
        z_search_step: int = 1,
        verbose: bool = True
    ) -> ICPAlignmentResult:
        """
        ICP 정밀 정합 수행 (Z축 그리드 탐색 포함)
        
        Args:
            source_pcd: 소스 포인트 클라우드 (이동 대상, CBCT)
            target_pcd: 타겟 포인트 클라우드 (목표 위치, FaceScan)
            z_search_range: Z축 탐색 범위 (min, max)
            z_search_step: Z축 탐색 간격
            verbose: 상세 출력 여부
        
        Returns:
            ICPAlignmentResult: ICP 정합 결과
        """
        if verbose:
            print("\n[ICP 정밀 정합] (Z축 그리드 탐색)")
            print("-" * 50)
            print(f"Z축 탐색 범위: {z_search_range}, 간격: {z_search_step}mm")
        
        best_icp_result: Optional[ICPResult] = None
        best_z_offset = 0
        
        # Z축 그리드 탐색
        for z in range(z_search_range[0], z_search_range[1], z_search_step):
            # Z축 이동
            z_transform = compute_translation_matrix([0, 0, z])
            source_z_moved = apply_transform(source_pcd, z_transform)
            
            # ICP 수행
            icp_result = self.icp_registration.register(
                source_z_moved,
                target_pcd,
                verbose=False
            )
            
            if best_icp_result is None or icp_result.fitness > best_icp_result.fitness:
                best_icp_result = icp_result
                best_z_offset = z
        
        if verbose:
            print(f"\n최적 Z 오프셋: {best_z_offset}mm")
            print(f"ICP Fitness: {best_icp_result.fitness:.6f}")
            print(f"ICP RMSE: {best_icp_result.inlier_rmse:.6f}")
            print(f"ICP Method: {best_icp_result.method}")
        
        # 최종 변환 행렬 계산: ICP @ Z이동
        z_transform = compute_translation_matrix([0, 0, best_z_offset])
        best_transform = best_icp_result.transformation @ z_transform
        
        # 변환 적용
        aligned_pcd = apply_transform(source_pcd, best_transform)
        
        if verbose:
            print(f"\n최종 변환 행렬 (Z이동 + ICP):")
            print(best_transform)
        
        return ICPAlignmentResult(
            aligned_pcd=aligned_pcd,
            transform_matrix=best_transform,
            fitness=best_icp_result.fitness,
            inlier_rmse=best_icp_result.inlier_rmse,
            method=best_icp_result.method,
            best_z_offset=best_z_offset
        )
    
    def run_icp_only(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        max_correspondence_distance: Optional[float] = None,
        verbose: bool = True
    ) -> ICPAlignmentResult:
        """
        ICP 정합만 수행 (Z축 탐색 없이)
        
        Args:
            source_pcd: 소스 포인트 클라우드
            target_pcd: 타겟 포인트 클라우드
            max_correspondence_distance: 최대 대응 거리 (None이면 config 값 사용)
            verbose: 상세 출력 여부
        
        Returns:
            ICPAlignmentResult: ICP 정합 결과
        """
        if verbose:
            print("\n[ICP 정밀 정합] (표면 기반)")
            print("-" * 50)
        
        # 임시로 max_correspondence_distance 변경
        original_distance = self.icp_registration.max_correspondence_distance
        if max_correspondence_distance is not None:
            self.icp_registration.max_correspondence_distance = max_correspondence_distance
        
        # ICP 수행
        icp_result = self.icp_registration.register(
            source_pcd,
            target_pcd,
            verbose=verbose
        )
        
        # 원래 값으로 복원
        self.icp_registration.max_correspondence_distance = original_distance
        
        # 변환 적용
        aligned_pcd = apply_transform(source_pcd, icp_result.transformation)
        
        if verbose:
            print(f"\nICP 결과:")
            print(f"  Fitness: {icp_result.fitness:.6f}")
            print(f"  RMSE: {icp_result.inlier_rmse:.6f}")
            print(f"  Method: {icp_result.method}")
        
        return ICPAlignmentResult(
            aligned_pcd=aligned_pcd,
            transform_matrix=icp_result.transformation,
            fitness=icp_result.fitness,
            inlier_rmse=icp_result.inlier_rmse,
            method=icp_result.method,
            best_z_offset=0
        )
    
    def execute(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        verbose: bool = True
    ) -> Tuple[AlignmentStepResult, ICPAlignmentResult]:
        """
        전체 정합 파이프라인 실행 (초기 정렬 + ICP)
        
        Args:
            source_pcd: 소스 포인트 클라우드 (CBCT)
            target_pcd: 타겟 포인트 클라우드 (FaceScan)
            z_search_range: Z축 탐색 범위
            z_search_step: Z축 탐색 간격
            verbose: 상세 출력 여부
        
        Returns:
            Tuple[initial_result, icp_result]:
                - initial_result: 초기 정렬 결과
                - icp_result: ICP 정합 결과
        """
        # 1. 초기 정렬
        initial_result = self.compute_initial_alignment(
            source_pcd, target_pcd, verbose
        )
        
        # 2. ICP 정밀 정합
        icp_result = self.perform_icp_registration(
            initial_result.aligned_pcd,
            target_pcd,
            verbose=verbose
        )
        
        return initial_result, icp_result

