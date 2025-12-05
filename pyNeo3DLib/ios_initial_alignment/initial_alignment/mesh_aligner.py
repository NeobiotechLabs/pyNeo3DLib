"""
메시 정렬 전체 프로세스를 조율하는 모듈
"""
import open3d as o3d
import numpy as np
import copy
from typing import Tuple, List, Optional
from .preprocessing import MeshPreprocessor
from .mesh_loader import MeshLoader
from .initial_alignment_finder import InitialAlignmentFinder
from .icp_registration import ICPRegistration
from .constants import (
    SamplingConfig,
    LogMessages
)


class MeshAligner:
    """메시 정렬을 위한 통합 클래스"""
    
    def __init__(self,
                 mesh_preprocessor: Optional[MeshPreprocessor] = None,
                 mesh_loader: Optional[MeshLoader] = None,
                 alignment_finder: Optional[InitialAlignmentFinder] = None,
                 icp_registration: Optional[ICPRegistration] = None):
        """
        Args:
            mesh_preprocessor: 메시 전처리기 (None이면 새로 생성)
            mesh_loader: 메시 로더 (None이면 새로 생성)
            alignment_finder: 초기 정렬 파인더 (None이면 새로 생성)
            icp_registration: ICP 정합기 (None이면 새로 생성)
        """
        self.mesh_preprocessor = mesh_preprocessor or MeshPreprocessor()
        self.mesh_loader = mesh_loader or MeshLoader()
        self.alignment_finder = alignment_finder or InitialAlignmentFinder(
            mesh_preprocessor=self.mesh_preprocessor
        )
        self.icp_registration = icp_registration or ICPRegistration()
    
    def align_from_files(self,
                        target_stl_path: str,
                        control_stl_path: str,
                        downsample_voxel_size: float = SamplingConfig.DEFAULT_VOXEL_SIZE,
                        verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """
        STL 파일로부터 메시를 로드하여 정렬
        
        Args:
            target_stl_path: 타겟 STL 파일 경로
            control_stl_path: 컨트롤 STL 파일 경로 (이 모델이 타겟으로 변환됨)
            downsample_voxel_size: 다운샘플링 복셀 크기
            verbose: 진행 상황 출력 여부
            
        Returns:
            Tuple[np.ndarray, dict]: (4x4 변환행렬, 결과 정보 딕셔너리)
        """
        # 1. 메시 로드 및 전처리
        if verbose:
            print(LogMessages.MESH_PREPROCESSING)
        
        target_mesh, control_mesh = self.mesh_loader.load_mesh_pair(
            target_stl_path, control_stl_path
        )
        
        # 2. 메시 객체로 정렬 수행
        return self.align_from_meshes(
            target_mesh, control_mesh, 
            downsample_voxel_size=downsample_voxel_size,
            verbose=verbose
        )
    
    def align_from_meshes(self,
                         target_mesh: o3d.geometry.TriangleMesh,
                         control_mesh: o3d.geometry.TriangleMesh,
                         downsample_voxel_size: float = SamplingConfig.DEFAULT_VOXEL_SIZE,
                         verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """
        메시 객체로부터 직접 정렬 수행
        
        Args:
            target_mesh: 타겟 메시 객체
            control_mesh: 컨트롤 메시 객체 (이 모델이 타겟으로 변환됨)
            downsample_voxel_size: 다운샘플링 복셀 크기
            verbose: 진행 상황 출력 여부
            
        Returns:
            Tuple[np.ndarray, dict]: (4x4 변환행렬, 결과 정보 딕셔너리)
        """
        # 1. 메시 전처리
        if verbose:
            print(LogMessages.MESH_PREPROCESSING)
        
        target_pcd = self.mesh_preprocessor.downsample_mesh(
            target_mesh, voxel_size=downsample_voxel_size
        )
        control_pcd = self.mesh_preprocessor.downsample_mesh(
            control_mesh, voxel_size=downsample_voxel_size
        )
        
        if verbose:
            print(f"타겟 포인트 수: {len(target_pcd.points)}")
            print(f"컨트롤 포인트 수: {len(control_pcd.points)}")
        
        # 2. 포인트 클라우드로 정렬 수행
        return self.align_from_pointclouds(
            target_pcd, control_pcd,
            downsample_voxel_size=downsample_voxel_size,
            verbose=verbose
        )
    
    def align_from_pointclouds(self,
                              target_pcd: o3d.geometry.PointCloud,
                              control_pcd: o3d.geometry.PointCloud,
                              downsample_voxel_size: float = SamplingConfig.DEFAULT_VOXEL_SIZE,
                              verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """
        포인트 클라우드로부터 직접 정렬 수행
        
        Args:
            target_pcd: 타겟 포인트 클라우드
            control_pcd: 컨트롤 포인트 클라우드
            downsample_voxel_size: 다운샘플링 복셀 크기 (결과 정보에 포함)
            verbose: 진행 상황 출력 여부
            
        Returns:
            Tuple[np.ndarray, dict]: (4x4 변환행렬, 결과 정보 딕셔너리)
        """
        # 1. 초기 변환 후보 찾기
        if verbose:
            print(f"\n{LogMessages.INITIAL_ALIGNMENT}")
        
        top_transforms = self.alignment_finder.find_best_transforms(
            control_pcd, target_pcd, verbose=verbose
        )
        
        # 2. ICP로 정밀 정렬
        if verbose:
            print(f"\n{LogMessages.ICP_REFINEMENT}")
        
        final_transform, final_rmse, final_fitness = self.refine_with_icp(
            control_pcd, target_pcd, top_transforms, verbose=verbose
        )
        
        # 3. 결과 정보 구성
        result_info = {
            'rmse': final_rmse,
            'fitness': final_fitness,
            'target_points': len(target_pcd.points),
            'control_points': len(control_pcd.points),
            'downsample_voxel_size': downsample_voxel_size
        }
        
        if verbose:
            print(f"\n{LogMessages.FINAL_RESULTS}")
            print(f"최종 RMSE: {final_rmse:.4f}")
            print(f"최종 Fitness: {final_fitness:.4f}")
            print("최종 변환 행렬:")
            print(final_transform)
        
        return final_transform, result_info
    
    def refine_with_icp(self,
                       source_pcd: o3d.geometry.PointCloud,
                       target_pcd: o3d.geometry.PointCloud,
                       initial_transforms: List[Tuple[float, np.ndarray]],
                       verbose: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        초기 변환들에 대해 ICP를 수행하여 최적의 변환 행렬 찾기
        
        Args:
            source_pcd: 소스 포인트 클라우드
            target_pcd: 타겟 포인트 클라우드
            initial_transforms: 초기 변환 후보들 [(RMSE, 변환행렬), ...]
            verbose: 진행 상황 출력 여부
            
        Returns:
            Tuple[np.ndarray, float, float]: (최종 변환행렬, RMSE, Fitness)
        """
        best_reg = None
        best_fitness = -1.0
        best_rank = None
        best_combined_transform = None
        
        for rank, (initial_rmse, T_init) in enumerate(initial_transforms, start=1):
            if verbose:
                print(f"\n▶ Rank {rank} 후보 ICP 시작 (초기 RMSE={initial_rmse:.4f})")
            
            # 깊은 복사
            src_copy = copy.deepcopy(source_pcd)
            tgt_copy = copy.deepcopy(target_pcd)
            
            # 초기 변환 적용
            src_copy.transform(T_init)
            
            # ICP 실행 (Point-to-Plane)
            try:
                reg_icp = self.icp_registration.run_multi_scale_with_rmse(
                    src_copy, tgt_copy, init_transformation=np.eye(4), verbose=verbose
                )
            except ValueError as e:
                if verbose:
                    print(f"Skipping ICP refinement due to error: {e}")
                continue

            if reg_icp is None:
                if verbose:
                    print(f"Rank {rank} ICP 결과가 None입니다. 이 후보를 건너뜝니다.")
                # None이면 이 후보는 유효하지 않으므로 다음 후보로 넘어감
                continue
            
            if verbose:
                print(f"Rank {rank} ICP RMSE: {reg_icp.inlier_rmse:.4f}")
                print(f"Rank {rank} ICP Fitness: {reg_icp.fitness:.4f}")
            
            # Fitness 기준으로 최적 결과 갱신
            if reg_icp.fitness > best_fitness:
                best_fitness = reg_icp.fitness
                best_reg = reg_icp
                best_rank = rank
                # 초기 변환과 ICP 변환을 결합
                best_combined_transform = reg_icp.transformation @ T_init
        
        if best_reg is None:
            # 모든 ICP 시도가 실패했을 경우 기본값 반환 또는 에러 처리
            if verbose:
                print("모든 ICP 시도가 실패했습니다. 기본값을 반환합니다.")
            return np.eye(4), 0.0, 0.0 # 적절한 기본값 또는 에러 처리

        if verbose:
            print(f"\n▶ 최적 ICP 결과")
            print(f"Rank {best_rank} 선택됨 (Fitness={best_fitness:.4f})")
        
        return best_combined_transform, best_reg.inlier_rmse, best_fitness


# ============================================================================
# 편의 함수들 (하위 호환성 유지)
# ============================================================================

def align_3d_meshes(target_stl_path: str,
                   control_stl_path: str,
                   downsample_voxel_size: float = SamplingConfig.DEFAULT_VOXEL_SIZE) -> Tuple[np.ndarray, dict]:
    """
    두 개의 STL 파일을 정렬하여 최적의 변환 행렬을 찾음
    (하위 호환성을 위한 편의 함수)
    
    Args:
        target_stl_path: 타겟 STL 파일 경로
        control_stl_path: 컨트롤 STL 파일 경로
        downsample_voxel_size: 다운샘플링 복셀 크기
        
    Returns:
        Tuple[np.ndarray, dict]: (4x4 변환행렬, 결과 정보 딕셔너리)
    """
    aligner = MeshAligner()
    return aligner.align_from_files(
        target_stl_path, control_stl_path,
        downsample_voxel_size=downsample_voxel_size,
        verbose=True
    )


def align_meshes_direct(target_mesh: o3d.geometry.TriangleMesh,
                       control_mesh: o3d.geometry.TriangleMesh,
                       downsample_voxel_size: float = SamplingConfig.DEFAULT_VOXEL_SIZE) -> Tuple[np.ndarray, dict]:
    """
    두 개의 메쉬 객체를 직접 받아서 정렬
    (하위 호환성을 위한 편의 함수)
    
    Args:
        target_mesh: 타겟 메쉬 객체
        control_mesh: 컨트롤 메쉬 객체
        downsample_voxel_size: 다운샘플링 복셀 크기
        
    Returns:
        Tuple[np.ndarray, dict]: (4x4 변환행렬, 결과 정보 딕셔너리)
    """
    aligner = MeshAligner()
    return aligner.align_from_meshes(
        target_mesh, control_mesh,
        downsample_voxel_size=downsample_voxel_size,
        verbose=True
    )

