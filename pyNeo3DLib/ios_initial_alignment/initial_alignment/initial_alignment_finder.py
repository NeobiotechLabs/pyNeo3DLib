"""
초기 정렬(Initial Alignment) 찾기를 담당하는 모듈
OBB(Oriented Bounding Box) 기반으로 변환 행렬 후보를 생성하고 평가
"""
import open3d as o3d
import numpy as np
import copy
from typing import List, Tuple
from .preprocessing import MeshPreprocessor
from .constants import (
    InitialAlignmentConfig
)


class InitialAlignmentFinder:
    """초기 정렬을 위한 변환 행렬 후보 탐색 클래스"""
    
    def __init__(self,
                 mesh_preprocessor: MeshPreprocessor = None,
                 top_k: int = InitialAlignmentConfig.TOP_K,
                 rmse_step: int = InitialAlignmentConfig.RMSE_STEP):
        """
        Args:
            mesh_preprocessor: 메시 전처리기 (None이면 새로 생성)
            top_k: 상위 몇 개의 후보를 반환할지
            rmse_step: RMSE 계산시 샘플링 간격
        """
        self.mesh_preprocessor = mesh_preprocessor or MeshPreprocessor()
        self.top_k = top_k
        self.rmse_step = rmse_step
    
    def find_best_transforms(self,
                            source_pcd: o3d.geometry.PointCloud,
                            target_pcd: o3d.geometry.PointCloud,
                            top_k: int = None,
                            verbose: bool = True) -> List[Tuple[float, np.ndarray]]:
        """
        OBB 기반으로 초기 변환 후보들을 찾고 RMSE로 평가
        
        Args:
            source_pcd: 소스 포인트 클라우드
            target_pcd: 타겟 포인트 클라우드
            top_k: 상위 몇 개의 후보를 반환할지 (None이면 초기화시 설정값 사용)
            verbose: 진행 상황 출력 여부
            
        Returns:
            List[Tuple[float, np.ndarray]]: (RMSE, 변환행렬) 튜플의 리스트 (RMSE 오름차순)
        """
        if top_k is None:
            top_k = self.top_k
        
        # 1. OBB 정보 계산
        center_target, axes_target, _ = self.mesh_preprocessor.compute_OBB_info(target_pcd)
        center_source, axes_source, _ = self.mesh_preprocessor.compute_OBB_info(source_pcd)
        
        # 2. 후보 회전 계산
        rotation_candidates = self.mesh_preprocessor.compute_rotation_candidates(
            axes_source, axes_target
        )
        
        # 3. 변환 행렬 리스트 생성
        transforms = [
            self.mesh_preprocessor.build_transform(R, center_source, center_target) 
            for R in rotation_candidates
        ]
        
        if verbose:
            print(f"생성된 변환 행렬 후보 개수: {len(transforms)}")
        
        # 4. RMSE 계산 및 후보 평가
        results = self._evaluate_transforms(
            source_pcd, target_pcd, transforms, verbose
        )
        
        # 5. RMSE 기준 정렬하여 상위 k개 반환
        results.sort(key=lambda x: x[0])
        top_results = results[:top_k]
        
        if verbose:
            print(f"\n▶ 상위 {top_k}개 변환행렬")
            for rank, (err, T) in enumerate(top_results, start=1):
                print(f"Rank {rank} - RMSE: {err:.4f}")
        
        return top_results
    
    def _evaluate_transforms(self,
                           source_pcd: o3d.geometry.PointCloud,
                           target_pcd: o3d.geometry.PointCloud,
                           transforms: List[np.ndarray],
                           verbose: bool = True) -> List[Tuple[float, np.ndarray]]:
        """
        변환 행렬 후보들을 RMSE로 평가
        
        Args:
            source_pcd: 소스 포인트 클라우드
            target_pcd: 타겟 포인트 클라우드
            transforms: 평가할 변환 행렬 리스트
            verbose: 진행 상황 출력 여부
            
        Returns:
            List[Tuple[float, np.ndarray]]: (RMSE, 변환행렬) 튜플의 리스트
        """
        results = []
        
        for i, T in enumerate(transforms):
            src_copy = copy.deepcopy(source_pcd)
            src_copy.transform(T)
            
            err = self.mesh_preprocessor.compute_rmse_bidirectional(
                src_copy, target_pcd, step=self.rmse_step
            )
            
            results.append((err, T))
            
            if verbose:
                print(f"Candidate {i}: RMSE = {err:.4f}")
        
        return results
    
    def find_best_transform_single(self,
                                  source_pcd: o3d.geometry.PointCloud,
                                  target_pcd: o3d.geometry.PointCloud,
                                  verbose: bool = True) -> Tuple[float, np.ndarray]:
        """
        최적의 단일 변환 행렬 반환 (top_k=1)
        
        Args:
            source_pcd: 소스 포인트 클라우드
            target_pcd: 타겟 포인트 클라우드
            verbose: 진행 상황 출력 여부
            
        Returns:
            Tuple[float, np.ndarray]: (최적 RMSE, 최적 변환행렬)
        """
        results = self.find_best_transforms(
            source_pcd, target_pcd, top_k=1, verbose=verbose
        )
        return results[0]
    
    def evaluate_single_transform(self,
                                 source_pcd: o3d.geometry.PointCloud,
                                 target_pcd: o3d.geometry.PointCloud,
                                 transform: np.ndarray) -> float:
        """
        단일 변환 행렬의 RMSE 평가
        
        Args:
            source_pcd: 소스 포인트 클라우드
            target_pcd: 타겟 포인트 클라우드
            transform: 평가할 변환 행렬
            
        Returns:
            float: RMSE 값
        """
        src_copy = copy.deepcopy(source_pcd)
        src_copy.transform(transform)
        
        return self.mesh_preprocessor.compute_rmse_bidirectional(
            src_copy, target_pcd, step=self.rmse_step
        )

