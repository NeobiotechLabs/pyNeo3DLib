"""
주축(Principal Axis) 계산을 담당하는 클래스
PCA와 관성 주축(Inertia) 분석을 통해 메쉬의 주축을 계산합니다.
"""

import numpy as np
import trimesh
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy.linalg import eigh


@dataclass
class PrincipalAxesResult:
    """주축 계산 결과를 담는 데이터 클래스"""
    principal_axes: np.ndarray
    max_variance_index: int
    max_variance_vector: np.ndarray
    centroid: np.ndarray


@dataclass
class InertiaAxesResult:
    """관성 주축 계산 결과를 담는 데이터 클래스"""
    principal_axes: np.ndarray
    eigenvalues: np.ndarray
    centroid: np.ndarray


@dataclass
class PCAResult:
    """PCA 분석 결과를 담는 데이터 클래스"""
    max_variance_axis: np.ndarray
    min_variance_axis: np.ndarray
    all_axes: np.ndarray
    variances: np.ndarray
    centroid: np.ndarray


class RotationAxisCalculator:
    """
    메쉬 버텍스의 주축을 계산하는 클래스
    
    주요 기능:
    - 관성 주축(Inertia Principal Axes) 계산
    - PCA를 통한 분산 기반 주축 계산
    - 분산이 가장 작은 축 찾기
    
    Example:
        >>> calculator = RotationAxisCalculator(verbose=True)
        >>> result = calculator.compute_principal_axes(vertices)
        >>> print(result.closest_axis_vector)
    """
    
    def __init__(self, verbose: bool = True):
        """
        RotationAxisCalculator 초기화
        
        Args:
            verbose: 상세 로그 출력 여부 (기본값: True)
        """
        self.verbose = verbose
    
    def compute_principal_axes(
        self,
        vertices: np.ndarray
    ) -> PrincipalAxesResult:
        """
        주축을 계산합니다.
        관성 주축과 PCA 분산 분석을 결합하여 가장 적합한 주축을 찾습니다.
        
        Args:
            vertices: 메쉬 버텍스 배열 (N x 3)
        
        Returns:
            PrincipalAxesResult: 주축 계산 결과
        """
        # 관성 주축 계산
        inertia_result = self.compute_inertia_axes(vertices)
        principal_axes = inertia_result.principal_axes
        centroid = inertia_result.centroid
        
        # 분산이 가장 큰 주축 계산
        pca_result = self.compute_pca_axes(vertices)
        max_variance_axis = pca_result.max_variance_axis

        if self.verbose:
            print(f"[INFO] Maximum variance axis: {max_variance_axis}")
        
        # principal_axes에서 max_variance_axis와 가장 가까운 주축 찾기
        max_variance_axis_index = np.argmax(np.abs(np.dot(principal_axes, max_variance_axis)))
        max_variance_axis_vector = principal_axes[max_variance_axis_index]
        
        return PrincipalAxesResult(
            principal_axes=principal_axes,
            max_variance_index=max_variance_axis_index,
            max_variance_vector=max_variance_axis_vector,
            centroid=centroid
        )
    
    def compute_inertia_axes(
        self,
        vertices: np.ndarray,
        faces: Optional[np.ndarray] = None
    ) -> InertiaAxesResult:
        """
        메시 버텍스로부터 회전관성 주축을 계산합니다.
        
        Args:
            vertices: 메쉬 버텍스 배열 (N x 3)
            faces: 메쉬 면 인덱스 배열 (선택사항, None이면 Convex Hull 사용)
        
        Returns:
            InertiaAxesResult: 관성 주축 계산 결과
        """
        if faces is not None:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            mesh = trimesh.convex.convex_hull(vertices)
        
        principal_axes = mesh.principal_inertia_vectors
        eigenvalues = mesh.principal_inertia_components
        centroid = mesh.center_mass
        
        if self.verbose:
            print(f'[INFO] Center of mass: {centroid}')
            print(f'[INFO] Principal moments of inertia (eigenvalues): {eigenvalues}')
            print(f'[INFO] Principal inertia axes (eigenvectors):\n{principal_axes}')
        
        return InertiaAxesResult(
            principal_axes=principal_axes,
            eigenvalues=eigenvalues,
            centroid=centroid
        )
    
    def compute_pca_axes(
        self,
        vertices: np.ndarray
    ) -> PCAResult:
        """
        PCA를 사용하여 메시 버텍스의 주축과 분산을 계산합니다.
        
        Args:
            vertices: 메쉬 버텍스 배열 (N x 3)
        
        Returns:
            PCAResult: PCA 분석 결과 (최소 분산 축, 모든 축, 분산값, 중심점)
        """
        centroid = np.mean(vertices, axis=0)
        centered_vertices = vertices - centroid
        covariance_matrix = np.cov(centered_vertices.T)
        eigenvalues, eigenvectors = eigh(covariance_matrix)
        
        # 고유값(분산) 기준 오름차순 정렬
        sorted_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        all_axes = eigenvectors
        variances = eigenvalues
        max_variance_axis = eigenvectors[:, 2]
        min_variance_axis = eigenvectors[:, 0]
        
        if self.verbose:
            print(f'[INFO] PCA center: {centroid}')
            print(f'[INFO] Principal component variances: {variances}')
        
        return PCAResult(
            max_variance_axis=max_variance_axis,
            min_variance_axis=min_variance_axis,
            all_axes=all_axes,
            variances=variances,
            centroid=centroid
        )
    

