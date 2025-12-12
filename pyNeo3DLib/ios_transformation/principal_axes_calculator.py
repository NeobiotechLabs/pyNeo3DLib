"""
주축 계산 모듈

PCA를 사용하여 메시의 주축을 계산하는 클래스와 함수들입니다.
"""

import numpy as np
import trimesh
from scipy.linalg import eigh
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pyNeo3DLib.fileLoader.mesh import Mesh


class PrincipalAxesCalculator:
    """
    주축 계산 클래스
    
    PCA를 통해 메시의 주축을 계산하고,
    분산이 가장 작은 주축을 찾습니다.
    """
    
    def compute(
        self, 
        vertices: np.ndarray
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        """
        주축을 계산합니다.
        
        Args:
            vertices: 메시의 정점 좌표 배열 (N x 3)
            
        Returns:
            Tuple[principal_axes, closest_axis_idx, closest_axis_vector, centroid]
        """
        # PCA를 통한 주축 계산
        principal_axes, _, centroid = compute_principal_axes_from_vertices(
            vertices, 
            verbose=True
        )
        
        # 분산이 가장 작은 주축 계산
        minimum_variance_axis, _, _ = compute_minimum_variance_axis_from_vertices(
            vertices,
            verbose=True
        )
        print(f"[INFO] Minimum variance axis: {minimum_variance_axis}")
        
        # principal_axes에서 minimum_variance_axis와 가장 가까운 주축 찾기
        closest_axis = np.argmax(np.abs(np.dot(principal_axes, minimum_variance_axis)))
        closest_axis_vector = principal_axes[closest_axis]
        print(f"[INFO] Closest axis index: {closest_axis}")
        print(f"[INFO] Closest axis vector: {closest_axis_vector}")
        
        return principal_axes, closest_axis, closest_axis_vector, centroid
    
    def compute_z_axis_vector(
        self,
        ios_mesh: "Mesh",
        closest_axis_vector: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Z축 벡터를 계산합니다.
        
        IOS 메시의 평균 법선 벡터와 주축 벡터의 내적을 계산하여
        방향을 결정합니다.
        
        Args:
            ios_mesh: IOS 메시 객체
            closest_axis_vector: PCA로 계산된 주축 벡터
            
        Returns:
            방향이 결정된 Z축 벡터
        """
        if ios_mesh.normals is None:
            ios_mesh._compute_normals()
        
        ios_normals = np.asarray(ios_mesh.normals)
        ios_normals_mean = np.mean(ios_normals, axis=0)
        print(f"[INFO] IOS mesh normal mean: {ios_normals_mean}")
        
        # 내적으로 방향 확인
        inner_product = np.dot(closest_axis_vector, ios_normals_mean)
        if inner_product > 0:
            print("[INFO] Same direction")
            return closest_axis_vector
        else:
            print("[INFO] Opposite direction")
            return -closest_axis_vector


def compute_principal_axes_from_vertices(
    vertices: np.ndarray, 
    faces: Optional[np.ndarray] = None, 
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    메시 버텍스로부터 회전관성 주축을 계산합니다.
    trimesh 라이브러리를 사용하여 물리적으로 정확한 관성 텐서를 계산합니다.
    
    Args:
        vertices: 메시의 정점 좌표 배열 (N x 3)
        faces: 메시의 면 정보 (선택사항, 제공시 더 정확한 계산)
        verbose: 계산 과정을 출력할지 여부
    
    Returns:
        principal_axes: 회전관성 주축 (3x3 행렬, 각 열이 주축 벡터)
        eigenvalues: 각 주축에 대한 관성 모멘트 값 (작은 순서대로)
        centroid: 메시의 무게중심
    
    Example:
        >>> vertices = mesh.vertices
        >>> axes, moments, center = compute_principal_axes_from_vertices(vertices)
        >>> print(f"Axis 1: {axes[:, 0]}")
    """
    # trimesh 객체 생성
    if faces is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        # faces가 없으면 convex hull 사용
        mesh = trimesh.convex.convex_hull(vertices)
    
    # trimesh의 내장 함수 사용
    principal_axes = mesh.principal_inertia_vectors
    eigenvalues = mesh.principal_inertia_components
    centroid = mesh.center_mass
    
    if verbose:
        print(f'[INFO] Center of mass: {centroid}')
        print(f'[INFO] Principal inertia moments (eigenvalues): {eigenvalues}')
        print(f'[INFO] Principal inertia axes (eigenvectors):')
        print(f'{principal_axes}')
        print(f'   - Axis 1 (min inertia): {principal_axes[:, 0]}')
        print(f'   - Axis 2 (mid inertia): {principal_axes[:, 1]}')
        print(f'   - Axis 3 (max inertia): {principal_axes[:, 2]}')
    
    return principal_axes, eigenvalues, centroid


def compute_minimum_variance_axis_from_vertices(
    vertices: np.ndarray, 
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA(주성분 분석)를 사용하여 메시 버텍스의 분산이 가장 작은 주축을 계산합니다.
    scipy를 사용하여 효율적으로 계산합니다.
    
    Args:
        vertices: 메시의 정점 좌표 배열 (N x 3)
        verbose: 계산 과정을 출력할지 여부
    
    Returns:
        minimum_variance_axis: 분산이 가장 작은 주축 벡터 (3,)
        all_axes: 모든 주성분 축 (3x3 행렬, 각 열이 주축 벡터, 분산 작은 순서)
        variances: 각 주축에 대한 분산 값 (작은 순서대로)
    
    Example:
        >>> vertices = mesh.vertices
        >>> min_axis, all_axes, variances = compute_minimum_variance_axis_from_vertices(vertices)
        >>> print(f"Minimum variance axis: {min_axis}")
    
    Note:
        PCA는 공분산 행렬을 사용하여 데이터의 주성분을 찾습니다.
        분산이 가장 작은 축은 데이터가 가장 평평한 방향을 나타냅니다.
    """
    # 1. 데이터의 중심 계산
    centroid = np.mean(vertices, axis=0)
    if verbose:
        print(f'[INFO] PCA center: {centroid}')
    
    # 2. 중심을 원점으로 이동
    centered_vertices = vertices - centroid
    
    # 3. 공분산 행렬 계산
    covariance_matrix = np.cov(centered_vertices.T)
    if verbose:
        print(f'[INFO] Covariance matrix:')
        print(f'{covariance_matrix}')
    
    # 4. scipy의 eigh로 고유값/고유벡터 계산 (대칭 행렬에 최적화)
    eigenvalues, eigenvectors = eigh(covariance_matrix)
    
    # 5. 고유값(분산)이 작은 순서대로 정렬
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # 6. 결과 추출
    all_axes = eigenvectors
    variances = eigenvalues
    minimum_variance_axis = eigenvectors[:, 0]
    
    if verbose:
        print(f'[INFO] Principal component variances (eigenvalues): {variances}')
        print(f'[INFO] Variance ratio: {variances / np.sum(variances) * 100}%')
        print(f'[INFO] Principal component axes (eigenvectors):')
        print(f'{all_axes}')
        print(f'   - Axis 1 (min variance): {all_axes[:, 0]} (variance: {variances[0]:.2f})')
        print(f'   - Axis 2 (mid variance): {all_axes[:, 1]} (variance: {variances[1]:.2f})')
        print(f'   - Axis 3 (max variance): {all_axes[:, 2]} (variance: {variances[2]:.2f})')
        print(f'[INFO] Minimum variance axis: {minimum_variance_axis}')
    
    return minimum_variance_axis, all_axes, variances
