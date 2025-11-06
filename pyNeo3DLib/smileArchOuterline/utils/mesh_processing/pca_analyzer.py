import numpy as np
from typing import Tuple, Optional

class PCAAnalyzer:
    """메쉬에 대한 PCA(주성분 분석) 계산을 처리하는 클래스"""

    def __init__(self):
        self.pca_eigenvalues: Optional[np.ndarray] = None
        self.pca_eigenvectors: Optional[np.ndarray] = None
        self.min_variance_axis: Optional[np.ndarray] = None
        self._pca_computed = False

    def compute_principal_axes(self, vertices: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        주성분 분석(PCA) 수행 - 분산 기반 분석
        
        공분산 행렬의 고유값 분해를 통해 메쉬의 주요 분산 방향을 계산합니다.
        
        Args:
            vertices: 메쉬의 정점 배열
            center: 메쉬의 중심점
            
        Returns:
            eigenvalues: 고윳값 배열 (오름차순, 3)
            eigenvectors: 고유벡터 행렬 (3 x 3, 각 열이 고유벡터)
            min_variance_axis: 최소 분산 축 (3,)
        """
        # 중심화된 좌표
        centered_vertices = vertices - center
        
        # 공분산 행렬 계산 및 고유값 분해
        cov_matrix = np.cov(centered_vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 결과 저장
        self.pca_eigenvalues = eigenvalues
        self.pca_eigenvectors = eigenvectors
        self.min_variance_axis = eigenvectors[:, 0]  # 최소 고윳값에 대응
        
        self._pca_computed = True
        
        return self.pca_eigenvalues, self.pca_eigenvectors, self.min_variance_axis
