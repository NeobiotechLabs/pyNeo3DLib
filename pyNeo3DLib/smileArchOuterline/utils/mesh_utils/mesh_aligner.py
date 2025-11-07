"""
메쉬 정렬을 위한 모듈

3D 메쉬 데이터의 주축 분석 및 정렬을 수행합니다.
PCA와 관성 텐서 분석을 통해 메쉬의 주요 방향성을 파악합니다.
"""

import numpy as np
import pyvista as pv
from typing import Tuple, Optional
from pathlib import Path


class MeshAligner:
    """
    3D 메쉬의 정렬 및 주축 분석을 수행하는 클래스
    
    메쉬를 원점 중심으로 정렬하고, PCA 및 관성 텐서 분석을 통해
    주요 방향 축을 계산합니다.
    
    Attributes:
        mesh (pv.PolyData): PyVista 메쉬 객체
        vertices (np.ndarray): 정점 좌표 배열 (N x 3)
        center (np.ndarray): 메쉬의 중심 좌표
        diagonal_length (float): 바운딩 박스의 대각선 길이
    """
    
    # 상수 정의
    EPSILON = 1e-9  # 0으로 나누기 방지
    AXIS_LENGTH_SCALE = 0.35  # 시각화 시 축 길이 스케일
    
    def __init__(self, stl_path: str):
        """
        STL 파일로부터 메쉬를 로드하고 초기화
        
        Args:
            stl_path: STL 파일 경로
            
        Raises:
            FileNotFoundError: STL 파일이 존재하지 않을 경우
        """
        self.stl_path = Path(stl_path)
        if not self.stl_path.exists():
            raise FileNotFoundError(f"STL 파일을 찾을 수 없습니다: {stl_path}")
        
        # 메쉬 로드 및 초기화
        self.mesh = pv.read(str(self.stl_path))
        self.vertices = np.asarray(self.mesh.points)
        
        # 메쉬를 원점 중심으로 평행이동
        self._center_mesh()
        
        # 바운딩 박스 대각선 길이 계산
        self.diagonal_length = self._compute_diagonal_length()
        
        # 분석 결과 캐시
        self._pca_computed = False
        self._inertia_computed = False

    def _center_mesh(self) -> None:
        """메쉬를 중심점이 원점이 되도록 평행이동"""
        center = self.vertices.mean(axis=0)
        self.mesh.points = self.vertices - center
        self.vertices = np.asarray(self.mesh.points)
        self.center = self.vertices.mean(axis=0)  # 원점에 가까움
    
    def _compute_diagonal_length(self) -> float:
        """
        바운딩 박스의 대각선 길이 계산
        
        Returns:
            대각선 길이
        """
        bounds = self.mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        min_bound = np.array([bounds[0], bounds[2], bounds[4]])
        max_bound = np.array([bounds[1], bounds[3], bounds[5]])
        return np.linalg.norm(max_bound - min_bound)
    
    def compute_pca(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        주성분 분석(PCA) 수행 - 분산 기반 분석
        
        공분산 행렬의 고유값 분해를 통해 메쉬의 주요 분산 방향을 계산합니다.
        
        Returns:
            eigenvalues: 고윳값 배열 (오름차순, 3)
            eigenvectors: 고유벡터 행렬 (3 x 3, 각 열이 고유벡터)
            min_variance_axis: 최소 분산 축 (3,)
        """
        # 중심화된 좌표
        centered_vertices = self.vertices - self.center
        
        # 공분산 행렬 계산 및 고유값 분해
        cov_matrix = np.cov(centered_vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # np.linalg.eigh는 이미 정규화된 고유벡터를 반환함
        
        # 결과 저장
        self.pca_eigenvalues = eigenvalues
        self.pca_eigenvectors = eigenvectors
        self.min_variance_axis = eigenvectors[:, 0]  # 최소 고윳값에 대응
        
        self._pca_computed = True
        
        return self.pca_eigenvalues, self.pca_eigenvectors, self.min_variance_axis

    def compute_principal_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        관성 텐서 기반 주축 분석
        
        정점들을 균등 질량 점으로 간주하고 관성 텐서를 계산하여
        회전 관성의 주축을 추출합니다.
        
        Returns:
            axis_lengths: 각 주축의 상대적 길이 (시각화용, 3)
            principal_axes: 주축 벡터들 (3 x 3, 각 열이 주축)
        """
        # 중심화된 좌표
        centered_vertices = self.vertices - self.center
        x, y, z = centered_vertices[:, 0], centered_vertices[:, 1], centered_vertices[:, 2]
        
        # 관성 텐서 성분 계산
        Ixx = np.sum(y**2 + z**2)
        Iyy = np.sum(x**2 + z**2)
        Izz = np.sum(x**2 + y**2)
        Ixy = -np.sum(x * y)
        Ixz = -np.sum(x * z)
        Iyz = -np.sum(y * z)
        
        # 관성 텐서 행렬 구성 (대칭 행렬)
        self.inertia_tensor = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz]
        ])
        
        # 고유값 분해 (오름차순)
        self.inertia_eigenvalues, self.principal_axes = np.linalg.eigh(self.inertia_tensor)
        # np.linalg.eigh는 이미 정규화된 고유벡터를 반환함
        
        # 시각화를 위한 축 길이 계산
        # 관성이 작을수록 길이가 길어짐 (회전이 용이한 방향)
        axis_lengths = 1.0 / np.sqrt(self.inertia_eigenvalues + self.EPSILON)
        axis_lengths = (axis_lengths / axis_lengths.max()) * (self.AXIS_LENGTH_SCALE * self.diagonal_length)
        self.axis_lengths = axis_lengths
        
        self._inertia_computed = True
        
        return self.axis_lengths, self.principal_axes
    
    
    def get_analysis_summary(self) -> dict:
        """
        분석 결과 요약 정보 반환
        
        Returns:
            분석 결과를 담은 딕셔너리
        """
        summary = {
            'mesh_path': str(self.stl_path),
            'num_vertices': len(self.vertices),
            'center': self.center.tolist(),
            'diagonal_length': float(self.diagonal_length),
        }
        
        if self._pca_computed:
            summary['pca_eigenvalues'] = self.pca_eigenvalues.tolist()
            summary['min_variance_axis'] = self.min_variance_axis.tolist()
        
        if self._inertia_computed:
            summary['inertia_eigenvalues'] = self.inertia_eigenvalues.tolist()
            summary['axis_lengths'] = self.axis_lengths.tolist()
        
        return summary