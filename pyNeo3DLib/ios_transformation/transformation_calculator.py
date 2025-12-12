"""
변환 행렬 계산 모듈

회전 및 이동 변환 행렬을 계산하는 클래스입니다.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyNeo3DLib.fileLoader.mesh import Mesh


class TransformationCalculator:
    """
    변환 행렬 계산 클래스
    
    회전 행렬, 이동 행렬, 결합 변환 행렬을 계산합니다.
    """
    
    def compute_rotation_matrix_to_standard_jaw(
        self, 
        x_axis: np.ndarray, 
        y_axis: np.ndarray, 
        z_axis: np.ndarray,
        is_upper: bool
    ) -> np.ndarray:
        """
        상악/하악에 맞는 회전 행렬을 계산합니다.
        
        Args:
            x_axis: 정규화된 x축 벡터
            y_axis: 정규화된 y축 벡터
            z_axis: 정규화된 z축 벡터
            is_upper: True면 상악(upper), False면 하악(lower)
            
        Returns:
            4x4 동차변환 행렬
        """
        if is_upper:
            target_axes = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
        else:
            target_axes = np.eye(3)
        
        return self._compute_rotation_matrix_to_standard(x_axis, y_axis, z_axis, target_axes)
    
    def _compute_rotation_matrix_to_standard(
        self, 
        x_axis: np.ndarray, 
        y_axis: np.ndarray, 
        z_axis: np.ndarray,
        target_axes: np.ndarray
    ) -> np.ndarray:
        """
        현재 좌표계를 목표 좌표계로 변환하는 회전 행렬을 계산합니다.
        
        입력 축 벡터들은 이미 정규화되고 직교하는 것으로 가정합니다.
        
        Args:
            x_axis, y_axis, z_axis: 현재 좌표계의 정규화된 축 벡터들
            target_axes: 목표 좌표계 (3x3 행렬, 각 열이 목표 축)
            
        Returns:
            4x4 동차변환 행렬
        """
        # 현재 좌표계 행렬 (각 열이 축 벡터)
        current_coordinate_system = np.column_stack([x_axis, y_axis, z_axis])
        
        # 회전 행렬 계산: R = Target @ Current^T
        # 정규 직교 행렬이므로 역행렬 = 전치 행렬
        rotation_matrix_3x3 = target_axes @ current_coordinate_system.T
        
        # 회전 행렬 검증
        det = np.linalg.det(rotation_matrix_3x3)
        is_orthogonal = np.allclose(rotation_matrix_3x3 @ rotation_matrix_3x3.T, np.eye(3), atol=1e-6)
        
        print(f"[INFO] Rotation matrix validation:")
        print(f"   Determinant: {det:.10f} (should be 1)")
        print(f"   Is orthogonal: {is_orthogonal}")
        
        if not is_orthogonal or abs(det - 1.0) > 1e-6:
            print(f"[WARNING] Rotation matrix is invalid!")
            print(f"   R @ R.T:")
            print(f"{rotation_matrix_3x3 @ rotation_matrix_3x3.T}")
        
        # 4x4 동차변환 행렬로 확장
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = rotation_matrix_3x3
        
        print(f"[INFO] Rotation matrix (4x4):")
        print(f"{rotation_matrix}")
        
        return rotation_matrix
    
    def compute_combined_transformation(
        self,
        rotation_matrix: np.ndarray,
        source_centroid: np.ndarray,
        target_centroid: np.ndarray
    ) -> np.ndarray:
        """
        회전 + 이동을 결합한 단일 4x4 동차 변환 행렬을 계산합니다.
        
        변환 순서:
        1. 소스 도심점을 원점으로 이동 (T1)
        2. 회전 변환 적용 (R)
        3. target_centroid로 이동 (T2)
        
        최종 변환: T2 @ R @ T1
        
        Args:
            rotation_matrix: 4x4 동차 변환 행렬
            source_centroid: 소스 메시의 무게중심 (회전 중심점)
            target_centroid: 목표 위치의 무게중심
            
        Returns:
            4x4 동차 변환 행렬
        """
        
        # 1단계: 소스 도심점을 원점으로 이동하는 변환 행렬 (T1)
        T1 = np.eye(4)
        T1[:3, 3] = -source_centroid
        
        # 2단계: 회전 변환 행렬 (R)
        R = rotation_matrix.copy()
        
        # 3단계: target_centroid로 이동하는 변환 행렬 (T2)
        T2 = np.eye(4)
        T2[:3, 3] = target_centroid
        
        # 4단계: 최종 변환 행렬 결합 (T2 @ R @ T1)
        combined_matrix = T2 @ R @ T1
        
        return combined_matrix
    
    def apply_transformation_to_mesh(
        self,
        mesh: "Mesh",
        transformation_matrix: np.ndarray
    ) -> "Mesh":
        """
        메시에 4x4 동차 변환 행렬을 적용합니다 (in-place).
        
        Args:
            mesh: 변환할 Mesh 객체
            transformation_matrix: 4x4 동차 변환 행렬
            
        Returns:
            변환된 Mesh 객체
        """
        # 정점을 동차 좌표로 변환 (N x 3 -> N x 4)
        vertices_homogeneous = np.hstack([
            mesh.vertices,
            np.ones((len(mesh.vertices), 1))
        ])
        
        # 변환 적용
        transformed_vertices_homogeneous = vertices_homogeneous @ transformation_matrix.T
        
        # 3D 좌표로 변환 (N x 4 -> N x 3)
        mesh.vertices = transformed_vertices_homogeneous[:, :3]
        
        return mesh
