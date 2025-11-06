import numpy as np
import pyvista as pv
from typing import Tuple
from pyNeo3DLib.smileArchOuterline.utils.common.vector_utils import VectorUtils


class MeshTransformer:
    """메쉬 변환 (전역 좌표계 정렬, 방향 정렬)을 담당하는 클래스"""

    def align_mesh_to_global_coordinates(
        self,
        input_mesh: pv.PolyData, 
        evec_x: np.ndarray, 
        evec_y: np.ndarray, 
        evec_z: np.ndarray
    ) -> pv.PolyData:
        """
        주축 벡터를 사용하여 메쉬를 글로벌 좌표계로 정렬합니다.
        
        목표 정렬:
        - evec_x → 글로벌 X축 (1, 0, 0)
        - evec_y → 글로벌 Y축 (0, 1, 0)
        - evec_z → 글로벌 Z축 (0, 0, 1)
        
        Args:
            input_mesh: 정렬할 메쉬
            evec_x: X축 정렬용 주축 벡터 (3,)
            evec_y: Y축 정렬용 주축 벡터 (3,)
            evec_z: Z축 정렬용 주축 벡터 (3,)
            
        Returns:
            aligned_mesh: 정렬된 메쉬
        """
        # 벡터 정규화 및 검증
        evec_x = VectorUtils.normalize_vector(evec_x)
        evec_y = VectorUtils.normalize_vector(evec_y)
        evec_z = VectorUtils.normalize_vector(evec_z)
        
        # 기저 행렬 구성 (3, 3)
        basis_matrix = np.array([evec_x, evec_y, evec_z], dtype=np.float64)
        
        # 역행렬 계산
        try:
            inverse_basis = np.linalg.inv(basis_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("기저 행렬이 특이행렬입니다. 벡터들이 선형 독립인지 확인하세요.")
        
        # 목표 좌표계 정의 (3, 3)
        target_matrix = np.array([
            [1.0, 0.0, 0.0],  # X축
            [0.0, 1.0, 0.0],  # Y축
            [0.0, 0.0, 1.0]   # Z축
        ], dtype=np.float64)
        
        # 회전 행렬 계산 및 적용
        rotation_matrix = np.matmul(inverse_basis, target_matrix)
        aligned_vertices = np.matmul(input_mesh.points, rotation_matrix)

        aligned_mesh = input_mesh.copy()
        aligned_mesh.points = aligned_vertices.astype(np.float64)
        
        return aligned_mesh

    def calculate_direction_vector(self, points: np.ndarray) -> np.ndarray:
        """
        주어진 포인트 배열의 방향 벡터를 계산합니다.
        주로 곡선의 시작점과 끝점을 잇는 벡터를 사용합니다.
        
        Args:
            points: 포인트 배열 (N, 3)
            
        Returns:
            np.ndarray: 정규화된 방향 벡터 (3,)
        """
        if len(points) < 2:
            raise ValueError("방향 벡터를 계산하려면 최소 두 개의 포인트가 필요합니다.")

        # 첫 번째와 마지막 포인트의 차이로 방향 벡터 계산
        direction_vector = points[-1] - points[0]
        return VectorUtils.normalize_vector(direction_vector)

    def align_points_to_z_axis(self, points: np.ndarray, direction_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        주어진 포인트들을 Z축 방향으로 정렬합니다.
        
        Args:
            points: 정렬할 포인트 배열 (N, 3)
            direction_vector: 포인트들의 현재 방향 벡터 (3,)
            
        Returns:
            Tuple[aligned_points, rotation_matrix]:
                - aligned_points: Z축에 정렬된 포인트 배열
                - rotation_matrix: 적용된 회전 행렬
        """
        target_z_axis = np.array([0, 0, 1])
        direction_vector = VectorUtils.normalize_vector(direction_vector)

        # 현재 방향 벡터와 목표 Z축 간의 회전축 및 각도 계산
        cross_product = np.cross(direction_vector, target_z_axis)
        rotation_axis = VectorUtils.normalize_vector(cross_product) if np.linalg.norm(cross_product) > VectorUtils.EPSILON else None
        
        if rotation_axis is None:
            # 이미 정렬되었거나 180도 반대 방향인 경우
            dot_product = np.dot(direction_vector, target_z_axis)
            if np.isclose(dot_product, -1.0):
                # 180도 회전이 필요한 경우 (예: [0,0,-1]을 [0,0,1]로)
                rotation_matrix = np.array([
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]  # Z축은 그대로 유지
                ], dtype=np.float64)
                aligned_points = np.dot(points, rotation_matrix.T)
                return aligned_points, rotation_matrix
            else:
                # 이미 정렬되어 있는 경우
                return points, np.eye(3)
        
        angle = np.arccos(np.dot(direction_vector, target_z_axis))
        angle_degrees = np.degrees(angle)
        
        # 회전 행렬 생성 (로드리게스 회전 공식)
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        I = np.eye(3)
        rotation_matrix = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        aligned_points = np.dot(points, rotation_matrix.T)
        
        return aligned_points, rotation_matrix
