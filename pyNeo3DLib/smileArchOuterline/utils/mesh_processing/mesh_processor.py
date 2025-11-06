"""
메시 처리 및 정렬을 담당하는 클래스
단일책임: 메시의 정렬, 회전, 중심점 계산
"""

import numpy as np
from typing import Tuple
# from .mesh_aligner import MeshAligner
from .mesh_alignment_manager import MeshAlignmentManager
from .mesh_filter import MeshFilter
from .mesh_transformer import MeshTransformer


class MeshProcessor:
    """메시 처리 및 정렬을 담당하는 클래스"""
    
    def __init__(self):
        self.mesh_aligner = MeshAlignmentManager()
        self.mesh_transformer = MeshTransformer()
        self.mesh_filter = MeshFilter()
    
    def perform_initial_alignment(self, mesh_path: str) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        1차 정렬을 수행합니다.
        
        Args:
            mesh_path: STL 메쉬 파일 경로
            
        Returns:
            Tuple[aligned_mesh, center, evec_x, evec_y, evec_z]:
                - aligned_mesh: 정렬된 메쉬
                - center: 메쉬 중심점
                - evec_x: X축 정렬 벡터
                - evec_y: Y축 정렬 벡터
                - evec_z: Z축 정렬 벡터
        """
        # 메쉬 로드 및 주축 계산
        # mesh_aligner = MeshAligner(mesh_path)
        # input_mesh = mesh_aligner.mesh
        # center = mesh_aligner.center.reshape(1, 3)
        # _, principal_evecs = mesh_aligner.compute_principal_axes()

        # MeshAlignmentManager를 사용하여 메쉬 로드 및 주축 계산
        self.mesh_aligner.load_mesh(mesh_path)
        input_mesh = self.mesh_aligner.mesh
        center = self.mesh_aligner.center.reshape(1, 3)
        _, principal_evecs, _ = self.mesh_aligner.compute_principal_axes()

        # 정렬 축 결정
        evec_x, evec_y, evec_z = self.mesh_aligner.determine_alignment_axes(
            input_mesh, center, principal_evecs
        )

        # 글로벌 좌표계로 정렬
        aligned_mesh = self.mesh_aligner.align_mesh_to_global_coordinates(
            input_mesh, evec_x, evec_y, evec_z
        )
        
        return aligned_mesh, center, evec_x, evec_y, evec_z
    
    def align_mesh_direction(self, mesh: object, points: np.ndarray) -> Tuple[object, np.ndarray, np.ndarray]:
        """
        메시와 포인트를 방향에 맞게 정렬합니다.
        
        Args:
            mesh: 정렬할 메시
            points: 정렬할 포인트들
            
        Returns:
            Tuple[rotated_mesh, aligned_points, rotation_matrix]:
                - rotated_mesh: 회전된 메시
                - aligned_points: 정렬된 포인트들
                - rotation_matrix: 회전 행렬
        """
        # 방향 벡터 계산
        direction_vector = self.mesh_transformer.calculate_direction_vector(points)
        
        # 포인트들을 Z축에 정렬
        aligned_points, rotation_matrix = self.mesh_transformer.align_points_to_z_axis(
            points, direction_vector
        )

        # 메시도 동일한 회전 적용
        rotated_mesh = mesh.copy()
        if rotation_matrix is not None:
            rotated_mesh.points = np.dot(mesh.points, rotation_matrix.T)

        return rotated_mesh, aligned_points, rotation_matrix
    
    def filter_and_center_mesh(self, mesh: object, points: np.ndarray) -> Tuple[object, np.ndarray, np.ndarray]:
        """
        메시와 포인트를 필터링하고 중심을 맞춥니다.
        
        Args:
            mesh: 필터링할 메시
            points: 필터링할 포인트들

        Returns:
            Tuple[filtered_mesh, centered_filtered_points, new_center]:
                - filtered_mesh: 필터링된 메시
                - centered_filtered_points: 중심이 맞춰진 필터링된 악궁 포인트들
                - new_center: 새로운 중심점
        """
        # Z값 기준 필터링
        filtered_points, z_threshold = self.mesh_filter.filter_points_by_z_threshold(points)
        
        # 중심점 재계산 및 정렬
        new_center = np.mean(filtered_points[[0, -1]], axis=0)
        
        # 메시와 포인트를 새로운 중심으로 이동
        centered_mesh = mesh.copy()
        centered_mesh.points -= new_center
        centered_filtered_points = filtered_points - new_center
        
        return centered_mesh, centered_filtered_points, new_center
