import numpy as np
import pyvista as pv
from typing import Tuple

class MeshProcessor:
    """치아 메쉬 처리를 위한 클래스"""
    
    def __init__(self, mesh_path: str):
        """
        메쉬 프로세서 초기화
        
        Args:
            mesh_path: STL 파일 경로
        """
        self.mesh_path = mesh_path
        self.mesh = pv.read(mesh_path)
        self.vertices = np.asarray(self.mesh.points, dtype=np.float32).reshape(-1, 3)
        self.mesh_center = np.array(self.mesh.center, dtype=np.float32).reshape(1, 3)
        
    def center_align_mesh(self) -> np.ndarray:
        """메쉬를 중심점으로 정렬"""
        return self.vertices - self.mesh_center
    
    def split_vertices_by_x_axis(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """x축 기준으로 버텍스를 좌우로 분리"""
        left_indices = np.where(vertices[:, 0] < 0)[0]
        right_indices = np.where(vertices[:, 0] > 0)[0]
        return left_indices, right_indices
    
    def find_lowest_z_points(self, vertices: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """좌우 영역에서 z축이 가장 낮은 점들을 찾기"""
        left_z_min_idx = np.argmin(vertices[left_indices, 2])
        right_z_min_idx = np.argmin(vertices[right_indices, 2])
        
        left_min_point = vertices[left_indices][left_z_min_idx]
        right_min_point = vertices[right_indices][right_z_min_idx]
        
        return left_min_point, right_min_point
    
    def calculate_direction_vector(self, left_point: np.ndarray, right_point: np.ndarray) -> np.ndarray:
        """두 점을 기준으로 방향벡터 계산"""
        direction = right_point - left_point
        return direction / np.linalg.norm(direction)
    
    def rotate_mesh_to_x_axis(self, direction_vector: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """방향벡터를 x축으로 정렬하도록 메쉬 회전"""
        reference_vector = np.array([1, 0, 0])
        
        # 사잇각 계산
        dot_product = np.dot(direction_vector, reference_vector)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # 회전 방향 결정
        cross_product = np.cross(direction_vector, reference_vector)
        if cross_product[1] < 0:  # y축 성분이 음수면 반대 방향으로 회전
            angle = -angle
        
        # Y축 기준 회전 행렬 생성
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ])
        
        return vertices @ rotation_matrix.T
    
    def calculate_polar_center(self, left_point: np.ndarray, right_point: np.ndarray) -> np.ndarray:
        """극좌표 중심점 계산"""
        origin = np.array([0, 0, 0], dtype=np.float32)
        return np.mean([left_point, right_point, origin], axis=0)
    
    def process_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        메쉬 전체 처리 파이프라인
        
        Returns:
            회전된 버텍스와 극좌표 중심점
        """
        # 1. 메쉬 중심 정렬
        centered_vertices = self.center_align_mesh()
        
        # 2. 첫 번째 방향벡터 계산을 위한 좌우 분리 및 최저점 찾기
        left_indices, right_indices = self.split_vertices_by_x_axis(centered_vertices)
        left_min_point, right_min_point = self.find_lowest_z_points(centered_vertices, left_indices, right_indices)
        
        # 3. 방향벡터 계산 및 메쉬 회전
        direction_vector = self.calculate_direction_vector(left_min_point, right_min_point)
        rotated_vertices = self.rotate_mesh_to_x_axis(direction_vector, centered_vertices)
        
        # 4. 회전된 메쉬에서 새로운 극좌표 중심점 계산
        rotated_left_indices, rotated_right_indices = self.split_vertices_by_x_axis(rotated_vertices)
        rotated_left_min, rotated_right_min = self.find_lowest_z_points(rotated_vertices, rotated_left_indices, rotated_right_indices)
        polar_center = self.calculate_polar_center(rotated_left_min, rotated_right_min)
        
        return rotated_vertices, polar_center