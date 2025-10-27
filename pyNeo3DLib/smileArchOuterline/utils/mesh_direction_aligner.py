"""
메시 방향 정렬을 담당하는 클래스
단일책임원칙에 따라 메시의 방향 정렬만을 담당합니다.
"""

import numpy as np
from typing import Tuple, Optional


class MeshDirectionAligner:
    """메시의 방향을 정렬하는 클래스"""
    
    def __init__(self, alignment_threshold: float = 0.8):
        """
        Args:
            alignment_threshold: 정렬 적용을 위한 내적값 임계값 (기본값: 0.8)
        """
        self.alignment_threshold = alignment_threshold
        self.z_axis = np.array([0, 0, 1])
    
    def align_points_to_z_axis(
        self, 
        points: np.ndarray, 
        direction_vector: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        주어진 점들을 direction_vector가 z축과 정렬되도록 회전시킵니다.
        
        Args:
            points: 회전시킬 점들 (N, 3)
            direction_vector: 정렬할 방향 벡터 (3,)
            
        Returns:
            Tuple[rotated_points, rotation_matrix]:
                - rotated_points: 회전된 점들
                - rotation_matrix: 적용된 회전 행렬 (회전이 적용되지 않은 경우 None)
        """
        # direction_vector 정규화
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        # z축과의 내적값 계산
        cos_angle = self._calculate_cosine_angle(direction_vector)
        
        print(f"direction_vector와 z축의 내적값: {cos_angle:.4f}")
        
        # 임계값 이상인 경우에만 회전 적용
        if cos_angle > self.alignment_threshold:
            angle = np.arccos(cos_angle)
            rotation_matrix = self._calculate_rotation_matrix(direction_vector, angle)
            
            if rotation_matrix is not None:
                rotated_points = np.dot(points, rotation_matrix.T)
                print(f"direction_vector를 z축과 정렬하기 위해 {np.degrees(angle):.2f}도 회전을 적용했습니다.")
                return rotated_points, rotation_matrix
            else:
                print("direction_vector가 이미 z축과 정렬되어 있습니다.")
                return points, None
        else:
            print(f"내적값이 {cos_angle:.4f}로 임계값({self.alignment_threshold}) 미만이므로 회전을 적용하지 않습니다.")
            return points, None
    
    def _calculate_cosine_angle(self, direction_vector: np.ndarray) -> float:
        """
        direction_vector와 z축 사이의 코사인 각도를 계산합니다.
        
        Args:
            direction_vector: 방향 벡터 (3,)
            
        Returns:
            float: 코사인 각도 (-1.0 ~ 1.0)
        """
        cos_angle = np.dot(direction_vector.flatten(), self.z_axis) / (
            np.linalg.norm(direction_vector) * np.linalg.norm(self.z_axis)
        )
        return np.clip(cos_angle, -1.0, 1.0)  # 수치적 안정성을 위해 클리핑
    
    def _calculate_rotation_matrix(
        self, 
        direction_vector: np.ndarray, 
        angle: float
    ) -> Optional[np.ndarray]:
        """
        로드리게스 회전 공식을 사용하여 회전 행렬을 계산합니다.
        
        Args:
            direction_vector: 회전축을 계산할 방향 벡터 (3,)
            angle: 회전 각도 (라디안)
            
        Returns:
            Optional[np.ndarray]: 회전 행렬 (3, 3) 또는 None (이미 정렬된 경우)
        """
        # 회전축 계산 (direction_vector와 z축의 외적)
        rotation_axis = np.cross(direction_vector.flatten(), self.z_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        # 회전축이 0에 가까우면 이미 정렬된 상태
        if rotation_axis_norm < 1e-6:
            return None
        
        rotation_axis = rotation_axis / rotation_axis_norm
        
        # 로드리게스 회전 공식을 사용한 회전 행렬 계산
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        
        rotation_matrix = (
            np.eye(3) + 
            np.sin(angle) * K + 
            (1 - np.cos(angle)) * np.dot(K, K)
        )
        
        return rotation_matrix
    
    def calculate_direction_vector(self, points: np.ndarray) -> np.ndarray:
        """
        점들의 중심에서 최대 z값을 가지는 점으로의 방향 벡터를 계산합니다.
        
        Args:
            points: 점들 (N, 3)
            
        Returns:
            np.ndarray: 정규화된 방향 벡터 (3,)
        """
        centroid = np.mean(points, axis=0)
        max_z_idx = np.argmax(points[:, 2])
        max_z_point = points[max_z_idx]
        
        direction_vector = max_z_point - centroid
        return direction_vector / np.linalg.norm(direction_vector)
