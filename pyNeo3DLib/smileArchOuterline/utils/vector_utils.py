import numpy as np
from typing import Optional

class VectorUtils:
    """벡터 관련 유틸리티 함수들을 제공하는 클래스"""

    @staticmethod
    def rotate_vector_around_axis(vector: np.ndarray, axis: np.ndarray, angle_degrees: float) -> np.ndarray:
        """
        주어진 축을 중심으로 벡터를 회전시킵니다.

        Args:
            vector: 회전시킬 벡터 (numpy array, shape: (3,))
            axis: 회전축 벡터 (numpy array, shape: (3,))
            angle_degrees: 회전 각도 (도 단위)

        Returns:
            rotated_vector: 회전된 벡터 (numpy array, shape: (3,))
        """
        # 입력 검증
        if axis is None:
            raise ValueError("회전축이 None입니다.")

        # axis를 1차원 배열로 변환
        axis = np.asarray(axis).flatten()

        if len(axis) != 3:
            raise ValueError(f"회전축은 3차원 벡터여야 합니다. 현재 형태: {axis.shape}")

        # vector를 1차원 배열로 변환
        vector = np.asarray(vector).flatten()

        if len(vector) != 3:
            raise ValueError(f"벡터는 3차원 벡터여야 합니다. 현재 형태: {vector.shape}")

        # 각도를 라디안으로 변환
        angle_rad = np.radians(angle_degrees)

        # 회전축 정규화
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            raise ValueError("회전축의 크기가 0입니다.")

        axis = axis / axis_norm

        # 로드리게스 회전 공식 사용
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # 외적 행렬 (skew-symmetric matrix)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        # 회전 행렬 계산: R = I + sin(θ)K + (1-cos(θ))K²
        I = np.eye(3)
        R = I + sin_angle * K + (1 - cos_angle) * np.dot(K, K)

        # 벡터 회전
        return np.dot(R, vector)
