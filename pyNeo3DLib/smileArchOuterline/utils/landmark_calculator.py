"""
랜드마크 계산 관련 기능을 담당하는 클래스
"""

import numpy as np
from typing import List
from .constants import AnalysisConstants


class LandmarkCalculator:
    """랜드마크 계산을 담당하는 클래스"""
    
    def __init__(self):
        self.landmark_decimal_places = AnalysisConstants.LANDMARK_DECIMAL_PLACES
    
    def calculate_landmarks(self, half_curve: np.ndarray) -> List[List[float]]:
        """
        악궁 곡선에 기반하여 랜드마크 포인트를 계산합니다.
        
        Args:
            half_curve: 반쪽 악궁 곡선 배열 (N, 3)
            
        Returns:
            landmark_points_list: 계산된 랜드마크 포인트 리스트
        """
        # Z축 범위 계산
        z_min = np.min(half_curve[:, 2])
        z_max = np.max(half_curve[:, 2])
        
        # 5개의 등간격 z 좌표 생성
        target_z_coords = np.linspace(z_max, z_min, 5)
        
        # 각 타겟 Z에 가장 가까운 포인트 찾기
        landmark_points = []
        for target_z in target_z_coords:
            diff = np.abs(half_curve[:, 2] - target_z)
            closest_index = np.argmin(diff)
            landmark_point = np.round(half_curve[closest_index, :], 2).reshape(1, 3)
            landmark_points.append(landmark_point)
        
        landmark_points = np.concatenate(landmark_points, axis=0)
        
        # Z축 정규화 (첫 번째 포인트 기준)
        landmark_points[:, 2] = landmark_points[:, 2] - landmark_points[0, 2]
        
        # Y축 제거 (X, Z만 사용)
        landmark_points = landmark_points[:, [0, 2]]
        
        # 양수화 및 첫 번째 포인트 제거
        landmark_points = np.abs(landmark_points)[1:]
        
        # 대칭 포인트 생성
        symmetric_points = landmark_points.copy()
        symmetric_points[:, 0] = -symmetric_points[:, 0]
        symmetric_points = symmetric_points[::-1]
        
        # 전체 랜드마크 생성 (좌측 + 중심 + 우측)
        total_landmark_points = np.concatenate([
            symmetric_points,
            np.zeros((1, 2)),
            landmark_points
        ], axis=0)
        
        # NumPy 배열을 Python 리스트로 변환
        landmark_points_list = [
            np.round([float(coord) for coord in point], 2).tolist()
            for point in total_landmark_points
        ]
        
        return landmark_points_list
