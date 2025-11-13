"""
곡선 샘플링 관련 기능을 담당하는 클래스
"""

import numpy as np
from typing import List
from .constants import AnalysisConstants
from .polar_sampler import PolarSampling
from .visualizer import VisualizeForTest


class CurveSampler:
    """곡선 샘플링을 담당하는 클래스"""
    
    def __init__(self):
        self.polar_start_angle = AnalysisConstants.POLAR_START_ANGLE
        self.polar_end_angle = AnalysisConstants.POLAR_END_ANGLE
        self.default_num_samples = AnalysisConstants.DEFAULT_NUM_SAMPLES
        self.landmark_decimal_places = AnalysisConstants.LANDMARK_DECIMAL_PLACES
    
    def subsample_points_uniformly(self, points: np.ndarray, target_count: int) -> np.ndarray:
        """
        주어진 점들을 균등하게 샘플링하여 목표 개수의 점을 반환합니다.
        첫 번째와 마지막 점을 포함하여 균등하게 분배합니다.
        
        Args:
            points: 샘플링할 포인트 클라우드 (numpy 배열, 형태: (n, 3))
            target_count: 목표 포인트 개수
            
        Returns:
            샘플링된 포인트들 (numpy 배열, 형태: (target_count, 3) 또는 (len(points), 3))
        """
        if len(points) < target_count:
            # 점의 개수가 target_count보다 적으면 모든 점 사용
            return np.array(points)
        else:
            # 첫 번째(0)와 마지막(-1) 인덱스를 포함하여 균등하게 분배
            indices = np.linspace(0, len(points) - 1, target_count, dtype=int)
            return np.array([points[i] for i in indices])
    
    def perform_polar_sampling(
        self,
        vertices: np.ndarray, 
        polar_center: np.ndarray, 
        angle_step: float = 1, 
        y_slice_mid: float = -1,
        y_offset: float = 0.5
    ) -> np.ndarray:
        """극좌표 샘플링 수행
        
        입력된 3D 메쉬 버텍스에서 극좌표계 기반으로 외곽점과 내곽점을 샘플링하고, 
        두 결과의 평균을 계산하여 치아의 중간 라인을 추출합니다.

        Args:
            vertices: 3D 메쉬의 버텍스 좌표 배열, 형태: (N, 3) [x, y, z]
            polar_center: 극좌표계의 중심점 좌표, 형태: (3,) [x, y, z]
            angle_step: 극좌표 샘플링 시 각도 간격(도). 기본값 1
            y_slice_mid: Y축 슬라이스 중심 위치. 기본값 -1
            y_offset: Y축 슬라이스 범위의 오프셋. 기본값 0.5

        Returns:
            샘플링된 중간 라인 포인트들, 형태: (M, 3) [x, y, z]
            외곽점과 내곽점의 평균으로 계산된 치아 아치의 중간 곡선을 나타냄
        """
        y_range = (y_slice_mid - y_offset, y_slice_mid + y_offset)
        polar_sampler = PolarSampling(polar_center)
        
        # 외곽점 샘플링
        outer_points = polar_sampler.polar_sampling(
            vertices, 
            angle_step=angle_step, 
            mode="farthest", 
            y_range=y_range,
            start_angle=self.polar_start_angle, 
            end_angle=self.polar_end_angle
        )
        
        # 내곽점 샘플링
        inner_points = polar_sampler.polar_sampling(
            vertices, 
            angle_step=angle_step, 
            mode="nearest", 
            y_range=y_range,
            start_angle=self.polar_start_angle, 
            end_angle=self.polar_end_angle
        )
        
        # 중간점 계산 (외곽과 내곽의 평균)
        return np.mean([outer_points, inner_points], axis=0)
    
    def calculate_curve_total_length(self, points: np.ndarray) -> float:
        """
        연속된 포인트들 사이의 유클리드 거리를 합산하여 곡선의 전체 길이를 계산합니다.
        
        Args:
            points: 곡선을 이루는 포인트 배열 (N, 3)
            
        Returns:
            total_length: 곡선의 전체 길이
        """
        if len(points) < 2:
            return 0.0
        
        # 각 연속된 포인트 쌍 간의 거리 계산 및 합산
        distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
        return np.sum(distances)
    
    def sample_points_by_arc_length(self, points: np.ndarray, num_samples: int) -> List[np.ndarray]:
        """
        곡선을 등호장(arc length) 기준으로 균등하게 샘플링합니다.
        
        곡선의 전체 길이를 측정하고, 등간격으로 나누어 각 구간에서 포인트를 추출합니다.
        
        Args:
            points: 샘플링할 포인트 배열 (N, 3)
            num_samples: 샘플링할 포인트 개수
            
        Returns:
            sampled_points: 샘플링된 포인트 리스트
        """
        if len(points) == 0:
            return []
        
        if len(points) <= num_samples:
            return [point for point in points]
        
        total_length = self.calculate_curve_total_length(points)
        segment_length = total_length / (num_samples - 1)
        
        sampled_points = [points[0]]  # 첫 번째 포인트
        accumulated_distance = 0.0
        
        # 등호장 기준으로 샘플링
        for i in range(len(points) - 1):
            distance = np.linalg.norm(points[i + 1] - points[i])
            accumulated_distance += distance
            
            # 누적 거리가 구간 길이를 초과하면 포인트 추출
            if accumulated_distance >= segment_length:
                sampled_points.append(points[i])
                accumulated_distance = 0.0
        
        # 마지막 포인트 추가
        sampled_points.append(points[-1])
        
        return sampled_points
    
    def compute_normalized_landmarks_and_arch_depth_molar_width(
        self,
        smoothed_points: np.ndarray, 
        num_samples: int = None
    ) -> tuple:
        """
        부드럽게 처리된 곡선 포인트에서 랜드마크와 arch_depth, molar_width를 계산합니다.
        
        처리 단계:
        1. 호장 길이 기준으로 균등 샘플링
        2. X값 기준 정렬
        3. Y값 제거 (X, Z만 사용)
        4. Z값 양수화
        5. 평균값을 빼서 중심을 원점으로 이동
        6. 소수점 반올림
        7. Python 네이티브 리스트로 변환
        
        Args:
            smoothed_points: 부드럽게 처리된 곡선 포인트 배열 (N, 3)
            num_samples: 샘플링할 포인트 개수 (기본값: DEFAULT_NUM_SAMPLES)
            
        Returns:
            landmark_points: 정규화된 랜드마크 포인트 리스트 [[x, z], ...] (num_samples개의 [x, z] 쌍)
            arch_depth: 치아 배열 곡선의 깊이
            molar_width: 치아 배열 곡선의 폭
        """
        if num_samples is None:
            num_samples = self.default_num_samples
        
        # 1. 호장 길이 기준으로 샘플링
        sampled_points = self.sample_points_by_arc_length(smoothed_points, num_samples)
        sampled_array = np.array(sampled_points)

        # 2. X값 기준 정렬
        sorted_indices = np.argsort(sampled_array[:, 0])
        sorted_points = sampled_array[sorted_indices]
        
        # 3. Y값 제거 (X, Z만 사용)
        points_xz = sorted_points[:, [0, 2]]
            
        # 4. 중심을 원점으로 이동 (평균값 빼기)
        mean_values = np.mean(points_xz, axis=0)
        landmark_points = points_xz - mean_values
        
        # 4. 반올림
        landmark_points = np.round(landmark_points, self.landmark_decimal_places)

        # 6. NumPy 배열을 리스트로 변환하고, 내부의 모든 요소를 Python 네이티브 타입으로 변환
        landmark_points_list = []
        for point in landmark_points:
            # NumPy float32/float64를 Python float으로 변환
            point_list = [float(coord) for coord in point]
            landmark_points_list.append(point_list)

        # arch_depth는 maxilla_spline_curve_points의 z값 중 최소값과 최대값의 차이
        arch_depth = abs(min(smoothed_points[:, 2]) - max(smoothed_points[:, 2]))
        arch_depth = np.round(arch_depth, 2)
        molar_width = abs(min(smoothed_points[:, 0]) - max(smoothed_points[:, 0]))
        molar_width = np.round(molar_width, 2)
        
        return landmark_points_list, arch_depth, molar_width
