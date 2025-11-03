"""
신호 처리 및 필터링 관련 기능을 담당하는 클래스
"""

import numpy as np
from .constants import AnalysisConstants


class SignalProcessor:
    """신호 처리 및 필터링을 담당하는 클래스"""
    
    def __init__(self):
        self.default_window_size = AnalysisConstants.DEFAULT_WINDOW_SIZE
    
    def remove_molar_outliers(self, result_points_array: np.ndarray, percentile_threshold: float = 2.5) -> np.ndarray:
        """
        대구치쪽 아웃라이어를 제거하는 함수
        
        Z값을 기준으로 하위 percentile_threshold%에 해당하는 점들을 제거하여
        대구치 부분의 노이즈를 제거합니다.
        
        Args:
            result_points_array: 입력 포인트 배열 (N, 3) [x, y, z]
            percentile_threshold: 제거할 하위 퍼센트 임계값 (기본값: 2.5)
            
        Returns:
            filtered_result_points_array: 아웃라이어가 제거된 포인트 배열 (M, 3)
        """
        # result_points_array를 z값만 남기고 나머지는 없애서 차원축소
        squized_into_zline_result_points_array = result_points_array[:, 2]
        print(f"Z값 배열 형태: {squized_into_zline_result_points_array.shape}")

        # z_result_points_array를 중복값 제거후 데이터 분포를 이용해서 -z방향으로 percentile_threshold% 떨어진값을 제거
        unique_z_result_points_array = np.unique(squized_into_zline_result_points_array)
        filtered_z_result_points_array = unique_z_result_points_array[
            unique_z_result_points_array > np.percentile(unique_z_result_points_array, percentile_threshold)
        ]

        # result_points_array에서 filtered_z_result_points_array에 해당하는 값을 찾아서하여 필터링
        # z값이 filtered_z_result_points_array에 포함되는 점들만 선택
        mask = np.isin(result_points_array[:, 2], filtered_z_result_points_array)
        filtered_result_points_array = result_points_array[mask]
        
        return filtered_result_points_array
    
    def moving_average_filter(self, points: np.ndarray, window_size: int = None) -> np.ndarray:
        """
        이동 평균 필터를 사용하여 포인트 클라우드를 부드럽게 만듭니다.
        
        Args:
            points: 입력 포인트 배열 (N, 3) [x, y, z]
            window_size: 이동 평균에 사용할 윈도우 크기 (기본값: DEFAULT_WINDOW_SIZE)
                        홀수를 권장 (중심점을 기준으로 양쪽 대칭)
            
        Returns:
            필터링된 포인트들 (N, 3)
        """
        if window_size is None:
            window_size = self.default_window_size
            
        if len(points) < window_size:
            return points
        
        filtered_points = np.zeros_like(points)
        half_window = window_size // 2
        
        for i in range(len(points)):
            # 윈도우 범위 계산 (경계 처리)
            start_idx = max(0, i - half_window)
            end_idx = min(len(points), i + half_window + 1)
            
            # 윈도우 내 포인트들의 평균 계산
            filtered_points[i] = np.mean(points[start_idx:end_idx], axis=0)
        
        return filtered_points
    
    def calculate_average_y_value(self, filtered_points):
        """
        필터링된 포인트들의 Y값 평균 계산
        
        Args:
            filtered_points: 필터링된 포인트들
            
        Returns:
            average_y_value: Y값 평균
        """
        y_values = filtered_points[:, 1]
        return np.mean(y_values)
