"""
치아 악궁 타입 분류기
단일책임: 유치악/무치악 판별 및 필터 파라미터 결정
"""

import numpy as np
from typing import Tuple
from ..curve_utils.polar_sampler import PolarSampling


class ArchTypeClassifier:
    """치아 악궁의 타입을 분류하는 클래스 (유치악/무치악)"""
    
    # 분류 임계값
    RMSE_THRESHOLD = 2.0
    
    # 필터 윈도우 크기
    DENTULOUS_WINDOW_SIZE = 50  # 유치악 (치아가 있는 경우)
    EDENTULOUS_WINDOW_SIZE = 20  # 무치악 (치아가 없는 경우)
    
    def __init__(self):
        """초기화"""
        pass
    
    def classify_arch_type(
        self, 
        points: np.ndarray, 
        z_min: float,
        start_angle: float = 80,
        end_angle: float = 100
    ) -> Tuple[str, int, float]:
        """
        악궁 타입을 분류합니다 (유치악/무치악).
        
        80~100도 범위에서 ymin과 farthest 모드로 샘플링한 결과의 RMSE를 계산하여
        치아의 유무를 판단합니다.
        
        Args:
            points: 포인트 클라우드 배열
            z_min: Z축 최소값
            start_angle: 시작 각도 (기본값: 80도)
            end_angle: 끝 각도 (기본값: 100도)
            
        Returns:
            Tuple[arch_type, filter_window_size, rmse]:
                - arch_type: "dentulous" (유치악) 또는 "edentulous" (무치악)
                - filter_window_size: 권장 필터 윈도우 크기
                - rmse: 계산된 RMSE 값
        """
        # 극좌표 샘플러 생성
        polar_sampler = PolarSampling(np.array([0, 0, z_min]))
        
        # ymin 모드로 샘플링 (치아의 절단연/교합면)
        ymin_samples = polar_sampler.polar_sampling(
            points,
            angle_step=1,
            mode="ymin",
            start_angle=start_angle,
            end_angle=end_angle,
            y_range=(-np.inf, np.inf)
        )
        
        # farthest 모드로 샘플링 (가장 먼 점)
        farthest_samples = polar_sampler.polar_sampling(
            points,
            angle_step=1,
            mode="farthest",
            start_angle=start_angle,
            end_angle=end_angle,
            y_range=(-np.inf, np.inf)
        )
        
        # RMSE 계산
        rmse = self._calculate_rmse(ymin_samples, farthest_samples)
        
        # 타입 분류
        if rmse < self.RMSE_THRESHOLD:
            arch_type = "dentulous"
            filter_window_size = self.DENTULOUS_WINDOW_SIZE
        else:
            arch_type = "edentulous"
            filter_window_size = self.EDENTULOUS_WINDOW_SIZE
        
        return arch_type, filter_window_size, rmse
    
    def _calculate_rmse(self, samples1: np.ndarray, samples2: np.ndarray) -> float:
        """
        두 샘플 세트 간의 RMSE(Root Mean Square Error)를 계산합니다.
        
        Args:
            samples1: 첫 번째 샘플 배열
            samples2: 두 번째 샘플 배열
            
        Returns:
            float: RMSE 값
        """
        return np.sqrt(np.mean((samples1 - samples2) ** 2))
    
    def get_filter_window_size(self, arch_type: str) -> int:
        """
        악궁 타입에 맞는 필터 윈도우 크기를 반환합니다.
        
        Args:
            arch_type: "dentulous" 또는 "edentulous"
            
        Returns:
            int: 필터 윈도우 크기
        """
        if arch_type == "dentulous":
            return self.DENTULOUS_WINDOW_SIZE
        else:
            return self.EDENTULOUS_WINDOW_SIZE

