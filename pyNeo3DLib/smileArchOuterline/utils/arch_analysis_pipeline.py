"""
치아 악궁 분석 파이프라인
단일책임: 악궁 분석 프로세스의 순차적 단계 관리
"""

import numpy as np
from typing import Tuple
from .curve_extractor import CurveExtractor
from .signal_processor import SignalProcessor
from .arch_type_classifier import ArchTypeClassifier
from .tangent_normal_from_curve import CurveTangentNormalCalculator
from .constants import AnalysisConstants


class ArchAnalysisPipeline:
    """악궁 분석의 각 단계를 순차적으로 처리하는 파이프라인 클래스"""
    
    def __init__(self):
        """초기화"""
        self.curve_extractor = CurveExtractor()
        self.signal_processor = SignalProcessor()
        self.arch_classifier = ArchTypeClassifier()
        self.tangent_calculator = CurveTangentNormalCalculator()
    
    def extract_initial_curve(
        self, 
        mesh_points: np.ndarray, 
        z_min: float
    ) -> np.ndarray:
        """
        1단계: 초기 곡선 추출 (극좌표 샘플링)
        
        Args:
            mesh_points: 메쉬 포인트 배열
            z_min: Z축 최소값
            
        Returns:
            np.ndarray: 추출된 곡선 포인트
        """
        return self.curve_extractor.extract_curve_by_polar_sampling(
            mesh_points, z_min
        )
    
    def smooth_curve(
        self, 
        curve_points: np.ndarray, 
        window_size: int = None
    ) -> np.ndarray:
        """
        2단계: 곡선 스무딩 (이동평균 필터)
        
        Args:
            curve_points: 곡선 포인트 배열
            window_size: 윈도우 크기 (None이면 기본값 사용)
            
        Returns:
            np.ndarray: 스무딩된 곡선 포인트
        """
        if window_size is None:
            window_size = AnalysisConstants.DEFAULT_WINDOW_SIZE
        
        return self.signal_processor.moving_average_filter(
            curve_points, 
            window_size=window_size
        )
    
    def expand_curve_outward(
        self, 
        curve_points: np.ndarray, 
        expand_distance: float = 5.0
    ) -> np.ndarray:
        """
        3단계: 곡선을 외측으로 확장
        
        곡선의 법선 방향으로 일정 거리만큼 확장하여
        메쉬 필터링에 사용할 경계선을 생성합니다.
        
        Args:
            curve_points: 곡선 포인트 배열
            expand_distance: 확장 거리 (기본값: 5.0)
            
        Returns:
            np.ndarray: 확장된 곡선 포인트
        """
        _, curve_normal = self.tangent_calculator.calculate_tangents_and_normals(
            curve_points
        )
        return curve_points - curve_normal * expand_distance
    
    def filter_mesh_by_curve_boundary(
        self, 
        mesh_points: np.ndarray, 
        boundary_curve: np.ndarray
    ) -> np.ndarray:
        """
        4단계: 곡선 경계를 기준으로 메쉬 필터링
        
        확장된 곡선의 X축 범위 내의 포인트만 남깁니다.
        
        Args:
            mesh_points: 메쉬 포인트 배열
            boundary_curve: 경계 곡선 포인트
            
        Returns:
            np.ndarray: 필터링된 메쉬 포인트
        """
        x_min = np.min(boundary_curve[:, 0])
        x_max = np.max(boundary_curve[:, 0])
        
        mask = (mesh_points[:, 0] > x_min) & (mesh_points[:, 0] < x_max)
        return mesh_points[mask]
    
    def classify_and_extract_final_curve(
        self, 
        filtered_points: np.ndarray, 
        z_min: float
    ) -> Tuple[np.ndarray, str, int, float]:
        """
        5단계: 치아 타입 분류 및 최종 곡선 추출
        
        Args:
            filtered_points: 필터링된 포인트 배열
            z_min: Z축 최소값
            
        Returns:
            Tuple[final_curve, arch_type, window_size, rmse]:
                - final_curve: 최종 스무딩된 곡선
                - arch_type: 악궁 타입 ("dentulous" or "edentulous")
                - window_size: 사용된 윈도우 크기
                - rmse: 분류 기준 RMSE 값
        """
        # 5-1: 2차 곡선 추출
        second_curve = self.extract_initial_curve(filtered_points, z_min)
        
        # 5-2: 치아 타입 분류
        arch_type, window_size, rmse = self.arch_classifier.classify_arch_type(
            filtered_points, z_min
        )
        
        # 5-3: 타입에 맞는 필터로 최종 스무딩
        final_curve = self.smooth_curve(second_curve, window_size)
        
        return final_curve, arch_type, window_size, rmse
    
    def remove_outliers(
        self, 
        points: np.ndarray, 
        percentile_threshold: float = None
    ) -> np.ndarray:
        """
        대구치 영역의 아웃라이어 제거
        
        Args:
            points: 포인트 배열
            percentile_threshold: 백분위수 임계값
            
        Returns:
            np.ndarray: 아웃라이어가 제거된 포인트 배열
        """
        if percentile_threshold is None:
            percentile_threshold = AnalysisConstants.MOLAR_OUTLIER_PERCENTILE_THRESHOLD
        
        return self.signal_processor.remove_molar_outliers(
            points, 
            percentile_threshold=percentile_threshold
        )

