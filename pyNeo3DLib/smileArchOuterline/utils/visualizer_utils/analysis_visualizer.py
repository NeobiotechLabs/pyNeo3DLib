"""
분석 결과 시각화를 담당하는 클래스
단일책임: 분석 결과의 시각화
"""

import numpy as np
from typing import List
from .visualizer import VisualizeForTest
from ..general_utils.constants import AnalysisConstants


class AnalysisVisualizer:
    """분석 결과 시각화를 담당하는 클래스"""
    
    def __init__(self):
        self.visualizer = VisualizeForTest()
    
    def visualize_precise_alignment_results(
        self, 
        mesh: object, 
        center_point: np.ndarray, 
        curve_points: np.ndarray
    ) -> None:
        """
        정밀정렬 결과를 시각화합니다.
        
        Args:
            mesh: 정렬된 메시
            center_point: 중심점
            curve_points: 곡선 포인트들
        """
        self.visualizer.visualize_mesh(mesh, color='yellow', opacity=0.5)
        self.visualizer.visualize_points(center_point, color='blue', point_size=5)
        self.visualizer.visualize_points(curve_points, color='red', point_size=5)
        self.visualizer.show()
    
    def visualize_analysis_results(
        self,
        z_min_point: np.ndarray, 
        processed_vertices: np.ndarray,
        smoothed_points: np.ndarray, 
        sampled_points: List[np.ndarray]
    ) -> None:
        """
        분석 결과를 시각화합니다.
        
        Args:
            z_min_point: Z 최소값 참조점
            processed_vertices: 전처리된 버텍스
            smoothed_points: 부드럽게 처리된 곡선 포인트
            sampled_points: 최종 샘플링된 포인트
        """
        # 참조점 (파란색)
        self.visualizer.visualize_points(
            z_min_point.reshape(1, -1), 
            color='blue', 
            point_size=10
        )
        
        # 전처리된 메쉬 (분홍색)
        self.visualizer.visualize_points(
            processed_vertices, 
            color='pink', 
            point_size=2
        )
        
        # 부드러운 곡선 (노란색)
        self.visualizer.visualize_points(
            smoothed_points, 
            color='yellow', 
            point_size=5
        )
        
        # 최종 샘플링 포인트 (녹색)
        self.visualizer.visualize_points(
            sampled_points, 
            color='green', 
            point_size=15
        )
        
        self.visualizer.show()
