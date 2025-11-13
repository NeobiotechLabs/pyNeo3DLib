"""
치아 악궁 분석을 담당하는 메인 클래스
리팩토링: 단일책임 원칙 준수 및 중복 로직 통합
"""

import numpy as np
from typing import List, Tuple
from .arch_analysis_coordinator import ArchAnalysisCoordinator


class ArchAnalyzer:
    """치아 악궁 분석을 담당하는 메인 클래스 (리팩토링됨)"""
    
    def __init__(self):
        self.coordinator = ArchAnalysisCoordinator()
    
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
        return self.coordinator.mesh_processor.perform_initial_alignment(mesh_path)

    def perform_precise_alignment(self, aligned_mesh: object, y_axis: np.ndarray) -> Tuple[np.ndarray, object, np.ndarray]:
        """
        정밀정렬을 수행합니다.
        
        Args:
            aligned_mesh: 1차 정렬된 메쉬
            y_axis: Y축 벡터
            
        Returns:
            Tuple[smoothed_points, rotated_mesh, new_center_point]:
                - smoothed_points: 정밀정렬된 곡선 포인트들
                - rotated_mesh: 정밀정렬된 메쉬
                - new_center_point: 새로운 중심점
        """
        return self.coordinator.perform_precise_alignment(aligned_mesh, y_axis)

    def extract_precise_curve_points(self, aligned_mesh: object, rotation_axis: np.ndarray) -> Tuple[np.ndarray, object, np.ndarray]:
        """
        정밀한 곡선 포인트를 추출합니다.
        
        Args:
            aligned_mesh: 정렬된 메쉬
            rotation_axis: 회전축
            
        Returns:
            Tuple[moving_average_points, filtered_aligned_mesh, center_point]:
                - moving_average_points: 이동평균 처리된 포인트들
                - filtered_aligned_mesh: 필터링된 메쉬
                - center_point: 중심점
        """
        return self.coordinator.extract_precise_curve_points(aligned_mesh, rotation_axis)

    def analyze_upper_IOS_scandata(
        self,
        mesh_path: str,
        visualize_result: bool = True
    ) -> Tuple[float, float, List[List[float]]]:
        """
        상악 IOS 스캔 데이터에서 치아 아치 곡선을 추출합니다.
        
        Args:
            mesh_path: STL 메쉬 파일 경로
            visualize_result: 결과 시각화 여부 (기본값: True)
            
        Returns:
            Tuple[arch_depth, molar_width, landmark_points]: 
                - arch_depth: 치아 배열 곡선의 깊이
                - molar_width: 치아 배열 곡선의 폭
                - landmark_points: 정규화된 랜드마크 포인트 리스트
        """
        return self.coordinator.analyze_upper_IOS_scandata(mesh_path, visualize_result)
