"""
치아 스캔 데이터에서 악궁(Dental Arch) 곡선을 추출하는 모듈

이 모듈은 IOS(Intraoral Scanner) 스캔 데이터를 분석하여
치아 악궁의 중심선을 추출하고 정규화된 랜드마크를 생성합니다.
"""

from typing import Tuple, List
from pyNeo3DLib.smileArchOuterline.utils.analysis.arch_analysis_coordinator import ArchAnalysisCoordinator


# ===== 기존 함수들은 별도 클래스로 분리됨 =====
# 모든 기능은 각각의 전용 클래스에서 처리됩니다:
# - RayCaster: 레이 캐스팅 관련 기능
# - MeshAlignmentManager: 메쉬 정렬 관련 기능  
# - SignalProcessor: 신호 처리 및 필터링 기능
# - CurveSampler: 곡선 샘플링 관련 기능
# - LandmarkCalculator: 랜드마크 계산 기능
# - ArchAnalyzer: 메인 분석 클래스


# ===== 메인 분석 함수 (래퍼) =====
def analyze_upper_IOS_scandata(
    mesh_path: str,
    visualize_result: bool = True
) -> Tuple[float, float, List[List[float]]]:
    """
    상악 IOS 스캔 데이터에서 치아 아치 곡선을 추출합니다.
    
    이 함수는 기존 인터페이스를 유지하기 위한 래퍼 함수입니다.
    실제 분석은 ArchAnalyzer 클래스에서 수행됩니다.
    
    Args:
        mesh_path: STL 메쉬 파일 경로
        visualize_result: 결과 시각화 여부 (기본값: True)
        
    Returns:
        Tuple[arch_depth, molar_width, landmark_points]: 
            - arch_depth: 치아 배열 곡선의 깊이
            - molar_width: 치아 배열 곡선의 폭
            - landmark_points: 정규화된 랜드마크 포인트 리스트
    """
    coordinator = ArchAnalysisCoordinator()
    return coordinator.analyze_upper_IOS_scandata(
        mesh_path=mesh_path,
        visualize_result=visualize_result
    )

