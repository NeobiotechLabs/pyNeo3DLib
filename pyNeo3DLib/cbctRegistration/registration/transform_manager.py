"""
변환 행렬 관리 모듈

여러 단계의 변환 행렬을 관리하고 누적 변환을 계산합니다.
"""
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class TransformManager:
    """
    변환 행렬 관리 클래스
    
    파이프라인의 각 단계에서 생성되는 변환 행렬을 추적하고
    누적 변환을 계산합니다.
    """
    
    # 각 단계별 변환 행렬
    facescan_to_smilearch: np.ndarray = field(default_factory=lambda: np.eye(4))
    rai_to_standard: np.ndarray = field(default_factory=lambda: np.eye(4))
    initial_alignment: np.ndarray = field(default_factory=lambda: np.eye(4))
    icp: np.ndarray = field(default_factory=lambda: np.eye(4))
    refinement: np.ndarray = field(default_factory=lambda: np.eye(4))
    
    def get_accumulated_transform(self, include_refinement: bool = True) -> np.ndarray:
        """
        누적 변환 행렬 계산 (표준 좌표계 기준)
        
        Args:
            include_refinement: 정제 변환 포함 여부
        
        Returns:
            np.ndarray: 누적 변환 행렬 (4x4)
        """
        # 표준 좌표계에서의 변환 순서: 초기정렬 → ICP → (정제)
        accumulated = self.initial_alignment.copy()
        accumulated = self.icp @ accumulated
        
        if include_refinement:
            accumulated = self.refinement @ accumulated
        
        return accumulated
    
    def get_final_transform(self, include_refinement: bool = True) -> np.ndarray:
        """
        최종 변환 행렬 계산 (RAI 좌표계 → FaceScan 좌표계)
        
        Args:
            include_refinement: 정제 변환 포함 여부
        
        Returns:
            np.ndarray: 최종 변환 행렬 (4x4)
        """
        # 전체 변환 순서: RAI→표준 → 초기정렬 → ICP → (정제)
        accumulated = self.get_accumulated_transform(include_refinement)
        return accumulated @ self.rai_to_standard
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        모든 변환 행렬을 딕셔너리로 변환
        
        Returns:
            Dict[str, np.ndarray]: 변환 행렬 딕셔너리
        """
        return {
            "facescan_to_smilearch": self.facescan_to_smilearch,
            "rai_to_standard": self.rai_to_standard,
            "initial_alignment": self.initial_alignment,
            "icp": self.icp,
            "refinement": self.refinement,
            "accumulated": self.get_accumulated_transform(include_refinement=False),
            "accumulated_refined": self.get_accumulated_transform(include_refinement=True),
            "final": self.get_final_transform(include_refinement=False),
            "final_refined": self.get_final_transform(include_refinement=True),
        }
    
    def print_summary(self, include_refinement: bool = True):
        """
        변환 행렬 요약 출력
        
        Args:
            include_refinement: 정제 변환 포함 여부
        """
        print("\n" + "=" * 60)
        print("변환 행렬 요약")
        print("=" * 60)
        
        print("\n[1단계] FaceScan → SmileArch:")
        print(self.facescan_to_smilearch)
        
        print("\n[2단계] RAI → 표준 좌표계:")
        print(self.rai_to_standard)
        
        print("\n[3단계] 초기 정렬:")
        print(self.initial_alignment)
        
        print("\n[4단계] ICP 정합:")
        print(self.icp)
        
        if include_refinement:
            print("\n[5단계] 정제 (Z축 회전):")
            print(self.refinement)
        
        print("\n" + "-" * 60)
        print("누적 변환 행렬:")
        print(self.get_accumulated_transform(include_refinement))
        
        print("\n최종 변환 행렬 (RAI → FaceScan):")
        print(self.get_final_transform(include_refinement))
        print("=" * 60)


__all__ = [
    "TransformManager",
]


