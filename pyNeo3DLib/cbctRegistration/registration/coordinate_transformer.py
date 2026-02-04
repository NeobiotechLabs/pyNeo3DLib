"""
좌표계 변환 처리 모듈

RAI 좌표계와 표준 좌표계 간의 변환을 담당합니다.

Note:
    - 기본 변환 함수들은 utils.py로 이동됨
    - 이 모듈은 좌표계 특화된 변환 기능만 제공
"""
import numpy as np
import open3d as o3d
from typing import Tuple

from ..utils import apply_transform, compute_translation_matrix


class CoordinateTransformer:
    """
    좌표계 변환을 담당하는 클래스
    
    주요 기능:
    - RAI → 표준 좌표계 변환
    - 코 중심 기반 원점 이동
    """
    
    # 기본 RAI → 표준 좌표계 변환 행렬 (Y축 기준 180도 회전)
    DEFAULT_RAI_TO_STANDARD = np.array([
        [-1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0, -1,  0],
        [ 0,  0,  0,  1]
    ], dtype=np.float64)
    
    def __init__(self, rai_to_standard_matrix: np.ndarray = None):
        """
        Args:
            rai_to_standard_matrix: RAI 좌표계를 표준 좌표계로 변환하는 4x4 행렬
        """
        self.rai_to_standard_matrix = (
            rai_to_standard_matrix 
            if rai_to_standard_matrix is not None 
            else self.DEFAULT_RAI_TO_STANDARD.copy()
        )
    
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """
        단일 포인트를 RAI에서 표준 좌표계로 변환
        
        Args:
            point: (3,) RAI 좌표
        
        Returns:
            (3,) 표준 좌표
        """
        point_h = np.append(point, 1)
        return (self.rai_to_standard_matrix @ point_h)[:3]
    
    def compute_combined_transform(
        self, 
        nose_center_rai: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RAI → 표준 좌표계 변환 + 코 중심을 원점으로 이동하는 결합 변환 행렬 계산
        
        Args:
            nose_center_rai: RAI 좌표계에서의 코 중심 (3,)
        
        Returns:
            Tuple[combined_transform, nose_center_std]:
                - combined_transform: 결합 변환 행렬 (4x4)
                - nose_center_std: 표준 좌표계에서의 코 중심 (변환 전)
        """
        # 코 중심을 표준 좌표계로 변환
        nose_center_std = self.transform_point(nose_center_rai)
        
        # 원점 이동 변환 행렬
        translation_matrix = compute_translation_matrix(-nose_center_std)
        
        # 결합 변환: 원점이동 @ RAI변환
        combined_transform = translation_matrix @ self.rai_to_standard_matrix
        
        return combined_transform, nose_center_std
    
    def transform_to_standard_with_origin(
        self, 
        pcd: o3d.geometry.PointCloud, 
        nose_center_rai: np.ndarray,
        verbose: bool = False
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        포인트 클라우드를 RAI에서 표준 좌표계로 변환 (코 중심을 원점으로)
        
        Args:
            pcd: RAI 좌표계의 포인트 클라우드
            nose_center_rai: RAI 좌표계에서의 코 중심
            verbose: 상세 출력 여부
        
        Returns:
            Tuple[pcd_transformed, combined_transform]:
                - pcd_transformed: 변환된 포인트 클라우드
                - combined_transform: 사용된 변환 행렬
        """
        combined_transform, nose_center_std = self.compute_combined_transform(nose_center_rai)
        
        if verbose:
            print(f"코 중심 (RAI): {nose_center_rai}")
            print(f"코 중심 (표준): {nose_center_std}")
            print(f"결합 변환 행렬:\n{combined_transform}")
        
        pcd_transformed = apply_transform(pcd, combined_transform)
        
        return pcd_transformed, combined_transform
    
    @classmethod
    def get_default_matrix(cls) -> np.ndarray:
        """기본 RAI → 표준 좌표계 변환 행렬 반환"""
        return cls.DEFAULT_RAI_TO_STANDARD.copy()

