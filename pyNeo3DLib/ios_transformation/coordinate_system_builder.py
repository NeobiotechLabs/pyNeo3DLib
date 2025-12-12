"""
좌표계 구축 모듈

직교 정규 좌표계를 구축하는 클래스입니다.
"""

import numpy as np
from typing import Tuple


class CoordinateSystemBuilder:
    """
    좌표계 구축 클래스
    
    그람-슈미트 직교 정규화를 사용하여 완벽한 직교 정규 기저를 생성합니다.
    """
    
    def build(
        self, 
        single_intersection_direction: np.ndarray, 
        closest_axis_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        좌표계를 구축합니다 (x, y, z 축).
        
        그람-슈미트 직교 정규화를 적용하여 완벽한 직교 정규 기저를 생성합니다.
        
        Args:
            single_intersection_direction: Y축이 될 방향 벡터
            closest_axis_vector: Z축이 될 방향 벡터
            
        Returns:
            정규화된 x_axis, y_axis, z_axis 벡터 (단위 벡터, 서로 직교)
        """
        print(f"[DEBUG] Input vectors:")
        print(f"   y_axis (original): {single_intersection_direction}, "
              f"norm: {np.linalg.norm(single_intersection_direction):.6f}")
        print(f"   z_axis (original): {closest_axis_vector}, "
              f"norm: {np.linalg.norm(closest_axis_vector):.6f}")
        print(f"   y dot z: {np.dot(single_intersection_direction, closest_axis_vector):.6f}")
        
        # 그람-슈미트 정규화로 직교 정규 기저 생성
        # y축을 우선으로 유지하고, z축을 조정한 후 x축을 재계산
        
        # 1. y축 정규화
        y_axis_vector = single_intersection_direction / np.linalg.norm(single_intersection_direction)
        
        # 2. z축을 y축에 직교하도록 조정 후 정규화
        z_orthogonal = closest_axis_vector - np.dot(closest_axis_vector, y_axis_vector) * y_axis_vector
        z_orthogonal_norm = np.linalg.norm(z_orthogonal)
        
        # 두 벡터가 평행한 경우 (collinear) 체크 - 0으로 나누기 방지
        if z_orthogonal_norm < 1e-10:
            raise ValueError(
                "closest_axis_vector와 single_intersection_direction이 평행(collinear)합니다. "
                "직교 좌표계를 구축할 수 없습니다. 서로 다른 방향의 벡터를 입력해주세요."
            )
        
        z_axis_vector = z_orthogonal / z_orthogonal_norm
        
        # 3. x축을 y축과 z축에 직교하도록 외적으로 재계산
        x_axis_vector = np.cross(y_axis_vector, z_axis_vector)
        
        # 정규화된 축 벡터 검증
        print(f"[INFO] Normalized axis vectors:")
        print(f"   x_axis: {x_axis_vector}, norm: {np.linalg.norm(x_axis_vector):.10f}")
        print(f"   y_axis: {y_axis_vector}, norm: {np.linalg.norm(y_axis_vector):.10f}")
        print(f"   z_axis: {z_axis_vector}, norm: {np.linalg.norm(z_axis_vector):.10f}")
        print(f"   x dot y: {np.dot(x_axis_vector, y_axis_vector):.10f} (should be 0)")
        print(f"   y dot z: {np.dot(y_axis_vector, z_axis_vector):.10f} (should be 0)")
        print(f"   z dot x: {np.dot(z_axis_vector, x_axis_vector):.10f} (should be 0)")
        
        return x_axis_vector, y_axis_vector, z_axis_vector
