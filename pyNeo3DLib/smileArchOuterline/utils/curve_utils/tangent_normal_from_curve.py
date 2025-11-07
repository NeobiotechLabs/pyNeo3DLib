import numpy as np
import pyvista as pv
from typing import Tuple

class CurveTangentNormalCalculator:
    """
    곡선의 접선과 법선 벡터를 계산 및 시각화하는 클래스
    """
    
    def __init__(self):
        """
        CurveTangentNormalCalculator 클래스의 생성자
        """
        pass
        
    def calculate_tangents_and_normals(self, curve: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        곡선의 접선과 법선 벡터를 계산하는 메서드
        
        Args:
            curve (numpy.ndarray): 스플라인 곡선의 점들 (N x 3 배열)
            sort_first (bool): 계산 전 점들을 기하학적으로 정렬할지 여부
            
        Returns:
            tuple: (접선 벡터 배열, 법선 벡터 배열)
        """

            
        # 접선 벡터와 법선 벡터를 저장할 배열 초기화    
        tangents = np.zeros_like(curve)
        normals = np.zeros_like(curve)
        
        # 각 점에서의 접선과 법선 계산
        for i in range(len(curve)-1):
            # 현재 점과 다음 점
            current_point = curve[i]
            next_point = curve[i+1]
            
            # 접선 벡터 계산 (다음 점 - 현재 점)
            tangent = next_point - current_point
            # 접선 벡터 정규화
            norm = np.linalg.norm(tangent)
            if norm > 1e-10:  # 0에 가까운 값으로 나누는 것 방지
                tangent_normalized = tangent / norm
            else:
                tangent_normalized = np.zeros_like(tangent)  # 또는 다른 기본 벡터 사용

            
            # y축 방향 벡터 [0,1,0]
            y_axis = np.array([0, 1, 0])
            
            # 법선 벡터 계산 (접선 벡터와 y축의 외적)
            normal = np.cross(tangent_normalized, y_axis)
            # 법선 벡터 정규화
            normal_normalized = normal / np.linalg.norm(normal)
            
            # 결과 저장
            tangents[i] = tangent_normalized
            normals[i] = normal_normalized
        
        # 마지막 점의 접선과 법선은 이전 점과 동일하게 설정
        tangents[-1] = tangents[-2]
        normals[-1] = normals[-2]
        
        return tangents, normals

  