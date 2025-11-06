"""
레이 캐스팅 관련 기능을 담당하는 클래스
"""

import numpy as np
import pyvista as pv
from typing import List
from pyNeo3DLib.smileArchOuterline.utils.common.constants import AnalysisConstants
from pyNeo3DLib.smileArchOuterline.utils.visualization.visualizer import VisualizeForTest
from pyNeo3DLib.smileArchOuterline.utils.common.vector_utils import VectorUtils
from pyNeo3DLib.smileArchOuterline.utils.ray_casting.point_cloud_ray_caster import PointCloudRayCaster


class RayCaster:
    """레이 캐스팅을 수행하는 클래스"""
    
    def __init__(self):
        self.ray_length = AnalysisConstants.RAY_LENGTH
        self.ray_scale_factor = AnalysisConstants.RAY_SCALE_FACTOR
    
    def ray_casting(self, mesh, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        레이 캐스팅 함수 (PyVista mesh 사용)
        
        Args:
            mesh: PyVista PolyData 객체
            origin: 레이 시작점 (numpy array, shape: (1,3))
            direction: 레이 방향 벡터 (numpy array, shape: (1,3))
        
        Returns:
            교차점 좌표 (numpy array, shape: (N, 3))
        """
        # 배열 형태 정규화
        origin_flat = origin.flatten()
        direction_flat = direction.flatten()
        
        # 방향 벡터 정규화
        direction_norm = direction_flat / np.linalg.norm(direction_flat)
        
        # 레이의 끝점 계산
        end_point = origin_flat + direction_norm * self.ray_length
        
        # PyVista ray_trace로 교차점 계산
        points, ind = mesh.ray_trace(origin_flat, end_point)
        
        if len(points) == 0:
            # 교차점이 없는 경우
            return np.array([]).reshape(0, 3)
        
        # 교차점들 반환
        return points

    
    def get_bidirectional_ray_points(
        self,
        input_mesh, 
        center: np.ndarray, 
        principal_evec: np.ndarray, 
        scale_factor: float = None
    ) -> np.ndarray:
        """
        주어진 주축 방향으로 양방향 레이 캐스팅을 수행하여 교차점들을 반환합니다.
        
        Args:
            input_mesh: PyVista PolyData 객체
            center: 레이 시작점 (numpy array, shape: (1,3))
            principal_evec: 주축 방향 벡터 (numpy array, shape: (3,) 또는 (3,1))
            scale_factor: 레이 방향 벡터의 스케일 팩터 (기본값: RAY_SCALE_FACTOR)
        
        Returns:
            total_points: 양방향 레이 캐스팅으로 얻은 모든 교차점 (numpy array, shape: (N, 3))
        """
        if scale_factor is None:
            scale_factor = self.ray_scale_factor
        
        # 방향 벡터를 1차원으로 변환
        evec = principal_evec.flatten()
        
        # 양방향 레이 방향 벡터 생성
        plus_direction = (evec * scale_factor).reshape(1, 3)
        minus_direction = (-evec * scale_factor).reshape(1, 3)
        
        # 양방향 레이 캐스팅 수행
        plus_points = self.ray_casting(input_mesh, center, plus_direction)
        minus_points = self.ray_casting(input_mesh, center, minus_direction)
        
        # 결과 합치기
        return np.concatenate([plus_points, minus_points], axis=0)
