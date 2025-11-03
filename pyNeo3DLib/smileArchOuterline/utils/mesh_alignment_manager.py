"""
메쉬 정렬 관련 기능을 담당하는 클래스
"""

import numpy as np
import pyvista as pv
from typing import Tuple, Optional
from .constants import AnalysisConstants
from .ray_caster import RayCaster


class MeshAlignmentManager:
    """메쉬 정렬을 관리하는 클래스"""
    
    def __init__(self):
        self.ray_caster = RayCaster()
        self.x_axis_intersection_count = AnalysisConstants.X_AXIS_INTERSECTION_COUNT
        self.y_axis_intersection_count = AnalysisConstants.Y_AXIS_INTERSECTION_COUNT
        self.z_axis_intersection_count = AnalysisConstants.Z_AXIS_INTERSECTION_COUNT
    
    def determine_alignment_axes(
        self,
        input_mesh, 
        center: np.ndarray, 
        principal_evecs: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        메쉬의 주축 벡터를 분석하여 정렬에 사용할 각 축을 결정합니다.
        
        양방향 레이 캐스팅의 교차점 개수를 기준으로:
        - 4개 교차점: X축 정렬용
        - 1개 교차점: Y축 정렬용
        - 2개 교차점: Z축 정렬용
        
        Args:
            input_mesh: PyVista PolyData 객체
            center: 메쉬 중심점 (1, 3)
            principal_evecs: 주축 벡터들 (3, 3)
            
        Returns:
            Tuple[x축용 벡터, y축용 벡터, z축용 벡터]
        """
        evec_x = None
        evec_y = None
        evec_z = None
        
        # 각 주축에 대해 레이 캐스팅 수행
        intersection_counts = []
        for i in range(3):
            intersection_points = self.ray_caster.get_bidirectional_ray_points(
                input_mesh, center, principal_evecs[:, i]
            )
            count = len(intersection_points)
            intersection_counts.append(count)

            # 교차점 개수가 self.y_axis_intersection_count 이면  교차점중 레이원점과 가장 가까운 점 하나를 추출 하고 (교차점-레이원점을 으로 벡터를 구하고 evec_y로 할당)
            if count == self.y_axis_intersection_count:
                closest_point = intersection_points[np.argmin(np.linalg.norm(intersection_points - center, axis=1))]
                evec_y = closest_point - center
                evec_y = evec_y / np.linalg.norm(evec_y)

            # 교차점 개수가 self.z_axis_intersection_count 이면  교차점중 레이원점과 가장 가까운 점 하나를 추출 하고 (교차점-레이원점을 으로 벡터를 구하고 evec_z로 할당)
            if count == self.z_axis_intersection_count:
                closest_point = intersection_points[np.argmin(np.linalg.norm(intersection_points - center, axis=1))]
                evec_z = closest_point - center
                evec_z = evec_z / np.linalg.norm(evec_z)


        # evec_y와 evec_z가 모두 결정된 후에 evec_x 계산
        if evec_y is not None and evec_z is not None:
            evec_x = np.cross(evec_y, evec_z)
            evec_x = evec_x / np.linalg.norm(evec_x)


        
        return evec_x, evec_y, evec_z
    
    def align_mesh_to_global_coordinates(
        self,
        input_mesh: pv.PolyData, 
        evec_x: np.ndarray, 
        evec_y: np.ndarray, 
        evec_z: np.ndarray
    ) -> np.ndarray:
        """
        주축 벡터를 사용하여 메쉬를 글로벌 좌표계로 정렬합니다.
        
        목표 정렬:
        - evec_x → 글로벌 X축 (1, 0, 0)
        - evec_y → 글로벌 Y축 (0, 1, 0)
        - evec_z → 글로벌 Z축 (0, 0, 1)
        
        Args:
            input_mesh: 정렬할 메쉬
            evec_x: X축 정렬용 주축 벡터
            evec_y: Y축 정렬용 주축 벡터
            evec_z: Z축 정렬용 주축 벡터
            
        Returns:
            aligned_mesh: 정렬된 메쉬
        """
        # 주축 벡터들로 기저 행렬 구성
        basis_matrix = np.array([evec_x, evec_y, evec_z]).reshape(3, 3)
        inverse_basis = np.linalg.inv(basis_matrix)
        
        # 목표 좌표계 정의
        target_matrix = np.array([
            [1, 0, 0],    # X축
            [0, 1, 0],   # -Y축
            [0, 0, 1]     # Z축
        ])
        
        # 회전 행렬 계산 및 적용
        rotation_matrix = np.matmul(inverse_basis, target_matrix)
        aligned_vertices = np.matmul(input_mesh.points, rotation_matrix)

        aligned_mesh = input_mesh.copy()
        aligned_mesh.points = np.asarray(aligned_vertices)
        
        return aligned_mesh
    
    def filter_mesh_by_z_threshold(self, aligned_mesh, filtered_points):
        """
        Z 임계값을 기준으로 메쉬 필터링
        
        Args:
            aligned_mesh: 정렬된 메쉬
            filtered_points: 필터링된 포인트들
            
        Returns:
            filtered_aligned_mesh: 필터링된 메쉬
        """
        z_min_point = np.min(filtered_points[:, 2])
        mask = aligned_mesh.points[:, 2] > z_min_point
        filtered_aligned_mesh = aligned_mesh.extract_points(mask)

        # 가장 큰 덩어리만 추출
        largest_component = filtered_aligned_mesh.extract_largest()

        # 중복된 면이나 점이 있는 경우 제거
        filtered_largest_component = largest_component.clean()
        
        return filtered_largest_component

