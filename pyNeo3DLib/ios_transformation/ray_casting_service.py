"""
레이캐스팅 서비스 모듈

메시에 대한 레이캐스팅 작업을 수행하는 클래스입니다.
"""

import numpy as np
import pyvista as pv
from typing import Any, Optional

from pyNeo3DLib.smileArchOuterline.utils.ray_caster import RayCaster


class RayCastingService:
    """
    레이캐스팅 서비스 클래스
    
    메시에 대해 레이캐스팅을 수행하여 교차점을 찾고,
    단일 교차점 방향을 결정합니다.
    """
    
    def __init__(self):
        self._ray_caster = RayCaster()
    
    def create_pyvista_mesh(
        self, 
        vertices: np.ndarray, 
        faces: np.ndarray
    ) -> Any:
        """
        PyVista 메시를 생성합니다.
        
        Args:
            vertices: 메시의 정점 배열
            faces: 메시의 면 정보
            
        Returns:
            PyVista PolyData 객체
        """
        faces_with_count = np.column_stack([np.full(len(faces), 3), faces])
        return pv.PolyData(vertices, faces_with_count)
    
    def find_surface_point_by_raycasting(
        self,
        pv_mesh: Any,
        origin: np.ndarray,
        direction: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        레이캐스팅으로 표면 포인트를 찾습니다.
        
        양방향으로 레이캐스팅하여 교차점을 찾습니다.
        
        Args:
            pv_mesh: PyVista 메시 객체
            origin: 레이 시작점
            direction: 레이 방향 벡터
            
        Returns:
            교차점 좌표, 없으면 None
        """
        # plus 방향 레이캐스팅
        plus_intersections = self._ray_caster.ray_casting(
            pv_mesh, origin.reshape(1, 3), direction.reshape(1, 3)
        )
        
        if len(plus_intersections) > 0:
            return plus_intersections[0]
        
        # minus 방향 레이캐스팅
        minus_intersections = self._ray_caster.ray_casting(
            pv_mesh, origin.reshape(1, 3), (-direction).reshape(1, 3)
        )
        
        if len(minus_intersections) > 0:
            return minus_intersections[0]
        
        return None
    
    def get_ray_casting_vector_to_centroid(
        self,
        mesh_vertices: np.ndarray,
        mesh_faces: np.ndarray,
        centroid: np.ndarray,
        axis_vector: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        도심점에서 axis_vector 방향으로 양방향 레이캐스팅하여 
        메쉬 표면 포인트를 검출하고, 검출한 포인트에서 도심점 방향으로 
        가는 벡터를 반환합니다.
        
        Args:
            mesh_vertices: 메시의 정점 배열
            mesh_faces: 메시의 면 정보
            centroid: 메시의 무게중심
            axis_vector: 레이캐스팅 방향 벡터
            
        Returns:
            교차점에서 도심점으로 가는 벡터, 교차점이 없으면 None
        """
        pv_mesh = self.create_pyvista_mesh(mesh_vertices, mesh_faces)
        
        surface_point = self.find_surface_point_by_raycasting(
            pv_mesh, centroid, axis_vector
        )
        
        if surface_point is None:
            return None
        
        return centroid - surface_point
    
    def find_single_intersection_direction(
        self, 
        mesh_vertices: np.ndarray,
        mesh_faces: np.ndarray,
        principal_axes: np.ndarray, 
        centroid: np.ndarray, 
        closest_axis_idx: int
    ) -> Optional[np.ndarray]:
        """
        레이캐스팅을 통해 단일 교차점을 가진 축 방향을 찾습니다.
        
        Args:
            mesh_vertices: 메시의 정점 배열
            mesh_faces: 메시의 면 정보
            principal_axes: 주성분 분석으로 얻은 주축들 (3x3)
            centroid: 메시의 무게중심
            closest_axis_idx: 제외할 축의 인덱스
            
        Returns:
            단일 교차점을 가진 축 방향 벡터, 없으면 None
        """
        pv_mesh = self.create_pyvista_mesh(mesh_vertices, mesh_faces)
        
        # 제외할 축을 제외한 나머지 축들
        remaining_axes_indices = [i for i in range(3) if i != closest_axis_idx]
        
        # 나머지 두 축에 대해 레이캐스팅 수행
        for axis_idx in remaining_axes_indices:
            axis_vector = principal_axes[axis_idx]
            
            # 단일 교차점 확인
            unit_vector = self._check_single_intersection(
                pv_mesh, centroid, axis_vector, axis_idx
            )
            
            if unit_vector is not None:
                return unit_vector
        
        print("[WARNING] Could not find axis with single intersection point.")
        return None
    
    def _check_single_intersection(
        self,
        pv_mesh: Any,
        centroid: np.ndarray,
        axis_vector: np.ndarray,
        axis_idx: int
    ) -> Optional[np.ndarray]:
        """
        특정 축에 대해 단일 교차점 여부를 확인합니다.
        
        Args:
            pv_mesh: PyVista 메시 객체
            centroid: 메시의 무게중심
            axis_vector: 축 방향 벡터
            axis_idx: 축 인덱스
            
        Returns:
            단일 교차점이면 단위 벡터, 아니면 None
        """
        # +방향 레이캐스팅
        plus_intersections = self._ray_caster.ray_casting(
            pv_mesh, centroid.reshape(1, 3), axis_vector.reshape(1, 3)
        )
        plus_has_intersection = len(plus_intersections) > 0
        
        # -방향 레이캐스팅
        minus_intersections = self._ray_caster.ray_casting(
            pv_mesh, centroid.reshape(1, 3), (-axis_vector).reshape(1, 3)
        )
        minus_has_intersection = len(minus_intersections) > 0
        
        # 교차점 개수 계산
        intersection_count = int(plus_has_intersection) + int(minus_has_intersection)
        print(f"  Axis {axis_idx} raycasting result: {intersection_count} initial intersection(s)")
        
        # 최초 교차점이 정확히 1개인 경우
        if intersection_count == 1:
            intersection_point = self._get_closest_intersection_point(
                plus_intersections if plus_has_intersection else minus_intersections,
                centroid
            )
            
            # 도심점에서 교차점 방향으로 나가는 단위벡터 계산
            direction_vector = intersection_point - centroid
            unit_vector = direction_vector / np.linalg.norm(direction_vector)
            
            print(f"[INFO] Found axis with single intersection: axis {axis_idx}")
            print(f"   Intersection point: {intersection_point}")
            print(f"   Unit vector from centroid to intersection: {unit_vector}")
            
            return unit_vector
        
        return None
    
    def _get_closest_intersection_point(
        self,
        intersections: np.ndarray,
        centroid: np.ndarray
    ) -> np.ndarray:
        """
        교차점들 중 도심점에 가장 가까운 점을 반환합니다.
        
        Args:
            intersections: 교차점 배열
            centroid: 메시의 무게중심
            
        Returns:
            가장 가까운 교차점
        """
        distances = np.linalg.norm(intersections - centroid, axis=1)
        closest_idx = np.argmin(distances)
        return intersections[closest_idx]
