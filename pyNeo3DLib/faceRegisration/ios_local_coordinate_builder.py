"""
레이캐스팅을 통해 단일 교차점을 가진 축 방향을 찾는 클래스
"""

import numpy as np
import pyvista as pv
from typing import Optional, Any, List

from .ray_caster import RayCaster


class IOSLocalCoordinateSystemBuilder:
    """레이캐스팅을 통해 단일 교차점을 가진 축 방향을 찾는 클래스"""
    
    def __init__(self):
        self.ray_caster = RayCaster()
    
    def build_local_coordinate_system(
        self, 
        pv_mesh: pv.PolyData,
        principal_axes: np.ndarray, 
        centroid: np.ndarray, 
        closest_axis_idx: int,
        closest_axis_vector: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        레이캐스팅을 통해 단일 교차점을 가진 축 방향을 찾고 로컬 좌표계를 구성합니다.
        
        Args:
            pv_mesh: PyVista PolyData 객체
            principal_axes: 주성분 분석으로 얻은 주축들 (3x3)
            centroid: 메시의 무게중심
            closest_axis_idx: 제외할 축의 인덱스 (0, 1, 2 중 하나)
            closest_axis_vector: Z축 계산에 사용할 주축 벡터
            
        Returns:
            (x_axis_vector, y_axis_vector, z_axis_vector) 튜플, 실패 시 None
            
        Raises:
            ValueError: 입력값이 유효하지 않은 경우
        """
        # 입력값 검증
        if pv_mesh is None:
            raise ValueError("pv_mesh가 None입니다.")
        
        if not isinstance(pv_mesh, pv.PolyData):
            raise ValueError(f"pv_mesh는 pv.PolyData 타입이어야 합니다. 현재 타입: {type(pv_mesh)}")
        
        if principal_axes is None or not isinstance(principal_axes, np.ndarray):
            raise ValueError("principal_axes는 numpy 배열이어야 합니다.")
        
        if principal_axes.shape != (3, 3):
            raise ValueError(f"principal_axes의 shape은 (3, 3)이어야 합니다. 현재: {principal_axes.shape}")
        
        if centroid is None or not isinstance(centroid, np.ndarray):
            raise ValueError("centroid는 numpy 배열이어야 합니다.")
        
        if centroid.shape != (3,):
            raise ValueError(f"centroid의 shape은 (3,)이어야 합니다. 현재: {centroid.shape}")
        
        if closest_axis_idx not in (0, 1, 2):
            raise ValueError(f"closest_axis_idx는 0, 1, 2 중 하나여야 합니다. 현재: {closest_axis_idx}")
        
        if closest_axis_vector is None or not isinstance(closest_axis_vector, np.ndarray):
            raise ValueError("closest_axis_vector는 numpy 배열이어야 합니다.")
        
        if closest_axis_vector.shape != (3,):
            raise ValueError(f"closest_axis_vector의 shape은 (3,)이어야 합니다. 현재: {closest_axis_vector.shape}")
        
        # 제외할 축을 제외한 나머지 축들
        remaining_axes_indices = [i for i in range(3) if i != closest_axis_idx]
        
        # 나머지 두 축에 대해 레이캐스팅 수행하여 Y축 벡터 찾기
        y_axis_vector = None
        for axis_idx in remaining_axes_indices:
            axis_vector = principal_axes[axis_idx]
            
            # 단일 교차점 확인
            unit_vector = self._check_single_intersection(
                pv_mesh, centroid, axis_vector, axis_idx
            )
            
            if unit_vector is not None:
                y_axis_vector = unit_vector
                break
        
        # Y축 벡터를 찾지 못한 경우
        if y_axis_vector is None:
            print("[WARNING] 단일 교차점을 가진 축을 찾지 못했습니다. 첫 번째 남은 축을 Y축으로 사용합니다.")
            fallback_axis_idx = remaining_axes_indices[0]
            y_axis_vector = principal_axes[fallback_axis_idx]
            y_axis_norm = np.linalg.norm(y_axis_vector)
            if y_axis_norm < 1e-10:
                raise ValueError("Y축 벡터가 영벡터입니다.")
            y_axis_vector = y_axis_vector / y_axis_norm
        
        # Z축 벡터 계산
        z_axis_vector = self._compute_z_axis_vector(pv_mesh, closest_axis_vector)
        if z_axis_vector is None:
            raise ValueError("Z축 벡터 계산에 실패했습니다.")
        
        # X축 벡터 계산 (Y x Z)
        x_axis_vector = np.cross(y_axis_vector, z_axis_vector)
        x_axis_norm = np.linalg.norm(x_axis_vector)
        
        if x_axis_norm < 1e-10:
            raise ValueError(
                f"X축 벡터가 영벡터입니다. Y축과 Z축이 평행할 수 있습니다. "
                f"Y축: {y_axis_vector}, Z축: {z_axis_vector}"
            )
        
        # X축 정규화
        x_axis_vector = x_axis_vector / x_axis_norm
        
        print(f"[INFO] 로컬 좌표계 구성 완료:")
        print(f"   X축: {x_axis_vector}")
        print(f"   Y축: {y_axis_vector}")
        print(f"   Z축: {z_axis_vector}")
        
        return x_axis_vector, y_axis_vector, z_axis_vector



    def _create_pyvista_mesh(
        self, 
        vertices: np.ndarray, 
        faces: np.ndarray
    ) -> pv.PolyData:
        """
        정점과 면 정보로 PyVista 메시를 생성합니다.
        
        Args:
            vertices: 메시의 정점 배열 (N, 3)
            faces: 메시의 면 정보 (M, 3)
            
        Returns:
            PyVista PolyData 객체
        """
        # PyVista 형식으로 면 정보 변환 (각 면 앞에 정점 개수 추가)
        faces_pv = np.hstack([
            np.full((faces.shape[0], 1), 3), 
            faces
        ]).flatten()
        
        return pv.PolyData(vertices, faces_pv)
    
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
            pv_mesh: PyVista PolyData 객체
            centroid: 메시의 무게중심
            axis_vector: 확인할 축 방향 벡터
            axis_idx: 축 인덱스 (로깅용)
            
        Returns:
            단일 교차점이 있는 경우 해당 방향의 단위 벡터, 없으면 None
        """
        # +방향 레이캐스팅
        plus_intersections = self.ray_caster.ray_casting(
            pv_mesh, centroid.reshape(1, 3), axis_vector.reshape(1, 3)
        )
        plus_has_intersection = len(plus_intersections) > 0
        
        # -방향 레이캐스팅
        minus_intersections = self.ray_caster.ray_casting(
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


    def _find_zero_and_one_intersection_vector(
        self,
        pv_mesh: Any,
        centroid: np.ndarray,
        axis_vectors: list[np.ndarray],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        특정 축에 대해 단일 교차점 여부를 확인합니다.
        
        Args:
            pv_mesh: PyVista PolyData 객체
            centroid: 메시의 무게중심
            axis_vectors: 확인할 축 방향 벡터 리스트
            
        Returns:
            단일 교차점이 있는 경우 해당 방향의 단위 벡터, 없으면 None, None
        """
        zero_intersection_vector = None
        one_intersection_vector = None

        for axis_vector in axis_vectors:

            # +방향 레이캐스팅
            plus_intersections = self.ray_caster.ray_casting(
                pv_mesh, centroid.reshape(1, 3), axis_vector.reshape(1, 3)
            )
            plus_has_intersection = len(plus_intersections) > 0
            
            # -방향 레이캐스팅
            minus_intersections = self.ray_caster.ray_casting(
                pv_mesh, centroid.reshape(1, 3), (-axis_vector).reshape(1, 3)
            )
            minus_has_intersection = len(minus_intersections) > 0
            
            # 교차점 개수 계산
            intersection_count = int(plus_has_intersection) + int(minus_has_intersection)

            if intersection_count == 0:
                zero_intersection_vector = axis_vector
            elif intersection_count == 1:
                one_intersection_vector = axis_vector
            else:
                continue

        return zero_intersection_vector, one_intersection_vector


    
    def _get_closest_intersection_point(
        self,
        intersections: np.ndarray,
        centroid: np.ndarray
    ) -> np.ndarray:
        """
        교차점들 중 도심점에 가장 가까운 점을 반환합니다.
        
        Args:
            intersections: 교차점 배열 (N, 3)
            centroid: 도심점
            
        Returns:
            가장 가까운 교차점
        """
        distances = np.linalg.norm(intersections - centroid, axis=1)
        closest_idx = np.argmin(distances)
        return intersections[closest_idx]


    def _compute_z_axis_vector(
        self,
        pv_mesh: pv.PolyData,
        closest_axis_vector: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Z축 벡터를 계산합니다.
        
        메시의 평균 법선 벡터와 주축 벡터의 내적을 계산하여
        방향을 결정합니다. 두 벡터가 같은 방향이면 주축 벡터를,
        반대 방향이면 주축 벡터의 반대 방향을 반환합니다.
        
        Args:
            pv_mesh: PyVista PolyData 객체
            closest_axis_vector: PCA로 계산된 주축 벡터
            
        Returns:
            방향이 결정된 Z축 벡터, 실패 시 None
        """
        try:
            # PyVista PolyData에서 법선 벡터 가져오기
            # point_normals가 없으면 compute_normals()로 계산
            if pv_mesh.point_normals is None or len(pv_mesh.point_normals) == 0:
                pv_mesh = pv_mesh.compute_normals(point_normals=True, cell_normals=False)
            
            ios_normals = np.asarray(pv_mesh.point_normals)
            
            if ios_normals is None or len(ios_normals) == 0:
                print("[ERROR] 메시에서 법선 벡터를 계산할 수 없습니다.")
                return None
            
            ios_normals_mean = np.mean(ios_normals, axis=0)
            
            # 평균 법선 벡터가 영벡터인 경우 처리
            normals_mean_norm = np.linalg.norm(ios_normals_mean)
            if normals_mean_norm < 1e-10:
                print("[WARNING] 평균 법선 벡터가 영벡터입니다. closest_axis_vector를 그대로 반환합니다.")
                return closest_axis_vector
            
            print(f"[INFO] IOS mesh normals mean: {ios_normals_mean}")
            
            # 내적으로 방향 확인
            inner_product = np.dot(closest_axis_vector, ios_normals_mean)
            if inner_product > 0:
                print("[INFO] Same direction")
                return closest_axis_vector
            else:
                print("[INFO] Opposite direction")
                return -closest_axis_vector
                
        except Exception as e:
            print(f"[ERROR] Z축 벡터 계산 중 오류 발생: {e}")
            return None

