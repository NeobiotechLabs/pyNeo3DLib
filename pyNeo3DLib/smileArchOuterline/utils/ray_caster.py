"""
레이 캐스팅 관련 기능을 담당하는 클래스
"""

import numpy as np
import pyvista as pv
from typing import List
from .constants import AnalysisConstants


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
    
    def rotate_vector_around_axis(self, vector: np.ndarray, axis: np.ndarray, angle_degrees: float) -> np.ndarray:
        """
        주어진 축을 중심으로 벡터를 회전시킵니다.
        
        Args:
            vector: 회전시킬 벡터 (numpy array, shape: (3,))
            axis: 회전축 벡터 (numpy array, shape: (3,))
            angle_degrees: 회전 각도 (도 단위)
        
        Returns:
            rotated_vector: 회전된 벡터 (numpy array, shape: (3,))
        """
        # 입력 검증
        if axis is None:
            raise ValueError("회전축이 None입니다.")
        
        # axis를 1차원 배열로 변환
        axis = np.asarray(axis).flatten()
        
        if len(axis) != 3:
            raise ValueError(f"회전축은 3차원 벡터여야 합니다. 현재 형태: {axis.shape}")
        
        # vector를 1차원 배열로 변환
        vector = np.asarray(vector).flatten()
        
        if len(vector) != 3:
            raise ValueError(f"벡터는 3차원 벡터여야 합니다. 현재 형태: {vector.shape}")
        
        # 각도를 라디안으로 변환
        angle_rad = np.radians(angle_degrees)
        
        # 회전축 정규화
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            raise ValueError("회전축의 크기가 0입니다.")
        
        axis = axis / axis_norm
        
        # 로드리게스 회전 공식 사용
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # 외적 행렬 (skew-symmetric matrix)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # 회전 행렬 계산: R = I + sin(θ)K + (1-cos(θ))K²
        I = np.eye(3)
        R = I + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        # 벡터 회전
        return np.dot(R, vector)






    def ray_casting_by_point_cloud(self, mesh, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        포인트 클라우드 버텍스를 이용한 레이 캐스팅 함수
        
        1. 원점에서 레이 방향으로 가장 가까운 점 추출
        2. 그 방향으로 조금 더 들어가서 원점 갱신
        3. 새로운 원점에서 나가는 방향(반대)으로 가장 가까운 점 찾기
        4. 결과적으로 두 개의 점 샘플링
        
        Args:
            mesh: PyVista PolyData 객체
            origin: 레이 시작점 (numpy array, shape: (1,3))
            direction: 레이 방향 벡터 (numpy array, shape: (1,3))
        
        Returns:
            교차점 좌표 (numpy array, shape: (N, 3)) - 최대 2개의 점
        """
        # 배열 형태 정규화
        origin_flat = origin.flatten()
        direction_flat = direction.flatten()
        
        # 방향 벡터 정규화
        direction_norm = direction_flat / np.linalg.norm(direction_flat)
        
        # 메시의 모든 버텍스 가져오기
        vertices = mesh.points
        
        # 첫 번째 점 찾기: 원점에서 레이 방향으로 가장 가까운 점
        first_point = self._find_closest_point_on_ray(
            vertices, origin_flat, direction_norm, forward=True
        )
        
        if first_point is None:
            return np.array([]).reshape(0, 3)
        
        # 원점 갱신: 첫 번째 점에서 방향으로 조금 더 들어감
        penetration_distance = 1.0  # 침투 거리 (mm 단위, 필요시 조정 가능)
        new_origin = first_point + direction_norm * penetration_distance
        
        # 두 번째 점 찾기: 새 원점에서 반대 방향으로 가장 가까운 점
        second_point = self._find_closest_point_on_ray(
            vertices, new_origin, -direction_norm, forward=True
        )
        
        # 결과 반환
        result_points = [first_point]
        if second_point is not None:
            result_points.append(second_point)
        
        return np.array(result_points)
    
    def _find_closest_point_on_ray(
        self, 
        vertices: np.ndarray, 
        origin: np.ndarray, 
        direction: np.ndarray, 
        forward: bool = True,
        distance_threshold: float = 5.0
    ) -> np.ndarray:
        """
        레이에 가장 가까운 버텍스를 찾습니다.
        
        Args:
            vertices: 메시의 모든 버텍스 (N, 3)
            origin: 레이 시작점 (3,)
            direction: 레이 방향 벡터 (정규화됨) (3,)
            forward: True면 방향으로, False면 반대 방향의 점만 고려
            distance_threshold: 레이로부터의 최대 허용 거리 (mm)
        
        Returns:
            가장 가까운 점의 좌표 (3,) 또는 None
        """
        # 각 버텍스에서 원점까지의 벡터
        to_vertices = vertices - origin  # (N, 3)
        
        # 레이 방향으로의 투영 거리 (스칼라 투영)
        projections = np.dot(to_vertices, direction)  # (N,)
        
        # forward=True인 경우 양의 투영만, False인 경우 모든 점 고려
        if forward:
            valid_mask = projections > 0
        else:
            valid_mask = np.ones(len(projections), dtype=bool)
        
        if not np.any(valid_mask):
            return None
        
        # 레이 위의 가장 가까운 점 계산
        projection_points = origin + projections[:, np.newaxis] * direction  # (N, 3)
        
        # 각 버텍스와 레이 사이의 거리 계산
        distances_to_ray = np.linalg.norm(vertices - projection_points, axis=1)  # (N,)
        
        # 거리 임계값 내에 있는 점들만 필터링
        valid_mask &= (distances_to_ray < distance_threshold)
        
        if not np.any(valid_mask):
            return None
        
        # 유효한 점들 중에서 원점에서 가장 가까운 점 찾기
        valid_indices = np.where(valid_mask)[0]
        distances_from_origin = np.linalg.norm(to_vertices[valid_indices], axis=1)
        closest_idx = valid_indices[np.argmin(distances_from_origin)]
        
        return vertices[closest_idx]




















    
    def perform_360_degree_ray_casting(
        self,
        input_mesh, 
        ray_origin: np.ndarray, 
        rotation_axis: np.ndarray, 
        initial_direction: np.ndarray, 
        angle_step: float = 5.0
    ) -> np.ndarray:
        """
        360도 회전 레이캐스팅을 수행하여 각 각도에서 레이 원점과 가장 가까운 점 2개를 선택합니다.
        
        Args:
            input_mesh: PyVista PolyData 객체
            ray_origin: 레이 시작점 (numpy array, shape: (3,))
            rotation_axis: 회전축 벡터 (numpy array, shape: (3,))
            initial_direction: 초기 레이 방향 벡터 (numpy array, shape: (3,))
            angle_step: 각도 간격 (도 단위, 기본값: 5.0)
        
        Returns:
            selected_points: 각 각도에서 선택된 가장 가까운 2개 포인트들 (numpy array, shape: (N, 3))
        """
        selected_points = []
        
        # 0도부터 360도까지 각도 간격으로 회전
        for angle in np.arange(0, 360, angle_step):
            # 현재 각도에서의 레이 방향 계산
            current_direction = self.rotate_vector_around_axis(initial_direction, rotation_axis, angle)
            
            # 레이 캐스팅 수행
            plus_direction = current_direction.reshape(1, 3)
            plus_points = self.ray_casting_by_point_cloud(input_mesh, ray_origin.reshape(1, 3), plus_direction)
            
            # 교차점이 2개 이상인 경우, 레이 원점과 가장 가까운 2개 선택
            if len(plus_points) >= 2:
                # 각 교차점과 레이 원점 사이의 거리 계산
                distances = np.linalg.norm(plus_points - ray_origin, axis=1)
                
                # 거리가 가장 가까운 2개 포인트의 인덱스 찾기
                closest_indices = np.argsort(distances)[:2]
                
                # 가장 가까운 2개 포인트 선택
                closest_points = plus_points[closest_indices]

                selected_points.append(closest_points)
        
        # 선택된 교차점들을 하나의 배열로 합치기
        if len(selected_points) > 0:
            return np.concatenate(selected_points, axis=0)
        else:
            return np.array([]).reshape(0, 3)
    
    def perform_height_based_ray_casting(self, aligned_mesh, rotation_axis, num_slices=5, angle_step=5):
        """
        높이별로 레이캐스팅하여 등고선 포인트 클라우드 추출
        
        Args:
            aligned_mesh: 정렬된 메쉬
            rotation_axis: 회전축
            num_slices: 슬라이스 개수
            angle_step: 각도 간격
        Returns:
            result_points_array: 등고선 포인트 클라우드 배열
        """
        # rotation_axis가 None인 경우 기본값 사용
        if rotation_axis is None:
            print("경고: rotation_axis가 None입니다. 기본 Y축을 사용합니다.")
            rotation_axis = np.array([0, 1, 0])
        
        # rotation_axis를 1차원 배열로 변환
        rotation_axis = np.asarray(rotation_axis).flatten()
        
        if len(rotation_axis) != 3:
            raise ValueError(f"rotation_axis는 3차원 벡터여야 합니다. 현재 형태: {rotation_axis.shape}")
        
        start_ray_origin = np.array([0, min(aligned_mesh.points[:, 1]), 0])
        end_ray_origin = np.array([0, max(aligned_mesh.points[:, 1]), 0])
        height_step = (end_ray_origin - start_ray_origin) / num_slices  
        direction = np.array([0, 1, 0])
        initial_direction = np.array([0, 0, -1])
        
        result_points_array = np.array([]).reshape(0, 3)
            
        for i in range(num_slices):
            stepped_start_ray_origin = start_ray_origin + height_step * i * direction
            
            ray_casting_results = self.perform_360_degree_ray_casting(
                aligned_mesh, stepped_start_ray_origin, rotation_axis, initial_direction, angle_step=angle_step
            )
            ray_casting_results_array = np.array(ray_casting_results).reshape(-1, 3)
            result_points_array = np.concatenate([result_points_array, ray_casting_results_array], axis=0)
        
        return result_points_array
