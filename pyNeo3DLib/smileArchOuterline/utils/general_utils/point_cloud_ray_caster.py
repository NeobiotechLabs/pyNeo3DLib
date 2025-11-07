import numpy as np
import pyvista as pv
from typing import List, Optional
from .constants import AnalysisConstants
from .vector_utils import VectorUtils


class PointCloudRayCaster:
    """포인트 클라우드 버텍스를 이용한 레이 캐스팅을 수행하는 클래스"""

    def __init__(self):
        self.ray_length = AnalysisConstants.RAY_LENGTH
        self.ray_scale_factor = AnalysisConstants.RAY_SCALE_FACTOR

    def ray_casting_by_point_cloud(self, vertices: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        포인트 클라우드 버텍스를 이용한 레이 캐스팅 함수

        1. 원점에서 레이 방향으로 가장 가까운 점 추출

        Args:
            vertices: 메시의 모든 버텍스 (N, 3)
            origin: 레이 시작점 (numpy array, shape: (1,3))
            direction: 레이 방향 벡터 (numpy array, shape: (1,3))

        Returns:
            교차점 좌표 (numpy array, shape: (N, 3)) - 최대 1개의 점
        """
        # 배열 형태 정규화
        origin_flat = origin.flatten()
        direction_flat = direction.flatten()

        # 방향 벡터 정규화
        direction_norm = direction_flat / np.linalg.norm(direction_flat)
        # 첫 번째 점 찾기: 원점에서 레이 방향으로 가장 가까운 점
        first_point = self._find_closest_point_on_ray(
            vertices, origin_flat, direction_norm, forward=True
        )

        if first_point is None:
            return np.array([]).reshape(0, AnalysisConstants.VECTOR_DIMENSION)

        # 결과 반환
        result_points = [first_point]


        return np.array(result_points)

    def _find_closest_point_on_ray(
        self,
        vertices: np.ndarray,
        origin: np.ndarray,
        direction: np.ndarray,
        forward: bool = True,
        distance_threshold: float = AnalysisConstants.RAY_DISTANCE_THRESHOLD
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
            valid_mask = np.ones(len(projections), dtype=AnalysisConstants.NUMPY_ONES_DTYPE_BOOL)

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
        vertices: np.ndarray,
        ray_origin: np.ndarray,
        rotation_axis: np.ndarray,
        initial_direction: np.ndarray,
        angle_step: float = AnalysisConstants.DEFAULT_ANGLE_STEP_RAY_CASTER
    ) -> np.ndarray:
        """
        360도 회전 레이캐스팅을 수행하여 각 각도에서 레이 원점과 가장 가까운 점 1개를 선택합니다.

        Args:
            vertices: 메시의 모든 버텍스 (N, 3)
            ray_origin: 레이 시작점 (numpy array, shape: (3,))
            rotation_axis: 회전축 벡터 (numpy array, shape: (3,))
            initial_direction: 초기 레이 방향 벡터 (numpy array, shape: (3,))
            angle_step: 각도 간격 (도 단위, 기본값: 5.0)

        Returns:
            selected_points: 각 각도에서 선택된 가장 가까운 1개 포인트들 (numpy array, shape: (N, 3))
        """
        selected_points = []

        # 0도부터 360도까지 각도 간격으로 회전
        for angle in np.arange(AnalysisConstants.ANGLE_360_START, AnalysisConstants.ANGLE_360_END, angle_step):
            # 현재 각도에서의 레이 방향 계산
            current_direction = VectorUtils.rotate_vector_around_axis(initial_direction, rotation_axis, angle)

            # 레이 캐스팅 수행
            plus_direction = current_direction.reshape(AnalysisConstants.SINGLE_ROW_SHAPE, AnalysisConstants.VECTOR_DIMENSION)
            plus_points = self.ray_casting_by_point_cloud(vertices, ray_origin.reshape(AnalysisConstants.SINGLE_ROW_SHAPE, AnalysisConstants.VECTOR_DIMENSION), plus_direction)

            # 교차점이 1개 이상인 경우, 레이 원점과 가장 가까운 1개 선택
            if len(plus_points) >= AnalysisConstants.MIN_POINTS_FOR_RAY_CASTING:
                # 각 교차점과 레이 원점 사이의 거리 계산
                distances = np.linalg.norm(plus_points - ray_origin, axis=1)

                # 거리가 가장 가까운 1개 포인트의 인덱스 찾기
                closest_index = np.argsort(distances)[:AnalysisConstants.MIN_POINTS_FOR_RAY_CASTING]

                # 가장 가까운 1개 포인트 선택
                closest_point = plus_points[closest_index]

                selected_points.append(closest_point)

        # 선택된 교차점들을 하나의 배열로 합치기
        if len(selected_points) > 0:
            return np.concatenate(selected_points, axis=0)
        else:
            return np.array([]).reshape(0, AnalysisConstants.VECTOR_DIMENSION)

    def perform_height_based_ray_casting(self, vertices: np.ndarray, rotation_axis, num_slices=AnalysisConstants.DEFAULT_NUM_SLICES_RAY_CASTER, angle_step=AnalysisConstants.DEFAULT_ANGLE_STEP_RAY_CASTER):
        """
        높이별로 레이캐스팅하여 등고선 포인트 클라우드 추출

        Args:
            vertices: 메시의 모든 버텍스 (N, 3)
            rotation_axis: 회전축
            num_slices: 슬라이스 개수
            angle_step: 각도 간격
        Returns:
            result_points_array: 등고선 포인트 클라우드 배열
        """
        # rotation_axis가 None인 경우 기본값 사용
        if rotation_axis is None:
            print("경고: rotation_axis가 None입니다. 기본 Y축을 사용합니다.")
            rotation_axis = np.array(AnalysisConstants.Y_AXIS_VECTOR)

        # rotation_axis를 1차원 배열로 변환
        rotation_axis = np.asarray(rotation_axis).flatten()

        if len(rotation_axis) != AnalysisConstants.VECTOR_DIMENSION:
            raise ValueError(f"rotation_axis는 {AnalysisConstants.VECTOR_DIMENSION}차원 벡터여야 합니다. 현재 형태: {rotation_axis.shape}")

        # min_y, max_y를 rotation_axis를 기준으로 계산
        projected_vertices = np.dot(vertices, rotation_axis)
        min_proj = np.min(projected_vertices)
        max_proj = np.max(projected_vertices)

        # 높이 슬라이스 계산
        height_step_val = (max_proj - min_proj) / num_slices

        result_points_array = np.array([]).reshape(0, AnalysisConstants.VECTOR_DIMENSION)
        initial_direction = np.array(AnalysisConstants.Z_AXIS_VECTOR_NEGATIVE) # 초기 방향 설정 (예시, 실제 데이터에 따라 조정 필요)

        for i in range(num_slices):
            # 현재 슬라이스 높이에 해당하는 평면 정의
            current_proj_val = min_proj + height_step_val * i

            # 슬라이스 평면과 가까운 버텍스 필터링
            # 여기서는 단순히 특정 높이 범위 내의 점을 선택하는 대신,
            # 해당 평면을 지나가는 레이를 쏘기 위한 원점을 설정합니다.
            # 레이 원점은 회전축에 수직인 평면 상에 있어야 하며, 이 평면의 높이는 current_proj_val에 해당합니다.

            # 간단화를 위해, 슬라이스 평면의 중심점을 레이 원점으로 가정합니다.
            # 실제 사용 시에는 메시의 중심점과 rotation_axis를 활용하여 더 정확한 레이 원점을 계산해야 합니다.
            # 여기서는 예시를 위해 메시의 바운딩 박스 중심을 사용하거나,
            # 아니면 initial_direction과 rotation_axis에 수직인 다른 벡터를 활용하여
            # 해당 높이의 적절한 시작점을 찾을 수 있습니다.

            # 임시 레이 원점 설정: 메시의 평균 x, z 값에 현재 높이 값을 부여
            # 이 부분은 실제 데이터 및 요구사항에 따라 크게 달라질 수 있습니다.
            mean_x = np.mean(vertices[:, 0])
            mean_z = np.mean(vertices[:, 2])
            # current_ray_origin = np.array([mean_x, current_y, mean_z])

            # rotation_axis에 따라 레이 원점을 설정하는 로직
            # 예: Y축이 회전축인 경우 XZ 평면을 따라 이동
            # X축이 회전축인 경우 YZ 평면을 따라 이동 등.

            # 일단 간단하게 ray_origin을 특정 높이에 위치시키기 위해 임의의 점을 사용합니다.
            # 이 부분은 `arch_curve_finder.py`에서 `perform_height_based_ray_casting`이 어떻게 호출되고
            # 어떤 `ray_origin`이 필요한지에 따라 달라질 수 있습니다.

            # 여기서는 예시를 위해 메시의 바운딩 박스 중심을 기준으로 높이를 조절하여 레이 원점을 설정합니다.
            bbox_center = np.mean(vertices, axis=0)
            # 회전축 방향으로 current_proj_val만큼 이동한 점을 레이 원점으로 사용
            ray_origin_for_slice = bbox_center + (current_proj_val - np.dot(bbox_center, rotation_axis)) * rotation_axis

            ray_casting_results = self.perform_360_degree_ray_casting(
                vertices, ray_origin_for_slice, rotation_axis, initial_direction, angle_step=angle_step
            )
            ray_casting_results_array = np.array(ray_casting_results).reshape(AnalysisConstants.LAST_INDEX, AnalysisConstants.VECTOR_DIMENSION)
            result_points_array = np.concatenate([result_points_array, ray_casting_results_array], axis=0)


        return result_points_array
