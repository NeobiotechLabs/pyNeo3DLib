import numpy as np
from typing import Optional, Tuple, Literal
from ..general_utils.constants import AnalysisConstants


class PolarSampling:
    def __init__(self, center: np.ndarray):
        self.center = center

    def polar_sampling(
        self,
        points: np.ndarray,
        angle_step: float = AnalysisConstants.POLAR_SAMPLER_DEFAULT_ANGLE_STEP,
        y_range: Optional[Tuple[float, float]] = AnalysisConstants.POLAR_SAMPLER_DEFAULT_Y_RANGE,
        start_angle: float = AnalysisConstants.POLAR_SAMPLER_DEFAULT_START_ANGLE,
        end_angle : float = AnalysisConstants.POLAR_SAMPLER_DEFAULT_END_ANGLE,
        mode: Literal["ymax", "ymin", "farthest", "nearest"] = AnalysisConstants.POLAR_SAMPLING_MODE_YMAX,
    ) -> np.ndarray:
        """
        극좌표계 기반으로 포인트 클라우드를 샘플링합니다.
        각도 구간별로 선택 기준(mode)에 따라 하나의 포인트를 선택합니다.

        Args:
            points (np.ndarray): (N, 3) 입력 포인트 클라우드
            angle_step (float): 샘플링 각도 간격(도). 기본 1
            y_range (Tuple[float, float] | None): y축 필터링 범위. None이면 [-5, 5]
            start_angle (float): 시작 각도(도). 기본 -90
            mode (Literal): 선택 기준
                - "ymax": y가 가장 큰 포인트
                - "ymin": y가 가장 작은 포인트
                - "farthest": 중심점으로부터 가장 먼 포인트
                - "nearest": 중심점으로부터 가장 가까운 포인트

        Returns:
            np.ndarray: (M, 3) 샘플링 포인트(각도 순 정렬)
        """
        if y_range is None:
            y_range = AnalysisConstants.POLAR_SAMPLER_DEFAULT_Y_RANGE

        if mode not in {AnalysisConstants.POLAR_SAMPLING_MODE_YMAX, AnalysisConstants.POLAR_SAMPLING_MODE_YMIN_POLAR, AnalysisConstants.POLAR_SAMPLING_MODE_FARTHEST_POLAR, AnalysisConstants.POLAR_SAMPLING_MODE_NEAREST_POLAR}:
            raise ValueError("mode는 'ymax', 'ymin', 'farthest', 'nearest' 중 하나여야 합니다.")

        # y 범위 필터링
        filtered_points = points[
            (points[:, AnalysisConstants.Y_AXIS_INDEX] >= y_range[0]) &
            (points[:, AnalysisConstants.Y_AXIS_INDEX] <= y_range[1])
        ]
        if filtered_points.size == 0:
            return np.empty((0, AnalysisConstants.VECTOR_DIMENSION), dtype=points.dtype)

        # 중심 기준 좌표(각도/거리 계산용)
        centered_points = filtered_points - self.center

        # x-z 평면에서 방위각(phi) [0, 2π)
        point_phis = np.arctan2(centered_points[:, AnalysisConstants.Z_AXIS_INDEX], centered_points[:, AnalysisConstants.X_AXIS_INDEX])
        point_phis[point_phis < 0] += AnalysisConstants.TWO_PI

        sampled_points = []
        sampled_angles = []

        # 각도 구간별 선택
        for phi in np.arange(start_angle, end_angle, angle_step):
            phi_rad = np.radians(phi)
            phi_range = np.radians(angle_step)

            # 현재 구간에 속하는 포인트
            angle_diff = np.abs(point_phis - phi_rad)
            mask = (angle_diff <= phi_range) | (np.abs(angle_diff - AnalysisConstants.TWO_PI) <= phi_range)
            points_in_range = filtered_points[mask]

            if points_in_range.size == 0:
                continue

            if mode == AnalysisConstants.POLAR_SAMPLING_MODE_YMAX:
                idx = np.argmax(points_in_range[:, AnalysisConstants.Y_AXIS_INDEX])
            elif mode == AnalysisConstants.POLAR_SAMPLING_MODE_YMIN_POLAR:
                idx = np.argmin(points_in_range[:, AnalysisConstants.Y_AXIS_INDEX])
            else:
                # 거리 기반 모드: 중심으로부터의 유클리드 거리

                deltas = points_in_range - self.center
                dists = np.sqrt(np.sum(deltas ** 2, axis=1))
                idx = np.argmax(dists) if mode == AnalysisConstants.POLAR_SAMPLING_MODE_FARTHEST_POLAR else np.argmin(dists) # mode == AnalysisConstants.POLAR_SAMPLING_MODE_NEAREST_POLAR

            sampled_points.append(points_in_range[idx])
            sampled_angles.append(phi)

        if not sampled_points:
            return np.empty((0, AnalysisConstants.VECTOR_DIMENSION), dtype=points.dtype)

        # 각도 순 정렬
        order = np.argsort(sampled_angles)
        sampled_points = np.asarray(sampled_points)[order]

        return sampled_points