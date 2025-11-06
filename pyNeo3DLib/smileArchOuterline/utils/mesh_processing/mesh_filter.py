"""
메시 필터링을 담당하는 클래스
단일책임원칙에 따라 메시의 필터링만을 담당합니다.
"""

import numpy as np
from typing import Tuple
import pyvista as pv


class MeshFilter:
    """메시의 필터링을 담당하는 클래스"""
    
    def __init__(self):
        """MeshFilter 초기화"""
        pass
    
    def filter_points_by_z_threshold(
        self, 
        points: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        z값 기준으로 점들을 필터링합니다.
        첫 번째와 마지막 점의 z값 절댓값을 비교하여 더 작은 값을 임계값으로 사용합니다.
        
        Args:
            points: 필터링할 점들 (N, 3)
            
        Returns:
            Tuple[filtered_points, threshold]:
                - filtered_points: 필터링된 점들
                - threshold: 사용된 z값 임계값
        """
        if len(points) == 0:
            return points, 0.0
        
        # 첫 번째와 마지막 점의 z값 절댓값 계산
        first_point_z = abs(points[0, 2])
        last_point_z = abs(points[-1, 2])
        
        # 더 작은 값을 임계값으로 선택
        threshold = min(first_point_z, last_point_z)
        
        # 임계값보다 큰 z값을 가지는 점들만 필터링
        filtered_points = points[points[:, 2] > -threshold]
        
        
        return filtered_points, threshold
    
    def filter_points_by_z_value(
        self, 
        points: np.ndarray, 
        z_threshold: float
    ) -> np.ndarray:
        """
        주어진 z값 임계값으로 점들을 필터링합니다.
        
        Args:
            points: 필터링할 점들 (N, 3)
            z_threshold: z값 임계값
            
        Returns:
            np.ndarray: 필터링된 점들
        """
        return points[points[:, 2] > -z_threshold]
    
    def filter_points_by_z_range(
        self, 
        points: np.ndarray, 
        z_min: float, 
        z_max: float
    ) -> np.ndarray:
        """
        z값 범위로 점들을 필터링합니다.
        
        Args:
            points: 필터링할 점들 (N, 3)
            z_min: 최소 z값
            z_max: 최대 z값
            
        Returns:
            np.ndarray: 필터링된 점들
        """
        return points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
    
    def remove_outliers_by_z_std(
        self, 
        points: np.ndarray, 
        std_threshold: float = 2.0
    ) -> np.ndarray:
        """
        z값의 표준편차를 기준으로 아웃라이어를 제거합니다.
        
        Args:
            points: 필터링할 점들 (N, 3)
            std_threshold: 표준편차 임계값 (기본값: 2.0)
            
        Returns:
            np.ndarray: 아웃라이어가 제거된 점들
        """
        if len(points) == 0:
            return points
        
        z_values = points[:, 2]
        z_mean = np.mean(z_values)
        z_std = np.std(z_values)
        
        # 표준편차 임계값 범위 내의 점들만 유지
        mask = np.abs(z_values - z_mean) <= std_threshold * z_std
        
        return points[mask]

    def filter_mesh_by_z_threshold(
        self, 
        aligned_mesh: pv.PolyData, 
        filtered_points: np.ndarray
    ) -> pv.PolyData:
        """
        Z 임계값을 기준으로 메쉬 필터링
        
        Args:
            aligned_mesh: 정렬된 메쉬
            filtered_points: 필터링된 포인트들 (N, 3)
            
        Returns:
            filtered_aligned_mesh: 필터링된 메쉬
        """
        if len(filtered_points) == 0:
            raise ValueError("filtered_points가 비어있습니다.")
        
        if filtered_points.shape[1] != 3:
            raise ValueError(f"filtered_points는 (N, 3) 형태여야 합니다. 현재 shape: {filtered_points.shape}")
        
        z_min_point = np.min(filtered_points[:, 2])
        mask = aligned_mesh.points[:, 2] > z_min_point
        filtered_aligned_mesh = aligned_mesh.extract_points(mask)

        # 가장 큰 덩어리만 추출
        largest_component = filtered_aligned_mesh.extract_largest()

        # 중복된 면이나 점이 있는 경우 제거
        filtered_largest_component = largest_component.clean()
        
        return filtered_largest_component
    
    def filter_points_by_distance_from_center(
        self, 
        points: np.ndarray, 
        center: np.ndarray = None,
        distance_threshold: float = None,
        percentile: float = 95.0
    ) -> np.ndarray:
        """
        중심점에서의 거리를 기준으로 점들을 필터링합니다.
        
        Args:
            points: 필터링할 점들 (N, 3)
            center: 중심점 (기본값: None, 자동 계산)
            distance_threshold: 거리 임계값 (기본값: None, 자동 계산)
            percentile: 거리 임계값 계산에 사용할 백분위수 (기본값: 95.0)
            
        Returns:
            np.ndarray: 필터링된 점들
        """
        if len(points) == 0:
            return points
        
        # 중심점 계산
        if center is None:
            center = np.mean(points, axis=0)
        
        # 거리 계산
        distances = np.linalg.norm(points - center, axis=1)
        
        # 임계값 계산
        if distance_threshold is None:
            distance_threshold = np.percentile(distances, percentile)
        
        # 임계값보다 가까운 점들만 유지
        mask = distances <= distance_threshold
        
        return points[mask]
    
    def filter_points_by_volume_density(
        self, 
        points: np.ndarray, 
        grid_size: float = 1.0,
        min_density: int = 5
    ) -> np.ndarray:
        """
        볼륨 밀도를 기준으로 점들을 필터링합니다.
        공간을 그리드로 나누고 각 그리드 셀의 점 밀도를 계산하여
        밀도가 낮은 영역의 점들을 제거합니다.
        
        Args:
            points: 필터링할 점들 (N, 3)
            grid_size: 그리드 셀 크기 (기본값: 1.0)
            min_density: 최소 밀도 임계값 (기본값: 5)
            
        Returns:
            np.ndarray: 필터링된 점들
        """
        if len(points) == 0:
            return points
        
        # 그리드 인덱스 계산
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # 그리드 크기 조정
        grid_indices = ((points - min_coords) / grid_size).astype(int)
        
        # 각 그리드 셀의 점 개수 계산
        unique_indices, counts = np.unique(grid_indices, axis=0, return_counts=True)
        
        # 밀도가 충분한 그리드 셀만 유지
        valid_indices = unique_indices[counts >= min_density]
        
        # 유효한 그리드 셀에 속하는 점들만 필터링
        mask = np.zeros(len(points), dtype=bool)
        for valid_idx in valid_indices:
            cell_mask = np.all(grid_indices == valid_idx, axis=1)
            mask |= cell_mask
        
        return points[mask]