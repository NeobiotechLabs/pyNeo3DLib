"""
레이캐스팅을 위한 격자 생성 모듈

격자 평면 생성 및 레이 시작점 계산을 담당합니다.
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np


class RayGridGenerator:
    """
    레이캐스팅을 위한 격자 생성 클래스
    
    주요 책임:
    - 레이 발사를 위한 격자 평면 생성
    - 격자 좌표계 설정 (u, v 축)
    - 격자점 3D 좌표 계산
    
    사용 예제:
    ```python
    generator = RayGridGenerator(
        grid_center=[77.7, 85.0, 94.23],
        grid_width_mm=80.0,
        grid_height_mm=100.0,
        grid_resolution=(50, 50),
        ray_direction=[0, -1, 0],
        ray_start_offset_mm=150.0,
    )
    
    grid_points = generator.generate_grid_points()  # (H, W, 3)
    ```
    """
    
    def __init__(
        self,
        grid_center: np.ndarray | list,
        grid_width_mm: float,
        grid_height_mm: float,
        grid_resolution: Tuple[int, int],
        ray_direction: np.ndarray | list,
        ray_start_offset_mm: float,
    ):
        """
        Parameters:
        -----------
        grid_center : np.ndarray | list
            격자 평면 중심 위치 [x, y, z] (mm)
            
        grid_width_mm : float
            격자 가로 폭 (mm)
            
        grid_height_mm : float
            격자 세로 폭 (mm)
            
        grid_resolution : Tuple[int, int]
            격자 해상도 (가로, 세로)
            
        ray_direction : np.ndarray | list
            레이 발사 방향 [x, y, z]
            
        ray_start_offset_mm : float
            격자 평면을 center에서 ray_direction 반대로 얼마나 뒤로 배치할지 (mm)
            
        """
        self.grid_center = np.array(grid_center, dtype=np.float64)
        self.grid_width_mm = grid_width_mm
        self.grid_height_mm = grid_height_mm
        self.grid_resolution = grid_resolution
        self.ray_direction = np.array(ray_direction, dtype=np.float64)
        self.ray_start_offset_mm = ray_start_offset_mm
        
        # 레이 방향 정규화
        self.ray_dir_normalized = self.ray_direction / np.linalg.norm(self.ray_direction)
        
        # 격자 축 계산
        self.u_axis, self.v_axis = np.array([1, 0, 0], dtype=np.float64), np.array([0, 0, 1], dtype=np.float64)
        
        # 격자 평면 중심 계산 (포인트클라우드 뒤쪽에 배치)
        self.grid_plane_center = self.grid_center - self.ray_dir_normalized * self.ray_start_offset_mm
    
    
    def generate_grid_points(self) -> np.ndarray:
        """
        격자점 3D 좌표 생성
        
        Returns:
        --------
        np.ndarray
            격자점 좌표 (H, W, 3)
        """
        grid_w, grid_h = self.grid_resolution
        
        # u, v 좌표 생성
        u_coords = np.linspace(-self.grid_width_mm/2, self.grid_width_mm/2, grid_w)
        v_coords = np.linspace(-self.grid_height_mm/2, self.grid_height_mm/2, grid_h)
        
        # 격자점 계산
        grid_points = np.zeros((grid_h, grid_w, 3), dtype=np.float64)
        for i, v in enumerate(v_coords):
            for j, u in enumerate(u_coords):
                grid_points[i, j] = (
                    self.grid_plane_center + 
                    u * self.u_axis + 
                    v * self.v_axis
                )
        
        return grid_points
    
    def get_ray_direction(self) -> np.ndarray:
        """정규화된 레이 방향 벡터 반환"""
        return self.ray_dir_normalized.copy()
    
    def get_grid_info(self) -> dict:
        """격자 정보 반환"""
        return {
            "grid_center": self.grid_center.copy(),
            "grid_plane_center": self.grid_plane_center.copy(),
            "grid_width_mm": self.grid_width_mm,
            "grid_height_mm": self.grid_height_mm,
            "grid_resolution": self.grid_resolution,
            "ray_direction": self.ray_dir_normalized.copy(),
            "u_axis": self.u_axis.copy(),
            "v_axis": self.v_axis.copy(),
        }


