"""
깊이 맵 시각화 모듈

레이캐스팅 결과를 3D로 시각화하는 기능을 담당합니다.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import open3d as o3d


class DepthMapVisualizer:
    """
    깊이 맵 시각화 클래스
    
    주요 책임:
    - 교차점 3D 시각화
    - 원본 포인트클라우드와 비교 시각화
    - 격자점 시각화
    
    사용 예제:
    ```python
    visualizer = DepthMapVisualizer(
        hit_points_array=hit_points_array,
        original_cloud=pts_face,
        grid_center=center,
    )
    
    visualizer.visualize_3d(
        show_original_cloud=True,
        show_center_point=True,
    )
    ```
    """
    
    def __init__(
        self,
        hit_points_array: np.ndarray,
        original_cloud: Optional[np.ndarray] = None,
        grid_points: Optional[np.ndarray] = None,
        grid_center: Optional[np.ndarray] = None,
    ):
        """
        Parameters:
        -----------
        hit_points_array : np.ndarray
            교차점 배열 (N, 3)
            
        original_cloud : Optional[np.ndarray]
            원본 포인트클라우드 (M, 3)
            
        grid_points : Optional[np.ndarray]
            격자점 좌표 (H, W, 3)
            
        grid_center : Optional[np.ndarray]
            격자 중심 좌표 [x, y, z]
        """
        self.hit_points_array = np.array(hit_points_array, dtype=np.float64)
        self.original_cloud = np.array(original_cloud, dtype=np.float64) if original_cloud is not None else None
        self.grid_points = np.array(grid_points, dtype=np.float64) if grid_points is not None else None
        self.grid_center = np.array(grid_center, dtype=np.float64) if grid_center is not None else None
        
        if self.hit_points_array.ndim != 2 or self.hit_points_array.shape[1] != 3:
            raise ValueError(f"hit_points_array는 (N, 3) 형태여야 합니다. 현재: {self.hit_points_array.shape}")
    
    def visualize_3d(
        self,
        show_original_cloud: bool = True,
        show_grid_points: bool = False,
        show_center_point: bool = True,
        window_name: str = "레이캐스팅 교차점",
        hit_point_size: float = 5.0,
        original_point_size: float = 1.0,
        background_color: tuple = (0.1, 0.1, 0.1),
        use_standard_coords: bool = True,
    ) -> None:
        """
        3D 시각화
        
        Parameters:
        -----------
        show_original_cloud : bool
            원본 포인트클라우드 표시 여부
            
        show_grid_points : bool
            격자점 표시 여부
            
        show_center_point : bool
            중심점 표시 여부 (녹색 구체)
            
        window_name : str
            창 이름
            
        hit_point_size : float
            교차점 크기
            
        original_point_size : float
            원본 포인트클라우드 크기
            
        background_color : tuple
            배경색 RGB (0~1 범위)
            
        use_standard_coords : bool
            True: 일반 오른손 좌표계 (X=Right, Y=Up, Z=Back)
            False: RAI 좌표계 (X=Right, Y=Anterior, Z=Inferior)
        """
        # RAI -> 일반 오른손 좌표계 변환 함수
        def rai_to_standard(points):
            """RAI (X,Y,Z) -> 표준 (-X,Y,-Z)"""
            if points is None:
                return None
            pts = np.array(points)
            if pts.ndim == 1:
                return np.array([-pts[0], pts[1], -pts[2]])
            else:
                return np.column_stack([-pts[:, 0], pts[:, 1], -pts[:, 2]])
        
        # 좌표 변환 적용
        if use_standard_coords:
            hit_points_vis = rai_to_standard(self.hit_points_array)
            original_cloud_vis = rai_to_standard(self.original_cloud) if self.original_cloud is not None else None
            grid_points_vis = rai_to_standard(self.grid_points.reshape(-1, 3)) if self.grid_points is not None else None
            grid_center_vis = rai_to_standard(self.grid_center) if self.grid_center is not None else None
        else:
            hit_points_vis = self.hit_points_array
            original_cloud_vis = self.original_cloud
            grid_points_vis = self.grid_points.reshape(-1, 3) if self.grid_points is not None else None
            grid_center_vis = self.grid_center
        
        geometries = []
        
        # 교차점 (빨간색)
        if len(hit_points_vis) > 0:
            pcd_hit = o3d.geometry.PointCloud()
            pcd_hit.points = o3d.utility.Vector3dVector(hit_points_vis)
            pcd_hit.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(pcd_hit)
        
        # 원본 포인트클라우드 (회색)
        if show_original_cloud and original_cloud_vis is not None:
            pcd_original = o3d.geometry.PointCloud()
            pcd_original.points = o3d.utility.Vector3dVector(original_cloud_vis)
            pcd_original.paint_uniform_color([0.8, 0.8, 0.8])
            geometries.append(pcd_original)
        
        # 격자점 (파란색)
        if show_grid_points and grid_points_vis is not None:
            pcd_grid = o3d.geometry.PointCloud()
            pcd_grid.points = o3d.utility.Vector3dVector(grid_points_vis)
            pcd_grid.paint_uniform_color([0.0, 0.0, 1.0])
            geometries.append(pcd_grid)
        
        # 중심점 (녹색 구체)
        if show_center_point and grid_center_vis is not None:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
            sphere.translate(grid_center_vis)
            sphere.paint_uniform_color([0.0, 1.0, 0.0])
            geometries.append(sphere)
        
        # 좌표축 (일반 오른손 좌표계)
        if len(hit_points_vis) > 0:
            axis_size = max(50.0, (hit_points_vis.max(axis=0) - hit_points_vis.min(axis=0)).max() * 0.5)
            axis_origin = hit_points_vis.mean(axis=0)
        elif grid_center_vis is not None:
            axis_size = 50.0
            axis_origin = grid_center_vis
        else:
            axis_size = 50.0
            axis_origin = np.array([0.0, 0.0, 0.0])
        
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size,
            origin=axis_origin
        )
        geometries.append(coord_frame)
        
        # 시각화
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        
        for geom in geometries:
            vis.add_geometry(geom)
        
        render_option = vis.get_render_option()
        render_option.point_size = hit_point_size
        render_option.background_color = np.array(background_color)
        
        vis.run()
        vis.destroy_window()
    
    def visualize_hit_points_only(
        self, 
        window_name: str = "교차점만 보기",
        point_size: float = 3.0,
        background_color: tuple = (0.1, 0.1, 0.1),
        use_standard_coords: bool = True,
    ) -> None:
        """
        교차점만 단독 시각화
        
        Parameters:
        -----------
        window_name : str
            창 이름
            
        point_size : float
            포인트 크기
            
        background_color : tuple
            배경색 RGB (0~1 범위)
            
        use_standard_coords : bool
            True: 일반 오른손 좌표계 (X=Right, Y=Up, Z=Back)
            False: RAI 좌표계 (X=Right, Y=Anterior, Z=Inferior)
        """
        if len(self.hit_points_array) == 0:
            print("교차점이 없습니다.")
            return
        
        # RAI -> 일반 오른손 좌표계 변환
        def rai_to_standard(points):
            """RAI (X,Y,Z) -> 표준 (-X,Y,-Z)"""
            pts = np.array(points)
            if pts.ndim == 1:
                return np.array([-pts[0], pts[1], -pts[2]])
            else:
                return np.column_stack([-pts[:, 0], pts[:, 1], -pts[:, 2]])
        
        # 좌표 변환 적용
        if use_standard_coords:
            hit_points_vis = rai_to_standard(self.hit_points_array)
        else:
            hit_points_vis = self.hit_points_array
        
        # Open3D로 시각화
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(hit_points_vis)
        pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 빨간색
        
        # 좌표축
        center = hit_points_vis.mean(axis=0)
        axis_size = max(50.0, (hit_points_vis.max(axis=0) - hit_points_vis.min(axis=0)).max() * 0.5)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size,
            origin=center
        )
        
        # 시각화
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)
        
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.array(background_color)
        
        vis.run()
        vis.destroy_window()


