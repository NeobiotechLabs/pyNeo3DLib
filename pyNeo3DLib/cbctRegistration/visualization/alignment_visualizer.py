"""
정합 결과 시각화 모듈

CBCT-FaceScan 정합 과정의 시각화를 담당합니다.
"""
import numpy as np
import open3d as o3d
import copy
from typing import List, Tuple, Optional

from ..config import VisualizationConfig


class AlignmentVisualizer:
    """
    정합 결과 시각화 클래스
    
    단일 책임: 정합 과정 및 결과의 시각화
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None, visualize: bool = False, verbose: bool = True):
        """
        Args:
            config: 시각화 설정 (None일 경우 기본값 사용)
            visualize: 시각화 활성화 여부
            verbose: 상세 출력 여부
        """
        self.config = config if config is not None else VisualizationConfig()
        self.enabled = visualize  # 메서드 이름과 충돌 방지를 위해 enabled 사용
        self.verbose = verbose
    
    def visualize(
        self,
        geometries: List,
        window_name: str = "Visualization"
    ) -> None:
        """
        일반 시각화
        
        Args:
            geometries: 시각화할 geometry 객체 리스트
            window_name: 창 이름
        """
        if self.enabled:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=window_name,
                width=self.config.window_width,
                height=self.config.window_height
            )
    
    def visualize_alignment(
        self,
        cbct_pcd: o3d.geometry.PointCloud,
        facescan_pcd: o3d.geometry.PointCloud,
        title: str,
        cbct_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        facescan_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        additional_geometries: Optional[List] = None
    ) -> None:
        """
        정렬 결과 시각화
        
        Args:
            cbct_pcd: CBCT 포인트 클라우드
            facescan_pcd: FaceScan 포인트 클라우드
            title: 시각화 창 제목
            cbct_color: CBCT 색상 (RGB, 0~1)
            facescan_color: FaceScan 색상 (RGB, 0~1)
            additional_geometries: 추가 geometry 리스트
        """
        cbct_vis = copy.deepcopy(cbct_pcd)
        cbct_vis.paint_uniform_color(cbct_color)
        
        facescan_vis = copy.deepcopy(facescan_pcd)
        facescan_vis.paint_uniform_color(facescan_color)
        
        geometries = [cbct_vis, facescan_vis]
        
        if additional_geometries:
            geometries.extend(additional_geometries)
        
        self.visualize(geometries, title)
    
    def visualize_final_result(
        self,
        cbct_pcd: o3d.geometry.PointCloud,
        facescan_mesh: o3d.geometry.TriangleMesh,
        cbct_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        facescan_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        window_name: str = "최종 정합 결과"
    ) -> None:
        """
        최종 정합 결과 시각화 (메쉬 포함)
        
        Args:
            cbct_pcd: CBCT 포인트 클라우드
            facescan_mesh: FaceScan 메쉬
            cbct_color: CBCT 색상
            facescan_color: FaceScan 색상
            window_name: 창 이름
        """
        cbct_vis = copy.deepcopy(cbct_pcd)
        cbct_vis.paint_uniform_color(cbct_color)
        
        facescan_vis = copy.deepcopy(facescan_mesh)
        facescan_vis.paint_uniform_color(facescan_color)
        
        self.visualize([facescan_vis, cbct_vis], window_name)
    
    def create_coordinate_frame(
        self,
        size: Optional[float] = None,
        origin: Optional[np.ndarray] = None
    ) -> o3d.geometry.TriangleMesh:
        """
        좌표축 프레임 생성
        
        Args:
            size: 좌표축 크기 (None이면 config 기본값)
            origin: 원점 위치 (None이면 [0, 0, 0])
        
        Returns:
            좌표축 메쉬
        """
        frame_size = size if size is not None else self.config.coordinate_frame_size
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        
        if origin is not None:
            frame.translate(origin)
        
        return frame
    
    def create_sphere_marker(
        self,
        center: np.ndarray,
        radius: float = 5.0,
        color: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    ) -> o3d.geometry.TriangleMesh:
        """
        구 마커 생성
        
        Args:
            center: 구 중심 좌표
            radius: 구 반경
            color: RGB 색상 (0~1)
        
        Returns:
            구 메쉬
        """
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(center)
        sphere.paint_uniform_color(color)
        return sphere
    
    def visualize_nose_points(
        self,
        cbct_pcd: o3d.geometry.PointCloud,
        facescan_pcd: o3d.geometry.PointCloud,
        cbct_nose_point: np.ndarray,
        facescan_nose_point: np.ndarray,
        title: str = "코 정점 포인트 비교",
        cbct_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        facescan_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        cbct_nose_color: Tuple[float, float, float] = (1.0, 0.5, 0.0),
        facescan_nose_color: Tuple[float, float, float] = (0.0, 0.5, 1.0)
    ) -> None:
        """
        CBCT와 FaceScan의 코 정점 포인트를 시각화
        
        Args:
            cbct_pcd: CBCT 포인트 클라우드
            facescan_pcd: FaceScan 포인트 클라우드
            cbct_nose_point: CBCT 코 정점 좌표
            facescan_nose_point: FaceScan 코 정점 좌표
            title: 시각화 창 제목
            cbct_color: CBCT 색상 (RGB, 0~1)
            facescan_color: FaceScan 색상 (RGB, 0~1)
            cbct_nose_color: CBCT 코 정점 마커 색상 (주황색)
            facescan_nose_color: FaceScan 코 정점 마커 색상 (하늘색)
        """
        import copy
        
        cbct_vis = copy.deepcopy(cbct_pcd)
        cbct_vis.paint_uniform_color(cbct_color)
        
        facescan_vis = copy.deepcopy(facescan_pcd)
        facescan_vis.paint_uniform_color(facescan_color)
        
        # 코 정점 마커 생성 (구 형태)
        cbct_nose_marker = self.create_sphere_marker(
            cbct_nose_point, 
            radius=3.0, 
            color=cbct_nose_color
        )
        
        facescan_nose_marker = self.create_sphere_marker(
            facescan_nose_point, 
            radius=3.0, 
            color=facescan_nose_color
        )
        
        # 좌표축 프레임 추가 (원점)
        coordinate_frame = self.create_coordinate_frame(size=20.0)
        
        geometries = [
            cbct_vis, 
            facescan_vis, 
            cbct_nose_marker, 
            facescan_nose_marker,
            coordinate_frame
        ]
        
        # 거리 계산 및 출력
        distance = np.linalg.norm(cbct_nose_point - facescan_nose_point)
        print(f"\n코 정점 포인트 거리: {distance:.2f} mm")
        print(f"  CBCT 코 정점: {cbct_nose_point}")
        print(f"  FaceScan 코 정점: {facescan_nose_point}")
        
        self.visualize(geometries, title)

