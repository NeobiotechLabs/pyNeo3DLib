"""
OBB(Oriented Bounding Box) 분석 및 좌표계 계산을 담당하는 모듈
"""
import open3d as o3d
import numpy as np
from typing import Tuple


class OBBAnalyzer:
    """OBB(Oriented Bounding Box) 분석 및 좌표계 계산을 담당하는 클래스"""
    
    def compute_obb_info(self, pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Open3D의 OBB 기반으로 중심 좌표, 축 방향, 각 축 크기를 계산
        
        Args:
            pcd: 입력 포인트 클라우드
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (중심점, 축 방향 행렬, 축 크기)
                - center: (3,) 중심 좌표
                - axes: (3, 3) 각 열이 주축 벡터 (X, Y, Z 축 방향)
                - extent: (3,) 각 축 방향의 길이
        """
        obb = pcd.get_oriented_bounding_box()
        
        center = obb.center     # 중심 좌표
        axes = obb.R.T          # 주축 방향 (열 벡터)
        extent = obb.extent     # 축 크기 (W, H, D)
        
        return center, axes, extent
    
    def get_coordinate_frame(self, pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        """
        포인트클라우드의 좌표계 정보를 반환
        
        Args:
            pcd: 입력 포인트 클라우드
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (중심점, 기저 행렬)
        """
        center, axes, _ = self.compute_obb_info(pcd)
        return center, axes
