import open3d as o3d
import numpy as np
from typing import Optional
from .constants import DistanceCalculationConfig


class DistanceCalculator:
    """거리 계산을 담당하는 클래스"""
    
    def __init__(self, default_step: int = DistanceCalculationConfig.DEFAULT_STEP):
        """
        Args:
            default_step: 기본 샘플링 간격
        """
        self.default_step = default_step
    
    def compute_rmse_bidirectional(self, 
                                   src_pcd: o3d.geometry.PointCloud, 
                                   tgt_pcd: o3d.geometry.PointCloud, 
                                   step: Optional[int] = None) -> float:
        """
        Chamfer Distance 기반 RMSE (양방향 최근접 거리).
        
        Args:
            src_pcd: 소스 포인트 클라우드
            tgt_pcd: 타겟 포인트 클라우드
            step: 샘플링 간격 (None이면 기본값 사용)
            
        Returns:
            float: 양방향 RMSE 값
        """
        if step is None:
            step = self.default_step
        
        src_pts = np.asarray(src_pcd.points)
        tgt_pts = np.asarray(tgt_pcd.points)

        tgt_kdtree = o3d.geometry.KDTreeFlann(tgt_pcd)
        src_kdtree = o3d.geometry.KDTreeFlann(src_pcd)

        # src → tgt
        dists_src2tgt = []
        for p in src_pts[::step]:
            _, _, dist = tgt_kdtree.search_knn_vector_3d(p, 1)
            dists_src2tgt.append(np.sqrt(dist[0]))

        # tgt → src
        dists_tgt2src = []
        for p in tgt_pts[::step]:
            _, _, dist = src_kdtree.search_knn_vector_3d(p, 1)
            dists_tgt2src.append(np.sqrt(dist[0]))

        # 양방향 평균 RMSE
        rmse = (np.mean(np.square(dists_src2tgt)) + np.mean(np.square(dists_tgt2src))) / 2
        return np.sqrt(rmse)
    
    def compute_chamfer_distance(self, 
                                src_pcd: o3d.geometry.PointCloud, 
                                tgt_pcd: o3d.geometry.PointCloud, 
                                step: Optional[int] = None) -> float:
        """
        Chamfer Distance 계산 (양방향 평균 거리)
        
        Args:
            src_pcd: 소스 포인트 클라우드
            tgt_pcd: 타겟 포인트 클라우드
            step: 샘플링 간격 (None이면 기본값 사용)
            
        Returns:
            float: Chamfer Distance 값
        """
        if step is None:
            step = self.default_step
        
        src_pts = np.asarray(src_pcd.points)
        tgt_pts = np.asarray(tgt_pcd.points)

        tgt_kdtree = o3d.geometry.KDTreeFlann(tgt_pcd)
        src_kdtree = o3d.geometry.KDTreeFlann(src_pcd)

        # src → tgt
        dists_src2tgt = []
        for p in src_pts[::step]:
            _, _, dist = tgt_kdtree.search_knn_vector_3d(p, 1)
            dists_src2tgt.append(np.sqrt(dist[0]))

        # tgt → src
        dists_tgt2src = []
        for p in tgt_pts[::step]:
            _, _, dist = src_kdtree.search_knn_vector_3d(p, 1)
            dists_tgt2src.append(np.sqrt(dist[0]))

        # 양방향 평균 거리
        chamfer_dist = (np.mean(dists_src2tgt) + np.mean(dists_tgt2src)) / 2
        return chamfer_dist
    
