"""
레이캐스팅 실행 모듈

포인트클라우드에 대한 레이캐스팅 및 교차점 탐색을 담당합니다.
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import open3d as o3d


class RayCaster:
    """
    레이캐스팅 실행 클래스
    
    주요 책임:
    - 포인트클라우드에 대한 KDTree 구축
    - 레이와 포인트클라우드의 교차점 탐색
    - 깊이 맵 생성
    
    사용 예제:
    ```python
    caster = RayCaster(
        pts_face=pts_face,
        search_radius_mm=3.0,
    )
    
    result = caster.cast_rays(
        grid_points=grid_points,
        ray_direction=ray_dir,
        max_ray_length=250.0,
    )
    
    depth_map = result["depth_map"]
    hit_points = result["hit_points"]
    ```
    """
    
    def __init__(
        self,
        pts_face: np.ndarray,
        search_radius_mm: float = 3.0,
    ):
        """
        Parameters:
        -----------
        pts_face : np.ndarray
            입력 포인트클라우드 (N, 3) 배열
            
        search_radius_mm : float
            레이 주변 탐색 반경 (mm)
        """
        self.pts_face = np.array(pts_face, dtype=np.float64)
        if self.pts_face.ndim != 2 or self.pts_face.shape[1] != 3:
            raise ValueError(f"pts_face는 (N, 3) 형태여야 합니다. 현재: {self.pts_face.shape}")
        
        self.search_radius_mm = search_radius_mm
        
        # KDTree 구축
        self._build_kdtree()
    
    def _build_kdtree(self) -> None:
        """포인트클라우드에 대한 KDTree 구축"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pts_face)
        self.kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    def cast_rays(
        self,
        grid_points: np.ndarray,
        ray_direction: np.ndarray,
        max_ray_length: float = 250.0,
        num_samples: int = 50,
        verbose: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        격자점에서 레이캐스팅 실행
        
        Parameters:
        -----------
        grid_points : np.ndarray
            격자점 좌표 (H, W, 3)
            
        ray_direction : np.ndarray
            정규화된 레이 방향 벡터 [x, y, z]
            
        max_ray_length : float
            최대 레이 길이 (mm)
            
        num_samples : int
            레이당 샘플링 포인트 개수
            
        verbose : bool
            진행 상황 출력 여부
            
        Returns:
        --------
        Dict containing:
            - depth_map: np.ndarray (H, W) - 깊이 맵
            - hit_points: np.ndarray (H, W, 3) - 교차점 3D 좌표
            - valid_mask: np.ndarray (H, W) - 유효 마스크
            - grid_points: np.ndarray (H, W, 3) - 격자점 좌표
        """
        grid_h, grid_w = grid_points.shape[:2]
        ray_dir = ray_direction / np.linalg.norm(ray_direction)
        
        # 결과 배열 초기화
        depth_map = np.full((grid_h, grid_w), np.nan, dtype=np.float64)
        hit_points = np.full((grid_h, grid_w, 3), np.nan, dtype=np.float64)
        valid_mask = np.zeros((grid_h, grid_w), dtype=bool)
        
        # 샘플링 거리
        sample_distances = np.linspace(0, max_ray_length, num_samples)
        
        # 레이캐스팅 실행
        total_rays = grid_h * grid_w
        for i in range(grid_h):
            for j in range(grid_w):
                ray_origin = grid_points[i, j]
                
                # 이 레이에서 가장 가까운 교차점 찾기
                best_depth, best_point = self._find_closest_intersection(
                    ray_origin=ray_origin,
                    ray_dir=ray_dir,
                    sample_distances=sample_distances,
                )
                
                if best_point is not None:
                    depth_map[i, j] = best_depth
                    hit_points[i, j] = best_point
                    valid_mask[i, j] = True
                
                # 진행 상황 출력
                if verbose and (i * grid_w + j + 1) % 500 == 0:
                    progress = (i * grid_w + j + 1) / total_rays * 100
                    print(f"  진행: {progress:.1f}% ({i * grid_w + j + 1}/{total_rays})")
        
        return {
            "depth_map": depth_map,
            "hit_points": hit_points,
            "valid_mask": valid_mask,
            "grid_points": grid_points,
        }
    
    def _find_closest_intersection(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        sample_distances: np.ndarray,
    ) -> tuple[float | None, np.ndarray | None]:
        """
        단일 레이에 대한 가장 가까운 교차점 찾기
        
        Parameters:
        -----------
        ray_origin : np.ndarray
            레이 시작점 [x, y, z]
            
        ray_dir : np.ndarray
            레이 방향 벡터 [x, y, z] (정규화됨)
            
        sample_distances : np.ndarray
            샘플링할 거리 배열
            
        Returns:
        --------
        tuple[float | None, np.ndarray | None]
            (깊이, 교차점) 또는 (None, None)
        """
        best_depth = np.inf
        best_point = None
        
        for dist in sample_distances:
            search_center = ray_origin + ray_dir * dist
            
            # 반경 내 포인트 탐색
            [k, idx, _] = self.kdtree.search_radius_vector_3d(
                search_center, 
                self.search_radius_mm
            )
            
            if k > 0:
                candidates = self.pts_face[idx]
                
                # 각 후보 포인트에 대해 레이와의 거리 계산
                for cand in candidates:
                    to_cand = cand - ray_origin
                    depth = np.dot(to_cand, ray_dir)
                    proj_point = ray_origin + depth * ray_dir
                    perp_dist = np.linalg.norm(cand - proj_point)
                    
                    # 유효한 교차점인지 확인
                    if (depth > 0 and 
                        perp_dist < self.search_radius_mm and 
                        depth < best_depth):
                        best_depth = depth
                        best_point = cand
        
        if best_point is not None:
            return best_depth, best_point
        else:
            return None, None
    
    def get_point_cloud(self) -> np.ndarray:
        """원본 포인트클라우드 반환"""
        return self.pts_face.copy()


