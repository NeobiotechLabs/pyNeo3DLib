"""
CBCT Depth Map Extractor - 메인 조정 클래스

레이캐스팅을 통한 표면 교차점 추출을 담당하는 메인 클래스입니다.
RayGridGenerator, RayCaster, DepthMapVisualizer를 조합하여 사용합니다.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import time
import numpy as np

from .ray_grid_generator import RayGridGenerator
from .ray_caster import RayCaster
from .depth_map_visualizer import DepthMapVisualizer


class CBCTDepthMapExtractor:
    """
    레이캐스팅을 통한 표면 교차점 추출 클래스 (리팩토링 버전)
    
    단일 책임 원칙에 따라 다음 컴포넌트들을 조합하여 사용합니다:
    - RayGridGenerator: 격자 생성
    - RayCaster: 레이캐스팅 실행
    - DepthMapVisualizer: 시각화
    
    사용 예제:
    ```python
    from cbctRegistration.processing.depth_map import CBCTDepthMapExtractor
    
    # 포인트클라우드 준비
    pts_face = np.array([[x, y, z], ...])  # (N, 3) 배열
    
    # 교차점 추출기 생성
    extractor = CBCTDepthMapExtractor(
        pts_face=pts_face,
        grid_center=[77.7, 85.0, 94.23],
        grid_width_mm=80.0,
        grid_height_mm=100.0,
        grid_resolution=(50, 50),
        ray_direction=[0, -1, 0],  # Y- 방향
        ray_start_offset_mm=150.0,
        search_radius_mm=3.0,
    )
    
    # 교차점 추출
    result = extractor.extract()
    
    # 결과 접근
    hit_points = result["hit_points_array"]  # (N, 3) 교차점 배열
    depth_map = result["depth_map"]          # (H, W) 깊이 맵
    
    # 저장 및 시각화
    extractor.save_hit_points("output.ply")
    extractor.visualize_3d()
    ```
    """
    
    def __init__(
        self,
        pts_face: np.ndarray,
        grid_center: np.ndarray | list,
        grid_width_mm: float,
        grid_height_mm: float,
        grid_resolution: Tuple[int, int] = (50, 50),
        ray_direction: np.ndarray | list = [0, -1, 0],
        ray_start_offset_mm: float = 150.0,
        search_radius_mm: float = 3.0,
    ):
        """
        Parameters:
        -----------
        pts_face : np.ndarray
            입력 포인트클라우드 (N, 3) 배열
            
        grid_center : np.ndarray | list
            격자 평면 중심 위치 [x, y, z] (mm)
            
        grid_width_mm : float
            격자 가로 폭 (mm)
            
        grid_height_mm : float
            격자 세로 폭 (mm)
            
        grid_resolution : Tuple[int, int]
            격자 해상도 (가로, 세로)
            예: (50, 50) = 2500개 레이
            
        ray_direction : np.ndarray | list
            레이 발사 방향 [x, y, z]
            예: [0, -1, 0] = Y- 방향 (전방->후방)
            
        ray_start_offset_mm : float
            격자 평면을 center에서 ray_direction 반대로 얼마나 뒤로 배치할지 (mm)
            
        search_radius_mm : float
            레이 주변 탐색 반경 (mm)
            
        grid_axis_u : Optional[np.ndarray | list]
            격자 가로축 방향 벡터 [x, y, z]
            None이면 ray_direction에 수직인 축 자동 계산
            
        grid_axis_v : Optional[np.ndarray | list]
            격자 세로축 방향 벡터 [x, y, z]
            None이면 ray_direction과 u축에 수직인 축 자동 계산
        """
        # 입력 포인트클라우드
        self.pts_face = np.array(pts_face, dtype=np.float64)
        if self.pts_face.ndim != 2 or self.pts_face.shape[1] != 3:
            raise ValueError(f"pts_face는 (N, 3) 형태여야 합니다. 현재: {self.pts_face.shape}")
        
        # 격자 파라미터 저장
        self.grid_center = np.array(grid_center, dtype=np.float64)
        self.grid_resolution = grid_resolution
        
        # 컴포넌트 초기화
        self.grid_generator = RayGridGenerator(
            grid_center=grid_center,
            grid_width_mm=grid_width_mm,
            grid_height_mm=grid_height_mm,
            grid_resolution=grid_resolution,
            ray_direction=ray_direction,
            ray_start_offset_mm=ray_start_offset_mm,
        )
        
        self.ray_caster = RayCaster(
            pts_face=pts_face,
            search_radius_mm=search_radius_mm,
        )
        
        # 결과 저장
        self._depth_result: Optional[Dict[str, np.ndarray]] = None
        self._visualizer: Optional[DepthMapVisualizer] = None
        
    def extract(self, verbose: bool = True) -> Dict[str, Any]:
        """
        레이캐스팅으로 교차점 추출
        
        Returns:
        --------
        Dict containing:
            - depth_map: np.ndarray (H, W) - 깊이 맵
            - hit_points: np.ndarray (H, W, 3) - 교차점 3D 좌표
            - hit_points_array: np.ndarray (N, 3) - 유효한 교차점만 (N개)
            - valid_mask: np.ndarray (H, W) - 유효 마스크
            - grid_points: np.ndarray (H, W, 3) - 격자점 좌표
        """
        if verbose:
            grid_info = self.grid_generator.get_grid_info()
            print("=" * 80)
            print("레이캐스팅 교차점 추출 시작")
            print("=" * 80)
            print(f"입력 포인트: {self.pts_face.shape[0]:,}개")
            print(f"격자 중심: {grid_info['grid_center']}")
            print(f"격자 크기: {grid_info['grid_width_mm']}mm x {grid_info['grid_height_mm']}mm")
            print(f"격자 해상도: {self.grid_resolution[0]} x {self.grid_resolution[1]}")
            print(f"레이 방향: {grid_info['ray_direction']}")
        
        t0 = time.time()
        
        # 1. 격자 생성
        grid_points = self.grid_generator.generate_grid_points()
        ray_direction = self.grid_generator.get_ray_direction()
        
        # 2. 레이캐스팅 실행
        self._depth_result = self.ray_caster.cast_rays(
            grid_points=grid_points,
            ray_direction=ray_direction,
            max_ray_length=250.0,
            num_samples=50,
            verbose=verbose,
        )
        
        if verbose:
            elapsed = time.time() - t0
            total_valid = np.sum(self._depth_result["valid_mask"])
            total_grid = self.grid_resolution[0] * self.grid_resolution[1]
            print(f"\n완료: {elapsed:.2f}초")
            print(f"교차점: {total_valid}/{total_grid}개 ({total_valid/total_grid*100:.1f}%)")
            print("=" * 80)
        
        # 결과 정리
        hit_points_array = self._get_hit_points_from_depth_map(self._depth_result)
        
        # 시각화 객체 생성
        self._visualizer = DepthMapVisualizer(
            hit_points_array=hit_points_array,
            original_cloud=self.pts_face,
            grid_points=grid_points,
            grid_center=self.grid_center,
        )
        
        result = {
            "depth_map": self._depth_result["depth_map"],
            "hit_points": self._depth_result["hit_points"],
            "hit_points_array": hit_points_array,
            "valid_mask": self._depth_result["valid_mask"],
            "grid_points": self._depth_result["grid_points"],
        }
        
        return result
    
    def _get_hit_points_from_depth_map(self, depth_result: Dict[str, np.ndarray]) -> np.ndarray:
        """Depth Map 결과에서 유효한 교차점만 추출"""
        hit_points = depth_result["hit_points"]
        valid_mask = depth_result["valid_mask"]
        return hit_points[valid_mask]
    
    def get_result(self) -> Dict[str, Any]:
        """
        마지막 추출 결과 반환
        
        Returns:
        --------
        Dict containing:
            - depth_map: np.ndarray (H, W) - 깊이 맵
            - hit_points: np.ndarray (H, W, 3) - 교차점 3D 좌표
            - hit_points_array: np.ndarray (N, 3) - 유효한 교차점만 (N개)
            - valid_mask: np.ndarray (H, W) - 유효 마스크
            - grid_points: np.ndarray (H, W, 3) - 격자점 좌표
        """
        if self._depth_result is None:
            raise RuntimeError("extract()를 먼저 실행하세요")
        
        hit_points_array = self._get_hit_points_from_depth_map(self._depth_result)
        
        return {
            "depth_map": self._depth_result["depth_map"],
            "hit_points": self._depth_result["hit_points"],
            "hit_points_array": hit_points_array,
            "valid_mask": self._depth_result["valid_mask"],
            "grid_points": self._depth_result["grid_points"],
        }

