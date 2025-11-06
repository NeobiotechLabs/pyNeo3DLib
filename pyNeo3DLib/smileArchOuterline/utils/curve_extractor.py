"""
치아 악궁 곡선 추출을 담당하는 클래스
단일책임: 레이캐스팅과 필터링을 통한 곡선 포인트 추출
"""

import numpy as np
from typing import Tuple
from .constants import AnalysisConstants
from .ray_caster import RayCaster
from .signal_processor import SignalProcessor
from .mesh_alignment_manager import MeshAlignmentManager
from .curve_sampler import CurveSampler
from .polar_sampler import PolarSampling
import time
from .visualizer import VisualizeForTest
import open3d as o3d


class CurveExtractor:
    """치아 악궁 곡선 추출을 담당하는 클래스"""
    
    def __init__(self):
        self.ray_caster = RayCaster()
        self.signal_processor = SignalProcessor()
        self.mesh_aligner = MeshAlignmentManager()
        self.curve_sampler = CurveSampler()
    
    def extract_contour_points(self, aligned_mesh: object, y_axis: np.ndarray) -> Tuple[np.ndarray, object]:
        """
        높이별 레이캐스팅을 통해 등고선 포인트를 추출합니다.
        
        Args:
            aligned_mesh: 정렬된 메쉬
            y_axis: Y축 벡터
            
        Returns:
            Tuple[filtered_result_points_array, filtered_aligned_mesh]: 
                - filtered_result_points_array: 필터링된 포인트 배열
                - filtered_aligned_mesh: 필터링된 메쉬
        """
        vertices = aligned_mesh.points

        time_start = time.time()

        # vertices를 다운샘플링 (Voxel Grid - 메시 형태 유지)
        vertices = self._voxel_downsample(vertices, voxel_size=1)
        # 레이캐스팅으로 등고선 포인트 클라우드 추출
        result_points_array = self.ray_caster.perform_height_based_ray_casting(
            vertices, y_axis, 
            num_slices=AnalysisConstants.DEFAULT_NUM_SLICES, 
            angle_step=AnalysisConstants.DEFAULT_ANGLE_STEP
        )


        time_end = time.time()
        print(f"다운샘플링 후 레이캐스팅 소요 시간: {time_end - time_start}")

        # 대구치 아웃라이어 제거
        filtered_result_points_array = self.signal_processor.remove_molar_outliers(
            result_points_array, 
            percentile_threshold=AnalysisConstants.MOLAR_OUTLIER_PERCENTILE_THRESHOLD
        )
 
        # 메쉬 필터링
        filtered_aligned_mesh = self.mesh_aligner.filter_mesh_by_z_threshold(
            aligned_mesh, filtered_result_points_array
        )

        print(f"filtered_aligned_mesh: {filtered_aligned_mesh.points.shape}")
        
        return filtered_result_points_array, filtered_aligned_mesh
    
    def filter_mesh_by_z_threshold(self, aligned_mesh: object, y_axis: np.ndarray) -> object:
        """
        높이별 레이캐스팅을 통해 등고선 포인트를 추출하고 메쉬를 필터링합니다.
        
        Args:
            aligned_mesh: 정렬된 메쉬
            y_axis: Y축 벡터
            
        Returns:
            object: 필터링된 메쉬
        """
        # 레이캐스팅으로 등고선 포인트 클라우드 추출
        result_points_array = self.ray_caster.perform_height_based_ray_casting(
            aligned_mesh, y_axis, 
            num_slices=AnalysisConstants.DEFAULT_NUM_SLICES, 
            angle_step=AnalysisConstants.DEFAULT_ANGLE_STEP
        )

        print(f"result_points_array shape: {result_points_array.shape}")
        
        # 대구치 아웃라이어 제거
        filtered_result_points_array = self.signal_processor.remove_molar_outliers(
            result_points_array, 
            percentile_threshold=AnalysisConstants.MOLAR_OUTLIER_PERCENTILE_THRESHOLD
        )

        
        # 메쉬 필터링
        filtered_aligned_mesh = self.mesh_aligner.filter_mesh_by_z_threshold(
            aligned_mesh, filtered_result_points_array
        )
        
        return filtered_aligned_mesh
    
    def extract_curve_by_polar_sampling(self, mesh_points: np.ndarray, z_min_point: float) -> np.ndarray:
        """
        극좌표 샘플링을 통해 곡선 포인트를 추출합니다.
        
        Args:
            mesh_points: 메쉬 포인트 배열
            z_min_point: Z 최소값
            
        Returns:
            np.ndarray: 극좌표 샘플링으로 추출된 곡선 포인트
        """
        polar_sampler = PolarSampling(np.array([0, 0, z_min_point]))
        polar_sampling_points = polar_sampler.polar_sampling(
            mesh_points,
            angle_step=1,
            mode="ymin",
            y_range=(-np.inf, np.inf)
        )
        
        return polar_sampling_points
    
    def extract_curve_by_curve_sampling(self, filtered_mesh: object, average_y_value: float, z_min_point: float) -> np.ndarray:
        """
        곡선 샘플러를 통해 곡선 포인트를 추출합니다.
        
        Args:
            filtered_mesh: 필터링된 메쉬
            average_y_value: 평균 Y값
            z_min_point: Z 최소값
            
        Returns:
            np.ndarray: 곡선 샘플링으로 추출된 곡선 포인트
        """
        polar_sampling_center = np.array([0, average_y_value, z_min_point])
        sampled_curve_points = self.curve_sampler.perform_polar_sampling(
            filtered_mesh.points, 
            polar_sampling_center, 
            angle_step=AnalysisConstants.DEFAULT_ANGLE_STEP_FULL_ARC, 
            y_slice_mid=average_y_value, 
            y_offset=AnalysisConstants.DEFAULT_Y_OFFSET
        )
        
        return sampled_curve_points
    
    def _voxel_downsample(self, points: np.ndarray, voxel_size: float = 0.5) -> np.ndarray:
        """
        Voxel Grid를 사용한 포인트 클라우드 다운샘플링 (Open3D 사용)
        메시의 형태를 유지하면서 공간적으로 균등하게 샘플링
        
        Args:
            points: (N, 3) 포인트 배열
            voxel_size: voxel 크기 (작을수록 더 많은 포인트 유지)
            
        Returns:
            다운샘플링된 포인트 배열
        """
        # Open3D PointCloud 객체 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Open3D의 최적화된 voxel downsampling 사용 (C++로 구현됨)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        # numpy 배열로 변환하여 반환
        return np.asarray(downsampled_pcd.points)
