"""
치아 악궁 분석 프로세스를 조율하는 클래스
단일책임: 전체 분석 프로세스의 조율 및 통합
"""

import numpy as np
from pyNeo3DLib.smileArchOuterline.utils import visualizer
import pyvista as pv
from typing import List, Tuple
from .curve_extractor import CurveExtractor
from .mesh_processor import MeshProcessor
from .analysis_visualizer import AnalysisVisualizer
from .signal_processor import SignalProcessor
from .curve_sampler import CurveSampler
from .constants import AnalysisConstants
from .tangent_normal_from_curve import CurveTangentNormalCalculator
from .polar_sampler import PolarSampling
from .visualizer import VisualizeForTest
from .ray_caster import RayCaster
from scipy.spatial import KDTree
import time


class ArchAnalysisCoordinator:
    """치아 악궁 분석 프로세스를 조율하는 클래스"""
    
    def __init__(self):
        self.curve_extractor = CurveExtractor()
        self.mesh_processor = MeshProcessor()
        self.visualizer = AnalysisVisualizer()
        self.signal_processor = SignalProcessor()
        self.curve_sampler = CurveSampler()
    
    def _align_mesh_with_arch_centroid(self, rotated_mesh: object) -> object:
        """
        아치 중심점을 기준으로 메쉬를 추가 정렬합니다.
        
        극좌표 샘플링으로 추출한 곡선의 중심점을 계산하고,
        해당 중심점이 Z축 방향을 향하도록 메쉬를 회전시킵니다.
        
        Args:
            rotated_mesh: 회전시킬 메쉬
            
        Returns:
            object: 중심점 기준으로 정렬된 메쉬
        """
        # 1단계: 극좌표 샘플링으로 곡선 추출
        z_min_point = min(rotated_mesh.points[:, 2])
        polar_sampling_points = self.curve_extractor.extract_curve_by_polar_sampling(
            rotated_mesh.points, z_min_point
        )
        
        # 2단계: 이동평균 필터링
        moving_average_points = self.signal_processor.moving_average_filter(
            polar_sampling_points,
            window_size=AnalysisConstants.DEFAULT_WINDOW_SIZE
        )
        
        # 3단계: 곡선의 시작점과 끝점을 잇는 직선 생성
        direction_vector = moving_average_points[-1] - moving_average_points[0]
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        points_on_line = np.linspace(moving_average_points[0], moving_average_points[-1], 100)
        points_on_line = points_on_line + direction_vector * 0.1
        
        # 4단계: 곡선의 중심점 계산
        centroid_point_of_arch = np.mean(moving_average_points, axis=0)
        centroid_point_of_arch = centroid_point_of_arch.reshape(1, 3)
        
        # 5단계: KDTree를 사용해 중심점에서 가장 가까운 선상의 점 찾기
        kdtree = KDTree(points_on_line)
        dist, idx = kdtree.query(centroid_point_of_arch, k=1)
        closest_point_on_line = points_on_line[idx].reshape(3)
        
        # 6단계: 선상의 점에서 중심점으로 가는 방향벡터 계산
        centroid_point_of_arch_flat = centroid_point_of_arch.reshape(3)
        direction_vector_from_closest_point_to_centroid = centroid_point_of_arch_flat - closest_point_on_line
        direction_vector_from_closest_point_to_centroid = direction_vector_from_closest_point_to_centroid / np.linalg.norm(direction_vector_from_closest_point_to_centroid)
        
        # 7단계: [0,0,1]과 방향벡터가 이루는 각도 계산 및 Y축 기준 회전
        angle = -1 * np.arccos(np.dot([0, 0, 1], direction_vector_from_closest_point_to_centroid))
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        rotated_mesh.points = rotated_mesh.points @ rotation_matrix
        
        return rotated_mesh
    
    def perform_precise_alignment(self, aligned_mesh: object, y_axis: np.ndarray) -> Tuple[np.ndarray, object, np.ndarray]:
        """
        정밀정렬을 수행합니다.
        
        Args:
            aligned_mesh: 1차 정렬된 메쉬
            y_axis: Y축 벡터
            
        Returns:
            Tuple[smoothed_points, rotated_mesh, new_center_point]:
                - smoothed_points: 정밀정렬된 곡선 포인트들
                - rotated_mesh: 정밀정렬된 메쉬
                - new_center_point: 새로운 중심점
        """
        # 1단계: 등고선 포인트 추출
        print(f"레이캐스팅에 사용할 Y축: {y_axis}")
        
        filtered_result_points_array, filtered_aligned_mesh = self.curve_extractor.extract_contour_points(
            aligned_mesh, y_axis
        )
        
        # 2단계: Y값 평균 계산
        average_y_value = self.signal_processor.calculate_average_y_value(filtered_result_points_array)
        print(f"average_y_value: {average_y_value}")
        
        # 3단계: 곡선 포인트 추출
        z_min_point = min(filtered_aligned_mesh.points[:, 2])
        sampled_curve_points = self.curve_extractor.extract_curve_by_curve_sampling(
            filtered_aligned_mesh, average_y_value, z_min_point
        )
        
        # 4단계: 이동평균 필터링
        smoothed_points = self.signal_processor.moving_average_filter(
            sampled_curve_points, 
            window_size=AnalysisConstants.DEFAULT_WINDOW_SIZE
        )
        
        # 5단계: 메시 방향 정렬
        rotated_mesh, smoothed_points, _ = self.mesh_processor.align_mesh_direction(
            aligned_mesh, smoothed_points
        )

        # 6단계: 필터링 및 중심 맞추기
        final_mesh, centered_filtered_points, new_center = self.mesh_processor.filter_and_center_mesh(
            rotated_mesh, smoothed_points
        )
        
        # 7단계: 아치 중심점 기준 추가 정렬
        final_mesh = self._align_mesh_with_arch_centroid(final_mesh)

        return final_mesh, centered_filtered_points, new_center
    

    def analyze_upper_IOS_scandata(
        self,
        mesh_path: str,
        visualize_result: bool = True
    ) -> Tuple[float, float, List[List[float]]]:
        """
        상악 IOS 스캔 데이터에서 치아 아치 곡선을 추출합니다.
        
        Args:
            mesh_path: STL 메쉬 파일 경로
            visualize_result: 결과 시각화 여부 (기본값: True)
            
        Returns:
            Tuple[arch_depth, molar_width, landmark_points]: 
                - arch_depth: 치아 배열 곡선의 깊이
                - molar_width: 치아 배열 곡선의 폭
                - landmark_points: 정규화된 랜드마크 포인트 리스트
        """
        y_axis = np.array([0, 1, 0])
        # 1단계: 1차 정렬 수행
        time_start = time.time()
        aligned_mesh, _, _, _, _ = self.mesh_processor.perform_initial_alignment(mesh_path)
        time_end = time.time()
        print(f"1차 정렬 시간: {time_end - time_start}")
        # 2단계: 정밀정렬 수행
        time_start = time.time()
        rotated_mesh, centered_filtered_points, _ = self.perform_precise_alignment(
            aligned_mesh, y_axis
        )
        time_end = time.time()
        print(f"정밀정렬 시간: {time_end - time_start}")






        # 대구치 아웃라이어 제거
        time_start = time.time()
        signal_processor = SignalProcessor()
        filtered_result_points_array = signal_processor.remove_molar_outliers(
            rotated_mesh.points, 
            percentile_threshold=AnalysisConstants.MOLAR_OUTLIER_PERCENTILE_THRESHOLD
        )
        time_end = time.time()
        print(f"대구치 아웃라이어 제거 시간: {time_end - time_start}")

        time_end = time.time()
                # 2단계: 극좌표 샘플링으로 곡선 추출
        z_min_point = min(rotated_mesh.points[:, 2])
        time_start = time.time()
        curve_extractor = CurveExtractor()
        polar_sampling_points = curve_extractor.extract_curve_by_polar_sampling(
            filtered_result_points_array, z_min_point
        )
        time_end = time.time()
        print(f"극좌표 샘플링 시간: {time_end - time_start}")


        time_start = time.time()
        # 3단계: 이동평균 필터링
        moving_average_points = self.signal_processor.moving_average_filter(
           polar_sampling_points ,
            window_size=AnalysisConstants.DEFAULT_WINDOW_SIZE
        )
        time_end = time.time()
        print(f"이동평균 필터링 시간: {time_end - time_start}")


        time_start = time.time()
        # curve tangent and normal calculation
        curve_tangent_normal_calculator = CurveTangentNormalCalculator()
        _, curve_normal = curve_tangent_normal_calculator.calculate_tangents_and_normals(moving_average_points)
        outer_expand_curve_points = moving_average_points - curve_normal * 5
        time_end = time.time()
        print(f"곡선 탄젠트 계산 시간: {time_end - time_start}")

        time_start = time.time()
        outer_expand_curve_points_min_point = np.min(outer_expand_curve_points[:, 0])
        outer_expand_curve_points_max_point = np.max(outer_expand_curve_points[:, 0])
        mask = (filtered_result_points_array[:, 0] > outer_expand_curve_points_min_point) & (filtered_result_points_array[:, 0] < outer_expand_curve_points_max_point)
        filtered_result_points_array = filtered_result_points_array[mask]
        time_end = time.time()
        print(f"메시 필터링 시간: {time_end - time_start}")

        time_start = time.time()

        # 2단계: 극좌표 샘플링으로 곡선 추출
        z_min_point = min(filtered_result_points_array[:, 2])
        polar_sampling_points_second = self.curve_extractor.extract_curve_by_polar_sampling(
            filtered_result_points_array, z_min_point
        )
        
        #  유치악, 무치악 판단해서 필터링 계수 변경시켜야함


        time_start = time.time()    
        # 80도에서 100도 사이의 포인트 추출 
        polar_sampler = PolarSampling(np.array([0, 0, z_min_point]))
        polar_sampling_points_80_100 = polar_sampler.polar_sampling(
            filtered_result_points_array,
            angle_step=1,
            mode="ymin",
            start_angle=80,
            end_angle=100,
            y_range=(-np.inf, np.inf)
        )

        polar_sampling_points_80_100_second = polar_sampler.polar_sampling(
            filtered_result_points_array,
            angle_step=1,
            mode="farthest",
            start_angle=80,
            end_angle=100,
            y_range=(-np.inf, np.inf)
        )
        time_end = time.time()
        print(f"polar_sampling_points_80_100 시간: {time_end - time_start}")

        time_start = time.time()
        # polar_sampling_points_80_100와 polar_sampling_points_80_100_second 의 RMSE 계산
        rmse = np.sqrt(np.mean((polar_sampling_points_80_100 - polar_sampling_points_80_100_second) ** 2))
        print(f"rmse: {rmse}")
    

        if rmse < 2:
            print("유치악")
            filter_window_size = 50
        else:
            print("무치악")
            filter_window_size = 20



        time_start = time.time()
        # 3단계: 정밀한 곡선 포인트 추출
        moving_average_points = self.signal_processor.moving_average_filter(
           polar_sampling_points_second ,
            window_size=filter_window_size
        )
        time_end = time.time()
        print(f"최종 이동평균 필터링 시간: {time_end - time_start}")
        
        # 4단계: 정규화된 랜드마크 계산
        landmark_points, arch_depth, molar_width = self.curve_sampler.compute_normalized_landmarks_and_arch_depth_molar_width(moving_average_points)
        
        # 6단계: 최종 시각화 (옵션)
        if visualize_result:
            self.visualizer.visualize_analysis_results(
                np.array([0,0,0]).reshape(1,3), 
                filtered_result_points_array, 
                moving_average_points, 
                self.curve_sampler.sample_points_by_arc_length(
                    moving_average_points, 
                    AnalysisConstants.DEFAULT_NUM_SAMPLES
                )
            )
        
        return arch_depth, molar_width, landmark_points
