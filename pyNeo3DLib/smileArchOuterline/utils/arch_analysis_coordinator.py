"""
치아 악궁 분석 프로세스를 조율하는 클래스
단일책임: 전체 분석 프로세스의 조율 및 통합
"""

import numpy as np
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


class ArchAnalysisCoordinator:
    """치아 악궁 분석 프로세스를 조율하는 클래스"""
    
    def __init__(self):
        self.curve_extractor = CurveExtractor()
        self.mesh_processor = MeshProcessor()
        self.visualizer = AnalysisVisualizer()
        self.signal_processor = SignalProcessor()
        self.curve_sampler = CurveSampler()
    
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
        final_mesh, final_points, new_center = self.mesh_processor.filter_and_center_mesh(
            rotated_mesh, smoothed_points
        )
        

        return final_points, final_mesh, new_center
    
    def extract_precise_curve_points(self, aligned_mesh: object, y_axis: np.ndarray) -> Tuple[np.ndarray, object, np.ndarray]:
        """
        정밀한 곡선 포인트를 추출합니다.
        
        Args:
            aligned_mesh: 정렬된 메쉬
            y_axis: Y축 벡터
            
        Returns:
            Tuple[moving_average_points, filtered_aligned_mesh, center_point]:
                - moving_average_points: 이동평균 처리된 포인트들
                - filtered_aligned_mesh: 필터링된 메쉬
                - center_point: 중심점
        """
        # 1단계: 등고선 포인트 추출
        filtered_aligned_mesh = self.curve_extractor.filter_mesh_by_z_threshold(
            aligned_mesh, y_axis
        )
   
        # 2단계: 극좌표 샘플링으로 곡선 추출
        z_min_point = min(filtered_aligned_mesh.points[:, 2])
        polar_sampling_points = self.curve_extractor.extract_curve_by_polar_sampling(
            filtered_aligned_mesh, z_min_point
        )

        
        # 3단계: 이동평균 필터링
        moving_average_points = self.signal_processor.moving_average_filter(
            polar_sampling_points,
            window_size=50
        )


        # curve tangent and normal calculation
        curve_tangent_normal_calculator = CurveTangentNormalCalculator()
        _, curve_normal = curve_tangent_normal_calculator.calculate_tangents_and_normals(moving_average_points)
        outer_expand_curve_points = moving_average_points - curve_normal * 5


        outer_expand_curve_points_min_point = np.min(outer_expand_curve_points[:, 0])
        outer_expand_curve_points_max_point = np.max(outer_expand_curve_points[:, 0])
        mask = (filtered_aligned_mesh.points[:, 0] > outer_expand_curve_points_min_point) & (filtered_aligned_mesh.points[:, 0] < outer_expand_curve_points_max_point)
        filtered_aligned_mesh = filtered_aligned_mesh.extract_points(mask)



        # 2단계: 극좌표 샘플링으로 곡선 추출
        z_min_point = min(filtered_aligned_mesh.points[:, 2])
        polar_sampling_points_second = self.curve_extractor.extract_curve_by_polar_sampling(
            filtered_aligned_mesh, z_min_point
        )

        #  유치악, 무치악 판단해서 필터링 계수 변경시켜야함

        # 80도에서 100도 사이의 포인트 추출 
        polar_sampler = PolarSampling(np.array([0, 0, z_min_point]))
        polar_sampling_points_80_100 = polar_sampler.polar_sampling(
            filtered_aligned_mesh.points,
            angle_step=1,
            mode="ymin",
            start_angle=80,
            end_angle=100,
            y_range=(-np.inf, np.inf)
        )

        polar_sampling_points_80_100_second = polar_sampler.polar_sampling(
            filtered_aligned_mesh.points,
            angle_step=1,
            mode="farthest",
            start_angle=80,
            end_angle=100,
            y_range=(-np.inf, np.inf)
        )

        # polar_sampling_points_80_100와 polar_sampling_points_80_100_second 의 RMSE 계산
        rmse = np.sqrt(np.mean((polar_sampling_points_80_100 - polar_sampling_points_80_100_second) ** 2))
        print(f"rmse: {rmse}")


        if rmse < 2:
            print("유치악")
            filter_window_size = 50
        else:
            print("무치악")
            filter_window_size = 20


        # 3단계: 이동평균 필터링
        moving_average_points = self.signal_processor.moving_average_filter(
           polar_sampling_points_second ,
            window_size=filter_window_size
        )


        return moving_average_points, filtered_aligned_mesh
    
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
        aligned_mesh, _, _, _, _ = self.mesh_processor.perform_initial_alignment(mesh_path)
        
        # 2단계: 정밀정렬 수행
        smoothed_points, rotated_mesh, _ = self.perform_precise_alignment(
            aligned_mesh, y_axis
        )


        
        # 3단계: 정밀한 곡선 포인트 추출
        precise_smoothed_points, precise_rotated_mesh = self.extract_precise_curve_points(
            rotated_mesh, y_axis
        )

        
        # 4단계: 정규화된 랜드마크 계산
        landmark_points, arch_depth, molar_width = self.curve_sampler.compute_normalized_landmarks_and_arch_depth_molar_width(smoothed_points)
        
        # 6단계: 최종 시각화 (옵션)
        if visualize_result:
            self.visualizer.visualize_analysis_results(
                np.array([0,0,0]).reshape(1,3), 
                precise_rotated_mesh.points, 
                precise_smoothed_points, 
                self.curve_sampler.sample_points_by_arc_length(
                    precise_smoothed_points, 
                    AnalysisConstants.DEFAULT_NUM_SAMPLES
                )
            )
        
        return arch_depth, molar_width, landmark_points
