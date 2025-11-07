"""
치아 악궁 분석 프로세스를 조율하는 클래스 (리팩토링됨)
단일책임: 전체 분석 프로세스의 조율 및 통합
"""

import numpy as np
from typing import List, Tuple
from scipy.spatial import KDTree

from ..mesh_utils.mesh_processor import MeshProcessor
from ..visualizer_utils.analysis_visualizer import AnalysisVisualizer
from ..curve_utils.curve_sampler import CurveSampler
from .constants import AnalysisConstants
from .arch_analysis_pipeline import ArchAnalysisPipeline
from .performance_timer import PerformanceTimer


class ArchAnalysisCoordinator:
    """치아 악궁 분석 프로세스를 조율하는 클래스 (리팩토링됨)"""
    
    def __init__(self, enable_timing: bool = False):
        """
        초기화
        
        Args:
            enable_timing: 성능 측정 활성화 여부
        """
        self.mesh_processor = MeshProcessor()
        self.visualizer = AnalysisVisualizer()
        self.curve_sampler = CurveSampler()
        self.pipeline = ArchAnalysisPipeline()
        self.timer = PerformanceTimer(enabled=enable_timing)
    
    def _align_mesh_with_arch_centroid(self, mesh: object) -> object:
        """
        아치 중심점을 기준으로 메쉬를 추가 정렬합니다.
        
        극좌표 샘플링으로 추출한 곡선의 중심점을 계산하고,
        해당 중심점이 Z축 방향을 향하도록 메쉬를 회전시킵니다.
        
        Args:
            mesh: 회전시킬 메쉬
            
        Returns:
            object: 중심점 기준으로 정렬된 메쉬
        """
        z_min = np.min(mesh.points[:, 2])
        
        # 1단계: 곡선 추출 및 스무딩
        curve = self.pipeline.extract_initial_curve(mesh.points, z_min)
        smoothed_curve = self.pipeline.smooth_curve(curve)
        
        # 2단계: 곡선의 시작점과 끝점을 잇는 직선 생성
        if smoothed_curve.shape[0] < 2:
            print("smoothed_curve가 2개 미만의 포인트를 가지고 있습니다. 방향을 계산할 수 없습니다. 기본 방향을 사용합니다.")
            direction = np.array([1.0, 0.0, 0.0])  # 기본 방향 벡터
        else:
            direction = smoothed_curve[AnalysisConstants.LAST_INDEX] - smoothed_curve[AnalysisConstants.FIRST_ELEMENT_INDEX]
            norm_direction = np.linalg.norm(direction)
            if norm_direction == 0:
                print("norm_direction이 0입니다. 방향을 정규화할 수 없습니다. 기본 방향을 사용합니다.")
                direction = np.array([1.0, 0.0, 0.0]) # 기본 방향 벡터
        
        line_points = np.linspace(smoothed_curve[AnalysisConstants.FIRST_ELEMENT_INDEX], smoothed_curve[AnalysisConstants.LAST_INDEX], AnalysisConstants.LINSPACE_NUM_POINTS)
        line_points = line_points + direction * AnalysisConstants.DIRECTION_OFFSET_FACTOR
        
        # 3단계: 곡선의 중심점 계산
        centroid = np.mean(smoothed_curve, axis=0).reshape(AnalysisConstants.SINGLE_ROW_SHAPE, AnalysisConstants.VECTOR_DIMENSION)
        
        # 4단계: KDTree를 사용해 중심점에서 가장 가까운 선상의 점 찾기
        kdtree = KDTree(line_points)
        _, idx = kdtree.query(centroid, k=1)
        closest_point = line_points[idx].reshape(AnalysisConstants.VECTOR_DIMENSION)
        
        # 5단계: 선상의 점에서 중심점으로 가는 방향벡터 계산
        centroid_flat = centroid.reshape(AnalysisConstants.VECTOR_DIMENSION)
        direction_to_centroid = centroid_flat - closest_point
        direction_to_centroid = direction_to_centroid / np.linalg.norm(direction_to_centroid)
        
        # 6단계: [0,0,1]과 방향벡터가 이루는 각도 계산 및 Y축 기준 회전
        angle = -np.arccos(np.dot(AnalysisConstants.Z_AXIS_VECTOR_POSITIVE, direction_to_centroid))
        rotation_matrix = np.array([
            [np.cos(angle), AnalysisConstants.ROTATION_MATRIX_ZERO, np.sin(angle)],
            [AnalysisConstants.ROTATION_MATRIX_ZERO, AnalysisConstants.ROTATION_MATRIX_ONE, AnalysisConstants.ROTATION_MATRIX_ZERO],
            [-np.sin(angle), AnalysisConstants.ROTATION_MATRIX_ZERO, np.cos(angle)]
        ])
        mesh.points = mesh.points @ rotation_matrix
        
        return mesh
    
    def perform_precise_alignment(
        self, 
        aligned_mesh: object, 
        y_axis: np.ndarray
    ) -> Tuple[object, np.ndarray, np.ndarray]:
        """
        정밀정렬을 수행합니다.
        
        Args:
            aligned_mesh: 1차 정렬된 메쉬
            y_axis: Y축 벡터
            
        Returns:
            Tuple[final_mesh, centered_points, new_center]:
                - final_mesh: 정밀정렬된 메쉬
                - centered_points: 중심 맞춤된 곡선 포인트들
                - new_center: 새로운 중심점
        """
        with self.timer.measure("등고선 포인트 추출"):
            contour_points, filtered_mesh = self.pipeline.curve_extractor.extract_contour_points(
                aligned_mesh, y_axis
            )
        
        with self.timer.measure("곡선 샘플링"):
            avg_y = self.pipeline.signal_processor.calculate_average_y_value(contour_points)
            z_min = np.min(filtered_mesh.points[:, 2])
            curve_points = self.pipeline.curve_extractor.extract_curve_by_curve_sampling(
                filtered_mesh, avg_y, z_min
            )
        
        with self.timer.measure("곡선 스무딩"):
            smoothed = self.pipeline.smooth_curve(curve_points)
        
        with self.timer.measure("메시 방향 정렬"):
            rotated_mesh, smoothed, _ = self.mesh_processor.align_mesh_direction(
                aligned_mesh, smoothed
            )
        
        with self.timer.measure("메시 중심 맞춤"):
            final_mesh, centered_points, new_center = self.mesh_processor.filter_and_center_mesh(
                rotated_mesh, smoothed
            )
        
        with self.timer.measure("아치 중심점 기준 정렬"):
            final_mesh = self._align_mesh_with_arch_centroid(final_mesh)
        
        return final_mesh, centered_points, new_center
    

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
        y_axis = np.array(AnalysisConstants.Y_AXIS_VECTOR)
        
        try:
            # 1단계: 1차 정렬 수행
            with self.timer.measure("1차 정렬"):
                aligned_mesh, _, _, _, _ = self.mesh_processor.perform_initial_alignment(mesh_path)
            
            if aligned_mesh is None:
                raise ValueError(f"Initial alignment failed for mesh: {mesh_path}")

            # 2단계: 정밀정렬 수행
            with self.timer.measure("정밀정렬"):
                aligned_mesh, _, _ = self.perform_precise_alignment(aligned_mesh, y_axis)
            
            # 3단계: 대구치 아웃라이어 제거
            with self.timer.measure("대구치 아웃라이어 제거"):
                mesh_points = self.pipeline.remove_outliers(aligned_mesh.points)
            
            if mesh_points.shape[0] == 0:
                raise ValueError("No points remaining after outlier removal.")

            # 4단계: 초기 곡선 추출 및 스무딩
            z_min = np.min(mesh_points[:, 2])
            with self.timer.measure("초기 곡선 추출 및 스무딩"):
                initial_curve = self.pipeline.extract_initial_curve(mesh_points, z_min)
                smoothed_curve = self.pipeline.smooth_curve(initial_curve)
            
            if smoothed_curve.shape[0] == 0:
                raise ValueError("Failed to extract or smooth initial curve.")

            # 5단계: 곡선 외측 확장 및 메시 필터링
            with self.timer.measure("곡선 기반 메시 필터링"):
                expanded_curve = self.pipeline.expand_curve_outward(smoothed_curve)
                filtered_points = self.pipeline.filter_mesh_by_curve_boundary(
                    mesh_points, expanded_curve
                )
            
            if filtered_points.shape[0] == 0:
                raise ValueError("No points remaining after curve-based mesh filtering.")

            # 6단계: 치아 타입 분류 및 최종 곡선 추출
            z_min = np.min(filtered_points[:, 2])
            with self.timer.measure("치아 타입 분류 및 최종 곡선 추출"):
                final_curve, arch_type, _, rmse = \
                    self.pipeline.classify_and_extract_final_curve(filtered_points, z_min)
            
            if final_curve.shape[0] == 0:
                raise ValueError("Failed to classify or extract final curve.")

            print(f"악궁 타입: {arch_type} (RMSE: {rmse:.4f})")
            
            # 7단계: 랜드마크 계산
            with self.timer.measure("랜드마크 계산"):
                landmark_points, arch_depth, molar_width = \
                    self.curve_sampler.compute_normalized_landmarks_and_arch_depth_molar_width(
                        final_curve
                    )
            
            # 8단계: 시각화 (옵션)
            if visualize_result:
                with self.timer.measure("시각화"):
                    sampled_points = self.curve_sampler.sample_points_by_arc_length(
                        final_curve, 
                        AnalysisConstants.DEFAULT_NUM_SAMPLES
                    )
                    self.visualizer.visualize_analysis_results(
                        np.array(AnalysisConstants.ORIGIN_POINT).reshape(AnalysisConstants.SINGLE_ROW_SHAPE, AnalysisConstants.VECTOR_DIMENSION),
                        filtered_points,
                        final_curve,
                        sampled_points
                    )
            
            # 성능 측정 요약 출력
            self.timer.print_summary()
            
            return arch_depth, molar_width, landmark_points
        except Exception as e:
            print(f"Error in analyze_upper_IOS_scandata: {e}")
            # 에러 발생 시 기본값 
            return 0.0, 0.0, [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
