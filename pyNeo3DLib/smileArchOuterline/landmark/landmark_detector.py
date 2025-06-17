import numpy as np
import scipy.interpolate as interpolate
from pyNeo3DLib.smileArchOuterline.visualize_utils.visualize_test import VisualizeForTest
from pyNeo3DLib.smileArchOuterline.mesh_utils.teeth_templete_loader import MeshLoader
from typing import List

class SmileArchOuterlineDetector:
    def __init__(self):
        """
        스마일 아치 랜드마크 감지기 클래스 초기화
        """
        pass
    
    def load_mesh(self, file_path):
        """
        STL 파일에서 메쉬를 로드하는 함수
            
        Args:
            file_path: STL 파일 경로
                
        Returns:
            pyvista.PolyData: PyVista 메쉬 객체
        """
        loader = MeshLoader(file_path)
        loader.load()
        _, pv_mesh = loader.to_pyvista()
        return pv_mesh  

    def _align_smile_arch_mesh(self, mesh, pin_height=2):
        """
        스마일 아치 메시를 정렬하는 함수
        
        Args:
            mesh: 정렬할 원본 메시
        
        Returns:
            rotated_mesh: 정렬된 메시
        """
        mesh_vertices = np.array(mesh.points).reshape(-1, 3)
        
        # 스마일 아치 트레이 부분만 추출
        filtered_mesh_vertices = self._extract_smile_arch_tray(mesh)

        # 스마일아치 트레이 고정핀 찾기
        min_z_point = filtered_mesh_vertices[np.argmin(filtered_mesh_vertices[:, 2])]
        pin_vertices = filtered_mesh_vertices[filtered_mesh_vertices[:, 2] < min_z_point[2]+pin_height]

        # pin_vertices에서 x_min 인덱스, x_max 인덱스, y_min 인덱스, y_max 인덱스 찾기
        x_min_index = np.argmin(pin_vertices[:, 0])
        x_max_index = np.argmax(pin_vertices[:, 0])
        y_min_index = np.argmin(pin_vertices[:, 1])
        y_max_index = np.argmax(pin_vertices[:, 1])

        x_min_point = pin_vertices[x_min_index]
        x_max_point = pin_vertices[x_max_index]
        y_min_point = pin_vertices[y_min_index]
        y_max_point = pin_vertices[y_max_index]

        # 방향 벡터 계산
        direction_vector_ymin_to_ymax = y_max_point - y_min_point
        direction_vector_ymin_to_ymax = direction_vector_ymin_to_ymax / np.linalg.norm(direction_vector_ymin_to_ymax)
        direction_vector_ymin_to_ymax = direction_vector_ymin_to_ymax.reshape(1, 3)

        direction_vector_ymin_to_xmax = x_max_point - y_min_point
        direction_vector_ymin_to_xmax = direction_vector_ymin_to_xmax / np.linalg.norm(direction_vector_ymin_to_xmax)
        direction_vector_ymin_to_xmax = direction_vector_ymin_to_xmax.reshape(1, 3)

        direction_vector_ymin_to_xmin = x_min_point - y_min_point
        direction_vector_ymin_to_xmin = direction_vector_ymin_to_xmin / np.linalg.norm(direction_vector_ymin_to_xmin)
        direction_vector_ymin_to_xmin = direction_vector_ymin_to_xmin.reshape(1, 3)

        # 좌표계 벡터 계산
        pin_up_vector = np.cross(direction_vector_ymin_to_xmax, direction_vector_ymin_to_xmin)
        pin_up_vector = pin_up_vector / np.linalg.norm(pin_up_vector)
        pin_up_vector = pin_up_vector.reshape(1, 3)

        pin_forward_vector = direction_vector_ymin_to_ymax 

        pin_right_vector = np.cross(pin_forward_vector, pin_up_vector)
        pin_right_vector = pin_right_vector / np.linalg.norm(pin_right_vector)
        pin_right_vector = pin_right_vector.reshape(1, 3)

        # 회전 행렬 생성
        rotation_matrix = np.vstack([pin_right_vector, pin_forward_vector, pin_up_vector]).T

        # 메시 회전
        rotated_mesh = mesh.copy()
        rotated_mesh.rotate_x(0)  # 회전 행렬을 직접 적용하기 위한 임시 방법
        rotated_points = np.dot(np.array(mesh.points), rotation_matrix)
        rotated_mesh.points = rotated_points
        
        return rotated_mesh

    def _calculate_smile_arch_measurements(self, spline_curve):
        """
        스마일 아치 스플라인 곡선에서 주요 측정값을 계산하는 함수
        
        Args:
            spline_curve: 생성된 스플라인 곡선 포인트 배열
        
        Returns:
            tuple: (arch_depth, molar_width, landmark_points) - 아치 깊이, 대구치 너비, 랜드마크 포인트 배열
        """
        # 측정값 계산
        arch_depth, molar_width = self._calculate_smile_arch_size(spline_curve)
        landmark_points = self._find_landmark_points(spline_curve)
        
        return arch_depth, molar_width, landmark_points

    def _export_landmark_points(self, landmark_points):
        """
        랜드마크 포인트를 출력 형식으로 변환하는 함수
        
        Args:
            landmark_points: 원본 랜드마크 포인트 배열
        
        Returns:
            list: 2D 좌표로 변환된 랜드마크 포인트 배열
        """
        landmark_points_for_export = []
        for point in landmark_points:
            landmark_points_for_export.append([float(round(abs(point[0]), 2)), float(round(abs(point[1]), 2))])
        
        return landmark_points_for_export


    def _print_smile_arch_results(self, arch_depth, molar_width, landmark_points):
        """
        스마일 아치 분석 결과를 콘솔에 출력하는 함수
        
        Args:
            arch_depth: 계산된 아치 깊이
            molar_width: 계산된 대구치 너비
            landmark_points: 계산된 랜드마크 포인트
        """
        print(f"대구치 너비 (Molar Width): {molar_width:.2f} mm")
        print(f"아치 깊이 (Arch Depth): {arch_depth:.2f} mm")
        print("랜드마크 포인트:")
        for i, point in enumerate(landmark_points):
            print(f"  포인트 {i+1}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")

    def _visualize_smile_arch(self, mesh, spline_curve, landmark_points):
        """
        스마일 아치 분석 결과를 시각화하는 함수
        
        Args:
            mesh: 정렬된 메쉬
            spline_curve: 생성된 스플라인 곡선
            landmark_points: 계산된 랜드마크 포인트
        """
        visualizer = VisualizeForTest()
        visualizer.visualize_mesh(mesh, color="white", opacity=1)
        visualizer.visualize_points(spline_curve, color="blue", point_size=3)
        visualizer.visualize_points(landmark_points, color="red", point_size=10)
        visualizer.show()

    def visualize_smile_arch_analysis(self, mesh, spline_curve, landmark_points):
        """
        스마일 아치 분석 결과를 출력하고 시각화하는 함수
        
        Args:
            mesh: 정렬된 메쉬
            spline_curve: 생성된 스플라인 곡선
            landmark_points: 계산된 랜드마크 포인트
            
        Returns:
            tuple: (arch_depth, molar_width) - 아치 깊이, 대구치 너비
        """
        # 대구치 너비, 아치 깊이 계산
        arch_depth, molar_width = self._calculate_smile_arch_size(spline_curve)
        
        # 결과 출력
        self._print_smile_arch_results(arch_depth, molar_width, landmark_points)
        
        # 시각화
        self._visualize_smile_arch(mesh, spline_curve, landmark_points)
        
        return arch_depth, molar_width

    def analyze_smile_arch(self, smile_arch_mesh):
        """
        메쉬 파일을 분석하여 스마일 아치 측정값을 계산하는 함수
        
        Args:
            mesh_file_path: 분석할 STL 파일 경로
            
        Returns:
            tuple: (arch_depth, molar_width, landmark_points, spline_curve) - 측정 결과
        """

        # 스마일아치 트레이 찾기
        filtered_mesh_vertices = self._extract_smile_arch_tray(smile_arch_mesh)

        # 평탄화 및 극좌표 샘플링 준비
        flattened_to_zeroZ_mesh_vertices = self._flatten_vertices_to_z_plane(filtered_mesh_vertices)
        polar_sampling_center_point = self._calculate_polar_sampling_center(flattened_to_zeroZ_mesh_vertices)

        # 극좌표 샘플링 수행
        smile_arch_outline = self._polar_coordinate_sampling(flattened_to_zeroZ_mesh_vertices, 
                                                            num_intervals=1000, 
                                                            center_point=polar_sampling_center_point)
        
        # 스플라인 곡선 생성
        control_points = self._generate_spline_control_points(smile_arch_outline)
        spline_curve = self._generate_spline_curve(control_points)

        # 측정값 계산
        arch_depth, molar_width, landmark_points = self._calculate_smile_arch_measurements(spline_curve)

        exported_landmark_points = self._export_landmark_points(landmark_points)

        # 결과 반환
        return arch_depth, molar_width, exported_landmark_points

    # 아래부터는 내부적으로만 사용되는 private 메서드들

    def _extract_smile_arch_tray(self, mesh, z_range_span=10):
        """
        스마일 아치 트레이 부분만 추출하는 함수
        
        Args:
            mesh: PyVista 메시 객체
            z_range_span: Z축 범위 크기
        
        Returns:
            filtered_mesh_vertices: 스마일 아치 트레이 부분만 추출된 정점 배열
        """
        # 메시에서 정점 배열 추출
        mesh_vertices = np.array(mesh.points).reshape(-1, 3)
        
        z_range_min = np.min(mesh_vertices[:, 2])
        z_range_max = z_range_span + z_range_min

        filtered_mesh_vertices = mesh_vertices[mesh_vertices[:, 2] >= z_range_min]
        filtered_mesh_vertices = filtered_mesh_vertices[filtered_mesh_vertices[:, 2] <= z_range_max]
        
        return filtered_mesh_vertices

    def _polar_coordinate_sampling(self, vertices, num_intervals=100, center_point=None):
        """
        극좌표 샘플링 함수
        
        Args:
            vertices: 샘플링할 정점 배열 (z=0으로 평탄화된 상태)
            num_intervals: 각도 구간 수 (기본값: 100)
            center_point: 극좌표의 중심점 (기본값: None, None인 경우 원점(0,0,0) 사용)
        
        Returns:
            sampled_vertices: 샘플링된 정점 배열
        """
        # 중심점 설정 (기본값: 원점)
        if center_point is None:
            center_point = np.array([0, 0, 0])
        
        # 중심점으로부터의 상대 좌표 계산
        relative_vertices = vertices - center_point
        
        # 중심점으로부터의 2D 거리 계산
        distances = np.sqrt(relative_vertices[:, 0]**2 + relative_vertices[:, 1]**2)
        
        # 각도 계산 (라디안)
        angles = np.arctan2(relative_vertices[:, 1], relative_vertices[:, 0])
        
        # 각도를 0~2π 범위로 조정 (360도)
        angles = np.where(angles < 0, angles + 2*np.pi, angles)
        
        # 각도 구간 정의
        interval_size = 2*np.pi / num_intervals
        interval_indices = np.floor(angles / interval_size).astype(int)
        
        # 각 구간에서 가장 먼 정점 찾기
        sampled_vertices = []
        
        for i in range(num_intervals):
            interval_mask = (interval_indices == i)
            if np.any(interval_mask):
                # 해당 구간에 정점이 있는 경우
                max_dist_idx = np.argmax(distances[interval_mask])
                point_indices = np.where(interval_mask)[0]
                sampled_vertices.append(vertices[point_indices[max_dist_idx]])
        
        return np.array(sampled_vertices)

    def _calculate_initial_control_points(self, smile_arch_outline):
        """
        스마일 아치에서 중요 랜드마크 포인트를 계산하는 함수
        
        Args:
            smile_arch_outline: 극좌표 샘플링된 스마일 아치 외곽선 점들
        
        Returns:
            tuple: (max_x_point, min_x_point, max_y_point) - 계산된 랜드마크 포인트
        """
        max_x_index = np.argmax(smile_arch_outline[:, 0])
        max_x_point = smile_arch_outline[max_x_index].copy()  # 복사본 생성

        min_x_index = np.argmin(smile_arch_outline[:, 0])
        min_x_point = smile_arch_outline[min_x_index].copy()  # 복사본 생성

        # x, y 좌표의 절댓값 계산
        abs_x_max = abs(max_x_point[0])
        abs_x_min = abs(min_x_point[0])
        abs_y_max = abs(max_x_point[1])
        abs_y_min = abs(min_x_point[1])

        # 절댓값의 평균 계산
        avg_abs_x = (abs_x_max + abs_x_min) / 2
        avg_abs_y = (abs_y_max + abs_y_min) / 2

        # max_x_point를 평균값으로 대체
        max_x_point[0] = avg_abs_x
        max_x_point[1] = -avg_abs_y  # 음수로 설정

        # min_x_point를 x=0 기준으로 대칭 (x 좌표만 부호 반전)
        min_x_point[0] = -avg_abs_x
        min_x_point[1] = -avg_abs_y  # 음수로 설정

        max_y_index = np.argmax(smile_arch_outline[:, 1])
        max_y_point = smile_arch_outline[max_y_index].copy()  # 복사본 생성
        max_y_point[0] = 0
        
        return max_x_point, min_x_point, max_y_point

    def _find_max_distance_point(self, point1, point2, outline_vertices):
        """
        두 점 사이의 영역에서 직선과 최대 거리를 가지는 점을 찾는 함수
        
        Args:
            point1: 첫 번째 점 (numpy array)
            point2: 두 번째 점 (numpy array)
            outline_vertices: 외곽선 정점 배열 (numpy array)
        
        Returns:
            max_distance_point: 직선과 최대 거리를 가지는 점
        """
        # x 좌표의 범위 계산
        x_min = min(point1[0], point2[0])
        x_max = max(point1[0], point2[0])
        
        # x 범위 내에 있는 정점들 필터링
        filtered_vertices = outline_vertices[(outline_vertices[:, 0] >= x_min) & 
                                             (outline_vertices[:, 0] <= x_max)]
        
        if len(filtered_vertices) == 0:
            return None
        
        # 두 점을 지나는 직선의 방정식 계산
        # 직선 방정식: ax + by + c = 0
        a = point2[1] - point1[1]
        b = point1[0] - point2[0]
        c = point2[0] * point1[1] - point1[0] * point2[1]
        
        # 각 정점에서 직선까지의 거리 계산
        # 거리 = |ax + by + c| / sqrt(a^2 + b^2)
        distances = np.abs(a * filtered_vertices[:, 0] + b * filtered_vertices[:, 1] + c) / np.sqrt(a**2 + b**2)
        
        # 최대 거리를 가진 정점 찾기
        max_distance_index = np.argmax(distances)
        max_distance_point = filtered_vertices[max_distance_index]
        
        return max_distance_point

    def _extract_vertices_on_plane(self, vertices, start_point, end_point, tolerance=0.1):
        """
        두 점으로 정의된 방향과 [0,0,1] 벡터의 외적으로 정의된 평면 위에 존재하는 버텍스들을 추출하는 함수
        평면의 normal_vector 방향에 있는 버텍스들만 추출
        
        Args:
            vertices: 전체 버텍스 배열 (numpy array)
            start_point: 시작점 (numpy array)
            end_point: 끝점 (numpy array)
            tolerance: 평면에 있다고 간주할 거리 임계값 (기본값: 0.1)
        
        Returns:
            plane_vertices: 평면 위에 있는 버텍스들의 배열
        """
        # 방향 벡터 계산
        direction_vector = end_point - start_point
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        # [0,0,1] 벡터
        up_vector = np.array([0, 0, 1])
        
        # 외적 계산으로 평면의 법선 벡터 구하기
        normal_vector = np.cross(up_vector,direction_vector)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        
        # 평면의 방정식 계수 구하기 (Ax + By + Cz + D = 0)
        A, B, C = normal_vector
        D = -np.dot(normal_vector, start_point)
        
        # 평면 방정식을 이용하여 법선 벡터 방향(양수 방향)에 있는 버텍스 찾기
        # 평면 방정식에서 Ax + By + Cz + D의 부호로 법선 벡터의 방향을 결정
        signed_distances = A * vertices[:, 0] + B * vertices[:, 1] + C * vertices[:, 2] + D
        
        # 법선 벡터 방향에 있는 버텍스만 선택 (양수 값)
        positive_side_vertices = vertices[signed_distances > 0]
        
        # 거리가 임계값 이하인 버텍스들 중에서 법선 벡터 방향에 있는 버텍스 추출
        absolute_distances = np.abs(signed_distances)
        normal_direction_vertices = vertices[(signed_distances > 0) & (absolute_distances >= tolerance)]
        
        return normal_direction_vertices

    def _find_max_distance_control_point(self, start_point = None, end_point = None, outline_vertices = None):
        """
        두 점 사이의 영역에서 필터링된 정점 중 직선과 최대 거리를 가지는 점을 찾는 함수
        
        Args:
            start_point: 시작점 (numpy array)
            end_point: 끝점 (numpy array)
            outline_vertices: 외곽선 정점 배열 (numpy array)
        
        Returns:
            max_distance_point: 직선과 최대 거리를 가지는 점
        """
        # 평면에 있는 정점 추출
        filtered_vertices = self._extract_vertices_on_plane(outline_vertices, start_point, end_point)
        
        # 최대 거리 점 찾기
        max_distance_point = self._find_max_distance_point(start_point, end_point, filtered_vertices)
        
        return max_distance_point

    def _normalize_points(self, point1, point2):
        """
        두 점의 x, y 좌표값을 절댓값의 평균으로 정규화하는 함수
        
        Args:
            point1: 첫 번째 점 (numpy array)
            point2: 두 번째 점 (numpy array)
        
        Returns:
            tuple: (normalized_point1, normalized_point2) - 정규화된 두 점
        """
        # 각 좌표의 절댓값 계산
        abs_x1 = abs(point1[0])
        abs_x2 = abs(point2[0])
        abs_y1 = abs(point1[1])
        abs_y2 = abs(point2[1])
        
        # 절댓값의 평균 계산
        avg_abs_x = (abs_x1 + abs_x2) / 2
        avg_abs_y = (abs_y1 + abs_y2) / 2
        
        # 점 복사
        normalized_point1 = point1.copy()
        normalized_point2 = point2.copy()
        
        # 첫 번째 점 정규화
        normalized_point1[0] = avg_abs_x if point1[0] >= 0 else -avg_abs_x
        normalized_point1[1] = avg_abs_y if point1[1] >= 0 else -avg_abs_y
        
        # 두 번째 점 정규화
        normalized_point2[0] = avg_abs_x if point2[0] >= 0 else -avg_abs_x
        normalized_point2[1] = avg_abs_y if point2[1] >= 0 else -avg_abs_y
        
        return normalized_point1, normalized_point2

    def _generate_spline_control_points(self, smile_arch_outline):
        """
        스마일 아치 외곽선에서 스플라인 곡선을 위한 제어점들을 생성하는 함수
        
        Args:
            smile_arch_outline: 극좌표 샘플링된 스마일 아치 외곽선 점들
        
        Returns:
            list: x값 기준으로 오름차순 정렬된 제어점 리스트
        """
        # 초기 제어점 계산
        control_point3, control_point1, control_point2 = self._calculate_initial_control_points(smile_arch_outline)
        
        # 1차 추가 제어점 계산
        control_point1_2 = self._find_max_distance_control_point(start_point=control_point1, end_point=control_point2, outline_vertices=smile_arch_outline)
        control_point2_3 = self._find_max_distance_control_point(start_point=control_point2, end_point=control_point3, outline_vertices=smile_arch_outline)
        
        # 제어점 정규화
        control_point1_2, control_point2_3 = self._normalize_points(control_point1_2, control_point2_3)
        control_point1, control_point3 = self._normalize_points(control_point1, control_point3)
        
        # 2차 추가 제어점 계산
        control_point2_23 = self._find_max_distance_control_point(start_point=control_point2, end_point=control_point2_3, outline_vertices=smile_arch_outline)
        control_point12_2 = self._find_max_distance_control_point(start_point=control_point1_2, end_point=control_point2, outline_vertices=smile_arch_outline)
        
        # 2차 제어점 정규화
        control_point12_2, control_point2_23 = self._normalize_points(control_point12_2, control_point2_23)
        
        # 3차 추가 제어점 계산
        control_point1_12 = self._find_max_distance_control_point(start_point=control_point1, end_point=control_point1_2, outline_vertices=smile_arch_outline)
        control_point23_3 = self._find_max_distance_control_point(start_point=control_point2_3, end_point=control_point3, outline_vertices=smile_arch_outline)
        
        # 3차 제어점 정규화
        control_point1_12, control_point23_3 = self._normalize_points(control_point1_12, control_point23_3)
        
        # 모든 제어점을 리스트에 추가
        control_points = [
            control_point1,
            control_point1_12,
            control_point1_2,
            control_point12_2,
            control_point2,
            control_point2_23,
            control_point2_3,
            control_point23_3,
            control_point3
        ]
        
        # x 좌표 기준으로 오름차순 정렬
        control_points.sort(key=lambda point: point[0])
        
        return control_points

    def _generate_spline_curve(self, control_points, num_points=1000):
        """
        제어점을 기반으로 스플라인 곡선을 생성하는 함수
        
        Args:
            control_points: 제어점 배열
            num_points: 생성할 곡선 포인트 수
        
        Returns:
            spline_points: 스플라인 곡선 포인트 배열
        """
        # 제어점 좌표 분리
        x = [p[0] for p in control_points]
        y = [p[1] for p in control_points]
        z = [p[2] for p in control_points]
        
        # 누적 거리를 기반으로 매개변수 생성
        distances = np.zeros(len(control_points))
        for i in range(1, len(control_points)):
            distances[i] = distances[i-1] + np.linalg.norm(
                np.array(control_points[i]) - np.array(control_points[i-1]))
        
        # 정규화된 매개변수
        u = distances / distances[-1]
        
        # 스플라인 생성 (3차 스플라인)
        tck, _ = interpolate.splprep([x, y, z], u=u, s=0, k=3)
        
        # 균일하게 분포된 매개변수 값에서 스플라인 평가
        u_new = np.linspace(0, 1, num_points)
        spline_points = np.column_stack(interpolate.splev(u_new, tck))
        
        return spline_points

    def _calculate_smile_arch_size(self, spline_curve):
        """
        스플라인 곡선에서 대구치 너비를 계산하는 함수
        
        Args:
            spline_curve: 생성된 스플라인 곡선 포인트 배열
        
        Returns:
            float: 계산된 대구치 너비 (mm)
        """
        # 스플라인 곡선 첫 번째 포인트의 x 값 절댓값의 2배
        y_max = np.max(spline_curve[:, 1])
        arch_depth = abs(y_max - abs(spline_curve[0][1]))
        molar_width = abs(spline_curve[0][0]) * 2
        
        return arch_depth, molar_width

    def _find_landmark_points(self, spline_curve):
        """
        아치 깊이를 4등분하여 주요 랜드마크 포인트를 찾는 함수
        
        Args:
            spline_curve: 생성된 스플라인 곡선 포인트 배열
        
        Returns:
            list: 랜드마크 포인트 배열 [pt1, pt2, pt3, pt4, pt5]
        """
        # y 범위 계산 (스플라인의 첫 포인트 y 값부터 y_max까지의 구간을 4등분)
        half_spline_curve = spline_curve[spline_curve[:, 0] <= 0]

        y_start = np.max(half_spline_curve[:, 1])
        y_end = np.min(half_spline_curve[:, 1])
        y_step = (y_start - y_end) / 4  # 4등분
        
        landmark_points = []
        
        # 첫 번째 랜드마크는 스플라인의 첫 포인트
        landmark_points.append([0, y_start, 0])
        
        # 나머지 4개의 랜드마크 포인트 찾기
        for i in range(1, 5):
            target_y = y_start - (y_step * i)
            
            # 타겟 y값에 가장 가까운 스플라인 포인트 찾기
            closest_idx = np.argmin(np.abs(half_spline_curve[:, 1] - target_y))
            landmark_points.append(half_spline_curve[closest_idx])
        
        return landmark_points

    def _flatten_vertices_to_z_plane(self, vertices):
        """
        정점 배열을 z=0 평면으로 평탄화하는 함수
        
        Args:
            vertices: 원본 정점 배열 (numpy array)
        
        Returns:
            flattened_vertices: z=0으로 평탄화된 정점 배열
        """
        flattened_vertices = vertices.copy()
        flattened_vertices[:, 2] = 0
        flattened_vertices = flattened_vertices.reshape(-1, 3)
        
        return flattened_vertices

    def _calculate_polar_sampling_center(self, vertices):
        """
        극좌표 샘플링의 중심점을 계산하는 함수
        
        Args:
            vertices: 평탄화된 정점 배열 (numpy array)
        
        Returns:
            numpy.ndarray: 극좌표 샘플링을 위한 중심점 [0, y_center, 0]
        """
        y_max = np.max(vertices[:, 1])
        y_min = np.min(vertices[:, 1])
        y_center = (y_max + y_min) / 2
        polar_sampling_center_point = np.array([0, y_center, 0])
        
        return polar_sampling_center_point


def main():
    """
    스마일 아치 랜드마크 감지 테스트를 위한 메인 함수
    """
    # 랜드마크 감지기 인스턴스 생성
    detector = SmileArchOuterlineDetector()
    
    # 메쉬 파일 분석
    mesh_file_path = "aligned_maxillary_smilearch.stl"
    arch_depth, molar_width, landmark_points, spline_curve, rotated_mesh = detector.analyze_smile_arch(mesh_file_path)
    
    # 랜드마크 포인트 내보내기 형식으로 변환
    exported_landmarks = detector.export_landmark_points(landmark_points)
    
    # 결과 시각화
    detector.visualize_smile_arch_analysis(rotated_mesh, spline_curve, landmark_points)

if __name__ == "__main__":
    main()






















