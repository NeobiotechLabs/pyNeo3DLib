import numpy as np
from typing import Tuple
from ..utils import VisualizeForTest, PolarSampling, SplineGenerator, MeshProcessor


def subsample_points_uniformly(points: np.ndarray, target_count: int) -> np.ndarray:
    """
    주어진 점들을 균등하게 샘플링하여 목표 개수의 점을 반환합니다.
    첫 번째와 마지막 점을 포함하여 균등하게 분배합니다.
    
    Args:
        points: 샘플링할 포인트 클라우드 (numpy 배열, 형태: (n, 3))
        target_count: 목표 포인트 개수
        
    Returns:
        샘플링된 포인트들 (numpy 배열, 형태: (target_count, 3) 또는 (len(points), 3))
    """
    if len(points) < target_count:
        # 점의 개수가 target_count보다 적으면 모든 점 사용
        return np.array(points)
    else:
        # 첫 번째(0)와 마지막(-1) 인덱스를 포함하여 균등하게 분배
        indices = np.linspace(0, len(points) - 1, target_count, dtype=int)
        return np.array([points[i] for i in indices])


def perform_polar_sampling(vertices: np.ndarray, polar_center: np.ndarray, angle_step: float = 1, y_slice_mid: float = -1, y_offset: float = 0.5) -> np.ndarray:
    """극좌표 샘플링 수행
    
    입력된 3D 메쉬 버텍스에서 극좌표계 기반으로 외곽점과 내곽점을 샘플링하고, 
    두 결과의 평균을 계산하여 치아의 중간 라인을 추출합니다.

    입력:
        vertices (np.ndarray): 3D 메쉬의 버텍스 좌표 배열, 형태: (N, 3) [x, y, z]
        polar_center (np.ndarray): 극좌표계의 중심점 좌표, 형태: (3,) [x, y, z]
        angle_step (float, optional): 극좌표 샘플링 시 각도 간격(도). 기본값 1
        y_slice_mid (float, optional): Y축 슬라이스 중심 위치. 기본값 -1
        y_offset (float, optional): Y축 슬라이스 범위의 오프셋. 기본값 0.5

    출력:
        np.ndarray: 샘플링된 중간 라인 포인트들, 형태: (M, 3) [x, y, z]
                   외곽점과 내곽점의 평균으로 계산된 치아 아치의 중간 곡선을 나타냄
    """
    
    # 극좌표 샘플링 각도 범위
    START_ANGLE = 0
    END_ANGLE = 180
    
    y_range = (y_slice_mid - y_offset, y_slice_mid + y_offset)
    
    polar_sampler = PolarSampling(polar_center)
    
    # 외곽과 내곽 샘플링
    outer_points = polar_sampler.polar_sampling(
        vertices, 
        angle_step=angle_step, 
        mode="farthest", 
        y_range=y_range,
        start_angle=START_ANGLE, 
        end_angle=END_ANGLE
    )
    
    inner_points = polar_sampler.polar_sampling(
        vertices, 
        angle_step=angle_step, 
        mode="nearest", 
        y_range=y_range,
        start_angle=START_ANGLE, 
        end_angle=END_ANGLE
    )
    
    # 평균 계산
    return np.mean([outer_points, inner_points], axis=0)


def calculate_landmarks(half_curve):
    """
    악궁 곡선에 기반하여 랜드마크 포인트를 계산합니다.
    
    Args:
        aligned_half_left_curve: 정렬된 악궁 곡선 배열
        arch_depth: 아치 깊이 값
        molar_width: 구치 폭 값
        
    Returns:
        landmark_points: 계산된 랜드마크 포인트 딕셔너리
    """

    landmark_points = []

    # aligned_half_left_curve의 y축 최솟값과 최댓값 찾기
    z_min = np.min(half_curve[:, 2])
    z_max = np.max(half_curve[:, 2])

    # 5개의 등간격 z 좌표 생성
    target_z_coords = np.linspace(z_min, z_max, 5)

    for target_z in target_z_coords:
        # target_z에 가장 가까운 half_curve의 포인트 찾기
        diff = np.abs(half_curve[:, 2] - target_z)
        closest_index = np.argmin(diff)
        landmark_point = np.round(half_curve[closest_index, :], 2).reshape(1, 3)
        landmark_points.append(landmark_point)

    landmark_points = np.concatenate(landmark_points, axis=0)
    landmark_points = landmark_points[np.argsort(landmark_points[:, 0])]

    # 랜드마크 포인트 좌표 조정(DB 서치시 비교가 쉽도록)
    landmark_points[:,2] -= landmark_points[0,2]
    landmark_points[:,0] -= landmark_points[0,0]
    
    # 음수 제거 
    landmark_points = np.abs(landmark_points)
    # y값은 없애고 2차원배열화 (x, z만 남김)
    landmark_points = landmark_points[:, [0, 2]]


    # NumPy 배열을 리스트로 변환하고, 내부의 모든 요소를 Python 네이티브 타입으로 변환
    landmark_points_list = []
    for point in landmark_points:
        # NumPy float32/float64를 Python float으로 변환
        point_list = np.round([float(coord) for coord in point], 2).tolist()
        landmark_points_list.append(point_list)

    return landmark_points_list



def analyze_upper_IOS_scandata(
    mesh_path: str,
    angle_step: float = 1, # 극좌표 샘플링 각도 간격
    y_slice_mid: float = -1, # Y축 슬라이스 중심 위치
    y_offset: float = 0.5, # Y축 슬라이스 범위의 오프셋
    target_point_count: int = 7, # 스플라인을 위한 타겟 포인트 개수
    spline_num_points: int = 200, # 생성할 스플라인 곡선의 포인트 개수
    visualize_result: bool = True # 결과를 시각화할지 여부
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    치아 아치 곡선을 추출하고 시각화하는 메인 처리 함수
    
    Args:
        mesh_path (str): STL 메쉬 파일 경로
        angle_step (float): 극좌표 샘플링 각도 간격. 기본값 1
        y_slice_mid (float): Y축 슬라이스 중심 위치. 기본값 -1
        y_offset (float): Y축 슬라이스 범위 오프셋. 기본값 0.5
        target_point_count (int): 스플라인을 위한 타겟 포인트 개수. 기본값 7
        spline_num_points (int): 생성할 스플라인 곡선의 포인트 개수. 기본값 200
        visualize_result (bool): 결과를 시각화할지 여부. 기본값 True
        
    Returns:
        Tuple containing:
            - rotated_vertices: 회전된 메쉬 버텍스
            - polar_center: 극좌표 중심점
            - average_sampled_points: 평균 샘플링된 포인트들
            - total_sampled_points_for_spline: 스플라인용 샘플링된 포인트들
            - spline_curve: 생성된 스플라인 곡선
            - mesh_center_origin: 원점 좌표
    """
    
    # 메쉬 전처리 (중심 정렬, 회전, 극좌표 중심점 계산)
    processor = MeshProcessor(mesh_path)
    rotated_vertices, polar_center = processor.process_mesh()

    # 극좌표 샘플링
    average_sampled_points = perform_polar_sampling(
        rotated_vertices, 
        polar_center, 
        angle_step=angle_step, 
        y_slice_mid=y_slice_mid, 
        y_offset=y_offset
    )

    # 첫 번째와 마지막 인덱스를 포함하여 target_point_count 개의 점을 균등하게 샘플링
    total_sampled_points_for_spline = subsample_points_uniformly(average_sampled_points, target_point_count)

    # 스플라인 곡선 생성
    spline_generator_for_arch_curve = SplineGenerator()
    spline_curve = spline_generator_for_arch_curve.create_spline_curve(
        total_sampled_points_for_spline, 
        num_points=spline_num_points
    )

    # 악궁 곡선 처리
    z_max_index = np.argmax(spline_curve[:,2])
    half_curve = spline_curve[0:z_max_index]
    arch_depth = round(np.abs(spline_curve[z_max_index, 2]-spline_curve[0, 2]),2)
    molar_width = round(2*np.abs(spline_curve[z_max_index, 0]-spline_curve[0, 0]),2)
    landmarks = calculate_landmarks(half_curve)
    
    # 원점 좌표
    mesh_center_origin = np.array([[0, 0, 0]], dtype=np.float32)

    # 시각화
    if visualize_result:
        visualize = VisualizeForTest()
        visualize.visualize_points(mesh_center_origin, color='blue', point_size=10)
        visualize.visualize_points(rotated_vertices, color='pink', point_size=2)
        visualize.visualize_points(average_sampled_points, color='green', point_size=5)
        visualize.visualize_points(total_sampled_points_for_spline, color='red', point_size=15)
        visualize.visualize_points(spline_curve, color='yellow', point_size=5)
        visualize.show()
    
    return arch_depth, molar_width, landmarks


def main():
    """메인 실행 함수"""
    mesh_path = "./analyzing_IOS/data/Upper 안지숙님 편집.stl"
    
    # 치아 아치 곡선 처리 실행
    arch_depth, molar_width, landmarks = analyze_upper_IOS_scandata(
        mesh_path=mesh_path,
        visualize_result=True
    )

    print(f"arch_depth: {arch_depth}")
    print(f"molar_width: {molar_width}")
    print(f"landmarks: {landmarks}")


if __name__ == "__main__":
    main()




















