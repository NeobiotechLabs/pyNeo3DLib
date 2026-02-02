"""
CBCT-FaceScan 정합 설정 관리
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class CBCTExtractionConfig:
    """CBCT 표면 추출 설정"""
    z_crop_top_ratio: float = 0.1  # 상부 제거 비율 (머리 윗부분)
    z_crop_bottom_ratio: float = 0.4  # 하부 제거 비율 (턱/목 아래)
    downsample_factor: int = 4  # 각 축 원본 해상도


@dataclass
class NoseEstimationConfig:
    """코 중심 추정 설정"""
    x_center_ratio_start: float = 0.35
    x_center_ratio_end: float = 0.65


@dataclass
class DepthMapConfig:
    """Depth Map 추출 설정"""
    grid_width_mm: float = 80.0
    grid_height_mm: float = 100.0
    grid_resolution: Tuple[int, int] = (50, 50)
    ray_direction: Tuple[float, float, float] = (0, -1, 0)  # Y- 방향
    ray_start_offset_mm: float = 150.0
    search_radius_mm: float = 3.0


@dataclass
class MeshSamplingConfig:
    """메쉬 샘플링 설정"""
    num_samples: int = 100_000  # 샘플링할 포인트 개수
    top_percent: float = 0.01  # 상위 포인트 비율 (코 끝 추정용)


@dataclass
class ICPConfig:
    """ICP 정합 설정"""
    max_correspondence_distance: float = 1.0  # mm
    max_iteration: int = 100
    relative_fitness: float = 1e-6
    relative_rmse: float = 1e-6
    normal_search_radius: float = 2.0  # 법선 추정 반경
    normal_max_nn: int = 30  # 법선 추정 최대 이웃 개수


@dataclass
class VisualizationConfig:
    """시각화 설정"""
    window_width: int = 1920
    window_height: int = 1080
    coordinate_frame_size: float = 50.0
    
    # 색상 설정 (RGB)
    color_cbct: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # 파란색
    color_facescan: Tuple[float, float, float] = (0.0, 1.0, 0.0)  # 초록색
    color_filtered: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # 빨간색
    color_nose: Tuple[float, float, float] = (0.0, 1.0, 0.0)  # 초록색


@dataclass
class CoordinateTransformConfig:
    """좌표계 변환 설정"""
    # LPS 좌표계 그대로 사용 (vtk.js와 동일)
    # pydicom으로 직접 좌표 계산하므로 변환 불필요
    @staticmethod
    def get_lps_to_standard_matrix() -> np.ndarray:
        return np.array([
            [-1,  0,  0,  0],  # X 반전
            [ 0, -1,  0,  0],  # Y 반전
            [ 0,  0,  1,  0],  # Z 유지
            [ 0,  0,  0,  1]
        ])
    
    # 하위 호환성을 위해 유지 (RAI 사용 시)
    @staticmethod
    def get_rai_to_standard_matrix() -> np.ndarray:
        return np.array([
            [-1,  0,  0,  0],
            [ 0,  1,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0,  0,  1]
        ])


@dataclass
class AlignmentConfig:
    """전체 정합 파이프라인 설정"""
    cbct_extraction: CBCTExtractionConfig = None
    nose_estimation: NoseEstimationConfig = None
    depth_map: DepthMapConfig = None
    mesh_sampling: MeshSamplingConfig = None
    icp: ICPConfig = None
    visualization: VisualizationConfig = None
    coordinate_transform: CoordinateTransformConfig = None
    
    def __post_init__(self):
        """기본값 초기화"""
        if self.cbct_extraction is None:
            self.cbct_extraction = CBCTExtractionConfig()
        if self.nose_estimation is None:
            self.nose_estimation = NoseEstimationConfig()
        if self.depth_map is None:
            self.depth_map = DepthMapConfig()
        if self.mesh_sampling is None:
            self.mesh_sampling = MeshSamplingConfig()
        if self.icp is None:
            self.icp = ICPConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.coordinate_transform is None:
            self.coordinate_transform = CoordinateTransformConfig()


