"""
initial_alignment 모듈
3D 메시의 초기 정렬(Initial Alignment)을 수행하는 모듈

주요 클래스:
    - MeshAligner: 메시 정렬 전체 프로세스 조율
    - MeshPreprocessor: 메시 전처리 (다운샘플링, 변환 등)
    - MeshLoader: 메시 파일 로딩 및 검증
    - InitialAlignmentFinder: OBB 기반 초기 정렬 탐색
    - ICPRegistration: ICP 정합 수행
    - OBBAnalyzer: OBB 분석 및 좌표계 계산
    - TransformCalculator: 변환 행렬 계산
    - DistanceCalculator: 거리 계산 (RMSE, Chamfer Distance)

주요 함수:
    - align_3d_meshes: STL 파일로부터 메시 정렬
    - align_meshes_direct: 메시 객체로부터 직접 정렬
"""

# 클래스 기반 API
from .mesh_aligner import MeshAligner
from .preprocessing import MeshPreprocessor
from .mesh_loader import MeshLoader
from .initial_alignment_finder import InitialAlignmentFinder
from .icp_registration import ICPRegistration
from .obb_analyzer import OBBAnalyzer
from .transform_calculator import TransformCalculator
from .distance_calculator import DistanceCalculator

# 편의 함수들
from .mesh_aligner import align_3d_meshes, align_meshes_direct

# 상수들
from .constants import (
    # 샘플링 관련
    SamplingConfig,
    DEFAULT_SAMPLE_POINTS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_VOXEL_SIZE,
    # ICP 관련
    ICPConfig,
    MultiScaleICPConfig,
    ICP_MAX_CORRESPONDENCE_DISTANCE,
    ICP_P2PLANE_MAX_CORRESPONDENCE_DISTANCE,
    ICP_MAX_ITERATIONS,
    # 초기 정렬 관련
    InitialAlignmentConfig,
    INITIAL_ALIGNMENT_TOP_K,
    INITIAL_ALIGNMENT_RMSE_STEP,
    # 거리 계산 관련
    DistanceCalculationConfig,
    DISTANCE_CALCULATION_DEFAULT_STEP,
    # OBB 관련
    OBBConfig,
    # 검증 관련
    ValidationConfig,
    # 메시지 관련
    ErrorMessages,
    LogMessages,
    # 경로 관련
    PathConfig,
)

__all__ = [
    # 클래스
    'MeshAligner',
    'MeshPreprocessor',
    'MeshLoader',
    'InitialAlignmentFinder',
    'ICPRegistration',
    'OBBAnalyzer',
    'TransformCalculator',
    'DistanceCalculator',
    
    # 주요 함수
    'align_3d_meshes',
    'align_meshes_direct',
    
    # 상수 클래스
    'SamplingConfig',
    'ICPConfig',
    'MultiScaleICPConfig',
    'InitialAlignmentConfig',
    'DistanceCalculationConfig',
    'OBBConfig',
    'ValidationConfig',
    'ErrorMessages',
    'LogMessages',
    'PathConfig',
    
    # 레거시 상수 (하위 호환성)
    'DEFAULT_SAMPLE_POINTS',
    'DEFAULT_RANDOM_SEED',
    'DEFAULT_VOXEL_SIZE',
    'ICP_MAX_CORRESPONDENCE_DISTANCE',
    'ICP_P2PLANE_MAX_CORRESPONDENCE_DISTANCE',
    'ICP_MAX_ITERATIONS',
    'INITIAL_ALIGNMENT_TOP_K',
    'INITIAL_ALIGNMENT_RMSE_STEP',
    'DISTANCE_CALCULATION_DEFAULT_STEP',
]
