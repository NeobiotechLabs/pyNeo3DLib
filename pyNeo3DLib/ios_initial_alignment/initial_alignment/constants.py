"""
initialAlign 모듈의 모든 상수와 설정값을 관리하는 파일
클래스 기반 구조로 일관된 상수 관리
"""

from typing import Final


# ============================================================================
# 샘플링 관련 상수
# ============================================================================

class SamplingConfig:
    """샘플링 관련 설정 상수"""
    DEFAULT_SAMPLE_POINTS: Final[int] = 100000
    """기본 포인트 샘플링 개수"""
    
    DEFAULT_RANDOM_SEED: Final[int] = 42
    """랜덤 시드 기본값 (결정론적 결과를 위해)"""
    
    DEFAULT_VOXEL_SIZE: Final[float] = 0.1
    """다운샘플링 기본 복셀 크기"""


# ============================================================================
# ICP 관련 상수
# ============================================================================

class ICPConfig:
    """ICP 정합 관련 설정 상수"""
    MAX_CORRESPONDENCE_DISTANCE_P2PLANE: Final[float] = 2.0
    """ICP 최근접 대응 최대 거리 임계값 (Point-to-Plane)"""
    
    MAX_ITERATIONS: Final[int] = 200
    """ICP 최대 반복 횟수"""

    RELATIVE_FITNESS_P2PLANE: Final[float] = 1e-07
    """ICP 정합의 상대적인 적합도 변화량 임계값 (Point-to-Plane)"""

    RELATIVE_RMSE_P2PLANE: Final[float] = 1e-07
    """ICP 정합의 상대적인 RMSE 변화량 임계값 (Point-to-Plane)"""
    
    NORMAL_ESTIMATION_RADIUS_MULTIPLIER: Final[float] = 2.0
    """법선 추정을 위한 반경 배율 (max_corr_dist * multiplier)"""
    
    NORMAL_ESTIMATION_MAX_NN: Final[int] = 40
    """법선 추정을 위한 최대 이웃 개수"""


# ============================================================================
# Multi-Scale ICP 관련 상수
# ============================================================================

class MultiScaleICPConfig:
    """Multi-Scale ICP 정합 관련 설정 상수"""
    TARGET_RMSE: Final[float] = 0.15
    """Multi-Scale ICP의 목표 RMSE 값"""

    RMSE_TOLERANCE: Final[float] = 1e-4
    """Multi-Scale ICP에서 RMSE 개선이 없다고 판단하는 임계값"""

    # (max_correspondence_dist, max_iteration) 튜플 리스트
    SCALES: Final[list[tuple[float, int]]] = [
        (2.0, 60),   # coarse
        (1.0, 80),   # mid
        (0.5, 120),  # fine
        # (0.3, 150), # 필요시 추가 가능
    ]
    """Multi-Scale ICP의 각 스케일별 (최대 대응 거리, 최대 반복 횟수) 설정"""


# ============================================================================
# 초기 정렬 관련 상수
# ============================================================================

class InitialAlignmentConfig:
    """초기 정렬 관련 설정 상수"""
    TOP_K: Final[int] = 2
    """초기 정렬 후보 중 상위 K개 선택"""
    
    RMSE_STEP: Final[int] = 10
    """초기 정렬 평가시 RMSE 계산을 위한 샘플링 간격"""


# ============================================================================
# 거리 계산 관련 상수
# ============================================================================

class DistanceCalculationConfig:
    """거리 계산 관련 설정 상수"""
    DEFAULT_STEP: Final[int] = 10
    """거리 계산시 기본 샘플링 간격 (속도 향상을 위해)"""
    
    RMSE_COMPUTATION_EPSILON: Final[float] = 1e-10
    """RMSE 계산시 0으로 나누기 방지를 위한 입실론 값"""


# ============================================================================
# OBB 관련 상수
# ============================================================================

class OBBConfig:
    """OBB 분석 관련 설정 상수"""
    MIN_DETERMINANT_THRESHOLD: Final[float] = 0.0
    """OBB 기반 회전 행렬의 최소 determinant 임계값 (reflection 제거용)"""


# ============================================================================
# 파일 검증 관련 상수
# ============================================================================

class ValidationConfig:
    """파일 검증 관련 설정 상수"""
    MIN_MESH_VERTICES: Final[int] = 3
    """유효한 메시로 간주하기 위한 최소 버텍스 개수"""
    
    MIN_POINTCLOUD_POINTS: Final[int] = 10
    """유효한 포인트클라우드로 간주하기 위한 최소 포인트 개수"""


# ============================================================================
# 에러 메시지
# ============================================================================

class ErrorMessages:
    """에러 메시지 상수"""
    FILE_NOT_FOUND: Final[str] = "파일을 찾을 수 없습니다: {path}"
    EMPTY_MESH: Final[str] = "메시가 비어있습니다: {path}"
    EMPTY_POINTCLOUD: Final[str] = "포인트클라우드가 비어있습니다"
    INVALID_MESH: Final[str] = "유효하지 않은 메시입니다: {path}"
    MESH_LOAD_FAILED: Final[str] = "메시 로드 실패: {path}"
    NO_VALID_TRANSFORMS: Final[str] = "유효한 변환 행렬을 찾을 수 없습니다"


# ============================================================================
# 로그 메시지
# ============================================================================

class LogMessages:
    """로그 메시지 상수"""
    MESH_PREPROCESSING: Final[str] = "=== 메시 전처리 ==="
    INITIAL_ALIGNMENT: Final[str] = "=== 초기 변환 후보 탐색 ==="
    ICP_REFINEMENT: Final[str] = "=== ICP 정밀 정렬 ==="
    FINAL_RESULTS: Final[str] = "=== 최종 결과 ==="
    PROGRAM_COMPLETE: Final[str] = "프로그램 완료!"
    TRANSFORM_SUCCESS: Final[str] = "변환 행렬이 성공적으로 계산되었습니다."


# ============================================================================
# 기본 파일 경로 (테스트용)
# ============================================================================

class PathConfig:
    """파일 경로 설정 상수"""
    DEFAULT_TARGET_PATH: Final[str] = "data/target_test.stl"
    DEFAULT_CONTROL_PATH: Final[str] = "data/control_test.stl"


# ============================================================================
# 하위 호환성을 위한 레거시 상수들
# ============================================================================

# 샘플링 관련
DEFAULT_SAMPLE_POINTS: Final[int] = SamplingConfig.DEFAULT_SAMPLE_POINTS
DEFAULT_RANDOM_SEED: Final[int] = SamplingConfig.DEFAULT_RANDOM_SEED
DEFAULT_VOXEL_SIZE: Final[float] = SamplingConfig.DEFAULT_VOXEL_SIZE

# ICP 관련
ICP_MAX_CORRESPONDENCE_DISTANCE: Final[float] = ICPConfig.MAX_CORRESPONDENCE_DISTANCE_P2PLANE
ICP_P2PLANE_MAX_CORRESPONDENCE_DISTANCE: Final[float] = ICPConfig.MAX_CORRESPONDENCE_DISTANCE_P2PLANE
ICP_MAX_ITERATIONS: Final[int] = ICPConfig.MAX_ITERATIONS

# 초기 정렬 관련
INITIAL_ALIGNMENT_TOP_K: Final[int] = InitialAlignmentConfig.TOP_K
INITIAL_ALIGNMENT_RMSE_STEP: Final[int] = InitialAlignmentConfig.RMSE_STEP

# 거리 계산 관련
DISTANCE_CALCULATION_DEFAULT_STEP: Final[int] = DistanceCalculationConfig.DEFAULT_STEP