"""
치아 스캔 데이터 분석에 사용되는 상수 정의
"""

import numpy as np

class AnalysisConstants:
    """분석에 사용되는 상수들"""
    VECTOR_DIMENSION = 3 # 벡터의 차원
    
    RAY_LENGTH = 1000.0  # 레이 캐스팅 길이
    RAY_SCALE_FACTOR = 100  # 레이 방향 벡터 스케일
    
    # 정렬 축 판별 기준 (교차점 개수)
    X_AXIS_INTERSECTION_COUNT = 4
    Y_AXIS_INTERSECTION_COUNT = 1
    Z_AXIS_INTERSECTION_COUNT = 2
    
    # 수치 안정성 및 비교를 위한 상수
    EPSILON = 1e-10
    NEAR_PARALLEL_DOT_PRODUCT_THRESHOLD = 0.9

    # 표준 기저 벡터 및 목표 정렬 행렬
    STANDARD_BASIS = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    TARGET_ALIGNMENT_MATRIX = np.array([
        [1.0, 0.0, 0.0],  # X축
        [0.0, 1.0, 0.0],  # Y축
        [0.0, 0.0, 1.0]   # Z축
    ], dtype=np.float64)

    # 극좌표 샘플링 설정 (공통)
    POLAR_START_ANGLE = 0
    POLAR_END_ANGLE = 180
    
    # 필터링 설정 (공통)
    DEFAULT_WINDOW_SIZE = 20
    DEFAULT_X_TOLERANCE = 0.1
    
    # 랜드마크 설정 (공통)
    DEFAULT_NUM_SAMPLES = 9
    LANDMARK_DECIMAL_PLACES = 2
    
    # 레이캐스팅 설정 (공통)
    DEFAULT_NUM_SLICES = 5
    DEFAULT_ANGLE_STEP = 10

    # 악궁 곡선 샘플링 설정 (공통)
    DEFAULT_ANGLE_STEP_FULL_ARC = 1
    
    # 아웃라이어 제거 설정 (공통)
    MOLAR_OUTLIER_PERCENTILE_THRESHOLD = 10
    
    # 극좌표 샘플링 설정 (공통)
    DEFAULT_Y_OFFSET = 0.5
    
    # 시각화 설정 (공통)
    DEFAULT_Z_MIN_POINT_Z = -50

    # Voxel Downsampling 설정 (CurveExtractor)
    DEFAULT_VOXEL_SIZE_EXTRACTOR = 1.0 # CurveExtractor에서 사용되는 기본 복셀 크기
    
    # 극좌표 샘플링 설정 (CurveExtractor)
    POLAR_SAMPLING_ANGLE_STEP_EXTRACTOR = 1 # CurveExtractor에서 사용되는 극좌표 샘플링 각도 단계
    POLAR_SAMPLING_MODE_YMIN = "ymin" # CurveExtractor에서 사용되는 극좌표 샘플링 모드 (ymin)
    POLAR_SAMPLING_Y_RANGE_INF = (float('-inf'), float('inf')) # CurveExtractor에서 사용되는 Y 범위 (무한대)

    # 극좌표 샘플링 중심점 좌표 (CurveExtractor)
    POLAR_SAMPLING_CENTER_XY_ZERO = [0, 0] # 극좌표 샘플링 중심점의 X, Y 좌표 (0,0)
    POLAR_SAMPLING_CENTER_X_ZERO = 0 # 극좌표 샘플링 중심점의 X 좌표 (0)

    # Curve Sampler 설정
    CURVE_SAMPLER_DEFAULT_ANGLE_STEP = 1.0 # CurveSampler에서 사용되는 극좌표 샘플링 각도 간격
    CURVE_SAMPLER_DEFAULT_Y_SLICE_MID = -1.0 # CurveSampler에서 사용되는 Y축 슬라이스 중심 위치
    CURVE_SAMPLER_DEFAULT_Y_OFFSET = 0.5 # CurveSampler에서 사용되는 Y축 슬라이스 범위 오프셋

    POLAR_SAMPLING_MODE_FARTHEST = "farthest" # 극좌표 샘플링 모드 (외곽점)
    POLAR_SAMPLING_MODE_NEAREST = "nearest" # 극좌표 샘플링 모드 (내곽점)

    MIN_POINTS_FOR_CURVE_LENGTH = 2 # 곡선 길이 계산을 위한 최소 포인트 개수

    # 랜드마크 계산 관련 인덱스 (CurveSampler)
    LANDMARK_X_INDEX = 0 # 랜드마크 계산 시 X축 인덱스
    LANDMARK_Z_INDEX = 2 # 랜드마크 계산 시 Z축 인덱스
    
    # 반올림 자릿수 (CurveSampler)
    ROUND_DECIMAL_PLACES_ARCH_METRICS = 2 # Arch depth, molar width 반올림 자릿수

    # Polar Sampler 설정
    POLAR_SAMPLER_DEFAULT_ANGLE_STEP = 1.0 # PolarSampling에서 사용되는 기본 각도 간격
    POLAR_SAMPLER_DEFAULT_START_ANGLE = 0.0 # PolarSampling에서 사용되는 기본 시작 각도
    POLAR_SAMPLER_DEFAULT_END_ANGLE = 180.0 # PolarSampling에서 사용되는 기본 끝 각도
    POLAR_SAMPLER_DEFAULT_Y_RANGE = (-5.0, 5.0) # PolarSampling에서 사용되는 기본 Y 범위

    POLAR_SAMPLING_MODE_YMAX = "ymax" # PolarSampling 모드 (y가 가장 큰 포인트)
    POLAR_SAMPLING_MODE_YMIN_POLAR = "ymin" # PolarSampling 모드 (y가 가장 작은 포인트) - CurveExtractor의 YMIN과 구분
    POLAR_SAMPLING_MODE_FARTHEST_POLAR = "farthest" # PolarSampling 모드 (중심점으로부터 가장 먼 포인트) - CurveSampler의 FARTHEST와 구분
    POLAR_SAMPLING_MODE_NEAREST_POLAR = "nearest" # PolarSampling 모드 (중심점으로부터 가장 가까운 포인트) - CurveSampler의 NEAREST와 구분

    TWO_PI = 2 * np.pi # 2 * 파이 값 (라디안)

    # 축 인덱스 (일반)
    X_AXIS_INDEX = 0 # X축 인덱스
    Y_AXIS_INDEX = 1 # Y축 인덱스
    Z_AXIS_INDEX = 2 # Z축 인덱스
    
    # 곡선 접선/법선 계산 설정
    Y_AXIS_VECTOR = [0, 1, 0] # Y축 방향 벡터
    LAST_INDEX = -1 # 리스트의 마지막 인덱스
    SECOND_LAST_INDEX = -2 # 리스트의 마지막에서 두 번째 인덱스

    # Window Size 설정
    HALF_WINDOW_DIVISOR = 2 # 윈도우 크기를 절반으로 나누는 데 사용

    # Arch Analysis Coordinator 설정
    LINSPACE_NUM_POINTS = 100 # np.linspace의 포인트 개수
    DIRECTION_OFFSET_FACTOR = 0.1 # 방향 오프셋 요소

    Z_AXIS_VECTOR_POSITIVE = [0, 0, 1] # 양의 Z축 방향 벡터

    ROTATION_MATRIX_ZERO = 0 # 회전 행렬의 0 값
    ROTATION_MATRIX_ONE = 1 # 회전 행렬의 1 값

    ORIGIN_POINT = [0, 0, 0] # 3D 공간의 원점
    SINGLE_ROW_SHAPE = 1 # 1행 배열을 위한 shape 값

    # Arch Analysis Pipeline 설정
    CURVE_EXPAND_DISTANCE = 5.0 # 곡선 확장 거리

    # 악궁 타입 분류 설정
    RMSE_CLASSIFICATION_THRESHOLD = 2.0 # 악궁 타입 분류를 위한 RMSE 임계값
    DENTULOUS_FILTER_WINDOW_SIZE = 50 # 유치악 필터 윈도우 크기
    EDENTULOUS_FILTER_WINDOW_SIZE = 20 # 무치악 필터 윈도우 크기
    CLASSIFY_START_ANGLE = 80.0 # 악궁 타입 분류 시작 각도
    CLASSIFY_END_ANGLE = 100.0 # 악궁 타입 분류 끝 각도
    ARCH_TYPE_DENTULOUS = "dentulous" # 유치악 타입 문자열
    ARCH_TYPE_EDENTULOUS = "edentulous" # 무치악 타입 문자열

    # Face Normal Filter 설정
    FACE_NORMAL_DEFAULT_TOLERANCE = 0.1 # 면 법선 필터의 기본 허용 오차
    DEFAULT_MAX_ANGLE_DEGREES = 10.0 # 기본 최대 허용 각도 (도)

    # 축 방향 벡터 (analyze_face_normals에서 사용)
    X_AXIS_VECTOR_POSITIVE = [1, 0, 0] # 양의 X축 방향 벡터

    # 시각화 색상 및 불투명도
    VIS_COLOR_LIGHTBLUE = 'lightblue'
    VIS_OPACITY_LOW = 0.3
    VIS_TITLE_ORIGINAL_MESH = "원본 메시"

    VIS_COLOR_RED = 'red'
    VIS_OPACITY_HIGH = 0.7
    VIS_TITLE_VERTICAL_FACES = "수직 방향 면들"

    VIS_COLOR_GREEN = 'green'
    VIS_OPACITY_NORMAL = 0.7 # 수평 면에도 0.7 사용 (일단 NORMAL로 명칭 통일)
    VIS_TITLE_HORIZONTAL_FACES = "수평 방향 면들"

    # 랜드마크 계산 설정
    NUM_LANDMARKS_TO_GENERATE = 5 # 생성할 랜드마크 포인트 개수
    LANDMARK_ROUND_DECIMAL_PLACES = 2 # 랜드마크 계산 시 반올림 자릿수
    FIRST_ELEMENT_INDEX = 0 # 리스트/배열의 첫 번째 요소 인덱스
    SECOND_ELEMENT_INDEX_START = 1 # 리스트/배열의 두 번째 요소부터 시작하는 인덱스
    SYMMETRIC_POINT_X_INDEX = 0 # 대칭 포인트의 X축 인덱스
    NUM_COLS_FOR_SYMMETRIC_POINTS = 2 # 대칭 포인트 생성을 위한 열 개수

    # Vector Utils 설정
    ZERO_MAGNITUDE = 0 # 벡터의 크기가 0인지 확인하는 상수
    MATRIX_DIMENSION_3X3 = 3 # 3x3 행렬의 차원

    # Mesh Aligner 설정
    MESH_ALIGNER_EPSILON = 1e-9 # MeshAligner에서 사용되는 0으로 나누기 방지 상수
    AXIS_VISUALIZATION_LENGTH_SCALE = 0.35 # 시각화 시 축 길이 스케일

    # 바운딩 박스 인덱스
    BBOX_X_MIN_INDEX = 0
    BBOX_X_MAX_INDEX = 1
    BBOX_Y_MIN_INDEX = 2
    BBOX_Y_MAX_INDEX = 3
    BBOX_Z_MIN_INDEX = 4
    BBOX_Z_MAX_INDEX = 5

    MIN_VARIANCE_AXIS_INDEX = 0 # 최소 분산 축 인덱스
    INERTIA_TENSOR_DIAGONAL_VALUE = 1.0 # 관성 텐서 계산 시 사용되는 상수

    # Mesh Filter 설정
    DEFAULT_Z_THRESHOLD = 0.0 # Z값 필터링 기본 임계값
    Z_STD_THRESHOLD = 2.0 # Z값 표준편차 아웃라이어 제거 임계값
    DISTANCE_FILTER_PERCENTILE = 95.0 # 거리 필터링 백분위수
    DEFAULT_GRID_SIZE = 1.0 # 볼륨 밀도 필터링 그리드 셀 크기
    DEFAULT_MIN_DENSITY = 5 # 볼륨 밀도 필터링 최소 밀도 임계값
    NUMPY_ZEROS_DTYPE_BOOL = False # np.zeros의 dtype=bool 값

    # Point Cloud Ray Caster 설정
    RAY_DISTANCE_THRESHOLD = 5.0 # 레이로부터의 최대 허용 거리
    NUMPY_ONES_DTYPE_BOOL = True # np.ones의 dtype=bool 값
    ANGLE_360_START = 0 # 360도 회전 시작 각도
    ANGLE_360_END = 360 # 360도 회전 끝 각도
    MIN_POINTS_FOR_RAY_CASTING = 1 # 레이 캐스팅을 위한 최소 포인트 개수
    DEFAULT_NUM_SLICES_RAY_CASTER = 5 # 레이 캐스터의 기본 슬라이스 개수
    DEFAULT_ANGLE_STEP_RAY_CASTER = 5 # 레이 캐스터의 기본 각도 단계
    Z_AXIS_VECTOR_NEGATIVE = [0, 0, -1] # 음의 Z축 방향 벡터

    # Mesh Direction Aligner 설정
    ALIGNMENT_THRESHOLD = 0.8 # 정렬 적용을 위한 내적값 임계값
    ROTATION_AXIS_NORM_THRESHOLD = 1e-6 # 회전축 크기 임계값 (거의 0)
    COSINE_ANGLE_CLIP_MIN = -1.0 # 코사인 각도 클리핑 최소값
    COSINE_ANGLE_CLIP_MAX = 1.0 # 코사인 각도 클리핑 최대값