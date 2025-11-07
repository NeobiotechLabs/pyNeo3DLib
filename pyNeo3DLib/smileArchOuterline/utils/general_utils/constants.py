"""
치아 스캔 데이터 분석에 사용되는 상수 정의
"""

class AnalysisConstants:
    """분석에 사용되는 상수들"""
    RAY_LENGTH = 1000.0  # 레이 캐스팅 길이
    RAY_SCALE_FACTOR = 100  # 레이 방향 벡터 스케일
    
    # 정렬 축 판별 기준 (교차점 개수)
    X_AXIS_INTERSECTION_COUNT = 4
    Y_AXIS_INTERSECTION_COUNT = 1
    Z_AXIS_INTERSECTION_COUNT = 2
    
    # 극좌표 샘플링 설정
    POLAR_START_ANGLE = 0
    POLAR_END_ANGLE = 180
    
    # 필터링 설정
    DEFAULT_WINDOW_SIZE = 20
    DEFAULT_X_TOLERANCE = 0.1
    
    # 랜드마크 설정
    DEFAULT_NUM_SAMPLES = 9
    LANDMARK_DECIMAL_PLACES = 2
    
    # 레이캐스팅 설정
    DEFAULT_NUM_SLICES = 5
    DEFAULT_ANGLE_STEP = 10

    # 악궁 곡선 샘플링 설정
    DEFAULT_ANGLE_STEP_FULL_ARC = 1

    
    # 아웃라이어 제거 설정
    MOLAR_OUTLIER_PERCENTILE_THRESHOLD = 10
    
    # 극좌표 샘플링 설정
    DEFAULT_Y_OFFSET = 0.5
    
    # 시각화 설정
    DEFAULT_Z_MIN_POINT_Z = -50