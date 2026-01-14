"""
치아 스캔 데이터 분석에 사용되는 상수 정의
"""

class AnalysisConstants:
    """분석에 사용되는 상수들"""
    RAY_LENGTH = 1000.0  # 레이 캐스팅 길이
    RAY_SCALE_FACTOR = 100  # 레이 방향 벡터 스케일


class PointCloudConstants:
    """포인트 클라우드 처리에 사용되는 상수들"""
    NORMAL_ESTIMATION_RADIUS = 0.1  # 노말 추정 반경
    NORMAL_ESTIMATION_MAX_NN = 30  # 노말 추정 최대 이웃 수
    ORIENT_NORMALS_K = 100  # 노말 방향 일관성 유지를 위한 k값
    DOWNSAMPLE_EVERY_K_POINTS = 2  # 다운샘플링 간격


class VisualizationConstants:
    """시각화에 사용되는 상수들"""
    WINDOW_WIDTH = 1920  # 시각화 창 너비
    WINDOW_HEIGHT = 1080  # 시각화 창 높이
    BACKGROUND_COLOR = [0.9, 0.9, 0.9]  # 배경색 (RGB)
    POINT_SIZE = 2.0  # 포인트 크기
    CAMERA_ZOOM = 0.8  # 카메라 줌 레벨
    CAMERA_FRONT = [0, -1, 0]  # 카메라 전방 방향
    CAMERA_UP = [0, 0, 1]  # 카메라 업 방향
    
    # 애니메이션 딜레이 (초)
    INIT_DELAY = 0.1  # 초기화 딜레이
    PAUSE_DELAY = 1.0  # 일시정지 딜레이
    ANIMATION_DELAY = 0.05  # 애니메이션 프레임 딜레이
    RENDER_LOOP_DELAY = 0.1  # 렌더링 루프 딜레이


class ICPConstants:
    """ICP 정합에 사용되는 상수들"""
    # ICP 거리 임계값 (3단계)
    DISTANCE_THRESHOLD_STAGE1 = 1.0  # 1차 ICP 거리 임계값 (coarse)
    DISTANCE_THRESHOLD_STAGE2 = 0.3  # 2차 ICP 거리 임계값 (medium)
    DISTANCE_THRESHOLD_STAGE3 = 0.05  # 3차 ICP 거리 임계값 (fine)
    
    # ICP 수렴 기준
    MAX_ITERATIONS = 1000  # ICP 최대 반복 횟수
    RELATIVE_FITNESS = 1e-6  # 상대적 적합도 수렴 기준
    RELATIVE_RMSE = 1e-6  # 상대적 RMSE 수렴 기준
    CONVERGENCE_TOLERANCE = 1e-6  # 변환 행렬 수렴 허용 오차
    
    # 시각화 관련
    VISUALIZATION_INTERVAL = 20  # 시각화 업데이트 간격 (반복 횟수)


class TextureConstants:
    """텍스처 처리에 사용되는 상수들"""
    BLACK_THRESHOLD = 10  # 검정색 투명 처리 임계값
    ALPHA_THRESHOLD = 128  # 투명 여부 판단 임계값 (50%)


class MeshCleaningConstants:
    """메시 정리에 사용되는 상수들"""
    MIN_CLUSTER_RATIO = 0.1  # 제거할 작은 클러스터의 최소 비율
    SMOOTH_ITERATIONS = 2  # Laplacian 스무딩 반복 횟수
    LAPLACIAN_LAMBDA = 0.5  # Laplacian 필터 람다 값


class NoiseRemovalConstants:
    """노이즈 제거에 사용되는 상수들"""
    NORMAL_ANGLE_THRESHOLD_DEGREES = 70  # 노말 각도 임계값 (도)


class RayCastingConstants:
    """레이캐스팅에 사용되는 상수들"""
    BOUNDARY_SCAN_SPAN = 3  # 경계 탐지 스캔 범위
    BOUNDARY_NUM_SAMPLES = 100  # 경계 탐지 샘플링 포인트 수


class IncisorAlignmentConstants:
    """중절치 정렬에 사용되는 상수들"""
    X_AXIS_CLIP_MIN = -3.0  # x축 클리핑 최소값
    X_AXIS_CLIP_MAX = 3.0  # x축 클리핑 최대값