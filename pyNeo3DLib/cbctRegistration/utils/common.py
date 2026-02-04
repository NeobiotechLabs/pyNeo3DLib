"""
CBCT 정합 유틸리티 함수 모듈

포인트 클라우드 변환 및 기본 연산을 위한 유틸리티 함수를 제공합니다.
"""
import numpy as np
import open3d as o3d
import copy
from typing import Optional, Tuple


def np_to_pcd(
    points: np.ndarray, 
    color: Optional[Tuple[float, float, float]] = None
) -> o3d.geometry.PointCloud:
    """
    numpy 배열을 Open3D PointCloud로 변환
    
    Args:
        points: (N, 3) 형태의 포인트 배열
        color: RGB 색상 (0~1 범위)
    
    Returns:
        o3d.geometry.PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def pcd_to_np(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Open3D PointCloud를 numpy 배열로 변환
    
    Args:
        pcd: Open3D PointCloud
    
    Returns:
        (N, 3) 형태의 numpy 배열
    """
    return np.asarray(pcd.points)


def apply_transform(
    pcd: o3d.geometry.PointCloud, 
    transform_matrix: np.ndarray,
    in_place: bool = False
) -> o3d.geometry.PointCloud:
    """
    포인트 클라우드에 변환 행렬 적용
    
    Args:
        pcd: 입력 포인트 클라우드
        transform_matrix: 4x4 변환 행렬
        in_place: True면 원본 수정, False면 복사본 반환
    
    Returns:
        변환된 포인트 클라우드
    """
    if in_place:
        pcd.transform(transform_matrix)
        return pcd
    else:
        pcd_copy = copy.deepcopy(pcd)
        pcd_copy.transform(transform_matrix)
        return pcd_copy


def compute_translation_matrix(translation_vector: np.ndarray) -> np.ndarray:
    """
    평행이동 변환 행렬 생성
    
    Args:
        translation_vector: (3,) 평행이동 벡터
    
    Returns:
        4x4 변환 행렬
    """
    matrix = np.eye(4)
    matrix[:3, 3] = translation_vector
    return matrix


def compute_center_alignment_transform(
    source_center: np.ndarray,
    target_center: np.ndarray
) -> np.ndarray:
    """
    소스 중심을 타겟 중심으로 이동하는 변환 행렬 계산
    
    Args:
        source_center: 소스 포인트 클라우드 중심 (3,)
        target_center: 타겟 포인트 클라우드 중심 (3,)
    
    Returns:
        4x4 변환 행렬
    """
    translation_vector = target_center - source_center
    return compute_translation_matrix(translation_vector)


def transform_point_homogeneous(
    point: np.ndarray, 
    transform_matrix: np.ndarray
) -> np.ndarray:
    """
    단일 포인트에 동차 좌표 변환 적용
    
    Args:
        point: (3,) 포인트 좌표
        transform_matrix: 4x4 변환 행렬
    
    Returns:
        (3,) 변환된 포인트 좌표
    """
    point_h = np.append(point, 1)  # homogeneous
    return (transform_matrix @ point_h)[:3]


def apply_transform_to_points(
    points: np.ndarray,
    transform_matrix: np.ndarray
) -> np.ndarray:
    """
    여러 포인트에 변환 행렬을 벡터화하여 적용 (빠른 버전)
    
    Args:
        points: (N, 3) 형태의 포인트 배열
        transform_matrix: 4x4 변환 행렬
    
    Returns:
        (N, 3) 형태의 변환된 포인트 배열
    """
    # 동차 좌표로 변환 (N, 4)
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    
    # 변환 적용 (벡터화)
    transformed_h = (transform_matrix @ points_h.T).T
    
    # 3D 좌표로 변환
    return transformed_h[:, :3]

