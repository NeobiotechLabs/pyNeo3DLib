"""평면·회전 등 공통 3D 기하 유틸."""

from __future__ import annotations

import math

import numpy as np


def as_f64(point: np.ndarray) -> np.ndarray:
    return np.asarray(point, dtype=np.float64)


def unit_vector(vector: np.ndarray, *, name: str) -> np.ndarray:
    v = as_f64(vector)
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        raise ValueError(f"{name} 벡터 길이가 0에 가까워 정의할 수 없습니다.")
    return v / norm


def rotate_point_about_axis(
    point: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """pivot을 지나고 axis 방향인 직선을 축으로 point를 angle_deg만큼 회전 (Rodrigues)."""
    k = unit_vector(axis, name="회전축")
    pivot = as_f64(pivot)
    v = as_f64(point) - pivot
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotated = v * cos_t + np.cross(k, v) * sin_t + k * np.dot(k, v) * (1.0 - cos_t)
    return pivot + rotated


def plane_from_three_points(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """세 점을 지나는 평면의 무게중심과 단위 법선."""
    p1, p2, p3 = as_f64(p1), as_f64(p2), as_f64(p3)
    centroid = (p1 + p2 + p3) / 3.0
    normal = np.cross(p2 - p1, p3 - p1)
    return centroid, unit_vector(normal, name="평면")


def point_along_normal(
    origin: np.ndarray,
    normal: np.ndarray,
    *,
    signed_offset_mm: float,
) -> np.ndarray:
    """단위 법선 방향으로 signed_offset_mm만큼 이동한 점."""
    n = unit_vector(normal, name="법선")
    return as_f64(origin) + n * float(signed_offset_mm)


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (as_f64(a) + as_f64(b)) / 2.0
