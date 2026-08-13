"""랜드마크 좌표 검증 (패키지 공통)."""

from __future__ import annotations

import numpy as np


class LandmarkValidationError(ValueError):
    """필수 랜드마크·좌표가 없거나 유효하지 않을 때."""


def validate_landmark_point(
    name: str,
    point: np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    arr = np.asarray(point, dtype=np.float64)
    if arr.shape != (3,):
        raise LandmarkValidationError(
            f"{source}: '{name}' 좌표 shape가 (3,)이 아님: {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise LandmarkValidationError(
            f"{source}: '{name}' 좌표에 NaN 또는 Inf가 있습니다: {arr!r}"
        )
    return arr.copy()


def validate_required_landmarks(
    landmarks: dict[str, np.ndarray],
    required: tuple[str, ...],
    *,
    source: str,
) -> dict[str, np.ndarray]:
    missing = [name for name in required if name not in landmarks]
    if missing:
        raise LandmarkValidationError(
            f"{source}: 필수 랜드마크 없음: {', '.join(missing)}"
        )
    return {
        name: validate_landmark_point(name, landmarks[name], source=source)
        for name in required
    }
