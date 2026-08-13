"""라벨맵 전용 연결 성분(post-CC) 필터링.

nnU-Net 예측 → 신경관 복원 직후, 최종 라벨맵에서 각 라벨별
연결 성분 크기를 기준으로 불필요한 작은 분절을 제거한다.

**설정 없이 바로 동작**: cfg=None 이면 ``DEFAULT_LABEL_CC_CONFIG``
자동 적용이다.

라벨 매핑 및 규칙:

    0 : 배경(background)
    1 : 상악두개골(maxillary skull) — 최대 1개 CC 유지
    2 : 하악골(mandible body) — 최대 1개 CC 유지
    3 : 신경관(neural canal) — 최대 상위 2개 CC 유지
    4 : 상악동(maxillary sinus) — 최대 상위 2개 CC 유지

사용:

    from dental_anatomy_segmentation_lib.postprocess_labelmap import (
        filter_labelmap_by_connected_components,
    )

    filtered = filter_labelmap_by_connected_components(labelmap)
    # cfg=None 이면 기본값 자동 적용
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelCcConfig:
    """각 라벨별로 최대 허용 연결 성분 수를 정의."""

    # { label_id: max_keep }
    # 가장 큰 N개의 연결 성분만 남기고, 나머지는 배경(0)으로 둡니다.
    # N <= 0 이면 해당 라벨은 필터하지 않고 그대로 둡니다.
    max_components: dict[int, int] | None = None

    def is_empty(self) -> bool:
        if self.max_components is None:
            return True
        return not any(v > 0 for v in self.max_components.values())


#: 기본 설정 — 상악두개골·하악골은 1개, 신경관·상악동은 상위 2개
DEFAULT_LABEL_CC_CONFIG = LabelCcConfig(
    max_components={
        1: 1,   # maxillary skull
        2: 1,   # mandible body
        3: 2,   # neural canal
        4: 2,   # maxillary sinus
    }
)


# ---------------------------------------------------------------------------
# Core filtering functions
# ---------------------------------------------------------------------------


def keep_largest_component(
    binary_mask: np.ndarray,
) -> np.ndarray:
    """
    이진 마스크에서 가장 큰 **하나**의 연결 성분만 남긴다.

    Parameters
    ----------
    binary_mask : np.ndarray[bool]
        입력 이진 마스크.

    Returns
    -------
    np.ndarray[bool]
        Largest connected component만 True인 마스크.
    """
    m = np.asarray(binary_mask, dtype=bool)
    if not m.any():
        return m

    labeled, n = ndimage.label(m)
    if n == 0 or n == 1:
        return m

    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    largest = int(np.argmax(sizes[1:]) + 1)
    return labeled == largest


def keep_top_k_components(
    binary_mask: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    이진 마스크에서 크기 상위 **k** 개의 연결 성분만 남긴다.

    Parameters
    ----------
    binary_mask : np.ndarray[bool]
        입력 이진 마스크.
    k : int
        유지할 최대 연결 성분 수. k <= 0 이면 필터 없이 반환.

    Returns
    -------
    np.ndarray[bool]
        Top-k components만 True 인 마스크.
    """
    m = np.asarray(binary_mask, dtype=bool)
    if k <= 0 or not m.any():
        return m

    labeled, n = ndimage.label(m)
    if n <= k:
        return m

    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    order = np.argsort(-sizes)[1:]  # 1-indexed labels sorted by size (desc)
    keep_labels = set(order[:k].tolist())
    return np.isin(labeled, list(keep_labels))


def filter_labelmap_by_connected_components(
    labelmap: np.ndarray,
    cfg: LabelCcConfig | None = None,
) -> np.ndarray:
    """
    전체 다중 라벨 맵을 라벨별 연결 성분 기준으로 필터링한다.

    각 foreground 라벨([1, 3, 4, 6 등]) 을 순서대로 개별 이진 마스크로
    추출 → top-N CC 필터 적용 → 다시 합산한다.

    Parameters
    ----------
    labelmap : np.ndarray[np.uint8 | np.intp]
        nnU-Net 예측 결과 (또는 그 이후 상태) 의 정수 라벨맵.
    cfg : LabelCcConfig, optional
        필터링 구성. None 이면 ``DEFAULT_LABEL_CC_CONFIG`` 를 사용.

    Returns
    -------
    np.ndarray
        필터링된 정수 라벨맵 (dtype 유지).
    """
    cfg = cfg or DEFAULT_LABEL_CC_CONFIG
    if cfg.is_empty():
        return labelmap.copy()

    max_cc = cfg.max_components
    result = np.zeros_like(labelmap, dtype=labelmap.dtype)

    # 각 라벨별 독립적으로 처리
    for label_id, max_k in max_cc.items():
        if max_k <= 0:
            continue
        mask = labelmap == label_id

        if max_k == 1:
            cleaned = keep_largest_component(mask)
        else:
            cleaned = keep_top_k_components(mask, max_k)

        result[cleaned] = label_id

    return result


__all__ = [
    "LabelCcConfig",
    "DEFAULT_LABEL_CC_CONFIG",
    "keep_largest_component",
    "keep_top_k_components",
    "filter_labelmap_by_connected_components",
]
