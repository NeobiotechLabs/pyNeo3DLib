"""이진 마스크 전처리: majority filter, small CC 제거, bbox."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def binary_majority_filter(mask: np.ndarray, size: int = 3) -> np.ndarray:
    if size <= 1 or not mask.any():
        return mask.astype(bool, copy=False)
    med = ndimage.median_filter(mask.astype(np.uint8), size=size)
    return med.astype(bool)


def remove_small_components(
    mask: np.ndarray, ratio: float = 0.05, min_voxels: int = 50
) -> tuple[np.ndarray, int, int]:
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(bool), 0, 0

    sizes = np.bincount(labeled.ravel())
    total = int(sizes[1:].sum())
    keep = np.zeros(n + 1, dtype=bool)
    for cid in range(1, n + 1):
        s = int(sizes[cid])
        others = total - s
        too_small_vs_others = others > 0 and s < ratio * others
        too_small_abs = s < min_voxels
        keep[cid] = not (too_small_vs_others or too_small_abs)

    return keep[labeled], n, int(keep.sum())


def bbox_slices(mask: np.ndarray, pad: int = 4):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = np.maximum(coords.min(axis=0) - pad, 0)
    maxs = np.minimum(coords.max(axis=0) + pad, np.array(mask.shape) - 1)
    return tuple(slice(int(a), int(b) + 1) for a, b in zip(mins, maxs))
