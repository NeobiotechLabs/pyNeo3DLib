"""좌/우 해부학적 분리 전담."""

from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy import ndimage


def _rl_axis_and_left_positive(affine: np.ndarray) -> tuple[int, bool]:
    codes = nib.aff2axcodes(affine)
    for i, c in enumerate(codes):
        if c in ("L", "R"):
            return i, c == "L"
    return 0, True


def split_left_right_masks(
    canal: np.ndarray, affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """신경관 CC를 mid-sagittal 기준 좌/우 마스크로 분리."""
    axis, left_is_positive = _rl_axis_and_left_positive(affine)
    mid = canal.shape[axis] / 2.0

    labeled, n = ndimage.label(canal)
    left = np.zeros_like(canal, dtype=bool)
    right = np.zeros_like(canal, dtype=bool)
    assign: dict[int, str] = {}

    for cid in range(1, n + 1):
        coords = np.argwhere(labeled == cid)
        centroid = coords.mean(axis=0)
        on_left = (
            centroid[axis] >= mid if left_is_positive else centroid[axis] < mid
        )
        side = "L" if on_left else "R"
        assign[cid] = side
        if on_left:
            left[labeled == cid] = True
        else:
            right[labeled == cid] = True

    meta = {
        "rl_axis": int(axis),
        "axcodes": list(nib.aff2axcodes(affine)),
        "mid": float(mid),
        "n_components": int(n),
        "assign": {str(k): v for k, v in assign.items()},
    }
    return left, right, meta
