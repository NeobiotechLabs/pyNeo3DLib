"""튜브 래스터화 · 라벨 합성 전담."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .centerline import arc_length_resample
from .config import LABEL_CANAL


def estimate_radius_mm(
    mask: np.ndarray,
    path_ijk: np.ndarray,
    spacing: np.ndarray,
    default_mm: float = 1.2,
) -> float:
    if not mask.any() or len(path_ijk) == 0:
        return default_mm
    dt = ndimage.distance_transform_edt(mask, sampling=spacing)
    radii = []
    shape = mask.shape
    for p in path_ijk:
        ijk = np.rint(p).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= shape):
            continue
        r = float(dt[tuple(ijk)])
        if r > 0:
            radii.append(r)
    if not radii:
        return default_mm
    return float(np.median(radii))


def rasterize_tube(
    shape: tuple[int, ...],
    path_ijk: np.ndarray,
    radius_mm: float,
    spacing: np.ndarray,
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if len(path_ijk) == 0:
        return out

    mean_sp = float(np.mean(spacing))
    pts = arc_length_resample(path_ijk, spacing_mm=0.5)
    idx = np.rint(pts).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, shape[0] - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, shape[1] - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, shape[2] - 1)
    out[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    if radius_mm <= 0:
        return out

    pad = int(np.ceil(radius_mm / mean_sp)) + 2
    mins = np.maximum(idx.min(axis=0) - pad, 0)
    maxs = np.minimum(idx.max(axis=0) + pad, np.array(shape) - 1)
    sl = tuple(slice(int(a), int(b) + 1) for a, b in zip(mins, maxs))
    crop = out[sl]
    dt = ndimage.distance_transform_edt(~crop, sampling=spacing)
    out[sl] = dt <= radius_mm
    return out


def merge_canal_into_label(
    data: np.ndarray,
    new_canal: np.ndarray,
    *,
    keep_original: bool,
) -> np.ndarray:
    canal0 = data == LABEL_CANAL
    if keep_original:
        new_canal = new_canal | canal0
    out = data.copy()
    out[canal0] = 0
    out[new_canal] = LABEL_CANAL
    return out
