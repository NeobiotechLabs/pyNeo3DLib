"""NIfTI 입출력 전담."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_label(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.dtype.kind == "f":
        data = np.rint(data).astype(np.uint8)
    else:
        data = data.astype(np.uint8, copy=False)
    return data, img.affine, img


def spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def save_label(
    path: Path,
    data: np.ndarray,
    affine: np.ndarray,
    ref_img: nib.Nifti1Image | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if ref_img is not None:
        hdr = ref_img.header.copy()
    else:
        hdr = None
    if hdr is not None:
        hdr.set_data_dtype(np.uint8)
        img = nib.Nifti1Image(data.astype(np.uint8), affine, hdr)
    else:
        img = nib.Nifti1Image(data.astype(np.uint8), affine)
    nib.save(img, str(path))
