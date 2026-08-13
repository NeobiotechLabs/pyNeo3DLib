"""
예측 라벨 NIfTI를 참조 볼륨(원본 CT NIfTI)의 ITK 격자에 **최인접**으로 맞춥니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import itk
import numpy as np


def _itk_images_same_grid(
    moving: itk.Image,
    reference: itk.Image,
    *,
    eps: float = 1e-5,
) -> bool:
    if itk.size(moving) != itk.size(reference):
        return False
    sm = np.asarray(moving.GetSpacing(), dtype=np.float64)
    sr = np.asarray(reference.GetSpacing(), dtype=np.float64)
    if np.max(np.abs(sm - sr)) > eps:
        return False
    om = np.asarray(moving.GetOrigin(), dtype=np.float64)
    or_ = np.asarray(reference.GetOrigin(), dtype=np.float64)
    if np.max(np.abs(om - or_)) > eps:
        return False
    dm = np.asarray(itk.GetArrayFromMatrix(moving.GetDirection()), dtype=np.float64).ravel()
    dr = np.asarray(itk.GetArrayFromMatrix(reference.GetDirection()), dtype=np.float64).ravel()
    return bool(np.max(np.abs(dm - dr)) <= eps)


def resample_label_nifti_to_reference_geometry(
    label_nifti: Union[str, Path],
    reference_nifti: Union[str, Path],
    output_nifti: Union[str, Path],
) -> Path:
    """
    ``label_nifti`` 를 ``reference_nifti`` 의 physical grid에 최인접으로 리샘플해 저장합니다.

    격자가 동일하면 복사만 합니다.
    """
    src = Path(label_nifti).resolve()
    ref = Path(reference_nifti).resolve()
    out = Path(output_nifti).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Label NIfTI not found: {src}")
    if not ref.is_file():
        raise FileNotFoundError(f"Reference NIfTI not found: {ref}")
    out.parent.mkdir(parents=True, exist_ok=True)

    moving = itk.imread(str(src))
    reference = itk.imread(str(ref))

    if _itk_images_same_grid(moving, reference):
        import shutil

        shutil.copy2(src, out)
        return out

    info = itk.template(moving)[1]
    pixel_type = info[0]
    dim = info[1]
    image_type = itk.Image[pixel_type, dim]
    interp = itk.NearestNeighborInterpolateImageFunction[image_type, itk.D].New()
    resampler = itk.ResampleImageFilter[image_type, image_type].New()
    resampler.SetInput(moving)
    resampler.SetInterpolator(interp)
    resampler.SetOutputParametersFromImage(reference)
    resampler.SetDefaultPixelValue(0)
    resampler.Update()
    output_image = resampler.GetOutput()
    itk.imwrite(output_image, str(out))
    return out


__all__ = ["resample_label_nifti_to_reference_geometry"]
