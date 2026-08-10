"""CBCT 볼륨 전처리(히스토그램·리샘플)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

import itk
import numpy as np
import SimpleITK as sitk

from .markups import spacing_to_scale_key

logger = logging.getLogger(__name__)


def correct_histo(
    filepath: str,
    outpath: str,
    min_percent: float = 0.01,
    max_percent: float = 0.95,
    i_min: int = -1500,
    i_max: int = 4000,
):
    logger.info("Correcting scan contrast: %s", filepath)
    input_img = sitk.ReadImage(filepath)
    input_img = sitk.Cast(input_img, sitk.sitkFloat32)
    img = sitk.GetArrayFromImage(input_img)

    img_min = np.min(img)
    img_max = np.max(img)
    img_range = img_max - img_min

    definition = 1000
    histo = np.histogram(img, definition)
    cum = np.cumsum(histo[0])
    cum = cum - np.min(cum)
    cum = cum / np.max(cum)

    res_high = np.argmax(cum > max_percent)
    res_max = (res_high * img_range) / definition + img_min

    res_low = np.argmax(cum > min_percent)
    res_min = (res_low * img_range) / definition + img_min

    res_min = max(res_min, i_min)
    res_max = min(res_max, i_max)

    img = np.where(img > res_max, res_max, img)
    img = np.where(img < res_min, res_min, img)

    output = sitk.GetImageFromArray(img)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)
    return output


def _resample_image(
    input_img, size, spacing, origin, direction, interpolator, vector_image_type
):
    resample_type = itk.ResampleImageFilter[vector_image_type, vector_image_type]
    resample_filter = resample_type.New()
    resample_filter.SetOutputSpacing(spacing.tolist())
    resample_filter.SetOutputOrigin(origin)
    resample_filter.SetOutputDirection(direction)
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetSize(size)
    resample_filter.SetInput(input_img)
    resample_filter.Update()
    return resample_filter.GetOutput()


def set_spacing(
    filepath: str,
    output_spacing=(0.5, 0.5, 0.5),
    outpath: str | Path | None = None,
):
    logger.info("Resample %s with spacing %s", filepath, output_spacing)
    img = itk.imread(filepath)

    spacing = np.array(img.GetSpacing())
    output_spacing = np.array(output_spacing)

    if not np.array_equal(spacing, output_spacing):
        size = itk.size(img)
        scale = spacing / output_spacing
        output_size = (np.array(size) * scale).astype(int).tolist()
        output_origin = img.GetOrigin()
        output_physical_size = np.array(output_size) * np.array(output_spacing)
        input_physical_size = np.array(size) * spacing
        output_origin = np.array(output_origin) - (
            output_physical_size - input_physical_size
        ) / 2.0

        img_info = itk.template(img)[1]
        pixel_type = img_info[0]
        pixel_dimension = img_info[1]
        vector_image_type = itk.Image[pixel_type, pixel_dimension]

        if any(seg in os.path.basename(filepath) for seg in ["seg", "Seg"]):
            interpolator_type = itk.NearestNeighborInterpolateImageFunction[
                vector_image_type, itk.D
            ]
        else:
            interpolator_type = itk.LinearInterpolateImageFunction[
                vector_image_type, itk.D
            ]

        interpolator = interpolator_type.New()
        resampled = _resample_image(
            img,
            output_size,
            output_spacing,
            output_origin,
            img.GetDirection(),
            interpolator,
            vector_image_type,
        )

        if outpath is not None:
            itk.imwrite(resampled, outpath)
        return resampled

    if outpath is not None:
        itk.imwrite(img, outpath)
    return img


def build_patient_scans(
    vol: str,
    spacing: List[float],
    scale_keys: List[str],
    temp_fold: str,
) -> Dict[str, str]:
    """볼륨별 멀티스케일 리샘플 경로 dict (scale_key → path)."""
    basename = os.path.basename(vol)
    scan_parts = basename.split(".")
    scans: Dict[str, str] = {}

    for sp, spac in zip(spacing, scale_keys):
        new_name = ""
        for i, element in enumerate(scan_parts):
            if i == 0:
                new_name = scan_parts[0] + "_scan_sp" + spac
            else:
                new_name += "." + element
        outpath = os.path.join(temp_fold, new_name)
        if not os.path.exists(outpath):
            temp_path = os.path.join(temp_fold, basename)
            if not os.path.exists(temp_path):
                correct_histo(vol, temp_path, 0.01, 0.99)
            set_spacing(temp_path, [sp, sp, sp], outpath)
        scans[spac] = outpath

    return scans


def build_patients_dict(
    vol: str, spacing: List[float], scale_keys: List[str], temp_fold: str
) -> dict:
    basename = os.path.basename(vol)
    return {
        basename: {
            "scan": vol,
            "scans": build_patient_scans(vol, spacing, scale_keys, temp_fold),
        }
    }


# 레거시 이름
CorrectHisto = correct_histo
SetSpacing = set_spacing
