"""
DICOM → NIfTI 전용 패키지.

한 import 경로로 모아 둠: 파사드·팩토리·저수준 ITK 헬퍼·추론 전 NIfTI·파이프라인용 변환기.

이 디렉터리만 복사해 쓸 때는 부모 경로를 ``PYTHONPATH``에 두고 ``import dicom_nifti`` 하면 됩니다
(상위 ``DentalSegmentatorLib`` 불필요).

예::

    from dicom_nifti import (
        DicomNiftiConversionFacade,
        VolumeConversionFactory,
        dicom_folder_to_pre_inference_nifti,
        load_dicom_folder_as_itk_image,
    )
"""

from .dicom_nifti_facade import DicomNiftiConversionFacade, DicomNiftiConversionResult
from .dicom_to_itk import (
    ProgressCallback,
    load_dicom_folder_as_itk_image,
    write_itk_image_as_nifti,
)
from .dicom_to_pre_inference_nifti import dicom_folder_to_pre_inference_nifti
from .factory import VolumeConversionFactory
from .folder_to_nifti_converter import DicomFolderToNiftiConverter

__all__ = [
    "DicomFolderToNiftiConverter",
    "DicomNiftiConversionFacade",
    "DicomNiftiConversionResult",
    "VolumeConversionFactory",
    "ProgressCallback",
    "dicom_folder_to_pre_inference_nifti",
    "load_dicom_folder_as_itk_image",
    "write_itk_image_as_nifti",
]
