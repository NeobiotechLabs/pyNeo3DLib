"""
파이프라인 ``DicomToNiftiConverter`` 프로토콜과 시그니처 호환 (구조적 서브타입).
``DicomNiftiConversionFacade`` 에 위임합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .dicom_to_itk import ProgressCallback
from .dicom_nifti_facade import DicomNiftiConversionFacade
from .factory import VolumeConversionFactory


class DicomFolderToNiftiConverter:
    """DICOM 폴더 → NIfTI (``DicomNiftiConversionFacade`` 위임)."""

    def __init__(
        self,
        *,
        facade: Optional[DicomNiftiConversionFacade] = None,
        progress: ProgressCallback = None,
    ) -> None:
        self._facade = facade or VolumeConversionFactory.create_dicom_nifti_facade(
            progress=progress
        )

    def convert_dicom_to_nifti(
        self,
        dicom_folder: Union[str, Path],
        output_nifti: Union[str, Path],
        *,
        patient_origin: bool = False,
    ) -> Path:
        return self._facade.convert_folder_to_nifti(
            dicom_folder,
            output_nifti,
            patient_origin=patient_origin,
        ).nifti_path


__all__ = ["DicomFolderToNiftiConverter"]
