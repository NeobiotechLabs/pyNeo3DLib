"""DICOM → NIfTI 고수준 파사드 (변환 하위 모듈 `dicom_to_itk` 사용)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from .dicom_to_itk import load_dicom_folder_as_itk_image, write_itk_image_as_nifti

ProgressCallback = Optional[Callable[[str], None]]


@dataclass(frozen=True)
class DicomNiftiConversionResult:
    """변환 결과 메타데이터.

    Attributes:
        nifti_path: 저장된 NIfTI 절대 경로.
    """

    nifti_path: Path


class DicomNiftiConversionFacade:
    """DICOM 폴더 읽기·ITK 볼륨 생성·NIfTI 쓰기를 캡슐화.

    생성은 `VolumeConversionFactory.create_dicom_nifti_facade()` 권장.
    """

    def __init__(self, progress: ProgressCallback = None):
        self._progress = progress

    def load_dicom_as_itk(
        self,
        dicom_folder: Union[str, Path],
        series_uid: Optional[str] = None,
        *,
        patient_origin: bool = False,
    ):
        """DICOM 폴더 → itk.Image (중간 표현). ``patient_origin``: DICOM 환자 원점 사용 여부."""
        return load_dicom_folder_as_itk_image(
            dicom_folder,
            series_uid=series_uid,
            progress=self._progress,
            patient_origin=patient_origin,
        )

    def write_itk_as_nifti(self, image, output_path: Union[str, Path]) -> Path:
        """itk.Image → 디스크 NIfTI."""
        path = Path(output_path).resolve()
        write_itk_image_as_nifti(image, path)
        return path

    def convert_folder_to_nifti(
        self,
        dicom_folder: Union[str, Path],
        output_nifti: Union[str, Path],
        series_uid: Optional[str] = None,
        *,
        patient_origin: bool = False,
    ) -> DicomNiftiConversionResult:
        """DICOM 폴더를 한 번에 NIfTI로 저장합니다. 기본 ITK 원점 (0,0,0)."""
        itk_image = self.load_dicom_as_itk(
            dicom_folder,
            series_uid=series_uid,
            patient_origin=patient_origin,
        )
        path = self.write_itk_as_nifti(itk_image, output_nifti)
        return DicomNiftiConversionResult(nifti_path=path)


__all__ = ["DicomNiftiConversionFacade", "DicomNiftiConversionResult", "ProgressCallback"]
