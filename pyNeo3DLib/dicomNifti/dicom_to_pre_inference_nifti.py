"""DICOM 폴더를 nnU-Net 등에 넣기 위한 **단일 볼륨 NIfTI** 로 저장합니다.

Windows에서 DICOM 트리 경로에 비ASCII가 있으면 ASCII 임시 폴더로 복사한 뒤 읽습니다.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, Optional, Union

from .dicom_nifti_facade import DicomNiftiConversionFacade, DicomNiftiConversionResult
from .factory import VolumeConversionFactory
from .windows_path_staging import cleanup_staging_roots, stage_directory_copy

ProgressCallback = Optional[Callable[[str], None]]


def dicom_folder_to_pre_inference_nifti(
    dicom_folder: Union[str, Path],
    output_nifti: Union[str, Path],
    *,
    series_uid: Optional[str] = None,
    progress: ProgressCallback = None,
    facade: Optional[DicomNiftiConversionFacade] = None,
    patient_origin: bool = False,
) -> DicomNiftiConversionResult:
    """DICOM 시리즈를 하나의 NIfTI로 씁니다.

    :param dicom_folder: DICOM 파일이 들어 있는 디렉터리
    :param output_nifti: 저장할 ``.nii`` / ``.nii.gz`` 경로 (부모 폴더는 없으면 생성)
    :param patient_origin: False(기본)면 ITK 원점 (0,0,0). True면 ImagePositionPatient.
    :param series_uid: 지정 시 해당 시리즈만 사용. None이면 시리즈 선택 규칙은
        ``load_dicom_folder_as_itk_image`` (가장 많은 인스턴스 등)에 따릅니다.
    :param facade: None이면 ``VolumeConversionFactory.create_dicom_nifti_facade`` 로 생성
    :returns: ``DicomNiftiConversionResult`` (``nifti_path``)

    이 함수는 **추론·메시를 실행하지 않습니다.** 산출 NIfTI만 디스크에 남습니다.
    """
    dicom_orig = Path(dicom_folder).expanduser().resolve()
    if not dicom_orig.is_dir():
        raise NotADirectoryError(f"Not a DICOM directory: {dicom_orig}")

    out = Path(output_nifti).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    f = facade or VolumeConversionFactory.create_dicom_nifti_facade(progress=progress)

    cleanup_roots: list[Optional[Path]] = []
    itk_image = None
    try:
        d_st = stage_directory_copy(
            dicom_orig,
            prefix="ds_dicom_",
            progress=progress,
            label="DICOM folder",
        )
        if d_st.cleanup_root is not None:
            cleanup_roots.append(d_st.cleanup_root)

        itk_image = f.load_dicom_as_itk(
            d_st.effective_path,
            series_uid=series_uid,
            patient_origin=patient_origin,
        )
        path = f.write_itk_as_nifti(itk_image, out)
        return DicomNiftiConversionResult(nifti_path=path)
    finally:
        cleanup_staging_roots(cleanup_roots)
        itk_image = None
        gc.collect()


__all__ = ["dicom_folder_to_pre_inference_nifti"]
