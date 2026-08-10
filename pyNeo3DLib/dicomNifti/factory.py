"""
볼륨·이미지 변환 관련 객체 생성 전용 팩토리.

DICOM→NIfTI 구현은 같은 패키지의 `dicom_to_itk`, `dicom_nifti_facade`에 두고,
외부에서는 이 클래스의 팩토리 메서드만 알면 됩니다.
"""

from __future__ import annotations

from .dicom_nifti_facade import DicomNiftiConversionFacade, ProgressCallback


class VolumeConversionFactory:
    """
    변환기(파사드 등)를 한 곳에서 생성합니다.

    예:
        facade = VolumeConversionFactory.create_dicom_nifti_facade(progress=print)
        result = facade.convert_folder_to_nifti(dicom_dir, out_nii)
    """

    __slots__ = ()

    @staticmethod
    def create_dicom_nifti_facade(progress: ProgressCallback = None) -> DicomNiftiConversionFacade:
        """DICOM 폴더 → ITK → NIfTI 파이프라인 파사드."""
        return DicomNiftiConversionFacade(progress=progress)


__all__ = ["VolumeConversionFactory"]
