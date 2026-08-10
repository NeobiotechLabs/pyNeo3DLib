"""
데이터 변환: DICOM 시리즈 → itk.Image / NIfTI 보조.

이 패키지에서 관리합니다 (itk 필요. Slicer 불필요).

``load_dicom_folder_as_itk_image`` 는 ITK GDCM(``GDCMSeriesFileNames`` + ``ImageSeriesReader``)으로
시리즈를 읽고, 간격·방향은 리더가 두고 ``patient_origin=False``(기본)일 때만
``Origin`` 을 ``(0,0,0)`` 으로 맞춥니다 (``patient_origin=True`` 면 DICOM 환자 원점 유지).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional, Union

from .windows_path_staging import ascii_system_temp_parent, path_string_is_pure_ascii

ProgressCallback = Optional[Callable[[str], None]]

__all__ = [
    "ProgressCallback",
    "load_dicom_folder_as_itk_image",
    "write_itk_image_as_nifti",
]


def _pick_gdcm_series_uid(names, series_uid: Optional[str]):
    """``GDCMSeriesFileNames`` 인스턴스에서 ``GetFileNames`` 에 넘길 시리즈 UID."""
    uids = names.GetSeriesUIDs()
    if not uids:
        raise ValueError("GDCM: 디렉터리에서 DICOM 시리즈를 찾지 못했습니다")
    if series_uid is not None:
        want = str(series_uid).strip()
        for u in uids:
            if str(u).strip() == want:
                return u
        raise ValueError(
            f"SeriesInstanceUID가 '{want}'인 시리즈가 없습니다. "
            f"발견된 UID 수: {len(uids)}"
        )
    # 인스턴스(파일) 수가 가장 많은 시리즈 선택
    return max(uids, key=lambda u: len(names.GetFileNames(u)))


def _log(progress: ProgressCallback, msg: str) -> None:
    if progress:
        progress(msg)


def load_dicom_folder_as_itk_image(
    folder: Union[str, Path],
    series_uid: Optional[str] = None,
    progress: ProgressCallback = None,
    *,
    patient_origin: bool = False,
):
    """
    DICOM 폴더에서 시리즈를 고르고 itk.Image[float, 3]로 반환합니다.

    **구현:** ITK ``GDCMSeriesFileNames``(하위 폴더 포함) + ``ImageSeriesReader``(GDCMImageIO).
    읽은 뒤 ``patient_origin=False``(기본)이면 ``SetOrigin((0,0,0))`` 만 적용하고
    spacing·direction은 리더 결과를 유지합니다.

    :param patient_origin: False(기본)면 ITK 원점 (0,0,0). True면 리더가 둔 DICOM 원점 유지.

    :param folder: DICOM 파일이 들어 있는 디렉터리
    :param series_uid: 지정 시 해당 ``SeriesInstanceUID``만 사용.
        None이면 인스턴스 수가 가장 많은 시리즈를 선택합니다.
    :param progress: 로그 콜백
    """
    import itk

    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    _log(progress, f"ITK GDCM series reader: {folder}")
    names = itk.GDCMSeriesFileNames.New()
    names.SetUseSeriesDetails(True)
    names.RecursiveOn()
    names.SetDirectory(str(folder))
    chosen_uid = _pick_gdcm_series_uid(names, series_uid)
    file_names = names.GetFileNames(chosen_uid)
    n_files = len(file_names)
    if n_files == 0:
        raise ValueError(f"GDCM: 시리즈에 DICOM 파일이 없습니다 (UID={chosen_uid})")

    reader = itk.ImageSeriesReader[itk.Image[itk.F, 3]].New()
    reader.SetImageIO(itk.GDCMImageIO.New())
    reader.SetFileNames(file_names)
    reader.Update()
    image = reader.GetOutput()
    image.DisconnectPipeline()

    if not patient_origin:
        image.SetOrigin((0.0, 0.0, 0.0))

    _log(
        progress,
        f"  series UID = {chosen_uid!r}, files = {n_files}, "
        f"size = {tuple(image.GetLargestPossibleRegion().GetSize())}, "
        f"spacing = {tuple(image.GetSpacing())}, "
        f"origin = {tuple(image.GetOrigin())}",
    )
    return image


def _windows_short_path(path: Path) -> Optional[Path]:
    """존재하는 경로의 8.3 짧은 경로 (ITK NIfTI 쓰기 우회)."""
    if sys.platform != "win32":
        return None
    p = Path(path).resolve()
    if not p.exists():
        return None
    import ctypes

    buf = ctypes.create_unicode_buffer(4096)
    rc = ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, len(buf))
    if rc == 0:
        return None
    return Path(buf.value)


def write_itk_image_as_nifti(image, output_path: Union[str, Path]) -> None:
    """itk.Image를 NIfTI(.nii 또는 .nii.gz)로 저장 (nnU-Net 등 후속 파이프라인용).

    Windows에서 비ASCII 경로는 ITK NIfTI writer가 실패할 수 있습니다.
    부모만 8.3 짧은 경로로 바꿔도 **파일명에 한글 등이 남으면** 깨지므로,
    전체 경로에 비ASCII가 있으면 ASCII-only 임시 파일에 쓴 뒤 ``move`` 합니다.
    """
    import shutil
    import tempfile

    import itk

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if sys.platform != "win32" or path_string_is_pure_ascii(path):
        itk.imwrite(image, str(path))
        return

    last_exc: Optional[BaseException] = None
    # 파일명이 ASCII일 때만: 짧은 경로 부모 + 원래 이름 (디스크 이동 없이 시도)
    name_ok = path_string_is_pure_ascii(Path(path.name))
    short_parent = _windows_short_path(path.parent.resolve()) if name_ok else None
    if short_parent is not None:
        try:
            itk.imwrite(image, str(short_parent / path.name))
            return
        except RuntimeError as e:
            last_exc = e

    tmp_dir = ascii_system_temp_parent()
    tmp_suffix = ".nii.gz" if str(path).lower().endswith(".gz") else path.suffix or ".nii.gz"
    fd, tmp_name = tempfile.mkstemp(suffix=tmp_suffix, prefix="itk_", dir=str(tmp_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        itk.imwrite(image, str(tmp_path))
        shutil.move(str(tmp_path), str(path))
    except Exception as e:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        msg = (
            f"Failed to save NIfTI: {path}\n"
            "ITK NIfTI write failed on Windows with this path (often non-ASCII paths). "
            "Use an ASCII-only output directory or enable library path staging."
        )
        raise RuntimeError(msg) from (last_exc if last_exc is not None else e)
