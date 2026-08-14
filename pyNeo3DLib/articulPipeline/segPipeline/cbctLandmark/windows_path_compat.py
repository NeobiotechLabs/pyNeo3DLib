"""Windows 비ASCII(한글 등) 경로 호환 유틸.

Windows에서 ITK / SimpleITK(nifti IO)는 경로에 비ASCII 문자가 포함되면
파일을 열지 못한다("Unable to open ... for reading"). 경로 중
**디렉토리 이름 또는 파일 이름 어느 하나라도** 비ASCII면 실패하므로,
볼륨·임시 폴더 경로를 판별해 ASCII 전용 임시 위치를 만드는 유틸을 제공한다.

cbctSeg 의 ``windows_native_path_staging`` 과 동일한 방식·규약이며,
Linux/macOS에서는 스테이징이 필요하지 않습니다.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Union

__all__ = [
    "ascii_system_temp_parent",
    "ascii_volume_name",
    "create_ascii_staging_root",
    "native_io_staging_needed",
    "path_is_pure_ascii",
]


def native_io_staging_needed() -> bool:
    """네이티브 IO 경로 스테이징이 필요한 플랫폼인지 (Windows 만 해당)."""
    return sys.platform == "win32"


def path_is_pure_ascii(path: Union[str, Path]) -> bool:
    """경로 문자열 전체가 ASCII만 사용하는지."""
    try:
        str(Path(path).expanduser().resolve()).encode("ascii")
        return True
    except (UnicodeEncodeError, OSError):
        return False


def ascii_system_temp_parent() -> Path:
    """대개 ASCII인 ``%SYSTEMROOT%\\Temp`` (사용자 TEMP 가 비ASCII일 때 우회)."""
    wr = os.environ.get("SYSTEMROOT", r"C:\Windows")
    t = Path(wr) / "Temp"
    try:
        t.mkdir(parents=True, exist_ok=True)
    except OSError:
        t = Path(tempfile.gettempdir())
    if path_is_pure_ascii(t):
        return t
    return Path(tempfile.gettempdir())


def create_ascii_staging_root(prefix: str) -> str:
    """ASCII 전용 임시 루트 폴더를 새로 만들어 경로(str)로 반환."""
    return tempfile.mkdtemp(prefix=prefix, dir=str(ascii_system_temp_parent()))


def ascii_volume_name(vol: Union[str, Path]) -> str:
    """ASCII 볼륨 파일명. ``.nii.gz`` 등 복합 확장자는 그대로 유지.

    예: ``이판임님 CT.nii.gz`` → ``volume.nii.gz``
    """
    return "volume" + "".join(Path(vol).suffixes)
