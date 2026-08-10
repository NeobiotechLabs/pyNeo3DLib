"""
Windows에서 ITK 등이 **비ASCII 경로**로 파일을 열지 못할 때 ASCII 임시 경로로 복사합니다.

Linux/macOS에서는 스테이징을 하지 않습니다. ``dicom_nifti`` 패키지 단독 사용을 위해
상위 패키지에 의존하지 않습니다.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

ProgressCallback = Optional[Callable[[str], None]]

# Windows 스테이징 기본 위치 (C: 용량 부족 방지). 환경변수로 덮어쓸 수 있음.
_STAGING_TEMP_ENV = "DICOM_NIFTI_STAGING_TEMP"
_DEFAULT_WINDOWS_STAGING_TEMP = Path(r"F:\Temp")


def path_string_is_pure_ascii(path: Path) -> bool:
    """경로 문자열 전체가 ASCII만 사용하는지."""
    try:
        str(path.expanduser().resolve()).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ascii_system_temp_parent() -> Path:
    """Windows 스테이징용 ASCII 임시 폴더.

    기본 ``F:\\Temp`` (C: 용량 부족 방지). ``DICOM_NIFTI_STAGING_TEMP`` 로 덮어쓰기 가능.
    F: 를 쓸 수 없으면 ``%SYSTEMROOT%\\Temp`` 로 폴백합니다.
    """
    if sys.platform != "win32":
        t = Path(tempfile.gettempdir())
        t.mkdir(parents=True, exist_ok=True)
        return t

    candidates: list[Path] = []
    env = os.environ.get(_STAGING_TEMP_ENV, "").strip()
    if env:
        candidates.append(Path(env))
    candidates.append(_DEFAULT_WINDOWS_STAGING_TEMP)
    wr = os.environ.get("SYSTEMROOT", r"C:\Windows")
    candidates.append(Path(wr) / "Temp")

    for t in candidates:
        try:
            t.mkdir(parents=True, exist_ok=True)
            return t
        except OSError:
            continue

    raise OSError(
        "No writable staging temp directory found "
        f"(tried: {', '.join(str(c) for c in candidates)})"
    )


def native_io_staging_enabled() -> bool:
    return sys.platform == "win32"


def directory_tree_has_non_ascii_path(
    root: Path,
    *,
    max_entries: int = 500_000,
) -> bool:
    """폴더 자체 또는 그 아래 어느 한 경로라도 비ASCII면 True."""
    if not native_io_staging_enabled():
        return False
    root = root.expanduser().resolve()
    if not path_string_is_pure_ascii(root):
        return True
    n = 0
    for p in root.rglob("*"):
        n += 1
        if n > max_entries:
            break
        if not path_string_is_pure_ascii(p):
            return True
    return False


@dataclass
class StagedPath:
    """``effective_path`` 로 읽기/쓰기. ``cleanup_root`` 가 있으면 처리 후 ``rmtree``."""

    effective_path: Path
    cleanup_root: Optional[Path]


def stage_directory_copy(
    src: Union[str, Path],
    *,
    prefix: str,
    progress: ProgressCallback = None,
    label: str = "folder",
) -> StagedPath:
    """
    필요 시 ``src`` 트리 전체를 ``ascii_system_temp_parent()`` 아래 ASCII 폴더로 복사.

    :returns: 스테이징 불필요 시 ``(src, None)``, 필요 시 ``(복사본 루트, 삭제할 상위 temp 루트)``
    """
    src = Path(src).expanduser().resolve()
    if not src.is_dir():
        raise NotADirectoryError(str(src))

    if not native_io_staging_enabled():
        return StagedPath(src, None)

    need = (not path_string_is_pure_ascii(src)) or directory_tree_has_non_ascii_path(
        src
    )
    if not need:
        return StagedPath(src, None)

    if progress:
        progress(
            f"Non-ASCII path workaround: copying {label} to an ASCII temp folder "
            f"(may take a while for large data)…"
        )

    tmp_root = Path(
        tempfile.mkdtemp(prefix=prefix, dir=str(ascii_system_temp_parent()))
    )
    inner = tmp_root / "data"
    try:
        shutil.copytree(src, inner, dirs_exist_ok=False)
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise
    return StagedPath(inner, tmp_root)


def cleanup_staging_roots(roots: List[Optional[Path]]) -> None:
    """여러 스테이징 루트를 삭제 (디렉터리는 rmtree, 단일 파일은 unlink)."""
    seen: set = set()
    for r in roots:
        if r is None:
            continue
        p = Path(r).resolve()
        if p in seen:
            continue
        seen.add(p)
        if not p.exists():
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink()
            except OSError:
                pass


__all__ = [
    "ProgressCallback",
    "StagedPath",
    "ascii_system_temp_parent",
    "cleanup_staging_roots",
    "directory_tree_has_non_ascii_path",
    "native_io_staging_enabled",
    "path_string_is_pure_ascii",
    "stage_directory_copy",
]
