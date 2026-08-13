"""
Windows에서 ITK / SimpleITK / nnU-Net 등이 **비ASCII(한글 등) 경로**로 파일을 열지 못할 때,
ASCII 전용 임시 경로로 **복사(staging)** 해서 쓰기 위한 유틸리티.

Linux/macOS에서는 스테이징을 하지 않습니다.
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


def path_string_is_pure_ascii(path: Path) -> bool:
    """경로 문자열 전체가 ASCII만 사용하는지."""
    try:
        str(path.expanduser().resolve()).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ascii_system_temp_parent() -> Path:
    """대개 ASCII인 ``%SYSTEMROOT%\\Temp`` (사용자 TEMP 가 한글일 때 우회)."""
    wr = os.environ.get("SYSTEMROOT", r"C:\Windows")
    t = Path(wr) / "Temp"
    t.mkdir(parents=True, exist_ok=True)
    return t


def native_io_staging_enabled() -> bool:
    return sys.platform == "win32"


def directory_tree_has_non_ascii_path(
    root: Path,
    *,
    max_entries: int = 500_000,
) -> bool:
    """
    폴더 자체 또는 그 아래 **어느 한 경로라도** 비ASCII면 True.

    (예: 영문 폴더 아래 파일 이름만 한글인 경우도 잡기 위함.)
    """
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


def file_needs_native_staging(path: Path) -> bool:
    """단일 파일 경로가 네이티브 IO에 비안전한지."""
    return native_io_staging_enabled() and not path_string_is_pure_ascii(
        Path(path).expanduser().resolve()
    )


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
    필요 시 ``src`` 트리 전체를 ``%SYSTEMROOT%\\Temp`` 아래 ASCII 폴더로 복사.

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


def _staging_filename_suffix(path: Path) -> str:
    """
    임시 파일 확장자. ``Path.suffix`` 만 쓰면 ``.nii.gz`` 가 ``.gz`` 로 잘리므로
    ``suffixes`` 를 이어 ``.nii.gz`` · ``.tar.gz`` 등 복합 확장자를 유지합니다.
    """
    suf = "".join(path.suffixes)
    return suf if suf else (path.suffix or "")


def stage_file_copy(
    src: Union[str, Path],
    *,
    prefix: str,
    progress: ProgressCallback = None,
    label: str = "file",
) -> StagedPath:
    """단일 파일을 ASCII 임시 경로로 복사 (필요할 때만)."""
    src = Path(src).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(str(src))

    if not file_needs_native_staging(src):
        return StagedPath(src, None)

    if progress:
        progress(f"Non-ASCII path workaround: copying {label} to an ASCII temp file…")

    parent = ascii_system_temp_parent()
    fd, name = tempfile.mkstemp(
        prefix=prefix,
        suffix=_staging_filename_suffix(src),
        dir=str(parent),
    )
    os.close(fd)
    tmp = Path(name)
    try:
        shutil.copy2(src, tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return StagedPath(tmp, tmp)


def create_ascii_work_staging(
    *,
    prefix: str,
    progress: ProgressCallback = None,
) -> StagedPath:
    """
    사용자 ``work_dir`` 가 비ASCII일 때 쓸 **빈** ASCII 작업 루트.

    :returns: ``(작업 루트, rmtree 할 temp 루트)`` — 작업 루트와 temp 루트가 동일.
    """
    if not native_io_staging_enabled():
        raise RuntimeError("create_ascii_work_staging is only supported on Windows.")

    if progress:
        progress(
            "Non-ASCII work directory: processing in an ASCII temp folder; "
            "results will be copied back."
        )

    root = Path(
        tempfile.mkdtemp(prefix=prefix, dir=str(ascii_system_temp_parent()))
    )
    return StagedPath(root, root)


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
    "create_ascii_work_staging",
    "directory_tree_has_non_ascii_path",
    "file_needs_native_staging",
    "native_io_staging_enabled",
    "path_string_is_pure_ascii",
    "stage_directory_copy",
    "stage_file_copy",
]
