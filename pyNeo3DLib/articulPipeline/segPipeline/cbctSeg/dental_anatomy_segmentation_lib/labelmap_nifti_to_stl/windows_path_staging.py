"""
Windows에서 ITK 등이 **비ASCII 경로**로 파일을 열지 못할 때 ASCII 임시 경로로 복사(staging).

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
    try:
        str(path.expanduser().resolve()).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ascii_system_temp_parent() -> Path:
    wr = os.environ.get("SYSTEMROOT", r"C:\Windows")
    t = Path(wr) / "Temp"
    t.mkdir(parents=True, exist_ok=True)
    return t


def native_io_staging_enabled() -> bool:
    return sys.platform == "win32"


def file_needs_native_staging(path: Path) -> bool:
    return native_io_staging_enabled() and not path_string_is_pure_ascii(
        Path(path).expanduser().resolve()
    )


@dataclass
class StagedPath:
    effective_path: Path
    cleanup_root: Optional[Path]


def _staging_filename_suffix(path: Path) -> str:
    suf = "".join(path.suffixes)
    return suf if suf else (path.suffix or "")


def stage_file_copy(
    src: Union[str, Path],
    *,
    prefix: str,
    progress: ProgressCallback = None,
    label: str = "file",
) -> StagedPath:
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


def cleanup_staging_roots(roots: List[Optional[Path]]) -> None:
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
    "file_needs_native_staging",
    "native_io_staging_enabled",
    "path_string_is_pure_ascii",
    "stage_file_copy",
]
