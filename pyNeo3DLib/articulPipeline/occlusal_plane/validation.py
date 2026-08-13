"""교합평면 파이프라인 파일 검증."""

from __future__ import annotations

from pathlib import Path


def require_existing_file(path: Path | str | None, *, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} 경로가 제공되지 않았습니다.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} 파일 없음: {resolved}")
    return resolved
