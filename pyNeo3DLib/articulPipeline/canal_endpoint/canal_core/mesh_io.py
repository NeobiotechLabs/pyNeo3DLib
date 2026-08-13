"""STL 메쉬 로드."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pyvista as pv

from .mesh import ensure_polydata


class MeshLoader(Protocol):
    def load(self, path: Path | str) -> pv.PolyData: ...


class StlMeshLoader:
    def load(self, path: Path | str) -> pv.PolyData:
        return ensure_polydata(pv.read(str(Path(path).expanduser().resolve())))

    def load_optional(self, path: Path | str | None) -> pv.PolyData | None:
        if path is None:
            return None
        return self.load(path)

    @staticmethod
    def resolve_optional(path: Path | str | None, label: str) -> Path | None:
        if path is None:
            return None
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{label} STL 없음: {resolved}")
        return resolved

    @staticmethod
    def resolve_required(path: Path | str | None, label: str) -> Path:
        if path is None:
            raise ValueError(f"{label} STL 경로가 필요합니다.")
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{label} STL 없음: {resolved}")
        return resolved
