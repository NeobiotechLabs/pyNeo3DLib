"""장면 범위(평면 크기) 계산."""

from __future__ import annotations

import numpy as np

from core.occlusal_plane.models import OcclusalMeshes


class SceneExtentCalculator:
    def __init__(self, *, min_mm: float = 80.0, scale: float = 1.15) -> None:
        self._min_mm = min_mm
        self._scale = scale

    def for_scene(
        self,
        landmarks: dict[str, np.ndarray],
        meshes: OcclusalMeshes,
    ) -> float:
        arrays: list[np.ndarray] = list(landmarks.values())
        for mesh in (meshes.upper_skull, meshes.mandibular_canal, meshes.mandible):
            if mesh is not None:
                arrays.append(np.asarray(mesh.points))
        for tooth in meshes.teeth:
            arrays.append(np.asarray(tooth.points))
        return self.compute(*arrays)

    def compute(self, *point_arrays: np.ndarray) -> float:
        if not point_arrays:
            return self._min_mm * self._scale
        stacked = np.vstack(point_arrays)
        extent = np.max(stacked, axis=0) - np.min(stacked, axis=0)
        return max(float(np.max(extent)), self._min_mm) * self._scale
