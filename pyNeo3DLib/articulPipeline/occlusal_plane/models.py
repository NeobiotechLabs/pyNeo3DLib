"""교합면 장면 도메인 모델."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv


@dataclass(frozen=True)
class OcclusalMeshes:
    upper_skull: pv.PolyData | None = None
    mandibular_canal: pv.PolyData | None = None
    mandible: pv.PolyData | None = None
    teeth: tuple[pv.PolyData, ...] = ()


@dataclass(frozen=True)
class OcclusalPlaneGeometry:
    center: np.ndarray
    normal: np.ndarray
    size_mm: float


@dataclass(frozen=True)
class OcclusalScene:
    cranial_landmarks: dict[str, np.ndarray]
    mef_landmarks: dict[str, np.ndarray]
    meshes: OcclusalMeshes
    msp_plane: OcclusalPlaneGeometry
    occlusal_plane_normal: np.ndarray
    mef_occlusal_plane: OcclusalPlaneGeometry | None = None
    mef_occlusal_plane_reference: np.ndarray | None = None
    ans_occlusal_plane: OcclusalPlaneGeometry | None = None
    ans_occlusal_plane_reference: np.ndarray | None = None
    mid_occlusal_plane: OcclusalPlaneGeometry | None = None
    mid_occlusal_plane_reference: np.ndarray | None = None
    mef_mid: np.ndarray | None = None
