"""계산 결과 → PyVista 시각화 장면 조립."""

from __future__ import annotations

import numpy as np

from core.canal_endpoint.mesh_io import StlMeshLoader
from core.occlusal_plane.compute import OcclusalComputeData, OcclusalPlaneInputs
from core.occlusal_plane.visualization.style import OcclusalSceneStyle
from core.occlusal_plane.models import OcclusalMeshes, OcclusalPlaneGeometry, OcclusalScene
from core.occlusal_plane.visualization.scene_extent import SceneExtentCalculator


def _plane_geometry(center: np.ndarray, normal: np.ndarray, size_mm: float) -> OcclusalPlaneGeometry:
    return OcclusalPlaneGeometry(center=center, normal=normal, size_mm=size_mm)


def build_extent_landmarks(
    cranial: dict[str, np.ndarray],
    mef: dict[str, np.ndarray],
    *,
    data: OcclusalComputeData,
) -> dict[str, np.ndarray]:
    g = data.geom
    return {
        **cranial,
        **mef,
        "P_occ_ans": g.p_occ_ans,
        "MeF_mid": g.mef_mid,
        "P_occ_mef": g.p_occ_mef,
        "P_occ_mid": g.p_occ_mid,
    }


def load_occlusal_meshes(
    inputs: OcclusalPlaneInputs,
    *,
    mesh_loader: StlMeshLoader | None = None,
) -> OcclusalMeshes:
    loader = mesh_loader or StlMeshLoader()
    canal_path = StlMeshLoader.resolve_required(inputs.mandibular_canal_path, "mandibular_canal")
    upper_path = StlMeshLoader.resolve_optional(inputs.upper_skull_path, "upper_skull")
    mandible_path = StlMeshLoader.resolve_optional(inputs.mandible_path, "mandible")
    teeth = tuple(loader.load(path) for path in inputs.teeth_paths)
    return OcclusalMeshes(
        upper_skull=loader.load_optional(upper_path),
        mandibular_canal=loader.load(canal_path),
        mandible=loader.load_optional(mandible_path),
        teeth=teeth,
    )


def build_occlusal_scene(
    data: OcclusalComputeData,
    inputs: OcclusalPlaneInputs,
    *,
    style: OcclusalSceneStyle | None = None,
    mesh_loader: StlMeshLoader | None = None,
) -> OcclusalScene:
    style = style or OcclusalSceneStyle()
    meshes = load_occlusal_meshes(inputs, mesh_loader=mesh_loader)
    extent = SceneExtentCalculator(
        min_mm=style.extent_min_mm,
        scale=style.plane_scale,
    )
    all_pts = build_extent_landmarks(data.cranial, data.mef, data=data)
    plane_size = extent.for_scene(all_pts, meshes)
    geom = data.geom
    n = geom.occlusal_normal

    return OcclusalScene(
        cranial_landmarks=data.cranial,
        mef_landmarks=data.mef,
        meshes=meshes,
        msp_plane=_plane_geometry(geom.msp.center, geom.msp.normal, plane_size),
        occlusal_plane_normal=n,
        mef_occlusal_plane=_plane_geometry(geom.p_occ_mef, n, plane_size),
        mef_occlusal_plane_reference=geom.p_occ_mef,
        ans_occlusal_plane=_plane_geometry(geom.p_occ_ans, n, plane_size),
        ans_occlusal_plane_reference=geom.p_occ_ans,
        mid_occlusal_plane=_plane_geometry(geom.p_occ_mid, n, plane_size),
        mid_occlusal_plane_reference=geom.p_occ_mid,
        mef_mid=geom.mef_mid,
    )
