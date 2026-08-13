"""VTK isosurface extraction via **PyVista** (label → mesh; no other backend)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def vtk_contour_binary_roi(
    mask_zyx: np.ndarray,
    origin_voxel_zyx: Tuple[int, int, int],
    spacing_zyx: Optional[Tuple[float, float, float]] = None,
):
    """
    Isosurface at 0.5 on a binary subvolume.

    ``mask_zyx`` 는 (Z, Y, X). ``origin_voxel_zyx`` 는 ``mask[0,0,0]`` 의 전역 격자 인덱스 ``(z0,y0,x0)``.

    - ``spacing_zyx`` 가 None: spacing 1 인 **연속 복셀 인덱스** 공간 (정점 = 전역 x,y,z 인덱스).
      월드 mm 는 호출측에서 ITK ``TransformContinuousIndexToPhysicalPoint``(동일 수식)으로 맞춤.
    - ``spacing_zyx`` 가 있으면 축 정렬 mm (``get_zooms`` 만 반영, 회전 없음 — 레거시).
    """
    import pyvista as pv

    z0, y0, x0 = (int(origin_voxel_zyx[0]), int(origin_voxel_zyx[1]), int(origin_voxel_zyx[2]))
    vol_vtk = np.transpose(mask_zyx.astype(np.float32, copy=False), (2, 1, 0))
    grid = pv.ImageData(dimensions=vol_vtk.shape)
    if spacing_zyx is None:
        grid.spacing = (1.0, 1.0, 1.0)
        grid.origin = (float(x0), float(y0), float(z0))
    else:
        sz, sy, sx = (float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2]))
        grid.spacing = (sx, sy, sz)
        grid.origin = (x0 * sx, y0 * sy, z0 * sz)
    grid.point_data["s"] = np.ascontiguousarray(vol_vtk).ravel(order="F")
    return grid.contour(isosurfaces=[0.5], scalars="s")


def polydata_to_trimesh(surf: Any) -> Optional[Any]:
    import trimesh

    if surf.n_points == 0 or surf.n_cells == 0:
        return None
    fc = np.asarray(surf.faces)
    if fc.size == 0 or fc.size % 4 != 0:
        return None
    faces = fc.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        return None
    tri = np.ascontiguousarray(faces[:, 1:4], dtype=np.int64)
    verts = np.ascontiguousarray(surf.points, dtype=np.float64)
    return trimesh.Trimesh(vertices=verts, faces=tri, process=False)
