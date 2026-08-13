"""
3D Slicer ``vtkBinaryLabelmapToClosedSurfaceConversionRule::CreateClosedSurface`` 의
**DecimatePro → WindowedSinc** 구간과 동일한 VTK 파라미터 (등고 추출 제외).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def trimesh_to_pyvista_polydata(mesh: Any):
    import trimesh

    if mesh is None or not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return None
    import pyvista as pv

    n = mesh.faces.shape[0]
    faces = np.hstack(
        [np.full((n, 1), 3, dtype=np.int64), mesh.faces.astype(np.int64)]
    ).ravel()
    return pv.PolyData(np.ascontiguousarray(mesh.vertices), faces)


def apply_slicer_style_mesh_ops(poly, *, decimation_factor: float, smoothing_factor: float):
    import vtk

    processing = poly

    if decimation_factor > 0.0:
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputData(processing)
        decimator.SetFeatureAngle(60)
        decimator.SplittingOff()
        decimator.PreserveTopologyOn()
        decimator.SetMaximumError(1)
        decimator.SetTargetReduction(decimation_factor)
        decimator.Update()
        processing = decimator.GetOutput()

    if smoothing_factor > 0.0:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(processing)
        pass_band = 10.0 ** (-4.0 * smoothing_factor)
        number_of_iterations = int(20 + smoothing_factor * 40)
        smoother.SetNumberOfIterations(number_of_iterations)
        smoother.SetPassBand(pass_band)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        processing = smoother.GetOutput()

    return processing


def slicer_style_polydata_to_trimesh(poly) -> Optional[Any]:
    import pyvista as pv

    from .mesh_vtk import polydata_to_trimesh

    return polydata_to_trimesh(pv.wrap(poly))


def slicer_style_postprocess_trimesh(
    mesh: Any,
    *,
    decimation_factor: float,
    smoothing_factor: float,
) -> Optional[Any]:
    pv_in = trimesh_to_pyvista_polydata(mesh)
    if pv_in is None:
        return None
    out = apply_slicer_style_mesh_ops(
        pv_in,
        decimation_factor=decimation_factor,
        smoothing_factor=smoothing_factor,
    )
    return slicer_style_polydata_to_trimesh(out)


__all__: list[str] = []
