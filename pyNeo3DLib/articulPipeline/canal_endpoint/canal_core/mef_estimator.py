"""LMeF·RMeF 추정 (단일 책임)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from shared.constants import MEF_LANDMARKS

from .config import CanalPipelineConfig
from .finder import CanalEndpointFinder
from .mesh import ensure_polydata
from .mesh_io import MeshLoader, StlMeshLoader
from .splitter import LeftRightCanalSplitter


class MandibularMefEstimator:
    def __init__(
        self,
        config: CanalPipelineConfig,
        *,
        splitter: LeftRightCanalSplitter | None = None,
        finder: CanalEndpointFinder | None = None,
        mesh_loader: MeshLoader | None = None,
    ) -> None:
        self._config = config
        self._splitter = splitter or LeftRightCanalSplitter(config.input)
        self._finder = finder or CanalEndpointFinder(config)
        self._mesh_loader = mesh_loader or StlMeshLoader()

    def estimate_from_path(self, stl_path: Path) -> dict[str, np.ndarray]:
        mesh = self._mesh_loader.load(stl_path)
        return self.estimate(mesh)

    def estimate(self, mesh: pv.PolyData) -> dict[str, np.ndarray]:
        mesh = ensure_polydata(mesh)
        canals = self._splitter.split(mesh)
        right_ep = self._finder.find(canals.right).min_y_endpoint()
        left_ep = self._finder.find(canals.left).min_y_endpoint()
        l_mef, r_mef = MEF_LANDMARKS
        return {
            r_mef: right_ep.point.copy(),
            l_mef: left_ep.point.copy(),
        }
