"""좌·우 신경관 분리."""

from __future__ import annotations

import pyvista as pv

from .config import InputConfig
from .mesh import ensure_polydata
from .models import SplitCanals


def _bbox_center_x(mesh: pv.PolyData) -> float:
    xmin, xmax, _, _, _, _ = mesh.bounds
    return 0.5 * (float(xmin) + float(xmax))


def _select_mef_canal_component(components: list[pv.PolyData]) -> pv.PolyData:
    """한쪽에 여러 조각이 있을 때 y 중심이 가장 작은 조각을 MeF 추정용으로 선택."""
    if len(components) == 1:
        return components[0]
    return min(components, key=lambda m: float(m.center[1]))


class LeftRightCanalSplitter:
    def __init__(self, config: InputConfig) -> None:
        self._config = config

    def split(self, mesh: pv.PolyData) -> SplitCanals:
        mesh = ensure_polydata(mesh)
        bodies = mesh.split_bodies()
        n_blocks = bodies.n_blocks

        if n_blocks < 2:
            raise ValueError(
                f"연결 컴포넌트가 2개 미만입니다 (found {n_blocks}). "
                f"STL: {self._config.resolve_canal_stl()}"
            )

        components = [ensure_polydata(bodies[i]) for i in range(n_blocks)]
        return self._split_by_bbox_x(components, mesh)

    def _split_by_bbox_x(
        self,
        components: list[pv.PolyData],
        full_mesh: pv.PolyData,
    ) -> SplitCanals:
        center_x = _bbox_center_x(full_mesh)

        right_parts: list[pv.PolyData] = []
        left_parts: list[pv.PolyData] = []
        for comp in components:
            if float(comp.center[0]) < center_x:
                right_parts.append(comp)
            else:
                left_parts.append(comp)

        if not right_parts or not left_parts:
            components.sort(key=lambda m: float(m.center[0]))
            mid = max(1, len(components) // 2)
            right_parts = components[:mid]
            left_parts = components[mid:]

        return SplitCanals(
            right=_select_mef_canal_component(right_parts),
            left=_select_mef_canal_component(left_parts),
        )
