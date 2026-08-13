"""교합평면 PyVista 시각화 (계산 모듈과 분리)."""

from __future__ import annotations

from core.occlusal_plane.visualization.style import OcclusalSceneStyle
from core.occlusal_plane.models import OcclusalScene
from core.occlusal_plane.visualization.scene import build_occlusal_scene
from core.occlusal_plane.visualization.visualizer import OcclusalPlaneVisualizer


def show_occlusal_scene(
    scene: OcclusalScene,
    *,
    style: OcclusalSceneStyle | None = None,
    visualizer: OcclusalPlaneVisualizer | None = None,
) -> None:
    viz = visualizer or OcclusalPlaneVisualizer(style)
    viz.show(scene)


__all__ = [
    "OcclusalPlaneVisualizer",
    "build_occlusal_scene",
    "show_occlusal_scene",
]
