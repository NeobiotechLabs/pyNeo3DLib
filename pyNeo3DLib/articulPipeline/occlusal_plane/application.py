"""계산 + 선택 시각화 오케스트레이션."""

from __future__ import annotations

from core.occlusal_plane.compute import OcclusalPlaneInputs, compute_occlusal_plane_data
from core.occlusal_plane.console_summary import print_occlusal_result
from core.occlusal_plane.result import OcclusalPlaneResult, occlusal_result_from_data
from core.occlusal_plane.visualization import (
    OcclusalPlaneVisualizer,
    build_occlusal_scene,
    show_occlusal_scene,
)
from core.occlusal_plane.visualization.style import OcclusalSceneStyle


def run_occlusal_plane(
    inputs: OcclusalPlaneInputs,
    *,
    show: bool = False,
    print_result: bool = True,
    json_output: bool = False,
    style: OcclusalSceneStyle | None = None,
) -> OcclusalPlaneResult:
    scene_style = style or OcclusalSceneStyle()
    data = compute_occlusal_plane_data(inputs)
    result = occlusal_result_from_data(data)
    if print_result:
        print_occlusal_result(result, json_output=json_output)
    if show:
        scene = build_occlusal_scene(data, inputs, style=scene_style)
        show_occlusal_scene(scene, style=scene_style, visualizer=OcclusalPlaneVisualizer(scene_style))
    return result
