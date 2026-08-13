"""PyVista 교합면 시각화."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv

from core.occlusal_plane.visualization.style import OcclusalSceneStyle
from core.shared.constants import CRANIAL_LANDMARKS
from core.occlusal_plane.models import OcclusalPlaneGeometry, OcclusalScene


@dataclass(frozen=True)
class _OptionalOcclusalPlaneDraw:
    show_attr: str
    plane_attr: str
    color_attr: str
    opacity_attr: str


_OPTIONAL_OCCLUSAL_PLANES = (
    _OptionalOcclusalPlaneDraw(
        "show_ans_occlusal_plane",
        "ans_occlusal_plane",
        "ans_occlusal_plane_color",
        "ans_occlusal_plane_opacity",
    ),
    _OptionalOcclusalPlaneDraw(
        "show_mef_occlusal_plane",
        "mef_occlusal_plane",
        "mef_occlusal_plane_color",
        "mef_occlusal_plane_opacity",
    ),
)

_REFERENCE_SPHERE_SPECS: tuple[tuple[str, str], ...] = (
    ("P_occ_ans", "ans_occlusal_plane_reference"),
    ("P_occ_mef", "mef_occlusal_plane_reference"),
    ("P_occ_mid", "mid_occlusal_plane_reference"),
)


class OcclusalPlaneVisualizer:
    def __init__(self, style: OcclusalSceneStyle | None = None) -> None:
        self._style = style or OcclusalSceneStyle()

    def show(self, scene: OcclusalScene) -> None:
        plotter = pv.Plotter(window_size=self._style.window_size)
        plotter.set_background(self._style.background)
        plotter.add_axes()

        self._add_meshes(plotter, scene)
        self._add_landmarks(plotter, scene)
        self._add_planes(plotter, scene)

        plotter.reset_camera()
        plotter.show()

    def _add_meshes(self, plotter: pv.Plotter, scene: OcclusalScene) -> None:
        mesh_map = {
            "upper_skull": scene.meshes.upper_skull,
            "mandible": scene.meshes.mandible,
            "mandibular_canal": scene.meshes.mandibular_canal,
        }
        for key, mesh in mesh_map.items():
            if mesh is None:
                continue
            spec = self._style.mesh_styles[key]
            plotter.add_mesh(
                mesh,
                color=spec["color"],
                opacity=spec["opacity"],
                smooth_shading=True,
            )
        for index, tooth in enumerate(scene.meshes.teeth):
            plotter.add_mesh(
                tooth,
                color=self._style.teeth_color,
                opacity=self._style.teeth_opacity,
                smooth_shading=True,
                name=f"tooth_{index}",
            )

    def _add_landmarks(self, plotter: pv.Plotter, scene: OcclusalScene) -> None:
        for name in CRANIAL_LANDMARKS:
            if name not in scene.cranial_landmarks:
                continue
            pos = scene.cranial_landmarks[name]
            self._add_landmark_sphere(plotter, name, pos)
            self._add_text_label(plotter, name, pos, offset=self._cranial_label_offset(name))

        for name, pos in scene.mef_landmarks.items():
            self._add_landmark_sphere(plotter, name, pos)
            self._add_text_label(
                plotter,
                name,
                pos,
                offset=self._mef_label_offset(name),
                color_key=name,
                label_name=name,
            )

        if self._style.show_occlusal_reference_spheres:
            for sphere_name, scene_attr in _REFERENCE_SPHERE_SPECS:
                pt = getattr(scene, scene_attr)
                if pt is not None:
                    self._add_landmark_sphere(plotter, sphere_name, pt)

    def _add_landmark_sphere(
        self,
        plotter: pv.Plotter,
        name: str,
        pos: np.ndarray,
    ) -> None:
        color = self._style.landmark_colors.get(name, "white")
        sphere = pv.Sphere(
            radius=self._style.sphere_radius_mm,
            center=pos,
            theta_resolution=self._style.sphere_resolution,
            phi_resolution=self._style.sphere_resolution,
        )
        plotter.add_mesh(
            sphere,
            color=color,
            opacity=1.0,
            smooth_shading=True,
            name=f"landmark_{name}",
        )

    def _cranial_label_offset(self, name: str) -> np.ndarray:
        offset = np.array(
            [0.0, 0.0, self._style.cranial_label_offset_superior_mm],
            dtype=np.float64,
        )
        sagittal = self._style.cranial_label_offset_sagittal_mm
        if name == "PNS":
            offset[1] = sagittal
        elif name == "ANS":
            offset[1] = -sagittal
        return offset

    def _mef_label_offset(self, landmark_key: str) -> np.ndarray:
        offset = np.array([0.0, 0.0, self._style.mef_label_offset_superior_mm], dtype=np.float64)
        lateral = self._style.mef_label_offset_lateral_mm
        if landmark_key == "LMeF":
            offset[0] = lateral
        elif landmark_key == "RMeF":
            offset[0] = -lateral
        return offset

    def _add_text_label(
        self,
        plotter: pv.Plotter,
        text: str,
        pos: np.ndarray,
        *,
        offset: np.ndarray,
        color_key: str | None = None,
        label_name: str | None = None,
    ) -> None:
        key = color_key or text
        name = label_name or text
        label_pos = np.asarray(pos, dtype=np.float64) + offset
        color = self._style.landmark_colors.get(key, self._style.landmark_colors.get(text, "white"))
        plotter.add_point_labels(
            [label_pos],
            [text],
            font_size=self._style.landmark_label_font_size,
            point_size=0,
            show_points=False,
            bold=True,
            text_color=color,
            shape_opacity=0.65,
            margin=4,
            name=f"label_{name}",
        )

    def _add_planes(self, plotter: pv.Plotter, scene: OcclusalScene) -> None:
        self._add_single_plane(
            plotter,
            scene.msp_plane,
            color=self._style.msp_plane_color,
            opacity=self._style.msp_plane_opacity,
        )
        for spec in _OPTIONAL_OCCLUSAL_PLANES:
            if not getattr(self._style, spec.show_attr):
                continue
            plane = getattr(scene, spec.plane_attr)
            if plane is None:
                continue
            self._add_single_plane(
                plotter,
                plane,
                color=getattr(self._style, spec.color_attr),
                opacity=getattr(self._style, spec.opacity_attr),
            )
        if scene.mid_occlusal_plane is not None:
            self._add_single_plane(
                plotter,
                scene.mid_occlusal_plane,
                color=self._style.mid_occlusal_plane_color,
                opacity=self._style.mid_occlusal_plane_opacity,
            )

    def _add_single_plane(
        self,
        plotter: pv.Plotter,
        plane: OcclusalPlaneGeometry,
        *,
        color: str,
        opacity: float,
    ) -> None:
        plane_mesh = pv.Plane(
            center=plane.center,
            direction=plane.normal,
            i_size=plane.size_mm,
            j_size=plane.size_mm,
        )
        plotter.add_mesh(
            plane_mesh,
            color=color,
            opacity=opacity,
            smooth_shading=False,
        )
