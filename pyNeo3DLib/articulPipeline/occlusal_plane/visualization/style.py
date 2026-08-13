"""PyVista 교합면 시각화 스타일."""

from __future__ import annotations

from dataclasses import dataclass, field

_DEFAULT_LANDMARK_COLORS = {
    "N": "#22c55e",
    "ANS": "#a855f7",
    "PNS": "#06b6d4",
    "LMeF": "#eab308",
    "RMeF": "#f97316",
    "MeF_mid": "#facc15",
    "P_occ_mef": "#fb923c",
    "P_occ_ans": "#ec4899",
    "P_occ_mid": "#14b8a6",
}


@dataclass(frozen=True)
class OcclusalSceneStyle:
    window_size: tuple[int, int] = (1200, 900)
    background: str = "white"
    sphere_radius_mm: float = 1.5
    sphere_resolution: int = 24
    msp_plane_opacity: float = 0.30
    msp_plane_color: str = "#38bdf8"
    mef_occlusal_plane_opacity: float = 0.40
    mef_occlusal_plane_color: str = "#f59e0b"
    ans_occlusal_plane_opacity: float = 0.35
    ans_occlusal_plane_color: str = "#a78bfa"
    mid_occlusal_plane_opacity: float = 0.45
    mid_occlusal_plane_color: str = "#14b8a6"
    show_mef_occlusal_plane: bool = False
    show_ans_occlusal_plane: bool = False
    show_occlusal_reference_spheres: bool = False
    plane_scale: float = 1.15
    extent_min_mm: float = 80.0
    landmark_colors: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_LANDMARK_COLORS))
    landmark_label_font_size: int = 14
    cranial_label_offset_superior_mm: float = 6.0
    cranial_label_offset_sagittal_mm: float = 5.0
    mef_label_offset_superior_mm: float = 5.0
    mef_label_offset_lateral_mm: float = 4.0
    mesh_opacity: float | None = None
    teeth_color: str = "white"
    teeth_opacity: float = 1.0

    @property
    def mesh_styles(self) -> dict[str, dict]:
        styles = {
            "upper_skull": {"color": "#e8e4dc", "opacity": 0.55, "label": "upper_skull"},
            "mandible": {"color": "#c4b5a0", "opacity": 0.7, "label": "mandible"},
            "mandibular_canal": {
                "color": "#64748b",
                "opacity": 0.9,
                "label": "mandibular_canal",
            },
        }
        if self.mesh_opacity is not None:
            return {key: {**spec, "opacity": self.mesh_opacity} for key, spec in styles.items()}
        return styles
