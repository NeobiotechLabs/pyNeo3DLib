"""Slicer markups JSON 및 좌표 변환."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import SimpleITK as sitk


def spacing_to_scale_key(sp) -> str:
    """
    spacing(예: 1.0, 0.3)을 모델 폴더 키('1', '0-3')로 변환.
    """
    sp = float(sp)
    if sp.is_integer():
        return str(int(sp))
    return ("%g" % sp).replace(".", "-")


def agent_index_zyx_to_lps_xyz(
    image: sitk.Image, pos_zyx
) -> Tuple[float, float, float]:
    """
    에이전트 (z, y, x) 복셀 인덱스 → LPS 물리 좌표 (mm).
    """
    iz, iy, ix = float(pos_zyx[0]), float(pos_zyx[1]), float(pos_zyx[2])
    physical = image.TransformContinuousIndexToPhysicalPoint((ix, iy, iz))
    return (float(physical[0]), float(physical[1]), float(physical[2]))


def volume_stem(volume_path: Union[str, Path]) -> str:
    """'case.nii.gz' → 'case', 'my.scan.nii.gz' → 'my.scan'."""
    name = Path(volume_path).name
    for suffix in (".nii.gz", ".nii", ".nrrd", ".gipl"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(volume_path).stem


def volume_output_dir_name(volume_path: Union[str, Path]) -> str:
    """출력 하위 폴더명. 'case.nii.gz' → 'case.nii' (평가 output 레이아웃과 동일)."""
    name = Path(volume_path).name
    if name.lower().endswith(".nii.gz"):
        return name[: -len(".gz")]
    if name.lower().endswith(".nii"):
        return name
    return volume_stem(volume_path)


def gen_control_points(groupe_data: Dict[str, Dict[str, float]]) -> list:
    lm_lst = []
    point_id = 0
    for landmark, data in groupe_data.items():
        point_id += 1
        lm_lst.append(
            {
                "id": str(point_id),
                "label": landmark,
                "description": "",
                "associatedNodeID": "",
                "position": [data["x"], data["y"], data["z"]],
                "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "selected": True,
                "locked": True,
                "visibility": True,
                "positionStatus": "preview",
            }
        )
    return lm_lst


def write_mrk_json(control_points: list, out_path: Union[str, Path]) -> None:
    file = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "locked": False,
                "labelFormat": "%N-%d",
                "controlPoints": control_points,
                "measurements": [],
                "display": {
                    "visibility": False,
                    "opacity": 1.0,
                    "color": [0.5, 0.5, 0.5],
                    "selectedColor": [
                        0.26666666666666669,
                        0.6745098039215687,
                        0.39215686274509806,
                    ],
                    "propertiesLabelVisibility": False,
                    "pointLabelsVisibility": True,
                    "textScale": 2.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 2.0,
                    "glyphSize": 5.0,
                    "useGlyphScale": True,
                    "sliceProjection": False,
                    "sliceProjectionUseFiducialColor": True,
                    "sliceProjectionOutlinedBehindSlicePlane": False,
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": False,
                    "snapMode": "toVisibleSurface",
                },
            }
        ],
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(file, f, ensure_ascii=False, indent=4)


# 평가·병합 출력용 control point 순서 (GT merged.mrk.json 과 동일)
MERGED_OUTPUT_ORDER = ("ANS", "PNS", "Pog", "Me", "N", "LMeF", "RMeF")

MERGED_DISPLAY_COLOR = [0.0, 1.0, 0.0]
MERGED_DISPLAY_SELECTED_COLOR = [0.2, 1.0, 0.2]
MERGED_DISPLAY_ACTIVE_COLOR = [0.0, 0.85, 0.0]
MERGED_SLICE_PROJECTION_COLOR = [0.0, 1.0, 0.0]


def merged_label_order(landmark_names: list[str]) -> tuple[str, ...]:
    preferred = [lm for lm in MERGED_OUTPUT_ORDER if lm in landmark_names]
    rest = sorted(set(landmark_names) - set(preferred))
    return tuple(preferred + rest)


def apply_merged_display_colors(markup: dict) -> None:
    display = markup.setdefault("display", {})
    display["color"] = list(MERGED_DISPLAY_COLOR)
    display["selectedColor"] = list(MERGED_DISPLAY_SELECTED_COLOR)
    display["activeColor"] = list(MERGED_DISPLAY_ACTIVE_COLOR)
    display["sliceProjectionColor"] = list(MERGED_SLICE_PROJECTION_COLOR)


def write_merged_mrk_json(
    landmark_coords: Dict[str, Dict[str, float]],
    out_path: Union[str, Path],
    *,
    label_order: tuple[str, ...] | None = None,
) -> Path:
    """
    모든 랜드마크를 하나의 Slicer fiducial JSON으로 저장.
    landmark_coords: {label: {"x", "y", "z"}} (LPS mm)
    """
    import logging

    names = list(landmark_coords.keys())
    order = label_order or merged_label_order(names)
    ordered: Dict[str, Dict[str, float]] = {}
    for lm in order:
        if lm in landmark_coords:
            ordered[lm] = landmark_coords[lm]
    for lm in sorted(set(names) - set(ordered)):
        ordered[lm] = landmark_coords[lm]

    markup = {
        "type": "Fiducial",
        "coordinateSystem": "LPS",
        "locked": False,
        "labelFormat": "%N-%d",
        "controlPoints": gen_control_points(ordered),
        "measurements": [],
        "display": {},
    }
    apply_merged_display_colors(markup)
    markup["lastUsedControlPointNumber"] = len(markup["controlPoints"])

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": [markup],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=4)
        f.write("\n")
    logging.getLogger(__name__).info("Saved merged landmarks: %s", path)
    return path


def load_mrk_landmarks(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Slicer .mrk.json에서 label → LPS [x,y,z] (mm)."""
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    markup = data["markups"][0]
    out: Dict[str, np.ndarray] = {}
    for cp in markup["controlPoints"]:
        label = cp["label"]
        pos = np.asarray(cp["position"], dtype=np.float64)
        if label in out:
            raise ValueError(f"{path}: 중복 label '{label}'")
        out[label] = pos
    return out


_LEGACY_ALIASES = {
    "GenControlePoint": "gen_control_points",
    "WriteJson": "write_mrk_json",
}


def __getattr__(name: str):
    if name in _LEGACY_ALIASES:
        from .compat import deprecate

        deprecate(name, _LEGACY_ALIASES[name])
        return globals()[_LEGACY_ALIASES[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
