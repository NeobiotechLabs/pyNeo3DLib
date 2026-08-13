"""하악골(mandible) 메쉬에서 LCo/RCo(과두점)를 추정하는 CLI.

사용 예:
    python find_mandible_condylle.py --input case01_mandible.stl --output condyles.mrk.json
    python find_mandible_condylle.py -i ./results -o condyles.mrk.json -v

하악골 메쉬 이름은 ``articulPipeline/structure_names.json`` 의 공통 규약을
따릅니다 (``{케이스이름}_mandible.stl``). ``--input`` 을 폴더로 지정하면
폴더 안에서 규약 이름의 하악골 메쉬를 자동으로 찾아 로드합니다.

동작:
1. --input 하악골 메쉬 로드 (폴더 지정 시 ``*_mandible.stl`` 탐색)
2. +x 최대 정점 -> LCo, -x 최소 정점 -> RCo
3. --output 경로에 과두점을 3D Slicer Markups(fiducial) JSON 형식으로 저장
4. --visualize 지정 시 하악골 + LCo/RCo 마커만 PyVista 창으로 표시
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

from structure_names import MANDIBLE, mesh_stem_suffix

MANDIBLE_COLOR = "#c8d6e5"
CONDYLE_COLOR = "lime"
PLOT_BACKGROUND = "#1a1a2e"

SLICER_MARKUPS_SCHEMA = (
    "https://raw.githubusercontent.com/slicer/slicer/master/"
    "Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#"
)
_IDENTITY_ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Condyles:
    left: np.ndarray
    right: np.ndarray
    left_index: int
    right_index: int

    @classmethod
    def from_mesh(cls, mesh: pv.PolyData) -> Condyles:
        points = np.asarray(mesh.points)
        left_idx = int(np.argmax(points[:, 0]))
        right_idx = int(np.argmin(points[:, 0]))
        return cls(
            left=points[left_idx],
            right=points[right_idx],
            left_index=left_idx,
            right_index=right_idx,
        )

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.left - self.right))


def resolve_input_path(input_path: Path) -> Path:
    """입력 경로를 하악골 메쉬 파일로 해석.

    폴더를 지정하면 공통 이름 규약(``structure_names.json``)에 맞는
    ``*_{mandible}.stl`` 파일을 폴더 안에서 찾아 반환합니다.
    """
    if input_path.is_dir():
        pattern = f"*{mesh_stem_suffix(MANDIBLE)}.stl"
        candidates = sorted(input_path.glob(pattern))
        if not candidates:
            raise FileNotFoundError(
                f"폴더에 하악골 메쉬({pattern})가 없습니다: {input_path}"
            )
        return candidates[0]
    return input_path


def load_mandible(path: Path) -> pv.PolyData:
    if not path.is_file():
        raise FileNotFoundError(f"하악골 메쉬 파일 없음: {path}")
    return pv.read(path)


def mesh_diagonal(mesh: pv.PolyData) -> float:
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    return float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))


def resolve_output_path(output: Path, input_path: Path) -> Path:
    """디렉터리로 지정되면 그 안에 <입력파일명>_condyles.mrk.json 으로 저장."""
    if output.is_dir() or str(output).endswith(("/", "\\")):
        return output / f"{input_path.stem}_condyles.mrk.json"
    return output


def _control_point(point_id: str, label: str, position: np.ndarray) -> dict:
    return {
        "id": point_id,
        "label": label,
        "description": "",
        "associatedNodeID": "",
        "position": [float(position[0]), float(position[1]), float(position[2])],
        "orientation": list(_IDENTITY_ORIENTATION),
        "selected": True,
        "locked": True,
        "visibility": True,
        "positionStatus": "preview",
    }


def save_condyles(condyles: Condyles, output: Path) -> Path:
    """LCo/RCo 를 3D Slicer Markups(fiducial) JSON 형식으로 저장."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "@schema": SLICER_MARKUPS_SCHEMA,
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "locked": False,
                "labelFormat": "%N-%d",
                "controlPoints": [
                    _control_point("1", "LCo", condyles.left),
                    _control_point("2", "RCo", condyles.right),
                ],
                "measurements": [],
                "display": {
                    "color": [0.0, 1.0, 0.0],
                    "selectedColor": [0.2, 1.0, 0.2],
                    "activeColor": [0.0, 0.85, 0.0],
                    "sliceProjectionColor": [0.0, 1.0, 0.0],
                },
                "lastUsedControlPointNumber": 2,
            }
        ],
    }
    output.write_text(
        json.dumps(payload, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def visualize(mandible: pv.PolyData, condyles: Condyles) -> None:
    marker_radius = max(mesh_diagonal(mandible) * 0.02, 1.0)

    plotter = pv.Plotter(title="Mandible Condyles — LCo / RCo")
    plotter.set_background(PLOT_BACKGROUND)
    plotter.add_mesh(
        mandible,
        color=MANDIBLE_COLOR,
        opacity=1.0,
        smooth_shading=True,
        name=MANDIBLE,
    )
    for label, position in (
        ("LCo (+x max)", condyles.left),
        ("RCo (-x min)", condyles.right),
    ):
        plotter.add_mesh(
            pv.Sphere(radius=marker_radius, center=position),
            color=CONDYLE_COLOR,
        )
        plotter.add_point_labels(
            [position],
            [label],
            font_size=14,
            point_color=CONDYLE_COLOR,
            text_color="white",
            shape_opacity=0.6,
            always_visible=True,
        )
    plotter.add_axes()
    plotter.show()


def print_summary(path: Path, mandible: pv.PolyData, condyles: Condyles) -> None:
    print(f"입력: {path}")
    print(f"정점 수: {mandible.n_points:,}")
    print(f"LCo (idx={condyles.left_index}): {condyles.left}")
    print(f"RCo (idx={condyles.right_index}): {condyles.right}")
    print(f"LCo-RCo 거리: {condyles.distance:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="하악골 메쉬에서 LCo/RCo 과두점을 추정하고 저장합니다."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help=(
            "하악골 메쉬 경로 (STL 등 PyVista가 읽을 수 있는 형식). "
            "폴더를 지정하면 공통 이름 규약(structure_names.json)의 "
            f"*{mesh_stem_suffix(MANDIBLE)}.stl 파일을 자동 탐색"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="과두점 저장 경로 (3D Slicer Markups JSON 형식, 예: condyles.mrk.json)",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="지정 시 하악골 + LCo/RCo 마커를 PyVista 창으로 표시",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = resolve_input_path(args.input)
    mandible = load_mandible(input_path)
    condyles = Condyles.from_mesh(mandible)
    print_summary(input_path, mandible, condyles)

    output = resolve_output_path(args.output, input_path)
    saved = save_condyles(condyles, output=output)
    print(f"과두점 저장: {saved}")

    if args.visualize:
        visualize(mandible, condyles)


if __name__ == "__main__":
    main()
