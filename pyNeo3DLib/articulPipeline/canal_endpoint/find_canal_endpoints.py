"""하악 신경관 STL에서 LMeF/RMeF 랜드마크를 추정해 저장하는 CLI.

사용 예:
    python find_canal_endpoints.py --input Mandibular_canal.stl --output mef.mrk.json
    python find_canal_endpoints.py -i Mandibular_canal.stl -o ./results/
    python find_canal_endpoints.py -i ./case_folder -o mef.mrk.json   (폴더: 신경관 STL 자동 탐색)
    python -m canal_endpoint.canal_core -i canal.stl -o mef.mrk.json   (articulPipeline 폴더에서)

동작:
1. --input 신경관 STL 로드 (폴더 지정 시 신경관 이름 패턴 자동 탐색)
2. 좌·우 신경관 분리 → 각쪽 끝점 쌍 추정 후 MeF(정신공) 끝점 선택 (y 최소)
3. --output 경로에 LMeF/RMeF 를 3D Slicer Markups(fiducial) JSON 형식으로 저장

랜드마크 이름은 ``articulPipeline/shared/constants.py`` 의 ``MEF_LANDMARKS``
공통 정의를 사용합니다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: 이 파일을 직접 실행(``python find_canal_endpoints.py``)할 때도 canal_endpoint·shared 패키지를
#: import 할 수 있도록 articulPipeline 폴더를 sys.path 에 추가합니다.
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from canal_endpoint.canal_core.cli_args import add_basic_canal_args
from canal_endpoint.canal_core.factory import CanalPipelineConfigFactory
from canal_endpoint.canal_core.mef_estimator import MandibularMefEstimator
from shared.constants import MEF_LANDMARKS

SLICER_MARKUPS_SCHEMA = (
    "https://raw.githubusercontent.com/slicer/slicer/master/"
    "Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#"
)
_IDENTITY_ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

#: 폴더 입력 시 신경관 STL 자동 탐색 패턴
#: (``shared/discovery.py`` 의 MANDIBULAR_CANAL_PATTERNS 과 동일 규약)
CANAL_STL_PATTERNS = (
    "Mandibular_canal.stl",
    "*[Mm]andibular*[Cc]anal*.stl",
    "*canal*.stl",
)


def resolve_input_path(input_path: Path) -> Path:
    """입력 경로를 신경관 STL 파일로 해석.

    폴더를 지정하면 ``CANAL_STL_PATTERNS`` 순서로 신경관 STL을 찾아 반환합니다.
    """
    input_path = input_path.expanduser()
    if input_path.is_dir():
        for pattern in CANAL_STL_PATTERNS:
            candidates = sorted(p for p in input_path.glob(pattern) if p.is_file())
            if candidates:
                return candidates[0]
        raise FileNotFoundError(
            f"폴더에 신경관 STL({', '.join(CANAL_STL_PATTERNS)})이 없습니다: {input_path}"
        )
    if not input_path.is_file():
        raise FileNotFoundError(f"신경관 STL 파일 없음: {input_path}")
    return input_path


def resolve_output_path(output: str, input_path: Path) -> Path:
    """디렉터리(''/'\\'로 끝나거나 이미 존재하는 폴더)면 그 안에
    <입력파일명>_mef.mrk.json 으로 저장."""
    if output.endswith(("/", "\\")) or Path(output).is_dir():
        return Path(output) / f"{input_path.stem}_mef.mrk.json"
    return Path(output)


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


def save_mef_landmarks(landmarks: dict[str, np.ndarray], output: Path) -> Path:
    """LMeF/RMeF 를 3D Slicer Markups(fiducial) JSON 형식으로 저장."""
    l_mef, r_mef = MEF_LANDMARKS
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
                    _control_point("1", l_mef, landmarks[l_mef]),
                    _control_point("2", r_mef, landmarks[r_mef]),
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


def print_summary(input_path: Path, landmarks: dict[str, np.ndarray]) -> None:
    l_mef, r_mef = MEF_LANDMARKS
    distance = float(np.linalg.norm(landmarks[l_mef] - landmarks[r_mef]))
    print(f"입력: {input_path}")
    print(f"{l_mef}: {np.round(landmarks[l_mef], 3)}")
    print(f"{r_mef}: {np.round(landmarks[r_mef], 3)}")
    print(f"LMeF-RMeF 거리: {distance:.3f} mm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="하악 신경관 STL에서 LMeF/RMeF 랜드마크를 추정하고 저장합니다."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help=(
            "신경관 STL 경로 (PyVista가 읽을 수 있는 형식). "
            "폴더를 지정하면 Mandibular_canal.stl 등 신경관 이름 패턴을 자동 탐색"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help=(
            "랜드마크 저장 경로 (3D Slicer Markups JSON 형식, 예: mef.mrk.json). "
            "폴더('/'로 끝나거나 존재하는 폴더)면 그 안에 <입력명>_mef.mrk.json 으로 저장"
        ),
    )
    add_basic_canal_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        input_path = resolve_input_path(args.input)
        config = CanalPipelineConfigFactory.build(
            input_path,
            curvature_percentile=args.curvature_percentile,
            cluster_distance_ratio=args.cluster_distance_ratio,
            expected_components=args.expected_components,
        )
        estimator = MandibularMefEstimator(config)
        landmarks = estimator.estimate_from_path(input_path)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"오류: {e}") from e
    print_summary(input_path, landmarks)

    output = resolve_output_path(args.output, input_path)
    saved = save_mef_landmarks(landmarks, output)
    print(f"랜드마크 저장: {saved}")


if __name__ == "__main__":
    main()
