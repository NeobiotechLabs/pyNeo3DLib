"""3개 랜드마크 마크업(랜드마크 / 콘다일 / MeF)을 하나의 mrk.json으로 통합하는 CLI.

사용 예:
    python merge_landmarks.py \
        --input landmarks.mrk.json condyles.mrk.json mef.mrk.json \
        --output merged.mrk.json

    python merge_landmarks.py \
        --input ./case01_merged.mrk.json ./case01_mandible_condyles.mrk.json ./case01_nerve_canal_mef.mrk.json \
        -o ./results/

동작:
1. --input 에 지정된 3개의 3D Slicer Markups JSON 을 순서대로 로드
   (순서: 랜드마크 → 콘다일 → MeF)
2. 각 파일의 controlPoints 를 하나로 합치고 id 를 1부터 다시 부여
   (같은 label 이 여러 파일에 있으면 첫 번째 것을 유지하고 경고 출력)
3. --output 경로에 입력과 동일한 형식(schema/display)의 통합 mrk.json 저장
   (--output 을 폴더로 지정하면 ``articulPipeline/mrk_output_names.json``
   공통 규약 이름 ``{케이스}_landmarks.mrk.json`` 으로 저장. 케이스 이름은
   첫 번째 입력(랜드마크) 파일명에서 ``_merged`` 접미어를 떼어 만듦)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mrk_output_names import mrk_filename

SLICER_MARKUPS_SCHEMA = (
    "https://raw.githubusercontent.com/slicer/slicer/master/"
    "Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#"
)
_IDENTITY_ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

_SOURCE_LABELS = ("랜드마크", "콘다일", "MeF")


def load_control_points(path: Path) -> list[dict]:
    """Slicer .mrk.json 에서 controlPoints 를 로드."""
    if not path.is_file():
        raise FileNotFoundError(f"마크업 파일 없음: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    markups = data.get("markups", [])
    if not markups:
        raise ValueError(f"markups 항목이 없습니다: {path}")

    points: list[dict] = []
    for markup in markups:
        points.extend(markup.get("controlPoints", []))
    if not points:
        raise ValueError(f"controlPoints 가 없습니다: {path}")
    return points


def normalize_control_point(cp: dict) -> dict:
    """입력 스키마와 동일한 키 구성으로 컨트롤 포인트를 정규화."""
    position = cp["position"]
    return {
        "id": "",  # 병합 후 일련번호로 다시 부여
        "label": cp.get("label", ""),
        "description": cp.get("description", ""),
        "associatedNodeID": cp.get("associatedNodeID", ""),
        "position": [float(position[0]), float(position[1]), float(position[2])],
        "orientation": [float(v) for v in cp.get("orientation", _IDENTITY_ORIENTATION)],
        "selected": bool(cp.get("selected", True)),
        "locked": bool(cp.get("locked", True)),
        "visibility": bool(cp.get("visibility", True)),
        "positionStatus": cp.get("positionStatus", "preview"),
    }


def merge_control_points(sources: list[list[dict]]) -> list[dict]:
    """여러 파일의 컨트롤 포인트를 순서대로 병합.

    - label 중복 시 첫 번째 등장만 유지(경고 출력)
    - id 는 1부터 다시 부여
    """
    merged: list[dict] = []
    seen_labels: set[str] = set()
    for cp in (raw for points in sources for raw in points):
        norm = normalize_control_point(cp)
        label = norm["label"]
        if label in seen_labels:
            print(f"[경고] 중복 label '{label}' 은(는) 첫 번째 항목만 유지합니다.")
            continue
        seen_labels.add(label)
        norm["id"] = str(len(merged) + 1)
        merged.append(norm)
    return merged


def build_payload(control_points: list[dict]) -> dict:
    """입력 mrk.json 과 동일한 형식의 통합 마크업 payload 생성."""
    return {
        "@schema": SLICER_MARKUPS_SCHEMA,
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "locked": False,
                "labelFormat": "%N-%d",
                "controlPoints": control_points,
                "measurements": [],
                "display": {
                    "color": [0.0, 1.0, 0.0],
                    "selectedColor": [0.2, 1.0, 0.2],
                    "activeColor": [0.0, 0.85, 0.0],
                    "sliceProjectionColor": [0.0, 1.0, 0.0],
                },
                "lastUsedControlPointNumber": len(control_points),
            }
        ],
    }


def save_merged(control_points: list[dict], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(control_points)
    output.write_text(
        json.dumps(payload, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def resolve_output_path(output: Path, landmark_input: Path) -> Path:
    """--output 이 폴더면 그 안에 {케이스}_landmarks.mrk.json 으로 저장.

    케이스 이름은 첫 번째 입력(랜드마크) 파일명에서 ``.mrk.json`` 확장자와
    ``_merged`` 접미어(``mrk_output_names.json`` 공통 규약)를 떼어 만들고,
    접미어가 없으면 파일명 그대로 사용.
    """
    if output.is_dir() or str(output).endswith(("/", "\\")):
        stem = landmark_input.name
        if stem.endswith(".mrk.json"):
            stem = stem[: -len(".mrk.json")]
        return output / mrk_filename(stem, "landmark_merge")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "랜드마크/콘다일/MeF 마크업 mrk.json 3개를 하나의 mrk.json 으로 통합합니다."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        nargs=3,
        required=True,
        metavar=("LANDMARK_MRK", "CONDYLE_MRK", "MEF_MRK"),
        help=(
            "입력 mrk.json 3개 경로 (순서: 랜드마크, 콘다일, MeF). "
            "예: --input case01_merged.mrk.json case01_mandible_condyles.mrk.json "
            "case01_nerve_canal_mef.mrk.json"
        ),
    )
    parser.add_argument(
        "-o",
        "-output",
        "--output",
        type=Path,
        required=True,
        dest="output",
        help=(
            "통합 mrk.json 저장 경로 (예: case01_landmarks.mrk.json). "
            "폴더로 지정하면 그 안에 {케이스}_landmarks.mrk.json 으로 저장"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    sources: list[list[dict]] = []
    for label, path in zip(_SOURCE_LABELS, args.input):
        points = load_control_points(path)
        print(f"{label} 입력: {path} ({len(points)}점)")
        sources.append(points)

    merged = merge_control_points(sources)
    output = resolve_output_path(args.output, args.input[0])
    saved = save_merged(merged, output)

    labels = ", ".join(cp["label"] for cp in merged)
    print(f"통합 랜드마크 {len(merged)}점: {labels}")
    print(f"저장: {saved}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)
