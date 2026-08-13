"""교합평면 계산 결과 콘솔 출력."""

from __future__ import annotations

import json

from core.occlusal_plane.result import (
    VECTOR_LABELS_KO,
    VECTOR_NAMES,
    OcclusalPlaneResult,
)


def format_occlusal_result_lines(result: OcclusalPlaneResult) -> list[str]:
    lines: list[str] = []
    for name in VECTOR_NAMES:
        xyz = getattr(result, name)
        label = VECTOR_LABELS_KO[name]
        coords = ", ".join(f"{v:.4f}" for v in xyz)
        lines.append(f"{name:16}  [{coords}]  # {label}")
    return lines


def print_occlusal_result(result: OcclusalPlaneResult, *, json_output: bool = False) -> None:
    """이름·설명과 함께 9개 벡터를 stdout에 출력."""
    if json_output:
        print(json.dumps(result.as_named_list(), ensure_ascii=False, indent=2))
        return

    print("# 좌표계: Slicer LPS (mm). 법선은 단위벡터.")
    for line in format_occlusal_result_lines(result):
        print(line)
