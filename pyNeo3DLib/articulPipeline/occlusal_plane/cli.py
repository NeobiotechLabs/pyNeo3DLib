"""교합면 계산·시각화 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.canal_endpoint.cli_args import add_basic_canal_args
from core.occlusal_plane.application import run_occlusal_plane
from core.occlusal_plane.compute import OcclusalPlaneInputs
from core.occlusal_plane.case_discovery import resolve_case_folder
from core.occlusal_plane.visualization.style import OcclusalSceneStyle
from core.occlusal_plane.result import OcclusalPlaneResult
from core.shared.validation import LandmarkValidationError


def _resolve_inputs(args: argparse.Namespace) -> OcclusalPlaneInputs:
    input_path = args.input_path.expanduser().resolve()

    if input_path.is_dir():
        try:
            resolved = resolve_case_folder(input_path)
        except (FileNotFoundError, NotADirectoryError, LandmarkValidationError) as e:
            raise SystemExit(f"오류: {e}")
        print(f"케이스 폴더: {resolved.case_dir}")
        print(f"  landmarks: {resolved.landmarks_path.name}")
        print(f"  canal:     {resolved.mandibular_canal_path.name}")
        if resolved.upper_skull_path is not None:
            print(f"  upper:     {resolved.upper_skull_path.name}")
        if resolved.mandible_path is not None:
            print(f"  mandible:  {resolved.mandible_path.name}")
        teeth_paths = () if args.no_teeth else resolved.teeth_paths
        if teeth_paths:
            print(f"  teeth:     {len(teeth_paths)}개")
        return OcclusalPlaneInputs(
            landmarks_path=resolved.landmarks_path,
            mandibular_canal_path=resolved.mandibular_canal_path,
            upper_skull_path=args.upper_skull or resolved.upper_skull_path,
            mandible_path=args.mandible or resolved.mandible_path,
            teeth_paths=teeth_paths,
            curvature_percentile=args.curvature_percentile,
            cluster_distance_ratio=args.cluster_distance_ratio,
            expected_components=args.expected_components,
        )

    if args.mandibular_canal is None:
        raise SystemExit(
            "오류: .mrk.json 파일을 직접 지정할 때는 --canal 이 필요합니다.\n"
            "케이스 폴더만 넘기려면 폴더 경로를 positional 인자로 주세요."
        )

    return OcclusalPlaneInputs(
        landmarks_path=input_path,
        upper_skull_path=args.upper_skull,
        mandibular_canal_path=args.mandibular_canal,
        mandible_path=args.mandible,
        curvature_percentile=args.curvature_percentile,
        cluster_distance_ratio=args.cluster_distance_ratio,
        expected_components=args.expected_components,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="두개골·하악·신경관 STL과 N/ANS/PNS 랜드마크, 교합 평면을 계산합니다.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="케이스 폴더(자동 탐색) 또는 N/ANS/PNS .mrk.json",
    )
    parser.add_argument("--upper-skull", type=Path, default=None, metavar="STL")
    parser.add_argument(
        "--mandibular-canal",
        "--canal",
        type=Path,
        default=None,
        dest="mandibular_canal",
        metavar="STL",
        help="하악 신경관 STL (폴더 지정 시 자동 탐색, .mrk.json 직접 지정 시 필수)",
    )
    parser.add_argument("--mandible", type=Path, default=None, metavar="STL")
    add_basic_canal_args(parser)
    parser.add_argument(
        "--show",
        action="store_true",
        help="PyVista 디버그 창 표시 (기본: 벡터만 stdout 출력)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=(1200, 900),
        metavar=("W", "H"),
        help="--show 일 때 창 크기",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="이름·라벨·xyz JSON 배열로 출력",
    )
    parser.add_argument(
        "--no-teeth",
        action="store_true",
        help="--show 일 때 tooth_*.stl 치아 메쉬 생략",
    )
    parser.add_argument(
        "--mesh-opacity",
        type=float,
        default=None,
        metavar="0-1",
        help="--show 일 때 골·신경관 메쉬 불투명도",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> OcclusalPlaneResult:
    try:
        args = parse_args(argv)
        inputs = _resolve_inputs(args)
        style = (
            OcclusalSceneStyle(
                window_size=tuple(args.window_size),
                mesh_opacity=args.mesh_opacity,
            )
            if args.show
            else None
        )
        return run_occlusal_plane(
            inputs,
            show=args.show,
            json_output=args.json,
            style=style,
        )
    except LandmarkValidationError as e:
        raise SystemExit(f"오류: {e}") from e
