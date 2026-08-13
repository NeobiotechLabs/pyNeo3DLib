"""CBCT 랜드마크 추론 CLI."""

from __future__ import annotations

import argparse
import os

from .landmarks import LABELS
from .pipeline import predict

_DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_landmarks(value: str) -> list[str]:
    items = [s.strip() for s in value.split(",") if s.strip()]
    if not items:
        raise argparse.ArgumentTypeError("랜드마크 목록이 비어 있습니다.")
    unknown = [lm for lm in items if lm not in LABELS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"알 수 없는 랜드마크: {', '.join(unknown)}. "
            f"--list-landmarks 로 전체 목록을 확인하세요."
        )
    return list(dict.fromkeys(items))


def print_available_landmarks() -> None:
    print("사용 가능한 랜드마크 (--landmarks / -l 에 쉼표로 지정):")
    for name in LABELS:
        print(f"  {name}")


def build_parser(default_root: str | None = None) -> argparse.ArgumentParser:
    root = default_root or _DEFAULT_ROOT
    parser = argparse.ArgumentParser(
        description="CBCT 랜드마크 추론",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="예: dental-cbct-predict -v vol.nii.gz -l Gn,Pog,B,RCo,LCo",
    )
    parser.add_argument("--volume", "-v", help="입력 볼륨 경로 (.nii.gz)")
    parser.add_argument(
        "--landmarks",
        "-l",
        type=parse_landmarks,
        metavar="NAMES",
        help="추론할 랜드마크 (쉼표 구분, 예: Gn,Pog,B,RCo,LCo)",
    )
    parser.add_argument(
        "--list-landmarks",
        action="store_true",
        help="지원 랜드마크 이름만 출력하고 종료",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(root, "outputs"),
        help="결과 저장 폴더 (기본: outputs/)",
    )
    parser.add_argument(
        "--models-dir",
        "-m",
        default=os.path.join(root, "models"),
        help="모델 폴더 (기본: models/)",
    )
    parser.add_argument(
        "--save-grouped",
        action="store_true",
        help="(선택) CB/L/U 그룹별 _lm_Pred_*.mrk.json 도 함께 저장",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_landmarks:
        print_available_landmarks()
        return 0

    if not args.volume:
        parser.error("--volume / -v 가 필요합니다.")
    if not args.landmarks:
        parser.error(
            "--landmarks / -l 가 필요합니다. 예: -l Gn,Pog,B  "
            "(전체 목록: --list-landmarks)"
        )

    results = predict(
        volume=args.volume,
        models_dir=args.models_dir,
        landmarks=args.landmarks,
        output_dir=args.output_dir,
        save_grouped=args.save_grouped,
    )

    print("\n── 추론 결과 (LPS mm) ──")
    for lm in args.landmarks:
        if lm in results:
            coord = results[lm]
            print(
                f"  {lm:<10} x={coord['x']:8.3f}  "
                f"y={coord['y']:8.3f}  z={coord['z']:8.3f}"
            )
        else:
            print(f"  {lm:<10} (탐색 실패 또는 결과 없음)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
