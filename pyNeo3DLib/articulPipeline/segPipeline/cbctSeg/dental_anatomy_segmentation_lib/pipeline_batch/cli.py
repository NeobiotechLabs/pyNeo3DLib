"""argparse 정의 — 입력/출력/모델 경로 (CLI 인자 또는 환경변수)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_batch.env_paths import ENV_INPUT, ENV_OUTPUT, ENV_NNUNET_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "NIfTI 세그멘테이션 (nnU-Net 라벨맵 생성). "
            f"경로는 CLI 인자 또는 환경변수({ENV_INPUT}, {ENV_OUTPUT}, {ENV_NNUNET_MODEL})로 지정."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="KEY=VALUE 형식 파일로 환경변수 보충. 생략 시 저장소 루트 .env.dental_cbct 자동 로드",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        metavar="FILE.or.DIR",
        help=f"입력 NIfTI 파일 (*.nii.gz) 또는 폴더. 미지정 시 환경변수 {ENV_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=f"출력 디렉터리. 미지정 시 환경변수 {ENV_OUTPUT}",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=f"nnU-Net 번들 루트. 미지정 시 환경변수 {ENV_NNUNET_MODEL}",
    )
    parser.add_argument(
        "--no-restore-mandibular",
        action="store_true",
        help="하악 신경관 복원 단계를 건너뜀",
    )
    parser.add_argument(
        "--no-export-meshes",
        action="store_true",
        help="메쉬 내보내기를 건너뜀 (기본 활성화)",
    )
    parser.add_argument(
        "--mesh-decimation",
        type=float,
        default=0.5,
        metavar="FACTOR",
        help="메쉬 단순화 비율 (0~1). 0.5=면적 절반, 1.0=변경 안 함 (기본 0.5)",
    )
    parser.add_argument(
        "--mesh-smoothing-iterations",
        type=int,
        default=15,
        metavar="N",
        help="Laplacian 스무딩 반복 횟수 (기본 15)",
    )
    parser.add_argument(
        "--mesh-smoothing-factor",
        type=float,
        default=0.5,
        metavar="FACTOR",
        help="스무딩 강도 (0~1, 클수록 강함, 기본 0.5)",
    )
    parser.add_argument(
        "--mesh-label-ids",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="메쉬로 변환할 라벨 ID 목록 (예: --mesh-label-ids 1 2 3). 생략 시 [1,2,3,4] 전체.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="요약만 출력",
    )
    return parser
