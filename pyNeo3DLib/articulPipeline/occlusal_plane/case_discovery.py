"""케이스 폴더에서 교합면 입력 파일 자동 탐색."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.landmarks import SlicerMarkupLandmarkReader
from core.shared.constants import CRANIAL_LANDMARKS
from core.shared.discovery import (
    MANDIBULAR_CANAL_PATTERNS,
    MANDIBLE_PATTERNS,
    MERGED_MRK_GLOB,
    MRK_GLOB,
    UPPER_SKULL_PATTERNS,
    discover_teeth_stls,
    find_landmark_json,
    first_hit,
)
from core.shared.validation import LandmarkValidationError


@dataclass(frozen=True)
class ResolvedCaseInputs:
    case_dir: Path
    landmarks_path: Path
    mandibular_canal_path: Path
    upper_skull_path: Path | None
    mandible_path: Path | None
    teeth_paths: tuple[Path, ...] = ()


def _require_cranial_landmark_json(directory: Path) -> Path:
    """N·ANS·PNS가 모두 있는 .mrk.json을 찾거나, 누락 시 명시적 예외."""
    landmarks = find_landmark_json(
        directory,
        CRANIAL_LANDMARKS,
        priority_globs=(MERGED_MRK_GLOB,),
    )
    if landmarks is not None:
        return landmarks

    candidates = sorted(p for p in directory.glob(MRK_GLOB) if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"{directory}: N/ANS/PNS 랜드마크 .mrk.json 없음 (*_merged.mrk.json 우선)"
        )

    reader = SlicerMarkupLandmarkReader()
    details: list[str] = []
    for path in candidates:
        try:
            raw = reader.read(path)
            missing = [name for name in CRANIAL_LANDMARKS if name not in raw]
            if missing:
                details.append(f"  {path.name}: 없음 — {', '.join(missing)}")
        except (ValueError, KeyError, OSError, TypeError) as exc:
            details.append(f"  {path.name}: 읽기 실패 ({exc})")

    raise LandmarkValidationError(
        f"{directory}: 필수 랜드마크 N, ANS, PNS가 모두 포함된 .mrk.json 없음 "
        f"(*_merged.mrk.json 우선).\n"
        + "\n".join(details)
    )


def resolve_case_folder(case_dir: Path) -> ResolvedCaseInputs:
    """케이스 폴더에서 landmarks·STL 경로를 자동으로 찾습니다."""
    directory = case_dir.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"케이스 폴더가 아님: {directory}")

    landmarks = _require_cranial_landmark_json(directory)

    canal = first_hit(directory, MANDIBULAR_CANAL_PATTERNS)
    if canal is None:
        raise FileNotFoundError(f"{directory}: Mandibular_canal.stl 없음")

    return ResolvedCaseInputs(
        case_dir=directory,
        landmarks_path=landmarks,
        mandibular_canal_path=canal,
        upper_skull_path=first_hit(directory, UPPER_SKULL_PATTERNS),
        mandible_path=first_hit(directory, MANDIBLE_PATTERNS),
        teeth_paths=discover_teeth_stls(directory),
    )
