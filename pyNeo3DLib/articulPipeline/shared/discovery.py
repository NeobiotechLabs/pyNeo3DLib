"""케이스 폴더·STL 파일 탐색 공통 유틸."""

from __future__ import annotations

from pathlib import Path

from core.landmarks import SlicerMarkupLandmarkReader

MERGED_MRK_GLOB = "*_merged.mrk.json"
MRK_GLOB = "*.mrk.json"

MANDIBULAR_CANAL_FALLBACK_PATTERNS = (
    "*[Mm]andibular*[Cc]anal*.stl",
    "*canal*.stl",
)
MANDIBULAR_CANAL_PATTERNS = ("Mandibular_canal.stl", *MANDIBULAR_CANAL_FALLBACK_PATTERNS)
UPPER_SKULL_PATTERNS = ("Upper_Skull.stl", "*[Uu]pper*[Ss]kull*.stl")
MANDIBLE_PATTERNS = ("Mandible.stl", "*[Mm]andible*.stl")
TOOTH_GLOB = "tooth_*.stl"


def discover_teeth_stls(directory: Path) -> tuple[Path, ...]:
    """케이스 폴더에서 치아 STL 목록을 반환 (파일명 정렬)."""
    return tuple(sorted(p for p in directory.glob(TOOTH_GLOB) if p.is_file()))


def _landmark_json_has_required_labels(
    path: Path,
    required_labels: tuple[str, ...],
    reader: SlicerMarkupLandmarkReader,
) -> bool:
    try:
        raw = reader.read(path)
        return bool(raw) and all(name in raw for name in required_labels)
    except (ValueError, KeyError, OSError, TypeError):
        return False


def find_landmark_json(
    directory: Path,
    required_labels: tuple[str, ...],
    *,
    priority_globs: tuple[str, ...] = (),
    reader: SlicerMarkupLandmarkReader | None = None,
) -> Path | None:
    """우선 glob → 필수 라벨이 모두 있는 .mrk.json 순으로 탐색."""
    directory = directory.expanduser().resolve()
    markup_reader = reader or SlicerMarkupLandmarkReader()

    for pattern in priority_globs:
        for path in sorted(p for p in directory.glob(pattern) if p.is_file()):
            if _landmark_json_has_required_labels(path, required_labels, markup_reader):
                return path

    for path in sorted(p for p in directory.glob(MRK_GLOB) if p.is_file()):
        if _landmark_json_has_required_labels(path, required_labels, markup_reader):
            return path
    return None


def mandibular_canal_patterns(primary: str = "Mandibular_canal.stl") -> tuple[str, ...]:
    return (primary, *MANDIBULAR_CANAL_FALLBACK_PATTERNS)


def first_hit(directory: Path, patterns: tuple[str, ...]) -> Path | None:
    """디렉터리(및 하위)에서 glob 패턴 순서대로 첫 파일을 반환."""
    for pattern in patterns:
        direct = sorted(p for p in directory.glob(pattern) if p.is_file())
        if direct:
            return direct[0]
        nested = sorted(p for p in directory.rglob(pattern) if p.is_file())
        if nested:
            return nested[0]
    return None


class CanalStlLocator:
    """케이스 폴더에서 하악 신경관 STL을 탐색."""

    def find(self, case_dir: Path, canal_glob: str = "Mandibular_canal.stl") -> Path | None:
        return first_hit(case_dir, mandibular_canal_patterns(canal_glob))
