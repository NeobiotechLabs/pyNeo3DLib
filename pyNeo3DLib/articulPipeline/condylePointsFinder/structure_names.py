"""해부학 구조 이름 공통 정의 — ``articulPipeline/structure_names.json`` 에서 로드.

articulPipeline 전체에서 STL 등 산출물을 저장/불러올 때 쓰는 구조 이름은
``articulPipeline/structure_names.json`` 한 곳에서 관리합니다. 이 모듈은 그
JSON을 상위 폴더에서 찾아 읽어 제공만 합니다.

이름을 바꾸려면 이 모듈이 아니라 JSON 파일을 수정하세요.

JSON 형식 (구조 이름 → 세그멘테이션 라벨 ID)::

    {
        "maxilla": 1,
        "mandible": 2,
        "nerve_canal": 3,
        "maxillary_sinus": 4
    }

사용 예::

    from structure_names import MANDIBLE, mesh_filename
    mesh_filename("case01", MANDIBLE)          # → "case01_mandible.stl"
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

#: articulPipeline 루트에 있는 공통 정의 파일 이름
_JSON_NAME = "structure_names.json"

#: 이 모듈이 참조해야 하는 필수 구조 이름
REQUIRED_STRUCTURES = ("maxilla", "mandible", "nerve_canal", "maxillary_sinus")


def _find_config_path() -> Path:
    """이 파일의 상위 폴더들을 거슬러 올라가며 공통 정의 JSON을 찾습니다."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _JSON_NAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{Path(__file__).resolve()} 의 상위 폴더에서 {_JSON_NAME} 을 찾을 수 없습니다"
    )


def _load_names() -> Dict[str, int]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    names = {str(name): int(label) for name, label in data.items()}
    missing = [s for s in REQUIRED_STRUCTURES if s not in names]
    if missing:
        raise KeyError(f"{CONFIG_PATH} 에 필수 구조 이름이 없습니다: {missing}")
    return names


#: 공통 정의 JSON 경로 (articulPipeline 루트)
CONFIG_PATH: Path = _find_config_path()

#: 구조 이름 → 세그멘테이션 라벨 ID (예: ``"mandible" → 2``)
STRUCTURE_NAMES: Dict[str, int] = _load_names()

#: 세그멘테이션 라벨 ID → 구조 이름 (예: ``2 → "mandible"``)
STRUCTURE_LABELS: Dict[int, str] = {v: k for k, v in STRUCTURE_NAMES.items()}

MAXILLA: str = "maxilla"
MANDIBLE: str = "mandible"
NERVE_CANAL: str = "nerve_canal"
MAXILLARY_SINUS: str = "maxillary_sinus"


def mesh_filename(stem: str, structure: str, suffix: str = "stl") -> str:
    """메쉬 파일 이름 규약: ``{케이스이름}_{구조이름}.{확장자}``.

    예: ``mesh_filename("case01", MANDIBLE)`` → ``"case01_mandible.stl"``
    """
    if structure not in STRUCTURE_NAMES:
        raise KeyError(
            f"알 수 없는 구조 이름: {structure!r} (허용: {sorted(STRUCTURE_NAMES)})"
        )
    return f"{stem}_{structure}.{suffix.lstrip('.')}"


def mesh_stem_suffix(structure: str) -> str:
    """저장된 메쉬에서 구조를 식별하는 접미 패턴: ``_{구조이름}`` (glob 에 사용)."""
    return f"_{structure}"


__all__ = [
    "CONFIG_PATH",
    "STRUCTURE_NAMES",
    "STRUCTURE_LABELS",
    "MAXILLA",
    "MANDIBLE",
    "NERVE_CANAL",
    "MAXILLARY_SINUS",
    "mesh_filename",
    "mesh_stem_suffix",
]
