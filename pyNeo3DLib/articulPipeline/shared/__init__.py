"""articulPipeline 공통 상수·유틸."""

from .constants import CRANIAL_LANDMARKS, MEF_LANDMARKS
from .mrk_output_names import (
    LANDMARK_MERGE_MRK_SUFFIX,
    LANDMARK_MRK_SUFFIX,
    MANDIBLE_CONDYLES_MRK_SUFFIX,
    MRK_OUTPUT_SUFFIXES,
    NERVE_CANAL_MEF_MRK_SUFFIX,
    case_stem,
    mrk_filename,
)
from .validation import LandmarkValidationError, validate_required_landmarks

__all__ = [
    "CRANIAL_LANDMARKS",
    "MEF_LANDMARKS",
    "LANDMARK_MERGE_MRK_SUFFIX",
    "LANDMARK_MRK_SUFFIX",
    "MANDIBLE_CONDYLES_MRK_SUFFIX",
    "MRK_OUTPUT_SUFFIXES",
    "NERVE_CANAL_MEF_MRK_SUFFIX",
    "case_stem",
    "mrk_filename",
    "MERGED_MRK_GLOB",
    "MRK_GLOB",
    "MANDIBULAR_CANAL_PATTERNS",
    "UPPER_SKULL_PATTERNS",
    "MANDIBLE_PATTERNS",
    "TOOTH_GLOB",
    "CanalStlLocator",
    "discover_teeth_stls",
    "find_landmark_json",
    "first_hit",
    "LandmarkValidationError",
    "validate_required_landmarks",
]

#: discovery 모듈에서 다시 export 하는 이름 (지연 로드).
#: discovery는 landmarks 리더 모듈이 필요해서 실제로 사용할 때만 import 합니다.
_DISCOVERY_NAMES = frozenset(
    {
        "MERGED_MRK_GLOB",
        "MRK_GLOB",
        "MANDIBULAR_CANAL_PATTERNS",
        "UPPER_SKULL_PATTERNS",
        "MANDIBLE_PATTERNS",
        "TOOTH_GLOB",
        "CanalStlLocator",
        "discover_teeth_stls",
        "find_landmark_json",
        "first_hit",
    }
)


def __getattr__(name: str):
    if name in _DISCOVERY_NAMES:
        from . import discovery

        return getattr(discovery, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
