"""랜드마크 산출물 mrk.json 파일 이름 공통 정의 — ``articulPipeline/mrk_output_names.json`` 에서 로드.

articulPipeline 전체에서 랜드마크 산출물 mrk.json 을 저장할 때 쓰는 파일 이름
접미어는 ``articulPipeline/mrk_output_names.json`` 한 곳에서 관리합니다. 이
모듈은 그 JSON을 상위 폴더에서 찾아 읽어 제공만 합니다.

이름을 바꾸려면 이 모듈이 아니라 JSON 파일을 수정하세요. (JSON이 없으면
``_DEFAULT_SUFFIXES`` 기본값을 사용합니다.)

JSON 형식 (산출물 종류 → 파일 이름 접미어)::

    {
        "landmark_inference": "_merged.mrk.json",
        "mandible_condyles": "_mandible_condyles.mrk.json",
        "nerve_canal_mef": "_nerve_canal_mef.mrk.json",
        "landmark_merge": "_landmarks.mrk.json"
    }

사용 예::

    from mrk_output_names import mrk_filename   # shared 에서는 shared.mrk_output_names
    mrk_filename("case01_mandible", "mandible_condyles")
    # → "case01_mandible_condyles.mrk.json"

이 모듈은 ``shared/``, ``condylePointsFinder/``, articulPipeline 루트에
동일한 복제본으로 존재합니다 (``structure_names.py`` 복제 규약과 동일 —
서브파이프라인별 import 환경이 격리되어 있어 공유 모듈을 직접 쓸 수 없음).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

#: articulPipeline 루트에 있는 공통 정의 파일 이름
_JSON_NAME = "mrk_output_names.json"

#: JSON 이 없을 때 사용할 기본 접미어 (JSON 내용과 동일한 규약)
_DEFAULT_SUFFIXES: Dict[str, str] = {
    "landmark_inference": "_merged.mrk.json",
    "mandible_condyles": "_mandible_condyles.mrk.json",
    "nerve_canal_mef": "_nerve_canal_mef.mrk.json",
    "landmark_merge": "_landmarks.mrk.json",
}

#: 산출물 종류 → 입력 파일 stem 에서 떼어낼 접미어.
#: ``mrk_output_names.json`` 의 키와 동기화할 것 (JSON 에서 유도 불가).
#: (통합은 랜드마크 입력 파일의 접미어 ``_merged`` 를 떼어 케이스명을 만든다)
_STRIP_SUFFIXES: Dict[str, str] = {
    "landmark_inference": "",
    "mandible_condyles": "_mandible",
    "nerve_canal_mef": "_nerve_canal",
    "landmark_merge": "_merged",
}


def _find_config_path() -> Path | None:
    """이 파일의 상위 폴더들을 거슬러 올라가며 공통 정의 JSON을 찾습니다."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _JSON_NAME
        if candidate.is_file():
            return candidate
    return None


def _load_suffixes() -> Dict[str, str]:
    """JSON 에서 접미어 사전 로드. 없거나 손상됐으면 기본값 사용."""
    config_path = _find_config_path()
    if config_path is None:
        return dict(_DEFAULT_SUFFIXES)
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        suffixes = {str(kind): str(suffix) for kind, suffix in data.items()}
    except (OSError, ValueError):
        return dict(_DEFAULT_SUFFIXES)
    # 필수 키가 없으면 기본값으로 보강
    for kind, default in _DEFAULT_SUFFIXES.items():
        suffixes.setdefault(kind, default)
    return suffixes


#: 산출물 종류 → 파일 이름 접미어 (예: ``"mandible_condyles" → "_mandible_condyles.mrk.json"``)
MRK_OUTPUT_SUFFIXES: Dict[str, str] = _load_suffixes()

#: 랜드마크 추론 결과 접미어 (예: ``case01_merged.mrk.json``)
LANDMARK_MRK_SUFFIX: str = MRK_OUTPUT_SUFFIXES["landmark_inference"]

#: 하악골 콘다일점(LCo/RCo) 접미어 (예: ``case01_mandible_condyles.mrk.json``)
MANDIBLE_CONDYLES_MRK_SUFFIX: str = MRK_OUTPUT_SUFFIXES["mandible_condyles"]

#: 하악 신경관 MeF(LMeF/RMeF) 접미어 (예: ``case01_nerve_canal_mef.mrk.json``)
NERVE_CANAL_MEF_MRK_SUFFIX: str = MRK_OUTPUT_SUFFIXES["nerve_canal_mef"]

#: 랜드마크+콘다일+MeF 통합 결과 접미어 (예: ``case01_landmarks.mrk.json``)
LANDMARK_MERGE_MRK_SUFFIX: str = MRK_OUTPUT_SUFFIXES["landmark_merge"]


def case_stem(stem: str, kind: str) -> str:
    """입력 메쉬 stem 에서 케이스 이름 추출.

    ``{케이스}_{구조}`` 형태면 구조 접미어를 떼어내고, 아니면 stem 그대로
    반환합니다. 예::

        case_stem("case01_mandible", "mandible_condyles")   # → "case01"
        case_stem("Mandibular_canal", "nerve_canal_mef")    # → "Mandibular_canal"
    """
    if kind not in _STRIP_SUFFIXES:
        raise KeyError(
            f"알 수 없는 산출물 종류: {kind!r} (허용: {sorted(_STRIP_SUFFIXES)})"
        )
    suffix = _STRIP_SUFFIXES[kind]
    if suffix and stem.endswith(suffix) and len(stem) > len(suffix):
        return stem[: -len(suffix)]
    return stem


def mrk_filename(stem: str, kind: str) -> str:
    """산출물 mrk.json 파일 이름: ``{케이스이름}{접미어}``.

    예: ``mrk_filename("case01_mandible", "mandible_condyles")``
    → ``"case01_mandible_condyles.mrk.json"``
    """
    if kind not in MRK_OUTPUT_SUFFIXES:
        raise KeyError(
            f"알 수 없는 산출물 종류: {kind!r} (허용: {sorted(MRK_OUTPUT_SUFFIXES)})"
        )
    return f"{case_stem(stem, kind)}{MRK_OUTPUT_SUFFIXES[kind]}"


__all__ = [
    "MRK_OUTPUT_SUFFIXES",
    "LANDMARK_MRK_SUFFIX",
    "MANDIBLE_CONDYLES_MRK_SUFFIX",
    "NERVE_CANAL_MEF_MRK_SUFFIX",
    "LANDMARK_MERGE_MRK_SUFFIX",
    "case_stem",
    "mrk_filename",
]
