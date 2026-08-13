"""랜드마크 산출물 mrk.json 파일 이름 — ``articulPipeline/mrk_output_names.json`` 에서 로드.

파일 이름 접미어는 ``articulPipeline/mrk_output_names.json`` 한 곳에서
관리합니다. 이 모듈은 그 JSON을 상위 폴더에서 찾아 읽어 제공만 합니다.
JSON이 없으면 기본값(``_merged.mrk.json``)을 사용합니다.

이름을 바꾸려면 이 모듈이 아니라 JSON 파일을 수정하세요.

cbctLandmark 는 서브프로세스(``python -m cbctLandmark.cli``,
PYTHONPATH=segPipeline)와 라이브러리 직결 import 양쪽에서 실행되므로,
``shared`` 등 외부 패키지에 의존하지 않도록 패키지 내부에 자체 로더를
둡니다 (``articulPipeline/mrk_output_names.py`` 로드 규약과 동일).
"""
from __future__ import annotations

import json
from pathlib import Path

#: articulPipeline 루트에 있는 공통 정의 파일 이름
_JSON_NAME = "mrk_output_names.json"

#: JSON 키 — cbctLandmark 는 랜드마크 추론 접미어만 사용
_KIND = "landmark_inference"

#: JSON 이 없을 때 사용할 기본 접미어 (JSON 내용과 동일한 규약)
_DEFAULT_SUFFIX = "_merged.mrk.json"


def _find_config_path() -> Path | None:
    """이 파일의 상위 폴더들을 거슬러 올라가며 공통 정의 JSON을 찾습니다."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _JSON_NAME
        if candidate.is_file():
            return candidate
    return None


def _load_suffix() -> str:
    """JSON 에서 랜드마크 추론 접미어 로드. 없거나 손상됐으면 기본값 사용."""
    config_path = _find_config_path()
    if config_path is None:
        return _DEFAULT_SUFFIX
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        suffix = data.get(_KIND)
    except (OSError, ValueError):
        return _DEFAULT_SUFFIX
    if not isinstance(suffix, str) or not suffix:
        return _DEFAULT_SUFFIX
    return suffix


#: 랜드마크 추론 결과 mrk.json 접미어 (예: ``case01_merged.mrk.json``)
LANDMARK_MRK_SUFFIX: str = _load_suffix()

__all__ = ["LANDMARK_MRK_SUFFIX"]
