"""입력·예측 NIfTI 전용 스크래치 (저장소 루트 ``.dentalseg_tmp`` 하위)."""

from __future__ import annotations

import uuid
from pathlib import Path


def _project_root_dir() -> Path:
    """``dental_anatomy_segmentation_lib`` 의 부모 = 프로젝트(저장소) 루트."""
    return Path(__file__).resolve().parent.parent.parent


def allocate_nifti_run_scratch_dir() -> Path:
    """
    실행마다 고유 폴더를 만들어 반환합니다.

    구조: ``<project>/.dentalseg_tmp/runs/<uuid>/``
    """
    base = _project_root_dir() / ".dentalseg_tmp" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex
    d.mkdir(parents=False, exist_ok=False)
    return d


__all__ = ["allocate_nifti_run_scratch_dir"]
