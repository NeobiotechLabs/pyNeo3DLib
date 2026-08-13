"""이미 완료된 케이스 건너뛰기 판별."""

from __future__ import annotations

from pathlib import Path


def expected_pred_nifti(vol: Path, case_dir: Path) -> Path:
    """배치에서 예측 라벨맵을 저장하는 파일명: ``{원본이름}_pred.nii.gz``."""
    stem = _volume_stem_from_nifti(vol)
    return case_dir / f"{stem}_pred.nii.gz"


def expected_centerline_json(vol: Path, case_dir: Path) -> Path:
    stem = _volume_stem_from_nifti(vol)
    return case_dir / f"{stem}_centerline.json"


def _volume_stem_from_nifti(path: Path) -> str:
    p = Path(path)
    name = p.name
    low = name.lower()
    if low.endswith(".nii.gz"):
        return name[:-7]
    if low.endswith(".nii"):
        return name[:-4]
    return p.stem


def case_output_complete(
    vol: Path,
    case_dir: Path,
    *,
    restore_mandibular: bool = False,
) -> bool:
    """배치는 항상 ``{원본이름}_pred.nii.gz`` 를 저장하므로 그 존재가 기본 판별 조건."""
    pred = expected_pred_nifti(vol, case_dir)
    if not (pred.is_file() and pred.stat().st_size > 0):
        return False

    if restore_mandibular:
        cl = expected_centerline_json(vol, case_dir)
        if not (cl.is_file() and cl.stat().st_size > 0):
            return False

    return True
