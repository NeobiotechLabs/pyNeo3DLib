"""NIfTI 파이프라인 결과 값 객체."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class DentalPipelineResult:
    """
    NIfTI 입력부터 최종 산출까지 돌린 결과.

    ``input_nifti`` 는 옵션으로 ``work_dir/input_volume.nii.gz`` 에 복사했을 때만 경로가 있고,
    아니면 ``None`` 입니다.

    ``prediction_nifti`` 는 정렬 라벨맵을 저장할 때만 경로가 있고,
    메시만 내보낼 때 등 생략하면 ``None`` 입니다.

    ``mesh_files`` 은 항상 비어 있습니다 (배치에서는 export_meshes=False).
    """

    input_nifti: Optional[Path]
    prediction_nifti: Optional[Path]
    mesh_files: List[Path]
    work_dir: Path
    landmarks_coordinates_json: Optional[Path] = None
    landmarks_mrk_json: Optional[Path] = None
    centerline_json: Optional[Path] = None


# 레거시 별칭
DicomToMeshesPipelineResult = DentalPipelineResult


__all__ = [
    "DentalPipelineResult",
    "DicomToMeshesPipelineResult",
    "LocalSegmentationMeshResult",
]
LocalSegmentationMeshResult = DentalPipelineResult  # noqa: E305
