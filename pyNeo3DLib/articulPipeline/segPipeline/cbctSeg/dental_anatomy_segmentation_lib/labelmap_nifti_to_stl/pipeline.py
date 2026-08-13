"""
라벨맵 처리 **공개 API**: 격자 정렬만 / 정렬+메시 한 번에.

외부 코드는 가능하면 이 모듈의 함수만 호출하는 것을 권장합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from .labelmap_to_mesh import export_meshes_from_label_nifti
from .resample_to_reference import resample_label_nifti_to_reference_geometry


@dataclass(frozen=True)
class LabelmapMeshPipelineResult:
    """``run_align_prediction_to_meshes`` 결과."""

    aligned_labelmap_nifti: Path
    mesh_files: List[Path]


def run_align_labelmap_to_reference(
    prediction_nifti: Union[str, Path],
    reference_nifti: Union[str, Path],
    output_aligned_nifti: Union[str, Path],
) -> Path:
    """
    예측 라벨 NIfTI를 참조 볼륨 NIfTI 격자에 최인접으로 맞춘 파일을 저장합니다.

    :returns: 저장된 ``output_aligned_nifti`` 절대 경로
    """
    return resample_label_nifti_to_reference_geometry(
        prediction_nifti,
        reference_nifti,
        output_aligned_nifti,
    )


def run_align_prediction_to_meshes(
    prediction_nifti: Union[str, Path],
    reference_nifti: Union[str, Path],
    mesh_output_dir: Union[str, Path],
    *,
    aligned_labelmap_path: Optional[Union[str, Path]] = None,
    dataset_json: Optional[Union[str, Path]] = None,
    mesh_formats: Iterable[str] = ("stl",),
    mesh_step_size: int = 1,
    mesh_label_names: Optional[Sequence[str]] = None,
    mesh_postprocess: bool = True,
    mesh_keep_largest_component: bool = True,
    mesh_decimation_factor: float = 0.5,
    mesh_smoothing_factor: float = 0.5,
) -> LabelmapMeshPipelineResult:
    """
    예측 라벨을 참조 CBCT NIfTI 격자에 맞춘 뒤, 라벨별 메시(STL 등)를 ``mesh_output_dir`` 에 저장합니다.

    ``aligned_labelmap_path`` 가 None이면 ``mesh_output_dir / "prediction_aligned_to_reference.nii.gz"`` 에 저장합니다.
    """
    mesh_out = Path(mesh_output_dir).resolve()
    mesh_out.mkdir(parents=True, exist_ok=True)

    if aligned_labelmap_path is None:
        aligned = mesh_out / "prediction_aligned_to_reference.nii.gz"
    else:
        aligned = Path(aligned_labelmap_path).expanduser().resolve()
        aligned.parent.mkdir(parents=True, exist_ok=True)

    run_align_labelmap_to_reference(prediction_nifti, reference_nifti, aligned)

    mesh_files = export_meshes_from_label_nifti(
        aligned,
        mesh_out,
        dataset_json=dataset_json,
        formats=mesh_formats,
        step_size=mesh_step_size,
        label_names=mesh_label_names,
        mesh_postprocess=mesh_postprocess,
        mesh_keep_largest_component=mesh_keep_largest_component,
        mesh_decimation_factor=mesh_decimation_factor,
        mesh_smoothing_factor=mesh_smoothing_factor,
    )
    return LabelmapMeshPipelineResult(
        aligned_labelmap_nifti=aligned.resolve(),
        mesh_files=mesh_files,
    )


__all__ = [
    "LabelmapMeshPipelineResult",
    "run_align_labelmap_to_reference",
    "run_align_prediction_to_meshes",
]
