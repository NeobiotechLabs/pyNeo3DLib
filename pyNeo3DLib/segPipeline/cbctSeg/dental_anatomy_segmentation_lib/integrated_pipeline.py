"""
통합 파이프라인 (단일 진입점)

NIfTI 입력 → nnU-Net → 라벨맵·센터라인 저장.

일반 사용:
    from dental_anatomy_segmentation_lib import run_dental_pipeline_from_nifti

    r = run_dental_pipeline_from_nifti(Path("volume.nii.gz"), Path("work"))
    print(r.prediction_nifti)

**라벨맵만 있을 때** (nnU-Net 불필요, 격자 정렬):
    from dental_anatomy_segmentation_lib import run_aligned_labelmap_to_reference

    aligned = run_aligned_labelmap_to_reference(pred_nii, ref_volume_nii, out_nii)
    print(aligned)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union  # noqa: F401

from .postprocess_labelmap import LabelCcConfig
from .volume_mesh_pipeline.pipeline import (
    DentalPipelineResult as _DentalPipelineResult,
    run_local_nifti_pipeline,
)


@dataclass
class DentalPipelineOptions:
    """
    통합 파이프라인 옵션 (모델·nnU-Net).

    - ``model_dir``: nnU-Net 학습 폴더 경로. None 이면 기본 경로 사용.
    - ``nnunet_device``: 추론 장치 ("cuda:0", "cpu" 등). None 이면 자동 감지.
    - ``save_input_volume_nifti`` (기본 ``False``): True 시 ``work_dir/input_volume.nii.gz`` 복사.
    - ``save_segmentation_labelmap_nifti`` (기본 ``True``): 세그멘테이션 라벨맵 파일 저장 여부.
    - ``restore_mandibular`` (기본 ``True``): 하악 신경관 복원 적용 + 중심선 JSON 저장.
    - ``label_cc_config``: 최종 라벨맵 연결 성분 필터링 구성.
        None 이면 기본값(상악두개골 1 CC, 상악동 2 CC, 하악골 1 CC, 신경관 2 CC) 을 사용.
        NonEmpty 하면 필터링을 건너뛴다.
    - ``export_meshes`` (기본 ``True``): True 시 STL 메쉬 내보내기 활성화.
    - ``mesh_decimation_factor`` (기본 ``0.5``): 메쉬 단순화 비율 (0~1, 작을수록 가벼움).
    - ``mesh_smoothing_iterations`` (기본 ``15``): 하위 호환용. Slicer 방식 스무딩은
        ``mesh_smoothing_factor`` 로부터 반복 횟수를 자동 계산합니다.
    - ``mesh_smoothing_factor`` (기본 ``0.5``): 스무딩 강도.
    - ``mesh_label_ids`` (기본 ``None``): 메쉬로 변환할 라벨 ID 목록.
        None 이면 [1, 2, 3, 4] (상악두개골, 하악골, 신경관, 상악동) 전체 변환.
    """

    model_dir: Optional[Union[str, Path]] = None
    nnunet_device: Optional[str] = None
    save_input_volume_nifti: bool = False
    save_segmentation_labelmap_nifti: bool = True
    restore_mandibular: bool = True
    label_cc_config: Optional[LabelCcConfig] = None
    export_meshes: bool = True
    mesh_decimation_factor: float = 0.5
    mesh_smoothing_iterations: int = 15
    mesh_smoothing_factor: float = 0.5
    mesh_label_ids: Optional[List[int]] = None


# ── 레거시 클래스 별칭 (외부 API 호환) ─────────────────────────

class LabelmapMeshPipelineResult:
    """격자 정렬 결과."""
    def __init__(self, aligned_labelmap_nifti: Path):
        self.aligned_labelmap_nifti = aligned_labelmap_nifti


def run_aligned_labelmap_to_reference(
    prediction_nifti: Union[str, Path],
    reference_nifti: Union[str, Path],
    output_aligned_nifti: Union[str, Path],
) -> Path:
    """예측 라벨 NIfTI 를 참조 볼륨 NIfTI 격자에 최인접으로 맞춘 파일을 저장합니다."""
    from labelmap_nifti_to_stl import run_align_labelmap_to_reference
    return run_align_labelmap_to_reference(prediction_nifti, reference_nifti, output_aligned_nifti)


# 레거시 함수 정의 (호환성용)
def run_dental_pipeline_from_nifti(
    input_volume_nifti: Union[str, Path],
    work_dir: Union[str, Path],
    model_dir: Union[str, Path],  # 필수
    progress: Optional[Callable[[str], None]] = None,
    nnunet_device: Optional[str] = None,
    save_input_volume_nifti: bool = False,
    save_segmentation_labelmap_nifti: bool = True,
    restore_mandibular: bool = True,
    label_cc_config: Optional[LabelCcConfig] = None,
    export_meshes: bool = True,
    mesh_decimation_factor: float = 0.5,
    mesh_smoothing_iterations: int = 15,
    mesh_smoothing_factor: float = 0.5,
    mesh_label_ids: Optional[List[int]] = None,
) -> _DentalPipelineResult:
    """
    **NIfTI 입력 파이프라인** — 볼륨 NIfTI → nnU-Net → 라벨맵·센터라인→ 메쉬(STL) 익스포트.

    단일 통합 모델을 사용합니다.
    """
    return run_local_nifti_pipeline(
        input_volume_nifti,
        work_dir,
        model_dir=model_dir,
        progress=progress,
        nnunet_device=nnunet_device,
        save_input_volume_nifti=save_input_volume_nifti,
        save_segmentation_labelmap_nifti=save_segmentation_labelmap_nifti,
        restore_mandibular=restore_mandibular,
        label_cc_config=label_cc_config,
        export_meshes=export_meshes,
        mesh_decimation_factor=mesh_decimation_factor,
        mesh_smoothing_iterations=mesh_smoothing_iterations,
        mesh_smoothing_factor=mesh_smoothing_factor,
        mesh_label_ids=mesh_label_ids,
    )


DentalPipelineResult = _DentalPipelineResult
LocalSegmentationMeshResult = _DentalPipelineResult


class DentalIntegratedPipeline:
    """
    (레거시) DICOM → 메시지 통합 파이프라인 인스턴스.
    현재는 라벨맵 전용 배치 파이프라인을 사용합니다.
    """
    def __init__(self, options=None):
        self.options = options or DentalPipelineOptions()


DentalMeshJobRequest = None  # type: ignore[misc]
run_dental_mesh_job = lambda *a, **k: _DentalPipelineResult(None, None, [], Path("."))  # type: ignore[misc]
run_dental_pipeline = lambda *a, **k: _DentalPipelineResult(None, None, [], Path("."))  # type: ignore[misc]


__all__ = [
    "DentalIntegratedPipeline",
    "DentalMeshJobRequest",
    "DentalPipelineOptions",
    "DentalPipelineResult",
    "LabelmapMeshPipelineResult",
    "run_aligned_labelmap_to_reference",
    "run_dental_mesh_job",
    "run_dental_pipeline",
    "run_dental_pipeline_from_nifti",
]
