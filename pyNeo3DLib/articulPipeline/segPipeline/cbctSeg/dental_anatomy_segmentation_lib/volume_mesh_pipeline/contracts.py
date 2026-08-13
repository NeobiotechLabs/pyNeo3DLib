"""
파이프라인 단계별 추상화(Protocol).

- **SRP**: 변환 / AI 추론 / 라벨맵 정렬을 각각 다른 구현으로 교체 가능
- **DIP**: 상위 파사드는 구체 클래스가 아니라 이 Protocol에만 의존
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Protocol, Sequence, Union

ProgressCallback = Optional[Callable[[str], None]]


class VolumeSegmentationPredictor(Protocol):
    """볼륨 NIfTI를 AI 세그멘테이션 모델에 넣어 라벨맵 NIfTI 생성."""

    def predict_segmentation(
        self,
        input_nifti: Union[str, Path],
        output_dir: Union[str, Path],
        model_training_dir: Union[str, Path],
        *,
        device: Optional[str] = None,
        checkpoint_name: str = "checkpoint_final.pth",
    ) -> Path:
        """예측 세그멘테이션 NIfTI 경로."""
        ...


class SegmentationMeshExporter(Protocol):
    """정수 라벨 NIfTI → 표면 메시 파일 목록."""

    def export_meshes(
        self,
        segmentation_nifti: Union[str, Path],
        output_dir: Union[str, Path],
        *,
        dataset_json: Optional[Union[str, Path]] = None,
        formats: Sequence[str] = ("stl",),
        step_size: int = 1,
        label_names: Optional[Sequence[str]] = None,
        mesh_postprocess: bool = True,
        mesh_keep_largest_component: bool = True,
        mesh_decimation_factor: float = 0.5,
        mesh_smoothing_factor: float = 0.5,
    ) -> List[Path]:
        ...


class LabelmapToReferenceAligner(Protocol):
    """예측 라벨 NIfTI를 참조 볼륨의 ITK 격자에 맞춥니다."""

    def align_prediction_to_reference(
        self,
        prediction_nifti: Union[str, Path],
        reference_nifti: Union[str, Path],
        output_nifti: Union[str, Path],
    ) -> Path:
        """저장된 ``output_nifti`` 경로."""
        ...


__all__ = [
    "LabelmapToReferenceAligner",
    "ProgressCallback",
    "SegmentationMeshExporter",
    "VolumeSegmentationPredictor",
]
