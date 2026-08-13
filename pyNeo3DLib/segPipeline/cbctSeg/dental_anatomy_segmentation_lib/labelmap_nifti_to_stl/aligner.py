"""예측 라벨맵을 참조 NIfTI 격자에 최인접 리샘플로 맞춥니다."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .resample_to_reference import resample_label_nifti_to_reference_geometry


class NearestNeighborLabelmapAligner:
    """dental_anatomy_segmentation_lib ``LabelmapToReferenceAligner`` Protocol 호환 기본 구현."""

    def align_prediction_to_reference(
        self,
        prediction_nifti: Union[str, Path],
        reference_nifti: Union[str, Path],
        output_nifti: Union[str, Path],
    ) -> Path:
        return resample_label_nifti_to_reference_geometry(
            prediction_nifti,
            reference_nifti,
            output_nifti,
        )


__all__ = ["NearestNeighborLabelmapAligner"]
