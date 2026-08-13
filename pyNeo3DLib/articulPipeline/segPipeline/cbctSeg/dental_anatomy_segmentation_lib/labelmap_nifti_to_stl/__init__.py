"""
라벨맵 NIfTI: **참조 격자 정렬**(최인접) + **STL/OBJ/PLY** (torch·nnU-Net 불필요).

**다른 프로젝트**: ``dental_anatomy_segmentation_lib/labelmap_nifti_to_stl/`` 폴더 통째로 복사 후 그 안에서 ``pip install -e .`` (또는 저장소 루트 ``pip install -e .``).

**공개 API** (이름만 기억해도 됨):

- ``run_align_labelmap_to_reference`` — 정렬된 라벨 NIfTI만 저장
- ``run_align_prediction_to_meshes`` — 정렬 + 메시까지
- ``export_meshes_from_label_nifti`` — 이미 격자가 맞는 라벨맵만 메시화

의존성: ``numpy``, ``itk``, ``trimesh``, ``vtk``, ``pyvista``.
"""

from __future__ import annotations

from .aligner import NearestNeighborLabelmapAligner
from .labelmap_to_mesh import export_meshes_from_label_nifti
from .pipeline import (
    LabelmapMeshPipelineResult,
    run_align_labelmap_to_reference,
    run_align_prediction_to_meshes,
)
from .resample_to_reference import resample_label_nifti_to_reference_geometry

__all__ = [
    "LabelmapMeshPipelineResult",
    "NearestNeighborLabelmapAligner",
    "export_meshes_from_label_nifti",
    "resample_label_nifti_to_reference_geometry",
    "run_align_labelmap_to_reference",
    "run_align_prediction_to_meshes",
]
