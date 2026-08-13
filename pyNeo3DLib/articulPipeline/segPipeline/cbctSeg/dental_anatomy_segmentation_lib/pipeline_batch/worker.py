"""단일 NIfTI 케이스 추론."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from pipeline_batch.env_paths import REPO_ROOT
from pipeline_batch.progress import prefixed_case_progress


def run_one_nifti_case(
    root_str: str,
    vol_str: str,
    case_dir_str: str,
    nnunet_device: Optional[str],
    model_dir_str: str,
    restore_mandibular: bool,
    export_meshes: bool = True,
    mesh_decimation_factor: float = 0.5,
    mesh_smoothing_iterations: int = 15,
    mesh_smoothing_factor: float = 0.5,
    mesh_label_ids: Optional[List[int]] = None,
) -> str:
    root = Path(root_str)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from dental_anatomy_segmentation_lib import run_dental_pipeline_from_nifti
    from dental_anatomy_segmentation_lib.pipeline_runtime import release_accelerator_memory

    vol = Path(vol_str)
    case_dir = Path(case_dir_str)
    case_dir.mkdir(parents=True, exist_ok=True)
    case_label = vol.name
    model_dir = Path(model_dir_str).resolve()
    try:
        result = run_dental_pipeline_from_nifti(
            vol,
            case_dir,
            progress=prefixed_case_progress(case_label),
            model_dir=model_dir,
            nnunet_device=nnunet_device,
            restore_mandibular=restore_mandibular,
            export_meshes=export_meshes,
            mesh_decimation_factor=mesh_decimation_factor,
            mesh_smoothing_iterations=mesh_smoothing_iterations,
            mesh_smoothing_factor=mesh_smoothing_factor,
            mesh_label_ids=mesh_label_ids,
        )
    finally:
        release_accelerator_memory()
    lab_part = (
        f"라벨맵 {result.prediction_nifti}"
        if result.prediction_nifti is not None
        else "라벨맵 NIfTI 생략"
    )
    cl_part = (
        f"센터라인 {result.centerline_json}"
        if result.centerline_json is not None
        else ("센터라인 생략" if not restore_mandibular else "센터라인 없음")
    )
    mesh_part = ""
    if result.mesh_files:
        mesh_names = ", ".join(f.name for f in result.mesh_files)
        mesh_part = f"\n  메쉬: {mesh_names}"
    return (
        f"완료 [{vol.name}]: device={nnunet_device or 'default'}, {lab_part}, {cl_part}{mesh_part} → {result.work_dir}"
    )


def default_repo_root_str() -> str:
    return str(REPO_ROOT.resolve())
