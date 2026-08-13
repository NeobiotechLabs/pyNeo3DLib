"""
NIfTI 볼륨 → nnU-Net → 라벨맵 + 신경관 복원 → 메쉬(STL) 익스포트.

중앙 진입점: ``run_local_nifti_pipeline`` (외부에서는 ``integrated_pipeline.run_dental_pipeline_from_nifti`` 사용 권장).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from ..labelmap_nifti_to_stl import export_meshes_from_label_nifti
from ..pipeline_runtime import install_batch_cleanup_handlers, release_accelerator_memory
from .nnunet_runner import run_nnunet_predict
from .result import DentalPipelineResult
from ..pipeline import LABEL_CANAL, restore_canal, write_centerline_json
from ..postprocess_labelmap import (
    LabelCcConfig,
    filter_labelmap_by_connected_components,
)
from ..restore_mandibular.config import RestoreConfig
from ..restore_mandibular.io_nifti import load_label, save_label, spacing_from_affine

#: 라벨 정수 값 → 메쉬 파일 이름 (dataset.json 없이 사용할 이름 매핑)
MESH_LABEL_FILENAMES: dict[int, str] = {
    1: "maxillary_skull",  # 상악두개골
    2: "mandible_body",    # 하악골
    3: "neural_canal",     # 신경관
    4: "maxillary_sinus",  # 상악동
}

#: 기본 메쉬 변환 라벨 — 볼륨에 없는 라벨은 건너뜀
DEFAULT_MESH_LABEL_IDS: List[int] = [1, 2, 3, 4]


def _label_id_from_default_mesh_stem(stem_name: str) -> Optional[int]:
    """``labelmap_nifti_to_stl`` 기본 파일 이름(``label_{id}``)에서 라벨 ID 파싱."""
    if not stem_name.startswith("label_"):
        return None
    try:
        return int(stem_name.split("_", 1)[1])
    except ValueError:
        return None


def export_final_labelmap_meshes(
    labelmap_nifti: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    stem: str,
    label_ids: Optional[List[int]] = None,
    decimation_factor: float = 0.5,
    smoothing_factor: float = 0.5,
    mesh_format: str = "stl",
) -> List[Path]:
    """
    최종 라벨맵 NIfTI를 ``labelmap_nifti_to_stl`` 로 라벨별 메시(STL)로 변환합니다.

    파일은 ``out_dir/{stem}_{라벨이름}.{format}`` 로 저장됩니다
    (예: ``case_maxillary_skull.stl``).

    - 라벨 이름: ``MESH_LABEL_FILENAMES`` 매핑 사용.
    - 후처리: 3D Slicer 방식 (DecimatePro 단순화 + WindowedSinc 스무딩).
      스무딩 반복 횟수는 ``smoothing_factor`` 로부터 자동 계산됩니다.
    """
    exported = export_meshes_from_label_nifti(
        labelmap_nifti,
        out_dir,
        formats=(mesh_format,),
        label_ids=label_ids or DEFAULT_MESH_LABEL_IDS,
        mesh_postprocess=True,
        mesh_keep_largest_component=True,
        mesh_decimation_factor=decimation_factor,
        mesh_smoothing_factor=smoothing_factor,
    )

    # ``labelmap_nifti_to_stl`` 는 dataset.json 없으면 ``label_{id}.{fmt}`` 로 저장 →
    # 프로젝트 관례인 ``{stem}_{라벨이름}.{fmt}`` 로 이름 변경.
    out = Path(out_dir).resolve()
    renamed: List[Path] = []
    for f in exported:
        lid = _label_id_from_default_mesh_stem(f.stem)
        name = MESH_LABEL_FILENAMES.get(lid, f.stem) if lid is not None else f.stem
        target = out / f"{stem}_{name}{f.suffix}"
        if target != f:
            f.replace(target)
        renamed.append(target)
    return renamed


def run_local_nifti_pipeline(
    input_volume_nifti: Union[str, Path],
    work_dir: Union[str, Path],
    *,
    model_dir: Union[str, Path],  # 필수
    progress: Optional[Callable[[str], None]] = None,
    nnunet_device: Optional[str] = None,
    nnunet_checkpoint_name: str = "checkpoint_final.pth",
    save_input_volume_nifti: bool = False,
    save_segmentation_labelmap_nifti: bool = True,
    restore_mandibular: bool = True,
    label_cc_config: Optional[LabelCcConfig] = None,
    export_meshes: bool = True,
    mesh_decimation_factor: float = 0.5,
    mesh_smoothing_iterations: int = 15,
    mesh_smoothing_factor: float = 0.5,
    mesh_label_ids: Optional[List[int]] = None,
) -> DentalPipelineResult:
    """
    NIfTI 볼륨부터 최종 결과까지: nnU-Net 추론 → 하악 신경관 복원 → CC 필터링 → 메쉬 생성.

    단일 통합 모델을 사용합니다.

    산출물:
    - ``work_dir/{원본이름}_pred.nii.gz`` — 세그멘테이션 라벨맵 (CC 필터링 적용)
    - ``work_dir/{원본이름}_centerline.json`` — 좌/우 신경관 중심선 월드 좌표 (mm)
    - ``work_dir/{원본이름}_{라벨이름}.stl`` — 세그멘테이션 요소별 STL 메쉬 (export_meshes=True 시)

    중간 스크래치는 temp 폴더에 두고 종료 시 정리됩니다.

    .. note::
        ``mesh_smoothing_iterations`` 는 하위 호환을 위해 유지됩니다. 메쉬 후처리가
        3D Slicer 방식(WindowedSinc)으로 바뀌면서 반복 횟수는 ``mesh_smoothing_factor``
        로부터 자동 계산됩니다.
    """
    from ..postprocess_labelmap import DEFAULT_LABEL_CC_CONFIG

    install_batch_cleanup_handlers()

    inp = Path(input_volume_nifti).resolve()
    user_wd = Path(work_dir).expanduser().resolve()
    user_wd.mkdir(parents=True, exist_ok=True)

    model = Path(model_dir).resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"Not an nnU-Net model directory: {model}")

    tmp_root = Path(tempfile.mkdtemp(prefix="dentaseg_", dir=tempfile.gettempdir()))
    try:
        pred_tmp = tmp_root / "nnunet_pred"
        pred_tmp.mkdir()

        vol_name = inp.name
        low = vol_name.lower()
        if low.endswith(".nii.gz"):
            stem = vol_name[:-7]
        elif low.endswith(".nii"):
            stem = vol_name[:-4]
        else:
            stem = vol_name.rsplit(".", 1)[0]
        pred_basename = f"{stem}_pred.nii.gz"

        def _log(msg: str) -> None:
            if progress:
                progress(msg)

        _log("nnU-Net inference…")

        # ── 1. nnU-Net 추론 (단일 모델) ──
        pred_nii = run_nnunet_predict(
            inp, pred_tmp, model,
            folds=("all",), checkpoint_name=nnunet_checkpoint_name, device=nnunet_device,
        )

        # ── 2. 신경관 복원 ──
        centerline_json_path: Optional[Path] = None
        final_pred = pred_nii
        if restore_mandibular:
            _log(f"Mandibular canal (신경관, label {LABEL_CANAL}) restoration…")
            data, aff, img = load_label(pred_nii)
            sp = spacing_from_affine(aff)
            canal_result = restore_canal(data, aff, sp, RestoreConfig())
            restored = pred_tmp / "canal_restored.nii.gz"
            save_label(restored, canal_result.label_out, aff, img)
            centerline_json_path = pred_tmp / "centerline.json"
            write_centerline_json(canal_result, centerline_json_path)
            final_pred = restored
            _log(
                f"Canal restore: {canal_result.canal_before} → "
                f"{canal_result.canal_after} (+{canal_result.added} voxels)"
            )

        # ── 3. 연결 성분(CC) 기반 라벨맵 후처리 ──
        postprocessed = final_pred
        use_cc = label_cc_config or DEFAULT_LABEL_CC_CONFIG
        if save_segmentation_labelmap_nifti and not use_cc.is_empty():
            _log("Applying connected-component labelmap postprocessing…")
            cc_data, cc_aff, cc_img = load_label(final_pred)
            filtered = filter_labelmap_by_connected_components(cc_data, cfg=use_cc)
            postprocessed = pred_tmp / "cc_postprocessed.nii.gz"
            save_label(postprocessed, filtered, cc_aff, cc_img)

        # ── 4. 산출물 복사 ──
        persisted_input: Optional[Path] = None
        if save_input_volume_nifti:
            dst = user_wd / "input_volume.nii.gz"
            shutil.copy2(inp, dst)
            persisted_input = dst.resolve()

        persisted_pred: Optional[Path] = None
        if save_segmentation_labelmap_nifti:
            dst = user_wd / pred_basename
            shutil.copy2(postprocessed, dst)
            persisted_pred = dst.resolve()

        persisted_centerline: Optional[Path] = None
        if restore_mandibular and centerline_json_path is not None and centerline_json_path.is_file():
            cl_dst = user_wd / f"{stem}_centerline.json"
            shutil.copy2(centerline_json_path, cl_dst)
            persisted_centerline = cl_dst.resolve()

        # ── 5. 메쉬(STL) 익스포트 ──
        mesh_files: List[Path] = []
        if export_meshes and persisted_pred is not None:
            _log("Exporting segmentation meshes (labelmap_nifti_to_stl)…")
            mesh_files = export_final_labelmap_meshes(
                persisted_pred,
                user_wd,  # NIfTI 와 같은 디렉토리에 저장
                stem=stem,  # 파일 접두사 → {stem}_{라벨이름}.stl
                label_ids=mesh_label_ids,
                decimation_factor=mesh_decimation_factor,
                smoothing_factor=mesh_smoothing_factor,
            )
            _log(f"Meshes exported: {len(mesh_files)} files")

        return DentalPipelineResult(
            input_nifti=persisted_input,
            prediction_nifti=persisted_pred,
            mesh_files=mesh_files,
            work_dir=user_wd,
            landmarks_coordinates_json=None,
            landmarks_mrk_json=None,
            centerline_json=persisted_centerline,
        )

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        release_accelerator_memory()


LocalSegmentationMeshResult = DentalPipelineResult

__all__ = [
    "LocalSegmentationMeshResult",
    "DentalPipelineResult",
    "run_local_nifti_pipeline",
]
