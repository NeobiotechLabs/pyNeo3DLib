"""
학습 결과 폴더(모델 루트/fold_0/checkpoint_final.pth 등)로 nnU-Net v2 추론.

Windows에서 입력 NIfTI·출력 폴더·모델 폴더 경로에 비ASCII가 있으면
ASCII 임시 경로로 복사한 뒤 추론합니다.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union

from ..pipeline_runtime import nnunet_sequential_inference_enabled, release_accelerator_memory
from ..windows_native_path_staging import (
    ascii_system_temp_parent,
    cleanup_staging_roots,
    native_io_staging_enabled,
    path_string_is_pure_ascii,
    stage_directory_copy,
    stage_file_copy,
)


def _ensure_nnunet_env_path_vars() -> None:
    """
    nnunetv2.paths 가 import 될 때 ``nnUNet_*`` 가 비어 있으면 경고를 출력합니다.
    로컬 체크포인트 추론만 쓸 때는 빈 플레이스홀더 디렉터리로 채워 두면 조용해지며,
    ``predict_from_files`` + ``initialize_from_trained_model_folder`` 동작에는 영향 없습니다.
    """
    keys = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
    if all(os.environ.get(k) for k in keys):
        return
    root = Path(tempfile.gettempdir()) / "dentalseg_nnunet_env"
    for k in keys:
        if not os.environ.get(k):
            d = root / k
            d.mkdir(parents=True, exist_ok=True)
            os.environ[k] = str(d.resolve())


def _nnunet_upstream_console_verbose() -> bool:
    """환경 변수 ``DENTAL_NNUNET_VERBOSE=1`` 이면 nnU-Net 원본 ``print``/경고를 그대로 둡니다."""
    return os.environ.get("DENTAL_NNUNET_VERBOSE", "").strip().lower() in ("1", "true", "yes")


def run_nnunet_predict(
    input_nifti: Union[str, Path],
    output_dir: Union[str, Path],
    model_training_dir: Union[str, Path],
    *,
    folds: Sequence[Union[int, str]] | Literal["all"] = (0,),
    checkpoint_name: str = "checkpoint_final.pth",
    device: Optional[str] = None,
    num_processes_preprocessing: int = 1,
    num_processes_segmentation_export: int = 1,
) -> Path:
    """
    단일 NIfTI 입력에 대해 세그멘테이션 NIfTI를 생성합니다.

    :param input_nifti: 입력 CT 볼륨 (.nii / .nii.gz)
    :param output_dir: 예측 저장 폴더 (생성됨)
    :param model_training_dir: ``dataset.json`` + ``plans.json`` + ``fold_*`` 가 있는 nnU-Net 결과 루트
    :returns: 생성된 세그멘테이션 NIfTI 경로 (첫 번째 매칭)

    기본적으로 nnU-Net 패키지가 뿌리는 잡다한 ``print``·구형 plans ``UserWarning`` 은 숨깁니다.
    디버깅 시 환경 변수 ``DENTAL_NNUNET_VERBOSE=1`` 을 켜면 원본 로그가 나갑니다.

    spawn 고아 방지: 기본은 단일 프로세스 추론(``predict_from_files_sequential``).
    예전처럼 spawn을 쓰려면 ``DENTAL_NNUNET_SEQUENTIAL=0`` 을 설정하세요.
    """
    inp = Path(input_nifti).resolve()
    out = Path(output_dir).resolve()
    model_dir = Path(model_training_dir).resolve()

    if not inp.is_file():
        raise FileNotFoundError(f"Input NIfTI not found: {inp}")
    if not (model_dir / "dataset.json").is_file():
        raise FileNotFoundError(f"No dataset.json in model directory: {model_dir}")

    cleanups: List[Optional[Path]] = []

    inp_st = stage_file_copy(inp, prefix="nnunet_in_", label="input NIfTI")
    if inp_st.cleanup_root is not None:
        cleanups.append(inp_st.cleanup_root)
    inp_eff = inp_st.effective_path

    m_st = stage_directory_copy(
        model_dir,
        prefix="nnunet_mdl_",
        label="nnU-Net model",
    )
    if m_st.cleanup_root is not None:
        cleanups.append(m_st.cleanup_root)
    model_eff = m_st.effective_path

    if not (model_eff / "dataset.json").is_file():
        cleanup_staging_roots(cleanups)
        raise FileNotFoundError(f"No dataset.json in model directory: {model_eff}")

    out_eff = out
    out_tmp: Optional[Path] = None
    if native_io_staging_enabled() and not path_string_is_pure_ascii(out):
        out_tmp = Path(
            tempfile.mkdtemp(prefix="nnunet_out_", dir=str(ascii_system_temp_parent()))
        )
        out_eff = out_tmp
        cleanups.append(out_tmp)
    else:
        out.mkdir(parents=True, exist_ok=True)

    try:
        return _run_nnunet_predict_core(
            inp_eff,
            out_eff,
            model_eff,
            out_user_root=out,
            out_was_staged=out_tmp is not None,
            folds=folds,
            checkpoint_name=checkpoint_name,
            device=device,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
        )
    finally:
        cleanup_staging_roots(cleanups)


def _run_nnunet_predict_core(
    inp_eff: Path,
    out_eff: Path,
    model_eff: Path,
    *,
    out_user_root: Path,
    out_was_staged: bool,
    folds: Sequence[Union[int, str]],
    checkpoint_name: str,
    device: Optional[str],
    num_processes_preprocessing: int,
    num_processes_segmentation_export: int,
) -> Path:
    _ensure_nnunet_env_path_vars()
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # 추론은 GPU 전용: CUDA 미설치 시 조용한 CPU 폴백 없이 즉시 실패.
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 지원 torch 가 설치되어 있지 않습니다 (CPU 전용 빌드 감지).\n"
                "GPU 버전으로 재설치하세요:\n"
                "    pip uninstall -y torch torchvision\n"
                "    pip install torch==2.11.0 torchvision "
                "--index-url https://download.pytorch.org/whl/cu128"
            )
        dev = torch.device("cuda")
    else:
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 디바이스가 요청되었지만 torch.cuda.is_available() 이 False 입니다. "
                "CUDA 지원 torch 빌드를 설치하세요."
            )

    perform_on_device = dev.type == "cuda"

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=perform_on_device,
        device=dev,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=not _nnunet_upstream_console_verbose(),
    )

    try:
        def _init_and_predict() -> None:
            # folds 가 "all" 인 경우 특수 처리
            folds_to_use = folds
            if isinstance(folds, str) and folds.lower() == "all":
                folds_to_use = ("all",)

            predictor.initialize_from_trained_model_folder(
                str(model_eff),
                use_folds=folds_to_use,
                checkpoint_name=checkpoint_name,
            )
            list_of_lists: List[List[str]] = [[str(inp_eff)]]
            if nnunet_sequential_inference_enabled():
                predictor.predict_from_files_sequential(
                    list_of_lists,
                    str(out_eff),
                    save_probabilities=False,
                    overwrite=True,
                )
            else:
                predictor.predict_from_files(
                    list_of_lists,
                    str(out_eff),
                    save_probabilities=False,
                    overwrite=True,
                    num_processes_preprocessing=num_processes_preprocessing,
                    num_processes_segmentation_export=num_processes_segmentation_export,
                )

        # nnU-Net upstream은 verbose=False 여도 print·UserWarning 을 씁니다.
        if _nnunet_upstream_console_verbose():
            _init_and_predict()
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    module=r"nnunetv2\.utilities\.plans_handling\.plans_handler",
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    _init_and_predict()

        pred_files = sorted(out_eff.rglob("*.nii.gz"))
        if not pred_files:
            pred_files = sorted(out_eff.rglob("*.nii"))
        if not pred_files:
            raise RuntimeError(f"No output NIfTI found after nnU-Net prediction: {out_eff}")

        pred = pred_files[0].resolve()

        if out_was_staged:
            out_user_root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(out_eff, out_user_root, dirs_exist_ok=True)
            pred = (out_user_root / pred.relative_to(out_eff)).resolve()

        return pred
    finally:
        del predictor
        release_accelerator_memory()


__all__ = ["run_nnunet_predict"]
