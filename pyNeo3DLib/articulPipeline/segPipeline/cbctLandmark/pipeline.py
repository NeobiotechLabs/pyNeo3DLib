"""
CBCT 볼륨에서 ALI 에이전트 기반으로 랜드마크 좌표를 추론하는 핵심 파이프라인.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from typing import Dict, List, Optional, Union

import torch

from .config import PredictConfig
from .inference import run_predict_pipeline
from .markups import spacing_to_scale_key, volume_output_dir_name, volume_stem
from .model_registry import get_brain_for_landmarks, validate_models
from .preprocessing import build_patients_dict
from . import constants as GV
from . import windows_path_compat as wpc
from .types import PredictResult

logger = logging.getLogger(__name__)


def predict(
    volume: str,
    models_dir: str,
    landmarks: List[str],
    output_dir: Optional[str] = None,
    spacing: Optional[List[float]] = None,
    agent_fov: Optional[List[int]] = None,
    speed_per_scale: Optional[List[int]] = None,
    spawn_radius: int = 10,
    focus_radius: int = 4,
    network: str = "DNet",
    temp_dir: Optional[str] = None,
    clear_temp: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
    strict: bool = False,
    return_details: bool = False,
    save_grouped: bool = False,
    save_merged: bool = True,
) -> Union[Dict[str, Dict[str, float]], PredictResult]:
    """
    CBCT 볼륨에서 랜드마크 좌표를 추론합니다.

    Parameters
    ----------
    return_details : bool
        True이면 PredictResult(성공 좌표 + failed 목록) 반환.
        False(기본)이면 기존과 동일한 dict 반환.

    Returns
    -------
    dict or PredictResult
        LPS 물리 좌표 (mm). 실패한 랜드마크는 dict 모드에서 제외되며,
        return_details=True 이면 ``failed`` 리스트에 포함됩니다.
    """
    cfg = PredictConfig(
        spacing=spacing if spacing is not None else [1.0, 0.3],
        agent_fov=agent_fov if agent_fov is not None else [64, 64, 64],
        speed_per_scale=speed_per_scale if speed_per_scale is not None else [1, 1],
        spawn_radius=spawn_radius,
        focus_radius=focus_radius,
        network=network,
        temp_dir=temp_dir,
        clear_temp=clear_temp,
        verbose=verbose,
        strict=strict,
        output_dir=output_dir,
        save_grouped=save_grouped,
        save_merged=save_merged,
    )

    vol = os.path.abspath(volume)
    if not os.path.isfile(vol):
        raise FileNotFoundError(f"입력 볼륨을 찾을 수 없습니다: {vol}")

    models_root = os.path.abspath(models_dir)
    if not os.path.isdir(models_root):
        raise FileNotFoundError(f"모델 디렉터리를 찾을 수 없습니다: {models_root}")

    if len(cfg.speed_per_scale) != len(cfg.spacing):
        raise ValueError(
            f"speed_per_scale 길이({len(cfg.speed_per_scale)})가 "
            f"spacing 길이({len(cfg.spacing)})와 같아야 합니다."
        )

    resolved_device = GV.resolve_device(device)
    scale_keys = [spacing_to_scale_key(s) for s in cfg.spacing]

    # ── Windows 비ASCII 경로 우회 ───────────────────────────────────────
    # ITK/SimpleITK 는 Windows 에서 경로에 비ASCII(한글 등) 문자가 있으면
    # 파일을 열지 못한다. 볼륨 경로 또는 임시 폴더 경로가 비ASCII면 볼륨을
    # ASCII 임시 위치로 복사해 처리한다. 출력(mrk.json)은 아래 out_path
    # 계산이 원본 ``vol`` 이름 기준이므로 원본 케이스명을 그대로 유지한다.
    staging_root: Optional[str] = None
    effective_vol = vol
    temp_base = cfg.temp_dir or os.path.join(os.path.dirname(vol), ".ali_temp")
    if wpc.native_io_staging_needed() and (
        not wpc.path_is_pure_ascii(vol) or not wpc.path_is_pure_ascii(temp_base)
    ):
        staging_root = wpc.create_ascii_staging_root(prefix="cbct_lm_")
        staged_vol = os.path.join(staging_root, wpc.ascii_volume_name(vol))
        shutil.copy2(vol, staged_vol)
        effective_vol = staged_vol
        if not cfg.temp_dir or not wpc.path_is_pure_ascii(cfg.temp_dir):
            temp_base = os.path.join(staging_root, ".ali_temp")
        print(
            f"참고: 경로에 비ASCII 문자가 있어 ASCII 임시 폴더에서 처리합니다 "
            f"({staging_root})",
            flush=True,
        )

    temp_root = os.path.abspath(temp_base)
    temp_fold = os.path.join(temp_root, "temp")
    os.makedirs(temp_fold, exist_ok=True)

    brain_dic, _ = get_brain_for_landmarks(models_root, landmarks, scale_keys)
    validate_models(brain_dic, landmarks, scale_keys)

    patients = build_patients_dict(
        effective_vol,
        cfg.spacing,
        scale_keys,
        temp_fold,
        patient_key=os.path.basename(vol),
    )

    stem = volume_stem(vol)
    base_out = os.path.abspath(cfg.output_dir) if cfg.output_dir else os.path.dirname(vol)
    out_path = os.path.join(base_out, volume_output_dir_name(vol))
    os.makedirs(out_path, exist_ok=True)

    try:
        predict_result = run_predict_pipeline(
            patients,
            landmarks,
            scale_keys,
            brain_dic,
            cfg,
            resolved_device,
            out_path,
        )
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)

    # 스테이징 루트 정리가 이미 temp_fold 까지 삭제한 경우 isdir 로 걸러냄
    if cfg.clear_temp and os.path.isdir(temp_fold):
        try:
            shutil.rmtree(temp_fold)
        except OSError as e:
            logger.warning("임시 폴더 삭제 실패: %s — %s", temp_fold, e.strerror)

    if return_details:
        return predict_result
    return predict_result.as_dict()
