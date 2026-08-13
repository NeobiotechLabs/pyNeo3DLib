"""
CBCT 볼륨에서 ALI 에이전트 기반으로 랜드마크 좌표를 추론하는 핵심 파이프라인.
"""
from __future__ import annotations

import logging
import os
import shutil
from typing import Dict, List, Optional, Union

import torch

from .config import PredictConfig
from .inference import run_predict_pipeline
from .markups import spacing_to_scale_key, volume_output_dir_name, volume_stem
from .model_registry import get_brain_for_landmarks, validate_models
from .preprocessing import build_patients_dict
from . import constants as GV
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

    temp_root = cfg.temp_dir or os.path.join(os.path.dirname(vol), ".ali_temp")
    temp_root = os.path.abspath(temp_root)
    temp_fold = os.path.join(temp_root, "temp")
    os.makedirs(temp_fold, exist_ok=True)

    brain_dic, _ = get_brain_for_landmarks(models_root, landmarks, scale_keys)
    validate_models(brain_dic, landmarks, scale_keys)

    patients = build_patients_dict(vol, cfg.spacing, scale_keys, temp_fold)

    stem = volume_stem(vol)
    base_out = os.path.abspath(cfg.output_dir) if cfg.output_dir else os.path.dirname(vol)
    out_path = os.path.join(base_out, volume_output_dir_name(vol))
    os.makedirs(out_path, exist_ok=True)

    predict_result = run_predict_pipeline(
        patients,
        landmarks,
        scale_keys,
        brain_dic,
        cfg,
        resolved_device,
        out_path,
    )

    if cfg.clear_temp:
        try:
            shutil.rmtree(temp_fold)
        except OSError as e:
            logger.warning("임시 폴더 삭제 실패: %s — %s", temp_fold, e.strerror)

    if return_details:
        return predict_result
    return predict_result.as_dict()
