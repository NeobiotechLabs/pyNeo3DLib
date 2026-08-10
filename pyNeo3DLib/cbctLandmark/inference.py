"""에이전트·환경 생성 및 추론 루프."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

import numpy as np
import torch

from . import constants as GV
from .agents import Agent
from .config import PredictConfig
from .environment import Environment
from .models import Brain, DNet, RNet
from .markups import agent_index_zyx_to_lps_xyz
from .types import LandmarkCoord, PredictResult

logger = logging.getLogger(__name__)

NETWORK_MAP = {"DNET": DNet, "RNET": RNet}


def resolve_network_type(network: str) -> Type:
    net_name = str(network).upper()
    if net_name not in NETWORK_MAP:
        raise ValueError(f'network 는 "DNet" 또는 "RNet" 이어야 합니다. got: {network}')
    return NETWORK_MAP[net_name]


def build_agents(
    landmarks: List[str],
    scale_keys: List[str],
    agent_fov: List[int],
    spawn_radius: int,
    speed_per_scale: List[int],
    focus_radius: int,
    verbose: bool,
) -> List[Agent]:
    agent_lst = []
    for label in landmarks:
        if verbose:
            logger.info("Generating agent for landmark: %s", label)
        agent_lst.append(
            Agent(
                targeted_landmark=label,
                movements=GV.MOVEMENTS,
                scale_keys=scale_keys,
                FOV=agent_fov,
                start_pos_radius=spawn_radius,
                speed_per_scale=speed_per_scale,
                focus_radius=focus_radius,
                verbose=verbose,
            )
        )
    logger.info("%d agent(s) ready", len(agent_lst))
    return agent_lst


def build_environments(
    patient_dic: dict,
    padding: np.ndarray,
    device: torch.device,
    verbose: bool,
) -> List[Environment]:
    env_lst = []
    for patient, data in patient_dic.items():
        if verbose:
            logger.info("Loading environment for patient: %s", patient)
        env = Environment(
            patient_id=patient,
            device=device,
            padding=padding,
            verbose=False,
        )
        env.LoadImages(data["scans"])
        env_lst.append(env)
    return env_lst


def attach_brains(
    agent_lst: List[Agent],
    brain_dic: dict,
    network_type: Type,
    scale_keys: List[str],
    device: torch.device,
) -> None:
    out_channels = len(GV.MOVEMENTS["id"])
    in_channels = 1024
    for agent in agent_lst:
        brain = Brain(
            network_type=network_type,
            network_scales=scale_keys,
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        brain.LoadModels(brain_dic[agent.target])
        agent.SetBrain(brain)


def run_inference(
    env_lst: List[Environment],
    agent_lst: List[Agent],
    landmarks: List[str],
    scale_keys: List[str],
) -> PredictResult:
    result = PredictResult()
    sk_final = scale_keys[-1]

    for environment in env_lst:
        for agent in agent_lst:
            agent.SetEnvironment(environment)
            steps = agent.Search()
            if steps == -1:
                result.failed.append(agent.target)
                logger.warning("Landmark search failed: %s", agent.target)

        ref = environment.get_reference_image(sk_final)
        for lm in landmarks:
            if lm in environment.predicted_landmarks:
                pos = environment.predicted_landmarks[lm]
                x, y, z = agent_index_zyx_to_lps_xyz(ref, pos)
                result.landmarks[lm] = LandmarkCoord(x=x, y=y, z=z)

    return result


def run_predict_pipeline(
    patients: dict,
    landmarks: List[str],
    scale_keys: List[str],
    brain_dic: dict,
    cfg: PredictConfig,
    device: torch.device,
    out_path: str,
) -> PredictResult:
    padding = np.array(cfg.agent_fov) / 2 + 1
    network_type = resolve_network_type(cfg.network)

    env_lst = build_environments(patients, padding, device, cfg.verbose)
    agent_lst = build_agents(
        landmarks,
        scale_keys,
        cfg.agent_fov,
        cfg.spawn_radius,
        cfg.speed_per_scale,
        cfg.focus_radius,
        cfg.verbose,
    )
    attach_brains(agent_lst, brain_dic, network_type, scale_keys, device)

    predict_result = run_inference(env_lst, agent_lst, landmarks, scale_keys)

    sk_final = scale_keys[-1]
    for environment in env_lst:
        environment.SavePredictedLandmarks(
            sk_final,
            out_path,
            save_grouped=cfg.save_grouped,
            save_merged=cfg.save_merged,
        )

    if cfg.strict and predict_result.failed:
        raise RuntimeError(
            f"랜드마크 탐색 실패: {', '.join(predict_result.failed)}"
        )

    return predict_result
