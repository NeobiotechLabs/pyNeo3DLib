"""학습 가중치(.pth) 탐색 및 검증."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def validate_models(brain_dic: dict, landmarks: list, scale_keys: list) -> None:
    missing = []
    for lm in landmarks:
        if lm not in brain_dic:
            missing.append(f"랜드마크 '{lm}': 모델 폴더 없음")
            continue
        for sk in scale_keys:
            if sk not in brain_dic[lm]:
                missing.append(f"랜드마크 '{lm}': 스케일 '{sk}' 가중치 없음")
    if missing:
        raise FileNotFoundError(
            "필요한 가중치가 없습니다:\n  - " + "\n  - ".join(missing)
        )


def get_brain_for_landmarks(
    models_root: str,
    landmarks: list,
    scale_keys: list,
) -> tuple[dict, dict]:
    """
    models_root/<팩>/<랜드마크>/<스케일>/*.pth 스캔.
    Returns: (brain_dic, chosen_pack_by_landmark)
    """
    root = os.path.abspath(models_root)
    packs = [
        d
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    ]
    brain_dic = {}
    chosen_pack_by_landmark = {}

    def scales_in_pack(pack: str, lm: str):
        scales = {}
        for sk in scale_keys:
            scale_dir = os.path.join(root, pack, lm, sk)
            if not os.path.isdir(scale_dir):
                return None
            pths = sorted(
                os.path.join(scale_dir, f)
                for f in os.listdir(scale_dir)
                if f.endswith(".pth")
            )
            if not pths:
                return None
            if len(pths) > 1:
                logger.warning("여러 .pth 중 첫 번째 사용: %s", pths[0])
            scales[sk] = pths[0]
        return scales

    def scales_flat(lm: str):
        scales = {}
        for sk in scale_keys:
            scale_dir = os.path.join(root, lm, sk)
            if not os.path.isdir(scale_dir):
                return None
            pths = sorted(
                os.path.join(scale_dir, f)
                for f in os.listdir(scale_dir)
                if f.endswith(".pth")
            )
            if not pths:
                return None
            scales[sk] = pths[0]
        return scales

    for lm in landmarks:
        candidates = []
        for pack in packs:
            s = scales_in_pack(pack, lm)
            if s is not None:
                candidates.append((pack, s))

        if not candidates:
            flat = scales_flat(lm)
            if flat is not None:
                brain_dic[lm] = flat
                chosen_pack_by_landmark[lm] = "(models 직하위)"
                continue
            raise FileNotFoundError(
                f"랜드마크 '{lm}': 사용 가능한 가중치 없음. "
                f"팩 후보: {packs} 또는 {os.path.join(root, lm, scale_keys[0])} 형태를 확인하세요."
            )

        if len(candidates) == 1:
            pack, scales = candidates[0]
        else:
            candidates.sort(key=lambda x: x[0])
            pack, scales = candidates[0]
            alt = [c[0] for c in candidates]
            logger.warning(
                "랜드마크 '%s': 가중치가 여러 팩에 존재 %s. 선택: '%s'.",
                lm,
                alt,
                pack,
            )

        brain_dic[lm] = scales
        chosen_pack_by_landmark[lm] = pack

    return brain_dic, chosen_pack_by_landmark


GetBrainForLandmarks = get_brain_for_landmarks
