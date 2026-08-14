"""추론 파이프라인 상수 및 디바이스 해석."""
from __future__ import annotations

import numpy as np
import torch

from .landmarks import GROUP_LABELS, LABELS, LABEL_GROUPES  # noqa: F401

#: 추론은 GPU 전용입니다. CUDA 미설치 시 조용한 CPU 폴백 대신 즉시 실패합니다.
DEFAULT_DEVICE = torch.device("cuda")

CUDA_INSTALL_HINT = (
    "CUDA 지원 torch 가 설치되어 있지 않습니다 (CPU 전용 빌드 감지).\n"
    "GPU 버전으로 재설치하세요:\n"
    "    pip uninstall -y torch torchvision\n"
    "    pip install torch==2.11.0 torchvision "
    "--index-url https://download.pytorch.org/whl/cu128"
)


def require_cuda() -> torch.device:
    """CUDA 디바이스를 반환. CUDA 미설치 시 안내 메시지와 함께 RuntimeError 발생."""
    if not torch.cuda.is_available():
        raise RuntimeError(CUDA_INSTALL_HINT)
    return torch.device("cuda")

SCALE_KEYS = ["1", "0-3"]

MOVEMENT_MATRIX_6 = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
)
MOVEMENT_ID_6 = ["Up", "Down", "Back", "Front", "Left", "Right"]
MOVEMENTS = {"id": MOVEMENT_ID_6, "mat": MOVEMENT_MATRIX_6}


def resolve_device(device=None) -> torch.device:
    """추론 디바이스 해석. 기본(None)·cuda 지정 시 CUDA 가용이 필수이며,
    미설치 상태에서는 CPU 로 폴백하지 않고 즉시 RuntimeError 를 발생시킵니다.
    명시적 ``device="cpu"`` 만 개발자 의도 지정으로 허용합니다.
    """
    if device is None:
        return require_cuda()
    dev = device if isinstance(device, torch.device) else torch.device(str(device))
    if dev.type == "cuda":
        return require_cuda()
    return dev
