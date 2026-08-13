"""추론 파이프라인 상수 및 디바이스 해석."""
from __future__ import annotations

import numpy as np
import torch

from .landmarks import GROUP_LABELS, LABELS, LABEL_GROUPES  # noqa: F401

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    if device is None:
        return DEFAULT_DEVICE
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))
