"""하위 호환: constants 로 이전됨."""

from __future__ import annotations

import warnings

from .constants import (  # noqa: F401
    DEFAULT_DEVICE,
    GROUP_LABELS,
    LABEL_GROUPES,
    LABELS,
    MOVEMENT_ID_6,
    MOVEMENT_MATRIX_6,
    MOVEMENTS,
    SCALE_KEYS,
    resolve_device,
)

warnings.warn(
    "pyNeo3DLib.cbctLandmark.global_var is deprecated; "
    "import from pyNeo3DLib.cbctLandmark.constants instead.",
    DeprecationWarning,
    stacklevel=2,
)

# 레거시: mutable DEVICE 별칭 (읽기 전용으로 유지)
DEVICE = DEFAULT_DEVICE
