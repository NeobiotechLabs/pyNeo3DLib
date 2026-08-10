"""
하위 호환용 re-export. 신규 코드는 markups, model_registry, preprocessing 을 직접 import 하세요.
"""

from __future__ import annotations

from .markups import (
    agent_index_zyx_to_lps_xyz,
    gen_control_points,
    load_mrk_landmarks,
    spacing_to_scale_key,
    volume_stem,
    write_mrk_json,
)
from .model_registry import GetBrainForLandmarks, get_brain_for_landmarks
from .preprocessing import (
    CorrectHisto,
    SetSpacing,
    build_patients_dict,
    correct_histo,
    set_spacing,
)

_LEGACY_EXPORTS = {
    "GenControlePoint": ("pyNeo3DLib.cbctLandmark.markups", "gen_control_points"),
    "WriteJson": ("pyNeo3DLib.cbctLandmark.markups", "write_mrk_json"),
}

__all__ = [
    "CorrectHisto",
    "GenControlePoint",
    "GetBrainForLandmarks",
    "SetSpacing",
    "WriteJson",
    "agent_index_zyx_to_lps_xyz",
    "build_patients_dict",
    "correct_histo",
    "gen_control_points",
    "get_brain_for_landmarks",
    "load_mrk_landmarks",
    "set_spacing",
    "spacing_to_scale_key",
    "volume_stem",
    "write_mrk_json",
]


def __getattr__(name: str):
    if name in _LEGACY_EXPORTS:
        from importlib import import_module

        from .compat import deprecate

        mod_name, attr = _LEGACY_EXPORTS[name]
        deprecate(name, f"{mod_name.rsplit('.', 1)[-1]}.{attr}")
        return getattr(import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
