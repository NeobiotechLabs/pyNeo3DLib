"""레거시 API 이름 deprecation."""
from __future__ import annotations

import warnings


def deprecate(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated; use {new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
