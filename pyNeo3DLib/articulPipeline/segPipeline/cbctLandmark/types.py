"""추론 설정·결과 타입."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class LandmarkCoord:
    x: float
    y: float
    z: float

    def as_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class PredictResult:
    """predict() 상세 반환값."""

    landmarks: Dict[str, LandmarkCoord] = field(default_factory=dict)
    failed: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {name: coord.as_dict() for name, coord in self.landmarks.items()}
