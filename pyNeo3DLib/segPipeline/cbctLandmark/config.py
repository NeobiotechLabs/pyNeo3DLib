"""추론 기본 설정."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PredictConfig:
    spacing: List[float] = field(default_factory=lambda: [1.0, 0.3])
    agent_fov: List[int] = field(default_factory=lambda: [64, 64, 64])
    speed_per_scale: List[int] = field(default_factory=lambda: [1, 1])
    spawn_radius: int = 10
    focus_radius: int = 4
    network: str = "DNet"
    clear_temp: bool = True
    verbose: bool = True
    strict: bool = False
    """True이면 하나라도 랜드마크 탐색 실패 시 RuntimeError."""

    temp_dir: Optional[str] = None
    output_dir: Optional[str] = None
    save_grouped: bool = False
    """CB/L/U 그룹별 Pred_*.mrk.json 저장 (기본: 저장 안 함)."""
    save_merged: bool = True
    """{케이스}_merged.mrk.json 단일 파일 저장."""
