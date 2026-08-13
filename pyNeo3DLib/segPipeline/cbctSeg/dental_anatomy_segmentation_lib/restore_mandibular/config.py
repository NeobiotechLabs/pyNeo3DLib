"""복원 파이프라인 설정 (한곳에서 파라미터 관리)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


LABEL_CANAL = 3


@dataclass
class RestoreConfig:
    majority_size: int = 3
    small_ratio: float = 0.005
    min_voxels: int = 80
    max_gap_mm: float = 15.0
    skeleton_sample_mm: float = 2.0
    gap_sample_mm: float = 2.5
    ma_window: int = 3
    resample_mm: float = 0.5
    radius_mm: float | None = None
    keep_original_canal: bool = True
    # viz
    viz: bool = False
    viz_side: str = "both"  # L | R | both
    viz_step_size: int = 2
    # I/O
    output_dir: Path | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.output_dir is not None:
            d["output_dir"] = str(self.output_dir)
        return d
