"""결과/중간산출물 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import networkx as nx
import numpy as np


@dataclass
class SideStats:
    side: str
    n_components_before: int = 0
    n_components_after: int = 0
    n_skeleton_voxels: int = 0
    n_endpoints: int = 0
    n_bridges: int = 0
    path_length_mm: float = 0.0
    path_n_points: int = 0
    radius_mm: float = 0.0
    restored_voxels: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SideArtifacts:
    """한 쪽(L/R) 파이프라인 중간 결과 — 시각화·디버깅용."""

    side: str
    offset: np.ndarray  # crop → full ijk
    crop_shape: tuple[int, ...]
    raw: np.ndarray
    after_majority: np.ndarray
    after_small_cc: np.ndarray
    skeleton: np.ndarray
    graph: nx.Graph | None = None
    endpoints: list[int] = field(default_factory=list)
    n_bridges: int = 0
    path_nodes: list[int] = field(default_factory=list)
    path_length_mm: float = 0.0
    controls_ijk: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    controls_ma_ijk: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    dense_ijk: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    restored_crop: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    restored_full: np.ndarray | None = None
    stats: SideStats | None = None


@dataclass
class PipelineResult:
    label_out: np.ndarray
    affine: np.ndarray
    spacing: np.ndarray
    split_meta: dict
    left: SideArtifacts | None
    right: SideArtifacts | None
    canal_before: int
    canal_after: int
    added: int
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "split": self.split_meta,
            "left": self.left.stats.to_dict() if self.left and self.left.stats else None,
            "right": self.right.stats.to_dict() if self.right and self.right.stats else None,
            "canal_voxels_before": self.canal_before,
            "canal_voxels_after": self.canal_after,
            "added_voxels": self.added,
            **self.meta,
        }
