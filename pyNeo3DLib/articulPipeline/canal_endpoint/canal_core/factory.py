"""CanalPipelineConfig 생성 (OCP: 파라미터 조합만 담당)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import (
    CanalPipelineConfig,
    CandidateExtractorMode,
    InputConfig,
    VisualizationConfig,
)
from .defaults import (
    DEFAULT_CANDIDATES,
    DEFAULT_CLUSTERING,
    DEFAULT_PAIR_SELECTION,
    DEFAULT_VISUALIZATION,
)


class CanalPipelineConfigFactory:
    @staticmethod
    def build(
        canal_stl_path: Path,
        *,
        candidate_extractor: CandidateExtractorMode = DEFAULT_CANDIDATES.mode,
        normal_divergence_percentile: float = DEFAULT_CANDIDATES.normal_divergence_percentile,
        curvature_percentile: float = DEFAULT_CANDIDATES.curvature_percentile,
        cluster_distance_ratio: float = DEFAULT_CLUSTERING.distance_ratio,
        top_geodesic_pairs: int = DEFAULT_PAIR_SELECTION.top_geodesic_pairs,
        expected_components: int = 2,
        visualization: VisualizationConfig | None = None,
    ) -> CanalPipelineConfig:
        return CanalPipelineConfig(
            input=InputConfig(
                canal_stl_path=canal_stl_path.expanduser().resolve(),
                expected_connected_components=expected_components,
            ),
            candidates=replace(
                DEFAULT_CANDIDATES,
                mode=candidate_extractor,
                normal_divergence_percentile=normal_divergence_percentile,
                curvature_percentile=curvature_percentile,
            ),
            clustering=replace(
                DEFAULT_CLUSTERING,
                distance_ratio=cluster_distance_ratio,
            ),
            pair_selection=replace(
                DEFAULT_PAIR_SELECTION,
                top_geodesic_pairs=top_geodesic_pairs,
            ),
            visualization=visualization or DEFAULT_VISUALIZATION,
        )
