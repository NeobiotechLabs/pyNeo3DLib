"""파이프라인 파라미터 — 단계별 dataclass로 그룹화."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class InputConfig:
    """입력 STL 경로."""

    canal_stl_path: Path
    expected_connected_components: int = 2

    def resolve_canal_stl(self) -> Path:
        return self.canal_stl_path.expanduser().resolve()


CandidateExtractorMode = Literal["normal_divergence", "curvature"]


@dataclass(frozen=True)
class CandidateExtractionConfig:
    """1단계: MeF 후보 정점 추출."""

    mode: CandidateExtractorMode = "normal_divergence"
    normal_divergence_percentile: float = 90.0
    normal_divergence_percentile_relaxed: float = 75.0
    min_candidates: int = 3
    curvature_percentile: float = 90.0
    curvature_type: str = "mean"


@dataclass(frozen=True)
class ClusteringConfig:
    """2단계: 후보 점 클러스터링."""

    distance_ratio: float = 0.06
    min_clusters: int = 3
    retry_shrink_factor: float = 0.65
    linkage: str = "average"
    min_distance_threshold_mm: float = 1e-3


@dataclass(frozen=True)
class EndpointPairSelectionConfig:
    """측지 상위 후보 쌍 중 중점 z(mean_z) 최대 쌍 선택."""

    top_geodesic_pairs: int = 3


@dataclass(frozen=True)
class VisualizationConfig:
    """PyVista 시각화."""

    window_size: tuple[int, int] = (1200, 900)
    background: str = "white"
    canal_opacity: float = 0.85
    right_canal_color: str = "red"
    left_canal_color: str = "blue"
    endpoint_colors: tuple[str, str] = ("yellow", "lime")
    endpoint_sphere_radius_mm: float = 1.25
    endpoint_sphere_resolution: int = 24


@dataclass(frozen=True)
class CanalPipelineConfig:
    """전체 파이프라인 설정."""

    input: InputConfig
    candidates: CandidateExtractionConfig = field(default_factory=CandidateExtractionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    pair_selection: EndpointPairSelectionConfig = field(
        default_factory=EndpointPairSelectionConfig
    )
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
