"""파이프라인 기본 설정값."""

from .config import (
    CandidateExtractionConfig,
    ClusteringConfig,
    EndpointPairSelectionConfig,
    VisualizationConfig,
)

DEFAULT_CANDIDATES = CandidateExtractionConfig(
    mode="normal_divergence",
    normal_divergence_percentile=90.0,
    normal_divergence_percentile_relaxed=75.0,
    curvature_percentile=90.0,
    curvature_type="mean",
)
DEFAULT_PAIR_SELECTION = EndpointPairSelectionConfig(top_geodesic_pairs=3)
DEFAULT_CLUSTERING = ClusteringConfig(
    distance_ratio=0.06,
    min_clusters=3,
    retry_shrink_factor=0.65,
    linkage="average",
    min_distance_threshold_mm=1e-3,
)
DEFAULT_VISUALIZATION = VisualizationConfig(
    window_size=(1200, 900),
    endpoint_sphere_radius_mm=1.25,
)
