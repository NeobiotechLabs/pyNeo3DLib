"""양끝점 추정 오케스트레이터 — 단계 조합만 담당."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from .config import CanalPipelineConfig
from .mesh import MeshGeodesicGraph, ensure_polydata
from .models import CanalEndpointAnalysis, CanalEndpoints, CandidateSet
from .pipeline_steps import (
    AgglomerativeCandidateClusterer,
    build_candidate_extractor,
    ClusterCenterVertexMapper,
    MaxMidpointZGeodesicPairSelector,
    rank_geodesic_pairs,
    select_pair_with_max_midpoint_z,
)
from .protocols import (
    CandidateClusterer,
    CandidateExtractor,
    ClusterRepresentativeMapper,
    EndpointPairSelector,
)


class CanalEndpointFinder:
    """개별 신경관 메쉬 → 양끝점 (의존성 주입으로 단계 교체 가능)."""

    def __init__(
        self,
        config: CanalPipelineConfig,
        candidate_extractor: CandidateExtractor | None = None,
        clusterer: CandidateClusterer | None = None,
        representative_mapper: ClusterRepresentativeMapper | None = None,
        pair_selector: EndpointPairSelector | None = None,
    ) -> None:
        self._config = config
        self._extractor = candidate_extractor or build_candidate_extractor(
            config.candidates
        )
        self._clusterer = clusterer or AgglomerativeCandidateClusterer(config.clustering)
        self._mapper = representative_mapper or ClusterCenterVertexMapper()
        self._pair_selector = pair_selector or MaxMidpointZGeodesicPairSelector(
            config.pair_selection
        )

    def find(self, mesh: pv.PolyData) -> CanalEndpoints:
        return self.analyze(mesh).endpoints

    def analyze(self, mesh: pv.PolyData, *, top_k: int = 3) -> CanalEndpointAnalysis:
        mesh = ensure_polydata(mesh)
        mesh_points = np.asarray(mesh.points, dtype=np.float64)

        candidates = self._extractor.extract(mesh)
        labels = self._clusterer.cluster(candidates.points)
        cluster_vertices = self._mapper.to_vertex_indices(mesh_points, candidates, labels)

        geodesic = MeshGeodesicGraph(mesh)
        vertex_a, vertex_b, geodesic_length = self._pair_selector.select_farthest_pair(
            cluster_vertices, geodesic, mesh_points
        )
        endpoints = CanalEndpoints(
            vertex_a=vertex_a,
            vertex_b=vertex_b,
            point_a=mesh_points[vertex_a].copy(),
            point_b=mesh_points[vertex_b].copy(),
            geodesic_length_mm=geodesic_length,
            n_candidates=int(candidates.points.shape[0]),
            n_clusters=len(cluster_vertices),
        )
        top_pairs = rank_geodesic_pairs(
            cluster_vertices,
            geodesic,
            mesh_points,
            top_k=top_k,
        )
        selected_pair = select_pair_with_max_midpoint_z(
            cluster_vertices,
            geodesic,
            mesh_points,
            top_k=self._config.pair_selection.top_geodesic_pairs,
        )
        return CanalEndpointAnalysis(
            endpoints=endpoints,
            candidate_vertex_indices=candidates.vertex_indices.copy(),
            cluster_vertex_indices=cluster_vertices,
            top_geodesic_pairs=top_pairs,
            selected_pair=selected_pair,
        )
