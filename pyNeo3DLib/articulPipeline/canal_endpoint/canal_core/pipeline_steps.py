"""파이프라인 단계별 구현체 (각 클래스 단일 책임)."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from sklearn.cluster import AgglomerativeClustering

from .config import (
    CandidateExtractionConfig,
    CandidateExtractorMode,
    ClusteringConfig,
    EndpointPairSelectionConfig,
)
from .mesh import vertex_normal_divergence
from .models import CandidateSet, GeodesicRepresentativePair
from .protocols import GeodesicDistanceCalculator


class NormalDivergenceCandidateExtractor:
    """표면 법선 divergence |div(n)| 가 큰 정점을 후보로 추출."""

    def __init__(self, config: CandidateExtractionConfig) -> None:
        self._config = config

    def extract(self, mesh: pv.PolyData) -> CandidateSet:
        points = np.asarray(mesh.points, dtype=np.float64)
        divergence = vertex_normal_divergence(mesh)
        magnitude = np.abs(divergence)
        valid = np.isfinite(magnitude)
        if not np.any(valid):
            raise ValueError("법선 divergence를 계산할 수 없습니다.")

        vertex_indices = self._indices_above_percentile(
            magnitude, valid, self._config.normal_divergence_percentile
        )
        if vertex_indices.size < self._config.min_candidates:
            relaxed = self._indices_above_percentile(
                magnitude,
                valid,
                self._config.normal_divergence_percentile_relaxed,
            )
            vertex_indices = np.unique(np.concatenate([vertex_indices, relaxed]))

        if vertex_indices.size < self._config.min_candidates:
            raise ValueError(
                f"법선 divergence 후보가 부족합니다 ({vertex_indices.size}개). "
                "percentile을 낮추거나 메쉬를 확인하세요."
            )

        return CandidateSet(
            points=points[vertex_indices],
            vertex_indices=vertex_indices,
        )

    @staticmethod
    def _indices_above_percentile(
        magnitude: np.ndarray,
        valid: np.ndarray,
        percentile: float,
    ) -> np.ndarray:
        threshold = float(np.nanpercentile(magnitude[valid], percentile))
        return np.where(valid & (magnitude >= threshold))[0]


class CurvatureConvexCandidateExtractor:
    """고곡률(|κ| 상위 percentile) 정점 후보 추출 (레거시)."""

    def __init__(self, config: CandidateExtractionConfig) -> None:
        self._config = config

    def extract(self, mesh: pv.PolyData) -> CandidateSet:
        points = np.asarray(mesh.points, dtype=np.float64)
        curvature = np.abs(
            np.asarray(mesh.curvature(curv_type=self._config.curvature_type), dtype=np.float64)
        )
        valid = np.isfinite(curvature)
        if not np.any(valid):
            raise ValueError("곡률을 계산할 수 없습니다.")

        threshold = float(
            np.nanpercentile(curvature[valid], self._config.curvature_percentile)
        )
        vertex_indices = np.where(valid & (curvature >= threshold))[0]
        return CandidateSet(points=points[vertex_indices], vertex_indices=vertex_indices)


def build_candidate_extractor(
    config: CandidateExtractionConfig,
) -> NormalDivergenceCandidateExtractor | CurvatureConvexCandidateExtractor:
    mode: CandidateExtractorMode = config.mode
    if mode == "normal_divergence":
        return NormalDivergenceCandidateExtractor(config)
    if mode == "curvature":
        return CurvatureConvexCandidateExtractor(config)
    raise ValueError(f"지원하지 않는 후보 추출 모드: {mode!r}")


class AgglomerativeCandidateClusterer:
    """후보 점 공간 클러스터링."""

    def __init__(self, config: ClusteringConfig) -> None:
        self._config = config

    def cluster(self, candidate_points: np.ndarray) -> np.ndarray:
        if candidate_points.shape[0] < 2:
            raise ValueError("클러스터링할 후보 점이 부족합니다.")

        distance_threshold = self._distance_threshold(candidate_points)
        labels = self._fit(candidate_points, distance_threshold)

        if len(np.unique(labels)) < self._config.min_clusters:
            distance_threshold *= self._config.retry_shrink_factor
            labels = self._fit(candidate_points, distance_threshold)

        return labels

    def _distance_threshold(self, points: np.ndarray) -> float:
        bbox_diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        return max(
            bbox_diagonal * self._config.distance_ratio,
            self._config.min_distance_threshold_mm,
        )

    def _fit(self, points: np.ndarray, distance_threshold: float) -> np.ndarray:
        return AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage=self._config.linkage,
        ).fit_predict(points)


class ClusterCenterVertexMapper:
    """cluster center → 가장 가까운 mesh vertex."""

    def to_vertex_indices(
        self,
        mesh_points: np.ndarray,
        candidates: CandidateSet,
        labels: np.ndarray,
    ) -> list[int]:
        representatives: list[int] = []

        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            center = candidates.points[mask].mean(axis=0)
            cluster_vertex_indices = candidates.vertex_indices[mask]
            local_nearest = int(
                np.argmin(
                    np.sum((mesh_points[cluster_vertex_indices] - center) ** 2, axis=1)
                )
            )
            representatives.append(int(cluster_vertex_indices[local_nearest]))

        return representatives


def select_pair_with_max_midpoint_z(
    cluster_vertex_indices: list[int],
    geodesic: GeodesicDistanceCalculator,
    mesh_points: np.ndarray,
    *,
    top_k: int,
) -> GeodesicRepresentativePair:
    """측지 상위 top_k 쌍 후보 중 중점 z(mean_z)가 가장 큰 쌍 1개."""
    candidates = rank_geodesic_pairs(
        cluster_vertex_indices,
        geodesic,
        mesh_points,
        top_k=top_k,
    )
    return max(candidates, key=lambda pair: pair.mean_z)


class MaxMidpointZGeodesicPairSelector:
    """측지 후보 쌍 중 중점 z 최대 쌍 선택."""

    def __init__(self, config: EndpointPairSelectionConfig) -> None:
        self._config = config

    def select_farthest_pair(
        self,
        cluster_vertex_indices: list[int],
        geodesic: GeodesicDistanceCalculator,
        mesh_points: np.ndarray,
    ) -> tuple[int, int, float]:
        selected = select_pair_with_max_midpoint_z(
            cluster_vertex_indices,
            geodesic,
            mesh_points,
            top_k=self._config.top_geodesic_pairs,
        )
        return selected.vertex_a, selected.vertex_b, selected.geodesic_length_mm


class FarthestGeodesicPairSelector:
    """cluster 대표 vertex 중 측지 거리 최대 쌍 선택 (레거시)."""

    def select_farthest_pair(
        self,
        cluster_vertex_indices: list[int],
        geodesic: GeodesicDistanceCalculator,
        mesh_points: np.ndarray,
    ) -> tuple[int, int, float]:
        _ = mesh_points
        if len(cluster_vertex_indices) < 2:
            raise ValueError("측지 거리 비교에 필요한 cluster가 2개 미만입니다.")

        best_a, best_b = cluster_vertex_indices[0], cluster_vertex_indices[1]
        best_distance = -1.0

        n = len(cluster_vertex_indices)
        for i in range(n):
            for j in range(i + 1, n):
                distance = geodesic.distance(cluster_vertex_indices[i], cluster_vertex_indices[j])
                if distance > best_distance:
                    best_distance = distance
                    best_a, best_b = cluster_vertex_indices[i], cluster_vertex_indices[j]

        return best_a, best_b, best_distance


def rank_geodesic_pairs(
    cluster_vertex_indices: list[int],
    geodesic: GeodesicDistanceCalculator,
    mesh_points: np.ndarray,
    *,
    top_k: int = 3,
) -> list[GeodesicRepresentativePair]:
    """대표점 쌍을 측지 거리 내림차순으로 정렬해 상위 top_k개 반환."""
    if len(cluster_vertex_indices) < 2:
        raise ValueError("측지 쌍 순위에 필요한 cluster가 2개 미만입니다.")

    pairs: list[GeodesicRepresentativePair] = []
    representatives = cluster_vertex_indices
    n = len(representatives)
    for i in range(n):
        for j in range(i + 1, n):
            vertex_a = representatives[i]
            vertex_b = representatives[j]
            point_a = mesh_points[vertex_a]
            point_b = mesh_points[vertex_b]
            distance = geodesic.distance(vertex_a, vertex_b)
            pairs.append(
                GeodesicRepresentativePair(
                    rank=0,
                    vertex_a=vertex_a,
                    vertex_b=vertex_b,
                    point_a=point_a.copy(),
                    point_b=point_b.copy(),
                    geodesic_length_mm=distance,
                    mean_y=float(0.5 * (point_a[1] + point_b[1])),
                    mean_z=float(0.5 * (point_a[2] + point_b[2])),
                )
            )

    pairs.sort(key=lambda item: item.geodesic_length_mm, reverse=True)
    limit = max(1, min(top_k, len(pairs)))
    return [
        GeodesicRepresentativePair(
            rank=rank,
            vertex_a=item.vertex_a,
            vertex_b=item.vertex_b,
            point_a=item.point_a,
            point_b=item.point_b,
            geodesic_length_mm=item.geodesic_length_mm,
            mean_y=item.mean_y,
            mean_z=item.mean_z,
        )
        for rank, item in enumerate(pairs[:limit], start=1)
    ]
