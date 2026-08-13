"""도메인 모델."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv


@dataclass(frozen=True)
class CandidateSet:
    """후보 정점 집합."""

    points: np.ndarray
    vertex_indices: np.ndarray


@dataclass(frozen=True)
class CanalEndpoint:
    """신경관 단일 끝점."""

    vertex: int
    point: np.ndarray


@dataclass(frozen=True)
class GeodesicRepresentativePair:
    """클러스터 대표점 쌍 + 측지 거리 (순위 포함)."""

    rank: int
    vertex_a: int
    vertex_b: int
    point_a: np.ndarray
    point_b: np.ndarray
    geodesic_length_mm: float
    mean_y: float
    mean_z: float


@dataclass(frozen=True)
class CanalEndpointAnalysis:
    """한쪽 신경관 파이프라인 중간·최종 결과 (시각화·디버그용)."""

    endpoints: CanalEndpoints
    candidate_vertex_indices: np.ndarray
    cluster_vertex_indices: list[int]
    top_geodesic_pairs: list[GeodesicRepresentativePair]
    selected_pair: GeodesicRepresentativePair

    def mef_endpoint(self) -> CanalEndpoint:
        """선정 쌍(z̄ 최대)에서 y 최소 → MeF."""
        return self.endpoints.mef_endpoint()


@dataclass(frozen=True)
class CanalEndpoints:
    """신경관 양끝점."""

    vertex_a: int
    vertex_b: int
    point_a: np.ndarray
    point_b: np.ndarray
    geodesic_length_mm: float
    n_candidates: int
    n_clusters: int

    def mef_endpoint(self) -> CanalEndpoint:
        """선정된 끝점 쌍 중 y가 작은 정점 (MeF)."""
        return self.min_y_endpoint()

    def min_y_endpoint(self) -> CanalEndpoint:
        """y값이 더 작은 끝점 1개 반환."""
        if float(self.point_a[1]) <= float(self.point_b[1]):
            return CanalEndpoint(vertex=self.vertex_a, point=self.point_a.copy())
        return CanalEndpoint(vertex=self.vertex_b, point=self.point_b.copy())

    def max_z_endpoint(self) -> CanalEndpoint:
        """z값이 더 큰 끝점 1개 반환."""
        if float(self.point_a[2]) >= float(self.point_b[2]):
            return CanalEndpoint(vertex=self.vertex_a, point=self.point_a.copy())
        return CanalEndpoint(vertex=self.vertex_b, point=self.point_b.copy())


@dataclass(frozen=True)
class SplitCanals:
    """좌·우로 분리된 신경관 메쉬."""

    right: pv.PolyData
    left: pv.PolyData
