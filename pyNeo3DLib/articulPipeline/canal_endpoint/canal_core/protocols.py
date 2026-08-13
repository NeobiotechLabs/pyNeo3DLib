"""의존성 역전(D) — 구현체 교체 가능한 계약."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pyvista as pv

from .models import CandidateSet, SplitCanals


class CanalSplitter(Protocol):
    def split(self, mesh: pv.PolyData) -> SplitCanals: ...


class CandidateExtractor(Protocol):
    def extract(self, mesh: pv.PolyData) -> CandidateSet: ...


class CandidateClusterer(Protocol):
    def cluster(self, candidate_points: np.ndarray) -> np.ndarray: ...


class ClusterRepresentativeMapper(Protocol):
    def to_vertex_indices(
        self,
        mesh_points: np.ndarray,
        candidates: CandidateSet,
        labels: np.ndarray,
    ) -> list[int]: ...


class GeodesicDistanceCalculator(Protocol):
    def distance(self, vertex_a: int, vertex_b: int) -> float: ...


class EndpointPairSelector(Protocol):
    def select_farthest_pair(
        self,
        cluster_vertex_indices: list[int],
        geodesic: GeodesicDistanceCalculator,
        mesh_points: np.ndarray,
    ) -> tuple[int, int, float]: ...
