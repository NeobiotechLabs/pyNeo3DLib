"""메쉬 전처리 및 측지 거리 그래프."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def ensure_polydata(mesh: pv.DataSet) -> pv.PolyData:
    if isinstance(mesh, pv.PolyData):
        return mesh.clean()
    try:
        surface = mesh.extract_surface(algorithm="dataset_surface")
    except TypeError:
        surface = mesh.extract_surface()
    return surface.triangulate().clean()


def nearest_vertex_index(points: np.ndarray, position: np.ndarray) -> int:
    return int(np.argmin(np.sum((points - position) ** 2, axis=1)))


def vertex_adjacency_lists(mesh: pv.PolyData) -> list[list[int]]:
    """삼각형 edge 기준 정점 이웃 목록."""
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    n_vertices = mesh.n_points
    neighbors: list[set[int]] = [set() for _ in range(n_vertices)]

    for tri in faces:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(tri[i]), int(tri[j])
                neighbors[a].add(b)
                neighbors[b].add(a)

    return [sorted(nb) for nb in neighbors]


def vertex_normal_divergence(mesh: pv.PolyData) -> np.ndarray:
    """정점별 법선장 surface divergence (이웃 edge 유한차분).

    div(n)_i ≈ (1/|N(i)|) Σ_{j∈N(i)} (n_j - n_i)·(p_j - p_i) / |p_j - p_i|²
    """
    mesh = mesh.compute_normals(
        point_normals=True,
        cell_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
    )
    points = np.asarray(mesh.points, dtype=np.float64)
    normals = np.asarray(mesh.point_normals, dtype=np.float64)

    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    edges_a = np.concatenate(
        [faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]]
    )
    edges_b = np.concatenate(
        [faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]]
    )
    edges = np.unique(np.stack([edges_a, edges_b], axis=1), axis=0)
    u = edges[:, 0]
    v = edges[:, 1]

    delta_p = points[v] - points[u]
    delta_n = normals[v] - normals[u]
    length_sq = np.sum(delta_p**2, axis=1)

    valid_edges = length_sq > 1e-12
    u = u[valid_edges]
    delta_p = delta_p[valid_edges]
    delta_n = delta_n[valid_edges]
    length_sq = length_sq[valid_edges]

    edge_vals = np.sum(delta_n * delta_p, axis=1) / length_sq

    sum_vals = np.bincount(u, weights=edge_vals, minlength=mesh.n_points)
    counts = np.bincount(u, minlength=mesh.n_points)

    divergence = np.full(mesh.n_points, np.nan, dtype=np.float64)
    valid_vertices = counts > 0
    divergence[valid_vertices] = sum_vals[valid_vertices] / counts[valid_vertices]

    return divergence


class MeshGeodesicGraph:
    """삼각형 edge 그래프 기반 측지 거리 계산 (단일 책임)."""

    def __init__(self, mesh: pv.PolyData) -> None:
        self._adjacency = self._build_adjacency(mesh)

    @staticmethod
    def _build_adjacency(mesh: pv.PolyData) -> csr_matrix:
        points = np.asarray(mesh.points, dtype=np.float64)
        faces = mesh.faces.reshape(-1, 4)[:, 1:4]
        n_vertices = mesh.n_points

        rows: list[int] = []
        cols: list[int] = []
        weights: list[float] = []

        for tri in faces:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = int(tri[i]), int(tri[j])
                    w = float(np.linalg.norm(points[a] - points[b]))
                    rows.extend((a, b))
                    cols.extend((b, a))
                    weights.extend((w, w))

        return csr_matrix(
            (weights, (rows, cols)),
            shape=(n_vertices, n_vertices),
        )

    def distance(self, vertex_a: int, vertex_b: int) -> float:
        distances = dijkstra(self._adjacency, directed=False, indices=vertex_a)
        dist = float(distances[vertex_b])
        if not np.isfinite(dist):
            raise ValueError(
                f"두 정점({vertex_a}, {vertex_b}) 사이 측지 경로가 없습니다."
            )
        return dist
