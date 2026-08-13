"""3D 스켈레톤 · 그래프 · 브릿지 · 최장 경로 전담."""

from __future__ import annotations

import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

NEIGH26 = [
    (di, dj, dk)
    for di in (-1, 0, 1)
    for dj in (-1, 0, 1)
    for dk in (-1, 0, 1)
    if not (di == 0 and dj == 0 and dk == 0)
]


def skeletonize_3d(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return skeletonize(mask.astype(bool))


def build_skeleton_graph(
    skel: np.ndarray, spacing: np.ndarray
) -> tuple[nx.Graph, dict[tuple[int, int, int], int]]:
    coords = [tuple(map(int, c)) for c in np.argwhere(skel)]
    index = {c: i for i, c in enumerate(coords)}
    g = nx.Graph()
    for i, c in enumerate(coords):
        g.add_node(i, ijk=c)

    shape = skel.shape
    for c, i in index.items():
        for d in NEIGH26:
            nijk = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if nijk not in index:
                continue
            j = index[nijk]
            if i >= j:
                continue
            if not (
                0 <= nijk[0] < shape[0]
                and 0 <= nijk[1] < shape[1]
                and 0 <= nijk[2] < shape[2]
            ):
                continue
            step = np.array(d, dtype=float) * spacing
            g.add_edge(i, j, weight=float(np.linalg.norm(step)))
    return g, index


def endpoints_of(g: nx.Graph) -> list[int]:
    return [n for n, deg in g.degree() if deg == 1]


def bridge_components(g: nx.Graph, spacing: np.ndarray, max_gap_mm: float) -> int:
    if g.number_of_nodes() == 0:
        return 0
    comps = list(nx.connected_components(g))
    if len(comps) <= 1:
        return 0

    comp_id = {}
    for cid, nodes in enumerate(comps):
        for n in nodes:
            comp_id[n] = cid

    ends: list[int] = []
    for nodes in comps:
        sub = g.subgraph(nodes)
        ep = endpoints_of(sub)
        if not ep:
            ep = list(nodes)[:1]
        ends.extend(ep)

    candidates: list[tuple[float, int, int]] = []
    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            u, v = ends[a], ends[b]
            if comp_id[u] == comp_id[v]:
                continue
            pu = np.array(g.nodes[u]["ijk"], dtype=float)
            pv = np.array(g.nodes[v]["ijk"], dtype=float)
            dist = float(np.linalg.norm((pu - pv) * spacing))
            if dist <= max_gap_mm:
                candidates.append((dist, u, v))
    candidates.sort(key=lambda x: x[0])

    parent = list(range(len(comps)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    n_bridges = 0
    for dist, u, v in candidates:
        if union(comp_id[u], comp_id[v]):
            g.add_edge(u, v, weight=dist, bridge=True)
            n_bridges += 1
            if len({find(i) for i in range(len(comps))}) == 1:
                break
    return n_bridges


def longest_geodesic_path(g: nx.Graph) -> tuple[list[int], float]:
    if g.number_of_nodes() == 0:
        return [], 0.0

    largest = max(nx.connected_components(g), key=len)
    sub = g.subgraph(largest).copy()
    ends = endpoints_of(sub)

    if len(ends) < 2:
        nodes = list(sub.nodes())
        if len(nodes) == 1:
            return nodes, 0.0
        if len(nodes) > 80:
            try:
                peri = nx.periphery(sub, weight="weight")
                ends = peri[:2] if len(peri) >= 2 else nodes[:2]
            except Exception:
                ends = nodes[:2]
        else:
            best_path: list[int] = []
            best_len = -1.0
            for u in nodes:
                lengths = nx.single_source_dijkstra_path_length(sub, u, weight="weight")
                v = max(lengths, key=lengths.get)
                if lengths[v] > best_len:
                    best_len = lengths[v]
                    best_path = nx.shortest_path(sub, u, v, weight="weight")
            return best_path, float(best_len)

    best_path: list[int] = []
    best_len = -1.0
    for i, u in enumerate(ends):
        lengths = nx.single_source_dijkstra_path_length(sub, u, weight="weight")
        for v in ends[i + 1 :]:
            if v not in lengths:
                continue
            if lengths[v] > best_len:
                best_len = lengths[v]
                best_path = nx.shortest_path(sub, u, v, weight="weight")
    if not best_path:
        return list(sub.nodes())[:1], 0.0
    return best_path, float(best_len)


def path_to_points(g: nx.Graph, path: list[int]) -> np.ndarray:
    if not path:
        return np.zeros((0, 3), dtype=float)
    return np.array([g.nodes[n]["ijk"] for n in path], dtype=float)
