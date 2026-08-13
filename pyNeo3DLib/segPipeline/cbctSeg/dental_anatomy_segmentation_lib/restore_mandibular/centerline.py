"""센터라인: 희소 샘플 · 이동평균 · polyline 조밀화 전담."""

from __future__ import annotations

import networkx as nx
import numpy as np

from .skeleton import path_to_points


def arc_length_resample(points: np.ndarray, spacing_mm: float = 0.5) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-6:
        return points[:1].copy()
    n = max(2, int(np.ceil(total / spacing_mm)) + 1)
    samples = np.linspace(0.0, total, n)
    out = np.zeros((n, 3), dtype=float)
    for dim in range(3):
        out[:, dim] = np.interp(samples, cum, points[:, dim])
    return out


def moving_average_keep_endpoints(points: np.ndarray, window: int = 3) -> np.ndarray:
    """양 끝점 고정, 중간점만 이동평균."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3 or window <= 1:
        return pts.copy()

    w = int(window)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 3:
        return pts.copy()

    half = w // 2
    out = pts.copy()
    for i in range(1, n - 1):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = pts[lo:hi].mean(axis=0)
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def _linear_equal_samples(
    p0_mm: np.ndarray, p1_mm: np.ndarray, sample_mm: float
) -> np.ndarray:
    chord = float(np.linalg.norm(p1_mm - p0_mm))
    if chord < 1e-6:
        return p0_mm.reshape(1, 3)
    n = max(2, int(np.ceil(chord / max(sample_mm, 1e-6))) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - t) * p0_mm + t * p1_mm


def path_to_control_points(
    g: nx.Graph,
    path: list[int],
    spacing: np.ndarray,
    *,
    skeleton_sample_mm: float = 2.0,
    gap_sample_mm: float = 2.5,
) -> np.ndarray:
    """최장경로 → 스켈레톤/갭 등간격 희소 컨트롤 (ijk)."""
    if not path:
        return np.zeros((0, 3), dtype=float)
    if len(path) == 1:
        return path_to_points(g, path)

    segments: list[tuple[str, list[int]]] = []
    i = 0
    while i < len(path):
        run = [path[i]]
        while i + 1 < len(path):
            u, v = path[i], path[i + 1]
            ed = g.get_edge_data(u, v) or {}
            if ed.get("bridge", False):
                break
            run.append(v)
            i += 1
        segments.append(("skel", run))
        if i + 1 >= len(path):
            break
        segments.append(("gap", [path[i], path[i + 1]]))
        i += 1

    def _skel_pts_mm(nodes: list[int]) -> np.ndarray:
        pts = np.array([g.nodes[n]["ijk"] for n in nodes], dtype=float) * spacing
        if len(pts) >= 2:
            pts = arc_length_resample(pts, spacing_mm=skeleton_sample_mm)
        return pts

    controls_mm: list[np.ndarray] = []
    for kind, nodes in segments:
        if kind == "skel":
            pts_mm = _skel_pts_mm(nodes)
            if len(controls_mm) and len(pts_mm):
                if np.linalg.norm(pts_mm[0] - controls_mm[-1][-1]) < 1e-6:
                    pts_mm = pts_mm[1:]
            if len(pts_mm):
                controls_mm.append(pts_mm)
            continue

        u, v = nodes
        p0 = np.array(g.nodes[u]["ijk"], dtype=float) * spacing
        p1 = np.array(g.nodes[v]["ijk"], dtype=float) * spacing
        gap_mm = _linear_equal_samples(p0, p1, gap_sample_mm)
        if len(controls_mm) and len(gap_mm):
            if np.linalg.norm(gap_mm[0] - controls_mm[-1][-1]) < 1e-6:
                gap_mm = gap_mm[1:]
        if len(gap_mm):
            controls_mm.append(gap_mm)

    if not controls_mm:
        return path_to_points(g, path)

    pts_mm = np.vstack(controls_mm)
    if len(pts_mm) >= 2:
        keep = np.ones(len(pts_mm), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1) > 1e-6
        pts_mm = pts_mm[keep]
    return pts_mm / spacing


def build_centerline(
    g: nx.Graph,
    path_nodes: list[int],
    spacing: np.ndarray,
    *,
    skeleton_sample_mm: float,
    gap_sample_mm: float,
    ma_window: int,
    resample_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        controls_ijk, controls_ma_ijk, dense_ijk
    """
    controls = path_to_control_points(
        g,
        path_nodes,
        spacing,
        skeleton_sample_mm=skeleton_sample_mm,
        gap_sample_mm=gap_sample_mm,
    )
    if len(controls) < 2:
        return controls, controls.copy(), controls.copy()

    controls_ma = moving_average_keep_endpoints(controls * spacing, window=ma_window) / spacing
    dense = arc_length_resample(controls_ma * spacing, spacing_mm=resample_mm) / spacing
    return controls, controls_ma, dense
