"""PyVista 단계별 3D 시각화 (선택적 의존성)."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import nibabel as nib
import networkx as nx
import numpy as np
from scipy import ndimage
from skimage import measure

from .models import PipelineResult, SideArtifacts


@dataclass
class StageView:
    name: str
    description: str
    meshes: list[tuple] = field(default_factory=list)  # (dataset, kwargs)


def _ijk_to_world(affine: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    ijk = np.asarray(ijk, dtype=float)
    if ijk.ndim == 1:
        ijk = ijk.reshape(1, 3)
    return nib.affines.apply_affine(affine, ijk)


def _local_affine(affine: np.ndarray, offset: np.ndarray) -> np.ndarray:
    aff = affine.copy()
    aff[:3, 3] = affine[:3, :3] @ offset + affine[:3, 3]
    return aff


def _mask_surf(mask, affine, *, step_size=2, color="crimson", opacity=0.55, name="m"):
    import pyvista as pv

    if mask is None or not np.any(mask):
        return []
    try:
        verts, faces, *_ = measure.marching_cubes(
            mask.astype(np.float32), level=0.5, step_size=step_size
        )
    except Exception:
        return []
    if len(verts) == 0 or len(faces) == 0:
        return []
    vw = _ijk_to_world(affine, verts)
    faces_pv = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)]
    ).ravel()
    mesh = pv.PolyData(vw, faces_pv)
    try:
        mesh = mesh.clean()
        if mesh.n_points > 0:
            mesh = mesh.smooth(n_iter=10, relaxation_factor=0.1)
    except Exception:
        pass
    return [(mesh, {"color": color, "opacity": opacity, "smooth_shading": True, "name": name})]


def _cc_surfs(mask, affine, *, step_size=2, opacity=0.65):
    labeled, n = ndimage.label(mask)
    out = []
    for cid in range(1, n + 1):
        rgb = colorsys.hsv_to_rgb((cid - 1) / max(n, 1), 0.75, 0.95)
        out.extend(
            _mask_surf(
                labeled == cid, affine, step_size=step_size, color=rgb, opacity=opacity, name=f"cc{cid}"
            )
        )
    return out


def _cloud(ijk, affine, *, color="yellow", point_size=12, name="pts"):
    import pyvista as pv

    if len(ijk) == 0:
        return []
    cloud = pv.PolyData(_ijk_to_world(affine, ijk))
    return [
        (
            cloud,
            {
                "color": color,
                "point_size": point_size,
                "render_points_as_spheres": True,
                "name": name,
            },
        )
    ]


def _polyline(ijk, affine, *, color="cyan", width=4, name="path"):
    import pyvista as pv

    if len(ijk) < 2:
        return _cloud(ijk, affine, color=color, name=name)
    pts = _ijk_to_world(affine, ijk)
    tube = pv.lines_from_points(pts).tube(radius=max(width * 0.05, 0.15))
    return [(tube, {"color": color, "opacity": 0.95, "smooth_shading": True, "name": name})]


def _graph_edges(g: nx.Graph, affine, *, color="#B0BEC5", bridge_color="orange", radius=0.1):
    import pyvista as pv

    out = []
    for u, v, data in g.edges(data=True):
        a = np.array(g.nodes[u]["ijk"], dtype=float)
        b = np.array(g.nodes[v]["ijk"], dtype=float)
        seg = np.vstack([_ijk_to_world(affine, a), _ijk_to_world(affine, b)])
        col = bridge_color if data.get("bridge") else color
        rad = radius * 1.6 if data.get("bridge") else radius
        out.append(
            (
                pv.lines_from_points(seg).tube(radius=rad),
                {"color": col, "opacity": 0.9, "name": f"e{u}_{v}"},
            )
        )
    return out


def build_stages_for_side(
    art: SideArtifacts, affine: np.ndarray, *, step_size: int = 2
) -> list[StageView]:
    tag = f"[{art.side}]"
    color = "#4FC3F7" if art.side == "L" else "#FF8A65"
    aff = _local_affine(affine, art.offset)
    stages: list[StageView] = []

    if art.raw.size == 0 or not np.any(art.raw):
        return [StageView(f"{tag} empty", "no canal", [])]

    stages.append(StageView(f"{tag} 1. Binary Mask", "raw side mask", _mask_surf(art.raw, aff, step_size=step_size, color=color)))
    stages.append(
        StageView(
            f"{tag} 2. Majority Filter",
            "binary majority",
            _mask_surf(art.after_majority, aff, step_size=step_size, color="#81C784"),
        )
    )
    stages.append(
        StageView(
            f"{tag} 3. Connected Components",
            "colored by CC",
            _cc_surfs(art.after_majority, aff, step_size=step_size),
        )
    )
    stages.append(
        StageView(
            f"{tag} 4. Small CC Removal",
            f"kept mask",
            _mask_surf(art.after_small_cc, aff, step_size=step_size, color="#AED581"),
        )
    )
    skel_pts = np.argwhere(art.skeleton) if art.skeleton.size else np.zeros((0, 3))
    stages.append(
        StageView(
            f"{tag} 5. Skeleton",
            f"voxels={len(skel_pts)}",
            _cloud(skel_pts, aff, color="white", point_size=8)
            + _mask_surf(art.after_small_cc, aff, step_size=step_size, color=color, opacity=0.12, name="ghost"),
        )
    )

    end_ijk = (
        np.array([art.graph.nodes[n]["ijk"] for n in art.endpoints], dtype=float)
        if art.graph is not None and art.endpoints
        else np.zeros((0, 3))
    )
    if art.graph is not None:
        stages.append(
            StageView(
                f"{tag} 6. Graph + Endpoints",
                f"bridges={art.n_bridges} endpoints={len(art.endpoints)}",
                _graph_edges(art.graph, aff)
                + _cloud(end_ijk, aff, color="red", point_size=16, name="ends")
                + _mask_surf(art.after_small_cc, aff, step_size=step_size, color=color, opacity=0.08, name="ghost"),
            )
        )

    stages.append(
        StageView(
            f"{tag} 7. Sparse Controls",
            f"n={len(art.controls_ijk)}",
            _cloud(art.controls_ijk, aff, color="#FFEE58", point_size=14)
            + _polyline(art.controls_ijk, aff, color="#FFEE58", width=3)
            + _mask_surf(art.after_small_cc, aff, step_size=step_size, color=color, opacity=0.1, name="ghost"),
        )
    )
    stages.append(
        StageView(
            f"{tag} 8. MA + Polyline",
            "endpoints fixed, middle MA-smoothed",
            _polyline(art.controls_ijk, aff, color="#90A4AE", width=2, name="before")
            + _cloud(art.controls_ma_ijk, aff, color="#E040FB", point_size=14)
            + _polyline(art.controls_ma_ijk, aff, color="#E040FB", width=5)
            + _mask_surf(art.after_small_cc, aff, step_size=step_size, color=color, opacity=0.1, name="ghost"),
        )
    )
    stages.append(
        StageView(
            f"{tag} 9. Dense Path",
            f"n={len(art.dense_ijk)}",
            _polyline(art.dense_ijk, aff, color="#00E5FF", width=5)
            + _mask_surf(art.after_small_cc, aff, step_size=step_size, color=color, opacity=0.1, name="ghost"),
        )
    )
    stages.append(
        StageView(
            f"{tag} 10. Restored Tube",
            f"vox={int(art.restored_crop.sum()) if art.restored_crop.size else 0}",
            _mask_surf(art.restored_crop, aff, step_size=step_size, color="#26A69A", opacity=0.7)
            + _polyline(art.controls_ma_ijk, aff, color="#E040FB", width=3),
        )
    )
    added = art.restored_crop & ~art.raw if art.restored_crop.size and art.raw.size else art.restored_crop
    stages.append(
        StageView(
            f"{tag} 11. Before vs After",
            f"path={art.path_length_mm:.1f}mm bridges={art.n_bridges}",
            _mask_surf(art.raw, aff, step_size=step_size, color="#EF5350", opacity=0.35, name="before")
            + _mask_surf(added, aff, step_size=1, color="#FFEE58", opacity=0.85, name="added")
            + _polyline(art.controls_ma_ijk, aff, color="#E040FB", width=4),
        )
    )
    return stages


def merge_stages(*lists: list[StageView]) -> list[StageView]:
    from collections import defaultdict

    if len(lists) == 1:
        return lists[0]
    buckets: dict[str, list[StageView]] = defaultdict(list)
    order: list[str] = []
    for lst in lists:
        for st in lst:
            key = st.name.split("] ", 1)[-1] if "] " in st.name else st.name
            if key not in buckets:
                order.append(key)
            buckets[key].append(st)
    merged = []
    for key in order:
        items = buckets[key]
        meshes = []
        descs = []
        for it in items:
            meshes.extend(it.meshes)
            descs.append(f"{it.name}: {it.description}")
        merged.append(StageView(key, " | ".join(descs), meshes))
    return merged


class StageViewer:
    def __init__(self, stages: list[StageView], title: str = "Canal Restore"):
        import pyvista as pv

        self.stages = stages
        self.idx = 0
        self.plotter = pv.Plotter(title=title)
        self.plotter.set_background("#1e1e1e")
        self._names: list[str] = []
        self._cam = False
        self.plotter.add_key_event("Left", self.prev)
        self.plotter.add_key_event("Right", self.next)
        self.plotter.add_key_event("a", self.prev)
        self.plotter.add_key_event("d", self.next)
        n = max(len(stages) - 1, 1)
        self.plotter.add_slider_widget(
            lambda v: self.show_stage(int(round(v))),
            rng=[0, n],
            value=0,
            title="Stage",
            pointa=(0.25, 0.92),
            pointb=(0.75, 0.92),
            style="modern",
            fmt="%.0f",
        )
        self.show_stage(0)

    def show_stage(self, idx: int):
        if not self.stages:
            return
        idx = int(np.clip(idx, 0, len(self.stages) - 1))
        self.idx = idx
        st = self.stages[idx]
        for name in self._names:
            try:
                self.plotter.remove_actor(name, render=False)
            except Exception:
                pass
        self._names.clear()
        for t in ("title", "desc"):
            try:
                self.plotter.remove_actor(t, render=False)
            except Exception:
                pass
        for mesh, kw in st.meshes:
            uniq = f"{kw.get('name', 'a')}_{len(self._names)}"
            k = dict(kw)
            k["name"] = uniq
            self.plotter.add_mesh(mesh, **k)
            self._names.append(uniq)
        self.plotter.add_text(
            f"[{idx + 1}/{len(self.stages)}] {st.name}",
            position="upper_left",
            font_size=12,
            color="white",
            name="title",
        )
        self.plotter.add_text(
            st.description + "\nLeft/Right or A/D : stage | Q : quit",
            position="lower_left",
            font_size=9,
            color="lightgray",
            name="desc",
        )
        if not self._cam:
            self.plotter.reset_camera()
            self._cam = True
        self.plotter.render()

    def next(self):
        self.show_stage(self.idx + 1)

    def prev(self):
        self.show_stage(self.idx - 1)

    def show(self):
        self.plotter.show()


def visualize_result(
    result: PipelineResult,
    *,
    side: str = "both",
    step_size: int = 2,
    title: str = "Canal Restore Stages",
) -> None:
    try:
        import pyvista  # noqa: F401
    except ImportError as e:
        raise ImportError("시각화에는 pyvista가 필요합니다: pip install pyvista") from e

    lists = []
    if side in ("L", "both") and result.left is not None:
        lists.append(build_stages_for_side(result.left, result.affine, step_size=step_size))
    if side in ("R", "both") and result.right is not None:
        lists.append(build_stages_for_side(result.right, result.affine, step_size=step_size))
    if not lists:
        print("시각화할 side 결과 없음")
        return
    stages = merge_stages(*lists)
    print(f"viz stages={len(stages)} (Left/Right to navigate)")
    StageViewer(stages, title=title).show()
