"""
세그멘테이션 NIfTI(정수 라벨) → STL / OBJ / PLY (라벨별 메시).

Windows에서 입력 NIfTI·dataset.json·출력 폴더에 비ASCII가 있으면 ASCII 임시 경로를 사용합니다.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import itk
import numpy as np

from .mesh_vtk import polydata_to_trimesh, vtk_contour_binary_roi
from .windows_path_staging import (
    ascii_system_temp_parent,
    cleanup_staging_roots,
    file_needs_native_staging,
    native_io_staging_enabled,
    path_string_is_pure_ascii,
    stage_file_copy,
)


def itk_continuous_index_xyz_to_physical_points(itk_image: Any, idx_xyz: np.ndarray) -> np.ndarray:
    idx_xyz = np.asarray(idx_xyz, dtype=np.float64)
    sp = np.asarray(itk_image.GetSpacing(), dtype=np.float64)
    og = np.asarray(itk_image.GetOrigin(), dtype=np.float64)
    d_flat = np.asarray(itk.GetArrayFromMatrix(itk_image.GetDirection()), dtype=np.float64)
    d = d_flat.reshape(3, 3)
    scaled = idx_xyz * sp
    return og + scaled @ d.T


def _safe_stem(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE)
    return s.strip("_") or "label"


def mesh_keep_largest_for_pipeline_label(
    display_name: str,
    *,
    mesh_keep_largest_component: bool = True,
) -> bool:
    if not mesh_keep_largest_component:
        return False
    raw = (display_name or "").strip()
    if not raw:
        return False
    return raw.lower() == "upper skull"


def postprocess_label_surface_mesh(
    mesh: Any,
    *,
    keep_largest_component: bool = True,
    mesh_decimation_factor: float = 0.5,
    mesh_smoothing_factor: float = 0.5,
) -> Optional[Any]:
    import trimesh

    from .slicer_style_mesh_ops import slicer_style_postprocess_trimesh

    if mesh is None or not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return None

    work = mesh.copy()

    if keep_largest_component:
        parts = work.split(only_watertight=False)
        if not parts:
            return None
        work = max(parts, key=lambda m: len(m.faces))
        if len(work.faces) == 0:
            return None

    if mesh_decimation_factor <= 0.0 and mesh_smoothing_factor <= 0.0:
        return work

    return slicer_style_postprocess_trimesh(
        work,
        decimation_factor=float(mesh_decimation_factor),
        smoothing_factor=float(mesh_smoothing_factor),
    )


def _resolve_ids_from_label_names(
    names: Iterable[str],
    id_to_name: Dict[int, str],
) -> Set[int]:
    want = {str(n).strip().lower() for n in names if str(n).strip()}
    if not want:
        return set()
    out: Set[int] = set()
    for lid, disp in id_to_name.items():
        if disp.strip().lower() in want:
            out.add(lid)
    return out


def label_roi_slices(
    data: np.ndarray,
    label_id: int,
    *,
    pad: int,
) -> Optional[tuple[slice, slice, slice]]:
    coords = np.argwhere(data == label_id)
    if coords.size == 0:
        return None
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    nz, ny, nx = data.shape
    p = max(0, pad)
    z0 = max(0, int(lo[0]) - p)
    z1 = min(nz, int(hi[0]) + p + 1)
    y0 = max(0, int(lo[1]) - p)
    y1 = min(ny, int(hi[1]) + p + 1)
    x0 = max(0, int(lo[2]) - p)
    x1 = min(nx, int(hi[2]) + p + 1)
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def export_meshes_from_label_nifti(
    segmentation_nifti: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    dataset_json: Optional[Union[str, Path]] = None,
    formats: Iterable[str] = ("stl", "obj", "ply"),
    label_ids: Optional[Iterable[int]] = None,
    label_names: Optional[Iterable[str]] = None,
    step_size: int = 1,
    roi_pad: int = 2,
    mesh_postprocess: bool = True,
    mesh_keep_largest_component: bool = True,
    mesh_decimation_factor: float = 0.5,
    mesh_smoothing_factor: float = 0.5,
) -> List[Path]:
    """
    정수 라벨 NIfTI → 라벨별 닫힌 메시.

    - 좌표: ITK로 읽어 연속 인덱스→mm (spacing·origin·direction).
    """
    seg_orig = Path(segmentation_nifti).resolve()
    out_user = Path(output_dir).resolve()
    out_user.mkdir(parents=True, exist_ok=True)

    cleanups: List[Optional[Path]] = []

    seg_st = stage_file_copy(
        seg_orig,
        prefix="mesh_seg_",
        label="segmentation NIfTI",
    )
    if seg_st.cleanup_root is not None:
        cleanups.append(seg_st.cleanup_root)
    seg_path = seg_st.effective_path

    dj_open: Optional[Path] = None
    if dataset_json is not None:
        dj = Path(dataset_json).resolve()
        if dj.is_file() and file_needs_native_staging(dj):
            dj_st = stage_file_copy(dj, prefix="mesh_dsj_", label="dataset.json")
            if dj_st.cleanup_root is not None:
                cleanups.append(dj_st.cleanup_root)
            dj_open = dj_st.effective_path
        else:
            dj_open = dj

    out_write = out_user
    out_tmp: Optional[Path] = None
    if native_io_staging_enabled() and not path_string_is_pure_ascii(out_user):
        out_tmp = Path(
            tempfile.mkdtemp(prefix="mesh_out_", dir=str(ascii_system_temp_parent()))
        )
        cleanups.append(out_tmp)
        out_write = out_tmp

    written: List[Path] = []

    try:
        itk_seg = itk.imread(str(seg_path))
        data_full = np.asarray(itk.array_from_image(itk_seg))
        step_f = float(step_size)
        if step_size > 1:
            data = data_full[::step_size, ::step_size, ::step_size]
        else:
            data = data_full

        id_to_name: Dict[int, str] = {}
        if dj_open is not None:
            with open(dj_open, encoding="utf-8") as f:
                meta = json.load(f)
            labels = meta.get("labels") or {}
            for name, lid in labels.items():
                if str(name).lower() == "background":
                    continue
                try:
                    id_to_name[int(lid)] = str(name)
                except (TypeError, ValueError):
                    continue

        present = {int(x) for x in np.unique(data.ravel()) if int(x) > 0}
        if label_ids is not None:
            ids = {int(x) for x in label_ids} & present
        elif label_names is not None:
            if not id_to_name:
                raise ValueError(
                    "label_names requires dataset_json with labels; "
                    "cannot resolve names without dataset.json"
                )
            ids = _resolve_ids_from_label_names(label_names, id_to_name) & present
        elif id_to_name:
            ids = set(id_to_name.keys()) & present
        else:
            ids = present

        fmt_set = {f.lower().strip(".") for f in formats}
        out_write.mkdir(parents=True, exist_ok=True)

        for lid in sorted(ids):
            roi = label_roi_slices(data, lid, pad=roi_pad)
            if roi is None:
                continue
            sub = data[roi]
            mask = (sub == lid).astype(np.float32)
            if mask.sum() == 0:
                continue
            z0 = int(roi[0].start or 0)
            y0 = int(roi[1].start or 0)
            x0 = int(roi[2].start or 0)
            try:
                surf = vtk_contour_binary_roi(
                    mask, (z0, y0, x0), spacing_zyx=None
                )
            except ValueError:
                continue
            mesh = polydata_to_trimesh(surf)
            if mesh is None:
                continue
            v = np.asarray(mesh.vertices, dtype=np.float64)
            idx_xyz = v * step_f
            mesh.vertices = itk_continuous_index_xyz_to_physical_points(itk_seg, idx_xyz)
            stem_name = id_to_name.get(lid, f"label_{lid}")
            base = _safe_stem(stem_name)
            if mesh_postprocess:
                keep_largest = mesh_keep_largest_for_pipeline_label(
                    stem_name,
                    mesh_keep_largest_component=mesh_keep_largest_component,
                )
                mesh = postprocess_label_surface_mesh(
                    mesh,
                    keep_largest_component=keep_largest,
                    mesh_decimation_factor=mesh_decimation_factor,
                    mesh_smoothing_factor=mesh_smoothing_factor,
                )
                if mesh is None:
                    continue

            for fmt in fmt_set:
                dest = out_write / f"{base}.{fmt}"
                mesh.export(str(dest))
                written.append(dest.resolve())

        if out_tmp is not None:
            shutil.copytree(out_write, out_user, dirs_exist_ok=True)
            written = [
                (out_user / p.relative_to(out_write)).resolve() for p in written
            ]

        return written
    finally:
        cleanup_staging_roots(cleanups)


__all__ = ["export_meshes_from_label_nifti"]
