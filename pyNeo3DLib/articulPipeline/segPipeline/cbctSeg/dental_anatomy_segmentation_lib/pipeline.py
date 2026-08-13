"""파이프라인 오케스트레이션 — restore_mandibular 단계 모듈을 조합만 한다."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .restore_mandibular.centerline import build_centerline
from .restore_mandibular.config import LABEL_CANAL, RestoreConfig
from .restore_mandibular.io_nifti import load_label, save_label, spacing_from_affine
from .restore_mandibular.laterality import split_left_right_masks
from .restore_mandibular.models import PipelineResult, SideArtifacts, SideStats
from .restore_mandibular.morphology import (
    bbox_slices,
    binary_majority_filter,
    remove_small_components,
)
from .restore_mandibular.reconstruct import (
    estimate_radius_mm,
    merge_canal_into_label,
    rasterize_tube,
)
from .restore_mandibular.skeleton import (
    bridge_components,
    build_skeleton_graph,
    endpoints_of,
    longest_geodesic_path,
    skeletonize_3d,
)


def process_side(
    mask: np.ndarray,
    spacing: np.ndarray,
    side: str,
    cfg: RestoreConfig,
) -> SideArtifacts:
    empty = np.zeros_like(mask, dtype=bool)
    empty_pts = np.zeros((0, 3), dtype=float)
    empty_stats = SideStats(side=side)

    if not mask.any():
        return SideArtifacts(
            side=side,
            offset=np.zeros(3),
            crop_shape=(0, 0, 0),
            raw=empty,
            after_majority=empty,
            after_small_cc=empty,
            skeleton=empty,
            restored_full=empty,
            stats=empty_stats,
        )

    sl = bbox_slices(mask, pad=max(4, cfg.majority_size + 2))
    assert sl is not None
    offset = np.array([s.start for s in sl], dtype=float)
    crop = mask[sl]

    majority = binary_majority_filter(crop, size=cfg.majority_size)
    cleaned, n_before, n_after = remove_small_components(
        majority, ratio=cfg.small_ratio, min_voxels=cfg.min_voxels
    )

    if not cleaned.any():
        art = SideArtifacts(
            side=side,
            offset=offset,
            crop_shape=crop.shape,
            raw=crop,
            after_majority=majority,
            after_small_cc=cleaned,
            skeleton=np.zeros_like(cleaned),
            restored_full=empty,
            stats=SideStats(
                side=side,
                n_components_before=n_before,
                n_components_after=0,
            ),
        )
        return art

    skel = skeletonize_3d(cleaned)
    g, _ = build_skeleton_graph(skel, spacing)
    n_bridges = bridge_components(g, spacing, max_gap_mm=cfg.max_gap_mm)
    ends = endpoints_of(g)
    path_nodes, path_len = longest_geodesic_path(g)

    controls, controls_ma, dense = build_centerline(
        g,
        path_nodes,
        spacing,
        skeleton_sample_mm=cfg.skeleton_sample_mm,
        gap_sample_mm=cfg.gap_sample_mm,
        ma_window=cfg.ma_window,
        resample_mm=cfg.resample_mm,
    )

    r = (
        cfg.radius_mm
        if cfg.radius_mm is not None
        else estimate_radius_mm(cleaned, dense, spacing)
    )
    tube = rasterize_tube(crop.shape, dense, r, spacing)
    restored_crop = tube | cleaned

    restored_full = np.zeros_like(mask, dtype=bool)
    restored_full[sl] = restored_crop

    stats = SideStats(
        side=side,
        n_components_before=n_before,
        n_components_after=n_after,
        n_skeleton_voxels=int(skel.sum()),
        n_endpoints=len(ends),
        n_bridges=n_bridges,
        path_length_mm=float(path_len),
        path_n_points=int(len(dense)),
        radius_mm=float(r),
        restored_voxels=int(restored_full.sum()),
    )

    return SideArtifacts(
        side=side,
        offset=offset,
        crop_shape=crop.shape,
        raw=crop,
        after_majority=majority,
        after_small_cc=cleaned,
        skeleton=skel,
        graph=g,
        endpoints=ends,
        n_bridges=n_bridges,
        path_nodes=path_nodes,
        path_length_mm=path_len,
        controls_ijk=controls,
        controls_ma_ijk=controls_ma,
        dense_ijk=dense,
        restored_crop=restored_crop,
        restored_full=restored_full,
        stats=stats,
    )


def restore_canal(
    data: np.ndarray,
    affine: np.ndarray,
    spacing: np.ndarray,
    cfg: RestoreConfig,
) -> PipelineResult:
    canal = data == LABEL_CANAL
    left_m, right_m, split_meta = split_left_right_masks(canal, affine)

    left = process_side(left_m, spacing, "L", cfg)
    right = process_side(right_m, spacing, "R", cfg)

    new_canal = np.zeros_like(canal, dtype=bool)
    if left.restored_full is not None:
        new_canal |= left.restored_full
    if right.restored_full is not None:
        new_canal |= right.restored_full

    label_out = merge_canal_into_label(
        data, new_canal, keep_original=cfg.keep_original_canal
    )
    canal_after = label_out == LABEL_CANAL

    return PipelineResult(
        label_out=label_out,
        affine=affine,
        spacing=spacing,
        split_meta=split_meta,
        left=left,
        right=right,
        canal_before=int(canal.sum()),
        canal_after=int(canal_after.sum()),
        added=int((canal_after & ~canal).sum()),
    )


def _side_centerline_world(side_art: SideArtifacts | None, affine: np.ndarray) -> list[list[float]]:
    """이동평균 후 중심선을 월드 좌표 (N,3) 리스트로 변환."""
    if side_art is None or len(side_art.controls_ma_ijk) == 0:
        return []
    ijk_full = side_art.controls_ma_ijk + side_art.offset
    world = nib.affines.apply_affine(affine, ijk_full)
    return [[float(x), float(y), float(z)] for x, y, z in world]


def centerline_payload(result: PipelineResult) -> dict:
    """좌/우 신경관 중심선 월드 좌표 JSON payload."""
    return {
        "left": _side_centerline_world(result.left, result.affine),
        "right": _side_centerline_world(result.right, result.affine),
        "unit": "mm",
    }


def write_centerline_json(result: PipelineResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(centerline_payload(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def restore_canal_labelmap_nifti(
    src_path: Path,
    dst_path: Path,
    cfg: RestoreConfig | None = None,
) -> PipelineResult:
    """라벨맵 NIfTI 파일 단위 복원: 읽기 → ``restore_canal`` → 저장. 결과를 반환."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    data, affine, img = load_label(src_path)
    spacing = spacing_from_affine(affine)
    result = restore_canal(data, affine, spacing, cfg or RestoreConfig())
    save_label(dst_path, result.label_out, affine, img)
    return result


__all__ = [
    "LABEL_CANAL",
    "RestoreConfig",
    "restore_canal",
    "restore_canal_labelmap_nifti",
    "centerline_payload",
    "write_centerline_json",
]
