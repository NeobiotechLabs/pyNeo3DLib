"""CLI — 복원 실행 + 선택적 시각화."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import RestoreConfig
from .io_nifti import load_label, save_label, spacing_from_affine
from ..pipeline import centerline_payload, restore_canal


def _iter_inputs(path: Path):
    if path.is_file():
        yield path
        return
    yield from sorted(path.glob("*.nii.gz"))
    yield from sorted(p for p in path.glob("*.nii") if not str(p).endswith(".nii.gz"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="canal_restore",
        description="끊긴 하악 신경관(라벨3) 복원 — SRP 모듈 파이프라인",
    )
    p.add_argument("input", type=Path, help="nii.gz 파일 또는 폴더")
    p.add_argument("-o", "--output", type=Path, default=None, help="출력 폴더")
    p.add_argument("--majority-size", type=int, default=3)
    p.add_argument("--small-ratio", type=float, default=0.005)
    p.add_argument("--min-voxels", type=int, default=80)
    p.add_argument("--max-gap-mm", type=float, default=15.0)
    p.add_argument("--skeleton-sample-mm", type=float, default=2.0)
    p.add_argument("--gap-sample-mm", type=float, default=2.5)
    p.add_argument("--ma-window", type=int, default=3)
    p.add_argument("--resample-mm", type=float, default=0.5)
    p.add_argument("--radius-mm", type=float, default=None)
    p.add_argument("--no-keep-original", action="store_true")
    # viz
    p.add_argument("--viz", action="store_true", help="PyVista 단계별 3D 시각화 ON")
    p.add_argument("--viz-side", choices=["L", "R", "both"], default="both")
    p.add_argument("--viz-step-size", type=int, default=2)
    return p


def config_from_args(args: argparse.Namespace) -> RestoreConfig:
    return RestoreConfig(
        majority_size=args.majority_size,
        small_ratio=args.small_ratio,
        min_voxels=args.min_voxels,
        max_gap_mm=args.max_gap_mm,
        skeleton_sample_mm=args.skeleton_sample_mm,
        gap_sample_mm=args.gap_sample_mm,
        ma_window=args.ma_window,
        resample_mm=args.resample_mm,
        radius_mm=args.radius_mm,
        keep_original_canal=not args.no_keep_original,
        viz=args.viz,
        viz_side=args.viz_side,
        viz_step_size=args.viz_step_size,
        output_dir=args.output,
    )


def process_one(src: Path, out_dir: Path, cfg: RestoreConfig) -> dict:
    data, affine, img = load_label(src)
    spacing = spacing_from_affine(affine)
    result = restore_canal(data, affine, spacing, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.name.replace(".nii.gz", "").replace(".nii", "")
    out_nii = out_dir / f"{stem}_canal_restored.nii.gz"
    out_json = out_dir / f"{stem}_canal_restored.json"

    save_label(out_nii, result.label_out, affine, img)

    # TypeScript 재구성용 최소 JSON: 좌/우 중심선 월드좌표 (N,3)
    payload = centerline_payload(result)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.viz:
        from .visualize import visualize_result

        visualize_result(
            result,
            side=cfg.viz_side,
            step_size=cfg.viz_step_size,
            title=f"Canal Restore - {src.name}",
        )

    return {
        "src": str(src),
        "output_nii": str(out_nii),
        "canal_voxels_before": result.canal_before,
        "canal_voxels_after": result.canal_after,
        "added_voxels": result.added,
        "left": result.left.stats.to_dict() if result.left and result.left.stats else {},
        "right": result.right.stats.to_dict() if result.right and result.right.stats else {},
        "centerline_world": payload,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    src = args.input
    if not src.exists():
        print(f"입력 없음: {src}", file=sys.stderr)
        return 1

    out_dir = cfg.output_dir
    if out_dir is None:
        out_dir = (src.parent if src.is_file() else src) / "canal_restored"

    files = list(_iter_inputs(src))
    if not files:
        print(f"nii 없음: {src}", file=sys.stderr)
        return 1

    print(f"input={len(files)} -> {out_dir} | viz={'ON' if cfg.viz else 'OFF'}", flush=True)
    for i, f in enumerate(files, 1):
        try:
            meta = process_one(f, out_dir, cfg)
            L, R = meta.get("left") or {}, meta.get("right") or {}
            print(
                f"[{i}/{len(files)}] OK {f.name} | "
                f"vox {meta['canal_voxels_before']}->{meta['canal_voxels_after']} "
                f"(+{meta['added_voxels']}) | "
                f"L bridge={L.get('n_bridges')} R bridge={R.get('n_bridges')}",
                flush=True,
            )
        except Exception as e:
            print(f"[{i}/{len(files)}] ERR {f.name}: {e}", flush=True)
            import traceback

            traceback.print_exc()

    print("done.", flush=True)
    return 0
