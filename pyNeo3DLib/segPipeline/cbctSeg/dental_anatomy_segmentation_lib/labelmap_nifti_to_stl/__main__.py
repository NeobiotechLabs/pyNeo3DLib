"""
python -m labelmap_nifti_to_stl <reference.nii.gz> <labelmap.nii.gz> <output_dir> [--dataset-json PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Nearest-neighbor resample labelmap NIfTI to reference volume grid, "
            "then export per-label meshes (STL/OBJ/PLY)."
        )
    )
    p.add_argument(
        "reference_nifti",
        type=Path,
        help="Reference volume NIfTI (e.g. CBCT) defining output grid",
    )
    p.add_argument(
        "labelmap_nifti",
        type=Path,
        help="Integer label segmentation NIfTI",
    )
    p.add_argument(
        "output_dir",
        type=Path,
        help="Output folder (aligned NIfTI + mesh files)",
    )
    p.add_argument(
        "--dataset-json",
        type=Path,
        default=None,
        help="Optional nnU-Net dataset.json (label id to display name for filenames)",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        default=["stl"],
        metavar="FMT",
        help="Mesh formats (default: stl). Example: --formats stl obj ply",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ref = args.reference_nifti.expanduser().resolve()
    lab = args.labelmap_nifti.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    dj = args.dataset_json.expanduser().resolve() if args.dataset_json else None

    if not ref.is_file():
        print(f"Reference NIfTI not found: {ref}", file=sys.stderr)
        return 1
    if not lab.is_file():
        print(f"Labelmap NIfTI not found: {lab}", file=sys.stderr)
        return 1
    if dj is not None and not dj.is_file():
        print(f"dataset.json not found: {dj}", file=sys.stderr)
        return 1

    from .pipeline import run_align_prediction_to_meshes

    result = run_align_prediction_to_meshes(
        prediction_nifti=lab,
        reference_nifti=ref,
        mesh_output_dir=out,
        dataset_json=dj,
        mesh_formats=tuple(args.formats),
    )
    print(f"aligned: {result.aligned_labelmap_nifti}")
    print(f"meshes ({len(result.mesh_files)}):")
    for p in result.mesh_files:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
