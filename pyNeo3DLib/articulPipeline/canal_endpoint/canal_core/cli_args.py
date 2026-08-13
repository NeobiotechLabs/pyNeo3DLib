"""신경관 파이프라인 CLI 인자 (공통)."""

from __future__ import annotations

import argparse

from .defaults import (
    DEFAULT_CANDIDATES,
    DEFAULT_CLUSTERING,
    DEFAULT_PAIR_SELECTION,
)


def add_basic_canal_args(parser: argparse.ArgumentParser) -> None:
    """교합평면 CLI용 최소 파이프라인 인자."""
    parser.add_argument(
        "--curvature-percentile",
        type=float,
        default=DEFAULT_CANDIDATES.curvature_percentile,
    )
    parser.add_argument(
        "--cluster-distance-ratio",
        type=float,
        default=DEFAULT_CLUSTERING.distance_ratio,
    )
    parser.add_argument("--expected-components", type=int, default=2)


def add_canal_pipeline_args(
    parser: argparse.ArgumentParser,
    *,
    include_top_geodesic_pairs: bool = False,
) -> None:
    """평가·비교 CLI용 파이프라인 인자."""
    parser.add_argument(
        "--candidate-extractor",
        choices=("normal_divergence", "curvature"),
        default="normal_divergence",
    )
    parser.add_argument("--normal-divergence-percentile", type=float, default=90.0)
    parser.add_argument(
        "--curvature-percentile",
        type=float,
        default=DEFAULT_CANDIDATES.curvature_percentile,
    )
    parser.add_argument(
        "--cluster-distance-ratio",
        type=float,
        default=DEFAULT_CLUSTERING.distance_ratio,
    )
    parser.add_argument("--expected-components", type=int, default=2)
    if include_top_geodesic_pairs:
        parser.add_argument(
            "--top-geodesic-pairs",
            type=int,
            default=DEFAULT_PAIR_SELECTION.top_geodesic_pairs,
        )
