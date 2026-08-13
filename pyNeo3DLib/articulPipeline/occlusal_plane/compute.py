"""교합평면 벡터 계산 (occlusal_plane 시각화 모듈 미사용).

``canal_endpoint`` 는 pyvista·scipy·scikit-learn 에 의존합니다.
시각화 없이 9개 벡터만 필요하면 ``compute_occlusal_plane`` 을 사용하세요.

코어 세트: ``core/`` (``landmarks`` + ``canal_endpoint`` + ``occlusal_plane``).
``evaluation/`` 은 필요 없습니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from core.canal_endpoint import CanalPipelineConfigFactory, MandibularMefEstimator
from core.landmarks import LandmarkSubsetSelector, SlicerMarkupLandmarkReader

from core.occlusal_plane.algorithm_config import (
    OCCLUSAL_ANS_ROTATION_DEG,
    OCCLUSAL_PNS_MSP_NORMAL_OFFSET_MM,
)
from core.shared.constants import CRANIAL_LANDMARKS
from core.occlusal_plane.plane_algorithms import (
    MidSagittalPlaneCalculator,
    OcclusalGeometry,
    OcclusalGeometryBuilder,
    OcclusalPlaneCalculator,
)
from core.occlusal_plane.result import OcclusalPlaneResult, occlusal_result_from_data
from core.occlusal_plane.validation import require_existing_file


@dataclass(frozen=True)
class OcclusalPlaneInputs:
    """교합평면 계산·시각화 입력."""

    landmarks_path: Path
    mandibular_canal_path: Path
    upper_skull_path: Path | None = None
    mandible_path: Path | None = None
    teeth_paths: tuple[Path, ...] = ()
    curvature_percentile: float = 90.0
    cluster_distance_ratio: float = 0.06
    expected_components: int = 2


@dataclass(frozen=True)
class OcclusalComputeData:
    """계산 중간 결과 (numpy). 시각화 모듈이 장면 조립에 사용."""

    cranial: dict[str, np.ndarray]
    mef: dict[str, np.ndarray]
    geom: OcclusalGeometry


def compute_occlusal_plane_data(
    inputs: OcclusalPlaneInputs,
    *,
    landmark_reader: SlicerMarkupLandmarkReader | None = None,
    subset_selector: LandmarkSubsetSelector | None = None,
    mef_estimator_factory: type[MandibularMefEstimator] = MandibularMefEstimator,
    msp_calculator: MidSagittalPlaneCalculator | None = None,
    occlusal_calculator: OcclusalPlaneCalculator | None = None,
) -> OcclusalComputeData:
    landmarks_path = require_existing_file(inputs.landmarks_path, label="랜드마크 JSON")
    reader = landmark_reader or SlicerMarkupLandmarkReader()
    selector = subset_selector or LandmarkSubsetSelector()

    cranial = selector.select(
        reader.read(landmarks_path),
        CRANIAL_LANDMARKS,
        source=landmarks_path,
    )

    canal_path = Path(inputs.mandibular_canal_path).resolve()
    config = CanalPipelineConfigFactory.build(
        canal_path,
        curvature_percentile=inputs.curvature_percentile,
        cluster_distance_ratio=inputs.cluster_distance_ratio,
        expected_components=inputs.expected_components,
    )
    mef = mef_estimator_factory(config).estimate_from_path(canal_path)

    msp = msp_calculator or MidSagittalPlaneCalculator()
    occ = occlusal_calculator or OcclusalPlaneCalculator(
        ans_rotation_deg=OCCLUSAL_ANS_ROTATION_DEG,
        pns_normal_offset_mm=OCCLUSAL_PNS_MSP_NORMAL_OFFSET_MM,
    )
    geom = OcclusalGeometryBuilder(
        msp_calculator=msp,
        occlusal_calculator=occ,
    ).build(cranial, mef)

    return OcclusalComputeData(cranial=cranial, mef=mef, geom=geom)


def compute_occlusal_plane(inputs: OcclusalPlaneInputs) -> OcclusalPlaneResult:
    """9개 3D 벡터를 list로 반환 (시각화 없음)."""
    return occlusal_result_from_data(compute_occlusal_plane_data(inputs))
