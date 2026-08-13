"""MSP·교합평면 법선·교합 기준점 알고리즘.

계산 순서 (두개골 랜드마크 N, ANS, PNS 필수):

  1. **시상정중면 (MSP)** — 세 점이 정의하는 평면
  2. **교합평면 법선** — MSP 법선 축으로 ANS를 회전한 뒤, PNS·PNS+MSP오프셋과 함께 3점 평면
  3. **교합 기준점** — 공통 법선 위에서 ANS / 이공(MeF, 신경관 STL 필수) / 그 중점으로 높이 오프셋
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.occlusal_plane.algorithm_config import (
    OCCLUSAL_ANS_HEIGHT_OFFSET_MM,
    OCCLUSAL_MEF_HEIGHT_OFFSET_MM,
)
from core.occlusal_plane.plane_math import (
    as_f64,
    midpoint,
    plane_from_three_points,
    point_along_normal,
    rotate_point_about_axis,
    unit_vector,
)


@dataclass(frozen=True)
class Plane3D:
    """무게중심과 단위 법선으로 표현하는 평면."""

    center: np.ndarray
    normal: np.ndarray

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray]:
        return self.center, self.normal


@dataclass(frozen=True)
class OcclusalGeometry:
    """MSP, 교합 법선, ANS·이공·중점 교합 기준점."""

    msp: Plane3D
    occlusal_normal: np.ndarray
    p_occ_ans: np.ndarray
    mef_mid: np.ndarray
    p_occ_mef: np.ndarray
    p_occ_mid: np.ndarray


# ---------------------------------------------------------------------------
# 1. 시상정중면 (MSP)
# ---------------------------------------------------------------------------


class MidSagittalPlaneCalculator:
    """N, ANS, PNS로 정의하는 시상정중면 (mid-sagittal plane)."""

    def compute(
        self,
        n: np.ndarray,
        ans: np.ndarray,
        pns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        plane = self.compute_plane(n, ans, pns)
        return plane.as_tuple()

    def compute_plane(self, n: np.ndarray, ans: np.ndarray, pns: np.ndarray) -> Plane3D:
        n, ans, pns = as_f64(n), as_f64(ans), as_f64(pns)
        centroid = (n + ans + pns) / 3.0
        normal = np.cross(ans - pns, n - pns)
        return Plane3D(center=centroid, normal=unit_vector(normal, name="MSP"))


# ---------------------------------------------------------------------------
# 2. 교합평면 법선
# ---------------------------------------------------------------------------


class OcclusalPlaneCalculator:
    """MSP 법선 축 기준 ANS 회전 + PNS 오프셋으로 교합평면 법선을 정한다."""

    def __init__(
        self,
        *,
        ans_rotation_deg: float = -6.0,
        pns_normal_offset_mm: float = 10.0,
    ) -> None:
        self._ans_rotation_deg = ans_rotation_deg
        self._pns_normal_offset_mm = pns_normal_offset_mm

    def compute_normal(
        self,
        ans: np.ndarray,
        pns: np.ndarray,
        msp_normal: np.ndarray,
    ) -> np.ndarray:
        n_msp = unit_vector(msp_normal, name="MSP 법선")
        pns = as_f64(pns)

        # PNS를 지나 MSP 법선 방향으로 ANS를 회전 → 교합면에 가까운 ANS 위치
        ans_on_occlusal_arc = rotate_point_about_axis(
            ans,
            pivot=pns,
            axis=n_msp,
            angle_deg=self._ans_rotation_deg,
        )
        # PNS에서 MSP 법선 방향으로 떨어진 세 번째 점 → 법선이 MSP에 수직 성분을 갖도록 함
        pns_along_msp = pns + self._pns_normal_offset_mm * n_msp

        _, normal = plane_from_three_points(ans_on_occlusal_arc, pns, pns_along_msp)
        return normal


# ---------------------------------------------------------------------------
# 3. 교합평면 위 기준점 (법선 방향 높이 오프셋)
# ---------------------------------------------------------------------------


def mef_midpoint(lmef: np.ndarray, rmef: np.ndarray) -> np.ndarray:
    """좌우 이공 평균 MeF_mid = (LMeF + RMeF) / 2."""
    return midpoint(lmef, rmef)


def occlusal_reference_from_ans(
    ans: np.ndarray,
    occlusal_normal: np.ndarray,
    *,
    height_offset_mm: float = OCCLUSAL_ANS_HEIGHT_OFFSET_MM,
) -> np.ndarray:
    """ANS에서 교합 법선 반대 방향으로 height_offset_mm 이동 → P_occ_ans."""
    return point_along_normal(
        ans,
        occlusal_normal,
        signed_offset_mm=-float(height_offset_mm),
    )


def occlusal_reference_from_mef(
    lmef: np.ndarray,
    rmef: np.ndarray,
    occlusal_normal: np.ndarray,
    *,
    height_offset_mm: float = OCCLUSAL_MEF_HEIGHT_OFFSET_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """MeF 중점에서 교합 법선 방향으로 height_offset_mm 이동 → (MeF_mid, P_occ_mef)."""
    mid = mef_midpoint(lmef, rmef)
    p_occ_mef = point_along_normal(
        mid,
        occlusal_normal,
        signed_offset_mm=float(height_offset_mm),
    )
    return mid, p_occ_mef


def occlusal_reference_midpoint(p_occ_ans: np.ndarray, p_occ_mef: np.ndarray) -> np.ndarray:
    """ANS·이공 교합 기준점의 중점 P_occ_mid."""
    return midpoint(p_occ_ans, p_occ_mef)


# ---------------------------------------------------------------------------
# 파이프라인: MSP → 법선 → 기준점
# ---------------------------------------------------------------------------


class OcclusalGeometryBuilder:
    """cranial·mef 랜드마크에서 OcclusalGeometry를 한 번에 계산."""

    def __init__(
        self,
        *,
        msp_calculator: MidSagittalPlaneCalculator | None = None,
        occlusal_calculator: OcclusalPlaneCalculator | None = None,
    ) -> None:
        self._msp = msp_calculator or MidSagittalPlaneCalculator()
        self._occlusal = occlusal_calculator or OcclusalPlaneCalculator()

    def build(
        self,
        cranial: dict[str, np.ndarray],
        mef: dict[str, np.ndarray],
    ) -> OcclusalGeometry:
        msp = self._msp.compute_plane(cranial["N"], cranial["ANS"], cranial["PNS"])

        occlusal_normal = self._occlusal.compute_normal(
            cranial["ANS"],
            cranial["PNS"],
            msp.normal,
        )

        p_occ_ans = occlusal_reference_from_ans(cranial["ANS"], occlusal_normal)

        mef_mid, p_occ_mef = occlusal_reference_from_mef(
            mef["LMeF"],
            mef["RMeF"],
            occlusal_normal,
        )
        p_occ_mid = occlusal_reference_midpoint(p_occ_ans, p_occ_mef)

        return OcclusalGeometry(
            msp=msp,
            occlusal_normal=occlusal_normal,
            p_occ_ans=p_occ_ans,
            mef_mid=mef_mid,
            p_occ_mef=p_occ_mef,
            p_occ_mid=p_occ_mid,
        )
