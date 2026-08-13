"""교합평면 계산 결과 — 벡터를 Python list로 반환 (시각화 무관)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from core.shared.validation import LandmarkValidationError

if TYPE_CHECKING:
    from core.occlusal_plane.compute import OcclusalComputeData


def _to_list3(vector: np.ndarray, *, name: str) -> list[float]:
    arr = np.asarray(vector, dtype=float).ravel()
    if arr.size != 3:
        raise LandmarkValidationError(
            f"{name} 좌표는 길이 3이어야 합니다 (got {arr.size}).",
        )
    return [float(x) for x in arr]


# ``as_list()`` / ``as_dict()`` 키 순서 (고정)
VECTOR_NAMES: tuple[str, ...] = (
    "N",
    "ANS",
    "PNS",
    "RMeF",
    "LMeF",
    "msp_center",
    "msp_normal",
    "occlusal_center",
    "occlusal_normal",
)

VECTOR_LABELS_KO: dict[str, str] = {
    "N": "Nasion (두개골)",
    "ANS": "ANS (전비극)",
    "PNS": "PNS (후비극)",
    "RMeF": "RMeF (우 하악 이공, 신경관 추정)",
    "LMeF": "LMeF (좌 하악 이공, 신경관 추정)",
    "msp_center": "시상정중면(MSP) 중심",
    "msp_normal": "시상정중면(MSP) 법선 (단위벡터)",
    "occlusal_center": "교합평면 중심 P_occ_mid (mm)",
    "occlusal_normal": "교합평면 법선 n_occ (단위벡터)",
}


@dataclass(frozen=True)
class OcclusalPlaneResult:
    """N/ANS/PNS/LMeF/RMeF 및 MSP·교합평면 중심·법선 (mm, LPS)."""

    N: list[float]
    ANS: list[float]
    PNS: list[float]
    RMeF: list[float]
    LMeF: list[float]
    msp_center: list[float]
    msp_normal: list[float]
    occlusal_center: list[float]
    occlusal_normal: list[float]

    def as_dict(self) -> dict[str, list[float]]:
        """이름 → [x, y, z] (mm LPS 또는 단위 법선)."""
        return {name: getattr(self, name) for name in VECTOR_NAMES}

    def as_list(self) -> list[list[float]]:
        """고정 순서의 9개 3D 벡터 리스트 (이름은 ``VECTOR_NAMES`` 참고)."""
        return [self.as_dict()[name] for name in VECTOR_NAMES]

    def as_named_list(self) -> list[dict[str, object]]:
        """[{\"name\": \"N\", \"label\": \"...\", \"xyz\": [x,y,z]}, ...]"""
        return [
            {
                "name": name,
                "label": VECTOR_LABELS_KO[name],
                "xyz": self.as_dict()[name],
            }
            for name in VECTOR_NAMES
        ]


def occlusal_result_from_data(data: OcclusalComputeData) -> OcclusalPlaneResult:
    cranial = data.cranial
    mef = data.mef
    geom = data.geom

    return OcclusalPlaneResult(
        N=_to_list3(cranial["N"], name="N"),
        ANS=_to_list3(cranial["ANS"], name="ANS"),
        PNS=_to_list3(cranial["PNS"], name="PNS"),
        RMeF=_to_list3(mef["RMeF"], name="RMeF"),
        LMeF=_to_list3(mef["LMeF"], name="LMeF"),
        msp_center=_to_list3(geom.msp.center, name="msp_center"),
        msp_normal=_to_list3(geom.msp.normal, name="msp_normal"),
        occlusal_center=_to_list3(geom.p_occ_mid, name="occlusal_center"),
        occlusal_normal=_to_list3(geom.occlusal_normal, name="occlusal_normal"),
    )
