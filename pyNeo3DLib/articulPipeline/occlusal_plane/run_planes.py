"""통합 랜드마크 JSON 을 입력받아 MSP·교합평면 계산 파이프라인.

CLI 사용 예:
    python run_planes.py --input ./case01_landmarks.mrk.json
    python run_planes.py --input ./case01_landmarks.mrk.json --stdout  (stdout 출력)

Python 모듈 임포트 사용 예:
    from core.occlusal_plane.run_planes import run_planes_from_landmarks

    # 랜드마크 dict → 계산 → 결과 dict 반환 (파일 I/O 없음)
    landmarks = {"N": [...], "ANS": [...], ...}
    result = run_planes_from_landmarks(landmarks)

중요 사항:
- 필수 랜드마크 (N, ANS, PNS, LMeF, RMeF) 중 하나라도 누락하면
  모든 평면을 NaN 으로 표기하고 computed=false 반환
- 좌표계는 통합 mrk.json 과 동일 (LPS mm)
"""
from __future__ import annotations

import argparse
import json
import sys
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np


# ── 필수 랜드마크 정의 ───────────────────────────────────────────────────

REQUIRED_CRANIAL_LANDMARKS = ("N", "ANS", "PNS")
REQUIRED_MEF_LANDMARKS = ("LMeF", "RMeF")
ALL_REQUIRED: tuple[str, ...] = REQUIRED_CRANIAL_LANDMARKS + REQUIRED_MEF_LANDMARKS

SLICER_MARKUPS_SCHEMA = (
    "https://raw.githubusercontent.com/slicer/slicer/master/"
    "Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#"
)


def _nan_vector() -> list[float]:
    return [math.nan, math.nan, math.nan]


def _load_mrkt_json(path: Path) -> dict:
    """Slicer Markups JSON 로드 → controlPoints 리스트 반환."""
    if not path.is_file():
        raise FileNotFoundError(f"마크업 파일 없음: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    markups = data.get("markups", [])
    if not markups:
        raise ValueError("markups 항목이 없습니다.")
    points = []
    for markup in markups:
        points.extend(markup.get("controlPoints", []))
    return points


def _extract_landmarks(points: list[dict]) -> dict[str, np.ndarray]:
    """controlPoints 에서 label → position 매핑 생성."""
    landmarks: dict[str, np.ndarray] = {}
    for cp in points:
        label = cp.get("label", "")
        position = cp.get("position", [])
        if label and len(position) >= 3:
            landmarks[label] = np.asarray(position[:3], dtype=np.float64)
    return landmarks


def _validate_required(
    landmarks: dict[str, np.ndarray],
    required: tuple[str, ...],
    *,
    context: str = "landmark",
) -> tuple[list[str], Optional[dict[str, np.ndarray]]]:
    """필수 랜드마크 검증 → 누락 목록 + 존재하는 랜드마크 딕트 반환."""
    missing: list[str] = []
    found: dict[str, np.ndarray] = {}
    for name in required:
        if name in landmarks:
            found[name] = landmarks[name]
        else:
            missing.append(name)
    if missing:
        return missing, None
    return [], found


def _vector_to_list(v: np.ndarray) -> list[float]:
    arr = np.asarray(v, dtype=float).ravel()
    return [float(x) for x in arr]


# ── MSP (시상정중면) 계산 ────────────────────────────────────────────────

def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)) / 2.0


def _unit_vector(vector: np.ndarray, *, name: str) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        raise ValueError(f"{name} 벡터 길이가 0 에 가까워 정의할 수 없습니다.")
    return v / norm


def _plane_from_three_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """세 점을 지나는 평면의 무게중심과 단위 법선."""
    p1, p2, p3 = np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64), np.asarray(p3, dtype=np.float64)
    centroid = (p1 + p2 + p3) / 3.0
    normal = np.cross(p2 - p1, p3 - p1)
    return centroid, _unit_vector(normal, name="평면")


def compute_msp(cranial: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """두개골 랜드마크 (N, ANS, PNS) 로 MSP 계산.

    원본(core/occlusal_plane/plane_algorithms.MidSagittalPlaneCalculator)과
    동일한 외적 순서 cross(ANS-PNS, N-PNS) 를 사용해 법선 방향을 일치시킨다.
    이 법선이 교합평면 ANS 회전의 축으로 쓰이므로 방향(부호)이 중요하다.

    Returns: (center, unit_normal)
    """
    n = cranial["N"]
    ans = cranial["ANS"]
    pns = cranial["PNS"]
    centroid, normal = _plane_from_three_points(pns, ans, n)
    return centroid, normal


# ── 교합평면 계산 ───────────────────────────────────────────────────────

def rotate_point_about_axis(
    point: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """pivot 를 지나고 axis 방향인 직선을 축으로 point 를 angle_deg 만큼 회전 (Rodrigues)."""
    k = _unit_vector(axis, name="회전축")
    pivot = np.asarray(pivot, dtype=np.float64)
    v = np.asarray(point, dtype=np.float64) - pivot
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotated = v * cos_t + np.cross(k, v) * sin_t + k * np.dot(k, v) * (1.0 - cos_t)
    return pivot + rotated


def point_along_normal(origin: np.ndarray, normal: np.ndarray, *, signed_offset_mm: float) -> np.ndarray:
    """단위 법선 방향으로 signed_offset_mm 만큼 이동한 점."""
    n = _unit_vector(normal, name="법선")
    return np.asarray(origin, dtype=np.float64) + n * float(signed_offset_mm)


# 원본 core/occlusal_plane/algorithm_config.py 상수와 반드시 동일하게 유지
OCCLUSAL_ANS_HEIGHT_OFFSET_MM = 29.2
OCCLUSAL_MEF_HEIGHT_OFFSET_MM = 22.4

ANS_ROTATION_DEG = -6.0
PNS_NORMAL_OFFSET_MM = 10.0


def compute_occlusal_normal(ans: np.ndarray, pns: np.ndarray, msp_normal: np.ndarray) -> np.ndarray:
    """MSP 법선 축 기준 ANS 회전 + PNS 오프셋으로 교합평면 법선 계산."""
    n_msp = _unit_vector(msp_normal, name="MSP 법선")
    pns = np.asarray(pns, dtype=np.float64)

    ans_on_occlusal_arc = rotate_point_about_axis(
        ans,
        pivot=pns,
        axis=n_msp,
        angle_deg=ANS_ROTATION_DEG,
    )
    pns_along_msp = pns + PNS_NORMAL_OFFSET_MM * n_msp

    _, normal = _plane_from_three_points(ans_on_occlusal_arc, pns, pns_along_msp)
    return normal


def occlusal_reference_from_ans(
    ans: np.ndarray,
    occlusal_normal: np.ndarray,
    *,
    height_offset_mm: float = OCCLUSAL_ANS_HEIGHT_OFFSET_MM,
) -> np.ndarray:
    """ANS 에서 교합 법선 반대 방향으로 height_offset_mm 이동 → P_occ_ans."""
    return point_along_normal(ans, occlusal_normal, signed_offset_mm=-float(height_offset_mm))


def occlusal_reference_from_mef(
    lmef: np.ndarray,
    rmef: np.ndarray,
    occlusal_normal: np.ndarray,
    *,
    height_offset_mm: float = OCCLUSAL_MEF_HEIGHT_OFFSET_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """MeF 중점에서 교합 법선 방향으로 height_offset_mm 이동 → (MeF_mid, P_occ_mef)."""
    mid = _midpoint(lmef, rmef)
    p_occ_mef = point_along_normal(mid, occlusal_normal, signed_offset_mm=float(height_offset_mm))
    return mid, p_occ_mef


def compute_occlusal_center(
    ans: np.ndarray,
    mef: dict[str, np.ndarray],
    occlusal_normal: np.ndarray,
) -> np.ndarray:
    """교합평면 중심점 계산.

    P_occ_mid = 평균(P_occ_ans, P_occ_mef)
    """
    p_occ_ans = occlusal_reference_from_ans(ans, occlusal_normal)
    mef_mid, p_occ_mef = occlusal_reference_from_mef(mef["LMeF"], mef["RMeF"], occlusal_normal)
    return _midpoint(p_occ_ans, p_occ_mef)


# ── 메인 파이프라인 ─────────────────────────────────────────────────────


def run_planes_from_landmarks(
    landmarks: dict[str, np.ndarray],
    *,
    verbose: bool = False,
) -> dict:
    """dict 형태로 전달된 랜드마크에서 MSP·교합평면 계산 (파일 입출력 없음).

    외부 앱에서 자주 호출할 때 사용합니다. 예컨대:

        landmarks = {
            "N": np.array([0.0, 0.0, 120.0]),
            "ANS": np.array([0.0, -45.0, 80.0]),
            "PNS": np.array([0.0, -60.0, 82.0]),
            "LMeF": np.array([-15.0, -55.0, 30.0]),
            "RMeF": np.array([15.5, -55.0, 30.0]),
        }
        result = run_planes_from_landmarks(landmarks)

    Args:
        landmarks: label → [x, y, z] (LPS mm) 매핑
        verbose: 상세 진행 메시지 출력 여부

    Returns:
        {
            "msp": {"center": [x,y,z], "normal": [x,y,z]},
            "occlusal": {"center": [x,y,z], "normal": [x,y,z]},
            "required": [...],
            "missing": [...],
            "computed": True/False,
            "error": null or "..."
        }
    """
    result: dict = {
        "msp": {"center": _nan_vector(), "normal": _nan_vector()},
        "occlusal": {"center": _nan_vector(), "normal": _nan_vector()},
        "required": list(ALL_REQUIRED),
        "missing": [],
        "computed": False,
        "error": None,
    }

    try:
        all_present, missing = validate_pipeline_inputs(landmarks)
        result["missing"] = missing

        if not all_present:
            if verbose:
                print(f"[경고] 누락된 랜드마크: {', '.join(missing)} → 모두 NaN", flush=True)
            result["error"] = f"누락된 랜드마크: {', '.join(missing)}"
            return result

        # Cranial 랜드마크 (N, ANS, PNS)
        cranial: dict[str, np.ndarray] = {k: landmarks[k] for k in REQUIRED_CRANIAL_LANDMARKS}
        mef: dict[str, np.ndarray] = {k: landmarks[k] for k in REQUIRED_MEF_LANDMARKS}

        if verbose:
            print("MSP 계산 시작...", flush=True)

        # 1. MSP 계산
        msp_center, msp_normal = compute_msp(cranial)

        if verbose:
            print("교합평면 계산 시작...", flush=True)

        # 2. 교합평면 법선 계산
        occlusal_normal = compute_occlusal_normal(cranial["ANS"], cranial["PNS"], msp_normal)

        # 3. 교합평면 중심 계산
        occlusal_center = compute_occlusal_center(cranial["ANS"], mef, occlusal_normal)

        result["msp"]["center"] = _vector_to_list(msp_center)
        result["msp"]["normal"] = _vector_to_list(msp_normal)
        result["occlusal"]["center"] = _vector_to_list(occlusal_center)
        result["occlusal"]["normal"] = _vector_to_list(occlusal_normal)
        result["computed"] = True

        if verbose:
            print("완료.", flush=True)
        return result

    except Exception as e:
        result["error"] = f"오류: {type(e).__name__}: {e}"
        return result


def load_landmarks(input_path: Path) -> dict[str, np.ndarray]:
    """마커업 JSON 파일에서 랜드마크 추출."""
    points = _load_mrkt_json(input_path)
    landmarks = _extract_landmarks(points)
    return landmarks


def validate_pipeline_inputs(landmarks: dict[str, np.ndarray]) -> tuple[bool, list[str]]:
    """모든 필수 랜드마크 존재 여부 검증."""
    missing_cranial, _ = _validate_required(landmarks, REQUIRED_CRANIAL_LANDMARKS, context="cranial")
    missing_mef, _ = _validate_required(landmarks, REQUIRED_MEF_LANDMARKS, context="mef")
    all_missing = missing_cranial + missing_mef
    return len(all_missing) == 0, all_missing


def run_planes_pipeline(
    input_path: Optional[Path] = None,
    *,
    landmarks: Optional[dict[str, np.ndarray]] = None,
    output_path: Optional[Path] = None,
    stdout: bool = False,
    verbose: bool = False,
) -> dict:
    """통합 랜드마크 → MSP·교합평면 계산.

    파일 경로 또는 dict 로직을 직접 받아 결과를 반환합니다.
    output_path 가 있고 stdout 이 False 면 파일을 저장하고,
    stdout 이 True 면 stdout 으로 JSON 을 내보냅니다.

    사용 예:
        # 파일에서 계산
        result = run_planes_pipeline(Path("case.mrk.json"))

        # 파일에 저장
        result = run_planes_pipeline(Path("case.mrk.json"), output_path=Path("out.json"))

        # stdout 으로 바로 출력
        result = run_planes_pipeline(Path("case.mrk.json"), stdout=True)

        # dict 에서 계산 (외부 앱)
        landmarks = {"N": ..., "ANS": ..., ...}
        result = run_planes_pipeline(landmarks=landmarks)
    """
    # landmark 데이터 로드 (파일 또는 dict)
    if landmarks is None:
        if input_path is None:
            raise ValueError("input_path 또는 landmarks 중 하나가 필요합니다.")
        try:
            landmarks = load_landmarks(input_path)
            if verbose:
                print(f"로드된 랜드마크: {list(landmarks.keys())}", flush=True)
        except FileNotFoundError as e:
            return {
                "msp": {"center": _nan_vector(), "normal": _nan_vector()},
                "occlusal": {"center": _nan_vector(), "normal": _nan_vector()},
                "required": list(ALL_REQUIRED),
                "missing": [],
                "computed": False,
                "error": str(e),
            }
    else:
        if verbose:
            print(f"직접 제공된 랜드마크: {list(landmarks.keys())}", flush=True)

    all_present, missing = validate_pipeline_inputs(landmarks)
    result: dict = {
        "msp": {"center": _nan_vector(), "normal": _nan_vector()},
        "occlusal": {"center": _nan_vector(), "normal": _nan_vector()},
        "required": list(ALL_REQUIRED),
        "missing": missing,
        "computed": False,
        "error": None,
    }

    if not all_present:
        if verbose:
            print(f"[경고] 누락된 랜드마크: {', '.join(missing)} → 모두 NaN", flush=True)
        result["error"] = f"누락된 랜드마크: {', '.join(missing)}"
        _write_output(result, output_path=output_path, stdout=stdout)
        return result

    # Cranial 랜드마크 (N, ANS, PNS)
    cranial: dict[str, np.ndarray] = {k: landmarks[k] for k in REQUIRED_CRANIAL_LANDMARKS}
    mef: dict[str, np.ndarray] = {k: landmarks[k] for k in REQUIRED_MEF_LANDMARKS}

    if verbose:
        print("MSP 계산 시작...", flush=True)

    # 1. MSP 계산
    msp_center, msp_normal = compute_msp(cranial)

    if verbose:
        print("교합평면 계산 시작...", flush=True)

    # 2. 교합평면 법선 계산
    occlusal_normal = compute_occlusal_normal(cranial["ANS"], cranial["PNS"], msp_normal)

    # 3. 교합평면 중심 계산
    occlusal_center = compute_occlusal_center(cranial["ANS"], mef, occlusal_normal)

    result["msp"]["center"] = _vector_to_list(msp_center)
    result["msp"]["normal"] = _vector_to_list(msp_normal)
    result["occlusal"]["center"] = _vector_to_list(occlusal_center)
    result["occlusal"]["normal"] = _vector_to_list(occlusal_normal)
    result["computed"] = True

    if verbose:
        print("완료.", flush=True)

    _write_output(result, output_path=output_path, stdout=stdout)
    return result


def _write_output(
    result: dict,
    output_path: Optional[Path] = None,
    stdout: bool = False,
) -> None:
    """결과를 파일 또는 stdout 으로 출력.

    stdout=True 가 우선 (외부 앱에서 자주 호출할 때 stdout 에 직접 결과 반환용).
    Args:
        result: 계산 결과 딕트
        output_path: 파일로 저장할 경로
        stdout: True 인 경우 stdout 에 JSON 출력 (--stdout 지정 시 이쪽 우선)
    """
    if stdout:
        # stdout 이 우선 ── output_path 는 무시하고 바로 stdout 으로 출력
        print(json.dumps(result, indent=4, ensure_ascii=False), end="", flush=True)
    elif output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    # stdout 도 output_path 도 없는 경우 → 아무것도 출력하지 않음 (호출 측에서 필요시 처리)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="통합 랜드마크 JSON 에서 MSP·교합평면 계산"
    )
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="통합 랜드마크 .mrk.json 파일 경로"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="결과 JSON 저장 경로 (미지정 시 stdout)"
    )
    parser.add_argument(
        "--stdout", action="store_true", default=False,
        help="-o 와 함께 사용할 때 파일 대신 stdout 으로 결과 출력",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="상세 진행 메시지 출력",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.is_file():
        print(f"[오류] 파일 없음: {args.input}", file=sys.stderr, flush=True)
        return 1

    result = run_planes_pipeline(
        args.input,
        output_path=args.output,
        stdout=args.stdout,
        verbose=args.verbose,
    )

    if result["computed"]:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
