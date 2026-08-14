"""정합(registration) 완료 후 통합 파이프라인 실행 훅.

``pyNeo3DLib.registration.Neo3DRegistration`` 의 정합이 완료된 뒤

    --input  : 입력 JSON 의 cbct.path (CBCT DICOM 시리즈 폴더)
    --output : 입력 JSON 의 pipeline_results.path

로 ``run_integrated_pipeline.py`` 를 서브프로세스 실행하고, 생성된
산출물을 수집해 콘솔 프린팅 + WebSocket 등으로 외부 전송할 수 있는
JSON 메시지를 만든다.

외부로 내보내는 최종 산출물 (케이스별):
- 생성 메쉬 STL 경로 : 상악동/상악골/하악골/신경관
- 통합 랜드마크 경로 : ``{케이스}_landmarks.mrk.json``
- 평면 결과 JSON 경로 : ``{케이스}_planes.json`` (시상정중면 · 교합평면)

이 모듈은 stdlib 만 사용한다 (torch/numpy 등 무거운 의존성 임포트 금지).
"""
from __future__ import annotations

import datetime
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Union

# ── 상수 ────────────────────────────────────────────────────────────────

#: 통합 파이프라인 오케스트레이터 스크립트
INTEGRATED_SCRIPT = Path(__file__).resolve().parent / "run_integrated_pipeline.py"

#: 평면 계산 결과 파일 접미어 (run_integrated_pipeline.PLANES_JSON_SUFFIX 와 동일)
PLANES_JSON_SUFFIX = "_planes.json"

#: 통합 랜드마크 파일 접미어 (run_integrated_pipeline.LANDMARKS_MRK_SUFFIX 와 동일)
LANDMARKS_MRK_SUFFIX = "_landmarks.mrk.json"

#: 세그멘테이션 생성 메쉬 STL 접미어 (structure_names.json 이름 규약과 동일)
_MESH_SUFFIXES = {
    "maxillary_sinus": "_maxillary_sinus.stl",
    "maxilla": "_maxilla.stl",
    "mandible": "_mandible.stl",
    "nerve_canal": "_nerve_canal.stl",
}

#: 메쉬 콘솔 출력용 표시 이름
_MESH_LABELS = {
    "maxillary_sinus": "상악동(maxillary_sinus)",
    "maxilla": "상악골(maxilla)",
    "mandible": "하악골(mandible)",
    "nerve_canal": "신경관(nerve_canal)",
}

#: 외부로 보고할 평면 키와 표시 이름
_PLANE_KEYS = ("msp", "occlusal")
_PLANE_LABELS = {"msp": "시상정중면(MSP)", "occlusal": "교합평면"}


# ── 명령어 빌드 · 실행 ──────────────────────────────────────────────────

def build_articul_command(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> list[str]:
    """통합 파이프라인 실행 명령어 빌드 (--input/--output)."""
    return [
        sys.executable,
        str(INTEGRATED_SCRIPT),
        "--input", str(input_dir),
        "--output", str(output_dir),
    ]


def _child_env() -> dict[str, str]:
    """자식 프로세스 환경변수 — GPU 숨김 제거 + UTF-8 강제.

    서버(fastserver) 는 임포트 시 ``CUDA_VISIBLE_DEVICES="-1"`` 을 설정해
    자식 프로세스에 GPU 가 보이지 않는다. 세그멘테이션은 GPU 전용이므로
    값을 완전히 숨기는 경우('-1', '') 만 제거하고, 정상적인 디바이스 선택
    값('0', '0,1' 등)은 그대로 둔다.
    """
    env = os.environ.copy()
    if env.get("CUDA_VISIBLE_DEVICES", "").strip() in ("-1", ""):
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def run_articul_pipeline(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> int:
    """통합 파이프라인을 서브프로세스로 실행하고 exit code 반환.

    긴 작업(세그멘테이션 등) 이므로 블로킹 호출이다 — 호출 측은 스레드
    (``asyncio.to_thread``) 에서 실행하는 것을 권장한다. 자식 프로세스의
    stdout/stderr 는 그대로 부모 콘솔로 상속된다.
    """
    cmd = build_articul_command(input_dir, output_dir)
    print(f"  ▶ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=_child_env())
    return result.returncode


# ── 결과 수집 · 출력 ────────────────────────────────────────────────────

def collect_plane_results(output_dir: Union[str, Path]) -> list[dict]:
    """``output_dir`` 의 ``*_planes.json`` 을 정렬 수집해 리스트로 반환.

    각 항목은 planes JSON 원본에 ``case`` (파일 stem) 와 ``planes_file``
    (절대 경로) 을 추가한 dict 이다. 읽기/파싱에 실패한 파일은 건너뛴다.
    """
    planes: list[dict] = []
    out = Path(output_dir)
    if not out.is_dir():
        return planes

    for path in sorted(out.glob(f"*{PLANES_JSON_SUFFIX}")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        data = dict(data)
        data["case"] = path.name[: -len(PLANES_JSON_SUFFIX)]
        data["planes_file"] = str(path)
        planes.append(data)
    return planes


def collect_case_artifacts(output_dir: Union[str, Path]) -> list[dict]:
    """``output_dir`` 에서 케이스별 세그멘테이션 산출물 경로를 수집한다.

    케이스 발견은 메쉬 STL 4 종(상악동/상악골/하악골/신경관) glob 의
    합집합으로 하며, 각 케이스마다 다음 경로를 담는다:

    - ``meshes`` : 구조 이름 → STL 절대 경로 (없으면 None)
    - ``landmarks_file`` : 통합 랜드마크 ``*_landmarks.mrk.json`` 경로
    - ``planes_file`` : 평면 결과 ``*_planes.json`` 경로 (시상정중면 · 교합평면)

    산출물이 하나도 없으면(세그멘테이션 실패 등) 빈 리스트를 반환한다.
    """
    out = Path(output_dir)
    if not out.is_dir():
        return []

    stems: set = set()
    for suffix in _MESH_SUFFIXES.values():
        for path in out.glob(f"*{suffix}"):
            if path.is_file():
                stems.add(path.name[: -len(suffix)])

    artifacts: list[dict] = []
    for stem in sorted(stems):
        meshes: dict = {}
        for key, suffix in _MESH_SUFFIXES.items():
            path = out / f"{stem}{suffix}"
            meshes[key] = str(path) if path.is_file() else None

        landmarks_path = out / f"{stem}{LANDMARKS_MRK_SUFFIX}"
        planes_path = out / f"{stem}{PLANES_JSON_SUFFIX}"
        artifacts.append({
            "case": stem,
            "meshes": meshes,
            "landmarks_file": str(landmarks_path) if landmarks_path.is_file() else None,
            "planes_file": str(planes_path) if planes_path.is_file() else None,
        })
    return artifacts


def print_case_artifacts(artifacts: list[dict]) -> None:
    """케이스별 생성 메쉬 · 통합 랜드마크 · 평면 JSON 경로를 콘솔에 프린팅."""
    line = "═" * 60
    print(f"\n{line}", flush=True)
    print(" 통합 파이프라인 산출물 — 메쉬 · 통합 랜드마크 · 평면", flush=True)
    print(line, flush=True)

    if not artifacts:
        print("  생성된 세그멘테이션 산출물(STL) 이 없습니다.", flush=True)

    for data in artifacts:
        print(f" Case: {data.get('case', 'unknown')}", flush=True)
        for key, label in _MESH_LABELS.items():
            path = (data.get("meshes") or {}).get(key)
            print(f"   {label} : {path or '없음'}", flush=True)
        print(f"   통합 랜드마크 : {data.get('landmarks_file') or '없음'}", flush=True)
        print(f"   평면 결과(MSP·교합평면) : {data.get('planes_file') or '없음'}", flush=True)
    print(line, flush=True)


def _fmt_vec(vec: Any) -> str:
    """[x, y, z] 벡터 표시 문자열 (NaN 도 그대로 표시)."""
    try:
        return "[" + ", ".join(f"{float(v):.4f}" for v in vec) + "]"
    except (TypeError, ValueError):
        return str(vec)


def print_plane_results(planes: list[dict]) -> None:
    """교합평면 · 시상정중면(MSP) 중심/법선 벡터를 콘솔에 프린팅."""
    line = "═" * 60
    print(f"\n{line}", flush=True)
    print(" 통합 파이프라인 결과 — 교합평면 · 시상정중면(MSP)", flush=True)
    print(line, flush=True)

    if not planes:
        print(f"  평면 결과 파일(*{PLANES_JSON_SUFFIX}) 이 없습니다.", flush=True)

    for data in planes:
        print(f" Case: {data.get('case', 'unknown')}", flush=True)
        if not data.get("computed", False):
            missing = data.get("missing") or []
            detail = (
                f"missing: {', '.join(map(str, missing))}"
                if missing
                else (data.get("error") or "computed=false")
            )
            print(f"   평면 계산 실패 ({detail}) → NaN 표기", flush=True)
            continue
        for key in _PLANE_KEYS:
            plane = data.get(key) or {}
            print(f"   {_PLANE_LABELS[key]}", flush=True)
            print(f"     중심 : {_fmt_vec(plane.get('center'))}", flush=True)
            print(f"     법선 : {_fmt_vec(plane.get('normal'))}", flush=True)
    print(line, flush=True)


def _json_safe_vec(vec: Any) -> Optional[list]:
    """JSON 전송용 벡터 정리 — NaN 은 null 로 변환 (JSON 표준 호환)."""
    if not isinstance(vec, (list, tuple)):
        return None
    cleaned: list = []
    for v in vec:
        try:
            f = float(v)
        except (TypeError, ValueError):
            cleaned.append(None)
            continue
        cleaned.append(None if math.isnan(f) else f)
    return cleaned


def planes_summary(planes: list[dict]) -> list[dict]:
    """외부 전송(WebSocket)용 요약 — case/computed/중심/법선 만 남긴다."""
    summary: list[dict] = []
    for data in planes:
        item: dict[str, Any] = {
            "case": data.get("case"),
            "computed": bool(data.get("computed", False)),
        }
        for key in _PLANE_KEYS:
            plane = data.get(key) or {}
            item[key] = {
                "center": _json_safe_vec(plane.get("center")),
                "normal": _json_safe_vec(plane.get("normal")),
            }
        if not item["computed"]:
            item["missing"] = data.get("missing") or []
            if data.get("error"):
                item["error"] = data["error"]
        summary.append(item)
    return summary


# ── WebSocket 메시지 빌더 ───────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def plane_success_message(
    planes: list[dict],
    output_dir: Union[str, Path],
    artifacts: Optional[list[dict]] = None,
) -> dict:
    """통합 파이프라인 완료 시 외부 전송 메시지.

    ``artifacts`` (``collect_case_artifacts`` 결과) 가 주어지면 케이스별
    생성 메쉬 STL · 통합 랜드마크 · 평면 JSON 파일 경로가 ``artifacts``
    키로 함께 포함된다.
    """
    msg: dict[str, Any] = {
        "type": "articul_planes_completed",
        "results": planes_summary(planes),
        "output_dir": str(output_dir),
        "timestamp": _timestamp(),
    }
    if artifacts is not None:
        msg["artifacts"] = artifacts
    return msg


def plane_failure_message(
    error: Any,
    output_dir: Union[str, Path],
    exit_code: Optional[int] = None,
) -> dict:
    """평면 계산 실패 시 외부 전송 메시지."""
    msg: dict[str, Any] = {
        "type": "articul_planes_failed",
        "error": str(error),
        "output_dir": str(output_dir),
        "timestamp": _timestamp(),
    }
    if exit_code is not None:
        msg["exit_code"] = exit_code
    return msg
