"""registration_bridge 단위 테스트.

브리지 모듈은 stdlib-only 이므로 무거운 의존성을 로드하지 않는다.
모든 테스트는 tmp_path 기반으로 파일 I/O 만 검증한다.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# ── 임포트 준비 ────────────────────────────────────────────────────────
_ARTICUL_DIR = Path(__file__).resolve().parent.parent  # .../articulPipeline/
if str(_ARTICUL_DIR) not in sys.path:
    sys.path.insert(0, str(_ARTICUL_DIR))

from registration_bridge import (
    INTEGRATED_SCRIPT,
    LANDMARKS_MRK_SUFFIX,
    PLANES_JSON_SUFFIX,
    build_articul_command,
    collect_case_artifacts,
    collect_plane_results,
    plane_failure_message,
    plane_success_message,
    planes_summary,
    print_case_artifacts,
    print_plane_results,
)

#: 세그멘테이션 메쉬 STL 접미어 (structure_names.json 규약)
_MESH_SUFFIXES = {
    "maxillary_sinus": "_maxillary_sinus.stl",
    "maxilla": "_maxilla.stl",
    "mandible": "_mandible.stl",
    "nerve_canal": "_nerve_canal.stl",
}


def _touch_meshes(tmp_path: Path, stem: str, structures=_MESH_SUFFIXES.keys()) -> None:
    for key in structures:
        (tmp_path / f"{stem}{_MESH_SUFFIXES[key]}").touch()


def _write_planes(tmp_path: Path, stem: str, data: dict) -> Path:
    p = tmp_path / f"{stem}{PLANES_JSON_SUFFIX}"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def _ok_planes_data() -> dict:
    return {
        "msp": {"center": [1.0, 2.0, 3.0], "normal": [0.0, 1.0, 0.0]},
        "occlusal": {"center": [4.0, 5.0, 6.0], "normal": [0.0, 0.0, 1.0]},
        "required": ["N", "ANS", "PNS", "LMeF", "RMeF"],
        "missing": [],
        "computed": True,
        "error": None,
    }


# ── build_articul_command ──────────────────────────────────────────────

def test_build_articul_command_contains_input_output():
    cmd = build_articul_command("C:/data/dicom", "C:/data/results")
    assert cmd[1] == str(INTEGRATED_SCRIPT)
    assert cmd[2] == "--input"
    assert cmd[3] == "C:/data/dicom"
    assert cmd[4] == "--output"
    assert cmd[5] == "C:/data/results"


def test_build_articul_command_accepts_path_objects():
    cmd = build_articul_command(Path("in"), Path("out"))
    assert cmd[3] == "in"
    assert cmd[5] == "out"


# ── collect_plane_results ──────────────────────────────────────────────

def test_collect_plane_results_reads_valid_files(tmp_path):
    _write_planes(tmp_path, "case01", _ok_planes_data())

    planes = collect_plane_results(tmp_path)

    assert len(planes) == 1
    assert planes[0]["case"] == "case01"
    assert planes[0]["computed"] is True
    assert planes[0]["msp"]["center"] == [1.0, 2.0, 3.0]
    assert planes[0]["planes_file"] == str(tmp_path / f"case01{PLANES_JSON_SUFFIX}")


def test_collect_plane_results_sorted_and_skips_broken(tmp_path):
    _write_planes(tmp_path, "caseB", _ok_planes_data())
    _write_planes(tmp_path, "caseA", _ok_planes_data())
    # 깨진 JSON — 건너뛰어야 함
    (tmp_path / f"broken{PLANES_JSON_SUFFIX}").write_text("{not json", encoding="utf-8")
    # dict 가 아닌 JSON — 건너뛰어야 함
    (tmp_path / f"list{PLANES_JSON_SUFFIX}").write_text("[1, 2]", encoding="utf-8")

    planes = collect_plane_results(tmp_path)

    assert [p["case"] for p in planes] == ["caseA", "caseB"]


def test_collect_plane_results_missing_dir(tmp_path):
    assert collect_plane_results(tmp_path / "not_there") == []


def test_collect_plane_results_does_not_mutate_file(tmp_path):
    _write_planes(tmp_path, "case01", _ok_planes_data())
    collect_plane_results(tmp_path)
    on_disk = json.loads(
        (tmp_path / f"case01{PLANES_JSON_SUFFIX}").read_text(encoding="utf-8")
    )
    assert "case" not in on_disk


# ── collect_case_artifacts ─────────────────────────────────────────────

def test_collect_case_artifacts_full_case(tmp_path):
    _touch_meshes(tmp_path, "case01")
    landmarks = tmp_path / f"case01{LANDMARKS_MRK_SUFFIX}"
    landmarks.touch()
    planes_file = _write_planes(tmp_path, "case01", _ok_planes_data())

    artifacts = collect_case_artifacts(tmp_path)

    assert len(artifacts) == 1
    item = artifacts[0]
    assert item["case"] == "case01"
    assert item["meshes"]["maxillary_sinus"] == str(tmp_path / "case01_maxillary_sinus.stl")
    assert item["meshes"]["maxilla"] == str(tmp_path / "case01_maxilla.stl")
    assert item["meshes"]["mandible"] == str(tmp_path / "case01_mandible.stl")
    assert item["meshes"]["nerve_canal"] == str(tmp_path / "case01_nerve_canal.stl")
    assert item["landmarks_file"] == str(landmarks)
    assert item["planes_file"] == str(planes_file)


def test_collect_case_artifacts_partial_meshes_none(tmp_path):
    # 하악골·상악골만 생성된 경우 — 나머지는 None 이어야 함
    _touch_meshes(tmp_path, "case02", structures=("mandible", "maxilla"))

    artifacts = collect_case_artifacts(tmp_path)

    assert len(artifacts) == 1
    item = artifacts[0]
    assert item["case"] == "case02"
    assert item["meshes"]["mandible"] == str(tmp_path / "case02_mandible.stl")
    assert item["meshes"]["maxilla"] == str(tmp_path / "case02_maxilla.stl")
    assert item["meshes"]["maxillary_sinus"] is None
    assert item["meshes"]["nerve_canal"] is None
    assert item["landmarks_file"] is None
    assert item["planes_file"] is None


def test_collect_case_artifacts_sorted_multi_case(tmp_path):
    _touch_meshes(tmp_path, "caseB")
    _touch_meshes(tmp_path, "caseA")

    artifacts = collect_case_artifacts(tmp_path)

    assert [a["case"] for a in artifacts] == ["caseA", "caseB"]


def test_collect_case_artifacts_empty_and_missing_dir(tmp_path):
    assert collect_case_artifacts(tmp_path) == []
    assert collect_case_artifacts(tmp_path / "not_there") == []


def test_collect_case_artifacts_ignores_unrelated_files(tmp_path):
    (tmp_path / "case01_merged.mrk.json").touch()
    (tmp_path / "case01_planes.json.bak").touch()
    (tmp_path / "notes.txt").touch()

    assert collect_case_artifacts(tmp_path) == []


def test_collect_case_artifacts_json_serializable(tmp_path):
    _touch_meshes(tmp_path, "case01")
    (tmp_path / f"case01{LANDMARKS_MRK_SUFFIX}").touch()
    _write_planes(tmp_path, "case01", _ok_planes_data())

    json.dumps(collect_case_artifacts(tmp_path), allow_nan=False)


# ── print_plane_results ────────────────────────────────────────────────

def test_print_plane_results_computed(tmp_path, capsys):
    planes = collect_plane_results(_write_planes(tmp_path, "case01", _ok_planes_data()).parent)
    print_plane_results(planes)
    out = capsys.readouterr().out

    assert "case01" in out
    assert "시상정중면(MSP)" in out
    assert "교합평면" in out
    assert "1.0000, 2.0000, 3.0000" in out  # msp 중심
    assert "0.0000, 0.0000, 1.0000" in out  # occlusal 법선


def test_print_plane_results_missing_landmarks(tmp_path, capsys):
    data = _ok_planes_data()
    data["computed"] = False
    data["missing"] = ["LMeF", "RMeF"]
    data["msp"] = {"center": [math.nan] * 3, "normal": [math.nan] * 3}
    data["occlusal"] = {"center": [math.nan] * 3, "normal": [math.nan] * 3}
    planes = collect_plane_results(_write_planes(tmp_path, "case02", data).parent)

    print_plane_results(planes)
    out = capsys.readouterr().out

    assert "평면 계산 실패" in out
    assert "LMeF, RMeF" in out


def test_print_plane_results_empty(capsys):
    print_plane_results([])
    out = capsys.readouterr().out
    assert "없습니다" in out


# ── print_case_artifacts ───────────────────────────────────────────────

def test_print_case_artifacts_full(tmp_path, capsys):
    _touch_meshes(tmp_path, "case01")
    (tmp_path / f"case01{LANDMARKS_MRK_SUFFIX}").touch()
    _write_planes(tmp_path, "case01", _ok_planes_data())
    artifacts = collect_case_artifacts(tmp_path)

    print_case_artifacts(artifacts)
    out = capsys.readouterr().out

    assert "case01" in out
    assert "상악동(maxillary_sinus)" in out
    assert "상악골(maxilla)" in out
    assert "하악골(mandible)" in out
    assert "신경관(nerve_canal)" in out
    assert str(tmp_path / "case01_mandible.stl") in out
    assert str(tmp_path / f"case01{LANDMARKS_MRK_SUFFIX}") in out
    assert str(tmp_path / f"case01{PLANES_JSON_SUFFIX}") in out


def test_print_case_artifacts_partial_shows_missing(tmp_path, capsys):
    _touch_meshes(tmp_path, "case02", structures=("mandible",))

    print_case_artifacts(collect_case_artifacts(tmp_path))
    out = capsys.readouterr().out

    assert str(tmp_path / "case02_mandible.stl") in out
    assert "없음" in out  # 누락 메쉬/랜드마크/평면 표기


def test_print_case_artifacts_empty(capsys):
    print_case_artifacts([])
    out = capsys.readouterr().out
    assert "없습니다" in out


# ── planes_summary ─────────────────────────────────────────────────────

def test_planes_summary_keeps_center_normal():
    data = _ok_planes_data()
    data["case"] = "case01"
    summary = planes_summary([data])

    assert summary[0]["case"] == "case01"
    assert summary[0]["computed"] is True
    assert summary[0]["msp"]["center"] == [1.0, 2.0, 3.0]
    assert summary[0]["occlusal"]["normal"] == [0.0, 0.0, 1.0]
    assert "missing" not in summary[0]


def test_planes_summary_nan_becomes_null():
    data = {
        "case": "case02",
        "computed": False,
        "missing": ["N"],
        "error": "누락된 랜드마크: N",
        "msp": {"center": [math.nan, math.nan, math.nan], "normal": [math.nan] * 3},
        "occlusal": {"center": [math.nan] * 3, "normal": [math.nan] * 3},
    }
    summary = planes_summary([data])

    assert summary[0]["computed"] is False
    assert summary[0]["msp"]["center"] == [None, None, None]
    assert summary[0]["missing"] == ["N"]
    assert summary[0]["error"] == "누락된 랜드마크: N"
    # JSON 직렬화 가능해야 함 (NaN 이 없으므로 allow_nan 불필요)
    json.dumps(summary, allow_nan=False)


# ── 메시지 빌더 ────────────────────────────────────────────────────────

def test_plane_success_message_shape():
    data = _ok_planes_data()
    data["case"] = "case01"
    msg = plane_success_message([data], "C:/out")

    assert msg["type"] == "articul_planes_completed"
    assert msg["output_dir"] == "C:/out"
    assert len(msg["results"]) == 1
    assert "timestamp" in msg
    # artifacts 미전달 시 키 자체가 없어야 함 (하위 호환)
    assert "artifacts" not in msg


def test_plane_success_message_with_artifacts(tmp_path):
    _touch_meshes(tmp_path, "case01")
    (tmp_path / f"case01{LANDMARKS_MRK_SUFFIX}").touch()
    _write_planes(tmp_path, "case01", _ok_planes_data())
    artifacts = collect_case_artifacts(tmp_path)
    planes = collect_plane_results(tmp_path)

    msg = plane_success_message(planes, str(tmp_path), artifacts)

    assert msg["type"] == "articul_planes_completed"
    assert len(msg["results"]) == 1
    assert msg["artifacts"] == artifacts
    item = msg["artifacts"][0]
    assert item["meshes"]["maxillary_sinus"].endswith("case01_maxillary_sinus.stl")
    assert item["landmarks_file"].endswith(f"case01{LANDMARKS_MRK_SUFFIX}")
    assert item["planes_file"].endswith(f"case01{PLANES_JSON_SUFFIX}")
    # 외부 전송 직렬화 가능 확인
    json.dumps(msg, allow_nan=False)


def test_plane_success_message_with_artifacts_but_no_planes(tmp_path):
    # 평면 계산 실패(computed=false) 해도 산출물 경로는 전송되어야 함
    _touch_meshes(tmp_path, "case01")
    artifacts = collect_case_artifacts(tmp_path)

    msg = plane_success_message([], str(tmp_path), artifacts)

    assert msg["results"] == []
    assert msg["artifacts"] == artifacts
    json.dumps(msg, allow_nan=False)


def test_plane_failure_message_shape():
    msg = plane_failure_message("세그멘테이션 실패", "C:/out", exit_code=2)

    assert msg["type"] == "articul_planes_failed"
    assert msg["error"] == "세그멘테이션 실패"
    assert msg["exit_code"] == 2


def test_plane_failure_message_without_exit_code():
    msg = plane_failure_message("오류", "C:/out")
    assert "exit_code" not in msg
