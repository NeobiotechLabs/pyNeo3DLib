"""run_integrated_pipeline.py 통합 테스트 - Stage 5(Plane) 포함."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

_ARTICUL_DIR = Path(__file__).resolve().parent.parent
if str(_ARTICUL_DIR) not in sys.path:
    sys.path.insert(0, str(_ARTICUL_DIR))

from run_integrated_pipeline import (
    PLANES_SCRIPT,
    PLANES_JSON_SUFFIX,
    LANDMARKS_MRK_SUFFIX,
    MANDIBLE_STL_SUFFIX,
    check_case_inputs,
    should_run_planes,
    build_planes_command,
    run_pipeline,
    main,
)


def _make_dummy_landmarks_mrk_file(tmp_path: Path, stem: str, landmarks: dict) -> Path:
    """랜드마크 마크업 파일을 생성."""
    points = [
        {"id": str(i + 1), "label": label, "position": pos, "orientation": [1.0] * 9, "positionStatus": "preview"}
        for i, (label, pos) in enumerate(landmarks.items())
    ]
    data = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": [{"type": "Fiducial", "coordinateSystem": "LPS", "controlPoints": points}],
    }
    path = tmp_path / f"{stem}{LANDMARKS_MRK_SUFFIX}"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _make_dummy_mandible_stl_file(tmp_path: Path, stem: str) -> Path:
    path = tmp_path / f"{stem}{MANDIBLE_STL_SUFFIX}"
    path.write_bytes(b"\x00dummy_stl_content_" + b"\x00" * 20)
    return path


def _make_dummy_planes_json_file(tmp_path: Path, stem: str, *, computed: bool, missing=None):
    if missing is None:
        missing = []
    if computed:
        data = {
            "msp": {"center": [0.0, -41.67, 94.0], "normal": [-1.0, 0.0, 0.0]},
            "occlusal": {"center": [0.38, -50.0, 55.0], "normal": [0.0, -0.03, -1.0]},
            "required": ["N", "ANS", "PNS", "LMeF", "RMeF"],
            "missing": [],
            "computed": True,
            "error": None,
        }
    else:
        data = {
            "msp": {"center": [math.nan, math.nan, math.nan], "normal": [math.nan, math.nan, math.nan]},
            "occlusal": {"center": [math.nan, math.nan, math.nan], "normal": [math.nan, math.nan, math.nan]},
            "required": ["N", "ANS", "PNS", "LMeF", "RMeF"],
            "missing": missing or ["LMeF"],
            "computed": False,
            "error": f"Missing landmarks: {', '.join(missing)}",
        }
    path = tmp_path / f"{stem}{PLANES_JSON_SUFFIX}"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _build_runner_with_result(tmp_path: Path, *, computed: bool, missing=None):
    """runner 생성 — plane 계산 시 JSON 결과 파일을 항상 생성."""
    if missing is None:
        missing = []

    def runner(cmd, verbose=False):
        if PLANES_SCRIPT.name in " ".join(cmd):
            _make_dummy_planes_json_file(tmp_path, "case01", computed=computed, missing=missing)
            return 0
        return 0

    return runner


class TestShouldRunPlanes:
    def test_should_run_when_landmarks_exist(self, tmp_path):
        (tmp_path / f"case01{LANDMARKS_MRK_SUFFIX}").touch()
        assert should_run_planes("case01", tmp_path) is True

    def test_skip_when_planes_already_done(self, tmp_path):
        (tmp_path / f"case01{LANDMARKS_MRK_SUFFIX}").touch()
        (tmp_path / f"case01{PLANES_JSON_SUFFIX}").touch()
        assert should_run_planes("case01", tmp_path) is False

    def test_force_rebuild(self, tmp_path):
        (tmp_path / f"case01{LANDMARKS_MRK_SUFFIX}").touch()
        (tmp_path / f"case01{PLANES_JSON_SUFFIX}").touch()
        assert should_run_planes("case01", tmp_path, force=True) is True

    def test_no_landmarks_skip(self, tmp_path):
        assert should_run_planes("case01", tmp_path) is False


class TestBuildPlanesCommand:
    def test_generates_correct_command(self, tmp_path):
        (tmp_path / f"case01{LANDMARKS_MRK_SUFFIX}").touch()
        cmd = build_planes_command("case01", tmp_path)
        assert cmd[0] == sys.executable
        assert str(PLANES_SCRIPT) in cmd
        input_arg = tmp_path / f"case01{LANDMARKS_MRK_SUFFIX}"
        output_arg = tmp_path / f"case01{PLANES_JSON_SUFFIX}"
        assert str(input_arg) in cmd
        assert str(output_arg) in cmd


class TestExceptionHandling:
    def test_partial_landmarks_nan_result(self, tmp_path):
        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()
        _make_dummy_mandible_stl_file(tmp_path, "case01")
        _make_dummy_landmarks_mrk_file(tmp_path, "case01", {"N": [0, 0, 10], "ANS": [0, -2, 5], "PNS": [0, 2, 5]})
        runner = _build_runner_with_result(tmp_path, computed=False, missing=["LMeF", "RMeF"])
        exit_code, failures = run_pipeline(dcm_dir, tmp_path, skip_seg=True, skip_planes=False, force=True, runner=runner)
        assert exit_code == 1
        assert "case01" in failures["plane"]

    def test_all_landmarks_present_success(self, tmp_path):
        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()
        _make_dummy_mandible_stl_file(tmp_path, "case01")
        _make_dummy_landmarks_mrk_file(tmp_path, "case01", {"N": [0, 0, 10], "ANS": [0, -2, 5], "PNS": [0, 2, 5], "LMeF": [-3, 0, 2], "RMeF": [3, 0, 2]})
        runner = _build_runner_with_result(tmp_path, computed=True, missing=[])
        exit_code, failures = run_pipeline(dcm_dir, tmp_path, skip_seg=True, skip_planes=False, force=True, runner=runner)
        assert exit_code == 0
        assert failures == {"seg": [], "condyle": [], "canal": [], "merge": [], "plane": []}

    def test_plane_calculation_fail(self, tmp_path):
        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()
        _make_dummy_mandible_stl_file(tmp_path, "case01")
        _make_dummy_landmarks_mrk_file(tmp_path, "case01", {"N": [0, 0, 10], "ANS": [0, -2, 5], "PNS": [0, 2, 5], "LMeF": [-3, 0, 2], "RMeF": [3, 0, 2]})

        def fail_runner(cmd, verbose=False):
            return 1 if PLANES_SCRIPT.name in " ".join(cmd) else 0

        exit_code, failures = run_pipeline(dcm_dir, tmp_path, skip_seg=True, skip_planes=False, force=True, runner=fail_runner)
        assert exit_code == 1
        assert "case01" in failures["plane"]

    def test_invalid_json_handling(self, tmp_path):
        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()
        _make_dummy_mandible_stl_file(tmp_path, "case01")
        _make_dummy_landmarks_mrk_file(tmp_path, "case01", {"N": [0, 0, 10], "ANS": [0, -2, 5], "PNS": [0, 2, 5], "LMeF": [-3, 0, 2], "RMeF": [3, 0, 2]})

        def bad_json_runner(cmd, verbose=False):
            if PLANES_SCRIPT.name in " ".join(cmd):
                bad_json_path = tmp_path / f"case01{PLANES_JSON_SUFFIX}"
                bad_json_path.write_text("not valid json {{{", encoding="utf-8")
                return 0
            return 0

        exit_code, failures = run_pipeline(dcm_dir, tmp_path, skip_seg=True, skip_planes=False, force=True, runner=bad_json_runner)
        assert exit_code == 1
        assert "case01" in failures["plane"]

    def test_graceful_continue_on_failure(self, tmp_path):
        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()

        # case01: 성공
        _make_dummy_mandible_stl_file(tmp_path, "case01")
        _make_dummy_landmarks_mrk_file(tmp_path, "case01", {"N": [0, 0, 10], "ANS": [0, -2, 5], "PNS": [0, 2, 5], "LMeF": [-3, 0, 2], "RMeF": [3, 0, 2]})

        # case02: 실패
        _make_dummy_mandible_stl_file(tmp_path, "case02")
        _make_dummy_landmarks_mrk_file(tmp_path, "case02", {"N": [0, 0, 10], "ANS": [0, -2, 5], "PNS": [0, 2, 5], "LMeF": [-3, 0, 2], "RMeF": [3, 0, 2]})

        def fail_runner(cmd, verbose=False):
            return 1 if PLANES_SCRIPT.name in " ".join(cmd) else 0

        exit_code, failures = run_pipeline(dcm_dir, tmp_path, skip_seg=True, skip_planes=False, force=True, runner=fail_runner)
        assert exit_code == 1
        assert failures["plane"] == ["case01", "case02"]

    def test_missing_landmarks_skips_plane(self, tmp_path):
        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()
        exit_code, failures = run_pipeline(dcm_dir, tmp_path, skip_seg=True, skip_planes=False, runner=lambda _cmd, _v: 0)
        assert exit_code == 2
        assert len(failures["plane"]) == 0


class TestCliMain:
    def test_help_output(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_missing_input_directory(self, tmp_path):
        non_existent = tmp_path / "no_such_folder"
        code = main(["-i", str(non_existent), "-o", str(tmp_path / "out")])
        assert code == 2


class TestCheckCaseInputs:
    def test_landmarks_mrk_check(self, tmp_path):
        (tmp_path / f"case01{LANDMARKS_MRK_SUFFIX}").touch()
        inputs = check_case_inputs("case01", tmp_path)
        assert inputs["mandible_stl"] is False
        assert inputs["canal_stl"] is False
        assert inputs["merged_mrk"] is False
        assert inputs["landmarks_mrk"] is True

    def test_all_files_present(self, tmp_path):
        (tmp_path / f"case01_mandible.stl").touch()
        (tmp_path / f"case01_nerve_canal.stl").touch()
        (tmp_path / f"case01_merged.mrk.json").touch()
        (tmp_path / f"case01_landmarks.mrk.json").touch()
        inputs = check_case_inputs("case01", tmp_path)
        assert all(inputs.values())
