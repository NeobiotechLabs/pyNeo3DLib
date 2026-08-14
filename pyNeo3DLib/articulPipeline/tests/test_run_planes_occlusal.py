"""run_planes.py 단위 테스트 — stdout 파일 입출력 외부화."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import numpy as np

# ── 모듈 경로 추가 (occlusal_plane 서브디렉토리) ────────────────────────
_OCP_DIR = Path(__file__).resolve().parent.parent / "occlusal_plane"  # .../articulPlane/occlusal_plane/
if str(_OCP_DIR) not in sys.path:
    sys.path.insert(0, str(_OCP_DIR))

from run_planes import (
    ALL_REQUIRED,
    REQUIRED_CRANIAL_LANDMARKS,
    REQUIRED_MEF_LANDMARKS,
    run_planes_from_landmarks,
    run_planes_pipeline,
    main,
)


# ── 헬퍼 ─────────────────────────────────────────────────────────────────

def _make_dummy_mrk_file(tmp_path: Path, stem: str, landmarks: dict) -> Path:
    """landmarks 마크업 더미 .mrk.json 생성."""
    points = [
        {
            "id": str(i + 1),
            "label": label,
            "position": pos,
            "orientation": [1.0] * 9,
            "positionStatus": "preview",
        }
        for i, (label, pos) in enumerate(landmarks.items())
    ]
    data = {
        "@schema": "test",
        "markups": [{"type": "Fiducial", "coordinateSystem": "LPS", "controlPoints": points}],
    }
    path = tmp_path / f"{stem}_landmarks.mrk.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _full_landmarks_dict() -> dict[str, list[float]]:
    return {"N": [0.5, -20.0, 120.0], "ANS": [0.5, -45.0, 80.0], "PNS": [0.5, -60.0, 82.0], "LMeF": [-15.0, -55.0, 30.0], "RMeF": [15.5, -55.0, 30.0]}


# ── A. run_planes_from_landmarks (dict → dict, 파일 I/O 없음) ─────────────

class TestRunPlanesFromLandmarks:
    def test_all_present(self):
        landmarks = {k: np.array(v, dtype=np.float64) for k, v in _full_landmarks_dict().items()}
        result = run_planes_from_landmarks(landmarks)
        assert result["computed"] is True
        assert result["missing"] == []
        assert result["error"] is None
        # NaN 이 아닌 유효한 값인지 확인
        for key in ("msp", "occlusal"):
            for subkey in ("center", "normal"):
                vals = result[key][subkey]
                assert all(not math.isnan(v) for v in vals), f"{key}.{subkey} should be valid"

    def test_missing_lme_f(self):
        """LMeF 누락 → computed=false, 모든 평면 NaN."""
        landmarks = {"N": np.array([0, 0, 10]), "ANS": np.array([0, -2, 5]), "PNS": np.array([0, 2, 5]), "RMeF": np.array([3, 0, 2])}
        result = run_planes_from_landmarks(landmarks)
        assert result["computed"] is False
        assert "LMeF" in result["missing"]
        assert "RMeF" in result["missing"] or "LMeF" in result["missing"]

    def test_missing_two_cranial(self):
        """두개골 랜드마크 두 개 이상 누락 → 계산 불가."""
        landmarks = {"N": np.array([0, 0, 10])}  # ANS, PNS, LMeF, RMeF 모두 없음
        result = run_planes_from_landmarks(landmarks)
        assert result["computed"] is False
        assert "ANS" in result["missing"]
        assert "PNS" in result["missing"]

    def test_empty_landmarks(self):
        """랜드마크가 비어 있으면 모두 누락."""
        result = run_planes_from_landmarks({})
        assert result["computed"] is False
        assert set(result["missing"]) == set(ALL_REQUIRED)

    def test_returns_numpy_arrays(self):
        """numpy.ndarray 형태의 landmark 도 정상 동작."""
        landmarks = {k: np.array(v, dtype=np.float64) for k, v in _full_landmarks_dict().items()}
        result = run_planes_from_landmarks(landmarks)
        assert result["computed"] is True

    def test_with_verbose_output(self, capsys):
        """verbose=True 시 stdout 에 출력."""
        landmarks = {k: np.array(v, dtype=np.float64) for k, v in _full_landmarks_dict().items()}
        run_planes_from_landmarks(landmarks, verbose=True)
        captured = capsys.readouterr()
        assert "MSP 계산 시작" in captured.out
        assert "교합평면 계산" in captured.out


# ── B. run_planes_pipeline (파일 또는 dict 지원) ──────────────────────────

class TestRunPlanesPipeline:
    def test_from_file_path(self, tmp_path: Path):
        mrk = _make_dummy_mrk_file(tmp_path, "case01", _full_landmarks_dict())
        result = run_planes_pipeline(mrk)
        assert result["computed"] is True
        assert result["missing"] == []

    def test_from_dict(self):
        landmarks = {k: np.array(v, dtype=np.float64) for k, v in _full_landmarks_dict().items()}
        result = run_planes_pipeline(landmarks=landmarks)
        assert result["computed"] is True
        assert result["missing"] == []

    def test_to_file(self, tmp_path: Path):
        mrk = _make_dummy_mrk_file(tmp_path, "case01", _full_landmarks_dict())
        out = tmp_path / "planes.json"
        result = run_planes_pipeline(mrk, output_path=out)
        assert out.is_file()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["computed"] is True

    def test_to_stdout(self, tmp_path: Path, capsys):
        mrk = _make_dummy_mrk_file(tmp_path, "case01", _full_landmarks_dict())
        result = run_planes_pipeline(mrk, stdout=True)
        captured = capsys.readouterr()
        assert '"computed": true' in captured.out
        assert result["computed"] is True

    def test_file_not_found(self, tmp_path: Path):
        non_existent = tmp_path / "no_such.mrk.json"
        result = run_planes_pipeline(non_existent)
        assert result["computed"] is False
        # FileNotFoundError 는 graceful 하게 처리 (에러 메시지에 파일 경로 포함)
        assert "no_such.mrk.json" in result.get("error", "")

    def test_no_input_specified_raises(self):
        with pytest.raises(ValueError, match="input_path 또는 landmarks"):
            run_planes_pipeline()

    def test_partial_landmarks_nan_result(self, tmp_path: Path):
        """랜드마크 일부만 있음 → NaN 표기."""
        partial = {"N": [0, 0, 10], "ANS": [0, -2, 5], "PNS": [0, 2, 5]}
        result = run_planes_pipeline(landmarks={k: np.array(v, dtype=np.float64) for k, v in partial.items()})
        assert result["computed"] is False
        assert any(x in result["missing"] for x in ["LMeF", "RMeF"])

    def test_output_and_stdout_conflict(self, tmp_path: Path, capsys):
        """output_path + stdout=True : stdout 가 우선."""
        mrk = _make_dummy_mrk_file(tmp_path, "case01", _full_landmarks_dict())
        out = tmp_path / "planes.json"
        result = run_planes_pipeline(mrk, output_path=out, stdout=True)
        captured = capsys.readouterr()
        # stdout 으로 결과 출력됨
        assert '"computed": true' in captured.out
        # 파일은 생성되지 않음 (stdout 이 우선)
        assert not out.is_file()


# ── C. CLI 테스트 ────────────────────────────────────────────────────────

class TestCliMain:
    def test_help(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_missing_input(self, tmp_path: Path):
        nonexistent = tmp_path / "notexist.mrk.json"
        code = main(["-i", str(nonexistent)])
        assert code == 1

    def test_file_output_default(self, tmp_path: Path):
        mrk = _make_dummy_mrk_file(tmp_path, "case01", _full_landmarks_dict())
        out = tmp_path / "result.json"
        code = main(["-i", str(mrk), "-o", str(out)])
        assert code == 0
        assert out.is_file()
        result = json.loads(out.read_text(encoding="utf-8"))
        assert result["computed"] is True

    def test_stdout_flag(self, tmp_path: Path, capsys):
        mrk = _make_dummy_mrk_file(tmp_path, "case01", _full_landmarks_dict())
        code = main(["-i", str(mrk), "--stdout"])
        assert code == 0
        captured = capsys.readouterr()
        assert '"computed": true' in captured.out

    def test_stdin_out_even_when_file_exists(self, tmp_path: Path, capsys):
        """--stdout 지정 시 output 파일이 있어도 stdout 출력."""
        mrk = _make_dummy_mrk_file(tmp_path, "case01", _full_landmarks_dict())
        pre_written = tmp_path / "pre.json"
        pre_written.write_text('{"pre": true}')
        # --stdout 과 -o 를 함께 쓰면 stdout 이 우선
        code = main(["-i", str(mrk), "-o", str(pre_written), "--stdout"])
        assert code == 0
        captured = capsys.readouterr()
        assert '"computed": true' in captured.out
        # 원래 파일은 유지됨 (overwrite 안됨)
        assert json.loads(pre_written.read_text(encoding="utf-8"))["pre"] is True


# ── D. 예외 케이스 ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_collinear_points_raise(self):
        """세 점이 일직선 위에 있으면 법선 길이 0 → 계산 실패."""
        collinear = {
            "N": np.array([0.0, 0.0, 0.0]),
            "ANS": np.array([0.0, 0.0, 5.0]),
            "PNS": np.array([0.0, 0.0, 10.0]),
            "LMeF": np.array([-3.0, 0.0, 2.0]),
            "RMeF": np.array([3.0, 0.0, 2.0]),
        }
        result = run_planes_from_landmarks(collinear)
        # collinear 이면 MSP 법선 norm ~ 0 → ValueError 발생
        assert result["computed"] is False
        assert result["error"] is not None

    def test_zero_length_vector_in_cross(self):
        """중복 포인트도 처리 잘 됨."""
        duplicate = {
            "N": np.array([0.0, 0.0, 10.0]),
            "ANS": np.array([0.0, 0.0, 10.0]),  # N 과 동일
            "PNS": np.array([0.0, 2.0, 5.0]),
            "LMeF": np.array([-3.0, 0.0, 2.0]),
            "RMeF": np.array([3.0, 0.0, 2.0]),
        }
        result = run_planes_from_landmarks(duplicate)
        assert result["computed"] is False
        assert result["error"] is not None

    def test_negative_coordinates(self):
        """음수 좌표도 정상 처리."""
        neg = {
            "N": np.array([-10.0, -20.0, -120.0]),
            "ANS": np.array([-10.0, -45.0, -80.0]),
            "PNS": np.array([-10.0, -60.0, -82.0]),
            "LMeF": np.array([-15.0, -55.0, -30.0]),
            "RMeF": np.array([-25.0, -55.0, -30.0]),
        }
        result = run_planes_from_landmarks(neg)
        assert result["computed"] is True
        # NaN 이 아닌지 확인
        for key in ("msp", "occlusal"):
            for subkey in ("center", "normal"):
                assert all(not math.isnan(v) for v in result[key][subkey])
