"""통합 파이프라인 예외(엣지) 케이스 테스트.

기존 test_integrated_pipeline*.py 에서 다루지 않는 예외 경로를 보완:

A. 케이스별 입력 누락 시 우아한 건너뛰기 (canal STL 없음 등)
B. 케이스/단계 간 실패 격리 (mixed failures)
C. 재실행(resume) 시나리오 — 실패 후 재실행, 성공 후 재실행
D. --skip-planes 플래그
E. Plane 단계 산출물 예외: 출력 파일 없음 / dict 가 아닌 JSON / missing=null
F. main() 의 catch-all 예외 처리, CLI 인자 오류
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ARTICUL_DIR = Path(__file__).resolve().parent.parent
if str(_ARTICUL_DIR) not in sys.path:
    sys.path.insert(0, str(_ARTICUL_DIR))

import run_integrated_pipeline as rip
from run_integrated_pipeline import (
    SEG_SCRIPT,
    CONDYLE_SCRIPT,
    CANAL_SCRIPT,
    MERGE_SCRIPT,
    PLANES_SCRIPT,
    MANDIBLE_STL_SUFFIX,
    CANAL_STL_SUFFIX,
    MERGED_MRK_SUFFIX,
    CONDYLES_MRK_SUFFIX,
    MEF_MRK_SUFFIX,
    LANDMARKS_MRK_SUFFIX,
    PLANES_JSON_SUFFIX,
    _clean_stem,
    run_pipeline,
    main,
)


# ── 헬퍼 ───────────────────────────────────────────────────────────────

def _touch(path: Path) -> None:
    path.touch()


def _make_case(tmp_path: Path, stem: str, *,
               mandible: bool = True,
               canal: bool = True,
               merged: bool = True,
               condyles: bool = False,
               mef: bool = False,
               landmarks: bool = False,
               planes: bool = False) -> None:
    """케이스 산출물을 선택적으로 생성."""
    if mandible:
        _touch(tmp_path / f"{stem}{MANDIBLE_STL_SUFFIX}")
    if canal:
        _touch(tmp_path / f"{stem}{CANAL_STL_SUFFIX}")
    if merged:
        _touch(tmp_path / f"{stem}{MERGED_MRK_SUFFIX}")
    if condyles:
        _touch(tmp_path / f"{stem}{CONDYLES_MRK_SUFFIX}")
    if mef:
        _touch(tmp_path / f"{stem}{MEF_MRK_SUFFIX}")
    if landmarks:
        _touch(tmp_path / f"{stem}{LANDMARKS_MRK_SUFFIX}")
    if planes:
        _touch(tmp_path / f"{stem}{PLANES_JSON_SUFFIX}")


def _script_of(cmd: list[str]) -> str:
    """cmd 가 호출하는 스크립트 파일명 반환."""
    return Path(cmd[1]).name


def _build_success_runner(tmp_path: Path, planes_json: dict | None = None):
    """모든 단계를 성공시키는 mock runner. 각 단계의 실제 산출물을 생성.

    planes_json: plane 단계에서 기록할 JSON 내용 (None 이면 computed=true 기본값).
    """
    history: list[list[str]] = []
    if planes_json is None:
        planes_json = {
            "msp": {"center": [0.0, 0.0, 0.0], "normal": [-1.0, 0.0, 0.0]},
            "occlusal": {"center": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, -1.0]},
            "required": ["N", "ANS", "PNS", "LMeF", "RMeF"],
            "missing": [],
            "computed": True,
            "error": None,
        }

    def runner(cmd: list[str], verbose: bool = False) -> int:
        history.append(cmd)
        script = _script_of(cmd)
        i_idx = cmd.index("-i")
        in_path = Path(cmd[i_idx + 1])
        o_idx = cmd.index("-o")
        out_path = Path(cmd[o_idx + 1])

        if script == CONDYLE_SCRIPT.name:
            stem = _clean_stem(in_path.stem)
            _touch(out_path / f"{stem}{CONDYLES_MRK_SUFFIX}")
        elif script == CANAL_SCRIPT.name:
            stem = _clean_stem(in_path.stem)
            _touch(out_path / f"{stem}{MEF_MRK_SUFFIX}")
        elif script == MERGE_SCRIPT.name:
            stem = in_path.name.replace(MERGED_MRK_SUFFIX, "")
            _touch(out_path / f"{stem}{LANDMARKS_MRK_SUFFIX}")
        elif script == PLANES_SCRIPT.name:
            import json
            out_path.write_text(json.dumps(planes_json), encoding="utf-8")
        return 0

    return runner, history


def _make_dcm(tmp_path: Path) -> Path:
    dcm = tmp_path / "dcm"
    dcm.mkdir(exist_ok=True)
    return dcm


# ════════════════════════════════════════════════════════════════════════
# A. 케이스별 입력 누락 → 우아한 건너뛰기
# ════════════════════════════════════════════════════════════════════════

class TestMissingInputsPerCase:
    def test_no_canal_stl_skips_canal_and_blocks_merge(self, tmp_path: Path):
        """nerve canal STL 없음 → canal 은 건너뛰고(실패 아님),
        merge 는 MEF mrk 누락으로 경고 + 실패 기록."""
        _make_case(tmp_path, "case01", canal=False)  # mandible + merged 만 존재

        runner, history = _build_success_runner(tmp_path)
        exit_code, failures = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True,
            skip_planes=True, runner=runner,
        )

        scripts = [_script_of(c) for c in history]
        assert CONDYLE_SCRIPT.name in scripts      # mandible 있으니 condyle 실행
        assert CANAL_SCRIPT.name not in scripts    # canal STL 없어 미실행
        assert MERGE_SCRIPT.name not in scripts    # MEF 누락으로 merge 미실행
        assert failures["canal"] == []             # 미실행은 실패로 기록 안 됨
        assert failures["merge"] == ["case01"]
        assert exit_code == 1

    def test_no_merged_mrk_blocks_merge_after_condyle_canal(self, tmp_path: Path):
        """segPipeline 의 merged.mrk.json 누락 → condyle/canal 은 실행,
        merge 는 merged 누락으로 경고 + 실패 기록."""
        _make_case(tmp_path, "case01", merged=False)

        runner, history = _build_success_runner(tmp_path)
        exit_code, failures = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True,
            skip_planes=True, runner=runner,
        )

        scripts = [_script_of(c) for c in history]
        assert CONDYLE_SCRIPT.name in scripts
        assert CANAL_SCRIPT.name in scripts
        assert MERGE_SCRIPT.name not in scripts
        assert failures["condyle"] == []
        assert failures["canal"] == []
        assert failures["merge"] == ["case01"]
        assert exit_code == 1

    def test_output_dir_created_when_missing(self, tmp_path: Path):
        """output 디렉토리가 없으면 자동 생성 (seg 실패 전 mkdir 확인)."""
        out = tmp_path / "new_output"
        assert not out.exists()

        def fail_seg(cmd, verbose=False):
            return 1

        run_pipeline(_make_dcm(tmp_path), out, runner=fail_seg)
        assert out.is_dir()


# ════════════════════════════════════════════════════════════════════════
# B. 케이스/단계 간 실패 격리
# ════════════════════════════════════════════════════════════════════════

class TestFailureIsolation:
    def test_mixed_failures_across_cases_and_stages(self, tmp_path: Path):
        """case01 condyle 실패, case02 canal 실패, case03 전체 성공.
        각 실패는 해당 단계·케이스에만 기록되고 파이프라인은 계속 진행."""
        for stem in ("case01", "case02", "case03"):
            _make_case(tmp_path, stem)

        history: list[list[str]] = []

        def runner(cmd: list[str], verbose: bool = False) -> int:
            history.append(cmd)
            script = _script_of(cmd)
            i_idx = cmd.index("-i")
            in_path = Path(cmd[i_idx + 1])
            o_idx = cmd.index("-o")
            out_path = Path(cmd[o_idx + 1])

            if script == CONDYLE_SCRIPT.name:
                if in_path.stem.startswith("case01"):
                    return 1  # case01 condyle 실패
                _touch(out_path / f"{_clean_stem(in_path.stem)}{CONDYLES_MRK_SUFFIX}")
            elif script == CANAL_SCRIPT.name:
                if in_path.stem.startswith("case02"):
                    return 1  # case02 canal 실패
                _touch(out_path / f"{_clean_stem(in_path.stem)}{MEF_MRK_SUFFIX}")
            elif script == MERGE_SCRIPT.name:
                stem = in_path.name.replace(MERGED_MRK_SUFFIX, "")
                _touch(out_path / f"{stem}{LANDMARKS_MRK_SUFFIX}")
            return 0

        exit_code, failures = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True,
            skip_planes=True, runner=runner,
        )

        assert failures["condyle"] == ["case01"]
        assert failures["canal"] == ["case02"]
        # condyle/canal 실패의 연쇄: 두 케이스 모두 merge 입력 누락 → merge 실패 기록
        assert failures["merge"] == ["case01", "case02"]
        # case03 은 merge 까지 성공
        assert (tmp_path / f"case03{LANDMARKS_MRK_SUFFIX}").is_file()
        assert exit_code == 1

    def test_seg_failure_blocks_everything(self, tmp_path: Path):
        """seg 실패 → discovery 조차 없이 즉시 종료 (exit 2)."""
        _make_case(tmp_path, "case01")

        def fail_seg(cmd, verbose=False):
            return 1

        exit_code, failures = run_pipeline(
            _make_dcm(tmp_path), tmp_path, runner=fail_seg,
        )
        assert exit_code == 2
        assert failures["seg"] == ["seg_failed"]
        # 이후 단계는 시도조차 안 됨
        assert failures["condyle"] == []
        assert failures["canal"] == []
        assert failures["merge"] == []
        assert failures["plane"] == []


# ════════════════════════════════════════════════════════════════════════
# C. 재실행(resume) 시나리오
# ════════════════════════════════════════════════════════════════════════

class TestResume:
    def test_resume_after_condyle_failure_reruns_only_failed(self, tmp_path: Path):
        """1 차 실행: condyle 실패, canal 성공.
        2 차 실행: condyle 만 재실행, canal 은 기존 산출물로 건너뛰기."""
        _make_case(tmp_path, "case01")

        # 1차: condyle 실패 runner
        def failing_runner(cmd: list[str], verbose: bool = False) -> int:
            if _script_of(cmd) == CONDYLE_SCRIPT.name:
                return 1
            # canal 성공 → mef 생성
            o_idx = cmd.index("-o")
            out_path = Path(cmd[o_idx + 1])
            i_idx = cmd.index("-i")
            in_path = Path(cmd[i_idx + 1])
            if _script_of(cmd) == CANAL_SCRIPT.name:
                _touch(out_path / f"{_clean_stem(in_path.stem)}{MEF_MRK_SUFFIX}")
            return 0

        code1, failures1 = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True,
            skip_planes=True, runner=failing_runner,
        )
        assert code1 == 1
        assert failures1["condyle"] == ["case01"]

        # 2차: 전부 성공 runner (resume)
        runner2, history2 = _build_success_runner(tmp_path)
        code2, failures2 = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True,
            skip_planes=True, runner=runner2,
        )
        scripts2 = [_script_of(c) for c in history2]
        assert CONDYLE_SCRIPT.name in scripts2      # 실패했던 condyle 재실행
        assert CANAL_SCRIPT.name not in scripts2    # 완료된 canal 은 건너뛰기
        assert MERGE_SCRIPT.name in scripts2        # merge 는 이제 가능
        assert code2 == 0
        assert all(len(v) == 0 for v in failures2.values())

    def test_resume_after_full_success_skips_planes(self, tmp_path: Path):
        """전체 성공 후 재실행: planes.json 이 이미 있으면 plane 단계는 건너뜀."""
        _make_case(tmp_path, "case01")

        runner1, history1 = _build_success_runner(tmp_path)
        code1, failures1 = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True, runner=runner1,
        )
        assert code1 == 0
        assert (tmp_path / f"case01{PLANES_JSON_SUFFIX}").is_file()

        runner2, history2 = _build_success_runner(tmp_path)
        code2, failures2 = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True, runner=runner2,
        )
        scripts2 = [_script_of(c) for c in history2]
        assert PLANES_SCRIPT.name not in scripts2   # plane 은 재실행 안 됨
        assert code2 == 0
        assert all(len(v) == 0 for v in failures2.values())

    def test_force_reruns_planes_even_when_done(self, tmp_path: Path):
        """--force 는 planes.json 이 있어도 plane 단계를 재실행."""
        _make_case(tmp_path, "case01", landmarks=True, planes=True)

        runner, history = _build_success_runner(tmp_path)
        code, _ = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True,
            force=True, runner=runner,
        )
        scripts = [_script_of(c) for c in history]
        assert PLANES_SCRIPT.name in scripts
        assert code == 0


# ════════════════════════════════════════════════════════════════════════
# D. --skip-planes 플래그
# ════════════════════════════════════════════════════════════════════════

class TestSkipPlanesFlag:
    def test_planes_never_run_even_with_landmarks(self, tmp_path: Path):
        """landmarks 가 있어도 skip_planes=True 면 plane 명령 자체를 실행 안 함."""
        _make_case(tmp_path, "case01", condyles=True, mef=True, landmarks=True)

        runner, history = _build_success_runner(tmp_path)
        code, failures = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True,
            skip_planes=True, runner=runner,
        )
        scripts = [_script_of(c) for c in history]
        assert PLANES_SCRIPT.name not in scripts
        assert failures["plane"] == []
        assert code == 0


# ════════════════════════════════════════════════════════════════════════
# E. Plane 단계 산출물 예외
# ════════════════════════════════════════════════════════════════════════

class TestPlaneOutputEdgeCases:
    def _setup_complete_case(self, tmp_path: Path) -> None:
        """plane 단계만 남기고 모든 산출물이 완료된 상태."""
        _make_case(tmp_path, "case01",
                   condyles=True, mef=True, landmarks=True)

    def test_runner_ok_but_no_planes_json(self, tmp_path: Path):
        """runner 가 exit 0 을 돌려도 planes.json 미생성 → 실패 기록."""
        self._setup_complete_case(tmp_path)

        def noop_runner(cmd, verbose=False):
            return 0  # 아무 파일도 만들지 않음

        code, failures = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True, runner=noop_runner,
        )
        assert failures["plane"] == ["case01"]
        assert code == 1

    def test_planes_json_null_does_not_crash(self, tmp_path: Path):
        """planes.json 내용이 `null` (dict 아님) → 파싱 오류로 실패 기록,
        전체 파이프라인은 크래시 없이 계속."""
        self._setup_complete_case(tmp_path)

        def null_json_runner(cmd, verbose=False):
            if _script_of(cmd) == PLANES_SCRIPT.name:
                o_idx = cmd.index("-o")
                Path(cmd[o_idx + 1]).write_text("null", encoding="utf-8")
            return 0

        code, failures = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True, runner=null_json_runner,
        )
        assert failures["plane"] == ["case01"]
        assert code == 1

    def test_planes_json_array_does_not_crash(self, tmp_path: Path):
        """planes.json 내용이 리스트 → 파싱 오류로 실패 기록."""
        self._setup_complete_case(tmp_path)

        def array_json_runner(cmd, verbose=False):
            if _script_of(cmd) == PLANES_SCRIPT.name:
                o_idx = cmd.index("-o")
                Path(cmd[o_idx + 1]).write_text("[1, 2, 3]", encoding="utf-8")
            return 0

        code, failures = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True, runner=array_json_runner,
        )
        assert failures["plane"] == ["case01"]
        assert code == 1

    def test_planes_json_missing_is_null(self, tmp_path: Path):
        """computed=false 인데 missing 이 null → join 크래시 없이 부분 실패 기록."""
        self._setup_complete_case(tmp_path)

        def null_missing_runner(cmd, verbose=False):
            if _script_of(cmd) == PLANES_SCRIPT.name:
                o_idx = cmd.index("-o")
                Path(cmd[o_idx + 1]).write_text(
                    '{"computed": false, "missing": null}', encoding="utf-8")
            return 0

        code, failures = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True, runner=null_missing_runner,
        )
        assert failures["plane"] == ["case01"]
        assert code == 1

    def test_non_dict_planes_json_does_not_block_other_cases(self, tmp_path: Path):
        """case01 의 planes.json 이 손상돼도 case02 plane 계산은 정상 수행."""
        self._setup_complete_case(tmp_path)
        # case02 도 완료 상태
        stem2 = "case02"
        _make_case(tmp_path, stem2, condyles=True, mef=True, landmarks=True)

        def runner(cmd, verbose=False):
            if _script_of(cmd) == PLANES_SCRIPT.name:
                o_idx = cmd.index("-o")
                out = Path(cmd[o_idx + 1])
                if out.name.startswith("case01"):
                    out.write_text("null", encoding="utf-8")  # 손상
                else:
                    out.write_text('{"computed": true, "missing": []}', encoding="utf-8")
            return 0

        code, failures = run_pipeline(
            _make_dcm(tmp_path), tmp_path, skip_seg=True, runner=runner,
        )
        assert failures["plane"] == ["case01"]   # case02 는 실패 목록에 없음
        assert code == 1


# ════════════════════════════════════════════════════════════════════════
# F. main() catch-all 및 CLI 예외
# ════════════════════════════════════════════════════════════════════════

class TestMainExceptionHandling:
    def test_main_returns_2_on_unexpected_exception(self, tmp_path: Path, monkeypatch):
        """runner(subprocess) 가 exit code 대신 예외를 던지면
        main 이 catch 해서 2 반환 (파이프라인이 미처리 예외로 죽지 않음)."""
        _make_dcm(tmp_path)

        def exploding_runner(cmd, verbose=False):
            raise RuntimeError("subprocess exploded")

        monkeypatch.setattr(rip, "run_subprocess", exploding_runner)
        code = main(["-i", str(tmp_path / "dcm"), "-o", str(tmp_path / "out")])
        assert code == 2

    def test_cli_missing_required_args_exits_2(self):
        """필수 인자(-i/-o) 누락 → argparse SystemExit(2)."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2

    def test_cli_seg_failure_exit_code(self, tmp_path: Path, monkeypatch):
        """CLI 로 seg 실행 실패 시 exit code 2 전파."""
        _make_dcm(tmp_path)
        monkeypatch.setattr(rip, "run_subprocess", lambda cmd, verbose=False: 1)
        code = main(["-i", str(tmp_path / "dcm"), "-o", str(tmp_path / "out")])
        assert code == 2
