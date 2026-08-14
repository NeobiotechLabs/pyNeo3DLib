"""통합 파이프라인 오케스트레이터 단위 테스트.

오케스트레이터 모듈은 stdlib-only 이므로 여기서는 torch/pyvista 등 무거운
의존성을 로드하지 않습니다. 모든 테스트는 tmp_path + mock runner 기반입니다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ── 임포트 준비 ────────────────────────────────────────────────────────
_ARTICUL_DIR = Path(__file__).resolve().parent.parent  # .../articulPipeline/
if str(_ARTICUL_DIR) not in sys.path:
    sys.path.insert(0, str(_ARTICUL_DIR))

from run_integrated_pipeline import (
    MERGE_SCRIPT,
    CONDYLE_SCRIPT,
    CANAL_SCRIPT,
    SEG_SCRIPT,
    MANDIBLE_STL_SUFFIX,
    CANAL_STL_SUFFIX,
    MERGED_MRK_SUFFIX,
    CONDYLES_MRK_SUFFIX,
    MEF_MRK_SUFFIX,
    LANDMARKS_MRK_SUFFIX,
    _FORBIDDEN_SEG_FLAGS,
    _clean_stem,
    discover_case_stems,
    check_case_inputs,
    should_run_condyle,
    should_run_canal,
    should_run_merge,
    cleanup_mrk_files,
    build_seg_command,
    build_condyle_command,
    build_canal_command,
    build_merge_command,
    run_subprocess,
    run_pipeline,
    main,
)

# ════════════════════════════════════════════════════════════════════════
# Fixtures —下游 스크립트들이 쓰는 파일명 규약과 정확히 일치해야 함
#   mandible :  {stem}{suffix}        = "case01_mandible.stl"
#   canal    :  {stem}_nerve_canal.stl = "case01_nerve_canal.stl"
#   mrk      :  {stem}_merged.mrk.json etc.
# ════════════════════════════════════════════════════════════════════════

def _make_case_files(tmp_path: Path, stem: str, *,
                     add_condyles: bool = False,
                     add_mef: bool = False,
                     add_merged: bool = True) -> None:
    """{stem} 의 테스트용 산출물들을 만든다."""
    # mandible STL: stem="case01" → "case01_mandible.stl"
    (tmp_path / f"{stem}{MANDIBLE_STL_SUFFIX}").touch()
    # nerve canal STL: stem="case01" → "case01_nerve_canal.stl"
    (tmp_path / f"{stem}{CANAL_STL_SUFFIX}").touch()
    if add_merged:
        (tmp_path / f"{stem}{MERGED_MRK_SUFFIX}").touch()
    if add_condyles:
        (tmp_path / f"{stem}{CONDYLES_MRK_SUFFIX}").touch()
    if add_mef:
        (tmp_path / f"{stem}{MEF_MRK_SUFFIX}").touch()


def _build_mock_runner(tmp_path: Path):
    """커맨드 기록용 목 runner · stdout 에는 print 안함."""
    history: list[list[str]] = []

    def _is_stage(cmd: list[str], target: str) -> bool:
        """cmd 의 인자 중 target 스크립트 이름이 있는지 확인.

        첫 번째 인자가 python 이고 두 번째가 목표 스크립트일 가능성이 높지만,
        하위 호환성을 위해 모든 인자를 문자열로 조인해 부분 일치로 찾는다.
        """
        # 절대 경로든 기본 이름든 찾기 위해 전체 명령어를 문자열로 비교
        return target in " ".join(cmd)

    def runner(cmd: list[str], verbose: bool = False) -> int:
        history.append(cmd)

        try:
            out_idx = cmd.index("-o")
            out_dir = Path(cmd[out_idx + 1])
        except (ValueError, IndexError):
            return 0

        if _is_stage(cmd, CONDYLE_SCRIPT.name):
            try:
                i_idx = cmd.index("-i")
                mesh_path = Path(cmd[i_idx + 1])
                raw = mesh_path.stem
            except (ValueError, IndexError):
                return 0
            clean = _clean_stem(raw)
            (out_dir / f"{clean}{CONDYLES_MRK_SUFFIX}").touch()

        elif _is_stage(cmd, CANAL_SCRIPT.name):
            try:
                i_idx = cmd.index("-i")
                mesh_path = Path(cmd[i_idx + 1])
                raw = mesh_path.stem
            except (ValueError, IndexError):
                return 0
            clean = _clean_stem(raw)
            (out_dir / f"{clean}{MEF_MRK_SUFFIX}").touch()

        elif _is_stage(cmd, MERGE_SCRIPT.name):
            try:
                i_idx = cmd.index("-i")
                first = Path(cmd[i_idx + 1])
                clean = first.stem.replace("_merged", "")
            except (ValueError, IndexError):
                clean = "merged"
            (out_dir / f"{clean}{LANDMARKS_MRK_SUFFIX}").touch()

        return 0

    return runner, history


# ── A. Discovery ──────────────────────────────────────────────────────

class TestCleanStem:
    def test_mandible_suffix(self):
        assert _clean_stem("case01_mandible") == "case01"

    def test_nerve_canal_suffix(self):
        assert _clean_stem("case01_nerve_canal") == "case01"

    def test_maxilla_suffix(self):
        assert _clean_stem("case02_maxilla") == "case02"

    def test_no_suffix(self):
        assert _clean_stem("standalone") == "standalone"

    def test_non_ascii(self):
        assert _clean_stem("케이스01_mandible") == "케이스01"


class TestDiscoverCaseStems:
    def test_single_case(self, tmp_path: Path):
        _make_case_files(tmp_path, "case01")
        assert discover_case_stems(tmp_path) == ["case01"]

    def test_multiple_cases_sorted(self, tmp_path: Path):
        _make_case_files(tmp_path, "zzz")
        _make_case_files(tmp_path, "aaa")
        _make_case_files(tmp_path, "mmm")
        assert discover_case_stems(tmp_path) == ["aaa", "mmm", "zzz"]

    def test_empty_raises(self, tmp_path: Path):
        (tmp_path / "other.stl").touch()
        with pytest.raises(FileNotFoundError):
            discover_case_stems(tmp_path)

    def test_non_ascii_stems(self, tmp_path: Path):
        _make_case_files(tmp_path, "케이스01")
        assert discover_case_stems(tmp_path) == ["케이스01"]

    def test_duplicates_not_possible_with_glob(self, tmp_path: Path):
        """glob 자체도 유니크 파일만 돌려서 dedupe 는 단순 확인."""
        _make_case_files(tmp_path, "case01")
        result = discover_case_stems(tmp_path)
        assert len(result) == 1


# ── B. Gating ─────────────────────────────────────────────────────────

class TestCheckCaseInputs:
    def test_all_present(self, tmp_path: Path):
        _make_case_files(tmp_path, "c1", add_condyles=True, add_mef=True)
        r = check_case_inputs("c1", tmp_path)
        assert all(v for k, v in r.items() if k != "landmarks_mrk")  # mandible/canal/merged 는 모두 True
        assert r["mandible_stl"] is True
        assert r["canal_stl"] is True
        assert r["merged_mrk"] is True

    def test_all_missing(self, tmp_path: Path):
        r = check_case_inputs("no_such", tmp_path)
        assert r["mandible_stl"] is False
        assert r["canal_stl"] is False
        assert r["merged_mrk"] is False

    def test_partial(self, tmp_path: Path):
        _make_case_files(tmp_path, "c2")  # mandible + canal + merged
        r = check_case_inputs("c2", tmp_path)
        assert r["mandible_stl"] is True
        assert r["canal_stl"] is True
        assert r["merged_mrk"] is True


class TestShouldRunCondyle:
    def test_ready(self, tmp_path: Path):
        _make_case_files(tmp_path, "c1")
        assert should_run_condyle("c1", tmp_path) is True

    def test_output_exists_skip(self, tmp_path: Path):
        _make_case_files(tmp_path, "c1", add_condyles=True)
        assert should_run_condyle("c1", tmp_path) is False

    def test_force_rebuild(self, tmp_path: Path):
        _make_case_files(tmp_path, "c1", add_condyles=True)
        assert should_run_condyle("c1", tmp_path, force=True) is True

    def test_no_mesh_skip(self, tmp_path: Path):
        assert should_run_condyle("ghost", tmp_path) is False


class TestShouldRunCanal:
    def test_ready(self, tmp_path: Path):
        _make_case_files(tmp_path, "c1")
        assert should_run_canal("c1", tmp_path) is True

    def test_output_exists_skip(self, tmp_path: Path):
        _make_case_files(tmp_path, "c1", add_mef=True)
        assert should_run_canal("c1", tmp_path) is False

    def test_force_rebuild(self, tmp_path: Path):
        _make_case_files(tmp_path, "c1", add_mef=True)
        assert should_run_canal("c1", tmp_path, force=True) is True

    def test_no_mesh_skip(self, tmp_path: Path):
        assert should_run_canal("ghost", tmp_path) is False


class TestShouldRunMerge:
    def test_all_three_present(self, tmp_path: Path):
        _make_case_files(tmp_path, "c1", add_condyles=True, add_mef=True)
        assert should_run_merge("c1", tmp_path) is True

    def test_merged_already_done_skip(self, tmp_path: Path):
        _make_case_files(tmp_path, "c1", add_condyles=True, add_mef=True)
        (tmp_path / "c1_landmarks.mrk.json").touch()
        assert should_run_merge("c1", tmp_path) is False

    def test_force_when_all_present_and_merges_done(self, tmp_path: Path):
        _make_case_files(tmp_path, "c1", add_condyles=True, add_mef=True)
        (tmp_path / "c1_landmarks.mrk.json").touch()
        assert should_run_merge("c1", tmp_path, force=True) is True

    def test_attempt_if_any_mrks_present(self, tmp_path: Path):
        """3 개 중 적어도 1 개 있으면 시도 (나머지 부족하면 merge 단계에서 스킵+경고)."""
        _make_case_files(tmp_path, "c1")
        assert should_run_merge("c1", tmp_path) is True

    def test_none_present(self, tmp_path: Path):
        assert should_run_merge("c1", tmp_path) is False


# ── C. Command Builders ───────────────────────────────────────────────

class TestBuildSegCommand:
    def test_minimal(self):
        cmd = build_seg_command(Path("/dcm"), Path("/out"))
        assert cmd[0] == sys.executable
        assert str(SEG_SCRIPT) in cmd
        # Windows 에서는 Path("/dcm") 가 "\\dcm" 으로 변환되므로 str(Path) 로 비교
        assert "-i" in cmd and str(Path("/dcm")) in cmd
        assert "-o" in cmd and str(Path("/out")) in cmd

    def test_model_dir(self):
        cmd = build_seg_command(Path("/dcm"), Path("/out"), model_dir=Path("/models"))
        assert "-m" in cmd and str(Path("/models")) in cmd

    def test_verbose_flag(self):
        cmd = build_seg_command(Path("/dcm"), Path("/out"), verbose=True)
        assert "-v" in cmd

    def test_forbidden_flags_absent(self):
        cmd = build_seg_command(Path("/dcm"), Path("/out"), verbose=True)
        for flag in _FORBIDDEN_SEG_FLAGS:
            assert flag not in cmd

    def test_patient_origin(self):
        cmd = build_seg_command(Path("/dcm"), Path("/out"), patient_origin=True)
        assert "--patient-origin" in cmd

    def test_no_keep_nifti(self):
        cmd = build_seg_command(Path("/dcm"), Path("/out"), no_keep_nifti=True)
        assert "--no-keep-nifti" in cmd

    def test_default_min_slices_omitted(self):
        cmd = build_seg_command(Path("/dcm"), Path("/out"))
        assert "--min-slices" not in cmd


class TestBuildCondyleCommand:
    def test_basic(self, tmp_path: Path):
        _make_case_files(tmp_path, "case01")
        cmd = build_condyle_command("case01", tmp_path)
        assert cmd[0] == sys.executable
        assert str(CONDYLE_SCRIPT) in cmd
        assert str(Path(tmp_path / "case01_mandible.stl")) in cmd
        assert str(tmp_path) in cmd


class TestBuildCanalCommand:
    def test_basic(self, tmp_path: Path):
        _make_case_files(tmp_path, "case01")
        cmd = build_canal_command("case01", tmp_path)
        assert cmd[0] == sys.executable
        assert str(CANAL_SCRIPT) in cmd
        assert str(Path(tmp_path / "case01_nerve_canal.stl")) in cmd
        assert str(tmp_path) in cmd


class TestBuildMergeCommand:
    def test_basic(self, tmp_path: Path):
        _make_case_files(tmp_path, "case01", add_condyles=True, add_mef=True)
        cmd = build_merge_command("case01", tmp_path)
        assert cmd[0] == sys.executable
        assert str(MERGE_SCRIPT) in cmd
        assert "-i" in cmd
        assert str(Path(tmp_path / "case01_merged.mrk.json")) in cmd
        assert str(Path(tmp_path / "case01_mandible_condyles.mrk.json")) in cmd
        assert str(Path(tmp_path / "case01_nerve_canal_mef.mrk.json")) in cmd
        assert "-o" in cmd and str(tmp_path) in cmd


# ── D. Orchestration (mock runner) ────────────────────────────────────

class TestRunPipelineHappyPath:
    def test_full_pipeline_skip_seg(self, tmp_path: Path):
        """skip-seg 후 mandible/canal/merged 존재 → condyle+canal+merge 실행.

        _make_case_files 는 입력 파일만 만들고, mock runner 가 condyle/canal 출력을
        생성해 게이트 전이가 실제 동작을 검증합니다.
        """
        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()
        # 입력만 미리 둠 (출력은 mock runner 가 생성)
        _make_case_files(tmp_path, "case01")

        runner, history = _build_mock_runner(tmp_path)
        exit_code, failures = run_pipeline(
            dcm_dir, tmp_path, skip_seg=True, runner=runner,
        )
        assert exit_code == 0
        assert failures == {"seg": [], "condyle": [], "canal": [], "merge": [], "plane": []}
        cmds_str = " ".join(" ".join(c) for c in history)
        assert CONDYLE_SCRIPT.name in cmds_str
        assert CANAL_SCRIPT.name in cmds_str
        assert MERGE_SCRIPT.name in cmds_str


class TestSkipSegNoCases:
    def test_exit_2_when_no_stls(self, tmp_path: Path):
        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()
        exit_code, _ = run_pipeline(dcm_dir, tmp_path, skip_seg=True)
        assert exit_code == 2


class TestCondyleFailCascade:
    def test_merge_skipped_when_condyle_fails(self, tmp_path: Path):
        """condyle 가 실패 → 출력이 없어서 merge 도 inputs 누락 → 스킵."""

        def fail_condyle(cmd, verbose=False):
            if CANAL_SCRIPT.name in " ".join(cmd):
                return 0  # canal 은 성공
            if MERGE_SCRIPT.name in " ".join(cmd):
                # merge 자체가 실패하도록
                return 1
            return 1  # condyle 실패

        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()
        _make_case_files(tmp_path, "case01")

        exit_code, failures = run_pipeline(dcm_dir, tmp_path, skip_seg=True, runner=fail_condyle)
        assert exit_code == 1
        assert "case01" in failures["condyle"]


class TestDryRun:
    def test_no_commands_executed(self, tmp_path: Path):
        """dry-run 시 discovery 도 하지 않음 (파일 없음 → exit 2) 또는
        발견해도 dry-run guard 가 명령 실행 막음."""
        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()
        # 파일을 만들지 않아서 discovery 가 FileNotFoundError → exit 2
        runner, history = _build_mock_runner(tmp_path)
        exit_code, _ = run_pipeline(dcm_dir, tmp_path, skip_seg=True, dry_run=True, runner=runner)
        assert exit_code == 2  # 발견 못 해서 early exit
        assert len(history) == 0

    def test_dry_run_does_not_run_after_discovery(self, tmp_path: Path):
        """파일이 있어서 discovery succeeded 해도 dry_run=True 면 merge 전 중단."""
        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()
        _make_case_files(tmp_path, "case01", add_condyles=True, add_mef=True)

        runner, history = _build_mock_runner(tmp_path)
        exit_code, _ = run_pipeline(dcm_dir, tmp_path, skip_seg=True, dry_run=True, runner=runner)
        # dry_run 은 condyle/canal 도 실행하지 않음 (그게 더 자연스러움)
        assert exit_code == 0
        # history 가 비어 있거나 condyle/canal 만 있음 (merge 는 절대 안 씀)
        cmds_str = " ".join(" ".join(c) for c in history)
        assert MERGE_SCRIPT.name not in cmds_str


class TestSegFailure:
    def test_exit_2_on_seg_failure(self, tmp_path: Path):
        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()

        def fail_runner(cmd, verbose=False):
            if SEG_SCRIPT.name in " ".join(cmd):
                return 1
            return 0

        exit_code, failures = run_pipeline(dcm_dir, tmp_path, runner=fail_runner)
        assert exit_code == 2
        assert failures["seg"] == ["seg_failed"]
        assert failures["condyle"] == []
        assert failures["canal"] == []
        assert failures["merge"] == []


# ── E. Cleanup (병합 후 원본 랜드마크 삭제) ────────────────────────────

class TestCleanupMrkFiles:
    def test_removes_all_three(self, tmp_path: Path):
        stem = "case01"
        (tmp_path / f"{stem}{MERGED_MRK_SUFFIX}").touch()
        (tmp_path / f"{stem}{CONDYLES_MRK_SUFFIX}").touch()
        (tmp_path / f"{stem}{MEF_MRK_SUFFIX}").touch()
        (tmp_path / f"{stem}{LANDMARKS_MRK_SUFFIX}").touch()  # merged 는 유지

        removed = cleanup_mrk_files(stem, tmp_path)
        assert len(removed) == 3
        assert not (tmp_path / f"{stem}{MERGED_MRK_SUFFIX}").is_file()
        assert not (tmp_path / f"{stem}{CONDYLES_MRK_SUFFIX}").is_file()
        assert not (tmp_path / f"{stem}{MEF_MRK_SUFFIX}").is_file()
        assert (tmp_path / f"{stem}{LANDMARKS_MRK_SUFFIX}").is_file()

    def test_skips_missing(self, tmp_path: Path):
        stem = "case01"
        (tmp_path / f"{stem}{MERGED_MRK_SUFFIX}").touch()
        # condyles, mef 는 없음
        removed = cleanup_mrk_files(stem, tmp_path)
        assert len(removed) == 1

    def test_empty_when_none_exist(self, tmp_path: Path):
        stem = "case01"
        removed = cleanup_mrk_files(stem, tmp_path)
        assert removed == []


class TestCleanupInPipeline:
    def test_cleanup_runs_on_success(self, tmp_path: Path):
        """merge 성공 시 원본 3개 mrk.json 이 자동으로 삭제됨."""
        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()
        _make_case_files(tmp_path, "case01")  # 입력만

        runner, history = _build_mock_runner(tmp_path)
        exit_code, failures = run_pipeline(
            dcm_dir, tmp_path, skip_seg=True, runner=runner,
        )
        assert exit_code == 0
        assert failures == {"seg": [], "condyle": [], "canal": [], "merge": [], "plane": []}
        # 원본 3개가 삭제되어야 함
        assert not (tmp_path / "case01_merged.mrk.json").is_file()
        assert not (tmp_path / "case01_mandible_condyles.mrk.json").is_file()
        assert not (tmp_path / "case01_nerve_canal_mef.mrk.json").is_file()

    def test_no_cleanup_when_flag_disabled(self, tmp_path: Path):
        """cleanup=False 면 원본 3개 mrk.json 이 그대로 유지됨."""
        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()
        _make_case_files(tmp_path, "case01")

        runner, history = _build_mock_runner(tmp_path)
        exit_code, failures = run_pipeline(
            dcm_dir, tmp_path, skip_seg=True, runner=runner, cleanup=False,
        )
        assert exit_code == 0
        assert failures == {"seg": [], "condyle": [], "canal": [], "merge": [], "plane": []}
        # 원본들이 남아있어야 함
        assert (tmp_path / "case01_merged.mrk.json").is_file()
        assert (tmp_path / "case01_mandible_condyles.mrk.json").is_file()
        assert (tmp_path / "case01_nerve_canal_mef.mrk.json").is_file()

    def test_merge_fail_does_not_cleanup(self, tmp_path: Path):
        """merge 실패 시 원본 삭제 안함."""

        def fail_merge(cmd, verbose=False):
            return 1  # merge 항상 실패

        dcm_dir = tmp_path / "dcm"
        dcm_dir.mkdir()
        _make_case_files(tmp_path, "case01", add_condyles=True, add_mef=True)

        exit_code, failures = run_pipeline(dcm_dir, tmp_path, skip_seg=True, runner=fail_merge)
        assert exit_code == 1
        # merge 가 실패했으므로 원본들 남아있어야 함
        assert (tmp_path / "case01_merged.mrk.json").is_file()
        assert (tmp_path / "case01_mandible_condyles.mrk.json").is_file()
        assert (tmp_path / "case01_nerve_canal_mef.mrk.json").is_file()
