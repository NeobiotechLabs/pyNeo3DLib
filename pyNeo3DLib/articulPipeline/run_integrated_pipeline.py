"""통합 파이프라인 오케스트레이터 — DICOM → 분할 + 랜드마크 통합.

사용 예:
    python run_integrated_pipeline.py \
        --input ./dcm_series/ --output ./results/

    python run_integrated_pipeline.py \
        --input ./dcm_series/ --output ./results/ --skip-seg  (기존 산출물 재사용)

    python run_integrated_pipeline.py \
        --input ./dcm_series/ --output ./results/ --dry-run -v  (미리 명령어 확인)

동작:
1. segPipeline: DICOM → NIfTI 변환, 랜드마크 추론(ANS/PNS/N), 분할(seg), STL 메쉬 추출
2. condylePointsFinder: mandible STL → LCo/RCo
3. canal_endpoint: nerve canal STL → LMeF/RMeF
4. merge_landmarks: 3개 mrk.json → {case}_landmarks.mrk.json (7개 포인트 전체)

케이스별 실패 시 다른 케이스는 계속 진행되며, 마지막에 요약이 출력됩니다.
이미 완료가 된 파일은 건너뛰며(--force 플래그 없으면) 재실행(resume) 가능합니다.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

# ── 상수 ────────────────────────────────────────────────────────────────

PIPELINE_ROOT = Path(__file__).resolve().parent

SEG_SCRIPT     = PIPELINE_ROOT / "segPipeline" / "run_segmentation_pipeline.py"
CONDYLE_SCRIPT = PIPELINE_ROOT / "condylePointsFinder" / "find_mandible_condylle.py"
CANAL_SCRIPT   = PIPELINE_ROOT / "canal_endpoint" / "find_canal_endpoints.py"
MERGE_SCRIPT   = PIPELINE_ROOT / "merge_landmarks.py"

MANDIBLE_STL_SUFFIX   = "_mandible.stl"
CANAL_STL_SUFFIX      = "_nerve_canal.stl"
MERGED_MRK_SUFFIX     = "_merged.mrk.json"
CONDYLES_MRK_SUFFIX   = "_mandible_condyles.mrk.json"
MEF_MRK_SUFFIX        = "_nerve_canal_mef.mrk.json"
LANDMARKS_MRK_SUFFIX  = "_landmarks.mrk.json"

#: 각 단계의 인자 전달을 막아야 하는 segPipeline 옵션 목록
_FORBIDDEN_SEG_FLAGS = ("--no-export-meshes", "--no-landmarks", "--no-restore-mandibular")


# ── Discovery ───────────────────────────────────────────────────────────

def _clean_stem(raw_stem: str) -> str:
    """메쉬 stem 에서 구조명 접미어 (_mandible) 를 제거한 기본 케이스 이름을 반환."""
    for suffix in (
        "_mandible", "_maxilla", "_nerve_canal", "_maxillary_sinus",
    ):
        if raw_stem.endswith(suffix):
            return raw_stem[: -len(suffix)]
    return raw_stem


def discover_case_stems(output_dir: Path) -> list[str]:
    """output_dir 에서 *_mandible.stl 을 찾아 정렬·중복 제거된 케이스 stem列表를 반환.

    케이스 이름은 metsh stem 에서 구조명 접미어(_mandible) 를 떼고,
    나머지를 케이스 ID 로 사용합니다.
    예: ``case01_mandible.stl`` → stem = ``case01``
    """
    pattern = "*_mandible.stl"
    stems = sorted({_clean_stem(p.stem) for p in output_dir.glob(pattern) if p.is_file()})
    if not stems:
        raise FileNotFoundError(
            f"'{output_dir}'에서 '*_mandible.stl' 파일을 찾지 못했습니다."
        )
    return stems


# ── Gating ──────────────────────────────────────────────────────────────

def check_case_inputs(case_stem: str, output_dir: Path) -> dict[str, bool]:
    """케이스별 필수 입력 파일 존재 여부를 반환.

    Returns: {"mandible_stl": bool, "canal_stl": bool, "merged_mrk": bool}
    """
    return {
        "mandible_stl": (output_dir / f"{case_stem}{MANDIBLE_STL_SUFFIX}").is_file(),
        "canal_stl": (output_dir / f"{case_stem}{CANAL_STL_SUFFIX}").is_file(),
        "merged_mrk": (output_dir / f"{case_stem}{MERGED_MRK_SUFFIX}").is_file(),
    }


def should_run_condyle(case_stem: str, output_dir: Path, *, force: bool = False) -> bool:
    """mandible STL 이 있고, 출력이 없을 때(True) 또는 force=True 일 때 True."""
    inputs = check_case_inputs(case_stem, output_dir)
    if not inputs["mandible_stl"]:
        return False
    if force:
        return True
    return not (output_dir / f"{case_stem}{CONDYLES_MRK_SUFFIX}").is_file()


def should_run_canal(case_stem: str, output_dir: Path, *, force: bool = False) -> bool:
    """nerve canal STL 이 있고, 출력이 없을 때(True) 또는 force=True 일 때 True."""
    inputs = check_case_inputs(case_stem, output_dir)
    if not inputs["canal_stl"]:
        return False
    if force:
        return True
    return not (output_dir / f"{case_stem}{MEF_MRK_SUFFIX}").is_file()


def should_run_merge(case_stem: str, output_dir: Path, *, force: bool = False) -> bool:
    """3개 mrk 가 모두 있고, merged 출력이 없을 때(True) 또는 force=True 일 때 True."""
    merged_already_done = (output_dir / f"{case_stem}{LANDMARKS_MRK_SUFFIX}").is_file()
    if force:
        # force 는 이미 완료된 것도 재실행
        return all((
            (output_dir / f"{case_stem}{MERGED_MRK_SUFFIX}").is_file(),
            (output_dir / f"{case_stem}{CONDYLES_MRK_SUFFIX}").is_file(),
            (output_dir / f"{case_stem}{MEF_MRK_SUFFIX}").is_file(),
        ))
    if merged_already_done:
        return False
    outputs_present = any(p.is_file() for p in [
        output_dir / f"{case_stem}{MERGED_MRK_SUFFIX}",
        output_dir / f"{case_stem}{CONDYLES_MRK_SUFFIX}",
        output_dir / f"{case_stem}{MEF_MRK_SUFFIX}",
    ])
    return outputs_present


# ── Command Builders (pure) ────────────────────────────────────────────

def build_seg_command(
    input_dir: Path,
    output_dir: Path,
    *,
    model_dir: Optional[Path] = None,
    no_keep_nifti: bool = False,
    min_slices: int = 50,
    patient_origin: bool = False,
    verbose: bool = False,
) -> list[str]:
    """segPipeline 호출 커맨드 빌드.

    주의: --no-export-meshes/--no-landmarks/--no-restore-mandibular 는
    다운스트림에서 STL/mrk.json 이 필요하므로 절대 전달하지 않습니다.
    """
    cmd = [sys.executable, str(SEG_SCRIPT)]
    cmd.extend(["-i", str(input_dir)])
    cmd.extend(["-o", str(output_dir)])

    if model_dir is not None:
        cmd.extend(["-m", str(model_dir)])
    if no_keep_nifti:
        cmd.append("--no-keep-nifti")
    if min_slices != 50:
        cmd.extend(["--min-slices", str(min_slices)])
    if patient_origin:
        cmd.append("--patient-origin")
    if verbose:
        cmd.append("-v")

    return cmd


def build_condyle_command(case_stem: str, output_dir: Path) -> list[str]:
    cmd = [sys.executable, str(CONDYLE_SCRIPT)]
    cmd.extend(["-i", str(output_dir / f"{case_stem}{MANDIBLE_STL_SUFFIX}")])
    cmd.extend(["-o", str(output_dir)])
    return cmd


def build_canal_command(case_stem: str, output_dir: Path) -> list[str]:
    cmd = [sys.executable, str(CANAL_SCRIPT)]
    cmd.extend(["-i", str(output_dir / f"{case_stem}{CANAL_STL_SUFFIX}")])
    cmd.extend(["-o", str(output_dir)])
    return cmd


def build_merge_command(case_stem: str, output_dir: Path) -> list[str]:
    cmd = [sys.executable, str(MERGE_SCRIPT)]
    cmd.extend(["-i"])
    cmd.extend([
        str(output_dir / f"{case_stem}{MERGED_MRK_SUFFIX}"),
        str(output_dir / f"{case_stem}{CONDYLES_MRK_SUFFIX}"),
        str(output_dir / f"{case_stem}{MEF_MRK_SUFFIX}"),
    ])
    cmd.extend(["-o", str(output_dir)])
    return cmd


# ── Execution ───────────────────────────────────────────────────────────

def _print(stage: str, case_stem: str, message: str, indent: int = 2) -> None:
    prefix = "  " * indent
    print(f"{prefix}[{stage}] {case_stem}: {message}", flush=True)


def run_subprocess(cmd: list[str], verbose: bool = False) -> int:
    """subprocess.run wrappers — exit code 반환."""
    if verbose or True:  # always show for clarity; can gate behind a flag later
        print(f"  ▶ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    *,
    skip_seg: bool = False,
    force: bool = False,
    dry_run: bool = False,
    model_dir: Optional[Path] = None,
    no_keep_nifti: bool = False,
    min_slices: int = 50,
    patient_origin: bool = False,
    verbose: bool = False,
    cleanup: bool = True,
    runner: Optional[Callable[[list[str], bool], int]] = None,
) -> tuple[int, dict[str, list[str]]]:
    """파이프라인 단계를 순서대로 실행.

    Args:
        runner: 주입 가능한 실행기. 테스트용 mock Runner가 subprocess 대신 호출됨.
                sig: ``runner(cmd: list[str], verbose: bool) -> exit_code``
                None이면 기본 subprocess 사용.
        cleanup: 병합 완료 후 원본 3개 mrk.json 파일을 삭제할지 여부 (기본 True).

    Returns:
        (exit_code, failures). failures 는 {"seg"|condyle|canal|merge": [케이스명...]}.
    """
    _runner = runner if runner is not None else run_subprocess

    output_dir.mkdir(parents=True, exist_ok=True)

    failures: dict[str, list[str]] = {
        "seg": [],
        "condyle": [],
        "canal": [],
        "merge": [],
    }
    total_cleaned = 0

    # ── Stage 1: Segmentation ──────────────────────────────────────────
    if not skip_seg:
        print("═══ Stage 1: Segmentation ═══", flush=True)
        seg_cmd = build_seg_command(
            input_dir, output_dir,
            model_dir=model_dir,
            no_keep_nifti=no_keep_nifti,
            min_slices=min_slices,
            patient_origin=patient_origin,
            verbose=verbose,
        )
        rc = _runner(seg_cmd, verbose=verbose)
        if rc != 0:
            print(f"[ERROR] segPipeline exited {rc}. 파이프라인을 중단합니다.", file=sys.stderr, flush=True)
            return 2, {"seg": ["seg_failed"], "condyle": [], "canal": [], "merge": []}
        print("[OK] Segmentation 완료.", flush=True)
    else:
        print("═══ Stage 1: Skip Segmentation (--skip-seg) ═══", flush=True)

    # ── Case Discovery ─────────────────────────────────────────────────
    try:
        case_stems = discover_case_stems(output_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 2, failures

    print(f"발견된 케이스: {len(case_stems)}개 — {', '.join(case_stems)}", flush=True)

    # ── Dry-run mode guard: stop before any stage execution ──────────────
    if dry_run:
        print("\n[--dry-run] 모드 — 실행하지 않고 종료.", flush=True)
        return 0, failures

    # ── Stage 2 & 3: Condyle + Canal (per-case, independent) ───────────
    for stem in case_stems:
        print(f"\n═══ Case: {stem} ═══", flush=True)

        # Condyle
        if should_run_condyle(stem, output_dir, force=force):
            condyle_cmd = build_condyle_command(stem, output_dir)
            rc = _runner(condyle_cmd, verbose=verbose)
            if rc != 0:
                failures["condyle"].append(stem)
                _print("condyle", stem, f"실패 (exit {rc})", 1)
            else:
                _print("condyle", stem, "완료", 1)
        else:
            existing = (output_dir / f"{stem}{CONDYLES_MRK_SUFFIX}").is_file()
            has_mesh = check_case_inputs(stem, output_dir)["mandible_stl"]
            if existing:
                _print("condyle", stem, "건너섬 (이미 완료)", 1)
            elif not has_mesh:
                _print("condyle", stem, "건너섬 (mandible STL 없음)", 1)
            else:
                _print("condyle", stem, "건러섬 (force=False 이며 이미 완료됨)", 1)

        # Canal
        if should_run_canal(stem, output_dir, force=force):
            canal_cmd = build_canal_command(stem, output_dir)
            rc = _runner(canal_cmd, verbose=verbose)
            if rc != 0:
                failures["canal"].append(stem)
                _print("canal", stem, f"실패 (exit {rc})", 1)
            else:
                _print("canal", stem, "완료", 1)
        else:
            existing = (output_dir / f"{stem}{MEF_MRK_SUFFIX}").is_file()
            has_mesh = check_case_inputs(stem, output_dir)["canal_stl"]
            if existing:
                _print("canal", stem, "건너섬 (이미 완료)", 1)
            elif not has_mesh:
                _print("canal", stem, "건너섬 (nerve canal STL 없음)", 1)
            else:
                _print("canal", stem, "건너섬 (force=False 이며 이미 완료됨)", 1)

    # ── Stage 4: Merge ─────────────────────────────────────────────────
    for stem in case_stems:
        if not should_run_merge(stem, output_dir, force=force):
            continue

        has_merged = (output_dir / f"{stem}{MERGED_MRK_SUFFIX}").is_file()
        has_condyles = (output_dir / f"{stem}{CONDYLES_MRK_SUFFIX}").is_file()
        has_mef = (output_dir / f"{stem}{MEF_MRK_SUFFIX}").is_file()

        if not (has_merged and has_condyles and has_mef):
            missing = []
            if not has_merged:
                missing.append(f"{stem}{MERGED_MRK_SUFFIX}")
            if not has_condyles:
                missing.append(f"{stem}{CONDYLES_MRK_SUFFIX}")
            if not has_mef:
                missing.append(f"{stem}{MEF_MRK_SUFFIX}")
            print(f"[경고] {stem}: 병합을 건너뜁니다 — 누락: {', '.join(missing)}", flush=True)
            failures["merge"].append(stem)
            continue

        print(f"\n═══ Merge: {stem} ═══", flush=True)
        merge_cmd = build_merge_command(stem, output_dir)
        rc = _runner(merge_cmd, verbose=verbose)
        if rc != 0:
            failures["merge"].append(stem)
            _print("merge", stem, f"실패 (exit {rc})", 1)
        else:
            _print("merge", stem, "완료", 1)
            if cleanup:
                n_removed = len(cleanup_mrk_files(stem, output_dir))
                total_cleaned += n_removed

    # ── Summary ────────────────────────────────────────────────────────
    _print_summary(failures, total_cleaned=total_cleaned)

    total_failures = sum(len(v) for v in failures.values())
    if total_failures == 0:
        return 0, failures
    return 1, failures


def cleanup_mrk_files(stem: str, output_dir: Path) -> list[Path]:
    """랜드마크 병합 후 원본 3개 mrk.json 파일을 삭제.

    Returns: 실제로 삭제한 Path 리스트
    """
    removed: list[Path] = []
    for suffix in (MERGED_MRK_SUFFIX, CONDYLES_MRK_SUFFIX, MEF_MRK_SUFFIX):
        p = output_dir / f"{stem}{suffix}"
        if p.is_file():
            p.unlink()
            removed.append(p)
            _print("cleanup", stem, f"제거: {p.name}", 2)
    return removed


def _print_summary(failures: dict[str, list[str]], total_cleaned: int = 0) -> None:
    print(f"\n{'='*50}", flush=True)
    print("═══ 실행 요약 ═══", flush=True)
    stages = [("seg", "Segmentation"), ("condyle", "Condyle Points"),
              ("canal", "Canal Endpoints"), ("merge", "Merge Landmarks")]
    for key, label in stages:
        items = failures.get(key, [])
        status = f"{len(items)}개 실패" if items else "모두 성공"
        print(f"  {label}: {status}", flush=True)
    if total_cleaned > 0:
        print(f"  원본 랜드마크: {total_cleaned}개 삭제", flush=True)
    print(f"{'='*50}", flush=True)


# ── CLI ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="통합 파이프라인: DICOM → 분할 + 랜드마크(STL) → 통합 mrk.json"
    )
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="DICOM 시리즈 폴더 (single-case 또는 batch root)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="결과 저장 폴더 (flat layout)"
    )
    # segPipeline passthrough
    parser.add_argument(
        "-m", "--model", type=Path, default=None,
        help="nnU-Net 모델 폴더 경로 (default: segPipeline/model/)"
    )
    parser.add_argument(
        "--no-keep-nifti", action="store_true", default=False,
        help="중간 .nii.gz 파일을 output 폴더에 유지하지 않음"
    )
    parser.add_argument(
        "--min-slices", type=int, default=50,
        help="DICOM 최소 슬라이스 수 제한 (default: 50)"
    )
    parser.add_argument(
        "--patient-origin", action="store_true", default=False,
        help="DICOM ImagePositionPatient 를 NIfTI 원점으로 사용"
    )
    # pipeline control
    parser.add_argument(
        "--skip-seg", action="store_true", default=False,
        help="segmentation 단계 생략 (output 폴더 기존 산출물만 사용)"
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="이미 완료된 파일도 강제 재실행"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="실행할 커맨드만 출력하고 실제로는 실행하지 않음"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="상세 진행 메시지 출력"
    )
    parser.add_argument(
        "--no-cleanup", action="store_true", default=False,
        help="병합 후 원본 랜드마크 파일 삭제 안함 (default: 삭제)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.is_dir():
        print(f"[ERROR] 입력 디렉토리가 없습니다: {args.input}", file=sys.stderr)
        return 2

    try:
        exit_code, failures = run_pipeline(
            args.input, args.output,
            skip_seg=args.skip_seg,
            force=args.force,
            dry_run=args.dry_run,
            model_dir=args.model,
            no_keep_nifti=args.no_keep_nifti,
            min_slices=args.min_slices,
            patient_origin=args.patient_origin,
            verbose=args.verbose,
            cleanup=not args.no_cleanup,
        )
        return exit_code
    except Exception as exc:
        print(f"[오류] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
