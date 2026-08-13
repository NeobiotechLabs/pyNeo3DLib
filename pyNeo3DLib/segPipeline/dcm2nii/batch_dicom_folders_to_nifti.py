#!/usr/bin/env python3
"""
지정한 루트 폴더의 **직계 하위 폴더**마다 DICOM(.dcm 등)을 읽어
``루트/output/*.nii.gz`` 로 저장합니다.

각 케이스는 기본적으로 **별도 Python 프로세스**에서 처리해 ITK 메모리가
케이스 종료 시 OS에 반환되도록 합니다.

케이스 폴더에 ``.zip`` 이 있으면 해당 폴더에 압축을 풀고, 하위에서
``.dcm`` 을 찾은 뒤 변환합니다.

예::

    python batch_dicom_folders_to_nifti.py F:\\data\\cases

구조::

    cases/
      patient_a/*.dcm
      patient_b/archive.zip   → patient_b/ 에 압축 해제 후 변환
    → cases/output/patient_a.nii.gz, patient_b.nii.gz
"""

from __future__ import annotations

import argparse
import gc
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Callable, Optional

ProgressCallback = Optional[Callable[[str], None]]


def _ensure_package_path() -> None:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_package_path()

from dicom_nifti import dicom_folder_to_pre_inference_nifti  # noqa: E402


def _safe_stem(name: str) -> str:
    """파일명에 쓸 수 없는 문자 제거·치환."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip())
    return s or "volume"


def _zip_files_in_folder(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".zip"
    )


def _folder_has_dicom(folder: Path) -> bool:
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".dcm":
            return True
    return False


def _count_dicom(folder: Path) -> int:
    return sum(
        1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".dcm"
    )


def _extract_zip_tolerant(zf: Path, dest: Path) -> None:
    """일반 해제 실패 시(로컬 헤더 UTF-16 이름 등) 이름 불일치 검사를 건너뛰고 추출."""
    try:
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(dest)
        return
    except (zipfile.BadZipFile, NotImplementedError, UnicodeDecodeError, OSError):
        pass

    import struct
    from zipfile import ZipExtFile

    def _open_skip_name_check(self, name, mode="r", pwd=None, *, force_zip64=False):
        if mode not in {"r", "U", "rU"}:
            raise ValueError("open() requires mode 'r'")
        pwd = pwd or self.pwd
        zinfo = self.getinfo(name) if isinstance(name, str) else name
        self._fileRefCnt += 1
        zef_file = zipfile._SharedFile(
            self.fp, zinfo.header_offset, self._fpclose, self._lock, lambda: self._writing
        )
        try:
            fheader = zef_file.read(zipfile.sizeFileHeader)
            if len(fheader) != zipfile.sizeFileHeader:
                raise zipfile.BadZipFile("Truncated file header")
            fheader = struct.unpack(zipfile.structFileHeader, fheader)
            if fheader[zipfile._FH_SIGNATURE] != zipfile.stringFileHeader:
                raise zipfile.BadZipFile("Bad magic number for file header")
            zef_file.read(fheader[zipfile._FH_FILENAME_LENGTH])
            if fheader[zipfile._FH_EXTRA_FIELD_LENGTH]:
                zef_file.read(fheader[zipfile._FH_EXTRA_FIELD_LENGTH])
            if zinfo.flag_bits & 0x1:
                raise NotImplementedError("encrypted zip not supported")
            return ZipExtFile(zef_file, mode, zinfo, pwd, True)
        except Exception:
            zef_file.close()
            raise

    orig_open = zipfile.ZipFile.open
    zipfile.ZipFile.open = _open_skip_name_check  # type: ignore[method-assign]
    try:
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(dest)
    finally:
        zipfile.ZipFile.open = orig_open  # type: ignore[method-assign]


def _extract_zips_into_folder(
    folder: Path,
    zips: list[Path],
    progress: ProgressCallback = None,
) -> None:
    for zf in zips:
        if progress:
            progress(f"[압축해제] {zf.name} -> {folder}")
        _extract_zip_tolerant(zf, folder)


def _prepare_case_folder(
    case_dir: Path,
    progress: ProgressCallback = None,
) -> None:
    """``.zip`` 이 있으면 케이스 폴더에 풀고, ``.dcm`` 존재 여부를 확인합니다."""
    case_dir = case_dir.expanduser().resolve()
    zips = _zip_files_in_folder(case_dir)
    # 이미 충분한 DICOM이 있으면 zip 해제를 건너뛰어 손상 zip으로 실패하는 것을 피함
    if zips and _count_dicom(case_dir) < 50:
        try:
            _extract_zips_into_folder(case_dir, zips, progress=progress)
        except Exception as e:
            if progress:
                progress(f"[압축해제 경고] {e}")
        if progress:
            progress(f"[DICOM 검색] {case_dir}")
    elif zips and progress:
        progress(f"[압축해제 생략] 이미 .dcm {_count_dicom(case_dir)}개 존재")

    if not _folder_has_dicom(case_dir):
        msg = f".dcm 파일을 찾지 못했습니다: {case_dir}"
        if zips:
            msg += f" (.zip {len(zips)}개 처리 후)"
        raise FileNotFoundError(msg)


def _release_memory() -> None:
    gc.collect()


def _best_dicom_folder(case_dir: Path) -> tuple[Path, int]:
    """케이스 트리에서 ``.dcm`` 이 가장 많은 폴더(직계 파일 기준)를 고릅니다."""
    best = case_dir
    best_n = sum(
        1 for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"
    )
    for sub in case_dir.rglob("*"):
        if not sub.is_dir():
            continue
        try:
            n = sum(
                1 for p in sub.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"
            )
        except OSError:
            continue
        if n > best_n:
            best, best_n = sub, n
    return best, best_n


def run_single_case(
    case_dir: Path,
    out_file: Path,
    *,
    patient_origin: bool = False,
    progress: ProgressCallback = None,
    min_slices: int = 50,
) -> None:
    case_dir = case_dir.expanduser().resolve()
    out_file = out_file.expanduser().resolve()

    _prepare_case_folder(case_dir, progress=progress)
    dicom_dir, n_slices = _best_dicom_folder(case_dir)
    if n_slices < min_slices:
        raise FileNotFoundError(
            f"CBCT 시리즈로 볼 슬라이스가 부족합니다 "
            f"(최대 {n_slices}장, 최소 {min_slices}장 필요): {case_dir}"
        )
    if progress and dicom_dir != case_dir:
        progress(f"[DICOM 폴더] {dicom_dir} ({n_slices}장)")
    try:
        dicom_folder_to_pre_inference_nifti(
            dicom_dir,
            out_file,
            progress=progress,
            patient_origin=patient_origin,
        )
    finally:
        _release_memory()


def _worker_argv(
    case_dir: Path,
    out_file: Path,
    *,
    patient_origin: bool,
    verbose: bool,
    min_slices: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-case",
        str(case_dir),
        "--worker-output",
        str(out_file),
        "--min-slices",
        str(min_slices),
    ]
    if patient_origin:
        cmd.append("--patient-origin")
    if verbose:
        cmd.append("-v")
    return cmd


def _run_case_subprocess(
    case_dir: Path,
    out_file: Path,
    *,
    patient_origin: bool,
    verbose: bool,
    min_slices: int,
) -> int:
    """케이스를 별도 프로세스에서 실행. stdout/stderr는 콘솔에 직접 연결해 인코딩 깨짐을 방지."""
    cmd = _worker_argv(
        case_dir,
        out_file,
        patient_origin=patient_origin,
        verbose=verbose,
        min_slices=min_slices,
    )
    result = subprocess.run(
        cmd,
        stdout=None,
        stderr=None,
    )
    return result.returncode


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--patient-origin",
        action="store_true",
        help="NIfTI 원점을 DICOM ImagePositionPatient 기준으로 둡니다 (기본: ITK 원점).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="진행 메시지 출력",
    )
    p.add_argument(
        "--min-slices",
        type=int,
        default=50,
        help="CBCT로 인정할 최소 .dcm 장수 (기본 50). 미만이면 건너뜀/실패.",
    )


def _run_worker_mode(args: argparse.Namespace) -> int:
    progress = print if args.verbose else None
    case_dir = args.worker_case.expanduser().resolve()
    out_file = args.worker_output.expanduser().resolve()

    if not case_dir.is_dir():
        print(f"오류: 디렉터리가 아닙니다: {case_dir}", file=sys.stderr)
        return 2

    if args.verbose:
        print(f"[변환] {case_dir} -> {out_file}")

    try:
        run_single_case(
            case_dir,
            out_file,
            patient_origin=args.patient_origin,
            progress=progress,
            min_slices=args.min_slices,
        )
        return 0
    except Exception as e:
        print(f"[실패] {case_dir}: {e}", file=sys.stderr)
        return 1


def main() -> int:
    p = argparse.ArgumentParser(
        description="루트 아래 각 하위 폴더의 DICOM을 하나의 .nii.gz로 변환해 루트/output에 저장합니다.",
    )
    p.add_argument(
        "root",
        nargs="?",
        type=Path,
        help="하위에 DICOM이 들어 있는 폴더들이 있는 루트 경로",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="저장 폴더 (기본: <루트>/output)",
    )
    p.add_argument(
        "--worker-case",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--worker-output",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--in-process",
        action="store_true",
        help="케이스마다 서브프로세스 대신 같은 프로세스에서 처리 (메모리 해제가 덜 확실함)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="출력 .nii.gz 가 이미 있으면 해당 케이스를 건너뜁니다",
    )
    _add_common_args(p)
    args = p.parse_args()

    if args.worker_case is not None or args.worker_output is not None:
        if args.worker_case is None or args.worker_output is None:
            p.error("--worker-case 와 --worker-output 은 함께 지정해야 합니다.")
        return _run_worker_mode(args)

    if args.root is None:
        p.error("root 경로가 필요합니다.")

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"오류: 디렉터리가 아닙니다: {root}", file=sys.stderr)
        return 2

    out_dir = (args.output_dir or (root / "output")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    subdirs = sorted(
        d for d in root.iterdir() if d.is_dir() and d.resolve() != out_dir
    )
    if not subdirs:
        print(f"처리할 하위 폴더가 없습니다: {root}", file=sys.stderr)
        return 1

    progress = print if args.verbose else None

    ok = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for sub in subdirs:
        stem = _safe_stem(sub.name)
        out_file = out_dir / f"{stem}.nii.gz"

        if args.skip_existing and out_file.is_file() and out_file.stat().st_size > 0:
            skipped += 1
            if args.verbose:
                print(f"[건너뜀] 이미 존재: {out_file.name}")
            continue

        # zip 이 없고 슬라이스가 부족하면 서브프로세스 없이 즉시 건너뜀
        zips = _zip_files_in_folder(sub)
        if not zips:
            _, n_slices = _best_dicom_folder(sub)
            if n_slices < args.min_slices:
                skipped += 1
                if args.verbose:
                    print(
                        f"[건너뜀] CBCT 슬라이스 부족 ({n_slices}<{args.min_slices}): {sub.name}"
                    )
                continue

        if args.in_process:
            if args.verbose:
                print(f"[변환] {sub} -> {out_file}")
            try:
                run_single_case(
                    sub,
                    out_file,
                    patient_origin=args.patient_origin,
                    progress=progress,
                    min_slices=args.min_slices,
                )
                ok += 1
            except Exception as e:
                failed.append((str(sub), str(e)))
                print(f"[실패] {sub}: {e}", file=sys.stderr)
            continue

        rc = _run_case_subprocess(
            sub,
            out_file,
            patient_origin=args.patient_origin,
            verbose=args.verbose,
            min_slices=args.min_slices,
        )
        if rc == 0:
            ok += 1
        else:
            failed.append((str(sub), f"처리 실패 (exit code {rc})"))

    print(
        f"완료: 성공 {ok}개, 건너뜀 {skipped}개, 실패 {len(failed)}개 → {out_dir}"
    )
    if failed:
        for path, err in failed:
            print(f"  - {path}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
