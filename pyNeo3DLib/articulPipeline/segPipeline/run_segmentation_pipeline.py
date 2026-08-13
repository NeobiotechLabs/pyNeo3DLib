#!/usr/bin/env python3
"""
DICOM 시리즈 → NIfTI 변환 → CBCT 세그멘테이션 통합 파이프라인.

입력:
    1. DCM 시리즈 폴더 경로 — ``.dcm`` 이 바로 들어 있는 폴더(단일 케이스) 또는
       케이스 폴더들이 들어 있는 루트(배치).
    2. 가중치 모델 경로 — ``--model`` 로 지정. 미지정 시 파이프라인 폴더 안
       ``model/`` (``segPipeline/model``)을 기본으로 사용하며, nnU-Net 번들
       폴더 또는 그 상위 폴더에서 ``dataset.json`` 을 자동 탐색합니다
       (바로 없으면 ``others_seg_model`` 등 하위 폴더). 같은 루트에서 랜드마크
       가중치(``landmarks_model``)도 함께 찾습니다.
    3. 최종 결과 저장 경로.

동작:
    1단계  ``dcm2nii/batch_dicom_folders_to_nifti.py`` 로 ``*.nii.gz`` 볼륨 생성
           (케이스별 서브프로세스 — ITK 메모리 반환). 임시 폴더는 **저장 경로
           안**에 만들고, 변환이 끝나면 nii.gz 를 저장 경로로 옮긴 뒤 임시
           폴더는 바로 삭제됩니다.
    2단계  ``cbctLandmark`` 로 nii.gz 생성 직후 랜드마크(기본 ANS,PNS,N) 좌표
           추정 (케이스별 서브프로세스 — 모델/GPU 메모리 반환). 가중치 폴더는
           파이프라인 폴더 안 ``model/`` (없으면 그 형제 ``model/``)에서
           ``landmarks_model`` 우선으로 자동 탐색하고 ``--landmark-models`` 로
           변경할 수 있습니다. 가중치를 찾지 못하면 건너뛰며
           ``--no-landmarks`` 로 생략할 수 있습니다.
    3단계  ``cbctSeg`` 세그멘테이션 (``pipeline_batch`` 워커)으로 라벨맵·센터라인·
           STL 메쉬 생성.
    결과물을 지정한 저장 경로에 **바로** 저장 — nii.gz 이름의 하위 폴더를
    만들지 않습니다. 모든 산출물은 ``{케이스이름}_...`` 접두사가 붙어
    같은 폴더에 저장해도 서로 충돌하지 않습니다.

산출물 (저장 경로 바로 아래):
    - ``{케이스}.nii.gz``            입력 볼륨 (기본 저장, --no-keep-nifti 시 삭제)
    - ``{케이스}_merged.mrk.json``   랜드마크 좌표 (Slicer 마크업, LPS mm — --landmark-models 지정 시)
    - ``{케이스}_pred.nii.gz``       세그멘테이션 라벨맵
    - ``{케이스}_centerline.json``   좌/우 신경관 중심선 (--no-restore-mandibular 시 생략)
    - ``{케이스}_{구조이름}.stl``    메쉬 (--no-export-meshes 시 생략). 구조 이름은
      ``articulPipeline/structure_names.json`` 공통 규약 사용:
      ``{케이스}_maxilla.stl``, ``{케이스}_mandible.stl``,
      ``{케이스}_nerve_canal.stl``, ``{케이스}_maxillary_sinus.stl``

예::

    # model/ 을 지정하지 않으면 segPipeline/model 에서 세그멘테이션·랜드마크 가중치를 자동 탐색
    python run_segmentation_pipeline.py --input D:\\data\\case01 --output D:\\results
    python run_segmentation_pipeline.py -i ./cases -m ./model -o ./out -v
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
DCM2NII_SCRIPT = PIPELINE_ROOT / "dcm2nii" / "batch_dicom_folders_to_nifti.py"
SEG_LIB_DIR = PIPELINE_ROOT / "cbctSeg" / "dental_anatomy_segmentation_lib"
# 가중치 기본 루트: 파이프라인 폴더 안 model/ (세그멘테이션 others_seg_model +
# 랜드마크 landmarks_model 이 함께 들어 있는 폴더)
DEFAULT_MODEL_ROOT = PIPELINE_ROOT / "model"
# 랜드마크 가중치 탐색 순서: 1) 파이프라인 폴더 안 model/  2) 그 형제 model/
DEFAULT_LANDMARK_ROOTS = [DEFAULT_MODEL_ROOT, PIPELINE_ROOT.parent / "model"]


def _safe_stem(name: str) -> str:
    """파일명에 쓸 수 없는 문자 제거·치환 (dcm2nii 규칙과 동일)."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip())
    return s or "volume"


def resolve_model_dir(model_dir: Path) -> Path:
    """모델 경로에서 nnU-Net 번들(``dataset.json`` 이 있는 폴더)을 찾아 반환.

    - 경로 바로 아래에 ``dataset.json`` 이 있으면 그대로 사용.
    - 없으면 ``others_seg_model`` 하위 폴더 우선 탐색.
    - 그래도 없으면 하위에서 ``dataset.json`` 을 찾아 그 폴더를 사용.
    """
    if (model_dir / "dataset.json").is_file():
        return model_dir

    preferred = model_dir / "others_seg_model"
    if (preferred / "dataset.json").is_file():
        return preferred

    found = sorted(
        model_dir.rglob("dataset.json"),
        key=lambda p: len(p.parts),  # 얕은 순서 우선
    )
    if found:
        return found[0].parent

    raise FileNotFoundError(
        f"모델 폴더에서 dataset.json(nnU-Net 번들)을 찾지 못했습니다: {model_dir}"
    )


def _has_landmark_weights(root: Path, landmarks: list[str]) -> bool:
    """root 바로 아래에 모든 랜드마크 가중치가 있는지.

    ``<팩>/<랜드마크>/<스케일>/*.pth`` 또는 ``<랜드마크>/<스케일>/*.pth``
    구조를 인정합니다.
    """
    if not root.is_dir():
        return False
    for lm in landmarks:
        if not (any(root.glob(f"*/{lm}/*/*.pth")) or any(root.glob(f"{lm}/*/*.pth"))):
            return False
    return True


def resolve_landmark_models_dir(models_root: Path, landmarks: list[str]) -> Path | None:
    """models_root 에서 cbctLandmark 가중치 폴더를 찾아 반환. 없으면 None.

    - models_root 자체에 가중치 구조가 있으면 그대로 사용.
    - 없으면 ``landmarks_model`` 하위 폴더 우선.
    - 그래도 없으면 하위 폴더 중 모든 랜드마크 가중치가 있는 폴더를 선택.
    """
    if _has_landmark_weights(models_root, landmarks):
        return models_root

    preferred = models_root / "landmarks_model"
    if _has_landmark_weights(preferred, landmarks):
        return preferred

    if models_root.is_dir():
        for sub in sorted(p for p in models_root.iterdir() if p.is_dir()):
            if _has_landmark_weights(sub, landmarks):
                return sub
    return None


def _has_direct_dicom_or_zip(folder: Path) -> bool:
    """폴더 '바로 아래'에 .dcm 파일이나 .zip 이 있는지 (단일 케이스 판별)."""
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in (".dcm", ".zip"):
            return True
    return False


def _run(cmd: list[str], verbose: bool) -> None:
    if verbose:
        print(f"[실행] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"단계 실패 (exit code {result.returncode}): {' '.join(cmd)}")


def convert_dicom_to_nifti(
    dcm_dir: Path,
    nii_dir: Path,
    *,
    min_slices: int,
    patient_origin: bool,
    verbose: bool,
) -> list[Path]:
    """1단계 — dcm2nii 스크립트를 서브프로세스로 실행해 ``nii_dir`` 에 NIfTI 생성."""
    common = ["--min-slices", str(min_slices)]
    if patient_origin:
        common.append("--patient-origin")
    if verbose:
        common.append("-v")

    if _has_direct_dicom_or_zip(dcm_dir):
        # 단일 케이스: 폴더 자체가 시리즈
        out_file = nii_dir / f"{_safe_stem(dcm_dir.name)}.nii.gz"
        cmd = [
            sys.executable,
            str(DCM2NII_SCRIPT),
            "--worker-case",
            str(dcm_dir),
            "--worker-output",
            str(out_file),
            *common,
        ]
        _run(cmd, verbose)
    else:
        # 배치: 하위 폴더들이 각각 케이스
        cmd = [
            sys.executable,
            str(DCM2NII_SCRIPT),
            str(dcm_dir),
            "-o",
            str(nii_dir),
            *common,
        ]
        _run(cmd, verbose)

    volumes = sorted(nii_dir.glob("*.nii.gz"))
    if not volumes:
        raise FileNotFoundError(f"생성된 NIfTI 파일이 없습니다: {nii_dir}")
    return volumes


def _flatten_landmark_outputs(vol: Path, output_dir: Path) -> None:
    """cbctLandmark 가 만든 하위 폴더(``{볼륨}.nii/``)의 .mrk.json 을 지정 경로로 옮김.

    cbctLandmark 는 ``{output}/{볼륨}.nii/{케이스}_merged.mrk.json`` 형태로
    하위 폴더를 만들어 저장하므로, 파이프라인의 '하위 폴더 없이 평평하게
    저장' 규칙에 맞게 파일만 ``output_dir`` 로 올리고 폴더는 삭제합니다.
    """
    name = vol.name
    if name.lower().endswith(".nii.gz"):
        sub_name = name[: -len(".gz")]
    elif name.lower().endswith(".nii"):
        sub_name = name
    else:
        sub_name = vol.stem
    sub_dir = output_dir / sub_name
    if not sub_dir.is_dir():
        return
    for mrk in sorted(sub_dir.glob("*.mrk.json")):
        shutil.move(str(mrk), str(output_dir / mrk.name))
    shutil.rmtree(sub_dir, ignore_errors=True)


def run_landmark_detection(
    volumes: list[Path],
    output_dir: Path,
    models_dir: Path,
    *,
    landmarks: str,
    verbose: bool,
) -> tuple[int, list[str]]:
    """2단계 — nii.gz 생성 직후 cbctLandmark 로 랜드마크 좌표 추정.

    ``python -m cbctLandmark.cli`` 를 서브프로세스로 실행해(모델/GPU 메모리
    반환 — 1단계 dcm2nii 서브프로세스와 같은 이유) 랜드마크를 찾고, 결과
    ``.mrk.json`` 을 ``output_dir`` 바로 아래로 옮겨 평평하게 저장합니다.
    """
    child_env = os.environ.copy()
    old_pp = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = str(PIPELINE_ROOT) + (
        os.pathsep + old_pp if old_pp else ""
    )

    print(
        f"랜드마크: 케이스 {len(volumes)}건 | 대상 {landmarks}\n"
        f"  모델: {models_dir}\n"
        f"  출력: {output_dir}  (.mrk.json, LPS mm)",
        flush=True,
    )

    n_ok = 0
    failed: list[str] = []
    for vol in volumes:
        print(f"── 랜드마크 탐색: {vol.name}", flush=True)
        cmd = [
            sys.executable,
            "-m",
            "cbctLandmark.cli",
            "--volume",
            str(vol),
            "--landmarks",
            landmarks,
            "--models-dir",
            str(models_dir),
            "--output-dir",
            str(output_dir),
        ]
        if verbose:
            print(f"[실행] {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, env=child_env)
        if result.returncode != 0:
            failed.append(vol.name)
            print(
                f"랜드마크 실패 [{vol.name}]: exit code {result.returncode}",
                file=sys.stderr,
                flush=True,
            )
            continue
        _flatten_landmark_outputs(vol, output_dir)
        n_ok += 1

    return n_ok, failed


def run_segmentation(
    volumes: list[Path],
    output_dir: Path,
    model_dir: Path,
    *,
    env_file: Path | None,
    restore_mandibular: bool,
    export_meshes: bool,
) -> tuple[int, list[str]]:
    """3단계 — 각 NIfTI 를 세그멘테이션해 ``output_dir`` 에 바로 저장.

    ``run_segmentation.py`` 가 사용하는 ``pipeline_batch`` 워커를 직접 호출하되,
    케이스 폴더(``outputs_root/vol.stem``)를 만들지 않고 ``output_dir`` 자체를
    작업 폴더로 사용해 산출물을 지정한 경로에 평평하게 저장합니다.
    """
    # pipeline_batch / dental_anatomy_segmentation_lib import 경로 준비
    for p in (str(SEG_LIB_DIR.parent), str(SEG_LIB_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # SEG_LIB_DIR 를 sys.path 에 추가한 뒤에야 import 가능 (정적 분석기는 해석 불가)
    from pipeline_batch.env_paths import load_env_file  # type: ignore[import-not-found]
    from pipeline_batch.progress import cuda_device_count  # type: ignore[import-not-found]
    from pipeline_batch.worker import run_one_nifti_case  # type: ignore[import-not-found]

    load_env_file(env_file)

    n_gpu = cuda_device_count()
    dev = "cuda:0" if n_gpu > 0 else None
    print(
        f"세그멘테이션: 케이스 {len(volumes)}건 | device={dev or 'cpu'}\n"
        f"  모델: {model_dir}\n"
        f"  출력: {output_dir}  (하위 폴더 없이 바로 저장)",
        flush=True,
    )

    n_ok = 0
    failed: list[str] = []
    for vol in volumes:
        print(f"── 시작: {vol.name}", flush=True)
        try:
            line = run_one_nifti_case(
                root_str=str(SEG_LIB_DIR),
                vol_str=str(vol),
                case_dir_str=str(output_dir),  # ← nii.gz 이름 폴더 없이 지정 경로에 저장
                nnunet_device=dev,
                model_dir_str=str(model_dir),
                restore_mandibular=restore_mandibular,
                export_meshes=export_meshes,
            )
            n_ok += 1
            print(line, flush=True)
        except Exception as exc:
            failed.append(vol.name)
            print(f"실패 [{vol.name}]: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    return n_ok, failed


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "DICOM 시리즈 → NIfTI (dcm2nii) → CBCT 세그멘테이션 (cbctSeg). "
            "결과는 지정한 저장 경로에 하위 폴더 없이 바로 저장됩니다."
        ),
    )
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        metavar="DIR",
        help="DCM 시리즈 폴더 (.dcm 폴더 또는 케이스 폴더들의 루트)",
    )
    p.add_argument(
        "--model",
        "-m",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            f"가중치 모델 경로 — 미지정 시 {DEFAULT_MODEL_ROOT} 사용. "
            "nnU-Net 번들 폴더 또는 그 상위 폴더. "
            "dataset.json 이 바로 없으면 others_seg_model 등 하위에서 자동 탐색"
        ),
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        metavar="DIR",
        help="최종 결과 저장 경로",
    )
    p.add_argument(
        "--landmark-models",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "cbctLandmark 가중치 폴더 — 미지정 시 --model 루트(기본 "
            f"{DEFAULT_MODEL_ROOT}) 에서 자동 탐색(landmarks_model 우선). "
            "찾으면 nii.gz 생성 후 랜드마크를 찾아 저장 경로에 "
            "{케이스}_merged.mrk.json 저장"
        ),
    )
    p.add_argument(
        "--no-landmarks",
        action="store_true",
        help="랜드마크 탐색 단계 건너뜀 (기본: 가중치 자동 탐색 후 실행)",
    )
    p.add_argument(
        "--landmarks",
        default="ANS,PNS,N",
        metavar="NAMES",
        help="찾을 랜드마크, 쉼표 구분 (기본: ANS,PNS,N)",
    )
    p.add_argument(
        "--no-keep-nifti",
        action="store_true",
        help="중간 산출물인 볼륨 .nii.gz 를 저장 경로에 남기지 않음 (기본: 저장)",
    )
    p.add_argument(
        "--min-slices",
        type=int,
        default=50,
        help="CBCT 로 인정할 최소 .dcm 장수 (기본 50, dcm2nii 옵션)",
    )
    p.add_argument(
        "--patient-origin",
        action="store_true",
        help="NIfTI 원점을 DICOM ImagePositionPatient 기준으로 둡니다 (dcm2nii 옵션)",
    )
    p.add_argument(
        "--no-restore-mandibular",
        action="store_true",
        help="하악 신경관 복원 단계를 건너뜀",
    )
    p.add_argument(
        "--no-export-meshes",
        action="store_true",
        help="메쉬(STL) 내보내기를 건너뜀",
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="KEY=VALUE 형식 파일로 환경변수 보충 (cbctSeg .env 와 동일 형식)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="진행 메시지 출력")
    args = p.parse_args()

    dcm_dir = args.input.expanduser().resolve()
    # 가중치 루트 (--model 미지정 시 segPipeline/model 기본)
    model_root = (
        args.model.expanduser().resolve()
        if args.model is not None
        else DEFAULT_MODEL_ROOT
    )
    output_dir = args.output.expanduser().resolve()

    if not dcm_dir.is_dir():
        print(f"오류: DCM 시리즈 폴더가 아닙니다: {dcm_dir}", file=sys.stderr)
        return 2
    if not model_root.is_dir():
        print(f"오류: 모델 폴더가 아닙니다: {model_root}", file=sys.stderr)
        return 2
    try:
        model_dir = resolve_model_dir(model_root)
    except FileNotFoundError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    # 임시 폴더는 지정된 저장 경로 안에 만들고 끝나면 삭제
    nii_dir = Path(tempfile.mkdtemp(prefix="nii_tmp_", dir=output_dir))
    try:
        # ── 1단계: DICOM → NIfTI ──
        print(f"[1/3] DICOM → NIfTI 변환: {dcm_dir} → {nii_dir}", flush=True)
        volumes = convert_dicom_to_nifti(
            dcm_dir,
            nii_dir,
            min_slices=args.min_slices,
            patient_origin=args.patient_origin,
            verbose=args.verbose,
        )
        print(f"  생성됨: {', '.join(v.name for v in volumes)}", flush=True)

        # 변환이 끝나면 nii.gz 를 저장 경로로 바로 옮기고 임시 폴더는 삭제
        # (--no-keep-nifti 시에만 세그멘테이션 입력으로 쓰다가 마지막에 삭제)
        if not args.no_keep_nifti:
            moved: list[Path] = []
            for vol in volumes:
                dst = output_dir / vol.name
                shutil.move(str(vol), str(dst))
                moved.append(dst)
            volumes = moved
            shutil.rmtree(nii_dir, ignore_errors=True)

        # ── 2단계: 랜드마크 탐색 (nii.gz 생성 직후) ──
        lm_ok = 0
        lm_failed: list[str] = []
        lm_enabled = False
        if args.no_landmarks:
            print("[2/3] --no-landmarks — 랜드마크 단계 건너뜀", flush=True)
        else:
            if args.landmark_models is not None:
                lm_roots = [args.landmark_models.expanduser().resolve()]
            else:
                # 세그멘테이션 가중치 루트(--model 또는 기본 segPipeline/model)를
                # 우선 탐색하고, 이어서 기본 후보 폴더들을 순서대로 탐색
                lm_roots = [model_root] + [
                    r for r in DEFAULT_LANDMARK_ROOTS if r != model_root
                ]
            lm_names = [s.strip() for s in args.landmarks.split(",") if s.strip()]
            lm_dir = None
            for lm_root in lm_roots:
                lm_dir = resolve_landmark_models_dir(lm_root, lm_names)
                if lm_dir is not None:
                    break
            if lm_dir is not None:
                print(
                    f"[2/3] 랜드마크 탐색 ({args.landmarks}) → {output_dir}\n"
                    f"      가중치 폴더: {lm_dir}",
                    flush=True,
                )
                lm_enabled = True
                lm_ok, lm_failed = run_landmark_detection(
                    volumes,
                    output_dir,
                    lm_dir,
                    landmarks=args.landmarks,
                    verbose=args.verbose,
                )
            else:
                roots_str = ", ".join(str(r) for r in lm_roots)
                print(
                    f"경고: {roots_str} 아래에서 랜드마크({args.landmarks}) 가중치를 "
                    f"찾지 못해 랜드마크 단계를 건너뜁니다",
                    file=sys.stderr,
                    flush=True,
                )

        # ── 3단계: 세그멘테이션 → 지정 경로에 바로 저장 ──
        print(f"[3/3] CBCT 세그멘테이션 → {output_dir}", flush=True)
        n_ok, failed = run_segmentation(
            volumes,
            output_dir,
            model_dir,
            env_file=args.env_file,
            restore_mandibular=not args.no_restore_mandibular,
            export_meshes=not args.no_export_meshes,
        )
    finally:
        shutil.rmtree(nii_dir, ignore_errors=True)

    summary = f"완료: 세그멘테이션 성공 {n_ok}건, 실패 {len(failed)}건"
    if lm_enabled:
        summary += f" | 랜드마크 성공 {lm_ok}건, 실패 {len(lm_failed)}건"
    print(f"{summary} → {output_dir}", flush=True)
    if failed:
        for name in failed:
            print(f"  - 세그멘테이션 실패: {name}", file=sys.stderr)
    if lm_failed:
        for name in lm_failed:
            print(f"  - 랜드마크 실패: {name}", file=sys.stderr)
    return 1 if failed or lm_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
