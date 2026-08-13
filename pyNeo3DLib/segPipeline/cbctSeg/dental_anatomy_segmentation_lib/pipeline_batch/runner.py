"""단일 NIfTI 또는 배치 추론 오케스트레이션."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from pipeline_batch.cli import build_parser
from pipeline_batch.env_paths import (
    ENV_INPUT,
    ENV_OUTPUT,
    load_env_file,
    optional_batch_dir,
    optional_nnunet_model_dir,
    preparse_env_file_arg,
)
from pipeline_batch.progress import (
    create_case_progress,
    cuda_device_count,
    try_tqdm_cls,
)
from pipeline_batch.worker import run_one_nifti_case

_repo = str(Path(__file__).resolve().parent.parent.resolve())
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def main() -> None:
    load_env_file(preparse_env_file_arg())

    cli = build_parser().parse_args()
    quiet = bool(cli.quiet)

    # ── 경로 해석 (CLI 인자 > 환경변수) ──
    input_path: Optional[Path]
    if cli.input is not None:
        input_path = cli.input.expanduser().resolve()
    else:
        input_path = optional_batch_dir(None, ENV_INPUT)

    outputs_root: Optional[Path]
    if cli.output_dir is not None:
        outputs_root = cli.output_dir.expanduser().resolve()
    else:
        outputs_root = optional_batch_dir(None, ENV_OUTPUT)

    model_dir = optional_nnunet_model_dir(cli.model_dir)

    missing = []
    if input_path is None:
        missing.append(f"입력: --input FILE 또는 환경변수 {ENV_INPUT}")
    if outputs_root is None:
        missing.append(f"출력: --output-dir DIR 또는 환경변수 {ENV_OUTPUT}")
    if model_dir is None:
        missing.append(f"모델: --model-dir DIR 또는 환경변수 {ENV_NNUNET_MODEL}")
    if missing:
        raise RuntimeError("경로가 지정되지 않았습니다:\n  - " + "\n  - ".join(missing))

    outputs_root.mkdir(parents=True, exist_ok=True)

    from dental_anatomy_segmentation_lib.pipeline_runtime import (
        install_batch_cleanup_handlers,
        release_accelerator_memory,
    )

    # ── 입력이 파일인지 폴더인지 확인 ──
    volumes: list[Path]
    if input_path.is_file():
        name_low = input_path.name.lower()
        if name_low.endswith(".nii.gz") or name_low.endswith(".nii"):
            volumes = [input_path]
        else:
            raise ValueError(f"입력 파일이 NIfTI (.nii / .nii.gz) 가 아닙니다: {input_path}")
    elif input_path.is_dir():
        volumes = sorted(input_path.glob("*.nii.gz"))
        if not volumes:
            raise FileNotFoundError(f"입력 디렉터리에 *.nii.gz 파일이 없습니다: {input_path}")
    else:
        raise FileNotFoundError(f"입력이 존재하지 않습니다: {input_path}")

    restore_mandibular = not cli.no_restore_mandibular

    install_batch_cleanup_handlers()

    n_gpu = cuda_device_count()
    dev: Optional[str] = "cuda:0" if n_gpu > 0 else None

    if not quiet:
        print(
            f"입력: {input_path}\n"
            f"출력: {outputs_root}\n"
            f"모델: {model_dir}\n"
            f"케이스 {len(volumes)}건 | device={dev or 'cpu'}",
            flush=True,
        )

    model_dir_str = str(model_dir.resolve())
    tick, close_bar = create_case_progress(len(volumes), "케이스")
    tqdm_cls = try_tqdm_cls()
    n_ok = 0
    n_fail = 0
    failed_names: list[str] = []

    try:
        for vol in volumes:
            vol_name = vol.name
            case_dir = outputs_root / vol.stem

            if not quiet:
                line = f"── 시작: {vol_name}"
                if tqdm_cls is not None:
                    tqdm_cls.write(line)
                else:
                    print(line, flush=True)

            try:
                mesh_ids = cli.mesh_label_ids
                if mesh_ids is not None:
                    mesh_ids = [int(m) for m in mesh_ids]

                line = run_one_nifti_case(
                    root_str=_repo,
                    vol_str=str(vol),
                    case_dir_str=str(case_dir),
                    nnunet_device=dev,
                    model_dir_str=model_dir_str,
                    restore_mandibular=restore_mandibular,
                    export_meshes=not cli.no_export_meshes,
                    mesh_decimation_factor=cli.mesh_decimation,
                    mesh_smoothing_iterations=cli.mesh_smoothing_iterations,
                    mesh_smoothing_factor=cli.mesh_smoothing_factor,
                    mesh_label_ids=mesh_ids,
                )
                n_ok += 1
                if not quiet:
                    if tqdm_cls is not None:
                        tqdm_cls.write(line)
                    else:
                        print(line, flush=True)
            except Exception as exc:
                n_fail += 1
                failed_names.append(vol_name)
                err_line = f"실패 [{vol_name}]: {type(exc).__name__}: {exc}"
                if tqdm_cls is not None:
                    tqdm_cls.write(err_line)
                else:
                    print(err_line, file=sys.stderr, flush=True)
            finally:
                release_accelerator_memory()
            tick()
    finally:
        close_bar()
        print(
            f"처리 요약: 성공 {n_ok}, 실패 {n_fail}, 전체 {len(volumes)}",
            flush=True,
        )
        if failed_names:
            print("실패 목록:", ", ".join(failed_names), flush=True)


if __name__ == "__main__":
    main()
