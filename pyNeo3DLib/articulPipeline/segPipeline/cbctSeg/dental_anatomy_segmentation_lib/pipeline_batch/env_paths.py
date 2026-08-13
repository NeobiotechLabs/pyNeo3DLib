"""env 파일·경로 해석 및 입출력/nnU-Net 모델 디렉터리 검증."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

# 저장소 루트 (``.env.dental_cbct`` 위치). ``dental_anatomy_segmentation_lib/pipeline_batch/`` 기준 ``parents[2]``.
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_INPUT = "DENTAL_CBCT_INPUT_DIR"
ENV_OUTPUT = "DENTAL_CBCT_OUTPUT_DIR"
ENV_NNUNET_MODEL = "DENTAL_NNUNET_MODEL_PATH"
DEFAULT_ENV_FILE = REPO_ROOT / ".env.dental_cbct"


def strip_env_value(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    return raw


def apply_env_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8-sig")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = strip_env_value(rest)
        if os.environ.get(key) is None:
            os.environ[key] = val


def load_env_file(cli_path: Optional[Path]) -> Optional[Path]:
    if cli_path is not None:
        p = cli_path.expanduser().resolve()
        if not p.is_file():
            hint = (
                f"  저장소 루트에 ``{DEFAULT_ENV_FILE.name}`` 를 만들고 "
                f"``{ENV_INPUT}=...`` / ``{ENV_OUTPUT}=...`` 등을 ``KEY=VALUE`` 로 적으세요."
            )
            raise FileNotFoundError(
                f"--env-file 경로에 파일이 없습니다: {p}\n{hint}"
            )
        apply_env_file(p)
        return p
    if DEFAULT_ENV_FILE.is_file():
        apply_env_file(DEFAULT_ENV_FILE.resolve())
        return DEFAULT_ENV_FILE.resolve()
    return None


def preparse_env_file_arg() -> Optional[Path]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", type=Path, default=None, metavar="PATH")
    args, _ = pre.parse_known_args()
    return args.env_file


def path_from_user_string(raw: str) -> Path:
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    s = s.replace("\\", "/")
    return Path(s).expanduser().resolve()


def optional_batch_dir(cli_path: Optional[Path], env_name: str) -> Optional[Path]:
    if cli_path is not None:
        return path_from_user_string(str(cli_path))
    raw = os.environ.get(env_name, "").strip()
    if raw:
        return path_from_user_string(raw)
    return None


def require_io_directories(
    cli_input: Optional[Path], cli_output: Optional[Path]
) -> tuple[Path, Path]:
    inputs_dir = optional_batch_dir(cli_input, ENV_INPUT)
    outputs_root = optional_batch_dir(cli_output, ENV_OUTPUT)
    if inputs_dir is not None and outputs_root is not None:
        return inputs_dir, outputs_root
    if inputs_dir is None and outputs_root is None:
        header = "입력·출력 경로가 모두 비어 있습니다."
    elif inputs_dir is None:
        header = "입력 경로가 비어 있습니다."
    else:
        header = "출력 경로가 비어 있습니다."
    lines = [header, "다음을 설정하세요:"]
    if inputs_dir is None:
        lines.append(f"  - 입력: 환경변수 {ENV_INPUT} 또는 --input-dir DIR")
    if outputs_root is None:
        lines.append(f"  - 출력: 환경변수 {ENV_OUTPUT} 또는 --output-dir DIR")
    lines.append(
        "  (셸에서 export 하거나, --env-file / 저장소 루트 .env.dental_cbct 에 두 키를 적으세요.)"
    )
    raise RuntimeError("\n".join(lines))


def optional_nnunet_model_dir(cli_path: Optional[Path]) -> Optional[Path]:
    if cli_path is not None:
        p = path_from_user_string(str(cli_path))
        if not p.is_dir():
            raise NotADirectoryError(f"--model-dir 이 유효한 폴더가 아닙니다: {p}")
        return p
    raw = os.environ.get(ENV_NNUNET_MODEL, "").strip()
    if not raw:
        return None
    p = path_from_user_string(raw)
    if not p.is_dir():
        hint = ""
        if os.name != "nt" and (("\\" in raw) or raw.strip().startswith("//")):
            hint = (
                " Linux에서는 `\\\\서버\\공유` 대신 SMB 마운트 후 "
                "`/mnt/.../others_seg_model` 같은 경로를 env에 넣으세요."
            )
        raise NotADirectoryError(
            f"환경변수 {ENV_NNUNET_MODEL} 이 가리키는 경로가 유효한 폴더가 아닙니다: {p}.{hint}"
        )
    return p
