"""
배치 파이프라인용 메모리·고아 프로세스 정리.

- ``release_accelerator_memory``: 케이스 간 Python GC + CUDA 캐시 비우기
- ``kill_owned_orphan_spawn_workers``: PPID=1 인 nnU-Net spawn 고아 종료
- ``install_batch_cleanup_handlers``: SIGINT/SIGTERM/atexit 시 고아 정리
- ``MemorySnapshot`` / ``log_memory_snapshot``: RSS·GPU 사용량 로그
"""

from __future__ import annotations

import atexit
import gc
import os
import signal
import subprocess
from dataclasses import dataclass
from typing import Optional

_HOOKS_INSTALLED = False


@dataclass(frozen=True)
class MemorySnapshot:
    process_rss_mib: float
    user_rss_mib: float
    mem_available_gib: Optional[float]
    swap_used_mib: Optional[float]
    gpu_lines: tuple[str, ...]


def env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def nnunet_sequential_inference_enabled() -> bool:
    """``DENTAL_NNUNET_SEQUENTIAL=0`` 이면 spawn 사용(빠름). 기본은 단일 프로세스 추론."""
    return env_flag("DENTAL_NNUNET_SEQUENTIAL", default=True)


def pipeline_memory_log_enabled() -> bool:
    """``DENTAL_PIPELINE_MEMORY_LOG=1`` 일 때만 ``[mem]`` RSS·GPU 스냅샷을 출력."""
    return env_flag("DENTAL_PIPELINE_MEMORY_LOG", default=False)


def release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except ImportError:
        pass


def _read_proc_rss_kib(pid: int) -> Optional[float]:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1])
    except OSError:
        return None
    return None


def _user_rss_kib(user: Optional[str] = None) -> float:
    user = user or os.environ.get("USER") or ""
    if not user:
        return 0.0
    try:
        out = subprocess.check_output(
            ["ps", "-u", user, "-o", "rss="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0.0
    total = 0.0
    for line in out.splitlines():
        line = line.strip()
        if line:
            total += float(line)
    return total


def _system_mem_available_kib() -> tuple[Optional[float], Optional[float]]:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            info = {}
            for line in f:
                key, value = line.split(":", 1)
                info[key] = float(value.split()[0])
        return info.get("MemAvailable"), info.get("SwapTotal", 0) - info.get("SwapFree", 0)
    except OSError:
        return None, None


def _gpu_memory_lines() -> tuple[str, ...]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ()

    lines: list[str] = []
    for raw in out.strip().splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 3:
            continue
        idx, used, total = parts
        lines.append(f"GPU{idx} {used}/{total} MiB")
    return tuple(lines)


def capture_memory_snapshot(*, user: Optional[str] = None) -> MemorySnapshot:
    proc_kib = _read_proc_rss_kib(os.getpid()) or 0.0
    user_kib = _user_rss_kib(user)
    avail_kib, swap_used_kib = _system_mem_available_kib()
    return MemorySnapshot(
        process_rss_mib=proc_kib / 1024.0,
        user_rss_mib=user_kib / 1024.0,
        mem_available_gib=(avail_kib / 1024.0 / 1024.0) if avail_kib is not None else None,
        swap_used_mib=(swap_used_kib / 1024.0) if swap_used_kib is not None else None,
        gpu_lines=_gpu_memory_lines(),
    )


def format_memory_snapshot(snapshot: MemorySnapshot) -> str:
    parts = [
        f"proc={snapshot.process_rss_mib:.0f}MiB",
        f"user={snapshot.user_rss_mib / 1024.0:.2f}GiB",
    ]
    if snapshot.mem_available_gib is not None:
        parts.append(f"avail={snapshot.mem_available_gib:.1f}GiB")
    if snapshot.swap_used_mib is not None:
        parts.append(f"swap={snapshot.swap_used_mib:.0f}MiB")
    if snapshot.gpu_lines:
        parts.append(" ".join(snapshot.gpu_lines))
    return ", ".join(parts)


def log_memory_snapshot(label: str, *, user: Optional[str] = None) -> MemorySnapshot:
    snap = capture_memory_snapshot(user=user)
    if pipeline_memory_log_enabled():
        print(f"[mem] {label}: {format_memory_snapshot(snap)}", flush=True)
    return snap


def _spawn_worker_pids(*, orphans_only: bool) -> list[int]:
    user = os.environ.get("USER") or ""
    if not user:
        return []
    try:
        out = subprocess.check_output(
            ["ps", "-u", user, "-ww", "-o", "pid=", "ppid=", "comm=", "args="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    pids: list[int] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        pid_s, ppid_s, comm, args = parts
        if not comm.startswith("python"):
            continue
        if "multiprocessing-fork" not in args and "spawn_main" not in args:
            continue
        try:
            pid = int(pid_s)
            ppid = int(ppid_s)
        except ValueError:
            continue
        if orphans_only and ppid != 1:
            continue
        pids.append(pid)
    return pids


def kill_owned_orphan_spawn_workers(*, signal_name: str = "TERM") -> int:
    """
    현재 사용자의 고아(PPID=1) nnU-Net spawn 워커를 종료합니다.

    :returns: 종료 시도한 PID 개수
    """
    if not env_flag("DENTAL_PIPELINE_KILL_ORPHAN_SPAWN", default=True):
        return 0

    pids = _spawn_worker_pids(orphans_only=True)
    if not pids:
        return 0

    sig = getattr(signal, "SIGKILL", 9) if signal_name.upper() == "KILL" else getattr(signal, "SIGTERM", 15)
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass

    if sig == signal.SIGTERM:
        remaining = _spawn_worker_pids(orphans_only=True)
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    return len(pids)


def _cleanup_on_exit(signum: Optional[int] = None) -> None:
    if signum is not None:
        try:
            name = signal.Signals(signum).name
        except (ValueError, AttributeError):
            name = str(signum)
        print(f"\n[pipeline_runtime] 종료 신호({name}) — 고아 spawn 정리…", flush=True)
    n = kill_owned_orphan_spawn_workers()
    if n:
        print(f"[pipeline_runtime] 고아 spawn {n}개 종료", flush=True)
    release_accelerator_memory()


def install_batch_cleanup_handlers() -> None:
    """배치 시작 시 한 번 호출. Ctrl+C·정상 종료 시 고아 spawn 정리."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    _HOOKS_INSTALLED = True

    atexit.register(_cleanup_on_exit)

    def _handler(signum: int, _frame) -> None:
        _cleanup_on_exit(signum)
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


__all__ = [
    "MemorySnapshot",
    "capture_memory_snapshot",
    "env_flag",
    "format_memory_snapshot",
    "install_batch_cleanup_handlers",
    "kill_owned_orphan_spawn_workers",
    "log_memory_snapshot",
    "nnunet_sequential_inference_enabled",
    "pipeline_memory_log_enabled",
    "release_accelerator_memory",
]
