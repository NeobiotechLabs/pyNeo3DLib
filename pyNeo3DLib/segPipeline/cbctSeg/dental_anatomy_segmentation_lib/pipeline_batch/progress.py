"""배치·케이스 진행 표시."""

from __future__ import annotations

import sys
from typing import Callable, Optional, Tuple


def cuda_device_count() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except ImportError:
        pass
    return 0


def try_tqdm_cls():
    try:
        from tqdm.auto import tqdm as tqdm_cls

        return tqdm_cls
    except ImportError:
        return None


def create_case_progress(
    total: int, desc: str
) -> tuple[Callable[[], None], Callable[[], None]]:
    tqdm_cls = try_tqdm_cls()
    if tqdm_cls is None:
        done = [0]

        def tick() -> None:
            done[0] += 1
            print(f"[{done[0]}/{total}] {desc}", flush=True)

        def close() -> None:
            pass

        return tick, close

    bar = tqdm_cls(total=total, desc=desc, unit="건", dynamic_ncols=True)

    def tick() -> None:
        bar.update(1)

    def close() -> None:
        bar.close()

    return tick, close


def prefixed_case_progress(case_label: str) -> Callable[[str], None]:
    def _emit(msg: str) -> None:
        line = f"[{case_label}] {msg}"
        tqdm_cls = try_tqdm_cls()
        if tqdm_cls is not None:
            tqdm_cls.write(line)
        else:
            print(line, file=sys.stderr, flush=True)

    return _emit
