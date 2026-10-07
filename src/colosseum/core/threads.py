"""Torch thread limits for worker / learner processes (R2-04, R6-04)."""

from __future__ import annotations

import os

import torch


def configure_torch_threads(num_threads: int, interop_threads: int = 1) -> None:
    """Set torch intra-op threads and (if still possible) inter-op threads.

    ``set_num_interop_threads`` may be called only once per process and only
    before any inter-op parallel work; later calls raise RuntimeError, which is
    ignored here (e.g. when a worker loop is run in-process by a test).
    """
    torch.set_num_threads(max(1, int(num_threads)))
    try:
        torch.set_num_interop_threads(max(1, int(interop_threads)))
    except RuntimeError:
        pass


def resolve_learner_threads(
    torch_threads: int | None,
    device: str,
    num_workers: int,
    worker_threads: int,
    num_learners: int,
    cpu_count: int | None = None,
) -> int:
    """Thread count for a learner process (spec block 2).

    Explicit ``learner.torch_threads`` wins. Otherwise: 2 on CUDA, and on CPU
    ``max(1, (cpu_count - num_workers * worker_threads) // num_learners)``.
    """
    if torch_threads is not None:
        return max(1, int(torch_threads))
    if str(device).startswith("cuda"):
        return 2
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return max(1, (cpus - num_workers * worker_threads) // max(1, num_learners))
