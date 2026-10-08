"""Global seeding of python, numpy and torch, and per-process seed streams."""

from __future__ import annotations

import hashlib
import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def apply_global_seed(seed: int | None) -> None:
    """Seed python, numpy and torch in this process. ``None`` leaves the RNGs alone."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Global seed set to {seed}")


def derive_seed(seed: int, role: str, index: int) -> int:
    """Deterministic 32-bit seed for process ``index`` of ``role``, derived from ``seed``.

    A hash of ``(seed, role, index)``: distinct roles and indices get independent
    streams, unlike additive offsets (workers use ``seed + worker_id * 1000``).
    """
    digest = hashlib.sha256(f"colosseum:{role}:{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def learner_seed(seed: int | None, agent_index: int) -> int | None:
    """Seed of the learner of the ``agent_index``-th trainable agent (None if unseeded)."""
    return None if seed is None else derive_seed(seed, "learner", agent_index)
