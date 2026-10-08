"""Global seeding of python, numpy and torch."""

from __future__ import annotations

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
