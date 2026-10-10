"""Opponent selection (spec block 5): matchmaker interface, built-in mixture matchmaker, PFSP
statistics and share schedules.

Exports are resolved lazily (PEP 562): ``colosseum.core.config`` imports
``colosseum.league.schedule``, and the matchmaker modules import ``colosseum.core.config``.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "BaseMatchmaker": "colosseum.league.base",
    "MatchmakerContext": "colosseum.league.base",
    "MixtureMatchmaker": "colosseum.league.mixture",
    "PfspStats": "colosseum.league.pfsp",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'colosseum.league' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
