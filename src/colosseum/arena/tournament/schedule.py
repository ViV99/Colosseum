"""Tournament schedules (SP4 spec block 3). T5.1: the ``Block`` type and the seed domains; T5.2 adds the builders.

Seeds: ``stage_seed(seed, stage, index)`` puts the stage's domain code in the top two bits of a 32-bit seed and
a keyed bijection of ``index`` in the low 30 bits, so the domains A (screening), B (confirmation) and P (pulse)
never share a seed and no seed repeats within a domain (indices < 2**30). Every candidate gets the same seed
for the same (stage, index): common random numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colosseum.core.types import Lineup

STAGES = ("A", "B", "P")
_STAGE_CODE = {"A": 1, "B": 2, "P": 3}
_LOW_BITS = 30
_LOW_MASK = (1 << _LOW_BITS) - 1


@dataclass(frozen=True)
class Block:
    """One seed, one composition and every admissible rotation (T5.2): the unit of scheduling, resume and
    bootstrap. ``lineups[i]`` is rotation ``rotations[i]``; every lineup carries ``seed``, ``block`` (= ``id``)
    and ``uid`` (= ``id + "/r<rotation>"``)."""

    id: str
    stage: str
    layout: str
    candidate: str
    composition: tuple[str, ...]
    index: int
    seed: int
    lineups: tuple[Lineup, ...]
    rotations: tuple[int, ...]

    @property
    def stratum(self) -> str:
        """The composition as one key: field members joined by ``+`` (``-`` for the homogeneous team)."""
        return "+".join(self.composition) or "-"


def stage_seed(seed: int, stage: str, index: int) -> int:
    """The episode seed of block ``index`` of ``stage`` (module docstring: disjoint domains, no repeats)."""
    if stage not in _STAGE_CODE:
        raise ValueError(f"stage_seed: stage must be one of {STAGES}, got {stage!r}")
    if not 0 <= int(index) <= _LOW_MASK:
        raise ValueError(f"stage_seed: index must be in [0, 2**30), got {index}")
    code = _STAGE_CODE[stage]
    a, b, c = (int(x) for x in np.random.SeedSequence([int(seed) % 2**63, code]).generate_state(3, np.uint32))
    x = (int(index) * (a | 1) + b) & _LOW_MASK      # odd multiplier + offset: a bijection mod 2**30
    x ^= x >> 15                                     # xorshift: a bijection
    x = (x * (c | 1)) & _LOW_MASK
    x ^= x >> 13
    return (code << _LOW_BITS) | x
