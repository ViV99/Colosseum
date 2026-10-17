"""A scripted Tron player (SP4 ladder, criterion 3.3): among straight / left / right it takes the move whose
cell is free and leads to the largest free area of its view window (flood fill; other heads count as
blocked), ties broken by the episode ``rng``; with probability ``epsilon`` it plays a uniformly random
action instead (``epsilon = 1``: a random player that often crashes)."""
from __future__ import annotations

from typing import Any

import numpy as np

from colosseum.players import ScriptedBot

# Moves in the rotated window (heading up, row 0): straight, turn left, turn right (TronGame's action order).
MOVES = ((-1, 0), (0, -1), (0, 1))


def free_area(blocked: np.ndarray, start: tuple[int, int]) -> int:
    """Cells reachable from ``start`` through free cells of ``blocked`` (4-neighbourhood); -1 if blocked."""
    h, w = blocked.shape
    if blocked[start]:
        return -1
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt not in seen and not blocked[nxt]:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen)


class FloodFillTronBot(ScriptedBot):
    def __init__(self, epsilon: float = 0.0) -> None:
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = float(epsilon)
        self._rng = np.random.default_rng()

    def reset(self, *, role: str, seat: int, layout: str, rng: np.random.Generator) -> None:
        self._rng = rng

    def act(self, obs: Any, mask: Any, info: Any) -> int:
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            return int(self._rng.integers(len(MOVES)))
        window = np.asarray(obs)
        blocked = (window[0] > 0) | (window[1] > 0)
        r = window.shape[1] // 2
        areas = np.array([free_area(blocked, (r + dr, r + dc)) for dr, dc in MOVES])
        return int(self._rng.choice(np.flatnonzero(areas == areas.max())))
