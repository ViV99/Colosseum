"""A scripted tic-tac-toe player (SP3 spec block 2): complete an own line if possible, else block the
opponent's, else take the center, else a random legal cell (from the episode's ``rng``)."""
from __future__ import annotations

from typing import Any

import numpy as np

from colosseum.players import ScriptedBot
from examples.tic_tac_toe.game import WINNING_LINES


class TicTacToeBot(ScriptedBot):
    def __init__(self) -> None:
        super().__init__()
        self._rng = np.random.default_rng()

    def reset(self, *, role: str, seat: int, layout: str, rng: np.random.Generator) -> None:
        self._rng = rng

    def act(self, obs: Any, mask: Any, info: Any) -> int:
        own = np.asarray(obs[0]).reshape(9) > 0.5
        opp = np.asarray(obs[1]).reshape(9) > 0.5
        legal = (np.asarray(mask, dtype=bool).reshape(9) if mask is not None
                 else np.asarray(obs[2]).reshape(9) > 0.5)
        for marks in (own, opp):        # win first, then block
            for line in WINNING_LINES:
                if sum(bool(marks[c]) for c in line) == 2:
                    free = [c for c in line if legal[c]]
                    if free:
                        return int(free[0])
        if legal[4]:
            return 4
        return int(self._rng.choice(np.flatnonzero(legal)))
