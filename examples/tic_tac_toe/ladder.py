"""Tic-tac-toe ladder bots of known strength (SP4 criterion 3.3): ``TicTacToeBot`` that plays a uniformly
random legal cell with probability ``epsilon``. A larger epsilon is weaker against any fixed field
(measured in docs/benchmarks.md, «Лесенка ботов»), so a tournament must rank them by epsilon."""
from __future__ import annotations

from typing import Any

import numpy as np

from examples.tic_tac_toe.bots import TicTacToeBot


class EpsilonTicTacToeBot(TicTacToeBot):
    def __init__(self, epsilon: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = float(epsilon)

    def act(self, obs: Any, mask: Any, info: Any) -> int:
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            legal = (np.asarray(mask, dtype=bool).reshape(9) if mask is not None
                     else np.asarray(obs[2]).reshape(9) > 0.5)
            return int(self._rng.choice(np.flatnonzero(legal)))
        return super().act(obs, mask, info)
