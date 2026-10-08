"""Matchmaking: build the match for one owner agent.

- SelfPlayMatchmaker: the owner's latest weights against its latest weights and
  its own checkpoints.
- PFSPMatchmaker (league): with probability ``self_play_ratio`` a self-play
  match, otherwise an arena of the owner plus N-1 opponents drawn by PFSP
  (with replacement) from the other trainable agents.

The coordinator decides which agent owns each env and shuffles the seats.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod

from colosseum.coordinator.agent_pool import AgentPool
from colosseum.coordinator.ratings import WinRateTracker
from colosseum.core.types import MatchConfig, PlayerSlot

logger = logging.getLogger(__name__)


def _new_match(slots: list[PlayerSlot], rng: random.Random) -> MatchConfig:
    return MatchConfig(match_id=f"{rng.getrandbits(48):012x}", env_config={}, player_slots=slots)


class BaseMatchmaker(ABC):
    """Builds one match whose training data belongs to ``owner``."""

    @abstractmethod
    def match_for(self, owner: str, num_players: int) -> MatchConfig:
        """Return a match with ``num_players`` slots.

        Slot 0 is always the owner's latest weights with ``collect_trajectories=True``.
        The coordinator shuffles the seats afterwards.
        """


class SelfPlayMatchmaker(BaseMatchmaker):
    """Self-play against the owner's own history.

    Slot 0: the owner's latest weights (collect). Every other slot: the latest weights
    (collect) with probability ``latest_prob``, otherwise a uniformly random checkpoint
    of the owner (no collect). Without checkpoints every slot is the latest weights.
    """

    def __init__(self, checkpoint_manager, latest_prob: float = 0.5,
                 rng: random.Random | None = None) -> None:
        self._ckpt_mgr = checkpoint_manager
        self._latest_prob = latest_prob
        self._rng = rng or random.Random()

    def self_play_slots(self, owner: str, num_players: int) -> list[PlayerSlot]:
        checkpoints = self._ckpt_mgr.list_checkpoints(owner)
        slots = [PlayerSlot(agent_id=owner, checkpoint_id=None, collect_trajectories=True)]
        for _ in range(1, num_players):
            if not checkpoints or self._rng.random() < self._latest_prob:
                slots.append(PlayerSlot(agent_id=owner, checkpoint_id=None, collect_trajectories=True))
            else:
                ckpt = self._rng.choice(checkpoints)
                slots.append(PlayerSlot(agent_id=owner, checkpoint_id=ckpt.checkpoint_id,
                                        collect_trajectories=False))
        return slots

    def match_for(self, owner: str, num_players: int) -> MatchConfig:
        return _new_match(self.self_play_slots(owner, num_players), self._rng)


class PFSPMatchmaker(BaseMatchmaker):
    """League matchmaker: self-play, or an N-player arena chosen by PFSP.

    PFSP weight of candidate ``c`` for owner ``o``: ``max(1e-6, (1 - wr(o, c)) ** p)``.
    Arena opponents are drawn with replacement. Every arena slot plays the latest
    weights and collects trajectories.
    """

    def __init__(self, agent_pool: AgentPool, checkpoint_manager, win_rate_tracker: WinRateTracker,
                 self_play_ratio: float = 0.5, pfsp_exponent: float = 1.0,
                 latest_prob: float = 0.5, rng: random.Random | None = None) -> None:
        self._pool = agent_pool
        self._win_rates = win_rate_tracker
        self._self_play_ratio = self_play_ratio
        self._pfsp_exponent = pfsp_exponent
        self._rng = rng or random.Random()
        self._self_play = SelfPlayMatchmaker(checkpoint_manager, latest_prob, self._rng)

    def pfsp_weights(self, owner: str, candidates: list[str]) -> list[float]:
        return [
            max(1e-6, (1.0 - self._win_rates.get_win_rate(owner, c)) ** self._pfsp_exponent)
            for c in candidates
        ]

    def select_opponents(self, owner: str, candidates: list[str], k: int) -> list[str]:
        return self._rng.choices(candidates, weights=self.pfsp_weights(owner, candidates), k=k)

    def match_for(self, owner: str, num_players: int) -> MatchConfig:
        others = [a.agent_id for a in self._pool.list_trainable() if a.agent_id != owner]
        if num_players < 2 or not others or self._rng.random() < self._self_play_ratio:
            return _new_match(self._self_play.self_play_slots(owner, num_players), self._rng)
        opponents = self.select_opponents(owner, others, num_players - 1)
        slots = [PlayerSlot(agent_id=owner, checkpoint_id=None, collect_trajectories=True)]
        slots += [PlayerSlot(agent_id=o, checkpoint_id=None, collect_trajectories=True) for o in opponents]
        return _new_match(slots, self._rng)
