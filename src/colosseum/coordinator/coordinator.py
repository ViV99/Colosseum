"""Coordinator: matchmaking, checkpoint storage and ratings of a training run.

Manages:
- the agent pool (trainable agents) and their roles;
- owner rotation over the trainable agents and one ``Lineup`` per env (``LineupMatchmaker``);
- checkpoint storage (learner checkpoint payloads -> ``CheckpointManager``; ``meta.json`` gets the
  agent's roles and their role signature);
- match results and per-layout ratings (``RatingBook``).
"""

from __future__ import annotations

import logging
import random
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path

from colosseum.coordinator.agent_pool import AgentPool
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.coordinator.matchmaker import LineupMatchmaker
from colosseum.coordinator.ratings import RatingBook
from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.roles import agent_role_spec, role_signature
from colosseum.core.types import Lineup, MatchResult
from colosseum.envs.game import GameSpec

logger = logging.getLogger(__name__)


class Coordinator:
    """Central coordinator of a single-machine training run."""

    def __init__(self, config: ColosseumConfig, spec: GameSpec, agent_roles: Mapping[str, Sequence[str]],
                 checkpoint_dir: str | Path) -> None:
        self._config = config
        self._spec = spec
        trainable = config.get_trainable_agent_ids()
        missing = [a for a in trainable if a not in agent_roles]
        if missing:
            raise ConfigError(f"Coordinator: no roles resolved for agents {missing}")
        self._agent_roles = {a: list(agent_roles[a]) for a in trainable}
        # One RNG for matchmaking and seat permutations: runs with the same seed get the same schedule.
        self._rng = random.Random(config.training.seed)
        self._agent_pool = AgentPool()
        for agent_id in trainable:
            self._agent_pool.register_trainable(agent_id)
        self._checkpoint_manager = CheckpointManager(base_dir=checkpoint_dir, pool_size=config.checkpoint.pool_size)
        self._ratings = RatingBook(spec, trainable)
        self._role_signatures = {a: role_signature(agent_role_spec(spec, roles))
                                 for a, roles in self._agent_roles.items()}
        self._match_results: deque[MatchResult] = deque(maxlen=10000)
        self._refresh_round = 0
        self._matchmaker = LineupMatchmaker(
            spec=spec, agent_roles=self._agent_roles, config=config.matchmaking,
            checkpoints=self._checkpoint_ids, win_rate=self._ratings.win_rate, rng=self._rng,
        )

    @property
    def agent_pool(self) -> AgentPool:
        return self._agent_pool

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        return self._checkpoint_manager

    @property
    def ratings(self) -> RatingBook:
        return self._ratings

    @property
    def matchmaker(self) -> LineupMatchmaker:
        return self._matchmaker

    @property
    def spec(self) -> GameSpec:
        return self._spec

    @property
    def agent_roles(self) -> dict[str, list[str]]:
        return {a: list(r) for a, r in self._agent_roles.items()}

    @property
    def refresh_round(self) -> int:
        return self._refresh_round

    def role_signature(self, agent_id: str) -> str:
        return self._role_signatures[agent_id]

    def _checkpoint_ids(self, agent_id: str) -> list[str]:
        return [c.checkpoint_id for c in self._checkpoint_manager.list_checkpoints(agent_id)]

    def next_round(self) -> None:
        """Advance the owner rotation. The launcher calls this once per match refresh."""
        self._refresh_round += 1

    def generate_lineups(self, num_envs: int, env_offset: int) -> list[Lineup]:
        """One lineup per env. Env ``e`` of this batch has global index ``g = env_offset + e``;
        its owner is ``agents[(g + refresh_round) % n_trainable]`` (SP1 rotation), so every
        trainable agent owns envs, and ownership rotates between refreshes."""
        agents = [a.agent_id for a in self._agent_pool.list_trainable()]
        if not agents:
            raise ValueError("Coordinator has no trainable agents")
        return [self._matchmaker.lineup_for(agents[(env_offset + e + self._refresh_round) % len(agents)])
                for e in range(num_envs)]

    def report_match_result(self, result: MatchResult) -> None:
        """Keep the result and update the ratings of its layout."""
        self._match_results.append(result)
        self._ratings.update(result)

    @property
    def match_results(self) -> list[MatchResult]:
        return list(self._match_results)

    def ratings_snapshot(self) -> dict:
        """``RatingBook.snapshot()``: JSON-serializable tables per layout."""
        return self._ratings.snapshot()

    def save_checkpoint_payload(self, payload: dict, meta_extra: dict | None = None) -> str:
        """Persist a learner checkpoint payload (see ``learner.make_checkpoint_payload``).

        ``meta.json`` gets ``final``, ``meta_extra``, and the agent's ``roles`` and
        ``role_signature``. The trainer state is kept only when ``checkpoint.save_optimizer``.
        """
        agent_id = payload["agent_id"]
        trainer_state = payload.get("trainer_state_bytes") if self._config.checkpoint.save_optimizer else None
        meta = {"final": bool(payload.get("final", False)), **(meta_extra or {}),
                "roles": list(self._agent_roles[agent_id]), "role_signature": self._role_signatures[agent_id]}
        return self._checkpoint_manager.save(
            agent_id=agent_id,
            policy_version=int(payload["policy_version"]),
            model_state=payload["model_state"],
            trainer_state=trainer_state,
            meta_extra=meta,
        )
