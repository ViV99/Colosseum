"""Coordinator: central orchestrator for the training pipeline.

Manages:
- Agent pool (trainable, frozen, scripted agents)
- Matchmaking (self-play, PFSP)
- Checkpoint storage (learner checkpoint payloads -> CheckpointManager)
- Match result tracking
- Pairwise ELO, win-rate matrix and latest-vs-past win rate (from per-seat results)
"""

from __future__ import annotations

import logging
import random
from collections import deque
from pathlib import Path

from colosseum.coordinator.agent_pool import AgentPool
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.coordinator.matchmaker import BaseMatchmaker, PFSPMatchmaker, SelfPlayMatchmaker
from colosseum.coordinator.ratings import EloRating, PastWinRate, WinRateTracker, pairwise_score
from colosseum.core.config import ColosseumConfig, TrainingPhase
from colosseum.core.types import LATEST_NETWORK_ID, MatchConfig, MatchResult

logger = logging.getLogger(__name__)


class Coordinator:
    """Central coordinator for training pipeline."""

    def __init__(self, config: ColosseumConfig, checkpoint_dir: str | Path | None = None) -> None:
        self._config = config
        # One RNG for matchmaking and seat shuffling: runs with the same seed get the same schedule.
        self._rng = random.Random(config.training.seed)
        self._agent_pool = AgentPool()
        for agent_id in config.get_trainable_agent_ids():
            self._agent_pool.register_trainable(agent_id)
        self._checkpoint_manager = CheckpointManager(
            base_dir=checkpoint_dir if checkpoint_dir is not None else config.checkpoint.dir,
            pool_size=config.self_play.pool_size,
        )
        self._match_results: deque[MatchResult] = deque(maxlen=10000)
        self._elo = EloRating()
        self._win_rates = WinRateTracker()
        self._past = PastWinRate()
        self._refresh_round = 0
        self._matchmaker: BaseMatchmaker = self._build_matchmaker()

    @property
    def agent_pool(self) -> AgentPool:
        return self._agent_pool

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        return self._checkpoint_manager

    @property
    def elo(self) -> EloRating:
        return self._elo

    @property
    def win_rates(self) -> WinRateTracker:
        return self._win_rates

    @property
    def past_win_rate(self) -> PastWinRate:
        return self._past

    @property
    def refresh_round(self) -> int:
        return self._refresh_round

    def next_round(self) -> None:
        """Advance the owner rotation. The launcher calls this once per match refresh."""
        self._refresh_round += 1

    def generate_match_configs(self, num_envs: int, env_offset: int) -> list[MatchConfig]:
        """One match per env. Env ``e`` of this batch has global index ``g = env_offset + e``.

        Its owner is ``agents[(g + refresh_round) % n_trainable]``, so every trainable
        agent owns envs in every phase, and ownership rotates between refreshes.
        """
        agents = [a.agent_id for a in self._agent_pool.list_trainable()]
        if not agents:
            raise ValueError("Coordinator has no trainable agents")
        num_players = self._config.env.num_players
        shuffle = self._config.self_play.shuffle_seats
        configs: list[MatchConfig] = []
        for e in range(num_envs):
            owner = agents[(env_offset + e + self._refresh_round) % len(agents)]
            match = self._matchmaker.match_for(owner, num_players)
            if shuffle:
                self._rng.shuffle(match.player_slots)
            configs.append(match)
        return configs

    def _build_matchmaker(self) -> BaseMatchmaker:
        sp = self._config.self_play
        if self._config.training.phase == TrainingPhase.LEAGUE:
            return PFSPMatchmaker(
                agent_pool=self._agent_pool,
                checkpoint_manager=self._checkpoint_manager,
                win_rate_tracker=self._win_rates,
                self_play_ratio=sp.self_play_ratio,
                pfsp_exponent=sp.pfsp_exponent,
                latest_prob=sp.latest_prob,
                rng=self._rng,
            )
        return SelfPlayMatchmaker(
            checkpoint_manager=self._checkpoint_manager,
            latest_prob=sp.latest_prob,
            rng=self._rng,
        )

    def save_checkpoint_payload(self, payload: dict, meta_extra: dict | None = None) -> str:
        """Persist a learner checkpoint payload (see ``learner.make_checkpoint_payload``).

        The trainer state is kept only when ``checkpoint.save_optimizer`` is true.
        """
        trainer_state = None
        if self._config.checkpoint.save_optimizer:
            trainer_state = payload.get("trainer_state_bytes")
        meta = {"final": bool(payload.get("final", False)), **(meta_extra or {})}
        return self._checkpoint_manager.save(
            agent_id=payload["agent_id"],
            policy_version=int(payload["policy_version"]),
            model_state=payload["model_state"],
            trainer_state=trainer_state,
            meta_extra=meta,
        )

    def report_match_result(self, result: MatchResult) -> None:
        """Update ratings from one finished match.

        Every pair of seats is compared by ``outcome`` (higher 1, equal 0.5, lower 0).
        - Different base agents: update the win-rate matrix and ELO. ELO deltas are
          computed from the pre-match ratings with K scaled by 1/(N-1).
        - Same agent, one seat ``latest`` and the other a checkpoint: update
          ``wr_vs_past`` from the latest seat's point of view.
        - Two ``latest`` seats of one agent carry no signal and are skipped.
        """
        self._match_results.append(result)
        seats = result.seats
        n = len(seats)
        if n < 2:
            return
        cross: list[tuple[str, str, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = seats[i], seats[j]
                score = pairwise_score(a.outcome, b.outcome)
                if a.agent_id != b.agent_id:
                    cross.append((a.agent_id, b.agent_id, score))
                elif a.network_id == LATEST_NETWORK_ID and b.network_id != LATEST_NETWORK_ID:
                    self._past.record(a.agent_id, score)
                elif b.network_id == LATEST_NETWORK_ID and a.network_id != LATEST_NETWORK_ID:
                    self._past.record(b.agent_id, 1.0 - score)
        for a_id, b_id, score in cross:
            self._win_rates.record_pair(a_id, b_id, score)
        if cross:
            self._elo.update_pairs(cross, k_scale=1.0 / (n - 1))

    @property
    def match_results(self) -> list[MatchResult]:
        return self._match_results

    def ratings_snapshot(self) -> dict:
        """JSON-serializable ratings of all trainable agents (persisted by the metrics hub)."""
        ids = [a.agent_id for a in self._agent_pool.list_trainable()]
        return {
            "elo": {a: self._elo.get(a) for a in ids},
            "win_rates": self._win_rates.get_win_rate_matrix(ids),
            "games": self._win_rates.get_games_matrix(ids),
            "wr_vs_past": {a: self._past.get(a) for a in ids},
            "past_games": {a: self._past.games(a) for a in ids},
        }
