"""Coordinator: central orchestrator for the training pipeline.

Manages:
- Agent pool (trainable, frozen, scripted agents)
- Matchmaking (self-play, PFSP)
- Checkpoint scheduling
- Match result tracking
- ELO ratings and win rate tracking
"""

from __future__ import annotations

import logging
from collections import deque

from colosseum.coordinator.agent_pool import AgentPool
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.coordinator.matchmaker import (
    BaseMatchmaker,
    PFSPMatchmaker,
    SelfPlayMatchmaker,
    SimpleSelfPlayMatchmaker,
)
from colosseum.coordinator.ratings import EloRating, WinRateTracker
from colosseum.core.config import ColosseumConfig, TrainingPhase
from colosseum.core.types import MatchConfig, MatchResult

logger = logging.getLogger(__name__)


class Coordinator:
    """Central coordinator for training pipeline."""

    def __init__(self, config: ColosseumConfig) -> None:
        self._config = config
        self._agent_pool = AgentPool()
        self._checkpoint_manager = CheckpointManager(
            base_dir=config.checkpoint.dir,
            pool_size=config.self_play.pool_size,
            save_optimizer=config.checkpoint.save_optimizer,
        )
        self._matchmaker: BaseMatchmaker | None = None
        self._match_results: deque[MatchResult] = deque(maxlen=10000)
        self._elo = EloRating()
        self._win_rates = WinRateTracker()

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

    def setup_matchmaker(self, agent_id: str) -> None:
        """Set up the matchmaker based on training phase and available resources."""
        phase = self._config.training.phase

        if phase == TrainingPhase.LEAGUE:
            self._matchmaker = PFSPMatchmaker(
                agent_pool=self._agent_pool,
                checkpoint_manager=self._checkpoint_manager,
                win_rate_tracker=self._win_rates,
                self_play_ratio=self._config.self_play.self_play_ratio,
                pfsp_exponent=self._config.self_play.pfsp_exponent,
                latest_prob=self._config.self_play.latest_prob,
            )
            trainable = self._agent_pool.list_trainable()
            logger.info(
                f"Using PFSPMatchmaker with {len(trainable)} trainable agents"
            )
        else:
            # Self-play phase
            checkpoints = self._checkpoint_manager.list_checkpoints(agent_id)
            if checkpoints:
                self._matchmaker = SelfPlayMatchmaker(
                    checkpoint_manager=self._checkpoint_manager,
                    latest_prob=self._config.self_play.latest_prob,
                )
                logger.info(f"Using SelfPlayMatchmaker with {len(checkpoints)} checkpoints")
            else:
                self._matchmaker = SimpleSelfPlayMatchmaker()
                logger.info("Using SimpleSelfPlayMatchmaker (no checkpoints yet)")

    def generate_match_configs(
        self,
        agent_id: str,
        num_envs: int,
    ) -> list[MatchConfig]:
        """Generate match configs using the current matchmaker."""
        num_players = self._config.env.num_players
        if self._matchmaker is None:
            self.setup_matchmaker(agent_id)
        return self._matchmaker.generate_matches(agent_id, num_envs, num_players)

    def maybe_save_checkpoint(
        self,
        agent_id: str,
        policy_version: int,
        state_dict: dict,
        optimizer_state: dict | None = None,
        metrics: dict | None = None,
    ) -> str | None:
        """Save checkpoint if policy_version is at a checkpoint interval."""
        interval = self._config.self_play.checkpoint_interval
        if interval > 0 and policy_version > 0 and policy_version % interval == 0:
            ckpt_id = self._checkpoint_manager.save(
                agent_id=agent_id,
                policy_version=policy_version,
                state_dict=state_dict,
                optimizer_state=optimizer_state,
                metrics=metrics,
            )
            # Update matchmaker to use new checkpoints
            self.setup_matchmaker(agent_id)
            return ckpt_id
        return None

    @staticmethod
    def _seat_outcomes_by_agent(result: MatchResult) -> dict[str, list[float]]:
        """Every seat's outcome, grouped by base agent id (no seat is dropped).

        One agent may hold several seats (self-play latest vs latest, an N-player
        arena that repeats agents); each seat contributes its own outcome.
        """
        by_agent: dict[str, list[float]] = {}
        for seat in result.seats:
            by_agent.setdefault(seat.agent_id, []).append(float(seat.outcome))
        return by_agent

    def report_match_result(self, result: MatchResult) -> None:
        """Record a per-seat result and update ratings.

        Seat outcomes are averaged per base agent (the unit PFSP selects over),
        then every pair of DIFFERENT agents updates the win-rate matrix and ELO.
        Same-agent pairs (latest vs its own checkpoint, or two seats of one
        agent) carry no cross-agent signal and are skipped. Pairwise seat
        ratings replace this in T5.2.
        """
        self._match_results.append(result)

        by_agent = self._seat_outcomes_by_agent(result)
        agg = {a: sum(v) / len(v) for a, v in by_agent.items()}

        agents = list(agg.keys())
        for i, a in enumerate(agents):
            for j, b in enumerate(agents):
                if i >= j:
                    continue  # each unordered pair once; skips same-agent
                outcome_a = agg[a]
                outcome_b = agg[b]

                self._win_rates.record(a, b, outcome_a)

                if outcome_a > outcome_b:
                    self._elo.update(a, b, draw=False)
                elif outcome_b > outcome_a:
                    self._elo.update(b, a, draw=False)
                else:
                    self._elo.update(a, b, draw=True)

    @property
    def match_results(self) -> list[MatchResult]:
        return self._match_results

    def get_ratings_summary(self) -> dict:
        """Get a summary of all agent ratings."""
        trainable = self._agent_pool.list_trainable()
        agent_ids = [a.agent_id for a in trainable]
        return {
            "elo": {aid: self._elo.get(aid) for aid in agent_ids},
            "win_rates": self._win_rates.get_win_rate_matrix(agent_ids),
        }
