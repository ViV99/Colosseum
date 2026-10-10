"""Coordinator: matchmaking, checkpoint storage and ratings of a training run.

Manages:
- the players (trainable, scripted, frozen agents) and their roles; owner rotation over the trainable agents
  in config order and one ``Lineup`` per env from the matchmaker (``colosseum.league``: the built-in
  ``MixtureMatchmaker`` or ``matchmaking.matchmaker_class``), each checked by ``check_lineup``; every
  finished match goes to ``matchmaker.on_result``;
- checkpoint storage (learner checkpoint payloads -> ``CheckpointManager``; ``meta.json`` gets the
  agent's roles and their role signature);
- match results, per-layout ratings of every player (``RatingBook``) and the PFSP statistics per player
  (``PfspStats``).
"""

from __future__ import annotations

import logging
import random
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.coordinator.ratings import RatingBook
from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.roles import agent_role_spec, role_signature
from colosseum.core.types import Lineup, MatchResult
from colosseum.envs.game import GameSpec
from colosseum.league.base import BaseMatchmaker, MatchmakerContext, load_matchmaker_class
from colosseum.league.lineups import check_lineup
from colosseum.league.mixture import MixtureMatchmaker, validate_matchmaking
from colosseum.league.pfsp import PfspStats

logger = logging.getLogger(__name__)


class Coordinator:
    """Central coordinator of a single-machine training run."""

    def __init__(self, config: ColosseumConfig, spec: GameSpec, player_roles: Mapping[str, Sequence[str]],
                 checkpoint_dir: str | Path, env_steps: Callable[[], int] = lambda: 0) -> None:
        """``player_roles``: roles of EVERY agent (``players.registry.resolve_player_roles``); ``env_steps``:
        the run's global env-step counter (share schedules, SP3 block 5)."""
        self._config = config
        self._spec = spec
        trainable = config.get_trainable_agent_ids()
        players = config.agent_ids()   # config order (implicit agent_0 first)
        missing = [a for a in players if a not in player_roles]
        if missing:
            raise ConfigError(f"Coordinator: no roles resolved for agents {missing}")
        self._trainable = list(trainable)
        self._player_roles = {a: list(player_roles[a]) for a in players}
        self._agent_roles = {a: list(player_roles[a]) for a in trainable}
        self._env_steps = env_steps
        # One RNG for matchmaking and seat permutations: runs with the same seed get the same schedule.
        self._rng = random.Random(config.training.seed)
        ckpt = config.checkpoint
        self._evicted: dict[str, list[str]] = {}   # evicted since the last take_evictions(), per agent
        self._pfsp = PfspStats({aid: config.get_agent_config(aid).matchmaking.pfsp.halflife_games
                                for aid in trainable})
        self._checkpoint_manager = CheckpointManager(
            base_dir=checkpoint_dir, keep_last=ckpt.keep_last, keep_every=ckpt.keep_every, interval=ckpt.interval,
            on_evict=self._on_evict,
        )
        self._ratings = RatingBook(spec, players)   # every player is a rating entity
        self._role_signatures = {a: role_signature(agent_role_spec(spec, roles))
                                 for a, roles in self._agent_roles.items()}
        self._match_results: deque[MatchResult] = deque(maxlen=10000)
        self._refresh_round = 0
        validate_matchmaking(spec, self._player_roles, config)
        self._context = MatchmakerContext.from_config(
            config, spec, self._player_roles, rng=self._rng, snapshots_fn=self._checkpoint_ids,
            pfsp_fn=self._pfsp.score, env_steps_fn=env_steps,
        )
        path = config.matchmaking.matchmaker_class
        matchmaker_cls = MixtureMatchmaker if path is None else load_matchmaker_class(path)
        try:
            self._matchmaker: BaseMatchmaker = matchmaker_cls(self._context)
        except ConfigError:
            raise
        except Exception as e:  # noqa: BLE001 - a user class failing in __init__ is a config problem
            raise ConfigError(f"matchmaking.matchmaker_class {path!r}: constructing it failed: "
                              f"{type(e).__name__}: {e}") from e

    @property
    def player_roles(self) -> dict[str, list[str]]:
        """Roles of every agent (trainable, scripted, frozen) in config order."""
        return {a: list(r) for a, r in self._player_roles.items()}

    @property
    def trainable_agents(self) -> list[str]:
        """Trainable agent ids in config order (the owner rotation order)."""
        return list(self._trainable)

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        return self._checkpoint_manager

    @property
    def ratings(self) -> RatingBook:
        return self._ratings

    @property
    def pfsp(self) -> PfspStats:
        """PFSP statistics per player (read by the matchmaker)."""
        return self._pfsp

    @property
    def matchmaker(self) -> BaseMatchmaker:
        return self._matchmaker

    @property
    def context(self) -> MatchmakerContext:
        """The matchmaker's read-only view of the run."""
        return self._context

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

    def _on_evict(self, agent_id: str, checkpoint_id: str) -> None:
        """Storage evicted a snapshot (spec block 4): new lineups no longer draw it (the matchmaker's
        candidates come from the store), its PFSP statistics are dropped, and workers are told to unload it
        (``take_evictions``)."""
        logger.debug(f"Snapshot {checkpoint_id} of {agent_id} evicted")
        self._pfsp.forget((agent_id, checkpoint_id))
        bucket = self._evicted.setdefault(agent_id, [])
        if checkpoint_id not in bucket:
            bucket.append(checkpoint_id)

    def take_evictions(self) -> dict[str, list[str]]:
        """Snapshots evicted since the previous call, per agent in eviction order (each id once)."""
        evicted, self._evicted = self._evicted, {}
        return evicted

    def import_snapshots(self, run_dir: str | Path) -> None:
        """Run-dir resume (spec block 4): carry the stored snapshots of every trainable agent of ``run_dir``
        into this run's store; a snapshot with another role signature is a ConfigError."""
        for aid in self._trainable:
            kept = self._checkpoint_manager.import_snapshots(Path(run_dir) / "checkpoints", aid,
                                                             expected_signature=self._role_signatures[aid])
            logger.info(f"Resume [{aid}]: snapshot pool carried over from {run_dir}: {kept}")

    def _checkpoint_ids(self, agent_id: str) -> list[str]:
        return [c.checkpoint_id for c in self._checkpoint_manager.list_checkpoints(agent_id)]

    def next_round(self) -> None:
        """Advance the owner rotation. The launcher calls this once per match refresh."""
        self._refresh_round += 1

    def generate_lineups(self, num_envs: int, env_offset: int) -> list[Lineup]:
        """One lineup per env. Env ``e`` of this batch has global index ``g = env_offset + e``;
        its owner is ``trainable[(g + refresh_round) % n_trainable]`` (SP1 rotation over the trainable
        agents in config order), so every trainable agent owns envs and ownership rotates between
        refreshes. Every lineup passes ``check_lineup`` (ValueError naming the matchmaker class)."""
        agents = self._context.trainable
        if not agents:
            raise ValueError("Coordinator has no trainable agents")
        who = type(self._matchmaker).__name__
        lineups = []
        for e in range(num_envs):
            lineup = self._matchmaker.lineup_for(agents[(env_offset + e + self._refresh_round) % len(agents)])
            check_lineup(self._context, lineup, who)
            lineups.append(lineup)
        return lineups

    def report_match_result(self, result: MatchResult) -> None:
        """Keep the result; update the ratings and the PFSP statistics of its layout, then
        ``matchmaker.on_result``."""
        self._match_results.append(result)
        self._ratings.update(result)
        self._pfsp.update(result)
        self._matchmaker.on_result(result)

    @property
    def match_results(self) -> list[MatchResult]:
        return list(self._match_results)

    def ratings_snapshot(self) -> dict:
        """``RatingBook.snapshot()`` with each layout's PFSP table under ``"pfsp"`` (JSON-serializable)."""
        snap = self._ratings.snapshot()
        pfsp = self._pfsp.snapshot()
        for layout, table in snap.items():
            table["pfsp"] = pfsp.get(layout, {})
        return snap

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
