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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from colosseum.arena.retention import SnapshotScore, snapshot_scores
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.coordinator.ratings import RatingBook
from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.roles import agent_role_spec, role_signature
from colosseum.core.types import Lineup, MatchResult
from colosseum.envs.game import GameSpec
from colosseum.league.base import BaseMatchmaker, MatchmakerContext, build_matchmaker, load_matchmaker_class
from colosseum.league.lineups import check_lineup, enabled_layouts
from colosseum.league.mixture import MixtureMatchmaker, validate_matchmaking
from colosseum.league.pfsp import PfspStats, compact_ratings

logger = logging.getLogger(__name__)

# The league state of a run (SP4 spec block 7.4): ratings and PFSP statistics, restored by a run-dir resume.
LEAGUE_STATE_FILE = "league_state.json"
LEAGUE_STATE_VERSION = 1


@dataclass(frozen=True)
class LeagueStateInfo:
    """What a loaded ``league_state.json`` says about the journal (SP5 R8)."""

    env_steps: int                  # the global env-step count it was written at
    journal_row: int | None         # the highest journal row id accounted in it (None: none yet, or an SP4 file)
    has_watermark: bool             # False: written before SP5 (no "journal_row" key)


def league_replay_window(info: LeagueStateInfo | None) -> tuple[int | None, int | None]:
    """``(after_row, min_env_steps)`` of the journal rows an in-place resume replays (SP5 T4.3, R8): no league state
    -> every detailed training match; an SP5 state -> the rows above its watermark; an SP4 state (no watermark: SP4
    accounted at intake, before the journal) -> the matches stamped after the env-step count it was written at
    (never one twice; a match stamped exactly at that count is not replayed)."""
    if info is None:
        return None, None
    if info.has_watermark:
        return info.journal_row, None
    return None, info.env_steps


class Coordinator:
    """Central coordinator of a single-machine training run."""

    def __init__(self, config: ColosseumConfig, spec: GameSpec, player_roles: Mapping[str, Sequence[str]],
                 checkpoint_dir: str | Path, env_steps: Callable[[], int] = lambda: 0,
                 run_id: str | None = None) -> None:
        """``player_roles``: roles of EVERY agent (``players.registry.resolve_player_roles``); ``env_steps``:
        the run's global env-step counter (share schedules, SP3 block 5); ``run_id``: the current run (SP4: only
        its final snapshot is protected; None keeps the SP3 rule)."""
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
        self._evict_listeners: list[Callable[[str, str], None]] = []   # e.g. the journal writer (SP4)
        self._evict_listener_failed = False   # the first failing listener call is logged with its traceback
        # The hub's periodic Announce drains evictions (take_evictions) so nodes can unload them. With
        # match_refresh_interval_sec 0 every cell's lineup is granted once for the run (TTL 0): a worker never stops
        # playing a snapshot its lineup names, so there is nothing for a node to unload and nothing is buffered (the
        # hub holds no evicted snapshot either: SnapshotHolds' bound is 0); a node's cache drops a snapshot as soon as
        # none of its lineups names it (node.lineups.prune_cache). SP5 T3.1 (carried item 4b).
        self._buffer_evictions = config.rollout.match_refresh_interval_sec > 0
        self._pfsp = PfspStats({aid: config.get_agent_config(aid).matchmaking.pfsp.halflife_games
                                for aid in trainable})
        self._ratings_source: Callable[[], object | None] | None = None   # top-k retention (SP4 T4.1)
        self._checkpoint_manager = CheckpointManager(
            base_dir=checkpoint_dir, keep_last=ckpt.keep_last, keep_every=ckpt.keep_every, interval=ckpt.interval,
            on_evict=self._on_evict, run_id=run_id, keep_top_k=ckpt.keep_top_k,
            top_k_min_games=ckpt.top_k_min_games, max_snapshots=ckpt.max_snapshots, scores_fn=self.snapshot_scores,
        )
        self._ratings = RatingBook(spec, players)   # every player is a rating entity
        self._role_signatures = {a: role_signature(agent_role_spec(spec, roles))
                                 for a, roles in self._agent_roles.items()}
        self._match_results: deque[MatchResult] = deque(maxlen=10000)
        self._refresh_round = 0
        self._seatable: Callable[[str], bool] = lambda agent_id: True     # R41: set_seatable (the hub)
        self._league_state: LeagueStateInfo | None = None                 # what load_league_state read (R8)
        validate_matchmaking(spec, self._player_roles, config)
        self._context = MatchmakerContext.from_config(
            config, spec, self._player_roles, rng=self._rng, snapshots_fn=self._checkpoint_ids,
            pfsp_fn=self._pfsp.score, env_steps_fn=env_steps, seatable_fn=lambda agent_id: self._seatable(agent_id),
        )
        path = config.matchmaking.matchmaker_class
        matchmaker_cls = MixtureMatchmaker if path is None else load_matchmaker_class(path)
        self._matchmaker: BaseMatchmaker = build_matchmaker(matchmaker_cls, self._context, path)

    def set_seatable(self, seatable: Callable[[str], bool]) -> None:
        """The matchmaker's runtime availability of trainable agents (``MatchmakerContext.seatable``; SP5 R41): the
        hub's ``_playable``, asked at every draw. Default: every agent."""
        self._seatable = seatable

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

    def add_evict_listener(self, listener: Callable[[str, str], None]) -> None:
        """Also call ``listener(agent_id, checkpoint_id)`` for every evicted snapshot (main loop; SP4: the journal
        writer folds the snapshot's matches). A raising listener is logged (ERROR) and never stops the eviction or
        the run."""
        self._evict_listeners.append(listener)

    def set_ratings_source(self, fn: Callable[[], object | None]) -> None:
        """Where top-k retention reads ratings from: a callable returning the latest immutable
        ``RatingsSnapshot`` (``LiveRatings.latest``) or None. Main loop only."""
        self._ratings_source = fn

    def snapshot_scores(self, agent_id: str) -> dict[str, SnapshotScore] | None:
        """Live-rating scores of ``agent_id``'s snapshots (``arena.retention.snapshot_scores``): layout weights from
        the agent's ``matchmaking.layouts``, quantile ``checkpoint.top_k_quantile``. None while there are no
        ratings yet (no source, as during the run-dir import, or no refit yet): retention then keeps every unrated
        snapshot as new and reruns after the first refit (``Launcher._reapply_retention``)."""
        snapshot = self._ratings_source() if self._ratings_source is not None else None
        if snapshot is None:
            return None
        weights = enabled_layouts(self._spec, self._config.get_agent_config(agent_id).matchmaking)
        return snapshot_scores(snapshot, agent_id, layout_weights=weights,
                               quantile=self._config.checkpoint.top_k_quantile)

    def _on_evict(self, agent_id: str, checkpoint_id: str) -> None:
        """Storage evicted a snapshot (spec block 4): new lineups no longer draw it (the matchmaker's
        candidates come from the store), its PFSP statistics are dropped, the evict listeners are told, and
        nodes are told to unload it (``take_evictions``, through the hub's ``Announce``; only with
        ``rollout.match_refresh_interval_sec > 0``: without a match refresh every lineup is granted once for the run,
        a snapshot it names is played to the end, and nothing would use the eviction)."""
        logger.debug(f"Snapshot {checkpoint_id} of {agent_id} evicted")
        self._pfsp.forget((agent_id, checkpoint_id))
        for listener in self._evict_listeners:
            try:
                listener(agent_id, checkpoint_id)
            except Exception:  # noqa: BLE001 - a listener (the journal) must never stop training
                logger.error(f"Evict listener {listener!r} failed for snapshot {checkpoint_id} of {agent_id}; "
                             f"training goes on", exc_info=not self._evict_listener_failed)
                self._evict_listener_failed = True
        if not self._buffer_evictions:
            return
        bucket = self._evicted.setdefault(agent_id, [])
        if checkpoint_id not in bucket:
            bucket.append(checkpoint_id)

    def take_evictions(self) -> dict[str, list[str]]:
        """Snapshots evicted since the previous call, per agent in eviction order (each id once)."""
        evicted, self._evicted = self._evicted, {}
        return evicted

    def import_snapshots(self, run_dir: str | Path) -> None:
        """Run-dir resume (spec block 4): carry the stored snapshots of every trainable agent of ``run_dir``
        into this run's store; a snapshot with another role signature is a ConfigError. Agents without
        snapshots in ``run_dir`` (e.g. added since) are skipped silently."""
        for aid in self._trainable:
            if not (Path(run_dir) / "checkpoints" / aid).is_dir():
                continue
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
        return [self.lineup_for(agents[(env_offset + e + self._refresh_round) % len(agents)])
                for e in range(num_envs)]

    def lineup_for(self, owner: str) -> Lineup:
        """The matchmaker's lineup for ``owner`` (SP5: the hub asks per cell, its rotation picks the owner); it
        passes ``check_lineup`` (ValueError naming the matchmaker class)."""
        lineup = self._matchmaker.lineup_for(owner)
        check_lineup(self._context, lineup, type(self._matchmaker).__name__)
        return lineup

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

    def ratings_snapshot(self, *, compact: bool = False) -> dict:
        """``RatingBook.snapshot()`` with each layout's PFSP table under ``"pfsp"`` (JSON-serializable); with
        ``compact`` the table is replaced by its ``pfsp_summary`` (the form of the metrics.jsonl records)."""
        snap = self._ratings.snapshot()
        pfsp = self._pfsp.snapshot()
        for layout, table in snap.items():
            table["pfsp"] = pfsp.get(layout, {})
        return compact_ratings(snap) if compact else snap

    def save_league_state(self, path: str | Path, *, env_steps: int, journal_row: int | None = None) -> None:
        """Write the ratings (``RatingBook``) and the PFSP statistics (with the forgotten snapshots) atomically to
        ``path`` (spec block 7.4). A custom matchmaker's own state (``on_result``) is not saved. ``journal_row``
        (SP5, R8): the highest journal row id whose result is accounted in this state (every row up to it is; None:
        no journal, or nothing journaled yet); ``load_league_state`` keeps it (``loaded_league_state``), an in-place
        resume replays the rows after it."""
        from colosseum.metrics.jsonl import write_json_atomic

        write_json_atomic(path, {"version": LEAGUE_STATE_VERSION, "env_steps": int(env_steps),
                                 "ratings": self._ratings.state_dict(), "pfsp": self._pfsp.state_dict(),
                                 "journal_row": None if journal_row is None else int(journal_row)})

    def load_league_state(self, path: str | Path) -> bool:
        """Restore a saved league state; False when ``path`` does not exist (an SP3 run dir, or a run that never saved
        it). Entries of agents that are not in this config are dropped (INFO). An unreadable file is a ConfigError.
        What the file says about the journal is kept for the replay of an in-place resume (``loaded_league_state``,
        SP5 R8)."""
        import json

        path = Path(path)
        if not path.is_file():
            return False
        try:
            state = json.loads(path.read_text())
            if state.get("version") != LEAGUE_STATE_VERSION:
                raise ValueError(f"version {state.get('version')!r}, expected {LEAGUE_STATE_VERSION}")
            players = list(self._player_roles)
            self._ratings.load_state_dict(state["ratings"], agents=players)
            self._pfsp.load_state_dict(state["pfsp"], agents=players)
            row = state.get("journal_row")
            info = LeagueStateInfo(env_steps=int(state.get("env_steps") or 0),
                                   journal_row=None if row is None else int(row), has_watermark="journal_row" in state)
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as e:
            raise ConfigError(f"Cannot read the league state {path} ({type(e).__name__}: {e}); move it away to "
                              f"resume with empty ratings and PFSP statistics") from e
        self._league_state = info
        return True

    @property
    def loaded_league_state(self) -> LeagueStateInfo | None:
        """What ``load_league_state`` read about the journal (None: no file was loaded)."""
        return self._league_state

    def league_state_journal_row(self) -> int | None:
        """The journal watermark of the loaded league state (None: none loaded, an SP4 file, or nothing accounted)."""
        return None if self._league_state is None else self._league_state.journal_row

    def replay_results(self, results: Iterable[MatchResult]) -> tuple[int, int]:
        """In-place resume (SP5 T4.3, R8): account journaled results the loaded league state does not hold yet, in
        commit order, exactly as ``report_match_result``. A result the current game cannot rate (a layout it no
        longer has, another outcome kind: the config changed with ``-c``) is skipped and counted (one WARNING).
        Returns ``(replayed, skipped)``."""
        replayed = skipped = 0
        for result in results:
            layouts = self._spec.layouts
            if result.layout not in layouts or result.outcome_kind != self._spec.outcome_kind(result.layout):
                if not skipped:
                    logger.warning(f"Resume: journaled match {result.uid} (layout {result.layout!r}, "
                                   f"{result.outcome_kind}) cannot be replayed into the ratings of this game; such "
                                   f"matches are skipped")
                skipped += 1
                continue
            self.report_match_result(result)
            replayed += 1
        return replayed, skipped

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
