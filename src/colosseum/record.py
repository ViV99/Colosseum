"""``colosseum record``: a player's decisions as behavioural-cloning data (spec block 7).

The player (a scripted or frozen agent of the config, or ``name=path`` for a checkpoint dir or a
``.pt``) plays on the eval engine (``play_lineups`` on ``MatchRunner``):
- without opponents it takes every seat of every layout it can fill, and every seat is recorded;
  a layout with a role the player does not play needs ``--against``;
- every opponent (``--against``) forms a pair with the player, scheduled like ``colosseum eval``
  (``schedule_lineups``: the orientation rotates per match, ``num_matches`` per pair and layout);
  only the player's seats are recorded.

Each (env, seat) collects its decisions in its own buffer; at the episode end the buffer goes,
whole, to the writer of the seat's role, so a seat-episode is contiguous in its file and its last
decision has ``dones = True``. Episodes the eval engine discards (extra episodes of envs left
without a scheduled lineup) are dropped. A writer produces ``<output>/<role>/part-NNNNN.pt`` in
the format of ``colosseum.bc.offline_bc`` (``observations``, ``actions``, ``action_masks`` when the
role has masks, ``dones``; the dtypes of the role's spaces) and ``<output>/record.json`` describes
the run, with the eval summary of the played matches.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.registry import env_spec, make_env
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.tree import Tree, tree_map
from colosseum.core.types import Lineup
from colosseum.envs.game import GameSpec, RoleSpec
from colosseum.eval import load_player, play_lineups, schedule_lineups, summarize
from colosseum.utils.fs import write_text_atomic
from colosseum.worker.buffers import put_row
from colosseum.worker.match_runner import ActRecord, EpisodeEnd

logger = logging.getLogger(__name__)

__all__ = ["DECISIONS_PER_FILE", "RECORD_FILE", "RECORD_FORMAT", "RecordObserver", "RoleWriter", "record"]

RECORD_FILE = "record.json"
RECORD_FORMAT = 1
DECISIONS_PER_FILE = 100_000      # a part file is written once this many decisions are pending

Decision = tuple[Tree, Tree, "Tree | None"]     # (observation, action, normalized mask) of one decision


class RoleWriter:
    """Collects whole seat-episodes of one role and writes them as BC part files."""

    def __init__(self, directory: Path, role: RoleSpec, decisions_per_file: int = DECISIONS_PER_FILE) -> None:
        self.directory = Path(directory)
        self._obs_spec = ObsSpec.from_space(role.observation_space)
        self._action_spec = ActionSpec.from_space(role.action_space)
        self._decisions_per_file = int(decisions_per_file)
        self._pending: list[list[Decision]] = []
        self._pending_decisions = 0
        self.files: list[str] = []
        self.seat_episodes = 0
        self.decisions = 0

    def add_episode(self, decisions: Sequence[Decision]) -> None:
        """One seat-episode, in order; empty episodes (a seat that never acted) are skipped."""
        if not decisions:
            return
        self._pending.append(list(decisions))
        self._pending_decisions += len(decisions)
        self.seat_episodes += 1
        self.decisions += len(decisions)
        if self._pending_decisions >= self._decisions_per_file:
            self.flush()

    def flush(self) -> None:
        """Write the pending seat-episodes as the next ``part-NNNNN.pt`` (atomic rename)."""
        n = self._pending_decisions
        if n == 0:
            return
        obs = self._obs_spec.allocate((n,))
        actions = self._action_spec.allocate_actions((n,))
        masks = self._action_spec.full_mask((n,)) if self._action_spec.has_masks else None
        dones = np.zeros(n, dtype=bool)
        i = 0
        for episode in self._pending:
            for o, a, m in episode:
                put_row(obs, i, o)
                put_row(actions, i, a)
                if masks is not None and m is not None:
                    put_row(masks, i, m)
                i += 1
            dones[i - 1] = True
        data: dict[str, Any] = {
            "observations": tree_map(torch.from_numpy, obs),
            "actions": tree_map(torch.from_numpy, actions),
            "dones": torch.from_numpy(dones),
        }
        if masks is not None:
            data["action_masks"] = tree_map(torch.from_numpy, masks)
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"part-{len(self.files):05d}.pt"
        tmp = self.directory / f".{name}.tmp"
        torch.save(data, tmp)
        os.replace(tmp, self.directory / name)
        self.files.append(name)
        self._pending.clear()
        self._pending_decisions = 0


class RecordObserver:
    """``MatchObserver`` for ``play_lineups``: buffers the decisions of ``player``'s seats per
    (env, seat) and hands every finished seat-episode to the writer of its role."""

    def __init__(self, writers: Mapping[str, RoleWriter], player: str) -> None:
        self._writers = writers
        self._player = player
        self._buffers: dict[tuple[int, int], list[Decision]] = {}

    def on_act(self, env: int, seat: int, record: ActRecord) -> None:
        if record.agent_id == self._player:
            self._buffers.setdefault((env, seat), []).append((record.obs, record.action, record.mask))

    def on_rewards(self, env: int, rewards: dict[int, float]) -> None:
        pass

    def on_terminated(self, env: int, seats: list[int]) -> None:
        pass

    def on_lineup_applied(self, env, old, new) -> None:
        pass

    def on_episode_end(self, env: int, end: EpisodeEnd) -> None:
        roles = {s.seat: s.role for s in end.result.seats}
        for key in sorted(k for k in self._buffers if k[0] == env):
            self._writers[roles[key[1]]].add_episode(self._buffers.pop(key))

    def on_episode_discarded(self, env: int) -> None:
        for key in [k for k in self._buffers if k[0] == env]:
            del self._buffers[key]


def _parse_player(text: str) -> tuple[str, str | None]:
    """``name`` (a scripted or frozen agent of the config) or ``name=path``."""
    name, sep, path = text.partition("=")
    if not name or (sep and not path):
        raise ConfigError(f"player {text!r}: expected an agent name of the config or name=path")
    return name, (path if sep else None)


def _schedule(spec: GameSpec, player: str, roles: Mapping[str, Sequence[str]], opponents: Sequence[str],
              layouts: Sequence[str] | None, num_matches: int) -> list[Lineup]:
    explicit = bool(layouts)
    chosen = list(dict.fromkeys(layouts)) if layouts else list(spec.layouts)
    unknown = sorted(set(chosen) - set(spec.layouts))
    if unknown:
        raise ConfigError(f"--layout {unknown}: not layouts of the game {sorted(spec.layouts)}")
    lineups: list[Lineup] = []
    skipped: list[str] = []
    for layout in chosen:
        if not opponents:
            missing = sorted({seat.role for seat in spec.layouts[layout]} - set(roles[player]))
            if missing:
                if explicit:
                    raise ConfigError(f"layout {layout!r}: player {player!r} does not play the roles {missing}; "
                                      f"add --against <player> for those seats")
                skipped.append(layout)
                continue
            lineups.extend(schedule_lineups(spec, layout, {player: roles[player]}, num_matches))
            continue
        found: list[Lineup] = []
        for opponent in opponents:
            pair = schedule_lineups(spec, layout, {player: roles[player], opponent: roles[opponent]}, num_matches)
            found.extend(lineup for lineup in pair if any(seat.agent_id == player for seat in lineup.seats))
        if not found and explicit:
            raise ConfigError(f"layout {layout!r}: {player!r} and {list(opponents)} cannot fill its seats together")
        lineups.extend(found)
    if skipped and lineups:
        logger.info("Layouts %s skipped: player %r does not play all of their roles (add --against to record them)",
                    skipped, player)
    if not lineups:
        hint = ("add --against <player> for the roles it does not play" if not opponents
                else "check the players' roles (agents.<id>.roles)")
        raise ConfigError(f"player {player!r} (roles {list(roles[player])}) cannot fill any layout of the game "
                          f"{sorted(spec.layouts)}: {hint}")
    return lineups


def record(config: ColosseumConfig, player: str, against: Sequence[str], *, layouts: Sequence[str] | None,
           num_matches: int, output: str | Path, num_envs: int = 8, seed: int | None = None,
           deterministic: bool = False, decisions_per_file: int = DECISIONS_PER_FILE) -> dict:
    """Record ``player``'s decisions into ``output`` (module docstring); returns the ``record.json`` content.

    With opponents and a layout of two or more teams among the chosen ones, an odd ``num_matches``
    is rounded up (like ``colosseum eval``), so the player plays every side equally often.
    """
    if num_matches < 1:
        raise ConfigError(f"num_matches must be >= 1, got {num_matches}")
    spec = env_spec(config)
    out = Path(output)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ConfigError(f"--output {out}: exists and is not empty; choose a new directory")
    p_name, p_path = _parse_player(player)
    opponents = [_parse_player(text) for text in against]
    names = [p_name] + [name for name, _ in opponents]
    reused = sorted({name for name in names if names.count(name) > 1})
    if reused:
        raise ConfigError(f"player names {reused} are used more than once; without --against the player takes "
                          f"every seat (a bot against itself)")
    models: dict[str, Any] = {}
    roles: dict[str, list[str]] = {}
    validated: set[str] = set()
    for option, (name, path) in [("--player", (p_name, p_path)), *(("--against", o) for o in opponents)]:
        models[name], roles[name] = load_player(config, name, path, spec=spec, validated=validated, option=option)
    chosen = list(layouts) if layouts else list(spec.layouts)
    if opponents and num_matches % 2 and any(spec.num_teams(name) >= 2 for name in chosen if name in spec.layouts):
        logger.info("num_matches %d is odd; rounded up, using %d per opponent and layout so the player plays "
                    "every side equally often", num_matches, num_matches + 1)
        num_matches += 1
    lineups = _schedule(spec, p_name, roles, [name for name, _ in opponents], layouts, num_matches)
    out.mkdir(parents=True, exist_ok=True)
    writers = {role: RoleWriter(out / role, spec.roles[role], decisions_per_file) for role in roles[p_name]}
    results = play_lineups(env_fn=lambda: make_env(config), models=models, lineups=lineups, num_envs=num_envs,
                           seed=seed, deterministic=deterministic, max_idle_steps=config.env.max_idle_steps,
                           observer=RecordObserver(writers, p_name))
    for writer in writers.values():
        writer.flush()
    report = summarize(spec, results, agents=names, num_matches=num_matches, deterministic=deterministic)
    played = {lineup.layout for lineup in lineups}
    content = {
        "format": RECORD_FORMAT,
        "game": config.env.env_class,
        "env_kwargs": config.env.kwargs,
        "player": p_name,
        "player_source": p_path,
        "against": [name for name, _ in opponents],
        "against_sources": {name: path for name, path in opponents},
        "layouts": [name for name in spec.layouts if name in played],
        "num_matches": num_matches,
        "matches": len(results),
        "seed": seed,
        "deterministic": deterministic,
        "seat_episodes": sum(w.seat_episodes for w in writers.values()),
        "decisions": sum(w.decisions for w in writers.values()),
        "roles": {role: {"seat_episodes": w.seat_episodes, "decisions": w.decisions, "files": list(w.files)}
                  for role, w in writers.items() if w.decisions},
        "summary": report.to_dict(),
    }
    write_text_atomic(out / RECORD_FILE, json.dumps(content, indent=2, default=str) + "\n")
    logger.info("Recorded %d decisions of %s (%d seat-episodes, %d matches) into %s", content["decisions"], p_name,
                content["seat_episodes"], content["matches"], out)
    return content
