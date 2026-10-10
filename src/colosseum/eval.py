"""Evaluation: inference-only matches on ``MatchRunner`` (spec block 8).

Engine
------
``play_lineups`` plays explicit ``Lineup``s with explicit players on the same ``MatchRunner`` core as
training: seat lifecycle, masks, per-seat model states, results by team. ``models`` maps every
``agent_id`` of the lineups to a ``PolicyModel``, a ``ScriptedPlayer`` or a ``ScriptedBot`` instance (a
prototype: every (agent, env, seat) plays a deep copy with ``game_spec`` set); ``network_id`` is ignored
(every seat uses ``models[agent_id]``). Every lineup is played once, to completion. Envs left without
a scheduled lineup keep playing their last one until the rest finish; those extra episodes are
discarded, so short episodes are not favoured. The in-process API is what the learning tests and the
future ``colosseum tournament`` (SP4) use.

Schedules (CLI, ``schedule_lineups``)
-------------------------------------
- Two or more teams, two or more agents: for every pair (a, b) and match m, team i gets the core
  ``(a, b)[(i + m) % 2]``; a team whose roles the core does not all play goes as a whole to the
  other agent of the pair, so every team is homogeneous. An orientation that cannot fill a team
  with one agent, or that leaves one of the two out, is replaced by the other orientation (so an
  asymmetric pair, hunter and prey, is never rotated); a pair with no valid orientation is not
  scheduled for the layout.
- Two or more teams, one agent: every team is that agent.
- One team (solo, cooperative): each agent alone (homogeneous team), then, with two or more
  agents and a team of two or more seats, every mixed composition (cross-play), rotating its
  seat orders over the matches.

Statistics (``summarize``), per layout, by the layout's outcome kind
--------------------------------------------------------------------
- ``wdl``: per pair, W/D/L from team ranks, win rate and score (draw = half) with 95% Wilson
  intervals (SP1), mean returns and length, and a per-side breakdown keyed by the team index of
  agent a. The reversed row is derived from the same counts. One agent: per-seat W/D/L and mean
  return. A lineup with a team mixing both agents is counted as ``unattributed``; ``schedule_lineups``
  no longer produces such lineups, only explicit ``play_lineups`` lineups can.
- ``rank``: per agent, mean team rank with a 95% normal interval and the share of first places;
  a pairwise "who ranked higher" table (``higher[a][b]``: score of a over b, draw = half).
  Teams mixing agents (only possible in explicit ``play_lineups`` lineups; ``schedule_lineups``
  does not produce them) are left out (the match is counted as ``unattributed``). One agent in
  every seat (SP1 solo mode): ``solo`` per seat (mean rank, share of first places, mean return).
- ``score``: per team composition (sorted agent ids joined by ``+``), mean team score with a 95%
  normal interval; homogeneous compositions are the per-agent results, the rest is cross-play.
- every kind: mean seat return per role and agent (``by_role``).
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import itertools
import json
import logging
import math
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from colosseum.coordinator.ratings import composition_key
from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.outcomes import pairwise_rank_score
from colosseum.core.roles import agent_role_spec
from colosseum.core.types import LATEST_NETWORK_ID, Lineup, MatchResult, SeatAssignment
from colosseum.core.validation import _check_model
from colosseum.envs.game import GameSpec, MultiAgentEnv
from colosseum.envs.vector import VectorEnv
from colosseum.networks.model import PolicyModel
from colosseum.players.scripted import ScriptedBot
from colosseum.worker.match_runner import EpisodeEnd, MatchObserver, MatchRunner, ScriptedPlayer

logger = logging.getLogger(__name__)

Z_95 = 1.959963984540054
SCORE_CI_METHOD = "wilson-on-score (draw = half win; conservative approximation)"


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def _eval_seat(agent_id: str) -> SeatAssignment:
    return SeatAssignment(agent_id=agent_id, network_id=LATEST_NETWORK_ID, collect=False)


def _pair_lineup(spec: GameSpec, layout: str, pair: tuple[str, str], orientation: int,
                 players: Mapping[str, Sequence[str]]) -> Lineup | None:
    """Orientation 0 or 1 of a pair: team i goes as a whole to its core ``pair[(i + orientation) % 2]``,
    or to the other agent when the core does not play every role of the team. ``None`` when a team
    cannot be filled by one agent or the lineup leaves one of the two out."""
    seat_specs = spec.layouts[layout]
    seats: list[SeatAssignment | None] = [None] * len(seat_specs)
    for team, members in enumerate(spec.teams(layout)):
        core = pair[(team + orientation) % 2]
        roles = {seat_specs[s].role for s in members}
        name = next((n for n in (core, pair[1 - (team + orientation) % 2]) if roles <= set(players[n])), None)
        if name is None:
            return None
        for s in members:
            seats[s] = _eval_seat(name)
    if {seat.agent_id for seat in seats} != set(pair):  # type: ignore[union-attr]
        return None
    return Lineup(layout=layout, seats=seats)  # type: ignore[arg-type]


def _pair_lineups(spec: GameSpec, layout: str, pair: tuple[str, str], num_matches: int,
                  players: Mapping[str, Sequence[str]]) -> list[Lineup]:
    """Match m uses orientation ``m % 2``, or the other one when that is invalid; ``[]`` if both are."""
    orientations = [_pair_lineup(spec, layout, pair, o, players) for o in (0, 1)]
    if all(lineup is None for lineup in orientations):
        return []
    picked = [orientations[m % 2] or orientations[1 - m % 2] for m in range(num_matches)]
    return [Lineup(layout, list(lineup.seats)) for lineup in picked]  # type: ignore[union-attr]


def _homogeneous(spec: GameSpec, layout: str, name: str, roles: Sequence[str]) -> Lineup | None:
    if any(seat.role not in roles for seat in spec.layouts[layout]):
        return None
    return Lineup(layout=layout, seats=[_eval_seat(name) for _ in spec.layouts[layout]])


def _cross_play_orders(spec: GameSpec, layout: str, composition: tuple[str, ...],
                       players: Mapping[str, Sequence[str]]) -> list[tuple[str, ...]]:
    roles = [seat.role for seat in spec.layouts[layout]]
    orders = sorted({order for order in itertools.permutations(composition)
                     if all(role in players[name] for name, role in zip(order, roles, strict=True))})
    return orders


def schedule_lineups(spec: GameSpec, layout: str, players: Mapping[str, Sequence[str]],
                     num_matches: int) -> list[Lineup]:
    """Lineups of one layout for ``players`` (name -> roles) by the CLI rules (module docstring).

    ``num_matches`` is per pair (two or more teams), per agent (one agent, or one team) and per
    mixed composition (cross-play). Returns ``[]`` when the players cannot fill the layout.
    """
    if layout not in spec.layouts:
        raise ValueError(f"schedule_lineups: unknown layout {layout!r}")
    if num_matches < 1:
        raise ValueError(f"schedule_lineups: num_matches must be >= 1, got {num_matches}")
    names = list(players)
    if not names:
        raise ValueError("schedule_lineups: no players")
    lineups: list[Lineup] = []
    if spec.num_teams(layout) >= 2 and len(names) >= 2:
        for pair in itertools.combinations(names, 2):
            lineups.extend(_pair_lineups(spec, layout, pair, num_matches, players))
        return lineups
    for name in names:
        lineup = _homogeneous(spec, layout, name, players[name])
        if lineup is not None:
            lineups.extend(Lineup(layout, list(lineup.seats)) for _ in range(num_matches))
    size = spec.layout_size(layout)
    if spec.num_teams(layout) == 1 and len(names) >= 2 and size >= 2:
        for composition in itertools.combinations_with_replacement(names, size):
            if len(set(composition)) < 2:
                continue
            orders = _cross_play_orders(spec, layout, composition, players)
            for m in range(num_matches if orders else 0):
                lineups.append(Lineup(layout, [_eval_seat(name) for name in orders[m % len(orders)]]))
    return lineups


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class _FixedModels:
    """``PlayerPool`` that serves ``models[agent_id]`` (a model or a ``ScriptedPlayer``) for any network id."""

    def __init__(self, models: Mapping[str, PolicyModel | ScriptedPlayer]) -> None:
        self._models = models

    def get(self, agent_id: str, network_id: str) -> PolicyModel | ScriptedPlayer | None:
        return self._models.get(agent_id)


def _copy_bot(prototype: ScriptedBot, spec: GameSpec) -> ScriptedBot:
    bot = copy.deepcopy(prototype)
    bot.game_spec = spec
    return bot


class _Collector:
    """``MatchObserver`` that keeps the result of every scheduled lineup and feeds the next one.

    ``inner`` (optional) receives every callback; ``on_episode_end`` only for scheduled episodes,
    and for the extra episodes of envs left without a lineup the optional ``on_episode_discarded(env)``.
    """

    def __init__(self, pending: deque[Lineup], scheduled: list[bool], inner: MatchObserver | None = None) -> None:
        self.runner: MatchRunner | None = None
        self.results: list[MatchResult] = []
        self._pending = pending
        self._scheduled = scheduled
        self._inner = inner

    def on_act(self, env, seat, record) -> None:
        if self._inner is not None:
            self._inner.on_act(env, seat, record)

    def on_rewards(self, env, rewards) -> None:
        if self._inner is not None:
            self._inner.on_rewards(env, rewards)

    def on_terminated(self, env, seats) -> None:
        if self._inner is not None:
            self._inner.on_terminated(env, seats)

    def on_lineup_applied(self, env, old, new) -> None:
        if self._inner is not None:
            self._inner.on_lineup_applied(env, old, new)

    def on_episode_start(self, env: int, layout: str, episode_seed: int | None) -> None:
        hook = getattr(self._inner, "on_episode_start", None)
        if hook is not None:
            hook(env, layout, episode_seed)

    def on_episode_end(self, env: int, end: EpisodeEnd) -> None:
        if self._scheduled[env]:
            self.results.append(end.result)
            if self._inner is not None:
                self._inner.on_episode_end(env, end)
        else:
            discard = getattr(self._inner, "on_episode_discarded", None)
            if discard is not None:
                discard(env)
        if self._pending:
            self.runner.set_next_lineup(env, self._pending.popleft())  # applied at this episode end
            self._scheduled[env] = True
        else:
            self._scheduled[env] = False


def play_lineups(
    *,
    env_fn: Callable[[], MultiAgentEnv],
    models: Mapping[str, PolicyModel | ScriptedBot | ScriptedPlayer],
    lineups: Sequence[Lineup],
    num_envs: int = 8,
    seed: int | None = None,
    deterministic: bool = False,
    max_idle_steps: int = 1000,
    observer: MatchObserver | None = None,
) -> list[MatchResult]:
    """Play every lineup once, to completion; one ``MatchResult`` per lineup, in completion order.

    Evaluation never collects: every seat is played with ``collect=False`` (copies of the lineups; the
    caller's are not changed), so a default ``SeatAssignment(agent_id)`` works for a scripted seat too.
    ``seed`` seeds the episode resets and a forked torch RNG, so the caller's global RNG is
    untouched. Models run in eval mode; their train/eval flags are restored on return. Scripted players
    need no eval mode; a ``ScriptedBot`` instance is never played itself (deep copies are).
    ``observer`` (optional) sees every decision and episode of the scheduled lineups; see ``_Collector``.
    """
    lineups = [Lineup(lineup.layout, [dataclasses.replace(seat, collect=False) for seat in lineup.seats])
               for lineup in lineups]
    if not lineups:
        return []
    unknown = sorted({seat.agent_id for lineup in lineups for seat in lineup.seats} - set(models))
    if unknown:
        raise ValueError(f"play_lineups: lineups use unknown agents {unknown}")
    if num_envs < 1:
        raise ValueError(f"play_lineups: num_envs must be >= 1, got {num_envs}")
    neural = [m for m in models.values() if isinstance(m, PolicyModel)]
    rng = torch.random.fork_rng(devices=[]) if seed is not None else contextlib.nullcontext()
    was_training = [(m, m.training) for model in neural for m in model.modules()]
    try:
        with rng:
            if seed is not None:
                torch.manual_seed(seed)
            for model in neural:
                model.eval()
            n = min(num_envs, len(lineups))
            pending = deque(lineups[n:])
            collector = _Collector(pending, [True] * n, inner=observer)
            vec_env = VectorEnv(env_fn, n)
            try:
                pool = {name: ScriptedPlayer(functools.partial(_copy_bot, player, vec_env.spec))
                        if isinstance(player, ScriptedBot) else player for name, player in models.items()}
                runner = MatchRunner(vec_env=vec_env, lineups=lineups[:n], models=_FixedModels(pool),
                                     observer=collector, seed=seed, max_idle_steps=max_idle_steps,
                                     deterministic=deterministic, context="eval, ", match_id_prefix="eval")
            except BaseException:
                vec_env.close()
                raise
            collector.runner = runner
            try:
                while len(collector.results) < len(lineups):
                    runner.step()
            finally:
                runner.close()
            return collector.results
    finally:
        for module, training in was_training:
            module.training = training


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_interval(successes: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for ``successes / n`` (fractional successes allowed); n=0 -> (0, 1)."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(0.0, p * (1.0 - p)) / n + z * z / (4.0 * n * n)) / denom
    return min(max(0.0, center - half), p), max(min(1.0, center + half), p)


def normal_interval(values: Sequence[float], z: float = Z_95) -> tuple[float, float, float]:
    """``(mean, low, high)`` with ``mean +- z * sd / sqrt(n)`` (ddof=1); n < 2 -> zero width."""
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(x.mean())
    if x.size < 2:
        return mean, mean, mean
    half = z * float(x.std(ddof=1)) / math.sqrt(x.size)
    return mean, mean - half, mean + half


def _team_agents(result: MatchResult) -> dict[int, set[str]]:
    teams: dict[int, set[str]] = defaultdict(set)
    for seat in result.seats:
        teams[seat.team].add(seat.agent_id)
    return teams


def _team_return(result: MatchResult, team: int) -> float:
    return float(np.mean([seat.reward for seat in result.seats if seat.team == team]))


@dataclass
class PairStats:
    """WDL counts of one unordered pair, from ``agent_a``'s side; ``per_side`` is keyed by a's team."""

    agent_a: str
    agent_b: str
    wins: int = 0
    draws: int = 0
    losses: int = 0
    return_a: float = 0.0
    return_b: float = 0.0
    total_length: int = 0
    per_side: dict[int, list[int]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.wins + self.draws + self.losses

    def add(self, result: MatchResult, team_a: int, team_b: int) -> None:
        ranks = {team.team: team.rank for team in result.teams}
        score = pairwise_rank_score(ranks[team_a], ranks[team_b])
        cell = self.per_side.setdefault(team_a, [0, 0, 0])
        idx = 0 if score == 1.0 else (1 if score == 0.5 else 2)
        cell[idx] += 1
        if idx == 0:
            self.wins += 1
        elif idx == 1:
            self.draws += 1
        else:
            self.losses += 1
        self.return_a += _team_return(result, team_a)
        self.return_b += _team_return(result, team_b)
        self.total_length += result.episode_length

    def reversed(self) -> PairStats:
        return PairStats(
            agent_a=self.agent_b, agent_b=self.agent_a, wins=self.losses, draws=self.draws, losses=self.wins,
            return_a=self.return_b, return_b=self.return_a, total_length=self.total_length,
            per_side={1 - side: [v[2], v[1], v[0]] for side, v in self.per_side.items()},
        )

    def to_row(self) -> dict[str, Any]:
        n = self.n
        points = self.wins + 0.5 * self.draws
        per_side = {}
        for side, (w, d, losses) in sorted(self.per_side.items()):
            m = w + d + losses
            per_side[str(side)] = {"n": m, "wins": w, "draws": d, "losses": losses,
                                   "score": (w + 0.5 * d) / m if m else 0.0}
        return {
            "agent_a": self.agent_a, "agent_b": self.agent_b, "n": n,
            "wins": self.wins, "draws": self.draws, "losses": self.losses,
            "win_rate": self.wins / n if n else 0.0, "win_rate_ci": list(wilson_interval(self.wins, n)),
            "score": points / n if n else 0.0, "score_ci": list(wilson_interval(points, n)),
            "mean_return_a": self.return_a / n if n else 0.0, "mean_return_b": self.return_b / n if n else 0.0,
            "mean_episode_length": self.total_length / n if n else 0.0,
            "per_side": per_side,
        }


def _by_role(results: Sequence[MatchResult]) -> dict[str, dict[str, dict[str, float]]]:
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        for seat in result.seats:
            acc[seat.role][seat.agent_id].append(float(seat.reward))
    return {role: {agent: {"n": len(v), "mean_return": float(np.mean(v))} for agent, v in sorted(agents.items())}
            for role, agents in sorted(acc.items())}


def _summarize_wdl(results: Sequence[MatchResult]) -> dict[str, Any]:
    pairs: dict[tuple[str, str], PairStats] = {}
    solo: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    solo_wdl: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    unattributed = 0
    for result in results:
        teams = _team_agents(result)
        names = sorted(set().union(*teams.values()))
        if len(names) == 1:
            ranks = {team.team: team.rank for team in result.teams}
            for seat in result.seats:
                solo[names[0]][seat.seat].append(float(seat.reward))
                other = min(rank for team, rank in ranks.items() if team != seat.team)
                own = ranks[seat.team]
                solo_wdl[names[0]][seat.seat][0 if own < other else (1 if own == other else 2)] += 1
            continue
        if len(names) != 2 or any(len(agents) != 1 for agents in teams.values()):
            unattributed += 1
            continue
        a, b = names
        team_a = next(t for t, agents in teams.items() if agents == {a})
        team_b = next(t for t, agents in teams.items() if agents == {b})
        pairs.setdefault((a, b), PairStats(a, b)).add(result, team_a, team_b)
    rows = []
    for pair in pairs.values():
        rows.append(pair.to_row())
        rows.append(pair.reversed().to_row())
    solo_rows = []
    for name, seats in solo.items():
        all_returns = [r for v in seats.values() for r in v]
        mean, low, high = normal_interval(all_returns)
        solo_rows.append({
            "agent": name, "n": len(all_returns), "mean_return": mean, "return_ci": [low, high],
            "per_seat": {str(s): {"n": len(v), "mean_return": float(np.mean(v)),
                                  "wins": solo_wdl[name][s][0], "draws": solo_wdl[name][s][1],
                                  "losses": solo_wdl[name][s][2]}
                         for s, v in sorted(seats.items())},
        })
    return {"pairs": rows, "solo": solo_rows, "unattributed": unattributed, "score_ci_method": SCORE_CI_METHOD}


def _summarize_rank(results: Sequence[MatchResult]) -> dict[str, Any]:
    ranks: dict[str, list[float]] = defaultdict(list)
    higher: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    solo: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    unattributed = 0
    for result in results:
        teams = _team_agents(result)
        team_rank = {team.team: team.rank for team in result.teams}
        names = set().union(*teams.values())
        if len(names) == 1:  # SP1 solo mode: one agent in every seat
            (name,) = names
            for seat in result.seats:
                solo[name][seat.seat].append((team_rank[seat.team], float(seat.reward)))
            continue
        if any(len(agents) != 1 for agents in teams.values()):
            unattributed += 1
        homogeneous = {t: next(iter(agents)) for t, agents in teams.items() if len(agents) == 1}
        for team, agent in homogeneous.items():
            ranks[agent].append(team_rank[team])
        for ta, tb in itertools.combinations(sorted(homogeneous), 2):
            a, b = homogeneous[ta], homogeneous[tb]
            if a == b:
                continue
            score = pairwise_rank_score(team_rank[ta], team_rank[tb])
            higher[a][b].append(score)
            higher[b][a].append(1.0 - score)
    agents = {}
    for agent, values in sorted(ranks.items()):
        mean, low, high = normal_interval(values)
        agents[agent] = {"n": len(values), "mean_rank": mean, "rank_ci": [low, high],
                         "first_place_rate": float(np.mean([v == 1.0 for v in values]))}
    table = {a: {b: {"n": len(v), "rate": float(np.mean(v))} for b, v in sorted(row.items())}
             for a, row in sorted(higher.items())}
    solo_rows = []
    for name, seats in sorted(solo.items()):
        all_returns = [ret for cells in seats.values() for _rank, ret in cells]
        mean, low, high = normal_interval(all_returns)
        solo_rows.append({
            "agent": name, "n": len(all_returns), "mean_return": mean, "return_ci": [low, high],
            "per_seat": {str(s): {"n": len(cells), "mean_rank": float(np.mean([r for r, _ in cells])),
                                  "first_place_rate": float(np.mean([r == 1.0 for r, _ in cells])),
                                  "mean_return": float(np.mean([ret for _, ret in cells]))}
                         for s, cells in sorted(seats.items())},
        })
    return {"agents": agents, "higher": table, "solo": solo_rows, "unattributed": unattributed}


def _summarize_score(results: Sequence[MatchResult]) -> dict[str, Any]:
    scores: dict[str, list[float]] = defaultdict(list)
    for result in results:
        scores[composition_key(seat.agent_id for seat in result.seats)].append(float(result.teams[0].score))
    compositions = {}
    for key, values in sorted(scores.items()):
        mean, low, high = normal_interval(values)
        compositions[key] = {"n": len(values), "mean_score": mean, "score_ci": [low, high],
                             "homogeneous": len(set(key.split("+"))) == 1}
    return {"compositions": compositions}


@dataclass
class EvalReport:
    """Result of an evaluation; ``to_dict()`` is the JSON written by ``--output``."""

    agents: list[str]
    num_matches: int
    deterministic: bool
    layouts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"agents": list(self.agents), "num_matches": self.num_matches, "deterministic": self.deterministic,
                "ci_level": 0.95, "layouts": self.layouts}

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def text(self) -> str:
        lines: list[str] = []
        for layout, report in self.layouts.items():
            lines.append(f"== layout {layout} ({report['outcome_kind']}, {report['n']} matches)")
            if report["outcome_kind"] == "wdl":
                for r in report["pairs"]:
                    lines.append(
                        f"{r['agent_a']:<16} vs {r['agent_b']:<16} n={r['n']:<5} W={r['wins']:<5} "
                        f"D={r['draws']:<5} L={r['losses']:<5} "
                        f"win_rate {r['win_rate']:.3f} [{r['win_rate_ci'][0]:.3f}, {r['win_rate_ci'][1]:.3f}]  "
                        f"score {r['score']:.3f} [{r['score_ci'][0]:.3f}, {r['score_ci'][1]:.3f}]")
                    for side, cell in r["per_side"].items():
                        lines.append(f"{'':<20}as team {side}: n={cell['n']} W={cell['wins']} "
                                     f"D={cell['draws']} L={cell['losses']} score={cell['score']:.3f}")
                for r in report["solo"]:
                    lines.append(f"{r['agent']:<16} alone: n={r['n']} mean_return {r['mean_return']:.3f} "
                                 f"[{r['return_ci'][0]:.3f}, {r['return_ci'][1]:.3f}]")
                if report["unattributed"]:
                    lines.append(f"({report['unattributed']} matches with mixed teams not attributed to a pair)")
            elif report["outcome_kind"] == "rank":
                for agent, r in report["agents"].items():
                    lines.append(f"{agent:<16} n={r['n']:<5} mean_rank {r['mean_rank']:.3f} "
                                 f"[{r['rank_ci'][0]:.3f}, {r['rank_ci'][1]:.3f}]  "
                                 f"first places {r['first_place_rate']:.3f}")
                for a, row in report["higher"].items():
                    for b, cell in row.items():
                        lines.append(f"{'':<4}{a} above {b}: {cell['rate']:.3f} (n={cell['n']})")
                for r in report["solo"]:
                    lines.append(f"{r['agent']:<16} alone: n={r['n']} mean_return {r['mean_return']:.3f} "
                                 f"[{r['return_ci'][0]:.3f}, {r['return_ci'][1]:.3f}]")
                    for seat, cell in r["per_seat"].items():
                        lines.append(f"{'':<20}seat {seat}: n={cell['n']} mean_rank {cell['mean_rank']:.3f} "
                                     f"first places {cell['first_place_rate']:.3f} "
                                     f"mean_return {cell['mean_return']:.3f}")
                if report["unattributed"]:
                    lines.append(f"({report['unattributed']} matches with mixed teams: only their "
                                 f"single-agent teams are counted)")
            else:
                for key, r in report["compositions"].items():
                    lines.append(f"{key:<24} n={r['n']:<5} mean_score {r['mean_score']:.3f} "
                                 f"[{r['score_ci'][0]:.3f}, {r['score_ci'][1]:.3f}]"
                                 f"{'' if r['homogeneous'] else '  (cross-play)'}")
            for role, agents in report["by_role"].items():
                for agent, cell in agents.items():
                    lines.append(f"{'':<4}role {role}: {agent} mean_return {cell['mean_return']:.3f} (n={cell['n']})")
        return "\n".join(lines)


def summarize(spec: GameSpec, results: Sequence[MatchResult], *, agents: Sequence[str] = (),
              num_matches: int = 0, deterministic: bool = False) -> EvalReport:
    """Aggregate match results into an :class:`EvalReport`, one section per layout."""
    by_layout: dict[str, list[MatchResult]] = defaultdict(list)
    for result in results:
        by_layout[result.layout].append(result)
    names = list(agents) or sorted({seat.agent_id for r in results for seat in r.seats})
    report = EvalReport(agents=names, num_matches=num_matches, deterministic=deterministic)
    unknown = sorted(set(by_layout) - set(spec.layouts))
    if unknown:
        raise ValueError(f"summarize: results of layouts {unknown} that the game does not have")
    for layout in (name for name in spec.layouts if name in by_layout):  # sections in the game's layout order
        layout_results = by_layout[layout]
        kind = spec.outcome_kind(layout)
        section: dict[str, Any] = {"outcome_kind": kind, "n": len(layout_results)}
        if kind == "wdl":
            section.update(_summarize_wdl(layout_results))
        elif kind == "rank":
            section.update(_summarize_rank(layout_results))
        else:
            section.update(_summarize_score(layout_results))
        section["by_role"] = _by_role(layout_results)
        report.layouts[layout] = section
    return report


# ---------------------------------------------------------------------------
# Loading agents and the CLI-level evaluation
# ---------------------------------------------------------------------------


def load_eval_model(config: ColosseumConfig, name: str, path: str | Path, *,
                    spec: GameSpec | None = None, validated: set[str] | None = None) -> tuple[PolicyModel, list[str]]:
    """Build and load one evaluation agent; returns ``(model in eval mode, roles)``.

    - A checkpoint dir (read with ``load_checkpoint_dir``: strict, never modified): roles and
      ``role_signature`` from its ``meta.json`` (required; the signature must match the game's
      spaces for those roles), architecture from its ``networks`` (else the agent's config).
    - A ``.pt`` state_dict: architecture and roles of ``agents.<name>`` when configured (a trainable
      agent's effective networks and roles; a frozen agent's ``networks`` / ``roles`` overrides), else the
      global ``networks`` with every role of the game (which must share one signature).

    Loading is ``players.registry.load_frozen`` + ``build_frozen_model`` (the loader frozen agents share).

    The built architecture is checked like ``validate_config`` checks an agent's model (``step``
    and ``unroll`` on the role's spaces), once per distinct (networks, roles): keys already in
    ``validated`` are skipped and new ones are added (``None``: always check).

    Problems raise ConfigError naming the path; a path that is neither a dir nor a ``.pt`` file
    raises FileNotFoundError.
    """
    from colosseum.core.registry import env_spec
    from colosseum.players.registry import build_frozen_model, load_frozen

    spec = spec if spec is not None else env_spec(config)
    frozen = load_frozen(config, name, path, spec)
    model = build_frozen_model(config, frozen, spec)
    role = agent_role_spec(spec, list(frozen.roles))
    key = json.dumps([frozen.networks, list(frozen.roles)], sort_keys=True)
    if validated is None or key not in validated:
        try:
            _check_model(model, role, None, f"agent {name!r}")
        except ConfigError as e:
            raise ConfigError(f"{frozen.networks_source}: {e}") from e
        if validated is not None:
            validated.add(key)
    return model, list(frozen.roles)


def load_player(config: ColosseumConfig, name: str, path: str | None, *, spec: GameSpec,
                validated: set[str] | None = None,
                option: str = "-a") -> tuple[PolicyModel | ScriptedPlayer, list[str]]:
    """One player of ``eval -a`` / ``record --player`` / ``record --against``: ``path=None`` is the scripted
    or frozen agent ``name`` of the config (a trainable agent needs a path: ConfigError naming ``option``),
    otherwise a checkpoint dir or a ``.pt`` (``load_eval_model``). Returns ``(player, roles)``.

    A path that is neither a directory nor a ``.pt`` file is a ConfigError."""
    from colosseum.players.registry import BotSpec, make_bot, resolve_player_roles

    if path is None:
        entry = config.agent_entry(name)          # ConfigError for an unknown name
        if entry.kind == "trainable":
            raise ConfigError(f"{option} {name}: '{name}' is a trainable agent, whose weights are not in the config; "
                              f"use {option} {name}=<checkpoint dir or .pt>")
        if entry.kind == "scripted":
            bot = BotSpec(entry.class_path, dict(entry.kwargs))
            make_bot(bot, spec)                    # a bad class or kwargs fail here, as a ConfigError
            roles = list(resolve_player_roles(config, spec)[name])
            return ScriptedPlayer(functools.partial(make_bot, bot, spec)), roles
        path = entry.path
        p = Path(path)
        if not (p.is_dir() or (p.is_file() and p.suffix == ".pt")):
            raise ConfigError(f"agents.{name}.path={path!r}: expected a checkpoint dir or a .pt file")
    else:
        p = Path(path)
        if not (p.is_dir() or (p.is_file() and p.suffix == ".pt")):
            raise ConfigError(f"{option} {name}={path}: expected a checkpoint dir or a .pt file")
    return load_eval_model(config, name, path, spec=spec, validated=validated)


def default_layouts(spec: GameSpec, players: Mapping[str, Sequence[str]]) -> list[str]:
    """Every layout of the game for which ``schedule_lineups`` has at least one lineup."""
    return [name for name in spec.layouts if schedule_lineups(spec, name, players, 1)]


def evaluate(config: ColosseumConfig, agents: Mapping[str, str | None], *, layouts: Sequence[str] | None,
             num_matches: int, seed: int | None = None, deterministic: bool = False,
             num_envs: int = 8) -> EvalReport:
    """Load ``agents`` (name -> checkpoint dir or ``.pt``; ``None`` = the scripted or frozen agent of that name
    in the config), schedule, play and summarize."""
    from colosseum.core.registry import env_spec, make_env

    spec = env_spec(config)
    models: dict[str, PolicyModel | ScriptedPlayer] = {}
    players: dict[str, list[str]] = {}
    validated: set[str] = set()  # each distinct architecture is checked once
    for name, path in agents.items():
        models[name], players[name] = load_player(config, name, path, spec=spec, validated=validated)
    chosen = list(dict.fromkeys(layouts)) if layouts else default_layouts(spec, players)
    unknown = sorted(set(chosen) - set(spec.layouts))
    if unknown:
        raise ConfigError(f"--layout {unknown}: not layouts of the game {sorted(spec.layouts)}")
    lineups: list[Lineup] = []
    for layout in chosen:
        layout_lineups = schedule_lineups(spec, layout, players, num_matches)
        if not layout_lineups:
            raise ConfigError(f"layout {layout!r}: the agents {sorted(players)} cannot fill its seats")
        lineups.extend(layout_lineups)
    if not lineups:
        raise ConfigError(f"no layout of the game can be filled by the agents {sorted(players)}")
    results = play_lineups(env_fn=lambda: make_env(config), models=models, lineups=lineups, num_envs=num_envs,
                           seed=seed, deterministic=deterministic, max_idle_steps=config.env.max_idle_steps)
    return summarize(spec, results, agents=list(agents), num_matches=num_matches, deterministic=deterministic)
