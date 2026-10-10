"""Per-interval aggregation of match results (by layout and role, played opponent shares) and system
throughput."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from colosseum.core.types import (
    FIXED_NETWORK_ID,
    LATEST_NETWORK_ID,
    OPPONENT_CATEGORIES,
    SOURCE_OWNER,
    MatchResult,
    SeatResult,
)

OPPONENT_TYPES = ("latest", "past", "arena", "anchor")
# Default cadence of worker stats (``rollout_worker_process(stats_interval_sec=...)``).
WORKER_STATS_INTERVAL_SEC = 2.0


def opponent_type(result: MatchResult, seat: SeatResult) -> str | None:
    """Kind of opposition a seat met; teammates are not opponents.

    'anchor' (a seat of another team is played by a scripted or frozen agent), 'arena' (a seat of another
    team plays another agent), 'past' (one plays a snapshot of the seat's agent), 'latest' (all play the
    agent's latest weights), or None (no other team: solo or cooperative layouts).
    """
    opponents = [s for s in result.seats if s.team != seat.team]
    if not opponents:
        return None
    if any(s.network_id == FIXED_NETWORK_ID for s in opponents):
        return "anchor"
    if any(s.agent_id != seat.agent_id for s in opponents):
        return "arena"
    if any(s.network_id != LATEST_NETWORK_ID for s in opponents):
        return "past"
    return "latest"


def opponent_draws(result: MatchResult) -> tuple[str, list[tuple[str, str | None]]] | None:
    """The matchmaker's draws behind a result: ``(owner, [(category, anchor or None) per opposing team])``.

    The owner's team is the team whose seats carry ``SOURCE_OWNER``; the owner is the only agent with
    latest weights on it. None for results without matchmaker sources (eval, tests) and for owner
    teams with the latest weights of several agents (``teammates: mixed``): their owner is ambiguous,
    and leaving them out does not bias the shares (teammates are drawn independently of opponents).
    The anchor of a team drawn from ``anchors`` is its most frequent agent on ``fixed`` seats.
    """
    by_team: dict[int, list[SeatResult]] = defaultdict(list)
    for seat in result.seats:
        by_team[seat.team].append(seat)
    owner_teams = [team for team, seats in by_team.items() if seats[0].source == SOURCE_OWNER]
    if len(owner_teams) != 1:
        return None
    owner_team = owner_teams[0]
    latest = {s.agent_id for s in by_team[owner_team] if s.network_id == LATEST_NETWORK_ID}
    if len(latest) != 1:
        return None
    draws: list[tuple[str, str | None]] = []
    for team in sorted(by_team):
        if team == owner_team:
            continue
        category = by_team[team][0].source
        if category not in OPPONENT_CATEGORIES:
            return None
        anchor = None
        if category == "anchors":
            fixed = Counter(s.agent_id for s in by_team[team] if s.network_id == FIXED_NETWORK_ID)
            anchor = fixed.most_common(1)[0][0] if fixed else None
        draws.append((category, anchor))
    return latest.pop(), draws


@dataclass
class _OpponentCell:
    """Opposing teams of one (owner, layout) and the categories / anchors they were drawn from."""

    teams: int = 0
    categories: Counter = field(default_factory=Counter)
    anchors: Counter = field(default_factory=Counter)

    def summary(self) -> dict[str, Any]:
        n = self.teams
        return {"teams": n, "categories": {c: self.categories[c] / n for c in OPPONENT_CATEGORIES},
                "anchors": {a: k / n for a, k in sorted(self.anchors.items())}}


def _wdl_counts() -> dict[str, list[int]]:
    return {t: [0, 0, 0] for t in OPPONENT_TYPES}


@dataclass
class _RoleCell:
    """Sums for one (agent, layout, role) cell of ``by_layout``."""

    seats: int = 0
    returns: float = 0.0
    lengths: float = 0.0
    team_scores: float = 0.0
    eliminated: int = 0
    wdl: dict[str, list[int]] = field(default_factory=_wdl_counts)

    def summary(self) -> dict[str, Any]:
        n = self.seats
        return {"episodes": n, "return_mean": self.returns / n, "length_mean": self.lengths / n,
                "team_score_mean": self.team_scores / n, "eliminated_frac": self.eliminated / n,
                "wdl": {t: list(v) for t, v in self.wdl.items()}}


class EpisodeAggregator:
    """Per agent, over the seats that play the agent's latest weights:

    - mean return and episode length, W/D/L by opponent type (win: the seat's team ranks strictly
      better than every other team; draw: ties the best one), seat counts;
    - ``by_layout[layout][role]``: episodes (seats), mean return, mean episode length, mean team
      score, the share of seats eliminated before the episode end, and W/D/L by opponent type;
    - ``opponents[layout]``: opposing teams drawn for the agent as data owner, the played share of
      every opponent category and of every anchor (``opponent_draws``; played, not drawn: a staged
      lineup can be replaced before it is played).
    """

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._returns: dict[str, list[float]] = defaultdict(list)
        self._lengths: dict[str, list[int]] = defaultdict(list)
        self._wdl: dict[str, dict[str, list[int]]] = defaultdict(_wdl_counts)
        self._seats: dict[str, list[int]] = defaultdict(list)
        # agent -> layout -> role -> sums
        self._by_layout: dict[str, dict[str, dict[str, _RoleCell]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(_RoleCell)))
        # owner -> layout -> opposing teams by category / anchor
        self._opponents: dict[str, dict[str, _OpponentCell]] = defaultdict(lambda: defaultdict(_OpponentCell))

    def add(self, result: MatchResult) -> None:
        ranks = {team.team: team.rank for team in result.teams}
        scores = {team.team: team.score for team in result.teams}
        length = int(result.episode_length)
        for seat in result.seats:
            if seat.network_id != LATEST_NETWORK_ID:
                continue
            agent_id = seat.agent_id
            self._returns[agent_id].append(float(seat.reward))
            self._lengths[agent_id].append(length)
            counts = self._seats[agent_id]
            while len(counts) <= seat.seat:
                counts.append(0)
            counts[seat.seat] += 1
            cell = self._by_layout[agent_id][result.layout][seat.role]
            cell.seats += 1
            cell.returns += float(seat.reward)
            cell.lengths += length
            cell.team_scores += float(scores.get(seat.team, 0.0))
            cell.eliminated += int(seat.eliminated_step is not None)
            kind = opponent_type(result, seat)
            if kind is None:
                continue
            own = ranks[seat.team]
            best_other = min(rank for team, rank in ranks.items() if team != seat.team)
            idx = 0 if own < best_other else (1 if own == best_other else 2)
            self._wdl[agent_id][kind][idx] += 1
            cell.wdl[kind][idx] += 1
        draws = opponent_draws(result)
        if draws is not None and draws[1]:          # cooperative layouts have no opposing team
            owner, teams = draws
            opponents = self._opponents[owner][result.layout]
            for category, anchor in teams:
                opponents.teams += 1
                opponents.categories[category] += 1
                if anchor is not None:
                    opponents.anchors[anchor] += 1

    def flush(self) -> dict[str, dict[str, Any]]:
        """Stats since the previous flush, per agent; resets the accumulators."""
        out: dict[str, dict[str, Any]] = {}
        for agent_id, returns in self._returns.items():
            by_layout = {
                layout: {role: cell.summary() for role, cell in sorted(roles.items())}
                for layout, roles in sorted(self._by_layout[agent_id].items())
            }
            out[agent_id] = {
                "episodes": len(returns),
                "return_mean": float(np.mean(returns)),
                "length_mean": float(np.mean(self._lengths[agent_id])),
                "wdl": {t: list(v) for t, v in self._wdl[agent_id].items()},
                "seat_counts": list(self._seats[agent_id]),
                "by_layout": by_layout,
                "opponents": {layout: cell.summary()
                              for layout, cell in sorted(self._opponents.get(agent_id, {}).items())},
            }
        self._reset()
        return out


class SystemStats:
    """Throughput and queue health between two snapshots (SP1 behavior, unchanged).

    Rate baselines start at ``initial_env_steps`` and ``initial_train_steps[agent]`` (0 for
    agents not listed): the counters a resumed run continues from, so the first snapshot
    of a resumed run measures only progress made since the start, not the resumed totals.
    Per-worker stats older than ``worker_timeout_sec`` (a dead or stuck worker) are forgotten.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic, *, initial_env_steps: int = 0,
                 initial_train_steps: dict[str, int] | None = None,
                 worker_timeout_sec: float = 3 * WORKER_STATS_INTERVAL_SEC) -> None:
        self._clock = clock
        self._last_t = clock()
        self._last_env_steps = int(initial_env_steps)
        self._last_train_steps: dict[str, int] = {a: int(s) for a, s in (initial_train_steps or {}).items()}
        self._train_steps: dict[str, int] = dict(self._last_train_steps)
        self._worker_timeout = float(worker_timeout_sec)
        self._workers: dict[int, tuple[float, dict]] = {}

    def on_train_step(self, agent_id: str, train_step: int) -> None:
        self._train_steps[agent_id] = max(int(train_step), self._train_steps.get(agent_id, 0))

    def on_worker_stats(self, stats: dict) -> None:
        self._workers[int(stats["worker_id"])] = (self._clock(), dict(stats))

    def snapshot(self, env_steps: int, queue_depths: dict[str, int]) -> dict[str, Any]:
        now = self._clock()
        dt = now - self._last_t
        self._workers = {w: v for w, v in self._workers.items() if now - v[0] <= self._worker_timeout}

        def rate(delta: float) -> float:
            return delta / dt if dt > 1e-3 else 0.0

        snap = {
            "env_steps": int(env_steps),
            "env_steps_per_sec": rate(env_steps - self._last_env_steps),
            "train_steps_per_sec": {
                a: rate(s - self._last_train_steps.get(a, 0)) for a, s in self._train_steps.items()
            },
            "queue_depths": dict(queue_depths),
            "parked_buffers": int(sum(int(w.get("parked_buffers", 0)) for _, w in self._workers.values())),
            "dropped_reward_episodes": int(sum(int(w.get("dropped_reward_episodes", 0))
                                               for _, w in self._workers.values())),
            "workers_reporting": len(self._workers),
        }
        self._last_t = now
        self._last_env_steps = int(env_steps)
        self._last_train_steps = dict(self._train_steps)
        return snap
