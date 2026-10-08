"""Evaluation: inference-only matches between agents and checkpoints.

Engine
------
``schedule_lineups`` turns agent names into a list of matches (one seat -> agent
lineup per match). ``play_matches`` plays them on a ``VectorEnv`` with batched
``act()`` inference per agent and one model ``State`` per (env, seat):

- the state starts at ``initial_state(1)`` at every episode start;
- it advances only when that seat acts (``info[p]["active"]``, default True);
  non-acting seats send the all-zeros action and keep their state;
- masks follow the worker's rules (``core/seat_info.py``): an acting seat whose
  action mask allows nothing raises ``EnvContractError``.

These are the same semantics as training rollouts, so stateful models (LSTM,
GRU, attention) are evaluated as the policy they were trained as.

Seat rotation: for a pair (a, b), match m gives seat s to ``(a, b)[(s + m) % 2]``.
With an even number of matches per pair each agent plays every seat equally
often. Every scheduled match is played to completion; extra episodes that idle
envs play while others finish are discarded, so short episodes are not favoured.

Statistics
----------
Pairwise (two or more agents in an N-player game, N >= 2). From agent a's side a
match is a win when a's mean seat outcome (``core.outcomes``) beats b's, a draw
when equal. Each pair reports:

- W/D/L and the win rate W/n with a 95% Wilson score interval;
- the score (W + D/2)/n with a 95% Wilson interval computed on the score as if it
  were a Bernoulli proportion. This is an approximation: a draw counts as half a
  win, and the per-match score variance (W + D/4)/n - s^2 never exceeds the
  Bernoulli variance s(1 - s), so the interval is conservative (never too narrow).
  Paired/rating-based selection (Bradley-Terry with bootstrap) is SP4;
- a per-seat breakdown keyed by the seats agent a occupied (e.g. "0", or "0,2");
- mean returns and mean episode length.

The reversed row (b vs a) is derived from the same counts, so the two rows never
contradict each other.

Solo (one agent, or a 1-player env): per agent, the mean episode return and the
mean outcome with 95% normal-approximation intervals ``mean +- 1.96 * sd / sqrt(n)``,
plus a per-seat breakdown when the env has several seats.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import math
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import ValidationError

from colosseum.coordinator.checkpoint_manager import check_model_state, load_checkpoint_dir
from colosseum.core.config import ColosseumConfig, NetworkConfig
from colosseum.core.errors import ConfigError
from colosseum.core.outcomes import player_outcomes
from colosseum.core.registry import build_model, validate_config
from colosseum.core.seat_info import acting_flags, check_masks, extract_masks
from colosseum.core.types import state_dict_from_numpy, state_dict_to_numpy
from colosseum.envs.base_env import BaseEnv
from colosseum.envs.vec_env import VectorEnv
from colosseum.networks.model import PolicyModel, act
from colosseum.networks.state import State, cat_batch, slice_batch

logger = logging.getLogger(__name__)

Z_95 = 1.959963984540054
SCORE_CI_METHOD = "wilson-on-score (draw = half win; conservative approximation)"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _eval_where(e: int, p: int) -> str:
    """Error-message context for eval env ``e``, seat ``p``."""
    return f"eval: env {e} seat {p}"


@dataclass(frozen=True)
class MatchRecord:
    """One finished match: agent name, outcome in [0, 1] and return per seat."""

    lineup: tuple[str, ...]
    outcomes: tuple[float, ...]
    returns: tuple[float, ...]
    length: int


def schedule_lineups(
    agent_names: Sequence[str], num_players: int, num_matches: int,
) -> list[tuple[str, ...]]:
    """Seat -> agent lineups for an evaluation.

    - One agent, or a 1-player env (solo): ``num_matches`` lineups per agent with
      the agent in every seat.
    - Otherwise (pairwise): for every pair (a, b) in ``itertools.combinations``
      order, ``num_matches`` lineups where match m gives seat s to
      ``(a, b)[(s + m) % 2]``. With even ``num_matches`` every agent plays every
      seat equally often.
    """
    names = list(agent_names)
    if not names:
        raise ValueError("schedule_lineups: no agents")
    if len(set(names)) != len(names):
        raise ValueError(f"schedule_lineups: duplicate agent names in {names}")
    if num_players < 1:
        raise ValueError(f"schedule_lineups: num_players must be >= 1, got {num_players}")
    if num_matches < 1:
        raise ValueError(f"schedule_lineups: num_matches must be >= 1, got {num_matches}")
    if len(names) == 1 or num_players == 1:
        return [tuple([name] * num_players) for name in names for _ in range(num_matches)]
    if num_matches % 2:
        logger.warning(
            "eval: num_matches=%d is odd, so seats are balanced only up to one match per pair",
            num_matches,
        )
    lineups: list[tuple[str, ...]] = []
    for a, b in itertools.combinations(names, 2):
        pair = (a, b)
        for m in range(num_matches):
            lineups.append(tuple(pair[(s + m) % 2] for s in range(num_players)))
    return lineups


def play_matches(
    models: dict[str, PolicyModel],
    env_fn: Callable[[], BaseEnv],
    lineups: Sequence[tuple[str, ...]],
    num_envs: int = 8,
    deterministic: bool = False,
    seed: int | None = None,
) -> list[MatchRecord]:
    """Play every lineup once, to completion; one record per lineup, in completion order.

    Each (env, seat) has its own model State: ``initial_state(1)`` at episode
    start, advanced only when the seat acts (``info[p]["active"]``, default True),
    reset at episode end. Non-acting seats send the zero action. Masks follow the
    worker's rules (``core/seat_info.py``): an acting seat without a legal action
    raises :class:`~colosseum.core.errors.EnvContractError`. Envs left without a
    scheduled match keep playing their last lineup until the rest finish; those
    extra episodes are discarded, so short episodes are not favoured.

    ``seed`` seeds the env resets and a forked torch RNG, so the caller's global
    RNG is untouched. Models run in eval mode; their train/eval flags are
    restored on return, also after an exception.
    """
    lineups = [tuple(lu) for lu in lineups]
    if not lineups:
        return []
    unknown = sorted({name for lu in lineups for name in lu} - set(models))
    if unknown:
        raise ValueError(f"play_matches: lineups use unknown agents {unknown}")
    if num_envs < 1:
        raise ValueError(f"play_matches: num_envs must be >= 1, got {num_envs}")
    # A seed must not leak into the caller: sample from a forked torch RNG. Only
    # the CPU generator is forked (``devices=[]``): eval feeds CPU tensors built
    # from numpy, so models run and sample on the CPU.
    rng = torch.random.fork_rng(devices=[]) if seed is not None else contextlib.nullcontext()
    # Restore every submodule's own flag (a caller may mix train/eval submodules).
    was_training = [(m, m.training) for model in models.values() for m in model.modules()]
    try:
        with rng:
            if seed is not None:
                torch.manual_seed(seed)
            for model in models.values():
                model.eval()
            vec_env = VectorEnv(env_fn, min(num_envs, len(lineups)))
            try:
                return _play(vec_env, models, lineups, deterministic, seed)
            finally:
                vec_env.close()
    finally:
        for module, training in was_training:
            module.training = training


def _play(
    vec_env: VectorEnv,
    models: dict[str, PolicyModel],
    lineups: list[tuple[str, ...]],
    deterministic: bool,
    seed: int | None,
) -> list[MatchRecord]:
    n_envs, n_players, spec = vec_env.num_envs, vec_env.num_players, vec_env.action_spec
    bad = [lu for lu in lineups if len(lu) != n_players]
    if bad:
        raise ValueError(f"play_matches: lineup {bad[0]} does not have {n_players} seats")

    pending = deque(lineups)
    playing: list[tuple[str, ...]] = [pending.popleft() for _ in range(n_envs)]
    counted = [True] * n_envs   # False once an env has no scheduled match left
    states: list[list[State]] = [
        [models[playing[e][p]].initial_state(1) for p in range(n_players)]
        for e in range(n_envs)
    ]
    returns = np.zeros((n_envs, n_players), dtype=np.float64)
    lengths = np.zeros(n_envs, dtype=np.int64)
    records: list[MatchRecord] = []

    obs, infos = vec_env.reset_all(seed=seed)
    while len(records) < len(lineups):
        acting = acting_flags(infos, n_envs, n_players)
        masks = extract_masks(infos, n_envs, n_players, spec)
        if masks is not None:
            check_masks(masks, acting, spec, _eval_where)
            masks = masks.reshape(n_envs, n_players, -1)

        actions = np.zeros((n_envs, n_players, *spec.action_shape), dtype=spec.numpy_dtype)
        groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for e in range(n_envs):
            for p in range(n_players):
                if acting[e, p]:
                    groups[playing[e][p]].append((e, p))
        for name, seats in groups.items():
            ei = [e for e, _ in seats]
            pi = [p for _, p in seats]
            obs_b = torch.from_numpy(np.ascontiguousarray(obs[ei, pi], dtype=np.float32))
            mask_b = None if masks is None else torch.from_numpy(np.ascontiguousarray(masks[ei, pi]))
            state_b = cat_batch([states[e][p] for e, p in seats])
            with torch.no_grad():
                out = act(models[name], obs_b, state_b, mask_b, deterministic=deterministic)
            actions[ei, pi] = out.actions.cpu().numpy().astype(spec.numpy_dtype, copy=False)
            for k, (e, p) in enumerate(seats):
                states[e][p] = None if out.state is None else slice_batch(out.state, k)

        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        returns += rewards
        lengths += 1

        for e in range(n_envs):
            if not (terminated[e] or truncated[e]):
                continue
            if counted[e]:
                terminal_infos = {
                    p: (infos[e].get(p, {}) or {}).get("terminal_info", {}) for p in range(n_players)
                }
                outcomes = player_outcomes(returns[e].tolist(), terminal_infos, n_players)
                records.append(MatchRecord(
                    lineup=playing[e],
                    outcomes=tuple(float(x) for x in outcomes),
                    returns=tuple(float(x) for x in returns[e]),
                    length=int(lengths[e]),
                ))
            if pending:
                playing[e] = pending.popleft()
            else:
                counted[e] = False   # keep the env busy with its last lineup; results ignored
            states[e] = [models[playing[e][p]].initial_state(1) for p in range(n_players)]
            returns[e] = 0.0
            lengths[e] = 0
    return records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_interval(successes: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion ``successes / n``.

    ``successes`` may be fractional (score with draws as half wins). The result
    always contains the point estimate; ``n == 0`` gives ``(0.0, 1.0)``.
    """
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(0.0, p * (1.0 - p)) / n + z * z / (4.0 * n * n)) / denom
    low = min(max(0.0, center - half), p)
    high = max(min(1.0, center + half), p)
    return low, high


def normal_interval(values: Sequence[float], z: float = Z_95) -> tuple[float, float, float]:
    """``(mean, low, high)`` with ``mean +- z * sd / sqrt(n)`` (sd with ddof=1).

    Fewer than two values give a zero-width interval; no values give zeros.
    """
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(x.mean())
    if x.size < 2:
        return mean, mean, mean
    half = z * float(x.std(ddof=1)) / math.sqrt(x.size)
    return mean, mean - half, mean + half


def _seat_key(seats: Sequence[int]) -> str:
    return ",".join(str(s) for s in seats)


@dataclass
class PairStats:
    """Counts for one unordered pair, from ``agent_a``'s side."""

    agent_a: str
    agent_b: str
    num_players: int
    wins: int = 0
    draws: int = 0
    losses: int = 0
    return_a: float = 0.0
    return_b: float = 0.0
    total_length: int = 0
    per_seat: dict[tuple[int, ...], list[int]] = field(default_factory=dict)  # a's seats -> [W, D, L]

    @property
    def n(self) -> int:
        return self.wins + self.draws + self.losses

    def add(self, record: MatchRecord) -> None:
        seats_a = tuple(s for s, name in enumerate(record.lineup) if name == self.agent_a)
        seats_b = tuple(s for s, name in enumerate(record.lineup) if name == self.agent_b)
        if not seats_a or not seats_b or len(seats_a) + len(seats_b) != len(record.lineup):
            raise ValueError(
                f"PairStats({self.agent_a}, {self.agent_b}): record lineup {record.lineup} "
                f"is not a lineup of this pair"
            )
        score_a = float(np.mean([record.outcomes[s] for s in seats_a]))
        score_b = float(np.mean([record.outcomes[s] for s in seats_b]))
        cell = self.per_seat.setdefault(seats_a, [0, 0, 0])
        if score_a > score_b:
            self.wins += 1
            cell[0] += 1
        elif score_a < score_b:
            self.losses += 1
            cell[2] += 1
        else:
            self.draws += 1
            cell[1] += 1
        self.return_a += float(np.mean([record.returns[s] for s in seats_a]))
        self.return_b += float(np.mean([record.returns[s] for s in seats_b]))
        self.total_length += record.length

    def reversed(self) -> PairStats:
        """The same counts seen from ``agent_b``'s side."""
        def complement(seats: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(s for s in range(self.num_players) if s not in seats)

        return PairStats(
            agent_a=self.agent_b, agent_b=self.agent_a, num_players=self.num_players,
            wins=self.losses, draws=self.draws, losses=self.wins,
            return_a=self.return_b, return_b=self.return_a, total_length=self.total_length,
            per_seat={complement(k): [v[2], v[1], v[0]] for k, v in self.per_seat.items()},
        )

    def to_row(self) -> dict[str, Any]:
        n = self.n
        points = self.wins + 0.5 * self.draws
        per_seat = {}
        for seats, (w, d, losses) in sorted(self.per_seat.items()):
            m = w + d + losses
            per_seat[_seat_key(seats)] = {
                "n": m, "wins": w, "draws": d, "losses": losses,
                "score": (w + 0.5 * d) / m if m else 0.0,
            }
        return {
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "n": n,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "win_rate": self.wins / n if n else 0.0,
            "win_rate_ci": list(wilson_interval(self.wins, n)),
            "score": points / n if n else 0.0,
            "score_ci": list(wilson_interval(points, n)),
            "mean_return_a": self.return_a / n if n else 0.0,
            "mean_return_b": self.return_b / n if n else 0.0,
            "mean_episode_length": self.total_length / n if n else 0.0,
            "per_seat": per_seat,
        }


@dataclass
class SoloStats:
    """Per-agent episode statistics for solo evaluation."""

    agent: str
    num_players: int
    returns: list[float] = field(default_factory=list)    # per match: mean over seats
    outcomes: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    seat_returns: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    seat_outcomes: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, record: MatchRecord) -> None:
        if set(record.lineup) != {self.agent}:
            raise ValueError(f"SoloStats({self.agent}): unexpected lineup {record.lineup}")
        self.returns.append(float(np.mean(record.returns)))
        self.outcomes.append(float(np.mean(record.outcomes)))
        self.lengths.append(record.length)
        for seat in range(len(record.lineup)):
            self.seat_returns[seat].append(record.returns[seat])
            self.seat_outcomes[seat].append(record.outcomes[seat])

    def to_row(self) -> dict[str, Any]:
        mean_ret, ret_lo, ret_hi = normal_interval(self.returns)
        mean_out, out_lo, out_hi = normal_interval(self.outcomes)
        per_seat = {
            str(seat): {
                "n": len(self.seat_returns[seat]),
                "mean_return": float(np.mean(self.seat_returns[seat])),
                "mean_outcome": float(np.mean(self.seat_outcomes[seat])),
            }
            for seat in sorted(self.seat_returns)
        }
        return {
            "agent": self.agent,
            "n": len(self.returns),
            "mean_return": mean_ret,
            "return_ci": [ret_lo, ret_hi],
            "mean_outcome": mean_out,
            "outcome_ci": [out_lo, out_hi],
            "mean_episode_length": float(np.mean(self.lengths)) if self.lengths else 0.0,
            "per_seat": per_seat,
        }


@dataclass
class EvalReport:
    """Result of an evaluation; ``to_dict()`` is the JSON schema written by ``--output``."""

    agents: list[str]
    num_players: int
    num_matches: int
    deterministic: bool
    pairs: list[PairStats] = field(default_factory=list)
    solo: list[SoloStats] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return "solo" if self.solo else "pairwise"

    def rows(self) -> list[dict[str, Any]]:
        """Both directions of every pair, derived from one set of counts."""
        out: list[dict[str, Any]] = []
        for pair in self.pairs:
            out.append(pair.to_row())
            out.append(pair.reversed().to_row())
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "agents": list(self.agents),
            "num_players": self.num_players,
            "num_matches_per_pair": self.num_matches,
            "deterministic": self.deterministic,
            "ci_level": 0.95,
            "score_ci_method": SCORE_CI_METHOD,
            "pairs": self.rows(),
            "solo": [s.to_row() for s in self.solo],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def summary(self) -> str:
        lines: list[str] = []
        if self.mode == "solo":
            header = (f"{'agent':<16} {'n':>6}  {'mean_return [95% CI]':<30} "
                      f"{'mean_outcome [95% CI]':<30} {'mean_len':>8}")
            lines += [header, "-" * len(header)]
            for s in self.solo:
                r = s.to_row()
                lines.append(
                    f"{r['agent']:<16} {r['n']:>6}  "
                    f"{r['mean_return']:>9.3f} [{r['return_ci'][0]:.3f}, {r['return_ci'][1]:.3f}]  "
                    f"{r['mean_outcome']:>6.3f} [{r['outcome_ci'][0]:.3f}, {r['outcome_ci'][1]:.3f}]  "
                    f"{r['mean_episode_length']:>8.1f}"
                )
            return "\n".join(lines)
        header = (f"{'agent_a':<16} {'agent_b':<16} {'n':>6} {'W':>6} {'D':>6} {'L':>6}  "
                  f"{'win_rate [95% CI]':<24} {'score [95% CI]':<24}")
        lines += [header, "-" * len(header)]
        for r in self.rows():
            lines.append(
                f"{r['agent_a']:<16} {r['agent_b']:<16} {r['n']:>6} {r['wins']:>6} "
                f"{r['draws']:>6} {r['losses']:>6}  "
                f"{r['win_rate']:.3f} [{r['win_rate_ci'][0]:.3f}, {r['win_rate_ci'][1]:.3f}]   "
                f"{r['score']:.3f} [{r['score_ci'][0]:.3f}, {r['score_ci'][1]:.3f}]"
            )
            for seats, cell in r["per_seat"].items():
                lines.append(
                    f"{'':<16}   seats {seats:<8} n={cell['n']:<5} W={cell['wins']:<5} "
                    f"D={cell['draws']:<5} L={cell['losses']:<5} score={cell['score']:.3f}"
                )
        lines.append(f"(score CI: {SCORE_CI_METHOD})")
        return "\n".join(lines)


def summarize(
    records: Sequence[MatchRecord],
    agent_names: Sequence[str],
    num_players: int,
    num_matches: int,
    deterministic: bool = False,
) -> EvalReport:
    """Aggregate match records into an :class:`EvalReport`."""
    names = list(agent_names)
    report = EvalReport(agents=names, num_players=num_players,
                        num_matches=num_matches, deterministic=deterministic)
    if len(names) == 1 or num_players == 1:
        solo = {name: SoloStats(name, num_players) for name in names}
        for record in records:
            solo[record.lineup[0]].add(record)
        report.solo = [solo[name] for name in names]
        return report
    pairs = {(a, b): PairStats(a, b, num_players) for a, b in itertools.combinations(names, 2)}
    for record in records:
        present = sorted(set(record.lineup), key=names.index)
        if len(present) != 2:
            raise ValueError(f"summarize: lineup {record.lineup} is not a pair lineup")
        pairs[(present[0], present[1])].add(record)
    report.pairs = list(pairs.values())
    return report


def evaluate(
    models: dict[str, PolicyModel],
    env_fn: Callable[[], BaseEnv],
    num_matches: int = 100,
    num_envs: int = 8,
    deterministic: bool = False,
    seed: int | None = None,
) -> EvalReport:
    """Schedule, play and summarize an evaluation.

    ``num_matches`` is per pair (pairwise) or per agent (solo).
    """
    names = list(models)
    probe = env_fn()
    try:
        num_players = probe.num_players
    finally:
        probe.close()
    lineups = schedule_lineups(names, num_players, num_matches)
    records = play_matches(models, env_fn, lineups, num_envs=num_envs,
                           deterministic=deterministic, seed=seed)
    return summarize(records, names, num_players, num_matches, deterministic)


# ---------------------------------------------------------------------------
# Loading agents
# ---------------------------------------------------------------------------


def load_eval_model(path: str | Path, config: ColosseumConfig) -> PolicyModel:
    """Build a model and load its weights for evaluation.

    ``path`` is either
    - a checkpoint directory ``<run>/checkpoints/<agent>/ckpt_v<N>/`` (read with
      ``load_checkpoint_dir``: strictly validated, never modified). If its
      ``meta.json`` has a ``networks`` section (the launcher writes it), the model
      is built from it, so agents with different architectures can be compared;
      that architecture is checked with ``validate_config`` against ``config.env``.
      Otherwise the model is built from ``config.networks``;
    - a ``.pt`` file with a plain ``state_dict`` (e.g. ``colosseum bc`` output),
      built from ``config.networks``.

    A malformed checkpoint, invalid ``networks``, an unreadable ``.pt`` or weights
    that do not fit the built model raise ConfigError; a path that is neither a
    directory nor a ``.pt`` file raises FileNotFoundError.
    """
    p = Path(path)
    model_config = config
    if p.is_dir():
        loaded = load_checkpoint_dir(p)
        networks = loaded["meta"].get("networks")
        if networks is not None:
            try:
                net_cfg = NetworkConfig.model_validate(networks)
            except ValidationError as e:
                raise ConfigError(f"Checkpoint {p}: invalid meta.json networks:\n{e}") from e
            model_config = config.model_copy(update={"networks": net_cfg})
            try:
                validate_config(model_config)
            except ConfigError as e:
                raise ConfigError(f"Checkpoint {p}: meta.json networks: {e}") from e
        model_state = loaded["model_state"]
    elif p.is_file() and p.suffix == ".pt":
        try:
            raw = torch.load(p, map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001 - any unpickling failure means unusable weights
            raise ConfigError(f"{p}: cannot load weights ({type(e).__name__}: {e})") from e
        if not isinstance(raw, dict) or not all(isinstance(v, torch.Tensor) for v in raw.values()):
            raise ConfigError(f"{p}: expected a state_dict of tensors")
        model_state = state_dict_to_numpy(raw)
    else:
        raise FileNotFoundError(f"{p}: expected a checkpoint directory or a .pt file")
    model = build_model(model_config)
    check_model_state(model, model_state, str(p))
    model.load_state_dict(state_dict_from_numpy(model_state))
    model.eval()
    return model
