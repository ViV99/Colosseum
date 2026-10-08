"""Evaluator: inference-only matchups between agents/checkpoints.

No training, just plays matches and collects win rates with
confidence intervals. Useful for comparing different agents or
checkpoints after training.

Supports N-player games: agents are assigned to player slots round-robin.
Pairwise results are extracted from N-player match outcomes.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import math
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from colosseum.core.outcomes import player_outcomes
from colosseum.core.seat_info import acting_flags, check_masks, extract_masks
from colosseum.envs.base_env import BaseEnv
from colosseum.envs.vec_env import VectorEnv
from colosseum.networks.model import PolicyModel, act
from colosseum.networks.state import State, cat_batch, slice_batch

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Results from evaluating two agents against each other."""

    agent_a: str
    agent_b: str
    num_matches: int
    wins_a: int
    wins_b: int
    draws: int
    win_rate_a: float
    win_rate_b: float
    ci_lower_a: float  # 95% confidence interval lower bound
    ci_upper_a: float  # 95% confidence interval upper bound
    avg_reward_a: float
    avg_reward_b: float
    avg_episode_length: float


@dataclass
class EvalMatrix:
    """Full evaluation matrix across all agent pairs."""

    agent_ids: list[str]
    results: dict[tuple[str, str], EvalResult] = field(default_factory=dict)

    def get(self, a: str, b: str) -> EvalResult | None:
        return self.results.get((a, b))

    def summary(self) -> str:
        """Format results as a table."""
        lines = []
        header = "Agent A vs Agent B | Matches | Win A | Win B | Draw | WR A (95% CI)"
        lines.append(header)
        lines.append("-" * len(header))
        for (a, b), r in sorted(self.results.items()):
            lines.append(
                f"{a:>10} vs {b:<10} | {r.num_matches:>7} | "
                f"{r.wins_a:>5} | {r.wins_b:>5} | {r.draws:>4} | "
                f"{r.win_rate_a:.3f} [{r.ci_lower_a:.3f}, {r.ci_upper_a:.3f}]"
            )
        return "\n".join(lines)


def _wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return 0.0, 1.0
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def evaluate_agents(
    agent_configs: dict[str, dict],
    env_fn: Callable[[], BaseEnv],
    model_factory: Callable[[], PolicyModel],
    num_matches: int = 100,
    num_envs: int = 8,
    model_factories: dict[str, Callable[[], PolicyModel]] | None = None,
    deterministic: bool = False,
) -> EvalMatrix:
    """Run evaluation matches between all agents.

    All agents play together in N-player matches (round-robin slot
    assignment). Pairwise results are extracted from match outcomes.

    Args:
        agent_configs: {agent_id: {"state_dict": dict}} — one entry per agent.
        env_fn: Factory to create a BaseEnv instance.
        model_factory: Default factory to create a PolicyModel.
        num_matches: Number of matches to play.
        num_envs: Number of parallel environments for evaluation.
        model_factories: Optional per-agent model factories. If provided,
            overrides ``model_factory`` for the corresponding agent.

    Returns:
        EvalMatrix with results for all pairs.
    """
    agent_ids = list(agent_configs.keys())
    matrix = EvalMatrix(agent_ids=agent_ids)

    # Load all models
    agents: list[tuple[str, PolicyModel]] = []
    for aid in agent_ids:
        cfg = agent_configs[aid]
        factory = (
            model_factories[aid]
            if model_factories and aid in model_factories
            else model_factory
        )
        net = factory()
        net.load_state_dict(cfg["state_dict"])
        net.eval()
        agents.append((aid, net))

    # Run matches with all agents
    pairwise_results = _run_matches(
        agents, env_fn, num_matches, num_envs, deterministic=deterministic,
    )

    for (a, b), result in pairwise_results.items():
        matrix.results[(a, b)] = result
        # Also store reversed
        matrix.results[(b, a)] = EvalResult(
            agent_a=b, agent_b=a, num_matches=result.num_matches,
            wins_a=result.wins_b, wins_b=result.wins_a, draws=result.draws,
            win_rate_a=result.win_rate_b, win_rate_b=result.win_rate_a,
            ci_lower_a=1 - result.ci_upper_a, ci_upper_a=1 - result.ci_lower_a,
            avg_reward_a=result.avg_reward_b, avg_reward_b=result.avg_reward_a,
            avg_episode_length=result.avg_episode_length,
        )

    return matrix


def _eval_where(e: int, p: int) -> str:
    """Error-message context for eval env ``e``, seat ``p``."""
    return f"eval: env {e} seat {p}"


def _run_matches(
    agents: list[tuple[str, PolicyModel]],
    env_fn: Callable[[], BaseEnv],
    num_matches: int,
    num_envs: int,
    deterministic: bool = False,
) -> dict[tuple[str, str], EvalResult]:
    """Run N-player matches with agents rotated across player slots.

    Unlike a fixed slot assignment (which would leave agents beyond
    ``num_players`` never playing), each env re-rolls its slot→agent assignment
    at the start of every match. When ``num_agents >= num_players`` each match
    uses a distinct random subset of agents; otherwise agents are repeated to
    fill the slots. Pairwise statistics are accumulated only over agents that
    actually co-occur in a match, so every pair eventually gathers samples.

    Only acting slots (``info["active"]``, default True) run inference, with
    the worker's mask rules (``core/seat_info.py``), and match outcomes prefer
    the env's authoritative signal (``info["rank"]`` / ``info["outcome"]``)
    over cumulative reward. Every (env, slot) keeps the model state of the
    agent playing it; the state is reset when the match ends.

    Args:
        agents: List of (agent_id, model) tuples.
        env_fn: Factory to create a BaseEnv instance.
        num_matches: Total matches to play.
        num_envs: Number of parallel environments.
        deterministic: If True, agents act greedily (distribution mode).

    Returns:
        dict mapping (agent_a, agent_b) -> EvalResult for each ordered pair (i < j).
    """
    import random

    vec_env = VectorEnv(env_fn, min(num_envs, num_matches))
    actual_envs = vec_env.num_envs
    num_players = vec_env.num_players
    action_spec = vec_env.action_spec
    num_agents = len(agents)

    # Pairwise tracking: keyed by (agent_idx_i, agent_idx_j) where i < j
    pair_wins_i: dict[tuple[int, int], int] = {}
    pair_wins_j: dict[tuple[int, int], int] = {}
    pair_draws: dict[tuple[int, int], int] = {}
    pair_reward_i: dict[tuple[int, int], float] = {}
    pair_reward_j: dict[tuple[int, int], float] = {}

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            pair_wins_i[(i, j)] = 0
            pair_wins_j[(i, j)] = 0
            pair_draws[(i, j)] = 0
            pair_reward_i[(i, j)] = 0.0
            pair_reward_j[(i, j)] = 0.0

    def _roll_assignment() -> list[int]:
        """Assign an agent index to each player slot for one match."""
        if num_agents >= num_players:
            return random.sample(range(num_agents), num_players)
        base = [k % num_agents for k in range(num_players)]
        random.shuffle(base)
        return base

    # Per-env slot→agent assignment, re-rolled on each match boundary.
    slot_assign: list[list[int]] = [_roll_assignment() for _ in range(actual_envs)]

    def _fresh_state(e: int, p: int) -> State:
        return agents[slot_assign[e][p]][1].initial_state(1)

    slot_states: dict[tuple[int, int], State] = {
        (e, p): _fresh_state(e, p) for e in range(actual_envs) for p in range(num_players)
    }

    matches_done = 0
    total_ep_length = 0

    obs, infos = vec_env.reset_all()
    ep_rewards = np.zeros((actual_envs, num_players))
    ep_lengths = np.zeros(actual_envs, dtype=int)

    while matches_done < num_matches:
        actions_np = np.zeros(
            (actual_envs, num_players, *action_spec.action_shape),
            dtype=action_spec.numpy_dtype,
        )
        acting = acting_flags(infos, actual_envs, num_players)
        masks = extract_masks(infos, actual_envs, num_players, action_spec)
        if masks is not None:
            check_masks(masks, acting, action_spec, _eval_where)
        obs_flat = obs.reshape(-1, *obs.shape[2:])

        # Group acting (env, slot) pairs by their assigned agent for batched inference.
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for e in range(actual_envs):
            for p in range(num_players):
                if acting[e, p]:
                    groups[slot_assign[e][p]].append((e, p))

        for agent_idx, slot_pairs in groups.items():
            model = agents[agent_idx][1]
            flat_idxs = [e * num_players + p for (e, p) in slot_pairs]
            obs_batch = torch.tensor(obs_flat[flat_idxs], dtype=torch.float32)
            mask_batch = None
            if masks is not None:
                mask_batch = torch.tensor(masks[flat_idxs], dtype=torch.bool)
            state = cat_batch([slot_states[(e, p)] for (e, p) in slot_pairs])
            with torch.no_grad():
                out = act(model, obs_batch, state, mask_batch, deterministic=deterministic)
            actions_arr = out.actions.numpy()
            for k, (e, p) in enumerate(slot_pairs):
                actions_np[e, p] = actions_arr[k]
                if out.state is not None:
                    slot_states[(e, p)] = slice_batch(out.state, k)

        next_obs, rewards, terminated, truncated, infos = vec_env.step(actions_np)

        for env_idx in range(actual_envs):
            if matches_done >= num_matches:
                break

            for p in range(num_players):
                ep_rewards[env_idx, p] += rewards[env_idx, p]
            ep_lengths[env_idx] += 1

            done = terminated[env_idx] or truncated[env_idx]
            if done:
                # Per-slot outcomes from env signal (rank/outcome) or reward.
                term_infos = {
                    p: infos[env_idx][p].get("terminal_info", {})
                    for p in range(num_players)
                }
                slot_outcomes = player_outcomes(
                    ep_rewards[env_idx], term_infos, num_players,
                )

                # Aggregate to the agents present in this match.
                agent_score: dict[int, list[float]] = defaultdict(list)
                agent_rewards: dict[int, float] = defaultdict(float)
                for p in range(num_players):
                    ai = slot_assign[env_idx][p]
                    agent_score[ai].append(slot_outcomes[p])
                    agent_rewards[ai] += ep_rewards[env_idx, p]
                score = {ai: sum(v) / len(v) for ai, v in agent_score.items()}

                present = sorted(score.keys())
                for a_pos in range(len(present)):
                    for b_pos in range(a_pos + 1, len(present)):
                        ia, ib = present[a_pos], present[b_pos]
                        i, j = (ia, ib) if ia < ib else (ib, ia)
                        si, sj = score[i], score[j]
                        pair_reward_i[(i, j)] += agent_rewards[i]
                        pair_reward_j[(i, j)] += agent_rewards[j]
                        if si > sj:
                            pair_wins_i[(i, j)] += 1
                        elif sj > si:
                            pair_wins_j[(i, j)] += 1
                        else:
                            pair_draws[(i, j)] += 1

                total_ep_length += ep_lengths[env_idx]
                matches_done += 1
                ep_rewards[env_idx] = 0.0
                ep_lengths[env_idx] = 0
                slot_assign[env_idx] = _roll_assignment()
                for p in range(num_players):
                    slot_states[(env_idx, p)] = _fresh_state(env_idx, p)

        obs = next_obs

    vec_env.close()

    # Build EvalResult for each pair
    results: dict[tuple[str, str], EvalResult] = {}
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            aid_i = agents[i][0]
            aid_j = agents[j][0]
            wi = pair_wins_i[(i, j)]
            wj = pair_wins_j[(i, j)]
            d = pair_draws[(i, j)]
            total = wi + wj + d
            wr_i = wi / max(1, total)
            wr_j = wj / max(1, total)
            ci_lo, ci_hi = _wilson_ci(wi, total)

            results[(aid_i, aid_j)] = EvalResult(
                agent_a=aid_i, agent_b=aid_j,
                num_matches=total,
                wins_a=wi, wins_b=wj, draws=d,
                win_rate_a=wr_i, win_rate_b=wr_j,
                ci_lower_a=ci_lo, ci_upper_a=ci_hi,
                avg_reward_a=pair_reward_i[(i, j)] / max(1, total),
                avg_reward_b=pair_reward_j[(i, j)] / max(1, total),
                avg_episode_length=total_ep_length / max(1, matches_done),
            )

    return results


# ---------------------------------------------------------------------------
# Engine (SP1 block 7): per-seat State, seat rotation, active-only inference
# ---------------------------------------------------------------------------


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
