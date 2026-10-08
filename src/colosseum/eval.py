"""Evaluator: inference-only matchups between agents/checkpoints.

No training, just plays matches and collects win rates with
confidence intervals. Useful for comparing different agents or
checkpoints after training.

Supports N-player games: agents are assigned to player slots round-robin.
Pairwise results are extracted from N-player match outcomes.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

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


def _extract_action_masks(
    infos: list[dict],
    num_envs: int,
    num_players: int,
    action_spec=None,
) -> np.ndarray | None:
    """Extract action masks from env info dicts into a flat array.

    Convention: info[env_idx][player_idx]["action_mask"] is a bool ndarray
    or dict of per-component bool ndarrays (for composite action spaces).
    Returns [num_envs * num_players, num_actions] bool array, or None if no masks.
    """
    if not infos:
        return None
    first_info = infos[0]
    if not isinstance(first_info, dict) or 0 not in first_info:
        return None
    if "action_mask" not in first_info[0]:
        return None

    masks = []
    for env_idx in range(num_envs):
        for p in range(num_players):
            raw = infos[env_idx][p]["action_mask"]
            if isinstance(raw, dict) and action_spec is not None:
                masks.append(action_spec.flatten_mask(raw))
            else:
                masks.append(raw)
    return np.array(masks, dtype=bool)


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

    Action masks (``info["action_mask"]``) are applied during inference, and
    match outcomes prefer the env's authoritative signal (``info["rank"]`` /
    ``info["outcome"]``) over cumulative reward. Every (env, slot) keeps the
    model state of the agent playing it; the state is reset when the match ends.

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
    from collections import defaultdict

    from colosseum.core.outcomes import player_outcomes

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
        masks = _extract_action_masks(
            infos, actual_envs, num_players, action_spec=action_spec,
        )
        obs_flat = obs.reshape(-1, *obs.shape[2:])

        # Group (env, slot) pairs by their assigned agent for batched inference.
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for e in range(actual_envs):
            for p in range(num_players):
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
