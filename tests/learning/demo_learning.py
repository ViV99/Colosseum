"""Support for the SP2 learning tests (T8.3) and ``scripts/units_experiment.py`` (T8.4).

- ``train_in_process``: the real ``RolloutLoop`` and ``APPO`` in one process, every chunk through
  ``to_payload``/``from_payload`` (fast tests);
- ``GreedyPolicy`` / ``ScriptedPolicy``: wrap a model's mode, or a numpy per-observation function,
  as a point-mass distribution. A greedy agent and the stochastic ``RandomPolicy`` can then meet in
  one ``play_lineups(..., deterministic=False)`` (``deterministic=True`` would make the random
  player always pick its first legal action);
- ``train_example`` / ``load_agent``: ``colosseum train`` on an example config (through
  ``cli_runner`` without its ``TINY`` settings), then the newest checkpoint of an agent, read-only
  through ``load_eval_model``;
- ``won`` / ``win_rate`` / ``mean_team_score``: outcome helpers over ``MatchResult``.
"""
from __future__ import annotations

import functools
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cli_runner
from cli_runner import REPO_ROOT, run_train
from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig, ColosseumConfig, load_config
from colosseum.core.registry import env_spec, make_env
from colosseum.core.run_dir import RESOLVED_CONFIG_FILE
from colosseum.core.specs import ActionGroup, ActionSpec
from colosseum.core.tree import tree_get, tree_index, tree_leaves, tree_map, tree_stack, tree_to_numpy, tree_to_torch
from colosseum.core.types import Lineup, MatchResult, SeatAssignment, TrajectoryChunk, WeightPayload
from colosseum.envs.game import MultiAgentEnv, RoleSpec
from colosseum.eval import load_eval_model, play_lineups
from colosseum.networks.heads import make_distribution
from colosseum.networks.model import PolicyModel, PolicyStep
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop
from game_helpers import RandomPolicy

# Seed of the slow tests; the calibration in T8.3 reruns them with 1 and 2.
LEARNING_SEED = int(os.environ.get("COLOSSEUM_LEARNING_SEED", "0"))
EVAL_SEED = 12345
POINT_LOGIT = 1.0e4          # logit gap of a point-mass categorical
POINT_LOG_STD = -20.0        # log std of a point-mass Gaussian
_CKPT_RE = re.compile(r"ckpt_v(\d+)")


# ----- models ---------------------------------------------------------------------------------
def random_model(role: RoleSpec) -> PolicyModel:
    """The uniformly random legal player of the SP2 tests."""
    return RandomPolicy(role)


def _point_logits(action: torch.Tensor, n: int) -> torch.Tensor:
    return (F.one_hot(action.long(), n).to(torch.float32) - 1.0) * POINT_LOGIT    # 0 at the action, -1e4 elsewhere


def _point_box(action: torch.Tensor, dim: int) -> dict[str, torch.Tensor]:
    return {"mean": action.to(torch.float32), "log_std": torch.full((dim,), POINT_LOG_STD, device=action.device)}


def _point_units(group: ActionGroup, action: Any) -> dict[str, Any]:
    units, params = group.units, {}
    for i, comp in enumerate(units.components):
        if units.per_unit_kind == "dict":
            a = action[comp.name]
        elif units.per_unit_kind == "multi_discrete":
            a = action[..., i]
        else:
            a = action
        params[comp.name] = _point_logits(a, comp.size) if comp.kind == "discrete" else _point_box(a, comp.size)
    return params


def point_params(spec: ActionSpec, actions: Any) -> Any:
    """``make_distribution`` params that put (numerically) all mass on ``actions`` (a torch tree, batch first)."""
    out: dict[str, Any] = {}
    for g in spec.groups:
        a = tree_get(actions, g.path) if g.path else actions
        if g.kind == "discrete":
            p = _point_logits(a, g.nvec[0])
        elif g.kind == "multi_discrete":
            p = torch.cat([_point_logits(a[:, i], n) for i, n in enumerate(g.nvec)], dim=-1)
        elif g.kind == "box":
            p = _point_box(a, g.box_dim)
        else:
            p = _point_units(g, a)
        if not spec.is_dict:
            return p
        node = out
        for key in g.path[:-1]:
            node = node.setdefault(key, {})
        node[g.path[-1]] = p
    return out


def _point_dist(spec: ActionSpec, actions: Any, action_mask: Any):
    dist = make_distribution(spec, point_params(spec, actions))
    return dist if action_mask is None else dist.apply_mask(action_mask)


class GreedyPolicy(PolicyModel):
    """Plays the mode of ``inner`` (with ``inner``'s state) as a point-mass distribution."""

    def __init__(self, inner: PolicyModel, action_spec: ActionSpec) -> None:
        super().__init__()
        self.inner = inner
        self.action_spec = action_spec

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu"):
        return self.inner.initial_state(batch_size, device)

    def reset_state(self, state, done):
        return self.inner.reset_state(state, done)

    @property
    def is_stateful(self) -> bool:
        return self.inner.is_stateful

    def step(self, obs, state, action_mask=None) -> PolicyStep:
        out = self.inner.step(obs, state, action_mask)
        return PolicyStep(_point_dist(self.action_spec, out.dist.mode(), action_mask), out.state)

    def unroll(self, *args, **kwargs):
        raise NotImplementedError("GreedyPolicy is for evaluation only")


class ScriptedPolicy(PolicyModel):
    """A numpy function ``fn(obs) -> action`` (one observation, env action format) as a stateless model."""

    def __init__(self, fn: Callable[[Any], Any], action_spec: ActionSpec) -> None:
        super().__init__()
        self.fn = fn
        self.action_spec = action_spec

    def step(self, obs, state, action_mask=None) -> PolicyStep:
        np_obs = tree_to_numpy(obs)
        batch = tree_leaves(np_obs)[0].shape[0]
        actions = [tree_map(np.asarray, self.fn(tree_index(np_obs, b))) for b in range(batch)]
        return PolicyStep(_point_dist(self.action_spec, tree_to_torch(tree_stack(actions)), action_mask), state)

    def unroll(self, *args, **kwargs):
        raise NotImplementedError("ScriptedPolicy is for evaluation only")


# ----- in-process training (fast tests) -------------------------------------------------------
@dataclass
class AgentSetup:
    roles: list[str]
    model_fn: Callable[[], PolicyModel]
    config: AlgorithmConfig
    action_spec: ActionSpec


def train_in_process(*, env_fn: Callable[[], MultiAgentEnv], agents: Mapping[str, AgentSetup],
                     lineups: Sequence[Lineup], chunk_length: int = 16, batch_chunks: int = 4,
                     max_updates: int = 300, solved: Callable[[dict[str, PolicyModel]], bool],
                     check_every: int = 10, seed: int = 0) -> int:
    """Collect with one ``RolloutLoop`` (one env per lineup) and train one ``APPO`` per agent.
    Every update trains every agent on exactly ``batch_chunks`` of its chunks. Returns the number
    of updates after which ``solved(models)`` first held (checked every ``check_every``), or -1."""
    torch.manual_seed(seed)
    algos = {a: APPO(s.model_fn(), s.config, s.action_spec, device="cpu") for a, s in agents.items()}
    pending: dict[str, list[TrajectoryChunk]] = {a: [] for a in agents}
    latest = {a: WeightPayload.from_model(a, algo.policy_version, algo.model) for a, algo in algos.items()}
    io = LoopIO(send_chunk=lambda c: pending[c.agent_id].append(TrajectoryChunk.from_payload(c.to_payload())),
                poll_weights=lambda agent_id: latest[agent_id])
    loop = RolloutLoop(worker_id=0, env_fn=env_fn, num_envs=len(lineups), chunk_length=chunk_length,
                       agent_ids=list(agents), agent_roles={a: s.roles for a, s in agents.items()},
                       model_factories={a: s.model_fn for a, s in agents.items()}, io=io, lineups=list(lineups),
                       weight_sync_interval=0.0, seed=seed)
    try:
        for update in range(1, max_updates + 1):
            while any(len(chunks) < batch_chunks for chunks in pending.values()):
                loop.step()
            for agent_id, algo in algos.items():
                batch = pending[agent_id][:batch_chunks]
                del pending[agent_id][:batch_chunks]
                algo.set_progress(update / max_updates)
                algo.train_step(batch)
                latest[agent_id] = WeightPayload.from_model(agent_id, algo.policy_version, algo.model)
            loop.sync_weights()
            if update % check_every == 0 and solved({a: algo.model for a, algo in algos.items()}):
                return update
    finally:
        loop.close()
    return -1


# ----- CLI training and checkpoints (slow tests, units experiment) ---------------------------
@dataclass
class TrainedRun:
    root: Path
    config: ColosseumConfig
    elapsed: float

    @classmethod
    def open(cls, root: Path) -> TrainedRun:
        return cls(Path(root), load_config(Path(root) / RESOLVED_CONFIG_FILE), 0.0)

    def env_fn(self) -> Callable[[], MultiAgentEnv]:
        return functools.partial(make_env, self.config)

    def records(self, kind: str) -> list[dict]:
        lines = (self.root / "metrics.jsonl").read_text().splitlines()
        return [r for r in map(json.loads, filter(str.strip, lines)) if r["kind"] == kind]

    def ratings(self) -> dict:
        """``ratings.json``: ``{"env_steps": N, "layouts": {<layout>: {...}}}``."""
        return json.loads((self.root / "ratings.json").read_text())

    def env_steps_per_sec(self) -> float:
        system = self.records("system")
        if len(system) < 2 or system[-1]["ts"] <= system[0]["ts"]:
            return float("nan")
        return (system[-1]["env_steps"] - system[0]["env_steps"]) / (system[-1]["ts"] - system[0]["ts"])


def train_cmd(config: Path, run_parent: Path, name: str, sets: Mapping[str, Any]) -> list[str]:
    """``cli_runner.train_cmd`` without the ``TINY`` settings: the config's own budget and sizes."""
    return cli_runner.train_cmd(config, run_parent, name, {k: str(v) for k, v in sets.items()}, tiny=False)


def train_example(name: str, tmp_path: Path, sets: Mapping[str, Any] | None = None,
                  timeout: float = 900.0) -> TrainedRun:
    """``colosseum train -c configs/examples/<name>.yaml`` with ``training.seed=LEARNING_SEED``."""
    config = REPO_ROOT / "configs" / "examples" / f"{name}.yaml"
    overrides = {k: str(v) for k, v in {"training.seed": LEARNING_SEED, **(sets or {})}.items()}
    start = time.monotonic()
    proc = run_train(config, tmp_path, name, overrides, timeout=timeout, tiny=False)
    elapsed = time.monotonic() - start
    assert proc.returncode == 0, proc.stderr[-3000:]
    run = TrainedRun.open(proc.root)
    run.elapsed = elapsed
    return run


def newest_checkpoint(root: Path, agent_id: str) -> Path:
    agent_dir = Path(root) / "checkpoints" / agent_id
    versions = {int(m.group(1)): d for d in agent_dir.iterdir()
                if d.is_dir() and (m := _CKPT_RE.fullmatch(d.name))} if agent_dir.is_dir() else {}
    assert versions, f"no checkpoints for {agent_id} in {root}"
    return versions[max(versions)]


def load_agent(run: TrainedRun, agent_id: str) -> tuple[PolicyModel, RoleSpec, ActionSpec]:
    """The newest checkpoint of ``agent_id`` (architecture and roles from its ``meta.json``)."""
    model, roles = load_eval_model(run.config, agent_id, newest_checkpoint(run.root, agent_id))
    model.eval()
    role = env_spec(run.config).roles[roles[0]]
    return model, role, ActionSpec.from_space(role.action_space)


def greedy_agent(run: TrainedRun, agent_id: str) -> tuple[GreedyPolicy, RoleSpec]:
    model, role, action_spec = load_agent(run, agent_id)
    return GreedyPolicy(model, action_spec), role


# ----- evaluation -----------------------------------------------------------------------------
def play(env_fn: Callable[[], MultiAgentEnv], models: Mapping[str, PolicyModel],
         lineups: Sequence[Lineup]) -> list[MatchResult]:
    return play_lineups(env_fn=env_fn, models=models, lineups=lineups, num_envs=8, seed=EVAL_SEED)


def rotating_lineups(layout: str, num_seats: int, agent: str, others: str, num_matches: int) -> list[Lineup]:
    """``agent`` in seat ``m % num_seats`` of match ``m``, ``others`` in every other seat (FFA, 1v1)."""
    return [Lineup(layout, [SeatAssignment(agent if s == m % num_seats else others) for s in range(num_seats)])
            for m in range(num_matches)]


def team_lineups(team_seats: Sequence[Sequence[int]], layout: str, agent: str, others: str,
                 num_matches: int) -> list[Lineup]:
    """``agent`` in every seat of team ``m % T`` of match ``m``, ``others`` in the other teams."""
    num_seats = sum(len(t) for t in team_seats)
    lineups = []
    for m in range(num_matches):
        mine = set(team_seats[m % len(team_seats)])
        lineups.append(Lineup(layout, [SeatAssignment(agent if s in mine else others) for s in range(num_seats)]))
    return lineups


def team_of(result: MatchResult, agent: str) -> int:
    teams = {s.team for s in result.seats if s.agent_id == agent}
    assert len(teams) == 1, f"{agent} plays teams {teams} in {result.match_id}"
    return teams.pop()


def won(result: MatchResult, agent: str) -> bool:
    """``agent``'s team is strictly first (a win in ``wdl``, a sole first place in ``rank``)."""
    ranks = {t.team: t.rank for t in result.teams}
    mine = team_of(result, agent)
    return all(ranks[mine] < r for team, r in ranks.items() if team != mine)


def win_rate(results: Sequence[MatchResult], agent: str) -> float:
    return sum(won(r, agent) for r in results) / len(results)


def mean_team_score(results: Sequence[MatchResult], agent: str) -> float:
    scores = [next(t.score for t in r.teams if t.team == team_of(r, agent)) for r in results]
    return float(np.mean(scores))
