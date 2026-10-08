"""Toy envs, probe models and queue helpers for data-flow / transition tests.

Imported by bare name (``from dataflow_helpers import ...``; ``tests/`` is on
``sys.path``, see ``conftest.py``), so spawned processes can unpickle the env
and model factories defined here.
"""

from __future__ import annotations

import os
import queue
from collections.abc import Callable
from dataclasses import dataclass, field

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.core.ipc import assert_no_tensors, put_latest
from colosseum.core.types import MatchResult, SeatResult, TrajectoryChunk, WeightPayload, WorkerCommand
from colosseum.envs.base_env import BaseEnv
from colosseum.networks.distributions import CategoricalDist
from colosseum.networks.model import PolicyModel, StepOutput
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop
from helpers import TinyMonolithicModel

OBS_DIM = 4
NUM_ACTIONS = 3


class TinyModel(TinyMonolithicModel):
    """Small stateless trainable model (linear policy + value) sized for the envs below."""

    def __init__(self, obs_dim: int = OBS_DIM, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__(obs_dim=obs_dim, num_actions=num_actions)


def make_tiny_model() -> TinyModel:
    return TinyModel()


class _Base(BaseEnv):
    """Shared plumbing: obs[p] = [env_id, ep, t, p], Discrete(NUM_ACTIONS)."""

    NUM_PLAYERS = 2

    def __init__(self, env_id: int = 0) -> None:
        self.env_id = env_id
        self.ep = -1
        self.t = 0
        self.log: list[dict] = []

    @property
    def num_players(self) -> int:
        return self.NUM_PLAYERS

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(-1e9, 1e9, (OBS_DIM,), np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(NUM_ACTIONS)

    def _obs(self) -> dict[int, np.ndarray]:
        return {p: np.array([self.env_id, self.ep, self.t, p], np.float32)
                for p in range(self.num_players)}


class ThreadProbeEnv(_Base):
    """1-player env whose observation reports the process's thread settings.

    obs = [torch.get_num_threads(), OMP_NUM_THREADS (or -1), torch.get_num_interop_threads(), t];
    4 steps per episode.
    """

    NUM_PLAYERS = 1

    def _probe(self) -> dict[int, np.ndarray]:
        omp = float(os.environ.get("OMP_NUM_THREADS", "-1"))
        return {0: np.array([torch.get_num_threads(), omp, torch.get_num_interop_threads(), self.t],
                            np.float32)}

    def reset(self, seed=None):
        self.ep += 1
        self.t = 0
        return self._probe(), {0: {}}

    def step(self, actions):
        self.t += 1
        done = self.t >= 4
        return self._probe(), {0: 0.0}, {0: done}, {0: False}, {0: {}}


class CheckedQueue(queue.Queue):
    """queue.Queue that rejects any item containing a torch.Tensor."""

    def put(self, item, block=True, timeout=None):
        assert_no_tensors(item, "queued item")
        super().put(item, block, timeout)

    def put_nowait(self, item):
        self.put(item, block=False)

    def cancel_join_thread(self) -> None:
        pass


def chunk_payload(T: int = 4, version: int = 0, agent_id: str = "a") -> dict:
    """A valid chunk payload with random content (for learner-loop tests only)."""
    rng = np.random.default_rng(version)
    return {
        "agent_id": agent_id,
        "observations": rng.standard_normal((T, OBS_DIM)).astype(np.float32),
        "actions": rng.integers(0, NUM_ACTIONS, T).astype(np.int64),
        "action_log_probs": np.full(T, -np.log(NUM_ACTIONS), np.float32),
        "rewards": np.zeros(T, np.float32),
        "dones": np.zeros(T, bool),
        "values": np.zeros(T, np.float32),
        "bootstrap_value": 0.0,
        "behavior_policy_version": int(version),
        "initial_state": None,
        "action_masks": None,
    }


class GridStepEnv(_Base):
    """Simultaneous-move env; every slot acts every step.

    obs[p] = [env_id, ep, t, p]; reward[p] = ep + 0.01 * t + 0.1 * p (or
    ``const_reward``). Episode ``ep`` lasts ``lengths[ep % len(lengths)]`` steps
    and ends with truncation if ``truncate_every`` and ``ep % truncate_every == 0``,
    else with termination. Every step is appended to ``self.log``.
    """

    def __init__(self, env_id: int = 0, lengths=(3,), truncate_every: int = 0,
                 const_reward: float | None = None, num_players: int = 2) -> None:
        super().__init__(env_id)
        self.lengths = tuple(lengths)
        self.truncate_every = truncate_every
        self.const_reward = const_reward
        self.NUM_PLAYERS = num_players

    def _length(self) -> int:
        return self.lengths[self.ep % len(self.lengths)]

    def reset(self, seed=None):
        self.ep += 1
        self.t = 0
        return self._obs(), {p: {} for p in range(self.num_players)}

    def step(self, actions):
        P = self.num_players
        pre_obs = self._obs()
        rew = {p: (self.const_reward if self.const_reward is not None
                   else self.ep + 0.01 * self.t + 0.1 * p) for p in range(P)}
        self.t += 1
        done = self.t >= self._length()
        trunc = done and self.truncate_every > 0 and self.ep % self.truncate_every == 0
        term = done and not trunc
        self.log.append({"ep": self.ep, "t": self.t - 1, "obs": pre_obs,
                         "actions": {p: int(actions[p]) for p in range(P)},
                         "active": {p: True for p in range(P)},
                         "rewards": rew, "done": done, "truncated": trunc})
        return (self._obs(), rew, {p: term for p in range(P)},
                {p: trunc for p in range(P)}, {p: {} for p in range(P)})


class EnvFactory:
    """Callable env factory: assigns env ids 0, 1, ... and keeps the instances."""

    def __init__(self, cls, **kwargs) -> None:
        self.cls = cls
        self.kwargs = kwargs
        self.created: list[BaseEnv] = []

    def __call__(self) -> BaseEnv:
        env = self.cls(env_id=len(self.created), **self.kwargs)
        self.created.append(env)
        return env


def publish_versions(q, n: int, done) -> None:
    """Spawn target: publish WeightPayload v1..vn into a size-1 mailbox as fast as possible."""
    for version in range(1, n + 1):
        assert put_latest(q, WeightPayload("a", version, {"w": np.full(64, version, dtype=np.float32)}))
    done.set()


class RecordingAlgorithm:
    """Minimal duck-typed algorithm for learner-loop tests: records what it was given.

    ``train_step`` stores the batch size, the chunks' behavior versions and the
    last progress value, then bumps ``policy_version``.
    """

    def __init__(self, start_version: int = 0) -> None:
        self._model = TinyModel()
        self._policy_version = start_version
        self._progress = 0.0
        self.batches: list[int] = []
        self.behavior_versions: list[list[int]] = []
        self.progress_at_train: list[float] = []

    @property
    def model(self) -> TinyModel:
        return self._model

    @property
    def network(self) -> TinyModel:
        return self._model

    @property
    def policy_version(self) -> int:
        return self._policy_version

    @property
    def is_off_policy(self) -> bool:
        return False

    def create_replay_buffer(self, capacity: int):
        return None

    def set_progress(self, progress: float) -> None:
        self._progress = float(progress)

    def compute_loss(self, chunks):
        return {}

    def train_step(self, chunks) -> dict[str, float]:
        self.batches.append(len(chunks))
        self.behavior_versions.append([c.behavior_policy_version for c in chunks])
        self.progress_at_train.append(self._progress)
        self._policy_version += 1
        return {"total_loss": 0.0}


@dataclass
class Collected:
    """In-memory LoopIO: everything the loop emits is appended to these lists.

    ``weights[agent_id]`` is a list of WeightPayloads handed out one per
    ``poll_weights`` call; ``commands`` are handed out one per ``poll_command``.
    """

    chunks: list[TrajectoryChunk] = field(default_factory=list)
    results: list[MatchResult] = field(default_factory=list)
    env_steps: list[int] = field(default_factory=list)
    weights: dict[str, list[WeightPayload]] = field(default_factory=dict)
    commands: list[WorkerCommand] = field(default_factory=list)

    def io(self) -> LoopIO:
        def poll_weights(agent_id: str) -> WeightPayload | None:
            pending = self.weights.get(agent_id)
            return pending.pop(0) if pending else None

        def poll_command() -> WorkerCommand | None:
            return self.commands.pop(0) if self.commands else None

        return LoopIO(send_chunk=self.chunks.append, poll_weights=poll_weights,
                      report_result=self.results.append, poll_command=poll_command,
                      add_env_steps=self.env_steps.append)


def make_loop(env_factory: Callable[[], BaseEnv],
              model_factory: Callable[[], PolicyModel] | dict[str, Callable[[], PolicyModel]],
              *, agent_ids=("a",), num_envs: int = 1, chunk_length: int = 4,
              collected: Collected | None = None, **kwargs) -> tuple[RolloutLoop, Collected]:
    """Build a RolloutLoop with in-memory I/O.

    ``model_factory`` is used for every agent, or is a dict ``{agent_id: factory}``.
    ``collected`` may be pre-filled (e.g. ``weights`` for the initial sync in
    ``RolloutLoop.__init__``); a fresh one is created otherwise. Other kwargs go
    to ``RolloutLoop``. Defaults: weight_sync_interval=0.0 (sync every step), seed=0.
    """
    col = collected if collected is not None else Collected()
    factories = (dict(model_factory) if isinstance(model_factory, dict)
                 else {a: model_factory for a in agent_ids})
    loop = RolloutLoop(
        worker_id=0, env_fn=env_factory, num_envs=num_envs, chunk_length=chunk_length,
        agent_ids=list(agent_ids), model_factories=factories,
        io=col.io(), weight_sync_interval=kwargs.pop("weight_sync_interval", 0.0),
        seed=kwargs.pop("seed", 0), **kwargs,
    )
    return loop, col


def run_until_chunks(loop: RolloutLoop, collected: Collected, n_chunks: int,
                     max_steps: int = 10_000) -> list[TrajectoryChunk]:
    """Step ``loop`` until ``collected`` holds at least ``n_chunks`` chunks; return the first ``n_chunks``.

    Raises ``AssertionError`` if ``max_steps`` loop steps are not enough.
    """
    for _ in range(max_steps):
        if len(collected.chunks) >= n_chunks:
            return collected.chunks[:n_chunks]
        loop.step()
    if len(collected.chunks) >= n_chunks:
        return collected.chunks[:n_chunks]
    raise AssertionError(f"RolloutLoop produced only {len(collected.chunks)} chunks after {max_steps} steps")


def add_to_counter(counter, n: int) -> None:
    """Spawn target: add 1 to a SharedCounter n times."""
    for _ in range(n):
        counter.add(1)


class ProbeModel(PolicyModel):
    """Deterministic-value test model.

    - logits are zeros (uniform over legal actions; the mask is applied);
    - ``value = obs @ obs_coeffs + state_coef * n`` where ``n`` counts this slot's
      ``step`` calls since its state was last reset (``stateful=True``; state
      ``{"n": [B, 1]}``), else 0;
    - every ``step`` call appends its batch size to ``self.calls``.
    """

    def __init__(self, obs_coeffs=(0.0, 0.0, 0.0, 0.0), state_coef: float = 0.0,
                 stateful: bool = False, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.register_buffer("obs_coeffs", torch.tensor(obs_coeffs, dtype=torch.float32))
        self.state_coef = float(state_coef)
        self.stateful = stateful
        self.num_actions = num_actions
        self.bias = nn.Parameter(torch.zeros(1))
        self.calls: list[int] = []

    def initial_state(self, batch_size: int, device="cpu"):
        if not self.stateful:
            return None
        return {"n": torch.zeros(batch_size, 1, device=device)}

    def step(self, obs, state, action_mask=None) -> StepOutput:
        self.calls.append(int(obs.shape[0]))
        batch = obs.shape[0]
        n = state["n"] if self.stateful else torch.zeros(batch, 1)
        value = obs.float() @ self.obs_coeffs + self.state_coef * n[:, 0] + self.bias * 0.0
        dist = CategoricalDist(torch.zeros(batch, self.num_actions), mask=action_mask)
        new_state = {"n": n + 1.0} if self.stateful else None
        return StepOutput(dist=dist, value=value, state=new_state)


def slot_transitions(chunks: list[TrajectoryChunk]) -> dict[tuple[int, int], list[dict]]:
    """Flatten chunks into per-(env_id, player) transition lists, in chunk order.

    Relies on obs = [env_id, ep, t, player] (the ``_Base`` envs).
    """
    out: dict[tuple[int, int], list[dict]] = {}
    for chunk in chunks:
        for i in range(chunk.chunk_length):
            o = chunk.observations[i].tolist()
            out.setdefault((int(o[0]), int(o[3])), []).append({
                "ep": int(o[1]), "t": int(o[2]), "action": int(chunk.actions[i]),
                "reward": float(chunk.rewards[i]), "done": bool(chunk.dones[i]),
                "value": float(chunk.values[i]), "log_prob": float(chunk.action_log_probs[i]),
            })
    return out


def reference_transitions(log: list[dict], num_players: int) -> dict[int, list[dict]]:
    """Straightforward reference: per player, one transition per acting step.

    A transition gets every reward from its own step until the player's next
    action (or the episode end); rewards before a player's first action in an
    episode go to that first transition; the last transition of each episode has
    ``done=True``. ``log`` is an env's step log (see ``GridStepEnv``).
    """
    out: dict[int, list[dict]] = {p: [] for p in range(num_players)}
    open_tr: dict[int, dict | None] = {p: None for p in range(num_players)}
    pending = {p: 0.0 for p in range(num_players)}
    for step in log:
        for p in range(num_players):
            if step["active"][p]:
                tr = {"ep": step["ep"], "t": step["t"], "action": step["actions"][p],
                      "reward": pending[p], "done": False}
                out[p].append(tr)
                open_tr[p] = tr
                pending[p] = 0.0
        for p in range(num_players):
            reward = float(step["rewards"][p])
            if open_tr[p] is not None:
                open_tr[p]["reward"] += reward
            else:
                pending[p] += reward
        if step["done"]:
            for p in range(num_players):
                if open_tr[p] is not None:
                    open_tr[p]["done"] = True
                open_tr[p] = None
                pending[p] = 0.0
    return out


class AlternatingWinEnv(_Base):
    """Turn-based 2-player env: player ``t % 2`` acts at step ``t``.

    - Episode ``ep`` lasts ``lengths[ep % len(lengths)]`` steps (default: 3 if
      ``ep`` is even, else 4). The player who makes the last move wins: +1 to the
      mover, -1 to the other.
    - In 4-step episodes player 1 gets +0.5 at step 0, before its first move.
    - After a step, the acting slot's mask is [True, True, False] and the
      non-acting slot's mask is all False. The reset info has only "active"
      (no mask: every action is legal at t=0). ``use_masks=False`` omits every
      mask (and the illegal-action check), so only "active" drives the worker.
    - The info returned on the terminal step is deliberately bogus (nobody
      active, all-false masks): the first decision of the next episode must use
      the reset info, otherwise the acting slot sees an empty mask (R2-14).
    - Raises if the mover plays action 2 when it is illegal (t > 0).
    """

    def __init__(self, env_id: int = 0, use_masks: bool = True, lengths=(3, 4)) -> None:
        super().__init__(env_id)
        self.use_masks = use_masks
        self.lengths = tuple(lengths)

    def reset(self, seed=None):
        self.ep += 1
        self.t = 0
        return self._obs(), {p: {"active": p == 0} for p in range(2)}

    def _length(self) -> int:
        return self.lengths[self.ep % len(self.lengths)]

    def _info(self) -> dict[int, dict]:
        mover = self.t % 2
        if not self.use_masks:
            return {p: {"active": p == mover} for p in range(2)}
        return {p: {"active": p == mover,
                    "action_mask": (np.array([True, True, False]) if p == mover
                                    else np.zeros(NUM_ACTIONS, dtype=bool))}
                for p in range(2)}

    def step(self, actions):
        mover = self.t % 2
        if self.use_masks and self.t > 0 and int(actions[mover]) == 2:
            raise AssertionError(f"illegal action from mover {mover} at t={self.t}")
        pre_obs = self._obs()
        rew = {0: 0.0, 1: 0.0}
        if self.t == 0 and self._length() == 4:
            rew[1] = 0.5
        self.t += 1
        done = self.t >= self._length()
        if done:
            rew = {mover: 1.0, 1 - mover: -1.0}
        self.log.append({"ep": self.ep, "t": self.t - 1, "obs": pre_obs,
                         "actions": {p: int(actions[p]) for p in range(2)},
                         "active": {p: p == mover for p in range(2)},
                         "rewards": rew, "done": done, "truncated": False})
        if done:
            info = {p: {"active": False, "action_mask": np.zeros(NUM_ACTIONS, dtype=bool),
                        "rank": 1 if p == mover else 2} for p in range(2)}
        else:
            info = self._info()
        return self._obs(), rew, {p: done for p in range(2)}, {p: False for p in range(2)}, info


class BadMaskEnv(AlternatingWinEnv):
    """Like AlternatingWinEnv, but at step 1 the ACTING slot gets an all-false mask."""

    def _info(self) -> dict[int, dict]:
        info = super()._info()
        if self.t == 1:
            info[1]["action_mask"] = np.zeros(NUM_ACTIONS, dtype=bool)
        return info


class InfoLeakEnv(_Base):
    """1-player env whose terminal step info carries stale keys the reset info lacks."""

    NUM_PLAYERS = 1

    def reset(self, seed=None):
        self.ep += 1
        self.t = 0
        return self._obs(), {0: {"active": True, "phase": "reset"}}

    def step(self, actions):
        self.t += 1
        done = self.t >= 2
        info = {0: {"active": False, "phase": "step",
                    "action_mask": np.array([False, True, False]),
                    "rank": 1, "outcome": 0.123}}
        return self._obs(), {0: 1.0}, {0: done}, {0: False}, info


class FFA4Env(_Base):
    """4-player simultaneous FFA, 2 steps per episode; seat p's episode reward is 3 - p.

    ``terminal_info(p)`` gives seat p's terminal info; the default reports rank p + 1.
    """

    NUM_PLAYERS = 4

    def __init__(self, env_id: int = 0, terminal_info: Callable[[int], dict] | None = None) -> None:
        super().__init__(env_id)
        self.terminal_info = terminal_info or (lambda p: {"rank": p + 1})

    def reset(self, seed=None):
        self.ep += 1
        self.t = 0
        return self._obs(), {p: {} for p in range(4)}

    def step(self, actions):
        self.t += 1
        done = self.t >= 2
        rew = {p: (float(3 - p) if done else 0.0) for p in range(4)}
        info = {p: (self.terminal_info(p) if done else {}) for p in range(4)}
        return self._obs(), rew, {p: done for p in range(4)}, {p: False for p in range(4)}, info


def two_seat_result(agent_a: str, outcome_a: float, agent_b: str, outcome_b: float,
                    network_b: str = "latest", match_id: str = "m") -> MatchResult:
    """A 2-seat MatchResult: seat 0 = agent_a (latest), seat 1 = agent_b (network_b)."""
    return MatchResult(match_id=match_id, episode_length=1, seats=[
        SeatResult(seat=0, agent_id=agent_a, network_id="latest", outcome=outcome_a, reward=outcome_a),
        SeatResult(seat=1, agent_id=agent_b, network_id=network_b, outcome=outcome_b, reward=outcome_b),
    ])
