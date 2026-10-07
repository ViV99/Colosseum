# SP1 Plan — Part B: Data flow (block 2) and Worker transitions (block 3)

This part covers spec blocks 2 and 3 (tasks T2.1–T2.6, T3.1–T3.4). Block 2 makes the process boundary safe and predictable:
- torch thread limits;
- numpy-only payloads on every queue and over gRPC;
- newest-wins weight delivery;
- exact learner batches;
- a global env-step budget that also drives the LR schedule;
- agent-owned rollout buffers that are parked, not discarded, on match re-assignment.

Block 3 rewrites how the worker turns env steps into transitions:
- a transition stays open until its slot acts again;
- no separate bootstrap forward;
- turn-based inference and reward/`done` routing;
- truncation bootstrapping;
- per-seat match results.

**Read `00-overview.md` first.** Its global constraints, file map and interface contract apply to every task below.

**Starting point.** Part A (`01-tooling-and-model.md`, T0.1–T1.7) is done:
- `.venv` exists and the test layout is `tests/unit|contract|integration|learning`;
- `RolloutLoop`/`LoopIO` live in `src/colosseum/worker/rollout_loop.py`;
- models are `PolicyModel`s with an opaque `State`;
- `TrajectoryChunk.initial_state` replaced `lstm_hidden`.

Where a task relies on a Part A detail that the contract does not pin down, its **Consumes** block says so.

**Shared test helpers.** This part builds one helper module, `tests/dataflow_helpers.py`. Every task adds to it, and each addition is shown in full in that task. It holds:
- toy envs and probe models;
- an in-memory `LoopIO`;
- a queue that rejects tensors.

Tests import it as `from tests.dataflow_helpers import ...`. Two things make this work: `tests/conftest.py` keeps the project root on `sys.path`, and spawned children inherit `sys.path`. So spawned processes can unpickle env and model factories defined there. Imports always go into the import block at the top of the module (ruff E402). New code is appended at the end.

---

### Task T2.1: Torch thread limits (worker, subprocess env children, learner auto)

Spec block 2 "Потоки torch" (R2-04, R6-04). Every worker process sets `torch.set_num_threads(rollout.torch_threads)` and `torch.set_num_interop_threads(1)` as its first action. `SubprocessVectorEnv` children run with `OMP_NUM_THREADS=1` and one torch thread. Learners get an automatic thread count.

**Files:**
- Create: `src/colosseum/core/threads.py`
- Create: `tests/dataflow_helpers.py`
- Modify: `src/colosseum/core/config.py` (`RolloutConfig`, `LearnerConfig`)
- Replace: `src/colosseum/worker/rollout_worker.py` (whole file)
- Modify: `src/colosseum/envs/subproc_vec_env.py` (`_worker_loop`, `SubprocessVectorEnv.__init__`)
- Modify: `src/colosseum/launcher.py` (`_worker_target`, `_learner_target`, process creation in `Launcher.launch`)
- Modify: `src/colosseum/distributed.py` (`_dist_worker_target`, `run_distributed_learner`)
- Test: `tests/unit/test_threads.py`, `tests/integration/test_worker_threads.py`

**Interfaces:**
- Consumes:
  - `RolloutLoop(*, worker_id, env_fn, num_envs, chunk_length, agent_ids, model_factories, io, gamma, weight_sync_interval, slot_agent_map, slot_network_map, collect_mask, checkpoint_state_dicts_by_agent, seed, vec_env_kind, subproc_workers)`, `RolloutLoop.run(should_stop, max_env_steps)`, `RolloutLoop.close()`, `RolloutLoop.stats`, `LoopIO(...)` (contract, T0.5);
  - `PolicyModel`, `StepOutput` (`colosseum.networks.model`);
  - `CategoricalDist(logits, mask=None)`;
  - `build_model(config)` (`colosseum.core.registry`).
  - Part A detail: the set of names other modules import from `colosseum.worker.rollout_worker`. Step 9 checks it with grep.
- Produces:
  - `colosseum.core.threads.configure_torch_threads(num_threads: int, interop_threads: int = 1) -> None`
  - `colosseum.core.threads.resolve_learner_threads(torch_threads: int | None, device: str, num_workers: int, worker_threads: int, num_learners: int, cpu_count: int | None = None) -> int`
  - `RolloutConfig.torch_threads: int = 1` (ge=1), `LearnerConfig.torch_threads: int | None = None` (ge=1)
  - `rollout_worker_process(*, worker_id, env_fn, num_envs, chunk_length, agent_ids, model_factories, trajectory_queues, weight_queues, stop_event, gamma=0.99, weight_sync_interval=5.0, torch_threads=1, max_env_steps=0, checkpoint_state_dicts_by_agent=None, slot_agent_map=None, slot_network_map=None, collect_mask=None, results_queue=None, command_queue=None, seed=None, vec_env_kind="sync", subproc_workers=None) -> None`. All arguments are keyword-only.
  - `colosseum.worker.rollout_worker._drain_commands(command_queue) -> WorkerCommand | None`. It merges `new_checkpoints` over all drained commands.
  - `launcher._worker_target(*, worker_id, config, agent_ids, agent_configs, trajectory_queues, weight_queues, stop_event, total_timesteps=0, checkpoint_state_dicts_by_agent=None, slot_network_map=None, collect_mask=None, slot_agent_map=None, results_queue=None, command_queue=None)`. Keyword-only.
  - `launcher._learner_target(*, ..., num_learners: int = 1)`. Keyword-only; existing parameters unchanged.
  - `tests/dataflow_helpers.py`: `OBS_DIM`, `NUM_ACTIONS`, `TinyModel`, `make_tiny_model`, `_Base`, `ThreadProbeEnv`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/unit/test_threads.py`:

```python
"""Torch thread limits: config fields and the learner auto-thread formula (T2.1)."""
import pytest
import torch
from pydantic import ValidationError

from colosseum.core.config import LearnerConfig, RolloutConfig
from colosseum.core.threads import configure_torch_threads, resolve_learner_threads


def test_config_thread_fields_defaults_and_validation():
    assert RolloutConfig().torch_threads == 1
    assert LearnerConfig().torch_threads is None
    with pytest.raises(ValidationError):
        RolloutConfig(torch_threads=0)
    with pytest.raises(ValidationError):
        LearnerConfig(torch_threads=0)


@pytest.mark.parametrize(
    "torch_threads, device, workers, worker_threads, learners, cpus, expected",
    [
        (None, "cpu", 4, 1, 1, 8, 4),   # (8 - 4*1) // 1
        (None, "cpu", 2, 1, 2, 8, 3),   # (8 - 2*1) // 2
        (None, "cpu", 4, 2, 1, 8, 1),   # max(1, 0)
        (None, "cpu", 8, 1, 3, 4, 1),   # negative remainder -> 1
        (None, "cuda", 4, 1, 1, 8, 2),
        (None, "cuda:1", 4, 1, 1, 8, 2),
        (5, "cpu", 4, 1, 1, 8, 5),      # an explicit value wins
        (3, "cuda", 4, 1, 1, 8, 3),
    ],
)
def test_resolve_learner_threads(torch_threads, device, workers, worker_threads, learners, cpus, expected):
    got = resolve_learner_threads(torch_threads, device, workers, worker_threads, learners, cpu_count=cpus)
    assert got == expected


def test_configure_torch_threads_sets_intra_op_threads():
    before = torch.get_num_threads()
    try:
        configure_torch_threads(2)
        assert torch.get_num_threads() == 2
        configure_torch_threads(1)  # a second call must not raise (inter-op already fixed)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)
```

- [ ] **Step 2: Run the unit tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_threads.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.core.threads'`.

- [ ] **Step 3: Create `src/colosseum/core/threads.py`**

```python
"""Torch thread limits for worker / learner processes (R2-04, R6-04)."""

from __future__ import annotations

import os
from typing import Optional

import torch


def configure_torch_threads(num_threads: int, interop_threads: int = 1) -> None:
    """Set torch intra-op threads and (if still possible) inter-op threads.

    ``set_num_interop_threads`` may be called only once per process and only
    before any inter-op parallel work; later calls raise RuntimeError, which is
    ignored here (e.g. when a worker loop is run in-process by a test).
    """
    torch.set_num_threads(max(1, int(num_threads)))
    try:
        torch.set_num_interop_threads(max(1, int(interop_threads)))
    except RuntimeError:
        pass


def resolve_learner_threads(
    torch_threads: Optional[int],
    device: str,
    num_workers: int,
    worker_threads: int,
    num_learners: int,
    cpu_count: Optional[int] = None,
) -> int:
    """Thread count for a learner process (spec block 2).

    Explicit ``learner.torch_threads`` wins. Otherwise: 2 on CUDA, and on CPU
    ``max(1, (cpu_count - num_workers * worker_threads) // num_learners)``.
    """
    if torch_threads is not None:
        return max(1, int(torch_threads))
    if str(device).startswith("cuda"):
        return 2
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return max(1, (cpus - num_workers * worker_threads) // max(1, num_learners))
```

- [ ] **Step 4: Add the config fields**

In `src/colosseum/core/config.py`, add this field to `class RolloutConfig` right after the `subproc_workers` field:

```python
    torch_threads: int = Field(
        default=1,
        ge=1,
        description="torch intra-op threads per worker process, set at process start "
                    "(inter-op threads are always 1). SubprocessVectorEnv children always "
                    "use 1 thread.",
    )
```

Add this field to `class LearnerConfig` right after the `pin_memory` field:

```python
    torch_threads: Optional[int] = Field(
        default=None,
        ge=1,
        description="torch threads per learner process. None = auto: 2 on CUDA; on CPU "
                    "max(1, (cpu_count - num_workers * rollout.torch_threads) // num_learners).",
    )
```

- [ ] **Step 5: Run the unit tests**

Run: `.venv/bin/python -m pytest tests/unit/test_threads.py -v`
Expected: PASS (10 tests).

- [ ] **Step 6: Create `tests/dataflow_helpers.py`**

```python
"""Toy envs, probe models and queue helpers for data-flow / transition tests.

Importable as ``tests.dataflow_helpers`` so spawned processes can unpickle the
env and model factories defined here.
"""

from __future__ import annotations

import os

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.envs.base_env import BaseEnv
from colosseum.networks.distributions import CategoricalDist
from colosseum.networks.model import PolicyModel, StepOutput

OBS_DIM = 4
NUM_ACTIONS = 3


class TinyModel(PolicyModel):
    """Small stateless trainable model (linear policy + value) for IPC tests."""

    def __init__(self, obs_dim: int = OBS_DIM, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.pi = nn.Linear(obs_dim, num_actions)
        self.v = nn.Linear(obs_dim, 1)

    def step(self, obs, state, action_mask=None) -> StepOutput:
        obs = obs.float()
        return StepOutput(CategoricalDist(self.pi(obs), mask=action_mask),
                          self.v(obs).squeeze(-1), None)


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

    obs = [torch.get_num_threads(), OMP_NUM_THREADS (or -1), ep, t]; 4 steps per episode.
    """

    NUM_PLAYERS = 1

    def _probe(self) -> dict[int, np.ndarray]:
        omp = float(os.environ.get("OMP_NUM_THREADS", "-1"))
        return {0: np.array([torch.get_num_threads(), omp, self.ep, self.t], np.float32)}

    def reset(self, seed=None):
        self.ep += 1
        self.t = 0
        return self._probe(), {0: {}}

    def step(self, actions):
        self.t += 1
        done = self.t >= 4
        return self._probe(), {0: 0.0}, {0: done}, {0: False}, {0: {}}
```

- [ ] **Step 7: Write the failing integration tests**

Create `tests/integration/test_worker_threads.py`:

```python
"""Spawned workers and SubprocessVectorEnv children run with the configured torch threads (T2.1)."""
import multiprocessing as mp
import os

import numpy as np

from colosseum.envs.subproc_vec_env import SubprocessVectorEnv
from colosseum.worker.rollout_worker import rollout_worker_process
from tests.dataflow_helpers import ThreadProbeEnv, make_tiny_model


def _observations(item) -> np.ndarray:
    """Chunk observations from a queue item (a TrajectoryChunk before T2.2, a payload dict after)."""
    if isinstance(item, dict):
        return np.asarray(item["observations"])
    return item.observations.numpy()


def _run_probe_worker(torch_threads: int, vec_env_kind: str) -> np.ndarray:
    ctx = mp.get_context("spawn")
    tq = ctx.Queue()
    wq = ctx.Queue(maxsize=1)
    stop = ctx.Event()
    proc = ctx.Process(
        target=rollout_worker_process,
        kwargs=dict(
            worker_id=0, env_fn=ThreadProbeEnv, num_envs=1, chunk_length=4,
            agent_ids=["a"], model_factories={"a": make_tiny_model},
            trajectory_queues={"a": tq}, weight_queues={"a": wq}, stop_event=stop,
            torch_threads=torch_threads, vec_env_kind=vec_env_kind, subproc_workers=1,
        ),
        daemon=False,  # a subprocess vec env spawns its own children
    )
    proc.start()
    try:
        item = tq.get(timeout=120)
    finally:
        stop.set()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
            proc.join()
    return _observations(item)


def test_spawned_worker_uses_rollout_torch_threads():
    obs = _run_probe_worker(torch_threads=2, vec_env_kind="sync")
    assert set(obs[:, 0].tolist()) == {2.0}


def test_subprocess_env_children_use_one_thread():
    obs = _run_probe_worker(torch_threads=2, vec_env_kind="subprocess")
    assert set(obs[:, 0].tolist()) == {1.0}   # torch.get_num_threads() in the env child
    assert set(obs[:, 1].tolist()) == {1.0}   # OMP_NUM_THREADS in the env child


def test_subproc_vec_env_children_threads_and_parent_env_restored():
    before = os.environ.get("OMP_NUM_THREADS")
    vec = SubprocessVectorEnv(ThreadProbeEnv, num_envs=2, num_workers=2)
    try:
        obs, _ = vec.reset_all()
        assert obs[:, 0, 0].tolist() == [1.0, 1.0]
        assert obs[:, 0, 1].tolist() == [1.0, 1.0]
    finally:
        vec.close()
    assert os.environ.get("OMP_NUM_THREADS") == before
```

- [ ] **Step 8: Run the integration tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/integration/test_worker_threads.py -v`
Expected:
- the two worker tests fail with `TypeError: rollout_worker_process() got an unexpected keyword argument 'model_factories'` or `'torch_threads'` (raised in the child; the parent then fails with `_queue.Empty`);
- the `SubprocessVectorEnv` test fails because `obs[:, 0, 0]` is the machine's default thread count.

- [ ] **Step 9: Check who imports from `rollout_worker.py`**

Run: `grep -rn "rollout_worker import\|rollout_worker\." src tests examples scripts --include=*.py`

The new file in Step 10 exports `rollout_worker_process`, `_drain_commands` and `LATEST_NETWORK_ID`. For every other imported name:
- If it is defined in `src/colosseum/worker/rollout_loop.py` (Part A moved the loop body there), add `from colosseum.worker.rollout_loop import <Name>  # noqa: F401` to the import block of the new file.
- If the importer is a test of a helper that no longer exists anywhere (for example `_apply_command` or `_sync_weights` of the old wrapper), delete that test. `RolloutLoop` owns this behaviour, and Part A's characterization tests cover it.

- [ ] **Step 10: Replace `src/colosseum/worker/rollout_worker.py`**

```python
"""Rollout worker process: a thin process wrapper around :class:`RolloutLoop`.

The wrapper limits torch threads (R2-04), turns queues into :class:`LoopIO`
callbacks and runs the loop until ``stop_event`` is set (or ``max_env_steps``
env steps were taken).
"""

from __future__ import annotations

import logging
import queue
from typing import Any, Callable, Optional, Union

from colosseum.core.threads import configure_torch_threads
from colosseum.core.types import MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand
from colosseum.envs.base_env import BaseEnv
from colosseum.networks.model import PolicyModel
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop

logger = logging.getLogger(__name__)

LATEST_NETWORK_ID = "latest"


def _drain_commands(command_queue: Any) -> Optional[WorkerCommand]:
    """Newest WorkerCommand, with ``new_checkpoints`` merged over all drained ones.

    Checkpoints are sent to a worker only once (as deltas), so a coalesced
    command must keep the checkpoints of the commands it replaces (R2-06).
    """
    latest: Optional[WorkerCommand] = None
    merged: dict[str, dict[str, Any]] = {}
    while True:
        try:
            cmd = command_queue.get_nowait()
        except queue.Empty:
            break
        for aid, ckpts in cmd.new_checkpoints.items():
            merged.setdefault(aid, {}).update(ckpts)
        latest = cmd
    if latest is not None:
        latest.new_checkpoints = merged
    return latest


def rollout_worker_process(
    *,
    worker_id: int,
    env_fn: Callable[[], BaseEnv],
    num_envs: int,
    chunk_length: int,
    agent_ids: list[str],
    model_factories: dict[str, Callable[[], PolicyModel]],
    trajectory_queues: dict[str, Any],
    weight_queues: dict[str, Any],
    stop_event: Any,
    gamma: Union[float, dict[str, float]] = 0.99,
    weight_sync_interval: float = 5.0,
    torch_threads: int = 1,
    max_env_steps: int = 0,
    checkpoint_state_dicts_by_agent: Optional[dict[str, dict[str, Any]]] = None,
    slot_agent_map: Optional[list[list[str]]] = None,
    slot_network_map: Optional[list[list[str]]] = None,
    collect_mask: Optional[list[list[bool]]] = None,
    results_queue: Any = None,
    command_queue: Any = None,
    seed: Optional[int] = None,
    vec_env_kind: str = "sync",
    subproc_workers: Optional[int] = None,
) -> None:
    """Worker process entry point (see module docstring)."""
    configure_torch_threads(torch_threads)

    def send_chunk(chunk: TrajectoryChunk) -> None:
        q = trajectory_queues[chunk.agent_id]
        while not stop_event.is_set():
            try:
                q.put(chunk, timeout=1.0)
                return
            except queue.Full:
                continue

    def poll_weights(agent_id: str) -> Optional[WeightPayload]:
        latest = None
        q = weight_queues[agent_id]
        while True:
            try:
                latest = q.get_nowait()
            except queue.Empty:
                return latest

    def report_result(result: MatchResult) -> None:
        try:
            results_queue.put_nowait(result)
        except queue.Full:
            logger.debug(f"Worker {worker_id}: results queue full, dropping {result.match_id}")

    def poll_command() -> Optional[WorkerCommand]:
        return _drain_commands(command_queue)

    io = LoopIO(
        send_chunk=send_chunk,
        poll_weights=poll_weights,
        report_result=report_result if results_queue is not None else None,
        poll_command=poll_command if command_queue is not None else None,
    )
    logger.info(
        f"Worker {worker_id}: starting with {num_envs} envs ({vec_env_kind}), "
        f"chunk_length={chunk_length}, agents={agent_ids}, torch_threads={torch_threads}"
    )
    loop = RolloutLoop(
        worker_id=worker_id, env_fn=env_fn, num_envs=num_envs, chunk_length=chunk_length,
        agent_ids=agent_ids, model_factories=model_factories, io=io, gamma=gamma,
        weight_sync_interval=weight_sync_interval, slot_agent_map=slot_agent_map,
        slot_network_map=slot_network_map, collect_mask=collect_mask,
        checkpoint_state_dicts_by_agent=checkpoint_state_dicts_by_agent, seed=seed,
        vec_env_kind=vec_env_kind, subproc_workers=subproc_workers,
    )
    try:
        loop.run(should_stop=stop_event.is_set, max_env_steps=max_env_steps)
    finally:
        loop.close()
        # Detach feeder threads of queues this worker produced to, so undrained
        # items (e.g. chunks a stopped learner never consumed) cannot block exit.
        for q in [*trajectory_queues.values(), results_queue]:
            if q is not None and hasattr(q, "cancel_join_thread"):
                q.cancel_join_thread()
    logger.info(f"Worker {worker_id}: finished. {loop.stats}")
```

- [ ] **Step 11: Limit threads in `SubprocessVectorEnv` children**

In `src/colosseum/envs/subproc_vec_env.py`, replace the first line of `_worker_loop`'s body (`vec_env = VectorEnv(env_fn, slice_size)`) with:

```python
    import torch

    # Env children only step envs: one torch thread each (R2-04). OMP_NUM_THREADS=1
    # is already in this process's environment (set by the parent before spawn).
    torch.set_num_threads(1)
    vec_env = VectorEnv(env_fn, slice_size)
```

In `SubprocessVectorEnv.__init__`, replace the spawn loop:

```python
        for (start, end) in self._slices:
            parent_conn, child_conn = self._ctx.Pipe()
            proc = self._ctx.Process(
                target=_worker_loop,
                args=(child_conn, env_fn, end - start, start),
                daemon=True,
            )
            proc.start()
            # Close the child end in the parent so EOF propagates correctly.
            child_conn.close()
            self._parent_conns.append(parent_conn)
            self._procs.append(proc)
```

with:

```python
        # Children must start with OMP_NUM_THREADS=1: they import torch while
        # unpickling env_fn, before _worker_loop runs (R2-04).
        prev_omp = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            for (start, end) in self._slices:
                parent_conn, child_conn = self._ctx.Pipe()
                proc = self._ctx.Process(
                    target=_worker_loop,
                    args=(child_conn, env_fn, end - start, start),
                    daemon=True,
                )
                proc.start()
                # Close the child end in the parent so EOF propagates correctly.
                child_conn.close()
                self._parent_conns.append(parent_conn)
                self._procs.append(proc)
        finally:
            if prev_omp is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = prev_omp
```

- [ ] **Step 12: Update the launcher**

In `src/colosseum/launcher.py`, replace the whole `_worker_target` function with:

```python
def _worker_target(
    *,
    worker_id: int,
    config: ColosseumConfig,
    agent_ids: list[str],
    agent_configs: dict[str, ColosseumConfig],
    trajectory_queues: dict[str, mp.Queue],
    weight_queues: dict[str, mp.Queue],
    stop_event: mp.Event,
    total_timesteps: int = 0,
    checkpoint_state_dicts_by_agent: Optional[dict] = None,
    slot_network_map: Optional[list[list[str]]] = None,
    collect_mask: Optional[list[list[bool]]] = None,
    slot_agent_map: Optional[list[list[str]]] = None,
    results_queue: Optional[mp.Queue] = None,
    command_queue: Optional[mp.Queue] = None,
) -> None:
    """Worker process entry point.

    Env and model factories are built INSIDE the process (``functools.partial``
    over top-level functions is picklable under spawn, which the subprocess
    vec_env needs to ship ``env_fn`` to its children).
    """
    import sys
    from functools import partial
    sys.path.insert(0, ".")

    from colosseum.core.registry import build_model
    from colosseum.worker.rollout_worker import rollout_worker_process

    worker_seed = None
    if config.training.seed is not None:
        worker_seed = config.training.seed + worker_id * 1000

    rollout_worker_process(
        worker_id=worker_id,
        env_fn=partial(_create_env, config.env.env_class, config.env.kwargs),
        num_envs=config.rollout.envs_per_worker,
        chunk_length=config.rollout.chunk_length,
        agent_ids=agent_ids,
        model_factories={aid: partial(build_model, agent_configs[aid]) for aid in agent_ids},
        trajectory_queues=trajectory_queues,
        weight_queues=weight_queues,
        stop_event=stop_event,
        gamma=config.algorithm.gamma,
        weight_sync_interval=config.rollout.weight_sync_interval_sec,
        torch_threads=config.rollout.torch_threads,
        max_env_steps=total_timesteps,
        checkpoint_state_dicts_by_agent=checkpoint_state_dicts_by_agent,
        slot_agent_map=slot_agent_map,
        slot_network_map=slot_network_map,
        collect_mask=collect_mask,
        results_queue=results_queue,
        command_queue=command_queue,
        seed=worker_seed,
        vec_env_kind=config.rollout.vec_env,
        subproc_workers=config.rollout.subproc_workers,
    )
```

In `_learner_target`:
- add `*,` as the first line of its parameter list;
- add `num_learners: int = 1,` as the last parameter;
- right after the block that resolves `device` (`device = config.learner.device` / `if device == "auto": ...`), insert:

```python
    from colosseum.core.threads import configure_torch_threads, resolve_learner_threads

    configure_torch_threads(resolve_learner_threads(
        config.learner.torch_threads, device, config.rollout.num_workers,
        config.rollout.torch_threads, num_learners,
    ))
```

In `Launcher.launch`, replace the learner process creation (`learner_proc = mp.Process(target=_learner_target, args=(...), daemon=True)`) with:

```python
            learner_proc = mp.Process(
                target=_learner_target,
                kwargs=dict(
                    agent_id=aid,
                    config=acfg,
                    trajectory_queue=trajectory_queues[aid],
                    weight_queues=weight_queues_per_agent[aid],
                    stop_event=self._stop_event,
                    metrics_queue=metrics_queue,
                    total_train_steps=total_train_steps,
                    checkpoint_queue=checkpoint_queues[aid],
                    checkpoint_interval=checkpoint_interval,
                    resume_state=resume_state,
                    num_learners=len(trainable_agents),
                ),
                daemon=True,
            )
```

Also replace the worker process creation (`worker_proc = mp.Process(target=_worker_target, args=(...), daemon=worker_daemon)`) with:

```python
            worker_proc = mp.Process(
                target=_worker_target,
                kwargs=dict(
                    worker_id=worker_id,
                    config=cfg,
                    agent_ids=trainable_agents,
                    agent_configs=agent_configs,
                    trajectory_queues=trajectory_queues,
                    weight_queues=worker_weight_queues,
                    stop_event=self._stop_event,
                    total_timesteps=cfg.training.total_timesteps // cfg.rollout.num_workers,
                    checkpoint_state_dicts_by_agent=ckpt_dicts_by_agent,
                    slot_network_map=slot_network_map,
                    collect_mask=collect_mask,
                    slot_agent_map=slot_agent_map,
                    results_queue=results_queue,
                    command_queue=command_queues[worker_id],
                ),
                daemon=worker_daemon,
            )
```

`_worker_target` now calls `build_model` directly. Leave the launcher's existing model-building helper (Part A's replacement for `_create_network`, used by `_learner_target`) unchanged.

- [ ] **Step 13: Update the distributed roles**

In `src/colosseum/distributed.py`, replace the body of `_dist_worker_target` from the line `from colosseum.core.registry import ...` to the end of the function with:

```python
    from colosseum.core.registry import build_model
    from colosseum.launcher import _create_env
    from colosseum.transport.grpc_transport import GRPCTransport
    from colosseum.weight_store.grpc_store import GRPCWeightStore
    from colosseum.worker.rollout_worker import rollout_worker_process

    max_mb = config.transport.grpc_max_message_mb
    store = GRPCWeightStore(weight_store_address, max_message_mb=max_mb)
    transports = {
        aid: GRPCTransport(learner_addresses[aid], max_message_mb=max_mb)
        for aid in agent_ids
    }

    worker_seed = None
    if config.training.seed is not None:
        worker_seed = config.training.seed + worker_id * 1000

    rollout_worker_process(
        worker_id=worker_id,
        env_fn=partial(_create_env, config.env.env_class, config.env.kwargs),
        num_envs=config.rollout.envs_per_worker,
        chunk_length=config.rollout.chunk_length,
        agent_ids=agent_ids,
        model_factories={aid: partial(build_model, agent_configs[aid]) for aid in agent_ids},
        trajectory_queues={aid: GRPCTrajectorySink(transports[aid], aid) for aid in agent_ids},
        weight_queues={aid: GRPCWeightSource(store, aid) for aid in agent_ids},
        stop_event=stop_event,
        gamma=config.algorithm.gamma,
        weight_sync_interval=config.rollout.weight_sync_interval_sec,
        torch_threads=config.rollout.torch_threads,
        max_env_steps=total_timesteps,
        slot_agent_map=slot_agent_map,
        seed=worker_seed,
        vec_env_kind=config.rollout.vec_env,
        subproc_workers=config.rollout.subproc_workers,
    )
```

In `run_distributed_learner`, right after the block that resolves `device`, insert:

```python
    from colosseum.core.threads import configure_torch_threads, resolve_learner_threads

    # The learner role does not know which workers share its machine, so with
    # learner.torch_threads unset it assumes none (num_workers=0).
    configure_torch_threads(resolve_learner_threads(
        acfg.learner.torch_threads, device, num_workers=0,
        worker_threads=acfg.rollout.torch_threads, num_learners=1,
    ))
```

- [ ] **Step 14: Update the remaining callers of the changed signatures**

Run: `grep -rn "_worker_target\|rollout_worker_process(" tests scripts --include=*.py`

Convert every call to keyword arguments. Before Part A, the callers were `tests/test_integration.py::test_worker_produces_chunks` and `tests/test_multi_agent.py::test_worker_multi_agent_routing`; after Part A they may live under `tests/integration/`. For example, the positional `args=(0, config, [agent_id], agent_configs, trajectory_queues, weight_queues, stop_event, 500)` becomes:

```python
    p = mp.Process(
        target=_worker_target,
        kwargs=dict(
            worker_id=0, config=config, agent_ids=[agent_id], agent_configs=agent_configs,
            trajectory_queues=trajectory_queues, weight_queues=weight_queues,
            stop_event=stop_event, total_timesteps=500,
        ),
        daemon=True,
    )
```

For the multi-agent routing test, also pass `slot_network_map=slot_network_map, collect_mask=collect_mask, slot_agent_map=slot_agent_map` by keyword and drop the positional `None`s. In any direct `rollout_worker_process(...)` call:
- rename `network_factories=` to `model_factories=`;
- rename `total_timesteps=` to `max_env_steps=`.

- [ ] **Step 15: Run the new tests**

Run: `.venv/bin/python -m pytest tests/unit/test_threads.py tests/integration/test_worker_threads.py -v`
Expected: PASS (13 tests).

- [ ] **Step 16: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: all pass.

- [ ] **Step 17: Commit**

```bash
git add src/colosseum/core/threads.py src/colosseum/core/config.py src/colosseum/worker/rollout_worker.py \
        src/colosseum/envs/subproc_vec_env.py src/colosseum/launcher.py src/colosseum/distributed.py \
        tests/dataflow_helpers.py tests/unit/test_threads.py tests/integration/test_worker_threads.py tests
git commit -m "perf: limit torch threads in workers, env children and learners"
```

---

### Task T2.2: Numpy payloads for all inter-process data

Spec block 2 "Передача между процессами" (R6-02, R3-07). Today torch tensors put on an `mp.Queue` travel as file descriptors that the sending process serves, so the receiver crashes when the sender has exited. After this task everything that crosses a process boundary is numpy plus primitives:
- chunks;
- weights;
- checkpoints and resume states;
- `WorkerCommand`s;
- results and metrics.

gRPC serializes the same payloads (the wire format changes in SP5).

**Files:**
- Create: `src/colosseum/core/ipc.py` (payload helpers; T2.3 and T2.5 extend it)
- Modify: `src/colosseum/core/types.py` (`state_dict_to_numpy`, `state_dict_from_numpy`, `TrajectoryChunk.to_payload/from_payload`, `WeightPayload`, `WorkerCommand`)
- Replace: `src/colosseum/transport/serialization.py`
- Modify: `src/colosseum/transport/grpc_transport.py`, `src/colosseum/transport/local.py`
- Replace: `src/colosseum/weight_store/shared_memory.py`
- Modify: `src/colosseum/worker/rollout_worker.py` (`send_chunk`), `src/colosseum/worker/rollout_loop.py` (weight/checkpoint loading)
- Modify: `src/colosseum/learner/learner.py` (resume, checkpoint, `_collect_chunks`, `_push_weights`)
- Modify: `src/colosseum/launcher.py` (`_derive_worker_configs`, checkpoint draining, resume state)
- Modify: `src/colosseum/distributed.py` (checkpoint drainer)
- Modify: `tests/dataflow_helpers.py`
- Test: `tests/unit/test_payloads.py`, `tests/unit/test_serialization.py`, `tests/contract/test_no_tensors_in_queues.py`
- Update: gRPC/distributed tests, tests that read worker queues, the checkpoint-queue monitor test

**Interfaces:**
- Consumes:
  - `state_to_numpy(state)` and `state_from_numpy(obj, device="cpu")` from `colosseum.networks.state` (T1.1); both map `None` to `None`;
  - `TrajectoryChunk` with `initial_state` (T1.5);
  - `rollout_worker_process` (T2.1).
  - Part A detail: the `load_state_dict(...)` calls inside `RolloutLoop` (Step 13 finds them with grep).
- Produces:
  - `colosseum.core.ipc.to_numpy_tree(obj) -> Any`, `from_numpy_tree(obj) -> Any`, `find_tensor(obj, path="item") -> str | None`, `assert_no_tensors(obj, what="item") -> None` (raises `TypeError` naming the path)
  - `colosseum.core.types.state_dict_to_numpy(sd) -> dict[str, np.ndarray]`, `state_dict_from_numpy(sd) -> dict[str, Tensor]`
  - `TrajectoryChunk.to_payload() -> dict[str, Any]`, `TrajectoryChunk.from_payload(payload) -> TrajectoryChunk` (contract)
  - `WeightPayload.state_dict: dict[str, np.ndarray]`, `WeightPayload.from_model(agent_id, policy_version, model)`, `WeightPayload.to_torch_state_dict()` (contract)
  - `WorkerCommand.new_checkpoints: dict[str, dict[str, dict[str, np.ndarray]]]` (contract)
  - `colosseum.transport.serialization.pack_payload(obj, compress=True) -> (bytes, bool)`, `unpack_payload(data, compressed) -> Any`, `serialize_chunk_payload`/`deserialize_chunk_payload`, `serialize_state_dict`/`deserialize_state_dict` (numpy), `serialize_chunk`/`deserialize_chunk` (TrajectoryChunk convenience)
  - Every item on a trajectory queue is now a chunk payload `dict`. `GRPCTransport.send_chunk` accepts a `TrajectoryChunk` or a payload dict. `TrajectoryServicer` puts payload dicts on its queue.
  - Checkpoint-queue items: `{"policy_version": int, "state_dict": dict[str, np.ndarray], "optimizer_state": numpy tree | None}`.
  - `tests/dataflow_helpers.py`: `CheckedQueue`, `chunk_payload(T=4, version=0, agent_id="a")`, `GridStepEnv`, `EnvFactory`

- [ ] **Step 1: Write the failing payload unit tests**

Create `tests/unit/test_payloads.py`:

```python
"""Numpy payload forms of chunks, weights, commands and optimizer states (T2.2)."""
import numpy as np
import pytest
import torch

from colosseum.core.ipc import assert_no_tensors, from_numpy_tree, to_numpy_tree
from colosseum.core.types import (
    TrajectoryChunk,
    WeightPayload,
    WorkerCommand,
    state_dict_from_numpy,
    state_dict_to_numpy,
)
from colosseum.weight_store.shared_memory import InMemoryWeightStore
from tests.dataflow_helpers import TinyModel


def _chunk(initial_state, masks: bool, T: int = 5) -> TrajectoryChunk:
    return TrajectoryChunk(
        agent_id="a", observations=torch.randn(T, 4), actions=torch.randint(0, 3, (T,)),
        action_log_probs=torch.randn(T), rewards=torch.randn(T),
        dones=torch.tensor([False, True, False, False, True]), values=torch.randn(T),
        bootstrap_value=torch.tensor(0.25), behavior_policy_version=7,
        initial_state=initial_state,
        action_masks=(torch.rand(T, 3) > 0.3) if masks else None,
    )


STATES = {
    "none": None,
    "lstm_like": {"h": torch.randn(1, 1, 8), "c": torch.randn(1, 1, 8)},
    "window_like": {"mem": torch.randn(1, 4, 8), "len": torch.tensor([3])},
}


def _assert_same_state(a, b) -> None:
    if a is None:
        assert b is None
        return
    assert set(a) == set(b)
    for key in a:
        assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key])


@pytest.mark.parametrize("state_name", list(STATES))
@pytest.mark.parametrize("masks", [True, False])
def test_chunk_payload_roundtrip(state_name, masks):
    chunk = _chunk(STATES[state_name], masks)
    payload = chunk.to_payload()
    assert_no_tensors(payload)
    back = TrajectoryChunk.from_payload(payload)
    for name in ("observations", "actions", "action_log_probs", "rewards", "dones", "values"):
        assert torch.equal(getattr(chunk, name), getattr(back, name)), name
    assert back.dones.dtype == torch.bool
    assert float(back.bootstrap_value) == pytest.approx(0.25)
    assert back.behavior_policy_version == 7 and back.agent_id == "a"
    _assert_same_state(chunk.initial_state, back.initial_state)
    if masks:
        assert torch.equal(chunk.action_masks, back.action_masks)
    else:
        assert back.action_masks is None


def test_weight_payload_is_numpy_and_loads_into_a_model():
    model = TinyModel()
    wp = WeightPayload.from_model("a", 3, model)
    assert_no_tensors(wp)
    assert all(isinstance(v, np.ndarray) for v in wp.state_dict.values())
    other = TinyModel()
    other.load_state_dict(wp.to_torch_state_dict())
    for key, value in model.state_dict().items():
        assert torch.equal(value, other.state_dict()[key])


def test_state_dict_numpy_roundtrip_handles_bfloat16():
    sd = {"w": torch.randn(2, 2).to(torch.bfloat16), "n": torch.tensor(5)}
    np_sd = state_dict_to_numpy(sd)
    assert np_sd["w"].dtype == np.float32 and np_sd["n"].dtype == np.int64
    back = state_dict_from_numpy(np_sd)
    assert torch.equal(back["w"], sd["w"].float()) and int(back["n"]) == 5


def test_worker_command_with_numpy_checkpoints_has_no_tensors():
    cmd = WorkerCommand(
        slot_agent_map=[["a"]], slot_network_map=[["ckpt_v1"]], collect_mask=[[False]],
        new_checkpoints={"a": {"ckpt_v1": state_dict_to_numpy(TinyModel().state_dict())}},
    )
    assert_no_tensors(cmd)


def test_optimizer_state_numpy_tree_roundtrip():
    model = TinyModel()
    opt = torch.optim.Adam(model.parameters())
    model.v(torch.randn(2, 4)).sum().backward()
    opt.step()
    tree = to_numpy_tree(opt.state_dict())
    assert_no_tensors(tree)
    restored = torch.optim.Adam(TinyModel().parameters())
    restored.load_state_dict(from_numpy_tree(tree))
    key = next(iter(opt.state_dict()["state"]))
    assert torch.equal(restored.state_dict()["state"][key]["exp_avg"],
                       opt.state_dict()["state"][key]["exp_avg"])


def test_assert_no_tensors_reports_the_path():
    with pytest.raises(TypeError, match=r"item\['x'\]\[1\]"):
        assert_no_tensors({"x": [1, torch.zeros(1)]})


def test_in_memory_weight_store_keeps_numpy():
    store = InMemoryWeightStore()
    store.put("a", WeightPayload.from_model("a", 2, TinyModel()))
    got = store.get("a")
    assert got.policy_version == 2 and store.get_version("a") == 2
    assert all(isinstance(v, np.ndarray) for v in got.state_dict.values())
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_payloads.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.core.ipc'`.

- [ ] **Step 3: Create `src/colosseum/core/ipc.py`**

```python
"""Inter-process helpers: newest-wins mailboxes, a shared step counter, payload checks.

Rule (spec block 2): nothing that crosses a process boundary may contain a
``torch.Tensor``. Tensors put on an ``mp.Queue`` are shared through file
descriptors served by the sending process, so the receiver crashes when the
sender has already exited (R6-02). Everything goes through numpy payloads.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np
import torch


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.numpy().copy()


def _numpy_to_tensor(a: np.ndarray) -> torch.Tensor:
    if not a.flags.writeable or not a.flags.c_contiguous:
        a = np.array(a, copy=True)
    return torch.from_numpy(a)


def to_numpy_tree(obj: Any) -> Any:
    """Copy of ``obj`` with every ``torch.Tensor`` replaced by a numpy array.

    Recurses into dict / list / tuple; other leaves are returned unchanged.
    """
    if isinstance(obj, torch.Tensor):
        return _tensor_to_numpy(obj)
    if isinstance(obj, dict):
        return {k: to_numpy_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_numpy_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_numpy_tree(v) for v in obj)
    return obj


def from_numpy_tree(obj: Any) -> Any:
    """Inverse of :func:`to_numpy_tree`: every ``np.ndarray`` becomes a tensor."""
    if isinstance(obj, np.ndarray):
        return _numpy_to_tensor(obj)
    if isinstance(obj, dict):
        return {k: from_numpy_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_numpy_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(from_numpy_tree(v) for v in obj)
    return obj


def find_tensor(obj: Any, path: str = "item") -> Optional[str]:
    """Return the path of the first ``torch.Tensor`` inside ``obj``, or None.

    Recurses into dict / list / tuple / set and dataclass instances.
    """
    if isinstance(obj, torch.Tensor):
        return path
    if isinstance(obj, dict):
        for k, v in obj.items():
            found = find_tensor(v, f"{path}[{k!r}]")
            if found is not None:
                return found
        return None
    if isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            found = find_tensor(v, f"{path}[{i}]")
            if found is not None:
                return found
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            found = find_tensor(getattr(obj, f.name), f"{path}.{f.name}")
            if found is not None:
                return found
    return None


def assert_no_tensors(obj: Any, what: str = "item") -> None:
    """Raise TypeError if ``obj`` contains a ``torch.Tensor`` anywhere."""
    found = find_tensor(obj, what)
    if found is not None:
        raise TypeError(
            f"torch.Tensor at {found}: inter-process data must be numpy/primitives "
            f"(use to_payload() / WeightPayload.from_model() / to_numpy_tree())"
        )
```

- [ ] **Step 4: Add payload support to `src/colosseum/core/types.py`**

Make sure the module imports `Mapping` from `typing`, `numpy as np` and `torch.nn as nn`; add them to the import block if they are missing:

```python
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
```

Insert these module-level functions between the imports and `class TrajectoryChunk`:

```python
def state_dict_to_numpy(state_dict: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Detached CPU numpy copies of a torch ``state_dict`` (bfloat16 -> float32)."""
    out: dict[str, np.ndarray] = {}
    for key, value in state_dict.items():
        t = value.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        out[key] = t.numpy().copy()
    return out


def state_dict_from_numpy(state_dict: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Torch tensors (CPU) from a numpy ``state_dict``; inverse of state_dict_to_numpy."""
    return {key: _array_to_tensor(np.asarray(value)) for key, value in state_dict.items()}


def _array_to_tensor(a: np.ndarray) -> torch.Tensor:
    if not a.flags.writeable or not a.flags.c_contiguous:
        a = np.array(a, copy=True)
    return torch.from_numpy(a)


def _tensor_to_array(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.numpy()
```

Add these two methods to `class TrajectoryChunk`, after `pin_memory`:

```python
    def to_payload(self) -> dict[str, Any]:
        """Numpy + primitives form for crossing a process boundary (spec block 2)."""
        from colosseum.networks.state import state_to_numpy

        return {
            "agent_id": str(self.agent_id),
            "observations": _tensor_to_array(self.observations),
            "actions": _tensor_to_array(self.actions),
            "action_log_probs": _tensor_to_array(self.action_log_probs),
            "rewards": _tensor_to_array(self.rewards),
            "dones": _tensor_to_array(self.dones),
            "values": _tensor_to_array(self.values),
            "bootstrap_value": float(self.bootstrap_value),
            "behavior_policy_version": int(self.behavior_policy_version),
            "initial_state": state_to_numpy(self.initial_state),
            "action_masks": (
                None if self.action_masks is None else _tensor_to_array(self.action_masks)
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TrajectoryChunk":
        """Rebuild a chunk from :meth:`to_payload` output (CPU tensors)."""
        from colosseum.networks.state import state_from_numpy

        masks = payload.get("action_masks")
        return cls(
            agent_id=str(payload["agent_id"]),
            observations=_array_to_tensor(np.asarray(payload["observations"])),
            actions=_array_to_tensor(np.asarray(payload["actions"])),
            action_log_probs=_array_to_tensor(np.asarray(payload["action_log_probs"])),
            rewards=_array_to_tensor(np.asarray(payload["rewards"])),
            dones=_array_to_tensor(np.asarray(payload["dones"])),
            values=_array_to_tensor(np.asarray(payload["values"])),
            bootstrap_value=torch.tensor(float(payload["bootstrap_value"]), dtype=torch.float32),
            behavior_policy_version=int(payload["behavior_policy_version"]),
            initial_state=state_from_numpy(payload.get("initial_state")),
            action_masks=None if masks is None else _array_to_tensor(np.asarray(masks)),
        )
```

Replace `class WorkerCommand` with:

```python
@dataclass
class WorkerCommand:
    """Runtime match-assignment update pushed from the coordinator to a worker.

    The slot maps are applied per env at its next episode boundary.
    ``new_checkpoints`` carries the checkpoint weights (numpy, see
    :func:`state_dict_to_numpy`) the worker does not have yet (deltas only).

    Attributes:
        slot_agent_map: ``[num_envs][num_players]`` -> agent_id.
        slot_network_map: ``[num_envs][num_players]`` -> ``"latest"`` or a checkpoint id.
        collect_mask: ``[num_envs][num_players]`` -> whether the slot collects trajectories.
        new_checkpoints: ``{agent_id: {checkpoint_id: numpy state_dict}}``.
    """

    slot_agent_map: list[list[str]] = field(default_factory=list)
    slot_network_map: list[list[str]] = field(default_factory=list)
    collect_mask: list[list[bool]] = field(default_factory=list)
    new_checkpoints: dict[str, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict)
```

Replace `class WeightPayload` with:

```python
@dataclass
class WeightPayload:
    """Model weights flowing from a learner to workers (numpy, never torch).

    Attributes:
        agent_id: The agent these weights belong to.
        policy_version: Monotonically increasing version (number of train steps).
        state_dict: ``model.state_dict()`` as numpy arrays (see :func:`state_dict_to_numpy`).
    """

    agent_id: str
    policy_version: int
    state_dict: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_model(cls, agent_id: str, policy_version: int, model: nn.Module) -> "WeightPayload":
        return cls(
            agent_id=agent_id,
            policy_version=int(policy_version),
            state_dict=state_dict_to_numpy(model.state_dict()),
        )

    def to_torch_state_dict(self) -> dict[str, torch.Tensor]:
        return state_dict_from_numpy(self.state_dict)
```

- [ ] **Step 5: Replace `src/colosseum/weight_store/shared_memory.py`**

```python
"""In-process and Manager-backed weight stores holding numpy state dicts.

``WeightPayload.state_dict`` is a ``dict[str, np.ndarray]`` (spec block 2), so the
stores keep it as is. Numpy arrays pickle safely across processes, unlike torch
tensors shared through file descriptors (R6-02).
"""

from __future__ import annotations

import threading
from typing import Optional

from colosseum.core.types import WeightPayload
from colosseum.weight_store.base import BaseWeightStore


class InMemoryWeightStore(BaseWeightStore):
    """Thread-safe in-memory weight store (backs the gRPC weight store server)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._weights: dict[str, dict] = {}
        self._versions: dict[str, int] = {}

    def put(self, agent_id: str, payload: WeightPayload) -> None:
        with self._lock:
            self._weights[agent_id] = dict(payload.state_dict)
            self._versions[agent_id] = payload.policy_version

    def get(self, agent_id: str) -> Optional[WeightPayload]:
        with self._lock:
            if agent_id not in self._weights:
                return None
            return WeightPayload(
                agent_id=agent_id,
                policy_version=self._versions[agent_id],
                state_dict=dict(self._weights[agent_id]),
            )

    def get_version(self, agent_id: str) -> int:
        with self._lock:
            return self._versions.get(agent_id, -1)


class SharedMemoryWeightStore(BaseWeightStore):
    """Weight store backed by a ``multiprocessing.Manager`` dict."""

    def __init__(self) -> None:
        import multiprocessing as mp

        self._manager = mp.Manager()
        self._weights = self._manager.dict()
        self._versions = self._manager.dict()
        self._lock = self._manager.Lock()

    def put(self, agent_id: str, payload: WeightPayload) -> None:
        with self._lock:
            self._weights[agent_id] = dict(payload.state_dict)
            self._versions[agent_id] = payload.policy_version

    def get(self, agent_id: str) -> Optional[WeightPayload]:
        with self._lock:
            if agent_id not in self._weights:
                return None
            state_dict = self._weights[agent_id]
            version = self._versions[agent_id]
        return WeightPayload(agent_id=agent_id, policy_version=version, state_dict=dict(state_dict))

    def get_version(self, agent_id: str) -> int:
        with self._lock:
            return self._versions.get(agent_id, -1)
```

- [ ] **Step 6: Run the payload tests**

Run: `.venv/bin/python -m pytest tests/unit/test_payloads.py -v`
Expected: PASS.

- [ ] **Step 7: Write the failing serialization tests**

Create `tests/unit/test_serialization.py`:

```python
"""gRPC serialization of numpy payloads (T2.2; wire format changes in SP5)."""
import numpy as np
import pytest
import torch

from colosseum.core.types import TrajectoryChunk
from colosseum.transport.serialization import (
    deserialize_chunk,
    deserialize_chunk_payload,
    deserialize_state_dict,
    pack_payload,
    serialize_chunk,
    serialize_chunk_payload,
    serialize_state_dict,
    unpack_payload,
)
from tests.dataflow_helpers import chunk_payload


@pytest.mark.parametrize("compress", [True, False])
def test_pack_unpack_nested_structures(compress):
    obj = {
        "a": np.arange(6, dtype=np.int64).reshape(2, 3),
        3: [np.float32(1.5), None, True, "s"],
        "t": (np.zeros(2, dtype=bool), 7),
        "n": {"x": np.ones((1, 1, 4), np.float32)},
    }
    data, compressed = pack_payload(obj, compress)
    assert compressed is compress
    back = unpack_payload(data, compressed)
    assert np.array_equal(back["a"], obj["a"]) and back["a"].dtype == np.int64
    assert back[3] == [1.5, None, True, "s"]
    assert isinstance(back["t"], tuple) and back["t"][1] == 7 and back["t"][0].dtype == bool
    assert back["n"]["x"].shape == (1, 1, 4)


def test_pack_rejects_tensors_and_object_arrays():
    with pytest.raises(TypeError):
        pack_payload({"x": torch.zeros(1)})
    with pytest.raises(TypeError):
        pack_payload({"x": np.array([object()], dtype=object)})


def test_chunk_payload_bytes_roundtrip_with_state():
    payload = chunk_payload(T=4, version=5)
    payload["initial_state"] = {"mem": np.ones((1, 2, 3), np.float32), "len": np.array([2])}
    payload["action_masks"] = np.ones((4, 3), dtype=bool)
    data, compressed = serialize_chunk_payload(payload)
    back = deserialize_chunk_payload(data, compressed)
    chunk = TrajectoryChunk.from_payload(back)
    assert chunk.behavior_policy_version == 5
    assert torch.equal(chunk.initial_state["len"], torch.tensor([2]))
    assert chunk.action_masks.dtype == torch.bool


def test_serialize_chunk_convenience_roundtrip():
    chunk = TrajectoryChunk.from_payload(chunk_payload(T=4, version=1))
    data, compressed = serialize_chunk(chunk)
    back = deserialize_chunk("a", 9, data, compressed)
    assert torch.equal(back.observations, chunk.observations)
    assert back.behavior_policy_version == 9


def test_state_dict_roundtrip():
    sd = {"w": np.random.randn(3, 3).astype(np.float32), "b": np.zeros(3, np.float32)}
    data, compressed = serialize_state_dict(sd)
    back = deserialize_state_dict(data, compressed)
    assert set(back) == {"w", "b"} and np.array_equal(back["w"], sd["w"])
```

- [ ] **Step 8: Add `CheckedQueue`, `chunk_payload`, `GridStepEnv` and `EnvFactory` to `tests/dataflow_helpers.py`**

Add to the import block:

```python
import queue
from typing import Optional

from colosseum.core.ipc import assert_no_tensors
```

Append to the end of the module:

```python
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
                 const_reward: Optional[float] = None, num_players: int = 2) -> None:
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
```

- [ ] **Step 9: Run the serialization tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_serialization.py -v`
Expected: collection error `ImportError: cannot import name 'deserialize_chunk_payload' from 'colosseum.transport.serialization'`.

- [ ] **Step 10: Replace `src/colosseum/transport/serialization.py`**

```python
"""Serialization for the gRPC transport: numpy payloads <-> bytes.

SP1 wire format (replaced in SP5):
    8-byte little-endian length N | N bytes of JSON skeleton | np.savez archive
The skeleton mirrors the payload structure; every numpy array is replaced by
``{"__nd__": i}`` (index into the archive). Dicts, lists and tuples are tagged
so that int dict keys and tuples survive JSON. Arrays are loaded with
``allow_pickle=False`` and no ``pickle`` is involved, so bytes from the network
cannot execute code. The whole blob is optionally lz4-compressed.
"""

from __future__ import annotations

import io
import json
import struct
from typing import Any

import lz4.frame
import numpy as np
import torch

from colosseum.core.types import TrajectoryChunk

_HEADER = struct.Struct("<Q")


def _encode(obj: Any, arrays: list[np.ndarray]) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            raise TypeError("object arrays cannot be serialized")
        arrays.append(obj)
        return {"__nd__": len(arrays) - 1}
    if isinstance(obj, torch.Tensor):
        raise TypeError("torch.Tensor in payload; convert with to_payload()/to_numpy_tree()")
    if isinstance(obj, dict):
        return {"__dict__": [[_encode(k, arrays), _encode(v, arrays)] for k, v in obj.items()]}
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v, arrays) for v in obj]}
    if isinstance(obj, list):
        return {"__list__": [_encode(v, arrays) for v in obj]}
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"cannot serialize {type(obj).__name__} in a payload")


def _decode(obj: Any, arrays: list[np.ndarray]) -> Any:
    if isinstance(obj, dict):
        if "__nd__" in obj:
            return arrays[obj["__nd__"]]
        if "__dict__" in obj:
            return {_decode(k, arrays): _decode(v, arrays) for k, v in obj["__dict__"]}
        if "__tuple__" in obj:
            return tuple(_decode(v, arrays) for v in obj["__tuple__"])
        if "__list__" in obj:
            return [_decode(v, arrays) for v in obj["__list__"]]
        raise ValueError(f"malformed payload skeleton: {obj!r}")
    return obj


def pack_payload(obj: Any, compress: bool = True) -> tuple[bytes, bool]:
    """Serialize a payload (numpy arrays + primitives in dict/list/tuple)."""
    arrays: list[np.ndarray] = []
    skeleton = json.dumps(_encode(obj, arrays)).encode("utf-8")
    buf = io.BytesIO()
    np.savez(buf, *arrays)
    data = _HEADER.pack(len(skeleton)) + skeleton + buf.getvalue()
    if compress:
        data = lz4.frame.compress(data)
    return data, compress


def unpack_payload(data: bytes, compressed: bool) -> Any:
    """Inverse of :func:`pack_payload`."""
    if compressed:
        data = lz4.frame.decompress(data)
    (n,) = _HEADER.unpack_from(data, 0)
    skeleton = json.loads(data[_HEADER.size:_HEADER.size + n].decode("utf-8"))
    with np.load(io.BytesIO(data[_HEADER.size + n:]), allow_pickle=False) as npz:
        arrays = [npz[f"arr_{i}"] for i in range(len(npz.files))]
    return _decode(skeleton, arrays)


def serialize_state_dict(state_dict: dict[str, np.ndarray], compress: bool = True) -> tuple[bytes, bool]:
    """Serialize a numpy state_dict (``WeightPayload.state_dict``)."""
    return pack_payload(dict(state_dict), compress)


def deserialize_state_dict(data: bytes, compressed: bool) -> dict[str, np.ndarray]:
    return unpack_payload(data, compressed)


def serialize_chunk_payload(payload: dict[str, Any], compress: bool = True) -> tuple[bytes, bool]:
    """Serialize a chunk payload (``TrajectoryChunk.to_payload()``)."""
    return pack_payload(payload, compress)


def deserialize_chunk_payload(data: bytes, compressed: bool) -> dict[str, Any]:
    return unpack_payload(data, compressed)


def serialize_chunk(chunk: TrajectoryChunk, compress: bool = True) -> tuple[bytes, bool]:
    """Serialize a TrajectoryChunk via its numpy payload."""
    return serialize_chunk_payload(chunk.to_payload(), compress)


def deserialize_chunk(
    agent_id: str,
    behavior_policy_version: int,
    data: bytes,
    compressed: bool,
) -> TrajectoryChunk:
    """Deserialize bytes from :func:`serialize_chunk` into a TrajectoryChunk."""
    payload = deserialize_chunk_payload(data, compressed)
    payload["agent_id"] = agent_id
    payload["behavior_policy_version"] = int(behavior_policy_version)
    return TrajectoryChunk.from_payload(payload)
```

- [ ] **Step 11: Send and receive payloads over gRPC and the local transport**

In `src/colosseum/transport/grpc_transport.py`, replace the serialization import with:

```python
from colosseum.transport.serialization import deserialize_chunk_payload, serialize_chunk_payload
```

Replace `TrajectoryServicer.SendChunks` with:

```python
    def SendChunks(self, request_iterator, context):
        """Decode each chunk into its numpy payload dict and queue it for the learner."""
        count = 0
        for proto_chunk in request_iterator:
            payload = deserialize_chunk_payload(proto_chunk.tensor_data, proto_chunk.compressed)
            payload["agent_id"] = proto_chunk.agent_id
            payload["behavior_policy_version"] = int(proto_chunk.behavior_policy_version)
            try:
                self._queue.put(payload, timeout=5.0)
                count += 1
            except queue.Full:
                logger.warning("Trajectory queue full, dropping chunk")
        return colosseum_pb2.SendChunksResponse(chunks_received=count)
```

In `class GRPCTransport`, replace `send_chunk` and `send_chunks_batch` with:

```python
    @staticmethod
    def _to_proto(agent_id: str, chunk: TrajectoryChunk | dict) -> colosseum_pb2.TrajectoryChunkProto:
        payload = chunk.to_payload() if isinstance(chunk, TrajectoryChunk) else chunk
        data, compressed = serialize_chunk_payload(payload)
        return colosseum_pb2.TrajectoryChunkProto(
            agent_id=agent_id,
            behavior_policy_version=int(payload["behavior_policy_version"]),
            tensor_data=data,
            compressed=compressed,
        )

    def send_chunk(self, agent_id: str, chunk: TrajectoryChunk | dict) -> None:
        """Send one chunk (a TrajectoryChunk or its payload dict) in one streaming RPC."""
        self._stub.SendChunks(iter([self._to_proto(agent_id, chunk)]))

    def send_chunks_batch(self, agent_id: str, chunks: list) -> int:
        """Send several chunks (TrajectoryChunks or payload dicts) in one streaming RPC."""
        response = self._stub.SendChunks(self._to_proto(agent_id, c) for c in chunks)
        return response.chunks_received
```

In `src/colosseum/transport/local.py`, replace `send_chunk` and `recv_chunk` with:

```python
    def send_chunk(self, agent_id: str, chunk: TrajectoryChunk) -> None:
        """Send the chunk's numpy payload to the agent's queue. Blocks if the queue is full."""
        if agent_id not in self._queues:
            raise KeyError(f"No channel for agent '{agent_id}'. Call create_channel() first.")
        self._queues[agent_id].put(chunk.to_payload())

    def recv_chunk(self, agent_id: str, timeout: Optional[float] = None) -> Optional[TrajectoryChunk]:
        """Receive and decode a chunk from the agent's queue (None on timeout)."""
        if agent_id not in self._queues:
            raise KeyError(f"No channel for agent '{agent_id}'. Call create_channel() first.")
        try:
            if timeout == 0:
                payload = self._queues[agent_id].get_nowait()
            else:
                payload = self._queues[agent_id].get(timeout=timeout)
        except Empty:
            return None
        return TrajectoryChunk.from_payload(payload)
```

`weight_store/grpc_store.py` needs no code change: `serialize_state_dict`/`deserialize_state_dict` now take and return numpy dicts, and `WeightPayload.state_dict` is numpy.

- [ ] **Step 12: Update the gRPC and distributed tests to numpy**

Run: `grep -rln "serialize_state_dict\|GRPCWeightSink\|serve_trajectory_receiver" tests`
Before Part A these were `tests/test_grpc.py` and `tests/test_distributed.py`.

In each of these files:
- add `import numpy as np` and `from colosseum.core.types import TrajectoryChunk` to the imports if missing;
- make the state-dict helpers return numpy:

```python
def _make_state_dict():          # test_grpc.py
    return {"w": np.random.randn(10, 10).astype(np.float32), "b": np.random.randn(10).astype(np.float32)}


def _state_dict():               # test_distributed.py
    return {"w": np.random.randn(4, 4).astype(np.float32), "b": np.random.randn(4).astype(np.float32)}
```

- replace every `torch.allclose(<a>.state_dict["w"], <b>["w"])` and `torch.allclose(sd["w"], sd2["w"])` / `torch.allclose(sd["b"], sd2["b"])` with the same `np.allclose(...)` call;
- wrap every `chunk_queue.get(timeout=2.0)` that reads from a `serve_trajectory_receiver` queue in `TrajectoryChunk.from_payload(...)`, for example:

```python
        received = TrajectoryChunk.from_payload(chunk_queue.get(timeout=2.0))
```

Run: `.venv/bin/python -m pytest tests/unit/test_serialization.py -v` and then the gRPC test files found by the grep.
Expected: PASS.

- [ ] **Step 13: Write the failing "no tensors in queues" contract test**

Create `tests/contract/test_no_tensors_in_queues.py`:

```python
"""No torch.Tensor ever crosses a process boundary (spec block 2, R6-02).

Real producers (rollout_worker_process, learner_process, Launcher refresh) run
in-process with CheckedQueues that raise on any tensor.
"""
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig, ColosseumConfig, LearnerConfig, load_config
from colosseum.core.types import TrajectoryChunk, WeightPayload, WorkerCommand, state_dict_to_numpy
from colosseum.learner.learner import learner_process
from colosseum.worker.rollout_worker import rollout_worker_process
from tests.dataflow_helpers import (
    CheckedQueue,
    EnvFactory,
    GridStepEnv,
    TinyModel,
    chunk_payload,
    make_tiny_model,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def restore_torch_threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def test_worker_queues_carry_only_numpy(restore_torch_threads):
    tq, wq, rq, cq = CheckedQueue(), CheckedQueue(maxsize=1), CheckedQueue(), CheckedQueue()
    wq.put(WeightPayload.from_model("a", 3, TinyModel()))
    cq.put(WorkerCommand(
        slot_agent_map=[["a", "a"]], slot_network_map=[["latest", "ckpt_v1"]],
        collect_mask=[[True, False]],
        new_checkpoints={"a": {"ckpt_v1": state_dict_to_numpy(TinyModel().state_dict())}},
    ))
    rollout_worker_process(
        worker_id=0, env_fn=EnvFactory(GridStepEnv, lengths=(3,)), num_envs=1, chunk_length=4,
        agent_ids=["a"], model_factories={"a": make_tiny_model},
        trajectory_queues={"a": tq}, weight_queues={"a": wq}, stop_event=threading.Event(),
        weight_sync_interval=0.0, max_env_steps=24, results_queue=rq, command_queue=cq,
    )
    payloads = [tq.get_nowait() for _ in range(tq.qsize())]
    assert payloads and all(isinstance(p, dict) for p in payloads)
    chunks = [TrajectoryChunk.from_payload(p) for p in payloads]
    assert chunks[-1].behavior_policy_version == 3
    assert rq.qsize() > 0


def test_learner_queues_carry_only_numpy():
    traj, wq, mq, ckq = CheckedQueue(), CheckedQueue(maxsize=1), CheckedQueue(), CheckedQueue()
    for version in range(4):
        traj.put(chunk_payload(T=4, version=version))
    stop = threading.Event()
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a",
        algorithm_factory=lambda: APPO(TinyModel(), AlgorithmConfig(), device="cpu"),
        trajectory_queue=traj, weight_queues=[wq],
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"),
        stop_event=stop, metrics_queue=mq, checkpoint_queue=ckq, checkpoint_interval=1,
    ))
    thread.start()
    deadline = time.monotonic() + 60
    while ckq.qsize() < 2 and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    ckpt = ckq.get_nowait()
    assert all(isinstance(v, np.ndarray) for v in ckpt["state_dict"].values())
    assert isinstance(wq.get_nowait(), WeightPayload)
    assert mq.qsize() >= 1


def test_refresh_commands_carry_numpy_checkpoints(tmp_path):
    from colosseum.coordinator.coordinator import Coordinator
    from colosseum.core.registry import build_model
    from colosseum.launcher import Launcher

    data = load_config(REPO / "configs/examples/tic_tac_toe.yaml").model_dump()
    data["checkpoint"]["dir"] = str(tmp_path / "ckpt")
    data["training"]["phase"] = "self_play"
    data["self_play"]["latest_prob"] = 0.0
    data["rollout"]["envs_per_worker"] = 4
    data["metrics"]["use_wandb"] = False
    cfg = ColosseumConfig(**data)
    coord = Coordinator(cfg)
    coord.agent_pool.register_trainable("agent_0")
    coord.checkpoint_manager.save("agent_0", 50, build_model(cfg).state_dict())
    coord.setup_matchmaker("agent_0")
    cq = CheckedQueue()
    Launcher(cfg)._refresh_worker_matches(coord, ["agent_0"], [cq], [{"agent_0": set()}])
    cmd = cq.get_nowait()
    state_dict = cmd.new_checkpoints["agent_0"]["ckpt_v50"]
    assert all(isinstance(v, np.ndarray) for v in state_dict.values())
```

- [ ] **Step 14: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/contract/test_no_tensors_in_queues.py -v`
Expected:
- `test_worker_queues_carry_only_numpy` fails with `TypeError: torch.Tensor at queued item.observations ...` (the worker still puts `TrajectoryChunk` objects);
- `test_learner_queues_carry_only_numpy` fails with `_queue.Empty`: the learner thread died with `TypeError: torch.Tensor at queued item.state_dict[...]`, which pytest reports as an unhandled thread exception;
- `test_refresh_commands_carry_numpy_checkpoints` fails with `TypeError: torch.Tensor at queued item.new_checkpoints[...]`.

- [ ] **Step 15: Send chunk payloads from the worker and load numpy weights in the loop**

In `src/colosseum/worker/rollout_worker.py`, replace `send_chunk` inside `rollout_worker_process` with:

```python
    def send_chunk(chunk: TrajectoryChunk) -> None:
        payload = chunk.to_payload()
        q = trajectory_queues[chunk.agent_id]
        while not stop_event.is_set():
            try:
                q.put(payload, timeout=1.0)
                return
            except queue.Full:
                continue
```

In `src/colosseum/worker/rollout_loop.py`, run `grep -n "load_state_dict" src/colosseum/worker/rollout_loop.py` and change every hit:
- the weight-sync load of a `WeightPayload`, `X.load_state_dict(payload.state_dict)`, becomes `X.load_state_dict(payload.to_torch_state_dict())`;
- every load of a checkpoint state dict that comes from `checkpoint_state_dicts_by_agent` or `WorkerCommand.new_checkpoints`, `X.load_state_dict(sd)`, becomes `X.load_state_dict(state_dict_from_numpy(sd))`;
- add `state_dict_from_numpy` to the module's import from `colosseum.core.types`.

- [ ] **Step 16: Make the learner put only numpy on its queues**

In `src/colosseum/learner/learner.py`, add to the import block:

```python
from colosseum.core.ipc import from_numpy_tree, to_numpy_tree
from colosseum.core.types import TrajectoryChunk, WeightPayload, state_dict_from_numpy, state_dict_to_numpy
```

(replace the existing `from colosseum.core.types import ...` line).

Replace the whole `if resume_state is not None:` block in `learner_process` with:

```python
    if resume_state is not None:
        # resume_state is numpy (it crossed the process boundary as a Process argument).
        algorithm.model.load_state_dict(state_dict_from_numpy(resume_state["state_dict"]))
        if resume_state.get("optimizer_state") is not None and hasattr(algorithm, "_optimizer"):
            algorithm._optimizer.load_state_dict(from_numpy_tree(resume_state["optimizer_state"]))
        train_step = resume_state.get("policy_version", 0)
        if hasattr(algorithm, "_policy_version"):
            algorithm._policy_version = train_step
        logger.info(f"Learner [{agent_id}]: resumed from checkpoint at step {train_step}")
```

Replace the checkpoint block (from `pv = algorithm.policy_version` through the `except Full:` branch that logs "checkpoint queue full") with:

```python
        pv = algorithm.policy_version
        if (checkpoint_queue is not None
                and checkpoint_interval > 0
                and pv > 0
                and pv % checkpoint_interval == 0):
            optimizer_state = getattr(algorithm, "optimizer_state_dict", None)
            try:
                checkpoint_queue.put_nowait({
                    "policy_version": pv,
                    "state_dict": state_dict_to_numpy(algorithm.model.state_dict()),
                    "optimizer_state": (
                        None if optimizer_state is None else to_numpy_tree(optimizer_state)
                    ),
                })
                logger.info(f"Learner [{agent_id}]: sent checkpoint at version {pv}")
            except Full:
                logger.warning(f"Learner [{agent_id}]: checkpoint queue full, skipping v{pv}")
```

Replace `_collect_chunks` and `_push_weights` with:

```python
def _collect_chunks(
    queue: mp.Queue,
    batch_size: int,
    timeout: float = 1.0,
) -> list[TrajectoryChunk]:
    """Collect up to batch_size chunk payloads from the queue and decode them.

    Waits up to `timeout` seconds for the first chunk, then collects
    remaining chunks non-blocking up to batch_size.
    """
    chunks: list[TrajectoryChunk] = []
    try:
        chunks.append(TrajectoryChunk.from_payload(queue.get(timeout=timeout)))
    except Empty:
        return chunks
    while len(chunks) < batch_size:
        try:
            chunks.append(TrajectoryChunk.from_payload(queue.get_nowait()))
        except Empty:
            break
    return chunks


def _push_weights(
    algorithm: BaseAlgorithm,
    agent_id: str,
    weight_queues: list,
) -> None:
    """Push current model weights (a numpy WeightPayload) to all worker weight queues."""
    payload = WeightPayload.from_model(agent_id, algorithm.policy_version, algorithm.model)
    for wq in weight_queues:
        try:
            wq.put_nowait(payload)
        except Full:
            pass  # workers will get the next weight update
```

- [ ] **Step 17: Convert checkpoints and resume states in the launcher and the distributed learner**

In `src/colosseum/launcher.py`, add to the import block:

```python
from colosseum.core.ipc import from_numpy_tree, to_numpy_tree
from colosseum.core.types import MatchConfig, state_dict_from_numpy, state_dict_to_numpy
```

(replace the existing `from colosseum.core.types import MatchConfig`).

Make these changes:
1. In `_derive_worker_configs`, change `agent_ckpts[ckpt_id] = sd` to `agent_ckpts[ckpt_id] = state_dict_to_numpy(sd)`. Workers receive these dicts as Process arguments and inside `WorkerCommand`s.
2. In `_monitor_loop`, replace the `coordinator.maybe_save_checkpoint(...)` call with:

```python
                        ckpt_id = coordinator.maybe_save_checkpoint(
                            agent_id=aid,
                            policy_version=ckpt_data["policy_version"],
                            state_dict=state_dict_from_numpy(ckpt_data["state_dict"]),
                            optimizer_state=from_numpy_tree(ckpt_data.get("optimizer_state")),
                        )
```

3. In `Launcher.launch`, change `resume_state = _resolve_resume_state(cfg, aid, coordinator)` to `resume_state = to_numpy_tree(_resolve_resume_state(cfg, aid, coordinator))`.

In `src/colosseum/distributed.py`, add `from colosseum.core.ipc import from_numpy_tree` to the imports and change `from colosseum.core.types import WeightPayload` to `from colosseum.core.types import WeightPayload, state_dict_from_numpy`. In `_drain_checkpoints`, replace the `coordinator_ckpt.save(...)` call with:

```python
                coordinator_ckpt.save(
                    agent_id=agent_id,
                    policy_version=data["policy_version"],
                    state_dict=state_dict_from_numpy(data["state_dict"]),
                    optimizer_state=from_numpy_tree(data.get("optimizer_state")),
                )
```

Also change the `GRPCTrajectorySink` docstring's first line to `"""``.put``-compatible sink that ships chunk payloads to a learner over gRPC.`.

- [ ] **Step 18: Update tests that read worker queues or fake checkpoint snapshots**

Run: `grep -rn "trajectory_queues\[.*\]\.get\|checkpoint_queues\[" tests --include=*.py`

- For every test that reads a chunk produced by a real worker, decode it. Before Part A these were `test_worker_produces_chunks` in `tests/test_integration.py` and `test_worker_multi_agent_routing` in `tests/test_multi_agent.py`. For example:

```python
            chunk = TrajectoryChunk.from_payload(trajectory_queues[agent_id].get(timeout=10))
```

and add `from colosseum.core.types import TrajectoryChunk` to that test.
- In `test_monitor_loop_per_agent_checkpoint_queues` (it puts a fake snapshot on a checkpoint queue), replace `state_dict = {"weight": torch.randn(3, 3)}` with:

```python
        state_dict = {"weight": np.random.randn(3, 3).astype(np.float32)}
```

and add `import numpy as np` to that test module.

- [ ] **Step 19: Run the contract test and the touched tests**

Run: `.venv/bin/python -m pytest tests/contract/test_no_tensors_in_queues.py tests/unit/test_payloads.py tests/unit/test_serialization.py tests/integration/test_worker_threads.py -v`
Expected: PASS.

- [ ] **Step 20: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: all pass.

- [ ] **Step 21: Commit**

```bash
git add src/colosseum/core/ipc.py src/colosseum/core/types.py src/colosseum/transport \
        src/colosseum/weight_store/shared_memory.py src/colosseum/worker src/colosseum/learner/learner.py \
        src/colosseum/launcher.py src/colosseum/distributed.py tests
git commit -m "fix: pass only numpy payloads between processes and over gRPC"
```

---

### Task T2.3: Newest-wins weight delivery (`put_latest` / `drain_latest`)

Spec block 2 "Веса, newest-wins" (R2-05, R3-10). Today the per-worker weight queue has `maxsize=2` and the learner drops a push when the queue is full. A worker draining it therefore loads one of the *first two* pushes since its last sync (repro: the learner publishes v1..v50 and the worker loads v2). After this task:
- each worker mailbox is a `maxsize=1` queue;
- the learner replaces a stale item;
- the worker takes the newest item.

**Files:**
- Modify: `src/colosseum/core/ipc.py` (add `put_latest`, `drain_latest`)
- Modify: `src/colosseum/learner/learner.py` (`_push_weights`)
- Modify: `src/colosseum/worker/rollout_worker.py` (`poll_weights`)
- Modify: `src/colosseum/launcher.py` (`_WEIGHT_QUEUE_SIZE = 1`)
- Modify: `tests/dataflow_helpers.py` (add `publish_versions`)
- Test: `tests/unit/test_ipc_latest.py`

**Interfaces:**
- Consumes: `WeightPayload` (numpy, T2.2); `_push_weights(algorithm, agent_id, weight_queues)` (learner, T2.2); `rollout_worker_process` (T2.1/T2.2).
- Produces:
  - `colosseum.core.ipc.put_latest(q, item, timeout: float = 1.0) -> bool`. It replaces a stale item, waits at most `timeout` seconds and returns `False` only if delivery failed. This extends the contract signature; see Contract notes.
  - `colosseum.core.ipc.drain_latest(q) -> Any | None`
  - `tests/dataflow_helpers.publish_versions(q, n, done)` (spawn target)

- [ ] **Step 1: Add the spawn publisher to `tests/dataflow_helpers.py`**

Change the ipc import line in the import block to:

```python
from colosseum.core.ipc import assert_no_tensors, put_latest
```

add `from colosseum.core.types import WeightPayload` to the import block, and append to the end of the module:

```python
def publish_versions(q, n: int, done) -> None:
    """Spawn target: publish WeightPayload v1..vn into a size-1 mailbox as fast as possible."""
    for version in range(1, n + 1):
        put_latest(q, WeightPayload("a", version, {"w": np.full(64, version, dtype=np.float32)}))
    done.set()
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_ipc_latest.py`:

```python
"""Newest-wins weight mailboxes (T2.3, regression for R2-05 / R3-10)."""
import multiprocessing as mp
import queue
import time

import numpy as np

from colosseum.core.ipc import drain_latest, put_latest
from colosseum.learner.learner import _push_weights
from tests.dataflow_helpers import TinyModel, publish_versions


def test_put_latest_replaces_the_stale_item():
    q = queue.Queue(maxsize=1)
    for version in range(5):
        assert put_latest(q, version)
    assert drain_latest(q) == 4
    assert drain_latest(q) is None


def test_drain_latest_returns_newest_of_many():
    q = queue.Queue()
    for version in range(3):
        q.put(version)
    assert drain_latest(q) == 2
    assert q.empty()


def test_put_latest_never_blocks_without_a_consumer():
    q = mp.get_context("spawn").Queue(maxsize=1)
    start = time.monotonic()
    for version in range(200):
        put_latest(q, version)
    assert time.monotonic() - start < 5.0
    got, deadline = None, time.monotonic() + 5.0
    while got != 199 and time.monotonic() < deadline:
        item = drain_latest(q)   # the last put may still be in the feeder thread
        if item is not None:
            got = item
        time.sleep(0.01)
    assert got == 199


def test_push_weights_leaves_only_the_newest_payload():
    class _Algo:
        def __init__(self):
            self.model = TinyModel()
            self.policy_version = 0

    algo, mailbox = _Algo(), queue.Queue(maxsize=1)
    for version in range(1, 6):
        algo.policy_version = version
        _push_weights(algo, "a", [mailbox])
    assert mailbox.get_nowait().policy_version == 5


def test_worker_gets_newest_weights_after_a_burst_of_publishes():
    """A learner process publishes v1..v50 between two syncs; the worker drains v50."""
    ctx = mp.get_context("spawn")
    mailbox, done = ctx.Queue(maxsize=1), ctx.Event()
    publisher = ctx.Process(target=publish_versions, args=(mailbox, 50, done))
    publisher.start()
    last = None
    try:
        assert done.wait(60)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            item = drain_latest(mailbox)
            if item is not None:
                last = item
            if last is not None and last.policy_version == 50:
                break
            time.sleep(0.01)
    finally:
        publisher.join(timeout=10)
    assert last is not None and last.policy_version == 50
    assert np.all(last.state_dict["w"] == 50)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_ipc_latest.py -v`
Expected: collection error `ImportError: cannot import name 'put_latest' from 'colosseum.core.ipc'`. The helper module import fails too.

- [ ] **Step 4: Add `put_latest` and `drain_latest` to `src/colosseum/core/ipc.py`**

Replace the import block with:

```python
from __future__ import annotations

import dataclasses
import queue
import time
from typing import Any, Optional

import numpy as np
import torch
```

Insert right after the import block (before `def _tensor_to_numpy`):

```python
def put_latest(q: Any, item: Any, timeout: float = 1.0) -> bool:
    """Publish ``item`` into a ``maxsize=1`` queue, replacing a stale item.

    Never blocks for longer than ``timeout`` seconds. Returns ``False`` only if
    the item could not be delivered within ``timeout`` (sustained contention).

    ``mp.Queue.put`` returns before its feeder thread has written the item into
    the pipe, so ``get_nowait`` can miss an item that is still in flight and the
    queue keeps reporting ``Full``. A short blocking ``get`` waits for that flush.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            pass
        try:
            q.get(timeout=0.01)  # evict the stale item (it may still be in flight)
        except queue.Empty:
            pass
        if time.monotonic() >= deadline:
            return False


def drain_latest(q: Any) -> Optional[Any]:
    """Return the newest item available in ``q`` (draining older ones), or None."""
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            return latest
```

- [ ] **Step 5: Use the mailbox in the learner, the worker and the launcher**

In `src/colosseum/learner/learner.py`, change the ipc import to `from colosseum.core.ipc import from_numpy_tree, put_latest, to_numpy_tree` and replace `_push_weights` with:

```python
def _push_weights(
    algorithm: BaseAlgorithm,
    agent_id: str,
    weight_queues: list,
) -> None:
    """Publish the current weights to every worker mailbox (newest wins)."""
    payload = WeightPayload.from_model(agent_id, algorithm.policy_version, algorithm.model)
    for wq in weight_queues:
        put_latest(wq, payload)
```

If `Full` is no longer used in `learner.py`, keep it anyway: the metrics and checkpoint `put_nowait` calls still catch `Full`.

In `src/colosseum/worker/rollout_worker.py`:
- add `from colosseum.core.ipc import drain_latest` to the import block;
- replace `poll_weights` inside `rollout_worker_process` with:

```python
    def poll_weights(agent_id: str) -> Optional[WeightPayload]:
        return drain_latest(weight_queues[agent_id])
```

In `src/colosseum/launcher.py`, change `_WEIGHT_QUEUE_SIZE = 2` to:

```python
_WEIGHT_QUEUE_SIZE = 1  # newest-wins mailbox per (agent, worker); see core.ipc.put_latest
```

`GRPCWeightSink` (distributed learner) only has `put_nowait`, which never raises `Full`, so `put_latest` publishes on the first try. `GRPCWeightSource` raises `queue.Empty` when the store has no newer version, so `drain_latest` works on it unchanged.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_ipc_latest.py -v`
Expected: PASS (5 tests).

- [ ] **Step 7: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/colosseum/core/ipc.py src/colosseum/learner/learner.py src/colosseum/worker/rollout_worker.py \
        src/colosseum/launcher.py tests/dataflow_helpers.py tests/unit/test_ipc_latest.py
git commit -m "fix: deliver the newest weights to workers through size-1 mailboxes"
```

---

### Task T2.4: Learner trains on exactly `batch_chunks` chunks (`collect_batch`)

Spec block 2 "Батч лёрнера" (R3-11, R1-11). Today `_collect_chunks` waits for the first chunk and then takes whatever is in the queue, so the effective batch is often one chunk. `collect_batch` blocks, in 0.5 s slices that check `stop_event`, until exactly `batch_size` chunks have arrived. It returns `None` on stop. Partial batches are never trained on.

**Files:**
- Modify: `src/colosseum/learner/learner.py` (add `collect_batch`, remove `_collect_chunks`, change the loop)
- Modify: `tests/dataflow_helpers.py` (add `RecordingAlgorithm`)
- Test: `tests/unit/test_collect_batch.py`

**Interfaces:**
- Consumes:
  - `TrajectoryChunk.from_payload` (T2.2);
  - `learner_process` keyword arguments `agent_id, algorithm_factory, trajectory_queue, weight_queues, config, stop_event, metrics_queue, checkpoint_queue, checkpoint_interval` (current signature);
  - `chunk_payload`, `CheckedQueue` (helpers, T2.2).
- Produces:
  - `colosseum.learner.learner.collect_batch(q, batch_size: int, stop_event, poll_interval: float = 0.5) -> list[TrajectoryChunk] | None` (contract). It raises `TypeError` for a queue item that is not a payload dict.
  - `tests/dataflow_helpers.RecordingAlgorithm(start_version: int = 0)`: a duck-typed algorithm with `model`, `network`, `policy_version`, `set_progress`, `train_step`, `compute_loss`, `create_replay_buffer`, `is_off_policy`. Its records: `batches: list[int]`, `behavior_versions: list[list[int]]`, `progress_at_train: list[float]`.

- [ ] **Step 1: Add `RecordingAlgorithm` to `tests/dataflow_helpers.py`**

Append to the end of the module:

```python
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
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_collect_batch.py`:

```python
"""collect_batch blocks for a full batch; the learner never trains on partial batches (T2.4)."""
import queue
import threading
import time

import pytest

from colosseum.core.config import LearnerConfig
from colosseum.learner.learner import collect_batch, learner_process
from tests.dataflow_helpers import CheckedQueue, RecordingAlgorithm, chunk_payload


def _collect_in_thread(q, batch_size, stop):
    out = {}
    thread = threading.Thread(
        target=lambda: out.setdefault("batch", collect_batch(q, batch_size, stop, poll_interval=0.05))
    )
    thread.start()
    return thread, out


def test_collect_batch_blocks_until_the_batch_is_full():
    q, stop = queue.Queue(), threading.Event()
    for version in range(2):
        q.put(chunk_payload(version=version))
    thread, out = _collect_in_thread(q, 3, stop)
    time.sleep(0.3)
    assert thread.is_alive()                       # 2 of 3 chunks: still waiting
    q.put(chunk_payload(version=2))
    thread.join(timeout=5)
    assert [c.behavior_policy_version for c in out["batch"]] == [0, 1, 2]


def test_collect_batch_returns_none_on_stop():
    q, stop = queue.Queue(), threading.Event()
    q.put(chunk_payload())
    thread, out = _collect_in_thread(q, 3, stop)
    time.sleep(0.2)
    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive() and out["batch"] is None


def test_collect_batch_rejects_non_payload_items():
    q = queue.Queue()
    q.put(object())
    with pytest.raises(TypeError):
        collect_batch(q, 1, threading.Event(), poll_interval=0.05)


def test_learner_trains_only_on_full_batches():
    algo = RecordingAlgorithm()
    traj, stop = CheckedQueue(), threading.Event()
    for version in range(7):
        traj.put(chunk_payload(version=version))
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, trajectory_queue=traj,
        weight_queues=[CheckedQueue(maxsize=1)], config=LearnerConfig(batch_chunks=3),
        stop_event=stop,
    ))
    thread.start()
    deadline = time.monotonic() + 10
    while len(algo.batches) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.3)        # the 7th chunk alone must not trigger a train step
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert algo.batches == [3, 3]
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_collect_batch.py -v`
Expected: collection error `ImportError: cannot import name 'collect_batch' from 'colosseum.learner.learner'`.

- [ ] **Step 4: Implement `collect_batch` and use it in the learner loop**

In `src/colosseum/learner/learner.py`, add `import queue` to the import block, change the typing import to `from typing import Any, Callable, Optional`, and replace the whole `_collect_chunks` function with:

```python
def collect_batch(
    q: Any,
    batch_size: int,
    stop_event: Any,
    poll_interval: float = 0.5,
) -> Optional[list[TrajectoryChunk]]:
    """Block until exactly ``batch_size`` chunk payloads arrived; decode them.

    Waits in ``poll_interval`` slices and returns None as soon as ``stop_event``
    is set (a partial batch is dropped: no training on incomplete batches).
    """
    chunks: list[TrajectoryChunk] = []
    while len(chunks) < batch_size:
        if stop_event.is_set():
            return None
        try:
            payload = q.get(timeout=poll_interval)
        except queue.Empty:
            continue
        if not isinstance(payload, dict):
            raise TypeError(
                f"trajectory queue item must be a chunk payload dict, got {type(payload).__name__}"
            )
        chunks.append(TrajectoryChunk.from_payload(payload))
    return chunks
```

In `learner_process`, replace:

```python
        # Collect a batch of chunks
        chunks = _collect_chunks(
            trajectory_queue,
            batch_size=config.batch_chunks,
            timeout=1.0,
        )

        if not chunks:
            continue
```

with:

```python
        # Block until exactly batch_chunks chunks arrived (None: stop requested).
        chunks = collect_batch(trajectory_queue, config.batch_chunks, stop_event)
        if chunks is None:
            break
```

Run: `grep -rn "_collect_chunks" src tests` and replace any remaining test use with `collect_batch`. A test that asserted partial-batch behaviour encodes the removed bug (R3-11): delete it.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_collect_batch.py tests/contract/test_no_tensors_in_queues.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/colosseum/learner/learner.py tests/dataflow_helpers.py tests/unit/test_collect_batch.py tests
git commit -m "fix: learner trains only on full batches of batch_chunks chunks"
```

---

### Task T2.5: Global env-step budget, `set_progress`, LR by progress, stop by budget

Spec block 2 "Общий бюджет" (R6-11, R1-11).
- `training.total_timesteps` counts env steps summed over all workers.
- Workers add their locally accumulated steps to one shared `mp.Value("q")` about every 0.5 s.
- The main process sets `stop_event` when the counter reaches the budget.
- Before every train step, each learner calls `algorithm.set_progress(min(1, counter / total_timesteps))`.
- APPO's learning rate becomes a function of progress (constant / linear / cosine). `setup_lr_schedule` and `total_train_steps` disappear.
- Distributed mode has no shared counter: the learner uses `consumed_samples / total_timesteps` and stops itself at the budget.
- Kickstart decay stays in train steps.

**Files:**
- Modify: `src/colosseum/core/ipc.py` (add `SharedCounter`, `BatchedCounter`)
- Modify: `src/colosseum/algorithms/base.py` (`set_progress`)
- Modify: `src/colosseum/algorithms/appo.py` (LR by progress; remove `setup_lr_schedule` and the scheduler)
- Modify: `src/colosseum/core/config.py` (`AlgorithmConfig.lr_schedule` description)
- Replace: `src/colosseum/learner/learner.py` (whole file)
- Modify: `src/colosseum/worker/rollout_worker.py` (`env_step_counter`), `src/colosseum/worker/rollout_loop.py` (call `add_env_steps`)
- Modify: `src/colosseum/launcher.py`, `src/colosseum/distributed.py`
- Modify: `tests/dataflow_helpers.py` (add `Collected`, `make_loop`, `add_to_counter`)
- Test: `tests/unit/test_budget.py`, `tests/unit/test_lr_progress.py`, `tests/integration/test_budget_stop.py`
- Update: the APPO LR tests and every caller of `setup_lr_schedule` / `total_train_steps` / `_worker_target(total_timesteps=...)`

**Interfaces:**
- Consumes:
  - `collect_batch` (T2.4);
  - `put_latest` (T2.3);
  - `RecordingAlgorithm`, `CheckedQueue`, `chunk_payload`, `GridStepEnv`, `EnvFactory`, `TinyModel`, `make_tiny_model` (helpers);
  - `LoopIO.add_env_steps` (contract).
  - Part A detail: the attribute where `RolloutLoop.__init__` stores its `io` argument. The code below uses `self._io`; T2.6 rewrites the loop with exactly that name.
- Produces:
  - `colosseum.core.ipc.SharedCounter(ctx=None)` with `.add(n)` and `.value` (contract); `BatchedCounter(counter, interval=0.5)` with `.add(n)` and `.flush()`
  - `BaseAlgorithm.set_progress(progress: float) -> None` (contract; default no-op)
  - `APPO.set_progress(progress)`, which sets the optimizer LR from `lr_schedule`. `APPO.setup_lr_schedule` is removed.
  - `learner_process(*, agent_id, algorithm_factory, trajectory_queue, weight_queues, config, stop_event, metrics_queue=None, checkpoint_queue=None, checkpoint_interval=0, resume_state=None, progress_counter=None, total_timesteps=0)`. This is the contract signature without `run_dir`, which T6.2 adds. Metrics gain `progress` and `consumed_samples`.
  - `rollout_worker_process(..., env_step_counter: SharedCounter | None = None, ...)`
  - `RolloutLoop.step()` calls `io.add_env_steps(num_envs)` every step
  - `launcher._worker_target(..., env_step_counter: SharedCounter | None = None, ...)` replaces `total_timesteps`; `launcher._learner_target(..., progress_counter=None, total_timesteps=0, num_learners=1)` replaces `total_train_steps`
  - `Launcher.env_steps_done -> int` (property)
  - `tests/dataflow_helpers.py`: `Collected` (fields `chunks`, `results`, `env_steps`, `weights`, `commands`; method `io() -> LoopIO`), `make_loop(env_factory, model_factory, *, agent_ids=("a",), num_envs=1, chunk_length=4, **kwargs) -> (RolloutLoop, Collected)`, `add_to_counter(counter, n)`

- [ ] **Step 1: Add the in-memory loop driver and the counter spawn target to `tests/dataflow_helpers.py`**

Add to the import block:

```python
from dataclasses import dataclass, field
from typing import Callable

from colosseum.core.types import MatchResult, TrajectoryChunk, WorkerCommand
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop
```

(`Optional` and `WeightPayload` are already imported; merge the names into the existing `typing` and `colosseum.core.types` import lines.)

Append to the end of the module:

```python
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
        def poll_weights(agent_id: str) -> Optional[WeightPayload]:
            pending = self.weights.get(agent_id)
            return pending.pop(0) if pending else None

        def poll_command() -> Optional[WorkerCommand]:
            return self.commands.pop(0) if self.commands else None

        return LoopIO(send_chunk=self.chunks.append, poll_weights=poll_weights,
                      report_result=self.results.append, poll_command=poll_command,
                      add_env_steps=self.env_steps.append)


def make_loop(env_factory: Callable[[], BaseEnv], model_factory: Callable[[], PolicyModel],
              *, agent_ids=("a",), num_envs: int = 1, chunk_length: int = 4,
              **kwargs) -> tuple[RolloutLoop, Collected]:
    """Build a RolloutLoop with in-memory I/O; every agent uses ``model_factory``.

    Defaults: weight_sync_interval=0.0 (sync every step), seed=0.
    """
    col = Collected()
    loop = RolloutLoop(
        worker_id=0, env_fn=env_factory, num_envs=num_envs, chunk_length=chunk_length,
        agent_ids=list(agent_ids), model_factories={a: model_factory for a in agent_ids},
        io=col.io(), weight_sync_interval=kwargs.pop("weight_sync_interval", 0.0),
        seed=kwargs.pop("seed", 0), **kwargs,
    )
    return loop, col


def add_to_counter(counter, n: int) -> None:
    """Spawn target: add 1 to a SharedCounter n times."""
    for _ in range(n):
        counter.add(1)
```

- [ ] **Step 2: Write the failing budget tests**

Create `tests/unit/test_budget.py`:

```python
"""Global env-step budget: shared counter, worker reporting, learner progress (T2.5)."""
import multiprocessing as mp
import threading
import time

import pytest
import torch

from colosseum.core.config import LearnerConfig
from colosseum.core.ipc import BatchedCounter, SharedCounter
from colosseum.learner.learner import learner_process
from colosseum.worker.rollout_worker import rollout_worker_process
from tests.dataflow_helpers import (
    CheckedQueue,
    EnvFactory,
    GridStepEnv,
    RecordingAlgorithm,
    add_to_counter,
    chunk_payload,
    make_loop,
    make_tiny_model,
)


@pytest.fixture
def restore_torch_threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def test_shared_counter_sums_across_spawned_processes():
    ctx = mp.get_context("spawn")
    counter = SharedCounter(ctx)
    procs = [ctx.Process(target=add_to_counter, args=(counter, 500)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    assert counter.value == 1000


def test_batched_counter_flushes_on_interval_and_on_demand():
    counter = SharedCounter()
    lazy = BatchedCounter(counter, interval=3600.0)
    lazy.add(5)
    lazy.add(7)
    assert counter.value == 0
    lazy.flush()
    assert counter.value == 12
    eager = BatchedCounter(counter, interval=0.0)
    eager.add(3)
    assert counter.value == 15


def test_rollout_loop_reports_env_steps_every_step():
    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(3,)), make_tiny_model, num_envs=2)
    for _ in range(5):
        loop.step()
    loop.close()
    assert col.env_steps == [2] * 5


def test_worker_adds_its_env_steps_to_the_shared_counter(restore_torch_threads):
    counter = SharedCounter()
    rollout_worker_process(
        worker_id=0, env_fn=EnvFactory(GridStepEnv, lengths=(3,)), num_envs=2, chunk_length=4,
        agent_ids=["a"], model_factories={"a": make_tiny_model},
        trajectory_queues={"a": CheckedQueue()}, weight_queues={"a": CheckedQueue(maxsize=1)},
        stop_event=threading.Event(), max_env_steps=20, env_step_counter=counter,
    )
    assert counter.value == 20


def _run_learner_until(algo, payloads, until, batch_chunks, **kwargs):
    traj, stop = CheckedQueue(), threading.Event()
    for payload in payloads:
        traj.put(payload)
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, trajectory_queue=traj,
        weight_queues=[CheckedQueue(maxsize=1)], config=LearnerConfig(batch_chunks=batch_chunks),
        stop_event=stop, **kwargs,
    ))
    thread.start()
    deadline = time.monotonic() + 10
    while not until() and time.monotonic() < deadline:
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_learner_progress_comes_from_the_shared_counter():
    algo, counter = RecordingAlgorithm(), SharedCounter()
    counter.add(50)
    _run_learner_until(algo, [chunk_payload()], lambda: len(algo.batches) >= 1, batch_chunks=1,
                       progress_counter=counter, total_timesteps=100)
    assert algo.progress_at_train == [0.5]


def test_learner_progress_is_capped_at_one():
    algo, counter = RecordingAlgorithm(), SharedCounter()
    counter.add(300)
    _run_learner_until(algo, [chunk_payload()], lambda: len(algo.batches) >= 1, batch_chunks=1,
                       progress_counter=counter, total_timesteps=100)
    assert algo.progress_at_train == [1.0]


def test_without_a_counter_progress_is_consumed_samples_and_the_learner_stops_itself():
    algo, traj = RecordingAlgorithm(), CheckedQueue()
    for _ in range(6):
        traj.put(chunk_payload(T=4))
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, trajectory_queue=traj,
        weight_queues=[CheckedQueue(maxsize=1)], config=LearnerConfig(batch_chunks=2),
        stop_event=threading.Event(), total_timesteps=16,
    ))
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()              # stopped by itself at 16 consumed samples
    assert algo.progress_at_train == [0.5, 1.0]
```

Create `tests/unit/test_lr_progress.py`:

```python
"""APPO learning rate as a function of training progress (T2.5)."""
import pytest

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig, LRSchedule
from colosseum.core.types import TrajectoryChunk
from tests.dataflow_helpers import TinyModel, chunk_payload


def _lr(algo) -> float:
    return algo._optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize("schedule, progress, expected", [
    (LRSchedule.CONSTANT, 0.7, 1e-3),
    (LRSchedule.LINEAR, 0.0, 1e-3),
    (LRSchedule.LINEAR, 0.25, 0.75e-3),
    (LRSchedule.LINEAR, 1.0, 0.0),
    (LRSchedule.LINEAR, 1.5, 0.0),
    (LRSchedule.COSINE, 0.5, 0.5e-3),
    (LRSchedule.COSINE, 1.0, 0.0),
])
def test_lr_is_a_function_of_progress(schedule, progress, expected):
    algo = APPO(TinyModel(), AlgorithmConfig(lr_schedule=schedule, learning_rate=1e-3), device="cpu")
    assert _lr(algo) == pytest.approx(1e-3)
    algo.set_progress(progress)
    assert _lr(algo) == pytest.approx(expected, abs=1e-12)


def test_train_step_does_not_change_the_lr():
    algo = APPO(TinyModel(), AlgorithmConfig(lr_schedule=LRSchedule.LINEAR, learning_rate=1e-3),
                device="cpu")
    algo.set_progress(0.25)
    algo.train_step([TrajectoryChunk.from_payload(chunk_payload(T=4)) for _ in range(2)])
    assert _lr(algo) == pytest.approx(0.75e-3)


def test_step_based_lr_schedule_is_gone():
    assert not hasattr(APPO, "setup_lr_schedule")
```

Create `tests/integration/test_budget_stop.py`:

```python
"""The launcher stops every process once the global env-step budget is reached (T2.5)."""
import time
from pathlib import Path

from colosseum.core.config import ColosseumConfig, load_config
from colosseum.launcher import Launcher

REPO = Path(__file__).resolve().parents[2]


def test_launch_stops_at_the_env_step_budget(tmp_path):
    data = load_config(REPO / "configs/examples/tic_tac_toe.yaml").model_dump()
    data["training"]["total_timesteps"] = 400
    data["rollout"]["num_workers"] = 1
    data["rollout"]["envs_per_worker"] = 2
    data["rollout"]["chunk_length"] = 8
    data["learner"]["batch_chunks"] = 2
    data["learner"]["queue_size"] = 16
    data["learner"]["device"] = "cpu"
    data["metrics"]["use_wandb"] = False
    data["checkpoint"]["dir"] = str(tmp_path / "ckpt")
    launcher = Launcher(ColosseumConfig(**data))
    start = time.monotonic()
    launcher.launch()
    assert launcher.env_steps_done >= 400
    assert time.monotonic() - start < 120
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_budget.py tests/unit/test_lr_progress.py tests/integration/test_budget_stop.py -v`
Expected:
- `test_budget.py`: collection error `ImportError: cannot import name 'BatchedCounter' from 'colosseum.core.ipc'`;
- `test_lr_progress.py`: `AttributeError: 'APPO' object has no attribute 'set_progress'`, and `test_step_based_lr_schedule_is_gone` fails;
- `test_budget_stop.py`: `AttributeError: 'Launcher' object has no attribute 'env_steps_done'`.

- [ ] **Step 4: Add the counters to `src/colosseum/core/ipc.py`**

Replace the import block with:

```python
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import queue
import time
from typing import Any, Optional

import numpy as np
import torch
```

Insert right after `drain_latest`:

```python
class SharedCounter:
    """A process-shared 64-bit counter (``mp.Value("q")``).

    Must be handed to child processes as a ``Process`` argument (not through a
    queue), like any ``multiprocessing`` synchronized value.
    """

    def __init__(self, ctx: Optional[mp.context.BaseContext] = None) -> None:
        ctx = ctx if ctx is not None else mp.get_context()
        self._val = ctx.Value("q", 0)

    def add(self, n: int) -> None:
        if n:
            with self._val.get_lock():
                self._val.value += int(n)

    @property
    def value(self) -> int:
        with self._val.get_lock():
            return int(self._val.value)


class BatchedCounter:
    """Accumulates increments locally and flushes them into a SharedCounter.

    Workers call :meth:`add` every env step; the shared (locked) counter is
    touched at most once per ``interval`` seconds. Call :meth:`flush` on exit.
    """

    def __init__(self, counter: SharedCounter, interval: float = 0.5) -> None:
        self._counter = counter
        self._interval = interval
        self._pending = 0
        self._last_flush = time.monotonic()

    def add(self, n: int) -> None:
        self._pending += int(n)
        if time.monotonic() - self._last_flush >= self._interval:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            self._counter.add(self._pending)
            self._pending = 0
        self._last_flush = time.monotonic()
```

- [ ] **Step 5: `set_progress` in the algorithm base class and APPO's LR by progress**

In `src/colosseum/algorithms/base.py`, add this method to `class BaseAlgorithm` (after the `policy_version` property):

```python
    def set_progress(self, progress: float) -> None:
        """Share (0..1) of the global env-step budget consumed so far.

        The learner calls this before every train step. The default ignores it;
        algorithms with schedules (APPO's learning rate) override it.
        """
        return None
```

In `src/colosseum/algorithms/appo.py`:
1. Add `import math` to the import block.
2. Delete the whole `setup_lr_schedule` method.
3. In `__init__`, replace

```python
        # LR scheduler (created in setup_lr_schedule when total_steps is known)
        self._lr_scheduler = None
```

with

```python
        # The LR is a function of training progress (share of the global env-step
        # budget), set by the learner through set_progress() before every train step.
        self._progress = 0.0
        self.set_progress(0.0)
```

4. In `train_step`, delete

```python
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

```

5. Add these methods right after the `policy_version` property:

```python
    def set_progress(self, progress: float) -> None:
        """Set the share (0..1) of the global env-step budget consumed so far.

        The optimizer LR follows ``config.lr_schedule``: constant, linear decay to
        0 at progress 1, or cosine decay to 0 at progress 1. (Kickstart decay
        stays in train steps.)
        """
        self._progress = min(1.0, max(0.0, float(progress)))
        lr = self._lr_at(self._progress)
        for group in self._optimizer.param_groups:
            group["lr"] = lr

    def _lr_at(self, progress: float) -> float:
        base = self._config.learning_rate
        if self._config.lr_schedule == LRSchedule.LINEAR:
            return base * (1.0 - progress)
        if self._config.lr_schedule == LRSchedule.COSINE:
            return base * 0.5 * (1.0 + math.cos(math.pi * progress))
        return base
```

In `src/colosseum/core/config.py`, change the `lr_schedule` field of `AlgorithmConfig` to:

```python
    lr_schedule: LRSchedule = Field(
        default=LRSchedule.LINEAR,
        description="LR schedule over training progress = env steps so far / training.total_timesteps.",
    )
```

- [ ] **Step 6: Replace `src/colosseum/learner/learner.py`**

```python
"""Learner process: receives trajectory chunks, trains the model, pushes weights.

Each learner owns one trainable agent. It:
1. collects exactly ``batch_chunks`` chunk payloads from its queue;
2. sets the algorithm's progress (share of the env-step budget) and trains;
3. publishes new weights to every worker's size-1 mailbox (newest wins);
4. sends checkpoint snapshots and metrics to the main process.

Everything it puts on a queue is numpy/primitives (no torch tensors).

Progress: in local mode ``progress = min(1, env_step_counter / total_timesteps)``
from the shared counter that workers increment. In distributed mode there is no
shared counter: ``progress = consumed_samples / total_timesteps``, where
``consumed_samples`` counts this learner's transitions, and the learner stops by
itself once ``consumed_samples >= total_timesteps``.
"""

from __future__ import annotations

import logging
import queue
from queue import Full
from typing import Any, Callable, Optional

import torch

from colosseum.algorithms.base import BaseAlgorithm
from colosseum.core.config import LearnerConfig
from colosseum.core.ipc import SharedCounter, from_numpy_tree, put_latest, to_numpy_tree
from colosseum.core.types import (
    TrajectoryChunk,
    WeightPayload,
    state_dict_from_numpy,
    state_dict_to_numpy,
)

logger = logging.getLogger(__name__)


def collect_batch(
    q: Any,
    batch_size: int,
    stop_event: Any,
    poll_interval: float = 0.5,
) -> Optional[list[TrajectoryChunk]]:
    """Block until exactly ``batch_size`` chunk payloads arrived; decode them.

    Waits in ``poll_interval`` slices and returns None as soon as ``stop_event``
    is set (a partial batch is dropped: no training on incomplete batches).
    """
    chunks: list[TrajectoryChunk] = []
    while len(chunks) < batch_size:
        if stop_event.is_set():
            return None
        try:
            payload = q.get(timeout=poll_interval)
        except queue.Empty:
            continue
        if not isinstance(payload, dict):
            raise TypeError(
                f"trajectory queue item must be a chunk payload dict, got {type(payload).__name__}"
            )
        chunks.append(TrajectoryChunk.from_payload(payload))
    return chunks


def _progress(
    progress_counter: Optional[SharedCounter],
    consumed_samples: int,
    total_timesteps: int,
) -> float:
    if total_timesteps <= 0:
        return 0.0
    done = progress_counter.value if progress_counter is not None else consumed_samples
    return min(1.0, done / total_timesteps)


def learner_process(
    *,
    agent_id: str,
    algorithm_factory: Callable[[], BaseAlgorithm],
    trajectory_queue: Any,
    weight_queues: list,
    config: LearnerConfig,
    stop_event: Any,
    metrics_queue: Any = None,
    checkpoint_queue: Any = None,
    checkpoint_interval: int = 0,
    resume_state: Optional[dict] = None,
    progress_counter: Optional[SharedCounter] = None,
    total_timesteps: int = 0,
) -> None:
    """Main learner loop (see module docstring)."""
    logger.info(f"Learner [{agent_id}]: starting on device={config.device}")
    algorithm = algorithm_factory()

    train_step = 0
    if resume_state is not None:
        train_step = _apply_resume_state(algorithm, resume_state, agent_id)

    total_chunks_received = 0
    consumed_samples = 0
    replay_buffer = algorithm.create_replay_buffer(config.queue_size * 4)
    _push_weights(algorithm, agent_id, weight_queues)

    while not stop_event.is_set():
        if (progress_counter is None and total_timesteps > 0
                and consumed_samples >= total_timesteps):
            logger.info(f"Learner [{agent_id}]: consumed {consumed_samples} samples; done")
            break

        chunks = collect_batch(trajectory_queue, config.batch_chunks, stop_event)
        if chunks is None:
            break
        total_chunks_received += len(chunks)
        consumed_samples += sum(c.chunk_length for c in chunks)

        progress = _progress(progress_counter, consumed_samples, total_timesteps)
        algorithm.set_progress(progress)

        if replay_buffer is not None:
            for chunk in chunks:
                replay_buffer.add(chunk)
            if len(replay_buffer) < config.batch_chunks:
                continue
            metrics = algorithm.train_step(replay_buffer.sample(config.batch_chunks))
        else:
            metrics = algorithm.train_step(chunks)
        train_step += 1
        metrics["progress"] = float(progress)

        if train_step % config.weight_push_interval == 0:
            _push_weights(algorithm, agent_id, weight_queues)

        pv = algorithm.policy_version
        if (checkpoint_queue is not None and checkpoint_interval > 0
                and pv > 0 and pv % checkpoint_interval == 0):
            _send_checkpoint(algorithm, agent_id, checkpoint_queue)

        if metrics_queue is not None:
            metrics["agent_id"] = agent_id
            metrics["train_step"] = train_step
            metrics["chunks_received"] = total_chunks_received
            metrics["consumed_samples"] = consumed_samples
            try:
                metrics_queue.put_nowait(metrics)
            except Full:
                pass  # metrics are best-effort

        if train_step % 10 == 0:
            logger.info(
                f"Learner [{agent_id}]: step={train_step}, chunks={total_chunks_received}, "
                f"progress={progress:.3f}, loss={metrics.get('total_loss', 0.0):.4f}"
            )

    logger.info(f"Learner [{agent_id}]: finished. Total train_steps={train_step}")


def _apply_resume_state(algorithm: BaseAlgorithm, resume_state: dict, agent_id: str) -> int:
    """Load weights (and optimizer state, if present) from a numpy ``resume_state``."""
    algorithm.model.load_state_dict(state_dict_from_numpy(resume_state["state_dict"]))
    if resume_state.get("optimizer_state") is not None and hasattr(algorithm, "_optimizer"):
        algorithm._optimizer.load_state_dict(from_numpy_tree(resume_state["optimizer_state"]))
    step = int(resume_state.get("policy_version", 0))
    if hasattr(algorithm, "_policy_version"):
        algorithm._policy_version = step
    logger.info(f"Learner [{agent_id}]: resumed from checkpoint at step {step}")
    return step


def _send_checkpoint(algorithm: BaseAlgorithm, agent_id: str, checkpoint_queue: Any) -> None:
    pv = algorithm.policy_version
    optimizer_state = getattr(algorithm, "optimizer_state_dict", None)
    try:
        checkpoint_queue.put_nowait({
            "policy_version": pv,
            "state_dict": state_dict_to_numpy(algorithm.model.state_dict()),
            "optimizer_state": None if optimizer_state is None else to_numpy_tree(optimizer_state),
        })
        logger.info(f"Learner [{agent_id}]: sent checkpoint at version {pv}")
    except Full:
        logger.warning(f"Learner [{agent_id}]: checkpoint queue full, skipping v{pv}")


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _push_weights(algorithm: BaseAlgorithm, agent_id: str, weight_queues: list) -> None:
    """Publish the current weights to every worker mailbox (newest wins)."""
    payload = WeightPayload.from_model(agent_id, algorithm.policy_version, algorithm.model)
    for wq in weight_queues:
        put_latest(wq, payload)
```

- [ ] **Step 7: Report env steps from the worker**

In `src/colosseum/worker/rollout_loop.py`, run `grep -n "add_env_steps" src/colosseum/worker/rollout_loop.py`. If `RolloutLoop.step()` does not call it yet, insert these lines immediately before `step()`'s `return` statement. Here `n` stands for the value `step()` returns, the number of env steps just taken; if Part A's `step()` returns an expression, assign it to `n` first. If Part A stores the `io` argument under another attribute name, use that name instead of `self._io`.

```python
        if self._io.add_env_steps is not None:
            self._io.add_env_steps(n)
```

In `src/colosseum/worker/rollout_worker.py`:
1. Change the ipc import to `from colosseum.core.ipc import BatchedCounter, SharedCounter, drain_latest`.
2. Add the parameter `env_step_counter: Optional[SharedCounter] = None,` to `rollout_worker_process`, right after `max_env_steps: int = 0,`.
3. Replace the `io = LoopIO(...)` statement with:

```python
    counter = BatchedCounter(env_step_counter) if env_step_counter is not None else None
    io = LoopIO(
        send_chunk=send_chunk,
        poll_weights=poll_weights,
        report_result=report_result if results_queue is not None else None,
        poll_command=poll_command if command_queue is not None else None,
        add_env_steps=counter.add if counter is not None else None,
    )
```

4. Replace the `try:` / `finally:` block that runs the loop with:

```python
    try:
        loop.run(should_stop=stop_event.is_set, max_env_steps=max_env_steps)
    finally:
        if counter is not None:
            counter.flush()
        loop.close()
        # Detach feeder threads of queues this worker produced to, so undrained
        # items (e.g. chunks a stopped learner never consumed) cannot block exit.
        for q in [*trajectory_queues.values(), results_queue]:
            if q is not None and hasattr(q, "cancel_join_thread"):
                q.cancel_join_thread()
```

- [ ] **Step 8: Wire the counter through the launcher**

In `src/colosseum/launcher.py`:

1. Change the ipc import to `from colosseum.core.ipc import SharedCounter, from_numpy_tree, to_numpy_tree`.
2. In `_worker_target`, replace the parameter `total_timesteps: int = 0,` with `env_step_counter: Optional[SharedCounter] = None,`. In its `rollout_worker_process(...)` call, replace `max_env_steps=total_timesteps,` with `env_step_counter=env_step_counter,`.
3. In `_learner_target`, delete the parameter `total_train_steps: int,` and add, right before `num_learners: int = 1,`:

```python
    progress_counter: Optional[SharedCounter] = None,
    total_timesteps: int = 0,
```

   In its `learner_process(...)` call, replace `total_train_steps=total_train_steps,` with:

```python
        progress_counter=progress_counter,
        total_timesteps=total_timesteps,
```

4. In `Launcher.__init__`, add after `self._all_queues: list[mp.Queue] = []`:

```python
        # Env steps taken by all workers together (spec block 2: global budget).
        self._env_step_counter = SharedCounter()
```

   and add this property to `class Launcher`, right after `__init__`:

```python
    @property
    def env_steps_done(self) -> int:
        """Env steps taken by all workers so far (the global budget counter)."""
        return self._env_step_counter.value
```

5. In `Launcher.launch`, delete the block that starts with `# Compute total train steps (same for all agents)` and ends with `total_train_steps = max(1, cfg.training.total_timesteps // env_steps_per_train_step,)`. In the learner `kwargs=dict(...)`, replace `total_train_steps=total_train_steps,` with:

```python
                    progress_counter=self._env_step_counter,
                    total_timesteps=cfg.training.total_timesteps,
```

   In the worker `kwargs=dict(...)`, replace `total_timesteps=cfg.training.total_timesteps // cfg.rollout.num_workers,` with `env_step_counter=self._env_step_counter,`.

6. In `_monitor_loop`, insert right before the final `time.sleep(0.5)` of the `while` body:

```python
            # Global env-step budget (spec block 2): all workers together have
            # taken training.total_timesteps env steps -> stop everything.
            if self.env_steps_done >= self._config.training.total_timesteps:
                logger.info(
                    f"Env-step budget reached ({self.env_steps_done} >= "
                    f"{self._config.training.total_timesteps}); stopping."
                )
                self._stop_event.set()
                break
```

- [ ] **Step 9: Distributed learner progress**

In `src/colosseum/distributed.py`, `run_distributed_learner`:
- delete the two lines that compute `env_steps_per_train_step` and `total_train_steps`;
- in the `learner_process(...)` call, replace `total_train_steps=total_train_steps,` with:

```python
            progress_counter=None,
            total_timesteps=config.training.total_timesteps,
```

Append this paragraph to the module docstring (before the closing `"""`):

```
Budget and progress: there is no shared env-step counter across machines. Each
distributed learner uses progress = consumed_samples / training.total_timesteps
(its own transitions, which drives the LR schedule) and stops by itself once
consumed_samples >= total_timesteps. Each worker stops after
total_timesteps / num_workers env steps. These numbers differ from the local
mode budget (env steps summed over workers); a single semantics comes with the
hub in SP5.
```

- [ ] **Step 10: Update callers of the removed APIs**

Run: `grep -rn "setup_lr_schedule\|total_train_steps\|_lr_scheduler" src tests scripts examples --include=*.py`
- Delete the three APPO LR tests that call `algo.setup_lr_schedule(...)`: `test_linear_lr_decay`, `test_cosine_lr_decay`, `test_constant_lr`. Before Part A they were in `tests/test_appo.py`. `tests/unit/test_lr_progress.py` replaces them.
- No other hit may remain in `src/`.

Run: `grep -rn "_worker_target" tests --include=*.py`
- In every `_worker_target` call, delete the `total_timesteps=...` keyword. These tests stop the worker through `stop_event`.

- [ ] **Step 11: Run the new tests**

Run: `.venv/bin/python -m pytest tests/unit/test_budget.py tests/unit/test_lr_progress.py tests/integration/test_budget_stop.py -v`
Expected: PASS.

- [ ] **Step 12: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: all pass. Launcher end-to-end tests now finish when the global budget is reached instead of after a fixed number of train steps.

- [ ] **Step 13: Commit**

```bash
git add src/colosseum/core/ipc.py src/colosseum/algorithms/base.py src/colosseum/algorithms/appo.py \
        src/colosseum/core/config.py src/colosseum/learner/learner.py src/colosseum/worker \
        src/colosseum/launcher.py src/colosseum/distributed.py tests
git commit -m "feat: global env-step budget drives stopping and the LR schedule"
```

---

### Task T2.6: Agent-owned buffers with parking; behavior version = first transition; policy-lag metric

Spec block 2 "Буферы без потерь при смене матча" and "Версии политики" (R2-12, R2-18, R1-10).
- Buffers belong to agents, not slots.
- When a slot at an episode boundary moves to another agent or stops collecting, its partial buffer is parked in the worker's pool for that agent. The buffer's last transition is already closed by `done`. The next slot that starts collecting for the agent takes the parked buffer first. No data is discarded on re-assignment.
- A chunk's `behavior_policy_version` is the weight version at its first transition.
- The learner logs `policy_lag_mean` and `policy_lag_max` per batch.

This task rewrites `rollout_loop.py` in full on top of the new `worker/slots.py`, with the same public API. T3.1–T3.4 then replace individual methods of this file. This task keeps today's transition semantics:
- every slot runs inference;
- only slots with `info["active"]` absent or true record a transition;
- a full buffer is sealed right after `env.step` with a separate bootstrap forward.

Block 3 changes these semantics.

**Files:**
- Create: `src/colosseum/worker/slots.py`
- Replace: `src/colosseum/worker/rollout_loop.py` (whole file)
- Modify: `src/colosseum/worker/rollout_worker.py` (import `LATEST_NETWORK_ID` from the loop)
- Modify: `src/colosseum/learner/learner.py` (policy-lag metrics)
- Modify: `tests/dataflow_helpers.py` (add `ProbeModel`, `slot_transitions`)
- Test: `tests/unit/test_buffer_pool.py`, `tests/contract/test_parking.py`, `tests/unit/test_policy_lag.py`
- Update: tests of the old `RolloutBuffer.append` / `_build_chunk` API; characterization tests that assert discard-on-re-assignment or seal-time versions

**Interfaces:**
- Consumes:
  - `act(model, obs, state, action_mask=None, deterministic=False) -> ActOutput`, `PolicyModel.initial_state(1)`, `PolicyModel.step(obs, state, action_mask)` (T1.2);
  - `cat_batch`, `slice_batch`, `tree_map`, `State` (T1.1);
  - `state_dict_from_numpy`, `TrajectoryChunk` (T2.2);
  - `LoopIO` (contract);
  - `player_outcomes(total_rewards, terminal_infos, num_players)` (`core/outcomes.py`);
  - `ActionSpec.from_space(space)` with `action_shape`, `numpy_dtype`, `flat_mask_size`, `components`, `flatten_mask`;
  - `VectorEnv` / `SubprocessVectorEnv`;
  - `Collected`, `make_loop`, `EnvFactory`, `GridStepEnv`, `TinyModel`, `RecordingAlgorithm`, `CheckedQueue`, `chunk_payload` (helpers).
  - Part A detail: if `tests/contract/harness.py` or Part A's tests read private attributes of `RolloutLoop`, Step 9 maps them to the names below.
- Produces:
  - `colosseum.worker.slots.RolloutBuffer(chunk_length, obs_shape, action_shape, action_dtype, mask_size=0)` with `steps`, `is_full`, `last_done`, `initial_state`, `first_version`, `begin_chunk(initial_state, policy_version)`, `open(obs, action, log_prob, value, mask, reward=0.0)`, `add_reward(r)`, `mark_done()`, `build_chunk(agent_id, bootstrap_value) -> TrajectoryChunk` (does not reset), `reset()`
  - `colosseum.worker.slots.BufferPool(chunk_length, obs_shape, action_shape, action_dtype, mask_size=0)` with `acquire(agent_id)`, `park(agent_id, buf)`, `parked_count()`, `parked_transitions(agent_id)` (contract plus constructor and `parked_transitions`)
  - `colosseum.worker.slots.SlotTrack(buffer, has_open=False, pending_reward=0.0, state=None)` (contract)
  - `colosseum.worker.rollout_loop.LATEST_NETWORK_ID = "latest"`
  - `RolloutLoop.stats` keys: `chunks_sent`, `env_steps`, `parked_buffers`, `recorded_transitions`, `recorded_transitions/<agent>`, `buffered_transitions/<agent>`
  - `RolloutLoop` private structure used by T3.1–T3.4:
    - `_vec_env`, `_models[agent][net_id]`, `_policy_versions`;
    - `_slot_agent_map`, `_slot_network_map`, `_collect_mask`, `_tracks[e][p]: SlotTrack`, `_pool: BufferPool`;
    - `_ep_rewards`, `_ep_lengths`, `_ep_counter`;
    - methods `_infer`, `_open_transitions`, `_after_env_step`, `_end_episode`, `_seal`, `_report_result`, `_resolve_model`, `_initial_state`, `_apply_pending_assignment`.
  - Learner metrics `policy_lag_mean`, `policy_lag_max`
  - `tests/dataflow_helpers.py`: `ProbeModel(obs_coeffs=(0,0,0,0), state_coef=0.0, stateful=False, num_actions=NUM_ACTIONS)` with `.calls: list[int]`; `slot_transitions(chunks) -> dict[(env_id, player), list[dict]]`

- [ ] **Step 1: Add `ProbeModel` and `slot_transitions` to `tests/dataflow_helpers.py`**

Append to the end of the module:

```python
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
```

- [ ] **Step 2: Write the failing unit tests for buffers and the pool**

Create `tests/unit/test_buffer_pool.py`:

```python
"""RolloutBuffer and BufferPool: agent-owned buffers with parking (T2.6)."""
import numpy as np
import pytest
import torch

from colosseum.worker.slots import BufferPool, RolloutBuffer, SlotTrack


def _buf(T=3, mask_size=0):
    return RolloutBuffer(chunk_length=T, obs_shape=(2,), action_shape=(), action_dtype=np.int64,
                         mask_size=mask_size)


def test_open_add_reward_done_and_build_chunk():
    b = _buf(T=3, mask_size=2)
    b.begin_chunk({"h": torch.ones(1, 4)}, policy_version=5)
    b.open(np.array([1, 2]), 1, -0.5, 0.25, np.array([True, False]), reward=0.5)
    b.add_reward(1.0)
    b.open(np.array([3, 4]), 0, -0.1, 0.5, None)
    b.mark_done()
    b.open(np.array([5, 6]), 1, -0.2, 0.75, None, reward=2.0)
    assert b.is_full
    c = b.build_chunk("a", bootstrap_value=0.9)
    assert c.agent_id == "a" and c.behavior_policy_version == 5
    assert c.rewards.tolist() == [1.5, 0.0, 2.0]
    assert c.dones.tolist() == [False, True, False] and c.dones.dtype == torch.bool
    assert c.actions.tolist() == [1, 0, 1]
    assert float(c.bootstrap_value) == pytest.approx(0.9)
    assert c.action_masks.tolist() == [[True, False], [True, True], [True, True]]
    assert torch.equal(c.initial_state["h"], torch.ones(1, 4))
    b.reset()
    assert b.steps == 0 and b.initial_state is None


def test_chunk_has_no_masks_when_none_were_given():
    b = _buf(T=1, mask_size=3)
    b.begin_chunk(None, 0)
    b.open(np.zeros(2), 0, 0.0, 0.0, None)
    assert b.build_chunk("a", 0.0).action_masks is None


def test_build_chunk_requires_full_buffer_and_begin_requires_empty():
    b = _buf(T=2)
    b.begin_chunk(None, 0)
    b.open(np.zeros(2), 0, 0.0, 0.0, None)
    with pytest.raises(RuntimeError):
        b.build_chunk("a", 0.0)
    with pytest.raises(RuntimeError):
        b.begin_chunk(None, 1)


def test_pool_prefers_parked_buffers_of_the_same_agent():
    pool = BufferPool(chunk_length=3, obs_shape=(2,), action_shape=(), action_dtype=np.int64)
    a1 = pool.acquire("a")
    a1.begin_chunk(None, 0)
    a1.open(np.zeros(2), 0, 0.0, 0.0, None)
    a1.mark_done()
    pool.park("a", a1)
    assert pool.parked_count() == 1 and pool.parked_transitions("a") == 1
    assert pool.acquire("b") is not a1
    assert pool.acquire("a") is a1
    assert pool.parked_count() == 0


def test_pool_recycles_empty_buffers_and_rejects_open_episodes():
    pool = BufferPool(chunk_length=3, obs_shape=(2,), action_shape=(), action_dtype=np.int64)
    empty = pool.acquire("a")
    pool.park("a", empty)
    assert pool.parked_count() == 0
    assert pool.acquire("b") is empty
    open_buf = pool.acquire("a")
    open_buf.begin_chunk(None, 0)
    open_buf.open(np.zeros(2), 0, 0.0, 0.0, None)
    with pytest.raises(RuntimeError, match="not done"):
        pool.park("a", open_buf)


def test_slot_track_defaults():
    t = SlotTrack(buffer=None)
    assert (t.has_open, t.pending_reward, t.state) == (False, 0.0, None)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_buffer_pool.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.worker.slots'`.

- [ ] **Step 4: Create `src/colosseum/worker/slots.py`**

```python
"""Per-slot rollout bookkeeping: agent-owned buffers with parking (spec blocks 2-3).

A :class:`RolloutBuffer` holds up to ``chunk_length`` consecutive transitions of
ONE agent. Each collecting (env, seat) slot holds one buffer exclusively, so two
seats of the same agent in one env write to different buffers.

When a slot stops collecting for an agent at an episode boundary (match
re-assignment), its partial buffer is *parked* in the worker's
:class:`BufferPool` for that agent; the next slot that starts collecting for the
agent takes the parked buffer first. Parked buffers always end with a
transition whose ``done`` flag is set, and the model state is reset on
``done``, so a chunk may mix episodes from different envs without any padding
and without losing data (R2-12).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from colosseum.core.types import TrajectoryChunk
from colosseum.networks.state import State, tree_map


class RolloutBuffer:
    """Pre-allocated numpy storage for one chunk of one agent's transitions.

    A transition is *opened* when the slot acts (:meth:`open`), receives rewards
    while it is open (:meth:`add_reward`) and may be marked as the last one of
    its episode (:meth:`mark_done`).
    """

    def __init__(
        self,
        chunk_length: int,
        obs_shape: tuple,
        action_shape: tuple,
        action_dtype: Any,
        mask_size: int = 0,
    ) -> None:
        T = chunk_length
        self.chunk_length = chunk_length
        self._obs = np.zeros((T, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((T, *action_shape), dtype=action_dtype)
        self._log_probs = np.zeros(T, dtype=np.float32)
        self._rewards = np.zeros(T, dtype=np.float32)
        self._dones = np.zeros(T, dtype=np.bool_)
        self._values = np.zeros(T, dtype=np.float32)
        self._masks: Optional[np.ndarray] = (
            np.ones((T, mask_size), dtype=np.bool_) if mask_size > 0 else None
        )
        self._has_masks = False
        self._cursor = 0
        self.initial_state: State = None
        self.first_version: int = 0

    # -- state ---------------------------------------------------------
    @property
    def steps(self) -> int:
        return self._cursor

    @property
    def is_full(self) -> bool:
        return self._cursor >= self.chunk_length

    @property
    def last_done(self) -> bool:
        """True if the buffer is empty or its last transition ends an episode."""
        return self._cursor == 0 or bool(self._dones[self._cursor - 1])

    # -- writing -------------------------------------------------------
    def begin_chunk(self, initial_state: State, policy_version: int) -> None:
        """Record the model state before the chunk's first transition and the
        policy version that produced it. Call only when ``steps == 0``."""
        if self._cursor != 0:
            raise RuntimeError("begin_chunk() on a non-empty buffer")
        self.initial_state = tree_map(lambda t: t.detach().clone(), initial_state)
        self.first_version = int(policy_version)

    def open(
        self,
        obs: np.ndarray,
        action: Any,
        log_prob: float,
        value: float,
        mask: Optional[np.ndarray],
        reward: float = 0.0,
    ) -> None:
        """Append a new (open) transition with an initial reward."""
        if self.is_full:
            raise RuntimeError("open() on a full buffer; seal it first")
        i = self._cursor
        self._obs[i] = obs
        self._actions[i] = action
        self._log_probs[i] = log_prob
        self._values[i] = value
        self._rewards[i] = reward
        self._dones[i] = False
        if self._masks is not None:
            if mask is not None:
                self._masks[i] = mask
                self._has_masks = True
            else:
                self._masks[i] = True
        self._cursor += 1

    def add_reward(self, reward: float) -> None:
        """Add ``reward`` to the most recent transition."""
        self._rewards[self._cursor - 1] += reward

    def mark_done(self) -> None:
        """Mark the most recent transition as the last one of its episode."""
        self._dones[self._cursor - 1] = True

    # -- sealing -------------------------------------------------------
    def build_chunk(self, agent_id: str, bootstrap_value: float) -> TrajectoryChunk:
        """Copy the (full) buffer into a TrajectoryChunk. Does not reset."""
        if not self.is_full:
            raise RuntimeError(f"build_chunk() on a partial buffer ({self._cursor}/{self.chunk_length})")
        return TrajectoryChunk(
            agent_id=agent_id,
            observations=torch.from_numpy(self._obs.copy()),
            actions=torch.from_numpy(self._actions.copy()),
            action_log_probs=torch.from_numpy(self._log_probs.copy()),
            rewards=torch.from_numpy(self._rewards.copy()),
            dones=torch.from_numpy(self._dones.copy()),
            values=torch.from_numpy(self._values.copy()),
            bootstrap_value=torch.tensor(float(bootstrap_value), dtype=torch.float32),
            behavior_policy_version=self.first_version,
            initial_state=self.initial_state,
            action_masks=(
                torch.from_numpy(self._masks.copy())
                if self._masks is not None and self._has_masks else None
            ),
        )

    def reset(self) -> None:
        self._cursor = 0
        self._has_masks = False
        self.initial_state = None
        self.first_version = 0


class BufferPool:
    """Per-worker pool of RolloutBuffers keyed by agent.

    ``acquire(agent)`` returns a parked partial buffer of that agent if there is
    one, else an empty recycled buffer, else a new one. ``park(agent, buf)``
    stores a buffer released by a slot at an episode boundary.
    """

    def __init__(
        self,
        chunk_length: int,
        obs_shape: tuple,
        action_shape: tuple,
        action_dtype: Any,
        mask_size: int = 0,
    ) -> None:
        self._spec = (chunk_length, tuple(obs_shape), tuple(action_shape), action_dtype, mask_size)
        self._parked: dict[str, list[RolloutBuffer]] = defaultdict(list)
        self._free: list[RolloutBuffer] = []

    def acquire(self, agent_id: str) -> RolloutBuffer:
        parked = self._parked.get(agent_id)
        if parked:
            return parked.pop(0)
        if self._free:
            return self._free.pop()
        return RolloutBuffer(*self._spec)

    def park(self, agent_id: str, buf: RolloutBuffer) -> None:
        if buf.steps == 0:
            buf.reset()
            self._free.append(buf)
            return
        if not buf.last_done:
            raise RuntimeError(
                f"cannot park a buffer of {agent_id!r} whose last transition is not done"
            )
        if buf.is_full:
            raise RuntimeError(f"cannot park a full buffer of {agent_id!r}; seal it first")
        self._parked[agent_id].append(buf)

    def parked_count(self) -> int:
        return sum(len(v) for v in self._parked.values())

    def parked_transitions(self, agent_id: str) -> int:
        return sum(b.steps for b in self._parked.get(agent_id, []))


@dataclass
class SlotTrack:
    """Rollout state of one (env, seat) slot."""

    buffer: Optional[RolloutBuffer]
    has_open: bool = False
    pending_reward: float = 0.0
    state: State = None
```

- [ ] **Step 5: Run the unit tests**

Run: `.venv/bin/python -m pytest tests/unit/test_buffer_pool.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Write the failing contract test for parking and versions**

Create `tests/contract/test_parking.py`:

```python
"""Parking partial buffers on match re-assignment loses no transitions (T2.6, R2-12).

The real RolloutLoop runs in-process; WorkerCommands flip slot agents and
collect flags every few steps. Every recorded transition must be either in a
sent chunk or still in a buffer (a slot's or a parked one), exactly once, and
chunks must stay sequences of one agent's transitions with episode boundaries.
"""
from colosseum.core.types import WeightPayload, WorkerCommand, state_dict_to_numpy
from tests.dataflow_helpers import EnvFactory, GridStepEnv, ProbeModel, TinyModel, make_loop

M1 = WorkerCommand(slot_agent_map=[["a", "b"], ["b", "a"]],
                   slot_network_map=[["latest", "latest"], ["latest", "latest"]],
                   collect_mask=[[True, True], [True, False]])
M2 = WorkerCommand(slot_agent_map=[["b", "b"], ["a", "a"]],
                   slot_network_map=[["latest", "latest"], ["latest", "latest"]],
                   collect_mask=[[True, False], [True, True]])


def test_reassignment_parks_buffers_without_losing_transitions():
    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(3, 5)), ProbeModel,
                          agent_ids=("a", "b"), num_envs=2, chunk_length=4,
                          slot_agent_map=[["a", "a"], ["b", "b"]])
    max_parked = 0
    for i in range(120):
        if i % 4 == 0:
            col.commands.append(M1 if (i // 4) % 2 == 0 else M2)
        loop.step()
        max_parked = max(max_parked, loop.stats["parked_buffers"])
    assert max_parked > 0, "the scenario must exercise parking"
    stats = loop.stats
    for aid in ("a", "b"):
        sent = sum(c.chunk_length for c in col.chunks if c.agent_id == aid)
        assert sent + stats[f"buffered_transitions/{aid}"] == stats[f"recorded_transitions/{aid}"]
    seen = set()
    for chunk in col.chunks:
        obs = chunk.observations.numpy()
        for i in range(chunk.chunk_length):
            key = tuple(obs[i].astype(int).tolist())       # (env_id, ep, t, player)
            assert key not in seen, f"transition {key} sent twice"
            seen.add(key)
            if i + 1 < chunk.chunk_length and not bool(chunk.dones[i]):
                # inside an episode the next row is the same slot's next step
                assert (obs[i + 1][0], obs[i + 1][1], obs[i + 1][3]) == (obs[i][0], obs[i][1], obs[i][3])
                assert obs[i + 1][2] == obs[i][2] + 1


def test_behavior_version_is_the_version_at_the_first_transition():
    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(50,)), TinyModel, num_envs=1, chunk_length=4)
    loop.step()                                  # transition t=0 recorded with version 0
    col.weights["a"] = [WeightPayload.from_model("a", 7, TinyModel())]
    loop.step()                                  # the sync at the end of this step loads v7
    for _ in range(10):
        loop.step()
    versions = [c.behavior_policy_version for c in col.chunks if int(c.observations[0, 3]) == 0]
    assert versions[0] == 0
    assert versions[1:] and all(v == 7 for v in versions[1:])


def test_command_checkpoint_is_loaded_and_used_from_the_next_episode():
    created = []

    def model_factory():
        model = ProbeModel()
        created.append(model)
        return model

    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(2,)), model_factory, num_envs=1, chunk_length=2)
    col.commands.append(WorkerCommand(
        slot_agent_map=[["a", "a"]], slot_network_map=[["latest", "ckpt_v9"]],
        collect_mask=[[True, False]],
        new_checkpoints={"a": {"ckpt_v9": state_dict_to_numpy(ProbeModel().state_dict())}},
    ))
    loop.step()
    assert len(created) == 2 and created[1].calls == []   # loaded, not used mid-episode
    loop.step()                                          # episode 0 ends -> assignment applied
    for _ in range(4):
        loop.step()
    assert created[1].calls == [1] * 4                   # the checkpoint plays seat 1
    late = [c for c in col.chunks if int(c.observations[0, 1]) >= 1]
    assert late and all(int(c.observations[0, 3]) == 0 for c in late)   # seat 1 no longer collects
```

- [ ] **Step 7: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/contract/test_parking.py -v`
Expected: `test_reassignment_parks_buffers_without_losing_transitions` fails with `KeyError: 'parked_buffers'` (or `'buffered_transitions/a'`). `test_behavior_version_is_the_version_at_the_first_transition` fails if Part A stamps the version at seal time. The checkpoint test may already pass.

- [ ] **Step 8: Replace `src/colosseum/worker/rollout_loop.py`**

```python
"""In-process rollout loop: vectorized envs, batched inference, per-slot transitions.

All I/O goes through :class:`LoopIO` callbacks, so the loop runs unchanged in a
worker process (``rollout_worker_process``) and in-process in tests.

Buffers are owned by agents (``worker/slots.py``): each collecting slot holds
one buffer, and on match re-assignment a slot's partial buffer is parked for
its agent instead of being discarded (spec block 2). A chunk's
``behavior_policy_version`` is the version at its FIRST transition.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from colosseum.core.action_spec import ActionSpec
from colosseum.core.outcomes import player_outcomes
from colosseum.core.types import (
    MatchResult,
    TrajectoryChunk,
    WeightPayload,
    WorkerCommand,
    state_dict_from_numpy,
)
from colosseum.envs.base_env import BaseEnv
from colosseum.envs.vec_env import VectorEnv
from colosseum.networks.model import PolicyModel, act
from colosseum.networks.state import State, cat_batch, slice_batch
from colosseum.worker.slots import BufferPool, RolloutBuffer, SlotTrack

logger = logging.getLogger(__name__)

LATEST_NETWORK_ID = "latest"


@dataclass
class LoopIO:
    """All I/O of the rollout loop goes through these callbacks."""

    send_chunk: Callable[[TrajectoryChunk], None]
    poll_weights: Callable[[str], Optional[WeightPayload]]
    report_result: Optional[Callable[[MatchResult], None]] = None
    poll_command: Optional[Callable[[], Optional[WorkerCommand]]] = None
    add_env_steps: Optional[Callable[[int], None]] = None


class RolloutLoop:
    """One worker's rollout loop over ``num_envs`` vectorized envs."""

    def __init__(
        self,
        *,
        worker_id: int,
        env_fn: Callable[[], BaseEnv],
        num_envs: int,
        chunk_length: int,
        agent_ids: list[str],
        model_factories: dict[str, Callable[[], PolicyModel]],
        io: LoopIO,
        gamma: float = 0.99,
        weight_sync_interval: float = 5.0,
        slot_agent_map: Optional[list[list[str]]] = None,
        slot_network_map: Optional[list[list[str]]] = None,
        collect_mask: Optional[list[list[bool]]] = None,
        checkpoint_state_dicts_by_agent: Optional[dict[str, dict[str, dict[str, np.ndarray]]]] = None,
        seed: Optional[int] = None,
        vec_env_kind: str = "sync",
        subproc_workers: Optional[int] = None,
    ) -> None:
        self.worker_id = worker_id
        self._io = io
        self._agent_ids = list(agent_ids)
        self._model_factories = model_factories
        self._chunk_length = chunk_length
        self._weight_sync_interval = weight_sync_interval
        self._gamma = float(gamma)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if vec_env_kind == "subprocess":
            from colosseum.envs.subproc_vec_env import SubprocessVectorEnv
            self._vec_env = SubprocessVectorEnv(env_fn, num_envs, num_workers=subproc_workers)
        elif vec_env_kind == "sync":
            self._vec_env = VectorEnv(env_fn, num_envs)
        else:
            raise ValueError(f"unknown vec_env_kind {vec_env_kind!r}")
        self._num_envs = num_envs
        self._num_players = self._vec_env.num_players
        self._action_spec = ActionSpec.from_space(self._vec_env.action_space)
        E, P = self._num_envs, self._num_players

        # Model pool: agent -> network id -> model ("latest" + frozen checkpoints).
        self._models: dict[str, dict[str, PolicyModel]] = {}
        self._policy_versions: dict[str, int] = {aid: 0 for aid in self._agent_ids}
        for aid in self._agent_ids:
            latest = model_factories[aid]()
            latest.eval()
            self._models[aid] = {LATEST_NETWORK_ID: latest}
        for aid, by_id in (checkpoint_state_dicts_by_agent or {}).items():
            for ckpt_id, sd in by_id.items():
                self._add_checkpoint(aid, ckpt_id, sd)
        self._warned_missing: set[tuple[str, str]] = set()

        # Live match assignment (re-assigned per env at episode boundaries).
        if slot_agent_map is None:
            slot_agent_map = [[self._agent_ids[0]] * P for _ in range(E)]
        if collect_mask is None:
            collect_mask = [[True] * P for _ in range(E)]
        if slot_network_map is None:
            # Non-collecting slots default to the agent's first checkpoint, if any.
            ckpts = checkpoint_state_dicts_by_agent or {}
            slot_network_map = [
                [
                    LATEST_NETWORK_ID
                    if collect_mask[e][p] or not ckpts.get(slot_agent_map[e][p])
                    else next(iter(ckpts[slot_agent_map[e][p]]))
                    for p in range(P)
                ]
                for e in range(E)
            ]
        self._slot_agent_map = [list(r) for r in slot_agent_map]
        self._slot_network_map = [list(r) for r in slot_network_map]
        self._collect_mask = [[bool(c) for c in r] for r in collect_mask]
        self._pending_maps: Optional[tuple[list, list, list]] = None

        self._obs, self._infos = self._vec_env.reset_all(seed=seed)
        self._pool = BufferPool(
            chunk_length=chunk_length,
            obs_shape=tuple(self._obs.shape[2:]),
            action_shape=tuple(self._action_spec.action_shape),
            action_dtype=self._action_spec.numpy_dtype,
            mask_size=self._action_spec.flat_mask_size,
        )
        self._tracks: list[list[SlotTrack]] = []
        for e in range(E):
            row = []
            for p in range(P):
                track = SlotTrack(buffer=None, state=self._initial_state(e, p))
                if self._collect_mask[e][p]:
                    track.buffer = self._pool.acquire(self._slot_agent_map[e][p])
                row.append(track)
            self._tracks.append(row)

        self._ep_rewards = np.zeros((E, P), dtype=np.float64)
        self._ep_lengths = np.zeros(E, dtype=np.int64)
        self._ep_counter = np.zeros(E, dtype=np.int64)
        self._env_steps = 0
        self._chunks_sent = 0
        self._recorded: dict[str, int] = defaultdict(int)
        self._last_weight_sync = time.monotonic()
        self.sync_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> int:
        """One vectorized env step for all envs. Returns env steps taken."""
        self._poll_command()
        obs = self._obs
        acting = self._acting_flags(self._infos)
        masks = self._extract_masks(self._infos)
        actions, log_probs, values, pre_states = self._infer(obs, masks)
        self._open_transitions(obs, actions, log_probs, values, masks, acting, pre_states)
        next_obs, rewards, terminated, truncated, infos = self._vec_env.step(actions)
        self._after_env_step(next_obs, rewards, terminated, truncated, infos)
        self._obs, self._infos = next_obs, infos
        n = self._num_envs
        self._env_steps += n
        if self._io.add_env_steps is not None:
            self._io.add_env_steps(n)
        if time.monotonic() - self._last_weight_sync >= self._weight_sync_interval:
            self.sync_weights()
        return n

    def sync_weights(self) -> None:
        """Pull the newest weights for every agent's latest model (non-blocking)."""
        for aid in self._agent_ids:
            payload = self._io.poll_weights(aid)
            if payload is not None:
                self._models[aid][LATEST_NETWORK_ID].load_state_dict(payload.to_torch_state_dict())
                self._policy_versions[aid] = int(payload.policy_version)
        self._last_weight_sync = time.monotonic()

    def run(self, should_stop: Callable[[], bool], max_env_steps: int = 0) -> None:
        while not should_stop():
            if max_env_steps > 0 and self._env_steps >= max_env_steps:
                break
            self.step()

    def close(self) -> None:
        self._vec_env.close()

    @property
    def stats(self) -> dict[str, int]:
        out = {
            "chunks_sent": self._chunks_sent,
            "env_steps": self._env_steps,
            "parked_buffers": self._pool.parked_count(),
            "recorded_transitions": sum(self._recorded.values()),
        }
        for aid in self._agent_ids:
            out[f"recorded_transitions/{aid}"] = self._recorded[aid]
            out[f"buffered_transitions/{aid}"] = self._buffered_transitions(aid)
        return out

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def _add_checkpoint(self, aid: str, ckpt_id: str, sd: dict[str, np.ndarray]) -> None:
        if aid not in self._models or ckpt_id in self._models[aid]:
            return
        model = self._model_factories[aid]()
        model.load_state_dict(state_dict_from_numpy(sd))
        model.eval()
        self._models[aid][ckpt_id] = model

    def _resolve_model(self, aid: str, net_id: str) -> PolicyModel:
        models = self._models[aid]
        model = models.get(net_id)
        if model is None:
            if (aid, net_id) not in self._warned_missing:
                self._warned_missing.add((aid, net_id))
                logger.warning(
                    f"Worker {self.worker_id}: network {net_id!r} of {aid!r} is not loaded; "
                    f"using {LATEST_NETWORK_ID!r}"
                )
            model = models[LATEST_NETWORK_ID]
        return model

    def _initial_state(self, e: int, p: int) -> State:
        model = self._resolve_model(self._slot_agent_map[e][p], self._slot_network_map[e][p])
        return model.initial_state(1)

    # ------------------------------------------------------------------
    # Commands / re-assignment
    # ------------------------------------------------------------------

    def _poll_command(self) -> None:
        if self._io.poll_command is None:
            return
        cmd = self._io.poll_command()
        if cmd is None:
            return
        for aid, ckpts in cmd.new_checkpoints.items():
            for ckpt_id, sd in ckpts.items():
                self._add_checkpoint(aid, ckpt_id, sd)
        if cmd.slot_agent_map:
            self._pending_maps = (
                [list(r) for r in cmd.slot_agent_map],
                [list(r) for r in cmd.slot_network_map],
                [[bool(c) for c in r] for r in cmd.collect_mask],
            )

    def _apply_pending_assignment(self, e: int) -> None:
        """Apply the staged assignment to env ``e`` (called at its episode end).

        A slot that stops collecting for its agent parks its partial buffer
        (whose last transition is done) in the pool; a slot that starts
        collecting acquires a buffer, preferring a parked one of its agent.
        """
        if self._pending_maps is None:
            return
        agents, nets, collect = self._pending_maps
        if e >= len(agents):
            return
        for p in range(self._num_players):
            track = self._tracks[e][p]
            old_aid = self._slot_agent_map[e][p]
            new_aid = agents[e][p]
            new_collect = collect[e][p]
            if track.buffer is not None and (not new_collect or new_aid != old_aid):
                self._pool.park(old_aid, track.buffer)
                track.buffer = None
            self._slot_agent_map[e][p] = new_aid
            self._slot_network_map[e][p] = nets[e][p]
            self._collect_mask[e][p] = new_collect
            if new_collect and track.buffer is None:
                track.buffer = self._pool.acquire(new_aid)

    def _buffered_transitions(self, aid: str) -> int:
        n = self._pool.parked_transitions(aid)
        for e in range(self._num_envs):
            for p in range(self._num_players):
                buf = self._tracks[e][p].buffer
                if buf is not None and self._slot_agent_map[e][p] == aid:
                    n += buf.steps
        return n

    # ------------------------------------------------------------------
    # Per-step pieces
    # ------------------------------------------------------------------

    def _acting_flags(self, infos: list[dict]) -> np.ndarray:
        """[E, P] bool: ``info[p]["active"]`` per slot; slots without the key act."""
        E, P = self._num_envs, self._num_players
        acting = np.ones((E, P), dtype=bool)
        for e in range(E):
            info_e = infos[e] if e < len(infos) else {}
            for p in range(P):
                info = info_e.get(p) if isinstance(info_e, dict) else None
                if isinstance(info, dict) and "active" in info:
                    acting[e, p] = bool(info["active"])
        return acting

    def _extract_masks(self, infos: list[dict]) -> Optional[np.ndarray]:
        """[E*P, mask_size] bool, or None if no slot provides ``action_mask``.

        Slots without a mask get an all-true row.
        """
        M = self._action_spec.flat_mask_size
        if M == 0:
            return None
        E, P = self._num_envs, self._num_players
        out: Optional[np.ndarray] = None
        for e in range(E):
            info_e = infos[e] if e < len(infos) else {}
            for p in range(P):
                info = info_e.get(p) if isinstance(info_e, dict) else None
                if not isinstance(info, dict) or info.get("action_mask") is None:
                    continue
                if out is None:
                    out = np.ones((E * P, M), dtype=bool)
                raw = info["action_mask"]
                if isinstance(raw, dict):
                    out[e * P + p] = self._action_spec.flatten_mask(raw)
                else:
                    out[e * P + p] = np.asarray(raw, dtype=bool).reshape(M)
        return out

    def _infer(self, obs: np.ndarray, masks: Optional[np.ndarray]):
        """Batched inference for every slot, grouped by (agent, network).

        Returns (actions [E,P,*A], log_probs [E,P], values [E,P], pre_states)
        where ``pre_states[(e, p)]`` is the slot's model state BEFORE this step.
        """
        E, P = self._num_envs, self._num_players
        spec = self._action_spec
        actions = np.zeros((E, P, *spec.action_shape), dtype=spec.numpy_dtype)
        log_probs = np.zeros((E, P), dtype=np.float32)
        values = np.zeros((E, P), dtype=np.float32)
        pre_states: dict[tuple[int, int], State] = {}
        masks3 = None if masks is None else masks.reshape(E, P, -1)

        groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        for e in range(E):
            for p in range(P):
                groups[(self._slot_agent_map[e][p], self._slot_network_map[e][p])].append((e, p))

        for (aid, net_id), slots in groups.items():
            model = self._resolve_model(aid, net_id)
            ei = [e for e, _ in slots]
            pi = [p for _, p in slots]
            obs_b = torch.from_numpy(np.ascontiguousarray(obs[ei, pi], dtype=np.float32))
            mask_b = None if masks3 is None else torch.from_numpy(np.ascontiguousarray(masks3[ei, pi]))
            state_b = cat_batch([self._tracks[e][p].state for e, p in slots])
            with torch.no_grad():
                out = act(model, obs_b, state_b, mask_b)
            actions[ei, pi] = out.actions.cpu().numpy().astype(spec.numpy_dtype, copy=False)
            log_probs[ei, pi] = out.log_probs.float().cpu().numpy()
            values[ei, pi] = out.values.float().cpu().numpy()
            for j, (e, p) in enumerate(slots):
                track = self._tracks[e][p]
                pre_states[(e, p)] = track.state
                track.state = None if out.state is None else slice_batch(out.state, j)
        return actions, log_probs, values, pre_states

    def _open_transitions(self, obs, actions, log_probs, values, masks, acting, pre_states) -> None:
        """Record a transition for every acting collecting slot.

        Slots that report ``info["active"] = False`` are not recorded.
        """
        E, P = self._num_envs, self._num_players
        masks3 = None if masks is None else masks.reshape(E, P, -1)
        for e in range(E):
            for p in range(P):
                if not (acting[e, p] and self._collect_mask[e][p]):
                    continue
                track = self._tracks[e][p]
                aid = self._slot_agent_map[e][p]
                buf = track.buffer
                if buf.steps == 0:
                    buf.begin_chunk(pre_states[(e, p)], self._policy_versions[aid])
                buf.open(
                    obs[e, p], actions[e, p], float(log_probs[e, p]), float(values[e, p]),
                    None if masks3 is None else masks3[e, p],
                )
                track.has_open = True
                self._recorded[aid] += 1

    def _after_env_step(self, next_obs, rewards, terminated, truncated, infos) -> None:
        """Give each transition recorded this step its reward and done flag;
        seal full buffers (bootstrap = V(next_obs), or 0 after a terminal step)."""
        E, P = self._num_envs, self._num_players
        self._ep_rewards += rewards
        self._ep_lengths += 1
        for e in range(E):
            done = bool(terminated[e] or truncated[e])
            for p in range(P):
                track = self._tracks[e][p]
                if not (self._collect_mask[e][p] and track.has_open):
                    continue
                buf = track.buffer
                buf.add_reward(float(rewards[e, p]))
                if done:
                    buf.mark_done()
                track.has_open = False
                if buf.is_full:
                    boot = 0.0 if done else self._bootstrap_value_forward(e, p, next_obs[e, p])
                    self._seal(self._slot_agent_map[e][p], buf, bootstrap_value=boot)
            if done:
                self._end_episode(e, infos[e])

    def _end_episode(self, e: int, info_e: dict) -> None:
        P = self._num_players
        for p in range(P):
            buf = self._tracks[e][p].buffer
            # A collecting slot that did not act on the last step still ends its
            # episode here: its last recorded transition gets done=True.
            if self._collect_mask[e][p] and buf is not None and buf.steps > 0:
                buf.mark_done()
        self._report_result(e, info_e)
        self._ep_rewards[e] = 0.0
        self._ep_lengths[e] = 0
        self._ep_counter[e] += 1
        self._apply_pending_assignment(e)
        for p in range(P):
            self._tracks[e][p].state = self._initial_state(e, p)

    def _bootstrap_value_forward(self, e: int, p: int, next_obs: np.ndarray) -> float:
        """V(next_obs) from the slot agent's latest model with the slot's state."""
        model = self._models[self._slot_agent_map[e][p]][LATEST_NETWORK_ID]
        obs_b = torch.from_numpy(np.ascontiguousarray(next_obs[None], dtype=np.float32))
        with torch.no_grad():
            out = model.step(obs_b, self._tracks[e][p].state)
        return float(out.value[0])

    def _seal(self, aid: str, buf: RolloutBuffer, bootstrap_value: float) -> None:
        chunk = buf.build_chunk(aid, bootstrap_value)
        buf.reset()
        self._io.send_chunk(chunk)
        self._chunks_sent += 1

    def _report_result(self, e: int, info_e: dict) -> None:
        if self._io.report_result is None:
            return
        P = self._num_players
        terminal_infos = {
            p: (info_e.get(p, {}) or {}).get("terminal_info", {}) for p in range(P)
        }
        rewards = self._ep_rewards[e]
        outcomes = player_outcomes(rewards, terminal_infos, P)
        player_outcome: dict[str, float] = {}
        total_rewards: dict[str, float] = {}
        for p in range(P):
            key = f"{self._slot_agent_map[e][p]}:{self._slot_network_map[e][p]}"
            player_outcome[key] = float(outcomes[p])
            total_rewards[key] = float(rewards[p])
        self._io.report_result(MatchResult(
            match_id=f"w{self.worker_id}_e{e}_ep{int(self._ep_counter[e])}",
            player_outcomes=player_outcome,
            total_rewards=total_rewards,
            episode_length=int(self._ep_lengths[e]),
        ))
```

In `src/colosseum/worker/rollout_worker.py`, replace `LATEST_NETWORK_ID = "latest"` and the `from colosseum.worker.rollout_loop import LoopIO, RolloutLoop` line with:

```python
from colosseum.worker.rollout_loop import LATEST_NETWORK_ID, LoopIO, RolloutLoop  # noqa: F401  (re-export)
```

- [ ] **Step 9: Adapt tests to the rewritten loop**

Run: `grep -rln "RolloutBuffer\|_build_chunk\|set_lstm_init" tests --include=*.py`
- Delete the tests that use the removed `RolloutBuffer.append` / `_build_chunk` API; `tests/unit/test_buffer_pool.py` replaces them.

Run: `grep -rn "loop\._\|_vec_env\|_networks\|_hidden" tests/contract/harness.py tests/contract tests/unit --include=*.py`
If Part A's harness or tests read private `RolloutLoop` attributes, rename them to this file's names:
- `_vec_env`;
- `_models[agent_id][net_id]`;
- `_policy_versions[agent_id]`;
- `_slot_agent_map`, `_slot_network_map`, `_collect_mask`;
- per-slot model state `_tracks[e][p].state`.

Run: `.venv/bin/python -m pytest tests/contract tests/unit -q`
Update any Part A characterization test that still encodes old behaviour:
- If it asserts that a re-assigned slot's partial buffer is discarded, assert instead that `loop.stats["parked_buffers"]` and the recorded/buffered/sent transition counts add up, as in `test_parking.py`.
- If it asserts that `behavior_policy_version` is the version at seal time, assert the version at the chunk's first transition.

- [ ] **Step 10: Emit the policy-lag metrics in the learner**

Create `tests/unit/test_policy_lag.py`:

```python
"""The learner reports policy lag = learner version - chunk behavior version (T2.6)."""
import threading
import time

import pytest

from colosseum.core.config import LearnerConfig
from colosseum.learner.learner import learner_process
from tests.dataflow_helpers import CheckedQueue, RecordingAlgorithm, chunk_payload


def test_learner_emits_policy_lag_mean_and_max():
    algo = RecordingAlgorithm(start_version=10)
    traj, metrics_q, stop = CheckedQueue(), CheckedQueue(), threading.Event()
    for version in (7, 10, 9, 10, 10, 8):
        traj.put(chunk_payload(version=version))
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, trajectory_queue=traj,
        weight_queues=[CheckedQueue(maxsize=1)], config=LearnerConfig(batch_chunks=3),
        stop_event=stop, metrics_queue=metrics_q,
    ))
    thread.start()
    deadline = time.monotonic() + 10
    while len(algo.batches) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=5)
    metrics = [metrics_q.get_nowait() for _ in range(metrics_q.qsize())]
    assert metrics[0]["policy_lag_mean"] == pytest.approx((3 + 0 + 1) / 3)   # learner at v10
    assert metrics[0]["policy_lag_max"] == 3.0
    assert metrics[1]["policy_lag_mean"] == pytest.approx((1 + 1 + 3) / 3)   # learner at v11
    assert metrics[1]["policy_lag_max"] == 3.0
```

Run: `.venv/bin/python -m pytest tests/unit/test_policy_lag.py -v`
Expected: FAIL with `KeyError: 'policy_lag_mean'`.

In `src/colosseum/learner/learner.py`, add `import numpy as np` to the import block. In `learner_process`, insert right after `consumed_samples += sum(c.chunk_length for c in chunks)`:

```python
        lags = [algorithm.policy_version - c.behavior_policy_version for c in chunks]
```

and insert right after `metrics["progress"] = float(progress)`:

```python
        metrics["policy_lag_mean"] = float(np.mean(lags))
        metrics["policy_lag_max"] = float(np.max(lags))
```

- [ ] **Step 11: Run the new tests**

Run: `.venv/bin/python -m pytest tests/unit/test_buffer_pool.py tests/contract/test_parking.py tests/unit/test_policy_lag.py -v`
Expected: PASS.

- [ ] **Step 12: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: all pass, including Part A's T1.6 contract test (the learner reproduces worker log-probs for the four cores). The rewritten loop records `initial_state` before each chunk's first transition and resets model state at episode end, exactly as before.

- [ ] **Step 13: Commit**

```bash
git add src/colosseum/worker/slots.py src/colosseum/worker/rollout_loop.py src/colosseum/worker/rollout_worker.py \
        src/colosseum/learner/learner.py tests
git commit -m "feat: agent-owned rollout buffers parked on re-assignment; policy-lag metrics"
```

---

### Task T3.1: Open transitions sealed on the slot's next action; no bootstrap forward

Spec block 3 "Шаг цикла", items 3–6 (R2-11, R1-04 partially). A collecting slot's transition stays open until that slot acts again, and rewards keep landing on it. A full buffer is sealed only when the slot acts again: `bootstrap_value` is the value from that batched inference. At episode end every open transition gets `done=True`, and a full buffer is sealed at once with `bootstrap_value = 0`. The separate per-chunk bootstrap forward (a third of all forward passes on tic-tac-toe, R2-11) is removed.

**Files:**
- Modify: `src/colosseum/worker/rollout_loop.py` (`step`, `_open_transitions`, `_after_env_step`, `_end_episode`; delete `_bootstrap_value_forward`)
- Modify: `tests/dataflow_helpers.py` (add `reference_transitions`)
- Test: `tests/contract/test_open_transitions.py`

**Interfaces:**
- Consumes: the `RolloutLoop` structure from T2.6 (`_tracks`, `_seal`, `_report_result`, `_apply_pending_assignment`, `_initial_state`, `_recorded`, `_policy_versions`); `RolloutBuffer.begin_chunk/open/add_reward/mark_done/is_full/steps`; `ProbeModel`, `slot_transitions`, `make_loop`, `EnvFactory`, `GridStepEnv` (helpers).
- Produces:
  - `RolloutLoop._after_env_step(self, rewards, terminated, truncated, infos)`. The `next_obs` argument is gone.
  - `RolloutLoop._end_episode(self, e, info_e)`, which marks open transitions done and seals full buffers with bootstrap 0.
  - Invariant used by T3.2/T3.3: `SlotTrack.has_open` is True from the slot's action until its next action or its episode's end.
  - `tests/dataflow_helpers.reference_transitions(log, num_players) -> dict[player, list[dict]]`.

- [ ] **Step 1: Add the reference construction to `tests/dataflow_helpers.py`**

Append to the end of the module:

```python
def reference_transitions(log: list[dict], num_players: int) -> dict[int, list[dict]]:
    """Straightforward reference: per player, one transition per acting step.

    A transition gets every reward from its own step until the player's next
    action (or the episode end); rewards before a player's first action in an
    episode go to that first transition; the last transition of each episode has
    ``done=True``. ``log`` is an env's step log (see ``GridStepEnv``).
    """
    out: dict[int, list[dict]] = {p: [] for p in range(num_players)}
    open_tr: dict[int, Optional[dict]] = {p: None for p in range(num_players)}
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
```

- [ ] **Step 2: Write the failing contract tests**

Create `tests/contract/test_open_transitions.py`:

```python
"""Open transitions: sealing on the slot's next action, no bootstrap forward (T3.1).

Real RolloutLoop in-process; ProbeModel's value is a known function of the
observation, and GridStepEnv logs everything it saw, so every chunk can be
compared with a straightforward reference built from the env log.
"""
import math

import numpy as np
import pytest

from tests.dataflow_helpers import EnvFactory, GridStepEnv, ProbeModel, make_loop, reference_transitions


def test_no_bootstrap_forward():
    created = []

    def model_factory():
        model = ProbeModel(obs_coeffs=(0.0, 0.0, 0.5, 0.25))
        created.append(model)
        return model

    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(3, 5)), model_factory,
                          num_envs=2, chunk_length=4)
    for _ in range(30):
        loop.step()
    assert len(col.chunks) >= 8
    # exactly one batched forward per env step (2 envs x 2 seats), nothing else
    assert created[0].calls == [4] * 30


def test_full_buffer_is_sealed_when_the_slot_acts_again():
    coeffs = (0.0, 0.0, 0.5, 0.25)
    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(10,)), lambda: ProbeModel(obs_coeffs=coeffs),
                          num_envs=1, chunk_length=4)
    for _ in range(4):
        loop.step()
    assert col.chunks == []                  # 4 transitions recorded, the 4th still open
    loop.step()
    assert len(col.chunks) == 2              # sealed by each seat's 5th action
    for chunk in col.chunks:
        p = int(chunk.observations[0, 3])
        assert float(chunk.bootstrap_value) == pytest.approx(0.5 * 4 + 0.25 * p)   # V(obs at t=4)


def test_full_buffer_ending_an_episode_is_sealed_at_once_with_zero_bootstrap():
    loop, col = make_loop(EnvFactory(GridStepEnv, lengths=(4,)), ProbeModel, num_envs=1, chunk_length=4)
    for _ in range(4):
        loop.step()
    assert len(col.chunks) == 2
    assert all(float(c.bootstrap_value) == 0.0 and bool(c.dones[-1]) for c in col.chunks)


def test_simultaneous_chunks_match_the_reference_construction():
    coeffs = (0.0, 0.1, 0.5, 0.25)          # value = 0.1*ep + 0.5*t + 0.25*player
    envs = EnvFactory(GridStepEnv, lengths=(3, 6, 2))
    loop, col = make_loop(envs, lambda: ProbeModel(obs_coeffs=coeffs), num_envs=2, chunk_length=4)
    for _ in range(40):
        loop.step()

    def value(tr, p):
        return coeffs[1] * tr["ep"] + coeffs[2] * tr["t"] + coeffs[3] * p

    by_slot = {}
    for chunk in col.chunks:
        by_slot.setdefault((int(chunk.observations[0, 0]), int(chunk.observations[0, 3])), []).append(chunk)
    assert set(by_slot) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for (env_id, p), chunks in by_slot.items():
        ref = reference_transitions(envs.created[env_id].log, 2)[p]
        for k, chunk in enumerate(chunks):
            rows = ref[4 * k: 4 * k + 4]
            assert [r["ep"] for r in rows] == chunk.observations[:, 1].int().tolist()
            assert [r["t"] for r in rows] == chunk.observations[:, 2].int().tolist()
            assert [r["action"] for r in rows] == chunk.actions.tolist()
            assert np.allclose([r["reward"] for r in rows], chunk.rewards.numpy())
            assert [r["done"] for r in rows] == chunk.dones.tolist()
            assert np.allclose([value(r, p) for r in rows], chunk.values.numpy(), atol=1e-5)
            assert np.allclose(chunk.action_log_probs.numpy(), -math.log(3), atol=1e-5)
            if rows[-1]["done"]:
                assert float(chunk.bootstrap_value) == 0.0
            else:
                assert float(chunk.bootstrap_value) == pytest.approx(value(ref[4 * k + 4], p), abs=1e-5)
            assert chunk.initial_state is None
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/contract/test_open_transitions.py -v`
Expected:
- `test_no_bootstrap_forward` fails: `created[0].calls` contains extra `1`s, the bootstrap forwards;
- `test_full_buffer_is_sealed_when_the_slot_acts_again` fails: the chunks are already sent after 4 steps;
- the reference test fails with `IndexError` (a chunk is sealed before the next step exists in the log);
- the zero-bootstrap test passes already.

- [ ] **Step 4: Rewrite the transition methods in `src/colosseum/worker/rollout_loop.py`**

Replace `step` with:

```python
    def step(self) -> int:
        """One vectorized env step for all envs. Returns env steps taken."""
        self._poll_command()
        obs = self._obs
        acting = self._acting_flags(self._infos)
        masks = self._extract_masks(self._infos)
        actions, log_probs, values, pre_states = self._infer(obs, masks)
        self._open_transitions(obs, actions, log_probs, values, masks, acting, pre_states)
        next_obs, rewards, terminated, truncated, infos = self._vec_env.step(actions)
        self._after_env_step(rewards, terminated, truncated, infos)
        self._obs, self._infos = next_obs, infos
        n = self._num_envs
        self._env_steps += n
        if self._io.add_env_steps is not None:
            self._io.add_env_steps(n)
        if time.monotonic() - self._last_weight_sync >= self._weight_sync_interval:
            self.sync_weights()
        return n
```

Replace `_open_transitions` with:

```python
    def _open_transitions(self, obs, actions, log_probs, values, masks, acting, pre_states) -> None:
        """For every acting collecting slot: if its buffer is full, seal it with
        ``bootstrap_value`` = the value from this step's inference; then open a
        new transition. The previous transition (if any) closes implicitly."""
        E, P = self._num_envs, self._num_players
        masks3 = None if masks is None else masks.reshape(E, P, -1)
        for e in range(E):
            for p in range(P):
                if not (acting[e, p] and self._collect_mask[e][p]):
                    continue
                track = self._tracks[e][p]
                aid = self._slot_agent_map[e][p]
                buf = track.buffer
                if buf.is_full:
                    self._seal(aid, buf, bootstrap_value=float(values[e, p]))
                if buf.steps == 0:
                    buf.begin_chunk(pre_states[(e, p)], self._policy_versions[aid])
                buf.open(
                    obs[e, p], actions[e, p], float(log_probs[e, p]), float(values[e, p]),
                    None if masks3 is None else masks3[e, p],
                )
                track.has_open = True
                self._recorded[aid] += 1
```

Replace `_after_env_step` and `_end_episode` with:

```python
    def _after_env_step(self, rewards, terminated, truncated, infos) -> None:
        """Add each collecting slot's reward to its open transition, then end
        finished episodes. Transitions stay open until the slot acts again."""
        E, P = self._num_envs, self._num_players
        self._ep_rewards += rewards
        self._ep_lengths += 1
        for e in range(E):
            for p in range(P):
                track = self._tracks[e][p]
                if self._collect_mask[e][p] and track.has_open:
                    track.buffer.add_reward(float(rewards[e, p]))
            if terminated[e] or truncated[e]:
                self._end_episode(e, infos[e])

    def _end_episode(self, e: int, info_e: dict) -> None:
        """Mark every open transition done; seal full buffers with bootstrap 0;
        report the result; apply a staged re-assignment; reset model states."""
        P = self._num_players
        for p in range(P):
            track = self._tracks[e][p]
            if self._collect_mask[e][p] and track.has_open:
                track.buffer.mark_done()
                track.has_open = False
                if track.buffer.is_full:
                    self._seal(self._slot_agent_map[e][p], track.buffer, bootstrap_value=0.0)
        self._report_result(e, info_e)
        self._ep_rewards[e] = 0.0
        self._ep_lengths[e] = 0
        self._ep_counter[e] += 1
        self._apply_pending_assignment(e)
        for p in range(P):
            self._tracks[e][p].state = self._initial_state(e, p)
```

Delete the method `_bootstrap_value_forward` entirely.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/contract/test_open_transitions.py tests/contract/test_parking.py -v`
Expected: PASS.

- [ ] **Step 6: Update characterization tests that counted forwards or expected eager sealing**

Run: `.venv/bin/python -m pytest tests/contract tests/unit -q`
Two kinds of failures are expected:
- A test that expects a chunk right after its T-th transition: run one more `loop.step()` before reading `col.chunks`, because the chunk is sealed when the slot acts again.
- A test that counts forward passes including bootstrap calls: drop the bootstrap calls from the expected count.

Bootstrap values do not change for these tests: the next action's value comes from the same model and state that the removed forward used.

- [ ] **Step 7: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/colosseum/worker/rollout_loop.py tests
git commit -m "perf: seal chunks on the slot's next action instead of a bootstrap forward"
```

---

### Task T3.2: Turn-based transitions: acting-only inference, pending rewards, mask rules, reset-info fix

Spec block 3, steps 1–5, "Маски" and "Утечка info после auto-reset" (R1-22, R2-02, ET-01, ET-17, R1-13, ET-08, R2-14, R2-15).
- Acting slots come from `info["active"]`; if the key is absent, the slot acts.
- Only acting slots run inference, grouped by (agent, network). The others send the zero action from `ActionSpec`, and their model state does not advance.
- Rewards that arrive before a slot's first action in an episode accumulate in `pending_reward` and are added to that first transition. Because transitions stay open (T3.1), the final rewards and `done` reach every collecting slot.
- Mask rules: an all-false mask row on a non-acting slot becomes all-true; on an acting slot it raises `EnvContractError` naming env, slot and step.
- `VectorEnv` auto-reset returns the reset info, so the first decision of a new episode no longer sees the terminal step's mask, `active`, `rank` or `outcome`. `SubprocessVectorEnv` children use `VectorEnv`, so the fix covers both.

**Files:**
- Modify: `src/colosseum/worker/rollout_loop.py` (`step`, new `_check_masks`, `_infer`, `_open_transitions`, `_after_env_step`, `_end_episode`)
- Modify: `src/colosseum/envs/vec_env.py` (auto-reset info)
- Modify: `src/colosseum/envs/base_env.py` (docstring: the turn-based convention)
- Modify: `tests/dataflow_helpers.py` (add `AlternatingEnv`, `BadMaskEnv`, `InfoLeakEnv`)
- Test: `tests/contract/test_turn_based.py`, `tests/unit/test_vec_env_reset_info.py`

**Interfaces:**
- Consumes:
  - `EnvContractError` (`colosseum.core.errors`, T1.4);
  - the T3.1 loop methods;
  - `SlotTrack.pending_reward`;
  - `ActionSpec.components` (`mask_offset`, `mask_size`, `name`);
  - `ProbeModel(stateful=True)`, `reference_transitions`, `slot_transitions`, `make_loop`, `EnvFactory` (helpers).
- Produces:
  - `RolloutLoop._check_masks(masks, acting)`;
  - `RolloutLoop._infer(obs, masks, acting)`, which infers for acting slots only.
  - `VectorEnv.step` info contract for a done env: `info[p] = reset_info[p] | {"terminal_observation": final_obs[p], "terminal_info": final_step_info[p]}`. The same holds for `SubprocessVectorEnv`.
  - `tests/dataflow_helpers.py`: `AlternatingEnv`, `BadMaskEnv`, `InfoLeakEnv`.

- [ ] **Step 1: Add the turn-based and info-leak envs to `tests/dataflow_helpers.py`**

Append to the end of the module:

```python
class AlternatingEnv(_Base):
    """Turn-based 2-player env: player ``t % 2`` acts at step ``t``.

    - Episode ``ep`` lasts 3 steps if ``ep`` is even, else 4. The player who makes
      the last move wins: +1 to the mover, -1 to the other.
    - In 4-step episodes player 1 gets +0.5 at step 0, before its first move.
    - After a step, the acting slot's mask is [True, True, False] and the
      non-acting slot's mask is all False. The reset info has only "active"
      (no mask: every action is legal at t=0).
    - The info returned on the terminal step is deliberately bogus (nobody
      active, all-false masks): the first decision of the next episode must use
      the reset info, otherwise the acting slot sees an empty mask (R2-14).
    - Raises if the mover plays action 2 when it is illegal (t > 0).
    """

    def reset(self, seed=None):
        self.ep += 1
        self.t = 0
        return self._obs(), {p: {"active": p == 0} for p in range(2)}

    def _length(self) -> int:
        return 3 if self.ep % 2 == 0 else 4

    def _info(self) -> dict[int, dict]:
        mover = self.t % 2
        return {p: {"active": p == mover,
                    "action_mask": (np.array([True, True, False]) if p == mover
                                    else np.zeros(NUM_ACTIONS, dtype=bool))}
                for p in range(2)}

    def step(self, actions):
        mover = self.t % 2
        if self.t > 0 and int(actions[mover]) == 2:
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


class BadMaskEnv(AlternatingEnv):
    """Like AlternatingEnv, but at step 1 the ACTING slot gets an all-false mask."""

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
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_vec_env_reset_info.py`:

```python
"""Auto-reset returns the reset info, not the terminal step's info (T3.2, R2-14)."""
import numpy as np
import pytest

from colosseum.envs.subproc_vec_env import SubprocessVectorEnv
from colosseum.envs.vec_env import VectorEnv
from tests.dataflow_helpers import InfoLeakEnv


@pytest.mark.parametrize("kind", ["sync", "subprocess"])
def test_auto_reset_info_is_the_reset_info(kind):
    if kind == "sync":
        vec = VectorEnv(InfoLeakEnv, 1)
    else:
        vec = SubprocessVectorEnv(InfoLeakEnv, 1, num_workers=1)
    try:
        vec.reset_all()
        _, _, terminated, _, infos = vec.step(np.zeros((1, 1), dtype=np.int64))
        assert not terminated[0] and infos[0][0]["phase"] == "step"   # mid-episode: step info
        _, _, terminated, _, infos = vec.step(np.zeros((1, 1), dtype=np.int64))
        assert terminated[0]
        info = infos[0][0]
        assert info["phase"] == "reset" and info["active"] is True
        for stale in ("action_mask", "rank", "outcome"):
            assert stale not in info
        assert info["terminal_info"]["rank"] == 1 and info["terminal_info"]["phase"] == "step"
        assert "terminal_info" not in info["terminal_info"]
        assert info["terminal_observation"][2] == 2.0      # obs = [env_id, ep, t, p] at t=2
    finally:
        vec.close()
```

Create `tests/contract/test_turn_based.py`:

```python
"""Turn-based transitions through the real RolloutLoop (T3.2; spec §3 contract test).

AlternatingEnv: player t%2 moves; the last mover wins (+1) and the other loses
(-1) on the same step. The loser's last recorded transition must carry -1 and
done=True, the winner's +1 and done=True, and non-acting steps produce no
transitions. ProbeModel(stateful) has value = number of the slot's own earlier
moves in the episode, which exposes any state advance on non-acting steps.
"""
import numpy as np
import pytest
import torch

from colosseum.core.errors import EnvContractError
from tests.dataflow_helpers import (
    AlternatingEnv,
    BadMaskEnv,
    EnvFactory,
    ProbeModel,
    make_loop,
    reference_transitions,
    slot_transitions,
)


def _run(num_envs=2, steps=40, chunk_length=2):
    created = []

    def model_factory():
        model = ProbeModel(state_coef=1.0, stateful=True)
        created.append(model)
        return model

    envs = EnvFactory(AlternatingEnv)
    loop, col = make_loop(envs, model_factory, num_envs=num_envs, chunk_length=chunk_length)
    for _ in range(steps):
        loop.step()
    return col, envs, created


def _moves_before(transitions: list[dict], idx: int) -> int:
    """Number of earlier transitions in the same episode as transitions[idx]."""
    ep = transitions[idx]["ep"]
    return sum(1 for tr in transitions[:idx] if tr["ep"] == ep)


def test_final_rewards_and_dones_reach_both_players():
    col, _, _ = _run()
    got = slot_transitions(col.chunks)

    def episode(player, ep):
        return [(t["t"], t["reward"], t["done"]) for t in got[(0, player)] if t["ep"] == ep]

    # 3-step episode: p0 moves at t0, t2 and wins; p1 moves at t1 and loses.
    assert episode(0, 0) == [(0, 0.0, False), (2, 1.0, True)]
    assert episode(1, 0) == [(1, -1.0, True)]
    # 4-step episode: p1 gets +0.5 before its first move, then wins at t3.
    assert episode(0, 1) == [(0, 0.0, False), (2, -1.0, True)]
    assert episode(1, 1) == [(1, 0.5, False), (3, 1.0, True)]


def test_chunks_match_the_reference_and_state_advances_only_on_own_moves():
    col, envs, created = _run()
    got = slot_transitions(col.chunks)
    for env_id in range(2):
        ref = reference_transitions(envs.created[env_id].log, 2)
        for p in range(2):
            mine = got[(env_id, p)]
            expected = ref[p][: len(mine)]
            assert [(r["ep"], r["t"], r["action"], r["done"]) for r in mine] == \
                   [(r["ep"], r["t"], r["action"], r["done"]) for r in expected]
            assert np.allclose([r["reward"] for r in mine], [r["reward"] for r in expected])
            for idx, tr in enumerate(mine):
                assert tr["value"] == pytest.approx(float(_moves_before(mine, idx)))
    # inference only for acting slots: one acting slot per env per step
    assert created[0].calls == [2] * 40


def test_initial_state_and_bootstrap_follow_the_slot_own_moves():
    col, envs, _ = _run(num_envs=1, steps=60, chunk_length=2)
    ref = reference_transitions(envs.created[0].log, 2)
    offset = {0: 0, 1: 0}
    for chunk in col.chunks:
        p = int(chunk.observations[0, 3])
        k = offset[p]
        offset[p] += chunk.chunk_length
        # state before the chunk's first move = number of the slot's earlier moves
        assert torch.equal(chunk.initial_state["n"], chunk.values[:1].view(1, 1))
        if bool(chunk.dones[-1]):
            assert float(chunk.bootstrap_value) == 0.0
        else:
            assert float(chunk.bootstrap_value) == pytest.approx(float(_moves_before(ref[p], k + 2)))


def test_empty_mask_on_an_acting_slot_raises():
    loop, _ = make_loop(EnvFactory(BadMaskEnv), ProbeModel, num_envs=1)
    loop.step()
    with pytest.raises(EnvContractError, match=r"env 0, slot 1, episode step 1"):
        loop.step()
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_vec_env_reset_info.py tests/contract/test_turn_based.py -v`
Expected:
- the vec-env tests fail on `assert stale not in info` (`'action_mask'` leaks);
- the turn-based tests fail with `ValueError: Expected parameter logits ...`: inference runs on the non-acting slot's all-false mask;
- `test_empty_mask_on_an_acting_slot_raises` fails with the same `ValueError` instead of `EnvContractError`.

- [ ] **Step 4: Return the reset info on auto-reset in `src/colosseum/envs/vec_env.py`**

In `VectorEnv.step`, replace:

```python
            # Auto-reset if the episode ended.
            if env_terminated or env_truncated:
                # Preserve terminal data in infos before resetting.
                for p in range(self.num_players):
                    original_info = {k: v for k, v in info_k[p].items()}
                    info_k[p]["terminal_observation"] = obs_k[p]
                    info_k[p]["terminal_info"] = original_info

                new_obs, new_info = env.reset()
                obs_dicts.append(new_obs)
                # Merge reset info into the returned infos (terminal data already stored).
                for p in range(self.num_players):
                    info_k[p].update(new_info[p])
                all_infos.append(info_k)
            else:
```

with:

```python
            # Auto-reset if the episode ended. The returned info is the RESET
            # info: it describes the observation the agents act on next (masks,
            # "active", ...). The final step's obs/info are kept only under
            # "terminal_observation" / "terminal_info" (R2-14).
            if env_terminated or env_truncated:
                new_obs, new_info = env.reset()
                obs_dicts.append(new_obs)
                merged: dict[int, dict] = {}
                for p in range(self.num_players):
                    out = dict(new_info.get(p, {}))
                    out["terminal_observation"] = obs_k[p]
                    out["terminal_info"] = dict(info_k[p])
                    merged[p] = out
                all_infos.append(merged)
            else:
```

In the same method's docstring, replace the `infos:` entry of `Returns:` with:

```
            infos:      list of num_envs info dicts (player_index -> info).
                        For an env that auto-reset, each player's info is the
                        RESET info plus ``"terminal_observation"`` (the final
                        observation) and ``"terminal_info"`` (the final step's info).
```

- [ ] **Step 5: Document the turn-based convention in `src/colosseum/envs/base_env.py`**

In the `BaseEnv` class docstring, replace

```
    For turn-based games: only the active player's action matters;
    other players can pass any valid action (ignored by env).
```

with

```
    Turn-based convention (used by the rollout worker):
    - ``info[p]["active"]`` (bool) marks the players who act on the next step;
      players without the key act. A non-acting player runs no inference, sends
      a zero action (ignored by the env) and its model state does not advance.
    - ``info[p]["action_mask"]`` may be all-false for a non-acting player; an
      acting player must have at least one legal action (else EnvContractError).
    - A player's rewards are credited to its last action until it acts again; at
      episode end the last transition of every collecting player gets done=True.
```

- [ ] **Step 6: Acting-only inference, pending rewards and mask rules in `src/colosseum/worker/rollout_loop.py`**

Add `from colosseum.core.errors import EnvContractError` to the import block.

Replace `step` with:

```python
    def step(self) -> int:
        """One vectorized env step for all envs. Returns env steps taken."""
        self._poll_command()
        obs = self._obs
        acting = self._acting_flags(self._infos)
        masks = self._extract_masks(self._infos)
        if masks is not None:
            self._check_masks(masks, acting)
        actions, log_probs, values, pre_states = self._infer(obs, masks, acting)
        self._open_transitions(obs, actions, log_probs, values, masks, acting, pre_states)
        next_obs, rewards, terminated, truncated, infos = self._vec_env.step(actions)
        self._after_env_step(rewards, terminated, truncated, infos)
        self._obs, self._infos = next_obs, infos
        n = self._num_envs
        self._env_steps += n
        if self._io.add_env_steps is not None:
            self._io.add_env_steps(n)
        if time.monotonic() - self._last_weight_sync >= self._weight_sync_interval:
            self.sync_weights()
        return n
```

Add this method right after `_extract_masks`:

```python
    def _check_masks(self, masks: np.ndarray, acting: np.ndarray) -> None:
        """Mask rules (R1-13, ET-08): an empty mask row (per discrete component)
        on a non-acting slot becomes all-true; on an acting slot it is an error."""
        P = self._num_players
        for comp in self._action_spec.components:
            if comp.mask_size == 0:
                continue
            lo, hi = comp.mask_offset, comp.mask_offset + comp.mask_size
            empty = ~masks[:, lo:hi].any(axis=1)
            for flat in np.flatnonzero(empty):
                e, p = divmod(int(flat), P)
                if acting[e, p]:
                    raise EnvContractError(
                        f"worker {self.worker_id}, env {e}, slot {p}, episode step "
                        f"{int(self._ep_lengths[e])}: action_mask has no legal action "
                        f"(component {comp.name!r}) for an acting slot"
                    )
                masks[flat, lo:hi] = True
```

Replace `_infer` with:

```python
    def _infer(self, obs: np.ndarray, masks: Optional[np.ndarray], acting: np.ndarray):
        """Batched inference for acting slots, grouped by (agent, network).

        Non-acting slots keep the default action (zeros) and their model state.
        Returns (actions [E,P,*A], log_probs [E,P], values [E,P], pre_states)
        where ``pre_states[(e, p)]`` is the acting slot's state BEFORE this step.
        """
        E, P = self._num_envs, self._num_players
        spec = self._action_spec
        actions = np.zeros((E, P, *spec.action_shape), dtype=spec.numpy_dtype)
        log_probs = np.zeros((E, P), dtype=np.float32)
        values = np.zeros((E, P), dtype=np.float32)
        pre_states: dict[tuple[int, int], State] = {}
        masks3 = None if masks is None else masks.reshape(E, P, -1)

        groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        for e in range(E):
            for p in range(P):
                if acting[e, p]:
                    groups[(self._slot_agent_map[e][p], self._slot_network_map[e][p])].append((e, p))

        for (aid, net_id), slots in groups.items():
            model = self._resolve_model(aid, net_id)
            ei = [e for e, _ in slots]
            pi = [p for _, p in slots]
            obs_b = torch.from_numpy(np.ascontiguousarray(obs[ei, pi], dtype=np.float32))
            mask_b = None if masks3 is None else torch.from_numpy(np.ascontiguousarray(masks3[ei, pi]))
            state_b = cat_batch([self._tracks[e][p].state for e, p in slots])
            with torch.no_grad():
                out = act(model, obs_b, state_b, mask_b)
            actions[ei, pi] = out.actions.cpu().numpy().astype(spec.numpy_dtype, copy=False)
            log_probs[ei, pi] = out.log_probs.float().cpu().numpy()
            values[ei, pi] = out.values.float().cpu().numpy()
            for j, (e, p) in enumerate(slots):
                track = self._tracks[e][p]
                pre_states[(e, p)] = track.state
                track.state = None if out.state is None else slice_batch(out.state, j)
        return actions, log_probs, values, pre_states
```

Replace `_open_transitions` with:

```python
    def _open_transitions(self, obs, actions, log_probs, values, masks, acting, pre_states) -> None:
        """For every acting collecting slot: seal a full buffer (bootstrap = this
        step's value), then open a new transition carrying the pending reward."""
        E, P = self._num_envs, self._num_players
        masks3 = None if masks is None else masks.reshape(E, P, -1)
        for e in range(E):
            for p in range(P):
                if not (acting[e, p] and self._collect_mask[e][p]):
                    continue
                track = self._tracks[e][p]
                aid = self._slot_agent_map[e][p]
                buf = track.buffer
                if buf.is_full:
                    self._seal(aid, buf, bootstrap_value=float(values[e, p]))
                if buf.steps == 0:
                    buf.begin_chunk(pre_states[(e, p)], self._policy_versions[aid])
                buf.open(
                    obs[e, p], actions[e, p], float(log_probs[e, p]), float(values[e, p]),
                    None if masks3 is None else masks3[e, p],
                    reward=track.pending_reward,
                )
                track.pending_reward = 0.0
                track.has_open = True
                self._recorded[aid] += 1
```

Replace `_after_env_step` and `_end_episode` with:

```python
    def _after_env_step(self, rewards, terminated, truncated, infos) -> None:
        """Add each collecting slot's reward to its open transition, or to its
        ``pending_reward`` before its first action in the episode; then end
        finished episodes."""
        E, P = self._num_envs, self._num_players
        self._ep_rewards += rewards
        self._ep_lengths += 1
        for e in range(E):
            for p in range(P):
                if not self._collect_mask[e][p]:
                    continue
                track = self._tracks[e][p]
                r = float(rewards[e, p])
                if track.has_open:
                    track.buffer.add_reward(r)
                else:
                    track.pending_reward += r
            if terminated[e] or truncated[e]:
                self._end_episode(e, infos[e])

    def _end_episode(self, e: int, info_e: dict) -> None:
        """Mark every open transition done; seal full buffers with bootstrap 0;
        drop rewards of slots that never acted; report the result; apply a staged
        re-assignment; reset model states."""
        P = self._num_players
        for p in range(P):
            track = self._tracks[e][p]
            if self._collect_mask[e][p] and track.has_open:
                track.buffer.mark_done()
                track.has_open = False
                if track.buffer.is_full:
                    self._seal(self._slot_agent_map[e][p], track.buffer, bootstrap_value=0.0)
            track.pending_reward = 0.0
        self._report_result(e, info_e)
        self._ep_rewards[e] = 0.0
        self._ep_lengths[e] = 0
        self._ep_counter[e] += 1
        self._apply_pending_assignment(e)
        for p in range(P):
            self._tracks[e][p].state = self._initial_state(e, p)
```

- [ ] **Step 7: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_vec_env_reset_info.py tests/contract/test_turn_based.py tests/contract/test_open_transitions.py tests/contract/test_parking.py -v`
Expected: PASS.

- [ ] **Step 8: Check the vec-env and worker suites for the old merge behaviour**

Run: `.venv/bin/python -m pytest tests -q -k "vec_env or subproc or rollout or worker"`
Expected: PASS. No existing test relies on step-info keys surviving an auto-reset; the old ones only check `terminal_observation`/`terminal_info`. If one does, it encodes R2-14: move its assertion to `info["terminal_info"]`.

- [ ] **Step 9: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add src/colosseum/worker/rollout_loop.py src/colosseum/envs/vec_env.py src/colosseum/envs/base_env.py tests
git commit -m "fix: turn-based rewards and dones reach every seat; acting-only inference; mask rules"
```

---

### Task T3.3: Truncation adds `γ·V(final_obs)` to the last reward

Spec block 3, step 5 (R1-03, R2-09). When an episode is truncated (`truncated and not terminated`), every open transition of a collecting slot gets `reward += gamma * V(final_obs)`:
- `V` comes from the slot's model, applied with the slot's current state to `info[p]["terminal_observation"]`;
- `VectorEnv` and `SubprocessVectorEnv` always provide `terminal_observation` on auto-reset (T3.2 keeps it);
- `gamma` is the slot agent's `algorithm.gamma`.

There is one batched forward per truncated episode and (agent, network) group, and none for terminated episodes. The transition is then terminal (`done=True`, bootstrap 0).

**Files:**
- Modify: `src/colosseum/worker/rollout_loop.py` (module docstring, `gamma` per agent in `__init__`, `_after_env_step`, `_end_episode`, new `_bootstrap_truncation`)
- Modify: `src/colosseum/launcher.py` (`_worker_target`: per-agent gamma), `src/colosseum/distributed.py` (`_dist_worker_target`: per-agent gamma)
- Test: `tests/contract/test_truncation.py`

**Interfaces:**
- Consumes:
  - the T3.2 loop;
  - `PolicyModel.step(obs, state)`;
  - `cat_batch`;
  - `EnvContractError`;
  - `info[p]["terminal_observation"]` from the vec envs;
  - `ProbeModel`, `GridStepEnv(truncate_every=..., const_reward=...)`, `make_loop`, `slot_transitions` (helpers).
- Produces:
  - `RolloutLoop(..., gamma: float | dict[str, float] = 0.99, ...)`. A dict maps every agent id to its gamma; this extends the contract, see Contract notes.
  - `RolloutLoop._end_episode(self, e, terminated: bool, truncated: bool, info_e)`.
  - `RolloutLoop._bootstrap_truncation(self, e, info_e)`.
  - `EnvContractError` if a truncated episode has no `terminal_observation`.

- [ ] **Step 1: Write the failing contract test**

Create `tests/contract/test_truncation.py`:

```python
"""Truncated episodes bootstrap V(final_obs) into the last reward (T3.3, R1-03, R2-09)."""
import pytest

from tests.dataflow_helpers import EnvFactory, GridStepEnv, ProbeModel, make_loop, slot_transitions


def test_truncation_adds_the_discounted_value_of_the_final_observation():
    created = []

    def model_factory():
        # value = t (from obs) + 10 * own moves so far: a wrong observation (the
        # next episode's first obs, t=0) or a reset state would change it.
        model = ProbeModel(obs_coeffs=(0.0, 0.0, 1.0, 0.0), state_coef=10.0, stateful=True)
        created.append(model)
        return model

    gamma = 0.9
    envs = EnvFactory(GridStepEnv, lengths=(5,), truncate_every=2, const_reward=1.0)
    loop, col = make_loop(envs, model_factory, num_envs=1, chunk_length=5, gamma=gamma)
    for _ in range(20):        # episodes 0 and 2 truncated, 1 and 3 terminated
        loop.step()
    got = slot_transitions(col.chunks)
    for p in range(2):
        last = {tr["ep"]: tr["reward"] for tr in got[(0, p)] if tr["done"]}
        final_value = 5.0 + 10.0 * 5     # terminal obs t=5, state after 5 own moves
        assert last[0] == pytest.approx(1.0 + gamma * final_value)
        assert last[2] == pytest.approx(1.0 + gamma * final_value)
        assert last[1] == pytest.approx(1.0)
        assert last[3] == pytest.approx(1.0)
    # 20 inference forwards (both seats) + exactly one forward per truncated episode
    assert created[0].calls == [2] * 22


def test_truncation_uses_the_gamma_of_the_slot_agent():
    envs = EnvFactory(GridStepEnv, lengths=(3,), truncate_every=1, const_reward=1.0)
    loop, col = make_loop(envs, lambda: ProbeModel(obs_coeffs=(0.0, 0.0, 1.0, 0.0)),
                          agent_ids=("a", "b"), num_envs=1, chunk_length=3,
                          gamma={"a": 0.5, "b": 0.9}, slot_agent_map=[["a", "b"]])
    for _ in range(3):
        loop.step()
    got = slot_transitions(col.chunks)
    assert got[(0, 0)][-1]["reward"] == pytest.approx(1.0 + 0.5 * 3.0)
    assert got[(0, 1)][-1]["reward"] == pytest.approx(1.0 + 0.9 * 3.0)
    assert got[(0, 0)][-1]["done"] and got[(0, 1)][-1]["done"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/contract/test_truncation.py -v`
Expected:
- the first test fails: the truncated episodes' last reward is `1.0` (no bootstrap), and `created[0].calls` has 20 entries;
- the second test fails on the `gamma` dict (`TypeError: float() argument must be ... 'dict'`) or on the reward.

- [ ] **Step 3: Implement truncation bootstrapping in `src/colosseum/worker/rollout_loop.py`**

Replace the module docstring with:

```python
"""In-process rollout loop: vectorized envs, batched inference, per-slot transitions.

All I/O goes through :class:`LoopIO` callbacks, so the loop runs unchanged in a
worker process (``rollout_worker_process``) and in-process in tests.

Transition model (spec block 3):
- Only *acting* slots (``info["active"]``; absent = all act) run inference; the
  others send the default action (zeros) and their model state is unchanged.
- A collecting slot's transition stays *open* until that slot acts again, so
  rewards produced on other players' turns land on it. Rewards that arrive
  before the slot's first action in an episode accumulate in ``pending_reward``
  and are added to that first transition.
- A full buffer is sealed when the slot acts again: ``bootstrap_value`` is the
  value from that inference. At episode end every open transition is marked
  ``done`` (truncation adds ``gamma * V(final_obs)`` to its reward) and a full
  buffer is sealed with ``bootstrap_value = 0``. No separate bootstrap forward.
- Buffers are owned by agents and parked on match re-assignment (block 2).
"""
```

Change the typing import to `from typing import Callable, Optional, Union`, change the constructor parameter `gamma: float = 0.99,` to `gamma: Union[float, dict[str, float]] = 0.99,`, and replace `self._gamma = float(gamma)` in `__init__` with:

```python
        if isinstance(gamma, dict):
            self._gammas = {aid: float(gamma[aid]) for aid in self._agent_ids}
        else:
            self._gammas = {aid: float(gamma) for aid in self._agent_ids}
```

Replace `_after_env_step` and `_end_episode` with:

```python
    def _after_env_step(self, rewards, terminated, truncated, infos) -> None:
        """Add each collecting slot's reward to its open transition, or to its
        ``pending_reward`` before its first action in the episode; then end
        finished episodes."""
        E, P = self._num_envs, self._num_players
        self._ep_rewards += rewards
        self._ep_lengths += 1
        for e in range(E):
            for p in range(P):
                if not self._collect_mask[e][p]:
                    continue
                track = self._tracks[e][p]
                r = float(rewards[e, p])
                if track.has_open:
                    track.buffer.add_reward(r)
                else:
                    track.pending_reward += r
            if terminated[e] or truncated[e]:
                self._end_episode(e, bool(terminated[e]), bool(truncated[e]), infos[e])

    def _end_episode(self, e: int, terminated: bool, truncated: bool, info_e: dict) -> None:
        """Close env ``e``'s episode: truncation bootstrap, ``done`` on every open
        transition (full buffers sealed with bootstrap 0), result report, staged
        re-assignment, model-state reset."""
        P = self._num_players
        if truncated and not terminated:
            self._bootstrap_truncation(e, info_e)
        for p in range(P):
            track = self._tracks[e][p]
            if self._collect_mask[e][p] and track.has_open:
                track.buffer.mark_done()
                track.has_open = False
                if track.buffer.is_full:
                    self._seal(self._slot_agent_map[e][p], track.buffer, bootstrap_value=0.0)
            track.pending_reward = 0.0
        self._report_result(e, info_e)
        self._ep_rewards[e] = 0.0
        self._ep_lengths[e] = 0
        self._ep_counter[e] += 1
        self._apply_pending_assignment(e)
        for p in range(P):
            self._tracks[e][p].state = self._initial_state(e, p)
```

Add this method right after `_end_episode`:

```python
    def _bootstrap_truncation(self, e: int, info_e: dict) -> None:
        """Add ``gamma * V(final_obs)`` to every open transition of env ``e``.

        One batched forward per (agent, network) group, with each slot's current
        model state, on ``info[p]["terminal_observation"]`` (R1-03, R2-09).
        """
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for p in range(self._num_players):
            if self._collect_mask[e][p] and self._tracks[e][p].has_open:
                groups[(self._slot_agent_map[e][p], self._slot_network_map[e][p])].append(p)
        for (aid, net_id), players in groups.items():
            try:
                final = np.stack([np.asarray(info_e[p]["terminal_observation"]) for p in players])
            except KeyError as exc:
                raise EnvContractError(
                    f"worker {self.worker_id}, env {e}: truncated episode without "
                    f"info['terminal_observation']"
                ) from exc
            model = self._resolve_model(aid, net_id)
            obs_b = torch.from_numpy(np.ascontiguousarray(final, dtype=np.float32))
            state_b = cat_batch([self._tracks[e][p].state for p in players])
            with torch.no_grad():
                out = model.step(obs_b, state_b)
            v = out.value.float().cpu().numpy()
            for j, p in enumerate(players):
                self._tracks[e][p].buffer.add_reward(self._gammas[aid] * float(v[j]))
```

- [ ] **Step 4: Pass each agent's gamma from the launcher and the distributed workers**

In `src/colosseum/launcher.py` `_worker_target` and in `src/colosseum/distributed.py` `_dist_worker_target`, replace `gamma=config.algorithm.gamma,` in the `rollout_worker_process(...)` call with:

```python
        gamma={aid: agent_configs[aid].algorithm.gamma for aid in agent_ids},
```

`rollout_worker_process` already declares `gamma: Union[float, dict[str, float]]` (T2.1) and passes it through.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/contract/test_truncation.py tests/contract/test_turn_based.py tests/contract/test_open_transitions.py tests/contract/test_parking.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/colosseum/worker/rollout_loop.py src/colosseum/launcher.py src/colosseum/distributed.py \
        tests/contract/test_truncation.py
git commit -m "fix: bootstrap gamma*V(final_obs) into the last reward of truncated episodes"
```

---

### Task T3.4: Per-seat match results (`SeatResult` / `MatchResult`)

Spec block 3 "Результат матча" (R2-08, R4-03, ET-04). Today the worker keys outcomes by `"agent:network"`. Seats that share a key overwrite each other: self-play latest-vs-latest, or an FFA where one agent holds two seats. After this task:
- the worker reports one `SeatResult` per seat, with seat, agent, network, outcome (from `core/outcomes.py`), episode reward and rank;
- the coordinator consumes the seats without collisions: it averages seat outcomes per base agent and keeps the existing pairwise ELO / win-rate math;
- full pairwise seat ratings come in T5.2.

**Files:**
- Modify: `src/colosseum/core/types.py` (replace `MatchResult`, add `SeatResult`)
- Modify: `src/colosseum/worker/rollout_loop.py` (`_report_result`)
- Modify: `src/colosseum/coordinator/coordinator.py` (`report_match_result`; `_base_agent` → `_seat_outcomes_by_agent`)
- Modify: `tests/dataflow_helpers.py` (add `FFA4Env`, `two_seat_result`)
- Test: `tests/contract/test_seat_results.py`
- Update: tests that build the old `MatchResult(player_outcomes=..., total_rewards=...)`

**Interfaces:**
- Consumes: `player_outcomes(total_rewards, terminal_infos, num_players)` (`core/outcomes.py`); the loop's `_slot_agent_map`, `_slot_network_map`, `_ep_rewards`, `_ep_lengths`, `_ep_counter` (T2.6); `Coordinator.report_match_result` (current).
- Produces:
  - `SeatResult(seat, agent_id, network_id, outcome, reward, rank=None)`, `MatchResult(match_id, seats, episode_length)` (contract);
  - `match_id` format `"w{worker_id}_e{env}_ep{episode_index}"`;
  - `Coordinator._seat_outcomes_by_agent(result) -> dict[str, list[float]]` (static);
  - `tests/dataflow_helpers.py`: `FFA4Env`, `two_seat_result(agent_a, outcome_a, agent_b, outcome_b, network_b="latest", match_id="m")`.

- [ ] **Step 1: Add the FFA env and a result builder to `tests/dataflow_helpers.py`**

Change the `colosseum.core.types` import line to also import `SeatResult`, then append to the end of the module:

```python
class FFA4Env(_Base):
    """4-player simultaneous FFA, 2 steps per episode; terminal rank of seat p is p + 1."""

    NUM_PLAYERS = 4

    def reset(self, seed=None):
        self.ep += 1
        self.t = 0
        return self._obs(), {p: {} for p in range(4)}

    def step(self, actions):
        self.t += 1
        done = self.t >= 2
        rew = {p: (float(3 - p) if done else 0.0) for p in range(4)}
        info = {p: ({"rank": p + 1} if done else {}) for p in range(4)}
        return self._obs(), rew, {p: done for p in range(4)}, {p: False for p in range(4)}, info


def two_seat_result(agent_a: str, outcome_a: float, agent_b: str, outcome_b: float,
                    network_b: str = "latest", match_id: str = "m") -> MatchResult:
    """A 2-seat MatchResult: seat 0 = agent_a (latest), seat 1 = agent_b (network_b)."""
    return MatchResult(match_id=match_id, episode_length=1, seats=[
        SeatResult(seat=0, agent_id=agent_a, network_id="latest", outcome=outcome_a, reward=outcome_a),
        SeatResult(seat=1, agent_id=agent_b, network_id=network_b, outcome=outcome_b, reward=outcome_b),
    ])
```

- [ ] **Step 2: Write the failing contract test**

Create `tests/contract/test_seat_results.py`:

```python
"""Per-seat match results from the real worker loop to the coordinator (T3.4).

4-seat FFA where agent "a" holds seats 0 and 2: no seat may be lost or
overwritten, neither in the worker's MatchResult nor in the coordinator.
"""
from pathlib import Path

import pytest

from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import ColosseumConfig, load_config
from tests.dataflow_helpers import EnvFactory, FFA4Env, ProbeModel, make_loop, two_seat_result

REPO = Path(__file__).resolve().parents[2]


def _coordinator(tmp_path, agents):
    data = load_config(REPO / "configs/examples/tic_tac_toe.yaml").model_dump()
    data["checkpoint"]["dir"] = str(tmp_path / "ckpt")
    data["metrics"]["use_wandb"] = False
    coord = Coordinator(ColosseumConfig(**data))
    for aid in agents:
        coord.agent_pool.register_trainable(aid)
    return coord


def test_worker_reports_every_seat_of_a_four_seat_ffa():
    loop, col = make_loop(EnvFactory(FFA4Env), ProbeModel, agent_ids=("a", "b", "c"), num_envs=1,
                          slot_agent_map=[["a", "b", "a", "c"]],
                          collect_mask=[[True, True, True, True]])
    loop.step()
    loop.step()
    assert len(col.results) == 1
    result = col.results[0]
    assert result.match_id == "w0_e0_ep0" and result.episode_length == 2
    assert [(s.seat, s.agent_id, s.network_id, s.rank) for s in result.seats] == [
        (0, "a", "latest", 1), (1, "b", "latest", 2), (2, "a", "latest", 3), (3, "c", "latest", 4)]
    assert [s.outcome for s in result.seats] == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0])
    assert [s.reward for s in result.seats] == pytest.approx([3.0, 2.0, 1.0, 0.0])


def test_coordinator_consumes_seats_without_collisions(tmp_path):
    loop, col = make_loop(EnvFactory(FFA4Env), ProbeModel, agent_ids=("a", "b", "c"), num_envs=1,
                          slot_agent_map=[["a", "b", "a", "c"]],
                          collect_mask=[[True, True, True, True]])
    loop.step()
    loop.step()
    result = col.results[0]
    by_agent = Coordinator._seat_outcomes_by_agent(result)
    assert set(by_agent) == {"a", "b", "c"}
    assert by_agent["a"] == pytest.approx([1.0, 1 / 3])
    assert by_agent["b"] == pytest.approx([2 / 3])
    assert by_agent["c"] == pytest.approx([0.0])
    coord = _coordinator(tmp_path, ("a", "b", "c"))
    coord.report_match_result(result)
    assert len(coord.match_results[-1].seats) == 4
    assert coord.elo.get("a") > coord.elo.get("c")   # a averages 2/3 over its two seats
    assert coord.elo.get("b") > coord.elo.get("c")


def test_same_agent_seats_carry_no_cross_agent_signal(tmp_path):
    coord = _coordinator(tmp_path, ("a",))
    coord.report_match_result(two_seat_result("a", 1.0, "a", 0.0, network_b="ckpt_v1"))
    assert coord.win_rates.get_win_rate("a", "a") == 0.5
    assert coord.elo.get("a") == coord.elo.initial_rating
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/contract/test_seat_results.py -v`
Expected: collection error `ImportError: cannot import name 'SeatResult' from 'colosseum.core.types'` (raised while importing `tests.dataflow_helpers`).

- [ ] **Step 4: Per-seat result types in `src/colosseum/core/types.py`**

Replace `class MatchResult` with:

```python
@dataclass
class SeatResult:
    """Outcome of one seat of a finished match.

    Attributes:
        seat: Seat (player slot) index in the env.
        agent_id: Base agent that played the seat.
        network_id: ``"latest"`` or a checkpoint id (``"ckpt_v<N>"``).
        outcome: In [0, 1] (1 = best), from :mod:`colosseum.core.outcomes`: the
            env's terminal ``outcome``/``rank`` if every seat has one, else
            derived from episode rewards.
        reward: Undiscounted episode return of the seat.
        rank: The env's terminal ``rank`` for the seat (1 = best), if provided.
    """

    seat: int
    agent_id: str
    network_id: str
    outcome: float
    reward: float
    rank: Optional[int] = None


@dataclass
class MatchResult:
    """A finished match, one :class:`SeatResult` per seat (no key collisions)."""

    match_id: str
    seats: list[SeatResult] = field(default_factory=list)
    episode_length: int = 0
```

- [ ] **Step 5: Report seats from the worker loop**

In `src/colosseum/worker/rollout_loop.py`, add `SeatResult` to the import from `colosseum.core.types` and replace `_report_result` with:

```python
    def _report_result(self, e: int, info_e: dict) -> None:
        if self._io.report_result is None:
            return
        P = self._num_players
        terminal_infos = {
            p: (info_e.get(p, {}) or {}).get("terminal_info", {}) for p in range(P)
        }
        rewards = self._ep_rewards[e]
        outcomes = player_outcomes(rewards, terminal_infos, P)
        seats = []
        for p in range(P):
            rank = terminal_infos[p].get("rank")
            seats.append(SeatResult(
                seat=p,
                agent_id=self._slot_agent_map[e][p],
                network_id=self._slot_network_map[e][p],
                outcome=float(outcomes[p]),
                reward=float(rewards[p]),
                rank=None if rank is None else int(rank),
            ))
        self._io.report_result(MatchResult(
            match_id=f"w{self.worker_id}_e{e}_ep{int(self._ep_counter[e])}",
            seats=seats,
            episode_length=int(self._ep_lengths[e]),
        ))
```

- [ ] **Step 6: Consume seats in the coordinator**

In `src/colosseum/coordinator/coordinator.py`, replace the static method `_base_agent` and the method `report_match_result` with:

```python
    @staticmethod
    def _seat_outcomes_by_agent(result: MatchResult) -> dict[str, list[float]]:
        """Every seat's outcome, grouped by base agent id (no seat is dropped).

        One agent may hold several seats (self-play latest vs latest, an N-player
        arena that repeats agents); each seat contributes its own outcome.
        """
        by_agent: dict[str, list[float]] = {}
        for seat in result.seats:
            by_agent.setdefault(seat.agent_id, []).append(float(seat.outcome))
        return by_agent

    def report_match_result(self, result: MatchResult) -> None:
        """Record a per-seat result and update ratings.

        Seat outcomes are averaged per base agent (the unit PFSP selects over),
        then every pair of DIFFERENT agents updates the win-rate matrix and ELO.
        Same-agent pairs (latest vs its own checkpoint, or two seats of one
        agent) carry no cross-agent signal and are skipped. Pairwise seat
        ratings replace this in T5.2.
        """
        self._match_results.append(result)

        by_agent = self._seat_outcomes_by_agent(result)
        agg = {a: sum(v) / len(v) for a, v in by_agent.items()}

        agents = list(agg.keys())
        for i, a in enumerate(agents):
            for j, b in enumerate(agents):
                if i >= j:
                    continue  # each unordered pair once; skips same-agent
                outcome_a = agg[a]
                outcome_b = agg[b]

                self._win_rates.record(a, b, outcome_a)

                if outcome_a > outcome_b:
                    self._elo.update(a, b, draw=False)
                elif outcome_b > outcome_a:
                    self._elo.update(b, a, draw=False)
                else:
                    self._elo.update(a, b, draw=True)
```

- [ ] **Step 7: Update the tests that build the old dict-keyed `MatchResult`**

Run: `grep -rn "player_outcomes=\|total_rewards=\|_base_agent" src tests --include=*.py`
- No hit may remain in `src/`. `colosseum.core.outcomes.player_outcomes(...)` is a different name: calls to that function stay.
- In each test hit, import `two_seat_result` from `tests.dataflow_helpers` and rewrite the result with the mapping below. Before Part A the hits were `test_coordinator_match_reporting` in `tests/test_ratings.py` and `test_coordinator_aggregates_composite_keys`, `test_coordinator_skips_same_agent_pairs`, `test_pfsp_uses_win_rates` in `tests/test_review_fixes.py`.
  - `MatchResult(match_id="m1", player_outcomes={"a0": 1.0, "a1": 0.0}, total_rewards={...}, episode_length=10)` → `two_seat_result("a0", 1.0, "a1", 0.0, match_id="m1")`
  - `player_outcomes={"agent_alpha:latest": 1.0, "agent_beta:latest": 0.0}` → `two_seat_result("agent_alpha", 1.0, "agent_beta", 0.0)`
  - `player_outcomes={"agent_alpha:latest": 1.0, "agent_alpha:ckpt_v1": 0.0}` → `two_seat_result("agent_alpha", 1.0, "agent_alpha", 0.0, network_b="ckpt_v1")`
  - `player_outcomes={"alpha:latest": 1.0, "beta:latest": 0.0}` → `two_seat_result("alpha", 1.0, "beta", 0.0, match_id="m1")`
  - `player_outcomes={"alpha:latest": 0.0, "gamma:latest": 1.0}` → `two_seat_result("alpha", 0.0, "gamma", 1.0, match_id="m2")`
- Remove the now-unused `from colosseum.core.types import MatchResult` lines from those tests.

- [ ] **Step 8: Run the tests**

Run: `.venv/bin/python -m pytest tests/contract/test_seat_results.py -v`
Expected: PASS (3 tests).

- [ ] **Step 9: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add src/colosseum/core/types.py src/colosseum/worker/rollout_loop.py src/colosseum/coordinator/coordinator.py tests
git commit -m "feat: per-seat match results from workers to the coordinator"
```

---

## Contract notes

1. **`put_latest(q, item, timeout: float = 1.0) -> bool`** (contract: `put_latest(q, item) -> None`, "never blocks").
   - **Problem.** `mp.Queue.put` returns before its feeder thread has written the item into the pipe. Right after a put, the queue reports `Full`, but `get_nowait` can still raise `Empty`. A strict put-nowait / get-nowait / put-nowait sequence therefore drops the newest weights whenever the previous item is still in flight, which is exactly the R2-05 symptom.
   - **Change.** `put_latest` retries with a 10 ms blocking `get` to evict the in-flight item and gives up after `timeout` seconds; in practice it returns within milliseconds. It returns whether the item was delivered, and callers ignore the value.
   - **Verified.** The spawn test publishes v1..v50 and the consumer gets v50 (10 of 10 runs).
2. **`RolloutLoop(gamma: float | dict[str, float] = 0.99)`** (contract: `gamma: float`). Truncation bootstrapping (T3.3) must use the gamma of the slot's agent, and `agents.<id>.algorithm` may override `gamma` per agent. A plain float still works. `rollout_worker_process` takes the same type, and `_worker_target` passes `{agent_id: agent_config.algorithm.gamma}`.
3. **`BufferPool` and `RolloutBuffer` details** (the contract names only `acquire`/`park`/`parked_count`):
   - `BufferPool(chunk_length, obs_shape, action_shape, action_dtype, mask_size=0)`;
   - `BufferPool.parked_transitions(agent_id)`;
   - the `RolloutBuffer` API: `begin_chunk`, `open`, `add_reward`, `mark_done`, `build_chunk`, `reset`, `steps`, `is_full`, `last_done`.

   Both classes live in `worker/slots.py`.
4. **Policy-lag metrics** (`policy_lag_mean`, `policy_lag_max`) are produced by `learner_process` (T2.6) from the batch's `behavior_policy_version`s and the learner's version before the train step. T4.6 lists them among the APPO metrics. T4.6 should not compute them a second time inside APPO.
5. **No LR scheduler object.** After T2.5 the LR is a pure function of `progress` (`APPO.set_progress`). T4.6's `BaseAlgorithm.state_dict()` should store `progress`, not "LR-scheduler" state. `load_state_dict` should call `set_progress(progress)` so that resume continues the LR.
6. **`learner_process` signature.** T2.5 implements the contract signature without `run_dir`, which T6.2 adds. Behaviour not stated in the contract: with `progress_counter=None` and `total_timesteps > 0` (distributed mode), the learner stops by itself once `consumed_samples >= total_timesteps`. Metrics gain `progress` and `consumed_samples`.
7. **Signatures not covered by the contract**, which later tasks (T6.2 logging, T6.5 signals) should extend rather than redefine:
   - `rollout_worker_process(*, ...)` (T2.1, keyword-only; full signature in T2.1/T2.5);
   - `launcher._worker_target(*, ..., env_step_counter=None, ...)`;
   - `launcher._learner_target(*, ..., progress_counter=None, total_timesteps=0, num_learners=1)`;
   - `Launcher.env_steps_done`.

   `LATEST_NETWORK_ID` now lives in `worker/rollout_loop.py` and is re-exported by `worker/rollout_worker.py`.
8. **Queue item formats** fixed by T2.2, for T5.3 and T6.3:
   - trajectory queues carry chunk payload dicts, and `collect_batch` raises `TypeError` on anything else;
   - checkpoint queues carry `{"policy_version": int, "state_dict": dict[str, np.ndarray], "optimizer_state": numpy tree | None}`;
   - learner `resume_state` Process arguments are numpy (`to_numpy_tree`).

   T5.3 replaces the checkpoint and resume payloads (`model_state`, `trainer_state`). The final checkpoint it adds must use the same numpy-only rule. `learner._apply_resume_state` must then read T5.3's keys.
9. **`_drain_commands` merges `new_checkpoints`** across all drained `WorkerCommand`s (worker side of R2-06). Checkpoints are sent once as deltas, so keeping only the newest command loses them. The launcher side ("mark as sent only after a successful put") stays in T5.3.
10. **`RolloutLoop.stats` keys:** `chunks_sent`, `env_steps`, `parked_buffers`, `recorded_transitions`, `recorded_transitions/<agent>`, `buffered_transitions/<agent>`. T6.3 can read `parked_buffers` for its `system` record.
11. **Files missing from the overview's file map:**
    - `src/colosseum/core/threads.py` (`configure_torch_threads`, `resolve_learner_threads`);
    - `tests/dataflow_helpers.py` (shared toy envs, probe models, `CheckedQueue`, `make_loop`).
12. **`Coordinator._base_agent` is removed.** `SeatResult.agent_id` is already the base agent. `_seat_outcomes_by_agent` is the minimal adapter; T5.2 replaces the rating math.
