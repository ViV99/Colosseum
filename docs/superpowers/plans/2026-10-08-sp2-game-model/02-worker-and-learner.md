# SP2 Plan — Part B: Worker (chunk v2, MatchRunner, RolloutLoop) and Learner (V-trace over slots, APPO v2)

This part covers spec blocks 4 (trajectories and chunk v2), 5 (worker side: `MatchRunner`, applying `Lineup`s, match results, episode-metric data) and 6 (algorithm), as tasks T3.1–T3.4 and T4.1–T4.4:

- T3.1 — `sp2` core types: chunk v2 and its numpy payload, lineups, results, commands, weights.
- T3.2 — rollout buffers with `act` / `boot` / `pad` slots and per-agent parking.
- T3.3 — `MatchRunner`: the match core shared by training and eval.
- T3.4 — `RolloutLoop` (observer + model pool over a `MatchRunner`), the worker process, and the chunk-structure contract tests.
- T4.1 — V-trace(λ) over slots.
- T4.2 — APPO v2 (unit modes, reductions over ACT slots, diagnostics) and kickstart v2.
- T4.3 — contract: the learner reproduces the worker's joint and per-unit log-probs at zero lag; the bootstrap is the learner's.
- T4.4 — the learner process on chunk v2.

**Read `00-overview.md` first.** Its global constraints, shadow package strategy, file map and interface contract apply to every task below. All new code lives under `src/colosseum/sp2/`; nothing outside `colosseum.sp2` and the new SP2 tests imports it.

**Starting point.** Part A (`01-foundations-and-model.md`) is done: T0.1, T1.1–T1.7, T2.1–T2.4 deliver the contract names of `colosseum.sp2.core.tree`, `.envs.spaces`, `.core.specs`, `.envs.game`, `.core.outcomes`, `.envs.contract`, `.envs.vector`, `.core.config`, `.networks.dist`, `.networks.{base,model,composed,heads,normalization}`, `.core.roles`, `.core.registry`, and `tests/game_helpers.py` with the toy games, `make_test_model`, `CORE_KINDS` and `RandomPolicy`. This part relies only on those contract names; what it additionally assumes about Part A is listed in "Contract notes" at the end.

**Reference prototype.** Every file and test of this part was run against Part A's implementation (scratch copy `/home/viv/.claude/jobs/0db917f9/tmp/plan_partB/integ`, not part of the repo): all tests below pass there, `ruff check` is clean. If the repository differs (Part A changed a detail), keep the intent, the contract and the tests, adapt the edit and say so in the commit body (overview, "Cross-part execution notes").

**Shared test kit.** This part extends two support modules; each addition is shown in full in the task that makes it:

- `tests/game_helpers.py` (created by Part A): T3.3 adds `Tick`, `TickGame` (a scripted `MultiAgentEnv` whose observations encode env, episode, step and seat), `DictModelPool`, `RecordingObserver`; T3.4 adds `NumpyOnlyQueue`; T4.2 adds `synthetic_chunk`; T4.4 adds `learner_role`, `chunk_v2_payload`, `learner_appo`, `RecordingLearnerAlgorithm`.
- `tests/contract/game_harness.py` (created by T3.4, extended by T4.3): an in-memory `LoopIO`, `GameFactory`, `make_loop`, slot decoders, and `learner_eval`.

When a task adds imports to one of these modules, put them into the module's import block at the top (ruff E402/I001), skipping names that are already imported; append the code at the end of the module. Both module names must be in `[tool.ruff.lint.isort] known-first-party` in `pyproject.toml` (T3.4 adds `game_harness`; it also adds `game_helpers` if Part A did not).

**Commands used below.** `PYTEST = .venv/bin/python -m pytest`. Full fast suite: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (must pass with zero warnings). Lint: `.venv/bin/ruff check .`.

**Slot notation in tests and prose.** A chunk's slots are spelled with letters: `A` open ACT, `T` terminal ACT, `B` BOOT at a chunk boundary (`reset_after=False`), `R` BOOT after a truncation (`reset_after=True`), `P` PAD. Example: `AATP` = two ACTs, a terminal ACT, a PAD.

---

### Task T3.1: `sp2` core types — chunk v2 and payloads, lineups, results, commands, weights

Spec block 4 ("Формат `TrajectoryChunk` v2"), block 5 ("Состав матча", "Результат матча"). The SP1 `PlayerSlot`, `MatchConfig` and the three slot maps of `WorkerCommand` are replaced by `SeatAssignment` / `Lineup`; `values` and `bootstrap_value` disappear from the chunk; trees with native dtypes cross the process boundary as numpy trees.

**Files:**
- Create: `src/colosseum/sp2/core/types.py`
- Test: `tests/unit/test_chunk_v2_types.py`

**Interfaces:**
- Consumes:
  - `colosseum.sp2.core.tree`: `Tree`, `tree_map(fn, tree, *rest)`, `tree_to_numpy(tree)`, `tree_to_torch(tree, device="cpu")` (T1.1);
  - `colosseum.core.ipc`: `numpy_to_tensor`, `tensor_to_numpy`, `assert_no_tensors` (unchanged);
  - `colosseum.networks.state`: `State`, `state_to_numpy`, `state_from_numpy`, `tree_map` (unchanged; model states keep SP1's pytree rules: tuples, namedtuples, lists, dicts).
- Produces (contract, exactly): `LATEST_NETWORK_ID`, `SLOT_ACT, SLOT_BOOT, SLOT_PAD = 0, 1, 2`, `TrajectoryChunk` (fields `agent_id, policy_version, initial_state, obs, global_state, actions, action_masks, kind, reward, terminal, reset_after, behavior_logp, behavior_unit_logp`; `num_slots`, `num_acts`, `to`, `pin_memory`, `to_payload`, `from_payload`), `SeatAssignment`, `Lineup`, `SeatResult`, `TeamResult`, `MatchResult`, `WorkerCommand(lineups, new_checkpoints={})`, `WeightPayload` (SP1), `state_dict_to_numpy`, `state_dict_from_numpy` (SP1).
  - Payload keys: exactly the 13 field names of `TrajectoryChunk`; trees as numpy trees, `kind`/`reward`/`terminal`/`reset_after`/`behavior_logp` as 1-D arrays, `behavior_unit_logp` `None` or `[S, K]`, `initial_state` via `state_to_numpy`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_chunk_v2_types.py`:

```python
"""SP2 core types: chunk v2 and its numpy payload, lineups, results, commands (T3.1)."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest
import torch

from colosseum.core.ipc import assert_no_tensors
from colosseum.sp2.core.types import (
    LATEST_NETWORK_ID,
    SLOT_ACT,
    SLOT_BOOT,
    SLOT_PAD,
    Lineup,
    MatchResult,
    SeatAssignment,
    SeatResult,
    TeamResult,
    TrajectoryChunk,
    WeightPayload,
    WorkerCommand,
    state_dict_from_numpy,
    state_dict_to_numpy,
)


def _chunk(*, with_gs: bool, with_masks: bool, with_units: bool, state: str) -> TrajectoryChunk:
    S = 4
    obs = {
        "grid": torch.randint(0, 255, (S, 3, 3), dtype=torch.uint8),
        "vec": torch.randn(S, 2),
    }
    actions = {"move": torch.randint(0, 3, (S,)), "units": torch.randint(0, 4, (S, 5))}
    masks = None
    if with_masks:
        masks = {
            "move": torch.ones(S, 3, dtype=torch.bool),
            "units": {"unit": torch.ones(S, 5, dtype=torch.bool), "action": torch.ones(S, 5, 4, dtype=torch.bool)},
        }
    initial_state = {
        "none": None,
        "dict": {"h": torch.randn(1, 1, 8), "c": torch.randn(1, 1, 8)},
        "tuple": (torch.randn(1, 3), torch.zeros(1, dtype=torch.long)),
    }[state]
    return TrajectoryChunk(
        agent_id="a",
        policy_version=7,
        initial_state=initial_state,
        obs=obs,
        global_state={"map": torch.randint(0, 2, (S, 4, 4), dtype=torch.int8)} if with_gs else None,
        actions=actions,
        action_masks=masks,
        kind=torch.tensor([SLOT_ACT, SLOT_ACT, SLOT_BOOT, SLOT_PAD], dtype=torch.int8),
        reward=torch.tensor([1.0, 0.5, 0.0, 0.0]),
        terminal=torch.tensor([False, False, False, False]),
        reset_after=torch.tensor([False, False, True, True]),
        behavior_logp=torch.tensor([-1.0, -2.0, 0.0, 0.0]),
        behavior_unit_logp=torch.randn(S, 6) if with_units else None,
    )


def _assert_trees_equal(a, b) -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b)
        for key in a:
            _assert_trees_equal(a[key], b[key])
    elif isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        for x, y in zip(a, b):
            _assert_trees_equal(x, y)
    elif a is None:
        assert b is None
    else:
        assert isinstance(b, torch.Tensor) and b.dtype == a.dtype, (a.dtype, getattr(b, "dtype", None))
        assert torch.equal(a, b)


def test_slot_kind_values_and_latest_id():
    assert (SLOT_ACT, SLOT_BOOT, SLOT_PAD) == (0, 1, 2)
    assert LATEST_NETWORK_ID == "latest"


@pytest.mark.parametrize("state", ["none", "dict", "tuple"])
@pytest.mark.parametrize("with_gs,with_masks,with_units", [
    (False, False, False), (True, True, True), (False, True, False), (True, False, True),
])
def test_chunk_payload_roundtrip_preserves_trees_and_dtypes(state, with_gs, with_masks, with_units):
    chunk = _chunk(with_gs=with_gs, with_masks=with_masks, with_units=with_units, state=state)
    payload = chunk.to_payload()
    assert_no_tensors(payload)
    payload = pickle.loads(pickle.dumps(payload))      # what an mp.Queue does
    assert payload["obs"]["grid"].dtype == np.uint8
    back = TrajectoryChunk.from_payload(payload)
    assert back.agent_id == "a" and back.policy_version == 7
    for f in dataclasses.fields(TrajectoryChunk):
        if f.name not in ("agent_id", "policy_version"):
            _assert_trees_equal(getattr(chunk, f.name), getattr(back, f.name))


def test_chunk_counts_slots_and_acts():
    chunk = _chunk(with_gs=False, with_masks=False, with_units=False, state="none")
    assert chunk.num_slots == 4
    assert chunk.num_acts == 2


def test_chunk_to_moves_every_tensor_including_state_leaves():
    chunk = _chunk(with_gs=True, with_masks=True, with_units=True, state="dict")
    moved = chunk.to("meta")
    leaves = [moved.kind, moved.reward, moved.terminal, moved.reset_after, moved.behavior_logp,
              moved.behavior_unit_logp, moved.obs["grid"], moved.obs["vec"], moved.global_state["map"],
              moved.actions["move"], moved.actions["units"], moved.action_masks["move"],
              moved.action_masks["units"]["unit"], moved.action_masks["units"]["action"],
              moved.initial_state["h"], moved.initial_state["c"]]
    assert all(t.device.type == "meta" for t in leaves)
    assert moved.obs["grid"].dtype == torch.uint8
    assert chunk.kind.device.type == "cpu"            # the original is untouched


def test_lineup_and_results_are_plain_picklable_dataclasses():
    lineup = Lineup(layout="2v2", seats=[SeatAssignment("a"), SeatAssignment("b", "ckpt_v3", collect=False)])
    assert lineup.seats[0].network_id == LATEST_NETWORK_ID and lineup.seats[0].collect is True
    result = MatchResult(
        match_id="w0_e1_ep2", layout="2p", outcome_kind="wdl",
        seats=[SeatResult(0, "player", 0, "a", "latest", 1.0),
               SeatResult(1, "player", 1, "b", "ckpt_v3", -1.0, eliminated_step=4)],
        teams=[TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)],
        episode_length=4,
    )
    assert result.seats[0].eliminated_step is None
    for obj in (lineup, result):
        assert_no_tensors(obj)
        assert pickle.loads(pickle.dumps(obj)) == obj


def test_worker_command_carries_lineups_and_numpy_checkpoints():
    cmd = WorkerCommand(
        lineups=[None, Lineup("solo", [SeatAssignment("a")])],
        new_checkpoints={"a": {"ckpt_v1": state_dict_to_numpy({"w": torch.ones(2)})}},
    )
    assert_no_tensors(cmd)
    assert WorkerCommand(lineups=[None]).new_checkpoints == {}


def test_weight_payload_and_state_dict_helpers_roundtrip():
    model = torch.nn.Linear(3, 2)
    payload = WeightPayload.from_model("a", 5, model)
    assert_no_tensors(payload)
    other = torch.nn.Linear(3, 2)
    other.load_state_dict(payload.to_torch_state_dict())
    assert torch.equal(other.weight, model.weight)
    sd = {"x": torch.ones(2, dtype=torch.bfloat16)}
    assert state_dict_from_numpy(state_dict_to_numpy(sd))["x"].dtype == torch.float32
```

- [ ] **Step 2: Run it and see it fail**

Run: `.venv/bin/python -m pytest tests/unit/test_chunk_v2_types.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.sp2.core.types'`.

- [ ] **Step 3: Implement the types**

`src/colosseum/sp2/core/types.py`:

```python
"""Core data types of the SP2 pipeline: chunk v2, lineups, match results, commands, weights.

Everything that crosses a process boundary has a numpy form: ``TrajectoryChunk.to_payload``,
``WeightPayload`` (numpy ``state_dict``), ``WorkerCommand`` (numpy checkpoints) and the plain
dataclasses ``Lineup`` / ``MatchResult``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from colosseum.core.ipc import numpy_to_tensor, tensor_to_numpy
from colosseum.networks.state import State, state_from_numpy, state_to_numpy
from colosseum.networks.state import tree_map as state_tree_map
from colosseum.sp2.core.tree import Tree, tree_map, tree_to_numpy, tree_to_torch

# Network id of an agent's current weights, as opposed to a checkpoint id ("ckpt_v<N>").
LATEST_NETWORK_ID = "latest"

# Values of TrajectoryChunk.kind (int8).
SLOT_ACT, SLOT_BOOT, SLOT_PAD = 0, 1, 2


def state_dict_to_numpy(state_dict: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Detached CPU numpy copies of a torch ``state_dict`` (bfloat16 -> float32)."""
    return {key: tensor_to_numpy(value) for key, value in state_dict.items()}


def state_dict_from_numpy(state_dict: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """CPU torch tensors from a numpy ``state_dict``; inverse of :func:`state_dict_to_numpy`."""
    return {key: numpy_to_tensor(value) for key, value in state_dict.items()}


def _map_optional(fn: Callable[[Any], Any], tree: Tree | None) -> Tree | None:
    return None if tree is None else tree_map(fn, tree)


@dataclass
class TrajectoryChunk:
    """``S`` consecutive slots of ONE agent (spec block 4).

    Each slot is one of:
    - ``SLOT_ACT``: a decision of a seat (observation, optional ``global_state``, mask,
      action, behavior log-probs, the reward accumulated while it was open, ``terminal``);
    - ``SLOT_BOOT``: only an observation (and ``global_state``) for the learner's bootstrap
      value; no action, no loss;
    - ``SLOT_PAD``: a filler that affects nothing.

    ``reset_after[s]`` resets the model state after slot ``s`` (terminal ACTs, truncation
    BOOTs and PADs). An ACT never sits in the last slot. Trees (``obs``, ``global_state``,
    ``actions``, ``action_masks``) have leaves ``[S, ...]`` with their native dtypes.
    ``behavior_unit_logp`` (``[S, K]``) exists only when the role's action has ``K > 1``
    deciders. ``initial_state`` is the model state before slot 0 (leaves ``[1, ...]``).
    """

    agent_id: str
    policy_version: int
    initial_state: State
    obs: Tree
    global_state: Tree | None
    actions: Tree
    action_masks: Tree | None
    kind: torch.Tensor
    reward: torch.Tensor
    terminal: torch.Tensor
    reset_after: torch.Tensor
    behavior_logp: torch.Tensor
    behavior_unit_logp: torch.Tensor | None

    @property
    def num_slots(self) -> int:
        return int(self.kind.shape[0])

    @property
    def num_acts(self) -> int:
        return int((self.kind == SLOT_ACT).sum())

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> TrajectoryChunk:
        return TrajectoryChunk(
            agent_id=self.agent_id,
            policy_version=self.policy_version,
            initial_state=state_tree_map(fn, self.initial_state),
            obs=tree_map(fn, self.obs),
            global_state=_map_optional(fn, self.global_state),
            actions=tree_map(fn, self.actions),
            action_masks=_map_optional(fn, self.action_masks),
            kind=fn(self.kind),
            reward=fn(self.reward),
            terminal=fn(self.terminal),
            reset_after=fn(self.reset_after),
            behavior_logp=fn(self.behavior_logp),
            behavior_unit_logp=None if self.behavior_unit_logp is None else fn(self.behavior_unit_logp),
        )

    def to(self, device: str | torch.device) -> TrajectoryChunk:
        """Copy with every tensor (state leaves included) on ``device``."""
        return self._apply(lambda t: t.to(device))

    def pin_memory(self) -> TrajectoryChunk:
        """Copy with every tensor in page-locked memory."""
        return self._apply(lambda t: t.pin_memory())

    def to_payload(self) -> dict[str, Any]:
        """Numpy trees + primitives only, for crossing a process boundary."""
        return {
            "agent_id": str(self.agent_id),
            "policy_version": int(self.policy_version),
            "initial_state": state_to_numpy(self.initial_state),
            "obs": tree_to_numpy(self.obs),
            "global_state": None if self.global_state is None else tree_to_numpy(self.global_state),
            "actions": tree_to_numpy(self.actions),
            "action_masks": None if self.action_masks is None else tree_to_numpy(self.action_masks),
            "kind": tensor_to_numpy(self.kind),
            "reward": tensor_to_numpy(self.reward),
            "terminal": tensor_to_numpy(self.terminal),
            "reset_after": tensor_to_numpy(self.reset_after),
            "behavior_logp": tensor_to_numpy(self.behavior_logp),
            "behavior_unit_logp": (
                None if self.behavior_unit_logp is None else tensor_to_numpy(self.behavior_unit_logp)
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TrajectoryChunk:
        """Rebuild a chunk (CPU tensors) from :meth:`to_payload` output."""
        gs, masks, unit_logp = payload["global_state"], payload["action_masks"], payload["behavior_unit_logp"]
        return cls(
            agent_id=str(payload["agent_id"]),
            policy_version=int(payload["policy_version"]),
            initial_state=state_from_numpy(payload["initial_state"]),
            obs=tree_to_torch(payload["obs"]),
            global_state=None if gs is None else tree_to_torch(gs),
            actions=tree_to_torch(payload["actions"]),
            action_masks=None if masks is None else tree_to_torch(masks),
            kind=numpy_to_tensor(payload["kind"]),
            reward=numpy_to_tensor(payload["reward"]),
            terminal=numpy_to_tensor(payload["terminal"]),
            reset_after=numpy_to_tensor(payload["reset_after"]),
            behavior_logp=numpy_to_tensor(payload["behavior_logp"]),
            behavior_unit_logp=None if unit_logp is None else numpy_to_tensor(unit_logp),
        )


@dataclass
class SeatAssignment:
    """Who plays one seat: an agent's latest weights or a checkpoint, and whether it collects."""

    agent_id: str
    network_id: str = LATEST_NETWORK_ID
    collect: bool = True


@dataclass
class Lineup:
    """One match composition: a layout of the game and one assignment per seat of it."""

    layout: str
    seats: list[SeatAssignment]


@dataclass
class SeatResult:
    """One seat of a finished match. ``reward`` is the undiscounted episode return."""

    seat: int
    role: str
    team: int
    agent_id: str
    network_id: str
    reward: float
    eliminated_step: int | None = None


@dataclass
class TeamResult:
    """One team of a finished match: rank (1 = best, ties share) and score."""

    team: int
    rank: float
    score: float


@dataclass
class MatchResult:
    """A finished match: one SeatResult per occupied seat, one TeamResult per team."""

    match_id: str
    layout: str
    outcome_kind: Literal["score", "wdl", "rank"]
    seats: list[SeatResult]
    teams: list[TeamResult]
    episode_length: int


@dataclass
class WorkerCommand:
    """Runtime update from the coordinator to one worker.

    ``lineups[e]`` replaces env ``e``'s lineup at its next episode end (``None`` = keep).
    ``new_checkpoints`` (``{agent_id: {checkpoint_id: numpy state_dict}}``) carries only the
    checkpoints the worker does not have yet; they are loaded at once.
    """

    lineups: list[Lineup | None]
    new_checkpoints: dict[str, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict)


@dataclass
class WeightPayload:
    """Model weights flowing from a learner to workers (numpy, never torch)."""

    agent_id: str
    policy_version: int
    state_dict: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_model(cls, agent_id: str, policy_version: int, model: nn.Module) -> WeightPayload:
        """Snapshot ``model``'s current weights as a numpy payload."""
        return cls(
            agent_id=agent_id,
            policy_version=int(policy_version),
            state_dict=state_dict_to_numpy(model.state_dict()),
        )

    def to_torch_state_dict(self) -> dict[str, torch.Tensor]:
        """CPU torch ``state_dict`` for ``model.load_state_dict``."""
        return state_dict_from_numpy(self.state_dict)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_chunk_v2_types.py -q`
Expected: `18 passed`.

- [ ] **Step 5: Full fast suite and lint**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`
Expected: all pass, zero warnings, `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add src/colosseum/sp2/core/types.py tests/unit/test_chunk_v2_types.py
git commit -m "feat(sp2): chunk v2 with act/boot/pad slots, lineups, team results, worker commands"
git push origin sp2-game-model
```

---

### Task T3.2: Rollout buffers with `act` / `boot` / `pad` slots and parking

Spec block 4 ("Слоты", "Содержимое не-act слотов", "Правила записи" 1 and 5, "Парковка"). The buffer enforces the slot invariants; *which* slot to write is decided by `RolloutLoop` (T3.4).

Invariants enforced here:
- an ACT never takes the last slot (`write_act` needs ≥ 2 free slots), so an open ACT always has room for its BOOT;
- a BOOT follows an open ACT; a PAD follows a slot that ends an episode (never an open ACT);
- BOOT and PAD slots get `ActionSpec.boot_mask()` (units: `unit=False`), zero actions, reward 0, `behavior_logp` 0; a PAD copies the previous slot's observation and `global_state` (zeros could give NaN in a user's attention encoder);
- terminal ACTs, truncation BOOTs and PADs have `reset_after=True`;
- `global_state` is required when the agent's role declares it; `unit_log_probs` is required when `K > 1`;
- buffers are parked only at an episode boundary (`ends_episode`) and never full; an empty buffer goes to the agent's free list. Parked and free buffers are kept **per agent** (agents may have different spaces).

**Files:**
- Create: `src/colosseum/sp2/worker/__init__.py` (empty), `src/colosseum/sp2/worker/buffers.py`
- Test: `tests/unit/test_slot_buffers.py`

**Interfaces:**
- Consumes: `ObsSpec.from_space`, `ObsSpec.allocate(leading)`, `ActionSpec.from_space`, `ActionSpec.allocate_actions(leading)`, `ActionSpec.full_mask(leading)`, `ActionSpec.boot_mask()`, `ActionSpec.has_masks`, `ActionSpec.num_deciders`, `ActionSpec.has_units` (T1.3); `Units` (T1.2); `tree_assign`, `tree_index`, `tree_map` (T1.1); `TrajectoryChunk`, `SLOT_*` (T3.1).
- Produces (contract): `BufferSpec(obs, action, global_state)`, `RolloutBuffer(num_slots, spec)` with `num_slots`, `slots_used`, `free_slots`, `is_full`, `has_open`, `ends_episode`, `num_acts`, `begin`, `write_act`, `add_reward`, `mark_terminal`, `write_boot`, `write_pad`, `build_chunk`, `reset`; `BufferPool(num_slots, specs)` with `acquire`, `park`, `parked_count`, `parked_acts`.
  - Additions (see "Contract notes"): `put_row(dst, idx, src)` (row assignment for a bare array or a dict tree; T3.3 uses it), `RolloutBuffer.reward_sum -> float`, `RolloutBuffer.spec`, `RolloutBuffer.initial_state`, `RolloutBuffer.policy_version`, `BufferPool.parked_reward(agent_id) -> float`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_slot_buffers.py`:

```python
"""RolloutBuffer act/boot/pad slots and BufferPool parking (T3.2)."""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.types import SLOT_ACT, SLOT_BOOT, SLOT_PAD
from colosseum.sp2.envs.spaces import Units
from colosseum.sp2.worker.buffers import BufferPool, BufferSpec, RolloutBuffer

OBS_SPACE = gymnasium.spaces.Dict({
    "grid": gymnasium.spaces.Box(0, 255, (2, 2), np.uint8),
    "vec": gymnasium.spaces.Box(-1.0, 1.0, (3,), np.float32),
})
GS_SPACE = gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32)


def _spec(*, units: bool = False, global_state: bool = False) -> BufferSpec:
    action = Units(3, gymnasium.spaces.Discrete(4)) if units else gymnasium.spaces.Discrete(3)
    return BufferSpec(
        obs=ObsSpec.from_space(OBS_SPACE),
        action=ActionSpec.from_space(action),
        global_state=ObsSpec.from_space(GS_SPACE) if global_state else None,
    )


def _obs(v: int) -> dict:
    return {"grid": np.full((2, 2), v, np.uint8), "vec": np.full(3, v / 10, np.float32)}


def _act(buf: RolloutBuffer, v: int, *, reward: float = 0.0, mask=None, gs=None, unit_lp=None) -> None:
    action = np.full(3, v % 4, np.int64) if unit_lp is not None else np.int64(v % 3)
    buf.write_act(_obs(v), gs, mask, action, -float(v), unit_lp, reward)


def test_act_never_takes_the_last_slot():
    buf = RolloutBuffer(3, _spec())
    _act(buf, 1)
    _act(buf, 2)
    assert buf.free_slots == 1 and buf.has_open
    with pytest.raises(RuntimeError, match="last slot"):
        _act(buf, 3)


def test_boot_requires_an_open_act_and_pad_requires_an_episode_end():
    buf = RolloutBuffer(4, _spec())
    with pytest.raises(RuntimeError):
        buf.write_boot(_obs(0), None, reset_after=False)
    with pytest.raises(RuntimeError):
        buf.write_pad()                       # empty
    _act(buf, 1)
    with pytest.raises(RuntimeError, match="BOOT"):
        buf.write_pad()                       # after an open ACT
    buf.mark_terminal()
    assert not buf.has_open and buf.ends_episode
    with pytest.raises(RuntimeError):
        buf.write_boot(_obs(2), None, reset_after=True)   # terminal ACT is not open
    buf.write_pad()


def test_rewards_go_to_the_open_act_and_terminal_sets_reset_after():
    buf = RolloutBuffer(4, _spec())
    _act(buf, 1, reward=0.5)
    buf.add_reward(1.0)
    buf.add_reward(-0.25)
    buf.mark_terminal()
    with pytest.raises(RuntimeError):
        buf.add_reward(1.0)
    buf.write_pad()
    _act(buf, 2, reward=2.0)
    buf.write_boot(_obs(3), None, reset_after=False)
    chunk = buf.build_chunk("a")
    assert chunk.kind.tolist() == [SLOT_ACT, SLOT_PAD, SLOT_ACT, SLOT_BOOT]
    assert chunk.reward.tolist() == [1.25, 0.0, 2.0, 0.0]
    assert chunk.terminal.tolist() == [True, False, False, False]
    assert chunk.reset_after.tolist() == [True, True, False, False]
    assert chunk.behavior_logp.tolist() == [-1.0, 0.0, -2.0, 0.0]
    assert buf.reward_sum == pytest.approx(3.25)


def test_boot_and_pad_contents():
    spec = _spec(units=True, global_state=True)
    buf = RolloutBuffer(5, spec)
    mask = {"unit": np.array([True, False, True]), "action": np.ones((3, 4), bool)}
    mask["action"][0, 1:] = False
    gs = np.array([0.1, 0.2], np.float32)
    buf.begin(None, 3)
    _act(buf, 1, mask=mask, gs=gs, unit_lp=np.array([-0.1, 0.0, -0.3], np.float32))
    buf.mark_terminal()
    buf.write_pad()                                       # copies slot 0's obs and global_state
    _act(buf, 2, mask=mask, gs=gs * 2, unit_lp=np.array([-0.2, 0.0, -0.4], np.float32))
    buf.write_boot(_obs(9), gs * 3, reset_after=True)     # truncation boot
    buf.write_pad()                                       # copies the boot's obs
    chunk = buf.build_chunk("a")
    assert chunk.kind.tolist() == [SLOT_ACT, SLOT_PAD, SLOT_ACT, SLOT_BOOT, SLOT_PAD]
    assert chunk.reset_after.tolist() == [True, True, False, True, True]
    assert chunk.obs["grid"].dtype == torch.uint8
    assert chunk.obs["grid"][:, 0, 0].tolist() == [1, 1, 2, 9, 9]
    assert torch.allclose(chunk.global_state[:, 0], torch.tensor([0.1, 0.1, 0.2, 0.3, 0.3]))
    # boot/pad masks are ActionSpec.boot_mask() (units: unit=False), actions are zeros
    for s in (1, 3, 4):
        assert not chunk.action_masks["unit"][s].any()
        assert chunk.action_masks["action"][s].all()
        assert int(chunk.actions[s].abs().sum()) == 0
        assert float(chunk.behavior_unit_logp[s].abs().sum()) == 0.0
    assert chunk.action_masks["unit"][0].tolist() == [True, False, True]
    assert chunk.behavior_unit_logp[2].tolist() == pytest.approx([-0.2, 0.0, -0.4])
    assert chunk.policy_version == 3 and chunk.num_acts == 2


def test_boot_and_pad_overwrite_stale_values_of_a_reused_buffer():
    buf = RolloutBuffer(2, _spec())
    mask = np.array([True, False, False])
    _act(buf, 2, mask=mask)
    buf.write_boot(_obs(4), None, reset_after=False)
    buf.build_chunk("a")
    buf.reset()
    _act(buf, 5, mask=np.array([False, True, False]), reward=1.0)
    buf.mark_terminal()
    buf.write_pad()
    chunk = buf.build_chunk("a")
    assert chunk.action_masks[1].tolist() == [True, True, True]
    assert int(chunk.actions[1]) == 0 and float(chunk.reward[1]) == 0.0


def test_unit_log_probs_are_required_only_for_multi_decider_actions():
    single = RolloutBuffer(3, _spec())
    _act(single, 1)                                      # K == 1: no unit log-probs
    multi = RolloutBuffer(3, _spec(units=True))
    with pytest.raises(ValueError, match="unit_log_probs"):
        multi.write_act(_obs(1), None, None, np.zeros(3, np.int64), -1.0, None, 0.0)


def test_global_state_is_required_when_the_role_declares_it():
    buf = RolloutBuffer(3, _spec(global_state=True))
    with pytest.raises(ValueError, match="global_state"):
        _act(buf, 1)


def test_begin_clones_state_and_only_on_an_empty_buffer():
    buf = RolloutBuffer(3, _spec())
    batched = torch.arange(6.0).reshape(3, 2)
    row = batched[1:2]
    buf.begin({"h": row}, 4)
    batched.zero_()
    assert buf.initial_state["h"].tolist() == [[2.0, 3.0]]
    assert buf.initial_state["h"].untyped_storage().nbytes() == 2 * 4
    _act(buf, 1)
    with pytest.raises(RuntimeError):
        buf.begin(None, 5)


def test_build_chunk_needs_a_full_buffer_and_reset_empties_it():
    buf = RolloutBuffer(2, _spec())
    _act(buf, 1)
    with pytest.raises(RuntimeError, match="partial"):
        buf.build_chunk("a")
    buf.write_boot(_obs(2), None, reset_after=False)
    assert buf.is_full and not buf.ends_episode
    buf.reset()
    assert buf.slots_used == 0 and buf.num_acts == 0 and buf.ends_episode


def test_ends_episode_skips_pads():
    buf = RolloutBuffer(5, _spec())
    _act(buf, 1)
    assert not buf.ends_episode
    buf.write_boot(_obs(2), None, reset_after=True)
    assert buf.ends_episode
    buf.write_pad()
    assert buf.ends_episode


def test_pool_prefers_parked_then_free_buffers_per_agent():
    pool = BufferPool(4, {"a": _spec(), "b": _spec(units=True)})
    a1 = pool.acquire("a")
    _act(a1, 1)
    with pytest.raises(RuntimeError, match="middle of an episode"):
        pool.park("a", a1)
    a1.mark_terminal()
    pool.park("a", a1)
    assert pool.acquire("a") is a1                   # a parked buffer comes back first
    fresh = pool.acquire("a")                        # nothing parked or free: a new buffer
    assert fresh is not a1
    pool.park("a", fresh)                            # an empty buffer becomes a free one
    pool.park("a", a1)
    assert pool.parked_count() == 1 and pool.parked_acts("a") == 1 and pool.parked_acts("b") == 0
    assert pool.acquire("b").spec.action.has_units  # never another agent's buffer
    assert pool.acquire("a") is a1                   # parked before free
    assert pool.acquire("a") is fresh
    assert pool.parked_count() == 0


def test_pool_refuses_full_buffers_and_unknown_agents():
    pool = BufferPool(2, {"a": _spec()})
    buf = pool.acquire("a")
    _act(buf, 1)
    buf.write_boot(_obs(2), None, reset_after=True)
    with pytest.raises(RuntimeError, match="full"):
        pool.park("a", buf)
    with pytest.raises(KeyError):
        pool.acquire("zzz")


def test_parked_reward_sums_unsent_rewards():
    pool = BufferPool(4, {"a": _spec()})
    buf = pool.acquire("a")
    _act(buf, 1, reward=1.5)
    buf.add_reward(0.5)
    buf.mark_terminal()
    pool.park("a", buf)
    assert pool.parked_reward("a") == pytest.approx(2.0)
    assert pool.parked_reward("b") == 0.0
```

- [ ] **Step 2: Run it and see it fail**

Run: `.venv/bin/python -m pytest tests/unit/test_slot_buffers.py -q`
Expected: `ModuleNotFoundError: No module named 'colosseum.sp2.worker'`.

- [ ] **Step 3: Implement the buffers**

Create the empty `src/colosseum/sp2/worker/__init__.py`, then `src/colosseum/sp2/worker/buffers.py`:

```python
"""Agent-owned rollout buffers with act / boot / pad slots and parking (spec block 4).

A :class:`RolloutBuffer` holds up to ``num_slots`` consecutive slots of ONE agent; each
collecting (env, seat) holds one buffer exclusively. The slot write rules (which slot to
write when a seat acts, is eliminated or its episode ends) are applied by ``RolloutLoop``;
this module enforces the invariants they rely on:

- an ACT never takes the last slot, so an open ACT always has room for its BOOT;
- BOOT follows an open ACT; PAD follows a slot that ends an episode (never an open ACT);
- BOOT and PAD slots carry ``ActionSpec.boot_mask()`` and zero actions; a PAD copies the
  previous slot's observation and ``global_state`` (zeros could produce NaN in user encoders);
- a buffer is parked only at an episode boundary (empty, or ending with a terminal ACT or
  a BOOT with ``reset_after``), never full (full buffers are sealed at once).

:class:`BufferPool` keeps parked and free buffers per agent (agents may have different
observation/action spaces, so buffers are never shared between agents).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from colosseum.networks.state import State
from colosseum.networks.state import tree_map as state_tree_map
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.tree import Tree, tree_assign, tree_index, tree_map
from colosseum.sp2.core.types import SLOT_ACT, SLOT_BOOT, SLOT_PAD, TrajectoryChunk


@dataclass(frozen=True)
class BufferSpec:
    """Shapes of one agent's slots: its role's observation, action and global-state specs."""

    obs: ObsSpec
    action: ActionSpec
    global_state: ObsSpec | None


def put_row(dst: Tree, idx: Any, src: Tree) -> None:
    """In place ``dst[idx] = src`` for a bare array or for every leaf of a dict tree."""
    if isinstance(dst, dict):
        tree_assign(dst, idx, src)
    else:
        dst[idx] = src


def _to_torch(tree: Tree) -> Tree:
    return tree_map(lambda a: torch.from_numpy(a.copy()), tree)


class RolloutBuffer:
    """Pre-allocated numpy storage for one chunk of one agent's slots."""

    def __init__(self, num_slots: int, spec: BufferSpec) -> None:
        if num_slots < 2:
            raise ValueError(f"a rollout buffer needs at least 2 slots, got {num_slots}")
        S = num_slots
        self.num_slots = S
        self.spec = spec
        self._obs = spec.obs.allocate((S,))
        self._gs = None if spec.global_state is None else spec.global_state.allocate((S,))
        self._actions = spec.action.allocate_actions((S,))
        self._zero_action = spec.action.allocate_actions(())
        self._masks = spec.action.full_mask((S,)) if spec.action.has_masks else None
        self._full_mask = spec.action.full_mask(()) if spec.action.has_masks else None
        self._boot_mask = spec.action.boot_mask() if spec.action.has_masks else None
        self._num_deciders = spec.action.num_deciders
        self._kind = np.zeros(S, dtype=np.int8)
        self._reward = np.zeros(S, dtype=np.float32)
        self._terminal = np.zeros(S, dtype=np.bool_)
        self._reset_after = np.zeros(S, dtype=np.bool_)
        self._logp = np.zeros(S, dtype=np.float32)
        self._unit_logp = (
            np.zeros((S, self._num_deciders), dtype=np.float32) if self._num_deciders > 1 else None
        )
        self._cursor = 0
        self._num_acts = 0
        self.initial_state: State = None
        self.policy_version = 0

    # -- state -----------------------------------------------------------
    @property
    def slots_used(self) -> int:
        return self._cursor

    @property
    def free_slots(self) -> int:
        return self.num_slots - self._cursor

    @property
    def is_full(self) -> bool:
        return self._cursor >= self.num_slots

    @property
    def has_open(self) -> bool:
        """The last slot is a non-terminal ACT (a transition still collecting rewards)."""
        i = self._cursor - 1
        return i >= 0 and self._kind[i] == SLOT_ACT and not self._terminal[i]

    @property
    def ends_episode(self) -> bool:
        """Empty, or the last non-PAD slot is a terminal ACT or a BOOT with ``reset_after``."""
        for i in range(self._cursor - 1, -1, -1):
            kind = self._kind[i]
            if kind == SLOT_PAD:
                continue
            if kind == SLOT_ACT:
                return bool(self._terminal[i])
            return bool(self._reset_after[i])
        return True

    @property
    def num_acts(self) -> int:
        return self._num_acts

    @property
    def reward_sum(self) -> float:
        """Sum of the rewards recorded in the used slots (only ACT slots carry rewards)."""
        return float(self._reward[: self._cursor].sum(dtype=np.float64))

    # -- writing ---------------------------------------------------------
    def begin(self, initial_state: State, policy_version: int) -> None:
        """Record the model state before slot 0 and the policy version that acts in it.

        Only on an empty buffer. State leaves are cloned: a seat's state is a row view of
        its inference group's batched state and would otherwise carry the whole batch.
        """
        if self._cursor != 0:
            raise RuntimeError("begin() on a non-empty buffer")
        self.initial_state = state_tree_map(lambda t: t.detach().clone(), initial_state)
        self.policy_version = int(policy_version)

    def write_act(
        self,
        obs: Tree,
        global_state: Tree | None,
        mask: Tree | None,
        action: Tree,
        log_prob: float,
        unit_log_probs: np.ndarray | None,
        reward: float,
    ) -> None:
        """Append an open ACT carrying ``reward`` (the seat's pending reward)."""
        if self.free_slots < 2:
            raise RuntimeError(
                f"write_act() with {self.free_slots} free slot(s): an ACT never takes the last slot"
            )
        i = self._cursor
        self._write_obs(i, obs, global_state)
        if self._masks is not None:
            put_row(self._masks, i, self._full_mask if mask is None else mask)
        put_row(self._actions, i, action)
        self._kind[i] = SLOT_ACT
        self._reward[i] = reward
        self._terminal[i] = False
        self._reset_after[i] = False
        self._logp[i] = log_prob
        if self._unit_logp is not None:
            if unit_log_probs is None:
                raise ValueError(f"unit_log_probs is required: the action has {self._num_deciders} deciders")
            self._unit_logp[i] = unit_log_probs
        self._cursor += 1
        self._num_acts += 1

    def add_reward(self, reward: float) -> None:
        """Add ``reward`` to the open ACT."""
        self._require_open("add_reward")
        self._reward[self._cursor - 1] += reward

    def mark_terminal(self) -> None:
        """Close the open ACT as the seat's last decision of the episode (bootstrap 0)."""
        self._require_open("mark_terminal")
        self._terminal[self._cursor - 1] = True
        self._reset_after[self._cursor - 1] = True

    def write_boot(self, obs: Tree, global_state: Tree | None, reset_after: bool) -> None:
        """Append a BOOT after the open ACT: the observation the learner bootstraps from."""
        self._require_open("write_boot")
        i = self._cursor
        self._write_obs(i, obs, global_state)
        self._write_non_act(i, SLOT_BOOT, reset_after)

    def write_pad(self) -> None:
        """Append a PAD after a slot that ends an episode; copies that slot's observations."""
        if self._cursor == 0:
            raise RuntimeError("write_pad() on an empty buffer")
        if self.has_open:
            raise RuntimeError("write_pad() after an open ACT; write a BOOT instead")
        if self.is_full:
            raise RuntimeError("write_pad() on a full buffer")
        i = self._cursor
        put_row(self._obs, i, tree_index(self._obs, i - 1))
        if self._gs is not None:
            put_row(self._gs, i, tree_index(self._gs, i - 1))
        self._write_non_act(i, SLOT_PAD, True)

    def _write_obs(self, i: int, obs: Tree, global_state: Tree | None) -> None:
        put_row(self._obs, i, obs)
        if self._gs is not None:
            if global_state is None:
                raise ValueError("global_state is required: the agent's role declares global_state_space")
            put_row(self._gs, i, global_state)

    def _write_non_act(self, i: int, kind: int, reset_after: bool) -> None:
        if self._masks is not None:
            put_row(self._masks, i, self._boot_mask)
        put_row(self._actions, i, self._zero_action)
        self._kind[i] = kind
        self._reward[i] = 0.0
        self._terminal[i] = False
        self._reset_after[i] = reset_after
        self._logp[i] = 0.0
        if self._unit_logp is not None:
            self._unit_logp[i] = 0.0
        self._cursor += 1

    def _require_open(self, what: str) -> None:
        if not self.has_open:
            raise RuntimeError(f"{what}() needs an open ACT as the last slot")

    # -- sealing ---------------------------------------------------------
    def build_chunk(self, agent_id: str) -> TrajectoryChunk:
        """Copy the full buffer into a TrajectoryChunk (torch, CPU). Does not reset."""
        if not self.is_full:
            raise RuntimeError(f"build_chunk() on a partial buffer ({self._cursor}/{self.num_slots})")
        return TrajectoryChunk(
            agent_id=agent_id,
            policy_version=self.policy_version,
            initial_state=self.initial_state,
            obs=_to_torch(self._obs),
            global_state=None if self._gs is None else _to_torch(self._gs),
            actions=_to_torch(self._actions),
            action_masks=None if self._masks is None else _to_torch(self._masks),
            kind=torch.from_numpy(self._kind.copy()),
            reward=torch.from_numpy(self._reward.copy()),
            terminal=torch.from_numpy(self._terminal.copy()),
            reset_after=torch.from_numpy(self._reset_after.copy()),
            behavior_logp=torch.from_numpy(self._logp.copy()),
            behavior_unit_logp=None if self._unit_logp is None else torch.from_numpy(self._unit_logp.copy()),
        )

    def reset(self) -> None:
        self._cursor = 0
        self._num_acts = 0
        self.initial_state = None
        self.policy_version = 0


class BufferPool:
    """Per-worker pool of RolloutBuffers keyed by agent.

    ``acquire(agent)`` returns a parked buffer of that agent if there is one, else a free
    (empty) buffer of that agent, else a new one. ``park(agent, buf)`` takes a buffer
    released by a seat at an episode boundary.
    """

    def __init__(self, num_slots: int, specs: Mapping[str, BufferSpec]) -> None:
        self._num_slots = num_slots
        self._specs = dict(specs)
        self._parked: dict[str, list[RolloutBuffer]] = defaultdict(list)
        self._free: dict[str, list[RolloutBuffer]] = defaultdict(list)

    def acquire(self, agent_id: str) -> RolloutBuffer:
        if agent_id not in self._specs:
            raise KeyError(f"no buffer spec for agent {agent_id!r}")
        if self._parked[agent_id]:
            return self._parked[agent_id].pop(0)
        if self._free[agent_id]:
            return self._free[agent_id].pop()
        return RolloutBuffer(self._num_slots, self._specs[agent_id])

    def park(self, agent_id: str, buf: RolloutBuffer) -> None:
        if buf.slots_used == 0:
            buf.reset()
            self._free[agent_id].append(buf)
            return
        if not buf.ends_episode:
            raise RuntimeError(f"cannot park a buffer of {agent_id!r} in the middle of an episode")
        if buf.is_full:
            raise RuntimeError(f"cannot park a full buffer of {agent_id!r}; seal it first")
        self._parked[agent_id].append(buf)

    def parked_count(self) -> int:
        return sum(len(v) for v in self._parked.values())

    def parked_acts(self, agent_id: str) -> int:
        return sum(b.num_acts for b in self._parked.get(agent_id, []))

    def parked_reward(self, agent_id: str) -> float:
        """Rewards recorded in ``agent_id``'s parked buffers (not yet sent)."""
        return sum(b.reward_sum for b in self._parked.get(agent_id, []))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_slot_buffers.py -q`
Expected: `13 passed`.

- [ ] **Step 5: Full fast suite and lint** (as in T3.1, Step 5).

- [ ] **Step 6: Commit**

```bash
git add src/colosseum/sp2/worker/__init__.py src/colosseum/sp2/worker/buffers.py tests/unit/test_slot_buffers.py
git commit -m "feat(sp2): rollout buffers with act/boot/pad slots and per-agent parking"
git push origin sp2-game-model
```

---

### Task T3.3: `MatchRunner` — the match core for training and eval

Spec block 5 ("`MatchRunner` — общее ядро матча", "Пул моделей передаёт вызывающий", "Состав матча", "Результат матча"), block 1 (lifecycle; the contract checks are `EpisodeTracker`'s, T1.5). Event order, fallbacks, seeds and match ids exactly as in the overview contract:

1. inference for all acting seats of all envs, grouped by `(agent_id, network_id)` (`networks.model.act`, policy only), then `on_act` per acting seat in `(env, seat)` order;
2. one `vec_env.step` for all envs (an env without acting seats gets `{}`);
3. per env in index order: `EpisodeTracker.on_step` → `on_rewards` (every reward of the step, also the elimination step) → `on_terminated` (if any) → if the episode is over: `on_episode_end` → apply the staged lineup (`on_lineup_applied`) → reset the env's model states;
4. one `vec_env.reset` for all finished envs with per-episode seeds (`SeedSequence([seed, env, episode_index])`, or `None` without a seed) → `EpisodeTracker.on_reset`.

The batch of a group is assembled per leaf with `ObsSpec.allocate((n,))` + `put_row` (T3.2), which also casts every leaf to its space's dtype (a `uint8` leaf stays `uint8`). Actions go to the env as numpy trees (one row of the batched action per seat; a `Discrete` action is a numpy scalar). `ActRecord.unit_log_probs` is set only when the role's action has `K > 1` deciders, `ActRecord.global_state` only when the role declares `global_state_space`. A lineup naming a network the pool cannot provide is seated as `SeatAssignment(agent_id, "latest", collect=True)` with one warning per `(agent, network)`; a lineup with an unknown layout, a wrong number of seats, or an agent without a latest model raises `ValueError`. `MatchRunner` computes each seat's return itself and builds `MatchResult` with `resolve_outcome` (default outcome = mean of the team's seat returns), `TeamResult` per team, `SeatResult.eliminated_step` = the episode step at which the seat was terminated. `match_id = f"{match_id_prefix}{env}_ep{k}"`. The `MatchResult` carries everything the episode metrics of spec block 5 need (layout, per-seat role, team, return and elimination step, per-team rank and score, episode length); aggregating them into `metrics.jsonl` / WandB is T5.3. `context` is a prefix that ends with `", "` (e.g. `"worker 3, "`); each env's `EpisodeTracker` gets `context=f"{context}env {e}"`, so contract errors read `"worker 3, env 1, seat 2, episode step 7, layout 4p: ..."`.

Test kit (`tests/game_helpers.py`): `TickGame` replays a list of `Tick`s every episode (tick 0 is the reset result, tick k the result of step k). Its observation of seat `s` at step `t` of episode `k` is `[tag, k, t, s, 0]`, the truncation `final_obs` is `[tag, k, t, s, 1]`, the `global_state` (optional) `[k, t + 0.5 s]`; `mask_fn(k, t, seat)` supplies masks. It is the exact-control game for every worker test of this part; Part A's toy games stay the realistic ones. (Part A already has a `ScriptedGame` that replays raw `StepResult`s for contract-violation tests; `TickGame` is a different tool and must not reuse that name.)

**Files:**
- Create: `src/colosseum/sp2/worker/match_runner.py`
- Modify: `tests/game_helpers.py` (imports; append the T3.3 block)
- Test: `tests/unit/test_match_runner.py`

**Interfaces:**
- Consumes: `VectorEnv(env_fn, num_envs)` / `SubprocessVectorEnv` with `num_envs`, `spec`, `reset(requests)`, `step(actions)`, `close()` (T1.6); `EpisodeTracker(spec, *, max_idle_steps, context)` with `layout`, `on_reset`, `on_step`, `live_seats()` (T1.5); `GameSpec.roles`, `.layouts`, `.layout_size`, `.teams`, `.outcome_kind`, `.role_of`, `StepResult`, `Outcome`, `RoleSpec`, `MultiAgentEnv` (T1.4); `resolve_outcome(outcome, teams, seat_returns, where)` (T1.4); `ObsSpec`, `ActionSpec` (T1.3); `tree_assign`, `tree_map`, `tree_to_numpy`, `tree_to_torch` (T1.1); `PolicyModel`, `act(model, obs, state, action_mask, deterministic) -> ActOutput(actions, log_probs, unit_log_probs, state)` (T2.3); `cat_batch`, `slice_batch` (`colosseum.networks.state`); `Lineup`, `SeatAssignment`, `SeatResult`, `TeamResult`, `MatchResult`, `LATEST_NETWORK_ID` (T3.1); `put_row` (T3.2); test kit: `make_test_model(role, core)` (T2.3).
- Produces (contract): `ModelPool`, `ActRecord`, `EpisodeEnd`, `MatchObserver`, `MatchRunner(*, vec_env, lineups, models, observer=None, seed=None, max_idle_steps=1000, deterministic=False, context="", match_id_prefix="m")` with `num_envs`, `spec`, `lineup(env)`, `set_next_lineup(env, lineup)`, `step() -> int`, `episodes_finished`, `close()` (closes the vector env).
  - Test kit: `Tick`, `TickGame(script, num_seats=1, *, action_space=None, global_state=False, mask_fn=None, tag=0, obs_dtype=np.float32)`, `SCRIPT_OBS_SPACE`, `SCRIPT_GS_SPACE`, `DictModelPool(models)`, `RecordingObserver` (`events`, `kinds(env=None)`).

- [ ] **Step 1: Add the test kit to `tests/game_helpers.py`**

Add to the import block (skip what is already imported):

```python
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import gymnasium
import numpy as np

from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult
```

Append at the end of the module:

```python
# ---------------------------------------------------------------------------
# Part B (T3.3): scripted game, dict model pool, recording observer
# ---------------------------------------------------------------------------


SCRIPT_OBS_SPACE = gymnasium.spaces.Box(-1e6, 1e6, (5,), np.float32)
SCRIPT_GS_SPACE = gymnasium.spaces.Box(-1e6, 1e6, (2,), np.float32)


@dataclass
class Tick:
    """What a TickGame returns at one episode step (tick 0 = the reset result)."""

    acting: Collection[int] = ()
    rewards: Mapping[int, float] = field(default_factory=dict)
    terminated: Collection[int] = ()
    over: bool = False
    truncated: bool = False
    outcome: Outcome | None = None


class TickGame(MultiAgentEnv):
    """Replays a fixed script of Ticks every episode (or cycles through several scripts).

    Observation of seat ``s`` at step ``t`` of episode ``k`` (``k`` counts this env's
    resets from 0): ``[tag, k, t, s, 0]``; its truncation ``final_obs`` is ``[tag, k, t, s, 1]``;
    ``global_state`` (if enabled) is ``[k, t + 0.5 * s]``. ``obs_dtype`` sets the observation
    Box dtype (e.g. ``np.uint8``). ``mask_fn(k, t, seat)`` gives the
    acting seats' masks (``None`` = no masks). ``log`` records ``(k, t, seed, actions)``
    for every reset (``actions`` None) and step.
    """

    def __init__(
        self,
        script: Sequence[Tick] | Sequence[Sequence[Tick]],
        num_seats: int = 1,
        *,
        action_space: gymnasium.Space | None = None,
        global_state: bool = False,
        mask_fn: Callable[[int, int, int], Any] | None = None,
        tag: int = 0,
        obs_dtype: Any = np.float32,
    ) -> None:
        self.scripts = [list(script)] if isinstance(script[0], Tick) else [list(s) for s in script]
        act = action_space if action_space is not None else gymnasium.spaces.Discrete(3)
        gs = SCRIPT_GS_SPACE if global_state else None
        self.obs_dtype = np.dtype(obs_dtype)
        obs = SCRIPT_OBS_SPACE if self.obs_dtype == np.float32 else gymnasium.spaces.Box(
            0, 255, (5,), self.obs_dtype)
        if num_seats == 1:
            self.spec = GameSpec.solo(obs, act, gs)
        else:
            self.spec = GameSpec.symmetric(num_seats, obs, act, gs)
        self.layout = next(iter(self.spec.layouts))
        self.num_seats = num_seats
        self.global_state_enabled = global_state
        self.mask_fn = mask_fn
        self.tag = tag
        self.k = -1
        self.t = 0
        self.eliminated: set[int] = set()
        self.log: list[tuple[int, int, int | None, dict | None]] = []

    def _obs(self, seat: int, final: bool = False) -> np.ndarray:
        return np.array([self.tag, self.k, self.t, seat, 1.0 if final else 0.0], self.obs_dtype)

    def _gs(self, seat: int) -> np.ndarray:
        return np.array([self.k, self.t + 0.5 * seat], np.float32)

    def _result(self, tick: Tick) -> StepResult:
        acting = set(tick.acting)
        res = StepResult(acting=acting, obs={s: self._obs(s) for s in acting})
        if self.mask_fn is not None:
            res.action_masks = {s: self.mask_fn(self.k, self.t, s) for s in acting}
        if self.global_state_enabled:
            res.global_state = {s: self._gs(s) for s in acting}
        return res

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.k += 1
        self.t = 0
        self.eliminated = set()
        self.log.append((self.k, 0, seed, None))
        return self._result(self.scripts[self.k % len(self.scripts)][0])

    def step(self, actions: dict[int, Any]) -> StepResult:
        self.t += 1
        self.log.append((self.k, self.t, None, dict(actions)))
        tick = self.scripts[self.k % len(self.scripts)][self.t]
        res = self._result(tick)
        res.rewards = dict(tick.rewards)
        res.terminated = set(tick.terminated)
        res.episode_over = tick.over
        res.truncated = tick.truncated
        res.outcome = tick.outcome
        self.eliminated |= res.terminated
        if tick.truncated:
            live = [s for s in range(self.num_seats) if s not in self.eliminated]
            res.final_obs = {s: self._obs(s, final=True) for s in live}
            if self.global_state_enabled:
                res.global_state = {s: self._gs(s) for s in live}
        return res


class DictModelPool:
    """ModelPool over a plain ``{(agent_id, network_id): model}`` dict."""

    def __init__(self, models: Mapping[tuple[str, str], Any]) -> None:
        self.models = dict(models)

    def get(self, agent_id: str, network_id: str):
        return self.models.get((agent_id, network_id))


class RecordingObserver:
    """MatchObserver that appends every event to ``events`` as a tuple."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_act(self, env, seat, record):
        self.events.append(("act", env, seat, record))

    def on_rewards(self, env, rewards):
        self.events.append(("rewards", env, dict(rewards)))

    def on_terminated(self, env, seats):
        self.events.append(("terminated", env, list(seats)))

    def on_episode_end(self, env, end):
        self.events.append(("end", env, end))

    def on_lineup_applied(self, env, old, new):
        self.events.append(("lineup", env, old, new))

    def kinds(self, env: int | None = None) -> list[str]:
        return [e[0] for e in self.events if env is None or e[1] == env]
```

- [ ] **Step 2: Write the failing test**

`tests/unit/test_match_runner.py`:

```python
"""MatchRunner: event order, grouped inference, results, lineups, seeds, fallbacks (T3.3)."""

from __future__ import annotations

import logging

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.core.errors import EnvContractError
from colosseum.networks.state import tree_leaves
from colosseum.sp2.core.types import Lineup, SeatAssignment
from colosseum.sp2.envs.game import Outcome
from colosseum.sp2.envs.spaces import Units
from colosseum.sp2.envs.vector import VectorEnv
from colosseum.sp2.networks.model import PolicyModel
from colosseum.sp2.worker.match_runner import MatchRunner
from game_helpers import DictModelPool, RecordingObserver, Tick, TickGame, make_test_model

TURNS = [
    Tick(acting={0}),
    Tick(acting={1}, rewards={1: 0.5}),
    Tick(acting={0}, rewards={0: 1.0}),
    Tick(over=True, rewards={0: 1.0, 1: -1.0}),
]


class CallCountingModel(PolicyModel):
    """Delegates to ``inner`` and records the batch size of every ``step``."""

    def __init__(self, inner: PolicyModel) -> None:
        super().__init__()
        self.inner = inner
        self.batch_sizes: list[int] = []

    def initial_state(self, batch_size, device="cpu"):
        return self.inner.initial_state(batch_size, device)

    def step(self, obs, state, action_mask=None):
        self.batch_sizes.append(int(obs.shape[0]))
        return self.inner.step(obs, state, action_mask)

    def unroll(self, *args, **kwargs):
        return self.inner.unroll(*args, **kwargs)

    def reset_state(self, state, done):
        return self.inner.reset_state(state, done)


def _env_fn(script, num_seats=1, **kwargs):
    return lambda: TickGame(script, num_seats, **kwargs)


def _runner(script, num_seats, lineups, models, *, num_envs=1, observer=None, seed=0, **game_kwargs):
    """A MatchRunner over a sync VectorEnv of TickGames; returns (runner, observer, games)."""
    games = []

    def env_fn():
        games.append(TickGame(script, num_seats, **game_kwargs))
        return games[-1]

    vec = VectorEnv(env_fn, num_envs)
    observer = observer if observer is not None else RecordingObserver()
    runner = MatchRunner(vec_env=vec, lineups=lineups, models=DictModelPool(models), observer=observer,
                         seed=seed, context="worker 0, ", match_id_prefix="w0_e")
    return runner, observer, games


def _role(num_seats=1, **game_kwargs):
    return next(iter(TickGame([Tick(acting={0})], num_seats, **game_kwargs).spec.roles.values()))


def test_events_follow_the_contract_order_and_the_result_is_per_team():
    model = make_test_model(_role(2))
    runner, obs, games = _runner(TURNS, 2, [Lineup("2p", [SeatAssignment("a"), SeatAssignment("b")])],
                               {("a", "latest"): model, ("b", "latest"): model})
    for _ in range(3):
        runner.step()
    assert obs.kinds() == ["act", "rewards", "act", "rewards", "act", "rewards", "end"]
    assert [(e[2], e[3].agent_id) for e in obs.events if e[0] == "act"] == [(0, "a"), (1, "b"), (0, "a")]
    assert [e[2] for e in obs.events if e[0] == "rewards"] == [{1: 0.5}, {0: 1.0}, {0: 1.0, 1: -1.0}]
    end = obs.events[-1][2]
    assert not end.truncated and end.live_seats == [0, 1] and end.final_obs is None
    res = end.result
    assert res.match_id == "w0_e0_ep0" and res.layout == "2p" and res.outcome_kind == "wdl"
    assert res.episode_length == 3
    assert [(s.seat, s.agent_id, s.network_id, s.reward, s.team) for s in res.seats] == [
        (0, "a", "latest", 2.0, 0), (1, "b", "latest", -0.5, 1)]
    assert [(t.team, t.rank, t.score) for t in res.teams] == [(0, 1.0, 2.0), (1, 2.0, -0.5)]
    assert runner.episodes_finished == 1
    # the env was reset right away for the next episode, and acting seat 0 sees its obs
    assert games[0].log[-1][:2] == (1, 0)


def test_act_records_carry_obs_mask_action_logprob_and_pre_state():
    role = _role(1)
    model = make_test_model(role, core="lstm")
    mask = np.array([True, False, True])
    script = [Tick(acting={0}), Tick(acting={0}), Tick(over=True)]
    runner, obs, games = _runner(script, 1, [Lineup("solo", [SeatAssignment("a")])], {("a", "latest"): model},
                               mask_fn=lambda k, t, s: mask)
    runner.step()
    runner.step()
    acts = [e[3] for e in obs.events if e[0] == "act"]
    assert len(acts) == 2
    first, second = acts
    assert first.obs.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0] and second.obs.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert first.mask.tolist() == mask.tolist()
    assert int(first.action) in (0, 2) and int(second.action) in (0, 2)
    assert games[0].log[1][3] == {0: first.action}
    assert first.unit_log_probs is None and first.global_state is None
    assert all(float(x.abs().sum()) == 0.0 for x in tree_leaves(first.pre_state))   # episode start
    assert any(float(x.abs().sum()) > 0.0 for x in tree_leaves(second.pre_state))   # advanced by act 1
    with torch.no_grad():
        step = model.step(torch.from_numpy(first.obs[None]), model.initial_state(1),
                          torch.from_numpy(mask[None]))
    assert first.log_prob == pytest.approx(float(step.dist.log_prob(torch.tensor([int(first.action)]))), abs=1e-6)


def test_units_actions_record_unit_log_probs_and_global_state_reaches_records():
    space = Units(3, gymnasium.spaces.Discrete(2))
    role = _role(1, action_space=space, global_state=True)
    model = make_test_model(role)
    script = [Tick(acting={0}), Tick(over=True)]
    runner, obs, _ = _runner(script, 1, [Lineup("solo", [SeatAssignment("a")])], {("a", "latest"): model},
                             action_space=space, global_state=True)
    runner.step()
    rec = obs.events[0][3]
    assert rec.unit_log_probs.shape == (3,) and rec.unit_log_probs.dtype == np.float32
    assert rec.log_prob == pytest.approx(float(rec.unit_log_probs.sum()), abs=1e-5)
    assert rec.action.shape == (3,) and rec.action.dtype == np.int64
    assert rec.global_state.tolist() == [0.0, 0.0]


def test_inference_is_grouped_by_agent_and_network():
    role = _role(2)
    a, b, b_old = (CallCountingModel(make_test_model(role)) for _ in range(3))
    script = [Tick(acting={0, 1}), Tick(acting={0, 1}), Tick(over=True)]
    lineups = [
        Lineup("2p", [SeatAssignment("a"), SeatAssignment("b")]),
        Lineup("2p", [SeatAssignment("a"), SeatAssignment("b", "ckpt_v1", collect=False)]),
        Lineup("2p", [SeatAssignment("b"), SeatAssignment("a")]),
    ]
    runner, obs, _ = _runner(script, 2, lineups, {("a", "latest"): a, ("b", "latest"): b,
                                                    ("b", "ckpt_v1"): b_old}, num_envs=3)
    runner.step()
    assert a.batch_sizes == [3] and b.batch_sizes == [2] and b_old.batch_sizes == [1]
    acts = [(e[1], e[2]) for e in obs.events if e[0] == "act"]
    assert acts == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]          # (env, seat) order
    assert [e[3].network_id for e in obs.events if e[0] == "act"] == ["latest"] * 3 + ["ckpt_v1", "latest", "latest"]


def test_rewards_of_the_elimination_step_come_before_on_terminated():
    script = [
        Tick(acting={0, 1, 2}),
        Tick(acting={0, 1}, rewards={2: -1.0, 0: 0.25}, terminated={2}),
        Tick(acting={0}, rewards={1: -1.0}, terminated={1}),
        Tick(over=True, rewards={0: 1.0}, outcome=Outcome(team_rank={0: 1, 1: 2, 2: 3})),
    ]
    model = make_test_model(_role(3))
    runner, obs, _ = _runner(script, 3, [Lineup("3p", [SeatAssignment("a")] * 3)], {("a", "latest"): model})
    for _ in range(3):
        runner.step()
    assert obs.kinds() == ["act"] * 3 + ["rewards", "terminated"] + ["act"] * 2 + ["rewards", "terminated"] + [
        "act", "rewards", "end"]
    assert [e[2] for e in obs.events if e[0] == "terminated"] == [[2], [1]]
    end = obs.events[-1][2]
    assert end.live_seats == [0]
    res = end.result
    assert res.outcome_kind == "rank"
    assert [s.eliminated_step for s in res.seats] == [None, 2, 1]
    assert [t.rank for t in res.teams] == [1.0, 2.0, 3.0]


def test_truncation_reports_final_obs_and_global_state_of_live_seats_only():
    script = [
        Tick(acting={0, 1, 2}),
        Tick(acting={0, 1}, terminated={2}),
        Tick(over=True, truncated=True, terminated={1}),
    ]
    model = make_test_model(_role(3, global_state=True))
    runner, obs, _ = _runner(script, 3, [Lineup("3p", [SeatAssignment("a")] * 3)], {("a", "latest"): model},
                             global_state=True)
    runner.step()
    runner.step()
    end = obs.events[-1][2]
    assert end.truncated and end.live_seats == [0]
    assert sorted(end.final_obs) == [0] and end.final_obs[0].tolist() == [0.0, 0.0, 2.0, 0.0, 1.0]
    assert sorted(end.final_global_state) == [0] and end.final_global_state[0].tolist() == [0.0, 2.0]


def test_next_lineup_is_applied_at_the_episode_end_and_states_reset():
    role = _role(2)
    model_a, model_b = make_test_model(role, core="gru"), make_test_model(role, core="gru")
    script = [Tick(acting={0, 1}), Tick(acting={0, 1}), Tick(over=True)]
    runner, obs, _ = _runner(script, 2, [Lineup("2p", [SeatAssignment("a"), SeatAssignment("a")])],
                             {("a", "latest"): model_a, ("b", "latest"): model_b})
    runner.step()
    new = Lineup("2p", [SeatAssignment("b"), SeatAssignment("a", collect=False)])
    runner.set_next_lineup(0, new)
    assert runner.lineup(0).seats[0].agent_id == "a"          # not mid-episode
    runner.step()
    kinds = obs.kinds()
    assert kinds[-2:] == ["end", "lineup"]
    _, _, old, applied = obs.events[-1]
    assert [s.agent_id for s in old.seats] == ["a", "a"]
    assert applied == new and runner.lineup(0) == new
    runner.step()                                             # first step of the next episode
    acts = [e[3] for e in obs.events if e[0] == "act"][-2:]
    assert [r.agent_id for r in acts] == ["b", "a"]
    assert all(float(x.abs().sum()) == 0.0 for r in acts for x in tree_leaves(r.pre_state))


def test_missing_network_falls_back_to_latest_collecting_with_one_warning(caplog):
    model = make_test_model(_role(2))
    script = [Tick(acting={0, 1}), Tick(over=True)]
    lineup = Lineup("2p", [SeatAssignment("a"), SeatAssignment("a", "ckpt_v9", collect=False)])
    with caplog.at_level(logging.WARNING, logger="colosseum.sp2.worker.match_runner"):
        runner, _, _ = _runner(script, 2, [lineup, lineup], {("a", "latest"): model}, num_envs=2)
    assert runner.lineup(0).seats[1] == SeatAssignment("a", "latest", True)
    assert runner.lineup(1).seats[1] == SeatAssignment("a", "latest", True)
    assert sum("ckpt_v9" in r.getMessage() for r in caplog.records) == 1


def test_lineups_are_validated():
    model = make_test_model(_role(2))
    script = [Tick(acting={0, 1}), Tick(over=True)]
    with pytest.raises(ValueError, match="layout"):
        _runner(script, 2, [Lineup("9p", [SeatAssignment("a")] * 9)], {("a", "latest"): model})
    with pytest.raises(ValueError, match="seats"):
        _runner(script, 2, [Lineup("2p", [SeatAssignment("a")])], {("a", "latest"): model})
    with pytest.raises(ValueError, match="no model"):
        _runner(script, 2, [Lineup("2p", [SeatAssignment("zzz")] * 2)], {("a", "latest"): model})
    with pytest.raises(ValueError, match="one lineup per env"):
        _runner(script, 2, [Lineup("2p", [SeatAssignment("a")] * 2)], {("a", "latest"): model}, num_envs=2)


def test_episode_seeds_are_deterministic_per_env_and_episode():
    script = [Tick(acting={0}), Tick(over=True)]
    model = make_test_model(_role(1))

    def seeds(seed):
        runner, _, games = _runner(script, 1, [Lineup("solo", [SeatAssignment("a")])] * 2,
                                 {("a", "latest"): model}, num_envs=2, seed=seed)
        for _ in range(2):
            runner.step()
        return [[entry[2] for entry in env.log if entry[3] is None] for env in games]

    first, again, other, none = seeds(7), seeds(7), seeds(8), seeds(None)
    assert first == again and first != other
    assert len({s for env in first for s in env}) == 6       # 2 envs x 3 resets, all distinct
    assert none == [[None] * 3, [None] * 3]


def test_match_ids_count_episodes_per_env():
    script = [Tick(acting={0}), Tick(over=True)]
    model = make_test_model(_role(1))
    runner, obs, _ = _runner(script, 1, [Lineup("solo", [SeatAssignment("a")])] * 2, {("a", "latest"): model},
                             num_envs=2)
    for _ in range(2):
        runner.step()
    ids = [e[2].result.match_id for e in obs.events if e[0] == "end"]
    assert ids == ["w0_e0_ep0", "w0_e1_ep0", "w0_e0_ep1", "w0_e1_ep1"]
    assert all(e[2].result.outcome_kind == "score" for e in obs.events if e[0] == "end")


def test_idle_ticks_step_the_env_with_no_actions():
    script = [Tick(acting={0}), Tick(acting=()), Tick(acting=()), Tick(acting={0}), Tick(over=True)]
    model = make_test_model(_role(1))
    runner, obs, games = _runner(script, 1, [Lineup("solo", [SeatAssignment("a")])], {("a", "latest"): model})
    for _ in range(4):
        runner.step()
    log = games[0].log
    assert log[1][3] == {0: obs.events[0][3].action}
    assert log[2][3] == {} and log[3][3] == {}            # idle ticks: step({})
    assert list(log[4][3]) == [0]
    assert obs.kinds().count("act") == 2


def test_env_contract_errors_carry_the_worker_and_env_context():
    script = [Tick(acting={0})] + [Tick(acting=())] * 5 + [Tick(over=True)]
    model = make_test_model(_role(1))
    vec = VectorEnv(_env_fn(script, 1), 2)
    runner = MatchRunner(vec_env=vec, lineups=[Lineup("solo", [SeatAssignment("a")])] * 2,
                         models=DictModelPool({("a", "latest"): model}), max_idle_steps=3, context="worker 4, ")
    with pytest.raises(EnvContractError, match="worker 4, env 0"):
        for _ in range(6):
            runner.step()


def test_deterministic_runner_takes_the_mode():
    role = _role(1)
    model = make_test_model(role)
    script = [Tick(acting={0})] * 20 + [Tick(over=True)]
    vec = VectorEnv(_env_fn(script, 1), 1)
    obs = RecordingObserver()
    runner = MatchRunner(vec_env=vec, lineups=[Lineup("solo", [SeatAssignment("a")])],
                         models=DictModelPool({("a", "latest"): model}), observer=obs, deterministic=True)
    for _ in range(5):
        runner.step()
    for rec in (e[3] for e in obs.events if e[0] == "act"):
        with torch.no_grad():
            dist = model.step(torch.from_numpy(rec.obs[None]), None).dist
        assert int(rec.action) == int(dist.mode()[0])
```

- [ ] **Step 3: Run it and see it fail**

Run: `.venv/bin/python -m pytest tests/unit/test_match_runner.py -q`
Expected: `ModuleNotFoundError: No module named 'colosseum.sp2.worker.match_runner'`.

- [ ] **Step 4: Implement `MatchRunner`**

`src/colosseum/sp2/worker/match_runner.py`:

```python
"""MatchRunner: the match core shared by training (RolloutLoop) and eval (spec block 5).

It owns a vector env, one :class:`Lineup` per env, one ``EpisodeTracker`` per env and the
model state of every occupied seat. Each :meth:`MatchRunner.step`:

1. runs inference for every acting seat of every env, grouped by ``(agent_id, network_id)``
   (policy only, ``networks.model.act``), then calls ``observer.on_act`` per acting seat in
   ``(env, seat)`` order;
2. steps every env once (``vec_env.step``; an env without acting seats gets ``{}``);
3. per env in index order: ``EpisodeTracker.on_step`` (contract checks, normalized masks)
   -> ``on_rewards`` (every reward of the step, including the elimination step) ->
   ``on_terminated`` (if any seat was eliminated) -> if the episode is over:
   ``on_episode_end`` (with the :class:`MatchResult`) -> apply the next lineup
   (``on_lineup_applied``) -> reset the env's model states;
4. resets all finished envs in one ``vec_env.reset`` with per-episode seeds.

A lineup naming a network the model pool cannot provide is seated as the agent's latest
weights with ``collect=True`` (SP1 rule; one warning per (agent, network)).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from colosseum.networks.state import State, cat_batch, slice_batch
from colosseum.sp2.core.outcomes import resolve_outcome
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.tree import Tree, tree_map, tree_to_numpy, tree_to_torch
from colosseum.sp2.core.types import (
    LATEST_NETWORK_ID,
    Lineup,
    MatchResult,
    SeatAssignment,
    SeatResult,
    TeamResult,
)
from colosseum.sp2.envs.contract import EpisodeTracker
from colosseum.sp2.envs.game import GameSpec, StepResult
from colosseum.sp2.envs.vector import SubprocessVectorEnv, VectorEnv
from colosseum.sp2.networks.model import PolicyModel, act
from colosseum.sp2.worker.buffers import put_row

logger = logging.getLogger(__name__)

__all__ = ["ActRecord", "EpisodeEnd", "MatchObserver", "MatchRunner", "ModelPool"]


class ModelPool(Protocol):
    def get(self, agent_id: str, network_id: str) -> PolicyModel | None: ...


@dataclass
class ActRecord:
    """One decision of one seat, as the observer sees it (numpy only, plus the model state)."""

    agent_id: str
    network_id: str
    obs: Tree
    global_state: Tree | None
    mask: Tree | None
    action: Tree
    log_prob: float
    unit_log_probs: np.ndarray | None
    pre_state: State


@dataclass
class EpisodeEnd:
    """How an env's episode ended. ``live_seats`` excludes every eliminated seat."""

    truncated: bool
    live_seats: list[int]
    final_obs: dict[int, Tree] | None
    final_global_state: dict[int, Tree] | None
    result: MatchResult


class MatchObserver(Protocol):
    def on_act(self, env: int, seat: int, record: ActRecord) -> None: ...
    def on_rewards(self, env: int, rewards: dict[int, float]) -> None: ...
    def on_terminated(self, env: int, seats: list[int]) -> None: ...
    def on_episode_end(self, env: int, end: EpisodeEnd) -> None: ...
    def on_lineup_applied(self, env: int, old: Lineup, new: Lineup) -> None: ...


@dataclass
class _RoleInfo:
    obs: ObsSpec
    action: ActionSpec
    has_global_state: bool


class _EnvState:
    """Per-env bookkeeping of the runner."""

    def __init__(self, tracker: EpisodeTracker, lineup: Lineup) -> None:
        self.tracker = tracker
        self.lineup = lineup
        self.next_lineup: Lineup | None = None
        self.result: StepResult | None = None
        self.masks: dict[int, Tree | None] = {}
        self.states: dict[int, State] = {}
        self.returns: list[float] = []
        self.eliminated_step: dict[int, int] = {}
        self.episode_index = 0
        self.length = 0


class MatchRunner:
    """Runs matches on a vector env for given lineups and a model pool (module docstring)."""

    def __init__(
        self,
        *,
        vec_env: VectorEnv | SubprocessVectorEnv,
        lineups: Sequence[Lineup],
        models: ModelPool,
        observer: MatchObserver | None = None,
        seed: int | None = None,
        max_idle_steps: int = 1000,
        deterministic: bool = False,
        context: str = "",
        match_id_prefix: str = "m",
    ) -> None:
        self._vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.spec: GameSpec = vec_env.spec
        if len(lineups) != self.num_envs:
            raise ValueError(f"need one lineup per env: {len(lineups)} lineups for {self.num_envs} envs")
        self._models = models
        self._observer = observer
        self._seed = seed
        self._deterministic = deterministic
        self._context = context
        self._prefix = match_id_prefix
        self._episodes_finished = 0
        self._warned_missing: set[tuple[str, str]] = set()
        self._roles = {
            name: _RoleInfo(
                obs=ObsSpec.from_space(role.observation_space),
                action=ActionSpec.from_space(role.action_space),
                has_global_state=role.global_state_space is not None,
            )
            for name, role in self.spec.roles.items()
        }
        self._envs: list[_EnvState] = []
        for e, lineup in enumerate(lineups):
            tracker = EpisodeTracker(self.spec, max_idle_steps=max_idle_steps, context=f"{context}env {e}")
            env_state = _EnvState(tracker, self._resolve(lineup, e))
            self._start_episode_state(env_state)
            self._envs.append(env_state)
        self._reset_envs(list(range(self.num_envs)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lineup(self, env: int) -> Lineup:
        """Env ``env``'s current lineup (after the missing-network fallback)."""
        return self._envs[env].lineup

    def set_next_lineup(self, env: int, lineup: Lineup) -> None:
        """Stage ``lineup`` for env ``env``; it is applied at the env's next episode end."""
        self._check_lineup(lineup, env)
        self._envs[env].next_lineup = lineup

    @property
    def episodes_finished(self) -> int:
        return self._episodes_finished

    def step(self) -> int:
        """One step of every env (see the module docstring). Returns the env steps taken."""
        actions = self._infer()
        results = self._vec_env.step(actions)
        finished: list[int] = []
        for e in range(self.num_envs):
            if self._after_step(e, actions[e], results[e]):
                finished.append(e)
        if finished:
            self._reset_envs(finished)
        return self.num_envs

    def close(self) -> None:
        self._vec_env.close()

    # ------------------------------------------------------------------
    # Lineups
    # ------------------------------------------------------------------

    def _check_lineup(self, lineup: Lineup, env: int) -> None:
        if lineup.layout not in self.spec.layouts:
            raise ValueError(
                f"{self._context}env {env}: lineup layout {lineup.layout!r} is not one of "
                f"{sorted(self.spec.layouts)}"
            )
        size = self.spec.layout_size(lineup.layout)
        if len(lineup.seats) != size:
            raise ValueError(
                f"{self._context}env {env}: lineup for layout {lineup.layout!r} has "
                f"{len(lineup.seats)} seats, the layout has {size}"
            )

    def _resolve(self, lineup: Lineup, env: int) -> Lineup:
        """Validate ``lineup`` and replace networks the pool cannot provide by latest + collect."""
        self._check_lineup(lineup, env)
        seats = []
        for assignment in lineup.seats:
            aid, net = assignment.agent_id, assignment.network_id
            if self._models.get(aid, net) is not None:
                seats.append(SeatAssignment(aid, net, assignment.collect))
                continue
            if self._models.get(aid, LATEST_NETWORK_ID) is None:
                raise ValueError(f"{self._context}env {env}: the model pool has no model for agent {aid!r}")
            if (aid, net) not in self._warned_missing:
                self._warned_missing.add((aid, net))
                logger.warning(
                    f"{self._context}network {net!r} of {aid!r} is not loaded; "
                    f"seating {LATEST_NETWORK_ID!r} (collecting) instead"
                )
            seats.append(SeatAssignment(aid, LATEST_NETWORK_ID, True))
        return Lineup(layout=lineup.layout, seats=seats)

    def _model(self, assignment: SeatAssignment) -> PolicyModel:
        model = self._models.get(assignment.agent_id, assignment.network_id)
        if model is None:
            raise RuntimeError(
                f"{self._context}model ({assignment.agent_id!r}, {assignment.network_id!r}) disappeared from the pool"
            )
        return model

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _infer(self) -> dict[int, dict[int, Tree]]:
        """Batched inference for all acting seats; ``on_act`` per seat in (env, seat) order."""
        groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        for e, env_state in enumerate(self._envs):
            for seat in sorted(env_state.result.acting):
                a = env_state.lineup.seats[seat]
                groups[(a.agent_id, a.network_id)].append((e, seat))

        actions: dict[int, dict[int, Tree]] = {e: {} for e in range(self.num_envs)}
        records: dict[tuple[int, int], ActRecord] = {}
        for (aid, net), seats in groups.items():
            model = self._models.get(aid, net)
            first_env, first_seat = seats[0]
            info = self._roles[self.spec.role_of(self._envs[first_env].tracker.layout, first_seat)]
            n = len(seats)
            obs_batch = info.obs.allocate((n,))
            mask_batch = info.action.full_mask((n,)) if info.action.has_masks else None
            for j, (e, seat) in enumerate(seats):
                env_state = self._envs[e]
                put_row(obs_batch, j, env_state.result.obs[seat])
                if mask_batch is not None:
                    put_row(mask_batch, j, env_state.masks[seat])
            state_batch = cat_batch([self._envs[e].states[seat] for e, seat in seats])
            out = act(model, tree_to_torch(obs_batch), state_batch,
                      None if mask_batch is None else tree_to_torch(mask_batch),
                      deterministic=self._deterministic)
            act_np = tree_to_numpy(out.actions)
            log_probs = out.log_probs.float().cpu().numpy()
            unit_lps = out.unit_log_probs.float().cpu().numpy() if info.action.num_deciders > 1 else None
            for j, (e, seat) in enumerate(seats):
                env_state = self._envs[e]
                action = tree_map(lambda leaf, j=j: leaf[j].copy(), act_np)
                actions[e][seat] = action
                gs = env_state.result.global_state
                records[(e, seat)] = ActRecord(
                    agent_id=aid,
                    network_id=net,
                    obs=env_state.result.obs[seat],
                    global_state=gs[seat] if info.has_global_state and gs is not None else None,
                    mask=env_state.masks.get(seat),
                    action=action,
                    log_prob=float(log_probs[j]),
                    unit_log_probs=None if unit_lps is None else unit_lps[j].copy(),
                    pre_state=env_state.states[seat],
                )
                env_state.states[seat] = None if out.state is None else slice_batch(out.state, j)
        if self._observer is not None:
            for key in sorted(records):
                self._observer.on_act(key[0], key[1], records[key])
        return actions

    def _after_step(self, e: int, actions: dict[int, Tree], result: StepResult) -> bool:
        """Process env ``e``'s step result; True if its episode ended."""
        env_state = self._envs[e]
        env_state.masks = env_state.tracker.on_step(actions, result)
        env_state.length += 1
        for seat, r in result.rewards.items():
            env_state.returns[seat] += float(r)
        obs = self._observer
        if obs is not None:
            obs.on_rewards(e, {int(s): float(r) for s, r in result.rewards.items()})
        if result.terminated:
            for seat in result.terminated:
                env_state.eliminated_step[seat] = env_state.length
            if obs is not None:
                obs.on_terminated(e, sorted(result.terminated))
        if not result.episode_over:
            env_state.result = result
            return False

        match_result = self._match_result(e, result)
        self._episodes_finished += 1
        if obs is not None:
            live = env_state.tracker.live_seats()
            gs = result.global_state
            obs.on_episode_end(e, EpisodeEnd(
                truncated=bool(result.truncated),
                live_seats=live,
                final_obs=dict(result.final_obs) if result.truncated and result.final_obs is not None else None,
                final_global_state=(
                    {s: gs[s] for s in live if s in gs} if result.truncated and gs is not None else None
                ),
                result=match_result,
            ))
        if env_state.next_lineup is not None:
            old, new = env_state.lineup, self._resolve(env_state.next_lineup, e)
            env_state.lineup, env_state.next_lineup = new, None
            if obs is not None:
                obs.on_lineup_applied(e, old, new)
        env_state.episode_index += 1
        self._start_episode_state(env_state)
        return True

    def _match_result(self, e: int, result: StepResult) -> MatchResult:
        env_state = self._envs[e]
        layout = env_state.tracker.layout
        teams = self.spec.teams(layout)
        where = f"{self._context}env {e}, layout {layout}"
        ranks, scores = resolve_outcome(result.outcome, teams, env_state.returns, where)
        seat_specs = self.spec.layouts[layout]
        seats = [
            SeatResult(
                seat=s,
                role=seat_specs[s].role,
                team=seat_specs[s].team,
                agent_id=a.agent_id,
                network_id=a.network_id,
                reward=float(env_state.returns[s]),
                eliminated_step=env_state.eliminated_step.get(s),
            )
            for s, a in enumerate(env_state.lineup.seats)
        ]
        return MatchResult(
            match_id=f"{self._prefix}{e}_ep{env_state.episode_index}",
            layout=layout,
            outcome_kind=self.spec.outcome_kind(layout),
            seats=seats,
            teams=[TeamResult(team=t, rank=float(ranks[t]), score=float(scores[t])) for t in range(len(teams))],
            episode_length=env_state.length,
        )

    def _start_episode_state(self, env_state: _EnvState) -> None:
        """Fresh per-episode bookkeeping and initial model states for the env's lineup."""
        seats = env_state.lineup.seats
        env_state.states = {s: self._model(a).initial_state(1) for s, a in enumerate(seats)}
        env_state.returns = [0.0] * len(seats)
        env_state.eliminated_step = {}
        env_state.length = 0

    def _episode_seed(self, e: int, k: int) -> int | None:
        if self._seed is None:
            return None
        return int(np.random.SeedSequence([int(self._seed), e, k]).generate_state(1)[0])

    def _reset_envs(self, envs: list[int]) -> None:
        requests = {
            e: (self._episode_seed(e, self._envs[e].episode_index), self._envs[e].lineup.layout) for e in envs
        }
        results = self._vec_env.reset(requests)
        for e in envs:
            env_state = self._envs[e]
            env_state.masks = env_state.tracker.on_reset(env_state.lineup.layout, results[e])
            env_state.result = results[e]
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_match_runner.py -q`
Expected: `14 passed`.

- [ ] **Step 6: Full fast suite and lint** (as in T3.1, Step 5).

- [ ] **Step 7: Commit**

```bash
git add src/colosseum/sp2/worker/match_runner.py tests/unit/test_match_runner.py tests/game_helpers.py
git commit -m "feat(sp2): MatchRunner with grouped inference, lineups, team results and per-episode seeds"
git push origin sp2-game-model
```

---

### Task T3.4: `RolloutLoop`, the worker process, chunk-structure and lifecycle contract tests

Spec block 4 ("Правила записи", "Следующий буфер", "Парковка", "Счёт данных"), block 5 (`RolloutLoop` as an observer, lineups via `WorkerCommand`, the missing-network fallback), section 6 (contract tests: seat lifecycle, chunk structure, parking without loss, reward accounting).

`RolloutLoop` is the `MatchRunner`'s model pool (`get(agent_id, network_id)`: each agent's latest model plus frozen checkpoints) and its observer. Slot rules it applies for every **collecting** seat (`_SeatTrack` = agent, buffer, pending reward):

- `on_act`: if the buffer has exactly one free slot — with an open ACT write a BOOT (the seat's current observation and `global_state`, `reset_after=False`), without one write a PAD; seal (send) the chunk and reset **the same buffer**; then `begin(pre_state, version)` if the buffer is empty, and `write_act` with the seat's pending reward. The "next buffer" at a chunk boundary is therefore always the seat's own buffer after the reset, never a parked one.
- `on_rewards`: to the open ACT, else to the seat's pending reward.
- `on_terminated`: the open ACT becomes terminal (`reset_after`). Because `MatchRunner` calls `on_rewards` first, the elimination step's reward lands on the ACT before it is closed.
- `on_episode_end`: every remaining open ACT (live seats; eliminated ones are already closed) gets a BOOT with `final_obs[seat]` / `final_global_state[seat]` and `reset_after=True` on truncation (sealed at once if that fills the buffer), else becomes terminal. A seat with a non-zero pending reward never acted in the episode: the reward is dropped and `dropped_reward_episodes` counts it. Then `report_result(end.result)`.
- `on_lineup_applied` (always at an episode end, after `on_episode_end`): a seat whose collecting agent changes (or that stops collecting) parks its buffer; a seat that starts collecting acquires a buffer (parked first).
- `begin` records the agent's current policy version, so a chunk's `policy_version` is the version at its first slot, also after parking.
- `WorkerCommand.new_checkpoints` are loaded at once; `WorkerCommand.lineups[e]` (non-`None`) is staged with `MatchRunner.set_next_lineup(e, ...)` and applied at env `e`'s next episode end.

`rollout_worker_process` is SP1's wrapper (thread limits, queues → `LoopIO`, newest-wins weights, non-blocking results, stats, `cancel_join_thread` on exit) with `agent_roles`, `lineups`, `max_idle_steps` added and `gamma`, `slot_*_map`, `collect_mask` removed. `_drain_commands` merges all drained commands: checkpoints are merged (deltas are sent once), lineups are merged per env (a later non-`None` lineup wins; `None` keeps the earlier one).

**Files:**
- Create: `src/colosseum/sp2/worker/rollout_loop.py`, `src/colosseum/sp2/worker/rollout_worker.py`, `tests/contract/game_harness.py`
- Modify: `tests/game_helpers.py` (imports; append the T3.4 block), `pyproject.toml` (`known-first-party`)
- Test: `tests/contract/test_chunk_v2_rules.py`, `tests/contract/test_rollout_loop_lineups.py`, `tests/unit/test_rollout_worker_v2.py`

**Interfaces:**
- Consumes: `MatchRunner`, `ActRecord`, `EpisodeEnd` (T3.3); `BufferPool`, `BufferSpec`, `RolloutBuffer` incl. `reward_sum` and `BufferPool.parked_reward` (T3.2); `agent_role_spec(spec, roles)` (T2.4); `ObsSpec`, `ActionSpec` (T1.3); `VectorEnv`, `SubprocessVectorEnv` (T1.6); `MultiAgentEnv` (T1.4); `Lineup`, `MatchResult`, `TrajectoryChunk`, `WeightPayload`, `WorkerCommand`, `state_dict_from_numpy`, `LATEST_NETWORK_ID` (T3.1); `BatchedCounter`, `SharedCounter`, `drain_latest`, `assert_no_tensors` (`colosseum.core.ipc`); `configure_torch_threads` (`colosseum.core.threads`); `WORKER_STATS_INTERVAL_SEC` (`colosseum.metrics.aggregator`; T5.3's `sp2` copy must keep this constant); test kit `Tick`, `TickGame`, `make_test_model` (T3.3, T2.3).
- Produces (contract): `LoopIO`, `RolloutLoop(...)` with `step`, `sync_weights`, `run`, `close`, `stats` (keys exactly `chunks_sent, env_steps, episodes, parked_buffers, dropped_reward_episodes, recorded_transitions, recorded_transitions/<agent>, buffered_transitions/<agent>`; transitions = ACT slots), `rollout_worker_process(...)`.
  - Additions (see "Contract notes"): `RolloutLoop.get(agent_id, network_id)` (the `ModelPool` method), the five `MatchObserver` methods, `RolloutLoop.buffered_reward(agent_id) -> float`; `report_worker_stats(q, worker_id, stats)` and `_drain_commands(q)` in `rollout_worker`.
  - Test kit: `NumpyOnlyQueue` (`tests/game_helpers.py`); `game_harness`: `Collected`, `GameFactory`, `Slot`, `lineup`, `make_loop`, `run_steps`, `run_until_chunks`, `kinds`, `slot_steps`, `seat_chunks`.

- [ ] **Step 1: Test kit**

In `pyproject.toml`, add `"game_harness"` (and `"game_helpers"` if missing) to `[tool.ruff.lint.isort] known-first-party`, keeping the list sorted:

```toml
known-first-party = ["cli_runner", "colosseum", "dataflow_helpers", "examples", "game_harness", "game_helpers", "harness", "helpers", "learning_envs", "ttt_eval"]
```

Add to the import block of `tests/game_helpers.py` (skip what is already imported):

```python
import queue

from colosseum.core.ipc import assert_no_tensors
```

and append:

```python
# ---------------------------------------------------------------------------
# Part B (T3.4): a queue that rejects tensors
# ---------------------------------------------------------------------------


class NumpyOnlyQueue(queue.Queue):
    """``queue.Queue`` that raises if an item contains a ``torch.Tensor`` (process-boundary rule)."""

    def put(self, item, block=True, timeout=None):
        assert_no_tensors(item, "queued item")
        super().put(item, block, timeout)

    def put_nowait(self, item):
        self.put(item, block=False)

    def cancel_join_thread(self) -> None:
        pass
```

Create `tests/contract/game_harness.py`:

```python
"""Drive the real SP2 RolloutLoop (and, from T4.3, the real APPO) in-process.

Chunks come from TickGame (tests/game_helpers.py), whose observations encode
``[env, episode, step, seat, is_final]``, so every slot of a chunk can be traced back to
the env step it came from.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from colosseum.sp2.core.types import (
    SLOT_ACT,
    SLOT_BOOT,
    SLOT_PAD,
    Lineup,
    MatchResult,
    SeatAssignment,
    TrajectoryChunk,
    WeightPayload,
    WorkerCommand,
)
from colosseum.sp2.worker.rollout_loop import LoopIO, RolloutLoop
from game_helpers import TickGame

KIND_LETTER = {SLOT_ACT: "A", SLOT_BOOT: "B", SLOT_PAD: "P"}


@dataclass
class Collected:
    """In-memory LoopIO: everything the loop emits is appended to these lists.

    ``weights[agent_id]`` holds WeightPayloads handed out one per ``poll_weights`` call;
    ``commands`` are handed out one per ``poll_command`` call.
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


class GameFactory:
    """Env factory: the i-th call builds ``TickGame(*games[i % n], tag=i)``.

    Each entry of ``games`` is a ``(script, num_seats)`` pair; ``game_kwargs`` go to every game.
    Sync vector envs build their envs in index order, so ``tag`` (the observation's first
    entry) is the env index. ``created`` keeps the built games.
    """

    def __init__(self, *games: tuple[Sequence[Any], int], **game_kwargs: Any) -> None:
        self.games = list(games)
        self.game_kwargs = game_kwargs
        self.created: list[TickGame] = []

    def __call__(self) -> TickGame:
        script, num_seats = self.games[len(self.created) % len(self.games)]
        game = TickGame(script, num_seats, tag=len(self.created), **self.game_kwargs)
        self.created.append(game)
        return game


def lineup(layout: str, *agents: str | SeatAssignment) -> Lineup:
    """``lineup("2p", "a", SeatAssignment("b", collect=False))``; strings are latest + collect."""
    return Lineup(layout, [a if isinstance(a, SeatAssignment) else SeatAssignment(a) for a in agents])


def make_loop(
    env_fn: Callable[[], Any],
    model_factories: Mapping[str, Callable[[], Any]],
    lineups: Sequence[Lineup],
    *,
    chunk_length: int = 4,
    collected: Collected | None = None,
    agent_roles: Mapping[str, Sequence[str]] | None = None,
    seed: int = 0,
    weight_sync_interval: float = 0.0,
    **kwargs: Any,
) -> tuple[RolloutLoop, Collected]:
    """A RolloutLoop with in-memory I/O over ``len(lineups)`` envs.

    ``agent_roles`` defaults to ``["player"]`` for every agent (TickGame's only role).
    ``collected`` may be pre-filled (e.g. ``weights`` for the initial sync).
    """
    col = collected if collected is not None else Collected()
    agent_ids = list(model_factories)
    loop = RolloutLoop(
        worker_id=0, env_fn=env_fn, num_envs=len(lineups), chunk_length=chunk_length,
        agent_ids=agent_ids,
        agent_roles=agent_roles if agent_roles is not None else {a: ["player"] for a in agent_ids},
        model_factories=dict(model_factories), io=col.io(), lineups=list(lineups),
        weight_sync_interval=weight_sync_interval, seed=seed, **kwargs,
    )
    return loop, col


def run_steps(loop: RolloutLoop, n: int) -> None:
    for _ in range(n):
        loop.step()


def run_until_chunks(loop: RolloutLoop, col: Collected, n_chunks: int,
                     max_steps: int = 10_000) -> list[TrajectoryChunk]:
    """Step until at least ``n_chunks`` chunks were sent; return the first ``n_chunks``."""
    for _ in range(max_steps):
        if len(col.chunks) >= n_chunks:
            return col.chunks[:n_chunks]
        loop.step()
    raise AssertionError(f"only {len(col.chunks)} chunks after {max_steps} steps")


def kinds(chunk: TrajectoryChunk) -> str:
    """Slot kinds as letters, e.g. ``"AAAB"``; a terminal ACT is ``"T"``, a BOOT with reset ``"R"``."""
    out = []
    for s in range(chunk.num_slots):
        letter = KIND_LETTER[int(chunk.kind[s])]
        if letter == "A" and bool(chunk.terminal[s]):
            letter = "T"
        elif letter == "B" and bool(chunk.reset_after[s]):
            letter = "R"
        out.append(letter)
    return "".join(out)


class Slot(NamedTuple):
    """A slot's TickGame observation: env tag, episode, step, seat, is_final."""

    env: int
    ep: int
    t: int
    seat: int
    final: int


def slot_steps(chunk: TrajectoryChunk) -> list[Slot]:
    """Decode every slot's TickGame observation."""
    return [Slot(*(int(round(float(x))) for x in chunk.obs[s])) for s in range(chunk.num_slots)]


def seat_chunks(col: Collected, seat: int, env: int = 0) -> list[TrajectoryChunk]:
    """Chunks whose first slot belongs to (``env``, ``seat``)."""
    return [c for c in col.chunks if slot_steps(c)[0].seat == seat and slot_steps(c)[0].env == env]
```

- [ ] **Step 2: Write the failing tests**

`tests/contract/test_chunk_v2_rules.py` (one test per slot rule, plus reward accounting):

```python
"""Chunk v2 slot rules on the real RolloutLoop + MatchRunner (spec block 4, T3.4).

TickGame observations are ``[env, episode, step, seat, is_final]``; ``slot_steps``
decodes them, ``kinds`` spells a chunk's slots (A = ACT, T = terminal ACT, B = BOOT,
R = BOOT with reset_after, P = PAD).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from colosseum.networks.state import tree_leaves
from colosseum.sp2.core.types import SeatAssignment, WorkerCommand
from game_harness import GameFactory, kinds, lineup, make_loop, run_steps, seat_chunks, slot_steps
from game_helpers import Tick, TickGame, make_test_model

ROLE1 = TickGame([Tick(acting={0})], 1).spec.roles["player"]
ROLE2 = TickGame([Tick(acting={0})], 2).spec.roles["player"]


def _solo(script, *, chunk_length=4, core="none", **game_kwargs):
    model = make_test_model(ROLE1, core=core)
    loop, col = make_loop(GameFactory((script, 1), **game_kwargs), {"a": lambda: model},
                          [lineup("solo", "a")], chunk_length=chunk_length)
    return loop, col


def _acts(n, *, reward=lambda t: 0.0):
    """Ticks 0..n-1 make seat 0 act; the step after the ACT at step t rewards it ``reward(t)``."""
    return [Tick(acting={0})] + [Tick(acting={0}, rewards={0: reward(t)}) for t in range(n - 1)]


def test_mid_episode_boundary_writes_a_boot_and_continues_in_the_same_buffer():
    script = _acts(10, reward=lambda t: t + 1.0) + [Tick(over=True, rewards={0: 10.0})]
    loop, col = _solo(script)
    run_steps(loop, 7)
    assert [kinds(c) for c in col.chunks] == ["AAAB", "AAAB"]
    first, second = col.chunks
    assert [s.t for s in slot_steps(first)] == [0, 1, 2, 3]      # the BOOT is step 3's observation...
    assert [s.t for s in slot_steps(second)] == [3, 4, 5, 6]     # ...and the next chunk's first ACT
    assert first.reward.tolist() == [1.0, 2.0, 3.0, 0.0]
    assert first.reset_after.tolist() == [False] * 4 and first.terminal.tolist() == [False] * 4
    assert float(first.behavior_logp[3]) == 0.0 and bool(first.behavior_logp[:3].lt(0).all())


def test_episode_end_by_rules_marks_the_open_act_terminal_and_pads():
    script = _acts(3) + [Tick(over=True, rewards={0: 5.0})]
    loop, col = _solo(script)
    run_steps(loop, 3)                        # the episode ends: terminal ACT at slot 2, no chunk yet
    assert col.chunks == []
    run_steps(loop, 1)                        # next episode's first ACT: PAD + seal, ACT into slot 0
    (chunk,) = col.chunks
    assert kinds(chunk) == "AATP"
    assert chunk.reward.tolist() == [0.0, 0.0, 5.0, 0.0]
    assert slot_steps(chunk)[3] == slot_steps(chunk)[2]   # the PAD copies the terminal ACT's obs
    assert chunk.reset_after.tolist() == [False, False, True, True]
    run_steps(loop, 3)
    assert kinds(col.chunks[1]) == "AATP" and slot_steps(col.chunks[1])[0].ep == 1


def test_chunks_span_episodes_and_end_with_a_boot_mid_episode():
    loop, col = _solo(_acts(2) + [Tick(over=True)])
    run_steps(loop, 4)
    (chunk,) = col.chunks
    assert kinds(chunk) == "ATAB"
    assert [(s.ep, s.t) for s in slot_steps(chunk)] == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_truncation_with_the_open_act_at_s_minus_2_writes_the_final_boot_and_seals():
    script = _acts(3, reward=lambda t: 1.0) + [Tick(over=True, truncated=True, rewards={0: 2.0})]
    loop, col = _solo(script)
    run_steps(loop, 3)
    (chunk,) = col.chunks                                     # sealed at the episode end
    assert kinds(chunk) == "AAAR"
    assert (slot_steps(chunk)[3].t, slot_steps(chunk)[3].final) == (3, 1)   # final_obs
    assert chunk.reward.tolist() == [1.0, 1.0, 2.0, 0.0]
    assert chunk.terminal.tolist() == [False] * 4             # truncation is not termination
    assert chunk.reset_after.tolist() == [False, False, False, True]


def test_truncation_with_the_open_act_at_s_minus_3_boots_then_pads_at_the_next_act():
    loop, col = _solo(_acts(2) + [Tick(over=True, truncated=True)])
    run_steps(loop, 2)
    assert col.chunks == []
    run_steps(loop, 1)                                        # next episode's first ACT: PAD + seal
    (chunk,) = col.chunks
    assert kinds(chunk) == "AARP"
    assert slot_steps(chunk)[2] == slot_steps(chunk)[3]
    assert (slot_steps(chunk)[2].t, slot_steps(chunk)[2].final) == (2, 1)
    run_steps(loop, 3)
    assert kinds(col.chunks[1]) == "AARP" and slot_steps(col.chunks[1])[0].ep == 1


def test_elimination_at_s_minus_2_is_terminal_with_its_step_reward_then_a_pad():
    script = [
        Tick(acting={0, 1}),
        Tick(acting={0, 1}, rewards={1: 0.5}),
        Tick(acting={0, 1}),
        Tick(acting={0}, rewards={0: 1.0, 1: -1.0}, terminated={1}),
        Tick(acting={0}),
        Tick(over=True, rewards={0: 1.0}),
    ]
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((script, 2)), {"a": lambda: model}, [lineup("2p", "a", "a")])
    run_steps(loop, 5)
    assert seat_chunks(col, 1) == []                          # terminal ACT at slot 2, slot 3 free
    run_steps(loop, 1)                                        # seat 1 acts in episode 1 -> PAD + seal
    (chunk,) = seat_chunks(col, 1)
    assert kinds(chunk) == "AATP"
    assert chunk.reward.tolist() == [0.5, 0.0, -1.0, 0.0]     # the elimination reward lands before the mark
    assert chunk.terminal.tolist() == [False, False, True, False]


def test_dead_teammate_keeps_its_open_act_until_the_final_reward():
    script = [
        Tick(acting={0, 1}),
        Tick(acting={0}, rewards={1: 0.25}),                  # seat 1 stops acting but stays live
        Tick(acting={0}, rewards={0: 1.0, 1: 0.25}),
        Tick(over=True, rewards={0: 3.0, 1: 3.0}),
    ]
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((script, 2)), {"a": lambda: model}, [lineup("2p", "a", "a")],
                          chunk_length=2)
    run_steps(loop, 4)                                        # episode 0, then seat 1 acts in episode 1
    (chunk,) = seat_chunks(col, 1)
    assert kinds(chunk) == "TP"
    assert chunk.reward.tolist() == [3.5, 0.0]                # every reward up to the episode end


def test_rewards_before_the_first_act_are_carried_into_it():
    script = [
        Tick(acting={0}),
        Tick(acting={1}, rewards={1: 0.5}),                   # seat 1's reward arrives with its first turn
        Tick(acting={0}, rewards={0: 1.0}),
        Tick(over=True, rewards={0: 1.0, 1: -1.0}),
    ]
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((script, 2)), {"a": lambda: model}, [lineup("2p", "a", "a")],
                          chunk_length=2)
    run_steps(loop, 5)
    assert [kinds(c) for c in seat_chunks(col, 0)] == ["AB", "TP"]
    assert [c.reward.tolist() for c in seat_chunks(col, 0)] == [[1.0, 0.0], [1.0, 0.0]]
    (seat1,) = seat_chunks(col, 1)
    assert kinds(seat1) == "TP"
    assert seat1.reward.tolist() == [-0.5, 0.0]               # 0.5 pending + (-1.0) final
    assert loop.stats["dropped_reward_episodes"] == 0


def test_a_seat_that_never_acts_drops_its_reward_and_is_counted():
    script = [
        Tick(acting={0}),
        Tick(acting={0}, rewards={1: 2.0}),
        Tick(over=True, rewards={0: 1.0, 1: 1.0}),
    ]
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((script, 2)), {"a": lambda: model}, [lineup("2p", "a", "a")])
    run_steps(loop, 6)                                        # 3 episodes
    stats = loop.stats
    assert stats["dropped_reward_episodes"] == 3
    assert stats["recorded_transitions"] == 6
    assert sum(c.num_acts for c in col.chunks) + stats["buffered_transitions/a"] == 6
    assert all(s.seat == 0 for c in col.chunks for s in slot_steps(c))


def test_global_state_is_recorded_for_acts_and_the_final_boot():
    loop, col = _solo(_acts(2) + [Tick(over=True, truncated=True)], global_state=True)
    run_steps(loop, 3)
    (chunk,) = col.chunks
    assert kinds(chunk) == "AARP"
    assert chunk.global_state[:, 1].tolist() == [0.0, 1.0, 2.0, 2.0]   # [k, t]; the PAD copies the BOOT


@pytest.mark.parametrize("core", ["lstm", "gru"])
def test_mid_episode_boundary_with_a_parked_buffer_continues_in_the_seat_buffer(core):
    long_ep = [Tick(acting={0, 1})] * 10 + [Tick(over=True)]
    short_ep = [Tick(acting={0, 1})] * 2 + [Tick(over=True)]
    model = make_test_model(ROLE2, core=core)
    loop, col = make_loop(GameFactory((long_ep, 2), (short_ep, 2)), {"a": lambda: model},
                          [lineup("2p", "a", "a"), lineup("2p", "a", "a")])
    col.commands.append(WorkerCommand(lineups=[None, lineup("2p", "a", SeatAssignment("a", collect=False))]))
    run_steps(loop, 2)                        # env 1's episode ends: its seat 1 buffer ("AT") is parked
    assert loop.stats["parked_buffers"] == 1
    run_steps(loop, 5)                        # env 0's seats cross a chunk boundary after 3 ACTs
    assert loop.stats["parked_buffers"] == 1  # ...and the parked buffer was not taken
    first, second = seat_chunks(col, 0, env=0)[:2]
    assert kinds(first) == "AAAB" and kinds(second) == "AAAB"
    assert [(s.env, s.t) for s in slot_steps(second)] == [(0, 3), (0, 4), (0, 5), (0, 6)]
    # the continuation starts from the state after the first chunk's 3 ACTs, not from zeros
    state = model.initial_state(1)
    with torch.no_grad():
        for s in range(3):
            state = model.step(first.obs[s:s + 1], state).state
    for got, want in zip(tree_leaves(second.initial_state), tree_leaves(state)):
        assert torch.allclose(got, want, atol=1e-6)
    assert any(float(x.abs().sum()) > 0 for x in tree_leaves(second.initial_state))


def test_reward_accounting_matches_the_env_except_documented_drops():
    script = [
        Tick(acting={0, 1, 2}),
        Tick(acting={0, 1}, rewards={0: 0.5, 2: -1.0, 3: 0.75}, terminated={2}),
        Tick(acting={0}, rewards={1: 0.25, 3: 0.75}),           # seat 1 is a dead teammate from here on
        Tick(acting={0}, rewards={0: 1.0, 1: 0.5}),
        Tick(over=True, truncated=True, rewards={0: 2.0, 1: 2.0}),
    ]
    episodes, length = 5, 4
    model = make_test_model(TickGame(script, 4).spec.roles["player"])
    loop, col = make_loop(GameFactory((script, 4)), {"a": lambda: model}, [lineup("4p", "a", "a", "a", "a")],
                          chunk_length=3)
    run_steps(loop, episodes * length)                         # stops exactly at an episode end
    per_episode = sum(sum(t.rewards.values()) for t in script)
    dropped = 0.75 + 0.75                                      # seat 3 never acts
    sent = sum(float(c.reward.sum()) for c in col.chunks)
    assert sent + loop.buffered_reward("a") == pytest.approx(episodes * (per_episode - dropped))
    assert loop.stats["dropped_reward_episodes"] == episodes
    acts = loop.stats["recorded_transitions/a"]
    assert acts == episodes * (3 + 2 + 1 + 1)
    assert sum(c.num_acts for c in col.chunks) + loop.stats["buffered_transitions/a"] == acts
    assert all(np.all(c.reward[c.kind != 0].numpy() == 0.0) for c in col.chunks)
```

`tests/contract/test_rollout_loop_lineups.py`:

```python
"""RolloutLoop lineups, commands, checkpoints, versions and parking without loss (T3.4)."""

from __future__ import annotations

import logging
import random

import pytest

from colosseum.sp2.core.types import SeatAssignment, WeightPayload, WorkerCommand, state_dict_to_numpy
from game_harness import GameFactory, kinds, lineup, make_loop, run_steps, slot_steps
from game_helpers import Tick, TickGame, make_test_model

ROLE2 = TickGame([Tick(acting={0})], 2).spec.roles["player"]


def _episode(length):
    return [Tick(acting={0, 1}, rewards={0: 1.0, 1: 1.0}) for _ in range(length)] + [
        Tick(over=True, rewards={0: 1.0, 1: -1.0})]


def _random_lineup(rng):
    seats = [SeatAssignment(rng.choice("ab"), collect=rng.random() < 0.7) for _ in range(2)]
    return lineup("2p", *seats)


def test_lineup_changes_park_buffers_without_losing_or_duplicating_acts():
    rng = random.Random(0)
    models = {"a": make_test_model(ROLE2), "b": make_test_model(ROLE2)}
    loop, col = make_loop(GameFactory((_episode(3), 2), (_episode(5), 2)),
                          {"a": lambda: models["a"], "b": lambda: models["b"]},
                          [lineup("2p", "a", "a"), lineup("2p", "b", "b")], chunk_length=4)
    max_parked = 0
    for i in range(200):
        if i % 3 == 0:
            col.commands.append(WorkerCommand(lineups=[_random_lineup(rng), _random_lineup(rng)]))
        loop.step()
        max_parked = max(max_parked, loop.stats["parked_buffers"])
        assert loop.stats["parked_buffers"] <= 2 * 2 * 2           # agents x envs x seats
    assert max_parked > 0, "the scenario must park buffers"
    stats = loop.stats
    for aid in "ab":
        sent = sum(c.num_acts for c in col.chunks if c.agent_id == aid)
        assert sent + stats[f"buffered_transitions/{aid}"] == stats[f"recorded_transitions/{aid}"]
    seen, resumed = set(), 0
    for chunk in col.chunks:
        slots, letters = slot_steps(chunk), kinds(chunk)
        if len({(s.env, s.seat) for s in slots}) > 1:
            resumed += 1
        for i, (slot, letter) in enumerate(zip(slots, letters)):
            if letter in "AT":
                key = (slot.env, slot.ep, slot.t, slot.seat)
                assert key not in seen, f"ACT {key} sent twice"
                seen.add(key)
            if i + 1 == len(slots):
                continue
            nxt = slots[i + 1]
            if letter == "A":       # an open ACT continues in the same (env, episode, seat)
                assert (nxt.env, nxt.ep, nxt.seat, nxt.t) == (slot.env, slot.ep, slot.seat, slot.t + 1)
            elif letters[i + 1] in "AT":   # after an episode end the next ACT starts an episode
                assert nxt.t == 0
    assert resumed > 0, "no chunk continues a parked buffer in another (env, seat)"


def test_command_checkpoint_is_loaded_at_once_and_seated_from_the_next_episode():
    created = []

    def factory():
        created.append(make_test_model(ROLE2))
        return created[-1]

    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": factory}, [lineup("2p", "a", "a")])
    ckpt = state_dict_to_numpy(make_test_model(ROLE2).state_dict())
    col.commands.append(WorkerCommand(
        lineups=[lineup("2p", "a", SeatAssignment("a", "ckpt_v9", collect=False))],
        new_checkpoints={"a": {"ckpt_v9": ckpt}},
    ))
    loop.step()
    assert len(created) == 2 and loop.get("a", "ckpt_v9") is created[1]     # loaded at once
    assert loop.get("a", "latest") is created[0]
    loop.step()                                              # episode 0 ends -> lineup applied
    run_steps(loop, 2)                                       # episode 1
    first, second = col.results
    assert [s.network_id for s in first.seats] == ["latest", "latest"]
    assert [s.network_id for s in second.seats] == ["latest", "ckpt_v9"]
    assert loop.stats["recorded_transitions"] == 2 * 2 + 2   # seat 1 stopped collecting in episode 1


def test_a_missing_checkpoint_is_replaced_by_latest_which_collects(caplog):
    model = make_test_model(ROLE2)
    with caplog.at_level(logging.WARNING):
        loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model},
                              [lineup("2p", "a", SeatAssignment("a", "ckpt_v3", collect=False))])
    run_steps(loop, 2)
    assert col.results[0].seats[1].network_id == "latest"
    assert loop.stats["recorded_transitions"] == 4          # both seats collect
    assert any("ckpt_v3" in r.getMessage() for r in caplog.records)


def test_chunk_version_is_the_version_at_its_first_slot_also_after_parking():
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((_episode(3), 2)), {"a": lambda: model, "b": lambda: model},
                          [lineup("2p", "a", "b")], chunk_length=8)
    col.commands.append(WorkerCommand(lineups=[lineup("2p", "a", SeatAssignment("b", collect=False))]))
    run_steps(loop, 4)                                       # episode 0 ends: b's 3 ACTs are parked
    assert loop.stats["parked_buffers"] == 1 and loop.stats["buffered_transitions/b"] == 3
    col.weights["b"] = [WeightPayload.from_model("b", 5, model)]
    col.commands.append(WorkerCommand(lineups=[lineup("2p", "a", "b")]))
    loop.step()                                              # loads b v5; the lineup waits for the episode end
    run_steps(loop, 3)                                       # episode 1 ends: seat 1 resumes the parked buffer
    assert loop.stats["parked_buffers"] == 0
    run_steps(loop, 8)
    b_chunks = [c for c in col.chunks if c.agent_id == "b"]
    assert b_chunks and b_chunks[0].policy_version == 0       # parked under v0, resumed under v5
    assert [s.ep for s in slot_steps(b_chunks[0])][:5] == [0, 0, 0, 2, 2]   # episode 1 was not collected


def test_results_are_reported_per_episode_and_stats_have_the_contract_keys():
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model, "b": lambda: model},
                          [lineup("2p", "a", "b"), lineup("2p", "b", "a")])
    run_steps(loop, 4)
    assert [r.match_id for r in col.results] == ["w0_e0_ep0", "w0_e1_ep0", "w0_e0_ep1", "w0_e1_ep1"]
    assert [s.agent_id for s in col.results[1].seats] == ["b", "a"]
    assert col.results[0].outcome_kind == "wdl" and col.results[0].teams[0].rank == 1.0
    assert set(loop.stats) == {
        "chunks_sent", "env_steps", "episodes", "parked_buffers", "dropped_reward_episodes",
        "recorded_transitions", "recorded_transitions/a", "recorded_transitions/b",
        "buffered_transitions/a", "buffered_transitions/b",
    }
    assert loop.stats["env_steps"] == 8 and loop.stats["episodes"] == 4
    assert col.env_steps == [2] * 4


@pytest.mark.parametrize("bad", ["unknown agent", "wrong layout"])
def test_bad_initial_lineups_fail_fast(bad):
    model = make_test_model(ROLE2)
    lu = lineup("2p", "zzz", "a") if bad == "unknown agent" else lineup("3p", "a", "a", "a")
    with pytest.raises(ValueError):
        make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model}, [lu])
```

`tests/unit/test_rollout_worker_v2.py`:

```python
"""SP2 rollout worker process: queues, numpy payloads, budget, commands (T3.4)."""

from __future__ import annotations

import queue
import threading
import time

import pytest
import torch

from colosseum.core.ipc import SharedCounter
from colosseum.sp2.core.types import Lineup, SeatAssignment, TrajectoryChunk, WorkerCommand
from colosseum.sp2.worker.rollout_worker import _drain_commands, rollout_worker_process
from game_helpers import NumpyOnlyQueue, Tick, TickGame, make_test_model

SCRIPT = [Tick(acting={0}), Tick(acting={0}, rewards={0: 1.0}), Tick(over=True, rewards={0: 1.0})]
ROLE = TickGame(SCRIPT, 1).spec.roles["player"]


@pytest.fixture
def restore_torch_threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def _env_fn():
    return TickGame(SCRIPT, 1)


def _model_factory():
    return make_test_model(ROLE)


def test_worker_runs_to_its_step_limit_and_sends_numpy_payloads(restore_torch_threads):
    traj, weights, results, stats = NumpyOnlyQueue(), NumpyOnlyQueue(), NumpyOnlyQueue(), NumpyOnlyQueue()
    counter = SharedCounter()
    rollout_worker_process(
        worker_id=3, env_fn=_env_fn, num_envs=2, chunk_length=3, agent_ids=["a"],
        agent_roles={"a": ["player"]}, model_factories={"a": _model_factory},
        trajectory_queues={"a": traj}, weight_queues={"a": weights}, stop_event=threading.Event(),
        max_env_steps=20, env_step_counter=counter, lineups=[Lineup("solo", [SeatAssignment("a")])] * 2,
        results_queue=results, stats_queue=stats, stats_interval_sec=0.0, seed=1,
    )
    assert counter.value == 20
    payloads = [traj.get_nowait() for _ in range(traj.qsize())]
    assert payloads and all(isinstance(p, dict) for p in payloads)
    chunk = TrajectoryChunk.from_payload(payloads[0])
    assert chunk.agent_id == "a" and chunk.num_slots == 3
    match_ids = [results.get_nowait().match_id for _ in range(results.qsize())]
    assert match_ids[:2] == ["w3_e0_ep0", "w3_e1_ep0"]
    worker_stats = stats.get_nowait()
    assert worker_stats["kind"] == "worker_stats" and worker_stats["worker_id"] == 3


def test_worker_stops_on_the_stop_event(restore_torch_threads):
    stop = threading.Event()
    traj = NumpyOnlyQueue(maxsize=1)               # fills up: send_chunk must still see the stop
    thread = threading.Thread(target=rollout_worker_process, kwargs=dict(
        worker_id=0, env_fn=_env_fn, num_envs=1, chunk_length=2, agent_ids=["a"],
        agent_roles={"a": ["player"]}, model_factories={"a": _model_factory},
        trajectory_queues={"a": traj}, weight_queues={"a": NumpyOnlyQueue()}, stop_event=stop,
        lineups=[Lineup("solo", [SeatAssignment("a")])],
    ))
    thread.start()
    deadline = time.monotonic() + 30
    while traj.qsize() < 1 and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert traj.qsize() == 1, "the worker never sent a chunk"
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_drain_commands_merges_checkpoints_and_lineups_per_env():
    q = queue.Queue()
    a = Lineup("solo", [SeatAssignment("a")])
    b = Lineup("solo", [SeatAssignment("b")])
    q.put(WorkerCommand(lineups=[a, a, None], new_checkpoints={"a": {"ckpt_v1": {}}}))
    q.put(WorkerCommand(lineups=[None, b], new_checkpoints={"a": {"ckpt_v2": {}}, "b": {"ckpt_v3": {}}}))
    cmd = _drain_commands(q)
    assert cmd.lineups == [a, b, None]
    assert cmd.new_checkpoints == {"a": {"ckpt_v1": {}, "ckpt_v2": {}}, "b": {"ckpt_v3": {}}}
    assert _drain_commands(q) is None
```

- [ ] **Step 3: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/contract/test_chunk_v2_rules.py tests/contract/test_rollout_loop_lineups.py tests/unit/test_rollout_worker_v2.py -q`
Expected: collection errors `ModuleNotFoundError: No module named 'colosseum.sp2.worker.rollout_loop'` (and `...rollout_worker`).

- [ ] **Step 4: Implement `RolloutLoop`**

`src/colosseum/sp2/worker/rollout_loop.py`:

```python
"""In-process rollout loop: a MatchRunner plus agent-owned chunk v2 buffers (spec blocks 4-5).

All I/O goes through :class:`LoopIO` callbacks, so the loop runs unchanged in a worker
process (``rollout_worker_process``) and in-process in tests.

``RolloutLoop`` is the :class:`MatchRunner`'s model pool (each agent's latest model plus
frozen checkpoints) and its observer. As observer it writes the slots of every collecting
seat (slot rules of spec block 4):

- **The seat acts.** With an open ACT and exactly one free slot: a BOOT with the current
  observation and ``global_state``, the chunk is sealed, and the new ACT goes into the same
  buffer after the reset (never into a parked buffer). Without an open ACT and exactly one
  free slot: a PAD, seal, then the ACT. Otherwise the ACT goes into the next slot (the
  previous transition closes implicitly). The first ACT carries the seat's pending reward.
- **Rewards** go to the seat's open ACT, or to its pending reward before its first ACT of
  the episode.
- **Elimination, or an episode that ends by the rules:** the open ACT becomes terminal.
- **Truncation:** a BOOT with ``final_obs`` / final ``global_state`` and ``reset_after``
  follows the open ACT.
- A chunk is sealed as soon as its last slot is taken (always a BOOT or a PAD).
- A seat that never acted in an episode drops its pending reward at the episode end
  (counted in ``stats["dropped_reward_episodes"]``).
- Buffers are owned by agents and parked when a lineup change at an episode end stops a
  seat from collecting for its agent; a seat that starts collecting takes a parked buffer
  of that agent first (``BufferPool``).
- A chunk's ``policy_version`` is the agent's version at its first slot.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from colosseum.sp2.core.roles import agent_role_spec
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.types import (
    LATEST_NETWORK_ID,
    Lineup,
    MatchResult,
    TrajectoryChunk,
    WeightPayload,
    WorkerCommand,
    state_dict_from_numpy,
)
from colosseum.sp2.envs.game import MultiAgentEnv
from colosseum.sp2.envs.vector import SubprocessVectorEnv, VectorEnv
from colosseum.sp2.networks.model import PolicyModel
from colosseum.sp2.worker.buffers import BufferPool, BufferSpec, RolloutBuffer
from colosseum.sp2.worker.match_runner import ActRecord, EpisodeEnd, MatchRunner

logger = logging.getLogger(__name__)

__all__ = ["LATEST_NETWORK_ID", "LoopIO", "RolloutLoop"]


@dataclass
class LoopIO:
    """All I/O of the rollout loop goes through these callbacks."""

    send_chunk: Callable[[TrajectoryChunk], None]
    poll_weights: Callable[[str], WeightPayload | None]
    report_result: Callable[[MatchResult], None] | None = None
    poll_command: Callable[[], WorkerCommand | None] | None = None
    add_env_steps: Callable[[int], None] | None = None  # global env-step budget counter


@dataclass
class _SeatTrack:
    """Rollout state of one collecting (env, seat)."""

    agent_id: str
    buffer: RolloutBuffer
    pending_reward: float = 0.0


class RolloutLoop:
    """One worker's rollout loop over ``num_envs`` envs (module docstring)."""

    def __init__(
        self,
        *,
        worker_id: int,
        env_fn: Callable[[], MultiAgentEnv],
        num_envs: int,
        chunk_length: int,
        agent_ids: list[str],
        agent_roles: Mapping[str, Sequence[str]],
        model_factories: Mapping[str, Callable[[], PolicyModel]],
        io: LoopIO,
        lineups: Sequence[Lineup],
        weight_sync_interval: float = 5.0,
        checkpoint_state_dicts_by_agent: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]] | None = None,
        seed: int | None = None,
        vec_env_kind: Literal["sync", "subprocess"] = "sync",
        subproc_workers: int | None = None,
        max_idle_steps: int = 1000,
    ) -> None:
        self.worker_id = worker_id
        self._io = io
        self._agent_ids = list(agent_ids)
        self._model_factories = dict(model_factories)
        self._weight_sync_interval = weight_sync_interval

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if vec_env_kind == "subprocess":
            vec_env = SubprocessVectorEnv(env_fn, num_envs, num_workers=subproc_workers)
        elif vec_env_kind == "sync":
            vec_env = VectorEnv(env_fn, num_envs)
        else:
            raise ValueError(f"unknown vec_env_kind {vec_env_kind!r}")
        spec = vec_env.spec

        specs: dict[str, BufferSpec] = {}
        for aid in self._agent_ids:
            role = agent_role_spec(spec, list(agent_roles[aid]))
            specs[aid] = BufferSpec(
                obs=ObsSpec.from_space(role.observation_space),
                action=ActionSpec.from_space(role.action_space),
                global_state=(
                    None if role.global_state_space is None else ObsSpec.from_space(role.global_state_space)
                ),
            )
        self._pool = BufferPool(chunk_length, specs)

        # Model pool: agent -> network id -> model ("latest" + frozen checkpoints).
        self._models: dict[str, dict[str, PolicyModel]] = {}
        self._policy_versions: dict[str, int] = {aid: 0 for aid in self._agent_ids}
        for aid in self._agent_ids:
            latest = self._model_factories[aid]()
            latest.eval()
            self._models[aid] = {LATEST_NETWORK_ID: latest}
        for aid, by_id in (checkpoint_state_dicts_by_agent or {}).items():
            for ckpt_id, sd in by_id.items():
                self._add_checkpoint(aid, ckpt_id, sd)
        self.sync_weights()   # initial weights and policy versions

        self._tracks: list[dict[int, _SeatTrack]] = [{} for _ in range(num_envs)]
        self._env_steps = 0
        self._episodes = 0
        self._chunks_sent = 0
        self._dropped_reward_episodes = 0
        self._recorded: dict[str, int] = defaultdict(int)
        self._runner = MatchRunner(
            vec_env=vec_env, lineups=lineups, models=self, observer=self, seed=seed,
            max_idle_steps=max_idle_steps, context=f"worker {worker_id}, ", match_id_prefix=f"w{worker_id}_e",
        )
        for e in range(num_envs):
            for seat, assignment in enumerate(self._runner.lineup(e).seats):
                if assignment.collect:
                    self._start_collecting(e, seat, assignment.agent_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> int:
        """One step of every env. Returns the env steps taken."""
        self._poll_command()
        n = self._runner.step()
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
        """Step until ``should_stop()`` or until ``max_env_steps`` env steps (0 = no limit)."""
        while not should_stop():
            if max_env_steps > 0 and self._env_steps >= max_env_steps:
                break
            self.step()

    def close(self) -> None:
        self._runner.close()

    @property
    def stats(self) -> dict[str, int]:
        out = {
            "chunks_sent": self._chunks_sent,
            "env_steps": self._env_steps,
            "episodes": self._episodes,
            "parked_buffers": self._pool.parked_count(),
            "dropped_reward_episodes": self._dropped_reward_episodes,
            "recorded_transitions": sum(self._recorded.values()),
        }
        for aid in self._agent_ids:
            out[f"recorded_transitions/{aid}"] = self._recorded[aid]
            out[f"buffered_transitions/{aid}"] = self._buffered_acts(aid)
        return out

    def buffered_reward(self, agent_id: str) -> float:
        """Rewards of ``agent_id`` recorded in buffers but not sent yet (seat + parked buffers)."""
        total = self._pool.parked_reward(agent_id)
        for tracks in self._tracks:
            for track in tracks.values():
                if track.agent_id == agent_id:
                    total += track.buffer.reward_sum
        return total

    # ------------------------------------------------------------------
    # ModelPool
    # ------------------------------------------------------------------

    def get(self, agent_id: str, network_id: str) -> PolicyModel | None:
        return self._models.get(agent_id, {}).get(network_id)

    def _add_checkpoint(self, aid: str, ckpt_id: str, sd: Mapping[str, np.ndarray]) -> None:
        if aid not in self._models or ckpt_id in self._models[aid]:
            return
        model = self._model_factories[aid]()
        model.load_state_dict(state_dict_from_numpy(sd))
        model.eval()
        self._models[aid][ckpt_id] = model
        logger.info(f"Worker {self.worker_id}: agent {aid}: loaded checkpoint {ckpt_id}")

    def _poll_command(self) -> None:
        """Load a command's new checkpoints now; stage its lineups for the envs' episode ends."""
        if self._io.poll_command is None:
            return
        cmd = self._io.poll_command()
        if cmd is None:
            return
        for aid, ckpts in cmd.new_checkpoints.items():
            for ckpt_id, sd in ckpts.items():
                self._add_checkpoint(aid, ckpt_id, sd)
        for e, lineup in enumerate(cmd.lineups):
            if lineup is not None and e < self._runner.num_envs:
                self._runner.set_next_lineup(e, lineup)

    # ------------------------------------------------------------------
    # MatchObserver
    # ------------------------------------------------------------------

    def on_act(self, env: int, seat: int, record: ActRecord) -> None:
        track = self._tracks[env].get(seat)
        if track is None:
            return
        buf = track.buffer
        if buf.free_slots == 1:
            if buf.has_open:
                buf.write_boot(record.obs, record.global_state, reset_after=False)
            else:
                buf.write_pad()
            self._seal(track.agent_id, buf)
        if buf.slots_used == 0:
            buf.begin(record.pre_state, self._policy_versions[track.agent_id])
        buf.write_act(record.obs, record.global_state, record.mask, record.action, record.log_prob,
                      record.unit_log_probs, track.pending_reward)
        track.pending_reward = 0.0
        self._recorded[track.agent_id] += 1

    def on_rewards(self, env: int, rewards: dict[int, float]) -> None:
        for seat, r in rewards.items():
            track = self._tracks[env].get(seat)
            if track is None:
                continue
            if track.buffer.has_open:
                track.buffer.add_reward(r)
            else:
                track.pending_reward += r

    def on_terminated(self, env: int, seats: list[int]) -> None:
        for seat in seats:
            track = self._tracks[env].get(seat)
            if track is not None and track.buffer.has_open:
                track.buffer.mark_terminal()

    def on_episode_end(self, env: int, end: EpisodeEnd) -> None:
        for seat, track in self._tracks[env].items():
            buf = track.buffer
            if buf.has_open:
                if end.truncated:
                    gs = None if end.final_global_state is None else end.final_global_state.get(seat)
                    buf.write_boot(end.final_obs[seat], gs, reset_after=True)
                    if buf.is_full:
                        self._seal(track.agent_id, buf)
                else:
                    buf.mark_terminal()
            if track.pending_reward != 0.0:
                self._dropped_reward_episodes += 1
                logger.debug(
                    f"Worker {self.worker_id}: env {env}, seat {seat}: dropping reward "
                    f"{track.pending_reward} of an episode the seat never acted in"
                )
            track.pending_reward = 0.0
        self._episodes += 1
        if self._io.report_result is not None:
            self._io.report_result(end.result)

    def on_lineup_applied(self, env: int, old: Lineup, new: Lineup) -> None:
        tracks = self._tracks[env]
        for seat in range(max(len(old.seats), len(new.seats))):
            assignment = new.seats[seat] if seat < len(new.seats) else None
            collector = assignment.agent_id if assignment is not None and assignment.collect else None
            track = tracks.get(seat)
            if track is not None and track.agent_id != collector:
                self._pool.park(track.agent_id, track.buffer)
                del tracks[seat]
            if collector is not None and seat not in tracks:
                self._start_collecting(env, seat, collector)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _start_collecting(self, env: int, seat: int, agent_id: str) -> None:
        if agent_id not in self._models:
            raise ValueError(f"worker {self.worker_id}: env {env}, seat {seat}: unknown agent {agent_id!r}")
        self._tracks[env][seat] = _SeatTrack(agent_id=agent_id, buffer=self._pool.acquire(agent_id))

    def _buffered_acts(self, aid: str) -> int:
        n = self._pool.parked_acts(aid)
        for tracks in self._tracks:
            for track in tracks.values():
                if track.agent_id == aid:
                    n += track.buffer.num_acts
        return n

    def _seal(self, aid: str, buf: RolloutBuffer) -> None:
        chunk = buf.build_chunk(aid)
        buf.reset()
        self._io.send_chunk(chunk)
        self._chunks_sent += 1
```

- [ ] **Step 5: Implement the worker process**

`src/colosseum/sp2/worker/rollout_worker.py` (copy of SP1's `worker/rollout_worker.py` with the changes described above):

```python
"""Rollout worker process: a thin process wrapper around :class:`RolloutLoop`.

The wrapper limits torch threads, turns queues into :class:`LoopIO` callbacks and runs the
loop until ``stop_event`` is set (or ``max_env_steps`` env steps were taken), adding its env
steps to the global budget counter. The loop itself (envs, inference, chunking) lives in
``colosseum.sp2.worker.rollout_loop`` and is testable in-process.
"""

from __future__ import annotations

import logging
import queue
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from colosseum.core.ipc import BatchedCounter, SharedCounter, drain_latest
from colosseum.core.threads import configure_torch_threads
from colosseum.metrics.aggregator import WORKER_STATS_INTERVAL_SEC
from colosseum.sp2.core.types import Lineup, MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand
from colosseum.sp2.envs.game import MultiAgentEnv
from colosseum.sp2.networks.model import PolicyModel
from colosseum.sp2.worker.rollout_loop import LATEST_NETWORK_ID, LoopIO, RolloutLoop

__all__ = ["LATEST_NETWORK_ID", "report_worker_stats", "rollout_worker_process"]

logger = logging.getLogger(__name__)


def _drain_commands(command_queue: Any) -> WorkerCommand | None:
    """Newest WorkerCommand, merged over all drained ones.

    Checkpoints are sent to a worker only once (as deltas), so ``new_checkpoints`` of all
    drained commands are merged. Lineups are merged per env: a later command's lineup wins,
    and a ``None`` entry keeps the earlier command's lineup for that env.
    """
    latest: WorkerCommand | None = None
    merged: dict[str, dict[str, Any]] = {}
    lineups: list[Lineup | None] = []
    while True:
        try:
            cmd = command_queue.get_nowait()
        except queue.Empty:
            break
        for aid, ckpts in cmd.new_checkpoints.items():
            merged.setdefault(aid, {}).update(ckpts)
        if len(cmd.lineups) > len(lineups):
            lineups.extend([None] * (len(cmd.lineups) - len(lineups)))
        for e, lineup in enumerate(cmd.lineups):
            if lineup is not None:
                lineups[e] = lineup
        latest = cmd
    if latest is None:
        return None
    return WorkerCommand(lineups=lineups, new_checkpoints=merged)


def report_worker_stats(q, worker_id: int, stats: dict) -> None:
    """Best-effort worker stats for the main process (system metrics). Never blocks."""
    try:
        q.put_nowait({"kind": "worker_stats", "worker_id": int(worker_id),
                      **{k: int(v) for k, v in stats.items()}})
    except queue.Full:
        pass


def rollout_worker_process(
    *,
    worker_id: int,
    env_fn: Callable[[], MultiAgentEnv],
    num_envs: int,
    chunk_length: int,
    agent_ids: list[str],
    agent_roles: Mapping[str, Sequence[str]],
    model_factories: Mapping[str, Callable[[], PolicyModel]],
    trajectory_queues: Mapping[str, Any],
    weight_queues: Mapping[str, Any],
    stop_event: Any,
    weight_sync_interval: float = 5.0,
    torch_threads: int = 1,
    max_env_steps: int = 0,
    env_step_counter: SharedCounter | None = None,
    checkpoint_state_dicts_by_agent: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]] | None = None,
    lineups: Sequence[Lineup],
    results_queue: Any = None,
    command_queue: Any = None,
    seed: int | None = None,
    vec_env_kind: Literal["sync", "subprocess"] = "sync",
    subproc_workers: int | None = None,
    stats_queue: Any = None,
    stats_interval_sec: float = WORKER_STATS_INTERVAL_SEC,
    max_idle_steps: int = 1000,
) -> None:
    """Worker process entry point (see module docstring).

    Chunk payloads (``TrajectoryChunk.to_payload()``) go to ``trajectory_queues[agent_id]``
    (blocking put that gives up when ``stop_event`` is set); weights are drained newest-wins
    from ``weight_queues[agent_id]``; results are put non-blocking on ``results_queue``;
    commands are drained (merged) from ``command_queue``. ``max_env_steps`` limits this
    worker's env steps (0 = until stopped). Env steps are added to ``env_step_counter``
    about every 0.5 s and once more on exit.
    """
    configure_torch_threads(torch_threads)

    def send_chunk(chunk: TrajectoryChunk) -> None:
        payload = chunk.to_payload()  # numpy only across processes
        q = trajectory_queues[chunk.agent_id]
        while not stop_event.is_set():  # waits in short slices so a stop is seen quickly
            try:
                q.put(payload, timeout=0.5)
                return
            except queue.Full:
                continue

    def poll_weights(agent_id: str) -> WeightPayload | None:
        return drain_latest(weight_queues[agent_id])

    def report_result(result: MatchResult) -> None:
        try:
            results_queue.put_nowait(result)
        except queue.Full:
            logger.debug(f"Worker {worker_id}: results queue full, dropping {result.match_id}")

    def poll_command() -> WorkerCommand | None:
        return _drain_commands(command_queue)

    counter = BatchedCounter(env_step_counter) if env_step_counter is not None else None
    io = LoopIO(
        send_chunk=send_chunk,
        poll_weights=poll_weights,
        report_result=report_result if results_queue is not None else None,
        poll_command=poll_command if command_queue is not None else None,
        add_env_steps=counter.add if counter is not None else None,
    )
    logger.info(
        f"Worker {worker_id}: starting with {num_envs} envs ({vec_env_kind}), "
        f"chunk_length={chunk_length}, agents={agent_ids}, torch_threads={torch_threads}"
    )
    loop = RolloutLoop(
        worker_id=worker_id, env_fn=env_fn, num_envs=num_envs, chunk_length=chunk_length,
        agent_ids=agent_ids, agent_roles=agent_roles, model_factories=model_factories, io=io,
        lineups=lineups, weight_sync_interval=weight_sync_interval,
        checkpoint_state_dicts_by_agent=checkpoint_state_dicts_by_agent, seed=seed,
        vec_env_kind=vec_env_kind, subproc_workers=subproc_workers, max_idle_steps=max_idle_steps,
    )
    last_stats = [time.monotonic()]

    def should_stop() -> bool:
        now = time.monotonic()
        if stats_queue is not None and now - last_stats[0] >= stats_interval_sec:
            last_stats[0] = now
            report_worker_stats(stats_queue, worker_id, loop.stats)
        return stop_event.is_set()

    try:
        loop.run(should_stop=should_stop, max_env_steps=max_env_steps)
    finally:
        if counter is not None:
            counter.flush()
        try:
            loop.close()
        finally:
            # Detach feeder threads of queues this worker produced to, so undrained
            # items (e.g. chunks a stopped learner never consumed) cannot block exit.
            for q in [*trajectory_queues.values(), results_queue, stats_queue]:
                if q is not None and hasattr(q, "cancel_join_thread"):
                    q.cancel_join_thread()
    logger.info(f"Worker {worker_id}: finished. {loop.stats}")
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/contract/test_chunk_v2_rules.py tests/contract/test_rollout_loop_lineups.py tests/unit/test_rollout_worker_v2.py -q`
Expected: `23 passed`.

- [ ] **Step 7: Full fast suite and lint** (as in T3.1, Step 5).

- [ ] **Step 8: Commit**

```bash
git add src/colosseum/sp2/worker/rollout_loop.py src/colosseum/sp2/worker/rollout_worker.py \
    tests/contract/game_harness.py tests/contract/test_chunk_v2_rules.py tests/contract/test_rollout_loop_lineups.py \
    tests/unit/test_rollout_worker_v2.py tests/game_helpers.py pyproject.toml
git commit -m "feat(sp2): RolloutLoop over MatchRunner with chunk v2 slot rules, worker process, contract tests"
git push origin sp2-game-model
```

---

### Task T4.1: V-trace(λ) over act / boot / pad slots

Spec block 6 ("V-trace по слотам"). Generalizes SP1's `algorithms/vtrace.py` (which stays untouched until T7.3):

- values come from the learner's network on every slot;
- on non-ACT slots ρ = c = 0, reward 0, `vs = V`, so a trace stops at a BOOT by itself;
- the next value of a non-terminal ACT at slot t is `V` of slot t+1 (an ACT of the same episode or its BOOT); a terminal ACT bootstraps 0; a PAD follows only an episode end;
- every selection is a `torch.where`, never a mask product: NaN in the value of an unused slot (e.g. a PAD) never reaches a target;
- `td` is the policy-gradient advantage `r_t + γ·vs_{t+1} − V_t` (no ρ factor; APPO decides whether to multiply by `clipped_rho`).

The test checks against a plain-Python slot-by-slot reference for random valid slot patterns (four (λ, ρ̄, c̄) settings), GAE(λ) equality at zero lag over ACT slots, a hand-computed BOOT/terminal case, and NaN isolation.

**Files:**
- Create: `src/colosseum/sp2/algorithms/__init__.py` (empty), `src/colosseum/sp2/algorithms/vtrace.py`
- Test: `tests/unit/test_vtrace_slots.py`

**Interfaces:**
- Consumes: `SLOT_ACT`, `SLOT_BOOT`, `SLOT_PAD` (T3.1; test only).
- Produces (contract): `VTraceOut(vs, td, clipped_rho)`, `compute_vtrace_slots(*, log_rhos, rewards, values, is_act, terminal, gamma, rho_bar=1.0, c_bar=1.0, lam=1.0) -> VTraceOut`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_vtrace_slots.py`:

```python
"""V-trace(lambda) over act/boot/pad slots against a slot-by-slot reference (T4.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from colosseum.sp2.algorithms.vtrace import VTraceOut, compute_vtrace_slots
from colosseum.sp2.core.types import SLOT_ACT, SLOT_BOOT, SLOT_PAD

LETTER_KIND = {"A": SLOT_ACT, "T": SLOT_ACT, "B": SLOT_BOOT, "R": SLOT_BOOT, "P": SLOT_PAD}


def random_pattern(S: int, rng: np.random.Generator) -> str:
    """A valid slot pattern: A open ACT, T terminal ACT, B boot (chunk end), R boot with reset, P pad."""
    out: list[str] = []
    while len(out) < S:
        if S - len(out) == 1:
            out.append("P")                       # only reached at an episode boundary
            break
        n, end = int(rng.integers(1, 5)), str(rng.choice(["T", "R"]))
        for _ in range(n):
            if S - len(out) == 1:
                out.append("B")                   # the open ACT continues in the next chunk
                break
            out.append("A")
        else:
            if end == "T":
                out[-1] = "T"
            else:
                out.append("R")
    return "".join(out)


def tensors(patterns: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    kind = torch.tensor([[LETTER_KIND[c] for c in p] for p in patterns], dtype=torch.int8).T
    terminal = torch.tensor([[c == "T" for c in p] for p in patterns]).T
    return kind, terminal


def reference(log_rho, r, v, pattern, gamma, rho_bar, c_bar, lam):
    """Plain-python V-trace over one column of slots."""
    S = len(pattern)
    vs, td, crho = list(v), [0.0] * S, [0.0] * S
    for t in reversed(range(S)):
        if pattern[t] not in "AT":
            continue
        rho = math.exp(min(20.0, max(-20.0, log_rho[t])))
        crho[t] = min(rho_bar, rho)
        c = lam * min(c_bar, rho)
        if pattern[t] == "T":
            v_next, acc_next = 0.0, 0.0
        else:
            v_next = v[t + 1]
            acc_next = vs[t + 1] - v[t + 1] if pattern[t + 1] in "AT" else 0.0
        vs[t] = v[t] + crho[t] * (r[t] + gamma * v_next - v[t]) + gamma * c * acc_next
    for t in range(S):
        if pattern[t] in "AT":
            nxt = 0.0 if pattern[t] == "T" else vs[t + 1]
            td[t] = r[t] + gamma * nxt - v[t]
    return vs, td, crho


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("lam,rho_bar,c_bar", [(1.0, 1.0, 1.0), (0.9, 1.0, 1.0), (0.95, 0.8, 1.2), (0.5, 2.0, 0.7)])
def test_matches_the_slot_reference(seed, lam, rho_bar, c_bar):
    rng = np.random.default_rng(seed)
    S, B, gamma = 12, 5, 0.97
    patterns = [random_pattern(S, rng) for _ in range(B)]
    kind, terminal = tensors(patterns)
    log_rhos = torch.tensor(rng.normal(0, 0.7, (S, B)), dtype=torch.float32)
    rewards = torch.tensor(rng.normal(0, 1, (S, B)), dtype=torch.float32)
    values = torch.tensor(rng.normal(0, 1, (S, B)), dtype=torch.float32)
    out = compute_vtrace_slots(log_rhos=log_rhos, rewards=rewards, values=values, is_act=kind == SLOT_ACT,
                               terminal=terminal, gamma=gamma, rho_bar=rho_bar, c_bar=c_bar, lam=lam)
    assert isinstance(out, VTraceOut)
    for b, pattern in enumerate(patterns):
        vs, td, crho = reference(log_rhos[:, b].tolist(), rewards[:, b].tolist(), values[:, b].tolist(),
                                 pattern, gamma, rho_bar, c_bar, lam)
        assert out.vs[:, b].tolist() == pytest.approx(vs, abs=1e-5), pattern
        assert out.td[:, b].tolist() == pytest.approx(td, abs=1e-5), pattern
        assert out.clipped_rho[:, b].tolist() == pytest.approx(crho, abs=1e-6), pattern


def _gae(pattern, r, v, gamma, lam):
    """GAE(lambda) over the ACT slots of one column; a BOOT gives the bootstrap value."""
    adv = [0.0] * len(pattern)
    running = 0.0
    for t in reversed(range(len(pattern))):
        if pattern[t] not in "AT":
            running = 0.0
            continue
        if pattern[t] == "T":
            delta, running = r[t] - v[t], 0.0
        else:
            delta = r[t] + gamma * v[t + 1] - v[t]
        running = delta + gamma * lam * running
        adv[t] = running
    return adv


@pytest.mark.parametrize("lam", [1.0, 0.95, 0.5, 0.0])
def test_zero_lag_equals_gae_over_act_slots(lam):
    rng = np.random.default_rng(7)
    S, B, gamma = 16, 6, 0.9
    patterns = [random_pattern(S, rng) for _ in range(B)]
    kind, terminal = tensors(patterns)
    rewards = torch.tensor(rng.normal(0, 1, (S, B)), dtype=torch.float32)
    values = torch.tensor(rng.normal(0, 1, (S, B)), dtype=torch.float32)
    out = compute_vtrace_slots(log_rhos=torch.zeros(S, B), rewards=rewards, values=values,
                               is_act=kind == SLOT_ACT, terminal=terminal, gamma=gamma, lam=lam)
    for b, pattern in enumerate(patterns):
        gae = _gae(pattern, rewards[:, b].tolist(), values[:, b].tolist(), gamma, lam)
        is_act = [c in "AT" for c in pattern]
        got = (out.vs[:, b] - values[:, b]).tolist()
        assert [g for g, a in zip(got, is_act) if a] == pytest.approx([g for g, a in zip(gae, is_act) if a],
                                                                       abs=1e-5)


def test_boot_cuts_the_trace_and_terminal_bootstraps_zero():
    # column 0: A A B (chunk end); column 1: A T P
    kind, terminal = tensors(["AAB", "ATP"])
    values = torch.tensor([[1.0, 1.0], [2.0, 2.0], [5.0, 100.0]])
    rewards = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    out = compute_vtrace_slots(log_rhos=torch.zeros(3, 2), rewards=rewards, values=values,
                               is_act=kind == SLOT_ACT, terminal=terminal, gamma=0.5)
    # column 0: vs_1 = 1 + 0.5 * 5 = 3.5; vs_0 = 1 + 0.5 * 3.5 = 2.75; the BOOT keeps its value
    assert out.vs[:, 0].tolist() == pytest.approx([2.75, 3.5, 5.0])
    # column 1: the terminal ACT ignores the PAD's value: vs_1 = 1, vs_0 = 1 + 0.5 * 1
    assert out.vs[:2, 1].tolist() == pytest.approx([1.5, 1.0])
    assert out.td[:, 1].tolist() == pytest.approx([1.0 + 0.5 * 1.0 - 1.0, 1.0 - 2.0, 0.0])
    assert out.clipped_rho[:, 0].tolist() == [1.0, 1.0, 0.0]


def test_nan_in_unused_slots_never_reaches_the_targets():
    rng = np.random.default_rng(3)
    S, B = 10, 8
    patterns = [random_pattern(S, rng) for _ in range(B - 2)] + ["AAATAAAATP", "AARAAAAARP"]
    kind, terminal = tensors(patterns)
    rewards = torch.tensor(rng.normal(0, 1, (S, B)), dtype=torch.float32)
    values = torch.tensor(rng.normal(0, 1, (S, B)), dtype=torch.float32)
    pad = kind == SLOT_PAD
    poisoned_values = torch.where(pad, torch.full_like(values, float("nan")), values)
    poisoned_rewards = torch.where(kind != SLOT_ACT, torch.full_like(rewards, float("nan")), rewards)
    poisoned_rhos = torch.where(kind != SLOT_ACT, torch.full_like(values, float("nan")), torch.zeros_like(values))
    kwargs = dict(is_act=kind == SLOT_ACT, terminal=terminal, gamma=0.99, lam=0.9)
    clean = compute_vtrace_slots(log_rhos=torch.zeros(S, B), rewards=rewards, values=values, **kwargs)
    dirty = compute_vtrace_slots(log_rhos=poisoned_rhos, rewards=poisoned_rewards, values=poisoned_values,
                                 **kwargs)
    act = kind == SLOT_ACT
    assert torch.isfinite(dirty.vs[act]).all() and torch.isfinite(dirty.td).all()
    assert torch.equal(dirty.vs[act], clean.vs[act])
    assert torch.equal(dirty.td, clean.td) and torch.equal(dirty.clipped_rho, clean.clipped_rho)


def test_random_patterns_respect_the_slot_invariants():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = random_pattern(int(rng.integers(2, 12)), rng)
        assert p[-1] in "BRP"                               # an ACT never takes the last slot
        for i, c in enumerate(p[1:], start=1):
            if c == "B":
                assert i == len(p) - 1 and p[i - 1] == "A"
            if c == "R":
                assert p[i - 1] == "A"
            if c == "P":
                assert i == len(p) - 1 and p[i - 1] in "TR"
```

- [ ] **Step 2: Run it and see it fail**

Run: `.venv/bin/python -m pytest tests/unit/test_vtrace_slots.py -q`
Expected: `ModuleNotFoundError: No module named 'colosseum.sp2.algorithms'`.

- [ ] **Step 3: Implement**

Create the empty `src/colosseum/sp2/algorithms/__init__.py`, then `src/colosseum/sp2/algorithms/vtrace.py`:

```python
"""V-trace(lambda) over chunk v2 slots (spec block 6).

Generalizes SP1's ``compute_vtrace`` (IMPALA, Espeholt et al. 2018, with the lambda variant
of Remark 2) to chunks of ACT / BOOT / PAD slots:

- values ``V`` come from the learner's network on every slot (``PolicyModel.unroll``);
- non-ACT slots have rho = c = 0, reward 0 and ``vs = V``, so a trace stops at a BOOT;
- the next value of a non-terminal ACT at slot t is ``V`` of slot t+1 (an ACT of the same
  episode or its BOOT); a terminal ACT bootstraps 0; a PAD follows only episode ends;
- everything is selected with ``torch.where`` (never by multiplying with a mask), so NaN in
  the values of slots that are not used (e.g. a PAD) never reaches the targets.

Math, on ACT slots (``cont_t`` = ACT and not terminal):
    rho_t = min(rho_bar, exp(log_rho_t)),  c_t = lam * min(c_bar, exp(log_rho_t))
    delta_t = rho_t * (r_t + gamma * cont_t * V_{t+1} - V_t)
    vs_t - V_t = delta_t + gamma * cont_t * c_t * (vs_{t+1} - V_{t+1})
    td_t = r_t + gamma * cont_t * vs_{t+1} - V_t           (the policy-gradient advantage)
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class VTraceOut(NamedTuple):
    vs: Tensor           # [S, B] value targets (== values on non-ACT slots)
    td: Tensor           # [S, B] r_t + gamma * (1 - terminal_t) * vs_{t+1} - V_t on ACT slots; 0 elsewhere
    clipped_rho: Tensor  # [S, B] min(rho_bar, rho_t) on ACT slots; 0 elsewhere


def compute_vtrace_slots(
    *,
    log_rhos: Tensor,
    rewards: Tensor,
    values: Tensor,
    is_act: Tensor,
    terminal: Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    lam: float = 1.0,
) -> VTraceOut:
    """V-trace(lambda) targets and TD advantages over ``[S, B]`` slots (module docstring).

    ``log_rhos`` is the scalar log importance ratio of each slot (from
    ``algorithm.unit_trace``; ignored where ``is_act`` is False). ``is_act`` and ``terminal``
    are bool. An ACT never sits in the last slot.
    """
    S = values.shape[0]
    zeros = torch.zeros_like(values)
    is_act = is_act.bool()
    cont = is_act & ~terminal.bool()
    log_rhos = torch.where(is_act, torch.clamp(log_rhos.to(values.dtype), -20.0, 20.0), zeros)
    rhos = torch.exp(log_rhos)
    clipped_rho = torch.where(is_act, torch.clamp(rhos, max=rho_bar), zeros)
    cs = torch.where(is_act, lam * torch.clamp(rhos, max=c_bar), zeros)
    rewards = torch.where(is_act, rewards.to(values.dtype), zeros)
    v_t = torch.where(is_act, values, zeros)
    next_values = torch.cat([values[1:], zeros[:1]], dim=0)
    v_next = torch.where(cont, next_values, zeros)
    deltas = torch.where(is_act, clipped_rho * (rewards + gamma * v_next - v_t), zeros)

    vs_minus_v = torch.zeros_like(values)
    acc = zeros[0]
    for t in reversed(range(S)):
        acc = torch.where(is_act[t], deltas[t] + gamma * cs[t] * torch.where(cont[t], acc, zeros[t]), zeros[t])
        vs_minus_v[t] = acc
    vs = torch.where(is_act, values + vs_minus_v, values)

    vs_next = torch.where(cont, torch.cat([vs[1:], zeros[:1]], dim=0), zeros)
    td = torch.where(is_act, rewards + gamma * vs_next - v_t, zeros)
    return VTraceOut(vs=vs, td=td, clipped_rho=clipped_rho)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_vtrace_slots.py -q`
Expected: `31 passed`.

- [ ] **Step 5: Full fast suite and lint** (as in T3.1, Step 5).

- [ ] **Step 6: Commit**

```bash
git add src/colosseum/sp2/algorithms/__init__.py src/colosseum/sp2/algorithms/vtrace.py tests/unit/test_vtrace_slots.py
git commit -m "feat(sp2): V-trace(lambda) over act/boot/pad slots"
git push origin sp2-game-model
```

---

### Task T4.2: APPO v2 (unit modes, reductions over ACT slots, diagnostics) and kickstart v2

Spec block 6 (as amended in spec section 8, items 8 and 9). `APPO` keeps SP1's structure (one `_prepare_batch` and one `_evaluate` path for training and `evaluate_chunks`, minibatches as slices over the chunk dimension, AMP/GradScaler, LR as a function of progress, `state_dict` keys) and changes the loss:

- **Batch.** Every chunk tree is stacked to `[S, B, ...]` (`tree_stack(axis=1)`); `None` trees stay `None`. `_select_chunks` slices every declared key (`obs, global_state, actions, action_masks, kind, reward, terminal, reset_after, behavior_logp, behavior_unit_logp`, and the state) and rejects unknown keys.
- **Evaluation.** `model.unroll(obs, initial_state, reset_after, action_masks, global_state=..., with_value=True)` over all slots; joint and per-decider log-probs, per-decider entropy and validity (`Distribution.unit_*` with the recorded actions).
- **Modes** (`resolve_modes(config, action_spec)`):

  | `K = num_deciders` | `ratio_mode` (`auto`) | `unit_trace` (`auto`) | `entropy_reduction` (`auto`) |
  |---|---|---|---|
  | 1 (no Units, or `Units(1, ...)` alone) | `joint` (always) | `joint` (always) | `sum` (always) |
  | > 1 (only possible with Units) | `per_unit` | `geo_mean` | `mean_valid` for `per_unit`, `sum` for `joint` |

  With `K == 1` every combination resolves to the joint path, so "K = 1 gives identical results in all modes" holds by construction (ruling, see "Contract notes": with one decider the per-decider and joint ratios coincide; keeping the joint path also keeps SP1's ρ-weighted advantage). Explicit values apply when `K > 1`; explicit `entropy_reduction` changes only the entropy/KL column.
- **V-trace input** (`unit_trace`): `joint` = joint log-ratio (sum over valid deciders); `geo_mean` = mean of the per-decider log-ratios over valid deciders (0 if none is valid); `none` = 0 with `rho_bar = c_bar = 1` (TD(λ)). `unit_trace` feeds only `compute_vtrace_slots`.
- **Policy loss** (`ratio_mode`): `joint` — PPO-clip on the joint ratio, advantage = `td × clipped_rho` (SP1); `per_unit` — PPO-clip per decider on the shared advantage `td` **without** ρ, step loss = mean over valid deciders, then mean over ACT slots. Advantage normalization (mean/std) over ACT slots.
- **Value loss**: mean over ACT slots of `(V − vs)²`. **Entropy**: per-decider entropy reduced per `entropy_reduction`, mean over ACT slots.
- **Everything** (losses, normalization, metrics) reduces over ACT slots via `torch.where`.
- **Diagnostics** (SP1 names kept: `approx_kl` (now the mean over valid deciders), `clip_fraction` (per decider), `rho_mean`, `rho_clip_frac`, `explained_variance`; new: `clip_fraction_joint`, `c_clip_frac`, `log_rho_abs_mean`, `log_rho_abs_p95` (per decider), `log_rho_joint_abs_mean`, `ess` (normalized ESS of the scalar ρ over ACT slots: `(Σρ)² / (N·Σρ²)`), `deciders_valid_mean`, `deciders_valid_max`, `boot_frac`, `pad_frac` (shares of all slots)). `rho_*` and `ess` use the scalar ρ chosen by `unit_trace`.
- **Normalizers** update once per `train_step` from the observations (and `global_state`) of ACT slots and BOOT slots with `reset_after`.
- `consumed_samples` counts ACT slots.

`KickstartLoss.compute` (keyword-only, contract signature) unrolls the teacher with `with_value=False`, computes the per-decider KL (`unit_kl`, forward = KL(teacher‖student)), reduces it over valid deciders (`sum` or `mean_valid`; APPO passes its resolved `entropy_reduction`) and averages over ACT slots.

`BaseAlgorithm` is copied unchanged except for the `TYPE_CHECKING` imports.

**Files:**
- Create: `src/colosseum/sp2/algorithms/base.py`, `src/colosseum/sp2/algorithms/appo.py`, `src/colosseum/sp2/bc/__init__.py` (empty), `src/colosseum/sp2/bc/kickstart.py`
- Modify: `tests/game_helpers.py` (imports; append the T4.2 block)
- Test: `tests/unit/test_appo_v2.py`, `tests/unit/test_kickstart_v2.py`

**Interfaces:**
- Consumes: `compute_vtrace_slots`, `VTraceOut` (T4.1); `TrajectoryChunk`, `SLOT_*` (T3.1); `AlgorithmConfig` with `ratio_mode`, `unit_trace`, `entropy_reduction`, `LRSchedule`, `LearnerConfig` (T1.7); `ActionSpec.num_deciders`, `.has_units`, `.has_masks`, `.full_mask`, `ObsSpec` (T1.3); `PolicyModel.unroll(..., global_state=, with_value=)`, `PolicyModel.update_normalizers(obs, global_state)`, `PolicyModel.step`, `UnrollOutput`, `Distribution.log_prob/unit_log_prob/unit_entropy/unit_valid/unit_kl/sample` (T2.1–T2.3); `tree_map`, `tree_stack`, `tree_to_torch` (T1.1); `cat_batch`, `slice_batch`, `state_to`, `tree_leaves` (`colosseum.networks.state`); test kit: `RolloutBuffer`, `BufferSpec` (T3.2), `make_test_model`, `CORE_KINDS` (T2.3), `Units` (T1.2), `RoleSpec` (T1.4).
- Produces (contract): `BaseAlgorithm`, `deep_cpu_copy` (SP1); `APPO(model, config, action_spec, device="cpu", pin_memory=False, kickstart=None)` with `train_step`, `compute_loss`, `evaluate_chunks -> (log_probs [S*B], values [S*B], unit_log_probs [S*B, K] | None)`, `model`, `policy_version`, `consumed_samples`, `set_progress`, `state_dict`, `load_state_dict`; `KickstartLoss(teacher, initial_lambda, decay_steps, direction)` with `compute(*, student_dist, obs, reset_after, state0, action_mask, actions, is_act, reduction)`, `current_lambda`, `step`, `state_dict`, `load_state_dict`, `teacher`, `to`.
  - Additions (see "Contract notes"): `resolve_modes(config, action_spec) -> (ratio_mode, unit_trace, entropy_reduction)`, `APPO.modes`.
  - Test kit: `synthetic_chunk(model, role, pattern, *, seed=0, agent_id="a", policy_version=0, logp_noise=0.0, random_units=True)`.

- [ ] **Step 1: Test kit**

Add to the import block of `tests/game_helpers.py` (skip what is already imported):

```python
import numpy as np
import torch

from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.tree import tree_map, tree_to_torch
from colosseum.sp2.worker.buffers import BufferSpec, RolloutBuffer
```

and append:

```python
# ---------------------------------------------------------------------------
# Part B (T4.2): synthetic chunk v2 from a model
# ---------------------------------------------------------------------------


def _random_leaves(spec: ObsSpec, rng: np.random.Generator):
    tree = spec.allocate(())

    def fill(leaf):
        if np.issubdtype(leaf.dtype, np.floating):
            return rng.normal(0.0, 1.0, leaf.shape).astype(leaf.dtype)
        if leaf.dtype == np.bool_:
            return rng.random(leaf.shape) < 0.5
        return rng.integers(0, 2, leaf.shape).astype(leaf.dtype)

    return tree_map(fill, tree)


def _random_unit_masks(mask, rng: np.random.Generator):
    """Copy of ``mask`` with every Units ``"unit"`` leaf randomized (p = 0.7, unit 0 always on)."""
    if not isinstance(mask, dict):
        return mask
    if set(mask) == {"unit", "action"}:
        unit = rng.random(mask["unit"].shape) < 0.7
        unit[..., 0] = True
        return {"unit": unit, "action": mask["action"].copy()}
    return {k: _random_unit_masks(v, rng) for k, v in mask.items()}


@torch.no_grad()
def synthetic_chunk(model, role, pattern: str, *, seed: int = 0, agent_id: str = "a",
                    policy_version: int = 0, logp_noise: float = 0.0, random_units: bool = True):
    """A TrajectoryChunk whose slots are spelled by ``pattern``, recorded like a worker would.

    Letters: ``A`` open ACT, ``T`` terminal ACT, ``B`` chunk-end BOOT, ``R`` truncation BOOT
    (``reset_after``), ``P`` PAD. Observations (and ``global_state`` if the role has one) are
    random; ACT actions are sampled from ``model``'s masked policy, stepped with the model
    state like on a worker (reset after T and R). ``logp_noise`` adds Gaussian noise to every
    valid decider's behavior log-prob (and their sum to the joint one) to make the chunk
    off-policy. With ``random_units`` Units masks get random ``unit`` rows.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    obs_spec = ObsSpec.from_space(role.observation_space)
    action_spec = ActionSpec.from_space(role.action_space)
    gs_spec = None if role.global_state_space is None else ObsSpec.from_space(role.global_state_space)
    buf = RolloutBuffer(len(pattern), BufferSpec(obs_spec, action_spec, gs_spec))
    state = model.initial_state(1)
    buf.begin(state, policy_version)
    K = action_spec.num_deciders
    for letter in pattern:
        obs = _random_leaves(obs_spec, rng)
        gs = None if gs_spec is None else _random_leaves(gs_spec, rng)
        if letter in "AT":
            mask = action_spec.full_mask(()) if action_spec.has_masks else None
            if mask is not None and random_units:
                mask = _random_unit_masks(mask, rng)
            obs_t = tree_to_torch(tree_map(lambda x: np.asarray(x)[None], obs))
            mask_t = None if mask is None else tree_to_torch(tree_map(lambda x: np.asarray(x)[None], mask))
            step = model.step(obs_t, state, mask_t)
            action_t = step.dist.sample()
            unit_lp = step.dist.unit_log_prob(action_t)[0].float().numpy()
            valid = step.dist.unit_valid(action_t)[0].numpy()
            noise = np.where(valid, rng.normal(0.0, logp_noise, K), 0.0).astype(np.float32)
            joint = float(step.dist.log_prob(action_t)[0]) + float(noise.sum())
            action = tree_map(lambda t: t[0].numpy(), action_t)
            buf.write_act(obs, gs, mask, action, joint, (unit_lp + noise) if K > 1 else None,
                          float(rng.normal()))
            state = step.state
            if letter == "T":
                buf.mark_terminal()
                state = model.initial_state(1)
        elif letter == "B":
            buf.write_boot(obs, gs, reset_after=False)
        elif letter == "R":
            buf.write_boot(obs, gs, reset_after=True)
            state = model.initial_state(1)
        elif letter == "P":
            buf.write_pad()
        else:
            raise ValueError(f"unknown slot letter {letter!r} in {pattern!r}")
    return buf.build_chunk(agent_id)
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_appo_v2.py`:

```python
"""APPO on chunk v2: unit modes, reductions over ACT slots, diagnostics, state (T4.2)."""

from __future__ import annotations

import copy
import itertools

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.networks.state import slice_batch
from colosseum.sp2.algorithms.appo import APPO, _select_chunks, resolve_modes
from colosseum.sp2.algorithms.vtrace import VTraceOut
from colosseum.sp2.core.config import AlgorithmConfig
from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.types import SLOT_ACT, TrajectoryChunk
from colosseum.sp2.envs.game import RoleSpec
from colosseum.sp2.envs.spaces import Units
from game_helpers import CORE_KINDS, make_test_model, synthetic_chunk

OBS = gymnasium.spaces.Box(-1.0, 1.0, (5,), np.float32)
DISCRETE = RoleSpec(OBS, gymnasium.spaces.Discrete(4))
UNITS = RoleSpec(OBS, Units(6, gymnasium.spaces.Discrete(3)))
UNITS_ONE = RoleSpec(OBS, Units(1, gymnasium.spaces.Discrete(3)))

DIAGNOSTICS = {
    "approx_kl", "clip_fraction", "clip_fraction_joint", "rho_mean", "rho_clip_frac", "c_clip_frac",
    "log_rho_abs_mean", "log_rho_abs_p95", "log_rho_joint_abs_mean", "ess", "deciders_valid_mean",
    "deciders_valid_max", "boot_frac", "pad_frac", "explained_variance",
}


def _patterns() -> list[str]:
    """4 chunks x 6 slots: 19 ACTs, 4 BOOTs (B or R) and 1 PAD."""
    return ["AAATAB", "ATAATP", "AARAAB", "AAAAAB"]


def _chunks(model, role, *, noise=0.3, seed=0, patterns=None):
    return [synthetic_chunk(model, role, p, seed=seed + i, logp_noise=noise)
            for i, p in enumerate(patterns or _patterns())]


def _algo(role, model, **cfg) -> APPO:
    return APPO(model, AlgorithmConfig(**cfg), ActionSpec.from_space(role.action_space), device="cpu")


def test_modes_resolve_auto_and_collapse_for_one_decider():
    units = ActionSpec.from_space(UNITS.action_space)
    single = ActionSpec.from_space(DISCRETE.action_space)
    one_unit = ActionSpec.from_space(UNITS_ONE.action_space)
    assert resolve_modes(AlgorithmConfig(), units) == ("per_unit", "geo_mean", "mean_valid")
    assert resolve_modes(AlgorithmConfig(ratio_mode="joint"), units) == ("joint", "geo_mean", "sum")
    assert resolve_modes(AlgorithmConfig(ratio_mode="joint", entropy_reduction="mean_valid", unit_trace="none"),
                         units) == ("joint", "none", "mean_valid")
    for spec in (single, one_unit):
        for r, t, e in itertools.product(("auto", "joint", "per_unit"), ("auto", "joint", "geo_mean", "none"),
                                         ("auto", "mean_valid", "sum")):
            cfg = AlgorithmConfig(ratio_mode=r, unit_trace=t, entropy_reduction=e)
            assert resolve_modes(cfg, spec) == ("joint", "joint", "sum")


@pytest.mark.parametrize("role", [DISCRETE, UNITS], ids=["discrete", "units"])
def test_compute_loss_returns_finite_losses_and_every_diagnostic(role):
    torch.manual_seed(0)
    model = make_test_model(role)
    losses = _algo(role, model).compute_loss(_chunks(model, role))
    assert {"total_loss", "policy_loss", "value_loss", "entropy"} | DIAGNOSTICS <= set(losses)
    for key, value in losses.items():
        assert torch.isfinite(value), key
    assert float(losses["boot_frac"]) == pytest.approx(4 / 24)
    assert float(losses["pad_frac"]) == pytest.approx(1 / 24)
    if role is UNITS:
        assert 1.0 <= float(losses["deciders_valid_mean"]) <= float(losses["deciders_valid_max"]) <= 6.0
    else:
        assert float(losses["deciders_valid_mean"]) == float(losses["deciders_valid_max"]) == 1.0


@pytest.mark.parametrize("role", [DISCRETE, UNITS_ONE], ids=["discrete", "units1"])
def test_one_decider_gives_identical_results_in_every_mode(role):
    torch.manual_seed(0)
    model = make_test_model(role)
    chunks = _chunks(model, role, noise=0.5)
    with torch.no_grad():
        reference = {k: float(v) for k, v in _algo(role, copy.deepcopy(model)).compute_loss(chunks).items()}
    for r, t, e in itertools.product(("joint", "per_unit"), ("joint", "geo_mean", "none"), ("mean_valid", "sum")):
        algo = _algo(role, copy.deepcopy(model), ratio_mode=r, unit_trace=t, entropy_reduction=e)
        with torch.no_grad():
            got = {k: float(v) for k, v in algo.compute_loss(chunks).items()}
        assert got == pytest.approx(reference, abs=1e-6, nan_ok=True), (r, t, e)


def _patched_vtrace(algo: APPO, rho: torch.Tensor):
    """Replace the clipped rho V-trace returns by ``rho`` (td and vs unchanged)."""
    inner = algo._compute_vtrace

    def patched(**kwargs):
        out = inner(**kwargs)
        return VTraceOut(vs=out.vs, td=out.td, clipped_rho=torch.where(kwargs["is_act"], rho, 0.0))

    algo._compute_vtrace = patched


def test_per_unit_does_not_multiply_the_advantage_by_rho_but_joint_does():
    torch.manual_seed(0)
    model = make_test_model(UNITS)
    chunks = _chunks(model, UNITS, noise=0.5)
    rho = torch.rand(6, len(chunks)) + 0.1
    for mode, changes in (("per_unit", False), ("joint", True)):
        plain = _algo(UNITS, copy.deepcopy(model), ratio_mode=mode).compute_loss(chunks)
        algo = _algo(UNITS, copy.deepcopy(model), ratio_mode=mode)
        _patched_vtrace(algo, rho)
        patched = algo.compute_loss(chunks)
        same = torch.allclose(plain["policy_loss"], patched["policy_loss"], atol=1e-7)
        assert same != changes, mode
        assert torch.allclose(plain["value_loss"], patched["value_loss"])


def _captured_log_rhos(algo: APPO, chunks):
    seen = {}
    inner = algo._compute_vtrace

    def spy(**kwargs):
        seen.update(kwargs)
        return inner(**kwargs)

    algo._compute_vtrace = spy
    algo.compute_loss(chunks)
    return seen


def test_unit_trace_sets_the_scalar_log_rho_of_vtrace():
    torch.manual_seed(0)
    model = make_test_model(UNITS)
    chunks = _chunks(model, UNITS, noise=0.5)
    algo = _algo(UNITS, copy.deepcopy(model))
    _, _, unit_lp = algo.evaluate_chunks(chunks)
    S, B = chunks[0].num_slots, len(chunks)
    behavior = torch.stack([c.behavior_unit_logp for c in chunks], dim=1).reshape(S * B, -1)
    kind = torch.stack([c.kind for c in chunks], dim=1).reshape(-1)
    valid = torch.stack([c.action_masks["unit"] for c in chunks], dim=1).reshape(S * B, -1)
    is_act = (kind == SLOT_ACT).unsqueeze(-1)
    diff = torch.where(valid & is_act, unit_lp - behavior, torch.zeros_like(unit_lp))
    joint = diff.sum(-1).reshape(S, B)
    geo = (diff.sum(-1) / valid.sum(-1).clamp(min=1)).reshape(S, B)
    for mode, expected, bars in (("joint", joint, (1.0, 1.0)), ("geo_mean", geo, (1.0, 1.0)),
                                 ("none", torch.zeros(S, B), (1.0, 1.0))):
        cfg = {"unit_trace": mode, "vtrace_rho_bar": 0.5, "vtrace_c_bar": 0.7}
        seen = _captured_log_rhos(_algo(UNITS, copy.deepcopy(model), **cfg), chunks)
        assert torch.allclose(seen["log_rhos"], expected, atol=1e-5), mode
        if mode == "none":
            assert (seen["rho_bar"], seen["c_bar"]) == bars
        else:
            assert (seen["rho_bar"], seen["c_bar"]) == (0.5, 0.7)


def test_entropy_reduction_sum_vs_mean_valid():
    torch.manual_seed(0)
    model = make_test_model(UNITS)
    chunks = [synthetic_chunk(model, UNITS, p, seed=i, random_units=False) for i, p in enumerate(_patterns())]
    total = _algo(UNITS, copy.deepcopy(model), entropy_reduction="sum").compute_loss(chunks)["entropy"]
    mean = _algo(UNITS, copy.deepcopy(model), entropy_reduction="mean_valid").compute_loss(chunks)["entropy"]
    assert float(mean.detach()) == pytest.approx(float(total.detach()) / 6, rel=1e-5)   # all 6 units valid


@pytest.mark.parametrize("core", ["none", "lstm"])
def test_losses_and_metrics_ignore_non_act_slots(core):
    torch.manual_seed(0)
    model = make_test_model(UNITS, core=core)
    chunks = _chunks(model, UNITS, patterns=["AATP", "ATAB", "AARP"])
    poisoned = []
    for c in chunks:
        bad = copy.deepcopy(c)
        pad = (bad.kind == 2).unsqueeze(-1)
        bad.obs = torch.where(pad, torch.full_like(bad.obs, float("nan")), bad.obs)
        bad.reward = torch.where(bad.kind != SLOT_ACT, torch.full_like(bad.reward, float("nan")), bad.reward)
        poisoned.append(bad)
    algo = _algo(UNITS, model)
    with torch.no_grad():
        clean, dirty = algo.compute_loss(chunks), algo.compute_loss(poisoned)
    for key in clean:
        assert torch.isfinite(dirty[key]), key
        assert float(dirty[key]) == pytest.approx(float(clean[key]), abs=1e-6, nan_ok=True), key


def test_value_targets_bootstrap_from_the_learners_boot_value():
    """Changing only the BOOT slot's observation changes the value loss: the bootstrap is V(boot)."""
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    chunk = synthetic_chunk(model, DISCRETE, "AAAB", seed=1)
    other = copy.deepcopy(chunk)
    other.obs[3] = other.obs[3] + 3.0
    algo = _algo(DISCRETE, model, vtrace_lambda=1.0)
    with torch.no_grad():
        a, b = algo.compute_loss([chunk]), algo.compute_loss([other])
    assert float(a["value_loss"]) != pytest.approx(float(b["value_loss"]))
    terminal = synthetic_chunk(model, DISCRETE, "AATP", seed=1)
    moved = copy.deepcopy(terminal)
    moved.obs[3] = moved.obs[3] + 3.0                          # a PAD after a terminal ACT
    with torch.no_grad():
        assert float(algo.compute_loss([terminal])["value_loss"]) == pytest.approx(
            float(algo.compute_loss([moved])["value_loss"]))


@pytest.mark.parametrize("core", CORE_KINDS)
@pytest.mark.parametrize("role", [DISCRETE, UNITS], ids=["discrete", "units"])
def test_evaluate_chunks_reproduces_behavior_log_probs_at_zero_lag(core, role):
    torch.manual_seed(0)
    model = make_test_model(role, core=core)
    chunks = _chunks(model, role, noise=0.0)
    lp, values, unit = _algo(role, model).evaluate_chunks(chunks)
    S, B = chunks[0].num_slots, len(chunks)
    kind = torch.stack([c.kind for c in chunks], dim=1).reshape(-1)
    act = kind == SLOT_ACT
    behavior = torch.stack([c.behavior_logp for c in chunks], dim=1).reshape(-1)
    assert lp.shape == values.shape == (S * B,)
    assert torch.allclose(lp[act], behavior[act], atol=1e-5)
    if role is UNITS:
        bu = torch.stack([c.behavior_unit_logp for c in chunks], dim=1).reshape(S * B, -1)
        assert torch.allclose(unit[act], bu[act], atol=1e-5)
    else:
        assert unit is None


def test_train_step_counts_act_slots_and_reports_metrics():
    torch.manual_seed(0)
    model = make_test_model(UNITS, core="gru")
    algo = _algo(UNITS, model, minibatch_chunks=2, num_epochs=2)
    before = [p.detach().clone() for p in model.parameters()]
    metrics = algo.train_step(_chunks(model, UNITS))
    assert algo.policy_version == 1
    assert algo.consumed_samples == sum(p.count("A") + p.count("T") for p in _patterns())
    assert DIAGNOSTICS | {"grad_norm", "skipped_updates", "policy_version", "lr", "total_loss"} <= set(metrics)
    assert all(isinstance(v, float) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_normalizers_update_once_from_act_and_reset_boot_slots(monkeypatch):
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    calls = []
    monkeypatch.setattr(model, "update_normalizers", lambda obs, gs=None: calls.append((obs.clone(), gs)))
    chunks = [synthetic_chunk(model, DISCRETE, p, seed=i) for i, p in enumerate(["AARP", "ATAB"])]
    _algo(DISCRETE, model, num_epochs=3, minibatch_chunks=1).train_step(chunks)
    assert len(calls) == 1
    obs, gs = calls[0]
    expected = [chunks[0].obs[0], chunks[1].obs[0], chunks[0].obs[1], chunks[1].obs[1], chunks[0].obs[2],
                chunks[1].obs[2]]                            # [S, B] row-major: slot 2 of chunk 1 is an ACT
    assert gs is None and obs.shape == (6, 5)
    assert torch.equal(obs, torch.stack(expected))


def test_minibatch_slice_equals_a_prepared_minibatch():
    torch.manual_seed(0)
    model = make_test_model(UNITS, core="lstm")
    chunks = _chunks(model, UNITS)
    algo = _algo(UNITS, model)
    idx = torch.tensor([2, 0])
    sliced = _select_chunks(algo._prepare_batch(chunks), idx)
    direct = algo._prepare_batch([chunks[2], chunks[0]])
    with torch.no_grad():
        a, b = algo._loss_from_batch(sliced), algo._loss_from_batch(direct)
    for key in a:
        assert float(a[key]) == pytest.approx(float(b[key]), abs=1e-6, nan_ok=True), key
    assert torch.equal(slice_batch(algo._prepare_batch(chunks)["initial_state"], idx)["h"],
                       direct["initial_state"]["h"])
    with pytest.raises(KeyError):
        _select_chunks({"kind": torch.zeros(2, 2), "bogus": torch.zeros(2, 2)}, torch.tensor([0]))


def test_state_dict_keys_and_resume_reproduce_the_next_update():
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    algo = _algo(DISCRETE, model)
    algo.train_step(_chunks(model, DISCRETE, seed=0))
    state = algo.state_dict()
    assert set(state) == {"optimizer", "progress", "scaler", "kickstart", "policy_version", "consumed_samples"}
    clone_model = copy.deepcopy(model)
    clone = _algo(DISCRETE, clone_model)
    clone.load_state_dict(state)
    batch = _chunks(model, DISCRETE, seed=10)
    torch.manual_seed(1)
    algo.train_step(batch)
    torch.manual_seed(1)
    clone.train_step(batch)
    for a, b in zip(model.parameters(), clone_model.parameters()):
        assert torch.equal(a, b)
    assert clone.policy_version == algo.policy_version == 2
    assert clone.consumed_samples == algo.consumed_samples


@pytest.mark.parametrize("schedule,progress,factor", [("constant", 0.5, 1.0), ("linear", 0.25, 0.75),
                                                      ("cosine", 0.5, 0.5)])
def test_lr_is_a_function_of_progress(schedule, progress, factor):
    model = make_test_model(DISCRETE)
    algo = _algo(DISCRETE, model, lr_schedule=schedule, learning_rate=1e-3)
    algo.set_progress(progress)
    assert algo.state_dict()["optimizer"]["param_groups"][0]["lr"] == pytest.approx(1e-3 * factor)


def test_payload_roundtrip_keeps_the_loss():
    torch.manual_seed(0)
    model = make_test_model(UNITS)
    chunks = _chunks(model, UNITS)
    algo = _algo(UNITS, model)
    with torch.no_grad():
        a = algo.compute_loss(chunks)
        b = algo.compute_loss([TrajectoryChunk.from_payload(c.to_payload()) for c in chunks])
    assert float(a["total_loss"]) == pytest.approx(float(b["total_loss"]), abs=1e-7)


@pytest.mark.gpu
@pytest.mark.parametrize("amp_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("core", ["none", "lstm"])
def test_amp_train_step_on_cuda_with_units(amp_dtype, core):
    torch.manual_seed(0)
    model = make_test_model(UNITS, core=core)
    chunks = _chunks(model, UNITS)
    algo = APPO(model, AlgorithmConfig(use_amp=True, amp_dtype=amp_dtype), ActionSpec.from_space(UNITS.action_space),
                device="cuda", pin_memory=True)
    metrics = algo.train_step(chunks)
    assert np.isfinite(metrics["total_loss"]) and algo.policy_version == 1
    lp, _, unit = algo.evaluate_chunks(chunks)
    assert lp.device.type == "cuda" and unit.shape[1] == 6
```

`tests/unit/test_kickstart_v2.py`:

```python
"""Kickstart v2: per-decider KL over ACT slots, teacher unrolled without value (T4.2)."""

from __future__ import annotations

import copy

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.networks.state import cat_batch
from colosseum.sp2.algorithms.appo import APPO
from colosseum.sp2.bc.kickstart import KickstartLoss
from colosseum.sp2.core.config import AlgorithmConfig
from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.tree import tree_map, tree_stack
from colosseum.sp2.core.types import SLOT_ACT
from colosseum.sp2.envs.game import RoleSpec
from colosseum.sp2.envs.spaces import Units
from game_helpers import make_test_model, synthetic_chunk

OBS = gymnasium.spaces.Box(-1.0, 1.0, (5,), np.float32)
UNITS = RoleSpec(OBS, Units(4, gymnasium.spaces.Discrete(3)))


def _batch(chunks):
    S, B = chunks[0].num_slots, len(chunks)
    stack = lambda get: tree_stack([get(c) for c in chunks], axis=1)  # noqa: E731
    actions = tree_map(lambda t: t.reshape(S * B, *t.shape[2:]), stack(lambda c: c.actions))
    return dict(obs=stack(lambda c: c.obs), reset_after=stack(lambda c: c.reset_after),
                action_mask=stack(lambda c: c.action_masks), actions=actions,
                is_act=stack(lambda c: c.kind) == SLOT_ACT,
                state0=cat_batch([c.initial_state for c in chunks]))


def _student_dist(model, batch):
    return model.unroll(batch["obs"], batch["state0"], batch["reset_after"], batch["action_mask"],
                        with_value=False).dist


def test_identical_teacher_gives_zero_kl_with_lstm_and_unit_masks():
    torch.manual_seed(0)
    student = make_test_model(UNITS, core="lstm")
    chunks = [synthetic_chunk(student, UNITS, p, seed=i) for i, p in enumerate(["AATP", "AARP"])]
    batch = _batch(chunks)
    loss = KickstartLoss(copy.deepcopy(student)).compute(student_dist=_student_dist(student, batch),
                                                          reduction="sum", **batch)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("reduction", ["sum", "mean_valid"])
def test_kl_is_reduced_over_valid_deciders_then_averaged_over_act_slots(direction, reduction):
    torch.manual_seed(0)
    student, teacher = make_test_model(UNITS), make_test_model(UNITS)
    chunks = [synthetic_chunk(student, UNITS, p, seed=i) for i, p in enumerate(["AATP", "ATAB"])]
    batch = _batch(chunks)
    s_dist = _student_dist(student, batch)
    kick = KickstartLoss(teacher, initial_lambda=0.5, direction=direction)
    loss = kick.compute(student_dist=s_dist, reduction=reduction, **batch)
    with torch.no_grad():
        t_dist = _student_dist(teacher, batch)
    kl = t_dist.unit_kl(s_dist, batch["actions"]) if direction == "forward" else s_dist.unit_kl(
        t_dist, batch["actions"])
    valid = s_dist.unit_valid(batch["actions"])
    per_slot = torch.where(valid, kl, 0.0).sum(-1)
    if reduction == "mean_valid":
        per_slot = per_slot / valid.sum(-1).clamp(min=1)
    act = batch["is_act"].reshape(-1)
    expected = 0.5 * per_slot[act].mean()
    assert float(loss.detach()) == pytest.approx(float(expected.detach()), rel=1e-5)
    assert float(loss.detach()) > 0


def test_teacher_is_unrolled_without_value_and_frozen():
    torch.manual_seed(0)
    student = make_test_model(UNITS)
    teacher = make_test_model(UNITS)
    calls = []
    original = teacher.unroll

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    teacher.unroll = spy
    kick = KickstartLoss(teacher)
    assert all(not p.requires_grad for p in teacher.parameters())
    batch = _batch([synthetic_chunk(student, UNITS, "AATP")])
    kick.compute(student_dist=_student_dist(student, batch), reduction="sum", **batch)
    assert calls and calls[0]["with_value"] is False and "global_state" not in calls[0]


def test_lambda_decays_and_state_round_trips():
    kick = KickstartLoss(make_test_model(UNITS), initial_lambda=1.0, decay_steps=4)
    for _ in range(3):
        kick.step()
    assert kick.current_lambda == pytest.approx(0.25)
    other = KickstartLoss(make_test_model(UNITS), initial_lambda=1.0, decay_steps=4)
    other.load_state_dict(kick.state_dict())
    assert other.current_lambda == pytest.approx(0.25)
    with pytest.raises(ValueError):
        KickstartLoss(make_test_model(UNITS), direction="sideways")


def test_appo_adds_kickstart_with_the_entropy_reduction_and_checks_the_state_layout():
    torch.manual_seed(0)
    student = make_test_model(UNITS, core="gru")
    teacher = make_test_model(UNITS, core="gru")
    spec = ActionSpec.from_space(UNITS.action_space)
    algo = APPO(student, AlgorithmConfig(), spec, kickstart=KickstartLoss(teacher, initial_lambda=0.5))
    chunks = [synthetic_chunk(student, UNITS, "AATP", seed=s) for s in range(2)]
    metrics = algo.train_step(chunks)
    assert metrics["kickstart_loss"] > 0 and metrics["kickstart_lambda"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="state layout"):
        APPO(make_test_model(UNITS, core="lstm"), AlgorithmConfig(), spec,
             kickstart=KickstartLoss(make_test_model(UNITS, core="none")))
```

- [ ] **Step 3: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_appo_v2.py tests/unit/test_kickstart_v2.py -q`
Expected: `ModuleNotFoundError: No module named 'colosseum.sp2.algorithms.appo'` (and `colosseum.sp2.bc`).

- [ ] **Step 4: Copy `BaseAlgorithm`**

Copy `src/colosseum/algorithms/base.py` to `src/colosseum/sp2/algorithms/base.py`; change only the `TYPE_CHECKING` block and one docstring line:

```python
if TYPE_CHECKING:
    from colosseum.sp2.core.types import TrajectoryChunk
    from colosseum.sp2.networks.model import PolicyModel
```

```python
    On-policy algorithms (APPO, PPO) train directly on incoming trajectory
    chunks (chunk v2: act / boot / pad slots). Off-policy algorithms (R2D2, DQN) add chunks to a replay buffer
```

- [ ] **Step 5: Kickstart v2**

Create the empty `src/colosseum/sp2/bc/__init__.py`, then `src/colosseum/sp2/bc/kickstart.py`:

```python
"""Online behavioral cloning via kickstarting (Schmitt et al., 2018), on chunk v2.

Adds ``lambda * KL`` between a frozen teacher policy and the student to the RL loss.
Lambda decays linearly from ``initial_lambda`` to 0 over ``decay_steps`` train steps.

Direction (``training.kickstart_kl``):
- ``"forward"`` (default): KL(teacher || student), mode-covering (Kickstarting, AlphaStar, VPT);
- ``"reverse"``: KL(student || teacher), mode-seeking.

The teacher is unrolled with ``with_value=False`` over the same ``[S, B]`` slots as the
student, with the chunks' action masks, ``reset_after`` flags and initial states. The KL is
computed per decider (``Distribution.unit_kl``, 0 where a decider is invalid), reduced over
deciders per ``reduction`` (``"sum"`` or ``"mean_valid"``) and averaged over ACT slots only.

In SP2 the teacher stays global and is built from the student's networks config, so both
share one state layout and the chunk's ``initial_state`` (recorded by the student's behavior
policy) is the teacher's ``state0``: exact when teacher == student, an approximation after.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from colosseum.networks.state import State
from colosseum.sp2.core.tree import Tree
from colosseum.sp2.networks.dist import Distribution
from colosseum.sp2.networks.model import PolicyModel

KLDirection = Literal["forward", "reverse"]


class KickstartLoss:
    """Decaying KL penalty between a frozen teacher and the student policy."""

    def __init__(
        self,
        teacher: PolicyModel,
        initial_lambda: float = 1.0,
        decay_steps: int = 50_000,
        direction: KLDirection = "forward",
    ) -> None:
        if direction not in ("forward", "reverse"):
            raise ValueError(f"kickstart direction must be 'forward' or 'reverse', got {direction!r}")
        self._teacher = teacher
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)
        self._initial_lambda = float(initial_lambda)
        self._decay_steps = max(1, int(decay_steps))
        self._direction: KLDirection = direction
        self._current_step = 0

    @property
    def teacher(self) -> PolicyModel:
        return self._teacher

    @property
    def direction(self) -> KLDirection:
        return self._direction

    @property
    def step_count(self) -> int:
        return self._current_step

    @property
    def current_lambda(self) -> float:
        """Current (decayed) lambda."""
        progress = min(1.0, self._current_step / self._decay_steps)
        return self._initial_lambda * (1.0 - progress)

    def step(self) -> None:
        """Advance the decay by one train step."""
        self._current_step += 1

    def to(self, device: str | torch.device) -> KickstartLoss:
        self._teacher.to(device)
        return self

    def state_dict(self) -> dict[str, int]:
        return {"step": self._current_step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._current_step = int(state["step"])

    def compute(
        self,
        *,
        student_dist: Distribution,
        obs: Tree,
        reset_after: Tensor,
        state0: State,
        action_mask: Tree | None,
        actions: Tree,
        is_act: Tensor,
        reduction: Literal["mean_valid", "sum"],
    ) -> Tensor:
        """Scaled kickstart loss ``lambda * mean over ACT slots of the reduced per-decider KL``.

        Args:
            student_dist: the student's masked distribution over ``S*B`` time-major rows,
                from the algorithm's main ``unroll``.
            obs: observation tree, leaves ``[S, B, ...]``.
            reset_after: ``[S, B]`` bool; the state is reset after slot s.
            state0: the teacher's initial state (leaves ``[B, ...]``).
            action_mask: mask tree, leaves ``[S, B, ...]``, or None.
            actions: the recorded actions, leaves ``[S*B, ...]`` (they gate ``only_if`` children).
            is_act: ``[S, B]`` bool; only ACT slots contribute.
            reduction: ``"sum"`` or ``"mean_valid"`` over the valid deciders of a slot.
        """
        lam = self.current_lambda
        if lam <= 0:
            return torch.zeros((), device=reset_after.device)
        with torch.no_grad():
            teacher_dist = self._teacher.unroll(obs, state0, reset_after.bool(), action_mask, with_value=False).dist
        if self._direction == "forward":
            kl = teacher_dist.unit_kl(student_dist, actions)
        else:
            kl = student_dist.unit_kl(teacher_dist, actions)
        kl = kl.float()                                             # [S*B, K]
        valid = student_dist.unit_valid(actions)
        per_slot = torch.where(valid, kl, torch.zeros_like(kl)).sum(-1)
        if reduction == "mean_valid":
            per_slot = per_slot / valid.sum(-1).clamp(min=1).to(per_slot.dtype)
        act = is_act.reshape(-1)
        mean = torch.where(act, per_slot, torch.zeros_like(per_slot)).sum() / act.sum().clamp(min=1)
        return lam * mean
```

- [ ] **Step 6: APPO v2**

`src/colosseum/sp2/algorithms/appo.py`:

```python
"""APPO on chunk v2: async PPO with V-trace over act / boot / pad slots (spec block 6).

- The learner's network computes every value (bootstrap included) with ``PolicyModel.unroll``
  over all slots of the batch; BOOT slots exist only for that.
- Every loss and diagnostic reduces over ACT slots only (``torch.where``, never a mask product).
- ``K`` = ``ActionSpec.num_deciders``. Three switches (``AlgorithmConfig``):
  - ``unit_trace`` (``auto | joint | geo_mean | none``): the scalar log-ratio fed to V-trace:
    sum of the per-decider log-ratios, their mean over valid deciders, or 0 with
    rho = c = 1 (TD(lambda)). ``auto`` = ``geo_mean`` with ``Units``, else ``joint``.
  - ``ratio_mode`` (``auto | joint | per_unit``): ``joint`` = PPO-clip on the joint ratio with
    the advantage multiplied by the clipped scalar rho (SP1); ``per_unit`` = PPO-clip per
    decider on the shared advantage WITHOUT the rho factor, mean over valid deciders, then
    mean over ACT slots. ``auto`` = ``per_unit`` with ``Units``, else ``joint``.
  - ``entropy_reduction`` (``auto | mean_valid | sum``): how entropy and kickstart KL reduce
    over deciders; ``auto`` = ``sum`` for ``joint``, ``mean_valid`` for ``per_unit``.
  With ``K == 1`` all modes collapse to the joint path (one decider: per-decider and joint
  quantities coincide, and the joint path keeps SP1's rho-weighted advantage).
- Observation normalizers update once per train step from ACT slots and BOOT slots with
  ``reset_after`` (a chunk-end BOOT repeats the next chunk's first ACT).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

import torch
from torch import Tensor

from colosseum.networks.state import cat_batch, slice_batch, state_to, tree_leaves
from colosseum.sp2.algorithms.base import BaseAlgorithm, deep_cpu_copy
from colosseum.sp2.algorithms.vtrace import compute_vtrace_slots
from colosseum.sp2.bc.kickstart import KickstartLoss
from colosseum.sp2.core.config import AlgorithmConfig, LRSchedule
from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.tree import tree_map, tree_stack
from colosseum.sp2.core.types import SLOT_ACT, SLOT_BOOT, SLOT_PAD, TrajectoryChunk
from colosseum.sp2.networks.model import PolicyModel, UnrollOutput

logger = logging.getLogger(__name__)

RatioMode = Literal["joint", "per_unit"]
UnitTrace = Literal["joint", "geo_mean", "none"]
EntropyReduction = Literal["mean_valid", "sum"]


def resolve_modes(config: AlgorithmConfig, action_spec: ActionSpec) -> tuple[RatioMode, UnitTrace, EntropyReduction]:
    """Resolve ``auto`` and the ``K == 1`` collapse (module docstring)."""
    if action_spec.num_deciders == 1:
        return "joint", "joint", "sum"
    ratio: RatioMode = config.ratio_mode if config.ratio_mode != "auto" else (
        "per_unit" if action_spec.has_units else "joint")
    trace: UnitTrace = config.unit_trace if config.unit_trace != "auto" else (
        "geo_mean" if action_spec.has_units else "joint")
    reduction: EntropyReduction = config.entropy_reduction if config.entropy_reduction != "auto" else (
        "sum" if ratio == "joint" else "mean_valid")
    return ratio, trace, reduction


def _check_teacher_state_layout(student: PolicyModel, teacher: PolicyModel) -> None:
    """The teacher reuses the student's chunk initial states, so the layouts must match."""
    student_shapes = [tuple(t.shape) for t in tree_leaves(student.initial_state(1))]
    teacher_shapes = [tuple(t.shape) for t in tree_leaves(teacher.initial_state(1))]
    if student_shapes != teacher_shapes:
        raise ValueError(
            "kickstart teacher must share the student's state layout "
            f"(student state leaves {student_shapes}, teacher {teacher_shapes}); "
            "build the teacher from the student's networks config"
        )


# Layout of every key ``APPO._prepare_batch`` produces (``_select_chunks`` rejects others).
_TIME_MAJOR_KEYS = frozenset({
    "obs", "global_state", "actions", "action_masks", "kind", "reward", "terminal", "reset_after",
    "behavior_logp", "behavior_unit_logp",
})                                   # trees / tensors with leaves [S, B, ...]
_STATE_KEY = "initial_state"         # State pytree, leaves [B, ...]


def _select_chunks(batch: dict, idx: Tensor) -> dict:
    """Rows ``idx`` of the chunk dimension of a prepared batch (``[S, B, ...]`` -> ``[S, b, ...]``)."""
    idx = idx.to(batch["kind"].device)
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if key == _STATE_KEY:
            out[key] = slice_batch(value, idx)
        elif key in _TIME_MAJOR_KEYS:
            out[key] = None if value is None else tree_map(lambda t: t[:, idx], value)
        else:
            raise KeyError(f"_select_chunks: batch key {key!r} has no declared layout")
    return out


def _act_mean(x: Tensor, is_act: Tensor, n_act: Tensor) -> Tensor:
    """Mean of ``x`` over ACT slots (``where``-selected: values elsewhere may be NaN)."""
    return torch.where(is_act, x, torch.zeros_like(x)).sum() / n_act


def _explained_variance(predicted: Tensor, target: Tensor, is_act: Tensor, n_act: Tensor) -> Tensor:
    """1 - Var(target - predicted) / Var(target) over ACT slots; 0 for a (near) constant target."""
    def var(x: Tensor) -> Tensor:
        mean = _act_mean(x, is_act, n_act)
        return _act_mean((x - mean) ** 2, is_act, n_act)

    var_target = var(target.float())
    ev = 1.0 - var(target.float() - predicted.float()) / var_target.clamp(min=1e-12)
    return torch.where(var_target < 1e-8, torch.zeros_like(ev), ev)   # no host sync


class APPO(BaseAlgorithm):
    """Async PPO with V-trace over chunk v2 slots (module docstring)."""

    def __init__(
        self,
        model: PolicyModel,
        config: AlgorithmConfig,
        action_spec: ActionSpec,
        device: str | torch.device = "cpu",
        pin_memory: bool = False,
        kickstart: KickstartLoss | None = None,
    ):
        self._model = model.to(device)
        if kickstart is not None:
            _check_teacher_state_layout(self._model, kickstart.teacher)
            kickstart.to(device)
        self._config = config
        self._action_spec = action_spec
        self._num_deciders = action_spec.num_deciders
        self._ratio_mode, self._unit_trace, self._entropy_reduction = resolve_modes(config, action_spec)
        self._device = device
        self._policy_version = 0
        self._consumed_samples = 0
        self._kickstart = kickstart
        self._pin_memory = pin_memory

        # AMP (automatic mixed precision)
        self._use_amp = config.use_amp and str(device).startswith("cuda")
        self._amp_dtype = getattr(torch, config.amp_dtype, torch.float16)
        self._scaler = torch.amp.GradScaler("cuda") if self._use_amp else None

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=config.learning_rate)
        self._zero_loss = torch.tensor(0.0, device=device)
        self._compute_vtrace = (
            torch.compile(compute_vtrace_slots) if config.use_torch_compile else compute_vtrace_slots
        )

        # The LR is a function of training progress (share of the global env-step budget),
        # set by the learner through set_progress() before every train step.
        self._progress = 0.0
        self.set_progress(0.0)

    @property
    def model(self) -> PolicyModel:
        return self._model

    @property
    def policy_version(self) -> int:
        return self._policy_version

    @property
    def consumed_samples(self) -> int:
        """Total ACT slots passed to train_step."""
        return self._consumed_samples

    @property
    def modes(self) -> tuple[str, str, str]:
        """Resolved ``(ratio_mode, unit_trace, entropy_reduction)``."""
        return self._ratio_mode, self._unit_trace, self._entropy_reduction

    def set_progress(self, progress: float) -> None:
        """Set the share (0..1) of the global env-step budget consumed so far (drives the LR)."""
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

    # ------------------------------------------------------------------
    # Batches
    # ------------------------------------------------------------------

    def _prepare_batch(self, chunks: list[TrajectoryChunk]) -> dict[str, Any]:
        """Stack chunks into ``[S, B, ...]`` trees and move them to the device once."""
        device = self._device
        use_pinning = self._pin_memory and str(device).startswith("cuda")

        def stack(get) -> Any:
            first = get(chunks[0])
            if first is None:
                return None
            return tree_stack([get(c) for c in chunks], axis=1)

        batch_cpu = {
            "obs": stack(lambda c: c.obs),
            "global_state": stack(lambda c: c.global_state),
            "actions": stack(lambda c: c.actions),
            "action_masks": stack(lambda c: c.action_masks),
            "kind": stack(lambda c: c.kind),
            "reward": stack(lambda c: c.reward),
            "terminal": stack(lambda c: c.terminal),
            "reset_after": stack(lambda c: c.reset_after),
            "behavior_logp": stack(lambda c: c.behavior_logp),
            "behavior_unit_logp": stack(lambda c: c.behavior_unit_logp),
        }

        def move(t: Tensor) -> Tensor:
            if use_pinning:
                return t.pin_memory().to(device, non_blocking=True)
            return t.to(device)

        batch: dict[str, Any] = {k: None if v is None else tree_map(move, v) for k, v in batch_cpu.items()}
        batch["initial_state"] = state_to(cat_batch([c.initial_state for c in chunks]), device)
        return batch

    def _autocast(self) -> torch.autocast:
        """AMP autocast context (a no-op unless AMP is enabled on CUDA)."""
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=self._use_amp)

    def _flat_actions(self, batch: dict) -> Any:
        S, B = batch["kind"].shape
        return tree_map(lambda t: t.reshape(S * B, *t.shape[2:]), batch["actions"])

    def _evaluate(self, batch: dict) -> tuple[UnrollOutput, dict[str, Tensor]]:
        """The one evaluation path (training and ``evaluate_chunks``): unroll the model over
        every slot under AMP autocast, from the chunks' initial states with the chunks'
        ``reset_after`` flags, masks and ``global_state``.

        Returns the ``UnrollOutput`` and float32 tensors: ``log_probs`` / ``values`` ``[S, B]``,
        ``unit_log_probs`` / ``unit_entropy`` ``[S, B, K]`` and bool ``unit_valid`` ``[S, B, K]``.
        """
        S, B = batch["kind"].shape
        K = self._num_deciders
        actions = self._flat_actions(batch)
        with self._autocast():
            out = self._model.unroll(
                batch["obs"], batch["initial_state"], batch["reset_after"].bool(), batch["action_masks"],
                global_state=batch["global_state"], with_value=True,
            )
            dist = out.dist
            evals = {
                "log_probs": dist.log_prob(actions).float().reshape(S, B),
                "values": out.value.float().reshape(S, B),
                "unit_log_probs": dist.unit_log_prob(actions).float().reshape(S, B, K),
                "unit_entropy": dist.unit_entropy(actions).float().reshape(S, B, K),
                "unit_valid": dist.unit_valid(actions).bool().reshape(S, B, K),
            }
        return out, evals

    @torch.no_grad()
    def evaluate_chunks(self, chunks: list[TrajectoryChunk]) -> tuple[Tensor, Tensor, Tensor | None]:
        """Current model's log-probs of the recorded actions, its values, and (``K > 1``)
        per-decider log-probs: ``[S*B]``, ``[S*B]``, ``[S*B, K]`` in time-major order
        (index ``s*B + b`` is slot ``s`` of ``chunks[b]``), computed exactly as in training.
        """
        _, ev = self._evaluate(self._prepare_batch(chunks))
        unit = ev["unit_log_probs"].reshape(-1, self._num_deciders) if self._num_deciders > 1 else None
        return ev["log_probs"].reshape(-1), ev["values"].reshape(-1), unit

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, chunks: list[TrajectoryChunk]) -> dict[str, Tensor]:
        """APPO loss and diagnostics for one minibatch of chunks (see ``_loss_from_batch``)."""
        return self._loss_from_batch(self._prepare_batch(chunks))

    def _loss_from_batch(self, batch: dict) -> dict[str, Tensor]:
        """APPO loss for one prepared minibatch (``_prepare_batch`` output or a slice of it).

        1. Unroll the model over every slot (``_evaluate``).
        2. Per-decider and joint log-ratios on ACT slots; the scalar V-trace log-ratio from
           ``unit_trace``.
        3. V-trace(lambda) over the slots (values of the learner's network everywhere).
        4. Policy loss per ``ratio_mode``; value MSE; entropy per ``entropy_reduction``.
        5. Optional kickstart KL with the same reduction.
        6. Diagnostics over ACT slots.
        """
        cfg = self._config
        K = self._num_deciders
        out, ev = self._evaluate(batch)
        kind = batch["kind"]
        is_act = kind == SLOT_ACT                                   # [S, B]
        n_act = is_act.sum().clamp(min=1).float()
        valid = ev["unit_valid"] & is_act.unsqueeze(-1)             # [S, B, K]
        n_valid = valid.sum(-1)                                     # [S, B]
        n_valid_f = n_valid.clamp(min=1).float()
        zeros_sb = torch.zeros_like(ev["values"])
        zeros_sbk = torch.zeros_like(ev["unit_log_probs"])

        behavior_unit = batch["behavior_unit_logp"] if K > 1 else batch["behavior_logp"].unsqueeze(-1)
        unit_log_ratio = torch.where(
            valid, torch.clamp(ev["unit_log_probs"] - behavior_unit, -20.0, 20.0), zeros_sbk)
        joint_log_ratio = torch.where(
            is_act, torch.clamp(ev["log_probs"] - batch["behavior_logp"], -20.0, 20.0), zeros_sb)

        # Scalar log-ratio for V-trace (unit_trace).
        with torch.no_grad():
            rho_bar, c_bar = cfg.vtrace_rho_bar, cfg.vtrace_c_bar
            if self._unit_trace == "joint":
                trace_log_rho = joint_log_ratio.detach()
            elif self._unit_trace == "geo_mean":
                trace_log_rho = unit_log_ratio.detach().sum(-1) / n_valid_f
            else:                                                   # "none": rho = c = 1
                trace_log_rho, rho_bar, c_bar = zeros_sb, 1.0, 1.0
            vt = self._compute_vtrace(
                log_rhos=trace_log_rho, rewards=batch["reward"], values=ev["values"].detach(), is_act=is_act,
                terminal=batch["terminal"].bool(), gamma=cfg.gamma, rho_bar=rho_bar, c_bar=c_bar,
                lam=cfg.vtrace_lambda,
            )

        # Advantages: joint mode weights them with the clipped scalar rho, per_unit does not.
        adv = vt.td * vt.clipped_rho if self._ratio_mode == "joint" else vt.td
        if cfg.normalize_advantages:
            mean = _act_mean(adv, is_act, n_act)
            var = _act_mean((adv - mean) ** 2, is_act, n_act) * n_act / (n_act - 1).clamp(min=1)
            normalized = (adv - mean) / (var.sqrt() + 1e-8)
            adv = torch.where(n_act > 1, normalized, adv)
        adv = torch.where(is_act, adv, zeros_sb).detach()

        eps = cfg.eps_clip
        if self._ratio_mode == "joint":
            ratio = torch.exp(joint_log_ratio)
            surr = torch.min(ratio * adv, torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv)
            policy_loss = -_act_mean(surr, is_act, n_act)
        else:
            unit_ratio = torch.exp(unit_log_ratio)
            unit_adv = adv.unsqueeze(-1)
            unit_surr = torch.min(unit_ratio * unit_adv, torch.clamp(unit_ratio, 1.0 - eps, 1.0 + eps) * unit_adv)
            step_surr = torch.where(valid, unit_surr, zeros_sbk).sum(-1) / n_valid_f
            policy_loss = -_act_mean(step_surr, is_act, n_act)

        value_loss = _act_mean((ev["values"] - vt.vs.detach()) ** 2, is_act, n_act)
        unit_entropy = torch.where(valid, ev["unit_entropy"], zeros_sbk).sum(-1)
        if self._entropy_reduction == "mean_valid":
            unit_entropy = unit_entropy / n_valid_f
        entropy = _act_mean(unit_entropy, is_act, n_act)
        total_loss = policy_loss + cfg.value_loss_coeff * value_loss - cfg.entropy_coeff * entropy

        kickstart_loss = self._zero_loss
        if self._kickstart is not None and self._kickstart.current_lambda > 0:
            with self._autocast():
                kickstart_loss = self._kickstart.compute(
                    student_dist=out.dist, obs=batch["obs"], reset_after=batch["reset_after"],
                    state0=batch["initial_state"], action_mask=batch["action_masks"],
                    actions=self._flat_actions(batch), is_act=is_act, reduction=self._entropy_reduction,
                )
            total_loss = total_loss + kickstart_loss

        result = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            **self._diagnostics(batch, ev, is_act, n_act, valid, n_valid, unit_log_ratio, joint_log_ratio,
                                trace_log_rho, vt.vs),
        }
        if self._kickstart is not None:
            result["kickstart_loss"] = kickstart_loss.detach()
            result["kickstart_lambda"] = torch.tensor(self._kickstart.current_lambda, device=total_loss.device)
        return result

    @torch.no_grad()
    def _diagnostics(self, batch, ev, is_act, n_act, valid, n_valid, unit_log_ratio, joint_log_ratio,
                     trace_log_rho, vs) -> dict[str, Tensor]:
        cfg = self._config
        eps = cfg.eps_clip
        n_dec = valid.sum().clamp(min=1).float()
        kind = batch["kind"]
        unit_ratio = torch.exp(unit_log_ratio)
        joint_ratio = torch.exp(joint_log_ratio)
        rho = torch.where(is_act, torch.exp(torch.clamp(trace_log_rho, -20.0, 20.0)), torch.zeros_like(trace_log_rho))
        abs_unit = unit_log_ratio.abs()
        nan = torch.full_like(abs_unit, float("nan"))
        sum_rho = rho.sum()
        sum_rho2 = (rho * rho).sum()

        def frac_dec(x: Tensor) -> Tensor:
            return torch.where(valid, x.float(), torch.zeros_like(abs_unit)).sum() / n_dec

        return {
            "approx_kl": frac_dec((unit_ratio - 1) - unit_log_ratio),
            "clip_fraction": frac_dec((unit_ratio - 1.0).abs() > eps),
            "clip_fraction_joint": _act_mean(((joint_ratio - 1.0).abs() > eps).float(), is_act, n_act),
            "rho_mean": _act_mean(rho, is_act, n_act),
            "rho_clip_frac": _act_mean((rho > cfg.vtrace_rho_bar).float(), is_act, n_act),
            "c_clip_frac": _act_mean((rho > cfg.vtrace_c_bar).float(), is_act, n_act),
            "log_rho_abs_mean": frac_dec(abs_unit),
            "log_rho_abs_p95": torch.nanquantile(torch.where(valid, abs_unit, nan).flatten(), 0.95),
            "log_rho_joint_abs_mean": _act_mean(joint_log_ratio.abs(), is_act, n_act),
            "ess": sum_rho * sum_rho / (n_act * sum_rho2).clamp(min=1e-12),
            "deciders_valid_mean": _act_mean(n_valid.float(), is_act, n_act),
            "deciders_valid_max": torch.where(is_act, n_valid, torch.zeros_like(n_valid)).max().float(),
            "boot_frac": (kind == SLOT_BOOT).float().mean(),
            "pad_frac": (kind == SLOT_PAD).float().mean(),
            "explained_variance": _explained_variance(ev["values"], vs, is_act, n_act),
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _clip_gradients(self) -> Tensor:
        """Clip to ``max_grad_norm``; return the total gradient norm BEFORE clipping."""
        return torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.max_grad_norm)

    def _update_normalizers(self, batch: dict) -> None:
        """Once per train step: observations of ACT slots and of BOOT slots with ``reset_after``."""
        kind = batch["kind"]
        sel = (kind == SLOT_ACT) | ((kind == SLOT_BOOT) & batch["reset_after"].bool())
        obs = tree_map(lambda t: t[sel], batch["obs"])
        gs = None if batch["global_state"] is None else tree_map(lambda t: t[sel], batch["global_state"])
        self._model.update_normalizers(obs, gs)

    def train_step(self, chunks: list[TrajectoryChunk]) -> dict[str, float]:
        """One training step: normalizer update, then num_epochs x minibatches of updates.

        Minibatches are slices of the step's batch over the chunk dimension B, so each
        chunk's slot sequence stays intact. Metrics are minibatch means read back with one
        host sync; ``grad_norm`` averages the finite pre-clipping norms; ``skipped_updates``
        counts the minibatches whose step the GradScaler skipped.
        """
        cfg = self._config
        sums: dict[str, Tensor] = {}
        num_updates = 0

        full_batch = self._prepare_batch(chunks)
        grad_norm_sum = torch.zeros((), dtype=torch.float64, device=full_batch["kind"].device)
        finite_updates = torch.zeros_like(grad_norm_sum)
        self._update_normalizers(full_batch)

        mb_size = cfg.minibatch_chunks if cfg.minibatch_chunks > 0 else len(chunks)
        for _epoch in range(cfg.num_epochs):
            indices = torch.randperm(len(chunks))
            for start in range(0, len(chunks), mb_size):
                losses = self._loss_from_batch(_select_chunks(full_batch, indices[start:start + mb_size]))
                total_loss = losses["total_loss"]

                self._optimizer.zero_grad()
                if self._scaler is not None:
                    self._scaler.scale(total_loss).backward()
                    self._scaler.unscale_(self._optimizer)
                    grad_norm = self._clip_gradients()
                    self._scaler.step(self._optimizer)      # skipped if the gradients are not finite
                    self._scaler.update()
                else:
                    total_loss.backward()
                    grad_norm = self._clip_gradients()
                    self._optimizer.step()

                finite = torch.isfinite(grad_norm)
                grad_norm_sum += torch.where(finite, grad_norm.detach().double(), 0.0)
                finite_updates += finite
                for key, value in losses.items():
                    value = value.detach().double()
                    sums[key] = sums[key] + value if key in sums else value
                num_updates += 1

        if self._kickstart is not None:
            self._kickstart.step()
        self._policy_version += 1
        self._consumed_samples += sum(c.num_acts for c in chunks)

        keys = list(sums)
        *totals, grad_norm_total, num_finite = torch.stack(
            [sums[k] for k in keys] + [grad_norm_sum, finite_updates]
        ).tolist()
        metrics = {k: v / max(1, num_updates) for k, v in zip(keys, totals)}
        metrics["grad_norm"] = grad_norm_total / num_finite if num_finite > 0 else float("nan")
        metrics["skipped_updates"] = float(num_updates - num_finite) if self._scaler is not None else 0.0
        metrics["policy_version"] = float(self._policy_version)
        metrics["lr"] = float(self._optimizer.param_groups[0]["lr"])
        return metrics

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Deep CPU copy of the training state (model weights excluded).

        Keys: ``optimizer``, ``progress``, ``scaler`` (None without AMP), ``kickstart`` (None
        without kickstart), ``policy_version``, ``consumed_samples``.
        """
        return deep_cpu_copy({
            "optimizer": self._optimizer.state_dict(),
            "progress": float(self._progress),
            "scaler": self._scaler.state_dict() if self._scaler is not None else None,
            "kickstart": self._kickstart.state_dict() if self._kickstart is not None else None,
            "policy_version": int(self._policy_version),
            "consumed_samples": int(self._consumed_samples),
        })

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore :meth:`state_dict` output (SP1 semantics, warnings on scaler/kickstart mismatch)."""
        state = deep_cpu_copy(state)
        self._optimizer.load_state_dict(state["optimizer"])
        self._load_optional_state("GradScaler", self._scaler, state["scaler"])
        self._load_optional_state("kickstart", self._kickstart, state["kickstart"])
        self._policy_version = int(state["policy_version"])
        self._consumed_samples = int(state["consumed_samples"])
        self.set_progress(float(state["progress"]))

    @staticmethod
    def _load_optional_state(name: str, component: Any, saved: dict | None) -> None:
        """Restore an optional component (GradScaler, kickstart) and warn on a mismatch."""
        if component is not None and saved is not None:
            component.load_state_dict(saved)
        elif saved is not None:
            logger.warning("Resume: the saved %s state is ignored because this run has no %s.", name, name)
        elif component is not None:
            logger.warning("Resume: no %s state was saved; this run's %s starts fresh.", name, name)
```

- [ ] **Step 7: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_appo_v2.py tests/unit/test_kickstart_v2.py -q -rs`
Expected: `35 passed, 4 skipped` (the 4 `gpu` tests are skipped without CUDA; add them to `docs/GPU_CHECKS.md` in T8.6).

- [ ] **Step 8: Full fast suite and lint** (as in T3.1, Step 5).

- [ ] **Step 9: Commit**

```bash
git add src/colosseum/sp2/algorithms/base.py src/colosseum/sp2/algorithms/appo.py src/colosseum/sp2/bc/__init__.py \
    src/colosseum/sp2/bc/kickstart.py tests/unit/test_appo_v2.py tests/unit/test_kickstart_v2.py tests/game_helpers.py
git commit -m "feat(sp2): APPO over chunk v2 with ratio_mode/unit_trace/entropy_reduction, ACT-only reductions, kickstart v2"
git push origin sp2-game-model
```

---

### Task T4.3: Contract — the learner reproduces the worker; the bootstrap is the learner's

Spec section 3 criterion 2 and section 6 ("Контракт"): the learner reproduces the worker's joint and per-unit log-probs at zero lag for the four cores on chunks with `boot`, `pad`, truncation and elimination; the bootstrap from `final_obs` / chunk-end observations is computed by the learner's network; `uint8` reaches the model on both sides.

Chunks come from the real `RolloutLoop` + `MatchRunner` on two `TickGame`s (3 seats each): one with an elimination (with a reward in the elimination step), a dead teammate and a truncation; one turn-based that ends by the rules. With `chunk_length=5` they contain `T`, `B`, `R` and `P` slots and chunks that start mid-episode with a non-zero state (asserted as preconditions). The learner's weights reach the worker through the initial weight sync, the chunks cross the numpy payload boundary, and the real `APPO.evaluate_chunks` re-evaluates them. A mutation check (observations shifted by one slot; initial states zeroed for stateful cores) proves the comparison can fail. The bootstrap test shifts the learner's value head after collection and checks, with λ = 0, that the target of every open ACT moves by exactly `γ·δ` — also where the next slot is a BOOT — and the target of every terminal ACT does not move.

**Files:**
- Modify: `tests/contract/game_harness.py` (imports; append `LearnerView`, `learner_eval`)
- Test: `tests/contract/test_learner_reproduces_worker.py`

**Interfaces:**
- Consumes: `APPO`, `evaluate_chunks` (T4.2); `compute_vtrace_slots` (T4.1); `RolloutLoop` via `game_harness.make_loop`, `GameFactory`, `Collected`, `run_until_chunks`, `kinds`, `lineup` (T3.4); `TickGame`, `Tick` (T3.3); `make_test_model`, `CORE_KINDS` (T2.3); `Units` (T1.2); `WeightPayload`, `SLOT_*` (T3.1); `ComposedModel.value` submodule (T2.3).
- Produces: test kit `LearnerView`, `learner_eval(model, chunks, role, **algo_config) -> LearnerView` (in `game_harness`).

- [ ] **Step 1: Extend the harness**

Add to the import block of `tests/contract/game_harness.py`:

```python
import torch

from colosseum.sp2.algorithms.appo import APPO
from colosseum.sp2.core.config import AlgorithmConfig
from colosseum.sp2.core.specs import ActionSpec
```

and append:

```python
@dataclass
class LearnerView:
    """Learner re-evaluation of chunks next to what the worker recorded (all time-major ``[S*B]``)."""

    log_probs: torch.Tensor
    values: torch.Tensor
    unit_log_probs: torch.Tensor | None
    worker_log_probs: torch.Tensor
    worker_unit_log_probs: torch.Tensor | None
    is_act: torch.Tensor


def learner_eval(model: Any, chunks: list[TrajectoryChunk], role: Any, **algo_config: Any) -> LearnerView:
    """Re-evaluate payload round-tripped ``chunks`` with a real APPO around ``model``."""
    chunks = [TrajectoryChunk.from_payload(c.to_payload()) for c in chunks]
    algo = APPO(model, AlgorithmConfig(**algo_config), ActionSpec.from_space(role.action_space), device="cpu")
    lp, values, unit = algo.evaluate_chunks(chunks)
    S, B = chunks[0].num_slots, len(chunks)
    worker_unit = None
    if chunks[0].behavior_unit_logp is not None:
        worker_unit = torch.stack([c.behavior_unit_logp for c in chunks], dim=1).reshape(S * B, -1)
    return LearnerView(
        log_probs=lp, values=values, unit_log_probs=unit,
        worker_log_probs=torch.stack([c.behavior_logp for c in chunks], dim=1).reshape(-1),
        worker_unit_log_probs=worker_unit,
        is_act=torch.stack([c.kind for c in chunks], dim=1).reshape(-1) == SLOT_ACT,
    )
```

- [ ] **Step 2: Write the contract tests**

`tests/contract/test_learner_reproduces_worker.py`:

```python
"""Contract: the learner reproduces the worker's joint and per-unit log-probs at zero lag on
chunks with BOOT, PAD, truncation and elimination, and bootstraps with its own values (T4.3).

Chunks come from the real RolloutLoop + MatchRunner (in-process), cross the numpy payload
boundary, and are re-evaluated by the real APPO (``unroll`` from the chunks' initial states
with their ``reset_after`` flags).
"""

from __future__ import annotations

import copy

import gymnasium
import numpy as np
import pytest
import torch
import torch.nn as nn

from colosseum.networks.state import tree_leaves
from colosseum.sp2.algorithms.vtrace import compute_vtrace_slots
from colosseum.sp2.core.types import SLOT_ACT, SLOT_BOOT, WeightPayload
from colosseum.sp2.envs.spaces import Units
from colosseum.sp2.networks.model import PolicyModel
from game_harness import Collected, GameFactory, kinds, learner_eval, lineup, make_loop, run_until_chunks
from game_helpers import CORE_KINDS, Tick, TickGame, make_test_model

TOL = 1e-5

# 3 seats: seat 2 is eliminated (with a reward in that step), seat 1 becomes a dead
# teammate, then the episode is truncated.
ELIMINATION_THEN_TRUNCATION = [
    Tick(acting={0, 1, 2}),
    Tick(acting={0, 1, 2}, rewards={0: 0.5}),
    Tick(acting={0, 1}, rewards={2: -1.0}, terminated={2}),
    Tick(acting={0, 1}),
    Tick(acting={0}, rewards={1: 0.25}),
    Tick(acting={0}),
    Tick(over=True, truncated=True, rewards={0: 1.0, 1: 1.0}),
]
# 3 seats take turns, the episode ends by the rules.
TURNS = [
    Tick(acting={0}), Tick(acting={1}), Tick(acting={2}), Tick(acting={0}, rewards={1: 0.5}),
    Tick(acting={1}), Tick(over=True, rewards={0: 1.0, 1: -1.0, 2: 0.0}),
]
UNITS_SPACE = Units(3, gymnasium.spaces.Discrete(3))


def _unit_mask(k: int, t: int, seat: int) -> dict:
    unit = np.array([(t + seat + i + k) % 3 != 0 for i in range(3)])
    action = np.ones((3, 3), bool)
    action[:, 2] = (t % 2 == 0)
    return {"unit": unit, "action": action}


def _collect(core: str, *, units: bool, n_chunks: int = 24, chunk_length: int = 5, obs_dtype=np.float32):
    game_kwargs = dict(action_space=UNITS_SPACE, mask_fn=_unit_mask) if units else {}
    role = TickGame(TURNS, 3, obs_dtype=obs_dtype, **game_kwargs).spec.roles["player"]
    torch.manual_seed(0)
    learner_model = make_test_model(role, core=core)
    col = Collected(weights={"a": [WeightPayload.from_model("a", 0, learner_model)]})
    factory = GameFactory((ELIMINATION_THEN_TRUNCATION, 3), (TURNS, 3), obs_dtype=obs_dtype, **game_kwargs)
    loop, col = make_loop(factory, {"a": lambda: make_test_model(role, core=core)},
                          [lineup("3p", "a", "a", "a"), lineup("3p", "a", "a", "a")],
                          chunk_length=chunk_length, collected=col)
    chunks = run_until_chunks(loop, col, n_chunks)
    loop.close()
    return learner_model, role, chunks


def _max_diff(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    return float((a[mask] - b[mask]).abs().max())


@pytest.mark.parametrize("core", CORE_KINDS)
@pytest.mark.parametrize("units", [False, True], ids=["discrete", "units"])
def test_learner_reproduces_worker_log_probs_at_zero_lag(core, units):
    model, role, chunks = _collect(core, units=units)
    letters = "".join(kinds(c) for c in chunks)
    for needed in "TBRP":
        assert needed in letters, f"no {needed!r} slot in {[kinds(c) for c in chunks]}"
    if core != "none":
        assert any(any(float(x.abs().sum()) > 0 for x in tree_leaves(c.initial_state)) for c in chunks), \
            "no chunk starts mid-episode with a non-zero state"
    view = learner_eval(model, chunks, role)
    assert _max_diff(view.log_probs, view.worker_log_probs, view.is_act) < TOL
    if units:
        act = view.is_act
        assert view.unit_log_probs.shape[1] == 3
        assert _max_diff(view.unit_log_probs, view.worker_unit_log_probs, act) < TOL
        assert any(not bool(c.action_masks["unit"][c.kind == SLOT_ACT].all()) for c in chunks)
    else:
        assert view.unit_log_probs is None and view.worker_unit_log_probs is None


@pytest.mark.parametrize("core", CORE_KINDS)
def test_mutated_chunks_are_detected(core):
    model, role, chunks = _collect(core, units=False)
    shifted = []
    for c in chunks:
        bad = copy.deepcopy(c)
        bad.obs = torch.roll(bad.obs, shifts=1, dims=0)
        shifted.append(bad)
    view = learner_eval(model, shifted, role)
    assert _max_diff(view.log_probs, view.worker_log_probs, view.is_act) > 1e-3
    if core != "none":
        zeroed = []
        for c in chunks:
            bad = copy.deepcopy(c)
            bad.initial_state = model.initial_state(1)
            zeroed.append(bad)
        view = learner_eval(model, zeroed, role)
        assert _max_diff(view.log_probs, view.worker_log_probs, view.is_act) > 1e-4


def _value_bias(model: PolicyModel) -> torch.Tensor:
    return [m for m in model.value.modules() if isinstance(m, nn.Linear)][-1].bias


def test_bootstrap_uses_the_learners_current_values():
    """With lambda = 0 the target of an ACT is ``r + gamma * V(next slot)`` (0 after a terminal
    ACT). Shifting the learner's value head by delta after collection must shift the target of
    every open ACT by gamma * delta, also when the next slot is a BOOT: the bootstrap is the
    learner's current V(boot), never a number recorded by the worker."""
    model, role, chunks = _collect("lstm", units=False)
    gamma, delta = 0.9, 0.75
    S, B = chunks[0].num_slots, len(chunks)
    kind = torch.stack([c.kind for c in chunks], dim=1)
    terminal = torch.stack([c.terminal for c in chunks], dim=1)
    rewards = torch.stack([c.reward for c in chunks], dim=1)
    is_act = kind == SLOT_ACT

    def targets(m):
        values = learner_eval(m, chunks, role).values.reshape(S, B)
        return compute_vtrace_slots(log_rhos=torch.zeros(S, B), rewards=rewards, values=values, is_act=is_act,
                                    terminal=terminal, gamma=gamma, lam=0.0).vs

    before = targets(model)
    shifted = copy.deepcopy(model)
    with torch.no_grad():
        _value_bias(shifted).add_(delta)
    change = targets(shifted) - before
    next_is_boot = torch.zeros_like(is_act)
    next_is_boot[:-1] = kind[1:] == SLOT_BOOT
    open_act, closed = is_act & ~terminal, is_act & terminal
    assert bool((open_act & next_is_boot).any()) and bool(closed.any())
    assert torch.allclose(change[open_act], torch.full_like(change[open_act], gamma * delta), atol=1e-5)
    assert torch.allclose(change[closed], torch.zeros_like(change[closed]), atol=1e-5)


class DtypeSpy(PolicyModel):
    """Delegates to ``inner``; records the observation dtype seen by ``step`` and ``unroll``."""

    def __init__(self, inner: PolicyModel) -> None:
        super().__init__()
        self.inner = inner
        self.seen: list[tuple[str, torch.dtype]] = []

    def initial_state(self, batch_size, device="cpu"):
        return self.inner.initial_state(batch_size, device)

    def step(self, obs, state, action_mask=None):
        self.seen.append(("step", obs.dtype))
        return self.inner.step(obs, state, action_mask)

    def unroll(self, obs, state0, reset_after, action_mask=None, global_state=None, with_value=True):
        self.seen.append(("unroll", obs.dtype))
        return self.inner.unroll(obs, state0, reset_after, action_mask, global_state, with_value)

    def reset_state(self, state, done):
        return self.inner.reset_state(state, done)


def test_uint8_observations_reach_the_model_on_both_sides():
    role = TickGame(TURNS, 3, obs_dtype=np.uint8).spec.roles["player"]
    worker_model = DtypeSpy(make_test_model(role))
    loop, col = make_loop(GameFactory((TURNS, 3), obs_dtype=np.uint8), {"a": lambda: worker_model},
                          [lineup("3p", "a", "a", "a")], chunk_length=4)
    chunks = run_until_chunks(loop, col, 3)
    assert {d for _, d in worker_model.seen} == {torch.uint8}
    assert all(c.obs.dtype == torch.uint8 for c in chunks)
    learner_model = DtypeSpy(copy.deepcopy(worker_model.inner))
    learner_eval(learner_model, chunks, role)
    assert ("unroll", torch.uint8) in learner_model.seen
```

- [ ] **Step 3: Run them**

Run: `.venv/bin/python -m pytest tests/contract/test_learner_reproduces_worker.py -q`
Expected: `14 passed`. These tests are written after the code they check (T3.4, T4.2), so they should pass at once; if one fails, the bug is in the worker or the learner — fix it there in this task (never relax the tolerance `1e-5` or a precondition) and say so in the commit body.

- [ ] **Step 4: Full fast suite and lint** (as in T3.1, Step 5).

- [ ] **Step 5: Commit**

```bash
git add tests/contract/game_harness.py tests/contract/test_learner_reproduces_worker.py
git commit -m "test(sp2): learner reproduces worker joint/per-unit log-probs on boot/pad/truncation/elimination chunks"
git push origin sp2-game-model
```

---

### Task T4.4: Learner process on chunk v2

Copy SP1's `learner/learner.py` to `sp2` with the minimal changes: `sp2` imports, `consumed_samples += sum(c.num_acts for c in chunks)` (ACT slots), policy lag from `c.policy_version`, and the docstrings that mention them. Everything else — exact `batch_chunks` batches, progress from the shared counter or `consumed_samples / total_timesteps`, newest-wins weight pushes, checkpoint payloads (numpy weights + trainer-state bytes), final checkpoint within `FINAL_CHECKPOINT_TIMEOUT_SEC = SHUTDOWN_GRACE_SEC - 2`, `_release_weight_queues` exit rules — stays SP1 behavior. The tests port SP1's learner tests (`tests/unit/test_collect_batch.py`, `test_policy_lag.py`, `test_budget.py` learner part, `test_checkpoint_store.py` learner part, `tests/contract/test_no_tensors_in_queues.py` learner part, `tests/integration/test_learner_process_exit.py`) to chunk v2 payloads under new basenames.

**Files:**
- Create: `src/colosseum/sp2/learner/__init__.py` (empty), `src/colosseum/sp2/learner/learner.py`
- Modify: `tests/game_helpers.py` (imports; append the T4.4 block)
- Test: `tests/unit/test_learner_v2.py`, `tests/integration/test_learner_v2_exit.py`

**Interfaces:**
- Consumes: `BaseAlgorithm` (T4.2), `APPO` (T4.2, tests), `LearnerConfig` (T1.7), `TrajectoryChunk.from_payload`, `num_acts`, `policy_version`, `WeightPayload`, `state_dict_*` (T3.1); `SharedCounter`, `put_latest`, `assert_no_tensors` (`colosseum.core.ipc`); `SHUTDOWN_GRACE_SEC`, `flush_queue`, `parent_alive` (`colosseum.utils.process`); test kit `NumpyOnlyQueue` (T3.4), `make_test_model` (T2.3).
- Produces (contract): `learner_process(*, agent_id, algorithm_factory, trajectory_queue, weight_queues, config, stop_event, metrics_queue=None, checkpoint_queue=None, checkpoint_interval=0, resume_state=None, progress_counter=None, total_timesteps=0, weight_sync_interval=5.0)`, `collect_batch`, `make_checkpoint_payload`, `send_checkpoint`, `apply_resume_state`, `resolve_device`, `FINAL_CHECKPOINT_TIMEOUT_SEC`.
  - Test kit: `learner_role(num_actions=3)`, `chunk_v2_payload(S=4, version=0, agent_id="a", *, pattern=None, num_actions=3)`, `learner_appo(num_actions=3, **algo_config)`, `RecordingLearnerAlgorithm(start_version=0)`.

- [ ] **Step 1: Test kit**

Add to the import block of `tests/game_helpers.py` (skip what is already imported):

```python
import math
from typing import Any

import gymnasium
import numpy as np

from colosseum.sp2.algorithms.appo import APPO
from colosseum.sp2.core.config import AlgorithmConfig
from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.types import SLOT_ACT, SLOT_BOOT, SLOT_PAD
from colosseum.sp2.envs.game import RoleSpec
```

and append:

```python
# ---------------------------------------------------------------------------
# Part B (T4.4): learner-loop test kit
# ---------------------------------------------------------------------------


LEARNER_OBS_DIM = 4
_KIND_OF = {"A": SLOT_ACT, "T": SLOT_ACT, "B": SLOT_BOOT, "R": SLOT_BOOT, "P": SLOT_PAD}


def learner_role(num_actions: int = 3) -> RoleSpec:
    """Box(4) observations, Discrete(num_actions) actions: the learner-loop tests' role."""
    return RoleSpec(gymnasium.spaces.Box(-1e9, 1e9, (LEARNER_OBS_DIM,), np.float32),
                    gymnasium.spaces.Discrete(num_actions))


def chunk_v2_payload(S: int = 4, version: int = 0, agent_id: str = "a", *, pattern: str | None = None,
                     num_actions: int = 3) -> dict:
    """A valid chunk v2 payload for ``learner_role(num_actions)`` with random content.

    ``pattern`` spells the slots like ``synthetic_chunk`` (default: ``S - 1`` ACTs and a BOOT).
    """
    rng = np.random.default_rng(version)
    pattern = pattern if pattern is not None else "A" * (S - 1) + "B"
    S = len(pattern)
    acts = np.array([c in "AT" for c in pattern])
    return {
        "agent_id": agent_id,
        "policy_version": int(version),
        "initial_state": None,
        "obs": rng.standard_normal((S, LEARNER_OBS_DIM)).astype(np.float32),
        "global_state": None,
        "actions": np.where(acts, rng.integers(0, num_actions, S), 0).astype(np.int64),
        "action_masks": np.ones((S, num_actions), np.bool_),
        "kind": np.array([_KIND_OF[c] for c in pattern], np.int8),
        "reward": np.where(acts, rng.standard_normal(S), 0.0).astype(np.float32),
        "terminal": np.array([c == "T" for c in pattern]),
        "reset_after": np.array([c in "TRP" for c in pattern]),
        "behavior_logp": np.where(acts, -math.log(num_actions), 0.0).astype(np.float32),
        "behavior_unit_logp": None,
    }


def learner_appo(num_actions: int = 3, **algo_config: Any) -> APPO:
    """A real APPO (CPU) around ``make_test_model(learner_role(num_actions))``."""
    role = learner_role(num_actions)
    return APPO(make_test_model(role), AlgorithmConfig(**algo_config), ActionSpec.from_space(role.action_space),
                device="cpu")


class RecordingLearnerAlgorithm:
    """Duck-typed algorithm for learner-loop tests: records what ``train_step`` was given."""

    def __init__(self, start_version: int = 0) -> None:
        self._model = make_test_model(learner_role())
        self._policy_version = start_version
        self._progress = 0.0
        self.batches: list[int] = []
        self.behavior_versions: list[list[int]] = []
        self.progress_at_train: list[float] = []

    @property
    def model(self):
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

    def train_step(self, chunks) -> dict[str, float]:
        self.batches.append(len(chunks))
        self.behavior_versions.append([c.policy_version for c in chunks])
        self.progress_at_train.append(self._progress)
        self._policy_version += 1
        return {"total_loss": 0.0}
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_learner_v2.py`:

```python
"""SP2 learner process on chunk v2 payloads (T4.4; ported from SP1's learner tests)."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest
import torch

from colosseum.core.ipc import SharedCounter, assert_no_tensors
from colosseum.sp2.core.config import LearnerConfig
from colosseum.sp2.core.types import TrajectoryChunk, WeightPayload
from colosseum.sp2.learner.learner import (
    FINAL_CHECKPOINT_TIMEOUT_SEC,
    _weight_flush_timeout,
    collect_batch,
    learner_process,
    make_checkpoint_payload,
    resolve_device,
)
from game_helpers import NumpyOnlyQueue, RecordingLearnerAlgorithm, chunk_v2_payload, learner_appo


def _collect_in_thread(q, batch_size, stop):
    out = {}
    thread = threading.Thread(
        target=lambda: out.setdefault("batch", collect_batch(q, batch_size, stop, poll_interval=0.05))
    )
    thread.start()
    return thread, out


def _run_learner(algo, payloads, until, *, batch_chunks, timeout=10.0, **kwargs):
    traj, stop = NumpyOnlyQueue(), threading.Event()
    for payload in payloads:
        traj.put(payload)
    kwargs.setdefault("weight_queues", [NumpyOnlyQueue(maxsize=1)])
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, trajectory_queue=traj,
        config=LearnerConfig(batch_chunks=batch_chunks, device="cpu"), stop_event=stop, **kwargs,
    ))
    thread.start()
    deadline = time.monotonic() + timeout
    while not until() and time.monotonic() < deadline and thread.is_alive():
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_collect_batch_blocks_until_full_and_decodes_chunk_v2():
    q, stop = queue.Queue(), threading.Event()
    for version in range(2):
        q.put(chunk_v2_payload(version=version))
    thread, out = _collect_in_thread(q, 3, stop)
    time.sleep(0.3)
    assert thread.is_alive()                       # 2 of 3 chunks: still waiting
    q.put(chunk_v2_payload(version=2, pattern="ARAR"))
    thread.join(timeout=5)
    batch = out["batch"]
    assert [c.policy_version for c in batch] == [0, 1, 2]
    assert all(isinstance(c, TrajectoryChunk) for c in batch)
    assert [c.num_acts for c in batch] == [3, 3, 2]


def test_collect_batch_returns_none_on_stop_and_rejects_non_payloads():
    q, stop = queue.Queue(), threading.Event()
    q.put(chunk_v2_payload())
    thread, out = _collect_in_thread(q, 3, stop)
    time.sleep(0.2)
    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive() and out["batch"] is None
    bad = queue.Queue()
    bad.put(object())
    with pytest.raises(TypeError):
        collect_batch(bad, 1, threading.Event(), poll_interval=0.05)


def test_learner_trains_only_on_full_batches():
    algo = RecordingLearnerAlgorithm()
    _run_learner(algo, [chunk_v2_payload(version=v) for v in range(7)], lambda: len(algo.batches) >= 2,
                 batch_chunks=3)
    time.sleep(0.1)
    assert algo.batches == [3, 3]


def test_learner_reports_policy_lag_from_chunk_versions():
    algo, metrics_q = RecordingLearnerAlgorithm(start_version=10), NumpyOnlyQueue()
    _run_learner(algo, [chunk_v2_payload(version=v) for v in (7, 10, 9, 10, 10, 8)],
                 lambda: len(algo.batches) >= 2, batch_chunks=3, metrics_queue=metrics_q)
    metrics = [metrics_q.get_nowait() for _ in range(metrics_q.qsize())]
    assert metrics[0]["policy_lag_mean"] == pytest.approx((3 + 0 + 1) / 3)
    assert metrics[0]["policy_lag_max"] == 3.0
    assert metrics[1]["policy_lag_mean"] == pytest.approx((1 + 1 + 3) / 3)


@pytest.mark.parametrize("counted,expected", [(50, 0.5), (300, 1.0)])
def test_progress_comes_from_the_shared_counter_and_is_capped(counted, expected):
    algo, counter = RecordingLearnerAlgorithm(), SharedCounter()
    counter.add(counted)
    _run_learner(algo, [chunk_v2_payload()], lambda: len(algo.batches) >= 1, batch_chunks=1,
                 progress_counter=counter, total_timesteps=100)
    assert algo.progress_at_train == [expected]


def test_without_a_counter_progress_counts_act_slots_and_the_learner_stops_itself():
    algo, traj = RecordingLearnerAlgorithm(), NumpyOnlyQueue()
    for _ in range(6):
        traj.put(chunk_v2_payload(S=4))           # "AAAB": 3 ACT slots per chunk
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, trajectory_queue=traj,
        weight_queues=[NumpyOnlyQueue(maxsize=1)], config=LearnerConfig(batch_chunks=2, device="cpu"),
        stop_event=threading.Event(), total_timesteps=12,
    ))
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()                  # stopped by itself at 12 consumed ACT slots
    assert algo.progress_at_train == [0.5, 1.0]


def test_learner_queues_carry_only_numpy_with_a_real_appo():
    wq, mq, ckq = NumpyOnlyQueue(maxsize=1), NumpyOnlyQueue(), NumpyOnlyQueue()
    algo = learner_appo()
    _run_learner(algo, [chunk_v2_payload(version=v, pattern="ARTP") for v in range(4)],
                 lambda: ckq.qsize() >= 2, batch_chunks=2, timeout=60, weight_queues=[wq], metrics_queue=mq,
                 checkpoint_queue=ckq, checkpoint_interval=1)
    ckpt = ckq.get_nowait()
    assert all(isinstance(v, np.ndarray) for v in ckpt["model_state"].values())
    assert isinstance(ckpt["trainer_state_bytes"], bytes)
    assert isinstance(wq.get_nowait(), WeightPayload)
    metrics = mq.get_nowait()
    assert metrics["consumed_samples"] == 4 and "ess" in metrics and "pad_frac" in metrics


def test_final_checkpoint_is_sent_on_stop():
    stop, ckq = threading.Event(), NumpyOnlyQueue()
    stop.set()
    learner_process(
        agent_id="a", algorithm_factory=learner_appo, trajectory_queue=NumpyOnlyQueue(),
        weight_queues=[NumpyOnlyQueue(maxsize=1)], config=LearnerConfig(batch_chunks=2, device="cpu"),
        stop_event=stop, checkpoint_queue=ckq, checkpoint_interval=100,
    )
    final = ckq.get_nowait()
    assert final["final"] is True and final["agent_id"] == "a" and final["policy_version"] == 0
    assert ckq.empty()
    assert FINAL_CHECKPOINT_TIMEOUT_SEC == pytest.approx(5.0)


def _resume_state(source) -> dict:
    payload = make_checkpoint_payload("a", source)
    state = {"model_state": payload["model_state"], "trainer_state": payload["trainer_state_bytes"],
             "policy_version": payload["policy_version"], "env_steps": 0, "source": "test"}
    assert_no_tensors(state)
    return state


def test_resume_continues_train_step_consumed_samples_and_lr_progress():
    source = learner_appo(lr_schedule="linear")
    for step in range(2):
        source.train_step([TrajectoryChunk.from_payload(chunk_v2_payload(version=step + v)) for v in range(2)])
    assert source.policy_version == 2 and source.consumed_samples == 12
    built = []

    def factory():
        built.append(learner_appo(lr_schedule="linear"))
        return built[-1]

    counter, mq = SharedCounter(), NumpyOnlyQueue()
    counter.add(400)
    traj = NumpyOnlyQueue()
    for version in range(2):
        traj.put(chunk_v2_payload(version=2 + version))
    stop = threading.Event()
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=factory, trajectory_queue=traj, weight_queues=[NumpyOnlyQueue(maxsize=1)],
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"), stop_event=stop,
        metrics_queue=mq, resume_state=_resume_state(source), progress_counter=counter, total_timesteps=1000,
    ))
    thread.start()
    deadline = time.monotonic() + 60
    while mq.qsize() < 1 and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=10)
    metrics = mq.get_nowait()
    assert metrics["train_step"] == 3 and metrics["consumed_samples"] == 18
    assert built[0].policy_version == 3 and built[0].consumed_samples == 18
    assert metrics["progress"] == pytest.approx(0.4)
    for key, value in source.model.state_dict().items():
        assert value.shape == built[0].model.state_dict()[key].shape


def test_resume_pushes_the_restored_weights():
    source = learner_appo()
    source.train_step([TrajectoryChunk.from_payload(chunk_v2_payload(version=v)) for v in range(2)])
    state = _resume_state(source)
    built, wq, stop = [], NumpyOnlyQueue(maxsize=1), threading.Event()
    stop.set()
    learner_process(
        agent_id="a", algorithm_factory=lambda: built.append(learner_appo()) or built[-1],
        trajectory_queue=NumpyOnlyQueue(), weight_queues=[wq], config=LearnerConfig(batch_chunks=2, device="cpu"),
        stop_event=stop, resume_state=state,
    )
    assert built[0].policy_version == 1
    for key, value in source.model.state_dict().items():
        assert torch.equal(built[0].model.state_dict()[key], value), key
    pushed = wq.get_nowait()
    assert pushed.policy_version == 1
    for key, value in state["model_state"].items():
        assert np.array_equal(pushed.state_dict[key], value), key


def test_weight_flush_timeout_and_device_resolution(monkeypatch):
    assert _weight_flush_timeout(5.0) == 60.0 and _weight_flush_timeout(90.0) == 270.0
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu" and resolve_device("cuda:1") == "cuda:1"
```

`tests/integration/test_learner_v2_exit.py` (spawns one learner process per test; the factory must be a module-level function so spawn can pickle it):

```python
"""SP2 learner exit vs. weight queues holding large numpy payloads (T4.4; port of SP1's T2.2 tests).

Numpy weight payloads are pickled in full, so a payload larger than the pipe buffer keeps
the queue's feeder thread blocked until a worker reads it:
- workers stopped (``stop_event`` set): the learner must not wait for them at exit;
- learner stopped by itself (``consumed_samples`` budget) while workers still run: the
  feeder must flush the whole message, or a live worker blocks forever on a truncated one.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time

from colosseum.sp2.core.config import LearnerConfig
from colosseum.sp2.learner.learner import learner_process
from game_helpers import chunk_v2_payload, learner_appo

BIG_NUM_ACTIONS = 8192      # policy head 16 x 8192 floats: ~0.5 MB per weight payload
WEIGHT_QUEUE_SIZE = 1       # newest-wins mailbox per (agent, worker), as the launcher uses
ONE_BATCH = 6               # total_timesteps: one batch of 2 chunks x 3 ACT slots, then the learner stops


def _big_appo():
    return learner_appo(num_actions=BIG_NUM_ACTIONS)


def _payload(version: int = 0) -> dict:
    return chunk_v2_payload(S=4, version=version, num_actions=BIG_NUM_ACTIONS)


def _start_learner(ctx, weights, stop, total_timesteps: int):
    traj = ctx.Queue()
    for version in range(2):
        traj.put(_payload(version))
    proc = ctx.Process(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=_big_appo, trajectory_queue=traj, weight_queues=[weights],
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"),
        stop_event=stop, total_timesteps=total_timesteps,
    ))
    proc.start()
    return proc, traj


def _cleanup(proc, *queues) -> None:
    if proc.is_alive():
        proc.kill()
        proc.join()
    for q in queues:
        q.cancel_join_thread()


def test_learner_exits_when_workers_stopped_and_weights_are_unread():
    ctx = mp.get_context("spawn")
    unread = ctx.Queue(maxsize=WEIGHT_QUEUE_SIZE)
    stop = ctx.Event()
    stop.set()
    proc, traj = _start_learner(ctx, unread, stop, total_timesteps=ONE_BATCH)
    try:
        proc.join(timeout=60)
        assert not proc.is_alive(), "learner hung at exit flushing an unread weight payload"
        assert proc.exitcode == 0
    finally:
        _cleanup(proc, unread, traj)


def test_live_slow_reader_gets_final_weights_after_learner_budget_exit():
    ctx = mp.get_context("spawn")
    weights = ctx.Queue(maxsize=WEIGHT_QUEUE_SIZE)
    stop = ctx.Event()
    received: list[int] = []

    def slow_reader() -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and 1 not in received:
            try:
                received.append(weights.get(timeout=1.0).policy_version)
            except Exception:  # queue.Empty
                continue
            time.sleep(1.0)

    proc, traj = _start_learner(ctx, weights, stop, total_timesteps=ONE_BATCH)
    reader = threading.Thread(target=slow_reader, daemon=True)
    reader.start()
    try:
        reader.join(timeout=90)
        assert not reader.is_alive(), f"reader blocked on a truncated weight payload (got {received})"
        assert received[-1] == 1, f"final weights (version 1) never arrived: {received}"
        proc.join(timeout=30)
        assert proc.exitcode == 0
    finally:
        _cleanup(proc, weights, traj)


def test_learner_budget_exit_does_not_deadlock_with_a_worker_blocked_on_chunks():
    ctx = mp.get_context("spawn")
    traj = ctx.Queue(maxsize=2)
    weights = ctx.Queue(maxsize=WEIGHT_QUEUE_SIZE)
    stop = ctx.Event()
    proc = ctx.Process(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=_big_appo, trajectory_queue=traj, weight_queues=[weights],
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"),
        stop_event=stop, total_timesteps=ONE_BATCH,
    ))
    received: list[int] = []
    worker_done = threading.Event()

    def worker() -> None:
        while not worker_done.is_set():
            try:
                traj.put(_payload(), timeout=0.5)
            except Exception:  # queue.Full: keep retrying, like send_chunk
                continue
            while True:
                try:
                    received.append(weights.get_nowait().policy_version)
                except Exception:  # queue.Empty
                    break

    proc.start()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        proc.join(timeout=60)
        assert not proc.is_alive(), f"learner and worker deadlocked (worker got versions {received})"
        assert proc.exitcode == 0
    finally:
        worker_done.set()
        thread.join(timeout=10)
        _cleanup(proc, weights, traj)
```

- [ ] **Step 3: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_learner_v2.py tests/integration/test_learner_v2_exit.py -q`
Expected: `ModuleNotFoundError: No module named 'colosseum.sp2.learner'`.

- [ ] **Step 4: Copy and adapt the learner**

Create the empty `src/colosseum/sp2/learner/__init__.py`. Copy `src/colosseum/learner/learner.py` to `src/colosseum/sp2/learner/learner.py` and make exactly these edits.

Module docstring, the distributed-progress sentence:

```python
counter: ``progress = consumed_samples / total_timesteps``, where
``consumed_samples`` counts this learner's ACT slots (decisions, not env steps;
one budget semantics for distributed mode is SP5), and the learner stops by
itself once ``consumed_samples >= total_timesteps``.
```

Imports (replace the three `colosseum.algorithms` / `colosseum.core.config` / `colosseum.core.types` lines; keep `colosseum.core.ipc` and `colosseum.utils.process`):

```python
from colosseum.core.ipc import SharedCounter, put_latest
from colosseum.sp2.algorithms.base import BaseAlgorithm
from colosseum.sp2.core.config import LearnerConfig
from colosseum.sp2.core.types import TrajectoryChunk, WeightPayload, state_dict_from_numpy, state_dict_to_numpy
from colosseum.utils.process import SHUTDOWN_GRACE_SEC, flush_queue, parent_alive
```

In `learner_process`, the batch bookkeeping:

```python
            consumed_samples += sum(c.num_acts for c in chunks)
            # Policy lag of the batch, measured before this train step bumps the version.
            lags = [algorithm.policy_version - c.policy_version for c in chunks]
```

In `collect_batch`, the first docstring line:

```python
    """Block until exactly ``batch_size`` chunk v2 payloads arrived; decode them.
```

The resulting file (for reference; it must match):

```python
"""Learner process: receives trajectory chunks, trains the model, pushes weights.

Each learner owns one trainable agent. It:
1. collects exactly ``batch_chunks`` chunk payloads (numpy,
   ``TrajectoryChunk.to_payload()``) from its queue and decodes them;
2. sets the algorithm's progress (share of the env-step budget) and trains;
3. publishes new weights (numpy ``WeightPayload``) to every worker's size-1
   mailbox (newest wins);
4. sends checkpoint payloads (numpy weights + trainer-state bytes, see
   ``make_checkpoint_payload``) and metrics to the main process, and a final
   snapshot when it stops.

Nothing this process puts on a queue contains a torch tensor (R6-02).

Progress and stopping: in local mode ``progress = min(1, env_step_counter /
total_timesteps)`` from the shared counter that workers increment; the main
process is the only budget authority and sets ``stop_event`` at the budget, so
the learner runs until ``stop_event``. In distributed mode there is no shared
counter: ``progress = consumed_samples / total_timesteps``, where
``consumed_samples`` counts this learner's ACT slots (decisions, not env steps;
one budget semantics for distributed mode is SP5), and the learner stops by
itself once ``consumed_samples >= total_timesteps``.
"""

from __future__ import annotations

import io
import logging
import multiprocessing as mp
import threading
import time
from collections.abc import Callable
from queue import Empty, Full
from typing import Any

import numpy as np
import torch

from colosseum.core.ipc import SharedCounter, put_latest
from colosseum.sp2.algorithms.base import BaseAlgorithm
from colosseum.sp2.core.config import LearnerConfig
from colosseum.sp2.core.types import TrajectoryChunk, WeightPayload, state_dict_from_numpy, state_dict_to_numpy
from colosseum.utils.process import SHUTDOWN_GRACE_SEC, flush_queue, parent_alive

logger = logging.getLogger(__name__)

# The final snapshot must be on the queue before the main process stops waiting for the
# children (SHUTDOWN_GRACE_SEC after stop_event) and starts terminating them.
FINAL_CHECKPOINT_TIMEOUT_SEC = SHUTDOWN_GRACE_SEC - 2.0


def make_checkpoint_payload(agent_id: str, algorithm: BaseAlgorithm, final: bool = False) -> dict:
    """Checkpoint snapshot that may cross a process boundary: numpy weights + bytes.

    The trainer state (optimizer, LR progress, scaler, kickstart, policy_version,
    consumed_samples) is serialized with ``torch.save`` into bytes, because it
    contains tensors (R6-02).
    """
    buf = io.BytesIO()
    torch.save(algorithm.state_dict(), buf)
    return {
        "agent_id": agent_id,
        "policy_version": int(algorithm.policy_version),
        "model_state": state_dict_to_numpy(algorithm.model.state_dict()),
        "trainer_state_bytes": buf.getvalue(),
        "final": bool(final),
    }


def send_checkpoint(q, payload: dict, block: bool, timeout: float = FINAL_CHECKPOINT_TIMEOUT_SEC) -> bool:
    """Put a checkpoint payload on ``q``. Returns False (and logs) if the queue stays full."""
    try:
        if block:
            q.put(payload, timeout=timeout)
        else:
            q.put_nowait(payload)
        return True
    except Full:
        level = logging.ERROR if block else logging.WARNING
        logger.log(level, f"Checkpoint queue full; dropped snapshot v{payload.get('policy_version')}")
        return False


def apply_resume_state(algorithm: BaseAlgorithm, resume_state: dict) -> None:
    """Load weights and trainer state produced by ``resolve_resume`` into ``algorithm``.

    Without a trainer state, only ``policy_version`` is restored (from meta.json), so
    checkpoint ids keep increasing after a resume.
    """
    model = algorithm.model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.load_state_dict(state_dict_from_numpy(resume_state["model_state"]))
    blob = resume_state.get("trainer_state")
    if blob is not None:
        algorithm.load_state_dict(torch.load(io.BytesIO(blob), map_location=device, weights_only=True))
    else:
        state = algorithm.state_dict()
        state["policy_version"] = int(resume_state.get("policy_version", 0))
        algorithm.load_state_dict(state)
    logger.info(f"Resumed from {resume_state.get('source')} at policy_version {algorithm.policy_version}")


def learner_process(
    *,
    agent_id: str,
    algorithm_factory: Callable[[], BaseAlgorithm],
    trajectory_queue: mp.Queue,
    weight_queues: list[mp.Queue],
    config: LearnerConfig,
    stop_event: mp.Event,
    metrics_queue: mp.Queue | None = None,
    checkpoint_queue: mp.Queue | None = None,
    checkpoint_interval: int = 0,
    resume_state: dict | None = None,
    progress_counter: SharedCounter | None = None,
    total_timesteps: int = 0,
    weight_sync_interval: float = 5.0,
) -> None:
    """Main learner process function (see module docstring).

    Args:
        agent_id: the trainable agent this learner owns
        algorithm_factory: factory to create the algorithm (includes network)
        trajectory_queue: queue to receive chunk payloads (``TrajectoryChunk.to_payload()``)
        weight_queues: list of queues to push weights to workers
        config: learner configuration
        stop_event: set to signal the learner to stop
        metrics_queue: optional queue to send metrics to the main process
        checkpoint_queue: optional queue for checkpoint payloads (``make_checkpoint_payload``)
            to the main process; a final snapshot is always sent when the loop ends
        checkpoint_interval: send a checkpoint every N policy versions (0 = only the final one)
        resume_state: optional ``checkpoint_manager.resolve_resume`` result (numpy
            weights, trainer-state bytes or None, policy_version) to resume from
        progress_counter: the global env-step counter (local mode); None in
            distributed mode, where progress is ``consumed_samples / total_timesteps``
        total_timesteps: ``training.total_timesteps`` (0 = no budget: progress stays 0)
        weight_sync_interval: the workers' ``rollout.weight_sync_interval_sec``; sizes
            the exit wait for pending weight payloads (see ``_weight_flush_timeout``)
    """
    logger.info(f"Learner [{agent_id}]: starting on device={resolve_device(config.device)}")

    algorithm = algorithm_factory()

    # On resume both counters continue: train_step from the restored policy_version,
    # consumed_samples (the distributed budget) from the trainer state.
    consumed_samples = 0
    if resume_state is not None:
        apply_resume_state(algorithm, resume_state)
        consumed_samples = int(algorithm.state_dict().get("consumed_samples", 0))
    train_step = int(algorithm.policy_version)

    total_chunks_received = 0
    last_ckpt_version = -1

    # Off-policy: create replay buffer if algorithm requires it
    replay_buffer = algorithm.create_replay_buffer(config.queue_size * 4)

    try:
        # Push initial weights to workers
        _push_weights(algorithm, agent_id, weight_queues)

        while not stop_event.is_set():
            if (progress_counter is None and total_timesteps > 0
                    and consumed_samples >= total_timesteps):
                logger.info(f"Learner [{agent_id}]: consumed {consumed_samples} samples; budget reached")
                break

            # Block until exactly batch_chunks chunks arrived (None: stop requested).
            chunks = collect_batch(trajectory_queue, config.batch_chunks, stop_event)
            if chunks is None:
                break
            total_chunks_received += len(chunks)
            consumed_samples += sum(c.num_acts for c in chunks)
            # Policy lag of the batch, measured before this train step bumps the version.
            lags = [algorithm.policy_version - c.policy_version for c in chunks]

            progress = _progress(progress_counter, consumed_samples, total_timesteps)
            algorithm.set_progress(progress)

            # Train step: off-policy adds to buffer, on-policy trains directly
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
            metrics["policy_lag_mean"] = float(np.mean(lags))
            metrics["policy_lag_max"] = float(np.max(lags))

            # Push updated weights to all workers at configured interval
            if train_step % config.weight_push_interval == 0:
                _push_weights(algorithm, agent_id, weight_queues)

            if checkpoint_queue is not None and checkpoint_interval > 0:
                pv = algorithm.policy_version
                if pv > 0 and pv % checkpoint_interval == 0 and pv != last_ckpt_version:
                    if send_checkpoint(checkpoint_queue, make_checkpoint_payload(agent_id, algorithm), block=False):
                        last_ckpt_version = pv

            if metrics_queue is not None:
                metrics["agent_id"] = agent_id
                metrics["train_step"] = train_step
                metrics["chunks_received"] = total_chunks_received
                metrics["consumed_samples"] = consumed_samples
                try:
                    metrics_queue.put_nowait(metrics)
                except Full:
                    pass  # Non-critical, don't block on metrics

            if train_step % 10 == 0:
                logger.info(
                    f"Learner [{agent_id}]: step={train_step}, chunks={total_chunks_received}, "
                    f"progress={progress:.3f}, loss={metrics.get('total_loss', 0.0):.4f}"
                )
    finally:
        _release_weight_queues(
            weight_queues, trajectory_queue, stop_event, timeout=_weight_flush_timeout(weight_sync_interval),
        )

    if checkpoint_queue is not None:
        # Final snapshot on every stop. The main process saves it before tearing children down (R3-07).
        if send_checkpoint(checkpoint_queue, make_checkpoint_payload(agent_id, algorithm, final=True),
                           block=True, timeout=FINAL_CHECKPOINT_TIMEOUT_SEC):
            logger.info(f"Learner [{agent_id}]: sent final checkpoint v{algorithm.policy_version}")
        # Wait until the snapshot is flushed into the pipe while the main process reads it.
        # Given up only once the main process is gone (it terminates us after its grace
        # period otherwise), so this process never hangs.
        flush_queue(checkpoint_queue, warn_after=SHUTDOWN_GRACE_SEC)

    logger.info(f"Learner [{agent_id}]: finished. Total train_steps={train_step}")


def _progress(
    progress_counter: SharedCounter | None,
    consumed_samples: int,
    total_timesteps: int,
) -> float:
    """Share of the budget used: the global counter if given, else ``consumed_samples``."""
    if total_timesteps <= 0:
        return 0.0
    done = progress_counter.value if progress_counter is not None else consumed_samples
    return min(1.0, done / total_timesteps)


# Shortest exit wait for workers to read pending weight payloads (see
# _weight_flush_timeout). A worker silent for longer is assumed gone (e.g.
# crashed), so its queue is abandoned.
_MIN_WEIGHT_FLUSH_TIMEOUT_SEC = 60.0


def _weight_flush_timeout(weight_sync_interval: float) -> float:
    """Exit wait for pending weight payloads: a live worker syncs weights every
    ``weight_sync_interval`` seconds (while this learner drains its chunks), so
    wait for several intervals, and never less than a minute."""
    return max(_MIN_WEIGHT_FLUSH_TIMEOUT_SEC, 3.0 * float(weight_sync_interval))


def _release_weight_queues(
    weight_queues: list,
    trajectory_queue,
    stop_event,
    *,
    timeout: float,
    poll: float = 0.1,
) -> None:
    """Let pending weight payloads reach the workers before this process exits.

    Numpy weight payloads are pickled in full, so a payload larger than the pipe
    buffer keeps an ``mp.Queue`` feeder thread blocked until a worker reads it.

    - ``stop_event`` set: workers are stopping and will never read it, so do not
      wait for the feeders (``cancel_join_thread``). This is the normal local
      exit: the main process sets ``stop_event`` at the global budget.
    - Otherwise (the learner raised while workers still run, or it stopped by
      itself at a ``consumed_samples`` budget with ``mp.Queue`` mailboxes; the
      distributed weight sinks have no feeder): wait until every feeder has flushed.
      Cancelling would kill a feeder mid-message, and a live worker would then
      block forever on the truncated payload. While waiting, keep discarding
      chunks: a worker blocked on this learner's full trajectory queue only
      syncs weights once its put succeeds. The wait ends early if
      ``stop_event`` gets set, and after ``timeout`` seconds the remaining
      feeders are abandoned (a reader silent that long is assumed dead).
    """
    mp_queues = [wq for wq in weight_queues if hasattr(wq, "join_thread")]
    if not stop_event.is_set():
        joiners = []
        for wq in mp_queues:
            wq.close()  # no more puts; join_thread() waits for the feeder to flush
            joiner = threading.Thread(target=wq.join_thread, daemon=True)
            joiner.start()
            joiners.append(joiner)
        deadline = time.monotonic() + timeout
        while any(j.is_alive() for j in joiners) and not stop_event.is_set():
            if not parent_alive():
                logger.warning("Learner: main process is gone; abandoning unread weight payloads")
                break
            if time.monotonic() > deadline:
                logger.warning(f"Learner: weight payloads unread after {timeout:.0f} s; abandoning them")
                break
            try:
                while True:
                    trajectory_queue.get_nowait()
            except Empty:
                pass
            stop_event.wait(poll)
    for wq in mp_queues:
        wq.cancel_join_thread()  # no-op for the feeders that already flushed


def resolve_device(device_str: str) -> str:
    """Resolve ``learner.device``: ``"auto"`` -> ``"cuda"`` if available else ``"cpu"``; others unchanged."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def collect_batch(
    q: Any,
    batch_size: int,
    stop_event: Any,
    poll_interval: float = 0.5,
) -> list[TrajectoryChunk] | None:
    """Block until exactly ``batch_size`` chunk v2 payloads arrived; decode them.

    Waits in ``poll_interval`` slices and returns None as soon as ``stop_event``
    is set (a partial batch is dropped: no training on incomplete batches).
    """
    chunks: list[TrajectoryChunk] = []
    while len(chunks) < batch_size:
        if stop_event.is_set():
            return None
        try:
            payload = q.get(timeout=poll_interval)
        except Empty:
            continue
        if not isinstance(payload, dict):
            raise TypeError(
                f"trajectory queue item must be a chunk payload dict, got {type(payload).__name__}"
            )
        chunks.append(TrajectoryChunk.from_payload(payload))
    return chunks


def _push_weights(
    algorithm: BaseAlgorithm,
    agent_id: str,
    weight_queues: list,
) -> None:
    """Publish the current weights to every worker mailbox (newest wins)."""
    payload = WeightPayload.from_model(agent_id, algorithm.policy_version, algorithm.model)
    for wq in weight_queues:
        if not put_latest(wq, payload):
            logger.debug(f"Learner [{agent_id}]: weight mailbox busy, v{payload.policy_version} not delivered")
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_learner_v2.py tests/integration/test_learner_v2_exit.py -q`
Expected: `15 passed` (the integration file takes ~15–30 s: three spawned learners).

- [ ] **Step 6: Full fast suite and lint** (as in T3.1, Step 5).

- [ ] **Step 7: Commit**

```bash
git add src/colosseum/sp2/learner/__init__.py src/colosseum/sp2/learner/learner.py tests/unit/test_learner_v2.py \
    tests/integration/test_learner_v2_exit.py tests/game_helpers.py
git commit -m "feat(sp2): learner process on chunk v2 (consumed_samples counts ACT slots)"
git push origin sp2-game-model
```

---

## Contract notes

Deviations from and additions to the overview's interface contract made by this part. None changes a contract signature; the additions are cross-task names that later parts may rely on.

**Additions (produced here, usable by later tasks):**

1. `colosseum.sp2.worker.buffers` (T3.2):
   - `put_row(dst: Tree, idx: Any, src: Tree) -> None` — `dst[idx] = src` for a bare array or every leaf of a dict tree. Used by `MatchRunner` (T3.3) to assemble inference batches. Reason: Part A's prototype of `tree_assign` rejects a bare-array `dst` (the contract text does not forbid it); Part B does not depend on that detail either way.
   - `RolloutBuffer.reward_sum -> float` (rewards in the used slots), `RolloutBuffer.spec`, `RolloutBuffer.initial_state`, `RolloutBuffer.policy_version`; `BufferPool.parked_reward(agent_id) -> float`. Used by `RolloutLoop.buffered_reward` and the reward-accounting contract test.
2. `colosseum.sp2.worker.rollout_loop.RolloutLoop` (T3.4): implements `ModelPool.get(agent_id, network_id)` and the five `MatchObserver` methods publicly (the contract says it is both); `buffered_reward(agent_id) -> float` (rewards recorded but not yet sent, seat + parked buffers) for reward accounting, next to the contract's `buffered_transitions/<agent>` stat.
3. `colosseum.sp2.worker.rollout_worker` (T3.4): `report_worker_stats(q, worker_id, stats)` (SP1 name kept) and `_drain_commands(command_queue)`. `_drain_commands` merges lineups per env (a later non-`None` lineup wins, `None` keeps an earlier one) instead of SP1's "newest command wins": the per-env `None` entries of `WorkerCommand.lineups` would otherwise erase staged lineups.
4. `colosseum.sp2.algorithms.appo` (T4.2): `resolve_modes(config, action_spec) -> (ratio_mode, unit_trace, entropy_reduction)` and `APPO.modes`. Ruling — **with `K == 1` every mode resolves to `("joint", "joint", "sum")`**. Why: spec block 6 states that with K = 1 all modes give the same result, but `per_unit` (no ρ factor) and `joint` (advantage × clipped ρ) differ off-policy even with one decider; collapsing makes the statement exact and keeps SP1's behavior for every non-units game. Cost if wrong: `Units(1, ...)` agents with `ratio_mode: per_unit` silently get the joint loss; revisit with the owner after the criterion-4 experiment.
5. `ess` metric = normalized effective sample size of the scalar V-trace ρ over ACT slots, `(Σρ)² / (N·Σρ²)` ∈ (0, 1]; `approx_kl` and `clip_fraction` are per decider (equal to SP1's at K = 1); `rho_mean` / `rho_clip_frac` / `c_clip_frac` use the scalar ρ chosen by `unit_trace`.
6. `MatchRunner` (T3.3): `context` is a prefix ending with `", "`; each env's `EpisodeTracker` gets `context=f"{context}env {e}"`. Bad lineups (unknown layout, wrong seat count, an agent without a latest model, a wrong number of lineups) raise `ValueError`. Actions are sent to envs as numpy trees, one row of the batched action per seat (`Discrete` → a numpy integer scalar). `EpisodeEnd.final_global_state` holds only the live seats. `MatchRunner.close()` closes the vector env. Per-episode seeds: `int(np.random.SeedSequence([seed, env, episode_index]).generate_state(1)[0])`.
7. `learner_process` (T4.4): `consumed_samples` counts ACT slots, so in distributed mode `progress = consumed_samples / total_timesteps` mixes decisions with env steps (documented in the module docstring; one budget semantics is SP5).
8. `colosseum.sp2.worker.rollout_worker` imports `WORKER_STATS_INTERVAL_SEC` from `colosseum.metrics.aggregator`. T5.3's `sp2` copy of the aggregator must keep this constant (after T7.3 the import path stays `colosseum.metrics.aggregator`).

**Test kit (shared, "do not duplicate"):**

- `tests/game_helpers.py`: `Tick`, `TickGame`, `SCRIPT_OBS_SPACE`, `SCRIPT_GS_SPACE`, `DictModelPool`, `RecordingObserver` (T3.3); `NumpyOnlyQueue` (T3.4; the SP2 successor of SP1's `dataflow_helpers.CheckedQueue`, which T7.3 deletes); `synthetic_chunk` (T4.2); `learner_role`, `chunk_v2_payload`, `learner_appo`, `RecordingLearnerAlgorithm` (T4.4; successors of `dataflow_helpers.chunk_payload` / `RecordingAlgorithm` for T5.x/T6.4 learner and launcher tests). `TickGame` is deliberately not named `ScriptedGame`: Part A's prototype already has a `ScriptedGame` (raw `StepResult` replay for contract-violation tests).
- `tests/contract/game_harness.py`: `Collected`, `GameFactory`, `Slot`, `lineup`, `make_loop`, `run_steps`, `run_until_chunks`, `kinds`, `slot_steps`, `seat_chunks` (T3.4); `LearnerView`, `learner_eval` (T4.3). Only tests under `tests/contract/` can import it (pytest puts that directory on `sys.path`); unit tests use `tests/game_helpers.py`.
- `pyproject.toml` `known-first-party` gains `game_harness` (and `game_helpers` if Part A did not add it).

**Requirements on Part A beyond the literal contract** (all met by Part A's prototype at the time of writing):

- `make_test_model(role, core)` supports a `Box` observation of shape `(5,)` with dtype `float32` **or `uint8`** (the encoder casts), a `Box(4)` observation, `Discrete(n)` actions, `Units(U, Discrete(n))` actions, and roles with a `global_state_space` (any value path is fine; T3.4 only collects such chunks). Its `ComposedModel` has a `value` submodule whose last `nn.Linear` has a bias (T4.3 shifts it).
- `ActionSpec.allocate_actions(())` and `ActionSpec.full_mask(())` accept an empty leading shape (zero action / mask of one slot); `ActionSpec.boot_mask()` returns `None` iff `has_masks` is False.
- `EpisodeTracker.on_step` applies `terminated` after the rewards and returns `{}` (or masks of acting seats) on `episode_over`; `live_seats()` excludes seats terminated in the current step; `layout` is set after `on_reset`; error messages start with the `context` string.
- `Distribution.unit_valid(actions)` is `True` for decider 0 when the action has non-units groups; `log_prob == unit_log_prob.sum(-1)`.
- `VectorEnv.reset(requests)` / `step(actions)` return `dict[int, StepResult]` keyed by env index.

**Dependency note:** T4.2's test kit (`synthetic_chunk`) uses `RolloutBuffer` (T3.2), so T4.2 depends on T3.2 in addition to the overview's list (the index order T3.x → T4.x already satisfies it).
