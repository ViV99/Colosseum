# SP2: Game Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Colosseum's game model so that every game structure (solo, 1v1 turn-based and simultaneous, one bot with many units, team vs team, FFA with elimination and variable player counts, asymmetric roles, cooperative) trains end-to-end. Bootstrap moves to the learner and a centralized critic is added.

**Architecture:**
- A new env contract (`GameSpec` + `MultiAgentEnv` + `StepResult`) replaces `BaseEnv`.
- Observations, actions and masks become trees (Dict observations, `Units` per-unit actions).
- One `MatchRunner` core (seat lifecycle, inference grouping, results) serves both training (`RolloutLoop`) and eval.
- Chunks hold `act`/`boot`/`pad` slots, so the learner computes every value (bootstrap included) with its current network, and an optional `global_state` reaches only the value path.
- Matchmaking, ratings and eval work on lineups: a layout (list of teams of seats with roles) plus one assignment per seat.
- The new code is built **next to** the old code in a temporary package `colosseum.sp2` with its own CLI (`python -m colosseum.sp2`). Task T7.3 moves it over the old code and deletes the legacy, so the test suite stays green after every task.

**Tech Stack:** Python 3.12, PyTorch ≥2.6 (CPU for development), numpy, gymnasium, pydantic v2, click, pytest (+pytest-timeout), ruff, uv. grpcio/protobuf optional (distributed mode).

**Spec:** `docs/superpowers/specs/2026-10-08-sp2-game-model-design.md`. Read it before starting any task. Research backing: `research/sp2_game_apis.md`, `research/sp2_per_unit_ppo.md`. SP1 decisions: `docs/superpowers/reports/2026-10-08-sp1-acceptance.md` (rulings at the end).

## Plan files

| File | Spec blocks | Tasks |
|---|---|---|
| `00-overview.md` (this file) | constraints, strategy, file map, interface contract, task index | — |
| `01-foundations-and-model.md` | 0 SP1 residuals, 1 env contract, 2 observations/actions, 3 model, config | T0.1, T1.1–T1.7, T2.1–T2.4 |
| `02-worker-and-learner.md` | 4 chunk v2, 5 worker/`MatchRunner`, 6 algorithm | T3.1–T3.4, T4.1–T4.4 |
| `03-league-eval-switch.md` | 7 matchmaking/ratings, 8 eval, 9 config/validate/checkpoints, BC, distributed, the switch | T5.1–T5.4, T6.1–T6.4, T7.1–T7.3 |
| `04-demos-docs-acceptance.md` | 10 demo envs, examples, docs, benchmarks; acceptance | T8.1–T8.7 |

## Global Constraints

These apply to every task.

- **Python and dependencies:** Python 3.12 (`requires-python = ">=3.11"` stays); `torch>=2.6`; no new runtime dependencies (gymnasium, numpy, pydantic, click, pyyaml are already declared). grpc stays in the optional extra `grpc`, wandb in `wandb`.
- **Environment:** run everything with `.venv/bin/python` (create it with `scripts/setup-dev.sh` if missing). Never install into system Python. CPU torch only from `--index-url https://download.pytorch.org/whl/cpu`, never with `--extra-index-url`.
- **No backward compatibility** with SP1 APIs, configs or checkpoints (owner's instruction). Legacy is deleted, not wrapped — but only in T7.3 (see "Shadow package strategy").
- **Green suite after every task:** the task's new tests pass, and `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` passes in full **with zero warnings**; `.venv/bin/ruff check .` passes.
- **Inter-process data:** nothing that crosses a process boundary (mp.Queue, pipes, gRPC) contains `torch.Tensor`. Use the numpy payload forms of the contract below (`TrajectoryChunk.to_payload`, `WeightPayload`, `WorkerCommand`, `MatchResult`, `StepResult` from subprocess envs).
- **dtypes are preserved** end to end for observations and `global_state` (a `uint8` leaf stays `uint8` in buffers, chunks, payloads and the learner batch; only the model casts).
- **Errors:** env contract violations raise `colosseum.core.errors.EnvContractError` with context `"worker W, env E, seat P, episode step K, layout L: ..."` (omit parts that do not apply). Config problems raise `colosseum.core.errors.ConfigError` with a fix hint. No silent fallbacks except those named in the spec (default outcome from rewards; missing network → latest with collect).
- **Tests:**
  - files only under pytest's `tmp_path`;
  - `@pytest.mark.gpu` for CUDA-only tests (auto-skipped), `@pytest.mark.slow` for tests longer than ~20 s;
  - at most 2 worker processes per test; `OMP_NUM_THREADS=1` (exported by `tests/conftest.py`);
  - no `__init__.py` under `tests/`; support modules are imported by bare name; **test file basenames and support-module names must be unique across the whole `tests/` tree** (pytest prepend import mode). New SP2 test files must not reuse an existing basename;
  - integration runs start processes through `tests/cli_runner.py` (process-group cleanup).
- **SP1 rulings stay in force** unless the spec (section 8, "Изменения правил SP1") changes them. Most likely to be broken by accident: agent id regex `[A-Za-z0-9_][A-Za-z0-9_-]*` and reserved ids `ratings`/`system`/`episodes`/`train`; strict explicit resume (`ConfigError` on malformed `meta.json`); `SHUTDOWN_GRACE_SEC = 7`, final-checkpoint put timeout = grace − 2, no child alive after 10 s; one module for seat/mask rules; eval loads checkpoints read-only (`load_checkpoint_dir`, never `CheckpointManager` on a user path); newest-wins weight queues; learner uses exactly `learner.batch_chunks` chunks per update; `training.total_timesteps` is a global env-step budget.
- **Language:** code, comments, docstrings, log messages, `README.md`, `CLAUDE.md` in English; new project docs (`docs/ENV_GUIDE.md`, additions to `docs/benchmarks.md`, `docs/GPU_CHECKS.md`) in Russian.
- **Commits:** conventional prefixes (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `perf:`); no attribution or co-author lines; at least one commit per task on branch `sp2-game-model`; push to `origin` after every task.
- **Style:** follow existing code style (type hints, dataclasses, pydantic models, module-level `logger = logging.getLogger(__name__)`, `from __future__ import annotations`).
- **Process (owner's instruction for SP2):** each task is implemented by one subagent on Opus with high effort (failing tests first, then code, then commit), reviewed by an Opus/high reviewer for spec compliance and quality; web research, if any, by Sonnet/high subagents.

## Shadow package strategy

The spec requires a green suite after every task, but almost every module changes (env API, model protocol, chunk format, worker, learner, coordinator, eval, config, CLI). Rewriting in place would break the old pipeline long before the new one works. So:

1. **New and rewritten modules live under `src/colosseum/sp2/`, mirroring their final location** (`colosseum/sp2/envs/game.py` → final `colosseum/envs/game.py`). Every `sp2` subdirectory has an `__init__.py` (empty, like the existing packages).
2. **Unchanged modules are imported from their current location** by `sp2` code. The list is in "File structure" below (e.g. `colosseum.core.errors`, `colosseum.core.ipc`, `colosseum.networks.cores`, `colosseum.networks.state`, `colosseum.utils.*`, `colosseum.metrics.jsonl`).
3. **A module that imports a changed module** (e.g. something importing `colosseum.core.config` or `colosseum.core.types`) gets an `sp2` copy when `sp2` code needs it. The task that needs it creates the copy and lists it under **Files**.
4. **Nothing outside `colosseum.sp2` and the new SP2 tests imports `colosseum.sp2`.** The old pipeline and its tests keep running untouched until T7.3.
5. **The new CLI** is `colosseum.sp2.cli` with `colosseum/sp2/__main__.py`, so `python -m colosseum.sp2 train|eval|validate|bc|run-learner|run-workers|serve-weight-store` works during development. `tests/cli_runner.py` gets a `module` parameter (default `"colosseum"`); SP2 tests pass `module="colosseum.sp2"` (T5.4).
6. **New SP2 tests** go into the normal tree (`tests/unit`, `tests/contract`, `tests/integration`, `tests/learning`) with new unique basenames, and import `colosseum.sp2.*`. New support modules: `tests/game_helpers.py` (toy `MultiAgentEnv` games, tiny models, `RandomPolicy`) and `tests/contract/game_harness.py`.
7. **Defaults that name classes** use the `sp2` path during development (e.g. `algorithm_class: "colosseum.sp2.algorithms.appo.APPO"`). Built-in class paths in examples and checkpoints' `meta.json` must never contain `colosseum.sp2` except through such defaults, which T7.3 rewrites.
8. **T7.3 (overlay)** deletes every legacy module, test and example listed there, moves `src/colosseum/sp2/**` onto `src/colosseum/**`, replaces `colosseum.sp2` with `colosseum` in every file (code, tests, configs, docs, string literals), switches `scripts/bench_throughput.py` to the new config schema, and leaves the suite green. After T7.3 the `sp2` package no longer exists; part `04-…` is written against the final paths (`colosseum.*`).

In this overview, contract paths are written as `colosseum.sp2.X`; after T7.3 they are `colosseum.X`.

## Deviations from the spec

- **Numbering:** spec section 7 calls the parts P0–P7. The plan uses T0 (P0), T1 (P1, plus config v2), T2 (P2), T3 (P3), T4 (P4), T5 (P5, plus launcher and CLI `train`), T6 (P6, plus BC and distributed), **T7 (new: the switch — port of SP1 integration coverage, new tic-tac-toe, overlay)** and T8 (P7: demos, examples, docs, acceptance).
- **Spec block 0 item 3** (move `_QueueReader` and queue helpers to `core/ipc.py`) is done in T0.1 on the old code, so the `sp2` launcher copy starts from the cleaned version. The `registry._reset_mask_row` duplication disappears with the old registry in T7.3.
- **Rating entities (spec block 7, "Сущность рейтинга — (агент, сеть), как в SP1"):** the plan follows SP1's actual code: ELO and the win-rate matrix are keyed by the base `agent_id` (a checkpoint of X playing Y counts as X vs Y); "latest of X vs a checkpoint of X" updates `wr_vs_past`; two `latest` seats of the same agent carry no signal and are skipped. The team-pair weighting of the spec is applied on top (see `member_pairs`).
- **`tic_tac_toe`, `chase`, `space_miners`** are rewritten as new files in their example directories (`game.py`, `models.py`) because the old `env.py`/`networks.py` are used by old tests until T7.3, which deletes the old files.
- **`ratings.json` and `ratings` records** are `{"env_steps": N, "layouts": {<layout>: {...}}}`, not the flat `{<layout>: ...}` sketched in spec block 7: a layout may legally be named `env_steps`, so layouts are nested (amendment R1).
- **Vector-env API shape:** spec block 1 sketches `step -> list[StepResult]` and `reset(env_index, seed, layout)`; the contract uses mapping in, dict out (`reset(requests)`, `step(actions)`), which batches resets into one IPC round. Same semantics.
- **Tree utilities:** spec block 2 says the `networks/state.py` utilities are generalized. The plan keeps `networks/state.py` for model `State` pytrees and adds `core/tree.py` for data trees (observations, actions, masks); merging them is not needed (YAGNI).
- **Distributed mode** (spec block 10): each worker env draws its layout from `matchmaking.layouts` once and keeps it for the worker's lifetime (no coordinator in distributed mode until SP5). Recorded as ruling PR-3.

---

## File structure

### Create (shadow package, final path = without `sp2/`)

Env contract and data model:
- `src/colosseum/sp2/__init__.py`, `src/colosseum/sp2/__main__.py`, and `__init__.py` in every `sp2` subpackage.
- `src/colosseum/sp2/core/tree.py` — tree utilities (T1.1).
- `src/colosseum/sp2/envs/spaces.py` — `Units` space (T1.2).
- `src/colosseum/sp2/core/specs.py` — `ObsSpec`, `ActionSpec`, mask rules (T1.3). Successor of `core/seat_info.py` and `core/action_spec.py`.
- `src/colosseum/sp2/envs/game.py` — `RoleSpec`, `SeatSpec`, `GameSpec`, `MultiAgentEnv`, `StepResult`, `Outcome` (T1.4).
- `src/colosseum/sp2/core/outcomes.py` — `resolve_outcome` (T1.4).
- `src/colosseum/sp2/envs/contract.py` — `EpisodeTracker`, `SeatPhase` (T1.5).
- `src/colosseum/sp2/envs/vector.py` — `VectorEnv`, `SubprocessVectorEnv` (T1.6).
- `src/colosseum/sp2/core/config.py` — config v2 (T1.7).
- `src/colosseum/sp2/core/types.py` — chunk v2, lineups, results, commands, weights (T3.1).

Model:
- `src/colosseum/sp2/networks/dist/__init__.py`, `base.py`, `leaf.py`, `units.py`, `tree.py` — distributions (T2.1, T2.2).
- `src/colosseum/sp2/networks/base.py` — `BaseEncoder`, `EncoderOutput`, `BasePolicy`, `BaseValue`, `BaseCriticEncoder` (T2.3).
- `src/colosseum/sp2/networks/model.py` — `PolicyModel`, `PolicyStep`, `UnrollOutput`, `ActOutput`, `act` (T2.3).
- `src/colosseum/sp2/networks/composed.py` — `ComposedModel` (T2.3).
- `src/colosseum/sp2/networks/heads.py` — `make_distribution` re-export, `UnitsHead`, `gridnet_to_units` (T2.3).
- `src/colosseum/sp2/networks/normalization.py` — `NormalizeObs` with a leaf path (T2.3).
- `src/colosseum/sp2/core/roles.py` — `role_signature`, `resolve_agent_roles`, `agent_role_spec` (T2.4).
- `src/colosseum/sp2/core/registry.py` — `import_class`, `build_model`, `make_env`, `env_spec` (T2.4); `validate_config` (T6.2).

Worker and learner:
- `src/colosseum/sp2/worker/buffers.py` — `BufferSpec`, `RolloutBuffer`, `BufferPool` (T3.2).
- `src/colosseum/sp2/worker/match_runner.py` — `MatchRunner`, `ModelPool`, `MatchObserver`, `ActRecord`, `EpisodeEnd` (T3.3).
- `src/colosseum/sp2/worker/rollout_loop.py` — `RolloutLoop`, `LoopIO` (T3.4).
- `src/colosseum/sp2/worker/rollout_worker.py` — `rollout_worker_process` (T3.4).
- `src/colosseum/sp2/algorithms/vtrace.py` — `compute_vtrace_slots`, `VTraceOut` (T4.1).
- `src/colosseum/sp2/algorithms/base.py`, `src/colosseum/sp2/algorithms/appo.py` (T4.2).
- `src/colosseum/sp2/bc/kickstart.py` (T4.2).
- `src/colosseum/sp2/learner/learner.py` (T4.4).

League, CLI, eval, BC, distributed:
- `src/colosseum/sp2/coordinator/matchmaker.py` (T5.1), `ratings.py` (T5.2), `coordinator.py` and `checkpoint_manager.py` (T5.3).
- `src/colosseum/sp2/metrics/aggregator.py`, `src/colosseum/sp2/metrics/hub.py` (T5.3; copies adapted to `MatchResult` v2 and per-layout ratings).
- `src/colosseum/sp2/core/run_dir.py` (T5.4; copy importing config v2).
- `src/colosseum/sp2/launcher.py`, `src/colosseum/sp2/cli.py` (T5.4; `eval`, `validate`, `bc`, distributed commands added in T6.x).
- `src/colosseum/sp2/eval.py` (T6.1).
- `src/colosseum/sp2/bc/offline_bc.py` (T6.3).
- `src/colosseum/sp2/distributed.py` and the `sp2` copies of `transport/serialization.py`, `transport/grpc_transport.py`, `weight_store/grpc_store.py` that need chunk v2 (T6.4).

Tests and examples:
- `tests/game_helpers.py` — toy games and tiny models (T1.4 creates; later tasks extend).
- `tests/contract/game_harness.py` (T3.4 creates; T4.3 extends).
- `examples/tic_tac_toe/game.py`, `examples/tic_tac_toe/models.py`, `configs/sp2/tic_tac_toe.yaml` (T7.2; the config moves to `configs/examples/` in T7.3).
- Demo envs `examples/{coin_grid,unit_harvest,team_tag,tron,predator_prey,coop_buttons}/` with `game.py`, `models.py`, and configs `configs/examples/<name>.yaml` (T8.1, T8.2).
- `examples/space_miners/game.py`, `models.py`; `examples/composite_action/game.py`, `models.py` (T8.5).
- `docs/ENV_GUIDE.md` (T8.6).

### Reused unchanged by `sp2` (import from the current path)

`colosseum.core.errors`, `colosseum.core.ipc` (after T0.1), `colosseum.core.threads`, `colosseum.networks.cores`, `colosseum.networks.state`, `colosseum.utils.*` (`fs`, `logging`, `process`, `seeding`), `colosseum.metrics.jsonl`, `colosseum.metrics.console`, `colosseum.metrics.wandb_logger` (if it only needs duck-typed config; otherwise copy in T5.3), `colosseum.coordinator.agent_pool`, `colosseum.weight_store.base`, `colosseum.weight_store.shared_memory`, `colosseum.transport.base`.

### Deleted in T7.3 (legacy)

`envs/base_env.py`, `envs/vec_env.py`, `envs/subproc_vec_env.py`, `core/action_spec.py`, `core/seat_info.py`, `networks/distributions.py`, and every old module that has an `sp2` replacement (it is overwritten by the move). Old tests that test removed behavior, `tests/helpers.py`, `tests/dataflow_helpers.py`, `tests/contract/harness.py` (unless still imported by surviving tests), `examples/*/env.py`, `examples/*/networks.py`, old `configs/examples/*.yaml` replaced by new ones.

---

## Interface contract

These are the cross-task names and signatures. A task that produces a name implements it exactly as written. A task that consumes it uses it exactly. If a task must deviate, it updates this section in the same commit and says so in the commit message. Paths are `colosseum.sp2.*` until T7.3.

Common aliases used below:

```python
Tree = Any        # np.ndarray | torch.Tensor | dict[str, Tree]; dict keys are str, insertion order is significant
State = Any       # colosseum.networks.state.State (unchanged from SP1)
```

### `colosseum.sp2.core.tree` (T1.1)

```python
def tree_map(fn: Callable[..., Any], tree: Tree, *rest: Tree) -> Tree: ...   # leaves aligned; structure of `tree`
def tree_leaves(tree: Tree) -> list[Any]: ...                                # depth-first, dict insertion order
def tree_paths(tree: Tree) -> list[tuple[str, ...]]: ...                     # same order as tree_leaves; () for a bare leaf
def tree_get(tree: Tree, path: tuple[str, ...]) -> Any: ...
def tree_stack(trees: Sequence[Tree], axis: int = 0) -> Tree: ...            # np.stack or torch.stack per leaf type
def tree_index(tree: Tree, idx: Any) -> Tree: ...                           # leaf[idx] for every leaf
def tree_assign(dst: Tree, idx: Any, src: Tree) -> None: ...                 # in place: dst_leaf[idx] = src_leaf
def tree_to_torch(tree: Tree, device: str | torch.device = "cpu") -> Tree: ...   # via core.ipc.numpy_to_tensor
def tree_to_numpy(tree: Tree) -> Tree: ...                                       # via core.ipc.tensor_to_numpy
def tree_same_structure(a: Tree, b: Tree) -> bool: ...
```

### `colosseum.sp2.envs.spaces` (T1.2)

```python
@dataclass(frozen=True)
class UnitComponent:
    name: str                         # Dict key, or "0".."C-1" for MultiDiscrete / "0" for Discrete or Box
    kind: Literal["discrete", "box"]
    size: int                         # categories (discrete) or dimension (box)

class Units(gymnasium.spaces.Space):
    def __init__(self, max_units: int,
                 per_unit: gymnasium.spaces.Discrete | gymnasium.spaces.MultiDiscrete
                           | gymnasium.spaces.Box | gymnasium.spaces.Dict,
                 only_if: Mapping[str, tuple[str, Collection[int]]] | None = None,
                 seed: int | None = None): ...
    max_units: int
    per_unit: gymnasium.spaces.Space
    per_unit_kind: Literal["discrete", "multi_discrete", "box", "dict"]
    components: tuple[UnitComponent, ...]           # natural order (Dict: space.spaces order)
    only_if: dict[str, tuple[str, frozenset[int]]]  # child -> (discrete parent in the same unit, allowed values)
    def sample(self, mask: Any = None) -> Any: ...
    def contains(self, x: Any) -> bool: ...
```

Action value of a `Units` group (env side, numpy): `per_unit_kind == "discrete"` → `int64[U]`; `"multi_discrete"` → `int64[U, C]`; `"box"` → `float32[U, d]`; `"dict"` → `{name: int64[U] | float32[U, d]}` (Dict values may only be `Discrete` or 1-D `Box`). Validation errors in the constructor: `ValueError` (e.g. `only_if` naming an unknown component, a non-discrete parent, a cycle, a nested `MultiDiscrete` inside `Dict`).

### `colosseum.sp2.core.specs` (T1.3)

```python
@dataclass(frozen=True)
class LeafSpec:
    path: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: np.dtype

class ObsSpec:                                        # observations and global_state
    leaves: tuple[LeafSpec, ...]
    @classmethod
    def from_space(cls, space: gymnasium.Space) -> ObsSpec: ...   # Box | Discrete | MultiBinary | MultiDiscrete | nested Dict
    def allocate(self, leading: tuple[int, ...]) -> Tree: ...      # zeros, numpy, dtypes preserved
    def check(self, value: Tree, where: str) -> None: ...          # structure + shapes; EnvContractError
    def signature(self) -> str: ...

@dataclass(frozen=True)
class ActionGroup:
    path: tuple[str, ...]                             # () when the action space is not a Dict
    kind: Literal["discrete", "multi_discrete", "box", "units"]
    nvec: tuple[int, ...]                             # discrete: (n,); multi_discrete: nvec; else ()
    box_dim: int                                      # box: d; else 0
    units: Units | None                               # units only
    mask_size: int                                    # discrete: n; multi_discrete: sum(nvec); units: sum of discrete component sizes; box: 0

class ActionSpec:
    groups: tuple[ActionGroup, ...]                   # natural order
    is_dict: bool
    has_units: bool
    has_masks: bool                                   # any group with mask_size > 0, or any units group
    num_deciders: int                                 # sum(max_units) + (1 if any non-units group else 0)
    @classmethod
    def from_space(cls, space: gymnasium.Space) -> ActionSpec: ...
    def allocate_actions(self, leading: tuple[int, ...]) -> Tree: ...       # zeros, numpy, int64/float32
    def full_mask(self, leading: tuple[int, ...] = ()) -> Tree | None: ...  # all allowed (units: unit=True); None if not has_masks
    def boot_mask(self) -> Tree | None: ...                                 # all allowed; units: unit=False
    def normalize_mask(self, raw: Tree | None, where: str) -> Tree | None: ...
        # missing leaves -> all-true; checks structure/shape/dtype(bool); EnvContractError
    def check_acting_mask(self, mask: Tree | None, where: str) -> None: ...
        # empty-row rule: a non-units discrete group with no legal action -> EnvContractError
    def signature(self) -> str: ...
```

Mask tree layout (mirrors the discrete parts of the action tree): `discrete` → `bool[n]`; `multi_discrete` → `bool[sum(nvec)]`; `box` → no leaf; `units` → `{"unit": bool[U], "action": bool[U, sum discrete sizes]}` (discrete components in natural order). Deciders: decider 0 is all non-units groups together (if any), then each units group in spec order contributes `max_units` deciders.

### `colosseum.sp2.envs.game` (T1.4)

```python
@dataclass(frozen=True)
class RoleSpec:
    observation_space: gymnasium.Space
    action_space: gymnasium.Space
    global_state_space: gymnasium.Space | None = None

@dataclass(frozen=True)
class SeatSpec:
    role: str
    team: int

@dataclass(frozen=True)
class GameSpec:
    roles: dict[str, RoleSpec]
    layouts: dict[str, tuple[SeatSpec, ...]]
    @property
    def max_seats(self) -> int: ...
    def layout_size(self, layout: str) -> int: ...
    def teams(self, layout: str) -> list[list[int]]: ...           # team index -> seats, ascending
    def num_teams(self, layout: str) -> int: ...
    def outcome_kind(self, layout: str) -> Literal["score", "wdl", "rank"]: ...   # 1 team / 2 / >= 3
    def role_of(self, layout: str, seat: int) -> str: ...
    def validate(self) -> None: ...                                # EnvContractError naming the problem
    @classmethod
    def solo(cls, obs: gymnasium.Space, act: gymnasium.Space,
             global_state: gymnasium.Space | None = None) -> GameSpec: ...           # layout "solo", role "player"
    @classmethod
    def symmetric(cls, num_players: int | Iterable[int], obs, act, global_state=None) -> GameSpec: ...
        # FFA: team = seat; layouts "2p", "3p", ...
    @classmethod
    def teams_of(cls, sizes: Sequence[int] | Iterable[Sequence[int]], obs, act, global_state=None) -> GameSpec: ...
        # layouts "2v2", "3v3", "2v1v1"; a single team [n] -> "coop<n>"

class MultiAgentEnv(ABC):
    spec: GameSpec
    @abstractmethod
    def reset(self, seed: int | None, layout: str) -> StepResult: ...
    @abstractmethod
    def step(self, actions: dict[int, Any]) -> StepResult: ...
    def close(self) -> None: ...

@dataclass
class Outcome:
    team_rank: dict[int, float] | None = None
    team_score: dict[int, float] | None = None

@dataclass
class StepResult:
    acting: set[int]
    obs: dict[int, Any]
    action_masks: dict[int, Any] = field(default_factory=dict)
    rewards: dict[int, float] = field(default_factory=dict)
    terminated: set[int] = field(default_factory=set)
    episode_over: bool = False
    truncated: bool = False
    final_obs: dict[int, Any] | None = None
    global_state: dict[int, Any] | None = None
    outcome: Outcome | None = None
    infos: dict[int, dict] = field(default_factory=dict)
```

Semantics: spec block 1 (lifecycle, rewards in the elimination step, waiting seats' observations ignored, `max_idle_steps`, end of episode, `global_state`, `reset`).

### `colosseum.sp2.core.outcomes` (T1.4)

```python
def resolve_outcome(outcome: Outcome | None, teams: list[list[int]],
                    seat_returns: Sequence[float], where: str = "") -> tuple[dict[int, float], dict[int, float]]: ...
    # -> (team_rank, team_score). Default score = mean of the team's seat returns; ranks from scores
    # (higher better; rank = 1 + number of strictly better teams, so ties share a rank).
    # Keys must be exactly the layout's teams, ranks/scores finite -> else EnvContractError.
def pairwise_rank_score(rank_a: float, rank_b: float) -> float: ...   # 1.0 if rank_a < rank_b, 0.5 if equal, else 0.0
```

### `colosseum.sp2.envs.contract` (T1.5)

```python
class SeatPhase(enum.Enum):
    EMPTY = "empty"
    LIVE = "live"
    ELIMINATED = "eliminated"

class EpisodeTracker:
    """Validates one env's StepResults against its GameSpec and tracks seat phases."""
    def __init__(self, spec: GameSpec, *, max_idle_steps: int = 1000, context: str = ""): ...
    layout: str | None
    episode_step: int
    episode_over: bool
    def on_reset(self, layout: str, result: StepResult) -> dict[int, Tree | None]: ...
    def on_step(self, actions: dict[int, Any], result: StepResult) -> dict[int, Tree | None]: ...
        # both return the normalized masks of the ACTING seats (None when the role has no masks)
    def phase(self, seat: int) -> SeatPhase: ...
    def live_seats(self) -> list[int]: ...
    def acting(self) -> set[int]: ...
```

`on_step` order: check `actions` keys == previous `acting`; check the result (every rule of spec block 1); apply rewards; then mark `terminated` seats ELIMINATED. `context` prefixes errors (e.g. `"worker 0, env 3"`); the tracker appends seat, episode step and layout.

### `colosseum.sp2.envs.vector` (T1.6)

```python
class VectorEnv:
    def __init__(self, env_fn: Callable[[], MultiAgentEnv], num_envs: int): ...
    num_envs: int
    spec: GameSpec                       # from env 0; every env must report an equal spec (EnvContractError otherwise)
    def reset(self, requests: Mapping[int, tuple[int | None, str]]) -> dict[int, StepResult]: ...  # env -> (seed, layout)
    def step(self, actions: Mapping[int, dict[int, Any]]) -> dict[int, StepResult]: ...           # steps the listed envs only
    def close(self) -> None: ...

class SubprocessVectorEnv:               # same interface; one IPC round per reset() / step() call
    def __init__(self, env_fn: Callable[[], MultiAgentEnv], num_envs: int, num_workers: int | None = None): ...
```

No auto-reset. Child processes apply `torch_threads`/`OMP` limits like SP1's `subproc_vec_env.py`.

### `colosseum.sp2.core.config` (T1.7)

Copy of SP1 `core/config.py` with these changes (everything else — `StrictModel`, agent-id rules, `deep_merge`, `parse_override_value`, `apply_overrides`, `load_config`, `get_agent_config`, `get_trainable_agent_ids` — keeps its SP1 behavior):

```python
class AlgorithmConfig(StrictModel):                       # SP1 fields, plus:
    algorithm_class: str = "colosseum.sp2.algorithms.appo.APPO"
    ratio_mode: Literal["auto", "joint", "per_unit"] = "auto"
    unit_trace: Literal["auto", "joint", "geo_mean", "none"] = "auto"
    entropy_reduction: Literal["auto", "mean_valid", "sum"] = "auto"

class EnvConfig(StrictModel):
    env_class: str
    kwargs: dict[str, Any] = {}
    max_idle_steps: int = Field(1000, ge=1)
    # num_players removed

class NetworkConfig(StrictModel):                         # SP1 fields, plus:
    critic_encoder_class: str | None = None               # composed models only

class RolloutConfig(StrictModel):                         # SP1 fields; chunk_length: Field(256, ge=2)

class TrainingConfig(StrictModel):                        # SP1 fields minus `phase` (TrainingPhase enum removed)

class MatchmakingConfig(StrictModel):
    mode: Literal["self_play", "league"] = "self_play"
    layouts: dict[str, float] = {}                        # empty = every layout, equal weights; weights > 0
    self_play_ratio: float = Field(0.5, ge=0, le=1)
    pfsp_exponent: float = Field(1.0, ge=0)
    latest_prob: float = Field(0.5, ge=0, le=1)
    teammates: Literal["self", "mixed"] = "self"
    teammate_self_prob: float = Field(0.5, ge=0, le=1)
    shuffle_seats: bool = True

class CheckpointConfig(StrictModel):
    interval: int = Field(1000, ge=1)                     # was self_play.checkpoint_interval
    pool_size: int = Field(20, ge=1)                      # was self_play.pool_size
    save_optimizer: bool = True

class AgentOverride(StrictModel):
    networks: dict[str, Any] | None = None
    algorithm: dict[str, Any] | None = None
    learner: dict[str, Any] | None = None
    roles: list[str] | None = None

class ColosseumConfig(StrictModel):
    algorithm, env, networks, rollout, learner, training, matchmaking, checkpoint, metrics, bc, transport, run, agents
    # self_play removed
    def get_agent_config(self, agent_id: str) -> ColosseumConfig: ...
    def get_trainable_agent_ids(self) -> list[str]: ...
    def agent_roles(self, agent_id: str) -> list[str] | None: ...   # agents.<id>.roles, None if omitted
```

### `colosseum.sp2.networks.dist` (T2.1, T2.2)

```python
class Distribution(ABC):
    batch_size: int
    num_deciders: int
    @abstractmethod
    def sample(self) -> Tree: ...
    @abstractmethod
    def mode(self) -> Tree: ...
    def log_prob(self, actions: Tree) -> Tensor: ...             # [B] = sum over valid deciders of unit_log_prob
    @abstractmethod
    def unit_log_prob(self, actions: Tree) -> Tensor: ...        # [B, K]; 0 where invalid
    @abstractmethod
    def unit_entropy(self, actions: Tree) -> Tensor: ...         # [B, K]; 0 where invalid
    @abstractmethod
    def unit_valid(self, actions: Tree) -> Tensor: ...           # [B, K] bool
    @abstractmethod
    def unit_kl(self, other: Distribution, actions: Tree) -> Tensor: ...   # [B, K] KL(self || other); 0 where invalid
    @abstractmethod
    def apply_mask(self, mask: Tree | None) -> Distribution: ...
    @classmethod
    @abstractmethod
    def cat(cls, dists: Sequence[Distribution]) -> Distribution: ...       # along the batch

# leaf distributions (T2.1); each is ONE decider when used alone
class CategoricalDist(Distribution):       # __init__(logits [B, n], mask: Tensor | None = None)
class MultiCategoricalDist(Distribution):  # __init__(logits [B, sum(nvec)], nvec: Sequence[int], mask: Tensor | None = None)
class DiagGaussianDist(Distribution):      # __init__(mean [B, d], log_std [B, d] | [d])

class UnitsDist(Distribution):             # (T2.2) __init__(group: ActionGroup, params: dict[str, Tensor | dict[str, Tensor]],
                                           #                 mask: dict[str, Tensor] | None = None)
                                           # params: discrete component -> logits [B, U, n];
                                           #         box component -> {"mean": [B, U, d], "log_std": [B, U, d] | [d]}
                                           # K = U; only_if gates by the given actions

class TreeDist(Distribution):              # (T2.1, units added in T2.2) composition along an ActionSpec
    def __init__(self, spec: ActionSpec, parts: dict[tuple[str, ...], Distribution]): ...   # group path -> distribution

def make_distribution(spec: ActionSpec, params: Tree) -> Distribution: ...
    # params mirror the action tree: discrete -> logits [B, n]; multi_discrete -> logits [B, sum(nvec)];
    # box -> {"mean": [B, d], "log_std": [B, d] | [d]}; units -> UnitsDist params (above)
    # returns a TreeDist (also for a single non-Dict group)
```

Rules: invalid positions use `torch.where`, never multiplication; masked logits use `torch.finfo(dtype).min`; a units decider is valid iff its `unit` mask is true and at least one component is valid (discrete component: non-empty mask row and `only_if` satisfied by the given action; box component: `only_if` satisfied); decider 0 (non-units groups) is always valid. Torch action tree from `sample()`: discrete `int64[B]`, multi_discrete `int64[B, C]`, box `float32[B, d]`, units as the env format with a leading `B`.

### `colosseum.sp2.networks.base`, `.model`, `.composed`, `.heads`, `.normalization` (T2.3)

```python
class EncoderOutput(NamedTuple):
    latent: Tensor                 # [B, D]
    aux: dict[str, Tensor]         # bypasses the core, e.g. unit embeddings [B, U, E]

class BaseEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, obs: Tree) -> Tensor | EncoderOutput: ...
    @property
    @abstractmethod
    def latent_dim(self) -> int: ...

class BasePolicy(nn.Module, ABC):
    @abstractmethod
    def forward(self, features: Tensor, aux: dict[str, Tensor]) -> Distribution: ...

class BaseValue(nn.Module, ABC):
    @abstractmethod
    def forward(self, features: Tensor) -> Tensor: ...            # [B]

class BaseCriticEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, global_state: Tree) -> Tensor: ...          # [B, G]
    @property
    @abstractmethod
    def output_dim(self) -> int: ...

class PolicyStep(NamedTuple):
    dist: Distribution             # batch [B]
    state: State

class UnrollOutput(NamedTuple):
    dist: Distribution             # batch [S*B], time-major (index s*B + b)
    value: Tensor | None           # [S*B]; None when with_value=False

class ActOutput(NamedTuple):
    actions: Tree                  # torch, batch [B]
    log_probs: Tensor              # [B]
    unit_log_probs: Tensor         # [B, K]
    state: State

class PolicyModel(nn.Module, ABC):
    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State: ...   # default None
    @abstractmethod
    def step(self, obs: Tree, state: State, action_mask: Tree | None = None) -> PolicyStep: ...
    @abstractmethod
    def unroll(self, obs: Tree, state0: State, reset_after: Tensor, action_mask: Tree | None = None,
               global_state: Tree | None = None, with_value: bool = True) -> UnrollOutput: ...
        # obs leaves [S, B, ...]; reset_after [S, B] bool (state reset AFTER slot s); masks [S, B, ...]
    def reset_state(self, state: State, done: Tensor) -> State: ...          # SP1 semantics
    def update_normalizers(self, obs: Tree, global_state: Tree | None = None) -> None: ...
        # default: every NormalizeObs submodule updates from its source tree and leaf path
    @property
    def is_stateful(self) -> bool: ...

@torch.no_grad()
def act(model: PolicyModel, obs: Tree, state: State, action_mask: Tree | None = None,
        deterministic: bool = False) -> ActOutput: ...

class ComposedModel(PolicyModel):
    def __init__(self, encoder: BaseEncoder, core: Core, policy_head: BasePolicy, value_head: BaseValue,
                 critic_encoder: BaseCriticEncoder | None = None): ...
    # submodules: encoder, core, policy, value, critic_encoder
    # value input = core features (⊕ critic_encoder(global_state) when set); unroll(with_value=True) with a
    # critic_encoder and global_state=None -> ValueError

class NormalizeObs(nn.Module):     # heads/encoders embed it; SP1 running-stat math
    def __init__(self, shape: Sequence[int], path: tuple[str, ...] = (),
                 source: Literal["obs", "global_state"] = "obs", eps: float = 1e-8, clip: float = 10.0): ...
    def forward(self, x: Tensor) -> Tensor: ...
    def update(self, x: Tensor) -> None: ...

class UnitsHead(nn.Module):        # unit features [B, U, F] -> UnitsDist params of one units group
    def __init__(self, group: ActionGroup, in_dim: int, hidden: int = 0): ...
    def forward(self, unit_features: Tensor) -> dict[str, Tensor | dict[str, Tensor]]: ...

def gridnet_to_units(x: Tensor) -> Tensor: ...    # [B, C, H, W] -> [B, H*W, C]
```

### `colosseum.sp2.core.roles`, `colosseum.sp2.core.registry` (T2.4; `validate_config` in T6.2)

```python
def role_signature(role: RoleSpec) -> str: ...     # ObsSpec/ActionSpec/global-state ObsSpec signatures
def resolve_agent_roles(config: ColosseumConfig, spec: GameSpec) -> dict[str, list[str]]: ...
    # roles given -> must exist and share one signature; omitted -> every role, which must share one
    # signature (else ConfigError "set agents.<id>.roles; roles with different spaces need separate agents")
def agent_role_spec(spec: GameSpec, roles: Sequence[str]) -> RoleSpec: ...   # spec.roles[roles[0]]

def import_class(dotted_path: str) -> type: ...    # SP1 behavior
def make_env(config: ColosseumConfig) -> MultiAgentEnv: ...
def env_spec(config: ColosseumConfig) -> GameSpec: ...    # instantiates once, validates the spec, closes
def build_model(agent_config: ColosseumConfig, role: RoleSpec) -> PolicyModel: ...
    # model_class -> cls(**inject, **networks.kwargs)
    # composed -> encoder(**inject, **kwargs); core(input_dim=encoder.latent_dim, **core.kwargs) (NoCore if None);
    #   critic_encoder(**inject, **kwargs) if set; policy(in_dim=core.output_dim, **inject, **kwargs);
    #   value(in_dim=core.output_dim + G, **kwargs)
    # inject = the subset of {observation_space, action_space, global_state_space, action_spec} that the
    # constructor accepts by name (SP1's `in_dim` probing rule)
def validate_config(config: ColosseumConfig) -> None: ...   # T6.2
```

### `colosseum.sp2.core.types` (T3.1)

```python
LATEST_NETWORK_ID = "latest"
SLOT_ACT, SLOT_BOOT, SLOT_PAD = 0, 1, 2          # values of TrajectoryChunk.kind (int8)

@dataclass
class TrajectoryChunk:
    agent_id: str
    policy_version: int                          # version at the first slot
    initial_state: State                         # leaves [1, ...]; None for stateless models
    obs: Tree                                    # torch, leaves [S, ...], dtypes preserved
    global_state: Tree | None
    actions: Tree                                # torch, leaves [S, ...]
    action_masks: Tree | None                    # None iff the role's ActionSpec has no masks
    kind: Tensor                                 # [S] int8
    reward: Tensor                               # [S] float32
    terminal: Tensor                             # [S] bool
    reset_after: Tensor                          # [S] bool
    behavior_logp: Tensor                        # [S] float32
    behavior_unit_logp: Tensor | None            # [S, K] float32, only when K > 1
    @property
    def num_slots(self) -> int: ...
    @property
    def num_acts(self) -> int: ...
    def to(self, device: str | torch.device) -> TrajectoryChunk: ...
    def pin_memory(self) -> TrajectoryChunk: ...
    def to_payload(self) -> dict[str, Any]: ...  # numpy trees + primitives only
    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TrajectoryChunk: ...

@dataclass
class SeatAssignment:
    agent_id: str
    network_id: str = LATEST_NETWORK_ID          # "latest" | "ckpt_v<N>"
    collect: bool = True

@dataclass
class Lineup:
    layout: str
    seats: list[SeatAssignment]                  # len == layout size

@dataclass
class SeatResult:
    seat: int
    role: str
    team: int
    agent_id: str
    network_id: str
    reward: float                                # undiscounted episode return
    eliminated_step: int | None = None

@dataclass
class TeamResult:
    team: int
    rank: float
    score: float

@dataclass
class MatchResult:
    match_id: str
    layout: str
    outcome_kind: Literal["score", "wdl", "rank"]
    seats: list[SeatResult]
    teams: list[TeamResult]
    episode_length: int

@dataclass
class WorkerCommand:
    lineups: list[Lineup | None]                 # one per worker env; None = unchanged
    new_checkpoints: dict[str, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict)

@dataclass
class WeightPayload: ...                         # identical to SP1 (from_model, to_torch_state_dict)
def state_dict_to_numpy(state_dict) -> dict[str, np.ndarray]: ...     # identical to SP1
def state_dict_from_numpy(state_dict) -> dict[str, Tensor]: ...       # identical to SP1
```

### `colosseum.sp2.worker.buffers` (T3.2)

```python
@dataclass(frozen=True)
class BufferSpec:
    obs: ObsSpec
    action: ActionSpec
    global_state: ObsSpec | None

class RolloutBuffer:
    def __init__(self, num_slots: int, spec: BufferSpec): ...
    num_slots: int
    @property
    def slots_used(self) -> int: ...
    @property
    def free_slots(self) -> int: ...
    @property
    def is_full(self) -> bool: ...
    @property
    def has_open(self) -> bool: ...        # last slot is a non-terminal ACT
    @property
    def ends_episode(self) -> bool: ...    # empty, or last non-PAD slot is a terminal ACT or a BOOT with reset_after
    @property
    def num_acts(self) -> int: ...
    def begin(self, initial_state: State, policy_version: int) -> None: ...   # only when empty; clones state leaves
    def write_act(self, obs: Tree, global_state: Tree | None, mask: Tree | None, action: Tree,
                  log_prob: float, unit_log_probs: np.ndarray | None, reward: float) -> None: ...
        # RuntimeError if free_slots < 2 (an ACT never takes the last slot)
    def add_reward(self, reward: float) -> None: ...      # to the open ACT
    def mark_terminal(self) -> None: ...                   # open ACT -> terminal + reset_after
    def write_boot(self, obs: Tree, global_state: Tree | None, reset_after: bool) -> None: ...   # needs has_open
    def write_pad(self) -> None: ...                       # copies the previous slot's obs/global_state
    def build_chunk(self, agent_id: str) -> TrajectoryChunk: ...   # only when is_full; does not reset
    def reset(self) -> None: ...

class BufferPool:
    def __init__(self, num_slots: int, specs: Mapping[str, BufferSpec]): ...   # agent_id -> spec
    def acquire(self, agent_id: str) -> RolloutBuffer: ...  # parked first, then free, then new; episode boundary only
    def park(self, agent_id: str, buf: RolloutBuffer) -> None: ...   # needs ends_episode and not is_full
    def parked_count(self) -> int: ...
    def parked_acts(self, agent_id: str) -> int: ...
```

Slot write rules (spec block 4) are applied by `RolloutLoop` (T3.4) using this API.

### `colosseum.sp2.worker.match_runner` (T3.3)

```python
class ModelPool(Protocol):
    def get(self, agent_id: str, network_id: str) -> PolicyModel | None: ...

@dataclass
class ActRecord:
    agent_id: str
    network_id: str
    obs: Tree                        # numpy
    global_state: Tree | None        # numpy; only when the seat's role declares global_state_space
    mask: Tree | None                # normalized numpy mask (EpisodeTracker)
    action: Tree                     # numpy, as sent to the env
    log_prob: float
    unit_log_probs: np.ndarray | None   # [K] float32 when K > 1
    pre_state: State                 # model state before this act, leaves [1, ...]

@dataclass
class EpisodeEnd:
    truncated: bool
    live_seats: list[int]            # LIVE at the end (eliminated seats excluded, incl. this step's)
    final_obs: dict[int, Tree] | None
    final_global_state: dict[int, Tree] | None
    result: MatchResult

class MatchObserver(Protocol):
    def on_act(self, env: int, seat: int, record: ActRecord) -> None: ...
    def on_rewards(self, env: int, rewards: dict[int, float]) -> None: ...
    def on_terminated(self, env: int, seats: list[int]) -> None: ...
    def on_episode_end(self, env: int, end: EpisodeEnd) -> None: ...
    def on_lineup_applied(self, env: int, old: Lineup, new: Lineup) -> None: ...

class MatchRunner:
    def __init__(self, *, vec_env: VectorEnv | SubprocessVectorEnv, lineups: Sequence[Lineup], models: ModelPool,
                 observer: MatchObserver | None = None, seed: int | None = None, max_idle_steps: int = 1000,
                 deterministic: bool = False, context: str = "", match_id_prefix: str = "m"): ...
    num_envs: int
    spec: GameSpec
    def lineup(self, env: int) -> Lineup: ...                       # current, after fallbacks
    def set_next_lineup(self, env: int, lineup: Lineup) -> None: ...   # applied at the env's next episode end
    def step(self) -> int: ...                                      # one step of every env; returns env steps
    @property
    def episodes_finished(self) -> int: ...
    def close(self) -> None: ...
```

`step()` order: (1) inference for all acting seats of all envs, grouped by `(agent_id, network_id)`, then `on_act` per acting seat in `(env, seat)` order; (2) `vec_env.step`; (3) per env in index order: `EpisodeTracker.on_step` → `on_rewards` (every reward of the step, including the elimination step) → `on_terminated` (if any) → if `episode_over`: `on_episode_end` → apply the next lineup (`on_lineup_applied`) → reset the env's model states; (4) one `vec_env.reset` for all finished envs with per-episode seeds → `EpisodeTracker.on_reset`. A lineup naming a network that `models.get` cannot provide is seated as `SeatAssignment(agent_id, "latest", collect=True)` with a one-time warning (SP1 rule). Episode seeds: `seed` given → deterministic per `(seed, env, episode index)` via `numpy.random.SeedSequence`; `None` → `None`. `match_id = f"{match_id_prefix}{env}_ep{k}"`.

### `colosseum.sp2.worker.rollout_loop`, `.rollout_worker` (T3.4)

```python
@dataclass
class LoopIO:                                      # SP1 names
    send_chunk: Callable[[TrajectoryChunk], None]
    poll_weights: Callable[[str], WeightPayload | None]
    report_result: Callable[[MatchResult], None] | None = None
    poll_command: Callable[[], WorkerCommand | None] | None = None
    add_env_steps: Callable[[int], None] | None = None

class RolloutLoop:                                 # a MatchObserver + ModelPool over its own MatchRunner
    def __init__(self, *, worker_id: int, env_fn: Callable[[], MultiAgentEnv], num_envs: int, chunk_length: int,
                 agent_ids: list[str], agent_roles: Mapping[str, Sequence[str]],
                 model_factories: Mapping[str, Callable[[], PolicyModel]], io: LoopIO, lineups: Sequence[Lineup],
                 weight_sync_interval: float = 5.0,
                 checkpoint_state_dicts_by_agent: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]] | None = None,
                 seed: int | None = None, vec_env_kind: Literal["sync", "subprocess"] = "sync",
                 subproc_workers: int | None = None, max_idle_steps: int = 1000): ...
    def step(self) -> int: ...
    def sync_weights(self) -> None: ...
    def run(self, should_stop: Callable[[], bool], max_env_steps: int = 0) -> None: ...
    def close(self) -> None: ...
    @property
    def stats(self) -> dict[str, int]: ...
        # chunks_sent, env_steps, episodes, parked_buffers, dropped_reward_episodes, recorded_transitions,
        # recorded_transitions/<agent>, buffered_transitions/<agent>   (transitions = ACT slots)

def rollout_worker_process(*, worker_id, env_fn, num_envs, chunk_length, agent_ids, agent_roles, model_factories,
                           trajectory_queues, weight_queues, stop_event, weight_sync_interval=5.0, torch_threads=1,
                           max_env_steps=0, env_step_counter=None, checkpoint_state_dicts_by_agent=None,
                           lineups, results_queue=None, command_queue=None, seed=None, vec_env_kind="sync",
                           subproc_workers=None, stats_queue=None, stats_interval_sec=WORKER_STATS_INTERVAL_SEC,
                           max_idle_steps=1000) -> None: ...
    # SP1 queue/signal/stats behavior; `gamma` removed (no worker-side bootstrap)
```

### `colosseum.sp2.algorithms.vtrace` (T4.1)

```python
class VTraceOut(NamedTuple):
    vs: Tensor               # [S, B] value targets (== values on non-ACT slots)
    td: Tensor               # [S, B] r_t + gamma * (1 - terminal_t) * vs_{t+1} - V_t on ACT slots; 0 elsewhere
    clipped_rho: Tensor      # [S, B] min(rho_bar, rho_t) on ACT slots; 0 elsewhere

def compute_vtrace_slots(*, log_rhos: Tensor, rewards: Tensor, values: Tensor, is_act: Tensor, terminal: Tensor,
                         gamma: float, rho_bar: float = 1.0, c_bar: float = 1.0, lam: float = 1.0) -> VTraceOut: ...
    # all [S, B]; log_rhos = scalar log-ratio per slot from unit_trace (ignored where not is_act);
    # an ACT never sits in the last slot; the next slot of a non-terminal ACT is an ACT of the same
    # episode or its BOOT; vs - V on non-ACT slots is 0, so traces stop at BOOT; terminal ACTs bootstrap 0;
    # non-ACT slots are selected with torch.where (NaN in their values never reaches the targets)
```

### `colosseum.sp2.algorithms.base`, `.appo`, `colosseum.sp2.bc.kickstart` (T4.2)

```python
class BaseAlgorithm(ABC): ...    # SP1 interface; compute_loss/train_step take chunk v2 lists

class APPO(BaseAlgorithm):
    def __init__(self, model: PolicyModel, config: AlgorithmConfig, action_spec: ActionSpec,
                 device: str | torch.device = "cpu", pin_memory: bool = False,
                 kickstart: KickstartLoss | None = None): ...
    def train_step(self, chunks: list[TrajectoryChunk]) -> dict[str, float]: ...
    def compute_loss(self, chunks: list[TrajectoryChunk]) -> dict[str, Tensor]: ...
    def evaluate_chunks(self, chunks: list[TrajectoryChunk]) -> tuple[Tensor, Tensor, Tensor | None]: ...
        # (log_probs [S*B], values [S*B], unit_log_probs [S*B, K] | None), time-major, exactly as in training
    # model, policy_version, consumed_samples (counts ACT slots), set_progress, state_dict/load_state_dict:
    # SP1 semantics; state_dict keys exactly: optimizer, progress, scaler, kickstart, policy_version, consumed_samples

class KickstartLoss:
    def __init__(self, teacher: PolicyModel, initial_lambda: float = 1.0, decay_steps: int = 50_000,
                 direction: Literal["forward", "reverse"] = "forward"): ...
    def compute(self, *, student_dist: Distribution, obs: Tree, reset_after: Tensor, state0: State,
                action_mask: Tree | None, actions: Tree, is_act: Tensor,
                reduction: Literal["mean_valid", "sum"]) -> Tensor: ...
    # teacher runs unroll(with_value=False); KL per decider (unit_kl) reduced per `reduction`, mean over ACT slots
    # current_lambda, step(), state_dict(), load_state_dict(), teacher, to(device): SP1 semantics
```

Train metrics (in addition to SP1's): `clip_fraction_joint`, `c_clip_frac`, `log_rho_abs_mean`, `log_rho_abs_p95`, `log_rho_joint_abs_mean`, `ess`, `deciders_valid_mean`, `deciders_valid_max`, `boot_frac`, `pad_frac`. Every loss and metric reduces over ACT slots only.

### `colosseum.sp2.learner.learner` (T4.4)

Copy of SP1 `learner/learner.py` on `sp2` types: same `learner_process(*, agent_id, algorithm_factory, trajectory_queue, weight_queues, config, stop_event, metrics_queue=None, checkpoint_queue=None, checkpoint_interval=0, resume_state=None, progress_counter=None, total_timesteps=0, weight_sync_interval=5.0)`, `collect_batch`, `make_checkpoint_payload`, `send_checkpoint`, `apply_resume_state`, `resolve_device`, `FINAL_CHECKPOINT_TIMEOUT_SEC`. `collect_batch` decodes chunk v2 payloads.

### `colosseum.sp2.coordinator.matchmaker` (T5.1)

```python
class LineupMatchmaker:
    def __init__(self, *, spec: GameSpec, agent_roles: Mapping[str, Sequence[str]], config: MatchmakingConfig,
                 checkpoints: Callable[[str], list[str]],             # agent -> checkpoint ids ("ckpt_v<N>")
                 win_rate: Callable[[str, str, str], float],         # (layout, a, b) -> P(a beats b)
                 rng: random.Random): ...
    def lineup_for(self, owner: str) -> Lineup: ...
    def playable_layouts(self, agent_id: str) -> list[str]: ...

def validate_matchmaking(spec: GameSpec, agent_roles: Mapping[str, Sequence[str]],
                         config: MatchmakingConfig) -> None: ...      # ConfigError (spec block 7 checks)
def permute_seats(spec: GameSpec, layout: str, seats: Sequence[SeatAssignment],
                  rng: random.Random) -> list[SeatAssignment]: ...    # teams with equal role composition; same-role seats
```

### `colosseum.sp2.coordinator.ratings` (T5.2)

```python
@dataclass(frozen=True)
class MemberPair:
    kind: Literal["cross", "past"]   # cross: different agent_ids; past: latest (a) vs a checkpoint of the same agent
    a: str
    b: str
    score_a: float
    weight: float                    # (1/(T-1)) / number of counted member pairs of this team pair

def member_pairs(result: MatchResult) -> list[MemberPair]: ...

class EloRating: ...                 # SP1 class plus update_weighted(pairs: Iterable[tuple[str, str, float, float]])
class WinRateTracker: ...            # SP1 class; record_pair(a, b, score, weight=1.0)
class PastWinRate: ...               # SP1 class; record(agent, score, weight=1.0)
class ScoreTracker:
    def update(self, agent_id: str, score: float) -> None: ...
    def summary(self) -> dict[str, dict[str, float]]: ...   # agent -> {"n", "mean", "ema", "ci_low", "ci_high"}
class CrossPlayTable:
    def update(self, composition: Sequence[str], score: float) -> None: ...   # multiset of agent ids
    def summary(self) -> dict[str, dict[str, float]]: ...   # "A+B" (sorted, "+"-joined) -> {"n", "mean"}

class RatingBook:
    def __init__(self, spec: GameSpec, agent_ids: Sequence[str], k_factor: float = 32.0,
                 initial_rating: float = 1200.0, past_window: int = 500): ...
    def update(self, result: MatchResult) -> None: ...
    def win_rate(self, layout: str, a: str, b: str) -> float: ...    # SP1 prior for unseen pairs
    def elo(self, layout: str, agent_id: str) -> float: ...
    def snapshot(self) -> dict[str, dict[str, Any]]: ...
        # {layout: {"elo", "win_rates", "games", "wr_vs_past", "past_games", "scores", "cross_play"}}
```

### `colosseum.sp2.coordinator.coordinator`, `.checkpoint_manager`; `colosseum.sp2.metrics.*` (T5.3)

```python
class Coordinator:
    def __init__(self, config: ColosseumConfig, spec: GameSpec, agent_roles: Mapping[str, Sequence[str]],
                 checkpoint_dir: str | Path): ...
    agent_pool: AgentPool
    checkpoint_manager: CheckpointManager
    ratings: RatingBook
    refresh_round: int
    def next_round(self) -> None: ...
    def generate_lineups(self, num_envs: int, env_offset: int) -> list[Lineup]: ...   # owner rotation as SP1
    def report_match_result(self, result: MatchResult) -> None: ...
    @property
    def match_results(self) -> list[MatchResult]: ...
    def ratings_snapshot(self) -> dict: ...                                # RatingBook.snapshot()
    def save_checkpoint_payload(self, payload: dict, meta_extra: dict | None = None) -> str: ...
        # meta.json additionally stores "roles" and "role_signature"
```

`CheckpointManager`, `load_checkpoint_dir`, `resolve_resume`, `check_model_state`, `read_weights_file`, `classify_resume_source`: SP1 behavior; `resolve_resume`/`load_checkpoint_dir` also return `"roles"` and `"role_signature"` from `meta.json`, and resume/eval compare the signature (`ConfigError` naming the path on mismatch). `EpisodeAggregator`/`MetricsHub`: SP1 behavior with per-layout and per-role breakdown (`episodes/<layout>/...`) and per-layout ratings (`ratings/<layout>/...`).

### `colosseum.sp2.launcher`, `colosseum.sp2.cli`, `colosseum.sp2.core.run_dir` (T5.4; CLI commands extended in T6.x)

`Launcher(config, run_dir, validated=False)`, `Launcher.launch() -> int`, `run_training(config_path, overrides=None) -> int`, `RunDir`: SP1 behavior (exit codes 0/1/130/143, signals, child-death detection, final checkpoints within the grace, `runs/<name>/` layout) on `sp2` types. The env's `GameSpec` comes from `registry.env_spec`, agent roles from `resolve_agent_roles`, initial lineups from `Coordinator.generate_lineups`. CLI commands and options: SP1's, plus `bc --agent A` and `eval --layout L` (repeatable); `colosseum.sp2.__main__` runs `colosseum.sp2.cli.main`.

### `colosseum.sp2.eval` (T6.1)

```python
def schedule_lineups(spec: GameSpec, layout: str, players: Mapping[str, Sequence[str]],
                     num_matches: int) -> list[Lineup]: ...      # players: name -> roles; CLI rules of spec block 8
def play_lineups(*, env_fn: Callable[[], MultiAgentEnv], models: Mapping[str, PolicyModel],
                 lineups: Sequence[Lineup], num_envs: int = 8, seed: int | None = None,
                 deterministic: bool = False, max_idle_steps: int = 1000) -> list[MatchResult]: ...
    # in-process API; Lineup.seats[*].agent_id is a key of `models`; network_id is ignored
def summarize(spec: GameSpec, results: Sequence[MatchResult]) -> EvalReport: ...
class EvalReport:
    def to_dict(self) -> dict: ...
    def text(self) -> str: ...
def evaluate(config: ColosseumConfig, agents: Mapping[str, str], *, layouts: Sequence[str] | None,
             num_matches: int, seed: int | None = None, deterministic: bool = False) -> EvalReport: ...
def load_eval_model(config: ColosseumConfig, name: str, path: str | Path) -> tuple[PolicyModel, list[str]]: ...
def wilson_interval(successes: float, n: int, z: float = Z_95) -> tuple[float, float]: ...   # SP1
def normal_interval(values: Sequence[float], z: float = Z_95) -> tuple[float, float, float]: ...   # SP1
```

### Test support (T1.4 creates `tests/game_helpers.py`; T2.3 and later extend)

```python
# toy games (MultiAgentEnv), each small and deterministic given the seed:
SoloCounterGame(length=8, truncate_at=None)            # solo; Box obs; Discrete(2); +1 for action 1
TurnTakingGame(length=6)                                # 2 seats alternate; rewards to the waiting seat; wdl
SimultaneousGame(length=5)                              # 2 seats act together
EliminationFFA(max_players=4, eliminate_at=None)        # layouts "2p","3p","4p"; scripted eliminations; rank
TeamDeadTeammateGame()                                  # "2v2"; one seat stops acting but keeps rewards until the end
UnitsGame(max_units=4, uint8_grid=True)                 # Dict obs (uint8 grid + entity list + mask), Units actions, births/deaths
AsymmetricGame()                                        # roles "hunter"/"prey" with different spaces
CoopGame(size=2)                                        # one team
GlobalStateGame()                                       # role with global_state_space
# models and policies:
CORE_KINDS = ("none", "lstm", "gru", "attention")
def make_test_model(role: RoleSpec, core: str = "none", hidden: int = 16) -> PolicyModel: ...   # T2.3
class RandomPolicy(PolicyModel): ...                    # T2.3; uniform over legal actions; stateless
def make_test_config(game: str, **overrides) -> ColosseumConfig: ...   # T5.4; tiny runs for integration tests
```

---

## Contract amendments (reconciled after all plan parts were written)

The four parts were written in parallel. Each ends with its own `## Contract notes`; those additions are accepted unless an item below overrides them. **The items below are binding** over the parts and over the contract above. Prefixes: A = `01-…`, B = `02-…`, C = `03-…`, D = `04-…`. Line numbers refer to the committed part files.

**Formats and names**
- **R1 — `ratings.json` shape (C wins).** `{"env_steps": N, "layouts": {<layout>: {"outcome_kind", "elo", "win_rates", "games", "wr_vs_past", "past_games", "scores", "cross_play", "role_win_rates"}}}`; the `ratings` records in `metrics.jsonl` carry the same `layouts` object. Part D changes: `"1v2" in run.ratings()` → `"1v2" in run.ratings()["layouts"]` (D:1683); `run.ratings()["coop2"]["cross_play"]` → `run.ratings()["layouts"]["coop2"]["cross_play"]` (D:1690, D:3237); D:1387, D contract note 7 and T8.6's README rows describe the nested shape.
- **R2 — episode metric keys (C wins).** Episode breakdowns are `episodes/<agent>/by_layout/<layout>/<role>/...` (record field `by_layout`), not `episodes/<layout>/...` as written in the T5.3 contract paragraph above.
- **R3 — `make_test_config` (C wins).** `tests/game_helpers.py::make_test_config(game: str, **sections) -> ColosseumConfig` is created in **T5.3** (not T5.4); `game` is a `TOY_GAMES` name; `sections` are top-level config dicts deep-merged onto the tiny defaults, except `agents`, which replaces.
- **R4 — `put_row` stays.** Part A's `tree_assign` keeps rejecting a bare-array destination; Part B's `colosseum.sp2.worker.buffers.put_row(dst, idx, src)` is a contract name for row writes into buffer trees. Both were executed as written; unifying them is not worth touching verified code.
- **R5 — `MatchRunner(context=...)`** must end with `", "` (B contract note); Part D's `demo_checks.random_matches` passes `context="demo, "`.
- **R6 — queue helpers** are public in `colosseum.core.ipc` after T0.1 (`QueueReader`, `release_command_queues`, `queue_depths`); the `sp2` launcher imports them.
- **R7 — `metrics/jsonl.py` gets an `sp2` copy** (C): its required-key tables describe record shapes that change. The "Reused unchanged" list above is amended accordingly.
- **R8 — `__main__` docstring.** `src/colosseum/sp2/__main__.py` keeps the SP1 docstring line "(used by the Docker/K8s entrypoints)", so the overlay does not drop it.

**Algorithm**
- **R9 — mode resolution at K == 1** (amends B's `resolve_modes` ruling). With one decider:
  - `auto` values resolve to `ratio_mode=joint`, `unit_trace=joint`, `entropy_reduction=sum`;
  - an explicit `ratio_mode: per_unit` collapses to `joint` with a one-time INFO log ("one decider: per_unit equals joint except for the rho factor; using joint");
  - explicit `unit_trace: geo_mean` and `entropy_reduction: mean_valid` collapse silently to `joint`/`sum` (identical at K = 1);
  - an explicit `unit_trace: none` is honoured (ρ = c = 1 is defined independently of K).
  T4.2 updates `test_modes_resolve_auto_and_collapse_for_one_decider` accordingly.
- **R10 — diagnostics (T4.2).** Add `log_rho_joint_abs_p95` to the train metrics and `DIAGNOSTICS`, plus a zero-lag test pinning `ess == 1`, `clip_fraction == clip_fraction_joint == 0`, `rho_clip_frac == c_clip_frac == 0`.
- **R11 — `unit_trace` default change in T8.4.** If the ruling changes the default, T8.4 also modifies `tests/unit/test_appo_v2.py` (the auto-resolution assertion; add it to T8.4's **Files**). T8.4's detection check becomes `grep -n "geo_mean" tests/unit/test_appo_v2.py`; T4.2's test is the pin, so no separate `test_unit_trace_default.py` is created.

**Coverage additions (gaps found by the cross-check)**
- **R12 — GPU successors.** T7.3 deletes SP1's `gpu` tests in `test_algorithm_state.py`, `test_appo_metrics.py`, `test_bc_trainer.py`, `test_kickstart_kl.py`. Add `gpu`-marked successors: in **T4.2** (`test_appo_v2.py`: GradScaler round trip, `state_dict` round trip on CUDA, an AMP step with a Dict + `uint8` observation and `global_state`, `pin_memory`; `test_kickstart_v2.py`: kickstart on CUDA with masks and an LSTM core) and in **T6.3** (`test_sp2_bc_trainer.py`: BC with tree data on CUDA). T7.3's coverage gate gets a "GPU successor" column. T8.6 rebuilds the `docs/GPU_CHECKS.md` table from `.venv/bin/python -m pytest -m gpu --collect-only -q` instead of editing rows.
- **R13 — episode metrics per layout (T5.3).** `by_layout[layout][role]` also carries `length_mean` and W/D/L by opponent type (`latest`/`past`/`arena`), with assertions.
- **R14 — env contract errors through `MatchRunner` (T3.3).** One parametrized test per `EpisodeTracker` error of spec block 1, driven through `MatchRunner` with `ScriptedGame`/`TickGame`, including idle overflow via `MatchRunner(max_idle_steps=...)`. **T5.4** asserts that `env.max_idle_steps` reaches `rollout_worker_process`.
- **R15 — elimination and truncation in the same step (T3.4).** `test_chunk_v2_rules.py` adds a case where a seat is terminated in the truncation step: its open ACT becomes terminal and no BOOT is written for it, while the other live seats get BOOT(final_obs).
- **R16 — `global_state` reaches only the value path, end to end (T4.3).** With `GlobalStateGame`: perturb the chunks' `global_state`; learner log-probs are unchanged and values change.

**Overlay (T7.3)**
- **R17 — deferred gate rows.** The coverage-gate rows for `learning/test_ttt_slow.py` + `ttt_eval.py` (successor T8.3) and `unit/test_examples.py` (successor T8.5) are "deferred to T8.3 / T8.5 (accepted gap)"; T7.3 still deletes those files and the old `space_miners` / `composite_action` `env.py`, `networks.py` and configs.
- **R18 — benchmark workload.** T7.3 replaces `NUM_PLAYERS` / `"num_players"` in `scripts/bench_throughput.py`'s `workload` dict with `"layout": "2p"` (and updates its unit test), so T8.4's "after" JSON has no SP1-only field.

**Rulings taken at the plan stage** (move them into the SP2 acceptance report's rulings ledger in T8.7)
- **PR-1 — K == 1 mode collapse** (R9). Why: keeps spec block 6's "при K = 1 все режимы дают один результат" exact and SP1 behaviour for every game without units. Cost if wrong: a `Units(1, …)` agent configured with `per_unit` trains with the joint loss (logged once).
- **PR-2 — a team's core always takes one seat; only the other seats follow `teammates`** (C's reading of spec block 7 item 4, "Остальные места команды"). The core's seat is drawn uniformly among the team's seats of a role the core plays; `shuffle_seats` is applied afterwards. Why: the owner always collects in its own team, which ownership rotation and the `coop_buttons` criterion need. Cost if wrong: `mixed` never fills the core's own seat with another agent.
- **PR-3 — distributed layouts are fixed per worker env** (see Deviations). Why: no coordinator in distributed mode until SP5. Cost if wrong: distributed runs of multi-layout games see a fixed layout mix decided at worker start.
- **PR-4 — `put_row` kept next to `tree_assign`** (R4). Cost if wrong: a small duplicate helper.

---

## Cross-part execution notes

- **Plan code vs. current code.** Every part was written against the code its author expected. When a task's "before" block does not match the repository, keep the intent, the contract and the task's tests, adapt the edit, and say so in the commit body. Never weaken a test to make it pass.
- **Copies from SP1.** When a task says "copy SP1 module X", copy the current file (after T0.1), then make only the listed changes. Keep SP1 docstrings where they still hold; fix them where they do not.
- **Contract amendments** (reconciled after all parts are written) are appended to this file and are binding over the parts' own "Contract notes".
- **Shared test kit.** Reuse `tests/game_helpers.py` and `tests/contract/game_harness.py`; do not create near-duplicates with different names.

---

## Task index and execution order

Execute **strictly in the order of this table** (amended after the cross-check; the order differs from the numbering at T6.2/T6.1, T7.2/T7.1 and T8.5/T8.3). Every task also lists its dependencies in its **Interfaces** block.

| Order | ID | Task | Depends on |
|---|---|---|---|
| 1 | T0.1 | SP1 residuals: second Ctrl+C, `bc` unreadable data, queue helpers to `core/ipc.py` | — |
| 2 | T1.1 | Tree utilities | T0.1 |
| 3 | T1.2 | `Units` space | T1.1 |
| 4 | T1.3 | `ObsSpec`, `ActionSpec`, mask rules | T1.2 |
| 5 | T1.4 | `GameSpec`, `MultiAgentEnv`, `StepResult`, `Outcome`, `resolve_outcome`, toy games | T1.3 |
| 6 | T1.5 | `EpisodeTracker`: seat lifecycle and contract checks | T1.4 |
| 7 | T1.6 | `VectorEnv`, `SubprocessVectorEnv` (no auto-reset) | T1.4 |
| 8 | T1.7 | Config v2 | T0.1 |
| 9 | T2.1 | Leaf distributions, `TreeDist`, `make_distribution` | T1.3 |
| 10 | T2.2 | `UnitsDist` (masks, `only_if`, deciders) | T2.1 |
| 11 | T2.3 | Model protocol, base classes, `ComposedModel` + critic, heads, `NormalizeObs`, test models | T2.2, T1.4, T1.5 |
| 12 | T2.4 | Roles resolution, `build_model`, `make_env`, `env_spec` | T2.3, T1.7, T1.4 |
| 13 | T3.1 | `sp2` types: chunk v2 + payloads, lineups, results, commands | T1.1 |
| 14 | T3.2 | Buffers with `act`/`boot`/`pad` slots and parking | T3.1, T1.3 |
| 15 | T3.3 | `MatchRunner` (+ R14) | T1.5, T1.6, T2.3, T3.1, T3.2 |
| 16 | T3.4 | `RolloutLoop` + worker process; chunk-structure and lifecycle contract tests (+ R15) | T3.2, T3.3, T2.4 |
| 17 | T4.1 | V-trace over slots | T3.1 |
| 18 | T4.2 | APPO v2 (unit modes, reductions, diagnostics) + kickstart v2 (+ R9, R10, R12) | T4.1, T3.1, T3.2, T2.3, T1.7 |
| 19 | T4.3 | Contract: learner reproduces worker log-probs (4 cores), bootstrap by learner (+ R16) | T3.4, T4.2 |
| 20 | T4.4 | Learner process on chunk v2 | T4.2, T3.4 |
| 21 | T5.1 | `LineupMatchmaker`, `validate_matchmaking`, `permute_seats` | T1.4, T1.7, T3.1 |
| 22 | T5.2 | Ratings per layout: team-pair ELO, win rates, past, scores, cross-play | T3.1, T1.4 |
| 23 | T5.3 | Coordinator, checkpoint manager (roles in meta), metrics aggregator/hub, `make_test_config` (+ R2, R3, R13) | T5.1, T5.2, T4.4, T2.4 |
| 24 | T5.4 | Launcher, run dir, CLI `train`; integration runs on toy games (+ R14) | T3.4, T4.4, T5.3, T2.4 |
| 25 | T6.2 | `validate_config`, CLI `validate` | T5.4, T5.1, T1.5 |
| 26 | T6.1 | Eval engine, reports, CLI `eval` | T3.3, T5.4, T6.2 |
| 27 | T6.3 | Offline BC on trees, CLI `bc --agent` (+ R12) | T2.4, T5.4, T6.2 |
| 28 | T6.4 | Distributed mode on chunk v2, CLI distributed commands | T4.4, T3.4, T5.4, T6.2 |
| 29 | T7.2 | New tic-tac-toe on the contract; SP1 fast learning tests on the new pipeline | T5.4, T6.2, T6.3 |
| 30 | T7.1 | Port SP1 integration coverage to the `sp2` CLI | T6.1–T6.4, T7.2 |
| 31 | T7.3 | Overlay: delete legacy, move `sp2` into place, rename, bench switch (+ R12, R17, R18) | T7.1, T7.2 |
| 32 | T8.1 | Demo envs: `coin_grid`, `unit_harvest`, `tron` (+ models, configs, smoke) | T7.3 |
| 33 | T8.2 | Demo envs: `team_tag`, `predator_prey`, `coop_buttons` (+ models, configs, smoke; R1) | T8.1 |
| 34 | T8.5 | Reference examples: `space_miners`, `chase` on the new contract | T8.1 |
| 35 | T8.3 | Learning tests: fast (units bandit, coop bandit) and slow (criterion 3; R1) | T8.1, T8.2 |
| 36 | T8.4 | Units experiment, throughput "after", `global_state` size (+ R11) | T8.3 |
| 37 | T8.6 | Docs: `ENV_GUIDE.md`, README, CLAUDE.md, `GPU_CHECKS.md` (+ R1, R12) | T8.4, T8.5 |
| 38 | T8.7 | Acceptance run against spec section 3; rulings PR-1..PR-4 go into the report | T8.6 |

Parallelism (optional): T1.6 ∥ T1.7; T5.1 ∥ T5.2; T8.2 ∥ T8.5. Tasks that touch `src/colosseum/sp2/cli.py` (T5.4, T6.1–T6.4) run in table order.
