# SP1: Foundation and Stabilization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make single-machine training in Colosseum correct, observable and robust on the current env API. Along the way, replace the LSTM/GRU-specific network code with a general stateful-model protocol.

**Architecture:**
- The IMPALA/APPO core is kept.
- The worker loop is extracted into an in-process testable `RolloutLoop`.
- Networks go behind a `PolicyModel` protocol with an opaque `State` pytree (cores: none / LSTM / GRU / window-attention).
- Data crosses process boundaries only as numpy payloads.
- Transitions stay "open" until the slot acts again, which fixes turn-based rewards and removes the bootstrap forward pass.
- League fixes, run directory, metrics and lifecycle are layered on top.

**Tech Stack:** Python 3.12, PyTorch ≥2.6 (CPU for development), numpy, gymnasium, pydantic v2, click, pytest (+pytest-timeout), ruff, uv, GitHub Actions. grpcio ≥1.78 / protobuf ≥6.31 are optional.

**Spec:** `docs/superpowers/specs/2026-10-08-sp1-stabilization-design.md`. Read it before starting any task. Finding IDs such as `R2-04` refer to `review/code/*.md`.

## Plan files

| File | Blocks | Tasks |
|---|---|---|
| `00-overview.md` (this file) | constraints, file map, interface contract, task index, order | — |
| `01-tooling-and-model.md` | 0 Tooling, 1 Model protocol | T0.1–T0.5, T1.1–T1.7 |
| `02-dataflow-and-transitions.md` | 2 Data flow, 3 Worker transitions | T2.1–T2.6, T3.1–T3.4 |
| `03-algorithm-and-eval.md` | 4 Algorithm, 7 Eval | T4.1–T4.6, T7.1–T7.2 |
| `04-league-observability-docs.md` | 5 League, 6 Observability/config/lifecycle, 8 Docs/examples/acceptance | T5.1–T5.4, T6.1–T6.5, T8.1–T8.4 |

## Global Constraints

These apply to every task.

- **Python and dependencies:** Python 3.12 (`requires-python = ">=3.11"` stays); `torch>=2.6`; `grpcio>=1.78`, `protobuf>=6.31`, `grpcio-tools>=1.78` in the optional extra `grpc`; `wandb` only in the optional extra `wandb`.
- **Environment:**
  - Run everything with `.venv/bin/python` (created by `scripts/setup-dev.sh` in T0.1). Never install into system Python.
  - On this machine (8 cores, 11 GB RAM, no GPU, ~600 KB/s network), CPU torch must come only from `--index-url https://download.pytorch.org/whl/cpu`, without `--extra-index-url`.
- **No backward compatibility:** config keys, checkpoint layout and APIs may change freely. Every example, config and test is updated in the same task that changes what it depends on.
- **Inter-process data:** nothing that crosses a process boundary (mp.Queue, pipes, gRPC) may contain `torch.Tensor`. Use the numpy payload forms defined in the contract below.
- **Tests:**
  - Write files only under pytest's `tmp_path` (or a run dir inside it).
  - GPU-only tests get `@pytest.mark.gpu` and are skipped without CUDA.
  - Tests longer than ~20 s get `@pytest.mark.slow`.
  - Every test has a pytest-timeout (default 300 s, configured in T0.2).
  - Use at most 2 worker processes in tests.
- **Definition of done for every task:** the task's new tests pass, and `.venv/bin/python -m pytest -m "not gpu and not slow" -q` passes in full.
- **Language:**
  - code, comments, docstrings and log messages in English;
  - `README.md` and `CLAUDE.md` stay in English;
  - new project docs `docs/GPU_CHECKS.md` and `docs/benchmarks.md` in Russian.
- **Commits:**
  - Conventional prefixes (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `perf:`).
  - No attribution or co-author lines.
  - At least one commit per task, on branch `sp1-stabilization`.
- **Style:** follow the existing code style (type hints, dataclasses, pydantic models, module-level `logger = logging.getLogger(__name__)`). `ruff check` must pass with the config from T0.3.

## Deviation from the spec

The spec places the extraction of `RolloutLoop` in block 3. The plan moves it to the end of block 0 (T0.5) as a pure refactor with characterization tests. Every later block needs to drive the real worker loop in-process for contract tests, so the extraction has to happen first.

---

## File structure (after SP1)

Create:
- `scripts/setup-dev.sh` — dev environment (uv, .venv, CPU/GPU torch).
- `scripts/bench_throughput.py` — throughput benchmark (updates/s, env steps/s for 1/2/4 workers).
- `.github/workflows/ci.yml` — ruff + pytest (not gpu, not slow) on CPU.
- `src/colosseum/networks/state.py` — `State` pytree utilities.
- `src/colosseum/networks/model.py` — `PolicyModel`, `StepOutput`, `UnrollOutput`, `ActOutput`, `act()`.
- `src/colosseum/networks/cores.py` — `Core`, `NoCore`, `LSTMCore`, `GRUCore`, `WindowAttentionCore`.
- `src/colosseum/networks/composed.py` — `ComposedModel`.
- `src/colosseum/worker/rollout_loop.py` — `RolloutLoop` (in-process worker loop, callbacks for I/O).
- `src/colosseum/worker/slots.py` — `SlotTrack`, `BufferPool` (agent-owned buffers, parking).
- `src/colosseum/core/ipc.py` — `put_latest`, `drain_latest`, `SharedCounter`.
- `src/colosseum/core/errors.py` — `EnvContractError`, `ConfigError`.
- `src/colosseum/core/run_dir.py` — `RunDir`.
- `src/colosseum/utils/__init__.py`, `src/colosseum/utils/logging.py` — `setup_process_logging`.
- `src/colosseum/metrics/jsonl.py` — `MetricsWriter`.
- `src/colosseum/metrics/aggregator.py` — `EpisodeAggregator`, `SystemStats`.
- `src/colosseum/metrics/console.py` — `ConsoleReporter`.
- `tests/unit/`, `tests/contract/`, `tests/integration/`, `tests/learning/` — new test layout. `tests/helpers.py` stays as the shared toy envs and models module.
- `tests/contract/harness.py` — helpers to drive `RolloutLoop` in-process and feed APPO.
- `docs/GPU_CHECKS.md`, `docs/benchmarks.md`.

Delete:
- `src/colosseum/networks/actor_critic.py` (replaced by `model.py`, `composed.py`, `cores.py`; removed in T1.7).
- `serve-trajectory` CLI command. It is a dead end: R3-24, removed in T6.5.

Modify (main ones):
- `core/types.py`, `core/config.py`, `core/registry.py`, `core/action_spec.py`, `core/outcomes.py`
- `algorithms/base.py`, `algorithms/appo.py`, `algorithms/vtrace.py`
- `networks/distributions.py`, `networks/normalization.py`, `networks/base.py`
- `worker/rollout_worker.py` — thin process wrapper around `RolloutLoop`
- `envs/vec_env.py`, `envs/subproc_vec_env.py`
- `learner/learner.py`, `launcher.py`, `distributed.py`, `cli.py`, `eval.py`
- `bc/offline_bc.py`, `bc/kickstart.py`
- `coordinator/coordinator.py`, `coordinator/matchmaker.py`, `coordinator/ratings.py`, `coordinator/checkpoint_manager.py`
- `metrics/wandb_logger.py`
- `transport/serialization.py`, `transport/grpc_transport.py`, `weight_store/*.py` (payload adaptation only)
- `examples/**`, `configs/examples/*.yaml`, `README.md`, `CLAUDE.md`, `pyproject.toml`

---

## Interface contract

These are the cross-task names and signatures. A task that produces a name must implement it exactly as written here. A task that consumes it must use it exactly. If a task needs to deviate, it must update this section in the same commit and say so in the commit message.

### `colosseum.networks.state` (T1.1)

```python
State = Any  # pytree: None | Tensor | tuple | list | dict[str, ...]; every Tensor leaf has batch on dim 0

def tree_map(fn: Callable[[Tensor], Tensor], state: State) -> State: ...
def tree_leaves(state: State) -> list[Tensor]: ...
def batch_size_of(state: State) -> int | None: ...                      # None if no leaves
def slice_batch(state: State, idx: int | Sequence[int] | Tensor) -> State: ...   # int idx keeps dim: [1,...]
def cat_batch(states: Sequence[State]) -> State: ...                     # concatenates on dim 0; all-None -> None
def where_done(done: Tensor, reset: State, state: State) -> State: ...   # done [B] bool; rows where done take reset
def state_to_numpy(state: State) -> Any: ...                             # same structure, np.ndarray leaves
def state_from_numpy(obj: Any, device: str | torch.device = "cpu") -> State: ...
def state_to(state: State, device: str | torch.device) -> State: ...
```

### `colosseum.networks.model` (T1.2)

```python
class StepOutput(NamedTuple):
    dist: Distribution      # batch [B]
    value: Tensor           # [B]
    state: State

class UnrollOutput(NamedTuple):
    dist: Distribution      # batch [T*B], time-major flatten (index t*B + b)
    value: Tensor           # [T*B], time-major flatten

class ActOutput(NamedTuple):
    actions: Tensor         # [B, *action_shape] (flat action layout from ActionSpec)
    log_probs: Tensor       # [B]
    values: Tensor          # [B]
    state: State

class PolicyModel(nn.Module, ABC):
    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State: ...  # default None
    @abstractmethod
    def step(self, obs: Tensor, state: State, action_mask: Tensor | None = None) -> StepOutput: ...
    def unroll(self, obs: Tensor, state0: State, dones: Tensor,
               action_mask: Tensor | None = None) -> UnrollOutput: ...
        # obs [T,B,...], dones [T,B] bool (dones[t]: episode ended AFTER step t), mask [T,B,A] | None
        # default: python loop of step(); after step t: state = reset_state(state, dones[t])
    def reset_state(self, state: State, done: Tensor) -> State: ...   # default where_done(done, initial_state(B), state)
    def update_normalizers(self, obs: Tensor) -> None: ...            # no-op until T4.2; from T4.2: update() every NormalizeObs submodule
    @property
    def is_stateful(self) -> bool: ...                                # initial_state(1) is not None

def act(model: PolicyModel, obs: Tensor, state: State,
        action_mask: Tensor | None = None, deterministic: bool = False) -> ActOutput: ...
```

### `colosseum.networks.cores` (T1.3)

```python
class Core(nn.Module, ABC):
    input_dim: int
    output_dim: int
    def initial_state(self, batch_size: int, device="cpu") -> State: ...
    @abstractmethod
    def step(self, x: Tensor, state: State) -> tuple[Tensor, State]: ...         # x [B,D] -> [B,O]
    def unroll(self, x: Tensor, state0: State, dones: Tensor) -> Tensor: ...      # x [T,B,D] -> [T,B,O]; default loop + reset
    def reset_state(self, state: State, done: Tensor) -> State: ...

class NoCore(Core):               # __init__(self, input_dim: int); state None; output_dim = input_dim
class LSTMCore(Core):             # __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1)
                                  # state {"h": [B,L,H], "c": [B,L,H]}
class GRUCore(Core):              # __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1)
                                  # state {"h": [B,L,H]}
class WindowAttentionCore(Core):  # __init__(self, input_dim: int, d_model: int = 64, window: int = 16,
                                  #          num_heads: int = 4, num_layers: int = 1)
                                  # state {"mem": [B,window,d_model] float, "len": [B] long}
```

### `colosseum.networks.composed` (T1.4)

```python
class ComposedModel(PolicyModel):
    def __init__(self, encoder: BaseEncoder, core: Core, policy_head: BasePolicy, value_head: BaseValue): ...
```

### `colosseum.core.registry` (T1.4)

```python
def import_class(dotted_path: str) -> type: ...           # unchanged
def build_model(config: ColosseumConfig) -> PolicyModel: ...
    # model_class set -> import_class(model_class)(**networks.kwargs)
    # else encoder(**kwargs) -> core(input_dim=encoder.latent_dim, **core.kwargs) (NoCore if core is None)
    #      -> heads: pass in_dim=core.output_dim if the constructor accepts `in_dim`, plus **kwargs
def validate_config(config: ColosseumConfig) -> None: ...  # raises ConfigError with a precise message
```

### `colosseum.core.config` (T1.4, T2.1, T2.5, T4.1, T4.3, T5.1, T6.1)

```python
class CoreConfig(BaseModel):                 # extra="forbid", populate_by_name=True
    class_path: str = Field(alias="class")
    kwargs: dict[str, Any] = {}

class NetworkConfig(BaseModel):
    model_class: str | None = None
    encoder_class: str | None = None
    core: CoreConfig | None = None
    policy_class: str | None = None
    value_class: str | None = None
    kwargs: dict[str, Any] = {}
    # validator: either model_class, or all of encoder_class/policy_class/value_class
    # removed: recurrent_type, recurrent_hidden_size, recurrent_num_layers

RolloutConfig.torch_threads: int = 1                         # T2.1 (ge=1)
LearnerConfig.torch_threads: int | None = None               # T2.1
AlgorithmConfig.vtrace_lambda: float = 1.0                   # T4.1 (ge=0, le=1); gae_lambda removed
TrainingConfig.kickstart_kl: Literal["forward", "reverse"] = "forward"   # T4.3
SelfPlayConfig.shuffle_seats: bool = True                    # T5.1
MetricsConfig.console_interval_sec: float = 10.0             # T6.3
class RunConfig(BaseModel): name: str | None = None; dir: str = "runs"   # T6.1; ColosseumConfig.run
CheckpointConfig: field `dir` removed in T6.2 (checkpoints live in the run dir), `save_optimizer` kept
ColosseumConfig.agents: dict[str, AgentOverride]  # AgentOverride.networks/algorithm/learner: dict[str, Any] | None
                                                  # deep-merged onto the global section BEFORE validation (T6.1)

def parse_override_value(raw: str) -> Any: ...                     # yaml.safe_load semantics (T6.1)
def apply_overrides(data: dict, overrides: dict[str, Any]) -> dict: ...   # unknown path -> ConfigError (T6.1)
def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ColosseumConfig: ...  # (T6.1)
ColosseumConfig.get_agent_config(agent_id: str) -> ColosseumConfig   # deep-merged effective config
ColosseumConfig.get_trainable_agent_ids() -> list[str]               # unchanged
```

### `colosseum.core.types` (T1.5, T2.2, T3.4)

```python
@dataclass
class TrajectoryChunk:
    agent_id: str
    observations: Tensor        # [T, *obs_shape]
    actions: Tensor             # [T, *action_shape]
    action_log_probs: Tensor    # [T]
    rewards: Tensor             # [T]  (truncation already includes gamma*V(final_obs), T3.3)
    dones: Tensor               # [T] bool: transition t is the last of its episode
    values: Tensor              # [T]
    bootstrap_value: Tensor     # scalar: value at the slot's next action, or 0.0 if last transition is terminal
    behavior_policy_version: int   # version when the chunk's FIRST transition was recorded
    initial_state: State = None    # model state before the first transition; leaves [1, ...]
    action_masks: Tensor | None = None   # [T, mask_size]
    @property
    def chunk_length(self) -> int: ...
    def to(self, device) -> "TrajectoryChunk": ...
    def pin_memory(self) -> "TrajectoryChunk": ...
    def to_payload(self) -> dict[str, Any]: ...          # numpy + primitives only (T2.2)
    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TrajectoryChunk": ...   # (T2.2)

@dataclass
class WeightPayload:                                     # (T2.2)
    agent_id: str
    policy_version: int
    state_dict: dict[str, np.ndarray]
    @classmethod
    def from_model(cls, agent_id: str, policy_version: int, model: nn.Module) -> "WeightPayload": ...
    def to_torch_state_dict(self) -> dict[str, Tensor]: ...

@dataclass
class SeatResult:                                        # (T3.4)
    seat: int
    agent_id: str
    network_id: str             # "latest" | "ckpt_v<N>"
    outcome: float              # in [0,1]; from core.outcomes (env rank/outcome, else reward-based)
    reward: float               # episode return of this seat
    rank: int | None = None

@dataclass
class MatchResult:                                       # (T3.4) replaces the dict-keyed version
    match_id: str
    seats: list[SeatResult]
    episode_length: int

@dataclass
class WorkerCommand:            # new_checkpoints become numpy state dicts (T2.2)
    slot_agent_map: list[list[str]]
    slot_network_map: list[list[str]]
    collect_mask: list[list[bool]]
    new_checkpoints: dict[str, dict[str, dict[str, np.ndarray]]]
```

`PlayerSlot` and `MatchConfig` are unchanged.

### `colosseum.core.ipc` (T2.3, T2.5)

```python
def put_latest(q: mp.Queue, item: Any, timeout: float = 1.0) -> bool: ...   # maxsize=1 queue; evicts a stale/in-flight item (see amendment B1)
def drain_latest(q: mp.Queue) -> Any | None: ...         # returns the newest available item or None
class SharedCounter:                                     # wraps mp.Value("q")
    def __init__(self, ctx: mp.context.BaseContext | None = None): ...
    def add(self, n: int) -> None: ...
    @property
    def value(self) -> int: ...
```

### `colosseum.core.errors` (T1.4, T3.2)

```python
class ColosseumError(Exception): ...
class ConfigError(ColosseumError): ...
class EnvContractError(ColosseumError): ...
```

### `colosseum.worker.rollout_loop` (T0.5; extended in T1.6, T2.x, T3.x)

```python
@dataclass
class LoopIO:                                   # all I/O goes through callbacks -> in-process testable
    send_chunk: Callable[[TrajectoryChunk], None]
    poll_weights: Callable[[str], WeightPayload | None]    # agent_id -> newest payload or None
    report_result: Callable[[MatchResult], None] | None = None
    poll_command: Callable[[], WorkerCommand | None] | None = None
    add_env_steps: Callable[[int], None] | None = None      # global budget counter (T2.5)

class RolloutLoop:
    def __init__(self, *, worker_id: int, env_fn: Callable[[], BaseEnv], num_envs: int, chunk_length: int,
                 agent_ids: list[str], model_factories: dict[str, Callable[[], PolicyModel]],
                 io: LoopIO, gamma: float | dict[str, float] = 0.99, weight_sync_interval: float = 5.0,
                 slot_agent_map: list[list[str]] | None = None,
                 slot_network_map: list[list[str]] | None = None,
                 collect_mask: list[list[bool]] | None = None,
                 checkpoint_state_dicts_by_agent: dict[str, dict[str, dict[str, np.ndarray]]] | None = None,
                 seed: int | None = None, vec_env_kind: str = "sync",
                 subproc_workers: int | None = None) -> None: ...
    def step(self) -> int: ...      # one vectorized env step for all envs; returns env steps taken (= num_envs)
    def sync_weights(self) -> None: ...
    def run(self, should_stop: Callable[[], bool], max_env_steps: int = 0) -> None: ...
    def close(self) -> None: ...
    @property
    def stats(self) -> dict[str, int]: ...   # chunks_sent, env_steps, parked_buffers, ...
```

### `colosseum.worker.slots` (T2.6, T3.1)

```python
class BufferPool:      # per-agent pool of RolloutBuffer; acquire() prefers parked partial buffers
    def acquire(self, agent_id: str) -> RolloutBuffer: ...
    def park(self, agent_id: str, buf: RolloutBuffer) -> None: ...
    def parked_count(self) -> int: ...

@dataclass
class SlotTrack:
    buffer: RolloutBuffer | None
    has_open: bool = False
    pending_reward: float = 0.0
    state: State = None
```

### `colosseum.algorithms.base` (T1.5, T2.5, T4.6)

```python
class BaseAlgorithm(ABC):
    @property
    def model(self) -> PolicyModel: ...
    @property
    def policy_version(self) -> int: ...
    @abstractmethod
    def train_step(self, chunks: list[TrajectoryChunk]) -> dict[str, float]: ...
    def set_progress(self, progress: float) -> None: ...     # 0..1 share of the global budget; drives LR schedule
    def state_dict(self) -> dict[str, Any]: ...              # optimizer, scheduler, scaler, kickstart_step,
                                                             # policy_version, consumed_samples (deep CPU copies)
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
    @property
    def is_off_policy(self) -> bool: ...                     # unchanged
    def create_replay_buffer(self, capacity: int) -> Any | None: ...   # unchanged
```

`APPO.__init__(self, model: PolicyModel, config: AlgorithmConfig, device="cpu", pin_memory=False, kickstart: KickstartLoss | None = None)`. `setup_lr_schedule(total_steps)` is removed: the LR is a function of `progress`.

### `colosseum.learner.learner` (T2.4, T2.5)

```python
def collect_batch(q, batch_size: int, stop_event, poll_interval: float = 0.5) -> list[TrajectoryChunk] | None: ...
    # blocks until exactly batch_size chunks (payloads decoded) or stop -> None
def learner_process(*, agent_id: str, algorithm_factory: Callable[[], BaseAlgorithm],
                    trajectory_queue, weight_queues: list, config: LearnerConfig, stop_event,
                    metrics_queue=None, checkpoint_queue=None, checkpoint_interval: int = 0,
                    resume_state: dict | None = None, progress_counter: SharedCounter | None = None,
                    total_timesteps: int = 0, run_dir: str | None = None) -> None: ...
```

### `colosseum.coordinator` (T5.1–T5.3)

```python
Coordinator.generate_match_configs(self, num_envs: int, env_offset: int) -> list[MatchConfig]
Coordinator.report_match_result(self, result: MatchResult) -> None
Coordinator.ratings_snapshot(self) -> dict   # {"elo": {...}, "win_rates": {a: {b: wr}}, "wr_vs_past": {agent: wr}}
WinRateTracker.record_pair(self, a: str, b: str, score_a: float) -> None    # score in {0, 0.5, 1}
EloRating.update_pair(self, a: str, b: str, score_a: float, k_scale: float = 1.0) -> None

class CheckpointManager:
    def __init__(self, base_dir: str | Path, pool_size: int = 20): ...
    def save(self, agent_id: str, policy_version: int, model_state: dict[str, np.ndarray],
             trainer_state: dict | None = None, meta_extra: dict | None = None) -> str: ...   # atomic
    def load_model(self, agent_id: str, checkpoint_id: str) -> dict[str, np.ndarray]: ...
    def load_trainer_state(self, agent_id: str, checkpoint_id: str) -> dict | None: ...
    def list_checkpoints(self, agent_id: str) -> list[CheckpointInfo]: ...
    def latest(self, agent_id: str) -> CheckpointInfo | None: ...

def resolve_resume(resume_from: str, agent_id: str) -> dict | None: ...
    # -> {"model_state": ..., "trainer_state": ... | None, "policy_version": int} (in checkpoint_manager.py)
```

### Run dir, logging, metrics (T6.2, T6.3)

```python
class RunDir:                                   # colosseum.core.run_dir
    root: Path; logs: Path; checkpoints: Path; metrics_path: Path; ratings_path: Path
    @classmethod
    def create(cls, config: ColosseumConfig, config_path: str | Path | None = None) -> "RunDir": ...
    def write_resolved_config(self, config: ColosseumConfig) -> Path: ...

def setup_process_logging(log_dir: str | Path | None, process_name: str,
                          console_level: int = logging.WARNING) -> None: ...   # colosseum.utils.logging

class MetricsWriter:                            # colosseum.metrics.jsonl
    def __init__(self, path: str | Path): ...
    def write(self, kind: str, **fields: Any) -> None: ...   # one JSON line {"ts", "kind", **fields}
    def close(self) -> None: ...
```

---

## Contract amendments (reconciled after all plan parts were written)

The four plan parts were written in parallel. Each part ends with its own "Contract notes" section, and the decisions below are binding when they differ from those notes or from the contract above. Prefixes: A = `01-…`, B = `02-…`, C = `03-…`, D = `04-…`.

**Model and algorithm**
- **A1 — T1.5 / T1.6 scope.** T1.5 switches the whole training path in one task: APPO, chunk format, learner, `RolloutLoop` state handling, launcher and distributed mode. T1.6 adds the four-core contract test, the mutation check and `harness.learner_eval`. No signatures change.
- **A2 — default `unroll`.**
  - Stateless models are evaluated with a single batched `step`; stateful models use the step loop.
  - New classmethod `Distribution.cat(dists) -> Distribution`, implemented for `CategoricalDist`, `DiagGaussianDist` and `CompositeDist`.
  - `ComposedModel` overrides `unroll` and never needs `cat`.
- **A3 — `ComposedModel` submodule names** are `encoder`, `core`, `policy`, `value`.
- **A4 — names introduced by part A:**
  - `APPO.evaluate_chunks(chunks) -> (log_probs [T*B], values [T*B])`;
  - `RolloutBuffer.initial_state` / `set_initial_state`;
  - `model_factories` (renamed from `network_factories`) and `colosseum.launcher._create_model`;
  - `TrajectoryChunk.to(device)` (replaces `to_device`);
  - test helpers in `tests/helpers.py` and `tests/contract/harness.py`.
- **A5 — initial weight sync.** `RolloutLoop` records the `policy_version` of the initial payload.
- **C1 — `update_normalizers`.** The base default from T4.2 calls `update()` on every `NormalizeObs` submodule (no-op when there are none). T1.2 ships a no-op that T4.2 replaces.
- **C2 / B5 — algorithm state.** There is no LR-scheduler object after T2.5; the LR is a function of `progress`.
  - `APPO.state_dict()` keys are exactly `optimizer`, `progress`, `scaler`, `kickstart`, `policy_version`, `consumed_samples`.
  - `load_state_dict` re-applies `set_progress`.
  - `deep_cpu_copy` lives in `colosseum.algorithms.base`.
- **C3 — kickstart teacher.** The teacher reuses the chunk's `initial_state`, so its state layout must match the student's; `APPO.__init__` checks this.
  - API: `KickstartLoss(teacher, initial_lambda, decay_steps, direction)`, `.compute(student_dist, observations, dones, state0, action_mask)`, `.state_dict()` / `.load_state_dict()`.
- **C4 — metric key.** The APPO learning-rate metric is `lr`; consumers also accept `learning_rate`.
- **C5 — other names introduced by part C:**
  - `compute_vtrace(..., lam=1.0)`;
  - `CategoricalDist.mask`;
  - `CompositeDist` keeps insertion order and exposes `.keys`, `.components`, `.flat_mask_size`;
  - `ActionSpec.component_names` and `ActionSpec.check_distribution(dist)` (called by `validate_config`);
  - `ColosseumConfig.bc: BCConfig(seq_len=64)`;
  - `OfflineBCTrainer(model, lr, device, seq_len)`;
  - `APPO.consumed_samples`;
  - eval API `MatchRecord`, `schedule_lineups`, `play_matches`, `wilson_interval`, `normal_interval`, `PairStats`, `SoloStats`, `EvalReport`, `summarize`, `evaluate`, `load_eval_model`;
  - CLI `-a/--agent name=path`.

**Data flow and worker**
- **B1 — `put_latest(q, item, timeout=1.0) -> bool`.** A put-nowait / get-nowait / put-nowait sequence loses the newest item while the previous one is still in the `mp.Queue` feeder thread. The function evicts with short blocking gets and gives up after `timeout`.
- **B2 — per-agent gamma.** `RolloutLoop` and `rollout_worker_process` accept `gamma: float | dict[str, float]`, keyed by agent.
- **B3 — buffers.**
  - `BufferPool(chunk_length, obs_shape, action_shape, action_dtype, mask_size=0)`;
  - `BufferPool.parked_transitions(agent_id)`;
  - `RolloutBuffer` API: `begin_chunk`, `open`, `add_reward`, `mark_done`, `build_chunk`, `reset`, `steps`, `is_full`, `last_done`, plus `initial_state` from A4.

  Everything lives in `worker/slots.py`, and B's T2.6 moves `RolloutBuffer` there.
- **B4 — policy lag.** `policy_lag_mean` / `policy_lag_max` are produced by `learner_process` (T2.6) and must not be duplicated in APPO.
- **B6 — signatures.** `rollout_worker_process(*, ...)` is keyword-only, as are `launcher._worker_target(*, ...)` and `launcher._learner_target(*, ...)`. `Launcher.env_steps_done` exists. `LATEST_NETWORK_ID` lives in `worker/rollout_loop.py` and is re-exported by `worker/rollout_worker.py`.
- **B7 — queue item formats after T2.2.**
  - Trajectory queues carry chunk payload dicts, and `collect_batch` raises `TypeError` on anything else.
  - Checkpoint queues carry numpy dicts, until T5.3 replaces them with `make_checkpoint_payload` (D2).
- **B8 — worker commands.** `_drain_commands` merges `new_checkpoints` over all drained commands. The command queue is never newest-wins (D9).
- **B9 — `RolloutLoop.stats` keys:** `chunks_sent`, `env_steps`, `parked_buffers`, `recorded_transitions`, `recorded_transitions/<agent>`, `buffered_transitions/<agent>`.
- **B10 — `SeatResult.agent_id`** is the base agent, and `Coordinator._base_agent` is removed.

**League, run dir, lifecycle**
- **D1 —** `checkpoint.dir` is removed in T6.2, not T6.1.
- **D2 — checkpoint payload and storage.**
  - Payload (T5.3): `make_checkpoint_payload` → `{agent_id, policy_version, model_state (numpy), trainer_state_bytes, final}`. The trainer state is `torch.save(algorithm.state_dict())` bytes; bytes are allowed on queues, torch tensors are not.
  - `CheckpointManager.save(..., trainer_state: dict | bytes | None, ...)`.
- **D3 —** `resolve_resume(...)` returns `{"model_state", "trainer_state": bytes | None, "policy_version", "env_steps", "source"}`. The main process continues the global `SharedCounter` from `env_steps`.
- **D4 / D5 — ratings.** `ratings_snapshot()` also returns `games` and `past_games`. `EloRating.update_pairs(pairs, k_scale)` applies all pairs of a match simultaneously.
- **D6 — Coordinator.**
  - `Coordinator(config, checkpoint_dir)`;
  - `next_round()` and `refresh_round`;
  - `save_checkpoint_payload(payload, meta_extra)` replaces `maybe_save_checkpoint`;
  - `past_win_rate`;
  - self-registration of trainable agents.
- **D7 — `RunDir`.** `RunDir.create(config, config_path=None, role=None)`, plus `RunDir.resolved_config_path` and `RunDir.open(root)`.
- **D8 — child processes.** `colosseum.utils.process.run_child(name, log_dir, fn, *args, **kwargs)`, `ProcessSupervisor`, `SHUTDOWN_GRACE_SEC = 7.0`, and the env vars `COLOSSEUM_LOG_DIR` / `COLOSSEUM_PROCESS_NAME`.
- **D9 — command queue.** `_COMMAND_QUEUE_SIZE = 1` with `put_nowait`; a full queue means the worker skips that round. Commands are never newest-wins.
- **D10 — exit code.** `Launcher(config, run_dir)`; `Launcher.launch() -> int` and `run_training(...) -> int` return the exit code. The `SharedCounter` is `Launcher._env_counter`.
- **D11 —** `rollout_worker_process(..., stats_queue=None, stats_interval_sec=2.0)` (T6.3).

**Additional files** (missing from the file map above):
- `src/colosseum/core/threads.py` (T2.1)
- `tests/dataflow_helpers.py` (T2.x)
- `src/colosseum/utils/process.py` (T6.2/T6.5)

## Cross-part execution notes

- **Plan code vs. current code.** Every part was written against the code state its author expected. Later tasks may meet code that already differs because an earlier task changed it. When a task's "before" block does not match, keep the intent, the contract (as amended above) and the task's tests, and adapt the edit. Say so in the commit message body. Never weaken a test to make it pass.
- **T2.2** must also:
  - update `tests/contract/harness.py::weights_payload` (A4) for the numpy `WeightPayload`;
  - remove `@pytest.mark.slow` from `tests/integration/test_pipelines.py` and `tests/integration/test_distributed_e2e.py`, except tests that take longer than ~20 s (A note 7).
- **T2.1** tests must assert on the explicit thread calls inside worker and learner processes, not on the `OMP_NUM_THREADS=1` that `tests/conftest.py` exports to children (A note 8).
- **T4.1** adds a test kit to `tests/helpers.py`. Reuse part A's helpers (`make_simple_model`, `make_core`, `CORE_KINDS`, `CountingEnv`, …) and `tests/contract/harness.py` where they already cover the need; do not create near-duplicates with different names.
- **T6.3** must switch `scripts/bench_throughput.py` from patching `colosseum.launcher.WandBLogger` to reading the run's `metrics.jsonl`, without changing the measured quantities or the CLI, so T8.3's "после" numbers stay comparable (A note 9).
- **T8.3** rewrites the README/CLAUDE.md passages that still mention `recurrent_*`, `actor_critic.py` and `evaluate_actions_recurrent` (A note 10).
- **Simulation copy.** Part A was simulated end to end in a scratch copy of the repo: `/tmp/claude-1000/-home-viv-dev-repos-Colosseum/0db917f9-6d82-45c2-8cea-f3910287244a/scratchpad/plan_partA/repo`. Implementers of T0.x–T1.x may diff against it when a step is unclear. It is a reference, not a source of truth over the plan text.

---

## Task index and execution order

The order is strict except where noted. Every task lists its own dependencies in its **Interfaces** block.

| ID | Task | Depends on |
|---|---|---|
| T0.1 | Dev environment script, dependency pins and extras, optional wandb import | — |
| T0.2 | Test infrastructure: spawn, markers, timeouts, tmp_path-only, new test layout | T0.1 |
| T0.3 | Ruff config + GitHub Actions CI | T0.2 |
| T0.4 | Throughput benchmark + `docs/benchmarks.md` baseline | T0.1 |
| T0.5 | Extract `RolloutLoop` (pure refactor) + characterization tests | T0.2 |
| T1.1 | `State` pytree utilities | T0.2 |
| T1.2 | `PolicyModel` protocol + `act()` | T1.1 |
| T1.3 | Cores: NoCore, LSTMCore, GRUCore, WindowAttentionCore | T1.2 |
| T1.4 | `ComposedModel`, `NetworkConfig`/`CoreConfig`, `build_model`, `validate_config`, errors module, helpers/examples migration | T1.3 |
| T1.5 | APPO on `PolicyModel.unroll`; `TrajectoryChunk.initial_state`; worker/learner/launcher/distributed switched to `PolicyModel` (amendment A1) | T1.4, T0.5 |
| T1.6 | Contract test: learner reproduces worker log-probs/values for 4 cores; mutation check; `harness.learner_eval` | T1.5 |
| T1.7 | Eval/BC/kickstart callers on `PolicyModel`; delete `actor_critic.py` | T1.6 |
| T2.1 | Torch thread limits (worker, subproc children, learner auto) | T1.7 |
| T2.2 | Numpy payloads for all inter-process data (chunks, weights, commands, checkpoints) | T2.1 |
| T2.3 | Newest-wins weight delivery (`put_latest`/`drain_latest`) | T2.2 |
| T2.4 | Learner trains on exactly `batch_chunks` (`collect_batch`) | T2.2 |
| T2.5 | Global env-step budget (`SharedCounter`), `set_progress`, LR by progress, stop by budget | T2.4 |
| T2.6 | Agent-owned buffers with parking; behavior version = first transition; policy-lag metric | T2.5 |
| T3.1 | Open transitions sealed on the slot's next action; no bootstrap forward | T2.6 |
| T3.2 | Turn-based: active-only inference, pending rewards, done to all collecting slots, mask rules, reset-info fix | T3.1 |
| T3.3 | Truncation: `γ·V(final_obs)` reward augmentation | T3.2 |
| T3.4 | `SeatResult`/`MatchResult` per seat (worker → coordinator) | T3.2 |
| T4.1 | `vtrace_lambda` (remove `gae_lambda`) | T3.3 |
| T4.2 | `NormalizeObs` explicit update; `update_normalizers` once per train step | T4.1 |
| T4.3 | Masked KL; kickstart forward KL, masks, unroll, reuse student dist | T4.2 |
| T4.4 | BC: masks, distribution-aware loss, stateful sequence training, strict action types | T4.3 |
| T4.5 | `ActionSpec` natural component order | T4.1 (independent of T4.2–T4.4) |
| T4.6 | Algorithm `state_dict`/`load_state_dict`; extra APPO metrics | T4.3 |
| T5.1 | Owner rotation, N-player arena, seat shuffle | T4.6, T3.4 |
| T5.2 | Pairwise ratings from seats; `wr_vs_past`; `ratings_snapshot` | T5.1 |
| T5.3 | `CheckpointManager` rewrite, final checkpoint, missing-ckpt fallback, `resolve_resume` | T4.6 |
| T5.4 | Integration: 2-agent self-play to budget; 3-agent league all pairs, seat balance | T5.2, T5.3, T6.5 |
| T6.1 | Config strictness, partial agent overrides, `--set` parsing, `run` section, seeds, validate num_players | T4.6 |
| T6.2 | `RunDir`, `setup_process_logging` in every process, resolved config | T6.1 |
| T6.3 | `metrics.jsonl`, episode/system aggregation, console reporter, `ratings.json` | T6.2, T5.2 |
| T6.4 | WandB per-agent step axes, optional import | T6.3 |
| T6.5 | CLI/lifecycle: sys.path, exit codes, signals, child-death detection, normal completion | T6.2 |
| T7.1 | Eval engine on `PolicyModel` with per-seat state and seat rotation | T4.6 |
| T7.2 | Eval statistics/CLI: W/D/L, Wilson CIs, per-seat, solo, meta-based models, JSON output | T7.1, T5.3 |
| T8.1 | Examples: tic-tac-toe `active` + mask, chase numpy fix, configs, attention example | T6.5, T7.2 |
| T8.2 | Learning tests: bandit/chain (fast), tic-tac-toe ≥80% (slow) | T8.1 |
| T8.3 | Docs: README, CLAUDE.md status + roadmap, `GPU_CHECKS.md`, benchmarks "after" | T8.2 |
| T8.4 | Acceptance run against spec §3 criteria | T8.3 |

Parallelism allowed after T4.6:
- one lane runs T5.x, another T6.x, a third T7.x;
- `launcher.py` is touched by T5.1, T5.3, T6.2, T6.3 and T6.5, so these tasks rebase on each other in index order;
- T5.4 waits for T6.5.
