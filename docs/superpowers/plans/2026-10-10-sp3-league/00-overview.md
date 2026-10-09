# SP3: Players, League, Warm Start — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A competition pipeline (scripted bot → `record` → `bc` → `train` with per-agent `init`, critic warm-up, kickstart (neural or scripted teacher) and a league of latest / PFSP snapshots / other agents / scripted and frozen anchors → `eval`) works end to end and is configured in one file; SP2 configs and checkpoints keep working.

**Architecture:**
- `agents.<id>.kind` (`trainable` | `scripted` | `frozen`) makes every seat-taker an agent. Seats of scripted/frozen agents use the reserved network id `"fixed"` and never collect.
- `MatchRunner` serves a *player pool*: neural players are batched as in SP2, scripted bots (`ScriptedBot`) run per (agent, env, seat) and see `obs`, `mask`, `infos[seat]`.
- A new package `colosseum.league` holds the matchmaker interface (`BaseMatchmaker` + `MatchmakerContext`), the built-in `MixtureMatchmaker` (opponent categories with shares, schedules and per-agent overrides), PFSP statistics per player and share schedules.
- Checkpoint storage keeps `keep_last` + every `keep_every`-th + the final snapshot, ships evictions to workers, and carries the pool over a run-dir resume.
- Warm start is per agent: `init` (weights, strict or partial), critic warm-up (value path only), kickstart from any frozen/scripted teacher (scripted = DAgger labels written by the worker into the chunk).
- `colosseum record` writes BC data of any player; `colosseum bc` reads it.

**Tech Stack:** Python 3.12, PyTorch ≥2.6 (CPU for development), numpy, gymnasium, pydantic v2, click, pytest (+pytest-timeout), ruff, uv. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-10-sp3-league-design.md` (Russian; binding authority). Read it before any task. SP2 records: `docs/superpowers/specs/2026-10-08-sp2-game-model-design.md`, `docs/superpowers/reports/2026-10-09-sp2-acceptance.md` (rulings at the end). SP1 rulings: end of `docs/superpowers/reports/2026-10-08-sp1-acceptance.md`.

## Plan files

| File | Spec blocks | Tasks |
|---|---|---|
| `00-overview.md` (this file) | constraints, file map, interface contract, task index | — |
| `01-players-storage.md` | compat fixtures; 1 agents/players config; 2 scripted bots; 3 player execution; 4 snapshot storage; PFSP statistics and rating entities (part of 5) | T0.1, T1.1–T1.5, T2.1–T2.3 |
| `02-league-warmstart.md` | 5 opponent selection; 6 warm start | T3.1–T3.3, T4.1–T4.5 |
| `03-record-demos-acceptance.md` | 7 record/BC/eval; 8 distributed guards; 9 team_tag; 10 APPO measurement; 11 docs; acceptance | T5.1–T5.2, T6.1–T6.7 |

## Global Constraints

These apply to every task.

- **Python and dependencies:** Python 3.12 (`requires-python = ">=3.11"` stays); `torch>=2.6`; no new runtime dependencies. grpc stays in the optional extra `grpc`, wandb in `wandb`.
- **Environment:** run everything with `.venv/bin/python` (create it with `scripts/setup-dev.sh` if missing). Never install into system Python. CPU torch only from `--index-url https://download.pytorch.org/whl/cpu`.
- **Backward compatibility (owner's instruction for SP3):** SP2 configs (the copies in `tests/fixtures/sp2/configs/`, T0.1) validate without edits; SP2 checkpoints load as frozen agents, as `init` and for resume. Old knobs are accepted only in the input YAML, translated, and announced by ONE warning each; they are never stored in the config model and never written to `config.resolved.yaml`. Old knob + its new replacement together = `ConfigError`.
- **Green suite after every task:** the task's new tests pass, and `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` passes in full **with zero warnings**; `.venv/bin/ruff check .` passes. (Tests that expect a deprecation warning catch it with `caplog`/`pytest.warns`; translation warnings go through `logging`, not `warnings.warn`, so they never reach `-rw`.)
- **Inter-process data:** nothing that crosses a process boundary (mp.Queue, pipes, gRPC, `mp.Process` kwargs) contains `torch.Tensor`. Weights travel as numpy state dicts; configs as pydantic models; scripted bots as `BotSpec` (class path + kwargs), never as instances.
- **dtypes are preserved** end to end (SP2 rule), including recorded BC data and `teacher_action`.
- **Errors:**
  - env contract violations: `colosseum.core.errors.EnvContractError` with context `"worker W, env E, seat P, episode step K, layout L: ..."` (SP2 format; omit parts that do not apply);
  - a scripted bot's illegal action or exception: the new `colosseum.core.errors.PlayerError` with the same context plus `agent <id>`;
  - config problems: `ConfigError` with a fix hint;
  - no silent fallbacks except the spec's: default outcome from rewards (SP2), missing network → latest with collect (SP1), the matchmaker runtime fallback (spec block 5).
- **Tests:**
  - files only under pytest's `tmp_path` (fixtures under `tests/fixtures/` are read-only inputs);
  - `@pytest.mark.gpu` for CUDA-only tests, `@pytest.mark.slow` for tests longer than ~20 s;
  - at most 2 worker processes per test; `OMP_NUM_THREADS=1` (exported by `tests/conftest.py`);
  - no `__init__.py` under `tests/`; support modules imported by bare name; **test file basenames and support-module names unique across `tests/`**. New SP3 test files use the prefix `test_sp3_`;
  - integration runs start processes through `tests/cli_runner.py`;
  - toy games and tiny models come from `tests/game_helpers.py` (extend it, do not duplicate it).
- **SP1/SP2 rulings stay in force** unless the spec (section 8, «Изменения правил SP2») changes them. Most likely to be broken by accident: agent id regex and reserved ids; strict explicit resume; `SHUTDOWN_GRACE_SEC = 7`; eval loads checkpoints read-only (`load_checkpoint_dir`, never `CheckpointManager` on a user path); newest-wins weight queues; the learner uses exactly `learner.batch_chunks` chunks per update; `training.total_timesteps` is a global env-step budget; one legality gate (`ActionSpec.first_illegal_action` / `units_component_valid`); error context format; owner rotation over trainable agents in config order.
- **Any config-schema change also updates `scripts/bench_throughput.py::_make_config`** (guarded by `tests/unit/test_bench_throughput.py`).
- **Language:** code, comments, docstrings, log messages, `README.md`, `CLAUDE.md` in English; `docs/LEAGUE_GUIDE.md`, `docs/ENV_GUIDE.md`, `docs/benchmarks.md`, acceptance report in Russian.
- **Commits:** conventional prefixes (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `perf:`); no attribution or co-author lines; at least one commit per task on branch `sp3-league`; push to `origin` after every task.
- **Style:** follow existing code style (type hints, dataclasses, pydantic models, `logger = logging.getLogger(__name__)`, `from __future__ import annotations`).
- **Process (owner's instruction):** each task is implemented by one subagent on Opus with high effort (failing tests first, then code, then commit), reviewed by an Opus/high reviewer; web research, if any, by Sonnet/high subagents.

## Deviations from the spec

- **Numbering:** spec section 7 names parts P0–P6; the plan uses T0 (P0), T1 (P1), T2 (P2), T3 (P3), T4 (P4), T5 (P5), T6 (P6 + acceptance).
- **Category tags:** spec block 5 says `Lineup` carries the category of each opposing team core. Seat permutation (`shuffle_seats`) moves whole teams, so the tag lives on every seat instead: `SeatAssignment.source` (and `SeatResult.source`); all seats of a team carry the same value (`"owner"` for the owner's team). Same information, survives permutation.
- **`infos` storage:** `MatchRunner` already keeps each env's latest `StepResult`; scripted seats read `result.infos.get(seat)` from it, so no extra storage exists (spec block 2's "stores infos only for scripted seats" holds trivially).
- **`coordinator/matchmaker.py` is deleted:** `enabled_layouts` and `permute_seats` move to `colosseum.league.lineups`; `LineupMatchmaker` is replaced by `colosseum.league.mixture.MixtureMatchmaker`.
- **`validate` output:** `registry.validate_config` returns a `ValidationReport` (lines to print: effective opponent mix, `init` reports) instead of `None`; callers that ignore the return value keep working.

---

## File structure

### Create
- `src/colosseum/players/__init__.py` — exports `ScriptedBot`, `RandomBot` (T1.2).
- `src/colosseum/players/scripted.py` — `ScriptedBot`, `RandomBot`, `bot_rng`, `check_bot_action` (T1.2).
- `src/colosseum/players/registry.py` — `BotSpec`, `FrozenSpec`, `FixedPlayers`, `resolve_player_roles`, `load_fixed_players`, `build_frozen_model`, `make_bot` (T1.2).
- `src/colosseum/league/__init__.py` — exports `BaseMatchmaker`, `MatchmakerContext`, `MixtureMatchmaker` (T3.2).
- `src/colosseum/league/schedule.py` — `ScheduleValue`, `schedule_value`, `schedule_points` (T3.1).
- `src/colosseum/league/pfsp.py` — `PlayerKey`, `PfspStats`, `pfsp_weight` (T2.3).
- `src/colosseum/league/lineups.py` — `enabled_layouts`, `permute_seats` (moved from `coordinator/matchmaker.py`), `check_lineup` (T3.2).
- `src/colosseum/league/base.py` — `AgentView`, `MatchmakerContext`, `BaseMatchmaker` (T3.2).
- `src/colosseum/league/mixture.py` — `MixtureMatchmaker`, `effective_mix`, `describe_mix`, `validate_matchmaking` (T3.2).
- `src/colosseum/learner/factory.py` — `TeacherSpec`, `InitState`, `resolve_teacher`, `resolve_init`, `build_algorithm` (T4.1; extended T4.2–T4.5).
- `src/colosseum/record.py` — `RecordObserver`, `RoleWriter`, `record` (T5.1).
- `examples/unit_harvest/bots.py`, `examples/tic_tac_toe/bots.py` — demo bots (T6.2).
- `tests/fixtures/sp2/` — SP2 config copies + SP2 checkpoint fixture + its generator (T0.1).
- `docs/LEAGUE_GUIDE.md` (T6.6).

### Modify (main ones)
- `src/colosseum/core/config.py` — agent kinds (T1.1), checkpoint retention (T2.1), matchmaking v3 (T3.1), `init`/`kickstart` sections (T4.1).
- `src/colosseum/core/types.py` — `FIXED_NETWORK_ID`, `SeatAssignment.source`, `SeatResult.source`, `WorkerCommand.evict`, `WeightPayload.teacher_active`, chunk `teacher_action`/`has_teacher` (T1.3, T2.2, T4.5).
- `src/colosseum/core/errors.py` — `PlayerError` (T1.2).
- `src/colosseum/worker/match_runner.py` (T1.3), `worker/rollout_loop.py` and `worker/rollout_worker.py` (T1.4, T2.2, T4.5), `worker/buffers.py` (T4.5).
- `src/colosseum/coordinator/coordinator.py` (T1.4, T2.1, T2.2, T3.3), `coordinator/checkpoint_manager.py` (T2.1), `coordinator/ratings.py` (T2.3); `coordinator/agent_pool.py` deleted (T1.4); `coordinator/matchmaker.py` deleted (T3.2).
- `src/colosseum/launcher.py` (T1.4, T2.1, T2.2, T3.3, T4.1–T4.5), `src/colosseum/distributed.py` (T4.1, T6.1).
- `src/colosseum/algorithms/appo.py` (T4.3–T4.5), `src/colosseum/bc/kickstart.py` (T4.4, T4.5), `src/colosseum/networks/model.py` + `networks/composed.py` (T4.3).
- `src/colosseum/eval.py` (T1.2, T1.5), `src/colosseum/bc/offline_bc.py` (T5.2), `src/colosseum/cli.py` (T1.5, T3.3, T5.1, T5.2), `src/colosseum/core/validation.py` (T1.5, T3.3, T4.2, T4.4).
- `src/colosseum/metrics/aggregator.py`, `metrics/hub.py` (T2.3, T3.3).
- `configs/examples/team_tag.yaml` (T6.3), `scripts/units_experiment.py` (T6.5), `scripts/bench_throughput.py` (every schema task).

---

## Interface contract

Names and signatures every task relies on. A task may add private helpers freely; it may not rename or re-type anything listed here without a contract amendment (below).

### `colosseum.core.errors` (T1.2)
```python
class PlayerError(ColosseumError):
    """A scripted player broke the rules (illegal action, exception, wrong structure)."""
```
(`ColosseumError` is the existing base class of `ConfigError` / `EnvContractError`; if the base has another name, subclass that.)

### `colosseum.core.types`
```python
LATEST_NETWORK_ID = "latest"                 # existing
FIXED_NETWORK_ID = "fixed"                   # T1.3: seats of scripted and frozen agents
SOURCE_OWNER = "owner"                       # T1.3: the owner's team
OPPONENT_CATEGORIES = ("latest", "snapshots", "rivals", "anchors", "fallback")   # T1.3

@dataclass
class SeatAssignment:                        # T1.3 adds `source`
    agent_id: str
    network_id: str = LATEST_NETWORK_ID
    collect: bool = True
    source: str = ""                         # "" (eval/tests), SOURCE_OWNER or one of OPPONENT_CATEGORIES;
                                             # every seat of a team carries its team's value

@dataclass
class SeatResult:                            # T1.3 adds `source` (copied from the lineup by MatchRunner)
    seat: int; role: str; team: int; agent_id: str; network_id: str; reward: float
    eliminated_step: int | None = None
    source: str = ""

@dataclass
class WorkerCommand:                         # T2.2 adds `evict`
    lineups: list[Lineup | None]
    new_checkpoints: dict[str, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict)
    evict: dict[str, list[str]] = field(default_factory=dict)   # agent -> checkpoint ids to unload

@dataclass
class WeightPayload:                         # T4.5 adds `teacher_active`
    agent_id: str; policy_version: int; state_dict: dict[str, np.ndarray] = ...
    teacher_active: bool = False             # True while a scripted kickstart teacher has lambda > 0

@dataclass
class TrajectoryChunk:                       # T4.5 adds two optional fields (default None, after the existing ones)
    ...
    teacher_action: Tree | None = None       # action tree [S, ...] (ActionSpec.allocate_actions layout)
    has_teacher: torch.Tensor | None = None  # [S] bool; True only on ACT slots that carry a teacher label
```
`to_payload` / `from_payload` / `_apply` carry both fields (None stays None); `validate_slot_structure` rejects `has_teacher` on a non-ACT slot.

### `colosseum.core.config`
T1.1 (agent kinds):
```python
AgentKind = Literal["trainable", "scripted", "frozen"]

class TrainableAgent(StrictModel):           # successor of AgentOverride (keep `AgentOverride = TrainableAgent` alias)
    kind: Literal["trainable"] = "trainable"
    roles: list[str] | None = None
    networks: dict[str, Any] | None = None
    algorithm: dict[str, Any] | None = None
    learner: dict[str, Any] | None = None
    matchmaking: dict[str, Any] | None = None    # T3.1 (field added in T1.1 as a free dict, validated in T3.1)
    init: dict[str, Any] | None = None           # T4.1 (same)
    kickstart: dict[str, Any] | None = None      # T4.1 (same)

class ScriptedAgent(StrictModel):
    kind: Literal["scripted"]
    class_path: str = Field(alias="class")       # dotted path to a ScriptedBot subclass
    kwargs: dict[str, Any] = {}
    roles: list[str] | None = None               # None = every role of the game (spaces may differ)

class FrozenAgent(StrictModel):
    kind: Literal["frozen"]
    path: str                                    # checkpoint dir (meta.json gives roles + networks) or .pt
    networks: dict[str, Any] | None = None       # only with a .pt path (partial override onto the global networks)
    roles: list[str] | None = None               # only with a .pt path

AgentEntry = Annotated[TrainableAgent | ScriptedAgent | FrozenAgent, Field(discriminator="kind")]
# a raw entry without "kind" gets kind="trainable" (before-validator); null entry = TrainableAgent()

class ColosseumConfig:
    agents: dict[str, AgentEntry]
    def get_trainable_agent_ids(self) -> list[str]   # trainable ids in config order; ["agent_0"] if none
    def fixed_agent_ids(self) -> list[str]           # scripted + frozen ids in config order
    def agent_entry(self, agent_id: str) -> TrainableAgent | ScriptedAgent | FrozenAgent   # implicit agent_0 -> TrainableAgent()
    def agent_kind(self, agent_id: str) -> AgentKind
    def get_agent_config(self, agent_id: str) -> ColosseumConfig   # trainable only; ConfigError for fixed kinds
    def agent_roles(self, agent_id: str) -> list[str] | None       # any kind
```
`StrictModel` gets `populate_by_name=True` (aliases `class`, `from`, `lambda`). Config dumps use `by_alias=True` (existing).

T2.1 (retention):
```python
class CheckpointConfig(StrictModel):
    interval: int = 1000
    keep_last: int = 20          # ge=1; raw `pool_size` -> keep_last (one warning); both -> ConfigError
    keep_every: int = 10         # ge=0; 0 = off
    save_optimizer: bool = True
```

T3.1 (matchmaking v3):
```python
ScheduleValue = float | dict[int, float]      # defined in colosseum.league.schedule, re-used here

class OpponentShares(StrictModel):
    latest: ScheduleValue = 0.7
    snapshots: ScheduleValue = 0.2
    rivals: ScheduleValue = 0.0
    anchors: ScheduleValue = 0.1

class PfspConfig(StrictModel):
    weighting: Literal["hard", "balanced", "uniform"] = "hard"
    exponent: float = 2.0         # ge=0
    halflife_games: float = 200.0 # gt=0

class MatchmakingConfig(StrictModel):
    opponents: OpponentShares = OpponentShares()
    anchors: list[str] | dict[str, ScheduleValue] | None = None   # None = every scripted and frozen agent
    pfsp: PfspConfig = PfspConfig()
    layouts: dict[str, float] = {}
    teammates: Literal["self", "mixed"] = "self"
    teammate_self_prob: float = 0.5
    shuffle_seats: bool = True                # global only
    matchmaker_class: str | None = None       # global only; dotted path to a BaseMatchmaker subclass

AGENT_MATCHMAKING_KEYS = frozenset({"opponents", "anchors", "pfsp", "layouts", "teammates", "teammate_self_prob"})
SP2_MATCHMAKING_KNOBS = ("mode", "self_play_ratio", "latest_prob", "pfsp_exponent")
def translate_sp2_matchmaking(raw: dict) -> dict      # pure: raw dict in, v3 raw dict out (spec block 5 formula)
```

T4.1 (warm start):
```python
class InitConfig(StrictModel):
    from_: str | None = Field(None, alias="from")   # .pt | checkpoint dir | run dir | frozen agent name
    strict: bool = True
    critic_warmup_steps: int = 0                     # ge=0

class KickstartConfig(StrictModel):
    teacher: str | None = None                       # frozen/scripted agent name | .pt | checkpoint dir
    lambda_: float = Field(1.0, alias="lambda")      # ge=0
    decay_steps: int = 50_000                        # ge=1
    kl: Literal["forward", "reverse"] = "forward"

class ColosseumConfig:
    init: InitConfig = InitConfig()
    kickstart: KickstartConfig = KickstartConfig()
_AGENT_SECTIONS = ("networks", "algorithm", "learner", "matchmaking", "init", "kickstart")
# raw training.kickstart_teacher / kickstart_lambda / kickstart_decay_steps / kickstart_kl -> top-level `kickstart`
# (one warning); together with a top-level `kickstart` -> ConfigError. TrainingConfig drops those fields.
```

### `colosseum.players` (T1.2)
```python
class ScriptedBot:
    game_spec: GameSpec            # set by the framework right after construction, before the first reset
    def reset(self, *, role: str, seat: int, layout: str, rng: np.random.Generator) -> None: ...   # default: no-op
    def act(self, obs: Tree, mask: Tree | None, info: Any) -> Tree: ...                             # abstract

class RandomBot(ScriptedBot):      # uniformly random legal action (core.validation.random_legal_action)

def bot_rng(episode_seed: int | None, seat: int, agent_id: str) -> np.random.Generator
    # SeedSequence([episode_seed, seat, crc32(agent_id)]) when episode_seed is not None, else fresh entropy
def check_bot_action(role: RoleSpec, action: Tree, mask: Tree | None, where: str) -> None
    # structure + space.contains + ActionSpec.first_illegal_action; PlayerError(f"{where}: ...") on failure

# colosseum.players.registry
@dataclass(frozen=True)
class BotSpec:  class_path: str; kwargs: dict[str, Any]          # picklable
@dataclass(frozen=True)
class FrozenSpec:
    agent_id: str; roles: tuple[str, ...]; networks: dict[str, Any]   # NetworkConfig dump (by_alias)
    model_state: dict[str, np.ndarray]; source: str
@dataclass(frozen=True)
class FixedPlayers:
    bots: dict[str, BotSpec]        # scripted agents
    frozen: dict[str, FrozenSpec]   # frozen agents
    roles: dict[str, tuple[str, ...]]   # every fixed agent's roles
def resolve_player_roles(config: ColosseumConfig, spec: GameSpec) -> dict[str, list[str]]   # EVERY agent, config order
def load_fixed_players(config: ColosseumConfig, spec: GameSpec) -> FixedPlayers   # main process; ConfigError on bad paths/signatures
def load_frozen(config: ColosseumConfig, agent_id: str, path: str, spec: GameSpec) -> FrozenSpec   # also used for teacher/init paths
def build_frozen_model(config: ColosseumConfig, frozen: FrozenSpec, spec: GameSpec) -> PolicyModel   # eval mode, weights loaded
def make_bot(bot: BotSpec, spec: GameSpec) -> ScriptedBot       # import, construct with kwargs, set game_spec; ConfigError if not a ScriptedBot
```
`eval.load_eval_model` delegates its checkpoint-dir / `.pt` logic to `load_frozen` + `build_frozen_model` (one loader).

### `colosseum.worker.match_runner` (T1.3)
```python
@dataclass(frozen=True)
class ScriptedPlayer:
    factory: Callable[[], ScriptedBot]       # e.g. functools.partial(make_bot, bot_spec, spec)

class PlayerPool(Protocol):                  # successor of ModelPool (keep `ModelPool = PlayerPool`)
    def get(self, agent_id: str, network_id: str) -> PolicyModel | ScriptedPlayer | None: ...

class MatchObserver(Protocol):               # unchanged five methods; plus OPTIONAL:
    # def on_episode_start(self, env: int, layout: str, episode_seed: int | None) -> None
    # called (if the observer defines it) after every env reset, after scripted bots were reset

@dataclass
class ActRecord:                             # adds `info`; for scripted seats log_prob=0.0, unit_log_probs=None, pre_state=None
    ...; info: Any = None

class MatchRunner:
    def next_lineup(self, env: int) -> Lineup | None     # the staged lineup, if any
```
Rules: scripted bot instances are keyed `(agent_id, env, seat)`, created lazily, reset after every env reset where they sit (`bot_rng(episode_seed, seat, agent_id)`), and get `result.infos.get(seat)`. `collect=True` only on `LATEST_NETWORK_ID` seats (ValueError otherwise). `_check_lineup` requires the pool entry of every `latest`/`fixed` seat; checkpoint seats keep the SP1 fallback at apply time.

### `colosseum.worker.rollout_loop` / `rollout_worker` (T1.4, T2.2, T4.5)
```python
RolloutLoop(..., fixed_players: FixedPlayers | None = None, teachers: Mapping[str, BotSpec] | None = None)
rollout_worker_process(..., fixed_players: FixedPlayers | None = None, teachers: Mapping[str, BotSpec] | None = None)
```
`teachers[agent_id]` = the scripted kickstart teacher of a trainable agent (T4.5). Evictions (T2.2): a command's `evict` ids are unloaded once no env's current or staged lineup uses them.

### `colosseum.coordinator.checkpoint_manager` (T2.1)
```python
class CheckpointManager:
    def __init__(self, base_dir, keep_last: int = 20, keep_every: int = 0, interval: int = 1,
                 on_evict: Callable[[str, str], None] | None = None) -> None
    def import_snapshots(self, src_checkpoints_dir: str | Path, agent_id: str) -> list[str]
        # hardlink (fallback copy) model.pt + meta.json of every ckpt_v* of src/agent_id; retention applied; ids returned
```
Retention after each save: keep the newest `keep_last`, every version with `version % (interval * keep_every) == 0` (if `keep_every > 0`), and every `final: true`; others are evicted (`on_evict(agent_id, checkpoint_id)` called). `trainer_state.pt` is deleted from checkpoints outside the newest `keep_last`.

### `colosseum.league` (T2.3, T3.1, T3.2)
```python
# league/schedule.py (T3.1)
ScheduleValue = float | dict[int, float]
def schedule_value(value: ScheduleValue, env_steps: int) -> float      # piecewise-linear, constant outside
def schedule_points(value: ScheduleValue) -> list[int]                  # breakpoints ([0] for a number)

# league/pfsp.py (T2.3)
PlayerKey = tuple[str, str]                                              # (agent_id, network_id)
def pfsp_weight(score: float, weighting: str, exponent: float) -> float  # floor 1e-6
class PfspStats:
    def __init__(self, halflife_by_agent: Mapping[str, float], prior: float = 0.5) -> None
    def update(self, result: MatchResult) -> None
    def score(self, layout: str, owner: str, player: PlayerKey) -> float
    def games(self, layout: str, owner: str, player: PlayerKey) -> float
    def forget(self, player: PlayerKey) -> None
    def snapshot(self) -> dict[str, dict[str, dict[str, dict[str, float]]]]   # {layout: {owner: {"agent@net": {"score", "games"}}}}

# league/lineups.py (T3.2)
def enabled_layouts(spec: GameSpec, config: MatchmakingConfig) -> dict[str, float]     # moved
def permute_seats(spec: GameSpec, layout: str, seats, rng) -> list[SeatAssignment]   # moved; `source` travels with its seat
def check_lineup(context: MatchmakerContext, lineup: Lineup, who: str) -> None          # ValueError naming `who`

# league/base.py (T3.2)
@dataclass(frozen=True)
class AgentView:  agent_id: str; kind: AgentKind; roles: frozenset[str]
class MatchmakerContext:
    spec: GameSpec
    agents: dict[str, AgentView]                     # every agent, config order
    trainable: list[str]                             # config order
    matchmaking: dict[str, MatchmakingConfig]        # effective config per trainable agent
    rng: random.Random
    def snapshots(self, agent_id: str) -> list[str]  # stored checkpoint ids, ascending version
    def pfsp_score(self, layout: str, owner: str, player: PlayerKey) -> float
    def env_steps(self) -> int
    def anchors(self, owner: str) -> dict[str, float]   # owner's effective anchors -> weight at env_steps()
class BaseMatchmaker(ABC):
    def __init__(self, context: MatchmakerContext) -> None
    @abstractmethod
    def lineup_for(self, owner: str) -> Lineup: ...
    def on_result(self, result: MatchResult) -> None: ...      # default no-op

# league/mixture.py (T3.2)
class MixtureMatchmaker(BaseMatchmaker): ...
def validate_matchmaking(spec: GameSpec, player_roles: Mapping[str, Sequence[str]], config: ColosseumConfig) -> None
def effective_mix(context: MatchmakerContext, owner: str, layout: str, team: int, env_steps: int) -> dict[str, float]
def describe_mix(config: ColosseumConfig, spec: GameSpec) -> list[str]   # lines for `validate`
```

### `colosseum.coordinator.coordinator` (T1.4, T2.1, T2.2, T3.3)
```python
class Coordinator:
    def __init__(self, config: ColosseumConfig, spec: GameSpec, player_roles: Mapping[str, Sequence[str]],
                 checkpoint_dir: str | Path, env_steps: Callable[[], int] = lambda: 0) -> None
    def generate_lineups(self, num_envs: int, env_offset: int) -> list[Lineup]   # every lineup passes check_lineup
    def report_match_result(self, result: MatchResult) -> None                    # ratings + PFSP + matchmaker.on_result
    def take_evictions(self) -> dict[str, list[str]]                              # T2.2: evicted since the last call
    def import_snapshots(self, run_dir: str | Path) -> None                       # T2.1: run-dir resume
    def ratings_snapshot(self) -> dict                                            # adds per-layout "pfsp" (T2.3)
```

### `colosseum.learner.factory` (T4.1–T4.5)
```python
@dataclass(frozen=True)
class TeacherSpec:
    kind: Literal["neural", "scripted"]
    source: str
    lambda_: float; decay_steps: int; kl: Literal["forward", "reverse"]
    frozen: FrozenSpec | None = None      # neural
    bot: BotSpec | None = None            # scripted
@dataclass(frozen=True)
class InitState:
    model_state: dict[str, np.ndarray]; strict: bool; source: str; report: list[str]
def resolve_teacher(config: ColosseumConfig, agent_id: str, spec: GameSpec) -> TeacherSpec | None
def resolve_init(config: ColosseumConfig, agent_id: str, spec: GameSpec) -> InitState | None
def build_algorithm(agent_config: ColosseumConfig, role_spec: RoleSpec, spec: GameSpec, *, device: str,
                    teacher: TeacherSpec | None) -> BaseAlgorithm
```
Launcher and `distributed.run_distributed_learner` both build algorithms through `build_algorithm` (T4.1).

### Models and APPO (T4.3–T4.5)
```python
class PolicyModel:
    def value_parameters(self) -> list[nn.Parameter]   # default: raise NotImplementedError; ComposedModel: value + critic_encoder
APPO(..., critic_warmup_steps: int = 0)                 # T4.3
KickstartLoss.compute_labels(student_dist, teacher_actions, has_teacher, is_act, reduction) -> Tensor   # T4.5
```

### `colosseum.record` (T5.1)
```python
def record(config: ColosseumConfig, player: str, against: Sequence[str], *, layouts: Sequence[str] | None,
           num_matches: int, output: str | Path, num_envs: int = 8, seed: int | None = None,
           deterministic: bool = False) -> dict   # the record.json content
```

### `colosseum.core.registry` / `validation` (T1.5, T3.3, T4.2)
```python
@dataclass
class ValidationReport:  lines: list[str] = field(default_factory=list)
def validate_config(config: ColosseumConfig) -> ValidationReport
```

### Test support (`tests/game_helpers.py`, extended by T1.2 onward)
- `RecordingBot(ScriptedBot)` — remembers every `reset` / `act` argument (for MatchRunner tests);
- `ConstantBot(ScriptedBot)` — always the same legal action (for DAgger tests);
- helpers that write a scripted/frozen agent into a config dict.

---

## Contract amendments

None yet. Amendments found while writing the parts or during the pre-flight scan are appended here as A1, A2, … and copied into the SDD workspace `constraints.md`.

## Cross-part execution notes

- T0.1 runs first, on unchanged SP2 code: its fixtures are the compatibility baseline every later task keeps green.
- T1.1 adds the `matchmaking`, `init`, `kickstart` fields to `TrainableAgent` as free dicts so later tasks only add validation.
- After T1.4 the launcher passes `FixedPlayers` to workers, but the SP2 matchmaker never seats fixed players until T3.2 replaces it; T1.x tests use explicit lineups.
- T3.2 deletes `coordinator/matchmaker.py`; update every import (distributed.py, validation.py, tests).
- T4.1 moves the learner's algorithm construction into `learner/factory.py`; T4.2–T4.5 extend it there (never in launcher.py / distributed.py directly).
- The slow tests (T6.3, T6.4) and measurements (T6.5) run long; their implementers record measurements in the task report, and thresholds become controller rulings.

## Task index and execution order

| Task | Title | Depends on |
|---|---|---|
| T0.1 | SP2 compatibility fixtures and baseline tests | — |
| T1.1 | Agent kinds in the config (`kind`, implicit `agent_0`) | T0.1 |
| T1.2 | `colosseum.players`: `ScriptedBot`, `RandomBot`, fixed-player loading, shared frozen loader | T1.1 |
| T1.3 | `MatchRunner` player pool (scripted seats, `infos`, `fixed`, `source`) | T1.2 |
| T1.4 | Fixed players in the worker, launcher and coordinator (`AgentPool` removed) | T1.3 |
| T1.5 | `eval -a name`, `play_lineups` with bots, `validate` of fixed agents | T1.4 |
| T2.1 | Snapshot retention (`keep_last`/`keep_every`/final) and run-dir resume import | T1.4 |
| T2.2 | Snapshot eviction on workers | T2.1 |
| T2.3 | PFSP statistics per player; fixed agents in ratings; `anchor` opponent type | T1.4 |
| T3.1 | Matchmaking config v3, schedules, SP2 knob translation, per-agent override | T2.3 |
| T3.2 | `colosseum.league`: `BaseMatchmaker`, `MixtureMatchmaker`, lineup checks | T3.1 |
| T3.3 | Coordinator/launcher integration, share metrics, `validate` mix printout | T3.2, T2.2 |
| T4.1 | `init`/`kickstart` config sections, `learner/factory.py` | T3.3 |
| T4.2 | `init`: sources, strict/partial, resume precedence, report | T4.1 |
| T4.3 | Critic warm-up | T4.2 |
| T4.4 | Neural teacher per agent (any architecture, state-layout rule) | T4.1 |
| T4.5 | Scripted teacher (DAgger): chunk labels, worker teachers, loss | T4.4, T4.3 |
| T5.1 | `colosseum record` | T1.5 |
| T5.2 | `bc` with several `--data` and record dirs | T5.1 |
| T6.1 | Distributed-mode guards | T4.5 |
| T6.2 | Demo bots, fast pipeline smoke, fast learning tests | T5.2, T4.5 |
| T6.3 | team_tag with anchors, single-run slow test | T6.2 |
| T6.4 | Slow pipeline test on unit_harvest, thresholds, pipeline vs scratch | T6.2 |
| T6.5 | APPO `Units` defaults: multi-seed measurement and rule | T6.1 |
| T6.6 | Docs (`LEAGUE_GUIDE`, `ENV_GUIDE`, README, CLAUDE.md), benchmark check | T6.3, T6.4, T6.5 |
| T6.7 | Acceptance report | T6.6 |

Execution order: the table order.
