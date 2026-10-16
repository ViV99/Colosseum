# SP4: Selection and Observability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the final stage of a competition works end to end:
- every training, tournament and pulse match lands in a bounded match log (`runs/<name>/matches.db`);
- snapshot ratings (Bradley–Terry with intervals) are computed from it, live during training and with bootstrap for reports;
- `colosseum tournament` compares candidates of any kind by a controlled design (common seeds, seat rotations) and selects the best in two stages (screening, then confirmation on fresh seeds);
- snapshots are kept by rating (top-k by lower bound), pinned, and capped (`max_snapshots`);
- an optional pulse process evaluates every new snapshot against a fixed field;
- `colosseum report` / `colosseum dashboard` show everything.

SP3 configs, checkpoints and run dirs keep working.

**Architecture:**
- A new package `colosseum.arena` holds the SP4 libraries. None of them touches process queues or the launcher, so SP5's hub can host them unchanged (spec block 8):
  - `arena.entities` — rating entities, version buckets;
  - `arena.comparisons` — team-pair sufficient statistics;
  - `arena.matchlog` — SQLite journal;
  - `arena.bt` — Bradley–Terry fit, Laplace, bootstrap, groups, anchor, G²;
  - `arena.live` — live refit loop;
  - `arena.retention` — the pure snapshot-retention function;
  - `arena.tournament.*` — schedules, engine, statistics, modes;
  - `arena.pulse` — the pulse process;
  - `arena.report.*` — data builders, HTML renderer, dashboard server.
- **Match records.** `MatchResult` / `SeatResult` gain optional fields: uid, source, seed, `env_steps`, block, per-seat weight versions at episode start and end. `Lineup` gains an optional explicit `seed`, `block` and `uid` (tournament / pulse).
- **Training main process** owns two threads:
  - the journal writer (`arena.matchlog.MatchLogWriter`), with a bounded queue, which also feeds an in-memory `ComparisonTable`;
  - the live rating refit (`arena.live.LiveRatings`), which publishes immutable snapshots.

  Retention, `on_evict`, `ratings.json` and metrics stay on the main loop. Threads never mutate coordinator or storage objects.
- **Snapshot storage** delegates the keep/evict decision to `arena.retention.retain(entries, scores, policy)`. `meta.json` gains `run_id` and `pinned`.
- **`colosseum tournament`** plays blocks (one seed × all admissible team rotations) in local worker processes through `MatchRunner`. It writes `source=tournament` matches (whole blocks per transaction) and reports from its own matches only.
- **`colosseum report`** builds a self-contained HTML file (own CSS/JS/SVG, data embedded, no CDN). **`colosseum dashboard`** serves the same page plus `/api/<section>` JSON with `http.server`.

**Tech Stack:** Python 3.12, PyTorch ≥ 2.6, numpy, gymnasium, pydantic v2, click, stdlib `sqlite3` / `http.server` / `threading`, pytest (+pytest-timeout), ruff, uv. **No new dependencies** (no scipy, pandas, openskill, streamlit or JS libraries).

**Spec:** `docs/superpowers/specs/2026-10-16-sp4-selection-design.md` (Russian; binding authority; approved by the owner 2026-10-16). Read it before any task. Research behind it: `research/sp4_ratings_selection.md`, `research/sp4_selection_practice.md`.

Earlier records:
- SP3 records: spec `docs/superpowers/specs/2026-10-10-sp3-league-design.md`, acceptance report `docs/superpowers/reports/2026-10-10-sp3-acceptance.md` (rulings in «Решения контроллера во время SP3»).
- SP2/SP1 rulings: the ends of `docs/superpowers/reports/2026-10-09-sp2-acceptance.md` and `docs/superpowers/reports/2026-10-08-sp1-acceptance.md`.
- GPU checks: `docs/superpowers/reports/2026-10-15-gpu-checks.md`.

## Plan files

| File | Spec blocks | Tasks |
|---|---|---|
| `00-overview.md` (this file) | constraints, file map, interface contract, task index | — |
| `01-matchlog-ratings.md` | §6 compat fixture; 1 match log; 2 ratings; 7 metrics and league state | T0.1, T1.1–T1.3, T2.1–T2.2, T3.1–T3.2 |
| `02-retention-tournament-pulse.md` | 4 retention; 3 tournament and selection; 6 pulse | T4.1–T4.2, T5.1–T5.4, T6.1 |
| `03-report-docs-acceptance.md` | 5 report and dashboard; 8 SP5 boundary; 9 docs; measurements; slow tests; acceptance | T7.1–T7.3, T8.1–T8.5 |

## Global Constraints

These apply to every task.

- **Python and dependencies:** Python 3.12 (`requires-python = ">=3.11"` stays); `torch>=2.6`. No new runtime or dev dependencies: `sqlite3`, `http.server`, `threading`, `json`, `hashlib` and numpy only. grpc stays in the extra `grpc`, wandb in `wandb`.
- **Environment (owner's CUDA machine, WSL2):**
  - prefix every `.venv/bin/python` / `.venv/bin/colosseum` / `.venv/bin/ruff` command with `env -u PYTHONPATH`. `~/.bashrc` sources ROS, whose `PYTHONPATH` breaks pytest at startup;
  - never install into system Python. CUDA torch is installed by `scripts/setup-dev.sh --gpu`; do not reinstall torch.
- **Clocks (WSL2 wall clock jumps by ~±9 s, GPU checks report):**
  - order data by counters (`env_steps`, `train_step`, row ids, versions), never by wall-clock `ts`;
  - measure durations, timeouts and cadences with `time.monotonic()`. Tests never assert on `ts` order or on wall-clock rates;
  - wall-clock timestamps are informational only.
- **Backward compatibility:**
  - SP3 configs (`configs/examples/*`, and `tests/fixtures/sp2/configs/*` as before) validate without edits;
  - an SP3 run dir (fixture of T0.1: no `matches.db`, no `league_state.json`) resumes, with an empty journal and empty league state plus INFO;
  - SP3 checkpoints play in tournaments as run snapshots and as frozen agents;
  - `MatchResult(...)`, `SeatResult(...)` and `Lineup(...)` built without the new fields keep working (defaults).
- **Green suite after every task:** the task's new tests pass; `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` passes in full **with zero warnings**; `env -u PYTHONPATH .venv/bin/ruff check .` passes.
  - Tasks that touch AMP / CUDA paths also run `pytest -m gpu -q -W error` (CUDA is present on this machine).
  - Tasks that touch training math, the worker or the learner run the slow suite once (`pytest -m slow -v`, ~7 min here).
- **Inter-process data:** nothing that crosses a process boundary (`mp.Queue`, pipes, `mp.Process` kwargs) contains `torch.Tensor`. Weights travel as numpy state dicts (pulse included), configs as pydantic models, bots as `BotSpec`.
- **Threads in the main process:**
  - the journal writer and the live-rating thread own their SQLite connections and their data;
  - they communicate with the main loop only through bounded queues and immutable published snapshots;
  - they never call `Coordinator`, `CheckpointManager`, `PfspStats` or `RatingBook` methods. Retention and `on_evict` run on the main loop.
- **SQLite:**
  - stdlib `sqlite3`; WAL mode; one connection per thread or process;
  - writers use `busy_timeout` and short transactions (one batch);
  - readers open read-only (`file:...?mode=ro`, `uri=True`) and never hold a transaction longer than one query.
- **Disk is bounded by construction** (spec principle 3): every growing store has a limit and a defined behaviour at the limit. A journal error never stops training (ERROR + counter), except schema/config errors at startup (`ConfigError`).
- **Errors:**
  - config problems: `ConfigError` with a fix hint;
  - env contract violations: `EnvContractError`; bot problems: `PlayerError` (SP2/SP3 formats);
  - no silent fallbacks beyond the spec's (and the inherited ones). Every lost match result is counted (`dropped_match_results`) and the first loss logs a WARNING.
- **Report / dashboard:**
  - no external resources: no CDN, web fonts or JS libraries. CSS / JS / SVG code lives in package data under `src/colosseum/arena/report/assets/`, and the page embeds its data as JSON;
  - the dashboard binds to `127.0.0.1` by default;
  - page text is English (code-side UI), docs are Russian.
- **Tests:**
  - files only under pytest's `tmp_path` (fixtures under `tests/fixtures/` are read-only inputs; copy before writing);
  - `@pytest.mark.gpu` for CUDA-only tests, `@pytest.mark.slow` for tests longer than ~20 s;
  - at most 2 worker processes per test (tournament and pulse processes count);
  - no `__init__.py` under `tests/`; support modules imported by bare name; **test basenames and support-module names unique across `tests/`**. New SP4 test files use the prefix `test_sp4_`;
  - integration runs start processes through `tests/cli_runner.py`;
  - toy games and tiny models come from `tests/game_helpers.py` (extend it, do not duplicate it);
  - synthetic statistics tests use fixed seeds and tolerance bands chosen so a correct implementation fails with probability < 1e-3 (say how in the test docstring).
- **SP1–SP3 rulings stay in force** unless the spec (§8 «Изменения правил SP1–SP3») changes them. Most likely to be broken by accident:
  - agent id regex and reserved ids;
  - strict explicit resume; `SHUTDOWN_GRACE_SEC = 7`; no process alive after 10 s;
  - eval loads checkpoints read-only (`load_checkpoint_dir`, never a `CheckpointManager` on a user path);
  - newest-wins weight queues; exactly `learner.batch_chunks` chunks per update; `training.total_timesteps` as the global env-step budget;
  - the legality gate; the error context format; owner rotation in config order;
  - PFSP statistics of evicted snapshots are never revived (SP3 #50); the retention index is updated before `on_evict` (SP3 #24);
  - `record` / `bc` / `eval` validate in the "play" scope (SP3 #48, #53, #54) — the tournament, `report` and `pin` use it too.
- **Config:** new sections `match_log`, `ratings`, `evaluation` and the new `checkpoint` keys are global only (no per-agent override in SP4). **Any config-schema change also updates `scripts/bench_throughput.py::_make_config`** (guarded by `tests/unit/test_bench_throughput.py`).
- **Language:** code, comments, docstrings, log messages, CLI help, report/dashboard UI text, `README.md`, `CLAUDE.md` in English; `docs/ARENA_GUIDE.md`, `docs/LEAGUE_GUIDE.md`, `docs/benchmarks.md`, `docs/GPU_CHECKS.md` and the acceptance report in Russian.
- **Commits:** conventional prefixes; no attribution or co-author lines; at least one commit per task on branch `sp4-selection`; push after every task (`git push`).
- **Style:** follow existing code (type hints, dataclasses, pydantic, `logger = logging.getLogger(__name__)`, `from __future__ import annotations`, module docstrings that state the contract).
- **Process (owner's instruction):** one Opus/high implementer subagent per task (TDD, commit, push; no subagents of its own), an Opus/high reviewer per task; web research, if any, by Sonnet/high subagents.

## Deviations from the spec

- **Numbering:** spec §7 parts P0–P8 map to task groups T0–T8 (P0 → T0.1, P1 → T1.x, …, P8 → T8.x); the acceptance is T8.5.
- **`rating_version` is computed by the journal, not carried by `SeatResult`.** The spec lists it among the new `SeatResult` fields. The plan computes it at write time from `version_start` and the run's `checkpoint.interval` (stored in `runs`) and keeps it in the `seats` table: one rule, one place. `SeatResult` carries `version_start` / `version_end` only.
- **Explicit seeds and block ids ride on `Lineup`:** `Lineup.seed`, `Lineup.block`, `Lineup.uid` (all optional). `MatchRunner` resets an env with `lineup.seed` when it is set (else the SP2/SP3 derived seed) and copies `block` / `uid` into the `MatchResult`. The spec says "`play_lineups` и `MatchRunner` получают явный seed на состав"; this is how.
- **Package `colosseum.arena`:** the spec (block 8) names the libraries generically; the plan puts them in one new package (module list in "File structure").

---

## File structure

### Create
- `src/colosseum/arena/__init__.py` — package docstring (what lives where); no eager heavy imports.
- `src/colosseum/arena/entities.py` — entity keys, version buckets, lineage, self-play test (T1.2).
- `src/colosseum/arena/comparisons.py` — `TeamKey`, `PairStat`, `ComparisonTable`, `match_pairs`, score statistics (T1.2).
- `src/colosseum/arena/matchlog.py` — schema, `MatchLog` (append / read / compact / import / sizes), `MatchLogWriter` thread (T1.2, T1.3).
- `src/colosseum/arena/bt.py` — `BTOptions`, `fit`, `laplace`, `bootstrap`, groups / components, anchor, `transitivity_test`, `score_summary` (T2.1).
- `src/colosseum/arena/live.py` — `LiveRatings` thread, `RatingsSnapshot`, the `bt` section of `ratings.json` (T2.2).
- `src/colosseum/arena/retention.py` — `SnapshotInfo`, `SnapshotScore`, `RetentionPolicy`, `RetentionDecision`, `retain` (T4.1).
- `src/colosseum/arena/tournament/__init__.py`, `players.py` (tournament players and entity names; T5.1), `engine.py` (block runner over worker processes; T5.1), `schedule.py` (blocks, rotations, compositions, seed domains; T5.2), `stats.py` (candidate scores, paired stratified bootstrap, `confirmed`, repeat flags; T5.3), `modes.py` (gauntlet / round-robin / halving / select; T5.3), `run.py` (tournament dir, settings check, resume, snapshot links, journal; T5.4).
- `src/colosseum/arena/pulse.py` — pulse process main and requests (T6.1).
- `src/colosseum/arena/report/__init__.py`, `data.py` (section builders; T7.1), `render.py` + `assets/{page.html,page.css,page.js}` (T7.2), `server.py` (T7.3).
- `src/colosseum/core/run_state.py` — `run_state.json` (`running` / `finished` / `failed`, pid) (T7.1).
- `tests/fixtures/sp3/` — SP3 run-dir fixture + generator (T0.1).
- `examples/<game>/ladder.py` (or in the existing `bots.py`) — ladder bots of known strength for two demo games (T8.2).
- `docs/ARENA_GUIDE.md` (T8.4).

### Modify (main ones)
- `src/colosseum/core/types.py`: `MatchResult`, `SeatResult`, `Lineup` new optional fields; `MATCH_SOURCES` (T1.1).
- `src/colosseum/core/config.py`:
  - `MatchLogConfig` (T1.1);
  - `RatingsConfig` (T2.2);
  - `CheckpointConfig` keys `keep_top_k`, `top_k_quantile`, `top_k_min_games`, `max_snapshots` (T4.1);
  - `EvaluationConfig` (T6.1).
- `src/colosseum/worker/match_runner.py` — versions per seat, explicit lineup seed / block / uid, source, uid prefix (T1.1).
- `src/colosseum/worker/rollout_loop.py`, `worker/rollout_worker.py` — version function, cumulative `dropped_results`, uid prefix with run id (T1.1); cumulative `dropped_reward_episodes` (T3.1).
- `src/colosseum/launcher.py`:
  - results queue 10 000, `env_steps` stamp (T1.1);
  - journal writer, compaction cadence, resume import (T1.3);
  - live ratings, BLAS threads (T2.2);
  - league state save / load (T3.2);
  - retention after refit (T4.1);
  - pulse process (T6.1);
  - `run_state.json` (T7.1).
- `src/colosseum/core/run_dir.py` — `run_id`, paths `matches_path`, `league_state_path`, `report_path`, `tournaments` (T1.3, T3.2, T7.1).
- `src/colosseum/coordinator/coordinator.py` — compact ratings snapshot (T2.2), league state (T3.2), retention scores provider (T4.1).
- `src/colosseum/coordinator/ratings.py`, `league/pfsp.py` — `state_dict` / `load_state_dict` (T3.2).
- `src/colosseum/coordinator/checkpoint_manager.py` — `retain` integration, `run_id` / `pinned` in `meta.json`, `pin`, `reapply_retention`, `max_snapshots` (T4.1).
- `src/colosseum/metrics/aggregator.py`, `metrics/hub.py`, `metrics/console.py`, `metrics/wandb_logger.py`:
  - `system` fields: `dropped_match_results`, journal sizes, degraded flag (T1.3);
  - compact `ratings` record + `bt` (T2.2);
  - monotonic counters (T3.1).
- `src/colosseum/algorithms/appo.py` — `rho_trunc_mass` / `c_trunc_mass` (T3.1).
- `src/colosseum/eval.py`, `src/colosseum/record.py` — shared player loading and agent-spec parsing; `(agent, network)` player pool; explicit seeds (T5.1).
- `src/colosseum/cli.py`:
  - `tournament` (T5.4), `pin` (T4.2), `report` (T7.2), `dashboard` (T7.3);
  - `bc` device log (T3.1).
- `src/colosseum/core/validation.py` — new sections, retention printout, pulse field checks (T2.2, T4.1, T6.1).
- `src/colosseum/distributed.py` — scope checks, `max_snapshots` FIFO (T4.1, T8.1).
- `scripts/bench_throughput.py` (every schema task), `scripts/units_experiment.py`, `scripts/measure_amp_logratio.py` (T3.1).
- Docs:
  - `README.md`, `docs/LEAGUE_GUIDE.md`, `docs/GPU_CHECKS.md`, `docs/benchmarks.md`, `CLAUDE.md` (T3.1 metric names; T8.4);
  - `configs/examples/unit_harvest_league.yaml` (T8.4).

---

## Interface contract

Names that cross task or part boundaries. Each part may add names in its own `## Contract notes`; conflicts are reconciled into "Contract amendments" below before execution.

### `colosseum.core.types` (T1.1)

```python
MATCH_SOURCES = ("train", "tournament", "pulse")

@dataclass
class Lineup:
    layout: str
    seats: list[SeatAssignment]
    seed: int | None = None      # explicit episode seed (tournament, pulse); None = MatchRunner's derived seed
    block: str | None = None     # tournament/pulse block id, copied into MatchResult.block
    uid: str | None = None       # deterministic match uid (tournament); None = runner-made uid

@dataclass
class SeatResult:                # existing fields unchanged, then:
    version_start: int | None = None   # weights version at episode start (latest: worker's version; ckpt_vN: N; fixed: None)
    version_end: int | None = None

@dataclass
class MatchResult:               # existing fields unchanged, then:
    uid: str = ""                # globally unique (run id / source / worker / env / episode, or the lineup's uid)
    source: str = "train"        # one of MATCH_SOURCES
    seed: int | None = None      # episode seed when known
    env_steps: int | None = None # stamped by the receiver (training main loop); pulse: the snapshot's env_steps
    block: str | None = None
```

### `colosseum.worker.match_runner.MatchRunner` (T1.1)
New keyword arguments, all with defaults that keep SP3 behaviour:
- `version_fn: Callable[[str], int | None] | None = None` — the current weights version of an agent's `latest` seats;
- `source: str = "train"`;
- `uid_prefix: str = ""` — prepended to the runner-made uid.

The version of a `ckpt_v<N>` seat is `N`; of a `fixed` seat `None`. `version_start` is read when the episode starts, `version_end` when it ends.

### `colosseum.core.config` (T1.1, T2.2, T4.1, T6.1)

```python
class MatchLogConfig(StrictModel):
    enabled: bool = True
    detail_window_mb: float = 512.0    # > 0
    min_free_disk_mb: float = 2048.0   # >= 0
    compact_interval_sec: float = 60.0 # > 0

class RatingsConfig(StrictModel):
    anchor: str | None = None          # an agent name of the config (checked in validate); None = default rule
    refit_interval_sec: float = 60.0   # > 0
    bootstrap_samples: int = 200       # >= 20

class CheckpointConfig(StrictModel):   # existing fields unchanged, then:
    keep_top_k: int = 0                # >= 0; > 0 needs match_log.enabled
    top_k_quantile: float = 0.1        # 0 < q < 0.5
    top_k_min_games: int = 200         # >= 0
    max_snapshots: int | None = 100    # >= 1 or None; < keep_last is a ConfigError

class EvaluationConfig(StrictModel):
    enabled: bool = False              # True needs match_log.enabled and a non-empty field
    agents: list[str] | None = None    # trainable agents; None = all
    field: list[str] | None = None     # agent names of the config; None = every scripted/frozen agent
    layouts: list[str] | None = None   # None = every enabled layout the agent plays
    every: int = 1                     # >= 1
    blocks: int = 25                   # >= 1, per field member (or composition)
    num_envs: int = 8                  # >= 1
    deterministic: bool = False
    max_backlog: int = 4               # >= 1

class ColosseumConfig:                 # new top-level sections
    match_log: MatchLogConfig
    ratings: RatingsConfig
    evaluation: EvaluationConfig
```

### `colosseum.arena.entities` (T1.2)

```python
EntityKey = str   # "agent@v<N>" (trainable seats), "agent" (scripted / frozen), tournament names (T5.1)

def snapshot_version(network_id: str) -> int | None              # "ckpt_v1200" -> 1200; "latest"/"fixed" -> None
def rating_version(network_id: str, version_start: int | None, interval: int) -> int | None
    # ckpt_vN -> N; latest -> floor(version_start / interval) * interval; fixed -> None
def entity_key(agent_id: str, rating_version: int | None) -> EntityKey   # None -> agent_id
def lineage(entity: EntityKey) -> str                             # "main@v1200" -> "main"; "b/main@v3" -> "b/main"
```

### `colosseum.arena.comparisons` (T1.2)

```python
@dataclass(frozen=True)
class TeamKey:
    team: int                                   # team index in the layout (side / roles; spec block 2)
    members: tuple[tuple[EntityKey, str], ...]  # sorted (entity, role) pairs

@dataclass
class PairStat:
    n: int = 0        # matches
    sw: float = 0.0   # sum of weights 1/(T-1)
    swy: float = 0.0  # sum of weight * outcome (1 / 0.5 / 0 from ranks)

class ComparisonTable:
    """Sufficient statistics of team pairs per (layout, source); plus score sums for one-team layouts."""
    def add_result(self, result: MatchResult, rating_versions: Sequence[int | None]) -> None
    def add(self, layout: str, source: str, a: TeamKey, b: TeamKey, stat: PairStat) -> None
    def merge(self, other: "ComparisonTable") -> None
    def layouts(self) -> list[str]
    def pairs(self, layout: str, sources: Collection[str]) -> Iterable[tuple[TeamKey, TeamKey, PairStat]]
    def scores(self, layout: str, sources: Collection[str]) -> Iterable[tuple[TeamKey, ScoreStat]]  # n, sum, sumsq
    def entity_games(self, layout: str, sources: Collection[str]) -> dict[EntityKey, int]  # self-play excluded

def match_pairs(result: MatchResult, rating_versions: Sequence[int | None]) -> list[tuple[TeamKey, TeamKey, float, float]]
    # (a, b, weight, outcome_a) for every pair of teams; [] for one-team layouts
```

### `colosseum.arena.matchlog` (T1.2, T1.3)

```python
MATCH_LOG_FILE = "matches.db"
SCHEMA_VERSION = 1

@dataclass(frozen=True)
class RunInfo:
    run_id: str
    name: str
    parent_run_id: str | None
    game: str
    config_hash: str
    checkpoint_interval: int

class MatchLog:
    def __init__(self, path: str | Path, run: RunInfo, *,
                 disk_free_fn: Callable[[Path], int] = ...,
                 min_free_disk_mb: float = 2048.0) -> None   # creates / migrates the schema; WAL
    @classmethod
    def open_readonly(cls, path: str | Path) -> "MatchLogReader"
    def append(self, results: Sequence[MatchResult]) -> AppendStats  # one transaction; counts-only under the disk guard
    def compact(self, *, window_bytes: int, max_batch: int = 10_000,
                evicted: Collection[EntityKey] = ()) -> CompactStats
    def import_parent(self, parent_path: str | Path) -> ImportStats  # counts + tournament + pulse; parent detail -> counts
    def comparison_table(self, sources: Collection[str] = ("train", "pulse"),
                         tournament: str | None = None) -> ComparisonTable
    def sizes(self) -> LogSizes                                      # file, wal, detail estimate, counts rows
    def close(self) -> None

class MatchLogReader:   # read-only API shared by report, dashboard, tournament
    def matches(self, *, agent: str | None = None, layout: str | None = None, source: str | None = None,
                tournament: str | None = None, limit: int = 1000) -> list[MatchRow]
    def comparison_table(self, sources=..., tournament=None) -> ComparisonTable
    def blocks(self, tournament: str) -> set[str]                    # block ids written whole
    def runs(self) -> list[RunInfo]
    def sizes(self) -> LogSizes

class MatchLogWriter:   # training main process: a thread with its own MatchLog connection
    def __init__(self, log_factory: Callable[[], MatchLog], *, queue_size: int = 100_000,
                 compact_interval_sec: float, window_bytes: int, table: ComparisonTable | None = None) -> None
    def start(self) -> None
    def submit(self, results: Sequence[MatchResult]) -> int          # returns the number dropped (queue full)
    def note_evicted(self, entities: Collection[EntityKey]) -> None
    def stop(self, timeout: float) -> WriterStats                    # flush, stop compaction, close
    @property
    def stats(self) -> WriterStats                                   # written, dropped, degraded, compactions, sizes
```

### `colosseum.arena.bt` (T2.1)

```python
@dataclass(frozen=True)
class BTOptions:
    prior_sigma: float          # logits; module constants chosen by T8.2 (rulings)
    smoothness: float           # random-walk precision between consecutive versions of one lineage
    anchor: str | None = None   # config anchor name or None (default rule, spec block 2)
    max_iter: int = 50
    tol: float = 1e-8

@dataclass
class BTGroup:
    layout: str
    group: str                  # role-group label (e.g. "default", "hunter")
    component: int
    entities: list[EntityKey]
    anchor: EntityKey | str     # entity or "mean-of-fixed"
    
@dataclass
class BTFit:
    layout: str
    groups: list[BTGroup]
    rating: dict[EntityKey, float]      # Elo units (400 / ln 10 per logit), anchor = 0 within its group/component
    se: dict[EntityKey, float]          # Laplace standard errors (Elo units)
    bias: dict[int, float]              # team index -> Elo units, sums to 0 over the layout's teams
    games: dict[EntityKey, int]         # comparisons with another entity (self-play excluded)
    iterations: int
    converged: bool

def fit(table: ComparisonTable, layout: str, *, sources: Collection[str], options: BTOptions,
        fixed_agents: Collection[str] = (), random_bot: str | None = None,
        warm: BTFit | None = None) -> BTFit
def laplace(fit_: BTFit, *, samples: int = 1000, rng: np.random.Generator,
            top_k: int = 0, quantile: float = 0.1) -> Uncertainty
def bootstrap(fit_: BTFit, data: BootstrapData, *, samples: int, rng: np.random.Generator,
              top_k: int = 0, quantile: float = 0.1) -> Uncertainty  # rows (Poisson) or blocks (paired, stratified)

@dataclass
class Uncertainty:
    ci95: dict[EntityKey, tuple[float, float]]
    lcb: dict[EntityKey, float]           # quantile `quantile`, in the lineage-contrast scale (spec block 4)
    p_best_lineage: dict[EntityKey, float]
    p_best_group: dict[EntityKey, float]
    p_top_k: dict[EntityKey, float]

def transitivity_test(table: ComparisonTable, fit_: BTFit, *, sources: Collection[str],
                      min_games: int = 20) -> TransitivityResult   # G2, dof, p_value, worst_pairs
def score_summary(table: ComparisonTable, layout: str, *, sources: Collection[str]) -> ScoreSummary  # one-team layouts
```

### `colosseum.arena.live` (T2.2)

```python
@dataclass(frozen=True)
class RatingsSnapshot:
    version: int                                   # increments per refit
    env_steps: int
    fits: Mapping[str, BTFit]                      # per layout
    uncertainty: Mapping[str, Uncertainty]
    scores: Mapping[str, ScoreSummary]

class LiveRatings:   # a thread in the training main process
    def __init__(self, table: ComparisonTable, *, interval_sec: float, options: BTOptions,
                 fixed_agents, random_bot, top_k: int, quantile: float, env_steps_fn: Callable[[], int]) -> None
    def start(self) -> None
    def stop(self, timeout: float) -> None
    def latest(self) -> RatingsSnapshot | None
def bt_section(snapshot: RatingsSnapshot) -> dict    # the "bt" key of ratings.json (spec block 2)
def bt_summary(snapshot: RatingsSnapshot, agents: Sequence[str]) -> dict  # compact per agent for the ratings record
```

### `colosseum.coordinator` (T3.2, T4.1)

- `RatingBook.state_dict() -> dict` / `load_state_dict(state, *, agents)`; `PfspStats.state_dict()` / `load_state_dict(state, *, agents)`. The forgotten set is included; unknown agents are dropped with INFO.
- `Coordinator.save_league_state(path)` / `load_league_state(path) -> bool`. `LEAGUE_STATE_FILE = "league_state.json"`.
- `Coordinator.ratings_snapshot(*, compact: bool = False)`: compact replaces each layout's `pfsp` table by `pfsp_summary` (T2.2).
- `CheckpointManager(..., run_id: str | None = None, keep_top_k: int = 0, top_k_min_games: int = 200, max_snapshots: int | None = None, scores_fn: Callable[[str], Mapping[str, SnapshotScore]] | None = None)`;
  - `pin(agent_id, checkpoint_id, pinned=True)`, `pinned(agent_id) -> list[str]`;
  - `reapply_retention(agent_id) -> list[str]` (evicted ids; main loop only).
  - `meta.json` gains `run_id` and `pinned`. `final` protects only `run_id == current run`.

### `colosseum.arena.retention` (T4.1)

```python
@dataclass(frozen=True)
class SnapshotInfo:
    checkpoint_id: str
    version: int
    final: bool
    run_id: str | None
    pinned: bool

@dataclass(frozen=True)
class SnapshotScore:
    lcb: float | None    # None = not rated yet
    games: int

@dataclass(frozen=True)
class RetentionPolicy:
    keep_last: int
    keep_every_period: int    # interval * keep_every; 0 = off
    keep_top_k: int
    top_k_min_games: int
    max_snapshots: int | None
    current_run_id: str | None

@dataclass(frozen=True)
class RetentionDecision:
    keep: list[str]
    evict: list[str]
    reasons: dict[str, str]   # checkpoint_id -> why kept / evicted (validate, logs)
    warnings: list[str]

def retain(entries: Sequence[SnapshotInfo], scores: Mapping[str, SnapshotScore], policy: RetentionPolicy,
           *, protect: str | None = None) -> RetentionDecision
```

### `colosseum.arena.tournament` (T5.1–T5.4)

- `players.TournamentPlayer`: `name` (entity key used in the journal: `main@v1200`, `alias/main@v1200`, `greedy`, `prev_sub#<hash8>`), `display`, `kind`, `roles`, `lineage`, plus how to build it (`snapshot dir`, `.pt` + networks, `BotSpec`, config agent).
- `players.resolve_candidates(...)` and `players.resolve_field(...)` follow the spec block 3 rules.
- `engine.play_blocks(config, spec, players, blocks, *, num_workers, num_envs, deterministic, on_block) -> None`: worker processes with lazy model loading and an LRU cache. `on_block(block_id, results)` is called once per finished whole block, in the tournament main process.
- `schedule.Block`: `id`, `stage` (`"A" | "B" | "P"`), `layout`, `candidate`, `composition`, `seed`, `lineups: list[Lineup]` (each with `seed`, `block`, `uid`).
- `schedule.gauntlet_blocks(spec, layout, candidates, field, roles, *, blocks, seed, stage) -> list[Block]`, `round_robin_blocks(...)`, `score_blocks(...)`, `stage_seed(seed, stage, index) -> int` (domains A / B / P never collide).
- `stats.candidate_scores(results_by_block, candidates, field, lineage) -> CandidateTable`; `stats.paired_bootstrap(...)`; `stats.confirmed(table_B, winner, runner_up, *, level=0.95) -> bool`; `stats.repeated_pairs(...)`.
- `modes.run(mode, ...) -> TournamentReport`; `TournamentReport.to_dict()` / `.text()`.
- `run.TournamentDir(path)`: `tournament.yaml` (resolved settings + seeds), the settings check, `snapshots/` links, `report.{json,txt,html}`.

### `colosseum.arena.pulse` (T6.1)

```python
@dataclass
class PulseRequest:              # main -> pulse (numpy only)
    agent_id: str
    checkpoint_id: str
    version: int
    env_steps: int
    state_dict: dict[str, np.ndarray]

def pulse_main(config: ColosseumConfig, *, requests: mp.Queue, results: mp.Queue, stop_event,
               log_dir: str, field_specs: ...) -> None
```

### `colosseum.arena.report` (T7.1–T7.3) and `colosseum.core.run_state` (T7.1)

- `data.build_report_data(run_dir: Path, *, include_tournaments: bool = True, limits: ReportLimits = ...) -> dict`. Sections: `summary`, `ratings`, `matrix`, `anchors`, `opponents`, `training`, `system`, `matches`, `tournaments` (spec block 5). `data.section(run_dir, name, **filters) -> dict` is used by the dashboard API.
- `render.render_html(data: dict, *, live: bool = False, refresh_sec: int = 10) -> str`; `render.write_report(run_dir, out=None) -> Path`.
- `server.serve(run_dir, *, host="127.0.0.1", port=8765, refresh_sec=10) -> None`; `GET /`, `GET /api/<section>`.
- `run_state.write_run_state(run_dir, status, pid)` / `read_run_state(run_dir) -> RunState`; statuses `running | finished | failed`.

### `colosseum.core.run_dir.RunDir` (T1.3, T3.2, T7.1)
New properties: `matches_path`, `league_state_path`, `report_path`, `tournaments_dir`, `run_state_path`, and `run_id` (uuid4 hex, created once per run dir and stored in `run_state.json`, or in a small `run.json` if T7.1 is not done yet — T1.3 decides and records it in its contract notes).

### Test support (`tests/game_helpers.py`)
- `SP3_RUN`, `copy_sp3_run(tmp_path)` (T0.1);
- `make_match_result(...)` builders for synthetic journals (T1.2);
- `synthetic_bt_matches(strengths, layout_kind, ...)` (T2.1);
- ladder bots (T8.2).

---

## Contract amendments

Binding; they override the contract above where they differ (reconciliation of the three parts, 2026-10-16). Pre-flight rulings P1, P2, … live in the SDD workspace `constraints.md`.

- **A1 (T1.1, T1.2).** The package `colosseum.arena` (`__init__.py`) and `arena/entities.py` are created by **T1.1** (`MatchRunner` needs `snapshot_version`); T1.2 adds `comparisons.py` and `matchlog.py`. `entities` also has `entity_version(entity) -> int | None`; `entity_key(agent_id, version)`.
- **A2 (T1.1; T4.1, T5.4, T6.1, T7.1).** The run id lives in `<run>/run.json` (`{"run_id": <uuid4 hex>}`, `run_dir.RUN_INFO_FILE`), not in `run_state.json`. `read_run_id(root)` never writes; `RunDir.run_id` is read-only (None until created); `RunDir.ensure_run_id()` is get-or-create; `Launcher._run_id` is set at the start of `launch()`. Launcher-side consumers use `self._run_id` (T4.1 `Coordinator(run_id=)`, T6.1 pulse prefix); code that may meet a run dir without `run.json` calls `ensure_run_id()` (T4.1 distributed learner, T5.4 `--run`, also on SP3 run dirs). `run_state.json` (T7.1) carries no run id.
- **A3 (T1.1).** Worker side: `LoopIO.report_result` returns False for a dropped result; `RolloutLoop.stats["dropped_results"]` (cumulative); `report_worker_stats(..., timeout=None)`; a final stats report and `flush_queue` of the results and stats queues at worker exit; `launcher._RESULTS_QUEUE_SIZE = 10_000`; `_drain_results` stamps `env_steps` only on results that have none.
- **A4 (T1.2).** `MatchLog(path, run: RunInfo | None, *, disk_free_fn=default_disk_free, min_free_disk_mb=2048.0)`: registering a `run_id` already in `runs` is idempotent and a second writer process works (WAL, `busy_timeout`); `run=None` means version buckets of width 1. `append(results, *, tournament: str | None = None) -> AppendStats(detailed, counted, duplicates, accepted, degraded)` stores the tournament id (column `matches.tournament`); only `source == "train"` matches without a tournament id are compacted or diverted to `counts` by the disk guard. `comparison_table(sources=RUN_SOURCES, tournament=None)`: `tournament=None` reads only matches outside tournaments, `tournament=T` only that tournament's detailed matches. `compact(*, window_bytes, max_batch=10_000, evicted=(), should_stop=...)`, `rating_versions(result)`, `degraded`, `import_parent(path) -> ImportStats(counts_rows, folded, copied)`; `MatchLogError`, `estimated_size`, `default_disk_free`.
- **A5 (T1.2).** `comparisons`: `ScoreStat(n, total, total_sq)`; `TeamKey.of / entities / encode / decode`; `ComparisonTable.add_score / copy / version / sources / as_dict`; `match_score`, `team_keys`, `pairs_of`, `score_of`, `coefficients`, `Pair`. `counts` keys are id-encoded. `LogSizes(file_bytes, wal_bytes, detail_bytes, detail_matches, other_matches, counts_rows, free_disk_bytes, degraded)` — there is no `counts_bytes` (T7.1 reads it only if present).
- **A6 (T1.2).** `MatchLogReader`: `matches(*, agent=None, version=None, layout=None, source=None, tournament=None, outcome=None, env_steps=None, limit: int | None = 1000)` newest first by row id (`limit=None`: every row; `outcome` ∈ `win|draw|loss` of `agent`'s team against the best other team, needs `agent`; `version` is a rating version; `env_steps` an inclusive range), `match_results(*, source, tournament=None) -> list[MatchResult]` oldest first, `comparison_table`, `blocks(tournament)`, `runs()`, `sizes()`, `close()`, context manager; a missing file reads as an empty journal.
- **A7 (T1.2).** `MatchRow(id, uid, run_id, source, layout, outcome_kind, length, seed, env_steps, block, tournament, worker, ts, seats: tuple[SeatRow, ...], teams: tuple[(team, rank, score), ...], replay=None)` with `to_dict()` = `{uid, run_id, source, layout, outcome_kind, episode_length, seed, env_steps, block, tournament, worker, replay, seats: [{seat, team, role, agent_id, network_id, entity, category, version_start, version_end, rating_version, reward, eliminated_step}], teams: [{team, rank, score}]}` (no `ts`; seats carry `entity` / `category`, not a player `kind`) and `to_result() -> (MatchResult, rating_versions)` (`match_id = uid`; a seat's `source` = its category).
- **A8 (T1.2).** `MatchLog` / `MatchLogReader.bootstrap_units(layout, sources) -> (counts rows [(a, b, PairStat)], [(block id | None, [(a, b, weight, outcome)])])` over matches outside tournaments — the input of `bt.bootstrap_data`.
- **A9 (T1.2, T5.2, T5.4, T6.1).** Block ids are `"<prefix>/<stage>/<layout>/<candidate>/<composition>/<index>"`, match uids `"<block id>/r<rotation>"`; the tournament id is the prefix (a tournament name: no `/`). T5.4 passes it explicitly: `append(block_results, tournament=block_id.split("/", 1)[0])`, one whole block per call; resume reads `blocks(name)` and `match_results(source="tournament", tournament=name)`. Pulse blocks have the prefix `pulse-<run id[:12]>` and no tournament id. Tournament seats: a run snapshot is `SeatAssignment(agent_id="[alias/]A", network_id="ckpt_v<N>")`, every other player `(<name>, "fixed")`, so `entity_key` yields the block-2 names; T5.4 opens `MatchLog(db, RunInfo, min_free_disk_mb=...)` (the `--run`'s run, or a tournament RunInfo under `--output`). Part 1's assumption "tournaments write with `MatchLog(path, None)` and network `latest`" is superseded.
- **A10 (T2.1).** `BTOptions(prior_sigma=PRIOR_SIGMA, smoothness=SMOOTHNESS, anchor=None, max_iter=50, tol=1e-8, dense_limit=DENSE_LIMIT, agent_order=())`; `DEFAULT_OPTIONS`; `default_options(anchor=None)`; `fit(table, layout, *, sources, options=DEFAULT_OPTIONS, fixed_agents=(), random_bot=None, agent_order=(), warm=None)` uses `agent_order or options.agent_order` for the "first trainable agent" anchor rule. Module constants `PRIOR_SIGMA`, `SMOOTHNESS` (provisional until T8.2), `ELO_PER_LOGIT`, `Z_95`, `DENSE_LIMIT`, `MEAN_OF_FIXED`, `TRANSITIVITY_P`; `chi2_sf`.
- **A11 (T2.1).** `BootstrapData(counts, matches, blocks)` (`blocks = {stratum: {key: [Contribution]}}`, keys drawn per stratum, the same draw for every contribution of a key), `Contribution`; `BootstrapData.from_table(table, layout, *, sources)` (rows only); `BootstrapData.from_blocks(results_by_block, *, strata=None)` (with `strata[block_id]`, a block's key is the id's last component, the seed index shared by every candidate: paired, stratified); `bootstrap_data(reader, layout, *, sources)` (counts rows, detailed training matches, pulse blocks with stratum = block id without its index). `--snapshots top:N` (T5.1) and the report (T7.1) use `bootstrap_data`; the tournament's BT (T5.3) uses `from_blocks`.
- **A12 (T2.1).** `Uncertainty(ci95, lcb, p_best_lineage, p_best_group, p_top_k, p_top_k_group, contrast_samples)`: `contrast_samples[entity]` are the per-draw lineage contrasts (Elo, float32) — the "draws" of T4.1's `combine_scores` (Part 2's name `Uncertainty.draws` is replaced by `contrast_samples`). `BTFit` adds `bias_se`, `approximate`, `params`, `design`, `options`, `hessian`, `group_of(entity)`; `BTGroup` adds `anchor_members`, `rule`, `note`; group labels are the `+`-joined sorted role names (`"player"`, `"hunter+prey"`). `TransitivityResult(g2, dof, p_value: float | None, flagged, pairs_tested, worst_pairs: [(a, b, residual, matches)])` with `to_dict()` = `{flag, g2, dof, p_value, pairs_tested, worst_pairs: [{a, b, team_a, team_b, residual, games}]}` (team labels `+`-joined member entities). `ScoreSummary(layout, entities: {entity: ScoreCell}, compositions, homogeneous)` with `by_entity` = `{entity: (n, mean, standard error)}` (every match the entity sat in) and `to_dict()` = `{layout, entities: {e: {n, mean, ci_low, ci_high}}, compositions: {k: {n, mean, ci_low, ci_high, homogeneous}}}`; `ScoreCell(n, mean, ci_low, ci_high)`.
- **A13 (T2.2).** `arena.live`: `LiveRatings(table, *, interval_sec, options, fixed_agents, random_bot, agent_order, top_k, quantile, env_steps_fn, samples=LAPLACE_SAMPLES, seed=0)` with `start`, `stop(timeout)`, `latest()`, `refit_now()`, `last_refit_sec`; `RatingsSnapshot(..., transitivity, refit_sec)`; `compute_snapshot`, `random_bot_agent(config)`, `rating_inputs(config) -> (BTOptions with ratings.anchor and the config's agent order, fixed agent ids, the first RandomBot agent)` — the one place that turns a config into fit inputs (launcher, report, `top:N`, scripts, slow tests); `console_bt`, `LAPLACE_SAMPLES`, `FINAL_REFIT_MAX_SEC`. The last refit runs at the end after the journal writer flushed, unless the previous refit took longer than `FINAL_REFIT_MAX_SEC`; the final `ratings.json` is written from it.
- **A14 (T2.2; T7.1, T8.3).** `ratings.json["bt"][layout][<group label>]` is a list of rows `{entity, lineage, version, rating, se, ci_low, ci_high, lcb, games, p_best (within the cell), p_best_lineage, [p_top_k], component, anchor: bool}` sorted by rating; per-layout extras under keys starting with `@`: `@bias` (`{"<team>": Elo}`), `@anchors` (`{"<label>/<component>": {anchor, members, rule}}`), `@transitivity` (= `TransitivityResult.to_dict()`), `@fit`, `@score`. This replaces Part 3's R-P1-6 shape; readers skip `@` keys.
- **A15 (T2.2).** `MetricsHub.maybe_tick / close(..., ratings: dict | Callable[[], dict], match_log=None, bt=None)` (the ratings snapshot is built only on a tick); `ConsoleReporter.format_line(..., bt=None)`; `league.pfsp.pfsp_summary`, `compact_ratings`; `Coordinator.ratings_snapshot(*, compact=False)`; `utils.process.BLAS_THREAD_VARS`, `limit_main_blas_threads()` (called first in `colosseum train`).
- **A16 (T1.3, T3.1).** The `system` record gains the required top-level `dropped_match_results` (workers' cumulative drops + the writer's) and `match_log` (`WriterStats.as_dict()`: `written, dropped, duplicates, write_errors, compactions, folded, degraded, pending` and the `LogSizes` fields, or null); the counts-only flag is `match_log.degraded`. `metrics.aggregator.CUMULATIVE_WORKER_KEYS`, `SystemStats.cumulative(key)`, `MetricsHub.system_cumulative(key)`; T3.1 adds `dropped_reward_episodes` to the cumulative keys (top level, monotonic).
- **A17 (T1.3, T2.2, T4.1, T6.1).** Launcher: attributes `_match_log_writer`, `_comparison_table`, `_live_ratings`; methods `_open_match_log(coordinator)`, `_on_snapshot_evicted`, `_match_log_stats`, `_stop_match_log`, `_open_live_ratings`, `_live_snapshot`, `_finish_live_ratings`. Order in `launch()`: `_open_metrics` → `_open_match_log` → `_open_live_ratings` → `_wire_retention` (T4.1) → `_start_children`. Order in `_finish_metrics`: `_drain_results` → `_pump_pulse` (T6.1) → `_drain_metrics` → `_stop_match_log` → `_finish_live_ratings` → `hub.close`.
- **A18 (T1.3, T7.1).** `RUN_SOURCES = ("train", "pulse")` is defined once, in `arena.matchlog`; `arena.report.data` imports it (and `entity_version` from `arena.entities`) instead of redefining it.
- **A19 (T1.3, T4.1).** `Coordinator.add_evict_listener(listener)` is T1.3's (listeners run right after PFSP forgets the snapshot, before the buffering early return); evictions reach the journal only through T1.3's `Launcher._on_snapshot_evicted`. T4.1 adds `Coordinator(..., run_id=None)`, `set_ratings_source(fn)`, `snapshot_scores(agent_id)` and no second listener.
- **A20 (T6.1).** Pulse results travel on their own queue `Launcher._pulse_results` (one message per snapshot, blocking put until the stop) and are submitted to the journal writer only (not to the coordinator or the episodes aggregator); `_pump_pulse` runs in the monitor loop, in the shutdown poll and in `_finish_metrics` before `_stop_match_log`. Part 1's assumption "pulse results come through the workers' results queue" is superseded (`_drain_results`' stamping rule A3 stays).
- **A21 (T3.2).** `Coordinator.save_league_state(path, *, env_steps)`, `load_league_state(path) -> bool`, `LEAGUE_STATE_FILE`, `LEAGUE_STATE_VERSION`; `RunDir.league_state_path`; `RatingBook` / `PfspStats` `state_dict()` / `load_state_dict(state, *, agents)`.
- **A22 (T4.1).** `arena.retention`: the contract's `retain` and dataclasses plus `combine_scores(agent_id, *, fits, uncertainty, scores, layout_weights, quantile)` (reads `BTFit.rating / games`, `Uncertainty.lcb / contrast_samples`, `ScoreSummary.by_entity`), `snapshot_scores(snapshot, agent_id, *, layout_weights, quantile)`, `describe_retention(checkpoint)`. T4.1 changes `_open_live_ratings` to `top_k=checkpoint.keep_top_k, quantile=checkpoint.top_k_quantile`. The new `CheckpointManager` arguments are keyword-only; `run_id=None` keeps the SP3 rule (every final protected).
- **A23 (T4.1, T4.2).** `checkpoint_manager.set_pinned`, `read_snapshots`, `link_or_copy` (public; `_link_or_copy` stays an alias), `parse_snapshot_ref`, `pin_snapshot`, `list_snapshots`, `SNAPSHOT_REF_RE`; `Launcher._wire_retention`, `_reapply_retention`; `distributed_checkpoint_manager(config, base_dir, run_id=None)`.
- **A24 (T5.1, T5.2).** `schedule.Block`, `STAGES`, `stage_seed` are created in T5.1, the builders in T5.2 and take `players: Mapping[name, SeatPlayer]` (plus `first_index`, `prefix`); `players.resolve_players(config, spec, request, *, fixed, link=None, rng=None, allow_empty_field=False)` replaces `resolve_candidates` / `resolve_field`; `TournamentPlayer` gains `agent_id`, `network_id`, `source`, `version`, `run`, `run_agent`; `engine.play_blocks(..., log_dir=None, source="tournament")`, `LazyPlayerPool`, `make_tasks`, `TournamentError`; shared eval/record names of Part 2 C9 (`eval.parse_agent_spec`, `balanced_match_count`, `load_frozen_player`, `play_lineups(..., pool=None, source=...)`, `players.registry.checkpoint_roles`, `ValidationReport.fixed`, `MatchRunner(..., snapshot_fallback=True)`).
- **A25 (T5.3).** `stats.candidate_scores(blocks, results_by_block, *, names, lineage, group_of, metric_of, deterministic)` → tables per (stage, layout, group); `CandidateTable.bootstrap_cells / bootstrap_scores` are the paired bootstrap; `confirmed(diff_replicates, *, level=0.95, conflict=False)`; `CandidateTable.repeated` / `repeated_conflict`; `modes.run(mode, *, spec, players, candidates, field, settings, play, bt_fn=None) -> TournamentReport`; `TournamentReport.to_dict()` is T5.3's report schema (`format, kind, name, mode, seed, …, tables, rounds, selection, bt, games, flags`); `modes.DEFAULT_BLOCKS / DEFAULT_HALVING_BLOCKS / DEFAULT_CONFIRM_BLOCKS / DEFAULT_FINALISTS` (T8.2 sets them); `modes.bt_section` builds `transitivity` from `TransitivityResult.to_dict()`.
- **A26 (T5.3; T7.1, T7.2, T8.2, T8.3).** `modes.report_view(data) -> dict` turns the `to_dict()` schema into the per-layout / per-group view `{name, mode, seed, [winner, confirmed], layouts: {layout: {outcome_kind, groups: {group: {candidates: [{name, display, lineage, score, score_ci, worst, bt, bt_ci, p_best, games, eliminated_round}], ranking, matrix: {c: {col: {score, ci, games}}}, rounds: [{round, blocks, survivors, eliminated}], select: {winner, runner_up, confirmed, diff, diff_ci, indistinguishable} | None, transitivity, repeated_pairs}}}}}`; idempotent on a view. `report.json` keeps the `to_dict()` schema; the run report's tournaments section, the tournament page and `arena_kit` (`find_report`, `select_snapshots`) apply `report_view`. This replaces Part 3's R-P2-1 as a statement about `to_dict()`.
- **A27 (T5.4, T7.2).** `TournamentDir.write_reports(report) -> list[Path]` is the single writer of `report.json`, `report.txt` and (through `arena.report.render.write_tournament_html(tournament_dir)`) `report.html`; T5.4 guards the import until T7.2, which removes the guard. There is no `render_tournament_html`.
- **A28 (T5.4).** CLI: `--blocks` (gauntlet / round-robin, per composition), `--halving-blocks` (halving / select, round 0), `--confirm-blocks`, `--finalists`, `--seed` (default 0), `--output DIR` (a run-dir-like layout `DIR/matches.db`, `DIR/tournaments/<name>/`; not with `--run`), `-c` optional with `--run`; a `select` ending with `confirmed: false` exits 0. `RunDir.tournaments_dir` is created by T5.4 (T7.1 adds only `run_state_path`, `report_path`). Part 3's R-P2-3 "`--blocks` sets select's first round" is replaced by `--halving-blocks`.
- **A29 (T6.1).** T6.1 creates `MetricsHub.set_system_value(key, value)` (merged at the top level of every later `system` record); the pulse counters are `pulse_skipped_snapshots`, `pulse_evaluated_snapshots`, `pulse_pending_snapshots` (Part 3's `pulse_skipped` / `pulse_evaluated` are these). `pulse.pulse_main(config, *, requests, results, stop_event, field_specs, field, agents, run_id)` (the contract's `log_dir` moves to `pulse_target(*, log_dir, **kwargs)`), `PulseRequest.final`, `PulseAgent`, `PulsePlan`, `pulse_plan`, `describe_pulse`, `pulse_blocks`, `PulseResults`, `PulseFeeder`, `PULSE_STATE_FILE`, `RunDir.pulse_state_path`; `runs/<name>/pulse_state.json`.
- **A30 (T7.1–T7.3).** `core.run_state`: `RUN_STATE_FILE`, `STATUSES`, `RunState(status, pid, exit_code, proc_start, extra)`, `write_run_state(run_dir, status, pid, *, exit_code=None)`, `read_run_state`, `process_start_ticks`, `process_alive`, `effective_status` (`running | stopped | finished | failed | unknown`), written only by `run_training`. `report.data`: `ReportBuilder`, `ReportLimits`, `SECTIONS`, `FILTERS`, `REPORT_FORMAT`, `MIN_CELL_GAMES`, `MetricsTail`, `thin`, `wilson`; `report.render`: `REPORT_FILE`, `asset`, `embed_json`, `is_run_dir`, `is_tournament_dir`, `tournament_page_data`, `write_tournament_html`; `report.server`: `make_server`, `DashboardServer`, `TABLE_MIN_AGE_SEC = 30.0`.
- **A31 (T7.1, T8.1, T8.2).** Test support: `tests/report_kit.py` (T7.1) and `tests/arena_kit.py` (T8.2), each added to `known-first-party`; ladder bots in `examples/tic_tac_toe/ladder.py` and `examples/tron/bots.py`; the scripts `measure_tournament_defaults.py`, `bt_crossval.py`, `measure_bt_cost.py`, `measure_pulse_cadence.py`; `bench_throughput.py` gets `CounterSampler`, `CLOCKS`, `SAMPLE_INTERVAL_S`, `load_tree_config`, `--clock`, `--tree` (T8.1). These are not in `tests/game_helpers.py` (overrides the overview's "ladder bots" bullet).
- **A32 (T8.3, T8.4).** After T8.3 the slow suite has 14 tests (SP3's 10 + ladder × 2 + pipeline + disk bound), about 20–25 min; the overview's "~7 min" applies until then.
- **A33 (index).** Task dependencies as in the task index below (changed: T3.1 → T1.3; T5.1 → T2.2, T4.1; T8.2 → T7.1).

## Cross-part execution notes

- T0.1 runs first, on unchanged `main` code: its fixture (an SP3 run dir) is the compatibility baseline every later task keeps green.
- T1.1 adds the record fields and the `match_log` config section; T1.2 is a pure library (no launcher changes); T1.3 wires the writer into training.
- Parts 2 and 3 consume `MatchLog`, `ComparisonTable` and `bt` only through the contract above. If an implementer needs a new public name, it goes into its task report as a contract note for the controller.
- T2.2's live ratings are the only `scores_fn` source for T4.1. Until T4.1 is done, `keep_top_k` stays rejected ("not supported yet"), the same way SP3 handled fields before their tasks.
- T5.1 changes `eval.py` / `record.py` shared code. Every SP2/SP3 eval/record test must stay green; behaviour changes are only the spec's (`.pt` overlay removed; explicit seeds optional).
- T8.2 (measurements) fixes module constants (`bt` prior / smoothness) and tournament defaults (`--blocks`, halving start, `--confirm-blocks`) as controller rulings. Earlier tasks use provisional values and say so in docstrings.
- Slow tests and measurements record their numbers in the task report; thresholds become controller rulings.

## Task index and execution order

| Task | Title | Depends on |
|---|---|---|
| T0.1 | SP3 run-dir fixture, compatibility baseline, result-flow tests on current behaviour | — |
| T1.1 | Match record fields, per-seat versions, explicit lineup seeds, counted result drops, `match_log` config | T0.1 |
| T1.2 | `arena.entities`, `arena.comparisons`, `arena.matchlog` (library: schema, append, counts, compaction, disk guard, import, reader) | T1.1 |
| T1.3 | Journal in training: writer thread, bounded queue, shutdown flush, compaction cadence, resume import, `system` journal fields, `run_id` | T1.2 |
| T2.1 | `arena.bt`: fit, bias by team index, groups and components, anchor, Laplace, bootstrap, G², score layouts | T1.2 |
| T2.2 | Live ratings: refit thread, `ratings.json` `bt`, compact `ratings` record, console and WandB, `ratings` config, BLAS threads | T1.3, T2.1 |
| T3.1 | `rho_trunc_mass` / `c_trunc_mass`, monotonic loss counters, `bc` device log | T1.3 |
| T3.2 | `league_state.json` (save, load on run-dir resume) | T2.2 |
| T4.1 | `arena.retention` + `CheckpointManager` (`run_id`, `pinned`, top-k, `max_snapshots`, finals), `validate` printout, distributed FIFO cap | T2.2 |
| T4.2 | `colosseum pin` | T4.1 |
| T5.1 | Tournament players and engine: shared player loading with `eval` / `record`, `(agent, network)` pool, explicit seeds, worker processes, eval / record dedup | T1.1, T2.2, T4.1 |
| T5.2 | Tournament schedules: blocks, admissible rotations, FFA compositions, `score` layouts, seed domains | T5.1 |
| T5.3 | Tournament statistics and modes: candidate scores, paired stratified bootstrap, halving, select, `confirmed`, repeat flags, several layouts | T5.2, T2.1 |
| T5.4 | `colosseum tournament`: CLI, several `--run`, snapshot selection, tournament dir and resume, snapshot links, env check, journal writes, `--pin` | T5.3, T4.2, T1.2 |
| T6.1 | Pulse process | T5.2, T1.3, T2.2 |
| T7.1 | Report data builders and `run_state.json` | T2.2, T5.4, T6.1 |
| T7.2 | HTML renderer and `colosseum report` | T7.1 |
| T7.3 | `colosseum dashboard` | T7.2 |
| T8.1 | Distributed guards and the bench config | T6.1, T4.1 |
| T8.2 | Ladder bots and measurements (tournament defaults, prior / smoothness by cross-validation, throughput by counters) | T5.4, T6.1, T7.1 |
| T8.3 | Slow tests: ladder on two games, pipeline with pulse and select, disk bound | T8.2, T7.2 |
| T8.4 | Docs: `ARENA_GUIDE.md`, `LEAGUE_GUIDE.md`, README, CLAUDE.md, example config | T8.3, T7.3 |
| T8.5 | Acceptance report | T8.4 |

Execution order: the table order (every dependency precedes its task in it; contract amendment A33).
