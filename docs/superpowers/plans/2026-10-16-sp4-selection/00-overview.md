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

(Filled after the three parts are written and reconciled; binding; they override the contract above where they differ. Pre-flight rulings P1, P2, … live in the SDD workspace `constraints.md`.)

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
| T3.1 | `rho_trunc_mass` / `c_trunc_mass`, monotonic loss counters, `bc` device log | T1.1 |
| T3.2 | `league_state.json` (save, load on run-dir resume) | T2.2 |
| T4.1 | `arena.retention` + `CheckpointManager` (`run_id`, `pinned`, top-k, `max_snapshots`, finals), `validate` printout, distributed FIFO cap | T2.2 |
| T4.2 | `colosseum pin` | T4.1 |
| T5.1 | Tournament players and engine: shared player loading with `eval` / `record`, `(agent, network)` pool, explicit seeds, worker processes, eval / record dedup | T1.1 |
| T5.2 | Tournament schedules: blocks, admissible rotations, FFA compositions, `score` layouts, seed domains | T5.1 |
| T5.3 | Tournament statistics and modes: candidate scores, paired stratified bootstrap, halving, select, `confirmed`, repeat flags, several layouts | T5.2, T2.1 |
| T5.4 | `colosseum tournament`: CLI, several `--run`, snapshot selection, tournament dir and resume, snapshot links, env check, journal writes, `--pin` | T5.3, T4.2, T1.2 |
| T6.1 | Pulse process | T5.2, T1.3, T2.2 |
| T7.1 | Report data builders and `run_state.json` | T2.2, T5.4, T6.1 |
| T7.2 | HTML renderer and `colosseum report` | T7.1 |
| T7.3 | `colosseum dashboard` | T7.2 |
| T8.1 | Distributed guards and the bench config | T6.1, T4.1 |
| T8.2 | Ladder bots and measurements (tournament defaults, prior / smoothness by cross-validation, throughput by counters) | T5.4, T6.1 |
| T8.3 | Slow tests: ladder on two games, pipeline with pulse and select, disk bound | T8.2, T7.2 |
| T8.4 | Docs: `ARENA_GUIDE.md`, `LEAGUE_GUIDE.md`, README, CLAUDE.md, example config | T8.3, T7.3 |
| T8.5 | Acceptance report | T8.4 |

Execution order: the table order.
