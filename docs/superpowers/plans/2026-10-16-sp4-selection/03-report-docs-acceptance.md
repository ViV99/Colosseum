# SP4 Plan — Part 3: report and dashboard, SP5 boundary, measurements, slow tests, docs, acceptance

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** Spec blocks 5, 8 and 9, the plan measurements of sections 3 and 8, the new slow tests and the acceptance of section 3:
- **T7.1** `colosseum.core.run_state` (`run_state.json`: running / finished / failed + pid, written by the launcher) and `colosseum.arena.report.data` (one JSON-ready dict per page section, built from a run dir: summary, ratings, matrix, anchors, opponents, training, system, matches, tournaments; `wdl` / `rank` / `score` layouts; size limits; counter ordering; running runs; SP3 run dirs) (block 5);
- **T7.2** `colosseum.arena.report.render` with its assets (`page.html`, `page.css`, `page.js`: own SVG charts, sortable and filterable tables, phone layout, light / dark), `colosseum report`, the tournament page (block 5; criterion 3.7);
- **T7.3** `colosseum.arena.report.server` and `colosseum dashboard` (block 5; criterion 3.7);
- **T8.1** distributed guards for `evaluation` / `keep_top_k`, the benchmark's pinned config with every SP4 section, the benchmark's counter clock (block 8; criteria 3.6, 3.9);
- **T8.2** ladder bots of known strength (tic_tac_toe, tron), `tests/arena_kit.py`, and the measurements that fix rulings: tournament default sizes, BT prior / smoothness by cross-validation, refit cost at ~2 000 entities, pulse cadence and the pipeline thresholds, throughput against `main` by counters (§3.3, §3.4, §3.6, §8 rulings 13);
- **T8.3** slow tests: the ladder on two games (3.3), the pipeline with pulse and `select` (3.4), the disk bound (3.5); the slow suite (3.11);
- **T8.4** documentation: `docs/ARENA_GUIDE.md` (new), `docs/LEAGUE_GUIDE.md` §12/§14/§15, `README.md`, `CLAUDE.md`, `configs/examples/unit_harvest_league.yaml` (block 9; criterion 3.10);
- **T8.5** acceptance against section 3, the report, the stop for the owner (section 7).

**Read `00-overview.md` first** (global constraints, file map, the binding interface contract). Names used here are the contract's. Parts 1 and 2 are consumed only through the contract; everything this part needs that the contract lacks is a requirement in `## Contract notes` (R-P1-*, R-P2-*), and accepted amendments in the overview win over this part. If an implementer finds a requirement met under another name, they use that name and say so in the task report.

**Execution order:** T7.1 → T7.2 → T7.3 → T8.1 → T8.2 → T8.3 → T8.4 → T8.5 (the index order). T8.2 and T8.3 are long measurement tasks (about 3 h and 1 h of machine time). They never run at the same time as each other or as any other CPU-heavy job (policy lag, and with it every learning number, depends on the load); before each measurement command check `uptime` (1-minute load average below 2.0 on the 20-core reference machine).

**Conventions used in every task.**
- Run every command from the repository root as `env -u PYTHONPATH .venv/bin/python ...` / `env -u PYTHONPATH .venv/bin/ruff ...` / `env -u PYTHONPATH .venv/bin/colosseum ...` (the owner's `~/.bashrc` sets a ROS `PYTHONPATH` that breaks pytest). The one exception is the `main` measurement of T8.2, which sets `PYTHONPATH` explicitly.
- "Full fast suite" = `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, **zero warnings**) followed by `env -u PYTHONPATH .venv/bin/ruff check .`.
- Every task ends with a commit on `sp4-selection` (conventional prefix, no attribution or co-author lines) and `git push`.
- New test files start with `test_sp4_`; new support modules under `tests/` (`report_kit.py` in T7.1, `arena_kit.py` in T8.2) get unique basenames and are added to `[tool.ruff.lint.isort] known-first-party` in `pyproject.toml` in the task that creates them.
- Integration runs start `train` through `tests/cli_runner.py`; `report`, `dashboard` and `tournament` CLI runs in tests go through `cli_runner.run_in_session` / `pipeline_kit.cli` (own session, killed on exit). At most 2 worker processes per test (tournament workers and the pulse process count); files only under `tmp_path`.
- Data order by counters (`train_step`, `env_steps`, versions, journal row order), never by `ts`; durations and timeouts with `time.monotonic()`. The machine's wall clock jumps (WSL2, GPU checks report §8): no test asserts on `ts` or wall-clock rates.
- Long commands (measurements, the slow suite) run in the background with output to a log file under the task's scratch directory (`nohup ... > log 2>&1 &`, or the tool's background mode); poll the log, never block a tool call for more than 10 minutes.

**Cross-task notes.**
- **The page contract.** `data.py` (T7.1) produces, `page.js` (T7.2) consumes one JSON shape (`REPORT_FORMAT = 1`), given in full in T7.1 («Report data shape»). Both sides are written against it; `report_kit.sample_report_data()` (T7.1) is a hand-made instance with every section, so the renderer is tested without a journal.
- **Already prototyped.** While writing this part, `page.html` / `page.css` / `page.js`, `data.py` (against stub arena modules with the contract's shapes), `server.py` and the run-state module were run end to end: headless Chrome (`--dump-dom`) drew all nine sections of both the sample data and the builder's output with no section error, the live page refreshed every section from `api/<section>`, and phone (390 px, dark) and desktop (1100 px, light) screenshots were checked by eye. The ladder bots were run on the real `MatchRunner` (`play_lineups`), and the benchmark's counter clock on this WSL2 machine (where `ts` windows fail). Code below is that code; implementers adapt it only where Parts 1 and 2 land with other names (contract notes).
- **Measurements and rulings.** The decision rules are fixed in this plan before any number is seen (`decide`, `choose`, `decide_cadence`, `decide_trend`, the slack rule of T8.3; unit-tested in T8.2). Implementers run the measurements and report raw numbers; the values they produce (tournament defaults, BT constants, `evaluation.every` / `blocks` of the pipeline test, `RHO_MIN`, `MAX_SECONDS`, `SLACK_MB`) are recorded by the controller as rulings ("что — почему — цена ошибки") in the SDD ledger, and T8.5 copies every ruling into the report.
- **Reference machine:** the owner's machine — WSL2, 20 cores, 25 GB RAM, RTX 4070 Ti; the slow suite of SP3 takes about 7 minutes there (423.96 s, GPU checks report). Budgets are sized for it.

---

### Task T7.1: Run state and report data builders

Spec block 5 («Разделы страницы», «Состояние запуска», «Размер страницы ограничен», the `score` rule, ordering by counters), principle «Порядок — по счётчикам», section 6 («отчёт и дашборд: построители данных для `wdl`, `rank` и `score`; состояние запуска по `run_state.json`»). The launcher records the run's state; `ReportBuilder` turns a run dir into one dict per section. Nothing in this task renders HTML.

**Files:**
- Create: `src/colosseum/core/run_state.py`
- Create: `src/colosseum/arena/report/__init__.py`, `src/colosseum/arena/report/data.py`
- Modify: `src/colosseum/core/run_dir.py` (properties `run_state_path`, `report_path`; `tournaments_dir` exists since T5.4; module docstring lists `run_state.json`, `report.html`)
- Modify: `src/colosseum/launcher.py` (`run_training` writes `run_state.json`)
- Create: `tests/report_kit.py` (support module); modify `pyproject.toml` (`known-first-party` += `report_kit`)
- Test: `tests/unit/test_sp4_run_state.py`, `tests/unit/test_sp4_report_data.py`

**Interfaces:**
- Consumes (contract): `MatchLog`, `MatchLog.open_readonly`, `MatchLogReader.matches / comparison_table / runs / sizes`, `RunInfo`, `MATCH_LOG_FILE` (T1.2/T1.3); `ComparisonTable.pairs / layouts`, `TeamKey.team / members`, `PairStat.n / sw / swy` (T1.2); `arena.entities.lineage` (T1.2); `bt.fit`, `bt.laplace`, `bt.bootstrap`, `bt.transitivity_test`, `bt.score_summary`, `BTFit.groups / rating / se / bias / games / iterations / converged`, `BTGroup.group / component / entities / anchor`, `Uncertainty.ci95 / lcb / p_best_lineage / p_best_group` (T2.1); the `bt` section of `ratings.json` (T2.2); `RatingsConfig.bootstrap_samples`, `EvaluationConfig.enabled / field` (T2.2, T6.1); `SeatResult.version_start / version_end`, `MatchResult.uid / source / seed / env_steps / block` (T1.1); `copy_sp3_run` (T0.1); `wr_vs_past_over_layouts` (`metrics/hub.py:53`); `write_text_atomic` (`utils/fs.py:9`); `load_config` (`core/config.py:1164`), `ColosseumConfig.agent_ids / agent_kind / fixed_agent_ids` (`core/config.py:976, 996, 1030`).
- Consumes (requirements, Contract notes): R-P1-1 `MatchLogReader.matches(..., outcome=, limit=None)` newest first and `MatchRow.to_dict()` / `to_result()`; R-P1-2 `MatchLogReader.close()`; R-P1-3 `bt.bootstrap_data(reader, layout, *, sources)`; R-P1-4 `TransitivityResult.to_dict()`, `ScoreSummary.to_dict()`; R-P1-5 `live.rating_inputs(config)`; R-P1-6 the `bt` section shape of `ratings.json` (Part 1's `bt_section`: a list of rows per group label, `@`-keyed extras; contract amendment A14); R-P1-7 `LogSizes` field names; R-P2-1 `arena.tournament.modes.report_view(report)` (contract amendment A26).
- Produces (contract + additions marked `*`):
  - `colosseum.core.run_state`: `RUN_STATE_FILE = "run_state.json"`, `STATUSES`, `RunState(status, pid, exit_code=None, proc_start=None, extra={})`, `write_run_state(run_dir, status, pid, *, exit_code=None) -> Path`, `read_run_state(run_dir) -> RunState | None`, `*process_start_ticks(pid)`, `*process_alive(state)`, `*effective_status(state) -> (status, detail)`;
  - `colosseum.arena.report.data`: `build_report_data(run_dir, *, include_tournaments=True, limits=None) -> dict`, `section(run_dir, name, **filters) -> dict`, `*ReportBuilder(run_dir, *, limits=None, include_tournaments=True, seed=0, table_min_age_sec=0.0)` with `build()`, `section(name, **filters)`, `close()`, `*ReportLimits(series_points=2000, entity_rows=2000, matches=1000, snapshot_matrix=100)`, `*SECTIONS`, `*FILTERS`, `*REPORT_FORMAT = 1`, `*RUN_SOURCES = ("train", "pulse")`, `*MIN_CELL_GAMES = 20`, `*MetricsTail`, `*thin`, `*entity_version`, `*wilson`;
  - `RunDir.run_state_path`, `RunDir.report_path` (`RunDir.tournaments_dir` is T5.4's);
  - `tests/report_kit.py`: `INTERVAL`, `VERSIONS`, `KINDS`, `make_results`, `write_run_dir`, `SyntheticRun`, `sample_report_data`.

**Report data shape** (`REPORT_FORMAT = 1`; the contract between `data.py` and `page.js`):

```text
{"format": 1, "title": str, "generated"?: str, "sections": {name: section}}      # every name of SECTIONS
summary:   {run, path, status: running|finished|failed|stopped|unknown, status_detail, pid, exit_code, env_steps,
            total_timesteps, budget_frac, env_steps_per_sec,
            agents: [{agent, kind, bt: {layout, entity, rating, ci: [lo, hi]} | null, elo, wr_vs_past}],
            parents: [run names], notes: [str]}
ratings:   {source: live|journal|none, uncertainty: laplace|bootstrap, env_steps, notes,
            layouts: {L: {outcome_kind, kind: "bt",
                          groups: [{label, component, anchor,
                                    entities: [{entity, lineage, version, kind, rating, ci, games, p_best_lineage,
                                                p_best_group, lcb}]   (sorted by rating, <= entity_rows),
                                    truncated: int,
                                    series: [{lineage, points: [[version, rating, lo, hi, games]]}]  (<= series_points),
                                    anchors: [{entity, rating, ci}]}],          # entities without a version
                          bias: {team: elo}, transitivity: {flag, g2, dof, p_value, worst_pairs: [{a, b, residual,
                          games}]} | null}
                     | {outcome_kind: "score", kind: "score", entities: [{entity, lineage, version, kind, n, mean, ci}],
                        compositions: [{composition, n, mean, ci, homogeneous}]}}}
matrix:    {min_games: 20, notes, layouts: {L: {kind: "bt", lineages: {axis: [names by rating], cells},
                                                snapshots: {agent, axis: [<= snapshot_matrix labels], cells,
                                                            thinned_from} | null}
                                            | {kind: "score", cross_play: [...as compositions]}}}
            cells[i][j] = {score, ci: [lo, hi], n} | null       (row i vs column j; n < min_games is drawn hatched)
anchors:   {source: pulse|train|none, biased: bool, notes,
            layouts: {L: {agent: {anchors: {anchor: [[version, score, lo, hi, n]]}, unevaluated: [versions]}}}}
opponents: {agents: {agent: {L: {env_steps: [...], categories: {cat: [...]}, anchors: {a: [...]}}}},
            pfsp: {L: {owner: [{player, score, games}]  (top 20 by games)}}, notes}
training:  {agents: {agent: {train_step: [...], metrics: {name: [...]}}}, default: [names], notes}
system:    {env_steps: [...], series: {dotted.name: [...]}, default: [names], latest: {dotted.name: float},
            free_disk_mb, journal: {file_mb, wal_mb, counts_rows?, detail_bytes?, counts_bytes?, counts_only} | null,
            notes}
matches:   {rows: [MatchRow.to_dict()] (<= matches, newest first), limit, filters: {agent, layout, source, outcome},
            notes}
tournaments: {items: [{name, path, report: <report.json of the tournament>}], notes}
```

Decisions (proposed rulings D1–D9 in the Contract notes): ratings of a `running` run (live pid) come from `ratings.json` (`bt`), of any other run from the journal with `ratings.bootstrap_samples` bootstrap replicas (spec block 2: bootstrap for reports), cached until the journal files change; a team belongs to a lineage (matrix, anchors) only if all its members belong to it (mixed teams are left out with a note); the lineage axis is ordered by the rating of the lineage's newest entity; the snapshot matrix shows the first trainable agent unless `agent=` is given; the anchors are the config's fixed agents (training games, marked "biased by PFSP") or, with `evaluation.enabled`, `evaluation.field` (pulse games only); journal flags of the `system` section are found by name suffix (`counts_only` / `degraded`); `run_state.json` says `finished` for exit codes 0 / 130 / 143 (final checkpoints were saved) and `failed` otherwise.

- [ ] **Step 1: Check that the new names are free and read the landed Part 1 / Part 2 names**

```bash
ls src/colosseum/core/run_state.py src/colosseum/arena/report tests/report_kit.py 2>/dev/null
find tests -name "test_sp4_run_state.py" -o -name "test_sp4_report_data.py"
grep -n "def matches\|def to_dict\|def to_result\|def close\|class LogSizes" -A3 src/colosseum/arena/matchlog.py | head -60
grep -n "def bootstrap_data\|def rating_inputs\|def bt_section\|def to_dict\|PRIOR_SIGMA\|SMOOTHNESS" src/colosseum/arena/bt.py src/colosseum/arena/live.py
grep -n "run_id\|matches_path\|league_state_path\|report_path\|tournaments_dir\|run_state_path" src/colosseum/core/run_dir.py
```

Expected: the first two commands print nothing. The greps show where R-P1-1 … R-P1-7 landed; where a name differs from this task's code, use the landed name and note it in the task report (a missing behaviour is reported to the controller before going on).

- [ ] **Step 2: Write the failing run-state tests**

Create `tests/unit/test_sp4_run_state.py`:

```python
"""run_state.json (SP4 spec block 5, T7.1): written by the launcher at start, normal end and failure;
readers decide liveness by the pid (and its start time on Linux), never by timestamps."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys

import pytest

from cli_runner import TTT_CONFIG, training_process, wait_for
from colosseum.core.run_state import (
    RUN_STATE_FILE,
    RunState,
    effective_status,
    process_alive,
    read_run_state,
    write_run_state,
)
from game_helpers import write_test_config


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_round_trip_keeps_keys_it_does_not_own(tmp_path):
    (tmp_path / RUN_STATE_FILE).write_text(json.dumps({"run_id": "abc", "status": "running", "pid": 1}))
    write_run_state(tmp_path, "finished", os.getpid(), exit_code=0)
    state = read_run_state(tmp_path)
    assert (state.status, state.pid, state.exit_code) == ("finished", os.getpid(), 0)
    assert state.extra == {"run_id": "abc"}
    assert json.loads((tmp_path / RUN_STATE_FILE).read_text())["run_id"] == "abc"


@pytest.mark.parametrize("content", [None, "not json", json.dumps({"status": "paused", "pid": 3}), "[]"])
def test_missing_or_malformed_files_read_as_none(tmp_path, content):
    if content is not None:
        (tmp_path / RUN_STATE_FILE).write_text(content)
    assert read_run_state(tmp_path) is None
    assert effective_status(None) == ("unknown", "no run_state.json")


def test_unknown_status_is_refused(tmp_path):
    with pytest.raises(ValueError, match="running"):
        write_run_state(tmp_path, "paused", os.getpid())


def test_liveness_by_pid_and_start_time_not_by_clock(tmp_path):
    write_run_state(tmp_path, "running", os.getpid())
    state = read_run_state(tmp_path)
    assert process_alive(state) and effective_status(state)[0] == "running"
    assert not process_alive(RunState("running", _dead_pid()))
    assert effective_status(RunState("running", _dead_pid()))[0] == "stopped"
    if state.proc_start is not None:        # Linux: a reused pid (other start time) is not the run
        assert not process_alive(RunState("running", os.getpid(), proc_start=state.proc_start + 1))
    assert effective_status(RunState("finished", 5, exit_code=130))[1].startswith("exit code 130")
    assert effective_status(RunState("failed", 5, exit_code=1)) == ("failed", "exit code 1")


def test_training_writes_running_then_finished_after_sigint(tmp_path):
    with training_process(TTT_CONFIG, tmp_path, overrides={"training.total_timesteps": "10000000"}) as (proc, root):
        assert wait_for(lambda: read_run_state(root) is not None, timeout=60)
        state = read_run_state(root)
        assert state.status == "running" and state.pid == proc.pid and process_alive(state)
        assert wait_for(lambda: (root / "metrics.jsonl").exists(), timeout=60)
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=60) == 130
    state = read_run_state(root)
    assert (state.status, state.exit_code) == ("finished", 130)


def test_a_launch_exception_marks_the_run_failed(tmp_path, monkeypatch, restore_root_logging, restore_global_rng):
    import colosseum.launcher as launcher

    def boom(self):
        raise RuntimeError("synthetic launch failure")

    monkeypatch.setattr(launcher.Launcher, "launch", boom)
    config = write_test_config(tmp_path / "cfg.yaml", "turns", run={"dir": str(tmp_path / "runs"), "name": "r"})
    with pytest.raises(RuntimeError, match="synthetic"):
        launcher.run_training(str(config))
    state = read_run_state(tmp_path / "runs" / "r")
    assert (state.status, state.exit_code, state.pid) == ("failed", 1, os.getpid())
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_run_state.py -q`

Expected: collection error `ModuleNotFoundError: No module named 'colosseum.core.run_state'`.

- [ ] **Step 3: The run-state module and the launcher**

Create `src/colosseum/core/run_state.py`:

```python
"""``runs/<name>/run_state.json``: whether a training run is running, finished or failed (SP4 spec block 5).

The launcher (``run_training``) writes ``running`` with its pid when the run dir exists, ``finished`` when
``launch`` returns 0 / 130 / 143 (budget reached, SIGINT, SIGTERM: final checkpoints were saved) and
``failed`` for any other exit code or an exception. Readers (report, dashboard) decide whether a
``running`` run is alive by its pid (``os.kill(pid, 0)``) and, on Linux, by the process start time from
``/proc/<pid>/stat`` recorded with it (a reused pid is not the run) — never by wall-clock timestamps.
Other keys of the file (e.g. ``run_id`` if T1.3 stored it here) are preserved on every rewrite.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colosseum.utils.fs import write_text_atomic

RUN_STATE_FILE = "run_state.json"
STATUSES = ("running", "finished", "failed")
_OWN_KEYS = ("status", "pid", "exit_code", "proc_start", "updated_ts")


@dataclass(frozen=True)
class RunState:
    status: str                     # one of STATUSES
    pid: int | None
    exit_code: int | None = None
    proc_start: int | None = None   # start time of ``pid`` in clock ticks since boot (Linux), pid-reuse guard
    extra: dict[str, Any] = field(default_factory=dict)


def process_start_ticks(pid: int) -> int | None:
    """Field 22 of ``/proc/<pid>/stat`` (start time in clock ticks after boot); None off Linux or if gone."""
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text()
        return int(text.rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _read_raw(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_run_state(run_dir: str | Path, status: str, pid: int, *, exit_code: int | None = None) -> Path:
    """Atomically (re)write ``run_state.json`` keeping keys this module does not own."""
    if status not in STATUSES:
        raise ValueError(f"run state must be one of {STATUSES}, got {status!r}")
    path = Path(run_dir) / RUN_STATE_FILE
    data = {k: v for k, v in _read_raw(path).items() if k not in _OWN_KEYS}
    data.update({"status": status, "pid": int(pid), "exit_code": exit_code,
                 "proc_start": process_start_ticks(pid), "updated_ts": time.time()})   # ts: informational only
    return write_text_atomic(path, json.dumps(data, indent=2, sort_keys=True))


def read_run_state(run_dir: str | Path) -> RunState | None:
    """The run's state, or None when the file is missing or malformed (SP3 run dirs have none)."""
    data = _read_raw(Path(run_dir) / RUN_STATE_FILE)
    if data.get("status") not in STATUSES:
        return None
    pid = data.get("pid")
    return RunState(status=data["status"], pid=int(pid) if isinstance(pid, int) else None,
                    exit_code=data.get("exit_code") if isinstance(data.get("exit_code"), int) else None,
                    proc_start=data.get("proc_start") if isinstance(data.get("proc_start"), int) else None,
                    extra={k: v for k, v in data.items() if k not in _OWN_KEYS})


def process_alive(state: RunState) -> bool:
    """Whether the process that wrote ``state`` still runs (pid exists and, where known, has the same start)."""
    if state.pid is None or state.pid <= 0:
        return False
    try:
        os.kill(state.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass                                   # exists, owned by another user
    if state.proc_start is not None:
        now = process_start_ticks(state.pid)
        if now is not None and now != state.proc_start:
            return False                       # the pid was reused by another process
    return True


def effective_status(state: RunState | None) -> tuple[str, str]:
    """``(status, detail)`` for display: running (alive), stopped (``running`` but the process is gone),
    finished, failed, unknown (no file)."""
    if state is None:
        return "unknown", "no run_state.json"
    if state.status == "running":
        if process_alive(state):
            return "running", f"pid {state.pid}"
        return "stopped", f"pid {state.pid} is gone without a final state (killed?)"
    code = "" if state.exit_code is None else f"exit code {state.exit_code}"
    if state.status == "finished" and state.exit_code in (130, 143):
        code += " (stopped by a signal; final checkpoints saved)"
    return state.status, code
```

In `src/colosseum/core/run_dir.py` add (`tournaments_dir` exists since T5.4) and list the files in the module docstring:

```python
    @property
    def run_state_path(self) -> Path:
        return self.root / "run_state.json"

    @property
    def report_path(self) -> Path:
        return self.root / "report.html"
```

In `src/colosseum/launcher.py::run_training` (`launcher.py:1009-1034`), replace the last lines from `print(f"Run directory: ...")` with:

```python
    print(f"Run directory: {run_dir.root}", flush=True)
    _record_run_state(run_dir, "running")
    try:
        code = Launcher(config, run_dir, validated=True).launch()
    except BaseException:
        _record_run_state(run_dir, "failed", exit_code=1)
        raise
    # 130 / 143: stopped by a signal after saving the final checkpoints (not a failure); 1: a dead child.
    _record_run_state(run_dir, "finished" if code in (0, 130, 143) else "failed", exit_code=code)
    if code == 0:
        logger.info(f"Training finished; outputs in {run_dir.root}")
    return code
```

and add next to `run_training`:

```python
def _record_run_state(run_dir: RunDir, status: str, *, exit_code: int | None = None) -> None:
    """run_state.json for report / dashboard (spec block 5); a failure to write it never stops a run."""
    from colosseum.core.run_state import write_run_state

    try:
        write_run_state(run_dir.root, status, os.getpid(), exit_code=exit_code)
    except OSError as e:
        logger.warning(f"cannot write {run_dir.root / 'run_state.json'}: {e}")
```

(`os` is imported by `launcher.py`; if not, add `import os`.) If T1.3 stored `run_id` in `run_state.json`, `write_run_state` keeps it (it rewrites only its own keys); if T1.3 created the file before this point, the first `write_run_state` call preserves its keys.

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_run_state.py -q`

Expected: 9 passed (the SIGINT test takes about 10 s).

- [ ] **Step 4: The synthetic run dirs (`tests/report_kit.py`)**

Create `tests/report_kit.py` and add `"report_kit"` to `known-first-party` in `pyproject.toml`:

```python
"""Synthetic run dirs and report data for the SP4 report and dashboard tests (T7.1-T7.3).

``write_run_dir`` builds a run dir the way a training run leaves it - ``config.resolved.yaml``,
``metrics.jsonl``, ``ratings.json``, ``matches.db`` (a real ``MatchLog``), ``run_state.json``,
``checkpoints/<agent>/ckpt_v<N>/`` - without training: matches between ``main`` (versions every
``INTERVAL``, strength rising with the version), ``greedy`` (strong scripted) and ``random`` are drawn from
known strengths with a fixed seed. Three outcome kinds: ``wdl`` (toy game ``turns``, layout ``2p``),
``rank`` (``ffa``, ``3p``), ``score`` (``solo``, ``solo``).

``sample_report_data`` is a hand-made report data dict covering every section (render tests need no journal).
"""
from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from colosseum.core.types import MatchResult, SeatResult, TeamResult
from game_helpers import make_test_config, scripted_agent

INTERVAL = 20
VERSIONS = tuple(range(0, 201, INTERVAL))
KINDS = {"wdl": ("turns", "2p", 2), "rank": ("ffa", "3p", 3), "score": ("solo", "solo", 1)}
FIXED_STRENGTH = {"greedy": 1.0, "random": 0.0}       # logits


def main_strength(version: int) -> float:
    return -1.0 + 2.5 * version / VERSIONS[-1]


@dataclass
class SyntheticRun:
    root: Path
    kind: str
    layout: str
    results: list[MatchResult]
    run_id: str


def _seat(seat: int, agent: str, version: int | None, *, latest: bool, offset: int = 3) -> SeatResult:
    if agent != "main":
        return SeatResult(seat, "player", seat, agent, "fixed", 0.0)
    if latest:     # latest weights: versions inside the bucket of ``version`` (rating_version = version)
        return SeatResult(seat, "player", seat, agent, "latest", 0.0, version_start=version + offset,
                          version_end=version + offset + 2)
    return SeatResult(seat, "player", seat, agent, f"ckpt_v{version}", 0.0, version_start=version,
                      version_end=version)


def make_results(kind: str, *, run_id: str, per_pair: int = 30, seed: int = 0, source: str = "train",
                 versions: Sequence[int] = VERSIONS) -> list[MatchResult]:
    """Matches of ``main@v`` against ``greedy`` and ``random`` (and, for ``rank``, all three together;
    for ``score``, homogeneous one-seat teams), outcomes drawn from the strengths; uids unique per call."""
    _, layout, teams = KINDS[kind]
    rng = np.random.default_rng(seed)
    results: list[MatchResult] = []

    def add(seats: list[SeatResult], ranks: list[float], scores: list[float]) -> None:
        i = len(results)
        block = f"P/{seats[0].version_start}/{i}" if source == "pulse" else None
        results.append(MatchResult(
            match_id=f"m{i}", layout=layout, outcome_kind="score" if teams == 1 else ("wdl" if teams == 2 else "rank"),
            seats=seats, teams=[TeamResult(t, ranks[t], scores[t]) for t in range(teams)], episode_length=5 + i % 4,
            uid=f"{run_id}/{source}/w0/e{i % 4}/ep{i}", source=source, seed=1000 + i, env_steps=10 * i, block=block))

    for v in versions:
        for k in range(per_pair):
            latest = k % 2 == 1
            if teams == 1:
                for agent in ("main", "greedy", "random"):
                    s = main_strength(v) if agent == "main" else FIXED_STRENGTH[agent]
                    score = float(2.0 + s + rng.normal(0.0, 0.5))
                    add([_seat(0, agent, v, latest=latest)], [1.0], [score])
                continue
            if teams == 2:
                for opp in ("greedy", "random"):
                    order = ["main", opp] if k % 4 < 2 else [opp, "main"]         # both sides
                    seats = [_seat(s, a, v, latest=latest) for s, a in enumerate(order)]
                    sa = main_strength(v) - FIXED_STRENGTH[opp]
                    p_main = 1.0 / (1.0 + math.exp(-sa))
                    draw = rng.random() < 0.1
                    main_wins = rng.random() < p_main
                    main_seat = order.index("main")
                    ranks = [1.0, 1.0] if draw else [0.0, 0.0]
                    if not draw:
                        ranks[main_seat] = 1.0 if main_wins else 2.0
                        ranks[1 - main_seat] = 2.0 if main_wins else 1.0
                    add(seats, ranks, [-r for r in ranks])
                continue
            order = ["main", "greedy", "random"]
            shift = k % 3
            order = order[shift:] + order[:shift]
            seats = [_seat(s, a, v, latest=latest) for s, a in enumerate(order)]
            noisy = [(main_strength(v) if a == "main" else FIXED_STRENGTH[a]) + rng.gumbel() for a in order]
            ranking = np.argsort(-np.asarray(noisy))
            ranks = [0.0] * 3
            for place, seat in enumerate(ranking):
                ranks[int(seat)] = float(place + 1)
            add(seats, ranks, [-r for r in ranks])
        if teams >= 2:          # latest against its own snapshot (self lineage: no lineage-matrix cell)
            for k in range(per_pair // 3):
                prev = max(0, v - INTERVAL)
                seats = [_seat(0, "main", v, latest=True), _seat(1, "main", prev, latest=False)]
                seats += [_seat(s, "random", None, latest=False) for s in range(2, teams)]
                ranks = [1.0, 2.0] + [3.0] * (teams - 2) if k % 2 else [2.0, 1.0] + [3.0] * (teams - 2)
                add(seats, ranks, [-r for r in ranks])
    return results


def _metrics_lines(kind: str, layout: str, *, train_records: int, decreasing_ts: bool) -> list[str]:
    lines = []
    for i in range(train_records):
        ts = 2.0e9 - i if decreasing_ts else 1.0e9 + i      # ts must not order anything (WSL2 clock jumps)
        lines.append({"ts": ts, "kind": "train", "agent": "main", "train_step": 10 * i, "total_loss": 1.0 / (1 + i),
                      "entropy": 1.0 - i / (2 * train_records), "rho_trunc_mass": 0.0, "lr": 1.0e-3})
    for i in range(40):
        lines.append({"ts": 1.0e9, "kind": "system", "env_steps": 1000 * i, "env_steps_per_sec": 900.0 + i,
                      "train_steps_per_sec": {"main": 5.0}, "queue_depths": {"main": 3}, "parked_buffers": 0,
                      "dropped_reward_episodes": 0, "dropped_match_results": 0, "workers_reporting": 1})
        if kind != "score":
            lines.append({"ts": 1.0e9, "kind": "episodes", "agent": "main", "env_steps": 1000 * i, "episodes": 10,
                          "return_mean": 0.1, "length_mean": 6.0, "wdl": {}, "seat_counts": [5, 5], "by_layout": {},
                          "opponents": {layout: {"teams": 10, "categories": {"latest": 0.5, "snapshots": 0.2,
                                                                              "rivals": 0.0, "anchors": 0.3,
                                                                              "fallback": 0.0},
                                                 "anchors": {"greedy": 0.2, "random": 0.1}}}})
    return [json.dumps(line) for line in lines]


def write_run_dir(tmp_path: Path, kind: str = "wdl", *, name: str = "synthetic", journal: bool = True,
                  status: str | None = "finished", pid: int | None = None, live_bt: bool = False,
                  evaluation: bool = False, pulse_versions: Sequence[int] = (), per_pair: int = 30,
                  train_records: int = 201, decreasing_ts: bool = True, partial_last_line: bool = False,
                  stored_versions: Sequence[int] = (160, 180, 200), bootstrap_samples: int = 20) -> SyntheticRun:
    """A finished (or ``status``) run dir of the given outcome ``kind`` under ``tmp_path / name``."""
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog, RunInfo
    from colosseum.core.run_state import write_run_state

    game, layout, _ = KINDS[kind]
    root = Path(tmp_path) / name
    root.mkdir(parents=True)
    # greedy is not a RandomBot, so the default anchor (the first RandomBot of the config) is ``random``
    agents = {"main": {}, "greedy": scripted_agent("game_helpers.ConstantBot"), "random": scripted_agent()}
    cfg = make_test_config(game, agents=agents, run={"name": name, "dir": str(tmp_path)},
                           checkpoint={"interval": INTERVAL}, ratings={"bootstrap_samples": bootstrap_samples},
                           evaluation={"enabled": evaluation})
    (root / "config.resolved.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json", by_alias=True),
                                                              sort_keys=False))
    run_id = uuid.uuid4().hex
    results = make_results(kind, run_id=run_id, per_pair=per_pair)
    if pulse_versions:
        results += make_results(kind, run_id=run_id, per_pair=4, seed=1, source="pulse", versions=pulse_versions)
    if journal:
        log = MatchLog(root / MATCH_LOG_FILE, RunInfo(run_id=run_id, name=name, parent_run_id=None, game=game,
                                                      config_hash="synthetic", checkpoint_interval=INTERVAL))
        try:
            for i in range(0, len(results), 500):
                log.append(results[i:i + 500])
        finally:
            log.close()
    text = "\n".join(_metrics_lines(kind, layout, train_records=train_records, decreasing_ts=decreasing_ts)) + "\n"
    if partial_last_line:
        text += '{"ts": 1.0, "kind": "train", "agent": "ma'
    (root / "metrics.jsonl").write_text(text)
    table = {"outcome_kind": "score" if kind == "score" else kind,
             "elo": {"main": 1250.0, "greedy": 1230.0, "random": 1010.0},
             "games": {"main": {"greedy": 100, "random": 100}}, "wr_vs_past": {"main": 0.6},
             "past_games": {"main": 50},
             "pfsp": {"main": {"greedy@fixed": {"score": 0.4, "games": 300.0},
                               "main@ckpt_v180": {"score": 0.6, "games": 40.0}}}}
    ratings = {"env_steps": 39000, "layouts": {layout: table}}
    if live_bt:
        # the shape of T2.2's live.bt_section (contract amendment A14): rows per role-group label, '@' extras
        def bt_row(entity, version, rating, lo, hi, games, lcb, p_lin, p_grp, anchor=False):
            return {"entity": entity, "lineage": entity.split("@v")[0], "version": version, "rating": rating,
                    "se": (hi - lo) / 3.92, "ci_low": lo, "ci_high": hi, "lcb": lcb, "games": games, "p_best": p_grp,
                    "p_best_lineage": p_lin, "component": 0, "anchor": anchor}

        ratings["bt"] = {layout: {
            "player": [bt_row("main@v200", 200, 123.0, 100.0, 146.0, 70, 105.0, 0.9, 0.6),
                       bt_row("greedy", None, 80.0, 60.0, 100.0, 700, 64.0, 1.0, 0.4),
                       bt_row("random", None, 0.0, 0.0, 0.0, 700, 0.0, 1.0, 0.0, anchor=True)],
            "@bias": {"0": 12.0, "1": -12.0},
            "@anchors": {"player/0": {"anchor": "random", "members": ["random"], "rule": "random_bot"}},
            "@transitivity": {"flag": False, "g2": 1.2, "dof": 1, "p_value": 0.27, "pairs_tested": 2,
                              "worst_pairs": []},
            "@fit": {"converged": True, "iterations": 5, "approximate": False}}}
    (root / "ratings.json").write_text(json.dumps(ratings))
    for v in stored_versions:
        (root / "checkpoints" / "main" / f"ckpt_v{v}").mkdir(parents=True)
    if status is not None:
        write_run_state(root, status, pid if pid is not None else os.getpid(),
                        exit_code=None if status == "running" else (0 if status == "finished" else 1))
    return SyntheticRun(root, kind, layout, results, run_id)


def sample_report_data() -> dict:
    """A report data dict with every section (wdl, rank and score layouts), for render tests."""
    versions = list(range(0, 2100, 100))
    main_rows = [{"entity": f"main@v{v}", "lineage": "main", "version": v, "kind": "trainable",
                  "rating": -50.0 + 8 * i, "ci": [-90.0 + 8 * i, -10.0 + 8 * i], "games": 50 + 10 * i,
                  "p_best_lineage": 0.05, "p_best_group": 0.04, "lcb": -80.0 + 8 * i} for i, v in enumerate(versions)]
    fixed = [{"entity": "greedy", "lineage": "greedy", "version": None, "kind": "scripted", "rating": 60.0,
              "ci": [40.0, 80.0], "games": 4000, "p_best_lineage": 1.0, "p_best_group": 0.2, "lcb": 45.0},
             {"entity": "random", "lineage": "random", "version": None, "kind": "scripted", "rating": 0.0,
              "ci": [0.0, 0.0], "games": 4000, "p_best_lineage": 1.0, "p_best_group": 0.0, "lcb": 0.0}]

    def cell(s: float, n: int) -> dict:
        return {"score": s, "ci": [max(0.0, s - 0.05), min(1.0, s + 0.05)], "n": n}

    snap_axis = [f"main@v{v}" for v in versions[::4]] + ["greedy", "random"]
    k = len(snap_axis)
    snap_cells = [[None if i == j else cell(0.5 + 0.03 * (j - i), 10 if (i + j) % 5 == 0 else 60) for j in range(k)]
                  for i in range(k)]
    steps = list(range(0, 200001, 4000))
    match_rows = [{"uid": f"r1/train/w0/e{i % 8}/ep{i}", "source": "pulse" if i % 3 == 0 else "train", "layout": "2p",
                   "outcome_kind": "wdl", "episode_length": 50, "seed": 1000 + i, "env_steps": 1000 * i, "block": None,
                   "replay": None,
                   "seats": [{"seat": 0, "team": 0, "role": "player", "agent_id": "main", "network_id": "latest",
                              "version_start": 100 * (i // 10), "version_end": 100 * (i // 10),
                              "rating_version": 100 * (i // 10), "reward": 1.0, "eliminated_step": None},
                             {"seat": 1, "team": 1, "role": "player", "agent_id": "greedy", "network_id": "fixed",
                              "version_start": None, "version_end": None, "rating_version": None, "reward": -1.0,
                              "eliminated_step": None}],
                   "teams": [{"team": 0, "rank": 1 + i % 2, "score": 1.0},
                             {"team": 1, "rank": 2 - i % 2, "score": 0.0}]}
                  for i in range(60)]
    tournament = {
        "name": "select-1", "mode": "select", "seed": 7, "winner": "main@v2000", "confirmed": True,
        "layouts": {"2p": {"outcome_kind": "wdl", "groups": {"default": {
            "candidates": [
                {"name": "main@v2000", "display": "main@v2000", "score": 0.71, "score_ci": [0.66, 0.76],
                 "worst": {"opponent": "greedy", "score": 0.49}, "bt": 120.0, "bt_ci": [90.0, 150.0], "p_best": 0.9,
                 "games": 400, "eliminated_round": None},
                {"name": "main@v1500", "display": "main@v1500", "score": 0.62, "score_ci": [0.57, 0.67],
                 "worst": {"opponent": "greedy", "score": 0.41}, "bt": 70.0, "bt_ci": [40.0, 100.0], "p_best": 0.1,
                 "games": 400, "eliminated_round": 1}],
            "ranking": ["main@v2000", "main@v1500"],
            "matrix": {"main@v2000": {"greedy": {"score": 0.49, "ci": [0.42, 0.56], "games": 200},
                                      "random": {"score": 0.93, "ci": [0.90, 0.96], "games": 200}}},
            "rounds": [{"round": 1, "blocks": 16, "survivors": ["main@v2000"], "eliminated": ["main@v1500"]}],
            "select": {"winner": "main@v2000", "confirmed": True, "indistinguishable": [], "diff_ci": [0.03, 0.15]},
            "transitivity": {"flag": False, "p_value": 0.4}, "repeated_pairs": []}}}}}
    return {
        "format": 1, "title": "synthetic-run", "generated": "colosseum report",
        "sections": {
            "summary": {"run": "synthetic-run", "path": "runs/synthetic-run", "status": "finished",
                        "status_detail": "exit code 0", "pid": 1234, "exit_code": 0, "env_steps": 160384,
                        "total_timesteps": 160000, "budget_frac": 1.0024, "env_steps_per_sec": 1620.5,
                        "agents": [{"agent": "main", "kind": "trainable", "elo": 1250.0, "wr_vs_past": 0.61,
                                    "bt": {"layout": "2p", "entity": "main@v2000", "rating": 110.0,
                                           "ci": [70.0, 150.0]}},
                                   {"agent": "greedy", "kind": "scripted", "elo": 1230.0, "wr_vs_past": None,
                                    "bt": {"layout": "2p", "entity": "greedy", "rating": 60.0, "ci": [40.0, 80.0]}},
                                   {"agent": "random", "kind": "scripted", "bt": None, "elo": 1010.0,
                                    "wr_vs_past": None}],
                        "parents": ["synthetic-parent"], "notes": []},
            "ratings": {"source": "journal", "uncertainty": "bootstrap", "env_steps": 160384, "notes": [], "layouts": {
                "2p": {"outcome_kind": "wdl", "kind": "bt", "bias": {"0": 12.0, "1": -12.0},
                       "transitivity": {"flag": True, "g2": 40.2, "dof": 12, "p_value": 0.0001,
                                        "worst_pairs": [{"a": "main@v300", "b": "greedy", "residual": 3.1,
                                                         "games": 120}]},
                       "groups": [{"label": "default", "component": 0, "anchor": "random",
                                   "entities": main_rows + fixed, "truncated": 0,
                                   "series": [{"lineage": "main", "points": [[r["version"], r["rating"], r["ci"][0],
                                                                              r["ci"][1], r["games"]]
                                                                             for r in main_rows]}],
                                   "anchors": [{"entity": f["entity"], "rating": f["rating"], "ci": f["ci"]}
                                               for f in fixed]}]},
                "4p": {"outcome_kind": "rank", "kind": "bt", "bias": {}, "transitivity": None,
                       "groups": [{"label": "default", "component": 0, "anchor": "random", "entities": fixed,
                                   "truncated": 3, "series": [], "anchors": []}]},
                "solo": {"outcome_kind": "score", "kind": "score",
                         "entities": [{"entity": "main@v100", "lineage": "main", "version": 100, "kind": "trainable",
                                       "n": 40, "mean": 3.2, "ci": [2.9, 3.5]}],
                         "compositions": [{"composition": "main@v100+random", "n": 20, "mean": 2.1, "ci": [1.8, 2.4],
                                           "homogeneous": False}]}}},
            "matrix": {"min_games": 20, "notes": [], "layouts": {
                "2p": {"kind": "bt",
                       "lineages": {"axis": ["main", "greedy", "random"],
                                    "cells": [[None, cell(0.48, 900), cell(0.97, 900)],
                                              [cell(0.52, 900), None, cell(0.99, 12)],
                                              [cell(0.03, 900), cell(0.01, 12), None]]},
                       "snapshots": {"agent": "main", "axis": snap_axis, "cells": snap_cells, "thinned_from": 23}},
                "solo": {"kind": "score", "cross_play": [{"composition": "main@v100+random", "n": 20, "mean": 2.1,
                                                          "ci": [1.8, 2.4], "homogeneous": False}]}}},
            "anchors": {"source": "train", "biased": True, "notes": [], "layouts": {"2p": {"main": {
                "unevaluated": [],
                "anchors": {"greedy": [[v, 0.3 + 0.01 * i, 0.25 + 0.01 * i, 0.35 + 0.01 * i, 40]
                                       for i, v in enumerate(versions)],
                            "random": [[v, 0.9, 0.85, 0.95, 40] for v in versions]}}}}},
            "opponents": {"notes": [], "agents": {"main": {"2p": {
                "env_steps": steps,
                "categories": {"latest": [0.5] * len(steps), "snapshots": [0.2] * len(steps),
                               "rivals": [0.0] * len(steps), "anchors": [0.3] * len(steps),
                               "fallback": [0.0] * len(steps)},
                "anchors": {"greedy": [0.15] * len(steps), "random": [0.15] * len(steps)}}}},
                "pfsp": {"2p": {"main": [{"player": "greedy@fixed", "score": 0.48, "games": 900.0},
                                         {"player": "main@ckpt_v300", "score": 0.7, "games": 120.0}]}}},
            "training": {"notes": [], "default": ["total_loss", "entropy", "rho_trunc_mass"], "agents": {"main": {
                "train_step": list(range(0, 1000, 10)),
                "metrics": {"total_loss": [1.0 / (1 + i) for i in range(100)],
                            "entropy": [1.5 - 0.005 * i for i in range(100)],
                            "rho_trunc_mass": [0.001 * (i % 7) for i in range(100)], "lr": [1.0e-3] * 100}}}},
            "system": {"notes": [], "env_steps": steps, "default": ["env_steps_per_sec", "dropped_match_results"],
                       "series": {"env_steps_per_sec": [1600.0 + (i % 5) * 10 for i in range(len(steps))],
                                  "dropped_match_results": [0.0] * len(steps),
                                  "queue_depths.main": [4.0] * len(steps)},
                       "latest": {"env_steps_per_sec": 1620.0, "dropped_match_results": 0.0,
                                  "dropped_reward_episodes": 0.0},
                       "free_disk_mb": 120000.0,
                       "journal": {"file_mb": 12.5, "wal_mb": 0.4, "counts_rows": 1200, "counts_only": False}},
            "matches": {"limit": 1000, "filters": {}, "notes": [], "rows": match_rows},
            "tournaments": {"notes": [], "items": [{"name": "select-1", "path": "tournaments/select-1",
                                                   "report": tournament}]},
        },
    }
```

`greedy` is a `ConstantBot` on purpose: the default anchor is the config's first `RandomBot` (spec block 2), which must be `random` for the strength-order test. Nothing here validates the config (only `load_config` reads it), so `ConstantBot` never plays.

- [ ] **Step 5: Write the failing report-data tests**

Create `tests/unit/test_sp4_report_data.py`:

```python
"""Report data builders (SP4 spec block 5, T7.1): every section for wdl / rank / score runs, size limits,
counter ordering, partial metrics lines, SP3 run dirs without a journal, live vs journal ratings, filters."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from colosseum.arena.report.data import (
    MIN_CELL_GAMES,
    SECTIONS,
    MetricsTail,
    ReportBuilder,
    ReportLimits,
    build_report_data,
    section,
    thin,
)
from game_helpers import copy_sp3_run
from report_kit import VERSIONS, write_run_dir


def _groups(data: dict, layout: str) -> list[dict]:
    return data["sections"]["ratings"]["layouts"][layout]["groups"]


def test_thin_keeps_first_and_last_and_bounds_the_count():
    assert thin(5, 10) == [0, 1, 2, 3, 4]
    idx = thin(10_001, 2000)
    assert len(idx) <= 2000 and idx[0] == 0 and idx[-1] == 10_000 and idx == sorted(set(idx))
    assert thin(7, 1) == [6]


@pytest.mark.parametrize("kind", ["wdl", "rank", "score"])
def test_every_section_is_built_for_each_outcome_kind(tmp_path, kind):
    run = write_run_dir(tmp_path, kind)
    data = build_report_data(run.root)
    assert data["format"] == 1 and set(data["sections"]) == set(SECTIONS)
    json.dumps(data, allow_nan=False)                       # embeddable as JSON
    ratings = data["sections"]["ratings"]
    assert ratings["source"] == "journal" and ratings["uncertainty"] == "bootstrap"
    layout = ratings["layouts"][run.layout]
    matrix = data["sections"]["matrix"]["layouts"][run.layout]
    if kind == "score":
        assert layout["kind"] == "score" and matrix["kind"] == "score"
        assert {r["lineage"] for r in layout["entities"]} == {"main", "greedy", "random"}
        assert layout["compositions"] and all(c["ci"][0] <= c["mean"] <= c["ci"][1] for c in layout["compositions"])
    else:
        (group,) = _groups(data, run.layout)
        names = {r["entity"] for r in group["entities"]}
        assert {"greedy", "random"} <= names and {f"main@v{v}" for v in VERSIONS} <= names
        (series,) = group["series"]
        assert series["lineage"] == "main" and [p[0] for p in series["points"]] == list(VERSIONS)
        assert {a["entity"] for a in group["anchors"]} == {"greedy", "random"}
        assert set(matrix["lineages"]["axis"]) == {"main", "greedy", "random"}
        assert matrix["snapshots"]["agent"] == "main"
        anchors = data["sections"]["anchors"]
        assert anchors["source"] == "train" and anchors["biased"] is True
        assert set(anchors["layouts"][run.layout]["main"]["anchors"]) == {"greedy", "random"}
        assert data["sections"]["opponents"]["agents"]["main"][run.layout]["categories"]["anchors"][0] == 0.3
        assert data["sections"]["opponents"]["pfsp"][run.layout]["main"][0]["player"] == "greedy@fixed"
    summary = data["sections"]["summary"]
    assert summary["status"] == "finished" and summary["env_steps"] == 39000
    assert {a["agent"]: a["kind"] for a in summary["agents"]} == {"main": "trainable", "greedy": "scripted",
                                                                 "random": "scripted"}
    assert data["sections"]["matches"]["rows"] and data["sections"]["tournaments"]["items"] == []


def test_strength_order_survives_into_ratings_and_matrix(tmp_path):
    run = write_run_dir(tmp_path, "wdl", per_pair=60)
    data = build_report_data(run.root)
    rating = {r["entity"]: r["rating"] for r in _groups(data, "2p")[0]["entities"]}
    assert rating["random"] == 0.0                              # the default anchor: the config's RandomBot
    assert rating[f"main@v{VERSIONS[-1]}"] > rating["main@v0"]
    assert rating["greedy"] > rating["random"]
    axis = data["sections"]["matrix"]["layouts"]["2p"]["lineages"]["axis"]
    assert axis.index("greedy") < axis.index("random")          # ordered by rating


def test_limits_bound_every_list_and_order_by_counters(tmp_path):
    run = write_run_dir(tmp_path, "wdl", train_records=300, decreasing_ts=True)
    limits = ReportLimits(series_points=20, entity_rows=4, matches=7, snapshot_matrix=5)
    data = build_report_data(run.root, limits=limits)
    steps = data["sections"]["training"]["agents"]["main"]["train_step"]
    assert len(steps) <= 20 and steps == sorted(steps) and steps[0] == 0 and steps[-1] == 2990
    assert len(data["sections"]["system"]["env_steps"]) <= 20
    (group,) = _groups(data, "2p")
    assert len(group["entities"]) == 4 and group["truncated"] == len(VERSIONS) + 2 - 4
    assert len(data["sections"]["matches"]["rows"]) == 7
    snap = data["sections"]["matrix"]["layouts"]["2p"]["snapshots"]
    assert len(snap["axis"]) <= 5 and snap["thinned_from"] > 5
    cells = data["sections"]["matrix"]["layouts"]["2p"]["lineages"]["cells"]
    assert all(len(row) == 3 for row in cells)


def test_few_games_are_marked_by_the_count_not_dropped(tmp_path):
    run = write_run_dir(tmp_path, "wdl", per_pair=4)
    snap = build_report_data(run.root)["sections"]["matrix"]["layouts"]["2p"]["snapshots"]
    counts = [c["n"] for row in snap["cells"] for c in row if c is not None]
    assert counts and min(counts) < MIN_CELL_GAMES


def test_metrics_tail_waits_for_a_complete_last_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"kind": "train", "train_step": 1}\n{"kind": "train", "train_st')
    tail = MetricsTail(path)
    assert [r["train_step"] for r in tail.poll()] == [1]
    with path.open("a") as fh:
        fh.write('ep": 2}\nnot json\n')
    assert [r["train_step"] for r in tail.poll()] == [1, 2] and tail.bad_lines == 1


def test_a_running_run_with_a_partial_line_is_read(tmp_path):
    run = write_run_dir(tmp_path, "wdl", status="running", partial_last_line=True)
    data = build_report_data(run.root)
    assert data["sections"]["summary"]["status"] == "running"
    assert data["sections"]["training"]["notes"] == []           # the partial line is not "malformed"


def test_live_ratings_come_from_ratings_json_while_running(tmp_path):
    run = write_run_dir(tmp_path, "wdl", status="running", live_bt=True)
    ratings = build_report_data(run.root)["sections"]["ratings"]
    assert ratings["source"] == "live" and ratings["uncertainty"] == "laplace"
    (group,) = ratings["layouts"]["2p"]["groups"]                  # the '@' extras are not groups
    rows = {r["entity"]: r for r in group["entities"]}
    assert rows["main@v200"]["rating"] == 123.0 and rows["main@v200"]["ci"] == [100.0, 146.0]
    assert rows["main@v200"]["p_best_group"] == 0.6 and group["anchor"] == "random" and group["label"] == "player"
    assert ratings["layouts"]["2p"]["bias"] == {"0": 12.0, "1": -12.0}
    assert ratings["layouts"]["2p"]["transitivity"]["flag"] is False


def test_a_dead_running_process_is_stopped_and_rated_from_the_journal(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    run = write_run_dir(tmp_path, "wdl", status="running", pid=proc.pid, live_bt=True)
    data = build_report_data(run.root)
    assert data["sections"]["summary"]["status"] == "stopped"
    assert data["sections"]["ratings"]["source"] == "journal"


def test_pulse_curves_and_unevaluated_snapshots(tmp_path):
    run = write_run_dir(tmp_path, "wdl", evaluation=True, pulse_versions=(160, 200), stored_versions=(160, 180, 200))
    anchors = build_report_data(run.root)["sections"]["anchors"]
    assert anchors["source"] == "pulse" and anchors["biased"] is False
    entry = anchors["layouts"]["2p"]["main"]
    assert [p[0] for p in entry["anchors"]["greedy"]] == [160, 200]
    assert entry["unevaluated"] == [180]


def test_sp3_run_dir_degrades_with_notes(tmp_path):
    root = copy_sp3_run(tmp_path)
    data = build_report_data(root)
    assert set(data["sections"]) == set(SECTIONS)
    assert data["sections"]["summary"]["status"] == "unknown"
    assert data["sections"]["ratings"]["source"] == "none" and data["sections"]["ratings"]["notes"]
    assert data["sections"]["matches"]["rows"] == [] and data["sections"]["matches"]["notes"]
    assert data["sections"]["training"]["agents"]                  # metrics.jsonl of SP3 still drawn
    json.dumps(data, allow_nan=False)


def test_section_filters_and_unknown_names(tmp_path):
    run = write_run_dir(tmp_path, "wdl", pulse_versions=(200,))
    greedy = section(run.root, "ratings", agent="greedy")
    assert {r["entity"] for r in greedy["layouts"]["2p"]["groups"][0]["entities"]} == {"greedy"}
    pulse = section(run.root, "matches", source="pulse", bogus="x")
    assert pulse["rows"] and {r["source"] for r in pulse["rows"]} == {"pulse"}
    assert pulse["filters"]["source"] == "pulse"
    with pytest.raises(KeyError):
        section(run.root, "nope")


def test_tournament_reports_are_listed(tmp_path):
    run = write_run_dir(tmp_path, "wdl")
    (run.root / "tournaments" / "t1").mkdir(parents=True)
    (run.root / "tournaments" / "t1" / "report.json").write_text(json.dumps({"name": "t1", "mode": "gauntlet"}))
    (run.root / "tournaments" / "t2").mkdir()
    out = build_report_data(run.root)["sections"]["tournaments"]
    assert [i["name"] for i in out["items"]] == ["t1"] and out["items"][0]["report"]["mode"] == "gauntlet"
    assert any("t2" in n for n in out["notes"])


def test_fits_are_cached_until_the_journal_changes(tmp_path, monkeypatch):
    from colosseum.arena import bt
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog, RunInfo
    from report_kit import INTERVAL, make_results

    run = write_run_dir(tmp_path, "wdl")
    calls = []
    real_fit = bt.fit
    monkeypatch.setattr(bt, "fit", lambda *a, **k: calls.append(1) or real_fit(*a, **k))
    builder = ReportBuilder(run.root)
    try:
        builder.section("ratings")
        builder.section("summary")
        assert len(calls) == 1
        log = MatchLog(run.root / MATCH_LOG_FILE, RunInfo(run_id=run.run_id, name=run.root.name, parent_run_id=None,
                                                          game="turns", config_hash="synthetic",
                                                          checkpoint_interval=INTERVAL))
        try:
            log.append(make_results("wdl", run_id=run.run_id + "-more", per_pair=2, seed=5))
        finally:
            log.close()
        builder.section("ratings")
        assert len(calls) == 2
    finally:
        builder.close()
    assert os.path.exists(run.root / MATCH_LOG_FILE)
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_report_data.py -q`

Expected: collection error `ModuleNotFoundError: No module named 'colosseum.arena.report'`.

- [ ] **Step 6: The data builders**

Create `src/colosseum/arena/report/__init__.py`:

```python
"""Run report and dashboard (SP4 spec block 5): ``data`` builds one JSON-ready dict per page section from a run
dir, ``render`` turns them into one self-contained HTML page, ``server`` serves the page and ``/api/<section>``
live. No module here touches process queues or the launcher (spec block 8: SP5's hub can host them)."""
```

Create `src/colosseum/arena/report/data.py`:

```python
"""Report data (SP4 spec block 5): one JSON-ready dict per page section, built from a run dir.

Sections (``SECTIONS``): summary, ratings, matrix, anchors, opponents, training, system, matches,
tournaments. The same builders serve ``colosseum report`` (``build_report_data``) and the dashboard
(``ReportBuilder.section`` behind ``/api/<section>``).

Rules:
- read-only: the match log is opened with ``MatchLog.open_readonly``; nothing in the run dir is written;
- order by counters (``train_step``, ``env_steps``, versions, journal row order), never by ``ts``
  (WSL2 wall clocks jump; ``ts`` is informational);
- bounded (``ReportLimits``): series thinned to 2 000 points (first and last kept), 2 000 entity rows
  per rating group, the latest 1 000 matches, at most 100 snapshots in the per-agent matrix;
- robust to a running run (a partial last line of ``metrics.jsonl`` is left for the next read) and to
  SP3 run dirs (no ``matches.db`` / ``run_state.json``): a section without its source carries a note;
- ratings: while the run is running (``run_state.json`` + live pid) the BT tables come from the
  ``bt`` section of ``ratings.json`` (live refit, Laplace intervals); otherwise they are fitted from the
  journal (sources ``train`` + ``pulse``) with ``ratings.bootstrap_samples`` bootstrap replicas and cached
  until the journal files change.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from colosseum.arena.entities import entity_version
from colosseum.arena.matchlog import RUN_SOURCES
from colosseum.core.run_state import effective_status, read_run_state

logger = logging.getLogger(__name__)

REPORT_FORMAT = 1
SECTIONS = ("summary", "ratings", "matrix", "anchors", "opponents", "training", "system", "matches",
            "tournaments")
MIN_CELL_GAMES = 20           # heat-map cells with fewer comparisons are drawn hatched ("few games")
PFSP_TOP = 20                 # rows of each owner's PFSP table in "opponents"
FILTERS = {"ratings": ("agent", "layout"), "matrix": ("agent", "layout"),
           "matches": ("agent", "layout", "source", "outcome")}
TRAIN_DEFAULT = ("total_loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                 "explained_variance", "rho_trunc_mass", "ess", "lr", "kickstart_loss", "kickstart_lambda")
SYSTEM_DEFAULT = ("env_steps_per_sec", "dropped_match_results", "dropped_reward_episodes")
_RECORD_META = ("ts", "kind", "agent", "train_step", "env_steps")
_Z95 = 1.959963984540054


@dataclass(frozen=True)
class ReportLimits:
    series_points: int = 2000
    entity_rows: int = 2000
    matches: int = 1000
    snapshot_matrix: int = 100


def thin(n: int, limit: int) -> list[int]:
    """Indices of at most ``limit`` evenly spaced items out of ``n``; the first and last are kept."""
    if n <= limit:
        return list(range(n))
    if limit <= 1:
        return [n - 1]
    return sorted({round(i * (n - 1) / (limit - 1)) for i in range(limit)})


def wilson(p: float, n: float) -> list[float]:
    """95% Wilson interval of a (fractional) score ``p`` over ``n`` comparisons; n <= 0 -> [0, 1]."""
    if n <= 0:
        return [0.0, 1.0]
    z2 = _Z95 * _Z95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = _Z95 * math.sqrt(max(0.0, p * (1 - p)) / n + z2 / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _flat(prefix: str, value: Any, out: dict[str, float]) -> None:
    """Numbers (bools as 0/1) of nested dicts as ``{"a.b": float}``."""
    if isinstance(value, Mapping):
        for k, v in value.items():
            _flat(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        out[prefix] = float(value)


def _as_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if hasattr(obj, "to_dict"):
        return dict(obj.to_dict())
    if is_dataclass(obj):
        return asdict(obj)
    return dict(obj)


def _file_sig(*paths: Path) -> tuple:
    """Change detector of files (size, mtime_ns); missing files count as (-1, -1). Not an ordering key."""
    sig = []
    for p in paths:
        try:
            s = p.stat()
            sig.append((s.st_size, s.st_mtime_ns))
        except OSError:
            sig.append((-1, -1))
    return tuple(sig)


class MetricsTail:
    """Incremental reader of ``metrics.jsonl``: only complete lines are parsed; a partial last line (a
    running run is writing it) waits for the next ``poll``. A malformed complete line is counted, not raised."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.offset = 0
        self.records: list[dict] = []
        self.bad_lines = 0

    def poll(self) -> list[dict]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return self.records
        if size < self.offset:                 # rewritten (another run with the same dir): start over
            self.offset, self.records, self.bad_lines = 0, [], 0
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return self.records
        for line in chunk[:end].split(b"\n"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                self.bad_lines += 1
                continue
            if isinstance(record, dict):
                self.records.append(record)
        self.offset += end + 1
        return self.records

    def of(self, kind: str) -> list[dict]:
        return [r for r in self.records if r.get("kind") == kind]


def _sorted_by(records: list[dict], key: str) -> list[dict]:
    """Stable sort by a counter (records without it keep file order at the end)."""
    return sorted(records, key=lambda r: (r.get(key) is None, r.get(key) or 0))


def _team_entity(team) -> str | None:
    entities = {e for e, _ in team.members}
    return entities.pop() if len(entities) == 1 else None


def _team_lineage(team) -> str | None:
    from colosseum.arena.entities import lineage

    names = {lineage(e) for e, _ in team.members}
    return names.pop() if len(names) == 1 else None


class ReportBuilder:
    """Builds report sections of one run dir, caching what is expensive (journal tables and fits) until the
    journal files change. Not thread-safe: the dashboard serializes calls with a lock."""

    def __init__(self, run_dir: str | Path, *, limits: ReportLimits | None = None,
                 include_tournaments: bool = True, seed: int = 0, table_min_age_sec: float = 0.0) -> None:
        self.run_dir = Path(run_dir)
        self.table_min_age_sec = float(table_min_age_sec)
        self.limits = limits or ReportLimits()
        self.include_tournaments = include_tournaments
        self.seed = seed
        self._metrics = MetricsTail(self.run_dir / "metrics.jsonl")
        self._config_cache: tuple[tuple, Any] | None = None
        self._reader = None
        self._reader_failed: str | None = None
        self._cache: dict[tuple, Any] = {}

    # ----- sources -----
    @property
    def journal_path(self) -> Path:
        from colosseum.arena.matchlog import MATCH_LOG_FILE

        return self.run_dir / MATCH_LOG_FILE

    def journal_sig(self) -> tuple:
        p = self.journal_path
        return _file_sig(p, p.with_name(p.name + "-wal"))

    def config(self):
        """The run's resolved config (None when missing or unreadable)."""
        from colosseum.core.run_dir import RESOLVED_CONFIG_FILE

        path = self.run_dir / RESOLVED_CONFIG_FILE
        sig = _file_sig(path)
        if self._config_cache is None or self._config_cache[0] != sig:
            cfg = None
            if path.exists():
                try:
                    from colosseum.core.config import load_config

                    cfg = load_config(path)
                except Exception as e:  # noqa: BLE001 - a report never fails on a config it cannot read
                    logger.warning(f"report: cannot load {path}: {e}")
            self._config_cache = (sig, cfg)
        return self._config_cache[1]

    def ratings_json(self) -> dict | None:
        path = self.run_dir / "ratings.json"
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def reader(self):
        """The read-only journal, or None (no ``matches.db``: an SP3 run, ``match_log.enabled: false``)."""
        if self._reader is not None:
            return self._reader
        if not self.journal_path.exists():
            return None
        try:
            from colosseum.arena.matchlog import MatchLog

            self._reader = MatchLog.open_readonly(self.journal_path)
            self._reader_failed = None
        except Exception as e:  # noqa: BLE001 - a broken journal degrades the sections, not the report
            self._reader_failed = f"the match log cannot be opened: {e}"
            return None
        return self._reader

    def _journal_note(self) -> str:
        if self._reader_failed:
            return self._reader_failed
        return "No match log in this run (an SP3 run dir, or match_log.enabled: false)."

    def table(self, sources: Sequence[str] = RUN_SOURCES):
        """The journal's comparison table for ``sources``; reloaded when the journal files changed, but not
        more often than every ``table_min_age_sec`` (the dashboard of a running run: the journal changes
        every second, a reload reads it whole)."""
        reader = self.reader()
        if reader is None:
            return None
        key = ("table", tuple(sources))
        sig, now = self.journal_sig(), time.monotonic()
        hit = self._cache.get(key)
        if hit is not None and (hit[0] == sig or now - hit[1] < self.table_min_age_sec):
            return hit[2]
        table = reader.comparison_table(sources=tuple(sources))
        self._cache[key] = (sig, now, table)
        return table

    def _drop_stale(self, kind: str, *rest) -> None:
        for k in [k for k in self._cache if k[0] == kind and k[1:1 + len(rest)] == rest]:
            del self._cache[k]

    def records(self, kind: str) -> list[dict]:
        self._metrics.poll()
        return self._metrics.of(kind)

    def status(self) -> tuple[str, str, Any]:
        state = read_run_state(self.run_dir)
        status, detail = effective_status(state)
        return status, detail, state

    def layout_kinds(self) -> dict[str, str]:
        """layout -> outcome kind, from ratings.json, else from the journal's first match of the layout."""
        kinds: dict[str, str] = {}
        for layout, table in ((self.ratings_json() or {}).get("layouts") or {}).items():
            kinds[layout] = str(table.get("outcome_kind", "wdl"))
        table = self.table()
        reader = self.reader()
        if table is not None:
            for layout in table.layouts():
                if layout not in kinds:
                    rows = reader.matches(layout=layout, limit=1)
                    kinds[layout] = str(rows[0].to_dict()["outcome_kind"]) if rows else "wdl"
        return dict(sorted(kinds.items()))

    def agent_kinds(self) -> dict[str, str]:
        cfg = self.config()
        return {aid: str(cfg.agent_kind(aid)) for aid in cfg.agent_ids()} if cfg is not None else {}

    def _kind_of(self, entity: str) -> str:
        from colosseum.arena.entities import lineage

        return self.agent_kinds().get(lineage(entity), "other")

    # ----- public -----
    def build(self) -> dict:
        sections = {name: self.section(name) for name in SECTIONS
                    if name != "tournaments" or self.include_tournaments}
        cfg = self.config()
        title = cfg.run.name if cfg is not None and cfg.run.name else self.run_dir.name
        return {"format": REPORT_FORMAT, "title": str(title), "sections": sections}

    def section(self, name: str, **filters: Any) -> dict:
        if name not in SECTIONS:
            raise KeyError(name)
        allowed = FILTERS.get(name, ())
        clean = {k: v for k, v in filters.items() if k in allowed and v not in (None, "")}
        return getattr(self, f"_{name}")(**clean)

    def close(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            finally:
                self._reader = None

    # ----- summary -----
    def _summary(self) -> dict:
        from colosseum.metrics.hub import wr_vs_past_over_layouts

        notes: list[str] = []
        status, detail, state = self.status()
        if state is None:
            notes.append("No run_state.json (an SP3 run dir or a distributed role): the state is unknown.")
        cfg = self.config()
        system = _sorted_by(self.records("system"), "env_steps")
        last = system[-1] if system else {}
        env_steps = last.get("env_steps")
        total = cfg.training.total_timesteps if cfg is not None else None
        rj = self.ratings_json() or {}
        layouts = rj.get("layouts") or {}
        ratings = self._ratings()
        agents = []
        for aid, kind in self.agent_kinds().items():
            elo = None
            for table in layouts.values():
                if sum((table.get("games") or {}).get(aid, {}).values()):
                    elo = (table.get("elo") or {}).get(aid)
                    break
            agents.append({"agent": aid, "kind": kind, "bt": self._newest_bt(ratings, aid), "elo": elo,
                           "wr_vs_past": wr_vs_past_over_layouts(layouts, aid) if layouts else None})
        parents: list[str] = []
        reader = self.reader()
        if reader is not None:
            own = self.run_dir.name
            parents = [r.name for r in reader.runs() if r.name != own]
        return {"run": self.run_dir.name, "path": str(self.run_dir), "status": status, "status_detail": detail,
                "pid": getattr(state, "pid", None), "exit_code": getattr(state, "exit_code", None),
                "env_steps": env_steps, "total_timesteps": total,
                "budget_frac": (env_steps / total) if env_steps is not None and total else None,
                "env_steps_per_sec": _finite(last.get("env_steps_per_sec")),
                "agents": agents, "parents": parents, "notes": notes}

    @staticmethod
    def _newest_bt(ratings: dict, agent: str) -> dict | None:
        for layout, lay in ratings["layouts"].items():
            for group in lay.get("groups", []):
                rows = [r for r in group["entities"] if r["lineage"] == agent]
                if rows:
                    row = max(rows, key=lambda r: (r["version"] is not None, r["version"] or 0))
                    return {"layout": layout, "entity": row["entity"], "rating": row["rating"], "ci": row["ci"]}
        return None

    # ----- ratings -----
    def _ratings(self, agent: str | None = None, layout: str | None = None) -> dict:
        status, _, _ = self.status()
        rj = self.ratings_json() or {}
        live = status == "running" and isinstance(rj.get("bt"), Mapping)
        notes: list[str] = []
        out_layouts: dict[str, dict] = {}
        table = self.table()
        if table is None and not live:
            notes.append(self._journal_note())
        for name, kind in self.layout_kinds().items():
            if layout and name != layout:
                continue
            if kind == "score":
                entry = self._score_layout(name)
            elif live:
                entry = self._live_layout(name, rj["bt"].get(name) or {})
            else:
                entry = self._fitted_layout(name, notes)
            if entry is None:
                continue
            entry["outcome_kind"] = kind
            if agent:
                entry = self._filter_agent(entry, agent)
            out_layouts[name] = entry
        if live:
            notes.append("Live ratings (ratings.json, refit every ratings.refit_interval_sec); Laplace intervals.")
        return {"source": "live" if live else ("journal" if table is not None else "none"),
                "uncertainty": "laplace" if live else ("bootstrap" if self._bootstrap_samples() else "laplace"),
                "env_steps": rj.get("env_steps"), "layouts": out_layouts, "notes": notes}

    @staticmethod
    def _filter_agent(entry: dict, agent: str) -> dict:
        entry = dict(entry)
        if "groups" in entry:
            entry["groups"] = [dict(g, entities=[e for e in g["entities"] if e["lineage"] == agent],
                                    series=[s for s in g["series"] if s["lineage"] == agent]) for g in entry["groups"]]
        if "entities" in entry:
            entry["entities"] = [e for e in entry["entities"] if e["lineage"] == agent]
        return entry

    def _bootstrap_samples(self) -> int:
        cfg = self.config()
        return int(cfg.ratings.bootstrap_samples) if cfg is not None else 200

    def _group_payload(self, label: str, component: int, anchor: str, rows: list[dict]) -> dict:
        rows = sorted(rows, key=lambda r: -(r["rating"] if r["rating"] is not None else -1e18))
        keep = rows[: self.limits.entity_rows]
        by_lineage: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["version"] is not None:
                by_lineage[r["lineage"]].append(r)
        series = []
        for name in sorted(by_lineage):
            pts = sorted(by_lineage[name], key=lambda r: r["version"])
            idx = thin(len(pts), self.limits.series_points)
            series.append({"lineage": name, "points": [[pts[i]["version"], pts[i]["rating"], pts[i]["ci"][0],
                                                        pts[i]["ci"][1], pts[i]["games"]] for i in idx]})
        anchors = [{"entity": r["entity"], "rating": r["rating"], "ci": r["ci"]} for r in rows if r["version"] is None]
        return {"label": label, "component": component, "anchor": anchor, "entities": keep,
                "truncated": len(rows) - len(keep), "series": series, "anchors": anchors}

    def _row(self, entity: str, rating: float, ci: Sequence[float], games: int, *, p_lin=None, p_grp=None,
             lcb=None) -> dict:
        from colosseum.arena.entities import lineage

        return {"entity": entity, "lineage": lineage(entity), "version": entity_version(entity),
                "kind": self._kind_of(entity), "rating": _finite(rating), "ci": [_finite(ci[0]), _finite(ci[1])],
                "games": int(games or 0), "p_best_lineage": _finite(p_lin), "p_best_group": _finite(p_grp),
                "lcb": _finite(lcb)}

    def _live_layout(self, layout: str, payload: Mapping) -> dict:
        """ratings.json ``bt[layout]`` (T2.2's ``bt_section``, contract amendment A14): ``{group label: [rows]}``,
        rows ``{"entity", "rating", "ci_low", "ci_high", "games", "lcb", "p_best" (within the cell),
        "p_best_lineage", "component", ...}``, plus extras under keys starting with ``@`` (``@bias``, ``@anchors``
        keyed ``"<label>/<component>"``, ``@transitivity``). One payload group per (label, component)."""
        anchors = payload.get("@anchors") or {}
        groups = []
        for label, entries in payload.items():
            if str(label).startswith("@") or not isinstance(entries, list):
                continue
            by_component: dict[int, list[dict]] = defaultdict(list)
            for d in entries:
                ci = d.get("ci") or [d.get("ci_low"), d.get("ci_high")]
                by_component[int(d.get("component", 0))].append(self._row(
                    str(d["entity"]), d.get("rating"), ci, d.get("games", 0), p_lin=d.get("p_best_lineage"),
                    p_grp=d.get("p_best_group", d.get("p_best")), lcb=d.get("lcb")))
            for component, rows in sorted(by_component.items()):
                anchor = (anchors.get(f"{label}/{component}") or {}).get("anchor", "")
                groups.append(self._group_payload(str(label), component, str(anchor), rows))
        trans = payload.get("@transitivity")
        return {"kind": "bt", "groups": groups, "bias": dict(payload.get("@bias") or {}),
                "transitivity": dict(trans) if isinstance(trans, Mapping) else None}

    def _fit(self, layout: str):
        """(fit, uncertainty, transitivity dict | None), cached until the journal changes."""
        key = ("fit", layout, self.journal_sig())
        if key in self._cache:
            return self._cache[key]
        self._drop_stale("fit", layout)
        from colosseum.arena import bt
        from colosseum.arena.live import rating_inputs

        cfg = self.config()
        table = self.table()
        options, fixed, random_bot = rating_inputs(cfg)
        fit_ = bt.fit(table, layout, sources=RUN_SOURCES, options=options, fixed_agents=fixed, random_bot=random_bot)
        rng = np.random.default_rng(self.seed)
        samples = self._bootstrap_samples()
        if samples > 0:
            data = bt.bootstrap_data(self.reader(), layout, sources=RUN_SOURCES)
            unc = bt.bootstrap(fit_, data, samples=samples, rng=rng)
        else:
            unc = bt.laplace(fit_, rng=rng)
        try:
            trans = bt.transitivity_test(table, fit_, sources=RUN_SOURCES).to_dict()
        except Exception as e:  # noqa: BLE001 - the diagnostic never hides the ratings
            logger.warning(f"report: transitivity test failed for {layout}: {e}")
            trans = None
        self._cache[key] = (fit_, unc, trans)
        return self._cache[key]

    def _fitted_layout(self, layout: str, notes: list[str]) -> dict | None:
        table = self.table()
        if table is None or layout not in table.layouts():
            return None
        fit_, unc, trans = self._fit(layout)
        groups = []
        for g in fit_.groups:
            rows = []
            for e in g.entities:
                r = fit_.rating[e]
                ci = unc.ci95.get(e) or (r - _Z95 * fit_.se[e], r + _Z95 * fit_.se[e])
                rows.append(self._row(e, r, ci, fit_.games.get(e, 0), p_lin=unc.p_best_lineage.get(e),
                                      p_grp=unc.p_best_group.get(e), lcb=unc.lcb.get(e)))
            groups.append(self._group_payload(g.group, g.component, str(g.anchor), rows))
        if not fit_.converged:
            notes.append(f"{layout}: the BT fit did not converge in {fit_.iterations} iterations.")
        return {"kind": "bt", "groups": groups, "bias": {str(k): v for k, v in fit_.bias.items()},
                "transitivity": trans}

    def _score_layout(self, layout: str) -> dict | None:
        """One-team layouts: mean score per entity and per team composition (``bt.score_summary``)."""
        table = self.table()
        if table is None:
            return None
        from colosseum.arena import bt
        from colosseum.arena.entities import lineage

        summary = bt.score_summary(table, layout, sources=RUN_SOURCES).to_dict()
        entities = [{"entity": e, "lineage": lineage(e), "version": entity_version(e), "kind": self._kind_of(e),
                     "n": int(d["n"]), "mean": _finite(d["mean"]), "ci": [_finite(d["ci_low"]), _finite(d["ci_high"])]}
                    for e, d in summary["entities"].items()]
        entities.sort(key=lambda r: -(r["mean"] if r["mean"] is not None else -1e18))
        comps = [{"composition": c, "n": int(d["n"]), "mean": _finite(d["mean"]),
                  "ci": [_finite(d["ci_low"]), _finite(d["ci_high"])], "homogeneous": bool(d.get("homogeneous"))}
                 for c, d in summary["compositions"].items()]
        return {"kind": "score", "entities": entities[: self.limits.entity_rows], "compositions": comps}

    # ----- matrix -----
    def _matrix(self, agent: str | None = None, layout: str | None = None) -> dict:
        notes: list[str] = []
        table = self.table()
        if table is None:
            return {"min_games": MIN_CELL_GAMES, "layouts": {}, "notes": [self._journal_note()]}
        ratings = self._ratings()
        out: dict[str, dict] = {}
        for name, kind in self.layout_kinds().items():
            if (layout and name != layout) or name not in table.layouts():
                continue
            if kind == "score":
                out[name] = {"kind": "score", "cross_play": ratings["layouts"].get(name, {}).get("compositions", [])}
                continue
            order = self._lineage_order(ratings["layouts"].get(name, {}))
            skipped = 0
            acc: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
            entity_acc: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
            for a, b, stat in table.pairs(name, RUN_SOURCES):
                la, lb = _team_lineage(a), _team_lineage(b)
                if la is None or lb is None:
                    skipped += 1
                    continue
                if la != lb:
                    _add(acc, la, lb, stat)
                ea, eb = _team_entity(a), _team_entity(b)
                if ea is not None and eb is not None and ea != eb:
                    _add(entity_acc, ea, eb, stat)
            axis = sorted({x for pair in acc for x in pair}, key=lambda x: order.get(x, (1, 0.0, x)))
            entry = {"kind": "bt", "lineages": {"axis": axis, "cells": _cells(axis, acc)}}
            trainable = [x for x in axis if self.agent_kinds().get(x) == "trainable"]
            target = agent if agent else (trainable[0] if trainable else None)
            entry["snapshots"] = self._snapshot_matrix(target, entity_acc, axis, order) if target else None
            if skipped:
                notes.append(f"{name}: {skipped} team pairs with members of several agents are not in the matrix.")
            out[name] = entry
        return {"min_games": MIN_CELL_GAMES, "layouts": out, "notes": notes}

    @staticmethod
    def _lineage_order(layout_ratings: Mapping) -> dict[str, tuple]:
        """Sort key per lineage: rating of its newest entity (descending), unrated last."""
        best: dict[str, tuple[int, float]] = {}
        for group in layout_ratings.get("groups", []):
            for r in group["entities"]:
                v = r["version"] if r["version"] is not None else math.inf
                if r["rating"] is not None and (r["lineage"] not in best or v > best[r["lineage"]][0]):
                    best[r["lineage"]] = (v, r["rating"])
        return {name: (0, -rating, name) for name, (_, rating) in best.items()}

    def _snapshot_matrix(self, agent: str, entity_acc, lineage_axis: list[str], order) -> dict:
        from colosseum.arena.entities import lineage

        versions = sorted({e for pair in entity_acc for e in pair if lineage(e) == agent},
                          key=lambda e: entity_version(e) or 0)
        others = [x for x in lineage_axis if x != agent]
        room = max(1, self.limits.snapshot_matrix - len(others))
        chosen = [versions[i] for i in thin(len(versions), room)]
        chosen_set = set(chosen)
        acc: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
        for (ea, eb), (n, sw, swy) in entity_acc.items():
            la, lb = lineage(ea), lineage(eb)
            xa = ea if la == agent else la
            xb = eb if lb == agent else lb
            if (la == agent and ea not in chosen_set) or (lb == agent and eb not in chosen_set) or xa == xb:
                continue
            cell = acc[(xa, xb)]
            cell[0] += n
            cell[1] += sw
            cell[2] += swy
        axis = chosen + others
        return {"agent": agent, "axis": axis, "cells": _cells(axis, acc), "thinned_from": len(versions) + len(others)}

    # ----- anchors -----
    def _anchors(self) -> dict:
        from colosseum.arena.entities import lineage

        cfg = self.config()
        notes: list[str] = []
        pulse = bool(cfg is not None and cfg.evaluation.enabled)
        source = "pulse" if pulse else "train"
        table = self.table((source,))
        if table is None:
            return {"source": "none", "biased": False, "layouts": {}, "notes": [self._journal_note()]}
        fixed = set(cfg.fixed_agent_ids()) if cfg is not None else set()
        field = set(cfg.evaluation.field or fixed) if pulse else fixed
        kinds = self.layout_kinds()
        layouts: dict[str, dict] = {}
        evaluated: dict[str, set[int]] = defaultdict(set)
        for name in table.layouts():
            if kinds.get(name) == "score":
                continue
            acc: dict[tuple[str, int, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
            for a, b, stat in table.pairs(name, (source,)):
                for x, y, flip in ((a, b, False), (b, a, True)):
                    ex, ly = _team_entity(x), _team_lineage(y)
                    vx = entity_version(ex) if ex is not None else None
                    if vx is None or ly not in field:
                        continue
                    cell = acc[(lineage(ex), vx, ly)]
                    cell[0] += stat.n
                    cell[1] += stat.sw
                    cell[2] += (stat.sw - stat.swy) if flip else stat.swy
            per_agent: dict[str, dict] = {}
            for (agent, version, anchor), (n, sw, swy) in sorted(acc.items()):
                if sw <= 0:
                    continue
                p = swy / sw
                per_agent.setdefault(agent, {"anchors": defaultdict(list), "unevaluated": []})
                per_agent[agent]["anchors"][anchor].append([version, p, *wilson(p, n), n])
                evaluated[agent].add(version)
            for d in per_agent.values():
                d["anchors"] = {k: [v[i] for i in thin(len(v), self.limits.series_points)]
                                for k, v in sorted(d["anchors"].items())}
            if per_agent:
                layouts[name] = per_agent
        if pulse:
            for layout_entry in layouts.values():
                for agent, d in layout_entry.items():
                    d["unevaluated"] = sorted(set(self._stored_versions(agent)) - evaluated[agent])
            if not layouts:
                notes.append("Pulse evaluation is enabled but has no results yet.")
        elif not fixed:
            notes.append("The config has no scripted or frozen agents (anchors).")
        return {"source": source, "biased": not pulse, "layouts": layouts, "notes": notes}

    def _stored_versions(self, agent: str) -> list[int]:
        root = self.run_dir / "checkpoints" / agent
        if not root.is_dir():
            return []
        out = []
        for d in root.iterdir():
            m = re.fullmatch(r"ckpt_v(\d+)", d.name)
            if m and d.is_dir():
                out.append(int(m.group(1)))
        return sorted(out)

    # ----- opponents -----
    def _opponents(self) -> dict:
        agents: dict[str, dict[str, dict]] = defaultdict(dict)
        for r in _sorted_by(self.records("episodes"), "env_steps"):
            for layout, cell in (r.get("opponents") or {}).items():
                d = agents[str(r.get("agent"))].setdefault(layout, {"env_steps": [], "categories": defaultdict(list),
                                                                     "anchors": defaultdict(list)})
                i = len(d["env_steps"])
                d["env_steps"].append(r.get("env_steps"))
                for group in ("categories", "anchors"):
                    values = cell.get(group) or {}
                    for k in set(values) | set(d[group]):
                        series = d[group][k]
                        series.extend([0.0] * (i - len(series)))
                        series.append(_finite(values.get(k)) or 0.0)
        for layouts in agents.values():
            for d in layouts.values():
                idx = thin(len(d["env_steps"]), self.limits.series_points)
                d["env_steps"] = [d["env_steps"][i] for i in idx]
                for group in ("categories", "anchors"):
                    d[group] = {k: [v[i] for i in idx] for k, v in sorted(d[group].items())}
        pfsp: dict[str, dict] = {}
        for layout, table in ((self.ratings_json() or {}).get("layouts") or {}).items():
            for owner, players in (table.get("pfsp") or {}).items():
                rows = [{"player": p, "score": _finite(v.get("score")), "games": _finite(v.get("games"))}
                        for p, v in players.items()]
                rows.sort(key=lambda r: -(r["games"] or 0))
                pfsp.setdefault(layout, {})[owner] = rows[:PFSP_TOP]
        notes = [] if agents else ["No 'episodes' records with opponent shares yet."]
        return {"agents": dict(agents), "pfsp": pfsp, "notes": notes}

    # ----- training -----
    def _training(self) -> dict:
        per_agent: dict[str, list[dict]] = defaultdict(list)
        for r in self.records("train"):
            per_agent[str(r.get("agent"))].append(r)
        agents = {}
        for agent, recs in sorted(per_agent.items()):
            recs = _sorted_by(recs, "train_step")
            idx = thin(len(recs), self.limits.series_points)
            recs = [recs[i] for i in idx]
            names = sorted({k for r in recs for k, v in r.items() if k not in _RECORD_META and _finite(v) is not None
                            and not isinstance(v, bool)})
            agents[agent] = {"train_step": [r.get("train_step") for r in recs],
                             "metrics": {k: [_finite(r.get(k)) for r in recs] for k in names}}
        notes = []
        if self._metrics.bad_lines:
            notes.append(f"{self._metrics.bad_lines} malformed lines of metrics.jsonl were skipped.")
        return {"agents": agents, "default": list(TRAIN_DEFAULT), "notes": notes}

    # ----- system -----
    def _system(self) -> dict:
        recs = _sorted_by(self.records("system"), "env_steps")
        idx = thin(len(recs), self.limits.series_points)
        flat = []
        for i in idx:
            row: dict[str, float] = {}
            _flat("", {k: v for k, v in recs[i].items() if k not in _RECORD_META}, row)
            flat.append(row)
        names = sorted({k for row in flat for k in row})
        latest: dict[str, float] = {}
        if recs:
            _flat("", {k: v for k, v in recs[-1].items() if k not in _RECORD_META}, latest)
        try:
            free_mb = shutil.disk_usage(self.run_dir).free / 2**20
        except OSError:
            free_mb = None
        journal = None
        if self.journal_path.exists():
            wal = self.journal_path.with_name(self.journal_path.name + "-wal")
            journal = {"file_mb": self.journal_path.stat().st_size / 2**20,
                       "wal_mb": wal.stat().st_size / 2**20 if wal.exists() else 0.0}
            reader = self.reader()
            if reader is not None:
                sizes = _as_dict(reader.sizes())
                for k in ("counts_rows", "detail_bytes", "counts_bytes"):
                    if k in sizes:
                        journal[k] = sizes[k]
            flags = [k for k in latest if k.endswith("counts_only") or k.endswith("degraded")]
            journal["counts_only"] = any(latest[k] for k in flags)
        return {"env_steps": [recs[i].get("env_steps") for i in idx], "series": {k: [r.get(k) for r in flat]
                                                                                   for k in names},
                "default": list(SYSTEM_DEFAULT), "latest": latest, "free_disk_mb": free_mb, "journal": journal,
                "notes": [] if recs else ["No 'system' records yet."]}

    # ----- matches -----
    def _matches(self, agent: str | None = None, layout: str | None = None, source: str | None = None,
                 outcome: str | None = None) -> dict:
        filters = {"agent": agent, "layout": layout, "source": source, "outcome": outcome}
        reader = self.reader()
        if reader is None:
            return {"rows": [], "limit": self.limits.matches, "filters": filters, "notes": [self._journal_note()]}
        notes = []
        if outcome and not agent:
            notes.append("The outcome filter needs an agent; ignored.")
            outcome = None
        rows = reader.matches(agent=agent, layout=layout, source=source, outcome=outcome, limit=self.limits.matches)
        return {"rows": [r.to_dict() for r in rows], "limit": self.limits.matches, "filters": filters, "notes": notes}

    # ----- tournaments -----
    def _tournaments(self) -> dict:
        items, notes = [], []
        root = self.run_dir / "tournaments"
        if self.run_dir.joinpath("tournament.yaml").exists() and self.run_dir.joinpath("report.json").exists():
            dirs = [self.run_dir]                      # a tournament dir itself (colosseum report <dir>)
        else:
            dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
        for d in dirs:
            path = d / "report.json"
            if not path.exists():
                notes.append(f"{d.name}: no report.json yet (running or interrupted).")
                continue
            try:
                from colosseum.arena.tournament.modes import report_view  # Part 2's schema -> per layout / group

                items.append({"name": d.name, "path": str(d), "report": report_view(json.loads(path.read_text()))})
            except (OSError, ValueError) as e:
                notes.append(f"{d.name}: report.json cannot be read ({e}).")
        return {"items": items, "notes": notes}


def _add(acc, a: str, b: str, stat) -> None:
    cell = acc[(a, b)]
    cell[0] += stat.n
    cell[1] += stat.sw
    cell[2] += stat.swy
    rev = acc[(b, a)]
    rev[0] += stat.n
    rev[1] += stat.sw
    rev[2] += stat.sw - stat.swy


def _cells(axis: Sequence[str], acc: Mapping[tuple[str, str], Sequence[float]]) -> list[list[dict | None]]:
    out = []
    for a in axis:
        row = []
        for b in axis:
            n, sw, swy = acc.get((a, b), (0, 0.0, 0.0))
            if sw <= 0:
                row.append(None)
            else:
                p = swy / sw
                row.append({"score": p, "ci": wilson(p, n), "n": int(n)})
        out.append(row)
    return out


def build_report_data(run_dir: str | Path, *, include_tournaments: bool = True,
                      limits: ReportLimits | None = None) -> dict:
    """Every section of the run's report (``ReportBuilder.build``)."""
    builder = ReportBuilder(run_dir, limits=limits, include_tournaments=include_tournaments)
    try:
        return builder.build()
    finally:
        builder.close()


def section(run_dir: str | Path, name: str, **filters: Any) -> dict:
    """One section (``/api/<name>`` of the dashboard); unknown names raise KeyError."""
    builder = ReportBuilder(run_dir)
    try:
        return builder.section(name, **filters)
    finally:
        builder.close()
```

Notes for the implementer:
- `ReportBuilder` imports `colosseum.arena.*` lazily (inside methods), so `import colosseum.arena.report.data` stays cheap (the dashboard imports it at startup) and an SP3 run dir without a journal never touches SQLite.
- `_fit` uses `ratings.bootstrap_samples` replicas (default 200, spec block 2: «bootstrap — для отчёта»); `report_kit.write_run_dir` sets 20 so the fast tests stay fast.
- `layout_kinds` takes outcome kinds from `ratings.json` (SP2 tables carry `outcome_kind`), else from the journal's first match of the layout.
- If R-P1-3 (`bt.bootstrap_data`) is not available, use `bt.laplace` in `_fit` and set `"uncertainty": "laplace"` (and say so in the task report; the controller decides whether Part 1 adds it).

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_report_data.py tests/unit/test_sp4_run_state.py -q`

Expected: all pass. If `test_strength_order_survives_into_ratings_and_matrix` fails on `rating["random"] == 0.0`, check T2.1's default anchor rule (first `RandomBot` of the config) before touching this test.

- [ ] **Step 7: Full fast suite and ruff**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `env -u PYTHONPATH .venv/bin/ruff check .`

Expected: all pass, no warnings; `All checks passed!`.

- [ ] **Step 8: Commit and push**

```bash
git add src/colosseum/core/run_state.py src/colosseum/core/run_dir.py src/colosseum/launcher.py \
  src/colosseum/arena/report/__init__.py src/colosseum/arena/report/data.py tests/report_kit.py pyproject.toml \
  tests/unit/test_sp4_run_state.py tests/unit/test_sp4_report_data.py
git commit -m "feat: run_state.json and report data builders for every report section"
git push
```

---

### Task T7.2: HTML renderer, assets and `colosseum report`

Spec block 5 («Команды»: `colosseum report runs/x [-o FILE]`, self-contained HTML, data embedded as JSON, own CSS / JS, SVG charts without libraries or CDN, opens offline; «Разделы страницы»; phone layout; light and dark themes; tournament reports in the page), criterion 3.7 (the report builds for demo-game runs of every outcome kind; all sections present, tested by data and markup), global constraint «Report / dashboard» (assets in package data, page text in English).

The page: a sticky header with the run name and a section menu; nine `<section id="sec-<name>">` containers that `page.js` fills from the embedded data; tables sort by a header click and filter by a text box, and scroll inside their own box (horizontal and, above 480 px, vertical); charts are SVG strings inserted into HTML elements (no namespace URL, no library): a line chart with a CI band, dashed anchor lines, crosshair tooltip and legend; a heat map (diverging blue / red around a neutral gray, hatched cells for fewer than `min_games`, tap for score / CI / games); a stacked area of opponent shares. Colors follow the entity, never the rank (fixed slot per name; fixed slot per opponent category). Light and dark via `prefers-color-scheme` (and `data-theme` on `<html>` if a viewer sets it). One column on phones (`max-width: 640px`); charts size their `viewBox` to the window so text stays readable. A section that fails to draw shows its error and its raw data instead of breaking the page.

**Files:**
- Create: `src/colosseum/arena/report/render.py`
- Create: `src/colosseum/arena/report/assets/page.html`, `page.css`, `page.js`
- Modify: `src/colosseum/cli.py` (`report` command)
- Modify: `src/colosseum/arena/tournament/run.py` (Part 2's T5.4: write `report.html` next to `report.json`, R-P2-2)
- Test: `tests/unit/test_sp4_report_render.py`, `tests/integration/test_sp4_report_cli.py`

**Package data.** `[tool.hatch.build.targets.wheel] packages = ["src/colosseum"]` (`pyproject.toml`) ships every file under the package directory, not only `.py` files: checked while writing this plan by building a wheel of a copy of the tree with `src/colosseum/arena/report/assets/page.js` added (`uv build --wheel --offline`), which contained `colosseum/arena/report/assets/page.js`. No `pyproject.toml` change is needed; the editable install reads the assets from the source tree. Step 6 repeats the check on the real tree.

**Interfaces:**
- Consumes: `build_report_data`, `ReportBuilder`, `REPORT_FORMAT`, `SECTIONS` (T7.1); `write_text_atomic`; `ConfigError`; Part 2's tournament dir (`tournament.yaml`, `report.json`; R-P2-1, R-P2-2).
- Produces: `colosseum.arena.report.render.render_html(data, *, live=False, refresh_sec=10) -> str`, `write_report(run_dir, out=None) -> Path` (contract), `*REPORT_FILE`, `*asset(name)`, `*embed_json(data)`, `*is_run_dir(path)`, `*is_tournament_dir(path)`, `*tournament_page_data(dir)`, `*write_tournament_html(dir) -> Path`; the CLI `colosseum report RUN [-o FILE]` (exit codes 0 / 1 / 2 / 130).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp4_report_render.py`:

```python
"""HTML renderer (SP4 spec block 5, T7.2): one self-contained file, data embedded as JSON, every section's
container present, no external resources, script-safe escaping; the page draws every section (headless
Chrome when available) and ``page.js`` is valid JavaScript (node when available)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from colosseum.arena.report.data import SECTIONS
from colosseum.arena.report.render import asset, embed_json, render_html, write_report, write_tournament_html
from colosseum.core.errors import ConfigError
from report_kit import sample_report_data, write_run_dir

ASSETS = ("page.html", "page.css", "page.js")
URL = re.compile(r"(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CHROME = next((shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser") if shutil.which(n)), None)


def _data_of(page: str) -> dict:
    m = re.search(r'<script type="application/json" id="colosseum-data">(.*?)</script>', page, re.S)
    return json.loads(m.group(1))


def test_page_embeds_the_data_and_has_every_section():
    data = sample_report_data()
    page = render_html(data)
    assert page.startswith("<!doctype html>") and "{{" not in page
    for name in SECTIONS:
        assert f'id="sec-{name}"' in page
    assert _data_of(page) == json.loads(json.dumps(data))
    config = json.loads(re.search(r'id="colosseum-config">(.*?)</script>', page, re.S).group(1))
    assert config == {"live": False, "refresh_sec": 10}
    assert json.loads(re.search(r'id="colosseum-config">(.*?)</script>', render_html(data, live=True, refresh_sec=3),
                                re.S).group(1)) == {"live": True, "refresh_sec": 3}


def test_no_external_resources_in_assets_or_page():
    for name in ASSETS:
        text = asset(name)
        assert not URL.search(text), name
        assert "@import" not in text and "<link" not in text and "src=" not in text, name
    page = render_html(sample_report_data())
    assert not URL.search(page) and "<script src" not in page and "<link" not in page


def test_strings_in_the_data_cannot_break_out_of_the_script_element():
    data = sample_report_data()
    evil = "</script><script>alert(1)</script>&\u2028"
    data["title"] = evil
    data["sections"]["summary"]["agents"][0]["agent"] = evil
    page = render_html(data)
    assert "<script>alert(1)" not in page
    assert _data_of(page)["sections"]["summary"]["agents"][0]["agent"] == evil
    assert "\\u003c/script" in embed_json({"x": "</script>"})


def test_non_finite_numbers_become_null():
    assert json.loads(embed_json({"a": float("nan"), "b": [float("inf"), 1.5], "c": (1, 2)})) == \
        {"a": None, "b": [None, 1.5], "c": [1, 2]}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_js_is_valid_javascript(tmp_path):
    path = tmp_path / "page.js"
    path.write_text(asset("page.js"))
    assert subprocess.run(["node", "--check", str(path)], capture_output=True, text=True).returncode == 0


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome")
def test_every_section_is_drawn_in_a_browser(tmp_path):
    page = tmp_path / "report.html"
    page.write_text(render_html(sample_report_data()))
    dom = subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--dump-dom",
                          "--virtual-time-budget=3000", page.as_uri()], capture_output=True, text=True,
                         timeout=60).stdout
    assert 'data-ready="1"' in dom
    assert dom.count('data-rendered="1"') == len(SECTIONS)
    assert "could not be drawn:" not in dom.split("<script")[0]
    assert dom.count("<svg") >= 6                       # rating, anchors, opponents, training, system, matrices


def test_write_report_default_path_and_output(tmp_path):
    run = write_run_dir(tmp_path, "rank")
    path = write_report(run.root)
    assert path == run.root / "report.html" and _data_of(path.read_text())["sections"]["ratings"]["layouts"]
    other = write_report(run.root, tmp_path / "x.html")
    assert other == tmp_path / "x.html" and other.exists()
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigError, match="neither a run directory"):
        write_report(tmp_path / "empty")


def test_a_tournament_dir_gets_a_tournament_only_page(tmp_path):
    tdir = tmp_path / "t1"
    tdir.mkdir()
    (tdir / "tournament.yaml").write_text("mode: select\n")
    report = sample_report_data()["sections"]["tournaments"]["items"][0]["report"]
    (tdir / "report.json").write_text(json.dumps(report))
    data = _data_of(write_tournament_html(tdir).read_text())
    assert set(data["sections"]) == {"tournaments"}
    assert data["sections"]["tournaments"]["items"][0]["report"]["winner"] == "main@v2000"
```

Create `tests/integration/test_sp4_report_cli.py`:

```python
"""``colosseum report`` on short real runs of demo games of every outcome kind (SP4 criterion 3.7, T7.2):
tic_tac_toe (wdl), tron (wdl + rank), coin_grid (score). The file opens offline: data embedded, every
section present, no external resources."""
from __future__ import annotations

import json
import re
import sys

import pytest

from cli_runner import REPO_ROOT, run_in_session, run_train
from colosseum.arena.report.data import SECTIONS

EXAMPLES = REPO_ROOT / "configs" / "examples"
URL = re.compile(r"(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@pytest.mark.parametrize(("config", "outcome"), [("tic_tac_toe", "wdl"), ("tron", "rank"), ("coin_grid", "score")],
                         ids=["wdl", "rank", "score"])
def test_report_of_a_demo_run(tmp_path, config, outcome):
    run = run_train(EXAMPLES / f"{config}.yaml", tmp_path, name=config)
    assert run.returncode == 0, run.stderr[-2000:]
    proc = run_in_session([sys.executable, "-m", "colosseum", "report", str(run.root)], timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    page = (run.root / "report.html").read_text()
    assert f"Report written to {run.root / 'report.html'}" in proc.stdout
    for name in SECTIONS:
        assert f'id="sec-{name}"' in page
    data = json.loads(re.search(r'id="colosseum-data">(.*?)</script>', page, re.S).group(1))
    layouts = data["sections"]["ratings"]["layouts"]
    assert outcome in {entry["outcome_kind"] for entry in layouts.values()}
    for entry in layouts.values():                     # score layouts: mean score, never BT
        assert entry["kind"] == ("score" if entry["outcome_kind"] == "score" else "bt")
    assert data["sections"]["summary"]["status"] == "finished"
    assert data["sections"]["matches"]["rows"] and data["sections"]["training"]["agents"]
    assert not URL.search(page) and len(page) < 5_000_000


def test_report_exit_codes(tmp_path):
    proc = run_in_session([sys.executable, "-m", "colosseum", "report", str(tmp_path)], timeout=60)
    assert proc.returncode == 1 and "Config error:" in proc.stderr
    proc = run_in_session([sys.executable, "-m", "colosseum", "report", str(tmp_path / "missing")], timeout=60)
    assert proc.returncode == 2
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_report_render.py -q`

Expected: collection error `ModuleNotFoundError: No module named 'colosseum.arena.report.render'`.

- [ ] **Step 2: The assets**

Create `src/colosseum/arena/report/assets/page.html` (placeholders `{{TITLE}}`, `{{CSS}}`, `{{JS}}`, `{{CONFIG}}`, `{{DATA}}` are filled by `render_html` in one pass):

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
{{CSS}}
</style>
</head>
<body>
<header class="top">
  <div class="top-row"><h1 id="page-title">{{TITLE}}</h1><span id="live-status" class="muted"></span></div>
  <nav id="nav"></nav>
</header>
<main id="main">
  <section id="sec-summary" class="sec"><h2>Summary</h2><div class="body"></div></section>
  <section id="sec-ratings" class="sec"><h2>Ratings</h2><div class="body"></div></section>
  <section id="sec-matrix" class="sec"><h2>Results matrix</h2><div class="body"></div></section>
  <section id="sec-anchors" class="sec"><h2>Anchors</h2><div class="body"></div></section>
  <section id="sec-opponents" class="sec"><h2>Opponents</h2><div class="body"></div></section>
  <section id="sec-training" class="sec"><h2>Training</h2><div class="body"></div></section>
  <section id="sec-system" class="sec"><h2>System</h2><div class="body"></div></section>
  <section id="sec-matches" class="sec"><h2>Matches</h2><div class="body"></div></section>
  <section id="sec-tournaments" class="sec"><h2>Tournaments</h2><div class="body"></div></section>
</main>
<div id="tip" class="tip" hidden></div>
<noscript><p class="note">This page draws its tables and charts with JavaScript; the data is embedded below as JSON.</p></noscript>
<script type="application/json" id="colosseum-config">{{CONFIG}}</script>
<script type="application/json" id="colosseum-data">{{DATA}}</script>
<script>
{{JS}}
</script>
</body>
</html>
```

Create `src/colosseum/arena/report/assets/page.css` (palette: the dataviz reference palette — categorical slots 1–8, diverging blue / red with a gray midpoint, status colors with an icon and a label; light and dark steps chosen separately):

```css
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11, 11, 11, 0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --div-pos: #2a78d6; --div-neg: #e34948; --div-mid: #f0efec; --few: #e1e0d9;
  --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255, 255, 255, 0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
    --div-pos: #3987e5; --div-neg: #e66767; --div-mid: #383835; --few: #2c2c2a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255, 255, 255, 0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
  --div-pos: #3987e5; --div-neg: #e66767; --div-mid: #383835; --few: #2c2c2a;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { background: var(--page); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.top { position: sticky; top: 0; z-index: 5; background: var(--page); border-bottom: 1px solid var(--ring); padding: 8px 16px; }
.top-row { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; }
h1 { font-size: 18px; margin: 0; overflow-wrap: anywhere; }
h2 { font-size: 16px; margin: 0 0 8px; }
h3 { font-size: 14px; margin: 16px 0 6px; color: var(--ink-2); overflow-wrap: anywhere; }
nav { display: flex; gap: 4px 12px; overflow-x: auto; white-space: nowrap; padding-top: 4px; font-size: 13px; }
nav a { color: var(--ink-2); text-decoration: none; }
nav a:hover { color: var(--ink); text-decoration: underline; }
main { max-width: 1100px; margin: 0 auto; padding: 8px 16px 48px; }
.sec { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 12px; margin: 12px 0; min-width: 0; }
.sec[hidden] { display: none; }
.muted { color: var(--muted); }
.note { color: var(--ink-2); font-size: 13px; margin: 4px 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; margin-bottom: 8px; }
.tile { border: 1px solid var(--ring); border-radius: 6px; padding: 8px; min-width: 0; }
.tile .k { color: var(--ink-2); font-size: 12px; }
.tile .v { font-size: 18px; font-weight: 600; overflow-wrap: anywhere; }
.status { display: inline-flex; align-items: center; gap: 6px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; background: var(--muted); }
.dot.good { background: var(--good); } .dot.warning { background: var(--warning); } .dot.critical { background: var(--critical); }
.flag { border-left: 4px solid var(--warning); padding: 6px 8px; margin: 6px 0; background: var(--page); }
.flag.critical { border-left-color: var(--critical); }
.flag.good { border-left-color: var(--good); }
.controls { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; margin: 4px 0 8px; }
.controls label { font-size: 13px; color: var(--ink-2); display: inline-flex; gap: 4px; align-items: center; }
select, input[type="search"] { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--axis); border-radius: 4px; padding: 2px 6px; max-width: 220px; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; max-width: 100%; }
.scroll.tbl { max-height: 480px; overflow-y: auto; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 3px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; cursor: pointer; user-select: none; position: sticky; top: 0; background: var(--surface); }
th.num, td.num { text-align: right; }
th[aria-sort="ascending"]::after { content: " \25B2"; }
th[aria-sort="descending"]::after { content: " \25BC"; }
.chart { position: relative; margin: 4px 0 8px; }
.chart svg { display: block; width: 100%; max-width: 680px; height: auto; overflow: visible; }
.chart text { fill: var(--muted); font-size: 11px; }
.chart .ttl { fill: var(--ink-2); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 12px; color: var(--ink-2); }
.legend i { display: inline-block; width: 14px; height: 3px; vertical-align: middle; margin-right: 4px; border-radius: 2px; }
.legend i.box { height: 10px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 8px 16px; }
.tip { position: fixed; z-index: 10; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--axis); border-radius: 6px; padding: 6px 8px; font-size: 12px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15); max-width: 280px; }
.tip b { font-weight: 600; }
.cellinfo { min-height: 1.4em; font-size: 13px; color: var(--ink-2); }
details > summary { cursor: pointer; color: var(--ink-2); }
pre { overflow-x: auto; font-size: 12px; }
.chips { display: flex; flex-wrap: wrap; gap: 4px 10px; }
.chips label { font-size: 12px; }
@media (max-width: 640px) {
  main { padding: 4px 16px 32px; }
  .sec { padding: 8px; }
  .grid2 { grid-template-columns: 1fr; }
}
@media (forced-colors: active) { .dot { border: 1px solid CanvasText; } }
```

Create `src/colosseum/arena/report/assets/page.js`:

```javascript
/* Colosseum run report (SP4 spec block 5): renders the embedded JSON data; with config.live, refreshes
   every section from api/<section> (colosseum dashboard). No external resources: SVG charts drawn here. */
"use strict";
(function () {
  const CFG = JSON.parse(document.getElementById("colosseum-config").textContent);
  const DATA = JSON.parse(document.getElementById("colosseum-data").textContent);
  const ORDER = ["summary", "ratings", "matrix", "anchors", "opponents", "training", "system", "matches", "tournaments"];
  const TITLES = { summary: "Summary", ratings: "Ratings", matrix: "Results matrix", anchors: "Anchors",
    opponents: "Opponents", training: "Training", system: "System", matches: "Matches", tournaments: "Tournaments" };
  const CATEGORIES = ["latest", "snapshots", "rivals", "anchors", "fallback"];
  const STATE = {};          // per-section UI state (selected layout, agent, metrics, filters, sort)
  const COLOR_OF = {};       // entity / lineage / series name -> slot (color follows the entity, never its rank)
  let uid = 0;

  // ---------- helpers ----------
  function esc(v) {
    return String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function num(v, d) {
    if (v === null || v === undefined || Number.isNaN(v)) return "-";
    if (typeof v !== "number") return String(v);
    if (d === undefined) d = Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 1 ? 2 : 3;
    return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function ci(c, d) { return c && c.length === 2 ? "[" + num(c[0], d) + ", " + num(c[1], d) + "]" : "-"; }
  function slot(name) {
    if (!(name in COLOR_OF)) COLOR_OF[name] = Object.keys(COLOR_OF).length % 8;
    return "var(--s" + (COLOR_OF[name] + 1) + ")";
  }
  function catColor(name) { return "var(--s" + (CATEGORIES.indexOf(name) % 8 + 1) + ")"; }   // fixed per category
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
      else if (k === "html") e.innerHTML = v;
      else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid !== null && kid !== undefined) e.append(kid instanceof Node ? kid : String(kid));
    return e;
  }
  function notes(list) { return (list || []).map((n) => h("p", { class: "note" }, n)); }
  function st(name, key, dflt) {
    STATE[name] = STATE[name] || {};
    if (!(key in STATE[name])) STATE[name][key] = dflt;
    return STATE[name][key];
  }
  function select(name, key, label, options, onchange) {
    const cur = st(name, key, options.length ? options[0] : "");
    const s = h("select", { "aria-label": label, on: { change: (e) => { STATE[name][key] = e.target.value; onchange(); } } },
      options.map((o) => h("option", { value: o, selected: o === cur }, o === "" ? "(all)" : o)));
    return h("label", {}, label, s);
  }
  function tile(k, v, cls) { return h("div", { class: "tile" + (cls ? " " + cls : "") }, h("div", { class: "k" }, k), h("div", { class: "v" }, v)); }
  function statusBadge(status) {
    const cls = { running: "good", finished: "good", failed: "critical", stopped: "warning" }[status] || "";
    return h("span", { class: "status" }, h("i", { class: "dot " + cls, "aria-hidden": "true" }), status || "unknown");
  }
  const TIP = document.getElementById("tip");
  function showTip(ev, rows) {
    TIP.replaceChildren(...rows.map((r) => h("div", {}, r[0] !== null ? h("b", {}, r[0]) : null, r[1] ? " " + r[1] : "")));
    TIP.hidden = false;
    const x = Math.min(ev.clientX + 12, window.innerWidth - TIP.offsetWidth - 8);
    const y = Math.min(ev.clientY + 12, window.innerHeight - TIP.offsetHeight - 8);
    TIP.style.left = Math.max(4, x) + "px";
    TIP.style.top = Math.max(4, y) + "px";
  }
  function hideTip() { TIP.hidden = true; }

  // ---------- table: sortable, filterable, scrolls inside its own box ----------
  function table(name, cols, rows, opts) {
    opts = opts || {};
    const key = opts.key || "table";
    const sort = st(name, key + ":sort", opts.sort || null);
    const box = h("div", {});
    const filt = st(name, key + ":filter", "");
    const input = opts.filter === false ? null : h("input", { type: "search", placeholder: "filter", value: filt, "aria-label": "filter rows",
      on: { input: (e) => { STATE[name][key + ":filter"] = e.target.value; draw(); } } });
    const wrap = h("div", { class: "scroll tbl" });
    function draw() {
      const f = (STATE[name][key + ":filter"] || "").toLowerCase();
      let rs = rows.filter((r) => !f || cols.some((c) => String(c.text ? c.text(r) : r[c.key] ?? "").toLowerCase().includes(f)));
      const s = STATE[name][key + ":sort"];
      if (s) {
        const c = cols.find((x) => x.key === s.key);
        const val = (r) => (c && c.sortv ? c.sortv(r) : r[s.key]);
        rs = rs.slice().sort((a, b) => {
          const x = val(a), y = val(b);
          if (x === y) return 0;
          if (x === null || x === undefined) return 1;
          if (y === null || y === undefined) return -1;
          return (x < y ? -1 : 1) * (s.dir === "desc" ? -1 : 1);
        });
      }
      const head = h("tr", {}, cols.map((c) => h("th", {
        class: c.num ? "num" : null, scope: "col",
        "aria-sort": s && s.key === c.key ? (s.dir === "desc" ? "descending" : "ascending") : null,
        on: { click: () => { const cur = STATE[name][key + ":sort"]; STATE[name][key + ":sort"] = { key: c.key, dir: cur && cur.key === c.key && cur.dir === "asc" ? "desc" : "asc" }; draw(); } },
      }, c.label)));
      const body = rs.map((r) => h("tr", {}, cols.map((c) => {
        const v = c.cell ? c.cell(r) : c.text ? c.text(r) : c.fmt ? c.fmt(r[c.key]) : r[c.key] ?? "-";
        return h("td", { class: c.num ? "num" : null }, v);
      })));
      wrap.replaceChildren(h("table", {}, h("thead", {}, head), h("tbody", {}, body)));
      count.textContent = rs.length === rows.length ? rows.length + " rows" : rs.length + " of " + rows.length + " rows";
    }
    const count = h("span", { class: "muted" });
    box.append(h("div", { class: "controls" }, input, count), wrap);
    draw();
    return box;
  }

  // ---------- scales and SVG charts (drawn as markup inserted into an HTML element) ----------
  function extent(vals) {
    let lo = Infinity, hi = -Infinity;
    for (const v of vals) if (typeof v === "number" && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (lo === Infinity) return [0, 1];
    if (lo === hi) return [lo - 1, hi + 1];
    return [lo, hi];
  }
  function ticks(lo, hi, n) {
    const step0 = (hi - lo) / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  function tickLabel(v) { return Math.abs(v) >= 1e6 ? (v / 1e6).toPrecision(3) + "M" : Math.abs(v) >= 1e4 ? (v / 1e3).toPrecision(3) + "k" : String(+v.toPrecision(4)); }

  /* series: [{name, pts: [[x, y, lo, hi], ...]}]; hlines: [{name, y}]; y0/y1 fix the y range (e.g. 0..1). */
  function chartWidth() { return Math.max(300, Math.min(640, window.innerWidth - 56)); }
  function lineChart(opts) {
    const W = Math.min(opts.width || 640, chartWidth()), H = opts.height || 240, L = 48, R = 12, T = 22, B = 34;
    const series = opts.series.filter((s) => s.pts.length);
    const box = h("div", { class: "chart" });
    if (!series.length) { box.append(h("p", { class: "note" }, opts.empty || "no data yet")); return box; }
    const xs = series.flatMap((s) => s.pts.map((p) => p[0]));
    const ys = series.flatMap((s) => s.pts.flatMap((p) => [p[1], p[2], p[3]])).concat((opts.hlines || []).map((l) => l.y));
    const [x0, x1] = extent(xs);
    let [y0, y1] = extent(ys);
    if (opts.y0 !== undefined) y0 = Math.min(y0, opts.y0);
    if (opts.y1 !== undefined) y1 = Math.max(y1, opts.y1);
    const sx = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
    const sy = (y) => T + (1 - (y - y0) / (y1 - y0 || 1)) * (H - T - B);
    let svg = "";
    for (const t of ticks(y0, y1, 4)) svg += `<line x1="${L}" x2="${W - R}" y1="${sy(t)}" y2="${sy(t)}" stroke="var(--grid)"/><text x="${L - 6}" y="${sy(t) + 4}" text-anchor="end">${esc(tickLabel(t))}</text>`;
    for (const t of ticks(x0, x1, 5)) svg += `<text x="${sx(t)}" y="${H - B + 16}" text-anchor="middle">${esc(tickLabel(t))}</text>`;
    svg += `<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="var(--axis)"/>`;
    svg += `<text class="ttl" x="${W - R}" y="${H - 4}" text-anchor="end">${esc(opts.xlabel || "")}</text>`;
    if (opts.ylabel) svg += `<text class="ttl" x="${L}" y="12">${esc(opts.ylabel)}</text>`;
    for (const l of opts.hlines || []) {
      svg += `<line x1="${L}" x2="${W - R}" y1="${sy(l.y)}" y2="${sy(l.y)}" stroke="var(--ink-2)" stroke-dasharray="4 4"/>`;
      svg += `<text x="${W - R}" y="${sy(l.y) - 4}" text-anchor="end">${esc(l.name)}</text>`;
    }
    for (const s of series) {
      const c = slot(s.name);
      const band = s.pts.filter((p) => typeof p[2] === "number" && typeof p[3] === "number");
      if (band.length > 1) {
        const up = band.map((p) => `${sx(p[0])},${sy(p[3])}`), dn = band.slice().reverse().map((p) => `${sx(p[0])},${sy(p[2])}`);
        svg += `<polygon points="${up.concat(dn).join(" ")}" fill="${c}" fill-opacity="0.16" stroke="none"/>`;
      }
      const pts = s.pts.filter((p) => typeof p[1] === "number");
      svg += `<polyline points="${pts.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round"/>`;
      if (pts.length <= 40) for (const p of pts) svg += `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="3.5" fill="${c}" stroke="var(--surface)" stroke-width="2"/>`;
    }
    svg += `<line class="xh" x1="0" x2="0" y1="${T}" y2="${H - B}" stroke="var(--ink-2)" visibility="hidden"/>`;
    svg += `<rect class="hit" x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/>`;
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.label || "chart")}">${svg}</svg>`;
    const svgEl = box.firstChild, xh = svgEl.querySelector(".xh");
    const allX = Array.from(new Set(xs)).sort((a, b) => a - b);
    function move(ev) {
      const r = svgEl.getBoundingClientRect();
      const x = x0 + ((ev.clientX - r.left) * (W / r.width) - L) / (W - L - R) * (x1 - x0);
      let best = allX[0];
      for (const v of allX) if (Math.abs(v - x) < Math.abs(best - x)) best = v;
      xh.setAttribute("x1", sx(best)); xh.setAttribute("x2", sx(best)); xh.setAttribute("visibility", "visible");
      const rows = [[null, (opts.xlabel || "x") + " " + num(best, 0)]];
      for (const s of series) {
        const p = s.pts.find((q) => q[0] === best);
        if (p) rows.push([num(p[1], opts.digits), s.name + (typeof p[2] === "number" ? " " + ci([p[2], p[3]], opts.digits) : "") + (p[4] !== undefined ? ", n " + p[4] : "")]);
      }
      showTip(ev, rows);
    }
    svgEl.querySelector(".hit").addEventListener("pointermove", move);
    svgEl.querySelector(".hit").addEventListener("pointerleave", () => { xh.setAttribute("visibility", "hidden"); hideTip(); });
    if (series.length > 1 || (opts.hlines || []).length) {
      box.append(h("div", { class: "legend" }, series.map((s) => h("span", {}, h("i", { style: "background:" + slot(s.name) }), s.name)),
        (opts.hlines || []).map((l) => h("span", {}, h("i", { style: "background:var(--ink-2)" }), l.name + " (dashed)"))));
    }
    return box;
  }

  /* x: [...]; parts: [{name, values: [...]}] shares stacked from 0 (fixed category order). */
  function stackedArea(x, parts, opts) {
    const W = chartWidth(), H = 200, L = 40, R = 12, T = 8, B = 30;
    const box = h("div", { class: "chart" });
    if (!x.length) { box.append(h("p", { class: "note" }, "no data yet")); return box; }
    const [x0, x1] = extent(x);
    const sx = (v) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R);
    const sy = (v) => T + (1 - v) * (H - T - B);
    let svg = "", base = x.map(() => 0);
    for (const t of [0, 0.5, 1]) svg += `<line x1="${L}" x2="${W - R}" y1="${sy(t)}" y2="${sy(t)}" stroke="var(--grid)"/><text x="${L - 6}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`;
    for (const t of ticks(x0, x1, 5)) svg += `<text x="${sx(t)}" y="${H - B + 16}" text-anchor="middle">${esc(tickLabel(t))}</text>`;
    for (const p of parts) {
      const top = base.map((b, i) => b + (p.values[i] || 0));
      const up = x.map((v, i) => `${sx(v)},${sy(top[i])}`), dn = x.slice().reverse().map((v, j) => `${sx(v)},${sy(base[x.length - 1 - j])}`);
      svg += `<polygon points="${up.concat(dn).join(" ")}" fill="${catColor(p.name)}" stroke="var(--surface)" stroke-width="1"/>`;
      base = top;
    }
    svg += `<text class="ttl" x="${W - R}" y="${H - 4}" text-anchor="end">${esc(opts.xlabel || "")}</text>`;
    svg += `<rect class="hit" x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/>`;
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.label || "shares")}">${svg}</svg>`;
    const svgEl = box.firstChild;
    svgEl.querySelector(".hit").addEventListener("pointermove", (ev) => {
      const r = svgEl.getBoundingClientRect();
      const xv = x0 + ((ev.clientX - r.left) * (W / r.width) - L) / (W - L - R) * (x1 - x0);
      let i = 0;
      for (let j = 0; j < x.length; j++) if (Math.abs(x[j] - xv) < Math.abs(x[i] - xv)) i = j;
      showTip(ev, [[null, (opts.xlabel || "x") + " " + num(x[i], 0)]].concat(parts.map((p) => [num(p.values[i], 2), p.name])));
    });
    svgEl.querySelector(".hit").addEventListener("pointerleave", hideTip);
    box.append(h("div", { class: "legend" }, parts.map((p) => h("span", {}, h("i", { class: "box", style: "background:" + catColor(p.name) }), p.name))));
    return box;
  }

  function mix(a, b, t) {
    const pa = a.match(/\w\w/g).map((v) => parseInt(v, 16)), pb = b.match(/\w\w/g).map((v) => parseInt(v, 16));
    return "rgb(" + pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",") + ")";
  }
  function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

  /* axis: labels; cells[i][j] = {score, ci, n} | null (row i vs column j); cells with n < minGames are hatched. */
  function heatmap(axis, cells, minGames, info, label) {
    const n = axis.length, cs = Math.max(14, Math.min(34, Math.floor(560 / Math.max(1, n)))), LW = 120, TH = 96;
    const W = LW + n * cs + 4, H = TH + n * cs + 4, id = "hatch" + ++uid;
    const pos = cssVar("--div-pos") || "#2a78d6", neg = cssVar("--div-neg") || "#e34948", mid = cssVar("--div-mid") || "#f0efec";
    let svg = `<defs><pattern id="${id}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="var(--few)"/><line x1="0" y1="0" x2="0" y2="6" stroke="var(--axis)" stroke-width="2"/></pattern></defs>`;
    axis.forEach((a, i) => {
      const short = a.length > 18 ? a.slice(0, 17) + "…" : a;
      svg += `<text x="${LW - 4}" y="${TH + i * cs + cs / 2 + 4}" text-anchor="end">${esc(short)}</text>`;
      svg += `<text transform="translate(${LW + i * cs + cs / 2 + 4},${TH - 4}) rotate(-60)">${esc(short)}</text>`;
    });
    cells.forEach((row, i) => row.forEach((c, j) => {
      if (!c) return;
      const few = c.n < minGames;
      const t = Math.min(1, Math.abs(c.score - 0.5) * 2);
      const fill = few ? `url(#${id})` : mix(mid, c.score >= 0.5 ? pos : neg, t);
      svg += `<rect data-i="${i}" data-j="${j}" x="${LW + j * cs + 1}" y="${TH + i * cs + 1}" width="${cs - 2}" height="${cs - 2}" rx="2" fill="${fill}"/>`;
    }));
    const box = h("div", { class: "chart" });
    box.innerHTML = `<div class="scroll"><svg viewBox="0 0 ${W} ${H}" style="width:${W}px;max-width:none" role="img" aria-label="${esc(label || "results matrix")}">${svg}</svg></div>`;
    const line = h("div", { class: "cellinfo" }, "Row vs column: score (blue = row wins more, red = fewer); hatched: fewer than " + minGames + " games. Tap a cell.");
    function describe(ev) {
      const r = ev.target.closest("rect[data-i]");
      if (!r) return null;
      const i = +r.dataset.i, j = +r.dataset.j, c = cells[i][j];
      return [axis[i] + " vs " + axis[j], "score " + num(c.score, 3) + " " + ci(c.ci, 3) + ", " + c.n + " games" + (c.n < minGames ? " (few games)" : "")];
    }
    const svgEl = box.querySelector("svg");
    svgEl.addEventListener("pointermove", (ev) => { const d = describe(ev); if (d) showTip(ev, [[d[1], ""], [null, d[0]]]); else hideTip(); });
    svgEl.addEventListener("pointerleave", hideTip);
    svgEl.addEventListener("click", (ev) => { const d = describe(ev); if (d) line.textContent = d[0] + ": " + d[1]; });
    box.append(line);
    if (info) box.append(h("p", { class: "note" }, info));
    return box;
  }

  // ---------- sections ----------
  const R = {};

  R.summary = function (s) {
    const out = [];
    const budget = s.budget_frac !== null && s.budget_frac !== undefined ? num(100 * s.budget_frac, 1) + " %" : "-";
    out.push(h("div", { class: "tiles" },
      tile("State", statusBadge(s.status)),
      tile("Env steps", num(s.env_steps, 0) + (s.total_timesteps ? " / " + num(s.total_timesteps, 0) : "")),
      tile("Budget", budget),
      tile("Env steps/s", num(s.env_steps_per_sec, 0)),
      tile("Agents", String((s.agents || []).length))));
    if (s.status_detail) out.push(h("p", { class: "note" }, s.status_detail));
    out.push(table("summary", [
      { key: "agent", label: "agent" }, { key: "kind", label: "kind" },
      { key: "bt", label: "BT ± CI (newest)", sortv: (r) => (r.bt ? r.bt.rating : null), text: (r) => (r.bt ? num(r.bt.rating, 0) + " " + ci(r.bt.ci, 0) + " (" + r.bt.entity + ", " + r.bt.layout + ")" : "-") },
      { key: "elo", label: "ELO", num: true, fmt: (v) => num(v, 0) },
      { key: "wr_vs_past", label: "wr_vs_past", num: true, fmt: (v) => num(v, 3) },
    ], s.agents || [], { filter: false }));
    if ((s.parents || []).length) out.push(h("p", { class: "note" }, "Resumed from: " + s.parents.join(" ← ")));
    return out.concat(notes(s.notes));
  };

  function layoutsOf(s) { return Object.keys(s.layouts || {}); }

  R.ratings = function (s) {
    const layouts = layoutsOf(s), out = [];
    if (!layouts.length) return [h("p", { class: "note" }, "No ratings yet.")].concat(notes(s.notes));
    const L = st("ratings", "layout", layouts[0]);
    const lay = s.layouts[L] || s.layouts[layouts[0]];
    const lineages = Array.from(new Set((lay.groups || []).flatMap((g) => g.entities.map((e) => e.lineage)).concat((lay.entities || []).map((e) => e.lineage)))).sort();
    out.push(h("div", { class: "controls" }, select("ratings", "layout", "layout", layouts, rerender("ratings")),
      select("ratings", "agent", "agent", [""].concat(lineages), rerender("ratings")),
      h("span", { class: "muted" }, "source: " + s.source + (s.uncertainty ? ", intervals: " + s.uncertainty : ""))));
    const agent = STATE.ratings.agent || "";
    if (lay.kind === "score") {
      out.push(h("h3", {}, L + " · mean score (one team)"));
      out.push(table("ratings", [
        { key: "entity", label: "entity" }, { key: "kind", label: "kind" }, { key: "mean", label: "mean score", num: true, fmt: (v) => num(v, 3) },
        { key: "ci", label: "95% CI", text: (r) => ci(r.ci, 3), sortv: (r) => (r.ci ? r.ci[0] : null) }, { key: "n", label: "matches", num: true },
      ], (lay.entities || []).filter((e) => !agent || e.lineage === agent), { key: "score", sort: { key: "mean", dir: "desc" } }));
      return out.concat(notes(s.notes));
    }
    if (lay.transitivity && lay.transitivity.flag) {
      out.push(h("div", { class: "flag" }, "Possible cycles: G² = " + num(lay.transitivity.g2, 1) + " on " + lay.transitivity.dof + " dof, p = " + num(lay.transitivity.p_value, 4) + ". Pairs with the largest residuals:"));
      out.push(table("ratings", [{ key: "a", label: "team A" }, { key: "b", label: "team B" }, { key: "residual", label: "residual", num: true, fmt: (v) => num(v, 2) }, { key: "games", label: "games", num: true }], lay.transitivity.worst_pairs || [], { key: "g2", filter: false }));
    }
    if (lay.bias && Object.keys(lay.bias).length) out.push(h("p", { class: "note" }, "Team bias (Elo): " + Object.entries(lay.bias).map(([t, b]) => "team " + t + " " + num(b, 0)).join(", ")));
    (lay.groups || []).forEach((g, gi) => {
      out.push(h("h3", {}, L + " · group " + g.label + (g.component ? " · component " + g.component : "") + " · anchor " + g.anchor));
      const series = (g.series || []).filter((x) => !agent || x.lineage === agent).map((x) => ({ name: x.lineage, pts: x.points }));
      out.push(lineChart({ series, hlines: (g.anchors || []).map((a) => ({ name: a.entity, y: a.rating })), xlabel: "version", ylabel: "BT (Elo units)", digits: 0, label: "rating by version", empty: "no trainable entities in this group" }));
      out.push(table("ratings", [
        { key: "entity", label: "entity" }, { key: "kind", label: "kind" },
        { key: "rating", label: "BT", num: true, fmt: (v) => num(v, 0) },
        { key: "ci", label: "95% CI", text: (r) => ci(r.ci, 0), sortv: (r) => (r.ci ? r.ci[0] : null) },
        { key: "games", label: "games", num: true },
        { key: "p_best_lineage", label: "P(best of agent)", num: true, fmt: (v) => num(v, 2) },
        { key: "p_best_group", label: "P(best)", num: true, fmt: (v) => num(v, 2) },
        { key: "lcb", label: "LCB", num: true, fmt: (v) => num(v, 0) },
      ], g.entities.filter((e) => !agent || e.lineage === agent), { key: "g" + gi, sort: { key: "rating", dir: "desc" } }));
      if (g.truncated) out.push(h("p", { class: "note" }, g.truncated + " more entities not shown (use the dashboard filters)."));
    });
    return out.concat(notes(s.notes));
  };

  R.matrix = function (s) {
    const layouts = layoutsOf(s), out = [];
    if (!layouts.length) return [h("p", { class: "note" }, "No matches yet.")].concat(notes(s.notes));
    const L = st("matrix", "layout", layouts[0]);
    const lay = s.layouts[L] || s.layouts[layouts[0]];
    out.push(h("div", { class: "controls" }, select("matrix", "layout", "layout", layouts, rerender("matrix"))));
    if (lay.kind === "score") {
      out.push(h("h3", {}, "Cross-play: team composition → mean score"));
      out.push(table("matrix", [{ key: "composition", label: "composition" }, { key: "mean", label: "mean score", num: true, fmt: (v) => num(v, 3) },
        { key: "ci", label: "95% CI", text: (r) => ci(r.ci, 3), sortv: (r) => (r.ci ? r.ci[0] : null) }, { key: "n", label: "matches", num: true },
        { key: "homogeneous", label: "homogeneous", text: (r) => (r.homogeneous ? "yes" : "no") }], lay.cross_play || [], { key: "cp", sort: { key: "mean", dir: "desc" } }));
      return out.concat(notes(s.notes));
    }
    if (lay.lineages && lay.lineages.axis.length) {
      out.push(h("h3", {}, "By agent (lineage), ordered by rating"));
      out.push(heatmap(lay.lineages.axis, lay.lineages.cells, s.min_games, null, "results by lineage"));
    }
    if (lay.snapshots && lay.snapshots.axis.length) {
      out.push(h("h3", {}, "Snapshots of " + lay.snapshots.agent));
      out.push(heatmap(lay.snapshots.axis, lay.snapshots.cells, s.min_games,
        lay.snapshots.thinned_from > lay.snapshots.axis.length ? "Showing " + lay.snapshots.axis.length + " of " + lay.snapshots.thinned_from + " entities (evenly spaced versions)." : null, "results by snapshot"));
    }
    return out.concat(notes(s.notes));
  };

  R.anchors = function (s) {
    const out = [];
    if (s.biased) out.push(h("div", { class: "flag" }, "From training games: biased by PFSP (opponents are chosen by the matchmaker, latest weights change). Enable evaluation (pulse) for an unbiased curve."));
    else if (s.source === "pulse") out.push(h("p", { class: "note" }, "From pulse evaluation: every snapshot against the same field on the same seeds."));
    for (const [L, agents] of Object.entries(s.layouts || {})) for (const [agent, a] of Object.entries(agents)) {
      out.push(h("h3", {}, L + " · " + agent));
      out.push(lineChart({ series: Object.entries(a.anchors).map(([name, pts]) => ({ name, pts })), xlabel: "version", ylabel: "score vs anchor", y0: 0, y1: 1, digits: 3, label: "score against anchors" }));
      if ((a.unevaluated || []).length) out.push(h("p", { class: "note" }, "Not evaluated: v" + a.unevaluated.join(", v")));
    }
    if (!Object.keys(s.layouts || {}).length) out.push(h("p", { class: "note" }, "No games against anchors yet."));
    return out.concat(notes(s.notes));
  };

  R.opponents = function (s) {
    const out = [];
    for (const [agent, layouts] of Object.entries(s.agents || {})) for (const [L, d] of Object.entries(layouts)) {
      out.push(h("h3", {}, agent + " · " + L + " · played opponent categories"));
      out.push(stackedArea(d.env_steps, CATEGORIES.filter((c) => c in d.categories).map((c) => ({ name: c, values: d.categories[c] })), { xlabel: "env steps", label: "opponent shares" }));
    }
    for (const [L, owners] of Object.entries(s.pfsp || {})) for (const [owner, rows] of Object.entries(owners)) {
      out.push(h("h3", {}, "PFSP · " + L + " · " + owner + " (top " + rows.length + " by games)"));
      out.push(table("opponents", [{ key: "player", label: "player" }, { key: "score", label: "EMA score", num: true, fmt: (v) => num(v, 3) }, { key: "games", label: "games", num: true, fmt: (v) => num(v, 0) }], rows, { key: L + owner, filter: false }));
    }
    if (!out.length) out.push(h("p", { class: "note" }, "No opponent data yet."));
    return out.concat(notes(s.notes));
  };

  function seriesPicker(name, all, dflt, after) {
    const chosen = st(name, "metrics", dflt.filter((m) => all.includes(m)));
    return h("details", {}, h("summary", {}, "series (" + chosen.length + " of " + all.length + ")"),
      h("div", { class: "chips" }, all.map((m) => h("label", {}, h("input", { type: "checkbox", checked: chosen.includes(m), on: { change: (e) => {
        const set = new Set(STATE[name].metrics);
        if (e.target.checked) set.add(m); else set.delete(m);
        STATE[name].metrics = all.filter((x) => set.has(x)); after();
      } } }), m))));
  }

  R.training = function (s) {
    const agents = Object.keys(s.agents || {}), out = [];
    if (!agents.length) return [h("p", { class: "note" }, "No train records yet.")].concat(notes(s.notes));
    const A = st("training", "agent", agents[0]);
    const d = s.agents[A] || s.agents[agents[0]];
    const all = Object.keys(d.metrics).sort();
    out.push(h("div", { class: "controls" }, select("training", "agent", "agent", agents, rerender("training")), seriesPicker("training", all, s.default || [], rerender("training"))));
    out.push(h("div", { class: "grid2" }, STATE.training.metrics.map((m) => h("div", {}, h("h3", {}, m),
      lineChart({ series: [{ name: m, pts: d.train_step.map((x, i) => [x, d.metrics[m][i]]).filter((p) => p[1] !== null) }], xlabel: "train step", height: 180, width: 360, label: m })))));
    return out.concat(notes(s.notes));
  };

  R.system = function (s) {
    const out = [], latest = s.latest || {};
    const pick = (k, d) => (k in latest ? num(latest[k], d) : "-");
    out.push(h("div", { class: "tiles" },
      tile("Env steps/s", pick("env_steps_per_sec", 0)),
      tile("Dropped match results", pick("dropped_match_results", 0)),
      tile("Dropped reward episodes", pick("dropped_reward_episodes", 0)),
      tile("Journal", s.journal ? num(s.journal.file_mb, 1) + " MB" + (s.journal.counts_only ? " · counts only" : "") : "-"),
      tile("Free disk", s.free_disk_mb !== null && s.free_disk_mb !== undefined ? num(s.free_disk_mb / 1024, 1) + " GB" : "-")));
    if (s.journal && s.journal.counts_only) out.push(h("div", { class: "flag critical" }, "Journal in counts-only mode: free disk below match_log.min_free_disk_mb."));
    const all = Object.keys(s.series || {}).sort();
    if (all.length) {
      out.push(h("div", { class: "controls" }, seriesPicker("system", all, s.default || [], rerender("system"))));
      out.push(h("div", { class: "grid2" }, STATE.system.metrics.map((m) => h("div", {}, h("h3", {}, m),
        lineChart({ series: [{ name: m, pts: s.env_steps.map((x, i) => [x, s.series[m][i]]).filter((p) => p[1] !== null) }], xlabel: "env steps", height: 180, width: 360, label: m })))));
    }
    if (s.journal) out.push(h("p", { class: "note" }, "Journal: " + Object.entries(s.journal).map(([k, v]) => k + " " + num(v)).join(", ")));
    return out.concat(notes(s.notes));
  };

  function composition(m) {
    const teams = {};
    for (const x of m.seats || []) (teams[x.team] = teams[x.team] || []).push(x.agent_id + (x.rating_version !== null && x.rating_version !== undefined ? "@v" + x.rating_version : "") + (x.network_id && x.network_id !== "latest" && x.network_id !== "fixed" ? " (" + x.network_id + ")" : ""));
    const ranks = {};
    for (const t of m.teams || []) ranks[t.team] = t;
    return Object.keys(teams).map((t) => "t" + t + " [" + (ranks[t] ? (m.outcome_kind === "score" ? "score " + num(ranks[t].score, 2) : "rank " + ranks[t].rank) : "?") + "]: " + teams[t].join(", ")).join(" | ");
  }
  function outcomeFor(m, agent) {
    if (!agent) return "";
    const mine = (m.seats || []).filter((x) => x.agent_id === agent).map((x) => x.team);
    if (!mine.length || m.outcome_kind === "score") return "";
    const rank = {};
    for (const t of m.teams || []) rank[t.team] = t.rank;
    const own = Math.min(...mine.map((t) => rank[t])), others = Object.keys(rank).filter((t) => !mine.includes(+t)).map((t) => rank[t]);
    if (!others.length) return "";
    const best = Math.min(...others);
    return own < best ? "win" : own === best ? "draw" : "loss";
  }
  R.matches = function (s) {
    const rows = s.rows || [], out = [];
    const opts = (f) => [""].concat(Array.from(new Set(rows.flatMap(f))).sort());
    const flt = s.filters || {};
    for (const [k, v] of Object.entries(flt)) if (v !== null && v !== undefined && !(STATE.matches && k in STATE.matches)) st("matches", k, String(v));
    out.push(h("div", { class: "controls" },
      select("matches", "agent", "agent", opts((m) => (m.seats || []).map((x) => x.agent_id)), refilter),
      select("matches", "layout", "layout", opts((m) => [m.layout]), refilter),
      select("matches", "source", "source", opts((m) => [m.source]), refilter),
      select("matches", "outcome", "outcome", ["", "win", "draw", "loss"], refilter)));
    const f = STATE.matches;
    const shown = rows.filter((m) => (!f.agent || (m.seats || []).some((x) => x.agent_id === f.agent)) && (!f.layout || m.layout === f.layout)
      && (!f.source || m.source === f.source) && (!f.outcome || outcomeFor(m, f.agent) === f.outcome));
    out.push(table("matches", [
      { key: "env_steps", label: "env steps", num: true, fmt: (v) => num(v, 0) },
      { key: "source", label: "source" }, { key: "layout", label: "layout" },
      { key: "composition", label: "teams (versions)", text: composition },
      { key: "seed", label: "seed", num: true, fmt: (v) => (v === null || v === undefined ? "-" : String(v)) },
      { key: "episode_length", label: "length", num: true },
      { key: "replay", label: "replay", cell: (r) => (r.replay && !/^\s*javascript:/i.test(r.replay) ? h("a", { href: r.replay }, "replay") : "-") },
      { key: "uid", label: "uid" },
    ], shown, { key: "m" }));
    out.push(h("p", { class: "note" }, "The latest " + rows.length + " matches" + (CFG.live ? " matching the filters." : " (filters apply to these).")));
    return out.concat(notes(s.notes));
  };
  function refilter() { if (CFG.live) refresh("matches"); else rerender("matches")(); }

  function tournamentGroup(name, g, gi) {
    const out = [];
    const cands = g.candidates || [];
    out.push(table(name, [
      { key: "display", label: "candidate", text: (r) => r.display || r.name },
      { key: "score", label: "score vs field", num: true, fmt: (v) => num(v, 3) },
      { key: "score_ci", label: "95% CI", text: (r) => ci(r.score_ci, 3), sortv: (r) => (r.score_ci ? r.score_ci[0] : null) },
      { key: "worst", label: "worst opponent", text: (r) => (r.worst ? r.worst.opponent + " " + num(r.worst.score, 3) : "-") },
      { key: "bt", label: "BT", num: true, fmt: (v) => num(v, 0) },
      { key: "bt_ci", label: "BT CI", text: (r) => ci(r.bt_ci, 0) },
      { key: "p_best", label: "P(best)", num: true, fmt: (v) => num(v, 2) },
      { key: "games", label: "games", num: true },
      { key: "eliminated_round", label: "out in round", num: true, fmt: (v) => (v === null || v === undefined ? "-" : String(v)) },
    ], cands, { key: "c" + gi, sort: { key: "score", dir: "desc" } }));
    if (g.matrix && Object.keys(g.matrix).length) {
      const field = Array.from(new Set(Object.values(g.matrix).flatMap((row) => Object.keys(row))));
      out.push(table(name, [{ key: "cand", label: "candidate \\ field" }].concat(field.map((f) => ({ key: f, label: f, text: (r) => (r[f] ? num(r[f].score, 3) + " " + ci(r[f].ci, 2) + " n " + r[f].games : "-") }))),
        Object.entries(g.matrix).map(([cand, row]) => Object.assign({ cand }, row)), { key: "mx" + gi, filter: false }));
    }
    if (g.select) {
      const sel = g.select;
      out.push(h("div", { class: "flag " + (sel.confirmed ? "good" : "") },
        h("span", { class: "status" }, h("i", { class: "dot " + (sel.confirmed ? "good" : "warning"), "aria-hidden": "true" }), sel.confirmed ? "confirmed" : "not confirmed"),
        " winner: " + sel.winner + (sel.diff_ci ? ", difference to the runner-up " + ci(sel.diff_ci, 3) : "") +
        (sel.confirmed ? "" : "; indistinguishable: " + (sel.indistinguishable || []).join(", "))));
    }
    for (const r of g.rounds || []) out.push(h("p", { class: "note" }, "Round " + r.round + " (" + r.blocks + " blocks): eliminated " + ((r.eliminated || []).join(", ") || "none")));
    if (g.transitivity && g.transitivity.flag) out.push(h("div", { class: "flag" }, "Possible cycles among the tournament's players (p = " + num(g.transitivity.p_value, 4) + ")."));
    for (const p of g.repeated_pairs || []) out.push(h("div", { class: "flag" }, "Games repeat (not enough randomness): " + (Array.isArray(p) ? p.join(" vs ") : String(p))));
    return out;
  }
  R.tournaments = function (s) {
    const out = [];
    for (const item of s.items || []) {
      const r = item.report || {};
      out.push(h("h3", {}, (item.name || r.name || "tournament") + " · " + (r.mode || "") + (r.seed !== undefined ? " · seed " + r.seed : "")));
      if (r.winner !== undefined) out.push(h("p", {}, "Selected: " + r.winner + (r.confirmed === undefined ? "" : r.confirmed ? " (confirmed)" : " (not confirmed)")));
      for (const [L, lay] of Object.entries(r.layouts || {})) {
        const groups = lay.groups || { default: lay };
        Object.entries(groups).forEach(([gname, g], gi) => {
          out.push(h("p", { class: "note" }, "layout " + L + (gname !== "default" ? " · group " + gname : "") + (lay.outcome_kind ? " · " + lay.outcome_kind : "")));
          out.push(...tournamentGroup("tournaments", g, (item.name || "") + L + gi));
        });
      }
      out.push(h("details", {}, h("summary", {}, "raw report"), h("pre", {}, JSON.stringify(r, null, 1))));
    }
    if (!out.length) out.push(h("p", { class: "note" }, "No tournaments in this run."));
    return out.concat(notes(s.notes));
  };

  // ---------- mount, navigation, live refresh ----------
  function mount(name) {
    const sec = document.getElementById("sec-" + name);
    if (!sec) return;
    const s = DATA.sections[name];
    if (!s) { sec.hidden = true; return; }
    sec.hidden = false;
    const body = sec.querySelector(".body");
    try {
      body.replaceChildren(...R[name](s));
    } catch (err) {
      body.replaceChildren(h("p", { class: "flag critical" }, "This section could not be drawn: " + err), h("details", {}, h("summary", {}, "data"), h("pre", {}, JSON.stringify(s, null, 1))));
    }
    body.dataset.rendered = "1";
  }
  function rerender(name) { return () => mount(name); }
  function query(name) {
    const f = STATE[name] || {}, keys = { ratings: ["layout", "agent"], matrix: ["layout", "agent"], matches: ["agent", "layout", "source", "outcome"] }[name] || [];
    const q = keys.filter((k) => f[k]).map((k) => encodeURIComponent(k) + "=" + encodeURIComponent(f[k])).join("&");
    return q ? "?" + q : "";
  }
  let inflight = 0;
  function refresh(name) {
    inflight++;
    return fetch("api/" + name + query(name), { cache: "no-store" }).then((r) => (r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status))))
      .then((s) => { if (JSON.stringify(s) !== JSON.stringify(DATA.sections[name])) { DATA.sections[name] = s; mount(name); } })
      .catch((e) => { document.getElementById("live-status").textContent = "live · update failed: " + e.message; })
      .finally(() => { inflight--; });
  }
  function tick() {
    if (inflight) return;
    Promise.all(ORDER.map(refresh)).then(() => {
      document.getElementById("live-status").textContent = "live · refreshed every " + CFG.refresh_sec + " s · last " + new Date().toLocaleTimeString();
    });
  }
  document.getElementById("nav").replaceChildren(...ORDER.filter((n) => DATA.sections[n]).map((n) => h("a", { href: "#sec-" + n }, TITLES[n])));
  for (const n of ORDER) mount(n);
  document.body.dataset.ready = "1";
  if (CFG.live) {
    document.getElementById("live-status").textContent = "live · refreshed every " + CFG.refresh_sec + " s";
    setInterval(tick, Math.max(2, CFG.refresh_sec) * 1000);
  } else if (DATA.generated) {
    document.getElementById("live-status").textContent = "static report · " + DATA.generated;
  }
})();
```

- [ ] **Step 3: The renderer**

Create `src/colosseum/arena/report/render.py`:

```python
"""Self-contained HTML of a run report (SP4 spec block 5): ``assets/page.{html,css,js}`` (package data) with
the report data embedded as JSON. No external resources: no CDN, fonts or libraries; charts are SVG drawn
by ``page.js``. The same page serves ``colosseum report`` (static), ``colosseum dashboard`` (``live=True``:
it refreshes every section from ``api/<section>``) and tournament reports (only the tournaments section).
"""

from __future__ import annotations

import html
import json
import math
import re
from importlib import resources
from pathlib import Path
from typing import Any

from colosseum.core.errors import ConfigError
from colosseum.utils.fs import write_text_atomic

REPORT_FILE = "report.html"
_PLACEHOLDER = re.compile(r"\{\{([A-Z]+)\}\}")


def asset(name: str) -> str:
    """Text of ``colosseum/arena/report/assets/<name>`` (installed as package data)."""
    return resources.files("colosseum.arena.report").joinpath("assets", name).read_text(encoding="utf-8")


def _clean(obj: Any) -> Any:
    """JSON-safe copy: non-finite floats -> None, tuples -> lists, unknown objects -> str."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if hasattr(obj, "item"):                       # numpy scalars
        return _clean(obj.item())
    return str(obj)


def embed_json(data: Any) -> str:
    """JSON safe inside ``<script type="application/json">``: ``<``, ``>``, ``&`` and U+2028/9 are escaped,
    so no string in the data can close the script element."""
    text = json.dumps(_clean(data), separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    return (text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def render_html(data: dict, *, live: bool = False, refresh_sec: int = 10) -> str:
    """The page for report ``data`` (``data.build_report_data``); placeholders are filled in one pass, so
    nothing inside the CSS, JS or data is ever substituted again."""
    title = str(data.get("title") or "Colosseum run")
    values = {
        "TITLE": html.escape(title),
        "CSS": asset("page.css"),
        "JS": asset("page.js"),
        "CONFIG": embed_json({"live": bool(live), "refresh_sec": int(refresh_sec)}),
        "DATA": embed_json(data),
    }
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], asset("page.html"))


def is_tournament_dir(path: Path) -> bool:
    return (path / "tournament.yaml").exists() and (path / "report.json").exists()


def is_run_dir(path: Path) -> bool:
    return (path / "metrics.jsonl").exists() or (path / "config.resolved.yaml").exists()


def tournament_page_data(tournament_dir: str | Path) -> dict:
    """Report data with only the tournaments section, for one tournament dir."""
    from colosseum.arena.report.data import REPORT_FORMAT, ReportBuilder

    tdir = Path(tournament_dir)
    builder = ReportBuilder(tdir)
    try:
        return {"format": REPORT_FORMAT, "title": f"tournament {tdir.name}",
                "sections": {"tournaments": builder.section("tournaments")}}
    finally:
        builder.close()


def write_tournament_html(tournament_dir: str | Path) -> Path:
    """``<tournament dir>/report.html`` next to its ``report.json`` (called by the tournament when it finishes)."""
    tdir = Path(tournament_dir)
    return write_text_atomic(tdir / REPORT_FILE, render_html(tournament_page_data(tdir)))


def write_report(run_dir: str | Path, out: str | Path | None = None) -> Path:
    """``colosseum report``: the HTML of a run dir (default ``<run>/report.html``) or of a tournament dir."""
    from colosseum.arena.report.data import build_report_data

    root = Path(run_dir)
    if is_tournament_dir(root):
        data = tournament_page_data(root)
    elif is_run_dir(root):
        data = build_report_data(root)
        data["generated"] = "colosseum report"
    else:
        raise ConfigError(f"{root} is neither a run directory (metrics.jsonl, config.resolved.yaml) nor a "
                          f"tournament directory (tournament.yaml, report.json)")
    target = Path(out) if out is not None else root / REPORT_FILE
    return write_text_atomic(target, render_html(data))
```

- [ ] **Step 4: The `report` command and the tournament page**

In `src/colosseum/cli.py`, after the `bc` command, add:

```python
@main.command("report")
@click.argument("run", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default=None, type=click.Path(dir_okay=False),
              help="HTML file to write (default: RUN/report.html)")
def report_cmd(run: str, output: str | None) -> None:
    """Build a self-contained HTML report of a run directory (or of a tournament directory): ratings, results
    matrix, anchors, opponents, training, system, matches and tournaments. The page needs no network.

    Exit code: 0 done, 1 not a run or tournament directory (or an unreadable run), 2 bad command-line
    arguments, 130 SIGINT.
    """
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if output is not None and not Path(output).parent.is_dir():
        raise click.BadParameter(f"directory {str(Path(output).parent)!r} does not exist", param_hint="'--output'")
    with _config_errors():
        from colosseum.arena.report.render import write_report

        path = write_report(run, output)
    click.echo(f"Report written to {path}")
```

In Part 2's `src/colosseum/arena/tournament/run.py`, `TournamentDir.write_reports` (the single place that writes `report.json` / `report.txt`, R-P2-2) already ends with a guarded call of `write_tournament_html(self.path)` (contract amendment A27): remove the `try` / `except ImportError` guard so that it reads

```python
        from colosseum.arena.report.render import write_tournament_html

        written.append(write_tournament_html(self.path))   # report.html: the same page, tournaments section only
        return written
```

If Part 2's tests assert that `report.html` is absent, update them (one line each; say so in the commit body).

- [ ] **Step 5: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_report_render.py tests/integration/test_sp4_report_cli.py tests/unit/test_sp4_report_data.py -v`

Expected: all pass; the node and Chrome tests pass on the reference machine (they skip where `node` / `google-chrome` are missing); each `test_report_of_a_demo_run[...]` takes about 10–15 s. Then look at the page once by eye (not a test):

```bash
env -u PYTHONPATH .venv/bin/python -c "import sys; sys.path.insert(0, 'tests'); from report_kit import sample_report_data; from colosseum.arena.report.render import render_html; open('/tmp/sp4-sample.html', 'w').write(render_html(sample_report_data()))"
google-chrome --headless=new --no-sandbox --disable-gpu --hide-scrollbars --window-size=390,5200 --force-dark-mode \
  --blink-settings=preferredColorScheme=0 --virtual-time-budget=3000 --screenshot=/tmp/sp4-phone.png file:///tmp/sp4-sample.html
google-chrome --headless=new --no-sandbox --disable-gpu --hide-scrollbars --window-size=1100,6000 \
  --virtual-time-budget=3000 --screenshot=/tmp/sp4-desktop.png file:///tmp/sp4-sample.html
```

Open both PNGs: no overlapping labels, tables scroll inside their boxes, charts readable in both themes. Delete the three files afterwards (`rm /tmp/sp4-sample.html /tmp/sp4-phone.png /tmp/sp4-desktop.png`).

- [ ] **Step 6: Package data in the wheel**

```bash
D=$(mktemp -d); env -u PYTHONPATH uv build --wheel --offline -o $D . >/dev/null && \
  env -u PYTHONPATH .venv/bin/python -c "import sys, zipfile, glob; n = zipfile.ZipFile(glob.glob(sys.argv[1] + '/*.whl')[0]).namelist(); print(sorted(x for x in n if 'report/assets' in x))" $D; rm -rf $D
```

Expected: `['colosseum/arena/report/assets/page.css', 'colosseum/arena/report/assets/page.html', 'colosseum/arena/report/assets/page.js']`. (If `uv build --offline` cannot find `hatchling` in the cache, run without `--offline` once; a missing build backend is not a failure of this task — report it.)

- [ ] **Step 7: Full fast suite and ruff**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `env -u PYTHONPATH .venv/bin/ruff check .`

Expected: all pass, no warnings; `All checks passed!`.

- [ ] **Step 8: Commit and push**

```bash
git add src/colosseum/arena/report/render.py src/colosseum/arena/report/assets src/colosseum/cli.py \
  src/colosseum/arena/tournament/run.py tests/unit/test_sp4_report_render.py tests/integration/test_sp4_report_cli.py
git commit -m "feat: colosseum report builds a self-contained HTML page with SVG charts"
git push
```

---

### Task T7.3: `colosseum dashboard`

Spec block 5 («`colosseum dashboard runs/x [--host 127.0.0.1] [--port 8765] [--refresh 10]`»: stdlib `http.server`, JSON per section at `/api/<section>`, the page refreshes every `--refresh` seconds, the journal read-only, BT from `ratings.json` while training runs and from the journal (cached until it changes) otherwise, `127.0.0.1` by default without authentication), section 6 («сервер на свободном порту отдаёт JSON разделов»), criterion 3.7.

`ThreadingHTTPServer` with one `ReportBuilder` behind a lock (the builder is not thread-safe; requests are short). `GET /` renders the page with `live=True`; `page.js` then fetches every section every `--refresh` seconds and redraws only the sections whose JSON changed, keeping the viewer's selections. `GET /api/<section>` passes only the section's filters (`FILTERS`). A running run's journal changes every second; the builder re-reads its comparison table at most every `TABLE_MIN_AGE_SEC = 30` s (the page still refreshes matches, training and system every `--refresh`). Any exception in a request answers 500 with JSON and the server goes on. Ctrl+C closes the socket and exits 130 (the CLI's `_interrupts`).

**Files:**
- Create: `src/colosseum/arena/report/server.py`
- Modify: `src/colosseum/cli.py` (`dashboard` command)
- Test: `tests/integration/test_sp4_dashboard.py`

**Interfaces:**
- Consumes: `ReportBuilder(..., table_min_age_sec=)`, `SECTIONS`, `FILTERS` (T7.1); `render_html`, `is_run_dir` (T7.2); `ConfigError`.
- Produces: `colosseum.arena.report.server.serve(run_dir, *, host="127.0.0.1", port=8765, refresh_sec=10) -> None` (contract), `*make_server(run_dir, *, host, port, refresh_sec) -> DashboardServer` (bound, not serving; `port=0` picks a free port, `server.port`), `*DashboardServer`, `*TABLE_MIN_AGE_SEC = 30.0`; the CLI `colosseum dashboard RUN [--host] [--port] [--refresh]` (exit codes 1 / 2 / 130).

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/test_sp4_dashboard.py`:

```python
"""``colosseum dashboard`` (SP4 spec block 5, T7.3): the live page and ``/api/<section>`` JSON on an
ephemeral port, filters, errors that do not stop the server, live BT while running, the CLI's port
handling and Ctrl+C (exit 130)."""
from __future__ import annotations

import json
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest

from cli_runner import REPO_ROOT, child_env
from colosseum.arena.report.data import SECTIONS
from colosseum.arena.report.server import make_server
from colosseum.core.errors import ConfigError
from report_kit import write_run_dir


@contextmanager
def serving(root, **kwargs):
    server = make_server(root, port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.port}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def _get(url: str) -> tuple[int, str, str]:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read().decode("utf-8")


def test_page_and_every_section_over_http(tmp_path):
    run = write_run_dir(tmp_path, "wdl", pulse_versions=(200,))
    with serving(run.root, refresh_sec=3) as (base, _):
        code, ctype, page = _get(base + "/")
        assert code == 200 and ctype.startswith("text/html")
        config = json.loads(re.search(r'id="colosseum-config">(.*?)</script>', page, re.S).group(1))
        assert config == {"live": True, "refresh_sec": 3}
        for name in SECTIONS:
            code, ctype, body = _get(f"{base}/api/{name}")
            assert code == 200 and ctype == "application/json", name
            assert isinstance(json.loads(body), dict)
        code, _, body = _get(base + "/api/matches?source=pulse&agent=main&bogus=1")
        rows = json.loads(body)["rows"]
        assert rows and {r["source"] for r in rows} == {"pulse"}
        code, _, body = _get(base + "/api/nope")
        assert code == 404 and "summary" in json.loads(body)["sections"]
        assert _get(base + "/../etc/passwd")[0] == 404


def test_a_failing_section_answers_500_and_the_server_keeps_serving(tmp_path, monkeypatch):
    run = write_run_dir(tmp_path, "wdl")
    with serving(run.root) as (base, server):
        monkeypatch.setattr(server.builder, "_matrix", lambda **kw: 1 / 0)
        code, _, body = _get(base + "/api/matrix")
        assert code == 500 and "division" in json.loads(body)["error"]
        assert _get(base + "/api/summary")[0] == 200


def test_a_running_run_is_rated_from_ratings_json(tmp_path):
    run = write_run_dir(tmp_path, "wdl", status="running", live_bt=True)
    with serving(run.root) as (base, _):
        ratings = json.loads(_get(base + "/api/ratings")[2])
        assert ratings["source"] == "live"
        assert json.loads(_get(base + "/api/summary")[2])["status"] == "running"


def test_make_server_refuses_a_non_run_dir_and_a_busy_port(tmp_path):
    with pytest.raises(ConfigError, match="not a run directory"):
        make_server(tmp_path)
    run = write_run_dir(tmp_path, "wdl")
    with serving(run.root) as (_, server):
        with pytest.raises(ConfigError, match="--port"):
            make_server(run.root, port=server.port)


def test_cli_prints_the_address_serves_and_exits_130_on_ctrl_c(tmp_path):
    run = write_run_dir(tmp_path, "wdl")
    proc = subprocess.Popen([sys.executable, "-m", "colosseum", "dashboard", str(run.root), "--port", "0",
                             "--refresh", "5"], cwd=REPO_ROOT, env=child_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        deadline, line = time.monotonic() + 60, ""
        while time.monotonic() < deadline and "Dashboard" not in line:
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready:
                line = proc.stdout.readline()
        m = re.search(r"http://127\.0\.0\.1:(\d+)/", line)
        assert m, line
        assert _get(f"http://127.0.0.1:{m.group(1)}/api/summary")[0] == 200
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=30) == 130
        assert "Interrupted" in proc.stderr.read()
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp4_dashboard.py -q`

Expected: collection error `ModuleNotFoundError: No module named 'colosseum.arena.report.server'`.

- [ ] **Step 2: The server**

Create `src/colosseum/arena/report/server.py`:

```python
"""``colosseum dashboard``: the run report served live (SP4 spec block 5).

``GET /`` is the report page in live mode (it refreshes every section from ``api/<section>`` every
``refresh_sec`` seconds); ``GET /api/<section>`` is one section as JSON (``data.SECTIONS``; filters as
query parameters: ``agent``, ``layout``, ``source``, ``outcome``). The journal is opened read-only; BT
tables come from ``ratings.json`` while the run is running, else from the journal, cached until the
journal files change (``ReportBuilder``). Binds 127.0.0.1 by default and has no authentication: reach
it from another machine or a phone through an SSH tunnel (docs/ARENA_GUIDE.md).
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from colosseum.arena.report.data import FILTERS, SECTIONS, ReportBuilder
from colosseum.arena.report.render import is_run_dir, render_html
from colosseum.core.errors import ConfigError

logger = logging.getLogger(__name__)

# A running run's journal changes every second; its comparison table is re-read at most this often.
TABLE_MIN_AGE_SEC = 30.0


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, run_dir: Path, address: tuple[str, int], refresh_sec: int) -> None:
        self.run_dir = Path(run_dir)
        self.refresh_sec = int(refresh_sec)
        self.builder = ReportBuilder(self.run_dir, table_min_age_sec=TABLE_MIN_AGE_SEC)
        self.lock = threading.Lock()          # ReportBuilder is not thread-safe
        super().__init__(address, _Handler)

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            with self.lock:
                self.builder.close()


class _Handler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        url = urlsplit(self.path)
        try:
            if url.path in ("/", "/index.html"):
                with self.server.lock:
                    page = render_html(self.server.builder.build(), live=True, refresh_sec=self.server.refresh_sec)
                self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))
            elif url.path.startswith("/api/") and url.path[5:] in SECTIONS:
                name = url.path[5:]
                query = parse_qs(url.query)
                filters = {k: query[k][-1] for k in FILTERS.get(name, ()) if k in query}
                with self.server.lock:
                    body = self.server.builder.section(name, **filters)
                self._send(200, "application/json", json.dumps(body, allow_nan=False, default=str).encode("utf-8"))
            else:
                self._send(404, "application/json", json.dumps({"error": f"not found: {url.path}",
                                                                "sections": list(SECTIONS)}).encode("utf-8"))
        except Exception as e:  # noqa: BLE001 - one bad request never stops the dashboard
            logger.exception(f"dashboard: {url.path} failed")
            self._send(500, "application/json", json.dumps({"error": str(e)}).encode("utf-8"))

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        logger.debug("dashboard %s - " + fmt, self.address_string(), *args)


def make_server(run_dir: str | Path, *, host: str = "127.0.0.1", port: int = 8765,
                refresh_sec: int = 10) -> DashboardServer:
    """A bound, not yet serving dashboard; ``port=0`` picks a free port (``server.port``)."""
    run_dir = Path(run_dir)
    if not is_run_dir(run_dir):
        raise ConfigError(f"{run_dir} is not a run directory (no metrics.jsonl or config.resolved.yaml)")
    try:
        return DashboardServer(run_dir, (host, int(port)), refresh_sec)
    except OSError as e:
        raise ConfigError(f"cannot listen on {host}:{port}: {e}; choose another --port") from None


def serve(run_dir: str | Path, *, host: str = "127.0.0.1", port: int = 8765, refresh_sec: int = 10) -> None:
    """Serve until KeyboardInterrupt (Ctrl+C), which propagates after the socket is closed."""
    server = make_server(run_dir, host=host, port=port, refresh_sec=refresh_sec)
    try:
        server.serve_forever()
    finally:
        server.server_close()
```

`make_server` uses `render.is_run_dir`, so `report` and `dashboard` agree on what a run dir is.

- [ ] **Step 3: The `dashboard` command**

In `src/colosseum/cli.py`, after the `report` command, add:

```python
@main.command("dashboard")
@click.argument("run", type=click.Path(exists=True, file_okay=False))
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Address to listen on. No authentication: keep 127.0.0.1 and use an SSH tunnel from other machines")
@click.option("--port", default=8765, type=click.IntRange(0, 65535), show_default=True,
              help="Port (0: any free port, printed at start)")
@click.option("--refresh", default=10, type=click.IntRange(min=1), show_default=True,
              help="Seconds between page refreshes")
def dashboard_cmd(run: str, host: str, port: int, refresh: int) -> None:
    """Serve the run's report live: the same page as 'colosseum report', refreshed from /api/<section> JSON.
    The match log is opened read-only; ratings come from ratings.json while the run trains.

    Exit code: 1 not a run directory or the port is taken, 2 bad command-line arguments, 130 Ctrl+C.
    """
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    with _config_errors():
        from colosseum.arena.report.server import make_server

        server = make_server(run, host=host, port=port, refresh_sec=refresh)
    click.echo(f"Dashboard of {run} at http://{host}:{server.port}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    finally:
        server.server_close()        # a KeyboardInterrupt propagates: "Interrupted", exit code 130
```

- [ ] **Step 4: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp4_dashboard.py -v`

Expected: 5 passed in well under 20 s each.

- [ ] **Step 5: Full fast suite and ruff**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `env -u PYTHONPATH .venv/bin/ruff check .`

Expected: all pass, no warnings; `All checks passed!`.

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/arena/report/server.py src/colosseum/cli.py tests/integration/test_sp4_dashboard.py
git commit -m "feat: colosseum dashboard serves the report live with per-section JSON"
git push
```

---

### Task T8.1: Distributed guards, the benchmark config and the benchmark's counter clock

Spec block 8 (distributed roles in SP4: no match results, no journal (documented); `evaluation.enabled` and `checkpoint.keep_top_k > 0` → `ConfigError` naming SP5 in `distributed.check_distributed_scope`; `max_snapshots` works in the distributed learner as FIFO), criterion 3.9, section 6 («распределённый режим»; «Бенчмарк: изменения схемы конфига обновляют `_make_config` … скорость считается по счётчикам»), criterion 3.6 («Если часы машины скачут (WSL2), замер идёт по счётчикам, а не по окнам `ts`»).

T4.1 (Part 2) already refuses `keep_top_k > 0` and gives `distributed_checkpoint_manager(config, base_dir, run_id=None)` the `max_snapshots` cap with FIFO order; T6.1 may already refuse `evaluation.enabled`. This task makes the SP4 scope complete and tested in one place, says once per distributed role that no match log is written, and pins the benchmark workload.

Throughput by counters (decision D10 in the Contract notes): `scripts/bench_throughput.py` gets a `--clock counters` mode (the new default): a `CounterSampler` thread in the benchmark process reads, every 0.25 s of `time.monotonic()`, the launcher's global env-step counter (`Launcher.env_steps_done`, `launcher.py:377`) and the newest `train` / `system` records of `metrics.jsonl` (for train steps and queue depth). The samples have the shapes of `load_samples`, so the window checks and rates (`validate_window`, `compute_rates`) are unchanged. `--clock ts` keeps the pre-SP4 behaviour. `--tree PATH` measures another checkout (a `main` worktree) with that tree's own `_make_config` (whose schema has no SP4 sections) while the measuring code is the branch's script; `PYTHONPATH=PATH/src` makes the tree's `colosseum` the imported one, and the script refuses otherwise. Checked while writing this plan on the reference machine (where `--clock ts` fails every run): `--workers 1 2 --duration 15 --warmup 5` gave complete windows (3621 / 7196 env steps/s, learner-bound).

**Files:**
- Modify: `src/colosseum/distributed.py` (`check_distributed_scope`: SP4 items; the INFO; the module docstring's scope paragraph)
- Modify: `scripts/bench_throughput.py` (`_make_config`: every SP4 section explicit; `CounterSampler`, `CLOCKS`, `SAMPLE_INTERVAL_S`, `load_tree_config`, `run_one(..., clock=, make_config=)`, `--clock`, `--tree`, the JSON's `clock` / `tree`)
- Modify: `tests/unit/test_bench_throughput.py`
- Test: `tests/unit/test_sp4_distributed_guards.py`

**Interfaces:**
- Consumes: `check_distributed_scope`, `distributed_checkpoint_manager(config, base_dir, run_id=None)` (SP3 T6.1, Part 2 T4.1); `MatchLogConfig`, `RatingsConfig`, `EvaluationConfig`, `CheckpointConfig.keep_top_k / top_k_quantile / top_k_min_games / max_snapshots` (T1.1, T2.2, T4.1, T6.1); `Launcher.env_steps_done`, `Launcher._stop_event`.
- Produces: the SP4 refusals of `check_distributed_scope`; `bench_throughput.CounterSampler(env_steps_fn, metrics_path, agent_id=AGENT_ID, interval=SAMPLE_INTERVAL_S, clock=time.monotonic)` with `start / stop / sample` and `train` / `system` lists, `CLOCKS`, `SAMPLE_INTERVAL_S`, `load_tree_config(tree)`, `run_one(num_workers, duration, warmup, *, clock="counters", make_config=None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp4_distributed_guards.py`:

```python
"""Distributed roles and SP4 (spec block 8, criterion 3.9; T8.1): pulse evaluation and top-k are refused
with a ConfigError naming SP5; the journal and ratings sections are accepted (distributed roles write no
match log: one INFO when it is enabled); ``max_snapshots`` caps the distributed learner's storage FIFO."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from colosseum.core.errors import ConfigError
from colosseum.distributed import check_distributed_scope, distributed_checkpoint_manager
from game_helpers import make_test_config, scripted_agent


@pytest.mark.parametrize(("sections", "needle"), [
    ({"evaluation": {"enabled": True},
      "agents": {"agent_0": {}, "bot": scripted_agent()}}, "evaluation.enabled"),
    ({"checkpoint": {"keep_top_k": 3}}, "keep_top_k"),
], ids=["pulse", "top-k"])
def test_sp4_settings_are_refused_with_a_pointer_to_sp5(sections, needle):
    config = make_test_config("turns", **sections)
    with pytest.raises(ConfigError, match="SP5") as info:
        check_distributed_scope(config)
    assert needle in str(info.value)


def test_journal_and_ratings_sections_are_accepted_and_the_missing_journal_is_noted(caplog):
    config = make_test_config("turns", match_log={"detail_window_mb": 64.0}, ratings={"refit_interval_sec": 5.0},
                              checkpoint={"max_snapshots": 7})
    with caplog.at_level(logging.INFO, logger="colosseum.distributed"):
        check_distributed_scope(config)
    notes = [r.getMessage() for r in caplog.records if "match log" in r.getMessage()]
    assert len(notes) == 1 and "SP5" in notes[0]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="colosseum.distributed"):
        check_distributed_scope(make_test_config("turns", match_log={"enabled": False}))
    assert not [r for r in caplog.records if "match log" in r.getMessage()]


def test_max_snapshots_caps_the_distributed_learner_oldest_keep_every_first(tmp_path):
    config = make_test_config("turns", checkpoint={"interval": 10, "keep_last": 2, "keep_every": 3,
                                                   "max_snapshots": 3})
    manager = distributed_checkpoint_manager(config, tmp_path / "checkpoints")
    state = {"w": np.zeros(2, np.float32)}
    for version in range(10, 101, 10):
        manager.save("agent_0", version, state, trainer_state={"step": version})
    kept = sorted(int(p.name.removeprefix("ckpt_v")) for p in (tmp_path / "checkpoints" / "agent_0").glob("ckpt_v*"))
    # keep_every: 30 / 60 / 90, newest: 90 / 100 -> 4 > 3, the oldest keep_every-only snapshot goes (FIFO)
    assert kept == [60, 90, 100]
```

(If T4.1 already has an equivalent FIFO test, keep both: this one goes through the config the learner uses.)

Append to `tests/unit/test_bench_throughput.py`:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_config_sets_every_section_and_every_sp4_key_explicitly(tmp_path):
    """Guard: a new config key never changes the pinned workload silently (SP1 T0.4 rule)."""
    from colosseum.core.config import ColosseumConfig

    cfg = bench._make_config(1, str(tmp_path))
    assert set(ColosseumConfig.model_fields) - cfg.model_fields_set <= {"agents", "bc"}
    for name in ("match_log", "ratings", "evaluation", "checkpoint"):
        section = getattr(cfg, name)
        assert section.model_fields_set == set(type(section).model_fields), name


def test_benchmark_workload_has_the_journal_on_and_pulse_and_top_k_off(tmp_path):
    """Criterion 3.6: throughput with the journal on and pulse off (and no top-k: no snapshot exists)."""
    cfg = bench._make_config(1, str(tmp_path))
    assert cfg.match_log.enabled and not cfg.evaluation.enabled and cfg.checkpoint.keep_top_k == 0


def test_counter_sampler_reads_the_counters_on_its_own_clock(tmp_path):
    from colosseum.metrics.jsonl import MetricsWriter

    path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(path)
    clock, env_steps = iter([1.0, 2.0, 3.0]), iter([100, 200, 300])
    sampler = bench.CounterSampler(lambda: next(env_steps), path, clock=lambda: next(clock))
    writer.write("train", agent="agent_0", train_step=1, chunks_received=8.0)
    writer.write("train", agent="other", train_step=9, chunks_received=8.0)
    writer.write("system", env_steps=0, env_steps_per_sec=0.0, train_steps_per_sec={}, queue_depths={"agent_0": 7},
                 parked_buffers=0, dropped_reward_episodes=0, workers_reporting=1)
    sampler.sample()
    sampler.sample()                                     # no new train step: no new train sample
    writer.write("train", agent="agent_0", train_step=2, chunks_received=16.0)
    writer._fh.write('{"ts": 1, "kind": "tra')          # a partial line waits
    writer._fh.flush()
    sampler.sample()
    writer.close()
    assert sampler.train == [(1.0, 1, 8), (3.0, 2, 16)]
    assert sampler.system == [(1.0, 100, 7), (2.0, 200, 7), (3.0, 300, 7)]
    rates = bench.compute_rates(sampler.train, sampler.system, 0.0, 10.0)
    assert rates["env_steps_per_s"] == pytest.approx(100.0)


def test_tree_config_needs_the_tree_on_the_python_path(tmp_path):
    assert callable(bench.load_tree_config(REPO_ROOT))   # this checkout's colosseum is the imported one
    with pytest.raises(SystemExit, match="PYTHONPATH"):
        bench.load_tree_config(tmp_path)
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_distributed_guards.py tests/unit/test_bench_throughput.py -q`

Expected: failures for the parts not done yet (`evaluation.enabled` not refused if T6.1 did not do it; no INFO; `CounterSampler` / `load_tree_config` missing; and, if some earlier schema task forgot the bench rule, the explicit-section guard).

- [ ] **Step 2: The distributed scope**

In `src/colosseum/distributed.py::check_distributed_scope` (`distributed.py:187-223`), next to the existing problems (keep T4.1's `keep_top_k` refusal if it is there; do not add a second one):

```python
    if config.evaluation.enabled:
        problems.append("evaluation.enabled (pulse evaluation runs in the training main process next to the match "
                        "log; across machines with SP5)")
    if config.checkpoint.keep_top_k > 0:
        problems.append(f"checkpoint.keep_top_k={config.checkpoint.keep_top_k} (top-k needs the match log and live "
                        f"ratings, which distributed roles do not have until SP5; checkpoint.max_snapshots works)")
```

and after the `if problems: raise ...` line:

```python
    if config.match_log.enabled:
        logger.info("distributed roles write no match log and no Bradley-Terry ratings (SP5: the hub); "
                    "match_log / ratings settings have no effect here, max_snapshots caps snapshots FIFO")
```

Extend the module docstring's scope paragraph with: "SP4: no match results reach a journal (no `matches.db`, no ratings); `evaluation.enabled` and `checkpoint.keep_top_k > 0` are refused (SP5); `checkpoint.max_snapshots` caps the learner's snapshots oldest-first among the `keep_every`-only ones."

- [ ] **Step 3: The benchmark**

In `scripts/bench_throughput.py::_make_config` (`bench_throughput.py:89-170`), give every SP4 section and key explicitly (values = the defaults; skip any a schema task already wrote, keep its values if they equal these):

```python
        checkpoint={
            "interval": 10**9,  # no checkpoints: every seat collects
            "keep_last": 10,
            "keep_every": 0,
            "save_optimizer": True,
            "keep_top_k": 0,
            "top_k_quantile": 0.1,
            "top_k_min_games": 200,
            "max_snapshots": 100,
        },
        match_log={"enabled": True, "detail_window_mb": 512.0, "min_free_disk_mb": 2048.0,
                   "compact_interval_sec": 60.0},   # criterion 3.6: the journal is on
        ratings={"anchor": None, "refit_interval_sec": 60.0, "bootstrap_samples": 200},
        evaluation={"enabled": False, "agents": None, "field": None, "layouts": None, "every": 1, "blocks": 25,
                    "num_envs": 8, "deterministic": False, "max_backlog": 4},   # criterion 3.6: pulse off
```

(If T6.1 named an `EvaluationConfig` field differently, use its names; the guard test compares with the model's fields.)

After `LEARNER_BOUND_QUEUE_FRACTION = 0.5` add:

```python
CLOCKS = ("counters", "ts")
# Counter clock: how often the sampler reads the env-step counter and metrics.jsonl (seconds of time.monotonic()).
SAMPLE_INTERVAL_S = 0.25


class CounterSampler:
    """Measurement by counters on ``time.monotonic()`` (GPU checks report §8: WSL2 wall clocks jump by ~9 s,
    so windows over the records' ``ts`` fail). Every ``interval`` seconds it records:

    - ``system``: ``(t, env_steps, queue_depth)``: the launcher's global env-step counter and the learner
      queue depth of the newest ``system`` record (-1 = unknown);
    - ``train``: ``(t, train_step, chunks_received)`` of the newest ``train`` record of ``agent_id``, once
      per new train step.

    The lists have the shapes ``load_samples`` returns, so ``validate_window`` / ``compute_rates`` apply.
    """

    def __init__(self, env_steps_fn, metrics_path: str | Path, agent_id: str = AGENT_ID,
                 interval: float = SAMPLE_INTERVAL_S, clock=time.monotonic) -> None:
        self.train: list[tuple[float, int, int]] = []
        self.system: list[tuple[float, int, int]] = []
        self._env_steps_fn = env_steps_fn
        self._path = Path(metrics_path)
        self._agent_id = agent_id
        self._interval = float(interval)
        self._clock = clock
        self._offset = 0
        self._last_train: tuple[int, int] | None = None
        self._depth = -1
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bench-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def sample(self) -> None:
        now = self._clock()
        self._read_metrics()
        self.system.append((now, int(self._env_steps_fn()), self._depth))
        if self._last_train is not None and (not self.train or self.train[-1][1] != self._last_train[0]):
            self.train.append((now, *self._last_train))

    def _read_metrics(self) -> None:
        """New complete lines of metrics.jsonl (a partial last line waits for the next sample)."""
        try:
            with self._path.open("rb") as fh:
                fh.seek(self._offset)
                chunk = fh.read()
        except OSError:
            return
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        self._offset += end + 1
        for line in chunk[:end].split(b"\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["kind"] == "train" and r["agent"] == self._agent_id:
                self._last_train = (int(r["train_step"]), int(r.get("chunks_received") or 0))
            elif r["kind"] == "system":
                self._depth = int(r["queue_depths"].get(self._agent_id, -1))

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.sample()


def load_tree_config(tree: str | Path):
    """``_make_config`` of the benchmark script of another checkout (``--tree``, e.g. a ``main`` worktree),
    whose ``src`` must be the ``colosseum`` this process imported (``PYTHONPATH=<tree>/src``)."""
    import importlib.util

    import colosseum

    tree = Path(tree).resolve()
    if not Path(colosseum.__file__).resolve().is_relative_to(tree / "src"):
        raise SystemExit(f"--tree {tree}: colosseum is imported from {Path(colosseum.__file__).parent}; run with "
                         f"PYTHONPATH={tree / 'src'} so the measured code is the tree's")
    spec = importlib.util.spec_from_file_location("bench_throughput_tree", tree / "scripts" / "bench_throughput.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))         # the tree's examples.* for this process and spawned children
    return module._make_config
```

Replace `run_one` with:

```python
def run_one(num_workers: int, duration: float, warmup: float, *, clock: str = "counters",
            make_config=None) -> dict:
    """Run one configuration; raises RuntimeError if the window is incomplete."""
    from colosseum.core.run_dir import RunDir
    from colosseum.launcher import Launcher

    make_config = make_config or _make_config
    with tempfile.TemporaryDirectory(prefix="bench-") as tmp:
        config = make_config(num_workers, tmp)
        run_dir = RunDir.create(config)
        launcher = Launcher(config, run_dir)
        sampler = CounterSampler(lambda: launcher.env_steps_done, run_dir.metrics_path) if clock == "counters" else None
        started = time.monotonic() if sampler is not None else time.time()  # ts mode: metrics.jsonl ``ts`` is Unix time
        timer = threading.Timer(warmup + duration, launcher._stop_event.set)
        timer.start()
        if sampler is not None:
            sampler.start()
        try:
            launcher.launch()
        finally:
            timer.cancel()
            if sampler is not None:
                sampler.stop()
        train, system = (sampler.train, sampler.system) if sampler is not None else load_samples(run_dir.metrics_path)
    start, end = started + warmup, started + warmup + duration
    error = validate_window(train, start, end, "learner") or validate_window(system, start, end, "system")
    if error is not None:
        raise RuntimeError(error)
    result = compute_rates(train, system, start, end)
    result["workers"] = num_workers
    return result
```

In the module docstring, after "Checkpoints are disabled, so every slot plays the latest policy and collects data.", add:

```text
Clocks (``--clock``): ``counters`` (default since SP4) samples the launcher's global env-step counter and
the newest ``train`` / ``system`` records on ``time.monotonic()`` in this process (``CounterSampler``);
``ts`` uses the records' wall-clock ``ts`` as before SP4. On machines whose wall clock jumps (WSL2: GPU
checks report §8) only ``counters`` works. ``--tree PATH`` measures another checkout (for example a ``main``
worktree) with its own ``_make_config``; run it with ``PYTHONPATH=PATH/src``.
```

In `main`: after `parser.add_argument("--json", ...)` add

```python
    parser.add_argument("--clock", choices=CLOCKS, default="counters",
                        help="counters: time.monotonic() samples of the counters (default); ts: records' ts")
    parser.add_argument("--tree", type=str, default=None,
                        help="measure another checkout with its own _make_config (run with PYTHONPATH=<tree>/src)")
```

then `make_config = load_tree_config(args.tree) if args.tree else _make_config` after `parse_args()`; print `f"clock: {args.clock}; tree: {args.tree or REPO_ROOT}"` after the workload line; call `run_one(n, args.duration, args.warmup, clock=args.clock, make_config=make_config)`; and write `{"machine", "workload", "clock": args.clock, "tree": str(args.tree or REPO_ROOT), "results"}` to `--json`.

- [ ] **Step 4: Run the tests and a short real measurement**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_distributed_guards.py tests/unit/test_bench_throughput.py tests/unit/test_sp3_distributed_guards.py tests/unit/test_sp2_distributed_roles.py tests/integration/test_sp2_distributed_e2e.py -q`

Expected: all pass (the SP2/SP3 distributed tests stay green: their configs leave `evaluation` off and `keep_top_k` at 0).

Then: `env -u PYTHONPATH .venv/bin/python scripts/bench_throughput.py --workers 1 --duration 15 --warmup 5`

Expected: one row, no "invalid measurement" (the counter clock works where `ts` windows fail).

- [ ] **Step 5: Full fast suite and ruff**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `env -u PYTHONPATH .venv/bin/ruff check .`

Expected: all pass, no warnings; `All checks passed!`.

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/distributed.py scripts/bench_throughput.py tests/unit/test_bench_throughput.py \
  tests/unit/test_sp4_distributed_guards.py
git commit -m "feat: distributed roles refuse pulse and top-k; benchmark pins SP4 sections and measures by counters"
git push
```

---

### Task T8.2: Ladder bots and the measurements behind the SP4 rulings

Criterion 3.3 (the ladder: one bot with a share of random moves, so the true order is known; «Число игр и допуски замеряются в плане и фиксируются как rulings»), block 3 («Размеры по умолчанию (`--blocks`, начальный блок `halving`, `--confirm-blocks`) замеряются в плане на «лесенке ботов»»), block 2 («Сила приора и гладкости — константы модуля, выбираются в плане по кросс-валидации на журналах демо-игр»), §8 risks («Подгонка BT медленная на тысячах сущностей … замер в плане»), criterion 3.4 (`evaluation.every` of the pipeline test, the trend threshold), criterion 3.6 (throughput against `main`, journal on, pulse off, by counters).

**Ladders (decision D11).** tic_tac_toe (1v1, layout `2p`): `EpsilonTicTacToeBot(epsilon)` = SP3's `TicTacToeBot` (win, block, center, random) that plays a uniformly random legal cell with probability `epsilon`. tron (FFA, layout `3p`): `FloodFillTronBot(epsilon)` takes, among straight / left / right, the move whose cell is free and opens the largest free area of its view window (flood fill; other heads count as blocked), ties broken by the episode RNG; with probability `epsilon` a uniformly random action. Candidates `e000`, `e025`, `e050`, `e100` (epsilon 0, 0.25, 0.5, 1); field `random` (`RandomBot`) and `ref` (the same bot at epsilon 0.1), never a candidate. Measured while writing this plan on the real `MatchRunner` (`play_lineups`, seat rotation; mean pairwise score against the field): tic_tac_toe 0.742 / 0.620 / 0.532 / 0.315 (400 games each), tron 3p 0.872 / 0.552 / 0.435 / 0.326 (300 games each); 2 800 games in about 5 s. A pure-numpy probe over 1 000 blocks per pair gave the same order with per-block SD 0.10–0.25.

**Files:**
- Create: `examples/tic_tac_toe/ladder.py`, `examples/tron/bots.py`
- Create: `tests/arena_kit.py` (support module); modify `pyproject.toml` (`known-first-party` += `arena_kit`)
- Create: `scripts/measure_tournament_defaults.py`, `scripts/bt_crossval.py`, `scripts/measure_bt_cost.py`, `scripts/measure_pulse_cadence.py`
- Test: `tests/unit/test_sp4_ladder_bots.py`, `tests/unit/test_sp4_measure_rules.py`, `tests/integration/test_sp4_ladder_smoke.py`
- Modify (with the measured values; rulings): `src/colosseum/arena/tournament/modes.py` (`DEFAULT_BLOCKS`, `DEFAULT_HALVING_BLOCKS`, `DEFAULT_CONFIRM_BLOCKS`), `src/colosseum/arena/bt.py` (`PRIOR_SIGMA`, `SMOOTHNESS`; R-P1-8), their docstrings ("provisional" → the ruling)
- Create: `docs/benchmarks/sp4-tournament-defaults.json`, `docs/benchmarks/sp4-bt-crossval.json`, `docs/benchmarks/sp4-bt-cost.json`, `docs/benchmarks/sp4-pulse-cadence.json`, `docs/benchmarks/sp4-pipeline-calibration.json`, `docs/benchmarks/sp4-throughput/{main-1,branch-1,main-2,branch-2}.json`; modify `docs/benchmarks.md`

**Interfaces:**
- Consumes: `colosseum tournament` (T5.4; R-P2-1 report shape, R-P2-3 sizes and flags), `TicTacToeBot` (`examples/tic_tac_toe/bots.py`), `TronGame` (`examples/tron/game.py`), `ScriptedBot`, `pipeline_kit.cli / write_config / pipeline_config / train / evaluate / UNIT_HARVEST / MAIN / RANDOM` (`tests/pipeline_kit.py`), `MatchLog.open_readonly`, `MatchRow.to_dict / to_result` (R-P1-1), `ComparisonTable.add_result`, `match_pairs` (T1.2), `bt.fit / laplace / bootstrap`, `BTOptions`, `BootstrapData.from_table` (R-P1-3), `live.rating_inputs` (R-P1-5), `build_report_data` (T7.1), the `system` counters `pulse_skipped_snapshots` / `pulse_evaluated_snapshots` (T6.1; R-P2-4, contract amendment A29), `scripts/bench_throughput.py --clock counters --tree` (T8.1).
- Produces: `examples.tic_tac_toe.ladder.EpsilonTicTacToeBot(epsilon=0.0)`, `examples.tron.bots.FloodFillTronBot(epsilon=0.0)`, `examples.tron.bots.free_area`, `examples.tron.bots.MOVES`; `arena_kit`: `EPSILONS`, `FIELD_EPSILON`, `FIELD`, `candidate_name`, `TRUE_ORDER`, `LadderGame`, `TIC_TAC_TOE`, `TRON`, `LADDER_GAMES`, `ladder_config`, `find_report`, `group_of`, `SelectRun`, `run_select`, `ladder_problems`, `PulseSettings`, `Trend`, `spearman`, `rating_trend`, `PulseRun`, `train_with_pulse`, `select_snapshots`; the four scripts with `decide`, `choose`, `decide_cadence`, `decide_trend`; the rulings.

- [ ] **Step 1: Check that the names are free**

```bash
ls examples/tic_tac_toe/ladder.py examples/tron/bots.py tests/arena_kit.py scripts/measure_tournament_defaults.py \
  scripts/bt_crossval.py scripts/measure_bt_cost.py scripts/measure_pulse_cadence.py 2>/dev/null
grep -n "DEFAULT_BLOCKS\|DEFAULT_HALVING_BLOCKS\|DEFAULT_CONFIRM_BLOCKS\|--blocks\|--confirm-blocks\|--halving" \
  src/colosseum/arena/tournament/modes.py src/colosseum/cli.py | head
grep -n "PRIOR_SIGMA\|SMOOTHNESS\|class BootstrapData\|def from_table\|pulse_skipped_snapshots\|pulse_evaluated_snapshots" -r src/colosseum | head
```

Expected: the first command prints nothing; the greps show the landed names of R-P1-3, R-P1-8, R-P2-3, R-P2-4 (use them if they differ).

- [ ] **Step 2: The bots and their tests**

Create `examples/tic_tac_toe/ladder.py`:

```python
"""Tic-tac-toe ladder bots of known strength (SP4 criterion 3.3): ``TicTacToeBot`` that plays a uniformly
random legal cell with probability ``epsilon``. A larger epsilon is weaker against any fixed field
(measured in docs/benchmarks.md, «Лесенка ботов»), so a tournament must rank them by epsilon."""
from __future__ import annotations

from typing import Any

import numpy as np

from examples.tic_tac_toe.bots import TicTacToeBot


class EpsilonTicTacToeBot(TicTacToeBot):
    def __init__(self, epsilon: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = float(epsilon)

    def act(self, obs: Any, mask: Any, info: Any) -> int:
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            legal = (np.asarray(mask, dtype=bool).reshape(9) if mask is not None
                     else np.asarray(obs[2]).reshape(9) > 0.5)
            return int(self._rng.choice(np.flatnonzero(legal)))
        return super().act(obs, mask, info)
```

Create `examples/tron/bots.py`:

```python
"""A scripted Tron player (SP4 ladder, criterion 3.3): among straight / left / right it takes the move whose
cell is free and leads to the largest free area of its view window (flood fill; other heads count as
blocked), ties broken by the episode ``rng``; with probability ``epsilon`` it plays a uniformly random
action instead (``epsilon = 1``: a random player that often crashes)."""
from __future__ import annotations

from typing import Any

import numpy as np

from colosseum.players import ScriptedBot

# Moves in the rotated window (heading up, row 0): straight, turn left, turn right (TronGame's action order).
MOVES = ((-1, 0), (0, -1), (0, 1))


def free_area(blocked: np.ndarray, start: tuple[int, int]) -> int:
    """Cells reachable from ``start`` through free cells of ``blocked`` (4-neighbourhood); -1 if blocked."""
    h, w = blocked.shape
    if blocked[start]:
        return -1
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt not in seen and not blocked[nxt]:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen)


class FloodFillTronBot(ScriptedBot):
    def __init__(self, epsilon: float = 0.0) -> None:
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = float(epsilon)
        self._rng = np.random.default_rng()

    def reset(self, *, role: str, seat: int, layout: str, rng: np.random.Generator) -> None:
        self._rng = rng

    def act(self, obs: Any, mask: Any, info: Any) -> int:
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            return int(self._rng.integers(len(MOVES)))
        window = np.asarray(obs)
        blocked = (window[0] > 0) | (window[1] > 0)
        r = window.shape[1] // 2
        areas = np.array([free_area(blocked, (r + dr, r + dc)) for dr, dc in MOVES])
        return int(self._rng.choice(np.flatnonzero(areas == areas.max())))
```

Create `tests/unit/test_sp4_ladder_bots.py`:

```python
"""Ladder bots of known strength (SP4 criterion 3.3, T8.2): epsilon is validated, epsilon 0 is the base bot,
the tron bot avoids walls, and against a fixed field (RandomBot + the bot at epsilon 0.1) the mean pairwise
score falls with epsilon on the real MatchRunner. Bands: at the plan's measurement (tic_tac_toe 0.742 /
0.532 / 0.315, tron 3p 0.872 / 0.435 / 0.326 for epsilon 0 / 0.5 / 1 over 400 / 300 games) adjacent gaps
are >= 0.1 with a per-game sd <= 0.5, so 200 / 150 games give an SE of the gap <= 0.05: an inversion is
a > 2 SE event (and the observed gaps are 2-4 times larger)."""
from __future__ import annotations

import functools

import numpy as np
import pytest

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.eval import play_lineups
from colosseum.players.registry import BotSpec, make_bot
from colosseum.worker.match_runner import ScriptedPlayer
from examples.tic_tac_toe.bots import TicTacToeBot
from examples.tic_tac_toe.game import TicTacToeGame
from examples.tic_tac_toe.ladder import EpsilonTicTacToeBot
from examples.tron.bots import FloodFillTronBot, free_area
from examples.tron.game import TronGame

TTT = "examples.tic_tac_toe.ladder.EpsilonTicTacToeBot"
TRON = "examples.tron.bots.FloodFillTronBot"


def _player(spec, cls: str, **kwargs) -> ScriptedPlayer:
    return ScriptedPlayer(functools.partial(make_bot, BotSpec(cls, kwargs), spec))


@pytest.mark.parametrize("cls", [EpsilonTicTacToeBot, FloodFillTronBot])
@pytest.mark.parametrize("epsilon", [-0.1, 1.5])
def test_epsilon_outside_the_unit_interval_is_refused(cls, epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        cls(epsilon=epsilon)


def test_epsilon_zero_plays_exactly_the_base_bot():
    rng = np.random.default_rng(0)
    for _ in range(50):
        board = rng.choice([0, 1, 2], size=9, p=[0.5, 0.25, 0.25])
        if not (board == 0).any():
            continue
        obs = np.stack([board == 1, board == 2, board == 0]).reshape(3, 3, 3).astype(np.float32)
        base, ladder = TicTacToeBot(), EpsilonTicTacToeBot(epsilon=0.0)
        for bot in (base, ladder):
            bot.game_spec = TicTacToeGame.spec
            bot.reset(role="player", seat=0, layout="2p", rng=np.random.default_rng(7))
        assert ladder.act(obs, board == 0, None) == base.act(obs, board == 0, None)


def test_tron_bot_never_takes_a_blocked_move_when_a_free_one_exists():
    window = np.zeros((2, 7, 7), np.uint8)
    window[0, 2, 3] = 1                      # wall straight ahead (heading up)
    window[0, 3, 2] = 1                      # wall on the left
    bot = FloodFillTronBot()
    bot.reset(role="player", seat=0, layout="2p", rng=np.random.default_rng(0))
    assert all(bot.act(window, None, None) == 2 for _ in range(10))          # turn right
    assert free_area(window[0] > 0, (2, 3)) == -1


def _mean_score(results, agent: str) -> float:
    total = 0.0
    for r in results:
        ranks = {t.team: t.rank for t in r.teams}
        (mine,) = {s.team for s in r.seats if s.agent_id == agent}
        others = [rank for team, rank in ranks.items() if team != mine]
        total += np.mean([1.0 if ranks[mine] < o else 0.5 if ranks[mine] == o else 0.0 for o in others])
    return total / len(results)


@pytest.mark.parametrize(("env_cls", "cls", "layout", "games"), [(TicTacToeGame, TTT, "2p", 200),
                                                                  (TronGame, TRON, "3p", 150)], ids=["ttt", "tron"])
def test_strength_falls_with_epsilon_against_a_fixed_field(env_cls, cls, layout, games):
    spec = env_cls().spec
    scores = []
    for eps in (0.0, 0.5, 1.0):
        models = {"c": _player(spec, cls, epsilon=eps), "random": _player(spec, "colosseum.players.RandomBot"),
                  "ref": _player(spec, cls, epsilon=0.1)}
        seats = [SeatAssignment(a, "fixed", False) for a in ("c", "random", "ref")]
        if layout == "2p":
            lineups = [Lineup(layout, [seats[0], seats[1 + (m // 2) % 2]][:: 1 if m % 2 == 0 else -1])
                       for m in range(games)]
        else:
            lineups = [Lineup(layout, seats[k % 3:] + seats[:k % 3]) for k in range(games)]
        results = play_lineups(env_fn=env_cls, models=models, lineups=lineups, num_envs=8, seed=3)
        scores.append(_mean_score(results, "c"))
    assert scores[0] > scores[1] > scores[2], scores
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_ladder_bots.py -q`

Expected: 8 passed in about 3 s (run while writing this plan: 8 passed in 2.98 s).

- [ ] **Step 3: `tests/arena_kit.py`, the scripts and their rules**

Create `tests/arena_kit.py` (add `"arena_kit"` to `known-first-party`):

```python
"""SP4 selection helpers for measurements and slow tests (T8.2, T8.3), driving the real CLI.

Ladder (criterion 3.3): scripted bots of known strength - one bot with a share ``epsilon`` of random moves
(``EPSILONS``; larger is weaker against any fixed field, measured in docs/benchmarks.md) - on two demo games:
tic_tac_toe (1v1, layout ``2p``) and tron (FFA, layout ``3p``). The field is the config's ``random``
(``RandomBot``) and ``ref`` (the same bot with ``FIELD_EPSILON``), never a candidate.

Pulse pipeline (criterion 3.4): ``unit_harvest_league`` trained from scratch with pulse evaluation, the
rating trend of its snapshots, ``tournament --mode select`` over the run's snapshots with the anchors as
the field, and ``eval`` of the winner against ``random``.

Report shape: ``arena.tournament.modes.report_view`` of the tournament's ``report.json`` (contract amendment
A26): ``layouts.<layout>.groups.<group>`` with ``candidates`` (rows: ``name``, ``score``, ``score_ci``, ``bt``,
``bt_ci``, ``games``, ...), ``ranking`` (final order, best first) and ``select`` (``winner``, ``confirmed``,
``indistinguishable``).
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import pipeline_kit as kit
from cli_runner import REPO_ROOT, TrainRun

EPSILONS = (0.0, 0.25, 0.5, 1.0)
FIELD_EPSILON = 0.1
FIELD = ("random", "ref")
RANDOM_BOT_CLASS = "colosseum.players.RandomBot"


def candidate_name(epsilon: float) -> str:
    """0.25 -> ``e025``."""
    return f"e{round(epsilon * 100):03d}"


TRUE_ORDER = tuple(candidate_name(e) for e in EPSILONS)     # strongest first


@dataclass(frozen=True)
class LadderGame:
    name: str
    config: str            # configs/examples/<config>.yaml
    bot_class: str         # ScriptedBot with an ``epsilon`` kwarg
    layout: str


TIC_TAC_TOE = LadderGame("tic_tac_toe", "tic_tac_toe", "examples.tic_tac_toe.ladder.EpsilonTicTacToeBot", "2p")
TRON = LadderGame("tron", "tron", "examples.tron.bots.FloodFillTronBot", "3p")
LADDER_GAMES = (TIC_TAC_TOE, TRON)


def ladder_config(game: LadderGame) -> dict:
    """The example config with the ladder agents only (candidates ``e000`` ... ``e100``, field ``random``, ``ref``)."""
    raw = yaml.safe_load((REPO_ROOT / "configs" / "examples" / f"{game.config}.yaml").read_text())
    agents: dict[str, Any] = {candidate_name(e): {"kind": "scripted", "class": game.bot_class, "kwargs": {"epsilon": e}}
                              for e in EPSILONS}
    agents["random"] = {"kind": "scripted", "class": RANDOM_BOT_CLASS}
    agents["ref"] = {"kind": "scripted", "class": game.bot_class, "kwargs": {"epsilon": FIELD_EPSILON}}
    raw["agents"] = agents
    return raw


def find_report(out: Path) -> dict:
    """The tournament's ``report.json`` somewhere under ``out`` (``--output`` dir or ``<run>/tournaments/<name>``)."""
    from colosseum.arena.tournament.modes import report_view

    found = sorted(Path(out).rglob("report.json"))
    assert found, f"no report.json under {out}"
    return report_view(json.loads(found[0].read_text()))


def group_of(report: Mapping, layout: str) -> dict:
    """The only role group of ``layout`` (the ladders are symmetric games)."""
    groups = report["layouts"][layout]["groups"]
    assert len(groups) == 1, f"expected one role group in {layout}, got {list(groups)}"
    return next(iter(groups.values()))


@dataclass
class SelectRun:
    report: dict
    group: dict
    seconds: float
    out: Path

    @property
    def ranking(self) -> list[str]:
        return list(self.group["ranking"])

    @property
    def winner(self) -> str:
        return self.group["select"]["winner"]

    @property
    def confirmed(self) -> bool:
        return bool(self.group["select"]["confirmed"])

    @property
    def rows(self) -> dict[str, dict]:
        return {row["name"]: row for row in self.group["candidates"]}


def run_select(game: LadderGame, workdir: Path, *, seed: int, blocks: int | None = None,
               confirm_blocks: int | None = None, num_workers: int = 2, num_envs: int = 8,
               timeout: float = 900.0, in_process: bool = False) -> SelectRun:
    """``colosseum tournament --mode select`` of the ladder candidates against the field (defaults unless
    ``blocks`` / ``confirm_blocks`` are given), into a fresh ``--output`` dir under ``workdir``."""
    workdir = Path(workdir)
    config = kit.write_config(workdir / f"ladder-{game.name}.yaml", ladder_config(game))
    out = workdir / f"select-{game.name}-s{seed}-b{blocks}-c{confirm_blocks}"
    args: list[Any] = ["tournament", "-c", config, "--mode", "select", "--layout", game.layout,
                       "--field", ",".join(FIELD), "--seed", seed, "--num-workers", num_workers,
                       "--num-envs", num_envs, "--output", out]
    for name in TRUE_ORDER:
        args += ["-a", name]
    if blocks is not None:
        args += ["--halving-blocks", blocks]          # select: the first halving round's blocks per composition
    if confirm_blocks is not None:
        args += ["--confirm-blocks", confirm_blocks]
    start = time.monotonic()
    result = kit.cli(args, timeout=timeout, in_process=in_process)
    seconds = time.monotonic() - start
    assert result.code == 0, f"tournament failed ({result.code}):\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    report = find_report(out)
    return SelectRun(report, group_of(report, game.layout), seconds, out)


def ladder_problems(run: SelectRun) -> list[str]:
    """Criterion 3.3 checks (empty = pass): true order, the strongest selected and confirmed, BT intervals
    of neighbours not contradicting the order (the stronger's interval is not entirely below the weaker's)."""
    problems = []
    if run.ranking != list(TRUE_ORDER):
        problems.append(f"ranking {run.ranking} != true order {list(TRUE_ORDER)}")
    if run.winner != TRUE_ORDER[0]:
        problems.append(f"winner {run.winner} != {TRUE_ORDER[0]}")
    if not run.confirmed:
        problems.append(f"not confirmed; indistinguishable {run.group['select'].get('indistinguishable')}")
    rows = run.rows
    for stronger, weaker in zip(TRUE_ORDER, TRUE_ORDER[1:]):
        hi_s, lo_w = rows[stronger]["bt_ci"][1], rows[weaker]["bt_ci"][0]
        if hi_s < lo_w:
            problems.append(f"BT intervals reverse the order: {stronger} {rows[stronger]['bt_ci']} is below "
                            f"{weaker} {rows[weaker]['bt_ci']}")
    return problems


# ---------------- pulse pipeline (criterion 3.4) ----------------

@dataclass(frozen=True)
class PulseSettings:
    every: int                       # evaluation.every (T8.2 ruling)
    blocks: int                      # evaluation.blocks (T8.2 ruling)
    num_workers: int = 1             # rollout workers: + the pulse process = 2 processes (test rule)
    total_timesteps: int | None = None   # None: the config's budget


@dataclass
class Trend:
    spearman: float                  # rank correlation of BT rating with version
    gain: float                      # mean rating of the newest third minus the oldest third (Elo units)
    points: list[tuple[int, float]] = field(default_factory=list)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rho (average ranks for ties); nan with fewer than 3 points or a constant input."""
    if len(x) < 3:
        return float("nan")

    def ranks(v):
        v = np.asarray(v, dtype=np.float64)
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=np.float64)
        for value in np.unique(v):
            tie = v == value
            r[tie] = r[tie].mean()
        return r

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def rating_trend(report_data: Mapping, *, layout: str = "2p", agent: str = kit.MAIN) -> Trend:
    """BT rating by version of ``agent``'s snapshots in ``layout`` (report data, ratings section)."""
    groups = report_data["sections"]["ratings"]["layouts"][layout]["groups"]
    points = sorted({(int(p[0]), float(p[1])) for g in groups for s in g["series"] if s["lineage"] == agent
                     for p in s["points"]})
    versions, ratings = [p[0] for p in points], [p[1] for p in points]
    third = max(1, len(points) // 3)
    gain = float(np.mean(ratings[-third:]) - np.mean(ratings[:third])) if points else float("nan")
    return Trend(spearman(versions, ratings), gain, points)


@dataclass
class PulseRun:
    run: TrainRun
    config: Path
    data: dict                       # build_report_data of the run
    trend: Trend
    pulse_versions: list[int]        # snapshots with pulse matches in the journal
    stored_versions: list[int]       # snapshot dirs at the end
    seconds: dict[str, float]


def train_with_pulse(workdir: Path, *, seed: int, settings: PulseSettings, name: str = "pulse") -> PulseRun:
    """``unit_harvest_league`` from scratch (no init, no kickstart) with pulse evaluation against the anchors."""
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
    from colosseum.arena.report.data import build_report_data

    workdir = Path(workdir)
    config = kit.write_config(workdir / f"{name}.yaml", kit.pipeline_config(kit.UNIT_HARVEST))
    sets: dict[str, Any] = {"training.seed": seed, "rollout.num_workers": settings.num_workers,
                            "evaluation.enabled": True, "evaluation.every": settings.every,
                            "evaluation.blocks": settings.blocks}
    if settings.total_timesteps is not None:
        sets["training.total_timesteps"] = settings.total_timesteps
    start = time.monotonic()
    run = kit.train(config, workdir, name, sets, timeout=1500.0)
    seconds = {"train": time.monotonic() - start}
    start = time.monotonic()
    data = build_report_data(run.root)
    seconds["report_data"] = time.monotonic() - start
    reader = MatchLog.open_readonly(run.root / MATCH_LOG_FILE)
    try:
        pulse = sorted({v for row in reader.matches(source="pulse", limit=None)
                        for seat in row.to_dict()["seats"] if seat["agent_id"] == kit.MAIN
                        and (v := seat["rating_version"]) is not None})
    finally:
        reader.close()
    stored = sorted(int(m.group(1)) for d in (run.root / "checkpoints" / kit.MAIN).iterdir()
                    if (m := re.fullmatch(r"ckpt_v(\d+)", d.name)))
    return PulseRun(run, config, data, rating_trend(data), pulse, stored, seconds)


def select_snapshots(run: PulseRun, *, seed: int, name: str = "sel", num_workers: int = 2,
                     timeout: float = 1200.0) -> dict:
    """``tournament --run <run> --agent main --snapshots all --field greedy,random --mode select``; the report."""
    args = ["tournament", "-c", run.config, "--run", run.run.root, "--agent", kit.MAIN, "--snapshots", "all",
            "--field", f"{kit.UNIT_HARVEST.bot},{kit.RANDOM}", "--mode", "select", "--seed", seed,
            "--num-workers", num_workers, "--name", name]
    result = kit.cli(args, timeout=timeout)
    assert result.code == 0, f"tournament failed ({result.code}):\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    from colosseum.arena.tournament.modes import report_view

    return report_view(json.loads((run.run.root / "tournaments" / name / "report.json").read_text()))
```

Create `scripts/measure_tournament_defaults.py`:

```python
"""Tournament default sizes on the bot ladders (SP4 criterion 3.3; plan T8.2).

For every grid point (first halving round ``--halving-blocks``, ``--confirm-blocks``) and seed, ``tournament --mode
select`` runs on both ladders (``tests/arena_kit.py``: tic_tac_toe 2p, tron 3p; 2 worker processes). Each run
is checked by ``arena_kit.ladder_problems``.

Decision rule (fixed before the measurement, ``decide``): the first grid point (smallest first) at which, on
both games, the winner is right in every seed and the order, ``confirmed`` and the BT-neighbour check each
hold in all seeds but at most one. Defaults: ``DEFAULT_HALVING_BLOCKS = max(8, blocks)``,
``DEFAULT_CONFIRM_BLOCKS = max(100, confirm)``, ``DEFAULT_BLOCKS`` (gauntlet, round-robin) ``= max(50, 2 *
blocks)``; the floors are T5.3's provisional values. They keep real selections (differences far smaller
than the ladder's) from defaulting to tiny tournaments. No passing point: report to the controller (no
defaults are set).

Usage::

    env -u PYTHONPATH .venv/bin/python scripts/measure_tournament_defaults.py --seeds 0 1 2 3 4 5 6 7 8 9 \
        --work-dir /tmp/sp4-ladder --json docs/benchmarks/sp4-tournament-defaults.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

GRID = ((8, 32), (16, 64), (32, 128))      # (first halving round blocks, confirm blocks), smallest first
FLOORS = {"halving_blocks": 8, "confirm_blocks": 100, "blocks": 50}   # Part 2's provisional defaults
CHECKS = ("winner", "order", "confirmed", "bt")


def classify(problems: list[str]) -> dict[str, bool]:
    """Which checks of ``arena_kit.ladder_problems`` passed."""
    return {"winner": not any(p.startswith("winner") for p in problems),
            "order": not any(p.startswith("ranking") for p in problems),
            "confirmed": not any(p.startswith("not confirmed") for p in problems),
            "bt": not any(p.startswith("BT") for p in problems)}


def decide(rows: list[dict], grid=GRID, floors=FLOORS) -> dict:
    """``rows``: ``{"game", "blocks", "confirm", "seed", "checks": {check: bool}, "seconds"}``."""
    games = sorted({r["game"] for r in rows})
    for blocks, confirm in grid:
        cell = [r for r in rows if (r["blocks"], r["confirm"]) == (blocks, confirm)]
        ok, summary = True, {}
        for game in games:
            runs = [r for r in cell if r["game"] == game]
            passed = {c: sum(r["checks"][c] for r in runs) for c in CHECKS}
            n = len(runs)
            summary[game] = {"runs": n, **passed, "mean_seconds": sum(r["seconds"] for r in runs) / max(n, 1)}
            ok &= n > 0 and passed["winner"] == n and all(passed[c] >= n - 1 for c in ("order", "confirmed", "bt"))
        if ok:
            return {"chosen": [blocks, confirm], "summary": summary,
                    "defaults": {"halving_blocks": max(floors["halving_blocks"], blocks),
                                 "confirm_blocks": max(floors["confirm_blocks"], confirm),
                                 "blocks": max(floors["blocks"], 2 * blocks)}}
    return {"chosen": None, "summary": None, "defaults": None}


def main() -> int:
    import arena_kit as kit

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/sp4-ladder"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    rows = []
    for blocks, confirm in GRID:
        for game in kit.LADDER_GAMES:
            for seed in args.seeds:
                work = args.work_dir / f"{game.name}-b{blocks}-c{confirm}-s{seed}"
                work.mkdir(parents=True, exist_ok=True)
                run = kit.run_select(game, work, seed=seed, blocks=blocks, confirm_blocks=confirm)
                problems = kit.ladder_problems(run)
                rows.append({"game": game.name, "blocks": blocks, "confirm": confirm, "seed": seed,
                             "checks": classify(problems), "problems": problems, "seconds": round(run.seconds, 1),
                             "ranking": run.ranking, "winner": run.winner, "confirmed": run.confirmed})
                print(json.dumps(rows[-1]), flush=True)
    decision = decide(rows)
    per_cell = defaultdict(dict)
    for r in rows:
        key = f"b{r['blocks']}-c{r['confirm']}"
        cell = per_cell[key].setdefault(r["game"], {c: 0 for c in CHECKS} | {"runs": 0})
        cell["runs"] += 1
        for c in CHECKS:
            cell[c] += r["checks"][c]
    print(json.dumps(per_cell, indent=1))
    print(json.dumps(decision, indent=1))
    if args.json:
        args.json.write_text(json.dumps({"grid": GRID, "floors": FLOORS, "rows": rows, "decision": decision},
                                        indent=2) + "\n")
    return 0 if decision["chosen"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `scripts/bt_crossval.py`:

```python
"""Bradley-Terry prior and smoothness by cross-validation (SP4 spec block 2; plan T8.2).

Journals: short training runs of four demo games (``--make-journals``: tic_tac_toe, tron, unit_harvest_league,
predator_prey; the journal is on by default). Every match of a journal goes to fold ``sha1(uid) % FOLDS``;
for each fold and grid point the fit on the other folds predicts the held-out team pairs
(``match_pairs``: weight ``1/(T-1)``, outcome 1 / 0.5 / 0) and scores them by the weighted log-loss.
Pairs with an entity the training folds never saw are skipped (counted). Score layouts have no pairs.

Decision rule (fixed before the measurement, ``choose``): the grid point with the smallest mean held-out
log-loss (games weighted equally); among the points within one standard error (over folds) of it, the one
with the largest ``smoothness`` (the spec wants few-game snapshots to lean on their neighbours: the most
smoothing that predicts as well), then the largest ``prior_sigma`` (the weakest pull towards the anchor).

Usage::

    env -u PYTHONPATH .venv/bin/python scripts/bt_crossval.py --make-journals --work-dir /tmp/sp4-cv
    env -u PYTHONPATH .venv/bin/python scripts/bt_crossval.py --work-dir /tmp/sp4-cv \
        --json docs/benchmarks/sp4-bt-crossval.json
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GAMES = {    # game -> (example config, extra --set)
    "tic_tac_toe": ("tic_tac_toe", {}),
    "tron": ("tron", {}),
    "unit_harvest_league": ("unit_harvest_league", {}),
    "predator_prey": ("predator_prey", {}),
}
PRIOR_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)            # prior standard deviation of a strength, logits
SMOOTH_GRID = (0.0, 1.0, 4.0, 16.0, 64.0)         # random-walk precision between consecutive versions
FOLDS = 5
ELO = 400.0 / math.log(10.0)                      # Elo units per logit (BTFit ratings are in Elo units)
SOURCES = ("train", "pulse")


def fold_of(uid: str) -> int:
    return int(hashlib.sha1(uid.encode("utf-8")).hexdigest(), 16) % FOLDS


def make_journals(work_dir: Path, seed: int) -> None:
    for game, (config, extra) in GAMES.items():
        if (work_dir / "runs" / game / "matches.db").exists():
            print(f"{game}: journal exists", flush=True)
            continue
        path = REPO_ROOT / "configs" / "examples" / f"{config}.yaml"
        cmd = [sys.executable, "-m", "colosseum", "train", "-c", str(path),
               "--set", f"run.dir={work_dir / 'runs'}", "--set", f"run.name={game}", "--set", f"training.seed={seed}",
               "--set", "rollout.num_workers=2"]
        for key, value in extra.items():
            cmd += ["--set", f"{key}={value}"]
        start = time.monotonic()
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        print(f"{game}: trained in {time.monotonic() - start:.0f} s", flush=True)


def load(journal: Path):
    """``(rating inputs, [(fold, result, rating_versions)])`` of one run's journal (detail rows only)."""
    from colosseum.arena.live import rating_inputs
    from colosseum.arena.matchlog import MatchLog
    from colosseum.core.config import load_config

    config = load_config(journal.parent / "config.resolved.yaml")
    reader = MatchLog.open_readonly(journal)
    try:
        rows = reader.matches(limit=None)
        items = []
        for row in rows:
            result, versions = row.to_result()
            if result.source in SOURCES:
                items.append((fold_of(result.uid), result, versions))
    finally:
        reader.close()
    return rating_inputs(config), items


def predict(fit_, a, b) -> float | None:
    """P(team a beats team b) under the fit; None if a member was not rated."""
    def strength(team) -> float | None:
        ratings = [fit_.rating.get(e) for e, _ in team.members]
        if any(r is None for r in ratings):
            return None
        return (sum(ratings) / len(ratings) + fit_.bias.get(team.team, 0.0)) / ELO

    sa, sb = strength(a), strength(b)
    if sa is None or sb is None:
        return None
    return 1.0 / (1.0 + math.exp(-(sa - sb)))


def evaluate(inputs, items, prior: float, smooth: float) -> dict:
    """Held-out log-loss per fold: ``{"loss": [...], "weight": [...], "skipped": n}``."""
    from colosseum.arena import bt
    from colosseum.arena.comparisons import ComparisonTable, match_pairs

    options, fixed, random_bot = inputs
    options = dataclasses.replace(options, prior_sigma=prior, smoothness=smooth)
    losses, weights, skipped = [], [], 0
    for k in range(FOLDS):
        table = ComparisonTable()
        for fold, result, versions in items:
            if fold != k:
                table.add_result(result, versions)
        loss = weight = 0.0
        fits = {}
        for fold, result, versions in items:
            if fold != k or result.outcome_kind == "score" or result.layout not in table.layouts():
                continue
            if result.layout not in fits:
                fits[result.layout] = bt.fit(table, result.layout, sources=SOURCES, options=options,
                                             fixed_agents=fixed, random_bot=random_bot)
            for a, b, w, y in match_pairs(result, versions):
                p = predict(fits[result.layout], a, b)
                if p is None:
                    skipped += 1
                    continue
                p = min(max(p, 1e-9), 1 - 1e-9)
                loss -= w * (y * math.log(p) + (1 - y) * math.log(1 - p))
                weight += w
        losses.append(loss)
        weights.append(weight)
    return {"loss": losses, "weight": weights, "skipped": skipped}


def choose(rows: list[dict]) -> dict:
    """The decision rule (module docstring). ``rows``: ``{"game", "prior", "smooth", "loss": [per fold],
    "weight": [per fold]}``; returns ``{"prior", "smooth", "mean", "se", "best": (prior, smooth)}``."""
    games = sorted({r["game"] for r in rows})
    by_point: dict[tuple[float, float], list[float]] = defaultdict(lambda: [0.0] * FOLDS)
    for r in rows:
        for k in range(FOLDS):
            per_game = r["loss"][k] / r["weight"][k] if r["weight"][k] else 0.0
            by_point[(r["prior"], r["smooth"])][k] += per_game / len(games)
    stats = {}
    for point, folds in by_point.items():
        mean = sum(folds) / FOLDS
        sd = math.sqrt(sum((f - mean) ** 2 for f in folds) / (FOLDS - 1))
        stats[point] = (mean, sd / math.sqrt(FOLDS))
    best = min(stats, key=lambda p: stats[p][0])
    limit = stats[best][0] + stats[best][1]
    eligible = [p for p in stats if stats[p][0] <= limit]
    prior, smooth = max(eligible, key=lambda p: (p[1], p[0]))
    return {"prior": prior, "smooth": smooth, "mean": stats[(prior, smooth)][0], "se": stats[(prior, smooth)][1],
            "best": list(best), "best_mean": stats[best][0], "best_se": stats[best][1]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--make-journals", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.make_journals:
        make_journals(args.work_dir, args.seed)
    rows = []
    for game in GAMES:
        journal = args.work_dir / "runs" / game / "matches.db"
        inputs, items = load(journal)
        print(f"{game}: {len(items)} matches", flush=True)
        for prior in PRIOR_GRID:
            for smooth in SMOOTH_GRID:
                start = time.monotonic()
                res = evaluate(inputs, items, prior, smooth)
                rows.append({"game": game, "prior": prior, "smooth": smooth, **res,
                             "seconds": round(time.monotonic() - start, 1)})
                mean = sum(res["loss"]) / max(sum(res["weight"]), 1e-12)
                print(f"  prior {prior:>4} smooth {smooth:>5}: log-loss {mean:.5f} (skipped {res['skipped']})",
                      flush=True)
    decision = choose(rows)
    print(json.dumps(decision))
    if args.json:
        args.json.write_text(json.dumps({"folds": FOLDS, "rows": rows, "decision": decision}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `scripts/measure_bt_cost.py`:

```python
"""Cost of the Bradley-Terry fit at about 2 000 entities (SP4 spec §8 risk "fit slow on thousands of entities";
plan T8.2). One BLAS thread, as in the training main process (spec block 2).

Synthetic 1v1 layout: 20 lineages x 100 versions + 3 fixed agents (2 003 entities); every entity meets 10
random others (20 matches each pair, outcomes from known strengths); timings: cold ``fit``, warm ``fit``
after 1 % more data, ``laplace`` (1 000 samples, top-k 5) and ``bootstrap`` (``--bootstrap`` replicas).

Rule (fixed before the measurement): live refits are acceptable if cold fit <= 30 s, warm fit <= 5 s and
laplace <= 5 s on the reference machine (refit every 60 s in a background thread); report bootstrap is
informational. Otherwise the controller decides (iterative solver task, or a larger default refit interval).

Usage::

    env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
        scripts/measure_bt_cost.py --json docs/benchmarks/sp4-bt-cost.json
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LINEAGES, VERSIONS, FIXED, OPPONENTS, PER_PAIR = 20, 100, ("random", "greedy", "old"), 10, 20


def synthetic_table(seed: int, extra: float = 0.0):
    from colosseum.arena.comparisons import ComparisonTable, PairStat, TeamKey

    rng = np.random.default_rng(seed)
    entities = [f"a{i}@v{100 * v}" for i in range(LINEAGES) for v in range(VERSIONS)] + list(FIXED)
    strength = {e: (0.0 if e == "random" else rng.normal(0.0, 1.0)) for e in entities}
    table = ComparisonTable()
    n_pairs = int(len(entities) * OPPONENTS * (1.0 + extra))
    for _ in range(n_pairs):
        a, b = rng.choice(len(entities), size=2, replace=False)
        ea, eb = entities[a], entities[b]
        p = 1.0 / (1.0 + np.exp(-(strength[ea] - strength[eb])))
        wins = float(rng.binomial(PER_PAIR, p))
        table.add("1v1", "train", TeamKey(0, ((ea, "player"),)), TeamKey(1, ((eb, "player"),)),
                  PairStat(n=PER_PAIR, sw=float(PER_PAIR), swy=wins))
    return table, len(entities)


def main() -> int:
    from colosseum.arena import bt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    options = bt.BTOptions(prior_sigma=bt.PRIOR_SIGMA, smoothness=bt.SMOOTHNESS)
    out = {"blas_threads": os.environ.get("OMP_NUM_THREADS")}
    table, n = synthetic_table(0)
    out["entities"] = n
    t = time.monotonic()
    cold = bt.fit(table, "1v1", sources=("train",), options=options, fixed_agents=FIXED, random_bot="random")
    out["fit_cold_s"] = time.monotonic() - t
    out["iterations_cold"] = cold.iterations
    more, _ = synthetic_table(0, extra=0.01)
    t = time.monotonic()
    warm = bt.fit(more, "1v1", sources=("train",), options=options, fixed_agents=FIXED, random_bot="random", warm=cold)
    out["fit_warm_s"] = time.monotonic() - t
    out["iterations_warm"] = warm.iterations
    t = time.monotonic()
    bt.laplace(cold, samples=1000, rng=np.random.default_rng(0), top_k=5)
    out["laplace_s"] = time.monotonic() - t
    if args.bootstrap:
        from colosseum.arena.bt import BootstrapData

        t = time.monotonic()
        bt.bootstrap(cold, BootstrapData.from_table(table, "1v1", sources=("train",)), samples=args.bootstrap,
                     rng=np.random.default_rng(0))
        out["bootstrap_s"] = time.monotonic() - t
        out["bootstrap_samples"] = args.bootstrap
    out["pass"] = out["fit_cold_s"] <= 30 and out["fit_warm_s"] <= 5 and out["laplace_s"] <= 5
    print(json.dumps(out, indent=1))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
```

Create `scripts/measure_pulse_cadence.py`:

```python
"""Pulse cadence and the pipeline-test thresholds (SP4 criterion 3.4; plan T8.2).

Phase 1 (``--every 1 2 5``, seed 0): ``unit_harvest_league`` from scratch with pulse (``tests/arena_kit.py``,
1 rollout worker + the pulse process), ``evaluation.blocks`` = ``--blocks``. Recorded: snapshots stored,
snapshots evaluated by pulse (journal), the ``system`` counters ``pulse_skipped_snapshots`` /
``pulse_evaluated_snapshots`` (T6.1),
training seconds. Rule ``decide_cadence``: the smallest ``every`` with ``pulse_skipped <= 1`` (the final
snapshot may be cut by the budget) and at least 6 evaluated snapshots.

Phase 2 (``--calibrate-seeds 0 1 2`` with the chosen ``every``): per seed the rating trend (Spearman rho of BT
rating with version, gain newest third - oldest third), ``tournament --mode select`` over the run's
snapshots (field: the anchors) and ``eval`` of the winner against ``random`` (60 matches, deterministic).
Rule ``decide_trend``: ``RHO_MIN = max(0.3, floor_0.05(min rho - 0.1))``; ``GAIN_MIN = 0``; the
winner-vs-random bar stays the SP3 one (0.95) and is reported, not tuned.

Usage::

    env -u PYTHONPATH .venv/bin/python scripts/measure_pulse_cadence.py --every 1 2 5 --blocks 25 \
        --work-dir /tmp/sp4-pulse --json docs/benchmarks/sp4-pulse-cadence.json
    env -u PYTHONPATH .venv/bin/python scripts/measure_pulse_cadence.py --every <chosen> --blocks 25 \
        --calibrate-seeds 0 1 2 --work-dir /tmp/sp4-pulse --json docs/benchmarks/sp4-pipeline-calibration.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MIN_EVALUATED = 6
RHO_FLOOR = 0.3


def _system_counter(run, name: str) -> int | None:
    records = [r for r in run.run.records("system") if name in r]
    return int(max(r[name] for r in records)) if records else None


def decide_cadence(rows: list[dict]) -> dict:
    for row in sorted(rows, key=lambda r: r["every"]):
        if (row["pulse_skipped"] or 0) <= 1 and row["evaluated"] >= MIN_EVALUATED:
            return {"every": row["every"], "row": row}
    return {"every": None, "row": None}


def decide_trend(rows: list[dict]) -> dict:
    rhos = [r["spearman"] for r in rows]
    rho_min = max(RHO_FLOOR, math.floor((min(rhos) - 0.1) / 0.05) * 0.05) if rhos else None
    return {"rho_min": rho_min, "gain_min": 0.0, "rhos": rhos, "gains": [r["gain"] for r in rows],
            "vs_random": [r["vs_random_win"] for r in rows], "confirmed": [r["confirmed"] for r in rows]}


def main() -> int:
    import arena_kit as kit
    import pipeline_kit as pk

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--every", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--blocks", type=int, default=25)
    parser.add_argument("--calibrate-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/sp4-pulse"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    out: dict = {"blocks": args.blocks}
    if not args.calibrate_seeds:
        rows = []
        for every in args.every:
            work = args.work_dir / f"every{every}"
            work.mkdir(parents=True, exist_ok=True)
            run = kit.train_with_pulse(work, seed=0, settings=kit.PulseSettings(every=every, blocks=args.blocks))
            rows.append({"every": every, "stored": run.stored_versions, "evaluated": len(run.pulse_versions),
                         "pulse_versions": run.pulse_versions,
                         "pulse_skipped": _system_counter(run, "pulse_skipped_snapshots"),
                         "pulse_evaluated": _system_counter(run, "pulse_evaluated_snapshots"),
                         "train_s": round(run.seconds["train"], 1), "spearman": run.trend.spearman,
                         "gain": run.trend.gain})
            print(json.dumps(rows[-1]), flush=True)
        out.update(rows=rows, decision=decide_cadence(rows))
    else:
        (every,) = args.every
        rows = []
        for seed in args.calibrate_seeds:
            work = args.work_dir / f"calib-s{seed}"
            work.mkdir(parents=True, exist_ok=True)
            run = kit.train_with_pulse(work, seed=seed, settings=kit.PulseSettings(every=every, blocks=args.blocks))
            report = kit.select_snapshots(run, seed=seed)
            group = kit.group_of(report, "2p")
            winner = group["select"]["winner"]
            version = int(winner.rsplit("@v", 1)[1])
            pairs = pk.evaluate(run.config, run.run.root / "checkpoints" / pk.MAIN / f"ckpt_v{version}", [pk.RANDOM],
                                work / "winner-eval.json", layout="2p", num_matches=60)
            rows.append({"seed": seed, "every": every, "spearman": run.trend.spearman, "gain": run.trend.gain,
                         "evaluated": len(run.pulse_versions), "winner": winner,
                         "confirmed": bool(group["select"]["confirmed"]),
                         "vs_random_win": pairs[pk.RANDOM]["win_rate"], "train_s": round(run.seconds["train"], 1)})
            print(json.dumps(rows[-1]), flush=True)
        out.update(rows=rows, decision=decide_trend(rows))
    print(json.dumps(out["decision"], indent=1))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `tests/unit/test_sp4_measure_rules.py`:

```python
"""Decision rules of the SP4 measurements (plan T8.2), fixed before any number is seen: tournament default
sizes, BT prior / smoothness by cross-validation, pulse cadence and the pipeline trend threshold."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from arena_kit import spearman

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


defaults = _script("measure_tournament_defaults")
crossval = _script("bt_crossval")
cadence = _script("measure_pulse_cadence")


def _rows(cell, game, n, fail=()):
    return [{"game": game, "blocks": cell[0], "confirm": cell[1], "seed": s, "seconds": 10.0,
             "checks": {c: not (s == 0 and c in fail) for c in defaults.CHECKS}} for s in range(n)]


def test_tournament_defaults_take_the_first_passing_cell_and_the_floors():
    rows = (_rows((8, 32), "ttt", 10, fail=("winner",)) + _rows((8, 32), "tron", 10)
            + _rows((16, 64), "ttt", 10, fail=("confirmed",)) + _rows((16, 64), "tron", 10, fail=("order",)))
    decision = defaults.decide(rows)
    assert decision["chosen"] == [16, 64]
    assert decision["defaults"] == {"halving_blocks": 16, "confirm_blocks": 100, "blocks": 50}
    small = defaults.decide(_rows((8, 32), "ttt", 3) + _rows((8, 32), "tron", 3))
    assert small["defaults"] == {"halving_blocks": 8, "confirm_blocks": 100, "blocks": 50}
    assert defaults.decide(_rows((8, 32), "ttt", 10, fail=("winner",)))["chosen"] is None
    big = defaults.decide(_rows((32, 128), "ttt", 5) + _rows((32, 128), "tron", 5))
    assert big["defaults"] == {"halving_blocks": 32, "confirm_blocks": 128, "blocks": 64}


def test_problem_strings_are_classified():
    checks = defaults.classify(["ranking ['e025', 'e000'] != true order", "not confirmed; indistinguishable"])
    assert checks == {"winner": True, "order": False, "confirmed": False, "bt": True}


def test_crossval_prefers_most_smoothing_then_weakest_prior_within_one_se():
    rows = []
    for prior in crossval.PRIOR_GRID:
        for smooth in crossval.SMOOTH_GRID:
            loss = 0.60 + (0.05 if (prior, smooth) == (0.5, 64.0) else 0.0) + (0.0 if smooth <= 16 else 0.02)
            rows.append({"game": "g", "prior": prior, "smooth": smooth, "weight": [100.0] * crossval.FOLDS,
                         "loss": [100.0 * (loss + 0.001 * k) for k in range(crossval.FOLDS)]})
    decision = crossval.choose(rows)
    assert (decision["prior"], decision["smooth"]) == (8.0, 16.0)


def test_pulse_cadence_and_trend_rules():
    rows = [{"every": 1, "pulse_skipped": 3, "evaluated": 9}, {"every": 2, "pulse_skipped": 1, "evaluated": 6},
            {"every": 5, "pulse_skipped": 0, "evaluated": 2}]
    assert cadence.decide_cadence(rows)["every"] == 2
    assert cadence.decide_cadence(rows[2:])["every"] is None
    trend = cadence.decide_trend([{"spearman": 0.83, "gain": 50.0, "vs_random_win": 1.0, "confirmed": False},
                                  {"spearman": 0.71, "gain": 40.0, "vs_random_win": 1.0, "confirmed": True}])
    assert trend["rho_min"] == pytest.approx(0.6)
    assert cadence.decide_trend([{"spearman": 0.2, "gain": 1.0, "vs_random_win": 1.0, "confirmed": True}])["rho_min"] \
        == pytest.approx(0.3)


def test_spearman_handles_ties_and_degenerate_inputs():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3], [1, 1, 2]) == pytest.approx(0.8660254, abs=1e-6)
    assert math.isnan(spearman([1, 2], [1, 2])) and math.isnan(spearman([1, 2, 3], [5, 5, 5]))
```

Create `tests/integration/test_sp4_ladder_smoke.py`:

```python
"""Fast tournament smoke on the tic-tac-toe ladder (SP4 section 6, "tournament ... with a short ladder";
T8.2): ``--mode select`` with tiny sizes finishes and its report has every key the slow ladder test and the
report page read. No thresholds."""
from __future__ import annotations

import arena_kit as kit


def test_select_on_the_ladder_reports_what_the_checks_read(tmp_path):
    run = kit.run_select(kit.TIC_TAC_TOE, tmp_path, seed=1, blocks=2, confirm_blocks=2, num_envs=4)
    assert set(run.ranking) == set(kit.TRUE_ORDER)
    assert run.winner in kit.TRUE_ORDER and isinstance(run.confirmed, bool)
    for name in kit.TRUE_ORDER:
        row = run.rows[name]
        assert {"score", "score_ci", "bt", "bt_ci", "games"} <= set(row)
    assert isinstance(kit.ladder_problems(run), list)
    assert (run.out / "report.html").exists() or list(run.out.rglob("report.html"))
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_measure_rules.py tests/integration/test_sp4_ladder_smoke.py -v`

Expected: all pass; the smoke test under 20 s (it is the fast suite's guard on the report keys the slow ladder test reads; if Part 2's report names a key differently, fix `arena_kit` here, once).

- [ ] **Step 4: Full fast suite, ruff, commit (before the long measurements)**

Run the full fast suite and ruff (expected: all pass, no warnings), then:

```bash
git add examples/tic_tac_toe/ladder.py examples/tron/bots.py tests/arena_kit.py pyproject.toml scripts/measure_*.py \
  scripts/bt_crossval.py tests/unit/test_sp4_ladder_bots.py tests/unit/test_sp4_measure_rules.py \
  tests/integration/test_sp4_ladder_smoke.py
git commit -m "feat: ladder bots of known strength and the SP4 measurement scripts with pre-fixed rules"
git push
```

- [ ] **Step 5: Tournament default sizes (about 30–45 min)**

Preconditions: nothing else CPU-heavy runs (`uptime`).

```bash
M=.superpowers/sdd/2026-10-16-sp4-selection/t82; mkdir -p $M
nohup env -u PYTHONPATH .venv/bin/python scripts/measure_tournament_defaults.py --seeds 0 1 2 3 4 5 6 7 8 9 \
  --work-dir $M/ladder --json docs/benchmarks/sp4-tournament-defaults.json > $M/defaults.log 2>&1 &
```

60 tournaments (3 grid points × 2 games × 10 seeds). When it finishes: exit code 0 and a `decision` with `chosen` and `defaults`. Set `DEFAULT_HALVING_BLOCKS`, `DEFAULT_CONFIRM_BLOCKS`, `DEFAULT_BLOCKS` in `arena/tournament/modes.py` to `decision["defaults"]` and replace "provisional" in their docstrings by "T8.2 ruling (ladder measurement, docs/benchmarks.md)". `DEFAULT_FINALISTS` stays 3 (spec block 3). If no grid point passes (exit 1): stop and report the per-cell table to the controller (a ladder or tournament problem, not a tuning knob).

- [ ] **Step 6: BT prior and smoothness by cross-validation (about 40–60 min)**

```bash
nohup env -u PYTHONPATH .venv/bin/python scripts/bt_crossval.py --make-journals --work-dir $M/cv \
  --json docs/benchmarks/sp4-bt-crossval.json > $M/cv.log 2>&1 &
```

Four trainings (the example configs, 2 workers, seed 0) and 4 × 25 grid points × 5 folds of fits. Set `PRIOR_SIGMA` / `SMOOTHNESS` in `arena/bt.py` to `decision["prior"]` / `decision["smooth"]` (docstrings: "T8.2 ruling (cross-validation)"). Then rerun T2.1's coverage tests with the new values: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit -k "bt" -q` — all pass. A failing coverage test with the chosen values is reported to the controller with the numbers (the rule's choice is not overridden silently; the controller decides between the CV choice and the coverage tolerance, as a ruling).

- [ ] **Step 7: Refit cost at about 2 000 entities (minutes)**

```bash
env -u PYTHONPATH .venv/bin/python scripts/measure_bt_cost.py --bootstrap 20 --json docs/benchmarks/sp4-bt-cost.json
```

Expected: `"pass": true` (cold fit ≤ 30 s, warm ≤ 5 s, Laplace ≤ 5 s with one BLAS thread). Otherwise report the numbers to the controller (an iterative-solver task or a larger default `ratings.refit_interval_sec` is a ruling, not done here).

- [ ] **Step 8: Pulse cadence and the pipeline thresholds (about 40–60 min)**

```bash
nohup env -u PYTHONPATH .venv/bin/python scripts/measure_pulse_cadence.py --every 1 2 5 --blocks 25 \
  --work-dir $M/pulse --json docs/benchmarks/sp4-pulse-cadence.json > $M/pulse.log 2>&1 &
# when done, with E = decision["every"]:
nohup env -u PYTHONPATH .venv/bin/python scripts/measure_pulse_cadence.py --every $E --blocks 25 \
  --calibrate-seeds 0 1 2 --work-dir $M/pulse --json docs/benchmarks/sp4-pipeline-calibration.json > $M/calib.log 2>&1 &
```

Record: per `every` the stored / evaluated snapshots, `pulse_skipped`, training seconds; per seed rho, gain, the select winner, `confirmed`, the winner's win rate against `random`. If no `every` passes with `--blocks 25`, repeat phase 1 with `--blocks 10` (one retry; a ruling). If a calibration seed's winner wins less than 0.95 against `random`, stop and report (criterion 3.4 would fail; not tuned here). The values for T8.3: `SETTINGS = PulseSettings(every=E, blocks=B)`, `RHO_MIN = decision["rho_min"]`.

- [ ] **Step 9: Throughput against `main` by counters (about 25 min)**

Preconditions: nothing else running; the branch committed. Runs alternate main / branch twice. `main` is measured by the branch's script with `--tree` (its own `_make_config`, its own code on `PYTHONPATH`).

```bash
REPO=$(pwd); B=$M/bench; mkdir -p $B
git worktree add --detach /tmp/colosseum-main main
(cd /tmp/colosseum-main && env PYTHONPATH=/tmp/colosseum-main/src $REPO/.venv/bin/python -c "import colosseum; print(colosseum.__file__)")
for i in 1 2; do
  uptime
  (cd /tmp/colosseum-main && env PYTHONPATH=/tmp/colosseum-main/src $REPO/.venv/bin/python $REPO/scripts/bench_throughput.py \
     --tree /tmp/colosseum-main --workers 1 2 4 --duration 60 --warmup 15 --json $B/main-$i.json)
  uptime
  env -u PYTHONPATH .venv/bin/python scripts/bench_throughput.py --workers 1 2 4 --duration 60 --warmup 15 --json $B/branch-$i.json
done
git worktree remove /tmp/colosseum-main
mkdir -p docs/benchmarks/sp4-throughput && cp $B/*.json docs/benchmarks/sp4-throughput/
```

Expected: the printed path is `/tmp/colosseum-main/src/colosseum/__init__.py`; every run writes its JSON (`"clock": "counters"`). Per worker count w: `main_w` / `branch_w` = mean env steps/s of the two runs. Criterion 3.6 holds if `branch_w ≥ 0.95 · main_w` for w = 1, 2, 4. If it fails: rerun the pair once (noise check); if it still fails, report to the controller (a performance fix is a separate task with a re-measurement; the criterion is never relaxed silently).

- [ ] **Step 10: `docs/benchmarks.md` and the rulings**

Append to `docs/benchmarks.md` (Russian; every `<...>` filled from the JSON files and logs):

```markdown
## Лесенка ботов и размеры турнира (SP4, критерий 3.3)

Боты известной силы: `EpsilonTicTacToeBot` (крестики-нолики, `2p`) и `FloodFillTronBot` (tron, FFA `3p`) с долей случайных ходов ε ∈ {0, 0.25, 0.5, 1}; поле — `random` (`RandomBot`) и `ref` (тот же бот с ε = 0.1). Средний попарный счёт против поля на `MatchRunner` (замер при написании плана): крестики-нолики 0.742 / 0.620 / 0.532 / 0.315, tron 0.872 / 0.552 / 0.435 / 0.326. `scripts/measure_tournament_defaults.py`: `tournament --mode select`, 10 сидов на точку сетки, 2 воркера. Сырые данные: [`benchmarks/sp4-tournament-defaults.json`](benchmarks/sp4-tournament-defaults.json). Дата: <date>.

| `--halving-blocks` (первый раунд) | `--confirm-blocks` | игра | победитель верен | порядок верен | confirmed | BT-соседи | среднее время, с |
|---|---|---|---|---|---|---|---|
<six rows>

Правило (зафиксировано до замера): первая точка, где на обеих играх победитель верен во всех сидах, а порядок, `confirmed` и проверка BT — во всех, кроме не более одного. Итог: <chosen> → `DEFAULT_HALVING_BLOCKS` = <…>, `DEFAULT_CONFIRM_BLOCKS` = <…>, `DEFAULT_BLOCKS` = <…>.

## Приор и гладкость Bradley–Terry (SP4, блок 2)

`scripts/bt_crossval.py`: журналы четырёх демо-игр (<матчей каждой>), 5 фолдов по хэшу uid матча, сетка `prior_sigma` × `smoothness`, взвешенный log-loss отложенных пар команд. Сырые данные: [`benchmarks/sp4-bt-crossval.json`](benchmarks/sp4-bt-crossval.json).

<table: the 25 grid points, mean log-loss over games ± SE; the best and the chosen marked>

Правило: минимум среднего log-loss; среди точек в пределах одной SE — наибольшая гладкость, затем наибольшая `prior_sigma`. Итог: `PRIOR_SIGMA` = <…>, `SMOOTHNESS` = <…>; тесты покрытия T2.1 с этими значениями зелёные.

## Стоимость подгонки BT (~2 000 сущностей)

`scripts/measure_bt_cost.py` (один поток BLAS): холодная подгонка <…> с (<…> итераций), тёплая <…> с, Лаплас (1 000 выборок) <…> с, bootstrap <…> реплик <…> с. Порог живого рейтинга (≤ 30 / 5 / 5 с) <выполнен / нет>.

## Оценка во время обучения и пороги конвейера (SP4, критерий 3.4)

`scripts/measure_pulse_cadence.py`: `unit_harvest_league` с нуля, 1 воркер + процесс pulse, поле — якоря. <table: every → stored, evaluated, pulse_skipped, train s>. Правило: наименьший `every` с `pulse_skipped ≤ 1` и не менее 6 оценёнными снимками → `evaluation.every` = <…>, `blocks` = <…>. Калибровка (сиды 0–2): ρ Спирмена <…>, прирост <…> Elo, победитель `select` <…> (confirmed <…>), против `random` <…>. Порог `RHO_MIN` = <…> (max(0.3, ⌊min − 0.1⌋ с шагом 0.05)).

## После SP4

Та же нагрузка (`scripts/bench_throughput.py`, `_make_config`: крестики-нолики, self-play, только latest, без чекпоинтов; в ветке журнал матчей включён, pulse выключен), замер по счётчикам (`--clock counters`: часы WSL2 скачут, окна по `ts` не работают). `main` (коммит <sha>, `--tree` из `git worktree`) и ветка `sp4-selection` (коммит <sha>) по очереди: main → ветка → main → ветка. Дата: <date>. Машина: WSL2, 20 ядер, лёрнер на <cpu/cuda по `learner.device` бенча: cpu>. Сырые данные: [`benchmarks/sp4-throughput/`](benchmarks/sp4-throughput/).

| Воркеры | main, шаги сред/с (1 / 2 / среднее) | ветка (1 / 2 / среднее) | ветка / main | апдейты/с ветки | узкое место |
|---|---|---|---|---|---|
| 1 | <…> | <…> | <…> % | <…> | <…> |
| 2 | <…> | <…> | <…> % | <…> | <…> |
| 4 | <…> | <…> | <…> % | <…> | <…> |
```

Report to the controller, for its rulings ("что — почему — цена ошибки"): the tournament defaults; `PRIOR_SIGMA` / `SMOOTHNESS`; the refit cost verdict; `evaluation.every` / `blocks` and `RHO_MIN` of the pipeline test; the throughput ratios; and every retry or deviation above.

- [ ] **Step 11: Full fast suite, ruff, commit and push**

Run the full fast suite and ruff (expected: all pass, no warnings), then:

```bash
git add src/colosseum/arena/tournament/modes.py src/colosseum/arena/bt.py docs/benchmarks.md docs/benchmarks/sp4-*.json \
  docs/benchmarks/sp4-throughput
git commit -m "perf: SP4 measurements: tournament defaults, BT prior and smoothness, refit cost, pulse cadence, throughput"
git push
```

---

### Task T8.3: Slow tests — ladder on two games, pipeline with pulse and `select`, disk bound

Criteria 3.3 (ladder), 3.4 (pipeline), 3.5 (disk), 3.11 (the whole slow suite green), section 6 («Медленные»). Three new slow test files, four tests. Their thresholds are T8.2's rulings (sizes, `every`, `RHO_MIN`) and two values measured here by rules fixed now (`MAX_SECONDS` of the ladder, `SLACK_MB` of the disk test).

**Budget.** The SP3 slow suite takes about 7 minutes on the reference machine. Targets for the new tests: ladder ≤ 2 × 2.5 min, pipeline ≤ 8 min (training with 1 rollout worker + pulse, then `select` and `eval`), disk ≤ 3 min. The whole slow suite: about 20–25 minutes. If a test runs over its target by more than 50 %, report it to the controller with the step timings (smaller budgets are a ruling with measurements, like SP3 T6.4).

**Decisions (Contract notes D12–D14).**
- Pipeline: training from scratch (no `init`, no kickstart): the rating trend of a warm-started run is flat by construction (it starts at the bot's level, SP3 T6.4). `confirmed` of `select` is printed but not asserted: the best snapshots of a converged run are legitimately indistinguishable, and the spec forbids a silent choice, not an unconfirmed report. What is asserted: the winner was evaluated on stage-B seeds (it is the stage-B winner by the report's schema) and beats `RandomBot` with win rate ≥ 0.95 (SP3's threshold).
- Disk: "ratings from the journal equal ratings from the full in-memory table" compares the BT fit of the compacted journal with the run's final live fit (`ratings.json` → `bt`), which T2.2 computes from the in-memory comparison table that saw every written result (R-P1-9: a final refit at shutdown). No separate "no-delete" journal mode is needed for this; if Part 1 provides one anyway, the test may use it in addition.
- Disk: the bound is `peak(matches.db + -wal) ≤ window + (final file − final detail estimate) + SLACK_MB`, i.e. the window plus what compaction never touches (counts, schema, pages not yet vacuumed) plus a slack for a WAL between checkpoints and one compaction batch.

**Files:**
- Create: `tests/learning/test_sp4_ladder_slow.py`, `tests/learning/test_sp4_pipeline_slow.py`, `tests/integration/test_sp4_disk_bound_slow.py`
- Modify: `README.md` / `CLAUDE.md` slow-suite counts are updated in T8.4 (not here)

**Interfaces:**
- Consumes: `arena_kit` (T8.2), `pipeline_kit.evaluate / MAIN / RANDOM`, `demo_learning.LEARNING_SEED`, `cli_runner.TTT_CONFIG / run_train / run_in_session`, `MatchLog.open_readonly`, `MatchLogReader.sizes / comparison_table / matches` (R-P1-1, R-P1-7), `bt.fit`, `live.rating_inputs` (R-P1-5), the final `ratings.json` `bt` section (R-P1-6, R-P1-9), `checkpoint.max_snapshots` (T4.1), `match_log.*` (T1.1), `colosseum report` (T7.2).
- Produces: the slow tests `test_select_ranks_the_ladder_and_confirms_the_strongest[tic_tac_toe|tron]`, `test_pulse_trend_select_and_report`, `test_journal_and_snapshots_stay_bounded_and_counts_stay_complete`; constants `MAX_SECONDS`, `SETTINGS`, `RHO_MIN`, `SLACK_MB` (rulings).

- [ ] **Step 1: Write the slow tests with T8.2's values**

Create `tests/learning/test_sp4_ladder_slow.py`:

```python
"""Slow ladder test (SP4 criterion 3.3): ``colosseum tournament --mode select`` with the default sizes ranks
the ladder bots (one bot with a share epsilon of random moves) in their true order on tic_tac_toe (1v1)
and tron (FFA, 3p), selects the strongest, confirms it on fresh seeds, and the BT intervals of neighbours
do not contradict the order. Sizes and the time bound: T8.2 rulings; changing one needs a ruling with
measurements."""
from __future__ import annotations

import json

import pytest

import arena_kit as kit

pytestmark = [pytest.mark.slow, pytest.mark.timeout(900)]

LADDER_SEED = 100            # not among the measurement seeds 0-9 of T8.2
# T8.2 ruling: <filled in T8.3 Step 1 from docs/benchmarks/sp4-tournament-defaults.json>
MAX_SECONDS = 300.0          # provisional; T8.3 sets max(60, 2 x the slowest seed at the defaults), rounded up to 30 s


@pytest.mark.parametrize("game", kit.LADDER_GAMES, ids=lambda g: g.name)
def test_select_ranks_the_ladder_and_confirms_the_strongest(tmp_path, game):
    run = kit.run_select(game, tmp_path, seed=LADDER_SEED)
    rows = {name: {k: run.rows[name][k] for k in ("score", "score_ci", "bt", "bt_ci", "games")}
            for name in kit.TRUE_ORDER}
    print(f"[ladder {game.name}] {run.seconds:.0f}s ranking {run.ranking} winner {run.winner} "
          f"confirmed {run.confirmed}\n{json.dumps(rows)}")
    assert kit.ladder_problems(run) == []
    assert run.seconds <= MAX_SECONDS
```

Create `tests/learning/test_sp4_pipeline_slow.py` (set `SETTINGS` and `RHO_MIN` from T8.2 Step 8, with the comment line naming the measured values):

```python
"""Slow SP4 pipeline test (criterion 3.4) on unit_harvest_league trained from scratch: ``train`` with pulse
evaluation -> the snapshots' BT rating rises with the version (Spearman rho and gain thresholds: T8.2
rulings) -> ``tournament --mode select`` over the run's snapshots with the anchors as the field picks a
winner that was re-evaluated on fresh seeds -> the winner beats ``RandomBot`` at least as often as the SP3
slow test requires (0.95) -> ``colosseum report`` of the run builds. 1 rollout worker + the pulse process
(2 processes, test rule); the tournament runs after training with 2 workers."""
from __future__ import annotations

import json
import sys

import pytest

import arena_kit as kit
import pipeline_kit as pk
from cli_runner import run_in_session
from demo_learning import LEARNING_SEED

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800), pytest.mark.usefixtures("restore_global_rng")]

# T8.2 rulings (docs/benchmarks/sp4-pulse-cadence.json, sp4-pipeline-calibration.json): <filled in T8.3 Step 1>
SETTINGS = kit.PulseSettings(every=1, blocks=25)   # provisional
RHO_MIN = 0.3                                      # provisional floor of the rule
GAIN_MIN = 0.0                                     # the newest third rates above the oldest third
VS_RANDOM_MIN_WIN = 0.95                           # SP3 slow pipeline test (T6.4 ruling), not tuned here


def test_pulse_trend_select_and_report(tmp_path):
    run = kit.train_with_pulse(tmp_path, seed=LEARNING_SEED, settings=SETTINGS)
    print(f"[sp4 pipeline] train {run.seconds['train']:.0f}s; stored {run.stored_versions}; pulse evaluated "
          f"{run.pulse_versions}; trend rho {run.trend.spearman:.3f} gain {run.trend.gain:.1f}")
    assert run.pulse_versions, "pulse evaluated no snapshot"
    assert run.data["sections"]["anchors"]["source"] == "pulse"
    assert run.trend.spearman >= RHO_MIN and run.trend.gain > GAIN_MIN

    report = kit.select_snapshots(run, seed=7)
    group = kit.group_of(report, "2p")
    select = group["select"]
    print(f"[sp4 pipeline] select: winner {select['winner']} confirmed {select['confirmed']} "
          f"indistinguishable {select.get('indistinguishable')}\n{json.dumps(group['ranking'])}")
    version = int(select["winner"].rsplit("@v", 1)[1])
    checkpoint = run.run.root / "checkpoints" / pk.MAIN / f"ckpt_v{version}"
    pairs = pk.evaluate(run.config, checkpoint, [pk.RANDOM], tmp_path / "winner-eval.json", layout="2p",
                        num_matches=60)
    vs_random = pairs[pk.RANDOM]
    print(f"[sp4 pipeline] winner vs random: win {vs_random['win_rate']:.3f} ({vs_random['n']} matches)")
    assert vs_random["win_rate"] >= VS_RANDOM_MIN_WIN

    proc = run_in_session([sys.executable, "-m", "colosseum", "report", str(run.run.root)], timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    page = (run.run.root / "report.html").read_text()
    assert 'id="sec-tournaments"' in page and '"sel"' in page
```

Create `tests/integration/test_sp4_disk_bound_slow.py`:

```python
"""Slow disk-bound test (SP4 criterion 3.5): a tic_tac_toe run far longer than a 1 MB detail window keeps
``matches.db`` within the window + what compaction never touches + a slack (T8.3 ruling), its counts are
complete (the BT fit from the compacted journal equals the run's final live fit over the in-memory
comparison table, which saw every written result) and every agent keeps at most ``max_snapshots``
snapshots. 2 rollout workers."""
from __future__ import annotations

import json
import re
import threading
import time

import pytest

from cli_runner import TTT_CONFIG, run_train
from colosseum.arena import bt
from colosseum.arena.live import rating_inputs
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
from colosseum.core.config import load_config

pytestmark = [pytest.mark.slow, pytest.mark.timeout(900)]

MB = 2**20
WINDOW_MB = 1.0
MAX_SNAPSHOTS = 5
# T8.3 ruling: <peak excess measured in Step 2>; provisional 4 MB (a WAL between checkpoints + one compaction batch)
SLACK_MB = 4.0
RATING_TOL_ELO = 0.5
OVERRIDES = {
    "training.total_timesteps": 400_000, "rollout.num_workers": 2, "rollout.envs_per_worker": 16,
    "match_log.detail_window_mb": WINDOW_MB, "match_log.compact_interval_sec": 2.0,
    "checkpoint.interval": 20, "checkpoint.keep_last": 3, "checkpoint.keep_every": 2,
    "checkpoint.max_snapshots": MAX_SNAPSHOTS, "ratings.refit_interval_sec": 5.0,
}


def test_journal_and_snapshots_stay_bounded_and_counts_stay_complete(tmp_path):
    db = tmp_path / "runs" / "disk" / MATCH_LOG_FILE
    peak = {"bytes": 0}
    stop = threading.Event()

    def poll():
        while not stop.wait(0.5):
            total = sum(p.stat().st_size for p in (db, db.with_name(db.name + "-wal")) if p.exists())
            peak["bytes"] = max(peak["bytes"], total)

    sampler = threading.Thread(target=poll, daemon=True)
    sampler.start()
    start = time.monotonic()
    try:
        run = run_train(TTT_CONFIG, tmp_path, name="disk", overrides=OVERRIDES, timeout=800, tiny=False)
    finally:
        stop.set()
        sampler.join(timeout=5)
    assert run.returncode == 0, run.stderr[-3000:]
    reader = MatchLog.open_readonly(db)
    try:
        sizes = reader.sizes()
        table = reader.comparison_table(sources=("train", "pulse"))
        detail = len(reader.matches(source="train", limit=None))
    finally:
        reader.close()
    total_pairs = sum(stat.n for a, b, stat in table.pairs("2p", ("train",)))
    untouched = sizes.file_bytes - sizes.detail_bytes
    print(f"[disk] {time.monotonic() - start:.0f}s; peak {peak['bytes'] / MB:.2f} MB; file {sizes.file_bytes / MB:.2f} "
          f"MB, detail {sizes.detail_bytes / MB:.2f} MB, untouched {untouched / MB:.2f} MB, counts rows "
          f"{sizes.counts_rows}; matches {total_pairs}, detail kept {detail}")
    assert total_pairs >= 3 * detail, "the run never outgrew the window: nothing was compacted"
    assert sizes.detail_bytes <= 1.1 * WINDOW_MB * MB
    assert peak["bytes"] <= WINDOW_MB * MB + untouched + SLACK_MB * MB

    config = load_config(run.root / "config.resolved.yaml")
    options, fixed, random_bot = rating_inputs(config)
    from_journal = bt.fit(table, "2p", sources=("train", "pulse"), options=options, fixed_agents=fixed,
                          random_bot=random_bot)
    final_bt = json.loads((run.root / "ratings.json").read_text())["bt"]["2p"]
    live = {e["entity"]: e["rating"] for label, group in final_bt.items() if not label.startswith("@")
            for e in group}                                  # T2.2's bt_section: rows per group, '@' extras
    assert set(live) == set(from_journal.rating)
    worst = max(abs(live[e] - from_journal.rating[e]) for e in live)
    print(f"[disk] {len(live)} entities; max |live - journal| = {worst:.4f} Elo")
    assert worst <= RATING_TOL_ELO

    for agent_dir in (run.root / "checkpoints").iterdir():
        kept = [d for d in agent_dir.iterdir() if re.fullmatch(r"ckpt_v\d+", d.name)]
        assert len(kept) <= MAX_SNAPSHOTS, (agent_dir.name, sorted(d.name for d in kept))
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/learning/test_sp4_ladder_slow.py tests/learning/test_sp4_pipeline_slow.py tests/integration/test_sp4_disk_bound_slow.py --collect-only -q` and `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" --collect-only -q | tail -1`

Expected: 4 slow tests collected; the fast-suite count did not change.

- [ ] **Step 2: First runs, `MAX_SECONDS` and `SLACK_MB`**

Preconditions: nothing else CPU-heavy (`uptime`).

```bash
M=.superpowers/sdd/2026-10-16-sp4-selection/t83; mkdir -p $M
env -u PYTHONPATH .venv/bin/python -m pytest tests/learning/test_sp4_ladder_slow.py tests/integration/test_sp4_disk_bound_slow.py \
  -m slow -v -s 2>&1 | tee $M/first.txt | grep -E "^\[|PASSED|FAILED|Error"
```

Rules (fixed now): `MAX_SECONDS = max(60, 2 × the slower of the two ladder runs)`, rounded up to 30 s; `SLACK_MB = max(4, ⌈2 × (peak − window − untouched) in MB⌉)` from the `[disk]` line (if the measured excess is negative, keep 4). Write both into the tests with a comment naming the measured numbers ("T8.3 ruling: ladder 41 s / 63 s → 150 s", "T8.3 ruling: peak excess 1.3 MB → 4 MB"). A ladder test that fails its checks (not its time) is reported to the controller with the printed report rows: T8.2's defaults passed 10 seeds, so a failure at seed 100 is a ruling question (more seeds, or larger sizes), never a silent seed change.

- [ ] **Step 3: The pipeline test**

```bash
env -u PYTHONPATH .venv/bin/python -m pytest tests/learning/test_sp4_pipeline_slow.py -m slow -v -s 2>&1 | tee $M/pipeline.txt \
  | grep -E "^\[|PASSED|FAILED|Error"
```

Expected: PASSED; the printed trend at or above `RHO_MIN`, the winner against `random` ≥ 0.95, the report built. Compare the printed numbers with T8.2's calibration; a much lower value means something changed since: investigate before going on.

- [ ] **Step 4: The whole slow suite (criterion 3.11)**

```bash
nohup env -u PYTHONPATH .venv/bin/python -m pytest -m slow -v -s -p no:cacheprovider > $M/slow.txt 2>&1 &
# later:
grep -E "^\[|PASSED|FAILED" $M/slow.txt; tail -3 $M/slow.txt
```

Expected: 14 passed (SP3's 10 — the 7 demo-game learning tests, the unit_harvest pipeline, 2 torch.compile checks — plus the 4 new ones). A failing SP2/SP3 learning test is reported with its printed numbers; the controller decides by ruling with measurements (rerun for noise first). Thresholds of earlier tests are never lowered silently.

- [ ] **Step 5: Full fast suite, ruff, commit and push**

Run the full fast suite and ruff (expected: all pass, no warnings), then:

```bash
git add tests/learning/test_sp4_ladder_slow.py tests/learning/test_sp4_pipeline_slow.py \
  tests/integration/test_sp4_disk_bound_slow.py
git commit -m "test: SP4 slow tests: bot ladder on two games, pulse pipeline with select, journal disk bound"
git push
```

Report to the controller: the four tests' printed lines and durations, the slow-suite total, `MAX_SECONDS` and `SLACK_MB` with their measurements (rulings).

---

### Task T8.4: Documentation, example config, CLAUDE.md

Spec block 9 and criterion 3.10: a new `docs/ARENA_GUIDE.md` (Russian) covering the journal, ratings, the tournament with a step-by-step recipe for choosing a submission, snapshot retention, pulse, report and dashboard (phone access through an SSH tunnel) and common mistakes; `docs/LEAGUE_GUIDE.md` §12 / §14 / §15 updated with links; `README.md` (commands, run-dir contents, metrics table, ratings paragraph, limitations, SP4 row); `configs/examples/unit_harvest_league.yaml` shows `evaluation` and `keep_top_k`; `CLAUDE.md` status, commands, research list and open items (English). Documentation describes what the code does now; every flag and default is checked against `--help` and the code, every number against `docs/benchmarks.md`.

**Files:**
- Create: `docs/ARENA_GUIDE.md`
- Modify: `docs/LEAGUE_GUIDE.md`, `README.md`, `CLAUDE.md`, `configs/examples/unit_harvest_league.yaml`, `docs/GPU_CHECKS.md` (the bench line: counters), `docs/ENV_GUIDE.md` (only if it mentions `dropped_reward_episodes` semantics or SP4-deferred items)
- Test: `tests/unit/test_sp4_docs.py`

**Interfaces:**
- Consumes: everything SP4 built (names from the overview contract and the amendments), the T8.2 / T8.3 numbers and rulings, `colosseum <command> --help`.
- Produces: documentation; one test that keeps the example config and the docs' command list honest.

- [ ] **Step 1: A doc test (failing first)**

Create `tests/unit/test_sp4_docs.py`:

```python
"""Docs and example config of SP4 (criterion 3.10, T8.4): the league example shows evaluation and keep_top_k and
still validates without pulse running by default; README and ARENA_GUIDE name every new command; every flag
the guide uses exists in the command's --help."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.config import load_config

REPO = Path(__file__).resolve().parents[2]
COMMANDS = ("tournament", "report", "dashboard", "pin")


def test_league_example_shows_evaluation_and_top_k_switched_off():
    raw = yaml.safe_load((REPO / "configs/examples/unit_harvest_league.yaml").read_text())
    assert raw["evaluation"]["enabled"] is False and "every" in raw["evaluation"]
    assert raw["checkpoint"]["keep_top_k"] == 0 and "max_snapshots" in raw["checkpoint"]
    cfg = load_config(REPO / "configs/examples/unit_harvest_league.yaml")
    assert not cfg.evaluation.enabled and cfg.match_log.enabled


@pytest.mark.parametrize("doc", ["README.md", "docs/ARENA_GUIDE.md"])
def test_docs_name_every_new_command(doc):
    text = (REPO / doc).read_text()
    for command in COMMANDS:
        assert f"colosseum {command}" in text, (doc, command)


@pytest.mark.parametrize("command", COMMANDS)
def test_every_flag_in_the_guide_exists(command):
    help_text = CliRunner().invoke(main, [command, "--help"]).output
    guide = (REPO / "docs/ARENA_GUIDE.md").read_text()
    used = set()
    for line in re.findall(rf"colosseum {command}\b(?:[^\n`\\]|\\\n)*", guide):   # with \-continued lines
        used |= set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", line))
    missing = sorted(flag for flag in used if flag not in help_text)
    assert not missing, (command, missing)
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_docs.py -q`

Expected: failures (no ARENA_GUIDE, the example has no `evaluation`).

- [ ] **Step 2: The example config**

In `configs/examples/unit_harvest_league.yaml`: add to `checkpoint:` (keep the existing keys and comments)

```yaml
  keep_top_k: 0           # e.g. 5: also keep the 5 best snapshots by the rating's lower bound (needs match_log)
  max_snapshots: 100      # cap per agent; null = no cap (SP3)
```

and a new section after `checkpoint:`:

```yaml
evaluation:               # pulse (docs/ARENA_GUIDE.md): every k-th new snapshot against a fixed field on fixed seeds
  enabled: false          # true: one more process (num_envs envs); unbiased "score vs field by version" in the report
  field: null             # null: every scripted / frozen agent (here greedy and random)
  every: 1
  blocks: 25              # blocks per field member (1v1: 2 games per block)
```

Also extend the header comment of the file with one line: `#   colosseum report runs/harvest-league            # HTML report (docs/ARENA_GUIDE.md)`.

Run: `env -u PYTHONPATH .venv/bin/colosseum validate -c configs/examples/unit_harvest_league.yaml` and `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_example_configs.py tests/unit/test_sp3_example_configs_v3.py -q`

Expected: `Config is valid.` (the retention printout of T4.1 lists the rule); the example-config tests pass (`unit_harvest_league.yaml` has no SP2 copy, so the v3 comparison test does not cover it).

- [ ] **Step 3: `docs/ARENA_GUIDE.md` (Russian)**

Structure and the facts each section must contain (write prose in the style of `docs/LEAGUE_GUIDE.md`: short paragraphs, YAML and shell blocks that run as written, numbers from `docs/benchmarks.md` with a link):

```markdown
# Арена: журнал матчей, рейтинги, турнир и отбор, хранение снимков, отчёт

Вступление: что дал SP4 и как части связаны (журнал → рейтинги → хранение top-k; турнир → выбор сабмита; pulse → несмещённая кривая; отчёт / дашборд). Ссылки на LEAGUE_GUIDE (лига, warm start) и ENV_GUIDE.

## 1. Что лежит в папке запуска
`matches.db` (+ `-wal`, `-shm`), `league_state.json`, `run_state.json`, `report.html`, `tournaments/<name>/` (`tournament.yaml`, `report.{json,txt,html}`, `snapshots/`, без `--run` — свой `matches.db` в `--output`); что из этого переживает resume (журнал — новый, с родителем; `league_state.json` загружается; снимки переносятся).

## 2. Журнал матчей
- что пишется (поля матча и мест: uid, источник train / tournament / pulse, seed, `env_steps`, блок, версии мест в начале и конце эпизода, `rating_version`, награды, ранги, счёт);
- окно подробных записей `match_log.detail_window_mb` и сжатие в `counts` (что теряется: seed, длина, награды старых обучающих матчей; рейтинги совпадают — замер T8.3); турнир и pulse хранятся подробно всегда;
- защита диска `min_free_disk_mb` (переход на счётчики, флаг в «Системе»);
- ориентиры заполнения окна (числа T8.3: крестики-нолики; оценка для ~20 матчей/с);
- resume (родитель, импорт `counts` / турниров / pulse; подробности остаются в журнале родителя); SP3-папка (журнал начинается пустым);
- запись теряется только со счётчиком `dropped_match_results` и WARNING; SIGKILL теряет очередь писателя;
- `match_log.enabled: false` (только онлайн-ELO; top-k и pulse тогда — ошибка конфига);
- чтение руками: `sqlite3 'file:runs/x/matches.db?mode=ro'` с двумя примерами запросов по схеме T1.2 (таблицы `matches`, `seats`, `entities`, `counts`);
- распределённые роли журнал не пишут (SP5).

## 3. Рейтинги
- сущности: `агент@v<B>` (корзина `⌊version_start / interval⌋ · interval`; снимок `ckpt_v<N>` — версия N; корзина v0 — не снимок), скриптовые / замороженные — по имени, линия = агент; партии против себя только в поправку `b`;
- модель: пары команд (вес 1/(T−1), исход 1 / 0.5 / 0), сила команды = среднее участников + поправка номера команды (сторона, роли); приор и гладкость (значения и замер — ссылка на benchmarks);
- шкала: Elo-единицы, ноль — якорь (`ratings.anchor`; по умолчанию первый `RandomBot`, иначе среднее фиксированных, иначе v0 первого агента; в каждой группе ролей и компоненте);
- группы ролей и мосты, компоненты связности — что можно сравнивать между собой;
- интервалы: живой (Лаплас) и в отчёте (bootstrap, `ratings.bootstrap_samples`); `LCB`; `P(лучший)` среди снимков агента и среди группы; «мало игр»;
- флаг «возможны циклы» (G², p < 0.01, пары с наибольшими остатками) и что с ним делать (матрица, турнир round-robin);
- раскладки `score`: средний счёт с ДИ и cross-play;
- онлайн-ELO SP2 — индикатор прогресса, не для отбора; обучающие партии смещены PFSP («смещено PFSP» в отчёте).

## 4. Турнир и выбор сабмита
- команды с примерами: `colosseum tournament -c cfg.yaml --run runs/x --snapshots top:20 --field greedy,random,prev_sub --mode select --seed 7 --pin`; `-a имя=путь`, `-a имя`, несколько `--run` с алиасами; `--layout`, `--deterministic`, `--num-workers`, `--num-envs`, `--name`, `--output`;
- режимы gauntlet / round-robin / halving / select; результат кандидата (средний попарный счёт против поля, линии весят поровну), худший соперник, BT по матчам турнира;
- дизайн: блоки (seed + все допустимые ротации), наборы seed A / B / P не пересекаются, FFA-составы, асимметричные роли (по группам), несколько раскладок, раскладки `score`;
- размеры по умолчанию (`--blocks`, начальный блок halving, `--confirm-blocks`) — значения и как они замерены (лесенка, ссылка);
- `confirmed` и «неразличимые»: что делать, если `confirmed: false` (больше `--confirm-blocks`, другой `--seed`, решение за вами — тихого выбора нет); «игры повторяются» при `--deterministic`;
- продолжение прерванного турнира тем же `--name`; турнир во время обучения (WAL, жёсткие ссылки на снимки); проверка секции `env`;
- **рецепт выбора сабмита по шагам**: (1) обучение с журналом и, при желании, `keep_top_k` и pulse; (2) посмотреть отчёт: кривая против якорей, рейтинг по версиям, флаг циклов; (3) `tournament --snapshots top:N` (или `every:K`) с полем из якорей и прошлого сабмита в режиме `select`; (4) прочитать отчёт турнира: победитель, `confirmed`, интервал разности; (5) `--pin` (или `colosseum pin`); (6) экспорт: путь к папке снимка для сабмита; (7) после сабмита — добавить его как frozen-агента (`prev_sub`) в поле следующих турниров.

## 5. Хранение снимков
- правило удержания (всегда: `keep_last`, финальный текущего запуска, закреплённые, только что сохранённый; `keep_every`; `keep_top_k` по `LCB` с `top_k_quantile`; защита новых до `top_k_min_games`); потолок `max_snapshots` и порядок вытеснения; WARNING, если защищённых больше потолка;
- финальные снимки импортированных запусков — обычные (частые resume больше не раздувают пул);
- `colosseum pin runs/x main@v1200 [--remove]`, `colosseum pin runs/x --list`; закрепление переживает resume;
- что печатает `validate`; распределённый лёрнер — FIFO без рейтингов.

## 6. Оценка во время обучения (pulse)
конфиг `evaluation` (все ключи с умолчаниями), что делает процесс (те же seed домена P для всех снимков, места не собирают данные, результаты с `source = pulse`), цена (один процесс, `num_envs` сред; бенчмарк скорости — без pulse), очередь `max_backlog` (пропуск старых снимков, INFO и счётчик в «Системе»), финальный снимок может не успеть, неоценённые помечены в отчёте; выбор `every` / `blocks` (замер T8.2 как пример).

## 7. Отчёт и дашборд
- `colosseum report runs/x [-o FILE]`: один HTML-файл без сети (можно переслать на телефон), разделы (таблица блока 5 спеки), ограничения размера (2 000 точек ряда, 2 000 строк, 1 000 матчей, 100 снимков в матрице);
- `colosseum dashboard runs/x [--host 127.0.0.1] [--port 8765] [--refresh 10]`: та же страница вживую, `/api/<раздел>` (JSON; фильтры `agent`, `layout`, `source`, `outcome`), рейтинги из `ratings.json` во время обучения;
- **доступ с другой машины и телефона**: только через SSH-туннель — `ssh -N -L 8765:127.0.0.1:8765 user@train-box`, затем `http://127.0.0.1:8765/` на ноутбуке; с телефона — SSH-клиент с пробросом порта (например, Termux: `ssh -N -L 8765:127.0.0.1:8765 user@host`) или просто `report.html` в мессенджере; без авторизации никогда не слушать `0.0.0.0` в чужой сети;
- состояние запуска (`run_state.json`, проверка pid) и «stopped»; часы WSL2 (порядок по счётчикам).

## 8. Частые ошибки
выбор сабмита по обучающему рейтингу или онлайн-ELO; сравнение рейтингов разных групп ролей без моста; `confirm-blocks` слишком мало (и выбор «на глаз» между неразличимыми); `--deterministic` в детерминированной игре (повторяющиеся партии); `keep_top_k` без журнала или `max_snapshots < keep_last` (ошибка конфига); pulse на перегруженной машине (пропуски снимков); турнир с другим `env` (ошибка); маленькое окно журнала с большим числом сущностей (размер `counts`); упорядочивание по `ts` на WSL2; дашборд на `0.0.0.0`.
```

Every command line in the guide must run against the code as written (Step 6 runs the recipe of §4 on a short run).

- [ ] **Step 4: `docs/LEAGUE_GUIDE.md`**

- §12 «Хранение снимков»: add the SP4 keys to the YAML block (`keep_top_k: 0`, `top_k_quantile: 0.1`, `top_k_min_games: 200`, `max_snapshots: 100`), replace «Финальный снимок не удаляется никогда» by "the final snapshot of the current run is never evicted; finals of runs this one resumed from become ordinary snapshots", replace the last sentence («Лучшие снимки по рейтингу (top-k) — в SP4 …») by a two-sentence summary of top-k / pin / cap with a link to `ARENA_GUIDE.md` §5.
- §14 «Метрики и рейтинги»: replace «Снимки засчитываются своему агенту (рейтинги снимков — SP4)» by a paragraph on the BT ratings of snapshots (journal, live in `ratings.json` → `bt`, console `bt <rating> ± <CI>`, report) with a link to `ARENA_GUIDE.md` §3; mention that `league_state.json` carries ELO / PFSP over a run-dir resume; the `ratings` records of `metrics.jsonl` now carry a PFSP summary (the full table stays in `ratings.json`).
- §15 «Ограничения»: replace the SP4 line by the SP4-out-of-scope items of spec §2 that a user meets (no distributed tournament / pulse / journal: SP5; no Swiss / SPRT gate / α-Rank; no Davidson / Plackett–Luce; no strategy-diversity top-k) and a link to `ARENA_GUIDE.md`; add to the distributed bullet: `evaluation.enabled` and `keep_top_k > 0` are config errors there, `max_snapshots` works FIFO.
- §13 (the `validate` printout): regenerate the two example outputs with the current code (they gain the retention and journal lines).

- [ ] **Step 5: `README.md` and `CLAUDE.md`**

README (English), each item checked against the code:
- Quick start / pipeline block: after the `eval` line add `colosseum report runs/harvest-league` and one sentence on the report.
- New section «Selection: match log, ratings, tournament, report» after «Evaluation»: the four commands with their synopses (`colosseum tournament ...`, `colosseum pin RUN [ENTITY] [--remove] [--list]`, `colosseum report RUN [-o FILE]`, `colosseum dashboard RUN [--host] [--port] [--refresh]`), five to eight lines on what each does, the SSH-tunnel note, a link to `docs/ARENA_GUIDE.md`.
- «What a run writes»: add `matches.db` (match log, bounded by `match_log.detail_window_mb`), `league_state.json`, `run_state.json`, `report.html` (written by `colosseum report`), `tournaments/<name>/`; `ratings.json` gains `bt`.
- `metrics.jsonl` table: `train` — `rho_trunc_mass` / `c_trunc_mass` (T3.1) instead of `rho_clip_frac` / `c_clip_frac`; `ratings` — PFSP summary instead of the full table, plus the compact `bt`; `system` — monotonic `dropped_reward_episodes`, `dropped_match_results`, the journal fields (sizes, counts-only flag) and the pulse counters. `meta.json`: `run_id`, `pinned`.
- «Ratings» paragraph: snapshot ratings (Bradley–Terry from the journal, live and in reports), the anchor rule, online ELO unchanged as a progress indicator, `league_state.json` over resume.
- Configuration reference: the `match_log`, `ratings`, `evaluation` sections and the new `checkpoint` keys (global only).
- «Distributed mode (limited)»: no match log or ratings; `evaluation.enabled` / `keep_top_k > 0` refused; `max_snapshots` FIFO.
- «Status and limitations»: the SP4 row content (match log, Bradley–Terry snapshot ratings with intervals, `colosseum tournament` with two-stage selection, top-k / pinned / capped snapshot storage, pulse, `report` / `dashboard`) with "(implemented on `sp4-selection`; acceptance pending)"; Known limitations: replace the SP4 item under **Players** by the remaining out-of-scope items (no distributed tournament or pulse until SP5; no Swiss / SPRT; FFA intervals from compacted training games are approximate).
- «Tests»: the fast-suite count (`pytest -m "not gpu and not slow" --collect-only -q | tail -1`), the slow suite "14 tests, about <T8.3 total> min" with the list, the gpu count.

CLAUDE.md (English):
- Research Documents: add `research/sp4_ratings_selection.md` — "SP4 research: rating models for our setting (Elo, BT, TrueSkill / OpenSkill, teams, FFA, asymmetric roles, anchors), online vs batch fitting, snapshot ratings, selecting the best with a given confidence (sample sizes, paired designs, winner's curse, successive halving), top-k retention" and `research/sp4_selection_practice.md` — "SP4 practice: how competition winners choose submissions, ladder rating systems, tournament formats and tools, match-log formats and sizes, league dashboards" (and `research/sp4_sims/` — the simulation scripts behind them).
- Phase 3 / Phase 4 / Eval Mode / Checkpoint Management / Metrics & Logging / Orchestration bullets: replace the SP4 "target" remarks with what exists (snapshot ratings, top-k, the tournament, the dashboard; TrueSkill-type online ratings are not planned — the live rating is BT on a timer, spec decision 1).
- Implementation Status: "State after SP4 (selection and observability), implemented on `sp4-selection`, acceptance pending"; test counts; SP4 records (spec, plan dir with its overview and amendments, the acceptance report path once T8.5 writes it); a «Works» bullet group for SP4 (match log, ratings, tournament, retention, pulse, report / dashboard, `run_state.json`, `league_state.json`, metrics changes, bench by counters).
- Commands: `tournament`, `pin`, `report`, `dashboard` with synopses; the measurement scripts of T8.2 under «Measurements».
- Partial / Not implemented / Roadmap / Open items: move the SP4 items that are done out of «Not implemented»; SP4 row of the Roadmap stays "in progress" until the merge (T8.5 Step 14 marks it done); «Open items» gets a «Parked during SP4» list from the SDD ledger's deferred items (filled again in T8.5).
- «Next step»: SP5 (parallel) and the SP4 acceptance.

- [ ] **Step 6: Run the README and guide commands on a short run**

```bash
D=.superpowers/sdd/2026-10-16-sp4-selection/t84; mkdir -p $D
env -u PYTHONPATH .venv/bin/colosseum train -c configs/examples/unit_harvest_league.yaml --set run.dir=$D/runs \
  --set run.name=docs --set training.total_timesteps=40000 --set evaluation.enabled=true --set rollout.num_workers=1
env -u PYTHONPATH .venv/bin/colosseum report $D/runs/docs
env -u PYTHONPATH .venv/bin/colosseum tournament -c configs/examples/unit_harvest_league.yaml --run $D/runs/docs \
  --snapshots last:3 --field greedy,random --mode select --seed 7 --num-workers 2 --pin
env -u PYTHONPATH .venv/bin/colosseum pin $D/runs/docs --list
env -u PYTHONPATH .venv/bin/colosseum dashboard $D/runs/docs --port 0 & DPID=$!; sleep 8; kill -INT $DPID; wait $DPID; echo "dashboard exit=$?"
rm -rf $D/runs
```

Expected: every command exits 0 (the dashboard 130 after the SIGINT), the tournament report names a winner, `pin --list` shows it pinned. Adjust the guide and README where a command or its output differs. Compare every flag in README and the guide with `colosseum <command> --help`.

- [ ] **Step 7: Run the doc test, the full fast suite and ruff; commit and push**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_docs.py -q`, then the full fast suite and ruff.

Expected: all pass, no warnings.

```bash
git add docs/ARENA_GUIDE.md docs/LEAGUE_GUIDE.md docs/GPU_CHECKS.md README.md CLAUDE.md \
  configs/examples/unit_harvest_league.yaml tests/unit/test_sp4_docs.py
git commit -m "docs: ARENA_GUIDE, league guide, README and CLAUDE.md for SP4; league example shows pulse and top-k"
git push
```

(Add `docs/ENV_GUIDE.md` only if it changed.)

---

### Task T8.5: Acceptance against spec section 3, report, stop for the owner

Spec section 3 (criteria 3.1–3.11) and section 7 («приёмка по разделу 3, отчёт в `docs/superpowers/reports/`, merge после явного одобрения владельца»). Each step checks one criterion with exact commands and an observable result; its outcome (pass / fail, numbers, node ids, paths) goes into the report.
- The whole-branch review (three area reviewers) and its single fix wave run before this task (Development Workflow in CLAUDE.md); the controller copies their records into `docs/superpowers/reports/<date>-sp4-review/` (README index plus the review and fix-wave reports) before Step 12.
- If a step fails, stop, write what failed into the report draft, and report to the controller. Do not "fix and continue" inside this task beyond documentation mismatches (those are fixed with a `docs:` commit and the step is rerun).
- The merge (Step 15) happens **only after the owner explicitly accepts the report** in their own message. No agent message counts as acceptance.

**Files:**
- Create: `docs/superpowers/reports/<date>-sp4-acceptance.md` (`<date>` = `date +%F` on the day the report is written)
- Possibly modify: `README.md`, `CLAUDE.md`, `docs/*.md` (documentation mismatches, `docs:` commits), `docs/benchmarks.md` + `docs/benchmarks/sp4-throughput/` (a re-measurement, Step 6)

**Interfaces:**
- Consumes: everything in parts 01–03; the SDD ledger `.superpowers/sdd/2026-10-16-sp4-selection/progress.md` (rulings and deferred items) and `constraints.md` (pre-flight rulings); the overview's «Contract amendments».
- Produces: the acceptance report; after the owner's acceptance, `main` containing SP4.

- [ ] **Step 1: Criterion 3.1 — fast suite, zero warnings, files only in `tmp_path`, ruff, ≤ 2 workers**

```bash
git status --porcelain                                     # must be empty before starting
A=.superpowers/sdd/2026-10-16-sp4-selection/t85; mkdir -p $A
timeout 3600 env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw -p no:cacheprovider 2>&1 | tee $A/fast.txt | tail -5; echo "exit=${PIPESTATUS[0]}"
grep -E "warnings summary|[0-9]+ warnings?( |$)" $A/fast.txt || echo "no warnings"
git status --porcelain --ignored | grep -vE "\.venv/|__pycache__|\.egg-info|\.ruff_cache|\.pytest_cache|\.superpowers/" || echo "clean"
env -u PYTHONPATH .venv/bin/ruff check .
for m in "not gpu and not slow" slow gpu; do echo "$m: $(env -u PYTHONPATH .venv/bin/python -m pytest -m "$m" --collect-only -q | tail -1)"; done
grep -rnE "num_workers[\"']?[:=] ?[\"']?[3-9]|--workers [3-9]|--num-workers[\"', ]+[3-9]|num_workers=[3-9]" tests/ || echo "no test uses more than 2 workers"
```

Expected: `exit=0`; `no warnings`; `clean` (no `runs/`, `matches.db`, `report.html`, `tournaments/`, `bc.pt`, `*.jsonl` in the repo); `All checks passed!`; the three counts equal those in README and CLAUDE.md; `no test uses more than 2 workers` (a hit that is a benchmark constant or a comment is fine — read it; a training test with pulse uses 1 rollout worker).

- [ ] **Step 2: Criterion 3.2 — contract tests**

```bash
env -u PYTHONPATH .venv/bin/python -m pytest $(ls tests/*/test_sp4_*.py | grep -v _slow) tests/contract -v -rA -m "not slow" 2>&1 | tee $A/contract.txt | tail -5
```

Expected: everything passes. For each item of criterion 3.2 / section 6 find the passing test(s) in `$A/contract.txt` and list their node ids in the report:

| Item (spec criterion 3.2 / section 6) | Task that wrote the test |
|---|---|
| every training match reaches the journal exactly once or is counted in `dropped_match_results`; seat versions at episode start and end, also when weights change mid-episode; a full worker or writer queue counts and warns; the writer queue is flushed at stop | T0.1, T1.1, T1.3 |
| compaction does not change ratings (journal vs full in-memory table); the window bound; untouched parts do not shrink the window; the disk guard switches to counts and back; resume import folds the parent's window into counts; read-only open while writing; two writers lose nothing | T1.2, T1.3 |
| Bradley–Terry recovers synthetic strengths (1v1 with side, teams, FFA, draws, asymmetric roles with and without a bridge): coverage of the 95 % intervals; finite estimates with few games / always losing; no cross-group comparison without a bridge; anchor per component; side bias; self-play only in `b`; anchor rule, replacement, `ConfigError`; smoothness; G² flag; score layouts | T2.1 |
| retention: `keep_last`, current-run final by `run_id`, pinned, `keep_every`, top-k by LCB, protection until `top_k_min_games`, the cap and its order (FIFO without top-k), the warning; pinning survives resume; `validate` checks of `max_snapshots` | T4.1, T4.2 |
| tournament: the same block seeds for every candidate; rotations equal and admissible; A / B / P disjoint; FFA pairs equally often; role groups and the mixed-group `ConfigError`; several layouts and score layouts; several `--run` with aliases; frozen and scripted teammates; resume by whole blocks and the settings `ConfigError`; snapshots survive eviction; the env check; repeated deterministic games; halving / select logic; `confirmed` by the paired stratified bootstrap | T5.1–T5.4 |
| pulse gets every new snapshot; its matches have `source = pulse` and the same seeds for different snapshots; backlog overflow skips the old ones; it dies with the run and its death stops the run | T6.1 |
| resume continues the journal and `league_state.json` | T1.3, T3.2 |
| report / dashboard: builders for `wdl`, `rank`, `score`; the HTML has the data and all sections; the server answers section JSON on a free port; run state by `run_state.json` | T7.1–T7.3 |
| distributed: `evaluation.enabled` and `keep_top_k > 0` → `ConfigError` with SP5; `max_snapshots` evicts the oldest in the distributed learner | T4.1, T8.1 |

An item without a passing test fails this criterion.

- [ ] **Step 3: Criteria 3.3, 3.4, 3.5, 3.11 — the slow suite**

```bash
nproc; uptime
env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp4_ladder_smoke.py tests/integration/test_sp4_report_cli.py tests/integration/test_sp4_dashboard.py -v --durations=0 2>&1 | tail -15
nohup env -u PYTHONPATH .venv/bin/python -m pytest -m slow -v -s -p no:cacheprovider > $A/slow.txt 2>&1 &
# when it finished (about 20-25 minutes):
grep -E "^\[|PASSED|FAILED" $A/slow.txt; tail -3 $A/slow.txt
```

Expected: the fast SP4 integration tests pass, each under 20 s; `nproc` = 20; all 14 slow tests PASS: the 7 demo-game learning tests, the SP3 unit_harvest pipeline, 2 torch.compile checks, the ladder on tic_tac_toe and tron (criterion 3.3: true order, the strongest selected and confirmed, BT neighbours consistent), the SP4 pipeline (criterion 3.4: trend ≥ `RHO_MIN`, the winner ≥ 0.95 against `random`, the report builds) and the disk bound (criterion 3.5: peak within the bound, `|live − journal|` ≤ 0.5 Elo, snapshots ≤ `max_snapshots`). Copy every printed `[...]` line into the report under its criterion and compare with T8.2 / T8.3; a value much lower than measured there means something changed since: investigate before going on.

- [ ] **Step 4: Criterion 3.6 — speed**

```bash
ls docs/benchmarks/sp4-throughput/
git log --oneline -1 -- docs/benchmarks/sp4-throughput
git diff --stat $(git log --format=%h -1 -- docs/benchmarks/sp4-throughput)..HEAD -- src/colosseum/worker src/colosseum/learner \
  src/colosseum/algorithms src/colosseum/envs src/colosseum/networks src/colosseum/core src/colosseum/coordinator \
  src/colosseum/league src/colosseum/arena src/colosseum/launcher.py src/colosseum/metrics
sed -n '/## После SP4/,/^## [^П]/p' docs/benchmarks.md | head -20
```

Expected: the «После SP4» table shows branch ≥ 95 % of `main` at 1 / 2 / 4 workers, measured by counters. If the `git diff --stat` lists changes since the measurement (e.g. the review fix wave), rerun T8.2 Step 9 (both `main` and the branch, alternating, by counters), update the table and the JSON (`docs:` commit) and use the new numbers.

- [ ] **Step 5: Criterion 3.7 — report and dashboard, the owner's phone check**

```bash
env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_report_render.py tests/unit/test_sp4_report_data.py \
  tests/integration/test_sp4_report_cli.py tests/integration/test_sp4_dashboard.py -v 2>&1 | tail -12
D=$A/demo; mkdir -p $D
env -u PYTHONPATH .venv/bin/colosseum train -c configs/examples/unit_harvest_league.yaml --set run.dir=$D --set run.name=accept \
  --set evaluation.enabled=true --set rollout.num_workers=1
env -u PYTHONPATH .venv/bin/colosseum tournament -c configs/examples/unit_harvest_league.yaml --run $D/accept \
  --snapshots all --field greedy,random --mode select --seed 7 --num-workers 2
env -u PYTHONPATH .venv/bin/colosseum report $D/accept && ls -la $D/accept/report.html
```

Expected: the tests pass (the Chrome / node tests run, not skip, on the reference machine); the run, the tournament and the report exit 0; `report.html` is a few MB at most. The controller publishes `$D/accept/report.html` as a private artifact page for the owner (the file is self-contained) and asks the owner to open it on the phone: every section readable, tables scroll inside their boxes, charts and the heat map usable, light and dark themes. Record the owner's answer in the report (criterion 3.7, «Владелец смотрит отчёт с телефона при приёмке»). Optionally demonstrate the dashboard through an SSH tunnel as in `ARENA_GUIDE.md` §7.

- [ ] **Step 6: Criterion 3.8 — SP3 compatibility**

```bash
env -u PYTHONPATH .venv/bin/python -m pytest $(grep -rlE "fixtures.{0,6}sp3|copy_sp3_run|SP3_RUN" tests --include="test_*.py") -v 2>&1 | tail -15
for f in configs/examples/*.yaml; do
  env -u PYTHONPATH .venv/bin/colosseum validate -c "$f" > $A/validate-$(basename $f).txt 2>&1 && echo "OK $f" || echo "FAIL $f"
done
```

Expected: the T0.1 compatibility tests pass (the SP3 run dir resumes with an empty journal and empty league state plus INFO; SP3 snapshots play in a tournament as run snapshots and as frozen agents; `MatchResult` / `SeatResult` without the new fields work); every example config `OK`. List the node ids in the report.

- [ ] **Step 7: Criterion 3.9 — distributed mode**

```bash
env -u PYTHONPATH .venv/bin/python -m pytest tests -v -k "distributed or grpc" 2>&1 | tail -15
R=$A/dist; rm -rf $R
env -u PYTHONPATH .venv/bin/colosseum serve-weight-store --port 50051 & WS=$!
env -u PYTHONPATH .venv/bin/colosseum run-learner -c configs/examples/tic_tac_toe.yaml --agent agent_0 --traj-port 50052 \
  --weight-store localhost:50051 --set run.dir=$R --set training.total_timesteps=20000 & LR=$!
sleep 10
env -u PYTHONPATH .venv/bin/colosseum run-workers -c configs/examples/tic_tac_toe.yaml --weight-store localhost:50051 \
  -l agent_0=localhost:50052 --set run.dir=$R --set training.total_timesteps=20000; echo "workers exit=$?"
kill $LR $WS; wait
ls $R; ls $R/*/ | head; find $R -name matches.db | wc -l
env -u PYTHONPATH .venv/bin/colosseum run-learner -c configs/examples/tic_tac_toe.yaml --agent agent_0 --traj-port 50052 \
  --weight-store localhost:50051 --set run.dir=$R --set checkpoint.keep_top_k=3; echo "top-k exit=$?"
env -u PYTHONPATH .venv/bin/colosseum run-workers -c configs/examples/unit_harvest_league.yaml --weight-store localhost:50051 \
  -l main=localhost:50052 --set run.dir=$R --set evaluation.enabled=true; echo "pulse exit=$?"
rm -rf $R
```

The weight store is already stopped for the last two commands on purpose: the scope check happens before any connection. Expected: the distributed tests pass (SP2, SP3 and `test_sp4_distributed_guards.py`); `workers exit=0`; the learner logged the INFO "distributed roles write no match log"; no `matches.db` anywhere (`0`); the last two commands fail fast with `exit=1` and a `Config error:` line naming SP5 (`keep_top_k`; `evaluation.enabled` and the scripted agents).

- [ ] **Step 8: Criterion 3.10 — documentation**

In a fresh shell (`env -i HOME=$HOME PATH=/usr/bin:/bin bash --noprofile --norc`), from the repo root, after `source .venv/bin/activate`, run the README «Competition pipeline» block and the new «Selection» commands exactly as written (into `runs/`), then:
- compare every flag in README and `docs/ARENA_GUIDE.md` with `colosseum <command> --help` (`tests/unit/test_sp4_docs.py` covers the guide's tournament / report / dashboard / pin lines);
- check that `docs/ARENA_GUIDE.md` has every topic of spec block 9 (`grep -n "^## " docs/ARENA_GUIDE.md`: journal, ratings, tournament with the submission recipe, retention, pulse, report and dashboard with the SSH tunnel, common mistakes);
- check that `docs/LEAGUE_GUIDE.md` §12 / §14 / §15 link to it and no longer say "SP4" for what exists;
- check that the README metrics table names `rho_trunc_mass`, `c_trunc_mass`, `dropped_match_results`, the monotonic `dropped_reward_episodes`, the PFSP summary and `bt`;
- check that README «Status and limitations» and CLAUDE.md «Implementation Status» / «Roadmap» / open items describe what Steps 1–7 observed.

Clean up: `rm -rf runs data bc.pt`.

- [ ] **Step 9: Hygiene**

```bash
git status --porcelain
git status --porcelain --ignored | grep -vE "\.venv/|__pycache__|\.egg-info|\.ruff_cache|\.pytest_cache|\.superpowers/" || echo "clean"
ps -eo pid,args | grep -E "colosseum|multiprocessing|bench_throughput|measure_|bt_crossval" | grep -v grep || echo "no leftover processes"
git worktree list
```

Expected: an empty status (except the report being written), `clean`, `no leftover processes`, only the main worktree.

- [ ] **Step 10: Collect the rulings and the deferred items**

```bash
ls .superpowers/sdd/
grep -n "Ruling:" .superpowers/sdd/2026-10-16-sp4-selection/progress.md | wc -l
grep -n "Ruling:" .superpowers/sdd/2026-10-16-sp4-selection/progress.md > $A/rulings.txt
grep -nE "Deferred|deferred|Minor|minor" .superpowers/sdd/2026-10-16-sp4-selection/progress.md > $A/deferred.txt
grep -nE "^- \*\*P[0-9]+" .superpowers/sdd/2026-10-16-sp4-selection/constraints.md > $A/preflight.txt
```

Every ruling goes into the report verbatim, in ledger order ("что — почему — цена ошибки"), grouped as: the spec's controller decisions and post-review clarifications (spec §8, for reference), the overview's contract amendments (A1…), pre-flight rulings (P1…), task rulings (T0.1 … T8.4, including the measured values: tournament defaults, `PRIOR_SIGMA` / `SMOOTHNESS`, refit-cost verdict, `evaluation.every` / `blocks`, `RHO_MIN`, `MAX_SECONDS`, `SLACK_MB`, throughput), review and fix-wave rulings, and the proposed rulings of this part's Contract notes (D1–D14) with the controller's decision on each. Every deferred minor goes into the appendix verbatim; the open items section lists those addressed to later sub-projects.

- [ ] **Step 11: Write the report**

Create `docs/superpowers/reports/<date>-sp4-acceptance.md` (Russian) with the structure of the SP3 report (`docs/superpowers/reports/2026-10-10-sp3-acceptance.md`):

```markdown
# SP4: приёмка по спеке §3 (T8.5)

- Ветка `sp4-selection`. Начало приёмки: HEAD `<sha>`. Правки документации во время приёмки: <shas или «нет»>. Отчёт — следующий коммит.
- Дата: <date>.
- Машина: WSL2, <nproc> ядер, <RAM> ГБ, <GPU>. Python <версия>, torch <версия>, `.venv`.
- Записи финального ревью и волны исправлений: [`<date>-sp4-review/`](<date>-sp4-review/README.md).
- Шаги 14–15 (статус «принято» и merge) не выполнялись: ждут явного одобрения владельца.

## Итог

<ВСЕ КРИТЕРИИ ВЫПОЛНЕНЫ / список невыполненных; одна фраза о вопросах владельцу>

| § | Критерий | Результат |
|---|---|---|
| 3.1 | Тесты | <PASS/FAIL: N passed, M deselected, время, предупреждений 0; ruff; дерево чистое; ≤ 2 воркеров; счёт fast / slow / gpu совпадает с README и CLAUDE.md> |
| 3.2 | Контрактные тесты | <число тестов; у каждого пункта есть зелёный тест> |
| 3.3 | Лесенка ботов | <обе игры: порядок, победитель, confirmed, BT-соседи; время; размеры по умолчанию (ruling)> |
| 3.4 | Конвейер | <ρ и прирост против RHO_MIN; победитель select, confirmed; против random; отчёт собран> |
| 3.5 | Диск | <пик против окна + нетронутое + запас; |live − журнал|; снимков ≤ max_snapshots> |
| 3.6 | Скорость | <шаги сред/с ветки и main при 1/2/4 воркерах по счётчикам, доля> |
| 3.7 | Отчёт и дашборд | <тесты (wdl/rank/score, разметка, Chrome, сервер); проверка владельцем с телефона: <ответ>> |
| 3.8 | Совместимость с SP3 | <фикстура SP3: resume, снимки в турнире; конфиги примеров N/N> |
| 3.9 | Распределённый режим | <тесты; localhost smoke; отказ с SP5 для top-k и pulse; журнала нет> |
| 3.10 | Документация | <ARENA_GUIDE (разделы), LEAGUE_GUIDE, README дословно, CLAUDE.md> |
| 3.11 | Slow-набор | <все 14 тестов, время> |

## §3.1 … §3.11
<по разделу на критерий: команда, вывод, node ids, числа>

## Замеры
### Лесенка ботов и размеры турнира (T8.2)
### Приор и гладкость BT (T8.2)
### Стоимость подгонки BT (T8.2)
### Оценка во время обучения и пороги конвейера (T8.2, T8.3)
### Диск (T8.3)
### Скорость (T8.2 и перезамер приёмки)

## Отклонения и замечания
<каждое отклонение от спеки или плана: бюджеты, пороги (только через ruling), пропущенные шаги и почему; поправки контракта A1…; предложения D1–D14 этой части и решения по ним>

## Гигиена
<шаг 9>

## Решения контроллера во время SP4 (все записи «Ruling:» из журнала, по порядку)
Каждая запись: что решено — почему — чем грозит, если решение неверно.
<шаг 10>

## Открытые пункты (отложены, не блокируют слияние)
<с адресатом SP5 / SP6 / «мелкое»; совпадает с «Parked during SP4» в CLAUDE.md>

## Вопросы владельцу
<только то, что требует его решения (например: `confirmed` не проверяется в тесте конвейера — D12; выбор сабмита из неразличимых финалистов); если нет — «нет»>

## Финальное ревью ветки
<ссылки на записи ревью и волны исправлений; что исправлено>

## Приложение. Отложенные замечания по задачам (из журнала контроллера, дословно)
<шаг 10>
```

Fill every section from Steps 1–10 and the T8.2 / T8.3 records. If the open items differ from CLAUDE.md's «Parked during SP4», fix CLAUDE.md in the same commit.

- [ ] **Step 12: Commit, push, publish and ask the owner**

```bash
git add docs/superpowers/reports/ README.md CLAUDE.md docs/
git commit -m "docs: SP4 acceptance report"
git push
```

The controller (main session) publishes the report as a private, phone-readable artifact page (owner's rule in CLAUDE.md «Working with the owner»), together with the link to the published `report.html` of Step 5, and sends the owner a short Russian summary: the criteria table first; each measured value against its threshold; the rulings with measured values (tournament defaults, BT constants, pipeline cadence and trend threshold, disk slack); every deviation; the open items; and the question «Принять SP4 и слить `sp4-selection` в `main`?».

**STOP here.** Wait for the owner's explicit "yes" in their own message. A reply from another agent, a summary or an earlier statement is not acceptance.

- [ ] **Step 13: After the owner's acceptance: status lines**

Update README's SP4 status row (✔, "accepted and merged <date>; acceptance report `docs/superpowers/reports/<date>-sp4-acceptance.md`"), CLAUDE.md «Implementation Status» / «Roadmap» (SP4 done) / «Next step», and the report's header ("**Владелец принял SP4 <date>** (<his words>)"):

```bash
git add README.md CLAUDE.md docs/superpowers/reports/*-sp4-acceptance.md
git commit -m "docs: SP4 accepted by the owner"
git push
```

- [ ] **Step 14: Merge and push (only after the owner's explicit acceptance)**

`--no-ff` keeps SP4 one unit in `main`'s history, so `git revert -m 1 <merge>` undoes it. SP5 runs in parallel: if `main` moved (an SP5 merge), merge it into `sp4-selection` first, rerun Step 1 and the slow suite on the result, and only then merge back; conflicts with SP5 code are resolved with the owner, not silently.

```bash
git checkout main
git pull --ff-only origin main
git merge --no-ff sp4-selection -m "Merge branch 'sp4-selection': SP4 selection and observability"
env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw
env -u PYTHONPATH .venv/bin/ruff check .
git push origin main
```

Expected: the merge completes without conflicts; the suite and ruff pass on the merge commit; the push succeeds. After the merge the controller deletes the SDD workspace (`.superpowers/sdd/2026-10-16-sp4-selection/`; everything durable is in `docs/`).

---

## Contract notes

> **Reconciled (pre-flight, 2026-10-16).** These notes are folded into `00-overview.md` «Contract amendments» A1–A33; the pre-flight rulings P1… are in the SDD workspace `constraints.md`. Where a note below differs from an amendment (for example a requirement met under another name), the amendment wins.

Requirements on Parts 1 and 2, details this part pins down, proposed rulings and proposed amendments to the overview contract. The controller folds accepted amendments into `00-overview.md` («Contract amendments») before the pre-flight scan.

**Requirements on Part 1** (the code of T7.1–T8.3 uses these names; if Part 1 lands them under other names, the implementer adapts and reports):

- **R-P1-1. Reader API of spec block 1** («матчи с фильтрами (агент, версия, раскладка, источник, исход, диапазон `env_steps`)»): `MatchLogReader.matches(*, agent=None, layout=None, source=None, outcome=None, version=None, env_steps=None, tournament=None, limit=1000) -> list[MatchRow]`, newest first (row order, never `ts`); `limit=None` returns every detailed row; `outcome` ∈ `win | draw | loss` relative to `agent`'s team (win: ranks strictly better than every other team). `MatchRow.to_dict()` returns `{uid, source, layout, outcome_kind, episode_length, seed, env_steps, block, worker, replay, seats: [{seat, team, role, agent_id, network_id, kind, version_start, version_end, rating_version, reward, eliminated_step}], teams: [{team, rank, score}]}` (`replay` None unless the env gave a link; the report draws it). `MatchRow.to_result() -> tuple[MatchResult, list[int | None]]` rebuilds the record and the per-seat rating versions (the cross-validation folds of T8.2 feed `ComparisonTable.add_result` / `match_pairs` with them).
- **R-P1-2.** `MatchLogReader.close()` (and context-manager use). Report, dashboard and scripts close every reader they open.
- **R-P1-3. Bootstrap inputs.** `bt.bootstrap_data(reader, layout, *, sources) -> BootstrapData` (rows with Poisson weights for training detail and `counts`, whole blocks for `pulse` / `tournament`, spec block 2) and `BootstrapData.from_table(table, layout, *, sources)` (rows only; the refit-cost measurement). Without the first, T7.1 falls back to Laplace intervals (stated in the report's notes).
- **R-P1-4.** `TransitivityResult.to_dict() -> {flag, g2, dof, p_value, worst_pairs: [{a, b, residual, games}]}` (`flag` = p < 0.01; team labels as strings) and `ScoreSummary.to_dict() -> {"entities": {entity: {n, mean, ci_low, ci_high}}, "compositions": {"a+b": {n, mean, ci_low, ci_high, homogeneous}}}`.
- **R-P1-5.** `colosseum.arena.live.rating_inputs(config) -> tuple[BTOptions, list[str], str | None]`: the options (module constants, `ratings.anchor`), the fixed agents and the default-anchor `RandomBot` of a config — the single place that turns a config into fit inputs (live ratings, report, scripts, the disk test).
- **R-P1-6. `ratings.json` → `bt`:** `{layout: {group label: {"component": int, "anchor": str, "entities": [{entity, rating, ci_low, ci_high, games, lcb, p_best_lineage, p_best_group}]}}}`, every rated entity, ratings as full-precision floats in Elo units. (T7.1 also accepts a bare list per group and `ci: [lo, hi]`.)
- **R-P1-7. Journal sizes.** `LogSizes` fields `file_bytes`, `wal_bytes`, `detail_bytes` (estimate of the detailed training matches the window bounds), `counts_rows`, `counts_bytes`; the `system` record carries the counts-only flag under a key ending in `counts_only` (bool, e.g. `match_log.counts_only`).
- **R-P1-8.** Module constants `bt.PRIOR_SIGMA` and `bt.SMOOTHNESS` (provisional until T8.2 Step 6) and `BTOptions` accepting them.
- **R-P1-9. Final live fit.** `LiveRatings.stop()` runs one last refit after the journal writer has flushed (every written result is in the in-memory table), and the final `ratings.json` is written from it (the hub's close comes after `LiveRatings.stop`). Fits of the same table and options agree within 0.5 Elo whatever the warm start (the disk test compares them).
- **R-P1-10. `run_id` and `run_state.json`.** If T1.3 keeps `run_id` in `run_state.json`, its key is `run_id` (T7.1's `write_run_state` preserves foreign keys); if in `run.json`, nothing to do.
- **R-P1-11. `system` names.** `dropped_match_results` (T1.3) and the monotonic `dropped_reward_episodes` (T3.1) at the top level of the record (the report's tiles read them; everything else is drawn generically by flattened name).
- **R-P1-12. Reopening a journal.** `MatchLog(path, run)` on an existing `matches.db` whose `runs` table already has `run.run_id` appends to it (no duplicate run row, no error): the report's cache test appends a second batch that way, as a tournament writing into a run's journal would.

**Requirements on Part 2:**

- **R-P2-1. Tournament report shape** (`TournamentReport.to_dict()`, `report.json`): `{name, mode, seed, winner?, confirmed?, layouts: {layout: {outcome_kind, groups: {group: {candidates: [{name, display, lineage, score, score_ci, worst: {opponent, score}, bt, bt_ci, p_best, games, eliminated_round}], ranking: [names, best first: stage-B order of the finalists, then the eliminated by later round and score], matrix: {candidate: {field member: {score, ci, games}}}, rounds: [{round, blocks, survivors, eliminated}], select: {winner, confirmed, indistinguishable, diff_ci} (select mode), transitivity: {flag, p_value, ...}, repeated_pairs: [[candidate, field member], ...]}}}}}`; a layout with one role group still has `groups` (one key). `arena_kit.ladder_problems` and the pipeline test read `ranking`, `select`, `candidates[].bt_ci`; the page reads the rest and shows unknown keys raw.
- **R-P2-2. One writer of the tournament dir's report files:** `TournamentDir.write_report(report)` writes `report.json` and `report.txt` (T7.2 adds `report.html` there); the dir has `tournament.yaml`. Without `--run`, `report.json` is under `--output` (the kit finds it with `rglob`).
- **R-P2-3. CLI and sizes:** `colosseum tournament` takes `-c` (optional with `--run`), `-a` (repeatable), `--run`, `--agent`, `--snapshots`, `--field a,b`, `--layout`, `--mode`, `--blocks` (in `halving` / `select` the first round's blocks per composition; default `DEFAULT_HALVING_BLOCKS` there and `DEFAULT_BLOCKS` in `gauntlet` / `round-robin`), `--confirm-blocks`, `--finalists`, `--seed`, `--num-workers`, `--num-envs`, `--deterministic`, `--name`, `--output`, `--pin`; the constants `DEFAULT_BLOCKS`, `DEFAULT_HALVING_BLOCKS`, `DEFAULT_CONFIRM_BLOCKS`, `DEFAULT_FINALISTS` live in `arena.tournament.modes` (as Part 2's header says) so T8.2 sets them in one place. A `select` that ends with `confirmed: false` exits 0.
- **R-P2-4. Pulse counters** in the `system` record: cumulative `pulse_evaluated` and `pulse_skipped` (T8.2's cadence rule reads them).
- **R-P2-5.** T4.1's `distributed_checkpoint_manager(config, base_dir, run_id=None)` applies `max_snapshots` FIFO (Part 2 has it); T8.1 adds the `evaluation.enabled` refusal only if T6.1 did not, and never a second `keep_top_k` refusal.

**Details this part pins down (additions; all additive):**

- **N1. `colosseum.core.run_state`:** `RUN_STATE_FILE`, `STATUSES`, `RunState(status, pid, exit_code, proc_start, extra)`, `write_run_state(run_dir, status, pid, *, exit_code=None)`, `read_run_state`, `process_start_ticks`, `process_alive`, `effective_status` (`running | stopped | finished | failed | unknown`). Written only by `run_training` (not by the benchmark's in-process `Launcher` nor by distributed roles: their run dirs show "unknown").
- **N2. `colosseum.arena.report.data`:** `ReportBuilder` (with `table_min_age_sec`), `ReportLimits`, `SECTIONS`, `FILTERS`, `REPORT_FORMAT`, `RUN_SOURCES`, `MIN_CELL_GAMES`, `MetricsTail`, `thin`, `entity_version`, `wilson`; the report data shape of T7.1.
- **N3. `colosseum.arena.report.render`:** `REPORT_FILE`, `asset`, `embed_json`, `is_run_dir`, `is_tournament_dir`, `tournament_page_data`, `write_tournament_html`; `colosseum report` accepts a tournament dir too (a page with the tournaments section only).
- **N4. `colosseum.arena.report.server`:** `make_server`, `DashboardServer` (`port`, `builder`, `lock`), `TABLE_MIN_AGE_SEC = 30.0`; `/api/<section>` with the section's filters; 404 / 500 as JSON.
- **N5. `scripts/bench_throughput.py`:** `CounterSampler`, `CLOCKS`, `SAMPLE_INTERVAL_S`, `load_tree_config`, `run_one(..., clock=, make_config=)`, `--clock`, `--tree`; the JSON gains `clock` and `tree`.
- **N6. Ladder and kits:** `examples.tic_tac_toe.ladder.EpsilonTicTacToeBot`, `examples.tron.bots.FloodFillTronBot` / `free_area` / `MOVES`; `tests/report_kit.py`, `tests/arena_kit.py`; the scripts `measure_tournament_defaults.py`, `bt_crossval.py`, `measure_bt_cost.py`, `measure_pulse_cadence.py` with their decision functions.

**Proposed rulings ("что — почему — цена ошибки"):**

- **D1. Where the report's BT tables come from.** A `running` run with a live pid: `ratings.json` (`bt`, Laplace); otherwise a fit of the journal with `ratings.bootstrap_samples` replicas, cached until the journal files' size / mtime change; the dashboard re-reads a running journal's table at most every 30 s. — Spec block 5 names exactly this split; a live fit per page load would compete with training for CPU. — Cost if wrong: a dashboard of a long run shows matrices up to 30 s old.
- **D2. Mixed teams.** Matrix and anchor curves count a team only if all its members belong to one lineage (one entity for snapshot cells); other team pairs are left out with a note. — A team of two agents has no single row in a lineage matrix. — Cost: with `teammates: mixed` the matrix shows fewer games than the ratings use.
- **D3. Lineage order** in the heat map: the rating of the lineage's newest entity, descending, unrated last. — "ordered by rating" needs one number per lineage; the newest is what the agent is now. — Cost: a lineage whose newest bucket is under-played sorts low.
- **D4. The per-agent snapshot matrix** shows the first trainable agent unless `agent=` is given (dashboard filter), its versions thinned evenly to `snapshot_matrix` minus the other lineages, which stay as extra rows / columns. — One agent per page keeps the file bounded. — Cost: other agents' snapshot matrices need the dashboard.
- **D5. Anchors section.** Without pulse: the config's scripted / frozen agents from training games, marked "biased by PFSP"; with `evaluation.enabled`: `evaluation.field` from pulse games only (no fallback to training games while pulse has no data). Stored snapshots without pulse games are listed as "not evaluated". — Spec block 5 and block 6. — Cost: a pulse run shows an empty anchors section until its first evaluation.
- **D6. Run states.** `finished` for exit codes 0 / 130 / 143 (final checkpoints were saved), `failed` otherwise; `stopped` is derived by readers when a `running` file's pid is gone or reused (Linux start ticks). — The spec's three states plus what the pid check reveals. — Cost: a SIGINT-stopped run reads "finished (exit code 130)", not "interrupted".
- **D7. "Few games"** in the heat map: fewer than 20 comparisons (the G² pair threshold of spec block 2), drawn hatched with the count in the tooltip; cell intervals are Wilson on the pair count. — One threshold for "too few to read" across the report. — Cost: FFA cells (fractional weights) get slightly too wide intervals.
- **D8. Matches section.** The latest 1 000 matching the filters (the journal filters, R-P1-1); the outcome filter needs an agent; a replay link is drawn as a link unless it is a `javascript:` URL. — Spec block 5 limits; outcome is relative to someone. — Cost: none known.
- **D9. Journal flags and free disk.** The counts-only flag is found by key suffix in the newest `system` record; free disk is measured when the report is built. — No coupling to T1.3's exact field names. — Cost: a renamed flag disappears from the tile silently (the series stays).
- **D10. Throughput by counters.** `bench_throughput.py --clock counters` (default) samples the global env-step counter and the newest records on `time.monotonic()`; `main` is measured by the branch's script with `--tree` and `main`'s own `_make_config`. — Spec criterion 3.6 allows counters when clocks jump; the GPU checks report found `ts` windows fail every run on this machine; one measuring script for both trees removes method differences. — Cost: the 0.25 s sampling adds up to ±0.25 s to a 60 s window (±0.4 %), equal for both trees.
- **D11. Ladders.** tic_tac_toe `2p` (`EpsilonTicTacToeBot`) and tron `3p` (`FloodFillTronBot`), epsilon ∈ {0, 0.25, 0.5, 1}, field `random` + `ref` (epsilon 0.1), candidates never in the field. — The spec's example epsilons; a fixed, separate field makes "true order" the order of the score against the field (measured monotone with gaps ≥ 0.09); tron covers FFA ranks with one composition per block. — Cost: the ladder is easier than real snapshot selection; T8.2's floors keep defaults from shrinking to the ladder's needs.
- **D12. Pipeline test.** Trains from scratch; asserts the trend, the winner against `RandomBot` (≥ 0.95) and the report, prints but does not assert `confirmed`. — A converged run's best snapshots are legitimately indistinguishable; the spec forbids a silent choice, which `confirmed: false` is not. — Cost: a regression that breaks `confirmed` in real runs is caught only by the ladder test (which asserts it).
- **D13. Disk test equality.** The compacted journal's fit is compared with the run's final live fit over the in-memory table (R-P1-9) within 0.5 Elo; no separate "no-delete" journal mode. — The live table is exactly the "full table in memory" of the spec and needs no test-only switch. — Cost: depends on R-P1-9; if T2.2 cannot refit at stop, Part 1 must add the no-delete test mode instead.
- **D14. Measurement rules** (`decide`, `choose`, `decide_cadence`, `decide_trend`, `MAX_SECONDS`, `SLACK_MB`) as written in T8.2 / T8.3, with Part 2's provisional sizes as floors and "most smoothing, then weakest prior within one SE" for the CV tie-break. — Fixed before the numbers, unit-tested; the CV tie-break serves the spec's intent (few-game snapshots lean on neighbours) without a strong pull to the anchor. — Cost: more smoothing than the strict optimum hides real jumps between neighbouring versions slightly.
- **D15. Package data needs no `pyproject.toml` change** (hatch ships every file under `src/colosseum`; checked by building a wheel). — Cost if wrong: an installed (non-editable) package has no assets; T7.2 Step 6 checks the real wheel.

**Proposed amendments to `00-overview.md`:**

- **A-P3.1 — File map, Create:** `tests/report_kit.py` (T7.1), `tests/arena_kit.py` (T8.2), `examples/tic_tac_toe/ladder.py` and `examples/tron/bots.py` (T8.2; the overview's "examples/<game>/ladder.py (or in the existing bots.py)" made concrete: tron has no `bots.py` yet), `scripts/measure_tournament_defaults.py`, `scripts/bt_crossval.py`, `scripts/measure_bt_cost.py`, `scripts/measure_pulse_cadence.py`, `docs/benchmarks/sp4-*.json`, `docs/benchmarks/sp4-throughput/`, `tests/unit/test_sp4_docs.py`. Modify: `scripts/bench_throughput.py` (T8.1: counter clock, `--tree`), `src/colosseum/arena/tournament/run.py` (T7.2: `report.html`), `src/colosseum/arena/tournament/modes.py` and `src/colosseum/arena/bt.py` (T8.2: measured constants).
- **A-P3.2 — Interface contract, `colosseum.arena.report`:** add `ReportBuilder`, `ReportLimits`, `SECTIONS`, `FILTERS`, `REPORT_FORMAT` (data), `write_tournament_html`, `is_run_dir` (render), `make_server`, `TABLE_MIN_AGE_SEC` (server); `colosseum.core.run_state`: `process_alive`, `effective_status`, `exit_code=` keyword. Reason: the dashboard and the tournament hook use them across tasks.
- **A-P3.3 — Test support:** the ladder bots are example bots (`examples/…`, usable by owners as calibration opponents) and the ladder / pulse drivers live in `tests/arena_kit.py`, not in `tests/game_helpers.py` (the overview's last bullet). Reason: they drive the CLI like `pipeline_kit`, which `game_helpers` does not.
- **A-P3.4 — Requirements R-P1-1 … R-P1-12 and R-P2-1 … R-P2-5** become contract text (names and shapes above). Reason: Parts 1 and 2 are written in parallel; these are the names Part 3's code uses.
- **A-P3.5 — Slow suite:** 14 tests after T8.3 (SP3's 10 + ladder × 2 + pipeline + disk), about 20–25 minutes on the reference machine; README / CLAUDE.md counts are updated in T8.4. Reason: the overview's "~7 min" applies until T8.3.

**Task size.** T8.2 is the longest task (code, then about 3 h of measurements). If the controller prefers shorter tasks, split it at Step 4: T8.2a (bots, kit, scripts, rule tests, smoke; Steps 1–4) and T8.2b (measurements and rulings; Steps 5–11). Nothing else needs splitting; T7.2's assets are given complete.
