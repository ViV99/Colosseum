<!-- Origin: .superpowers/sdd/2026-10-16-sp4-selection/fix-wave-A-report.md in the git-ignored SDD workspace (deleted after the SP4 merge); copied verbatim on 2026-10-17. -->
<!-- Frozen record: never edit. -->

# SP4 final fix wave, part A: report

Worktree `/home/viv/dev/repos/Colosseum-wt-fwa`, branch `sp4-fw-a` from `167d9cc`. Nothing pushed. Python was run as
`env -u PYTHONPATH PYTHONPATH=<worktree>/src OMP_NUM_THREADS=1 .venv/bin/python`; `colosseum.__file__` =
`/home/viv/dev/repos/Colosseum-wt-fwa/src/colosseum/__init__.py`.

**Status: DONE**

## Commits

| commit | subject | findings |
|---|---|---|
| `0ff1414` | fix: workers wait for every agent's first weights before any match (no random-weight v0 games in the journal) | A-M3 |
| `dd86575` | fix: the first-weights wait polls every 0.1 s (fewer weight-store RPCs while a distributed learner starts) | A-M3 |
| `e875460` | fix: an unset checkpoint.max_snapshots follows keep_last over 100, so SP3 configs keep working | A-M7 |
| `9ea03bc` | fix: the compact ratings record's best is the best stored snapshot | A-M4 |
| `acad578` | fix: colosseum pin reports a snapshot evicted while it is pinned as a config error | M8 |
| `78bdfc9` | fix: the journal writer's abandon WARNING says an in-flight batch may still be committed | M1 |
| `1257edc` | docs: version_start is the weights version at the env's reset | M2 |
| `fd17698` | test: drop a no-op assert from the BT compaction test | M10 |
| `e8909a7` | fix: league-state INFO lines name the real reason for each drop | M11 |
| `c9cda0a` | docs: ARENA_GUIDE on conservative FFA intervals, weak G2 in FFA and per-layout lineage contrasts | M5, M9 |
| `76c4041` | docs: CLAUDE.md parked SP4 items after fix-wave part A | M1, M2, M3, M4, M7, M8, M11 (doc) |

## Per finding

### M3 (ruling A-M3): startup race — fixed
- **Design chosen: an explicit option.** `RolloutLoop(wait_for_first_weights: Callable[[], bool] | None = None)` takes
  a stop predicate. None, the default, keeps the single non-blocking initial poll, for in-process loops and tests.
  With a predicate, the initial sync waits **before the MatchRunner is built**, so no env is reset and no episode
  starts. This matters because `version_start` is read at reset (M2), so waiting after the construction would still
  journal the first episodes as v0.
  - The wait polls only the agents still missing, in slices of `FIRST_WEIGHTS_POLL_SEC` = 0.1 s.
  - It logs one INFO after `FIRST_WEIGHTS_INFO_SEC` = 5 s: "Worker W: waiting for the first weights of [...] from
    their learners; no match starts before they arrive".
  - It stops when the predicate is true. The factory weights then stay, and `run` with the same sticky stop returns
    before its first step, so no match is played.
- `rollout_worker_process(wait_for_first_weights: bool = False)` passes `stop_event.is_set` when the flag is set.
  `launcher._worker_main` and `distributed._dist_worker_main` pass `True`: on in the worker processes, local and
  distributed.
- Tests: new `tests/contract/test_sp4_first_weights.py` (8 tests).
  - The default starts at once, as v0.
  - A waiting loop polls 5 times with no reset (each poll records the games' reset count: always 0). It resets
    after the payload arrives, loads the payload's tensors, and every result has `version_start` 7 / 3, never 0.
  - A stop during the wait ends it, and the loop then plays nothing: 0 env steps, no results, no chunks.
  - One INFO line.
  - The worker process with the wait on, in-process: 0.5 s with no env step, chunk or result, so nothing reaches the
    journal. After the push, the results have `version_start` = the pushed version.
  - A worker process stopped while it waits exits within 5 s.
  - Wiring for `_worker_main` and `_dist_worker_main`.
- Adapted tests (they drive `_worker_target` without a learner): `tests/integration/test_sp2_pipelines.py` (2 tests)
  and `tests/integration/test_sp3_fixed_players_worker.py` now push the learner's initial weights first.
- Docs: one ARENA_GUIDE §2 line. CLAUDE.md parked list: "done in the final fix wave".

### M7 (ruling A-M7): SP3 compat of max_snapshots — fixed
- `CheckpointConfig._check_cap`: when `max_snapshots` is not in `model_fields_set` and its default (100) is below
  `keep_last`, the cap becomes `keep_last`, with one INFO line.
  - An SP2 `pool_size` goes through the before-validator into `keep_last`, so it is covered too.
  - An explicit `max_snapshots < keep_last` stays a ConfigError. An explicit `null` stays "no cap".
  - The raised value is stored, so a re-validated dump (`get_agent_config`) logs nothing more. The test checks this.
- Tests in `tests/unit/test_sp4_sp3_compat.py`:
  - An SP3-form config (a copy of the SP3 `tic_tac_toe.yaml` with `keep_last: 150`) loads with `max_snapshots`
    150 and exactly one INFO line, and `colosseum validate` exits 0 printing "at most 150 per agent".
  - SP2 `pool_size: 150` behaves the same.
  - A parametrized test of the cases that do not change: `keep_last` 50 → 100; explicit null → None; explicit 200 →
    200; no INFO in any of them.
  - An explicit 120 with `keep_last` 150 is a ConfigError.
- Docs: ARENA_GUIDE §5 (yaml comment, the cap paragraph, §8 "явный"), LEAGUE_GUIDE §12 yaml comment, README config
  table.

### M4 (ruling A-M4): compact `best` — fixed with the real stored set
- **The stored set is known cheaply:** `bt_summary` is called by the metrics hub in the main process, not by the
  live thread, and the main process's `CheckpointManager` keeps an in-memory index. So `best` is the **best stored
  snapshot**, not the "best version" fallback.
- `bt_summary(snapshot, agents, stored=None)`: `best` is the highest-rated (or highest mean score) entity of the
  agent with version > 0 and version in `stored[agent]`.
  - `stored=None` (a hub without a source) means every version > 0.
  - A layout with no rated stored snapshot has no `best` key.
- `MetricsHub(stored_snapshots=callable)` calls it at a tick, inside the existing failure guard.
- `Launcher._stored_snapshot_versions()` reads `coordinator.checkpoint_manager.list_checkpoints(aid)` for the
  coordinator's trainable agents, and returns None before the coordinator exists.
- Tests (`tests/unit/test_sp4_live_ratings.py`):
  - A table where v0 is the strongest entity: without a stored set, best = v200; with `{100}` (v200 evicted),
    best = v100; with `{0, 300}`, no best. `latest` is always v200.
  - The hub writes the stored-set best.
  - The launcher reports `{agent_0: {20, 40}}` after two saves, and None without a coordinator.
- Docs: README metrics table, ARENA_GUIDE §3 «Где видно».

### M8: pin race with eviction — fixed
- `pin_snapshot` catches `OSError` and `ConfigError` from `set_pinned`.
  - If the dir is gone, it raises a ConfigError: "snapshot X@vN was evicted by the training run while it was being
    pinned; a copy may still be in <run>/tournaments/*/snapshots/ ...; see colosseum pin <run> --list".
  - Otherwise it re-raises the ConfigError, or wraps the OSError ("writing meta.json failed").
  - After a **pin** write, it re-checks that the dir still exists. After an unpin write it does not: an eviction
    there is the expected outcome.
- Tests (`tests/integration/test_sp4_pin_cli.py`):
  - Eviction before the write and after the write, both through the CLI: exit 1, a SystemExit and not a raw OSError,
    no traceback, no "Pinned", the hint present.
  - An unpin followed by an eviction exits 0 with "Unpinned".
- Docs: the `pin` CLI docstring, ARENA_GUIDE §5 «Закрепление».

### M1: abandon WARNING accounting (T1.3 deferred) — fixed
- The writer thread records what it is doing in `_activity`: opening, waiting for results, writing N results, adding
  to the live table, compacting, reading sizes, closing.
- When `stop()` gives up, there are two WARNINGs:
  - With results in flight: "(it was writing a batch of N results); they are counted as dropped
    (dropped_match_results), but the thread may still commit them: an over-count of at most 2000 results".
  - With nothing in flight: "(it was compacting the journal; no results in flight, e.g. waiting for the journal's
    write lock)". The old "the 0 results" text is gone, and this WARNING does not claim any drop.
- The docstrings of `WriterStats` (the one exception to the invariant), `MatchLogWriter` (accounting) and `stop` say
  the same. So does ARENA_GUIDE §2 «Потери».
- Tests (`tests/unit/test_sp4_matchlog_writer.py`):
  - The in-flight test now asserts the new wording and the bound.
  - New: a writer stuck in compaction at the deadline. Dropped 0, the WARNING names compaction and "no results in
    flight", and contains neither "0 results" nor "counted as dropped".

### M2: `version_start` docstring — fixed (text)
- `SeatResult`, the `MatchRunner` module docstring and the `_EnvState` comment, and the `RolloutLoop` docstring now
  say: the version at the env's reset; the first decision can use weights one sync newer.
- The same line is in ARENA_GUIDE §2. No test: text only.

### M10: leftovers — fixed
- Dropped `assert PairStat(1, 1.0, 1.0).n == 1` from `tests/unit/test_sp4_bt.py` and its now-unused import.
- `_link_or_copy` and `ratings_snapshot(compact=True)` are kept (contract, A23).

### M11: league-state INFO wording (T3.2 deferred) — fixed
- Launcher: "has no league_state.json (an SP3 run dir, or a run that never saved it, e.g. killed before its first
  metrics tick); ...". The `Coordinator.load_league_state` docstring matches.
- `PfspStats.load_state_dict`: an owner still in the config but no longer trainable gets its own INFO ("owners that
  are no longer trainable agents"). The return value is unchanged: the union of the dropped agents.
- `RatingBook.load_state_dict`: a layout whose outcome kind changed gets its own INFO ("... whose outcome kind
  changed are dropped: 2p (saved rank, now wdl)"), separate from "layouts this game does not have".
- Tests (`tests/unit/test_sp4_league_state.py`, `tests/integration/test_sp4_sp3_resume.py`): both PFSP reasons are
  asserted. A new test covers a changed outcome kind: its own line, the layout not restored. The resume test checks
  the new launcher wording.
- Docs: ARENA_GUIDE §1 resume notes.

### M5: FFA intervals — fixed (doc)
- ARENA_GUIDE §3 «Интервалы и вероятности» has a new bullet:
  - In FFA the intervals are conservative. Probe: sd(z) 0.74 on 4-team FFA, about 1.35× too wide; 1v1 0.91, 2v2
    0.89; never too narrow. The cause is the composite likelihood.
  - G² has little power there: pair weights are 1/(T−1). Read the matrix instead.

### M9: lineage contrasts per layout — fixed (doc)
- ARENA_GUIDE §5, rule 3 (top-k): contrasts are computed per layout, over the snapshots rated in it. So a snapshot
  rated in only some layouts is compared against different references, and its `LCB` is only approximately
  comparable.

## RED / GREEN

All commands run from the worktree with the prefix above; `-p no:cacheprovider`.

- **M3**
  - RED: `pytest tests/contract/test_sp4_first_weights.py`: 6 failed, 2 passed.
    - `AttributeError: ... rollout_loop has no attribute 'FIRST_WEIGHTS_POLL_SEC'` (×3).
    - The process test: `assert (False, 0, 0, 0) == (True, 0, 0, 0)`. The thread died on the unknown kwarg.
    - `KeyError: 'wait_for_first_weights'` (×2, wiring).
  - GREEN: 8 passed (`-W error`).
  - Adapted `_worker_target` tests: 3 passed.
  - Worker-area files (result flow, lineups, worker v2, worker threads, launcher checkpoints, fixed-players wiring,
    distributed roles, DAgger worker): 70 passed.
- **M7**
  - RED: the new compat tests: 2 failed with `checkpoint.max_snapshots (100) is smaller than checkpoint.keep_last
    (150)` (the keep_last 150 and pool_size 150 cases); the 4 others passed.
  - GREEN: `tests/unit/test_sp4_sp3_compat.py` 35 passed. Docs, retention, bench, distributed guards and config
    tests: 252 passed.
- **M4**
  - RED: 3 failed: `assert 'main@v0' == 'main@v200'`; `MetricsHub.__init__() got an unexpected keyword argument
    'stored_snapshots'`; `'Launcher' object has no attribute '_stored_snapshot_versions'`.
  - GREEN: `test_sp4_live_ratings.py` + `test_sp4_live_ratings_run.py`: 26 passed.
- **M8**
  - RED: 2 failed. Before the write: the CLI result held a raw `FileNotFoundError(2, ...)`. After the write: exit 0,
    "Pinned".
  - GREEN: `test_sp4_pin_cli.py` 7 passed.
- **M1**
  - RED: 2 failed with the old text: "...did not stop within 1.0 s; the 2000 results it was writing are counted as
    dropped..." and "...0.6 s; the 0 results it was writing...".
  - GREEN: `test_sp4_matchlog_writer.py` 15 passed.
- **M11**
  - RED: 2 failed. The changed outcome kind was logged as "layouts this game does not have"; there was no "owners
    that are no longer trainable" line.
  - GREEN: `test_sp4_league_state.py` + `test_sp4_sp3_resume.py` 7 passed.
- **M2, M5, M9:** text only. **M10:** test-only. `test_sp4_bt.py` 33 passed, ruff clean.

## Full fast suite and ruff

- Fast suite run once at the end in the worktree (`pytest -m "not gpu and not slow" -q -rw`, at commit `dd86575`):
  **2143 passed, 39 deselected in 659.97 s, zero warnings, zero failures** (no flake re-runs needed).
- `ruff check .`: all checks passed.
- Slow and GPU suites not run: no change touches the training math, the learner or AMP; the worker change is the
  startup wait only (the fast suite's spawned-worker, pipeline and distributed e2e tests cover it).

## Deviations and proposed rulings («что — почему — цена ошибки»)

1. **A-M3: the option is a stop predicate on `RolloutLoop` and a bool on `rollout_worker_process`, default off in
   both.**
   - Why: the ruling allows an explicit option. Defaulting it on in `rollout_worker_process` would make every
     in-process test caller (8 of them) push weights or hang. The production callers (`_worker_main`,
     `_dist_worker_main`) pass `True`, and wiring tests guard both.
   - Cost if wrong: a future third worker entry point that forgets the flag plays factory weights as v0 again, as
     before the fix.
2. **A-M3: a stop during the wait still finishes building the loop** (the MatchRunner resets the envs once), and
   `run` returns before its first step.
   - Why: no half-built object and no new exception path in the worker's cleanup (`finally`). No match is played or
     journaled.
   - Cost if wrong: one env reset during shutdown, inside `SHUTDOWN_GRACE_SEC` for any reasonable env.
3. **A-M3: a worker now waits for as long as its learner takes to start.**
   - Why: this is the ruling. A learner that dies before pushing is a dead child, and the supervisor sets
     `stop_event`. A distributed worker whose learner never publishes waits until stopped, with the INFO line.
   - Cost if wrong: a distributed worker started long before its learner idles visibly instead of playing random
     weights.
4. **A-M4: the real stored set is used** (the checkpoint manager's index in the main process), so the docs say "best
   stored snapshot", not "best version". `stored=None`, a hub without a source, falls back to versions > 0.
   - Cost if wrong: none for the metrics. The WandB keys `ratings/bt/<agent>/<layout>/best/*` are now missing in a
     layout until a stored snapshot is rated.
5. **A-M7: the raised cap is stored in the validated config**, so `config.resolved.yaml` shows `max_snapshots: 150`
   as an explicit value.
   - Why: one INFO line per config, not one per re-validation.
   - Cost if wrong: a later run that reuses that resolved file with a lower `keep_last` keeps the cap at 150 (a
     higher cap, harmless).
6. **M8: the post-write existence check is only for pinning**, not for unpinning. After an unpin, an eviction is the
   intended effect.
   - Cost if wrong: none.
7. **M1: beyond the wording, the thread records its current operation** (a string attribute written by the thread
   only), so the WARNING can name compaction, opening, sizes or the live-table feed.
   - Cost if wrong: one attribute write per operation.
8. **CLAUDE.md:** the parked items fixed here move into "done in the final fix wave" notes, in one separate docs
   commit (`76c4041`) rather than in each fix commit.
   - Why: parts B and C edit the same list, so one small hunk is easier to merge.
   - Cost if wrong: an integration conflict in three lines of CLAUDE.md.
