<!-- Origin: .superpowers/sdd/2026-10-16-sp4-selection/final-review-A.md in the git-ignored SDD workspace (deleted after the SP4 merge); copied verbatim on 2026-10-17. -->
<!-- Frozen record: never edit. -->

# SP4 final whole-branch review — Area A

Journal, ratings, live ratings, league state, retention, result flow into them (incl. FX1), metrics, `colosseum pin`.
Range `1366bec..589d464` (`sp4-selection`); the working tree also has T8.3's uncommitted doc edits, which I did not review.

## Method

- I read the full diffs of `arena/{matchlog,comparisons,entities,bt,live,retention,__init__}.py`,
  `coordinator/{checkpoint_manager,coordinator,ratings}.py`, `league/pfsp.py`, the journal / live / retention / league-state /
  results-drain parts of `launcher.py`, `worker/{match_runner,rollout_loop,rollout_worker}.py`, `core/{config,run_dir,types}.py`,
  `metrics/*`, `algorithms/appo.py` and `cli.py pin`.
- I read the area's SP4 tests and the SP2/SP3 tests SP4 changed (the result-flow contract, writer, matchlog, bt,
  retention / snapshot store, live ratings, league state, journal runs, SP3 resume). I checked them against the spec
  (blocks 1, 2, 4, 7, 8), A1–A23, P3/P14/P15/P20, C1/C2/C3/C8/C9/C10 and the ledger.
- I traced every exactly-once / counted path by hand: worker batcher → results queue → `_drain_results` → writer
  queue → append / compaction / disk guard → stop / abandon / dead writer. I also traced import and resume, and
  checkpoint-dir resume.
- Probes, each single-process, `OMP_NUM_THREADS=1`, under 30 s, scratch in `final-review/A/`:
  1. read-only open of a WAL journal after a clean close (works; SQLite 3.53.1).
  2. compaction cost: 40 000 tic-tac-toe-like matches appended at about 85 000/s; folding 30 000 matches takes
     4 batches, 0.37 s (0.09 s per 10 000-match batch). So compaction does not starve the writer queue on fast games.
  3. Laplace calibration: z = (estimate − truth) / se over 30 seeds of three of the coverage scenarios:

     | scenario | sd(z) | coverage |
     |---|---|---|
     | 1v1 with side and draws | 0.91 | 0.976 |
     | 2v2 | 0.89 | 0.975 |
     | 4-team FFA | 0.74 | 0.989 |

     No interval is too narrow. FFA intervals are about 1.35× too wide (composite likelihood; see M5).
- I ran three of the area's unit test files (retention, comparisons, matchlog writer): 39 passed, 5 s.

## Strengths

- **Exactly once or counted holds on every path I traced.**
  - The batcher counts every result of a dropped batch, and the final report counts the drops of the exit flush.
  - `_drain_results` feeds each result once and stamps `env_steps` only when it is missing.
  - The writer queue counts overflow.
  - Writer errors count the whole batch.
  - A dead writer or a failed open counts everything as dropped.
  - `stop()` counts what was queued and what was in flight.
  - Compaction folds and deletes in one transaction and bumps `compact_generation`; readers retry across it.
  - The disk guard folds straight into counts and marks the results accepted.
  - `import_parent` runs in one transaction with an `imported:<id>` marker, so a second import is skipped with an INFO line.
  - A checkpoint-dir resume imports nothing, and an SP3 run dir gives an INFO line and an empty journal.
  - The in-memory live table is read once, before the writer starts, and is then fed only accepted results. Nothing is
    counted twice in the live ratings.
  - `WriterStats` keeps the invariant written + duplicates + dropped + pending = submitted.
- **The BT math matches the spec.**
  - Bias: `b = Cβ` per team index, summing to zero.
  - Groups: the role union-find bridges groups through any entity that played both roles; a self-play row that
    identifies `b` correctly counts as a bridge.
  - Components: per-cell anchors with the full rule chain and notes.
  - Dense Laplace up to `DENSE_LIMIT` = 2000 parameters, with exact `Var(θ_e − c·θ)`, Cholesky draws and
    `cov(δ) = H⁻¹`.
  - Perturb-and-solve: `cov(r) = XᵀWX + I/σ² + s·DᵀD = H`, 400 draws, a seed per layout from CRC32.
  - G² is refit on the tested pairs (near-MLE, no smoothness), with exact rank-based degrees of freedom.
  - The bootstrap is Poisson for counts and matches and a multinomial over keys per stratum for blocks, the same
    draw for every contribution of a key. The decomposition check refuses data that does not match the fitted table.
  - Score layouts get a normal CI and no interval below two matches.
  - The coverage tests follow C1 and state their < 1e-3 rationale.
- **Retention matches spec block 4.** The pure `retain` protects keep_last, the current run's final (by `run_id`),
  pinned snapshots and the snapshot just saved. Then come keep_every, top-k by LCB, and new snapshots protected by
  `top_k_min_games`, then the cap order a/b/c. `scores=None` ("no ratings yet") differs from `{}`, and the 2× cap
  before the first refit is the ruled deviation. The immediate first refit on a resume, together with the
  "no source → None" rule of `Coordinator.snapshot_scores` during the pool import, closes the T4.1 bug class. A test
  checks this (`test_imported_rated_snapshots_survive_until_the_first_refit_of_a_resumed_run`). Pins are re-read from
  disk before every decision and written by atomic replace, so a hard-linked parent copy is not touched.
- **Threads.**
  - The writer opens its own connection in its thread; the launcher's startup connection is closed before the writer
    starts.
  - The live-ratings thread has no SQLite; it copies a locked table and publishes immutable snapshots.
  - Retention and `on_evict` run on the main loop.
  - Every error is logged at ERROR (the traceback once) and counted. Nothing is swallowed silently.
- **SP3 compatibility.**
  - New `MatchResult` / `SeatResult` / `Lineup` fields are appended with defaults.
  - SP3 snapshots without `run_id` / `pinned` are imported as ordinary snapshots.
  - SP3 run-dir resume is tested end to end.
  - The changed SP2/SP3 tests change only what the spec changed: the `bt` key, `pfsp_summary`, finals of parent
    runs as ordinary snapshots, batched results.
- **Metrics.**
  - `rho_trunc_mass` / `c_trunc_mass` use the trace ρ and the bars V-trace applies, masked to ACT slots, and are
    exactly 0 with `unit_trace: none`.
  - The cumulative counters take the maximum per worker and sum over every worker that ever reported.
  - The old metric names survive only in historical text (C10) and in the report's legacy map.

## Issues

### Critical

None.

### Important

None. I tried to substantiate several candidates (writer starvation by compaction, live-table double counting on a
resume, eviction of the parent's rated snapshots on a resume, consistency of reads across a compaction, stale
blocks via the import). Each was disproved by reading the code or by a probe.

### Minor

**M1. The abandon WARNING misstates the accounting** (`arena/matchlog.py:1223-1225`; ledger T1.3 deferred).
- When `stop()` gives up on a stuck thread, the in-flight batch is counted as dropped, but the thread's `append` may
  still commit. The batch is then journaled and also counted in `dropped_match_results`. The over-count is at most
  `WRITE_BATCH`; `test_a_batch_still_in_flight_at_the_stop_deadline_is_counted_as_dropped` pins this behaviour.
- The WARNING says the results "are counted as dropped". When the thread is stuck in compaction, it reports "the 0
  results it was writing".
- Realistic trigger: `BUSY_TIMEOUT_MS` (10 s) is longer than `SHUTDOWN_GRACE_SEC` (7 s). A writer waiting on a
  tournament process's write lock at shutdown is abandoned.
- Fix: word the WARNING and the docstring as "counted as dropped; they may still be committed (an over-count of at most
  WRITE_BATCH)", and name compaction when `in_flight == 0`.

**M2. The `version_start` docstring does not say it is read at reset** (`core/types.py:294`,
`worker/match_runner.py:577`; ledger T1.1 deferred; CLAUDE.md lists it as "due in the final fix wave").
- `version_start` is read right after the reset, before that step's weight sync, so an episode's first decision can
  use weights one sync newer. The SeatResult / MatchRunner docstrings should say "version at reset".

**M3. Startup race: workers play their own factory weights as version 0 until the learner's first push**
(`worker/rollout_loop.py:203-212`, `learner/learner.py:169`).
- The worker builds a fresh `latest` model and polls the weight queue once without waiting. The learner pushes its
  initial weights only after building its algorithm and applying resume or init.
- If the worker is ready first, it plays up to `rollout.weight_sync_interval_sec` (default 5 s) of episodes with
  random weights. The journal records them with `version_start = 0`, so they are rated as bucket `v0`.
- Scenario: the README pipeline (`init.from: bc.pt`), or a run-dir resume of an `init.from` run. The rating of
  `main@v0` (the start-weights entity, and the default anchor when a config has no fixed agent) mixes in random-weight
  games, about 2000 per second on tic-tac-toe.
- The race is pre-existing for training data (V-trace corrects it); its effect on ratings is new. It usually goes the
  learner's way.
- Fix (cheap, but on the worker path): ship the numpy init / resume weights the launcher already holds as the
  workers' initial `latest`, or block the first sync briefly. Otherwise note it in the acceptance report.

**M4. The compact `ratings` record's `best` is the best entity, not the best stored snapshot** (`arena/live.py:337`).
- Spec block 2 asks for "рейтинг его последней корзины и лучшего снимка". `best` can be bucket `v0` (no snapshot) or
  the bucket of an evicted snapshot.
- It is a progress indicator only. Rename it to "best version" in the docs, or restrict it to versions > 0.

**M5. FFA uncertainty is conservative; the G² test has little power in FFA** (`arena/bt.py`; spec «В FFA … приближённые»).
- Probe: sd(z) = 0.74 on the 4-team FFA scenario, so the Laplace intervals are about 1.35× too wide. 1v1 and 2v2 are
  at 0.91 and 0.89.
- G² uses the weighted sums `sw = n/(T−1)`, so in a T-team FFA it is about `1/(T−1)` of the unweighted deviance. It
  rarely flags cycles there.
- Both are the safe direction, and the spec calls them approximate. The coverage test checks only the lower side.
  Recommend a line in ARENA_GUIDE §3 («как читать интервалы») with the measured factor.

**M6. Pulse bootstrap strata include the candidate** (`arena/bt.py:284`, per A11).
- The run-level bootstrap does not pair snapshots on the pulse's common seeds P. Contrasts between snapshots, and
  P(best) / LCB within a lineage, come out wider than the common-seed design allows.
- Spec block 6, «Реплики bootstrap для pulse — по блокам, как у турнира», can be read as paired. A11 is binding and
  the error is conservative.
- Note it for the owner. If pairing is wanted: use the stratum without the candidate component.

**M7. An SP3 config with `checkpoint.keep_last > 100`, or an SP2 config with `pool_size > 100`, no longer validates
without edits** (`core/config.py:627-632`).
- The `max_snapshots` default of 100 and the rule "`max_snapshots < keep_last` → ConfigError" are both spec
  requirements, but together they contradict «конфиги SP3 продолжают работать» for such configs. The error message
  has a clear hint.
- Suggest a line in LEAGUE_GUIDE's SP3 notes. Alternatively, an unset `max_snapshots` could take
  `max(100, keep_last)` with an INFO line (needs the owner).

**M8. `colosseum pin` races with eviction** (`coordinator/checkpoint_manager.py:268`; ledger T4.2 deferred).
- A snapshot evicted between `read_checkpoint_meta` and the write raises a raw `FileNotFoundError`, and the CLI
  prints a traceback.
- A pin that lands between the trainer's `_refresh_pins` and `_evict` is reported "Pinned" while the snapshot is gone
  a moment later. The window is milliseconds.
- Fix: catch `OSError` in `pin_snapshot` and raise a ConfigError ("evicted meanwhile; it may still be in
  tournaments/*/snapshots"), and re-check that the dir exists after the write. This narrows the second case.

**M9. Lineage contrasts are mixed across layouts whose rated version sets differ** (`arena/retention.py:240-258`).
- Each layout's contrast is relative to its own lineage mean. A snapshot rated in only one layout is compared with
  snapshots rated in several against different references.
- Rare: early in a layout, or with zero-weight layouts, which are already documented. Document it, or compute the
  contrasts over the versions common to the counted layouts.

**M10. Leftovers.**
- `tests/unit/test_sp4_bt.py:455`: `assert PairStat(1, 1.0, 1.0).n == 1` is a no-op.
- `checkpoint_manager._link_or_copy` (an alias A23 keeps, used nowhere) and `Coordinator.ratings_snapshot(compact=True)`
  (unused in production, T2.2 deferred). Keep them per the contract, or drop them.

**M11. League-state INFO wording** (`launcher.py:1114`, `coordinator/ratings.py`; ledger T3.2 deferred).
- "(an SP3 run dir)" is also printed for an SP4 run dir that never wrote `league_state.json`.
- Two drop reasons are worded wrongly.
- Cheap text fixes.

## Deferred minors to fix now (ledger references)

**Fix in the single fix wave** (text or small code changes; none touches the training math):
1. **T1.1:** the `version_start` docstring "version at reset" (M2). CLAUDE.md already promises it in the final wave.
2. **T1.3:** the abandon WARNING / docstring "may still commit (over-count ≤ WRITE_BATCH)" and the "0 results"
   wording when stuck in compaction (M1). This is the one place where the "exactly once or counted" text is inexact.
3. **T4.2:** `FileNotFoundError` → ConfigError in `pin_snapshot`, plus a post-write existence check (M8). Raw
   tracebacks in a user command.
4. **T3.2:** the two INFO wordings (M11). Optional, trivial.

**Leave parked:** the rest of my area's ledger items.
- T1.3 `_take` off-by-one; `_match_log_stats()` per iteration.
- T2.1 bias count from the data.
- T2.2 double join, unused `compact=True`, immutability by convention, GIL measurement.
- T3.1 masses under geo_mean and without ACT slots; the masked 0/1e-12 gives 0, no NaN.
- T3.2 surviving PFSP rows after a parent SIGKILL; `forgotten` growth; e2e gaps.
- T4.1 items (documented in ARENA_GUIDE §5).
- FX1: a lost final stats report; a malformed drain.

Each is either unreachable on supported paths or cosmetic.

**Carried-forward checks in my area:**
- T0.1 (final on an interval version): handled by the replace-and-keep-pin path.
- T1.1 → T1.3 (e2e run id in uids): `test_journaled_seats_carry_versions_and_uids_the_run_id`.
- T2.1 → T4.1 (draw-length truncation): `combine_scores`.
- T2.1 → T2.2 (`FINAL_REFIT_MAX_SEC`): ruling.
- T3.2 → T4.1 (SP2 resume flake): budget 30000 / interrupt 6000.

All are closed.

## Recommendations

1. Put the M3 startup race and the M5 / M6 statistical caveats in the acceptance report's open items, unless the
   owner wants M3 fixed now.
2. Add a cheap upper-side calibration guard for the 1v1 Laplace coverage, e.g. `sd(z) ≥ 0.75` over the existing 20
   replications. It would catch a future over-wide regression; FFA stays exempt per M5.
3. Before the acceptance, check that T8.3's pending doc edits (CLAUDE.md / README / ARENA_GUIDE are modified in the
   working tree) keep the FX1 statements consistent with the code: 1 000 batches of at most 256 results or 0.1 s;
   a full queue drops a whole batch and counts it per result.

## Assessment

**Ready to merge: With fixes.** The fixes are the Minor items above (M1, M2, M8, optionally M11) in the single fix
wave. No Critical or Important issue in Area A.
