<!-- Origin: .superpowers/sdd/2026-10-16-sp4-selection/final-review-C.md in the git-ignored SDD workspace (deleted after the SP4 merge); copied verbatim on 2026-10-17. -->
<!-- Frozen record: never edit. -->

# SP4 final whole-branch review — AREA C (report, dashboard, run state, measurements, slow tests, docs)

Reviewer: area C final reviewer, also T8.3's task reviewer (controller ruling, ledger line 210).
Range: `1366bec..167d9cc` (HEAD of `sp4-selection`), area C paths only.
Scratch probes (git-ignored): `.superpowers/sdd/2026-10-16-sp4-selection/final-review/C/probe_layouts.py`, `probe_table_cost.py`.

## What I verified

Verified by reading the code end to end, unless a probe or test is named.

- **Page.** `render.py`, `page.html`, `page.js`, `page.css`.
  - Every `innerHTML` write. All three are SVG builders: data reaches them only through `esc()` or through arithmetic, which coerces strings to `NaN`.
  - Every `h()` attribute write.
  - The replay protocol allow-list.
  - `embed_json`'s escapes of `<`, `>`, `&`, U+2028 and U+2029, and the one-pass placeholder substitution.
  - `html.escape` of the title.
- **Dashboard.** `server.py`.
  - Exact-match routing with no static file path.
  - The Host-header guard.
  - The builder queue and result limits.
  - The stale, building, busy and failed states.
  - Shutdown.
  - Probe in headless Chrome: a live dashboard on a two-layout run dir, before and after one refresh (Important 1).
- **Report data.** `data.py`.
  - Counter ordering.
  - Each size limit.
  - Laplace intervals while running, bootstrap when stopped.
  - SP3 run dirs.
  - The single journal read behind the fit, its bootstrap and the G2 test.
  - The anchors and unevaluated logic.
  - The matrix orientation (`_add` and the `flip` in `_pairwise_anchors`).
  - Probe: the report leaves `matches.db-shm` / `-wal` in the run dir (Minor 3).
- **Run state.** `run_state.py` and `run_training`'s writes.
- **Measurement scripts.** The bench (`CounterSampler`, `--tree`, the pinned `_make_config`), `bt_crossval` (folds, `predict`, `choose`), `measure_tournament_defaults`, `measure_pulse_cadence`, `measure_bt_cost`, and their rule tests.
- **Slow tests.** The three new files against the T8.3 brief, rulings R1–R6, C2, C14 (a) and A32.
  - The T8.3 logs (`t83/*.txt`). The ladder rows are byte-identical between the first run and the slow-suite run, so the ladder test is deterministic.
- **Docs.**
  - `README.md`, `CLAUDE.md`, `docs/ARENA_GUIDE.md`, the SP4 diffs of `docs/LEAGUE_GUIDE.md`, `docs/benchmarks.md` and `docs/GPU_CHECKS.md`, `configs/examples/unit_harvest_league.yaml` and `tests/unit/test_sp4_docs.py`.
  - Claims checked against the code: queue and batch constants, the `Match results:` log line, the validate lines (I ran `colosseum validate` on the league example, with and without pulse and top-k), the retention text, the console `bt` format, `worst_pairs[:5]`, the tournament CLI defaults, the BT constants and the log file names.
  - Counts: 2121 fast + 14 slow + 25 gpu = 2160, which matches the collection.
  - No `⟨T8…⟩` marker is left.
- **Tests run.** The report, render, measure-rules, docs and run-state unit tests: 80 passed.

## Strengths

- **The page is safe by construction.**
  - Data reaches the DOM only through `textContent` / `setAttribute` (`h()`), or through the SVG template code, where labels pass through `esc()` and coordinates through arithmetic.
  - Replay links pass a URL-parser protocol allow-list.
  - The embedded JSON cannot close its `<script>` element.
  - The dashboard serves exactly `/`, `/index.html` and `/api/<SECTIONS>`, so no path traversal is possible. It binds to loopback by default, logs a WARNING otherwise, and answers 403 to a foreign `Host` (DNS rebinding).
- **A non-confirmed winner is never shown as confirmed.**
  - The page tests `r.confirmed === true` at the top. In the group view it draws "not confirmed: best stage B score …" with the indistinguishable finalists.
  - `report.txt` prints `NOT confirmed`.
  - Only `select` produces a selection (`modes.py:391–410`).
- **Report data follows its contract.**
  - Ordering is by `train_step` / `env_steps` / row id (the fixture writes decreasing `ts` to prove it).
  - Every list has its spec limit.
  - A live pid gives Laplace intervals from `ratings.json`; a stopped run gets a bootstrap fit of the journal, cached by the journal's signature.
  - The fit, its bootstrap and the G² test share one journal read (`BootstrapData.to_table`).
  - Non-finite values become `null`.
  - SP3 run dirs give notes instead of errors.
  - "Unevaluated" is decided from the journal, so the parent run's pulse games count.
- **The dashboard's concurrency design is sound.**
  - One builder thread owns the non-thread-safe builder.
  - Requests that ask for the same build share it.
  - The pending queue and the result cache are bounded (32 / 64).
  - A slow cold fit answers stale or 202 "building" instead of blocking.
  - A dead builder answers 500 rather than hanging.
- **The measurements are honest.**
  - Every decision rule is fixed in the code before the data and unit-tested: `decide`, `choose` with the paired SE as ruled, `decide_cadence`, `decide_trend`.
  - The bench's counter clock samples the launcher's shared `mp.Value` env-step counter on `time.monotonic()`, which is the same mechanism on `main` (`env_steps_done` is unchanged since `1366bec`).
  - `bt_crossval.predict` uses the model's own linear predictor across rating cells.
- **The slow tests assert what criteria 3.3, 3.4, 3.5 and 3.11 need, and their thresholds are the rulings' values.**
  - The ladder is deterministic (identical rows in two runs), so it has no flake risk. Its time bound has a 10× margin.
  - Pipeline thresholds sit far from the observed values: rho ≥ 0.85 against 0.982–1.000, gain > 0 against ≥ 1375 Elo, vs random ≥ 0.95 against 1.000.
  - The stage-B freshness assertion of C14 (a) holds by construction of the seed domains.
- **The docs carry measured facts with sources.**
  - Every T8.3 number traces to `task-T8.3-report.md`.
  - The FX1 text is consistent in all four places.
  - `benchmarks.md` marks the throughput table as pre-FX1, and the re-measure is due at acceptance.
  - Every carried-forward note from the ledger is present:
    - T4.1: layout weights; top-k on short runs.
    - T5.1a: the frozen `.pt` overlay; the SP3 dedup items closed.
    - T5.2: FFA neighbour positions.
    - T7.2: the unbounded tournaments section.
    - T8.1: the console-only distributed INFO.
    - T8.2b: the prior caveat.
    - FX1: the headroom beyond 4 workers.

## Issues

### Critical

None.

### Important

**I1. Live dashboard: after the first refresh, Ratings and Results matrix are locked to one layout, and there is no control for another agent's snapshot matrix.**
- **Where:** `src/colosseum/arena/report/assets/page.js:623–627` (`query`), `:315/:318` and `:365/:367` (layout selects that only re-render from `DATA`), `:362–386` (`R.matrix` has no agent select).
- **What is wrong.**
  - In live mode `tick()` fetches each section with `query(name)`. For `ratings` and `matrix` that query includes `layout`, and `STATE.<section>.layout` is always set, because `st()` defaults it to the first layout at the first draw.
  - The server then returns only that layout (`data.py:446`, `:638`, as `FILTERS` allows).
  - The re-drawn section builds its layout `<select>` from the layouts in the payload, which is now one. Changing the select only re-renders from `DATA`.
  - From the first tick on, the user can never see another layout's ratings or matrix.
  - Separately, the snapshot heat map shows the first trainable agent only. The page has no agent control, although the API supports `api/matrix?agent=`.
- **Why it matters.**
  - Spec block 5 requires the Ratings table "фильтр по агенту и раскладке" and the snapshot map "по снимкам выбранного агента". Criterion 3.7 says the dashboard shows "ту же страницу и данные вживую".
  - Pre-flight ruling D4 accepted the static report's one-agent matrix with the cost "other agents' snapshot matrices need the dashboard", but the dashboard page cannot show them.
  - `docs/ARENA_GUIDE.md:433` says "остальные агенты — в дашборде, фильтр `agent`", which is true only for hand-typed API URLs.
  - Every multi-layout game is affected: tron 2p/3p/4p, `tic_tac_toe_multi.yaml`, any competition with several player counts. Owner priority 5 names exactly these games.
- **Failure scenario (reproduced).**
  - Setup: a run dir with layouts `2p` and `3p` (`report_kit.write_run_dir("wdl")` plus `make_results("rank")` appended to its journal), then `make_server(port=0, refresh_sec=1)`, then headless Chrome `--dump-dom`.
  - Result at `--virtual-time-budget=500`: the Ratings and Matrix layout selects list `['2p', '3p']`.
  - Result at `--virtual-time-budget=9000` (after a refresh): they list `['2p']` only, and the live status shows "last …".
  - Probe: `final-review/C/probe_layouts.py`.
  - For a real tron run, after 10 s the dashboard can show only the 2p ratings and matrix. The 3p / 4p ratings, the anchors' cells by layout and every non-first agent's snapshot matrix are unreachable from the page.
- **Fix (small, page only).**
  - In `query()`, drop `layout` from the `ratings` / `matrix` keys and choose the layout client-side. The full payload is already bounded per layout, and the server filter stays for API users.
  - Alternatively, make the layout select call `refresh()` in live mode, and have the server always return the list of layout names.
  - Add an `agent` select to `R.matrix`, shown in live mode, over the trainable lineages of the lineage axis, with `refresh("matrix")` on change. This also closes T7.1's deferred "matrix agent filter does not filter score cross-play", if the score branch applies it.
  - Add the missing test: a live-page DOM test (`make_server(port=0, refresh_sec=1)` on a two-layout run dir, headless Chrome with a budget past one refresh). Assert both layouts remain options and that the matrix agent control produces `?agent=`. Today no test exercises `page.js` in live mode (`test_sp4_dashboard.py` tests only the HTTP API).
  - Fix ARENA_GUIDE:433 to match.

### Minor

**M1. Disk slow test: the end-state check is weaker than it needs to be (R3), and two cheap exact checks are missing.**
- Where: `tests/integration/test_sp4_disk_bound_slow.py:76–80, 92–100`.
- **R3 is sound but loose.** The end bound `detail ≤ window + SLACK_MB` (14 MB) plus `total ≥ 3 × detail kept` still catches:
  - compaction that never runs or never deletes (ratio 1);
  - compaction that drops or corrupts the counts (the ratings would move by many Elo).
- **Not caught:** a compaction that keeps a window up to about 4–6× too large. With about 60k matches, detail stays ≤ total/3 (≈ 6 MB) and ≤ 14 MB, and the peak stays under its bound. The compaction contract tests cover the window arithmetic, so this is not a hole in the suite, only in this test.
- **Cheap strengthenings:**
  - assert per-entity games equal, exactly: `live` rows carry `games`, so compare them with `from_journal.games`. This is an exact "counts complete" check, stronger than 0.5 Elo;
  - track the maximum number of `ckpt_v*` dirs per agent in the existing 0.5 s poll thread (spec 3.5: "не превышает", during the run, not only at the end);
  - optionally, after the run, call `MatchLog(db, None).compact(window_bytes=WINDOW)` on the copy and assert `detail ≤ 1.1 × window` with unchanged ratings. That is the brief's original 1.1× check, applied where it is valid.

**M2. `MetricsTail` memory grows with the run's length.**
- Where: `src/colosseum/arena/report/data.py:143–186`.
- It keeps every `train` / `system` / `episodes` record as a Python dict, and the first poll reads the whole remaining file into one `bytes` and splits it.
- Estimate: at the default `log_interval` 10 and about 20 updates/s that is about 170k train records per day per agent. A multi-day run's dashboard or report then holds GBs, and the first poll briefly holds about 2× the file size.
- The docstring's "bounded memory" claim (data.py:21–23) is about kinds, not counts.
- Suggestion: keep per-agent numeric columns, thinned on ingest (stride doubling by counter), and read in fixed-size chunks.

**M3. A "read-only" report leaves `matches.db-shm` / `matches.db-wal` in the run dir.**
- Verified by probe: after `build_report_data` on a finished run dir, both files exist.
- SQLite creates them for a `mode=ro` connection to a WAL database. By the same mechanism, a run dir on a read-only mount, or one owned by another user, gives the note "the match log cannot be opened".
- The claims at `data.py:8` ("nothing in the run dir is written") and `docs/ARENA_GUIDE.md:415` («ничего не меняет в папке запуска … кроме самого `report.html`») are slightly inaccurate.
- Fix: a doc sentence. Opening with `immutable=1` is unsafe while a writer may exist.

**M4. `colosseum report` is all or nothing.**
- `ReportBuilder.build()` (`data.py:352–355`) does not isolate sections, so one failing section aborts the whole static report with a traceback.
- The dashboard isolates sections (500 for that section only).
- Example: an unexpected `bt.bootstrap` ValueError on an odd journal gives no page at all, although eight sections were fine.
- Suggestion: catch per section in `build()` and put a `{"notes": ["This section failed: …"]}` placeholder, as `server._page` does. Exit code 0 with the note, or 1 after writing the page: a ruling.

**M5. Tournaments section unbounded** (T7.2 deferred; documented in ARENA_GUIDE §7). Park, see below.

**M6. `--refresh 1` refreshes every 2 s.**
- Where: `page.js:647` (`Math.max(2, …)`); the CLI help says "Seconds between page refreshes" and ARENA_GUIDE says "раз в `--refresh` секунд".
- Fix: either `IntRange(min=2)` on the CLI, or say so in the help.

**M7. The pipeline slow test's report assertion is weak.**
- Where: `tests/learning/test_sp4_pipeline_slow.py:75`.
- `f'"{TOURNAMENT}"' in page` matches any JSON string equal to `"sel"`.
- Suggestion: parse the embedded data (as `test_sp4_report_render._data_of` does) and assert `sections.tournaments.items` has an item named `sel` with a `select` group. Cheap.

**M8. The Ratings table's `LCB` column is on another scale than the BT column next to it.**
- Where: `page.js:355`. `LCB` is a lineage-contrast quantile (ARENA_GUIDE §3), shown beside BT ratings with no hint, so a reader will take it as the lower bound of the rating.
- Fix: a `title` on the header ("lower bound of the deviation from the lineage mean; used by top-k").

**M9. CLAUDE.md state lines for T8.5.**
- `CLAUDE.md:411` («Next step») still lists the slow tests (T8.3) and the drops diagnostic as pending.
- `CLAUDE.md:311` (Throughput) quotes the SP4 numbers as criterion 3.6 without saying they predate FX1 and are re-measured at the acceptance (ledger line 183; `benchmarks.md` says so).
- The parked learner metrics-queue item (ledger line 184: "named in the acceptance report") is not yet in «Parked during SP4».
- All three belong to T8.5's status refresh, as the brief expects.

**M10. `measure_bt_cost`'s pass rule is inline and untested.**
- Where: `scripts/measure_bt_cost.py:85`.
- It is three comparisons, fixed in the docstring before the measurement. Mentioned for completeness; no action needed.

## T8.3 task review (controller ruling: area C reviewer)

**Brief against what was built.**
- All four tests exist with the brief's names, constants and `pytestmark` timeouts.
- Deviations, each a recorded ruling:
  - R3: end bound;
  - R4: `match_refresh_interval_sec: 1.0`, `len(live) ≥ 2`;
  - R5: the stage-B freshness assertion from the journal;
  - R2: `SLACK_MB` from the first run of the corrected test;
  - added printouts, and `with MatchLog.open_readonly(...)` (R-P1-2).
- No production code changed. The docs commit fills all 7 markers with the report's numbers.

**Constraints.**
- C14 (a): the pipeline test asserts stage B on fresh seeds (winner's B blocks > 0, 0 reused seeds) and the winner vs random ≥ 0.95, and prints `confirmed` without asserting it. Met.
- C2: the compacted journal's fit is compared with the final live fit over the in-memory table. Met. The final refit cannot be skipped at about 50 entities: under 2 s, dense path, and `finish()` joins a running refit first.
- A32: 14 tests. R6 corrects the duration to about 10 min (measured 567 s). Met.

**Rulings R1–R6:** sound.
- R2: 13 vs 15 MB changes nothing material; both leave about 6 MB of margin over the largest excess seen.
- R3: see M1. It weakens only the window-scale sensitivity, which the contract tests cover. The ratio check and the rating equality still catch a broken compaction.
- R4: needed. Without it the equality check was vacuous.

**Flake risk per threshold.**

| Test | Check | Risk |
|---|---|---|
| ladder | order / winner / confirmed / BT | none (deterministic) |
| ladder | `MAX_SECONDS` 60 | about 10× over the measured 6 s |
| pipeline | `RHO_MIN` 0.85 | low (calibration 0.982–1.000, T8.3 1.000 twice) |
| pipeline | `GAIN_MIN` 0 | ≥ 1375 Elo observed |
| pipeline | vs random 0.95 over 60 matches | 1.000 observed |
| pipeline | stage-B freshness | by construction |
| disk | ratio ≥ 3 | 7.7–109 observed; about 4.5 on a 2× slower machine |
| disk | peak and end bounds | about 6 MB margin |
| disk | 0.5 Elo | 0.0000 observed (same table and options) |
| disk | `len(live) ≥ 2` | 49–52 observed |

**Conventions.**
- Files only under `tmp_path`.
- At most 2 worker processes per test: disk 2 rollout workers; pipeline 1 worker + pulse, then a tournament with 2.
- Durations by `time.monotonic()`; no `ts` assertions.
- The report runs through `run_in_session`.

**Verdict:** Approved. M1 and M7 are optional strengthenings for the fix wave.

## Deferred minors to fix now (single fix wave)

**My area, cheap (one line or text each):**
- T7.1: `evaluation.agents: []` is listed as "every trainable agent unevaluated" (`data.py:737`: use `is not None`, as `pulse_plan` does). One line.
- T8.2a: `tests/unit/test_sp4_ladder_bots.py:7–8` says z ≥ 3.9 and ≤ 5e-5. The values are 0.116 / 0.030 = 3.87 and about 5.4e-5 per gap, about 3.3e-4 for six gaps, so the < 1e-3 conclusion stands. Text only: the docstring of a statistics test must state its numbers right.
- T7.1: "the matrix agent filter does not filter score cross-play". Fold it into the I1 fix if the new agent control drives the score branch too.

**Park (documented or cosmetic):**
- T7.2: the unbounded tournaments embed. It is documented in ARENA_GUIDE §7, and a cap needs an ordering by counter (journal block row ids), which is more than a one-liner.
- T7.2: per-anchor shares are not drawn (the spec asks only for the category mix).
- T7.1: score caps apply before the agent filter; the fit reads fresh while the table is cached.
- All T8.1 bench minors.
- T8.4: none left (all fixed in its round).

## Recommendations

1. Fix I1 in the wave together with its live-page DOM test (the area's one real defect). Re-check that ARENA_GUIDE §7 then describes the page as it is.
2. Take M1's exact per-entity games equality and the max-snapshots-during-run sample into the disk test: about 6 lines, and no new measurement, because the thresholds do not change. M7 likewise.
3. Leave M2–M4 and M6–M8 as recorded minors unless the wave has room. M4 (per-section isolation of the static report) is the most useful of them for real users.
4. T8.5: refresh CLAUDE.md «Next step» / Throughput / «Parked during SP4» (M9), and copy R1–R6 and the area C deferred minors into the acceptance report.

## Assessment

**Ready to merge: With fixes.**
- Fix I1, with its test, before the acceptance. Criterion 3.7's "the same page and data live" fails on every multi-layout game until then.
- Everything else in area C is correct, safe and well-tested. The slow tests and the docs match the rulings and the code after FX1 and T8.3.
