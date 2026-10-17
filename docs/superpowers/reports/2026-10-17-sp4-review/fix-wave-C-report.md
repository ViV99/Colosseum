<!-- Origin: .superpowers/sdd/2026-10-16-sp4-selection/fix-wave-C-report.md in the git-ignored SDD workspace (deleted after the SP4 merge); copied verbatim on 2026-10-17. -->
<!-- Frozen record: never edit. -->

# SP4 final fix wave — part C report

Worktree `/home/viv/dev/repos/Colosseum-wt-fwc`, branch `sp4-fw-c`, base `167d9cc`. Nothing pushed.
Python as `env -u PYTHONPATH PYTHONPATH=<worktree>/src OMP_NUM_THREADS=1 /home/viv/dev/repos/Colosseum/.venv/bin/python`.
`colosseum.__file__` = `/home/viv/dev/repos/Colosseum-wt-fwc/src/colosseum/__init__.py`.

Status: **DONE**

## Commits (in order)

| SHA | Subject | Findings |
|---|---|---|
| 60b4d8c | fix: an empty evaluation.agents list marks no snapshot unevaluated in the report | T7.1 deferred (`is not None`) |
| 32be417 | fix: the report's metrics tail keeps bounded memory over a run of any length | C-M2 |
| 261324d | fix: colosseum report isolates sections: a failing one is a note, the page is written, exit 1 | C-M4 |
| d0e0905 | fix: the live dashboard keeps its layout and agent choices; the matrix gets an agent select | C-I1, T7.1 deferred (cross-play agent filter) |
| 22d4557 | fix: colosseum dashboard --refresh is at least 2 seconds | C-M6 |
| cdbe814 | fix: the Ratings table's LCB header explains its scale | C-M8 |
| 5a765a3 | docs: a report leaves SQLite's matches.db-shm / -wal in the run dir | C-M3 |
| cbc0ede | test: the pipeline slow test reads the report's tournament item, not any string | C-M7 |
| fa6da3a | docs: the ladder bots test states its statistics right | T8.2a deferred |
| 1bfb158 | test: the disk slow test checks games exactly, the window on a compacted copy, the cap during the run | C-M1 |
| ab84418 | fix: the truncated-entities note no longer points to dashboard filters | follows C-I1 (see deviations) |

## Per finding

- **C-I1 — fixed** (d0e0905, ab84418).
  - `page.js`, live mode:
    - `query()` sends nothing for `ratings` and only `agent` for `matrix` (`matches` unchanged).
    - Ratings (layout and agent) and the matrix layout are chosen on the page from the full answer.
    - `R.matrix` has an `agent` select, live only, with options `[""] + s.agents`. A change calls `refresh("matrix")`, which requests `api/matrix?agent=`.
    - Only the answer to a section's latest request is drawn (a sequence number per section).
    - The live status shows the effective interval, `max(2, refresh_sec)`.
  - `data.py`:
    - The matrix section has a new `agents` field: the config's trainable agents, the same whatever the filter.
    - The score branch applies `_filter_agent`, so `agent=` filters the cross-play too (T7.1 deferred).
  - Docs: ARENA_GUIDE §7 (the snapshot-map limit line and the dashboard paragraph) describes the page after the fix.
  - Tests:
    - `test_sp4_dashboard.py::test_the_live_page_keeps_its_layout_and_agent_choices_across_refreshes`. Headless Chrome against `make_server(refresh_sec=2)` on `report_kit.write_multi_run_dir`: layouts `2p` and `3p`, trainable agents `main` and `second`.
      - An injected script waits for the first completed refresh, then picks Ratings layout `3p` and matrix agent `second` through the selects' change events.
      - `builder.section` is wrapped to record calls and to append a counter note, so every refresh redraws.
      - Asserted: both layouts stay options, `3p` stays selected, the agent select offers `"", main, second` with `second` selected, "Snapshots of second" is drawn, the server saw `("matrix", {"agent": "second"})`, and no ratings or matrix call carried `layout`.
    - `test_sp4_report_data.py::test_the_matrix_lists_the_trainable_agents_and_draws_the_chosen_ones_snapshots`.
    - The cross-play assertion added to `test_score_compositions_are_capped_and_filtered_by_agent`.
  - New `report_kit` helpers: `write_run_dir(trainable=...)`, `as_agent`, `append_results`, `write_multi_run_dir`, `page_data`, `select_groups`.
- **C-M1 — fixed** (1bfb158).
  - The disk test now asserts:
    - the per-entity `games` of the live fit equal the journal fit's exactly;
    - a SQLite-backup copy of the final journal, `MatchLog(copy, None).compact(window_bytes=1 MB)`, has `detail ≤ 1.1 × window`, the same games and the same ratings (`COMPACT_TOL_ELO = 1e-3`);
    - the 0.5 s poll thread tracks the most `ckpt_v*` dirs per agent during the run, and that number is ≤ `max_snapshots` (spec 3.5).
  - Thresholds are unchanged.
  - Run alone twice:
    - run 1, 26 s: games equal, compacted detail 1.00 MB at a 1 MB window, compacted vs journal 0.000000 Elo, live vs journal 0.0000 Elo, at most 5 snapshots, 0 listings over the cap;
    - run 2, 22 s: the same figures.
  - ARENA_GUIDE §2 says what the test checks.
  - The re-list rule is a deviation; see ruling R-C1.
- **C-M2 — fixed** (32be417).
  - `MetricsTail` thins each (kind, agent) series on ingest by its counter (`_Series`). The counter is `train_step` for `train` and `env_steps` otherwise.
  - Stride doubling: one record per counter bucket is kept, the one with the smallest counter. When more than `max_per_series` buckets are kept, the bucket width doubles and pairs of neighbouring buckets merge.
  - The newest record is always kept. Records without a counter: the newest `cap` of them.
  - Bound: ≤ `2·cap + 1` records per series, stated in the docstring (in practice `cap + 1`).
  - `ReportBuilder` passes `2 × series_points`, so the page's own `thin` still gets its 2 000 points. The order and the counter keys are unchanged, because consumers sort by counter as before.
  - Reads go in `READ_CHUNK_BYTES` (1 MiB) chunks, up to the size seen at the poll's start. A line longer than a chunk and a partial last line are both handled.
  - Measured with tracemalloc: a 120 MB `metrics.jsonl` (200k train records) keeps 3 908 records, about 12 MB now and 17 MB at peak.
  - Tests:
    - `test_the_metrics_tail_thins_each_series_by_its_counter`: 20 polls, three series, bound, first and last kept, sorted and even spacing;
    - `test_the_report_builder_bounds_its_metrics_tail_by_the_page_limits`;
    - `test_the_metrics_tail_reads_in_fixed_size_chunks`: 7-byte chunks give the same records, offsets, malformed line and completed partial line.
- **C-M3 — fixed** (5a765a3).
  - Text only, in `data.py`'s docstring and ARENA_GUIDE §7.
  - It says that a `mode=ro` reader of a WAL journal leaves `matches.db-shm` and `-wal`, and that for the same reason a run dir the reader cannot write gives "the match log cannot be opened".
  - Both claims checked by probe: after the report the files are there, and a `chmod 555` run dir gives the note.
- **C-M4 — fixed** (261324d).
  - `ReportBuilder.build()` isolates each section.
    - A failure is logged with its traceback and replaced by `failed_section(error)`, which is `{"notes": ["This section failed: <Type>: <msg>"]}`. The dashboard's `_page` now uses the same function.
    - Failures are recorded in `builder.failed_sections`.
  - `write_report` writes the page, then raises `ReportSectionsFailed(path, failed)`.
  - The CLI prints `Report written to <path>, but N section(s) failed and show a note instead: <name> (<error>); … (tracebacks in the log above)` on stderr and exits 1. Its help text says so.
  - ARENA_GUIDE §7 and README say so too.
  - Tests:
    - `test_sp4_report_render.py::test_a_failing_section_is_a_note_and_the_page_is_still_written`;
    - `test_sp4_report_cli.py::test_report_exits_1_naming_the_failed_sections_after_writing_the_page` (in-process CliRunner).
- **C-M6 — fixed** (22d4557).
  - **Chosen option: `click.IntRange(min=2)`**, with the help text "Seconds between page refreshes (at least 2)". ARENA_GUIDE says "не меньше 2".
  - The page keeps its `Math.max(2, …)` guard for direct `make_server` users.
  - Test: `test_cli_exit_codes_for_a_non_run_dir_and_bad_arguments` now checks that `--refresh 0` and `--refresh 1` exit 2.
- **C-M7 — fixed** (cbc0ede).
  - The pipeline slow test parses the embedded data with `report_kit.select_groups(page, TOURNAMENT)`. It asserts that the tournaments item named after the tournament has exactly one select group, and that this group's winner is the tournament's `select["winner"]`. That is slightly stronger than the brief asked.
  - Not run (a ~2 min training). The helper was checked in two fast tests instead:
    - on a page built by `write_report` from the sample tournament `sel` in `test_a_tournament_output_dir_builds_a_page`: `page_data(page) == _data_of(page)`, the winner is `main@v2000`, and a missing name raises;
    - on the run report of a real `select` tournament (the `to_dict()` schema through `report_view`) in `test_sp4_tournament_cli.py::test_select_on_a_run_writes_the_journal_and_reports_and_pins_the_confirmed_winner`: the winner is `main@v30`.
- **C-M8 — fixed** (cdbe814).
  - Table columns accept a `title`. The LCB header carries "lower bound of the deviation from the lineage mean; used by top-k".
  - Asserted in `test_every_section_is_drawn_in_a_browser`.
- **T7.1 deferred, `evaluation.agents: []` — fixed** (60b4d8c).
  - `plan` uses `is not None`, as `pulse_plan` does. The `field` default line uses the same rule for consistency; an empty `field` is a config error anyway.
  - Test: `test_an_empty_evaluation_agent_list_marks_no_snapshot_unevaluated`.
- **T7.1 deferred, matrix agent filter vs score cross-play — fixed** with C-I1 (d0e0905).
- **T8.2a deferred, docstring — fixed** (fa6da3a). It now reads: z = 0.116 / 0.030 = 3.87; about 5.5e-5 per gap (normal tail at z = 3.867); at most 6 × 5.5e-5 = 3.3e-4 < 1e-3 for the six gaps. See R-C6 on 5.4e-5 vs 5.5e-5.
- **M5, M9, M10:** not mine. M5 and M10 are parked by the review; M9 is left for T8.5, as the dispatch says.

## RED / GREEN

The command for all of these: `<python> -m pytest -q <file> -k <test>`.

- **T7.1.**
  - RED: `AssertionError … Left contains 1 more item: {'main': [160, 180, 200]}`.
  - GREEN: `test_sp4_report_data.py` 28 passed.
- **M2.**
  - RED, three failures:
    - `TypeError: MetricsTail.__init__() got an unexpected keyword argument 'max_per_series'`;
    - `AssertionError: assert 600 <= ((2 * 10) + 1)` (the builder kept every record);
    - `AttributeError: … has no attribute 'READ_CHUNK_BYTES'`.
  - GREEN: 31 passed.
- **M4.**
  - RED:
    - `ImportError: cannot import name 'ReportSectionsFailed'`;
    - the CLI test failed with `assert ('anchors' in '')`: the exception propagated, there was no message and no page.
  - GREEN: 2 passed.
- **I1.**
  - RED in the data tests:
    - the cross-play was unfiltered (13 compositions where `['greedy']` was expected);
    - `KeyError: 'agents'`.
  - RED in the DOM test: `data-missing="matrix/agent "` (no agent select).
  - A second RED view, with the missing-select check disabled temporarily: `assert (['2p', '3p'], []) == (['2p', '3p'], ['3p'])`. The `3p` choice was lost after a refresh.
  - GREEN: the dashboard, report data and render tests, 58 passed. The DOM test passed 2 more times, about 1.5 s each.
- **M6.**
  - RED: `subprocess.TimeoutExpired` on `--refresh 1`: the server started.
  - GREEN: 1 passed in 1.4 s.
- **M8.**
  - RED: 1 failed (no title).
  - GREEN: render tests, 16 passed.
- **M1, M7, M3, T8.2a:** test strengthening or text, so no RED. M1's two runs are listed above.

## Suite

- `env -u PYTHONPATH PYTHONPATH=<wt>/src OMP_NUM_THREADS=1 <python> -m pytest -m "not gpu and not slow" -q -rw` in the worktree:
  - **2129 passed, 39 deselected, 0 warnings, 640.85 s.**
  - That is 2121 + 8 new tests: 5 report data, 1 render, 1 report CLI, 1 dashboard DOM.
  - It ran once, after the last commit. A first start was stopped early by me to land ab84418 first.
- `ruff check .`: all checks passed.
- No slow test was run except the disk test, alone, twice; both passed.

## Deviations and proposed rulings (что — почему — цена ошибки)

- **R-C1 (M1, snapshot cap during the run).**
  - Что: a listing over `max_snapshots` counts only when two more listings, 0.1 s apart, are over it too.
  - Почему:
    - `CheckpointManager.save` puts the new `ckpt_v<N>` in place before `_retain` renames the evicted one out of the `ckpt_v*` namespace. For a few ms the dir holds cap + 1.
    - With about 50 saves in a 22 s run and 44 polls, one 0.5 s listing could catch that window in roughly 5–18 % of runs. That is a flake, not a spec violation.
    - A real over-cap persists until the next retention pass. Both runs saw 0 listings over the cap.
  - Цена ошибки: an over-cap that lasts less than about 0.2 s goes unseen. If the owner reads spec 3.5 as "never, even during a save", the save should evict before placing the new dir (`checkpoint_manager.py`, area A).
- **R-C2 (I1 scope).**
  - Что:
    - The Ratings refresh drops `agent` too, not only `layout`. Both are filtered on the page.
    - The matrix payload gets an additive `agents` field.
    - The agent select exists in live mode only, and "(all)" means the server default: the top-rated trainable agent's snapshot map and every composition.
  - Почему:
    - A server-side agent filter has the same defect as the layout one: the select's options are rebuilt from the filtered answer.
    - The static page has one agent's map only (ruling D4).
  - Цена ошибки: none functionally. The API keeps every filter.
- **R-C3 (follows I1, ab84418).**
  - Что: the truncated-entities note "(use the dashboard filters)" now says which rows are left out: "(the lowest rated)" or "(the lowest mean scores)".
  - Почему: the 2 000 kept rows are cut before any agent filter, on the server as well. No filter ever brought the cut rows back, and the live page now filters on the page.
  - Цена ошибки: none; text only.
- **R-C4 (M4 API).**
  - Что: `write_report` raises `ReportSectionsFailed` after writing. `build_report_data` (used by `arena_kit`) isolates sections and does not raise; a failure is an ERROR log line plus the note.
  - Почему: one rule for both callers. The CLI needs the failure; a kit test keeps asserting on the section contents, so it still fails on a placeholder.
  - Цена ошибки: a kit that only checks `set(sections)` would no longer notice a failing section; none does today.
- **R-C5 (M2 design).**
  - Что: buckets are set by the counter's value (`floor(counter / width)`), not by the record's ordinal, and the newest record is always kept.
  - Почему: this follows the ruling's "by its counter". Spacing stays even for irregular `env_steps`, and the live tiles (`latest`, the summary's env steps) keep the newest record.
  - Цена ошибки: records with equal counters collapse to one per bucket. This is harmless for line charts.
- **R-C6 (T8.2a numbers).**
  - Что: the docstring says 5.5e-5 per gap, where the dispatch said "about 5.4e-5".
  - Почему: Φ(−0.116/0.030) = Φ(−3.867) = 5.52e-5, while 5.4e-5 is Φ(−3.87) at the rounded z. The six-gap bound is 3.3e-4 either way.
  - Цена ошибки: none.

## Observation (not fixed, outside the findings)

- In a run dir the reader cannot write, the note reads `the match log cannot be opened: … is not a match log (attempt to write a readonly database)`. The words "is not a match log" come from `MatchLogReader` and are misleading for a permission problem. This is minor and outside area C's report code; it is up to the controller whether to park it.
