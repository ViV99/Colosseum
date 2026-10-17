<!-- Origin: .superpowers/sdd/2026-10-16-sp4-selection/fix-wave-2-report.md in the git-ignored SDD workspace (deleted after the SP4 merge); copied verbatim on 2026-10-17. -->
<!-- Frozen record: never edit. -->

# FW-2 report (follow-up to the SP4 final fix wave: five Minor findings from the FW-B / FW-C re-reviews)

Status: DONE. Branch `sp4-selection`, pushed (`7faf179..0abc4bd`).

## Commits
- `8d19b29` fix: round-robin checks every layout's role groups before playing; the failure search drains once more (B-a, B-b)
- `0abc4bd` fix: the report's snapshot map, Ratings compositions and matrix agent select follow the agent filter (C-N1, C-N2, C-N3)

## Per item

### B-a: round-robin mixed-group check (`src/colosseum/arena/tournament/modes.py`, `_play_mode`)
- Fix: the round-robin branch now runs `check_role_groups` on every layout first and only then plays the layouts (two loops), as the gauntlet's `_groups` does. Error text unchanged.
- Test: `tests/unit/test_sp4_tournament_modes.py::test_round_robin_checks_the_role_groups_of_every_layout_before_playing_any`. Roles `a`, `b`, `c`; layout `aa` (both participants play `a`, one group), layout `bc` (one plays `b`, the other `c`, two groups). It asserts the ConfigError names `'bc'` and `FakeArena.calls == []`. The results source is the only path to the journal (`run.BlockRunner` in the CLI), so a source that is never called means nothing was played or journaled. The docstring says so.
- RED: `pytest tests/unit/test_sp4_tournament_modes.py -k every_layout_before` failed with `AssertionError: assert [['t/A/aa/p@v1/q@v1/0', 't/A/aa/p@v1/q@v1/1']] == []`. Layout `aa` was played before the error, as expected.
- GREEN: the whole file passed, 23 tests.

### B-b: race in `_error_after_failure` (`src/colosseum/arena/tournament/engine.py`)
- Fix: after the alive check finds no worker alive, the function calls `_drain(results)` once more, as `_collect` does. Each late message goes to `handle`. An `"error"` message found there is returned as its typed error (`_typed_error`), keeping its class. Then the loop breaks. The docstring explains the gap. The commit body explains the change.
- Test: `tests/unit/test_sp4_tournament_collect.py::test_a_peer_that_exits_right_after_an_empty_read_still_has_its_last_block_journaled`. It is deterministic, not timing-based: a fake queue gives `[GAP, ("done", 1, {})]` and a fake supervisor reports no worker alive (`_AllExited`). The handled list must be `[1]`. A second case, `[GAP, done, error]`, must raise `PlayerError`.
- RED: `assert ([] == [1]`: the peer's last block was lost.
- GREEN: the collect file passed, 4 tests.

### C-N1: snapshot map for an agent with no entities (`src/colosseum/arena/report/data.py`, `_matrix`)
- Fix: if the target agent has no entity in the layout's entity pairs, the layout gets `snapshots: None` and the note `"<layout>: <agent> has no snapshots in this layout."`. The target is the one given, or the default (the top-rated trainable agent). An unknown `?agent=` falls under the same rule. The page already skips a null map and draws the section notes, so the page code needed no change. The docstring is updated.
- Tests:
  - Data: `tests/unit/test_sp4_report_data.py::test_the_matrix_lists_the_trainable_agents_and_draws_the_chosen_ones_snapshots` is extended.
    - For `second` in `3p`: `snapshots` is None, the lineage map is still present, and the notes are exactly `["3p: second has no snapshots in this layout."]`.
    - For `agent=nobody`: both layouts are None, with one note each.
    - For the default (no filter): `3p` still maps `main`, with no notes.
  - Page: `tests/integration/test_sp4_dashboard.py::test_the_live_page_keeps_its_layout_and_agent_choices_across_refreshes` gets one assertion. The note is drawn in the matrix section of the live DOM (headless Chrome).
- RED:
  - Data test: `assert ({'agent': 'second', 'axis': ['main', 'greedy', 'random'], ...} is None)`, the lineage-map copy.
  - Live page test (run with data.py temporarily reverted by `git stash`): `assert '3p: second has no snapshots in this layout.' in '<section id="sec-matrix" ...'` failed.
- GREEN: both pass. The stash was popped before the commit.

### C-N2: Ratings compositions ignore the agent filter (`page.js`, `R.ratings`)
- Fix: two helpers, `lineageOf` (removes the `@v<N>` suffix, like `entities.lineage`) and `hasMember` (`composition.split("+")`). The score branch now draws only `(lay.compositions || []).filter((c) => !agent || hasMember(c, agent))`, the same rule as the server's `_filter_agent`. This applies in live and static mode, since both filter Ratings on the page.
- Test: `tests/unit/test_sp4_report_render.py::test_the_ratings_agent_filter_also_filters_the_score_compositions`. It is a static-page DOM test. The solo layout gets a `greedy` entity and a `greedy+greedy` composition. The test runs three times:
  - no choice: both compositions are drawn;
  - pick `main`: only `main@v100+random`;
  - pick `greedy`: only `greedy+greedy`.

  Each choice is made through the select's own change event. `_dom` gained optional `live` / `before` / `act` script injection.
- RED: after picking `main`, `'greedy+greedy' is contained here: ...<td>greedy+greedy</td>...`.
- GREEN: pass.

### C-N3: the agent select vanishes during a build (`page.js`, `R.matrix`)
- Fix: `STATE.matrix.known_agents` keeps the last `agents` list. An answer without an `agents` array (the 202 placeholder) draws the select from that list. The selected agent stays in `STATE.matrix.agent` as before.
- Test: `tests/unit/test_sp4_report_render.py::test_the_matrix_agent_select_stays_while_the_agents_map_is_building`. A live-mode page is loaded from a file, with a `fetch` stub injected before page.js. `api/matrix?agent=second` answers the dashboard's placeholder shape `{building, section, notes}` with status 202; every other request gets the embedded section back. The test picks `second`, and the placeholder is drawn: the building note is there and the snapshot map is gone. The agent select must still have the options `["", "main", "second"]`, with `second` selected. There is no server and no timing.
- RED: `AssertionError: the agent select is gone`.
- GREEN: pass.

Page security: no `innerHTML`. New values go only through `h()` text and attributes. No query building was added; the existing `encodeURIComponent` path is unchanged.

## Docs (ARENA_GUIDE, Russian)
- §4, the `round-robin` row: the group check covers every layout before the first game.
- §7:
  - an agent without snapshots in a layout gets no map, only the note;
  - the Ratings agent also filters the score cross-play compositions;
  - during "building", the Matrix agent list keeps the last agents and the choice.

## Suite
- Focused, after the fixes: the modes, collect, render, data, dashboard and tournament engine tests passed, 92 tests in 22.8 s (`fw2/green-focused.txt`).
- Full fast suite, run once before the commits: `env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (I also passed `-p no:cacheprovider`) gave **2168 passed, 39 deselected in 528.15 s**, with zero warnings and zero failures (`fw2/full-suite.txt`).
- `env -u PYTHONPATH .venv/bin/ruff check .`: All checks passed.
- The RED output of N2 and N3 is in `fw2/red-c-n2-n3.txt`.

## Self-review / concerns
- C-N1 also covers the default target. In the rare case where the top-rated trainable lineage in the axis has no single-entity team, you now get the note instead of the lineage-map copy. This is the same rule; the brief named only the chosen agent.
- The note carries the layout prefix (`3p: ...`). Matrix notes are section-wide, which matches the existing `"{layout}: N team pairs ..."` note.
- Nothing else changed. No concerns.
