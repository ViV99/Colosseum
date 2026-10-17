<!-- Origin: .superpowers/sdd/2026-10-16-sp4-selection/fix-wave-B-report.md in the git-ignored SDD workspace (deleted after the SP4 merge); copied verbatim on 2026-10-17. -->
<!-- Frozen record: never edit. -->

# SP4 final fix wave, part B — report

Worktree `/home/viv/dev/repos/Colosseum-wt-fwb`, branch `sp4-fw-b`, from `167d9cc`. Not pushed. Python checked:
`colosseum.__file__` = `/home/viv/dev/repos/Colosseum-wt-fwb/src/colosseum/__init__.py`.

## Findings

| id | status | commit | tests |
|---|---|---|---|
| I1 | fixed (ruling B-I1) | `4d4a779`, state lines `11fa30c` | `test_one_team_games_need_no_field_and_two_team_games_still_do` (integration/test_sp4_tournament_cli.py), `test_one_team_games_need_no_pulse_field_and_two_team_games_still_do` (unit/test_sp4_pulse.py) |
| M1 | fixed | `2e01c5f` (+ CLAUDE.md parked line in `0bc107f`) | `test_round_robin_marks_repeated_pairs_under_deterministic` (unit/test_sp4_tournament_modes.py) |
| M2 | fixed (ruling B-M2) | `742f776` | new unit/test_sp4_tournament_collect.py (3 tests); integration/test_sp4_tournament_engine.py unchanged and green |
| M3 | fixed: eviction → ConfigError (not skip) | `921425b` | `test_journal_and_eviction_errors_are_config_errors_with_their_context` (integration/test_sp4_tournament_cli.py) |
| M4 | fixed (+ T6.1 deferred wording) | `eaa6580` | `test_the_texts_about_the_final_snapshot_say_a_local_run_never_evaluates_it` (unit/test_sp4_pulse.py) |
| M6 | fixed (ruling B-M6) | `ed099ec` | `test_a_retried_start_never_plays_the_snapshot_links_of_a_failed_one` (integration/test_sp4_tournament_cli.py) |
| M8 | fixed (ruling B-M8) | `ac15777`, state lines `11fa30c` | `test_round_robin_refuses_mixed_role_groups_and_a_pairless_participant_has_no_score` (unit/test_sp4_tournament_modes.py) |
| M9 | fixed | `d5afc5d` | `test_a_command_line_pt_whose_roles_differ_suggests_the_config_entry_or_a_checkpoint_dir` (unit/test_sp4_eval_dedup.py) |
| M10 | fixed | `0bc107f` | `test_pin_says_evicted_only_when_the_snapshot_is_gone` (integration/test_sp4_tournament_cli.py) |
| T5.1a one-liner (doubled `--agent`) | fixed, still present before | `d5afc5d` | `test_a_bad_agent_spec_names_its_option_once` (unit/test_sp4_eval_dedup.py) |

### What changed, per finding
- **I1.** `run._new_tournament` computes the layouts before resolving players and passes `allow_empty_field = mode == "round-robin" or every chosen layout has one team` (gauntlet, halving, select). `pulse_plan` raises the empty-field error per agent only when one of its planned layouts has two or more teams (equivalent to «no planned layout has two or more teams»). Error texts unchanged. ARENA_GUIDE: one sentence in the tournament section (§4 «Поле»), one in the pulse section (§6). `evaluation.field: []` (explicit empty list) is still a config-level error («or null for all»): the config validator has no spec; `null` gives the empty field on a solo game.
- **M1.** `_round_robin_tables` keeps each block's signature; under `settings.deterministic` a pair with ≥ 2 blocks and one signature gets `repeated: true` in both matrix cells, Wilson(x, 1) (and its mirror 1 − hi, 1 − lo), and both rows list it in `repeated`; `flags.repeated_pairs` lists each pair once (schedule order: a before b), so the text report prints «Games repeat …». ARENA_GUIDE's `--deterministic` line names round-robin pairs.
- **M8.** Round-robin calls `check_role_groups` per layout before scheduling; more than one group → ConfigError listing the groups (overlapping groups keep the schedule's own ConfigError). A participant with no pair gets `score: None`, sorts last, prints `-` in `report_text`. ARENA_GUIDE's mode table row says both. The table's `group` stays `"*"` (unchanged for area C's page).
- **M2.** The result loop is `engine._collect`: `first_failure` before every message; on a failure `stop.set()` first, then `_error_after_failure` keeps calling `handle` (journal) for `done` messages while it looks for the typed error, bounded by `STOP_GRACE_SEC` (earlier once no worker is alive and the queue is empty). The «all exited with tasks unplayed» branch now yields to a nonzero exit (the next iteration names the dead worker), which keeps `test_a_dead_worker_names_its_log` green.
- **M3.** `_journal_errors(db, name)` wraps the tournament body (`run_tournament` → `_run`): `MatchLogError` / `sqlite3.Error` → ConfigError naming the tournament and the journal (blocks kept for a rerun). `TournamentDir.link`: `FileNotFoundError` → ConfigError `--run R: snapshot ckpt_vN of A (path) was evicted by the run while the tournament started; rerun`. **Choice: ConfigError, not skip with a WARNING** (see rulings). ARENA_GUIDE: two sentences next to «Турнир во время обучения».
- **M4.** `describe_pulse`, the `evaluation.every` description, the pulse module docstring and the ARENA_GUIDE example line now say «the final one is offered too, but a local run stops before it is played: evaluate it with colosseum tournament». Folded T6.1 deferred: `PulseFeeder.skipped_by_backlog()`; `_finish_pulse` says «(the backlog skipped it)» for a final pushed out of the backlog, else «(the run stopped first)»; ARENA_GUIDE mentions the variant; the CLAUDE.md parked line is removed.
- **M6.** `_new_tournament` empties `snapshots/` before linking (no `tournament.yaml` yet, nothing references it). `link` reuses a complete dir only when each file is `os.path.samefile` or byte-equal (`filecmp.cmp(shallow=False)`) to the source; otherwise it is replaced.
- **M9.** `_pt_roles(user_path=...)` (from `load_frozen(use_entry=False)`): the hint for a command-line `.pt` suggests `-a <name>` (the config's frozen entry, whose roles apply to its own path) or, without such an entry, adding a frozen agent; a checkpoint dir; or a trainable agent's name. The config entry's own path keeps the old hint.
- **M10.** `_pin` uses the «evicted by the run» text only when `stored_dir` is missing after the failure; otherwise «could not be pinned in <run> (<error>)»; both keep the pointer to the tournament's copy.
- **T5.1a one-liner.** `parse_agent_spec(option=None)` gives no prefix; `eval` and `tournament` use it (click names the option). `record` keeps `--player` / `--against` prefixes (its errors are not click BadParameters).

## RED / GREEN

Command for focused runs: `env -u PYTHONPATH PYTHONPATH=<wt>/src OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q <test>`.

- I1 RED: tournament `AssertionError: Config error: the field is empty: name opponents with --field (agents of the config, -a players or snapshots), or add scripted / frozen agents to the config` (exit 1); pulse `ConfigError: evaluation.field is empty: the config has no scripted / frozen agents; ...`. GREEN: 2 passed.
- M1 RED: `KeyError: 'repeated'` (no mark in the cell; the reviewer's probe on the same code showed `ci95 [1.0, 1.0]`, `flags []`). GREEN: test_sp4_tournament_modes.py 21 passed.
- M8 RED: `Failed: DID NOT RAISE ConfigError`. GREEN: 22 passed.
- M2 RED (after a refactor-only extraction of the loop into `_collect`, no behaviour change): `assert [(1, False)] == [(1, True), (2, True)]` (stop not set, the peer's block after the gap discarded); bound test `assert ((163432.26 - 163428.24) < 1.5)` (4 s); typed-error test got `TournamentError` instead of `PlayerError`. GREEN: collect + engine integration 8 passed. (A first draft of the bound test used an endless stream and hung on the old code; it was stopped and replaced by a 4 s stream.)
- M6 RED: `os.path.samefile(link, second/.../ckpt_v5/model.pt)` False after the retry (the failed start's link to the first run was reused). GREEN: test_sp4_tournament_cli.py 19 passed.
- M3 RED: schema mismatch escaped as `MatchLogError('... is a match log of schema version 99, not 1')` (CliRunner exception, no "Config error"); probe (scratchpad `m3/probe_m3.py`): `OperationalError('database is locked')` and `FileNotFoundError(2, 'No such file or directory')` escaped raw. GREEN: test_sp4_tournament_cli.py 20 passed; the lock message reads `Config error: tournament 'b': the match log .../matches.db cannot be used (OperationalError: database is locked); blocks already in it are kept: rerun with the same --name ...`.
- M4 RED: `'the final one always' is contained` in the validate line. GREEN: test_sp4_pulse.py + test_sp4_docs.py 30 passed.
- M9 RED: `'-a bc' in "...: agent 'bc': a .pt file plays every role ... set agents.bc.roles (kind: frozen), name the agent after a configured agent, or use a checkpoint dir"` False. GREEN: eval_dedup + sp3_players + sp2_eval_cli 38 passed.
- T5.1a RED: `"-a 'x='" in "... Invalid value for '--agent': -a 'x=': expected ..."`. GREEN: 32 passed (with sp2_eval_cli and the usage-error parametrization).
- M10 RED: `--pin: the winner main@v10 was evicted by the run ... (no snapshots of agent 'main' ...)` for a malformed meta.json. GREEN: tournament CLI + pin CLI 25 passed.

## Suite

Full fast suite once at the end in the worktree (HEAD `11fa30c`): `env -u PYTHONPATH PYTHONPATH=<wt>/src OMP_NUM_THREADS=1 .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw -p no:cacheprovider` → **2134 passed, 39 deselected in 667.54 s, zero warnings, zero failures** (2121 before + 13 new tests). `ruff check .` → All checks passed. No flaky reruns were needed. Slow suite not run (no training-math, worker or learner change).

## Deviations and proposed rulings

- **M3: an evicted snapshot at the start is a ConfigError, not a skip** — что: `link` raises «... was evicted by the run while the tournament started; rerun» — почему: the selection the user asked for never changes silently (a skipped `--field` snapshot would change the field, a skipped candidate the ranking), nothing is written before the links (`tournament.yaml` comes after them, and B-M6 makes a rerun start from an empty `snapshots/`), and the race window is seconds — цена ошибки: a user with very aggressive retention reruns once more; a skip could be added later for candidates only.
- **M3: one wrapper around the whole tournament body** — что: `_journal_errors` wraps `_run` (also the `_store` appends inside `play_blocks`), not each journal call — почему: every journal touch (the startup reads, `MatchLog(...)`, the per-block append) gets the same context with one code path; sqlite is used nowhere else in the command — цена ошибки: a non-journal sqlite error would also be named as the journal's (none exists in this command).
- **M2: the search for the typed error waits up to `STOP_GRACE_SEC` (2 s) even when the dead worker sent none** — что: an OOM-killed worker now raises after ≤ 2 s (earlier when all workers have exited) instead of after the first 1 s gap — почему: ruling B-M2 bounds the search by `STOP_GRACE_SEC` and the peers' late blocks are journaled meanwhile — цена ошибки: ≤ 1 s more before the error on a crash.
- **M2: blocks a peer finishes after the grace are not journaled** — что: after `_error_after_failure` returns, the `finally` waits for the workers but does not drain their late `done` messages — почему: the ruling bounds the journaling by `STOP_GRACE_SEC`; workers finish their current task after the stop, which can take longer — цена ошибки: one task per peer replayed on resume (as before the wave, but bounded to one task instead of the rest of the tournament).
- **M8: the round-robin table's `group` stays `"*"`** — что: one group is enforced but not written into the table — почему: area C's page and the report view key round-robin tables by `"*"`; the wave parts run in parallel — цена ошибки: a cosmetic label; changing it later is additive.
- **M8: `score: null` is defensive** — что: with one role group per layout every pair is admissible or none is (then `round_robin_blocks` raises), so the null case is not reachable through `run`; the test drives `_round_robin_tables` directly — почему: the ruling asks for null instead of 0.0 — цена ошибки: none.
- **I1: `evaluation.field: []` stays a config error** — что: only `field: null` (the default) gives an empty pulse field on a one-team game — почему: the config validator cannot see the spec, and its message already says «or null for all» — цена ошибки: one config edit for a user who wrote `[]`.
- **M4 commit also changes code** (`PulseFeeder.skipped_by_backlog`, the `_finish_pulse` reason) — the T6.1 deferred wording folded in as the review suggested; launcher.py is also FW-A's file (different function), a cherry-pick conflict is unlikely.
- **Docs touched by this part:** ARENA_GUIDE §4 (empty field, round-robin row, `--deterministic` line, live-training paragraph), §6 (pulse empty field, validate example line, final-snapshot INFO variant); CLAUDE.md (tournament / pulse state lines; parked items removed: round-robin repeated pairs, `_pin` wording, doubled `--agent`, backlog-dropped final). Parts A and C also edit ARENA_GUIDE and CLAUDE.md: conflicts, if any, are line-local.
