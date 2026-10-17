<!-- Origin: .superpowers/sdd/2026-10-16-sp4-selection/final-review-B.md in the git-ignored SDD workspace (deleted after the SP4 merge); copied verbatim on 2026-10-17. -->
<!-- Frozen record: never edit. -->

# SP4 final whole-branch review — Area B (tournament, pulse, eval/players, distributed guards)

Reviewer: area B, range `1366bec..589d464` (`sp4-selection`). Read end to end: `arena/tournament/{players,schedule,engine,stats,modes,run,__init__}.py`, `arena/pulse.py`, the pulse parts of `launcher.py` (`launch`, `_start_children`, `_monitor_loop`, `_save_checkpoint`, `_pump_pulse`, `_write_pulse_state`, `_finish_pulse`, `_finish_metrics`, `_shutdown`, `_release_queues`), `cli.py` (`tournament`, `eval`, `record`), `eval.py` (`parse_agent_spec`, `balanced_match_count`, `play_lineups`, `load_frozen_player`, `load_player`), `record.py`, `players/registry.py` (`load_frozen(use_entry)`, `_pt_roles`), `core/validation.py` (SP4 checks), `distributed.py`, `utils/process.py`, the `MatchRunner` changes they rely on, and the area's tests. Cross-checked against spec blocks 3, 6, 8, §8, the overview amendments A9, A11, A20, A24–A29, constraints P/C rulings, Part 2 rulings D1–D21 and the ledger.

Probes (single process, `OMP_NUM_THREADS=1`, scratch `final-review/B/`): `probe_solo.py` (tournament on a solo game), `probe_pulse_solo.py` (pulse plan on a solo game), `probe_rr.py` (round-robin vs gauntlet under `--deterministic`), the numpy-before-BLAS-limit check, and the unit test files `test_sp4_tournament_{schedule,stats,modes}.py`, `test_sp4_eval_dedup.py`, `test_sp4_distributed_guards.py` (57 passed). No multi-process run.

## Strengths

- **Scores are attributed by name everywhere.** `stats.block_values` maps every team to a player through the seat's `(agent_id, network_id)` (`_team_names`, ValueError if a team mixes players or the candidate holds ≠ 1 team). Rows, the halving ranking (`_halving` maps `comb.candidates` → points by name, the T5.3 fix), `combine` (index lookup per table), `select_winner`, `_table_dict`, `_combined_dict`, the BT columns (`by_name`) and `report_view` all key by name. Journal reuse round-trips `network_id` (`MatchRow.to_result`), so reused and freshly played blocks are scored identically. FFA multisets, asymmetric shifts, 2v2 teams and score cross-play (column = `composition[0]`, value = the team score) are handled.
- **Admissibility does not depend on the field order** (sorted composition key; the first ordering with the most shifts among the sorted distinct permutations). Within a role group every candidate has the same admissible shifts, so paired blocks are structurally identical. Every unschedulable composition is an INFO plus a `DroppedComposition` in the report. A candidate with no composition at all is a ConfigError.
- **Seed domains are disjoint by construction.** `stage_seed` puts the domain code (A = 1, B = 2, P = 3) in the top two bits and a keyed bijection of the index in the low 30 bits. Stage A (with halving rounds continuing `first_index`), stage B and the pulse never share a seed, and indices never repeat within a domain. Training seeds were already full uint32, so no new seed range reaches the envs.
- **Resume is safe against stale blocks.** It compares the request (mode, every size, seed, deterministic, layouts, absolute player paths, `--set`, `env`) and refuses a name whose blocks are in the journal without a `tournament.yaml` (T5.4 fix, checked before any write). It reuses only blocks whose uid set equals the schedule's. Players come from the stored yaml, so `top:N` does not move. Blocks are appended one transaction each and never compacted. The test `no_play` patches `play_blocks` to prove there is no replay.
- **`select` is correct.**
  - The winner is the stage B point maximum.
  - `confirmed` is `quantiles(diff)[0] > 0` over a paired stratified bootstrap: one multinomial draw per stratum, shared by every candidate. Repeated-column conflicts veto it.
  - A lone finalist is never confirmed. Stage A blocks never enter stage B tables.
  - The text, `report.json` and `report_view` all print `NOT confirmed` / `confirmed: false` honestly.
  - `--pin` acts only on a confirmed run-snapshot winner and pins `winner.run` / `winner.run_agent` / `winner.version`, which is the right run also for an aliased second `--run`.
- **The pulse holds its contract.** It plays a fixed field on stage P seeds (`training.seed or 0`), every snapshot on the same blocks and the same torch seed per task. Its seats never collect. Results go to the journal writer only (A20 / C8; the unit test fails on any coordinator call).
  - The backlog is bounded: the oldest request is dropped (INFO + `pulse_skipped_snapshots`). The request queue is one slot, and the weights of in-flight requests are not kept.
  - The stop is checked before every env step and in `put_until_stop`. The process follows the child rules: `run_child`, PDEATHSIG, a crash stops the run, `logs/pulse.log`.
  - The final snapshot is offered after the stop and never handed out, as ruled. The final pending count and an INFO name it.
- **The engine shuts down cleanly.** It starts `min(num_workers, tasks)` spawn workers, plays whole blocks per task, and checks `check_blocks` (unique uids, block seed and id, seats ⊆ players) before any worker starts. Worker errors keep their class (MRO lookup), get a match uid and log path, and are re-raised in the main process. On every exit the workers are stopped (event, grace, terminate, kill) and the queues closed without join.
- **The distributed guards match spec block 8.** `evaluation.enabled` and `keep_top_k > 0` are a ConfigError naming SP5 (the pulse is also caught by the fixed-agents rule). `match_log` / `ratings` give one INFO. `max_snapshots` works FIFO with the learner's `run_id` (`ensure_run_id`).
- **The eval / record dedup keeps SP3 behaviour.** It uses one parser and one rounding rule. Frozen agents are loaded once (`ValidationReport.fixed`). `use_entry=False` drops only the frozen overlay for a user path, and a trainable name still gives its architecture (D20). The help strings say so (T8.4). Exit codes are unchanged: `eval` gives 2 through `BadParameter`, `record` gives 1. `limit_main_blas_threads` really runs before numpy is imported (`import colosseum.cli` leaves numpy unimported).
- **Carried-forward items are closed:**
  - T1.1 lineup-per-episode, by `check_blocks`;
  - T5.1b absolute paths, by `_absolute_options` / `_absolute_player`;
  - T5.2 → T5.3: de-duplicated lists, once-per-tournament INFO, dropped compositions in the report;
  - T2.1 consistent `top:N` read, by `bootstrap_data(...).to_table`;
  - T5.3 → T7.x marks, in `report_view`.

## Issues

### Critical

None.

### Important

**I1. A tournament and a pulse cannot run on a solo or cooperative (`score`-only) game unless the config has an unused scripted / frozen agent.**
- **Where:**
  - `src/colosseum/arena/tournament/players.py:376-378` (`if not field_players and not allow_empty_field: raise ConfigError("the field is empty ...")`);
  - `src/colosseum/arena/tournament/run.py:389-391` (`allow_empty_field=options.mode == "round-robin"`);
  - `src/colosseum/arena/pulse.py:149-151` (`if not field: raise ConfigError("evaluation.field is empty ...")`).
- **What is wrong.** Spec block 3 says that a `score` layout block is "один seed и однородная команда кандидата" and that only "С `--field` добавляются составы «кандидат + участник поля»". So a one-team layout needs no field, and in a one-seat layout (`score_blocks`: `len(seat_roles) >= 2` is false) a field member is never even seated. The code refuses an empty field in every mode except round-robin, and round-robin itself refuses one-team layouts (C4). The pulse refuses an empty field the same way, although spec block 6 defines score-layout pulse games as the homogeneous team plus optional field compositions.
- **Failure scenario.** `configs/examples/coin_grid.yaml` and `coop_buttons.yaml` have no scripted or frozen agent.
  - `colosseum tournament --run runs/coin --mode select` gives `Config error: the field is empty: name opponents with --field ...`. Probe `final-review/B/probe_solo.py` reproduces it on `write_fake_run(..., "solo")`.
  - `colosseum train ... --set evaluation.enabled=true` on such a config gives `ConfigError evaluation.field is empty` (probe `probe_pulse_solo.py`).
  - A solo game is owner priority 5. The only workarounds are undocumented: add a dummy bot that never plays, or move one snapshot into `--field`, which removes it from the candidates. The error text ("name opponents") is misleading for a game without opponents.
- **Fix:**
  - Tournament: allow an empty field when every tournament layout has one team. Pass `allow_empty_field = mode == "round-robin" or all(spec.num_teams(L) == 1 for L in chosen layouts)`. This needs `_layouts` computed before `resolve_players`, or a check of the spec's layouts when `--layout` is absent.
  - Pulse: in `pulse_plan`, raise only when some planned layout has two or more teams.
  - `score_blocks` already works with `field=[]`.
  - Add one test per path, plus one ARENA_GUIDE sentence.

### Minor

**M1. `round-robin --deterministic` claims certainty for repeated games.** Ledger T5.3 deferred minor; this is the fix-wave candidate under "Deferred minors to fix now".
- **Where:** `src/colosseum/arena/tournament/modes.py:437-463` (`_round_robin_tables`).
- **What is wrong.** Every pair whose blocks all repeat gets `ci95 = [x, x]` with no `repeated` mark and no entry in `flags.repeated_pairs`. The gauntlet marks such a pair and widens its interval to Wilson of one observation. Probe `probe_rr.py`, 1v1, deterministic, strengths ±100:
  - round-robin: `{'score': 1.0, 'ci95': [1.0, 1.0], 'blocks': 10}`, flags `[]`;
  - gauntlet: `ci95 [0.207, 1.0]`, `repeated: True`.
- **Why it matters.** Spec block 3: «отчёт помечает «игры повторяются: недостаточно случайности». Такие пары в ДИ … считаются одним блоком».
- **Fix.** `block_values` already returns the signature. Under `settings.deterministic`, flag a pair with ≥ 2 blocks and one signature, give its cell Wilson(x, 1), and add it to `repeated_pairs`. About 10 lines plus a test.

**M2. A dead worker is noticed late while others keep streaming, and its peers' finished blocks are then thrown away.**
- **Where:** `src/colosseum/arena/tournament/engine.py:340-355` and `277-285`.
- **What is wrong.** `first_failure` is checked only on `queue.Empty` (a 0.2 s gap). `_error_after_failure` then reads for up to 1 s *per message* and discards every `("done", …)` message, while the other workers keep playing, because `stop` is set only in `finally`.
- **Failure scenario.** tic-tac-toe, `--num-workers 4`, tasks of about 50 ms, and worker 2 is OOM-killed. The main process journals the others' blocks until a gap; then `_error_after_failure` drains for the rest of the tournament. Every remaining block is played, not journaled, and replayed on resume, and only then does `TournamentError` appear. There is no hang and no corruption, but the compute is lost.
- **Fix:**
  - check `first_failure` once per loop iteration (not only on Empty);
  - in `_error_after_failure`, set `stop` first;
  - keep handling `done` messages (journal them) while looking for the typed error, bounded by `STOP_GRACE_SEC`.

**M3. Journal and filesystem errors in `tournament` escape as raw tracebacks.**
- **Where:** `src/colosseum/arena/tournament/run.py:356-366` (`_journal_blocks`), `329-334` (`_journal_run`), `451-462`; `TournamentDir.link` `166-183`; `src/colosseum/arena/matchlog.py:132` (`MatchLogError(Exception)` is not a `ConfigError`).
- **What is wrong.** These paths raise errors the CLI's `_config_errors` does not catch:
  - a journal of another schema version, or a corrupt file, gives `MatchLogError` from `open_readonly`;
  - a long training lock gives `sqlite3.OperationalError`;
  - a snapshot evicted between `read_snapshots` and `link_or_copy` gives `FileNotFoundError`. This race is on the supported "tournament while training" path: `--snapshots all` / `top:N` resolve for seconds while retention runs.

  All of them print a Python traceback (exit 1). Only `top_snapshot_lcb` wraps `MatchLogError`.
- **Fix.** Wrap `MatchLogError` / `sqlite3.Error` into a `ConfigError` naming the journal. In `link`, turn `FileNotFoundError` into a `ConfigError` ("ckpt_vN was evicted by the run while the tournament started; rerun"), or skip that snapshot with a WARNING.

**M4. The pulse's user-facing text says the final snapshot is always evaluated.**
- **Where:** `src/colosseum/arena/pulse.py:187` (`describe_pulse`: "(the final one always)"), `src/colosseum/core/config.py:702` (`evaluation.every` description: "Evaluate every k-th saved snapshot (the final one always)."), `pulse.py:4-5` (module docstring); the ARENA_GUIDE:393 example quotes the line.
- **What is wrong.** By the T6.1 ruling the final snapshot of a local run is *never* evaluated. ARENA_GUIDE:399 explains this correctly, but the `validate` printout and the schema description promise the opposite.
- **Fix.** Reword to "the final one is offered too; local runs stop before it is played (evaluate it with `colosseum tournament`)" and update the ARENA_GUIDE example line.

**M5. Worker processes are respawned for every layout of every halving round, and layouts run one after another.**
- **Where:** `src/colosseum/arena/tournament/modes.py:266`, `387`, `394` call `ctx.run` per layout. Each `BlockRunner.__call__` → `play_blocks` spawns `W` fresh processes: torch import, `env_spec`, the LRU model cache is lost.
- **Why it matters.** A three-layout `select` with four halving rounds spawns 15 worker generations. Each layout's small round under-fills the workers (`make_tasks` gives fewer tasks than `W`). This is a performance issue only.
- **Fix.** Collect every layout (and every group) of a round into one `ctx.run(...)` call.

**M6. `TournamentDir.link` trusts an existing complete link dir by path alone.**
- **Where:** `run.py:168-170`.
- **Failure scenario.** A first start fails after linking (for example, a `--layout` typo). A retry with the same `--name` passes a different second `--run` dir with the same basename. `snapshots/1-<name>/…/ckpt_vN` is reused, and the old run's weights play under the new run's names.
- **Fix.** In `_new_tournament`, where nothing references `snapshots/` yet, empty it first, or compare `os.path.samefile` / `meta.json` before reuse.

**M7. Two concurrent starts of the same tournament name are not excluded.**
- **Where:** `run.py:211-215`. `BlockRunner._store` ignores `AppendStats.duplicates`.
- **What is wrong.** Both processes pass the "no yaml" check, write `tournament.yaml` (last writer wins) and append the same uids. The second process's report is computed from its own games, while the journal holds the first's.
- **Fix.** Create `tournament.yaml` with `O_EXCL`, or hold a lock file, and WARN or raise when a just-played block comes back as duplicates.

**M8. `round-robin` ignores role groups.**
- **Where:** `modes.py:376-381` (`_groups` / `check_role_groups` are not called) and `455-460` (a participant without a playable pair gets `score 0.0`, `opponents 0`, ranked last).
- **What is wrong.** In an asymmetric game a hunter-only and a prey-only participant are ranked in one row list. Spec block 3: «Смешанные группы в одной таблице — ConfigError с перечнем групп».
- **Fix.** Either split the rows by group or refuse mixed groups, and give a participant with no pair `score: null`.

**M9. A misleading hint for a `.pt` given on the command line under a frozen agent's name.**
- **Where:** `src/colosseum/players/registry.py:118-126` (`_pt_roles`).
- **What is wrong.** With `use_entry=False`, `-a bc=x.pt` in a game whose roles have different spaces fails with "set agents.bc.roles (kind: frozen) …". But `agents.bc.roles` is exactly what a user path ignores (D20).
- **Fix.** For a command-line path, suggest `-a bc` (the config entry) or a checkpoint dir.

**M10. `_pin` words every `pin_snapshot` ConfigError as "evicted by the run".**
- **Where:** `run.py:511-515`. Ledger T5.4 deferred minor.
- **What is wrong.** A malformed `meta.json` is also reported as an eviction.
- **Fix.** Use the "evicted" text only when `stored_dir` is missing, else pass the error through.

Not raised: the pulse strata in `bt.bootstrap_data` are per snapshot (`…/<agent>@v<N>/<composition>`), so the run's BT bootstrap of pulse blocks is not paired across snapshots. That makes it conservative, not narrow, and it follows A11 as written. This belongs to area A, if anywhere.

## Deferred minors to fix now (ledger references)

- **T5.3: round-robin under `--deterministic` never detects repeated pairs** (= M1). Fix in the wave: it is a spec block 3 requirement and the report shows a zero-width interval as certainty. Cheap and local.
- **T5.4: `_pin` wording** (= M10). Optional; one condition.
- **T6.1: a backlog-dropped final reported as "the run stopped first".** Optional wording, fold it into M4's rewording of the final-snapshot texts.
- **T5.1a: "--agent" named twice in a bad `-a` message.** Optional one-liner (`param_hint` vs the `-a` prefix of `parse_agent_spec`).

Park the rest:
- T5.1a docstrings and the `fixed.frozen` validated mark;
- T5.1b repeat-episode uid in an error;
- T5.2 dead `or '-'`, the (T-1)! orderings, the FFA neighbours (already documented in ARENA_GUIDE);
- T5.3 untyped halving ValueError (unreachable);
- T5.4 second config read;
- T6.1 aborted-worker late `put_nowait`;
- T8.1 console-only distributed INFO (SP5).

## Recommendations

1. Fix I1 in the final wave (small, one test each for the tournament and the pulse). Without it, priority 5's solo game type has no selection path that works out of the box.
2. Fix M1 with I1. Both are statistics and report honesty, local to `modes.py`.
3. M2 and M3 are robustness polish for long tournaments. M2's fix is about 10 lines in `engine.play_blocks`; worth doing if the wave has room, otherwise park.
4. M4 is a text fix (two strings and one doc line), cheap enough for the wave.
5. Park M5–M9 (M5 into SP6 performance; M6, M7 and M8 as unscheduled minors; M9 into the wording batch).

## Assessment

**Ready to merge: With fixes.** No Critical findings. The core statistics, seeds, resume, `confirmed` / `--pin`, the pulse contract, the engine lifecycle and the distributed guards are correct as read and probed. I1 (solo / score-only games cannot run a tournament or a pulse without a dummy agent) should be fixed before the merge. M1 (an honest round-robin interval under `--deterministic`) and M4 (the final-snapshot wording) are cheap additions to the same wave.
