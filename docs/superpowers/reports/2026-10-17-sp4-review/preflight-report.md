<!-- Origin: .superpowers/sdd/2026-10-16-sp4-selection/preflight-report.md in the git-ignored SDD workspace (deleted after the SP4 merge); copied verbatim on 2026-10-17. -->
<!-- Frozen record: never edit. -->

# SP4 plan: reconciliation and pre-flight report (2026-10-16)

Inputs: `docs/superpowers/plans/2026-10-16-sp4-selection/00-overview.md` and parts 01–03 (as written in parallel), the approved spec, the SP3 process. Outputs: contract amendments A1–A33 in `00-overview.md`, pre-flight rulings P1–P22 in `constraints.md`, the plan edits below. Line numbers are those of the committed plan files.

## Summary

- **Amendments:** 33 (A1–A33), binding, in `00-overview.md` «Contract amendments»; the contract text above them is unchanged. Each part's `## Contract notes` starts with a pointer saying the amendments win.
- **Pre-flight rulings:** 22 (P1–P22), in `constraints.md` with "what — why — cost if wrong".
- **Mismatches fixed:** 28 (cross-part names / shapes / missing producers, double definitions, ordering, three real bugs that would have crashed: `TeamKey` in `report.json`, `@` keys read as role groups, the disk test's `@bias` KeyError).
- **Probe (scratch tree from the plan's code, `scratchpad/sp4/preflight/wt`):** the T1.2 + T2.1 library tests (49, incl. the two new ones), the T5.2 + T5.3 tests (26, incl. the new `report_view` test, against Part 1's real `bt`), 11 of 12 retention tests (the 12th needs T4.1's config keys) pass; `ruff check` passes on every whole-file code block of the three parts (after P1 and two fixes of my own additions found by it). Live: `bt_section` with `@transitivity = to_dict()` and `rating_inputs` checked by a script.

## Changes per part

### Part 1 (`01-matchlog-ratings.md`)
- T1.2 Interfaces "Produces" (≈ l. 1330): new reader / row names.
- T1.2 `matchlog.py`: imports `SeatResult`, `TeamResult`; `MatchRow.replay`, `to_dict()` (l. 2390), `to_result()` (l. 2404); `_Store.bootstrap_units` (l. 2589); `MatchLogReader.__enter__/__exit__` (l. 3057), `match_results` (l. 3068); `matches(limit: int | None)` (l. 3096) reading `m.replay`.
- T1.2 test `test_rows_round_trip_results_and_bootstrap_units` (l. 1826).
- T2.1 "Produces" (≈ l. 4008): `agent_order`, `default_options`, `from_table`, `from_blocks`, `bootstrap_data`, `to_dict`, `by_entity`.
- T2.1 `bt.py`: `Mapping`, `TYPE_CHECKING`, `match_pairs`, `snapshot_version` imports; `BTOptions.agent_order` (l. 4602), `default_options` (l. 4608); `BootstrapData.from_table` / `from_blocks` (l. 4707–4713), `bootstrap_data` (l. 4729); `TransitivityResult.worst_pairs` with match counts and `to_dict()` (l. 4754); `ScoreSummary.by_entity` (l. 4778), `to_dict()` (l. 4783); `fit` uses `agent_order or options.agent_order` (l. 5092); residuals carry `stat.n`.
- T2.1 test `test_json_forms_options_and_bootstrap_inputs` (l. 4387).
- T2.2 `live.py`: `rating_inputs(config)` (l. 5843); `@transitivity = test.to_dict()` (l. 6004); launcher `_open_live_ratings` uses `rating_inputs`; test asserts `rating_inputs`; "Produces" updated.
- T1.1 Step 10: slow suite once (P9). One wrapped signature line (P1 style).
- `## Contract notes`: reconciliation pointer.

### Part 2 (`02-retention-tournament-pulse.md`)
- Cross-task notes (l. 30–34): `ensure_run_id`, explicit tournament id (A9).
- T4.1 Consumes / Produces (l. 67–73), grep (l. 81), expected text (l. 88): `contrast_samples`, T1.3's evict listener.
- T4.1 test stand-ins `_Unc.contrast_samples` (l. 219, 436); `combine_scores` reads `contrast_samples` (l. 760, 784).
- T4.1 `CheckpointManager` error message wrapped (l. 944, P1).
- T4.1 Step 8: no duplicate `_evict_listeners` / `add_evict_listener` / loop (l. 1067); `_build_coordinator(run_id=self._run_id)` and `_open_live_ratings(top_k, quantile)` (l. 1087); `_wire_retention` only sets the ratings source (l. 1091); wiring order; distributed learner `run_dir.ensure_run_id()` (l. 1167).
- T5.1 `test_sp4_tournament_players.py`: `def pick` instead of a 132-char lambda (l. 1670); `eval.py` comment wrapped (P1); Step 10 slow suite once (P9).
- T5.1 `top_snapshot_lcb`: `rating_inputs`, reader as context manager, `bt.bootstrap_data` (l. 2378); `_random_bot` removed; Consumes updated.
- T5.3: `bt_section` transitivity via `to_dict()`; `_jsonable` removed; `report_view` (l. 4377, idempotent, `rated=rated` binding); test `test_the_report_view_by_layout_and_role_group` (l. 3638); Consumes / Produces updated.
- T5.4: `BlockRunner._store` appends with `tournament=block_id.split("/", 1)[0]` (l. 4946); `write_reports` guards `write_tournament_html(self.path)` (l. 4926); `_journal_run` uses `ensure_run_id()` (l. 5034); `tournaments_dir` owned here (l. 4518, 5310); Consumes updated; frozen-teammate case (l. 4709, P10).
- T6.1: Files (+`metrics/hub.py`), Consumes / Produces; test `test_system_values_reach_every_later_system_record` (l. 5497); `run_id=self._run_id` (l. 6067); `_pump_pulse` before `_stop_match_log` (l. 6122); hub `set_system_value` (l. 6125); commit stages `hub.py`.
- `## Contract notes`: reconciliation pointer.

### Part 3 (`03-report-docs-acceptance.md`)
- T7.1 Files / Consumes / Produces (l. 43–54): `tournaments_dir` is T5.4's; A14 / A26 named.
- T7.1 `RunDir` step: only `run_state_path`, `report_path` (≈ l. 320).
- `tests/report_kit.py` `live_bt` fixture in T2.2's real shape (`bt_row`, l. 553–561); live test asserts group, anchor, bias, transitivity.
- `data.py`: imports `entity_version` / `RUN_SOURCES` (l. 971); duplicates removed; `_live_layout` rewritten (l. 1383); `_tournaments` applies `report_view` (l. 1722).
- T7.2 Step 4 (l. 2829): `TournamentDir.write_reports`, remove the guard.
- T8.2 `arena_kit`: docstring, `find_report` / `select_snapshots` apply `report_view` (l. 3851, 4029), `--halving-blocks` (l. 3899); grid text and the benchmarks header; pulse counter names (Consumes, grep, docstring, l. 4516).
- T8.3 disk test skips `@` keys (l. 4982).
- `## Contract notes`: reconciliation pointer.

### Overview (`00-overview.md`)
- «Contract amendments» A1–A33 (l. 507–543).
- Task index: T3.1 → T1.3; T5.1 → T1.1, T2.2, T4.1; T8.2 → T5.4, T6.1, T7.1; execution-order sentence.

## Reconciliation table (requirement / addition → resolution)

| Item | Resolution |
|---|---|
| R1 `RunDir.run_id` get-or-create | `ensure_run_id()` (T1.1) at T4.1 distributed / T5.4; `Launcher._run_id` at T4.1 coordinator, T6.1 pulse; `run_id` property stays read-only (A2, P16) |
| R2 launcher attributes, shutdown order | met (`_match_log_writer`, `_live_ratings`); order fixed in A17 |
| R3 `MatchLogReader.close()` | met; context manager added (A6) |
| R4 tournaments in the journal | explicit `append(..., tournament=)`; T5.4 passes the block prefix; `match_results` added; idempotent run registration met (A4, A6, A9) |
| R5 `Uncertainty.draws` | Part 1's `contrast_samples`; Part 2 renamed (A12) |
| R6 `ScoreSummary.by_entity` | property added (A12; every match the entity sat in) |
| R7 `BootstrapData.from_table` | added (A11); `top:N` uses `bootstrap_data` instead (P3) |
| R8 `BootstrapData.from_blocks` | added; key = block id's last component (A11) |
| R9 `bt.default_options` | added (A10) |
| R10 `MetricsHub.set_system_value` | created by T6.1 (A29, P8) |
| R11 `p_value: float | None`, JSON `worst_pairs` | met; `TransitivityResult.to_dict()`; T5.3 uses it (A12, P2) |
| R12 `render_tournament_html` | replaced by `write_tournament_html(dir)` (A27, P19) |
| R-P1-1 reader API, `to_dict`, `to_result`, `limit=None` | added; seats carry `entity` / `category` (no `kind`, unused by `page.js`) (A6, A7) |
| R-P1-2 close / context manager | added (A6) |
| R-P1-3 `bootstrap_data`, `from_table` | added (`bootstrap_units` on the reader) (A8, A11) |
| R-P1-4 `to_dict` of transitivity / score summary | added; worst pairs carry `games` (A12) |
| R-P1-5 `rating_inputs` | added as a 3-tuple; the agent order travels in `BTOptions.agent_order` so every 3-tuple call site gets the live anchor rule (A10, A13) |
| R-P1-6 `bt` shape | Part 1's list-per-group + `@` extras kept (spec block 2 wording); Part 3 reader fixed (A14, P4, P5) |
| R-P1-7 `LogSizes` | met except `counts_bytes` (absent; T7.1 reads it optionally); flag `match_log.degraded` (A5, A16) |
| R-P1-8 constants | met (A10) |
| R-P1-9 final live fit | met with Part 1 ruling 12's condition (≤ `FINAL_REFIT_MAX_SEC`) (A13) |
| R-P1-10 run id location | `run.json` (A2) |
| R-P1-11 `system` names | met (A16) |
| R-P1-12 reopening a journal | met (`INSERT OR IGNORE`) (A4) |
| R-P2-1 tournament report shape | `modes.report_view` adapter in T5.3; consumers apply it (A26, P21) |
| R-P2-2 one writer of report files | `write_reports` (A27) |
| R-P2-3 CLI and sizes | met except select's round 0 is `--halving-blocks` (A28, P11) |
| R-P2-4 pulse counters | names are `pulse_*_snapshots` (A29, P12) |
| R-P2-5 distributed FIFO / refusals | met (T4.1 `keep_top_k`, T8.1 `evaluation.enabled`) |
| Part 1 additions 1–19 | A1–A5, A12–A18, A21 |
| Part 2 C1–C12 | A19, A22–A25, A27–A29 (C3's `add_evict_listener` is T1.3's: A19, P14) |
| Part 2 V1 / V2 / V3 | V1 → A20; V2 → index (A33, P18); V3 → controller (list below) |
| Part 3 N1–N6, A-P3.1–A-P3.5 | A30–A32; A-P3.4 superseded by the amendments where names differ |
| Part 1 assumptions 1–6 | 1 → P15; 2 → superseded by A9; 3 → superseded by A20; 4 → A14, A16; 5, 6 → met |
| Double definitions | `add_evict_listener` (P14), `tournaments_dir` (P13), `RUN_SOURCES` / `entity_version` (P6), `_random_bot` (P3) |

## Spec coverage

- §3 criteria 3.1–3.11: mapped to tasks in T8.5's table (unchanged); 3.4 and 3.5 depend on rulings D12 / D13 of Part 3 (below).
- §6 test items: all covered; the frozen-teammate case was missing (P10); "smoke tests on demo configs check the journal" counted as covered by T1.3 and T7.2 (P22).
- §9 trace items: all land in a task or are marked out of SP4 in the spec itself (W/D/L in three places, copy / hard-link tests, mixed-role shares → "не в SP4").

## Proposed rulings for the controller

Spec-changing ones first (they alter something the owner approved, or its wording); each line: the ruling — recommendation.

### Change or bend the approved spec (owner should see them)
1. **Part 3 D12 — the pipeline slow test does not assert `confirmed`.** Spec criterion 3.4 says `select` "выбирает снимок, подтверждённый на свежих seed". — Modify: assert the stage-B winner and its ≥ 0.95 win rate as written, print `confirmed`, and ask the owner whether 3.4 means "confirmed: true" (then the test must assert it, possibly with a larger `--confirm-blocks`). Owner decision.
2. **Part 3 D13 — the disk test compares the compacted journal with the final live fit instead of a "no-delete journal test mode".** Spec criterion 3.5 / §6 name the test mode. — Accept (the in-memory table is exactly the full table; no test-only switch in production code); tell the owner.
3. **Part 1 ruling 10 — only "not a database / malformed" files are quarantined; any other open error at start-up is a `ConfigError`.** Spec risk table: "если файл не открывается — переименовать его, начать новый журнал". — Accept (renaming a locked or unreadable but valid journal would lose data; the error is at start-up, before training); tell the owner.
4. **Part 2 D7 — `round-robin` only on two-team layouts.** Spec block 3 lists round-robin without that limit. — Accept (FFA "all pairs" has no natural seating); document in ARENA_GUIDE; tell the owner.
5. **Part 2 D10 — `--output DIR` is laid out like a run dir (`DIR/matches.db`, `DIR/tournaments/<name>/`).** Spec: "всё в --output DIR". — Accept (several tournaments per DIR, one renderer path).
6. **Part 2 D12 — `--pin` pins only a confirmed winner that is a run snapshot (WARNING otherwise).** Spec: "--pin закрепляет победителя select". — Accept ("тихого выбора нет"); tell the owner.
7. **Part 2 D17 — seeds are a keyed 32-bit bijection with the stage in the top bits, not `SeedSequence([seed, stage, …])`.** — Accept (domains disjoint by construction, which the spec asks for); note the wording difference.
8. **Part 2 V1 — pulse results on their own queue, not "через очередь результатов".** — Accept (blocking put; keeps pulse matches out of the online ELO / PFSP, which the spec requires).
9. **Part 1 ruling 11 — the main process may need up to `SHUTDOWN_GRACE_SEC` more after the children are gone (writer flush).** SP1's "no process alive after 10 s" is about children. — Accept; record that the main process's exit can take up to ~14 s in the worst case.
10. **Part 1 ruling 17 — `docs/benchmarks.md` keeps its historical `c_clip_frac` tables, explained in the text.** Spec block 7.1 says the file is updated. — Accept (history is not rewritten).

### Within the spec (interpretations and plan choices)
- **Part 1 rulings** — 1: Bradley–Terry coverage band ≥ 0.85 over 20 replications (nominal 0.95, tolerance 0.10; false-fail < 1e-3) — accept; let T8.2 tighten it after fixing prior / smoothness. 2: self-play = zero net coefficient — accept. 3: a trainable anchor = its lowest version — accept. 4: last fallback anchor = first entity by name — accept. 5: bias prior, smoothness across components — accept. 6: G² dof = pairs − (entities − cells) − (T − 1) — accept. 7: compacted rows bootstrapped as Poisson wins / losses — accept (the reading of "пуассоновские веса по строкам" that does not resample a whole row as one unit). 8: "faster than expected" = untouched parts above max(window, 64 MiB) — accept. 9: counts-only training matches keep no uid — accept. 12: refit cadence (first after one interval, skip unchanged, final if last ≤ 2 s) — accept. 13: no journal → no live ratings — accept. 14: compact record "best" = highest rating — accept. 15: league state loaded before the pool import; malformed file → `ConfigError` — accept. 16: BLAS env vars around `import numpy` — accept. 18: stamp `env_steps` only when missing — accept.
- **Part 2** — D1 top-k among all rated — accept. D2 over the cap, unrated `keep_every` snapshots go before rated ones — accept (the spec's order sentence is ambiguous here). D3 manager without run id protects every final — accept. D4 overflow WARNING once per agent and process — accept. D5 repeated deterministic pairs = one block (Wilson of one observation; no confirmation across a repeated column) — accept. D6 halving keep rule; eliminated candidates only in rounds (no CI) — accept (A26's view shows their last score). D8 score-layout cross-play = one field seat — accept. D9 tournament BT on all stages, non-snapshots fixed for the anchor — accept. D11 resume reuses stored players; execution settings not compared — accept. D13 layout weights from the first `--run` agent — accept. D14 naming / alias rules — accept. D15 `every:K` from the newest — accept. D16 `--seed` default 0, pulse seeds from `training.seed` — accept. D18 worker tasks of whole blocks ≥ 4·num_envs — accept. D19 pulse always offered the final snapshot; journal only — accept. D20 `-a <trainable>=<.pt>` keeps the trainable architecture — accept. D21 `pin` idempotent, no manager on the run dir — accept.
- **Part 3** — D1 BT source of the report (live while running, else journal bootstrap, 30 s cache) — accept. D2 mixed teams left out of matrix / anchors — accept. D3 lineage order by newest entity — accept. D4 snapshot matrix of the first trainable agent — accept. D5 anchors section pulse-only with pulse — accept. D6 run states (130 / 143 = finished; `stopped` derived) — accept. D7 "few games" < 20 — accept. D8 matches section filters — accept. D9 flags by key suffix — accept (the key is `match_log.degraded`, A16; could be exact). D10 throughput by counters with `--tree` — accept. D11 ladders (tic_tac_toe 2p, tron 3p; epsilons 0 / 0.25 / 0.5 / 1; field random + ref) — accept. D14 measurement rules fixed before the numbers — accept. D15 no `pyproject` change for package data — accept (T7.2 checks the wheel).
- **Process** — Part 2 V3 (split T5.1 at Step 5 into T5.1a / T5.1b) — recommend split: it is the largest task (shared loading + players + engine). Part 3 T8.2 split at Step 4 (code / ~3 h of measurements) — recommend split. Part 3 A-P3.5: slow suite 14 tests, ~20–25 min after T8.3 — accept (A32).

## Open risks

- **Probe coverage.** The probe ran library tasks only (T1.2, T2.1, T5.2, T5.3, the pure retention function). Launcher / worker / CLI tasks (T1.1, T1.3, T2.2 integration, T4.1, T5.1, T5.4, T6.1, T7.x, T8.x) were checked by reading and by `ruff` on their whole-file blocks; their edit snippets were not executed. Part 1's author dry-ran T0.1–T3.2 before this reconciliation; my changes to those tasks are additive (new functions and tests) except `worst_pairs` (4-tuple) and `@transitivity`, both covered by the probe.
- **Bootstrap inputs vs a live journal.** `bt.bootstrap` raises on a team pair the fit does not know. The report fits from `reader.comparison_table` and resamples `reader.bootstrap_units` in two reads; if a training run writes between them (a run whose pid check says "not running" but still writes, or a tournament — excluded by `tournament IS NULL`), a new pair can appear. T7.1 uses the journal only for non-running runs (D1), so the window is small; an implementer may build the table from the same units to remove it.
- **`report_view` semantics for several layouts.** Per-layout rankings come from per-layout tables; the `select` winner is decided on the layout-weighted combination. With several layouts the per-layout `ranking[0]` may differ from `select.winner`. The ladder and pipeline tests use one layout.
- **Frozen teammate test (P10).** Checked: `players.resolve_players` looks `--field` names up among the `-a` players by their given name (`users[name]`), so `--field mate` resolves; the column assertion uses `startswith("mate")` because the journal name is `mate#<hash>`.
- **Final live refit.** The disk test (D13) relies on the final refit (Part 1 ruling 12: skipped when the last refit took > 2 s). Fine for tic_tac_toe; a slower machine could skip it and the test would compare against an older fit.
- **Spec-changing items 1–2 above** decide whether criteria 3.4 and 3.5 are met as written; T8.3 / T8.5 should not start before the controller (and, for D12, the owner) rules.
