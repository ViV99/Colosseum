> Origin: `.superpowers/sdd/2026-10-10-sp3-league/fix-wave-report.md` (SP3 SDD workspace, git-ignored, deleted after the merge). Copied 2026-10-10.
> Frozen record; do not edit. Copied verbatim (no absolute scratch paths to replace).

# SP3 final fix wave — report

Branch `sp3-league`, range `c31eda9..6fa1406` (18 commits, pushed). One implementer, no subagents.

## Final verification
- Full fast suite: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` → **1689 passed, 32 deselected in 396.98 s, 0 warnings** (log: `fixwave/fullsuite.log`).
- `.venv/bin/ruff check .` → All checks passed.
- Collection: 10 slow, 22 GPU (`pytest --collect-only -m slow|gpu`). CLAUDE.md / README counts updated (1662 → 1689).
- Slow suite and training runs: not run (per brief; acceptance runs them). Note: the slow pipeline test now runs from one config and with a seeded BC (item 1/15) — acceptance should confirm its thresholds still hold.

## Items

### 1. One-config pipeline (A-I1 = C-I1) — `084a710`
- `core.validation.validate_config(config, *, scope: Literal["train","play"]="train", players: Collection[str] | None = None)`; `registry.validate_config` passes both through. "play" scope skips: matchmaking checks (`validate_matchmaking`, custom matchmaker class, `describe_mix`), the warm start of trainable agents (`_check_critic_warmup`, kickstart teachers, `init` — moved into `_check_warm_start`), and every fixed agent not in `players`. `players` with the train scope or an unknown scope → ValueError.
- **Deviation (necessary):** the play scope also skips the matchmaking checks. They need every anchor's roles, i.e. reading every frozen agent's file (default `anchors: null` = all fixed agents, so `bc_net` would be read). record/bc/eval never use the matchmaker. Documented in the docstring.
- `players.registry.resolve_player_roles(config, spec, players=None)` / `load_fixed_players(config, spec, players=None)`: an optional subset of fixed agents (trainable always resolved). `eval.load_player` resolves only the named scripted agent's roles, so other frozen agents are not read.
- CLI: `record`, `bc` and `eval` take `--set` (same parser/help); eval passes the bare `-a` names, record passes the `--player` / `--against` names without a path (`record._parse_player` → public `record.parse_player`), and bc passes none.
- Pipeline kit: `run_pipeline` writes ONE config (bc_net `path: <workdir>/bc.pt`, init, kickstart) before bc.pt exists and uses it for record → bc → train → eval. The fast smoke now covers the one-config form.
- Docs: README "Competition pipeline", LEAGUE_GUIDE §8 and the `unit_harvest_league.yaml` header comment show the one-config form, and the `--set` form stays mentioned. README record/bc/eval usage lines and the CLAUDE.md command lines show `--set` / `--seed`. Also fixed README "BC about 17 s" → 9 s (C minor 3, `docs/benchmarks.md`), because it is in the rewritten paragraph.
- Tests:
  - new `tests/integration/test_sp3_one_config.py`, covering two cases:
    - `record --player bot`, `bc --agent main` and full `validate` pass on one config with frozen `bc_net` (`path: bc.pt`, absent), `init.from: bc.pt` and `kickstart.teacher: bc_net`. `eval -a bot -a main=<moved .pt>` works without bc.pt. `eval -a bc_net` without the file fails naming bc_net, and works with `--set agents.bc_net.path=...`.
    - `--set` reaches record, bc and eval (parametrized).
  - `tests/unit/test_sp3_validate_players.py::test_the_play_scope_reads_only_the_named_fixed_agents_and_skips_warm_start`.
- RED: `TypeError: validate_config() got an unexpected keyword argument 'scope'`; `Usage: main record/bc/eval [OPTIONS]` (no `--set`). GREEN: 4 passed, and the focused area run (`-k "validate or eval or record or bc or players or cli or pipeline or one_config or entry_points or fixed"`) gave 351 passed.

### 2. Critic warm-up updates value-path normalizers (C-I2) — `0e878c4`
- `APPO._update_normalizers(batch, value_path_only=warmup)`: during the warm-up only `model.update_normalizers(None, global_state)`. `PolicyModel.update_normalizers(obs: Tree | None, ...)` skips obs normalizers for `obs=None`; its docstring is updated.
- Wording updated in the APPO module docstring, the `InitConfig.critic_warmup_steps` description, README, LEAGUE_GUIDE §8 and CLAUDE.md.
- Test: `test_warmup_trains_only_the_value_path_and_keeps_the_workers_policy_bit_identical` now expects the `critic_encoder.norm.rms.*` count and mean to change, while every other non-value tensor (`encoder.norm.*` included) stays bit-identical.
- RED: `assert array(0.0001) > array(0.0001)`. GREEN: 9 passed, and the warm-up/normalizer/APPO subset gave 66 passed, 12 skipped (CUDA). The GPU twin needs no change: `make_test_model` has no normalizers.

### 3. PFSP statistics of evicted snapshots stay gone (B-I1) — `c3b8d90`
- `PfspStats._forgotten`: `forget` adds the key, and `_record` skips forgotten players. Module docstring updated.
- Tests in `test_sp3_pfsp_stats.py`:
  - `test_a_late_result_never_brings_a_forgotten_snapshot_back`;
  - `test_late_results_after_evictions_keep_the_coordinator_table_small` (the reviewer's v10..v40 scenario through the coordinator).
- RED: 2 failed (`'a@ckpt_v10'` back in the snapshot). GREEN: 14 passed.

### 4. SP2 behaviour changes documented (B-I2 = C-I3) — `01ac5e2`
- LEAGUE_GUIDE §2 last sentence: the translation is exact (an example behaves like its SP2 copy loaded by SP3), but league behaviour changed, with a pointer to §16.
- LEAGUE_GUIDE §16 lists every change:
  - the new default mix;
  - snapshots chosen by PFSP (EMA, not cumulative ratings);
  - opponent snapshots in asymmetric games (`predator_prey` ~20 %);
  - `mode: self_play` with several agents meets others' snapshots;
  - league mode drawn per opposing team;
  - `keep_last` + `keep_every: 10` + final (disk growth), and pool import on resume;
  - plus the two existing notes and how to get the SP2 mix back explicitly.
- README `matchmaking` row: one sentence pointing to §16 (README has no compatibility paragraph).

### 5. `play_lineups` forces `collect=False` (T6.2 ruling) — `df3367d`
- Copies of the lineups with `dataclasses.replace(seat, collect=False)`; the caller's lineups are not changed. Docstring updated. `test_sp3_fast_learning.py` eval seats no longer pass `collect=False`.
- Test: `test_play_lineups_never_collects_so_default_seat_assignments_work_for_bots` (default `SeatAssignment("bot")`, caller's lineups untouched).
- RED: `ValueError: eval, env 0, seat 1: agent 'bot' is a scripted player with coll...`. GREEN: 15 passed (eval players, eval engine, fast learning).

### 6. Validation helpers (T1.5 ruling) — `860fd37`
- `_reset_layout(env, config, layout)` and `_step_env(env, config, layout, actions, step)` are shared by `_exercise_env` and `_check_scripted_players`, with identical wording. Covered by the existing validate tests: 40 passed.

### 7. T3.1 wording, inf/nan, `--set` hint — `5728d34`
- `PfspConfig.exponent` and `halflife_games` use `allow_inf_nan=False`, and so do the `matchmaking.layouts` weights. Shares and anchor weights already rejected inf/nan (probed).
- `_check_override_path`: when the parent annotation can hold a dict (anchor map, schedule), the error is: `'matchmaking.anchors' is a map or a schedule, not a section; set the whole value, e.g. --set matchmaking.anchors={random: ...}`.
- `TrainableAgent` docstring, the `get_agent_config` docstring and its ConfigError (networks / algorithm / learner / matchmaking / init / kickstart), the `agents` field description and the module header (config v3) are updated.
- LEAGUE_GUIDE §10 describes the real merge (`opponents` / `pfsp` per key; `anchors` / `layouts` / teammates replaced whole; `init` / `kickstart` per key). It also notes that shares are relative weights and gives the `--set` whole-value hint.
- Tests:
  - `test_v3_bounds` (+5 cases: inf/nan exponent and halflife, inf layout weight);
  - `test_setting_inside_an_anchor_map_or_a_schedule_hints_at_the_whole_value` (2 cases).
- RED: 3 DID NOT RAISE + 2 regex mismatches. GREEN: 120 passed, and the config/agent-kind/matchmaking/warmstart subset gave 344 passed.

### 8. T2.2 eviction (`_notify_evicted`, `Coordinator._evicted`) — `653ad43`
- Every callback error after the first is logged with `logger.exception` before the first is raised.
- The coordinator buffers evictions only when `rollout.match_refresh_interval_sec > 0` (the only drainer is the launcher's refresh). PFSP forgetting is unaffected, and `take_evictions` semantics are unchanged for refresh users.
- Tests:
  - `test_eviction_callback_errors_after_the_first_are_logged`;
  - `test_without_match_refresh_the_coordinator_keeps_no_evictions`.
- RED: `[] == ['callback failed for ckpt_v20']`; `{'agent_0': [...]} == {}`. GREEN: 16 passed.

### 9. T2.1 import log — `c675fe2`
- "Imported N snapshots" counts only linked or copied snapshots (`(k already present)` is appended). `Coordinator.import_snapshots` skips agents without a `checkpoints/<id>` dir in the old run, so there is no "carried over: []" line for them.
- Test: `test_the_import_log_counts_only_linked_snapshots_and_skips_absent_agents`.
- RED: the log said "Imported 2" for a re-import. GREEN: 13 passed (with the SP2 resume integration test, which still finds "snapshot pool carried over").

### 10. Dead aliases — `a3993fa`
- Deleted `core.config.AgentOverride` and `worker.match_runner.ModelPool` (plus `__all__`). `test_sp3_agent_kinds.py` no longer imports the alias, and the `game_helpers` docstring now says `PlayerPool`. 20 passed.

### 11. T3.2/T3.3 mixture and construction helper — `286a6c5`
- The unreachable zero-weight-anchor widening in `_opponent_core` is removed; the existing RuntimeError covers it.
- `lineup_for` raises RuntimeError for "no playable layout". A non-trainable owner keeps its KeyError: the brief named only the layout case, and an existing test asserts KeyError for the owner.
- New `league.base.build_matchmaker(cls, context, path)`: one wrapping, and a class's own ConfigError passes through. It is used by `Coordinator` and `_check_matchmaker_class`; before this, validate double-wrapped a user ConfigError.
- `mixture.CATEGORIES = OPPONENT_CATEGORIES[:4]`, `FALLBACK = OPPONENT_CATEGORIES[4]`.
- Tests:
  - `test_the_runtime_fallback_never_seats_a_zero_weight_anchor`;
  - `test_lineup_for_without_a_playable_layout_is_a_runtime_error`;
  - `test_the_coordinator_and_validate_report_a_failing_matchmaker_init_alike` (RuntimeError and own-ConfigError cases).
- RED: DID NOT RAISE; KeyError; the own ConfigError was wrapped by validate. GREEN: 46 passed.

### 12. Distributed warning names `teammates: mixed` (T6.1) — `094d522`
- One warning when the mix is reduced and/or `teammates: mixed` is set ("matchmaking.teammates: mixed is ignored (...)"). Docstrings updated.
- Test: `test_the_one_warning_also_says_that_mixed_teammates_are_ignored` (latest-only and default mix).
- RED: no warning / no teammates text. GREEN: 23 passed.

### 13. `_check_critic_warmup` wraps import errors (T4.3) — `860fd37`
- An `import_class` failure becomes a ConfigError naming `algorithm.algorithm_class`.
- Test: new case in `test_validate_rejects_a_warmup_the_agent_cannot_do`.
- RED: `ModuleNotFoundError`. GREEN: 10 passed.

### 14. T4.5 DAgger — `200088d`
- `kickstart_label_frac` is computed outside the lambda/warm-up gate: it is the labelled share of ACT slots whenever a scripted teacher's labels are in the batch.
- `_teacher_label` uses `self._episode_layout[env]` directly, with an assert and no fallback.
- Tests:
  - `test_appo_dagger_loss_with_units_is_the_bc_nll_over_labeled_act_slots` (K=4, BC reference `per_sample_nll[labelled].mean()`, label fraction, finite gradients);
  - `test_the_label_share_is_reported_during_the_critic_warmup`.
- RED: `0.0 == 0.6` for the warm-up case. The Units test passed at once: it is coverage of a path that already worked, as the review measured. GREEN: 8 passed, and `-k "dagger or teacher"` gave 62 passed.

### 15. `colosseum bc --seed` — `084a710`
- `--seed` seeds torch and numpy before the model is built, which covers the initial weights, the window offset and the minibatch order (all from torch's global RNG). The pipeline kit's `bc()` requires a seed and `run_pipeline` passes its seed, so the slow pipeline test and `pipeline_vs_scratch` are seeded. The README recipe passes `--seed 0`.
- Test: `test_bc_seed_makes_the_weights_reproducible` (same seed → equal weights, other seed → different).
- RED: no separate log (the option did not exist; same class as the `--set` usage error). GREEN: 15 passed (with the validate players tests).

### 16. `record.py` docstring states P8 — `6cddb66`

### 17. Live example configs validated in the fast suite — `dc85637`
- **Finding:** this already existed. `tests/unit/test_example_configs.py::test_every_example_config_validates` runs the full `colosseum validate` (full `validate_config`) on every `configs/examples/*.yaml` in the fast suite, `team_tag.yaml` and `unit_harvest_league.yaml` included, and `space_miners` is skipped without Box2D. Review B's M5 missed it.
- I only strengthened it: it now asserts the output ends with "Config is valid.". 19 passed.

### 18. CLAUDE.md — `51337cc`, `080128b`, `6fa1406`
- «Parked during SP3» no longer lists the items this wave closed: the validate reset/step duplication, `play_lineups` collect, matchmaker construction, CATEGORIES, KeyError, zero-weight fallback, bc `--seed`, label_frac during warm-up, `--set` into maps, `pfsp.exponent` inf, the critic warm-up import error, `_evicted` growth, `_notify_evicted` logging, the K>1 DAgger test and distributed `teammates: mixed`.
- Updated the warm-up, distributed and smoke notes, the command lines, and the test counts (1689 / 10 / 22; README fast-suite count too).

## Self-review / concerns
- Play scope skips matchmaking checks (item 1 deviation, explained above). A matchmaking error is now reported by `validate` / `train`, not by record/bc/eval. That matches the ruling's "cost if wrong".
- `bc --seed` sets the global torch/numpy seed of the process; harmless for the CLI, while in-process CliRunner tests already seed what they need.
- `KeyError` kept for a non-trainable owner in `lineup_for` (item 11, see above).
- Not done (outside the 18 items): C minor 1 (`kl: reverse` with a scripted teacher), C minor 4 (`teacher_active` during warm-up), A M-2/M-3, B M1/M2/M12/M13 stay parked as decided.
