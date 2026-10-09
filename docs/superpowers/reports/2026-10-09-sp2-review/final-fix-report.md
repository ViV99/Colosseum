> Origin: `.superpowers/sdd/2026-10-08-sp2-game-model/final-fix-report.md` (SP2 SDD workspace, git-ignored, deleted after the merge). Copied 2026-10-09.
> Frozen record; do not edit. Only absolute scratch paths were replaced with `<scratch>/`.

# SP2 final fix wave — report

Branch `sp2-game-model`, base `7d55974`. One commit per item group (no attribution lines). TDD: a failing test first for
every behaviour change (RED output below), then the fix (GREEN). Evidence files: `<scratch>/finalfix/`.
Full fast suite and ruff were run once at the end (after all code commits); focused tests ran per commit.

## Items

### 1. A / I-1 — `UnitsHead` crashes on natural component names — `a6a2faa`
- Change: `src/colosseum/networks/heads.py` keys `heads` / `log_std` by component position (`c0`, `c1`, ...); the output
  dict keeps the user's component names. Docstring says so. No other code depended on the old keys (grep).
- Tests (new):
  - `tests/unit/test_units_head.py::test_units_head_accepts_natural_component_names` (`type`, `to`, `a.b`, `move`)
  - `tests/unit/test_units_head.py::test_natural_component_names_build_step_unroll_and_round_trip_the_state_dict`
    (`make_test_model` with an LSTM core → `act` per step → `unroll` reproduces per-decider log-probs → `state_dict`
    loaded into a fresh model reproduces them again).
- RED: `pytest tests/unit/test_units_head.py` → `KeyError: "attribute 'type' already exists"`; `2 failed, 2 passed`.
- GREEN: `4 passed in 0.27s`.

### 2. C / I-1 — eval scheduling of partly overlapping role sets — `d83ad0f`
- Change: `src/colosseum/eval.py`. `_pair_lineup(spec, layout, pair, orientation, players)` fills each team as a whole:
  its core `pair[(team + orientation) % 2]` if it plays every role of the team, else the other agent if it does, else
  the orientation is invalid; an orientation that leaves one agent out is invalid. New `_pair_lineups`: match m uses
  orientation `m % 2`, or the other when that one is invalid; `[]` only if both are invalid. `default_layouts` keeps
  probing `schedule_lineups(..., 1)`, which is now equivalent to any count (fallback), so it uses the same predicate.
  Module docstring and README eval bullet updated (one sentence on the one-orientation case).
- Behaviour change to note: pairs whose only fill mixes agents inside a team (e.g. `a: [att]`, `b: [def]` with teams
  `(att, def | att, def)`) are no longer scheduled (before: played, then counted as `unattributed`); spec block 8
  requires homogeneous teams. Explicit `play_lineups` is unchanged.
- Tests (new, `tests/unit/test_sp2_eval_schedule.py`):
  - `::test_partly_overlapping_roles_fall_back_to_the_valid_orientation` (1v1: 4 × `[gen, defn]`; `default_layouts == ["1v1"]`)
  - `::test_partly_overlapping_roles_never_mix_a_team` (2v2 → `[]`; `default_layouts` of the 1v1+2v2 game `== ["1v1"]`)
  - `::test_mixed_role_teams_are_filled_by_one_agent_each`
- RED: 3 failed (`assert [] == [['gen','defn'], ...]`; 2v2 returned the mixed `[gen, gen, gen, defn]` lineups).
- GREEN: `11 passed`; all eval test files (`tests/*/test*eval*.py`): `48 passed`.

### 3. A / M-1 (T6.2) — validate context order — `3f32974`
- Change: `src/colosseum/core/validation.py::_check_spaces` takes the step and writes
  `validate, seat S, episode step N, layout L` (same order as the tracker). Docs: ENV_GUIDE sentence about the
  "other order" replaced; the item removed from CLAUDE.md «Parked during SP2» and from the acceptance report's
  «Открытые пункты» (README only states the global order, already correct).
- Test (tightened): `tests/unit/test_sp2_validate.py::test_observation_outside_the_space_is_an_env_contract_error`
  now matches `^validate, seat 0, episode step 2, layout solo: observation is not in the role's space`.
- RED: `Regex pattern did not match. Actual message: "validate, layout solo, episode step 2, seat 0: observation is not in ..."`.
- GREEN: `26 passed`.

### 4. A / M-2 (T1.4) — `resolve_outcome` mixed key types — `52ca7c6`
- Change: `src/colosseum/core/outcomes.py`: key listing in the error uses `_keys()` (sorted when comparable, else the
  env's order) → `EnvContractError("...: outcome.team_rank keys [0, 'b'] must be exactly the layout's teams ...")`.
- Test: `tests/unit/test_team_outcomes.py::test_bad_outcomes_raise_with_context[outcome4-...]` (new parameter).
- RED: `TypeError: '<' not supported between instances of 'str' and 'int'`. GREEN: see item 5.

### 5. A / M-3 (T1.3) — `ObsSpec.check` ragged / `None` / dict leaves — `52ca7c6`
- Change: `src/colosseum/core/specs.py::ObsSpec._check_leaf` rejects `None` and mappings at a leaf
  (`leaf turn is None, expected an array of shape ()`), and a ragged nesting (`np.shape` `ValueError`) becomes
  `EnvContractError("...: leaf <root> is not an array (ragged nesting?), expected shape (2, 2): ...")`.
- Test (new): `tests/unit/test_obs_action_specs.py::test_obs_spec_check_rejects_ragged_and_missing_leaves`.
- RED: `ValueError: setting an array element with a sequence...`. GREEN (items 4+5): `34 passed`; tracker/contract/toy
  games: `236 passed`.

### 6. A / M-4 (T1.6) — dead `SubprocessVectorEnv` child — `18d1abd`
- Change: `src/colosseum/envs/vector.py::_round` catches send failures (`BrokenPipeError`/`EOFError`/`OSError`) and
  `recv` failures (`EOFError`/`OSError`), still collects the other children's replies (protocol stays in sync), then
  raises `RuntimeError("env child W (envs A..B) died (exit code N), e.g. a crash in native env code or an OOM kill")`;
  the child is joined (bounded, `_JOIN_TIMEOUT`) to read its exit code.
- Test (new): `tests/integration/test_game_subproc_vector_env.py::test_a_dead_child_is_reported_with_its_env_range_and_exit_code`
  (`os._exit(7)` in `step`; checks the recv path, the send path on the next call, and that the other child still answers).
- RED: `EOFError`, `1 failed`. GREEN: file `10 passed`; with `test_sp2_worker_threads.py`: `13 passed`.

### 7. B Minor 2 + C (ruling T8.6) — dropped rewards visible — `c5225e6`
- Change:
  - `src/colosseum/worker/rollout_loop.py`: first drop per worker → `logger.warning("worker W, env E, seat S, episode
    step N, layout L: agent 'a' loses reward R of an episode in which this seat never acted; further drops on this
    worker are only counted (dropped_reward_episodes in the system metrics); see docs/ENV_GUIDE.md, rewards of
    waiting seats")`; later drops at DEBUG. Episode step = `MatchResult.episode_length`.
  - `src/colosseum/metrics/aggregator.py::SystemStats.snapshot` sums `dropped_reward_episodes` over reporting workers
    (cumulative per worker) → every `system` record in `metrics.jsonl`, and WandB `system/dropped_reward_episodes`
    (the hub flattens the whole record). `metrics/jsonl.py::REQUIRED_KEYS["system"]` includes it.
  - Docs: ENV_GUIDE "Награды ждущему месту" paragraph rewritten to the real behaviour; README `system` row lists the key.
- Tests:
  - `tests/contract/test_chunk_v2_rules.py::test_a_seat_that_never_acts_drops_its_reward_and_is_counted` (exactly one
    WARNING over 3 dropping episodes, exact context prefix, ENV_GUIDE pointer)
  - `tests/unit/test_sp2_metrics.py::test_system_stats_rates_and_resume_baselines` (sum over 2 workers = 7)
  - `tests/unit/test_sp2_metrics.py::test_hub_forwards_layout_namespaces_to_wandb` (`system/dropped_reward_episodes`)
  - schema: `tests/integration/test_sp2_metrics_outputs.py` (REQUIRED_KEYS on a real run) passes.
- RED: `assert 0 == 1` (no WARNING), `KeyError: 'dropped_reward_episodes'`, `KeyError: 'system/dropped_reward_episodes'`;
  `3 failed, 22 passed`. GREEN: with metrics outputs, bench, lineups contract: `50 passed`.
- `scripts/bench_throughput.py::_make_config` untouched: no config-schema change.

### 8. C / M-1 — duplicated distributed tests — `d691de9`
- Deleted from `tests/unit/test_sp2_launcher_lifecycle.py`: `test_run_workers_returns_an_exit_code` (2 params),
  the `fake_learner_role` fixture, `test_run_learner_returns_128_plus_signum` (2 params),
  `test_run_learner_setup_failure_leaves_signal_handlers_untouched`. The originals in
  `tests/unit/test_sp2_distributed_roles.py` (with `pytest.importorskip("grpc")`) remain. Both files: `43 passed`.

### 9. FIX-2b — `2fb77c3`
- `serve-weight-store` (`src/colosseum/cli.py`): after `server.stop(0)` the KeyboardInterrupt is re-raised, so
  `_interrupts` prints `Interrupted` and exits 130 (controller ruling); a lost startup Ctrl-C keeps serving until the
  next Ctrl-C, then also 130. README exit-code bullet updated (no test asserted 0).
  - Test (new): `tests/unit/test_sp2_launcher_lifecycle.py::test_ctrl_c_stops_serve_weight_store_with_exit_130`
    (fake server, `importorskip("grpc")`). RED: `assert 0 == 130`. GREEN: `1 passed`.
- `tests/unit/test_process_lifecycle.py`: autouse fixture `_no_lost_interrupt_leaks` calls `take_lost_interrupt()`
  before and after each test.
- `tests/integration/test_sp2_lifecycle.py`: `test_ctrl_c_inside_string_exec_during_startup_still_exits_130[exec/eval]`
  and `test_ctrl_c_lost_in_a_weakref_callback_during_startup_still_exits_130` extend `child_env()["PYTHONPATH"]`
  (tests/ dir kept) instead of replacing it. These + `test_process_lifecycle.py`: `24 passed`.

### 10. B optional — `a461d75`
- Minor 4: `learner.collect_batch(..., agent_id=None)`; `learner_process` passes its `agent_id`; a foreign chunk raises
  `ValueError("learner of agent 'a' received a chunk of agent 'b'; check the workers' learner addresses (run-workers -l AGENT=HOST:PORT)")`.
  Test (new): `tests/unit/test_learner_v2.py::test_collect_batch_rejects_a_chunk_of_another_agent`.
  RED: `TypeError: collect_batch() got an unexpected keyword argument 'agent_id'`. GREEN: `1 passed`; learner/BC/
  distributed files: `55 passed, 2 skipped` (gRPC e2e skipped without the extra, as before).
- Minor 3: `bc/offline_bc.py` module docstring: only K > 1 decisions without a valid decider are skipped; with K = 1
  every decision counts (an absent single unit adds NLL 0). Docstring only.

### Docs and acceptance report — `b4d6f93`
- README / CLAUDE.md fast-test count 1243 → 1248 (collect-only at final HEAD: fast 1248/1275, slow 9, gpu 18).
- Acceptance report: «Итог» notes the final review and wave; §3.1 has a «Повтор после финальной волны» block; the
  «Кандидаты финального ревью» paragraph replaced by a pointer; new section «Финальное ревью ветки» (severity counts
  per area, one line per item with commit, what stays parked); ledger entries 38 (T8.7 review folded into the final
  review, verbatim) and 39 (serve-weight-store 130 ruling). Open items (5) = CLAUDE.md «Parked during SP2» (5).

## Final verification
- Full fast suite at `a461d75` (all code changes; the later commit `b4d6f93` is docs only):
  `OMP_NUM_THREADS=1 .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw -p no:cacheprovider`
  → `1248 passed, 27 deselected in 343.50s (0:05:43)`, exit 0; grep for a warnings summary: none (0 warnings).
- `.venv/bin/ruff check .` → `All checks passed!` (also at `b4d6f93`).
- `git status --porcelain --ignored`: only `__pycache__`, `.venv`, caches, `.superpowers` (no runs/ or stray files).
- Collect-only at the final HEAD: fast `1248/1275 (27 deselected)`, slow `9/1275`, gpu `18/1275`.
  Delta vs 1243: +10 new tests (2 units head, 3 eval schedule, 1 outcome param, 1 obs check, 1 dead child,
  1 serve-weight-store, 1 foreign chunk), −5 deleted duplicates.
- Slow tests not run (no training math touched). All commits pushed to `origin/sp2-game-model`.

## Commits
a6a2faa, d83ad0f, 3f32974, 52ca7c6, 18d1abd, c5225e6, d691de9, 2fb77c3, a461d75, b4d6f93.

## Self-review / concerns
- Item 2 changes one eval behaviour beyond the reviewer's two cases: a pair that can only fill a layout with mixed-agent
  teams (each agent plays a different role of the same team) is now not scheduled for that layout (naming it with
  `--layout` gives the existing "cannot fill its seats" ConfigError) instead of producing `unattributed` matches.
  This follows spec block 8 ("команды заполняются однородно"); one-team (cross-play) layouts are unaffected.
- Item 7: the `system` sum covers workers that reported within the stats timeout, like `parked_buffers`; in
  distributed mode there is no metrics hub, so only the WARNING applies there.
- `final-fix-report.md` is in `.superpowers/` (not committed, as for the other SDD files).

## Addendum after the re-review (docs only)
- Commit `62157a6` `docs: park the system dropped-reward counter semantics (SP4); eval docstring`.
- ENV_GUIDE and README: `system` `dropped_reward_episodes` = sum of the latest counts of recently reporting workers (can
  drop when a worker stalls, can read 0 at shutdown); the worker-log WARNING is the reliable signal.
- CLAUDE.md «Parked during SP2» + report «Открытые пункты» item 6: N-1 parked for SP4; «Финальное ревью ветки» notes the
  re-review (all 10 items addressed, N-1 parked).
- `eval.py` module docstring: mixed-team (`unattributed`) lineups come only from explicit `play_lineups`.
- Checks: `ruff check .` clean; T8.6 Step 7 link check "links ok". Its `.gitignore` line check prints nothing because the
  entry is `/eval.json` (anchored, pre-existing since T8.6), not `eval.json`; unchanged.
