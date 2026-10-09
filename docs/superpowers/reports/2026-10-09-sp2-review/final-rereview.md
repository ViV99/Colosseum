> Origin: `.superpowers/sdd/2026-10-08-sp2-game-model/final-rereview.md` (SP2 SDD workspace, git-ignored, deleted after the merge). Copied 2026-10-09.
> Frozen record; do not edit. Only absolute scratch paths were replaced with `<scratch>/`.

# SP2 final fix wave — scoped re-review

Range `7d55974..b4d6f93` (10 commits), package `review-final-fix.diff`. Inputs: `final-fix-brief.md` (items 1–10 and
controller rulings), `final-review-{A,B,C}.md` (only the items the brief names), `final-fix-report.md`.
Read-only. Checks run: `ruff check .` (clean); `--collect-only` per marker (1248/1275 fast, 9 slow, 18 gpu); two scratch
probes in `<scratch>/final-rereview/` (`sched.py` for the eval schedule; an inline
`SystemStats` probe). The full suite was not re-run. The fix report shows RED/GREEN output for every behaviour change
and a full fast run at `a461d75`: 1248 passed, 0 warnings.

### Finding Verdicts

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | A / I-1 `UnitsHead` natural names | **ADDRESSED** | `src/colosseum/networks/heads.py:31-42`: `heads`/`log_std` keyed `c{i}`, output keyed by `c.name`. Tests `tests/unit/test_units_head.py::test_units_head_accepts_natural_component_names` (`type`, `to`, `a.b`, `move`) and `::test_natural_component_names_build_step_unroll_and_round_trip_the_state_dict` (LSTM, act → unroll parity → `load_state_dict` into a fresh model). |
| 2 | C / I-1 eval partial role overlap | **ADDRESSED** | `src/colosseum/eval.py:86-113` (`_pair_lineup` fills each team with one agent; `_pair_lineups` falls back to the other orientation and returns `[]` only if both are invalid), `:147`, `default_layouts` `:668-670` (the probe with `num_matches=1` now covers both orientations through the fallback). Tests: `tests/unit/test_sp2_eval_schedule.py:94-112` (1v1 → 4 × `[gen, defn]`; 2v2 → `[]`; `default_layouts == ["1v1"]`). |
| 3 | A / M-1 validate context order | **ADDRESSED** | `src/colosseum/core/validation.py:92-110` (`validate, seat S, episode step N, layout L`), call `:137`. Test anchored with `^` in `tests/unit/test_sp2_validate.py`. ENV_GUIDE §9 updated. Item removed from CLAUDE.md «Parked during SP2» (now 5 items) and from the report's «Открытые пункты» (now 5). |
| 4 | A / M-2 `resolve_outcome` mixed keys | **ADDRESSED** | `src/colosseum/core/outcomes.py:23-28` (`_keys`: sorted, or the env's order on `TypeError`), used at `:34`. New parameter in `tests/unit/test_team_outcomes.py::test_bad_outcomes_raise_with_context` (`[0, 'b']`). |
| 5 | A / M-3 `ObsSpec.check` ragged/None | **ADDRESSED** | `src/colosseum/core/specs.py:142-155`: `None` and mappings at a leaf are rejected; `np.shape` `ValueError` → `EnvContractError` naming the leaf path. Test `tests/unit/test_obs_action_specs.py::test_obs_spec_check_rejects_ragged_and_missing_leaves`. |
| 6 | A / M-4 dead `SubprocessVectorEnv` child | **ADDRESSED** | `src/colosseum/envs/vector.py:224-251`: send/recv failures → `RuntimeError("env child W (envs A..B) died (exit code N) ...")`; the other children's replies are still collected, so the pipe protocol stays in sync; join bounded by `_JOIN_TIMEOUT = 5.0`. `close()` already tolerates dead pipes (`:283-295`). Test `tests/integration/test_game_subproc_vector_env.py::test_a_dead_child_is_reported_with_its_env_range_and_exit_code` covers the recv path, the send path and a surviving child. |
| 7 | B Minor 2 + C — dropped rewards visible | **ADDRESSED** (one Minor semantic issue below, N-1) | WARNING once per `RolloutLoop` (= per worker process) at `src/colosseum/worker/rollout_loop.py:322-334`, later drops at DEBUG. The context has worker, env, seat, episode step (`MatchResult.episode_length`), layout, agent and reward, plus pointers to the metric and ENV_GUIDE. `SystemStats.snapshot` sums the counter (`src/colosseum/metrics/aggregator.py:171-172`). `REQUIRED_KEYS["system"]` includes it (`metrics/jsonl.py`). WandB `system/dropped_reward_episodes` is tested. ENV_GUIDE and README rows updated. Contract test: exactly one WARNING over 3 dropping episodes, exact prefix. |
| 8 | C / M-1 duplicate tests | **ADDRESSED** | `d691de9` deletes the three copies and the `fake_learner_role` fixture from `tests/unit/test_sp2_launcher_lifecycle.py`. The originals remain in `tests/unit/test_sp2_distributed_roles.py:102-165`, gRPC-guarded at `:119`. Ruff is clean, so no unused imports are left. |
| 9 | FIX-2b | **ADDRESSED** | Autouse `_no_lost_interrupt_leaks` in `tests/unit/test_process_lifecycle.py` (before and after each test). Both integration tests extend `child_env()["PYTHONPATH"]` (`tests/integration/test_sp2_lifecycle.py`; `child_env` always sets the key, `tests/cli_runner.py:42`). `serve-weight-store` re-raises after `server.stop(0)` (`src/colosseum/cli.py:347-352`), so `_interrupts` exits 130 with `Interrupted`. Test `test_ctrl_c_stops_serve_weight_store_with_exit_130` (gRPC-guarded). README exit-code bullet updated. Ruling 39 recorded. |
| 10 | B optional (BC docstring, agent check) | **ADDRESSED** | `src/colosseum/bc/offline_bc.py` docstring (skipping only with K > 1). `collect_batch(..., agent_id=)` check at `src/colosseum/learner/learner.py:359-361`, passed from `learner_process` at `:178`. Test `tests/unit/test_learner_v2.py::test_collect_batch_rejects_a_chunk_of_another_agent`. |
| — | Docs/report: counts, «Финальное ревью ветки» | **ADDRESSED** | `--collect-only` at HEAD: 1248/1275 fast, 9 slow, 18 gpu. README `:369` and CLAUDE.md `:249` say 1248 / 9 / 18; no stale 1243 is left outside the historical §3.1 line. Severity table matches the reviews (A 0/1/11 = M-1..M-11; B 0/0/6; C 0/1/6 = M-1..M-6). Ten item lines with the right commits. Ledger 38 and 39 added. The parked list matches the reviews' unfixed items. Open items (5) = CLAUDE.md «Parked during SP2» (5). |

### Focus checks requested by the dispatch

**UnitsHead state_dict keys.** Only `UnitsHead` changed its keys; every caller holds it as a submodule:
`examples/unit_harvest/models.py:48`, `examples/space_miners/models.py:49`, `tests/game_helpers.py:578`,
`tests/learning/sp2_bandits.py:107`, `tests/unit/test_policy_model_v2.py:135`. Nothing indexes `head.heads[<name>]`
or `log_std[<name>]` from outside (grep). Checkpoints, BC output and the kickstart teacher are all saved from and
loaded into a model built by the same code, so they round-trip inside SP2. The eval architecture check
(`checkpoint_manager.check_model_state`) compares key sets of the built model against the file. A checkpoint written
before `a6a2faa` (e.g. a local `runs/` dir from earlier SP2 runs) therefore fails loudly with a "missing / unexpected
keys" `ConfigError`, never silently. No `.pt` files are tracked in git, and no backward compatibility is promised.

**Eval orientation.**
- Probe `sched.py`: with two symmetric agents, ffa4 (4 teams × 1 seat) and 2v2 give every agent every seat exactly
  2× in 4 matches. That is the same balance as before, because when both orientations are valid match m still uses
  `m % 2`, and the CLI still rounds odd counts up (`cli.py:209-212`).
- The one-team (coop / cross-play) path is untouched code: it gives 2 homogeneous lineups per agent plus the
  rotating mixed compositions, as before.
- A rank/FFA layout with partly overlapping roles, `(att | def | def)` with gen/defn, alternates its two valid
  orientations. defn can never sit in the att seat, so perfect seat balance is impossible there by construction.
- The behaviour change the implementer flagged follows spec block 8 («команды заполняются однородно»). Only layouts
  that the two agents could fill only by sharing a team are now skipped; before, those matches were played and then
  counted as `unattributed`.
- Naming such a layout with `--layout` gives the existing `cannot fill its seats` ConfigError. Explicit
  `play_lineups` is unchanged.

**Learner agent check.**
- Local mode: `rollout_worker.send_chunk` routes by `trajectory_queues[chunk.agent_id]`
  (`worker/rollout_worker.py:107-109`). The chunk's id comes from `_seal(track.agent_id, ...)` →
  `build_chunk(aid)`. Buffers are owned by agents, and parked buffers are only re-acquired by the same agent. The
  learner gets its own key (`launcher.py:267`), so the check can never fire here.
- Distributed: `run-learner --agent A` passes `agent_id=A` (`distributed.py:329`). Workers send each chunk to the
  address given for its own `agent_id`, so only a wrong `-l` mapping fires it. That is the intended case: the learner
  dies loudly and the role exits 1.
- Kickstart, BC and the bench script do not call `collect_batch`.

**serve-weight-store.** `server.stop(0)` does not block, then the `KeyboardInterrupt` goes through `_interrupts` →
`sys.exit(130)`, well inside 10 s. A Ctrl-C lost during startup keeps the server serving until the next Ctrl-C, which
README now states. SIGTERM is unchanged: default action, 143 in the shell. The 0 / 1 / 130 / 143 rules of the other
commands are not touched by the diff.

**Dropped-reward WARNING.** One flag per `RolloutLoop`, so one WARNING per worker process; later drops are DEBUG, so
there is no per-episode spam at INFO. The context is complete. The metric semantics are the subject of N-1.

### New Breakage in the Fix Diff

**N-1 (Minor) — the "cumulative" `dropped_reward_episodes` can go down, and the final `system` record can show 0.**
- Where: `src/colosseum/metrics/aggregator.py:158` and `:171-172`. `snapshot()` first prunes workers whose last
  stats are older than `worker_timeout_sec` (3 × 2 s = 6 s), then sums the counter over the survivors.
- When it happens:
  - A worker blocked > 6 s, for example in `send_chunk` on a full learner queue.
  - A `worker_stats` message dropped on a full queue (`report_worker_stats` is best effort).
  - Shutdown. Workers send no final stats, and `_finish_metrics` → `hub.close` writes the last `system` record after
    the children have stopped. That can be up to the 7 s grace, longer than the 6 s timeout, after the last report.
- Probe: report 5 at t = 0 → snapshot at t = 2 gives 5; at t = 8.5 it gives **0**.
- Why it misleads: in WandB the series dips. The final `metrics.jsonl` record, the one a user is most likely to read
  as "the total", may say 0. ENV_GUIDE promises «накопительно по всем воркерам». The pruning is right for gauges
  (`parked_buffers`, `workers_reporting`) and wrong for a counter.
- Minimal correct semantics: a counter that never decreases within a run.
  - Keep a separate per-worker map that the timeout does not prune, e.g. in `on_worker_stats`:
    `self._dropped[w] = max(self._dropped.get(w, 0), stats.get("dropped_reward_episodes", 0))`.
  - In `snapshot`, report `sum(self._dropped.values())`.
  - This is safe because workers are fixed at start and a dead worker stops the run, so nothing is double-counted.
  - Document it as "seat-episodes since this process started" (it restarts at 0 on resume, like the other worker
    stats).
  - A ~3-line change plus a one-line test: a value survives a snapshot after the timeout.
- Not blocking: the WARNING is the primary signal and fires regardless; dips are transient during a run. If it is
  not fixed now, park it with the SP4 observability items and soften the ENV_GUIDE wording.

No other new breakage found:
- `outcomes._keys` keeps sorted output for comparable keys, so the existing messages are unchanged.
- `ObsSpec._check_leaf` accepts Python scalars (`np.shape(3) == ()`); the probe in the test confirms it.
- The `vector._round` return order is preserved.
- `test_sp2_launcher_lifecycle.py` still uses its remaining imports (ruff is clean).

### Out-of-Scope Observations

- `eval.py:31,34`: the module docstring still describes `unattributed` mixed-team matches. That stays correct for
  explicit `play_lineups`, but `schedule_lineups` can no longer produce them.

### Verdict

**Fix round:** All findings addressed, no new Critical/Important breakage. One new Minor (N-1: the
`dropped_reward_episodes` sum drops workers after the 6 s stats timeout, so the "cumulative" counter can dip and the
final `system` record can read 0). Suggested fix: a sticky per-worker max not subject to pruning (~3 lines plus a
test), or park it with the SP4 observability items and soften the ENV_GUIDE wording.
