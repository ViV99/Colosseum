# SP5 Plan — Part 4: Learners on any node, failures, in-place resume, lifecycle

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** This part owns the following items of `docs/superpowers/specs/2026-10-17-sp5-distributed-design.md` (Russian; the binding authority; read it fully first):
- **Block 1 «Лёрнер» and 2.6** (learner placement and slots): one learner per agent on any node, chosen by `learner.device` (`auto` / `cpu` / `cuda` / `cuda:N` / `mps`) and `learner.placement.{node,tags,strict}`; GPU slots one per GPU (`--gpus`), CPU learner slots (`--cpu-learners`), `--no-learners`; train's local node takes every learner (ruling R18); an agent without a slot is `pending` (its seats do not collect; it plays with its last weights when it has any, else it is not drawn) — **T4.1**.
- **7.2** (learner generations): every assignment has a new generation; announcements, checkpoints, recovery points, chunks and seat results of a superseded generation are fenced by the version cap (ruling R16); a learner that kept running on a cut-off machine is told to stop when the machine returns — **T4.1** (recovery points in T4.2).
- **5.2** (checkpoints and recovery points through the hub): recovery points every `learner.recovery_interval_sec` (300 s) in `runs/<name>/recovery/<agent>/`, overwritten, never in the pool; trainer state kept as bytes by the hub and read by a learner only with `weights_only=True`; a restart before the first recovery point starts from the hub-start source (`init` or the resumed snapshot); versions continue from the maximum + 1 — **T4.2**.
- **7.1** (what is automated): worker / evaluator restarts (≤ 3 per 10 min, new connections, new epochs, `worker_restarts`, slot disabled after the limit), learner restarts (≤ 3 per 30 min from the recovery point with a new intake address, then `failed`: the others go on, exit 1 at the end), machine loss (`node_lost_after_sec`: cells and unfinished matches dropped), learner relocation after `learner_relocate_after_sec` (new generation, placement preference unless `strict`, else `pending`), a short network cut (session resumed, same cells) — **T4.2**; hub away, bad data, full `/dev/shm` — **T4.4**; contract errors with the node in their context — **T4.4**.
- **7.4** (stop): the final snapshot by a dedicated big-data call that a full queue cannot drop — **T4.2**; the stop flow (budget / Ctrl+C / SIGTERM, `stop_run`, finals, result flush, `shutdown_grace_sec` 20 s, the local 10 s guarantee kept, nodes stay connected unless `--once`) — **T4.4**.
- **5.4** (in-place resume): `hub --resume RUN` / `train --resume RUN` continue the same dir and run id; refused while `run_state.json` names a live process; config precedence (saved config + `agents_state.json` without `-c`; with `-c` a diff WARNING and strict checks); `--set`; restored counters, epochs (R7: the next block), league (R8: replay of the journal rows after the watermark), snapshots, recovery points; nodes reconnect by run id; SP4 run dirs continued in place (journal migration, counters from `metrics.jsonl` and the journal, learners from snapshots); `training.resume_from` unchanged (child run) — **T4.3**; and from **4.3** / **4.4**: the journal `node` column with a schema version and the in-place migration (also on import), the WARNING when resuming at or past the budget — **T4.3**.
- **7.5** exit codes (`hub` / `train`, `join`), **7.6** monotonic clocks (seeds are T2.2's), **7.7** node logs and the 64 KB tails in the run dir, **7.8** portability (parent death, process liveness, shm dir, socket paths) — **T4.4**.
- Controller rulings: **R7** (epoch blocks continue after a resume, T4.3), **R8** (accounting after commit; its replay, T4.3), **R9** (learner ids and versions: generation per assignment, version floor, hub-stamped `env_steps`; T4.1 / T4.2), **R16** (fencing by the version cap, T4.1).

Not in this part: chunk routing, the relay and the data port (T3.2), weights across machines and `GetLatest` (T3.1), `max_policy_lag` (T3.3), evaluator processes and pulse on blocks (T5.1: evaluators get T4.2's restart policy by construction), `join` / registration checks / `nodes …` (T6.x), the `system` key contract, `nodes.json`, the «Nodes» section (T7.1), `validate`'s placement lines (`placement.describe_plan`, T7.2), agents on the fly and `agents_state.json` (T8.2), the recipes of 7.3 «не автоматизируется» (docs, T7.5).

**What the part delivers.** After T4.4:
- the hub places every trainable agent's learner on a node slot of the right device, keeps it there while the node lives, gives every assignment a new generation and fences everything a superseded generation sends; agents without a slot are `pending` and play with hub-held ("parked") weights when they have any;
- learners send recovery points to the hub; a crashed learner comes back from the newest one on its node with a new generation and new versions above every version ever used; a learner that keeps crashing makes its agent `failed` (the others train on; exit 1 at the end); workers and evaluators that die are restarted with new epochs, and a slot that keeps dying is switched off;
- a lost machine's cells disappear at once and its learners move to another node after `learner_relocate_after_sec`; a returning machine's stale learner is stopped;
- `train --resume RUN` / `hub --resume RUN` continue a run dir in place (also an SP4 one): same run id, journal (migrated to schema 2 with a `node` column), counters, epochs, league (replayed from the journal watermark), snapshots, recovery points and learners;
- the hub and nodes survive a hub crash: nodes keep playing their lineups, learners keep training through the direct data plane, results wait in the node buffer and arrive once after `hub --resume`; after `hub_lost_timeout_sec` nodes stop their processes and keep reconnecting;
- the distributed stop collects every learner's final snapshot within `distributed.shutdown_grace_sec`; every exit code of spec 7.5; remote processes' last 64 KB of log in `runs/<name>/logs/nodes/<node>/<proc>.log.tail`; parent-death and liveness work without `/proc` and `prctl` (macOS by design).

**Read `00-overview.md` first** (global constraints, file map, the binding interface contract). Names used here are the contract's; additions and deviations are listed in `## Contract notes` at the end of this part.

**Conventions used in every task.**
- Run every command from the repository root as `env -u PYTHONPATH .venv/bin/python ...` / `env -u PYTHONPATH .venv/bin/ruff ...` (the owner's machine: ROS sets `PYTHONPATH`).
- "Full fast suite" means `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, zero warnings), followed by `env -u PYTHONPATH .venv/bin/ruff check .`. "Slow suite" means `env -u PYTHONPATH .venv/bin/python -m pytest -m slow -v`, run once, alone, in the background on a quiet machine; "GPU suite" means `env -u PYTHONPATH .venv/bin/python -m pytest -m gpu -q -W error`.
- New test files have basenames that start with `test_sp5_` (unique across `tests/`); no `__init__.py`; support modules by bare name (`game_helpers`, `cli_runner`, `hub_kit`, `node_kit`, `cluster_kit`, `fault_proxy`, `net_kit`).
- Every file a test writes is under `tmp_path`; the autouse `_isolate_colosseum_dirs` (T1.3) redirects `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `COLOSSEUM_SHM_DIR` into `tmp_path`; two nodes of one test get two identity roots (`NodeOptions.identity_root`). Fixtures under `tests/fixtures/` are read-only (`copy_sp4_run(tmp_path)`).
- At most 2 loaded CPU processes per fast test (workers, learners, evaluators; hub and node processes are not counted), at most 3 per slow test; `OMP_NUM_THREADS=1`. A test that needs more than about 20 s is `@pytest.mark.slow`.
- Hub logic is tested on pure objects (`colosseum.hub.placement`, `hub.recovery`, `hub.resume`, `hub.state`) and with fake nodes (`hub_kit.FakeNode`: in-process clients of the real hub server on loopback); node logic with a fake hub (`node_kit.FakeHub`) (R13). Real processes only where the process boundary is the subject (kills, signals, exit codes, CLI resume). CLI processes start through `tests/cli_runner.py` (`run_in_session` for `hub`).
- Timeouts, cadences, leases and waits use `time.monotonic()`; tests wait by counters and files (`hub_state.json`, `metrics.jsonl`, the journal, received messages) with generous timeouts, and never assert on wall-clock values or rates. Distributed timeouts in tests are shrunk through the config (`distributed.heartbeat_sec: 0.1`, `node_lost_after_sec: 0.5`, `learner_relocate_after_sec: 1.0`, `hub_lost_collect_sec: 1.0`, `hub_lost_timeout_sec: 2.0`, `shutdown_grace_sec: 5.0`, `learner.recovery_interval_sec: 0.5`); module constants that are not config keys (`WORKER_RESTARTS`, `LEARNER_RESTARTS`, `HUB_LOST_RETRY_SEC`, `RECONNECT_*`) are monkeypatched only for in-process nodes.
- Messages go through `logging` (never `warnings.warn`), so `-rw` stays empty.
- Commit messages use conventional prefixes and carry no attribution lines; push after every task (`git push`, branch `sp5-distributed` or the task's worktree branch).

**Cross-task notes.**
- **The code these tasks edit does not exist yet.** After T2.5 the hub is `colosseum.hub.hub.Hub` (the evolved `Launcher`, R5) with inbox handlers (`_on_session_opened`, `_on_session_closed`, `_on_lineup_request`, `_on_result_batch`, `_on_metrics_batch`, `_on_upload`, `_on_version`, `_on_event`, `_on_admin`) and main-loop deadlines (node loss, TTL bookkeeping, commit intake, retention, metrics tick + `hub_state.json`, budget, signals); the node is `colosseum.node.agent.NodeAgent` with `ChildSupervisor`, the results buffer, lineup pull, cache fetch and the session client; T3.1–T3.3 add version distribution, routing, relay and lag rules. Each task below gives the behaviour, the interface and the edit points by function / class; the implementer adapts the edits to the code as it then is. Pure modules (`hub/placement.py`, `hub/recovery.py`, `hub/resume.py`, `node/uploads.py`) and the library edits of `arena/matchlog.py`, `arena/retention.py`, `coordinator/coordinator.py`, `core/run_state.py`, `utils/process.py` are given in full.
- **T4.1 → T4.2 → (T4.3 ‖ T4.4).** T4.1 creates the learner table that T4.2 drives on failures; T4.3 and T4.4 run in parallel worktrees. Their split of shared files (overview task index): `hub/hub.py` — T4.3 owns the start / resume functions (`run_training(resume=)`, `run_hub(resume=)`, `Hub._resume_hub_state`, `_load_hub_state`, `_replay_journal`, `_resume_start_sources`), T4.4 owns `_shutdown`, `_stop_nodes`, `_exit_code_after_shutdown`, `_signal_exit_code`, `_write_log_tail`, `_count_drop`; `cli.py` — only T4.3; `node/agent.py`, `node/session.py`, `node/procs.py`, `node/logs.py`, `utils/process.py`, `core/run_state.py` — only T4.4; `arena/matchlog.py`, `arena/retention.py`, `coordinator/coordinator.py`, `core/run_dir.py` — only T4.3. If both need a one-line hook in the other's function, the later one to integrate adds it during the cherry-pick.
- **Learner states and who decides.** The hub's `LearnerTable` (T4.1) is the only authority on where a learner runs and with which generation; nodes run what the hub assigns (`start_learner` / `stop_learner` commands and `Welcome.learners`, the desired set on every (re)connect). Restart limits are counted by the node (`ChildSupervisor.may_restart`, `LEARNER_RESTARTS` / `WORKER_RESTARTS`), but a learner restart is always a new assignment from the hub (a new generation, a start point the hub holds); while the hub is away a dead learner stays down (ruling P6).
- **Parked weights.** An agent without a learner (pending, relocating, failed) is played with weights the hub holds (the newest of its recovery point, its newest stored snapshot, its `init` weights, its `training.resume_from` state), announced as a `VersionInfo` with `node = 0`, `relay_only = True`, `generation = 0` and served by `Download(kind="latest")` from `runs/<name>/blobs/latest-<agent>-parked.frame` (T4.1, contract notes). Generation 0 is never stale (T2.1's `is_stale` rule for unknown generations).
- **Start points of a learner** (one rule, three tasks): a recovery point of this run with a version not below the hub-start source's (T4.2); else the hub-start source: the newest stored snapshot with `trainer_state.pt` on an in-place resume (T4.3), the agent's `training.resume_from` state on a fresh run (T4.1), its `init` weights, or factory weights. `LearnerAssignment.weights_blob` names it (`start:recovery`, `start:resume`, `start:ckpt_v<N>`, `init`, ``""``); every `start:*` blob is a checkpoint frame (`encode_checkpoint`: weights + trainer-state bytes), so the node has one decode path.
- **Version floor** of every assignment = max(`VersionRegistry.version_floor(agent)`, the newest stored snapshot's version + 1, the recovery point's version + 1) on the hub, and on the node additionally the agent's `latest` pointer in the machine's cache + 1 (R9; no version number ever names two different weights).
- **Tests rely on these kit capabilities** (T2.3 / T2.4 / T3.1 / T3.2 create the kits; a task that finds a capability missing adds it to the kit and lists it in its report): `hub_kit.FakeNode` (register; open a session with a given `Hello` (slots, hardware GPUs, learners, data addresses, name, tags); send any `NodeMessage`; `received` HubMessages and `wait_for(predicate, timeout)`; `upload(header, data) -> UploadReply`; `download(BlobRequest) -> (bytes, meta_json)`; `close_session()`; `pause_heartbeats()` / `resume_heartbeats()`); `node_kit.FakeHub` (scripted `Welcome`, `send_command(...)`, blob serving by `(kind, agent, ident)`, recorded `NodeMessage`s and uploads, `drop_session()`); `cluster_kit` (`hub_in_thread(config, run_dir, options)` with `.hub`, `.address`, `.run_dir`, `.stop()`, `.result()`; `node_in_thread(options)` with `.agent`, `.stop()`, `.result()`; `wait_until(predicate, timeout)`; `read_hub_state(root)`); `fault_proxy.TcpFaultProxy(target)` with `drop()`, `delay(sec)`, `cut(sec)`, `restore()`, `port`, and `retarget(port)` (added by T4.3 if missing).

**Verified while writing this part.** A scratch package (`src/` of `3022e6a` copied to the scratchpad, with stand-ins for the parts of earlier tasks it needs: T1.4's `LearnerConfig.placement` / `recovery_interval_sec`, a header-only subset of T1.1's frame layout (`encode_frame`, `peek_meta`), T1.2's `MatchResult.node`, T2.1's `HUB_STATE_FILE` / `load_hub_state`) ran the full code of this part against the real repo code with `env -u PYTHONPATH .venv/bin/python`:
- `hub/placement.py` with `test_sp5_placement.py` (26 tests), `hub/recovery.py` with `test_sp5_recovery_store.py` (5), `node/uploads.py` with `test_sp5_upload_queue.py` (6), `hub/resume.py` with `test_sp5_resume_plan.py` (10), the journal schema 2 with `test_sp5_matchlog_v2.py` (6, against a real SP4 journal: an SP4 run dir made by `colosseum train` of the unchanged tree, standing in for the T0.1 fixture), the league replay with `test_sp5_league_replay.py` (7: a coordinator that loads the state and replays the rows after the watermark equals one that accounted everything, for SP5, SP4 and missing league states), retention with `test_sp5_retention_resume.py` (2), the portability helpers with `test_sp5_portability.py` (6): all green, `ruff check` clean;
- the existing SP4 suites against the patched library modules: `test_sp4_matchlog.py`, `test_sp4_matchlog_writer.py`, `test_sp4_retention.py`, `test_sp4_run_state.py`, `test_process_lifecycle.py`, `test_sp4_report_data.py`, `test_sp4_tournament_collect.py` (125 passed) and the training-run integration tests `test_sp4_journal_runs.py`, `test_sp4_sp3_resume.py`, `test_sp4_retention_runs.py` (6 passed: real `colosseum train` runs writing a schema-2 journal); one expected failure, `test_sp4_league_state.py::test_the_coordinator_saves_and_loads_league_state`, which pins the league-state keys (T2.5 adds `journal_row` and updates it);
- `check_resumed_agents` on that real SP4 run dir: the saved config passes, a config with another `networks.kwargs` is refused with the agent and the snapshot named.
- Not verified (the hub / node code of T2.x–T3.x does not exist yet): the `VersionRegistry` parking additions, the learner's recovery cadence test (written against T2.2's `ChunkIntake` / `LearnerOutputs` protocols), and every integration test of the hub and node; they are specified by name, setup and assertions.

**Traceability (spec item → task → test).**

| Spec | Item | Task | Test (file :: test) |
|---|---|---|---|
| 1 | learner of an agent on any node, by device | T4.1 | `test_sp5_placement` :: `test_one_learner_per_gpu_and_cpu_slots_count`, `test_cuda_index_is_honoured_across_nodes`; `test_sp5_learner_placement` :: `test_devices_and_slots_decide_where_learners_start` |
| 1 | `auto` → GPU slot if any GPU node, else CPU | T4.1 | `test_sp5_placement` :: `test_auto_takes_cuda_when_any_node_offers_cuda_else_cpu`, `test_auto_waits_for_a_busy_gpu_instead_of_taking_cpu` |
| 1 | `placement.node` / `tags` preferred, `strict` | T4.1 | `test_sp5_placement` :: `test_preferred_node_first_then_any_other`, `test_strict_never_leaves_the_preferred_nodes`, `test_tags_must_all_match` |
| 1 | a learner never moves while its machine lives | T4.1 | `test_sp5_placement` :: `test_a_learner_never_moves_to_its_preference_while_its_node_lives` |
| 2.6 | GPU slot per GPU, `--gpus`, CPU learners, `--no-learners` | T4.1 | `test_sp5_placement` :: `test_no_learner_nodes_offer_nothing`, `test_capacity_from_hello_reads_gpu_kinds` |
| 2.6 | local node of `train` takes every learner (R18) | T4.1 | `test_sp5_placement` :: `test_local_node_takes_every_learner_on_one_gpu_like_before`; `test_sp5_learner_gpu` :: `test_two_agents_share_the_local_gpu` |
| 2.6 | `pending`: no collecting; last weights if any, else not in lineups | T4.1 | `test_sp5_learner_placement` :: `test_an_agent_without_a_slot_is_pending_then_starts_when_a_slot_appears`, `test_a_pending_agent_plays_with_parked_weights`, `test_a_pending_agent_without_weights_is_not_drawn` |
| 4.5 | versions never go back after a restart | T4.1, T4.2 | `test_sp5_learner_placement` :: `test_version_floor_continues_above_announced_and_stored_versions`; `test_sp5_learner_start` :: `test_the_floor_is_above_the_cached_pointer`; `test_sp5_hub_failures` :: `test_a_restarted_learner_never_reuses_a_version` |
| 7.2 | generations; stale announcements / checkpoints / recovery points / results / chunks refused; old learner stopped | T4.1, T4.2 | `test_sp5_learner_placement` :: `test_a_late_announcement_of_an_old_generation_is_fenced`, `test_stale_generation_checkpoints_and_results_are_refused`, `test_a_reconnecting_node_keeps_its_current_learner`; `test_sp5_hub_failures` :: `test_a_recovery_point_of_a_stale_generation_is_refused`; `test_sp5_relocation` :: `test_a_long_cut_relocates_the_learner_and_stops_the_old_one` |
| 5.2 | snapshots via the hub; final by a dedicated call | T4.2 | `test_sp5_upload_queue` (all); `test_sp5_node_restarts` :: `test_a_final_snapshot_is_never_lost_behind_periodic_ones` |
| 5.2 | recovery points (interval, overwritten, not in the pool, bytes, `weights_only`) | T4.2 | `test_sp5_recovery_store` (all); `test_sp5_learner_recovery` (all); `test_sp5_hub_failures` :: `test_a_learner_death_restarts_it_from_the_recovery_point` |
| 5.2 | restart before the first recovery point: `init` / resume source, versions continue | T4.2 | `test_sp5_hub_failures` :: `test_a_restart_before_any_recovery_point_uses_the_start_source` |
| 7.1 | worker / evaluator restart ≤ 3 / 10 min, new connections and epochs, slot off | T4.2 | `test_sp5_node_restarts` :: `test_a_killed_worker_restarts_with_a_new_incarnation`, `test_a_worker_that_keeps_dying_disables_its_slot`; `test_sp5_hub_failures` :: `test_a_disabled_slot_releases_its_cells`; `test_sp2_lifecycle` :: `test_a_killed_worker_is_restarted_and_its_log_named` |
| 7.1 | learner restart ≤ 3 / 30 min, new intake address, then `failed`, exit 1 | T4.2 | `test_sp5_node_restarts` :: `test_a_killed_learner_is_reported_and_its_siblings_keep_running`, `test_the_learner_restart_limit_reports_learner_failed`; `test_sp5_hub_failures` :: `test_a_failed_learner_stops_its_agent_but_not_the_run`; `test_sp2_lifecycle` :: `test_learner_crash_in_train_step_gives_exit_1` |
| 7.1 | machine loss: cells and unfinished matches dropped, rejoin | T4.2 | `test_sp5_hub_failures` :: `test_machine_loss_drops_cells_and_parks_its_learner`; `test_sp5_relocation` :: `test_a_cut_shorter_than_node_lost_changes_nothing` |
| 7.1 | relocation after `learner_relocate_after_sec`, else pending | T4.2 | `test_sp5_hub_failures` :: `test_relocation_after_the_wait_moves_the_learner`, `test_relocation_without_a_free_slot_waits_as_pending`, `test_strict_placement_waits_for_its_node`; `test_sp5_relocation` :: `test_a_long_cut_relocates_the_learner_and_stops_the_old_one` |
| 7.1 | hub away: lineups past TTL, learning on, results buffered, `hub_lost_collect_sec`, `hub_lost_timeout_sec` | T4.4 | `test_sp5_hub_away` (all); `test_sp5_hub_restart_slow` :: `test_kill_the_hub_and_resume_in_place_with_a_live_node` |
| 7.1 | bad data dropped, counted, one WARNING per node | T4.4 | `test_sp5_bad_data` (all) |
| 7.1 | full `/dev/shm` / disk: ERROR with hint, no work taken | T4.4 | `test_sp5_hub_away` :: `test_a_full_cache_stops_taking_work_until_space_returns` |
| 7.1 | contract / player error: run stops, exit 1, node in the context | T4.4 | `test_sp5_stop_and_exit` :: `test_a_player_error_on_a_remote_node_stops_the_run_with_its_context` |
| 7.4 | stop: `stop_run`, finals, results flushed, grace, nodes stay connected | T4.4 | `test_sp5_stop_and_exit` :: `test_budget_stop_collects_final_snapshots_and_the_node_stays`, `test_a_once_node_exits_0_after_the_run` |
| 7.4 | local 10 s guarantee kept | T4.4 | `test_sp2_lifecycle` :: `test_sigterm_stops_all_descendants_within_10s` (unchanged); `test_sp5_stop_and_exit` :: `test_the_local_grace_applies_without_remote_nodes` |
| 7.5 | exit codes `hub` / `train` / `join` | T4.4 | `test_sp5_stop_and_exit` :: `test_hub_exit_codes`, `test_node_exit_codes` |
| 7.6 | monotonic clocks only | T4.4 | `test_sp5_portability` :: `test_sp5_packages_measure_time_with_the_monotonic_clock` |
| 7.7 | node logs; 64 KB tails in the run dir; both paths named | T4.4 | `test_sp5_log_tails` (all) |
| 7.8 | parent death, liveness, shm dir, socket paths portable | T4.4 | `test_sp5_portability` (all) |
| 5.4 | in-place resume of the same dir and run id | T4.3 | `test_sp5_resume_in_place` :: `test_train_resume_continues_the_run_in_place` |
| 5.4 | refused while the process lives | T4.3 | `test_sp5_resume_plan` :: `test_refused_while_the_run_process_lives_and_allowed_after`; `test_sp5_resume_in_place` :: `test_resume_is_refused_while_the_run_is_alive` |
| 5.4 | config precedence, `-c` diff WARNING, strict checks, `--set` | T4.3 | `test_sp5_resume_plan` :: `test_without_c_the_saved_config_with_set_and_the_dir_pinned`, `test_with_c_the_new_config_and_a_diff_warning`; `test_sp5_resume_in_place` :: `test_a_changed_architecture_is_refused_before_anything_is_written`, `test_with_c_the_diff_is_logged_and_an_agent_can_be_added` |
| 5.4 | counters, epochs (R7), league replay (R8), snapshots, recovery points, learners | T4.3 | `test_sp5_league_replay` (all); `test_sp5_resume_in_place` :: `test_train_resume_continues_the_run_in_place`; `test_sp5_hub_restart_slow` |
| 5.4 | SP4 run dir continued in place | T4.3 | `test_sp5_resume_in_place` :: `test_an_sp4_run_dir_continues_in_place`; `test_sp5_resume_plan` :: `test_sp4_start_env_steps_and_the_budget_warning` |
| 5.4 | `training.resume_from` unchanged (child run) | T4.3 | `test_sp5_resume_plan` :: `test_resume_from_is_not_applied_again`; `test_sp4_sp3_resume`, `test_sp4_league_state_resume` (unchanged, still green) |
| 4.3 | journal `node` column, schema version, migration in place and on import | T4.3 | `test_sp5_matchlog_v2` (all) |
| 4.4 | WARNING when resuming at / past the budget | T4.3 | `test_sp5_resume_plan` :: `test_sp4_start_env_steps_and_the_budget_warning` |

---
### Task T4.1: Learner placement, devices, `pending`, start / stop with generations, fencing

Spec block 1 «Лёрнер» (placement by device, preference, never moves while its machine lives), 2.6 (slots and `pending`), 4.5 («номера версий никогда не идут назад»), 4.7 (no collecting seats without a learner), 7.2 (generations); rulings R9, R16 (the version cap), R18 (the local node takes every learner). The hub gets a learner table and a placement pass: every trainable agent's learner is assigned to a slot of the right device on an online node with a new generation, version floor and start point (`start_learner`); a node that reconnects is reconciled (its learners of the current generation are adopted, everything else is stopped); agents without a slot are `pending`, do not collect, and play with hub-held weights when they have any; everything a superseded generation sends is fenced. The node runs what it is assigned: it starts the learner child on the assigned device with the generation and floor, stops superseded ones, and treats `Welcome.learners` as the desired set.

**Split point:** dispatch (a) Steps 1–6 (the library, the registry additions, the hub side and its tests), (b) Steps 7–11 (the node side, the GPU test, suites, commit). (b) needs (a) merged.

**Files:**
- Create: `src/colosseum/hub/placement.py`
- Modify: `src/colosseum/hub/state.py` (`VersionRegistry.park` / `unpark` / `parked`; `HubStateData.learners`)
- Modify: `src/colosseum/hub/hub.py` (learner table, `_node_capacities`, `_place_learners`, `_start_learner`, `_stop_learner`, `_learner_start_source`, `_version_floor`, `_park` / `_drop_parked`, reconcile in `_on_session_opened`, `_on_version`, fencing in `_on_upload`, `_lineup_for_cell`, `_on_metrics_batch` stamps, `hub_state.json` learners, `pending_agents`)
- Modify: `src/colosseum/hub/server.py` (the `Download` servicer: kind `latest` from a parked frame, kind `start`; `BlobStore.parked_path`, `BlobStore.start_path`)
- Modify: `src/colosseum/net/proto/colosseum_v1.proto` + regenerated `colosseum_v1_pb2*.py` (`Event.generation`; `scripts/gen_proto.sh`)
- Modify: `src/colosseum/node/agent.py` (commands `start_learner` / `stop_learner`, `Welcome.learners` as the desired set, start-blob fetch), `src/colosseum/node/learner_main.py` (the assignment: device, generation, floor, start state, seed)
- Modify: `src/colosseum/learner/learner.py` (`send_final`), `src/colosseum/utils/seeding.py` (`learner_seed(..., generation=1)`)
- Test: `tests/unit/test_sp5_placement.py`, `tests/unit/test_sp5_version_parking.py`, `tests/integration/test_sp5_learner_placement.py`, `tests/integration/test_sp5_learner_start.py`, `tests/integration/test_sp5_learner_gpu.py`

**Interfaces:**
- Consumes: `VersionRegistry`, `VersionInfo`, `NodeRegistry` / `NodeEntry` (`slots`, `hardware`, `tags`, `data_path`), `CellTable`, `OwnerRotation`, `ResultIntake`, `HubStateData` (T2.1); `SessionHandle.send`, `BlobStore`, inbox events (T2.3); `Hub` handlers and `LocalNodeOptions` (T2.5); `LearnerAssignment`, `Command`, `RunningLearner`, `AgentState`, `Welcome`, `Announce` (T1.2); `encode_weights` / `WeightsMeta` / `encode_checkpoint` / `decode_checkpoint` (T1.1); `WeightCacheReader.latest_version` (T1.5); `ChildSupervisor`, `node.learner_main.learner_target` (T2.4); `learner_process(version_floor=...)` (T2.2); `LearnerConfig.device` / `placement` (T1.4); T3.1's version distribution and `GetLatest`; T3.2's collect rule and `LearnerRoute`s.
- Produces (contract `colosseum.hub.placement` plus additions marked *):
  - `LearnerSlot`, `NodeCapacity` (*`mps`), `PlacementPlan` (*`new(current)`), `place(...)`; *`parse_device`, *`requested_slot`, *`capacity_from_hello`, *`LearnerEntry`, *`Reconciled`, *`LearnerTable`, *`LEARNER_STATUSES`, *`TABLE_STATE_VERSION`, *`CPU` / `CUDA` / `MPS`;
  - *`VersionRegistry.park(info)`, *`unpark(agent_id)`, *`parked(agent_id)`; *`HubStateData.learners` (`LearnerTable.state(...)`);
  - *`BlobStore.parked_path(agent_id)`, *`BlobStore.start_path(agent_id, ident)`; `Download(kind="latest")` serves a parked frame; *`Download(kind="start", ident=...)`;
  - *`Event.generation` (int64, learner events); *the `LearnerAssignment.weights_blob` grammar (`""` | `init` | `start:resume` | `start:recovery` | `start:ckpt_v<N>`; `trainer_state_blob` stays `""`);
  - *`learner_process(..., send_final: Callable[[], bool] | None = None)`; *`utils.seeding.learner_seed(seed, agent_index, generation=1)`;
  - `train` records gain `learner_node`, `generation`; `system` value `pending_agents`; *hub counter `lineup_redraw_failed`; *`colosseum.hub.hub.LINEUP_REDRAWS = 16`.

- [ ] **Step 1: Write the failing library tests**

Create `tests/unit/test_sp5_placement.py`:
```python
"""Learner placement and the learner table (SP5 T4.1; spec blocks 1, 2.6, 7.1, 7.2): device kinds, GPU / CPU / MPS
slots, the local node's unlimited slots (R18), preferences and ``strict``, stability, config-order priority,
pending reasons; the table's statuses, relocation by the node's last heartbeat, reconciliation of a reconnecting
node, restarts and failures, and its hub_state form."""
from __future__ import annotations

import pytest

from colosseum.core.config import LearnerConfig
from colosseum.hub.placement import (
    LearnerSlot,
    LearnerTable,
    NodeCapacity,
    capacity_from_hello,
    parse_device,
    place,
    requested_slot,
)


def _cfg(device: str = "auto", *, node: str | None = None, tags: tuple[str, ...] = (), strict: bool = False):
    return LearnerConfig(device=device, placement={"node": node, "tags": list(tags), "strict": strict})


def _node(number: int, name: str | None = None, *, gpus=(), cpu: int = 0, tags=(), unlimited=False, mps=False):
    return NodeCapacity(number, name or f"n{number}", tuple(tags), tuple(gpus), cpu, unlimited=unlimited, mps=mps)


# ---- devices ------------------------------------------------------------------------------------------------


def test_parse_device_kinds():
    assert parse_device("cpu") == ("cpu", None)
    assert parse_device("cuda") == ("cuda", None)
    assert parse_device("cuda:3") == ("cuda", 3)
    assert parse_device("mps") == ("mps", None)
    with pytest.raises(ValueError, match="expected auto, cpu, cuda, cuda:N or mps"):
        parse_device("tpu")


def test_auto_takes_cuda_when_any_node_offers_cuda_else_cpu():
    assert requested_slot("auto", [_node(1, cpu=1), _node(2, gpus=(0,))]) == ("cuda", None)
    assert requested_slot("auto", [_node(1, cpu=1)]) == ("cpu", None)
    assert requested_slot("auto", [_node(1, gpus=(0,), mps=True)]) == ("cpu", None)   # MPS only by name


def test_auto_waits_for_a_busy_gpu_instead_of_taking_cpu():
    nodes = [_node(1, gpus=(0,), cpu=1)]
    plan = place({"a": _cfg("auto"), "b": _cfg("auto")}, nodes, current={})
    assert plan.assigned == {"a": LearnerSlot(1, "cuda:0")}
    assert plan.pending == ["b"]
    assert "no free CUDA GPU slot" in plan.reasons["b"]


# ---- slots --------------------------------------------------------------------------------------------------


def test_one_learner_per_gpu_and_cpu_slots_count():
    nodes = [_node(1, gpus=(0, 1), cpu=1)]
    agents = {"a": _cfg("cuda"), "b": _cfg("cuda"), "c": _cfg("cuda"), "d": _cfg("cpu"), "e": _cfg("cpu")}
    plan = place(agents, nodes, current={})
    assert plan.assigned == {"a": LearnerSlot(1, "cuda:0"), "b": LearnerSlot(1, "cuda:1"),
                             "d": LearnerSlot(1, "cpu")}
    assert plan.pending == ["c", "e"]
    assert "CPU learner slot" in plan.reasons["e"]


def test_cuda_index_is_honoured_across_nodes():
    nodes = [_node(1, gpus=(0,)), _node(2, gpus=(0, 1))]
    plan = place({"a": _cfg("cuda:1")}, nodes, current={})
    assert plan.assigned == {"a": LearnerSlot(2, "cuda:1")}
    plan = place({"a": _cfg("cuda:3")}, nodes, current={})
    assert plan.pending == ["a"] and "slot on GPU 3" in plan.reasons["a"]


def test_mps_slots_only_for_mps_and_never_for_cuda():
    nodes = [_node(1, gpus=(0,), mps=True)]
    assert place({"a": _cfg("mps")}, nodes, current={}).assigned == {"a": LearnerSlot(1, "mps")}
    plan = place({"a": _cfg("cuda")}, nodes, current={})
    assert plan.pending == ["a"]


def test_local_node_takes_every_learner_on_one_gpu_like_before():
    """R18: train's local node is unlimited; with one GPU every GPU learner shares cuda:0 (as before SP5)."""
    nodes = [_node(1, gpus=(0,), cpu=0, unlimited=True)]
    agents = {"a": _cfg("auto"), "b": _cfg("auto"), "c": _cfg("cpu")}
    plan = place(agents, nodes, current={})
    assert plan.assigned == {"a": LearnerSlot(1, "cuda:0"), "b": LearnerSlot(1, "cuda:0"),
                             "c": LearnerSlot(1, "cpu")}
    assert plan.pending == []


def test_local_node_without_gpu_runs_auto_on_cpu():
    plan = place({"a": _cfg("auto"), "b": _cfg("auto")}, [_node(1, unlimited=True)], current={})
    assert plan.assigned == {"a": LearnerSlot(1, "cpu"), "b": LearnerSlot(1, "cpu")}


def test_local_node_spreads_over_its_gpus_least_used_first():
    nodes = [_node(1, gpus=(0, 1), unlimited=True)]
    plan = place({"a": _cfg("cuda"), "b": _cfg("cuda"), "c": _cfg("cuda")}, nodes, current={})
    assert [plan.assigned[a].device for a in "abc"] == ["cuda:0", "cuda:1", "cuda:0"]


def test_no_learner_nodes_offer_nothing():
    nodes = [capacity_from_hello(1, "laptop", (), {"gpus": [0], "cpu_learners": 1, "no_learners": True},
                                 {0: "cuda"}, unlimited=False)]
    plan = place({"a": _cfg("cpu")}, nodes, current={})
    assert plan.pending == ["a"] and plan.reasons["a"] == "no online node offers learner slots"


def test_capacity_from_hello_reads_gpu_kinds():
    cap = capacity_from_hello(3, "mac", ("m1",), {"gpus": [0], "cpu_learners": 1}, {0: "mps"}, unlimited=False)
    assert cap == NodeCapacity(3, "mac", ("m1",), (0,), 1, unlimited=False, mps=True)


# ---- preferences --------------------------------------------------------------------------------------------


def test_preferred_node_first_then_any_other():
    nodes = [_node(1, "pc", cpu=1), _node(2, "laptop", cpu=1)]
    plan = place({"a": _cfg("cpu", node="laptop")}, nodes, current={})
    assert plan.assigned == {"a": LearnerSlot(2, "cpu")}
    plan = place({"b": _cfg("cpu"), "a": _cfg("cpu", node="pc")}, nodes, current={})
    # b (first in config order) takes pc's slot; a's preferred node is full, so it goes elsewhere
    assert plan.assigned == {"b": LearnerSlot(1, "cpu"), "a": LearnerSlot(2, "cpu")}


def test_strict_never_leaves_the_preferred_nodes():
    nodes = [_node(1, "pc", cpu=1), _node(2, "laptop", cpu=1)]
    plan = place({"b": _cfg("cpu", node="pc"), "a": _cfg("cpu", node="pc", strict=True)}, nodes, current={})
    assert plan.assigned == {"b": LearnerSlot(1, "cpu")}
    assert plan.pending == ["a"] and "strict" in plan.reasons["a"]
    plan = place({"a": _cfg("cpu", node="cloud", strict=True)}, nodes, current={})
    assert plan.reasons["a"].startswith("no online node matches learner.placement (node 'cloud', strict)")


def test_tags_must_all_match():
    nodes = [_node(1, cpu=1, tags=("fast",)), _node(2, cpu=1, tags=("fast", "big"))]
    plan = place({"a": _cfg("cpu", tags=("fast", "big"))}, nodes, current={})
    assert plan.assigned == {"a": LearnerSlot(2, "cpu")}


# ---- stability ----------------------------------------------------------------------------------------------


def test_current_slots_are_kept_and_count_against_capacity():
    nodes = [_node(1, gpus=(0,), cpu=1), _node(2, gpus=(0,))]
    current = {"b": LearnerSlot(1, "cuda:0")}
    plan = place({"a": _cfg("cuda"), "b": _cfg("cuda")}, nodes, current=current)
    assert plan.assigned == {"a": LearnerSlot(2, "cuda:0"), "b": LearnerSlot(1, "cuda:0")}
    assert plan.new(current) == {"a": LearnerSlot(2, "cuda:0")}


def test_a_learner_never_moves_to_its_preference_while_its_node_lives():
    nodes = [_node(1, "pc", cpu=1), _node(2, "laptop", cpu=1)]
    current = {"a": LearnerSlot(1, "cpu")}
    plan = place({"a": _cfg("cpu", node="laptop")}, nodes, current=current)
    assert plan.assigned == current and plan.new(current) == {}


def test_relocate_places_again_on_the_online_nodes():
    """Criterion 11: a cpu agent preferring the laptop moves to the PC's CPU slot when the laptop is gone."""
    nodes = [_node(1, "pc", gpus=(0,), cpu=1)]                     # the laptop (node 2) is not online
    current = {"main": LearnerSlot(1, "cuda:0"), "cpu_agent": LearnerSlot(2, "cpu")}
    agents = {"main": _cfg("auto"), "cpu_agent": _cfg("cpu", node="laptop")}
    plan = place(agents, nodes, current=current, relocate={"cpu_agent"})
    assert plan.assigned == {"main": LearnerSlot(1, "cuda:0"), "cpu_agent": LearnerSlot(1, "cpu")}


def test_placement_is_deterministic():
    nodes = [_node(3, cpu=1), _node(1, cpu=1), _node(2, gpus=(1, 0))]
    agents = {"a": _cfg("cpu"), "b": _cfg("cpu"), "c": _cfg("cuda")}
    first = place(agents, nodes, current={})
    assert first == place(agents, list(reversed(nodes)), current={})
    assert first.assigned == {"a": LearnerSlot(1, "cpu"), "b": LearnerSlot(3, "cpu"), "c": LearnerSlot(2, "cuda:0")}


# ---- the table ----------------------------------------------------------------------------------------------


def test_table_statuses_from_pending_to_running():
    table = LearnerTable(["a", "b"])
    assert table.to_place() == ["a", "b"] and table.collecting() == []
    table.assign("a", LearnerSlot(1, "cpu"), generation=1)
    assert table["a"].status == "starting" and table.can_collect("a")    # R11 holds its lineups for its weights
    assert table.announced("a", 2) is False                       # another generation: no change
    assert table.announced("a", 1) is True and table["a"].status == "running"
    assert table.collecting() == ["a"] and table.to_place() == ["b"] and not table.can_collect("b")
    with pytest.raises(ValueError, match="not above"):
        table.assign("a", LearnerSlot(1, "cpu"), generation=1)


def test_node_loss_relocates_after_the_wait_measured_from_the_last_heartbeat():
    table = LearnerTable(["a"])
    table.assign("a", LearnerSlot(2, "cpu"), generation=1)
    table.announced("a", 1)
    assert table.node_lost(2, last_heartbeat=100.0) == ["a"]
    assert table["a"].status == "relocating" and not table.can_collect("a")
    assert table.current() == {"a": LearnerSlot(2, "cpu")}        # still awaited: its slot is kept
    assert table.relocation_due(219.0, after_sec=120.0) == []
    assert table.relocation_due(220.0, after_sec=120.0) == ["a"]
    assert table["a"].status == "pending" and table["a"].slot is None


def test_reconcile_adopts_the_current_generation_and_stops_the_rest():
    table = LearnerTable(["a", "b", "c"])
    table.assign("a", LearnerSlot(2, "cpu"), generation=3)
    table.announced("a", 3)
    table.node_lost(2, last_heartbeat=10.0)
    table.assign("b", LearnerSlot(1, "cuda:0"), generation=5)       # b lives on node 1
    out = table.reconcile(2, {"a": 3, "b": 4, "zz": 1}, announced={"a": 3})
    assert out.adopted == ["a"] and table["a"].status == "running"
    assert sorted(out.stop) == [("b", 4), ("zz", 1)]
    assert out.gone == []


def test_reconcile_after_relocation_stops_the_old_learner():
    """Spec 7.2: a learner that kept running on a cut-off machine is told to stop when the machine returns."""
    table = LearnerTable(["a"])
    table.assign("a", LearnerSlot(2, "cpu"), generation=1)
    table.announced("a", 1)
    table.node_lost(2, last_heartbeat=0.0)
    table.relocation_due(200.0, after_sec=120.0)
    table.assign("a", LearnerSlot(1, "cpu"), generation=2)
    out = table.reconcile(2, {"a": 1}, announced={"a": 1})
    assert out.stop == [("a", 1)] and out.adopted == []
    assert table["a"].slot == LearnerSlot(1, "cpu") and table["a"].generation == 2


def test_reconcile_marks_learners_the_node_no_longer_runs_as_pending():
    table = LearnerTable(["a"])
    table.assign("a", LearnerSlot(2, "cpu"), generation=1)
    out = table.reconcile(2, {}, announced={})
    assert out.gone == ["a"] and table["a"].status == "pending" and table.to_place() == ["a"]


def test_reconcile_keeps_a_starting_learner_starting():
    table = LearnerTable(["a"])
    table.assign("a", LearnerSlot(2, "cpu"), generation=7)
    out = table.reconcile(2, {"a": 7}, announced={"a": 6})
    assert out.adopted == ["a"] and table["a"].status == "starting"


def test_died_restarts_only_the_current_generation_and_fail_is_final():
    table = LearnerTable(["a"])
    table.assign("a", LearnerSlot(1, "cuda:0"), generation=2)
    table.announced("a", 2)
    assert table.died("a", 1) is None                              # an old incarnation: ignored
    assert table.died("a", 2) == LearnerSlot(1, "cuda:0")
    table.assign("a", LearnerSlot(1, "cuda:0"), generation=3)
    table.fail("a", "3 restarts in 1800 s")
    assert table["a"].status == "failed" and table.failed() == ["a"] and table.to_place() == []
    assert table.current() == {}
    with pytest.raises(ValueError, match="failed"):
        table.assign("a", LearnerSlot(1, "cpu"), generation=4)
    table.set_pending("a", "x")
    assert table["a"].status == "failed"


def test_table_state_round_trip_on_resume_waits_for_remote_nodes_only():
    table = LearnerTable(["a", "b", "c", "e"])
    table.assign("a", LearnerSlot(2, "cpu"), generation=4)
    table.announced("a", 4)
    table.assign("b", LearnerSlot(1, "cuda:0"), generation=2)
    table.fail("b", "limit")
    table.assign("e", LearnerSlot(1, "cpu"), generation=7)          # on the local node (dies with the hub)
    state = table.state(local_nodes={1})
    assert state["agents"]["a"] == {"status": "running", "node": 2, "device": "cpu", "generation": 4, "reason": "",
                                    "local": False}
    assert state["agents"]["e"]["local"] is True
    resumed = LearnerTable(["a", "b", "c", "d", "e"])
    assert resumed.load_resumed(state, now=50.0) == ["a"]
    assert resumed["a"].status == "relocating" and resumed["a"].lost_since == 50.0
    assert resumed["a"].generation == 4 and resumed["b"].status == "pending" and resumed["b"].generation == 2
    assert resumed["e"].status == "pending" and resumed["e"].generation == 7
    assert resumed.to_place() == ["b", "c", "d", "e"]
    out = resumed.reconcile(2, {"a": 4}, announced={"a": 4})
    assert out.adopted == ["a"] and resumed["a"].status == "running"
```

Create `tests/unit/test_sp5_version_parking.py`:
```python
"""Hub-held ("parked") weights in the version registry (SP5 T4.1; spec 2.6 «соперником он играет последними
весами»): a parked version is the agent's latest until a learner version is accepted again or it is unparked; it
never lowers the version floor, its generation 0 is never stale, and it is not saved in hub_state.json (the hub
parks again at start)."""
from __future__ import annotations

import json

import pytest

from colosseum.hub.state import VersionInfo, VersionRegistry

PARKED = VersionInfo("a", 0, 3, 100, "p3", 0, relay_only=True)


def _accepted(version: int, generation: int) -> VersionInfo:
    return VersionInfo("a", generation, version, 100, f"d{version}", 2)


def test_a_parked_version_is_latest_until_a_learner_version_is_accepted():
    reg = VersionRegistry()
    g = reg.new_generation("a")
    assert reg.announce(_accepted(5, g)) == "accepted"
    reg.park(PARKED)
    assert reg.latest("a") == PARKED and reg.parked("a") == PARKED
    assert reg.version_floor("a") == 6                       # parking never lowers the floor
    assert reg.is_stale("a", 0, 3) is False                   # generation 0 (hub-held) is never stale
    g2 = reg.new_generation("a")
    assert reg.announce(_accepted(6, g2)) == "accepted"
    assert reg.parked("a") is None and reg.latest("a").version == 6


def test_unpark_restores_the_last_accepted_version():
    reg = VersionRegistry()
    reg.announce(_accepted(5, reg.new_generation("a")))
    reg.park(PARKED)
    assert reg.unpark("a") == PARKED
    assert reg.latest("a").version == 5 and reg.unpark("a") is None


def test_parked_weights_are_hub_held_and_not_saved():
    reg = VersionRegistry()
    with pytest.raises(ValueError, match="node 0"):
        reg.park(VersionInfo("a", 0, 3, 1, "x", 2))
    with pytest.raises(ValueError, match="generation 0"):
        reg.park(VersionInfo("a", 4, 3, 1, "x", 0))
    reg.park(PARKED)
    again = VersionRegistry.from_state(json.loads(json.dumps(reg.state())))
    assert again.parked("a") is None and again.latest("a") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_placement.py tests/unit/test_sp5_version_parking.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.hub.placement'`, and `AttributeError: 'VersionRegistry' object has no attribute 'park'`.

- [ ] **Step 3: Write `src/colosseum/hub/placement.py`**

```python
"""Learner placement and the hub's learner table (SP5 spec blocks 1, 2.6, 7.1, 7.2; task T4.1).

Pure: no processes, no network, no clock (callers pass ``now`` from ``time.monotonic()`` of the hub). The hub's
main loop owns one ``LearnerTable`` and calls ``place`` whenever the set of nodes or the set of agents that need a
learner changes.

**Slots** (spec 2.6). A node offers one GPU learner slot per GPU index in ``NodeCapacity.gpus`` (CUDA, or MPS when
``mps``) and ``cpu_learners`` CPU learner slots; a node started with ``--no-learners`` offers none. ``unlimited``
(train's local node, ruling R18) takes any number of learners on each of its GPUs and on the CPU; on such a node a
GPU learner goes to its least used GPU (ties: the lower index), so one GPU takes every learner, as before SP5.

**Devices** (``learner.device``):
- ``cpu`` -> a CPU slot;
- ``cuda`` -> any CUDA GPU slot; ``cuda:N`` -> the GPU slot with index N (on any node that has that index);
- ``mps`` -> an MPS slot;
- ``auto`` -> a CUDA GPU slot when any online node offers CUDA GPU slots (free or taken), else a CPU slot, like
  ``learner.resolve_device("auto")`` on one machine; MPS only when asked for by name.

**Preference** (``learner.placement``): ``node`` (a node name) and ``tags`` (every one of them) mark the preferred
nodes. Without ``strict`` another node with a matching free slot is used when no preferred node has one; with
``strict`` only preferred nodes are used. Within a tier: ascending node number, then ascending GPU index.

**Stability:** a learner never moves while its node lives: ``current`` slots are kept as they are (also while their
node is lost but not yet past ``learner_relocate_after_sec``); only the agents in ``relocate`` are placed again.
Agents are placed in the order of ``agents`` (config order), so the earlier agent wins a contested slot.

**The table** (``LearnerTable``) holds per trainable agent the status of its learner:
``pending`` (no slot; its seats do not collect) -> ``starting`` (``start_learner`` sent with a new generation; its
seats already collect and it owns cells: a lineup reaches a worker only once the learner's first weights are in
the machine's cache, R11) -> ``running`` (the hub accepted a version announcement of that generation) and
``relocating`` (its node is lost; it waits ``learner_relocate_after_sec`` from the node's last heartbeat for the
node to come back, then becomes ``pending``) or ``failed`` (the node's restart limit is spent; never placed again in
this hub incarnation). Every assignment has a new generation (the caller takes it from
``VersionRegistry.new_generation``); ``reconcile`` compares a reconnecting node's running learners with the table
(adopt the current generation, stop anything else).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

CPU, CUDA, MPS = "cpu", "cuda", "mps"
_CUDA_INDEX_RE = re.compile(r"cuda:(\d+)")
LEARNER_STATUSES = ("pending", "starting", "running", "relocating", "failed")
LearnerStatus = Literal["pending", "starting", "running", "relocating", "failed"]
TABLE_STATE_VERSION = 1


@dataclass(frozen=True)
class LearnerSlot:
    node: int
    device: str                   # "cpu" | "cuda:N" | "mps"


@dataclass(frozen=True)
class NodeCapacity:
    node: int
    name: str
    tags: tuple[str, ...]
    gpus: tuple[int, ...]         # GPU indices offered for learners (Slots.gpus; [] with --no-learners)
    cpu_learners: int             # CPU learner slots (Slots.cpu_learners; 0 with --no-learners)
    unlimited: bool = False       # train's local node (R18): any number of learners per GPU and on the CPU
    mps: bool = False             # the GPU slots are MPS (macOS), not CUDA


@dataclass(frozen=True)
class PlacementPlan:
    assigned: dict[str, LearnerSlot]      # every agent with a slot: the kept ones and the newly placed ones
    pending: list[str]                    # agents without a slot, in the order of ``agents``
    reasons: dict[str, str]               # pending agent -> why (one line, for logs and ``nodes list``)

    def new(self, current: Mapping[str, LearnerSlot]) -> dict[str, LearnerSlot]:
        """The slots that differ from ``current`` (agents to start)."""
        return {a: s for a, s in self.assigned.items() if current.get(a) != s}


def parse_device(device: str) -> tuple[str, int | None]:
    """``learner.device`` -> (kind, GPU index or None) for every value but ``auto``; ValueError otherwise."""
    if device in (CPU, CUDA, MPS):
        return device, None
    match = _CUDA_INDEX_RE.fullmatch(device)
    if match:
        return CUDA, int(match.group(1))
    raise ValueError(f"learner.device {device!r}: expected auto, cpu, cuda, cuda:N or mps")


def requested_slot(device: str, nodes: Sequence[NodeCapacity]) -> tuple[str, int | None]:
    """The slot kind ``device`` needs on this cluster (``auto`` resolved over ``nodes``, module docstring)."""
    if device == "auto":
        return (CUDA, None) if any(n.gpus and not n.mps for n in nodes) else (CPU, None)
    return parse_device(device)


def _preferred(node: NodeCapacity, placement: Any) -> bool:
    if placement is None:
        return True
    if placement.node is not None and node.name != placement.node:
        return False
    return all(tag in node.tags for tag in placement.tags)


def _has_preference(placement: Any) -> bool:
    return placement is not None and (placement.node is not None or bool(placement.tags))


class _Usage:
    """Slots taken per node (kept and newly placed learners)."""

    def __init__(self, nodes: Sequence[NodeCapacity]) -> None:
        self.nodes = {n.node: n for n in nodes}
        self.gpu: dict[int, dict[int, int]] = {n.node: {i: 0 for i in n.gpus} for n in nodes}
        self.cpu: dict[int, int] = {n.node: 0 for n in nodes}

    def take(self, slot: LearnerSlot) -> None:
        if slot.node not in self.nodes:          # a kept slot on a lost node holds nothing
            return
        if slot.device == CPU:
            self.cpu[slot.node] += 1
            return
        index = 0 if slot.device == MPS else parse_device(slot.device)[1]
        counts = self.gpu[slot.node]
        if index in counts:
            counts[index] += 1

    def free_slot(self, node: NodeCapacity, kind: str, index: int | None) -> LearnerSlot | None:
        if kind == CPU:
            if node.unlimited or self.cpu[node.node] < node.cpu_learners:
                return LearnerSlot(node.node, CPU)
            return None
        if (kind == MPS) != node.mps:
            return None
        counts = self.gpu[node.node]
        candidates = [i for i in sorted(counts) if index is None or i == index]
        if not node.unlimited:
            candidates = [i for i in candidates if counts[i] == 0]
        if not candidates:
            return None
        best = min(candidates, key=lambda i: (counts[i], i))
        return LearnerSlot(node.node, MPS if node.mps else f"{CUDA}:{best}")


def _what(kind: str, index: int | None) -> str:
    if kind == CPU:
        return "CPU learner slot"
    if kind == MPS:
        return "MPS slot"
    return "CUDA GPU slot" if index is None else f"slot on GPU {index}"


def place(agents: Mapping[str, Any], nodes: Sequence[NodeCapacity], *, current: Mapping[str, LearnerSlot],
          relocate: Collection[str] = ()) -> PlacementPlan:
    """Slots for every agent of ``agents`` (agent -> its ``LearnerConfig``, in config order) on the online
    ``nodes`` (module docstring). ``current`` slots are kept unless the agent is in ``relocate``; a kept slot
    counts against its node's capacity. Agents of ``agents`` without a kept slot are placed in order; the rest are
    pending with a reason. Deterministic for equal inputs."""
    ordered = sorted(nodes, key=lambda n: n.node)
    usage = _Usage(ordered)
    assigned: dict[str, LearnerSlot] = {}
    for agent in agents:
        slot = current.get(agent)
        if slot is not None and agent not in relocate:
            assigned[agent] = slot
            usage.take(slot)
    pending: list[str] = []
    reasons: dict[str, str] = {}
    offering = [n for n in ordered if n.unlimited or n.gpus or n.cpu_learners > 0]
    for agent, cfg in agents.items():
        if agent in assigned:
            continue
        placement = getattr(cfg, "placement", None)
        if not offering:
            pending.append(agent)
            reasons[agent] = "no online node offers learner slots"
            continue
        kind, index = requested_slot(cfg.device, ordered)
        preferred = [n for n in offering if _preferred(n, placement)]
        strict = bool(getattr(placement, "strict", False)) and _has_preference(placement)
        tiers = [preferred] if strict else [preferred, [n for n in offering if n not in preferred]]
        slot = None
        for tier in tiers:
            for node in tier:
                slot = usage.free_slot(node, kind, index)
                if slot is not None:
                    break
            if slot is not None:
                break
        if slot is None:
            pending.append(agent)
            if strict and not preferred:
                reasons[agent] = f"no online node matches learner.placement ({_describe(placement)}, strict)"
            else:
                where = f" on the nodes matching learner.placement ({_describe(placement)}, strict)" if strict else ""
                reasons[agent] = f"no free {_what(kind, index)}{where} (learner.device {cfg.device})"
            continue
        assigned[agent] = slot
        usage.take(slot)
    return PlacementPlan(assigned=assigned, pending=pending, reasons=reasons)


def _describe(placement: Any) -> str:
    parts = []
    if placement.node is not None:
        parts.append(f"node {placement.node!r}")
    if placement.tags:
        parts.append(f"tags {list(placement.tags)}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------------------------------------------
# The learner table
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class LearnerEntry:
    agent_id: str
    status: LearnerStatus = "pending"
    slot: LearnerSlot | None = None
    generation: int = 0                 # the current (or last) assignment's generation; 0 = never assigned
    lost_since: float | None = None     # hub monotonic: the last heartbeat of the slot's node (relocating only)
    reason: str = ""                    # why pending / relocating / failed
    assignments: int = 0                # assignments in this hub incarnation (learner_restarts = sum - 1 each)


@dataclass(frozen=True)
class Reconciled:
    """``LearnerTable.reconcile``: the node's learners the hub keeps, the ones it must stop, and the agents whose
    learner this node should run but does not (they became pending)."""

    adopted: list[str] = field(default_factory=list)
    stop: list[tuple[str, int]] = field(default_factory=list)          # (agent, generation)
    gone: list[str] = field(default_factory=list)


class LearnerTable:
    """Per trainable agent: the status of its learner (module docstring). Main loop only."""

    def __init__(self, agents: Sequence[str]) -> None:
        self._entries: dict[str, LearnerEntry] = {a: LearnerEntry(a) for a in agents}

    def __contains__(self, agent: str) -> bool:
        return agent in self._entries

    def __getitem__(self, agent: str) -> LearnerEntry:
        return self._entries[agent]

    def entries(self) -> list[LearnerEntry]:
        return list(self._entries.values())

    def to_place(self) -> list[str]:
        """Agents waiting for a slot, in config order."""
        return [a for a, e in self._entries.items() if e.status == "pending"]

    def current(self) -> dict[str, LearnerSlot]:
        """Slots of every learner that exists or is awaited (starting, running, relocating)."""
        return {a: e.slot for a, e in self._entries.items()
                if e.status in ("starting", "running", "relocating") and e.slot is not None}

    def on_node(self, node: int) -> list[str]:
        return [a for a, e in self._entries.items()
                if e.slot is not None and e.slot.node == node and e.status in ("starting", "running", "relocating")]

    def can_collect(self, agent: str) -> bool:
        """The agent's seats collect and it owns cells while its learner starts or runs (spec 4.7)."""
        entry = self._entries.get(agent)
        return entry is not None and entry.status in ("starting", "running")

    def collecting(self) -> list[str]:
        """Agents that can collect, in config order (the owner rotation's agents)."""
        return [a for a, e in self._entries.items() if e.status in ("starting", "running")]

    def assign(self, agent: str, slot: LearnerSlot, generation: int) -> None:
        """A ``start_learner`` with ``generation`` goes to ``slot``'s node: ``starting``."""
        entry = self._entries[agent]
        if entry.status == "failed":
            raise ValueError(f"learner of {agent!r} failed; it is not placed again")
        if generation <= entry.generation:
            raise ValueError(f"generation {generation} of {agent!r} is not above {entry.generation}")
        entry.status, entry.slot, entry.generation = "starting", slot, int(generation)
        entry.lost_since, entry.reason = None, ""
        entry.assignments += 1

    def announced(self, agent: str, generation: int) -> bool:
        """The hub accepted a version of ``generation``: a ``starting`` learner of that generation is ``running``.
        True when the status changed."""
        entry = self._entries.get(agent)
        if entry is None or entry.generation != generation or entry.status != "starting":
            return False
        entry.status = "running"
        return True

    def node_lost(self, node: int, last_heartbeat: float) -> list[str]:
        """``node`` is lost: its learners wait for it (``relocating``); returns them."""
        moved = []
        for agent in self.on_node(node):
            entry = self._entries[agent]
            if entry.status != "relocating":
                entry.status, entry.lost_since = "relocating", float(last_heartbeat)
                entry.reason = f"node {node} lost"
                moved.append(agent)
        return moved

    def relocation_due(self, now: float, after_sec: float) -> list[str]:
        """``relocating`` learners whose node stayed away ``after_sec`` since its last heartbeat become ``pending``
        (the caller places them again with a new generation); returns them."""
        due = []
        for agent, entry in self._entries.items():
            if entry.status == "relocating" and entry.lost_since is not None and now - entry.lost_since >= after_sec:
                entry.status, entry.slot, entry.lost_since = "pending", None, None
                entry.reason = f"relocating: its node stayed away {after_sec:g} s"
                due.append(agent)
        return due

    def reconcile(self, node: int, running: Mapping[str, int], *, announced: Mapping[str, int]) -> Reconciled:
        """A node (re)connected with ``running`` = {agent: generation} of its live learners (``Hello.learners``).
        ``announced`` = {agent: generation of the newest accepted version} (``VersionRegistry``).

        - a learner of the agent's current generation on its slot's node is adopted: ``running`` when a version of
          that generation was accepted, else ``starting`` (a ``relocating`` one is back);
        - any other learner (a stale generation, an agent placed elsewhere, failed or unknown) is stopped;
        - an agent whose learner this node should run (starting / running / relocating here) but does not becomes
          ``pending`` (``gone``): the caller places it again with a new generation."""
        out = Reconciled()
        for agent, generation in running.items():
            entry = self._entries.get(agent)
            ours = (entry is not None and entry.slot is not None and entry.slot.node == node
                    and entry.generation == generation and entry.status in ("starting", "running", "relocating"))
            if not ours:
                out.stop.append((agent, int(generation)))
                continue
            entry.status = "running" if announced.get(agent) == generation else "starting"
            entry.lost_since, entry.reason = None, ""
            out.adopted.append(agent)
        for agent in self.on_node(node):
            if agent not in out.adopted:
                entry = self._entries[agent]
                entry.status, entry.slot, entry.lost_since = "pending", None, None
                entry.reason = f"node {node} came back without it"
                out.gone.append(agent)
        return out

    def died(self, agent: str, generation: int) -> LearnerSlot | None:
        """The learner of ``generation`` died and its node may restart it: the slot to restart on (the caller
        assigns a new generation there), or None when the report is about an old generation (ignored)."""
        entry = self._entries.get(agent)
        if entry is None or entry.generation != generation or entry.status not in ("starting", "running"):
            return None
        return entry.slot

    def fail(self, agent: str, reason: str) -> None:
        """The restart limit is spent (spec 7.1): ``failed``, slot released, never placed again."""
        entry = self._entries[agent]
        entry.status, entry.slot, entry.lost_since, entry.reason = "failed", None, None, reason

    def failed(self) -> list[str]:
        return [a for a, e in self._entries.items() if e.status == "failed"]

    def set_pending(self, agent: str, reason: str) -> None:
        entry = self._entries[agent]
        if entry.status != "failed":
            entry.status, entry.slot, entry.lost_since, entry.reason = "pending", None, None, reason

    def state(self, local_nodes: Collection[int] = ()) -> dict:
        """``hub_state.json`` ``learners`` (JSON-ready); ``local_nodes``: numbers of nodes that die with the hub
        (train's local node), so a resume does not wait for them."""
        return {"version": TABLE_STATE_VERSION, "agents": {
            a: {"status": e.status, "node": None if e.slot is None else e.slot.node,
                "device": None if e.slot is None else e.slot.device, "generation": e.generation, "reason": e.reason,
                "local": e.slot is not None and e.slot.node in local_nodes}
            for a, e in self._entries.items()}}

    def load_resumed(self, state: Mapping[str, Any] | None, *, now: float) -> list[str]:
        """In-place resume (T4.3): learners the previous hub incarnation had (starting, running or relocating) on a
        remote node wait for it as ``relocating`` from ``now`` (the node reconnects and is reconciled, or they
        relocate after ``learner_relocate_after_sec``); learners of a local node (it died with the hub) and
        ``failed`` agents are ``pending`` at once (failed ones are retried); generations continue; agents not in
        this table are dropped. Returns the awaited agents."""
        awaited = []
        for agent, saved in (state or {}).get("agents", {}).items():
            entry = self._entries.get(agent)
            if entry is None:
                continue
            entry.generation = max(entry.generation, int(saved.get("generation") or 0))
            if (saved.get("status") in ("starting", "running", "relocating") and saved.get("node") is not None
                    and not saved.get("local")):
                entry.status = "relocating"
                entry.slot = LearnerSlot(int(saved["node"]), str(saved["device"]))
                entry.lost_since = float(now)
                entry.reason = "waiting for its node after the hub restart"
                awaited.append(agent)
        return awaited


def capacity_from_hello(node: int, name: str, tags: Sequence[str], slots: Mapping[str, Any],
                        gpu_kinds: Mapping[int, str], *, unlimited: bool) -> NodeCapacity:
    """A ``NodeCapacity`` from a node's ``Hello.slots`` (as a dict: ``gpus``, ``cpu_learners``, ``no_learners``) and
    its GPU kinds by index (``Hello.hardware.gpus``)."""
    if slots.get("no_learners"):
        return NodeCapacity(node, name, tuple(tags), (), 0, unlimited=False)
    gpus = tuple(sorted(int(i) for i in slots.get("gpus", ())))
    mps = bool(gpus) and all(gpu_kinds.get(i) == MPS for i in gpus)
    return NodeCapacity(node, name, tuple(tags), gpus, int(slots.get("cpu_learners", 0)), unlimited=unlimited,
                        mps=mps)
```

- [ ] **Step 4: Parked weights in `VersionRegistry` (`src/colosseum/hub/state.py`) and `HubStateData.learners`**

Add to `VersionRegistry` (T2.1's class; keep everything else):
```python
    # __init__ and from_state: self._parked: dict[str, VersionInfo] = {}   (never part of state())

    def park(self, info: VersionInfo) -> None:
        """Serve ``info`` as ``info.agent_id``'s latest until a learner version of the agent is accepted again
        (``announce``) or ``unpark``: hub-held weights of an agent without a running learner (``node == 0``,
        ``generation == 0``; SP5 T4.1). ``max_version``, the floor and the caps are unchanged; not saved."""
        if info.node != 0:
            raise ValueError(f"parked weights of {info.agent_id!r} are hub-held: node 0, not {info.node}")
        if info.generation != 0:
            raise ValueError(f"parked weights of {info.agent_id!r} carry generation 0, not {info.generation}")
        self._parked[info.agent_id] = info

    def unpark(self, agent_id: str) -> VersionInfo | None:
        """Stop serving the parked weights (the agent's learner was adopted again); returns them."""
        return self._parked.pop(agent_id, None)

    def parked(self, agent_id: str) -> VersionInfo | None:
        return self._parked.get(agent_id)
```
and change two existing methods: `latest(agent_id)` returns `self._parked.get(agent_id)` when present, else the accepted latest as before; `announce(info)` pops `self._parked[info.agent_id]` when it returns `"accepted"`. `is_stale(agent, generation, version)` must return False for generation 0 or None (T2.1's "unknown generation" rule; assert it in the test above, fix T2.1's code if it differs).

`HubStateData` gains `learners: dict = field(default_factory=dict)` (`LearnerTable.state(...)`), saved and loaded like the other fields (`load_hub_state` of a file without it gives `{}`); `HUB_STATE_VERSION` stays 1 (an added key with a default).

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_placement.py tests/unit/test_sp5_version_parking.py tests/unit/test_sp5_hub_state.py -q` (the last is T2.1's file; its basename may differ — run T2.1's state tests). Expected: all pass.

- [ ] **Step 5: The hub side (`src/colosseum/hub/hub.py`, `src/colosseum/hub/server.py`, the proto)**

Behaviour, by edit point (the handler names are T2.5's, R5):

1. **Proto.** In `colosseum_v1.proto` add `int64 generation = <next free number>;` to `message Event` (the learner generation of a `proc_died` / `learner_failed` / `child_error` event of a learner; 0 otherwise). Regenerate with `scripts/gen_proto.sh` and commit the generated files.

2. **State.** In `Hub.launch`, after the coordinator: `self._learners = LearnerTable(cfg.get_trainable_agent_ids())`; `self._parked: dict[str, int] = {}` (agent → parked version); `self._unplayable: set[str]`; `self._pending_reasons: dict[str, str]`; `self._start_sources: dict[str, tuple[str, int, str]]` (agent → (weights_blob, version, description)), filled by `_resume_start_sources()` (below; T4.3 replaces its body for an in-place resume).

3. **`_node_capacities()`**: one `capacity_from_hello(entry.number, entry.name, entry.tags, entry.slots, {g["index"]: g["kind"] for g in entry.hardware.get("gpus", ())}, unlimited=entry.data_path == "local")` per `NodeRegistry.online()` entry (R18: the local node of `train` is unlimited). `entry.slots` / `entry.hardware` are the `Hello` fields as dicts (T2.1); if T2.1 stores the protobuf, convert with `MessageToDict(..., preserving_proto_field_name=True)`.

4. **`_place_learners()`** (main loop only; cheap, idempotent): `agents = {a: cfg.get_agent_config(a).learner for a in trainable if self._learners[a].status != "failed"}`; `current = self._learners.current()`; `plan = place(agents, self._node_capacities(), current=current)`; for each `(agent, slot)` in `plan.new(current)`: `self._start_learner(agent, slot)`; for each pending agent: `self._learners.set_pending(agent, plan.reasons[agent])`, an INFO `Learner of <agent>: pending (<reason>)` when the reason changed since the last pass, and `self._park(agent)`; then `self._hub_metrics.set_system_value("pending_agents", [a for a in trainable if status in ("pending", "relocating")])`. Called from `_on_session_opened` (after the reconcile), from `_on_session_closed` only when the registry marks the node lost (T4.2 adds the loss handling), from the metrics tick, and by T4.2 after deaths and relocations.
   **`train` without `--listen`** (only the local node can ever exist): an agent still `pending` after the first placement pass with the local node online stops the run with exit 1 and `ERROR Learner of <agent> cannot run on this machine: <reason>` (e.g. `learner.device: cuda` without CUDA, `cuda:1` on a one-GPU machine); with `hub` or `train --listen` it waits for a node.

5. **`_start_learner(agent, slot)`**: `generation = self._versions.new_generation(agent)`; `self._learners.assign(agent, slot, generation)`; `blob, version, what = self._learner_start_source(agent)`; send `Command(kind="start_learner", learner=LearnerAssignment(agent_id=agent, generation=generation, version_floor=self._version_floor(agent), device=slot.device, weights_blob=blob, trainer_state_blob="", env_steps=self._budget.value))` on the session of node `slot.node`; INFO `Learner of <agent>: generation <g> on node <n> (<name>), device <device>, from <what>, versions from <floor>`; `self._announce_agents()` (an `Announce` whose `agents` carry `AgentState(agent, "trainable", status, start_env_steps=0)` with status `active` for starting / running, `pending` for pending / relocating, `failed`).
   **`_learner_start_source(agent)`**: in this task the hub-start source only: `self._start_sources[agent]`, computed once in `launch` by `_resume_start_sources()`: an agent with a `training.resume_from` state (`self._resume_states[agent]`, T2.5's `_resolve_resume`) → `("start:resume", state["policy_version"], f"{state['source']} (policy_version N)")`, writing `BlobStore.start_path(agent, "resume")` once with `encode_checkpoint({"agent_id": agent, "policy_version": v, "model_state": state["model_state"], "trainer_state_bytes": state["trainer_state"] or b"", "final": False}, kind="recovery", generation=0)` (an empty trainer state = weights only, as SP3's `.pt` resume); else an agent with `init` (`self._init_states[agent]`, already a `Welcome.fixed` blob of kind `init`) → `("init", 0, "init.from <path>")`; else `("", 0, "factory weights")`. T4.2 puts the recovery point in front, T4.3 changes the hub-start source of an in-place resume. If T2.5 shipped the resume state to the local node another way, replace it with this scheme (one start path for local and remote nodes).
   **`_version_floor(agent)`** = `max(self._versions.version_floor(agent), latest_snapshot_version(agent) + 1)` with `latest_snapshot_version` from `CheckpointManager.latest(agent)` (−1 without snapshots); T4.2 adds the recovery point.

6. **`_stop_learner(node, agent, generation, reason)`**: `Command(kind="stop_learner", learner=LearnerAssignment(agent_id=agent, generation=generation), reason=reason)` on that node's session, at most once per `(node, agent, generation)`; reasons `superseded` (the node must not upload a final snapshot) and, in T4.4, `stop_run`.

7. **`_on_session_opened`** (before the `Welcome` is built): `out = self._learners.reconcile(node, {l.agent_id: l.generation for l in hello.learners}, announced={a: info.generation for a in trainable if (info := self._versions.latest(a)) is not None and info.node != 0})`; for every adopted agent `self._versions.unpark(agent)` + `self._drop_parked(agent)`; `Welcome.learners` = a `LearnerAssignment` (generation, device, the floor and blob it was started with) for every agent whose table slot is on this node and whose status is starting or running — **the node's desired learner set**; after the `Welcome` send `_stop_learner(node, agent, generation, "superseded")` for every `out.stop`; then `_place_learners()` (the `out.gone` agents are pending again). INFO per adopted / stopped learner.

8. **`_on_version`** (the `VersionAnnounce` handler of T2.5 / T3.1): after `result = self._versions.announce(info)`: `"accepted"` → `self._learners.announced(agent, info.generation)` (INFO `Learner of <agent>: running on node <n> (generation <g>)` on the first one) and `self._drop_parked(agent)`; `"stale_generation"` → count it (`dropped_announcements.stale_generation`, kept in `hub_state.json` counters), WARNING once per (node, reason) and `_stop_learner(node, agent, info.generation, "superseded")`. The rebroadcast of accepted versions stays T3.1's.

9. **`_on_upload`** (kinds `checkpoint` / `final`): before decoding, refuse with `UploadReply(accepted=False, reason="stale_generation")` (counted `dropped_uploads.stale_generation`, WARNING once per node) when `header.generation != self._versions.generation(agent)` or the uploading node is not the agent's slot node; nothing is saved and the spooled file is deleted.

10. **Results** need no new code: `ResultIntake.check` (T2.1) applies the cap rule with `self._versions`; this task's tests pin it end to end.

11. **Collecting and owners.** Every place where T2.5 / T3.2 decide "this agent has a running learner" (the owner rotation's agent list, the `collect` flag of seats) uses `self._learners.can_collect(agent)` (starting or running: a lineup with a starting learner's seats waits on the node until its first weights are cached, R11).

12. **Playability and lineups.** `_playable(agent)` = a fixed agent, or a trainable agent that can collect or has parked weights. `_lineup_for_cell(owner)` (where T2.5 calls `Coordinator.lineup_for`): up to `LINEUP_REDRAWS = 16` draws; a draw with a `latest` seat of an unplayable trainable agent is rejected and drawn again (rejection sampling: the result is the matchmaker's distribution conditioned on playable agents); after 16 rejections the cell gets no grant in this pass (it keeps its lineup and asks again at its next request), `lineup_redraw_failed` += 1 (a hub counter in `hub_state.json`, shown by T7.1) and one WARNING per owner: `matchmaking for <owner>: <n> draws in a row named agents without weights (<agents>); no lineup this pass — give them init weights or a learner slot`. Seats of agents that cannot collect get `collect=False` (T3.2's rule).

13. **Parking.** `_park(agent)` (no-op when already parked or the agent can collect): the source is the newest by version of (T4.2: the recovery point), the newest stored snapshot (`CheckpointManager.latest(agent)` + `load_model`), the `init` weights (version 0), the `training.resume_from` state; none → `self._unplayable.add(agent)` and one INFO `<agent> has no weights yet: it is not drawn into lineups until its learner runs`. Otherwise write `BlobStore.parked_path(agent)` atomically (`write_bytes` to a temp name + `os.replace`) with `encode_weights(state_dict, WeightsMeta(agent, "latest", generation=0, policy_version=N))`, `self._versions.park(VersionInfo(agent, 0, N, size, blake2b16(frame), node=0, relay_only=True))` and broadcast it in an `Announce` (T3.1's broadcast). `_drop_parked(agent)`: delete the file (POSIX: a servicer that has it open keeps reading) and forget it. Parking happens when an agent is left pending by a placement pass, when it becomes relocating or failed (T4.2), never for starting / running agents.

14. **`Download` servicer** (`server.py`): kind `latest` — when `BlobStore.parked_path(agent)` exists, peek its header (`peek_meta`): its `policy_version <= request.since_version` → one `BlobPart(not_modified=True)`, else the file in parts with `meta_json` of its `WeightsMeta`; when it does not exist, T3.1's relayed-latest path. Kind `start` — ident `resume` / `ckpt_v<N>` → `BlobStore.start_path(agent, ident)` (`runs/<name>/blobs/start-<agent>-<ident>.frame`), T4.2 adds ident `recovery`; a missing file → `NOT_FOUND`. Servicer threads only read files.

15. **Metrics.** `_on_metrics_batch`: every `TrainMetrics` of the agent's current generation becomes the `train` record dict with `learner_node` (the slot's node number) and `generation` added; a `TrainMetrics` of another generation is dropped (DEBUG; a stale learner's numbers would mix into the curves). `hub_state.json` (each metrics tick and at the end) gets `learners = self._learners.state(local_nodes={e.number for e in registry entries with data_path "local"})`.

16. **`hub --idle`** (T5.2) has no trainable agents: the table is empty and nothing above runs.

- [ ] **Step 6: Hub integration tests (`tests/integration/test_sp5_learner_placement.py`)**

Setup shared by the tests (module docstring: "Learner placement in the hub (SP5 T4.1) with fake nodes: devices and slots, pending agents and parked weights, reconciliation, generation fencing, version floors, metrics stamps."): an in-process hub (`cluster_kit.hub_in_thread`, mode `hub`, loopback, `--insecure` off: the test identity under `tmp_path`) on a toy config from `game_helpers.make_test_config("turns", ...)` with trainable agents `a`, `b` (and per test a scripted `rnd`), `distributed.heartbeat_sec: 0.1`, `node_lost_after_sec: 0.5`, `metrics.log_interval` small enough for a tick per 0.2 s; `hub_kit.FakeNode`s register, open sessions with the `Hello` fields each test names, and heartbeat until told otherwise. Reads: `received` commands of each FakeNode, `cluster_kit.read_hub_state(run_dir)["learners"]`, `metrics.jsonl`, `logs/main.log`. Tests:

1. `test_devices_and_slots_decide_where_learners_start` — `a.learner.device: cuda`, `b: cpu`; FakeNode `n1` slots `{gpus: [0], cpu_learners: 0}` with hardware GPU 0 kind `cuda`, `n2` `{gpus: [], cpu_learners: 1}`: `n1` receives exactly one `start_learner` (`a`, generation 1, device `cuda:0`, version_floor 0, weights_blob `""`, env_steps 0), `n2` one for `b` (device `cpu`); `hub_state` learners: both `starting` on those nodes.
2. `test_an_agent_without_a_slot_is_pending_then_starts_when_a_slot_appears` — both agents `cpu`, `n1` with `cpu_learners: 1`: `a` starts, `b` is pending: `system.pending_agents == ["b"]` in the next `system` record, INFO `Learner of b: pending (no free CPU learner slot (learner.device cpu))`; FakeNode `n2` with `cpu_learners: 1` connects → it receives `start_learner(b, generation 1)`; the next record has `pending_agents == []`.
3. `test_a_pending_agent_plays_with_parked_weights` — `b` has `init.from` = a `.pt` the test writes (`torch.save` of `build_model(...)` weights, as `test_sp3_init.py` does) and no slot: an `Announce` carries `VersionInfo(agent_id="b", generation=0, version=0, node=0, relay_only=True)`; `FakeNode.download(BlobRequest(kind="latest", agent_id="b", since_version=-1))` returns a weights frame whose `decode_weights` meta is kind `latest`, policy_version 0 and whose arrays equal the `.pt`; `since_version=0` → a single `not_modified` part; the `LineupGrants` `n1` receives (cells requested with `LineupRequest`) contain seats of `b` with `collect == False`, and none owned by `b`.
4. `test_a_pending_agent_without_weights_is_not_drawn` — the same without `init`: over 40 grants no seat names `b`, `lineup_redraw_failed == 0` while `a`'s mix has other categories (`opponents: {latest: 0.5, rivals: 0.5}`); with `opponents: {rivals: 1.0}` for `a` the grants stop, `hub_state` counter `lineup_redraw_failed > 0` and one WARNING `draws in a row named agents without weights (b)`.
5. `test_the_first_accepted_announcement_makes_a_learner_running` — `n1` sends `VersionAnnounce(a, generation 1, version 0)` → `hub_state` learners `a` `running`; INFO `Learner of a: running on node <n1>`; a second session of `n1` gets a `Welcome` whose `learners` hold `a` with generation 1.
6. `test_a_reconnecting_node_keeps_its_current_learner` — `a` running on `n1`; `n1.close_session()` and a new session within `node_lost_after_sec` with `Hello.learners = [RunningLearner(a, 1, 3)]`: no `stop_learner`, no new `start_learner`, generation still 1, `Welcome.learners` lists `a`.
7. `test_a_late_announcement_of_an_old_generation_is_fenced` — `a` running on `n1` (generation 1, versions 0..5 accepted); `n1` reconnects with `Hello.learners = []` (its learner process restarted) → `a` pending → placed again on `n1` with generation 2 and `version_floor == 6`; then `n1` sends `VersionAnnounce(a, generation 1, version 9)` → not in any `Announce`, `hub_state` counter `dropped_announcements.stale_generation == 1`, `n1` receives `stop_learner(a, generation 1, reason "superseded")`; a `Hello` naming `RunningLearner(a, 7, 0)` (an unknown generation) on another reconnect → `stop_learner(a, 7)`.
8. `test_stale_generation_checkpoints_and_results_are_refused` — continuing 7: `n1.upload(UploadHeader(kind="checkpoint", agent_id="a", generation=1, version=10, ...))` → `UploadReply(accepted=False, reason="stale_generation")`, no `checkpoints/a/ckpt_v10/`; a `ResultBatch` with one result whose seat of `a` has `generation=1, version_end=9` is dropped (`dropped_match_results_by_reason.stale_generation == 1` in `hub_state` counters, not in the journal) while one with `generation=1, version_end=5` (≤ cap(1) = 5) is journaled; an upload from `n2` (not `a`'s slot node) with the current generation is refused the same way.
9. `test_version_floor_continues_above_announced_and_stored_versions` — `a` generation 1 announces versions 0..12 and uploads `checkpoint` version 20 (saved: `checkpoints/a/ckpt_v20/`); `n1` reconnects without `a` → the new `start_learner` has generation 2, `version_floor == 21`.
10. `test_train_records_carry_learner_node_and_generation` — `n1` sends `MetricsBatch(train=[TrainMetrics(agent_id="a", generation=1, train_step=3, values={"total_loss": 0.5})])` → a `train` record of `a` with `learner_node == n1.number`, `generation == 1`; the same with `generation=0` gives no record.
11. `test_train_without_listen_fails_fast_on_an_impossible_device` — `colosseum train` (through `cli_runner`) of a TINY tic-tac-toe config with `--set learner.device=cuda:7`: exit code 1, stderr has `cannot run on this machine` and `slot on GPU 7`; nothing else starts (no `worker-0.log`).

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_placement.py tests/unit/test_sp5_version_parking.py tests/integration/test_sp5_learner_placement.py -q`
Expected: all pass (about 20 s).

- [ ] **Step 7: The node side (`src/colosseum/node/agent.py`, `src/colosseum/node/learner_main.py`, `src/colosseum/learner/learner.py`, `src/colosseum/utils/seeding.py`)**

1. **`start_learner`** (a `Command` on the session, or an entry of `Welcome.learners` the node does not run): at most one learner per agent; a running learner of the agent with a lower generation is stopped first (`Stop(reason="superseded")`, no final snapshot). Fetch the start blob before spawning: `""` → none; `init` → the agent's `init` blob from the cache (`Welcome.fixed`, already fetched); `start:<ident>` → `Download(BlobRequest(kind="start", agent_id=agent, ident=ident))` through the transfer pool into memory, `decode_checkpoint(buf, cap=payload_byte_cap(max_message_mb))` → `resume_state = {"model_state": payload["model_state"], "trainer_state": payload["trainer_state_bytes"] or None, "policy_version": payload["policy_version"], "env_steps": assignment.env_steps, "source": f"{ident} of the hub"}`. A failed fetch is retried at the next main-loop pass (WARNING once); the learner is not started without its start point. Floor: `version_floor = max(assignment.version_floor, cached + 1)` with `cached = WeightCacheReader.latest_version(agent)[0]` when the machine's cache has a `latest` pointer for the agent (no reuse of a version a dead incarnation published locally without announcing it). Spawn `learner-<agent>` (a new incarnation: a new intake address, T2.2) with the assignment's device, generation and floor; INFO `learner-<agent>: generation <g>, device <d>, versions from <floor>, start <blob or "factory">`.
2. **`stop_learner(agent, generation, reason)`**: only when the running learner of `agent` has that generation (else DEBUG and ignore); send `Stop(reason=reason)`; with `superseded` the learner exits without a final snapshot (below); grace `distributed.shutdown_grace_sec`, then terminate.
3. **`Welcome.learners` is the desired set** on every (re)connect: learners of the node that are not in it (or have another generation) are stopped (`superseded`); listed ones the node does not run are started as in 1; listed ones it runs with the same generation keep running (same process).
4. **`Hello.learners`** = `RunningLearner(agent, generation, last published version)` of every live learner child (T2.4 may already fill it; the generation now comes from the assignment).
5. **`node.learner_main.learner_target`**: takes `device`, `generation`, `version_floor`, `resume_state`, `init_state`, `env_steps` from the assignment kwargs: the agent config's `learner.device` is replaced by the assigned device before `resolve_device` and `resolve_learner_threads` (`cuda:N` → that GPU); `learner_process(..., version_floor=version_floor, send_final=lambda: intake.stop_reason != "superseded", ...)`; the seed is `learner_seed(training.seed, agent_index, generation)`.
6. **`learner_process(..., send_final=None)`** (`learner/learner.py`): the final snapshot on stop is sent only when `send_final is None or send_final()`; `ConnectionIntake` (T2.2) keeps the `reason` of the `Stop` it read in `stop_reason` (None before any).
7. **`utils/seeding.py`**:
```python
def learner_seed(seed: int | None, agent_index: int, generation: int = 1) -> int | None:
    """Seed of the learner of the ``agent_index``-th trainable agent (None if unseeded). Generation 1 keeps the
    SP4 stream; every later generation (a restart, a relocation, a resumed hub) gets its own (SP5 T4.1)."""
    if seed is None:
        return None
    return derive_seed(seed, "learner", agent_index) if generation <= 1 else \
        derive_seed(seed, f"learner-g{int(generation)}", agent_index)
```

- [ ] **Step 8: Node tests (`tests/integration/test_sp5_learner_start.py`)**

Module docstring: "The node runs the learners it is assigned (SP5 T4.1): device, generation and version floor; the start point of a `start:*` blob; the floor above the machine's cache; superseded and stale stops; the desired set of `Welcome.learners`." Setup: `node_kit.FakeHub` (in-process server) with a toy config (`game_helpers` "turns", trainable `a`, `learner.batch_chunks: 2`), a `NodeAgent` in a thread (`cluster_kit.node_in_thread`, `NodeOptions(workers=1, cpu_learners=1, identity_root=tmp_path / "node")`): one learner child and one worker child (2 loaded processes). Tests:

1. `test_start_learner_spawns_the_learner_with_device_generation_and_floor` — `FakeHub.send_command(start_learner(a, generation=3, version_floor=17, device="cpu"))` → the FakeHub records `VersionAnnounce(agent_id="a", generation=3, version>=17)`; after `FakeHub.drop_session()` the next `Hello.learners == [RunningLearner("a", 3, v)]` with `v >= 17`; `learner-a.log` (node log dir) has `generation 3, device cpu, versions from 17`.
2. `test_the_floor_is_above_the_cached_pointer` — before the command, publish a `latest` of `a` at version 40 into the node's cache (`WeightCacheWriter(shm_dir(), cache_prefix(ca8, run_id, node))` with a frame of the right weights, `node_kit` knows the prefix) → `start_learner(floor 0)` → the first announce has version ≥ 41.
3. `test_a_start_frame_restores_weights_and_trainer_state` — FakeHub serves `("start", "a", "resume")` = `encode_checkpoint(make_checkpoint_payload("a", algo), kind="recovery", generation=0)` of the agent's real algorithm (`colosseum.learner.factory.build_algorithm(config, role_spec, spec, device="cpu")`) after `set_policy_version(algo, 9)` → `start_learner(weights_blob="start:resume", version_floor=5)`: the first announced version is 9; the `latest` segment the learner published equals the frame's `model_state` array by array (`WeightCacheReader.read_latest`); `learner-a.log` has `Resumed from resume of the hub at policy_version 9`.
4. `test_stop_learner_of_another_generation_is_ignored` — generation 3 runs; `stop_learner(a, 2, "superseded")` → still announcing; `stop_learner(a, 3, "superseded")` → the learner exits (its pid gone), no `Upload` of kind `final` reaches the FakeHub, the next `Hello.learners == []`.
5. `test_a_newer_start_replaces_the_running_learner` — `start_learner(a, 4)` while generation 3 runs: generation 3 exits without a final, generation 4 announces; the learner's intake address of the new incarnation ends with `-i1` (the node log line naming it), the worker's chunks go to it (its log reports chunks received).
6. `test_welcome_is_the_desired_learner_set` — generation 3 runs; `FakeHub.drop_session()` and the next `Welcome.learners == []` → the learner stops (no final); in a second phase a `Welcome` listing `a` generation 5 → a learner of generation 5 starts; a `Welcome` listing the running generation keeps the same pid.

- [ ] **Step 9: The GPU test (`tests/integration/test_sp5_learner_gpu.py`, `@pytest.mark.gpu`, skipped without CUDA)**

1. `test_auto_puts_the_local_learner_on_the_gpu` — `colosseum train` of TINY tic-tac-toe with `learner.device: auto`: exit 0; `hub_state.json` learners `agent_0` device `cuda:0`; `learner-agent_0.log` has `device=cuda:0`; `train` records carry `generation == 1` and `learner_node` = the local node's number.
2. `test_two_agents_share_the_local_gpu` — `configs/examples/tic_tac_toe_multi.yaml` with TINY overrides and `learner.device: auto`: both learners on `cuda:0` (R18), exit 0.

- [ ] **Step 10: Focused tests, full fast suite + ruff, slow suite, GPU suite**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_placement.py tests/unit/test_sp5_version_parking.py tests/integration/test_sp5_learner_placement.py tests/integration/test_sp5_learner_start.py -q -rw`, then the full fast suite and ruff, then the slow suite (alone, in the background), then the GPU suite. Expected: all green; zero warnings. SP2–SP4 tests that read `learner_seed(seed, i)` with two arguments are unchanged (generation defaults to 1).

- [ ] **Step 11: Commit and push**

```bash
git add src/colosseum/hub/placement.py src/colosseum/hub/state.py src/colosseum/hub/hub.py src/colosseum/hub/server.py \
        src/colosseum/net/proto src/colosseum/node/agent.py src/colosseum/node/learner_main.py \
        src/colosseum/learner/learner.py src/colosseum/utils/seeding.py \
        tests/unit/test_sp5_placement.py tests/unit/test_sp5_version_parking.py \
        tests/integration/test_sp5_learner_placement.py tests/integration/test_sp5_learner_start.py \
        tests/integration/test_sp5_learner_gpu.py
git commit -m "feat: learner placement on node slots, pending agents with parked weights, generations and fencing"
git push
```

---
### Task T4.2: Recovery points, restarts, machine loss, relocation, the final snapshot by its own call

Spec 5.2 («Точка восстановления», trainer state as bytes, `weights_only=True`, restart before the first point), 7.1 (rows: worker / evaluator death, learner crash, machine with workers lost, machine with the learner lost, network cut), 7.2 (recovery points of a stale generation refused), 7.4 («Финальный снимок отправляется отдельным вызовом больших данных и не теряется из-за полной очереди»); R9. Learners send a recovery point every `learner.recovery_interval_sec`; the hub stores the newest in `runs/<name>/recovery/<agent>/` and starts every later generation of the agent from it. A learner that dies is reported by its node and re-assigned by the hub on the same slot with a new generation (≤ 3 restarts per 30 min, counted by the node); then the agent is `failed`. Workers and evaluators are restarted by their node (≤ 3 per 10 min) with new connections and new epochs; then the slot is off. A lost machine's cells go at once, its learners wait `learner_relocate_after_sec` and move. Checkpoint uploads get their own queue on the node, in which a final snapshot is never dropped.

**Split point:** dispatch (a) Steps 1–7 (recovery points: library, learner, node uploads, hub store, start source; learner restarts and `failed`), (b) Steps 8–13 (worker / evaluator restarts and disabled slots, machine loss and relocation, the SP2 lifecycle test ports, the relocation tests, suites, commit).

**Files:**
- Create: `src/colosseum/hub/recovery.py`, `src/colosseum/node/uploads.py`
- Modify: `src/colosseum/learner/learner.py` (recovery cadence, `clock`), `src/colosseum/node/learner_main.py` (passes `recovery_interval_sec`)
- Modify: `src/colosseum/node/agent.py` (`UploadQueue` for every checkpoint frame, the restart policy, learner / worker / evaluator events, pending events across reconnects)
- Modify: `src/colosseum/hub/hub.py` (`_on_upload` recovery and final, `_on_event` deaths and limits, node loss → relocating, the relocation deadline, the start source and floor with the recovery point, all-agents-failed stop, exit code 1 for failed agents), `src/colosseum/hub/server.py` (`Download(kind="start", ident="recovery")`), `src/colosseum/hub/state.py` (`CellTable.drop_worker`), `src/colosseum/core/run_dir.py` (`RunDir.recovery_dir`, docstring)
- Modify: `tests/integration/test_sp2_lifecycle.py` (two tests change with spec 7.1, Step 11)
- Test: `tests/unit/test_sp5_recovery_store.py`, `tests/unit/test_sp5_upload_queue.py`, `tests/unit/test_sp5_learner_recovery.py`, `tests/integration/test_sp5_node_restarts.py`, `tests/integration/test_sp5_hub_failures.py`, `tests/integration/test_sp5_relocation.py`

**Interfaces:**
- Consumes: T4.1's `LearnerTable` (`died`, `fail`, `node_lost`, `relocation_due`, `reconcile`), `_place_learners`, `_start_learner`, `_park`, `_learner_start_source`, `_version_floor`; `ChildSupervisor.may_restart`, `WORKER_RESTARTS = (3, 600.0)`, `LEARNER_RESTARTS = (3, 1800.0)`, `node.logs.log_tail` (T2.4); `NodeOutputs.send_checkpoint(kind=...)`, `CheckpointHeader` (T2.2); `UploadHeader` / `UploadReply`, `send_parts` (T1.2, T2.3); `peek_meta`, `MAX_HEADER_BYTES`, `encode_checkpoint` (T1.1); `NodeRegistry.lost`, `CellTable` (T2.1); T2.5's node-loss handling.
- Produces (contract plus additions marked *):
  - *`colosseum.hub.recovery`: `RECOVERY_DIR`, `RECOVERY_FRAME`, `RECOVERY_META`, `RecoveryPoint`, `RecoveryStore(run_root)` (`path`, `get`, `accept`), `read_frame_meta(path)`;
  - *`colosseum.node.uploads`: `FINAL` / `RECOVERY` / `PERIODIC`, `UPLOAD_CONCURRENCY = 2`, `Upload`, `UploadQueue` (`offer`, `take`, `done`, `waiting`, `drop_waiting`, `replaced`);
  - `learner_process(..., recovery_interval_sec=...)` (contract) and *`clock: Callable[[], float] = time.monotonic`;
  - *`CellTable.drop_worker(node, worker) -> list[CellEntry]`; *`RunDir.recovery_dir`;
  - hub counters `learner_restarts`, `worker_restarts` (contract `system` keys, T7.1 shows them) and *`dropped_uploads` (`{"stale_generation", "invalid", "not_newer"}`); *`system` value `failed_agents`;
  - *Event kinds the hub handles: `proc_died` (worker / evaluator / learner, with `generation` for learners), `learner_failed`, `slot_disabled`.

- [ ] **Step 1: Write the failing library tests**

Create `tests/unit/test_sp5_recovery_store.py` (the frame helper `encode_frame` is T1.1's: a checkpoint frame's header carries `kind="checkpoint"` and meta `{"kind", "agent_id", "generation", "policy_version", "final"}`; if T1.1 names the meta keys differently, adapt `_frame` and `RecoveryStore._check` together):
```python
"""Recovery points in the hub (SP5 T4.2, spec 5.2 and 7.2): accepted only from the current generation and never
older than the stored one, header-checked without decoding, moved into place atomically, overwritten, with a
meta.json copy; unreadable files are ignored with a WARNING."""
from __future__ import annotations

import json
import logging

import numpy as np

from colosseum.hub.recovery import RECOVERY_META, RecoveryStore, read_frame_meta
from colosseum.net.frame import encode_frame


def _frame(agent="main", generation=2, version=40, kind="recovery") -> bytes:
    meta = {"kind": kind, "agent_id": agent, "generation": generation, "policy_version": version, "final": False}
    tree = {"w": np.arange(6, dtype=np.float32), "trainer_state_bytes": np.frombuffer(b"\x80\x02xyz", np.uint8)}
    return encode_frame(tree, kind="checkpoint", meta=meta)


def _spool(tmp_path, data: bytes, name: str = "up-1") -> object:
    path = tmp_path / ".incoming" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    return path


def test_accept_moves_the_frame_into_place_and_writes_meta(tmp_path):
    store = RecoveryStore(tmp_path)
    spooled = _spool(tmp_path, _frame())
    assert store.accept("main", spooled, generation=2, version=40, env_steps=1234, current_generation=2) == (True, "")
    assert not spooled.exists()
    point = store.get("main")
    assert (point.generation, point.version, point.env_steps) == (2, 40, 1234)
    assert point.path == tmp_path / "recovery" / "main" / "recovery.frame"
    assert json.loads((point.path.parent / RECOVERY_META).read_text()) == {"generation": 2, "version": 40,
                                                                             "env_steps": 1234}
    assert read_frame_meta(point.path) == ("checkpoint", {"kind": "recovery", "agent_id": "main", "generation": 2,
                                                          "policy_version": 40, "final": False})


def test_stale_generation_and_older_points_are_refused(tmp_path):
    store = RecoveryStore(tmp_path)
    store.accept("main", _spool(tmp_path, _frame(version=40)), generation=2, version=40, env_steps=1,
                 current_generation=2)
    stale = _spool(tmp_path, _frame(generation=1, version=99), "up-2")
    assert store.accept("main", stale, generation=1, version=99, env_steps=2, current_generation=2) == (
        False, "stale_generation")
    assert not stale.exists()
    older = _spool(tmp_path, _frame(version=30), "up-3")
    assert store.accept("main", older, generation=2, version=30, env_steps=3, current_generation=2) == (
        False, "not_newer")
    newer = _spool(tmp_path, _frame(generation=3, version=55), "up-4")
    assert store.accept("main", newer, generation=3, version=55, env_steps=4, current_generation=3) == (True, "")
    assert (store.get("main").generation, store.get("main").version) == (3, 55)


def test_a_header_that_does_not_match_the_upload_is_invalid(tmp_path):
    store = RecoveryStore(tmp_path)
    for i, data in enumerate((_frame(agent="other"), _frame(version=41), _frame(kind="periodic"), b"not a frame")):
        spooled = _spool(tmp_path, data, f"bad-{i}")
        assert store.accept("main", spooled, generation=2, version=40, env_steps=0, current_generation=2) == (
            False, "invalid")
    assert store.get("main") is None


def test_the_frame_header_wins_over_a_stale_meta_copy(tmp_path):
    store = RecoveryStore(tmp_path)
    store.accept("main", _spool(tmp_path, _frame(version=40)), generation=2, version=40, env_steps=500,
                 current_generation=2)
    (tmp_path / "recovery" / "main" / RECOVERY_META).write_text(json.dumps(
        {"generation": 2, "version": 10, "env_steps": 9}))           # a crash between the two writes
    point = store.get("main")
    assert (point.version, point.env_steps) == (40, 0)


def test_an_unreadable_point_is_ignored_with_a_warning(tmp_path, caplog):
    path = tmp_path / "recovery" / "main" / "recovery.frame"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"CSTF\x01\x00")
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.recovery"):
        assert RecoveryStore(tmp_path).get("main") is None
    assert "unreadable" in caplog.text
```

Create `tests/unit/test_sp5_upload_queue.py`:
```python
"""The node's checkpoint uploads (SP5 T4.2, spec 5.2 and 7.4): finals are never dropped and go first, recovery
points and periodic snapshots wait one deep (a replaced periodic snapshot is counted), one upload per agent at a
time, failed uploads wait again, refused ones are not retried, the stop deadline drops what is left."""
from __future__ import annotations

import pytest

from colosseum.node.uploads import UploadQueue


def _kinds(uploads):
    return [(u.agent_id, u.kind, u.version) for u in uploads]


def test_finals_first_then_recovery_then_periodic_one_per_agent():
    q = UploadQueue(concurrency=4)
    q.offer("a", "periodic", 1, 10, b"p10")
    q.offer("a", "recovery", 1, 12, b"r12")
    q.offer("b", "periodic", 1, 5, b"bp5")
    q.offer("a", "final", 1, 13, b"f13")
    started = q.take()
    assert _kinds(started) == [("a", "final", 13), ("b", "periodic", 5)]
    assert q.take() == []                                     # a is busy, b has nothing else
    for upload in started:
        assert q.done(upload.ticket, True) is upload
    assert _kinds(q.take()) == [("a", "recovery", 12)]
    assert q.done(999, True) is None


def test_a_newer_periodic_snapshot_replaces_the_waiting_one_and_is_counted():
    q = UploadQueue()
    for version in (10, 20, 30):
        q.offer("a", "periodic", 1, version, b"x")
    assert q.replaced == 2 and q.waiting("periodic") == 1
    assert _kinds(q.take()) == [("a", "periodic", 30)]
    q.offer("a", "recovery", 1, 31, b"r")
    q.offer("a", "recovery", 1, 32, b"r")
    assert q.replaced == 2                                    # recovery points are replaced silently


def test_finals_are_never_replaced_and_retried_after_a_failure():
    q = UploadQueue(concurrency=1)
    t1 = q.offer("a", "final", 1, 13, b"f1")
    q.offer("a", "final", 2, 40, b"f2")                       # another generation's final: both kept
    assert _kinds(q.take()) == [("a", "final", 13)]
    q.done(t1, False)                                          # the hub is away
    assert _kinds(q.take()) == [("a", "final", 13)]           # first again
    q.done(t1, True)
    assert _kinds(q.take()) == [("a", "final", 40)]


def test_a_failed_periodic_upload_waits_again_unless_a_newer_one_came():
    q = UploadQueue()
    t = q.offer("a", "periodic", 1, 10, b"x")
    q.take()
    q.done(t, False)
    assert q.waiting("periodic") == 1 and q.replaced == 0
    t = q.take()[0].ticket
    q.offer("a", "periodic", 1, 20, b"y")
    q.done(t, False)
    assert q.replaced == 1 and _kinds(q.take()) == [("a", "periodic", 20)]


def test_a_refused_upload_is_not_retried_and_the_concurrency_bound_holds():
    q = UploadQueue(concurrency=2)
    for agent in "abc":
        q.offer(agent, "periodic", 1, 1, b"x")
    started = q.take()
    assert len(started) == 2
    q.done(started[0].ticket, False, retry=False)              # refused: stale_generation
    assert q.waiting() == 2 and len(q.take()) == 1


def test_drop_waiting_at_the_stop_deadline():
    q = UploadQueue(concurrency=1)
    q.offer("a", "final", 1, 1, b"f")
    q.offer("b", "final", 1, 1, b"g")
    q.offer("b", "periodic", 1, 1, b"p")
    q.take()
    dropped = q.drop_waiting()
    assert sorted((u.agent_id, u.kind) for u in dropped) == [("b", "final"), ("b", "periodic")]
    assert q.waiting() == 1                                    # the one in flight
    with pytest.raises(ValueError):
        q.offer("a", "weird", 1, 1, b"")
```

Create `tests/unit/test_sp5_learner_recovery.py` (written against T2.2's `ChunkIntake` / `LearnerOutputs` protocols; the `weights_only` test was run against today's `apply_resume_state`):
```python
"""Recovery points from the learner (SP5 T4.2, spec 5.2): one every learner.recovery_interval_sec on the learner's
monotonic clock, never twice without a new train step, sent non-blocking as kind "recovery"; none without an
interval; the final snapshot on stop unless the learner was superseded; a learner restored from a recovery point
continues its optimizer state and publishes max(restored version, floor); a trainer state that needs pickle is
refused (weights_only=True)."""
from __future__ import annotations

import io
import pickle
import queue
import threading
import time

import pytest
import torch

from colosseum.core.config import LearnerConfig
from colosseum.core.types import TrajectoryChunk
from colosseum.learner.learner import apply_resume_state, learner_process, make_checkpoint_payload
from game_helpers import chunk_v2_payload, learner_appo


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class ListIntake:
    """``ChunkIntake`` over prepared chunks; every ``get`` advances the fake clock by one second."""

    def __init__(self, payloads, clock: FakeClock) -> None:
        self._chunks = [TrajectoryChunk.from_payload(p) for p in payloads]
        self._clock = clock
        self.stop_reason = None

    @property
    def depth(self) -> int:
        return len(self._chunks)

    @property
    def dropped(self) -> dict[str, int]:
        return {}

    def get(self, timeout: float) -> TrajectoryChunk:
        self._clock.t += 1.0
        if not self._chunks:
            time.sleep(min(timeout, 0.01))
            raise queue.Empty
        return self._chunks.pop(0)


class RecordingOutputs:
    def __init__(self) -> None:
        self.weights, self.checkpoints, self.metrics = [], [], []

    def publish_weights(self, payload) -> None:
        self.weights.append(payload)

    def send_checkpoint(self, payload: dict, *, kind: str, block: bool) -> bool:
        self.checkpoints.append((kind, block, payload))
        return True

    def send_metrics(self, metrics: dict) -> None:
        self.metrics.append(metrics)


def _run(n_chunks: int, *, interval, algo=None, resume_state=None, version_floor: int = 0, send_final=None):
    clock, outputs, stop = FakeClock(), RecordingOutputs(), threading.Event()
    intake = ListIntake([chunk_v2_payload(version=v) for v in range(n_chunks)], clock)
    algo = algo if algo is not None else learner_appo()
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, intake=intake, outputs=outputs,
        config=LearnerConfig(batch_chunks=1, device="cpu"), stop_event=stop, resume_state=resume_state,
        version_floor=version_floor, recovery_interval_sec=interval, clock=clock, send_final=send_final))
    thread.start()
    deadline = time.monotonic() + 60
    while len(outputs.metrics) < n_chunks and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    return outputs, algo


def test_recovery_points_follow_the_interval_and_a_new_train_step():
    outputs, _ = _run(10, interval=2.5)
    kinds = [(kind, block, payload["policy_version"]) for kind, block, payload in outputs.checkpoints]
    assert kinds == [("recovery", False, 3), ("recovery", False, 6), ("recovery", False, 9), ("final", True, 10)]
    recovery = outputs.checkpoints[0][2]
    assert recovery["final"] is False and isinstance(recovery["trainer_state_bytes"], bytes)


def test_no_recovery_points_without_an_interval():
    outputs, _ = _run(4, interval=None)
    assert [kind for kind, _, _ in outputs.checkpoints] == ["final"]


def test_a_superseded_learner_sends_no_final_snapshot():
    outputs, _ = _run(2, interval=None, send_final=lambda: False)
    assert outputs.checkpoints == []


def test_a_restored_recovery_point_continues_the_optimizer_and_the_versions():
    first, _ = _run(7, interval=2.5)
    point = first.checkpoints[1][2]                                  # the recovery point at version 6
    state = {"model_state": point["model_state"], "trainer_state": point["trainer_state_bytes"],
             "policy_version": point["policy_version"], "env_steps": 0, "source": "recovery point v6"}
    second, algo = _run(0, interval=None, resume_state=state, version_floor=8)
    assert second.weights[0].policy_version == 8                     # max(restored 6, floor 8)
    final = second.checkpoints[-1][2]
    saved = torch.load(io.BytesIO(point["trainer_state_bytes"]), weights_only=True)
    again = torch.load(io.BytesIO(final["trainer_state_bytes"]), weights_only=True)
    assert saved["optimizer"]["state"].keys() == again["optimizer"]["state"].keys()
    for key, tensors in saved["optimizer"]["state"].items():
        for name, value in tensors.items():
            assert torch.equal(value, again["optimizer"]["state"][key][name]), (key, name)


class _Evil:
    def __reduce__(self):
        return (print, ("unpickled",))


def test_a_trainer_state_that_needs_pickle_is_refused():
    payload = make_checkpoint_payload("a", learner_appo())
    buf = io.BytesIO()
    torch.save({"x": _Evil()}, buf)
    with pytest.raises(pickle.UnpicklingError):
        apply_resume_state(learner_appo(), {"model_state": payload["model_state"], "trainer_state": buf.getvalue(),
                                            "policy_version": 3, "source": "x"})
```
(`APPO.state_dict()` carries the optimizer under `"optimizer"`; if its key differs, compare that entry.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_recovery_store.py tests/unit/test_sp5_upload_queue.py tests/unit/test_sp5_learner_recovery.py -q`
Expected: `ModuleNotFoundError` for `colosseum.hub.recovery` and `colosseum.node.uploads`; `TypeError: learner_process() got an unexpected keyword argument 'clock'` (the `weights_only` test already passes: it pins today's behaviour).

- [ ] **Step 3: Write `src/colosseum/hub/recovery.py` and `src/colosseum/node/uploads.py`**

`src/colosseum/hub/recovery.py`:
```python
"""Recovery points of the learners (SP5 spec block 5.2; task T4.2).

``runs/<name>/recovery/<agent>/recovery.frame`` holds the newest recovery point of each agent: one checkpoint frame
(``colosseum.net.chunks.encode_checkpoint(..., kind="recovery")``: the model weights and the trainer state, the
latter as one uint8 leaf of ``torch.save`` bytes). The hub never decodes the arrays and never unpickles anything:
it checks the frame header only (``colosseum.net.frame.peek_meta``) and moves the spooled upload into place with
one ``os.replace`` (atomic), so a reader (a ``Download`` servicer thread, a resumed hub) sees the old or the new
point, never a mix. ``meta.json`` next to it is a readable copy (``{"generation", "version", "env_steps"}``) written
after the frame; the frame header is the authority when the two disagree (a crash between the two writes).

A recovery point is not a snapshot: it is never in the opponent pool, never rated, never retained or evicted; it is
overwritten by the next one. Accepted only from the agent's current learner generation (spec 7.2) and never older
than the stored one. Main loop only (servicer threads only read ``path(agent)``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from colosseum.net.frame import MAX_HEADER_BYTES, FrameError, peek_meta
from colosseum.utils.fs import write_text_atomic

logger = logging.getLogger(__name__)

RECOVERY_DIR = "recovery"
RECOVERY_FRAME = "recovery.frame"
RECOVERY_META = "meta.json"
_HEAD = 16


@dataclass(frozen=True)
class RecoveryPoint:
    agent_id: str
    generation: int
    version: int
    env_steps: int
    size: int
    path: Path


def read_frame_meta(path: str | Path) -> tuple[str, dict]:
    """``peek_meta`` of a frame file (reads the header only). FrameError for a file that is not a frame."""
    with open(path, "rb") as fh:
        head = fh.read(_HEAD)
        if len(head) < _HEAD:
            raise FrameError("truncated")
        length = int.from_bytes(head[8:16], "little")
        if length > MAX_HEADER_BYTES:
            raise FrameError("header_too_large")
        return peek_meta(head + fh.read(length))


class RecoveryStore:
    """The recovery points of a run dir (module docstring)."""

    def __init__(self, run_root: str | Path) -> None:
        self._root = Path(run_root) / RECOVERY_DIR

    def path(self, agent_id: str) -> Path:
        return self._root / agent_id / RECOVERY_FRAME

    def get(self, agent_id: str) -> RecoveryPoint | None:
        """The stored point (from the frame header; ``env_steps`` from meta.json, 0 when missing), or None (none
        yet, or an unreadable file: WARNING)."""
        path = self.path(agent_id)
        if not path.is_file():
            return None
        try:
            kind, meta = read_frame_meta(path)
            if kind != "checkpoint" or meta.get("kind") != "recovery" or meta.get("agent_id") != agent_id:
                raise FrameError("not_a_recovery_point")
            generation, version = int(meta["generation"]), int(meta["policy_version"])
        except (OSError, FrameError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Recovery point {path} is unreadable ({type(e).__name__}: {e}); ignored")
            return None
        env_steps = 0
        try:
            saved = json.loads((path.parent / RECOVERY_META).read_text())
            if int(saved.get("version", -1)) == version and int(saved.get("generation", -1)) == generation:
                env_steps = int(saved.get("env_steps") or 0)
        except (OSError, ValueError, TypeError):
            pass
        return RecoveryPoint(agent_id, generation, version, env_steps, path.stat().st_size, path)

    def accept(self, agent_id: str, spooled: str | Path, *, generation: int, version: int, env_steps: int,
               current_generation: int) -> tuple[bool, str]:
        """Store the uploaded frame ``spooled`` (in the run dir's ``.incoming/``) as ``agent_id``'s recovery point.
        Returns ``(accepted, reason)``; reasons: ``stale_generation`` (not the agent's current learner
        generation), ``invalid`` (the header is not a recovery checkpoint of this agent, generation and version),
        ``not_newer`` (older than the stored point). A refused file is deleted."""
        spooled = Path(spooled)
        reason = self._check(agent_id, spooled, generation, version, current_generation)
        if reason:
            spooled.unlink(missing_ok=True)
            return False, reason
        target = self.path(agent_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(spooled, target)
        write_text_atomic(target.parent / RECOVERY_META, json.dumps(
            {"generation": int(generation), "version": int(version), "env_steps": int(env_steps)}) + "\n")
        return True, ""

    def _check(self, agent_id: str, spooled: Path, generation: int, version: int, current_generation: int) -> str:
        if int(generation) != int(current_generation):
            return "stale_generation"
        try:
            kind, meta = read_frame_meta(spooled)
        except (OSError, FrameError):
            return "invalid"
        if (kind != "checkpoint" or meta.get("kind") != "recovery" or meta.get("agent_id") != agent_id
                or meta.get("generation") != generation or meta.get("policy_version") != version):
            return "invalid"
        stored = self.get(agent_id)
        if stored is not None and version < stored.version:
            return "not_newer"
        return ""
```

`src/colosseum/node/uploads.py`:
```python
"""Checkpoint uploads of a node (SP5 spec 5.2, 7.4; task T4.2): which learner checkpoint frame goes to the hub next.

A learner hands its node three kinds of checkpoint frames (``CheckpointHeader`` + frame on its control connection);
the node uploads each with its own ``Hub/Upload`` call (never on the session, never through a bounded queue that
could drop a final snapshot: spec 7.4). Per agent:

- ``final``: one per learner generation; never replaced, never dropped by the queue; a failed upload is retried
  until it is delivered or the caller gives up at the stop deadline (``drop_waiting``);
- ``recovery``: at most one waiting; a newer one replaces it (only the newest recovery point matters);
- ``periodic``: at most one waiting; a newer one replaces it and the replaced one is counted in ``replaced`` (that
  snapshot never reaches the pool: the hub is slower than the learner's ``checkpoint.interval``).

At most one upload per agent is in flight (frames of one agent arrive in order) and at most ``concurrency`` per
node. Order: finals, then recovery points, then periodic snapshots; within a kind the oldest offer first. A failed
upload (the hub is away, the call broke) waits again, unless a newer frame of its kind arrived meanwhile (then it
is dropped; a periodic one is counted). A refused upload (the hub answered ``accepted: false``) is never retried.
Pure and single-threaded: the node's main loop owns it, the transfer pool runs the calls and posts the outcome.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

FINAL, RECOVERY, PERIODIC = "final", "recovery", "periodic"
_PRIORITY = {FINAL: 0, RECOVERY: 1, PERIODIC: 2}
UPLOAD_CONCURRENCY = 2


@dataclass(frozen=True)
class Upload:
    ticket: int
    agent_id: str
    kind: str                 # final | recovery | periodic
    generation: int
    version: int
    frame: bytes


class UploadQueue:
    """The node's checkpoint uploads (module docstring)."""

    def __init__(self, concurrency: int = UPLOAD_CONCURRENCY) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._concurrency = int(concurrency)
        self._next = 1
        self._finals: dict[str, deque[Upload]] = {}
        self._latest: dict[tuple[str, str], Upload] = {}
        self._flight: dict[int, Upload] = {}
        self.replaced = 0          # periodic snapshots that never left the node (replaced by a newer one)

    def offer(self, agent_id: str, kind: str, generation: int, version: int, frame: bytes) -> int:
        """Queue a frame; returns its ticket."""
        if kind not in _PRIORITY:
            raise ValueError(f"unknown checkpoint kind {kind!r}")
        upload = Upload(self._next, agent_id, kind, int(generation), int(version), frame)
        self._next += 1
        if kind == FINAL:
            self._finals.setdefault(agent_id, deque()).append(upload)
        else:
            if (agent_id, kind) in self._latest and kind == PERIODIC:
                self.replaced += 1
            self._latest[(agent_id, kind)] = upload
        return upload.ticket

    def take(self) -> list[Upload]:
        """The uploads to start now (marked in flight)."""
        busy = {u.agent_id for u in self._flight.values()}
        waiting = [q[0] for q in self._finals.values() if q] + list(self._latest.values())
        waiting.sort(key=lambda u: (_PRIORITY[u.kind], u.ticket))
        started = []
        for upload in waiting:
            if len(self._flight) >= self._concurrency:
                break
            if upload.agent_id in busy:
                continue
            if upload.kind == FINAL:
                self._finals[upload.agent_id].popleft()
            else:
                del self._latest[(upload.agent_id, upload.kind)]
            self._flight[upload.ticket] = upload
            busy.add(upload.agent_id)
            started.append(upload)
        return started

    def done(self, ticket: int, ok: bool, *, retry: bool = True) -> Upload | None:
        """The outcome of an upload: ``ok`` (delivered and accepted), or failed (``retry``: the call broke or the
        hub is away; ``retry=False``: the hub refused it). Returns the upload (None for an unknown ticket)."""
        upload = self._flight.pop(ticket, None)
        if upload is None or ok or not retry:
            return upload
        if upload.kind == FINAL:
            self._finals.setdefault(upload.agent_id, deque()).appendleft(upload)
        elif (upload.agent_id, upload.kind) not in self._latest:
            self._latest[(upload.agent_id, upload.kind)] = upload
        elif upload.kind == PERIODIC:
            self.replaced += 1
        return upload

    def waiting(self, kind: str | None = None) -> int:
        """Frames not yet delivered (waiting or in flight), of ``kind`` or all."""
        items = [*self._flight.values(), *self._latest.values(), *(u for q in self._finals.values() for u in q)]
        return sum(1 for u in items if kind is None or u.kind == kind)

    def drop_waiting(self) -> list[Upload]:
        """Give up on everything not in flight (the stop deadline passed); returns what was dropped."""
        dropped = [u for q in self._finals.values() for u in q] + list(self._latest.values())
        self._finals.clear()
        self._latest.clear()
        return dropped
```

- [ ] **Step 4: The learner (`src/colosseum/learner/learner.py`, `src/colosseum/node/learner_main.py`)**

In `learner_process` add the keyword `clock: Callable[[], float] = time.monotonic` (module docstring: "recovery points: every `recovery_interval_sec` on `clock` (None: off), only after a new train step, non-blocking"). After the resume and before the loop: `last_recovery_at = clock()`, `last_recovery_step = train_step`. After the periodic-checkpoint block of each train step:
```python
            if (recovery_interval_sec is not None and train_step > last_recovery_step
                    and clock() - last_recovery_at >= recovery_interval_sec):
                outputs.send_checkpoint(make_checkpoint_payload(agent_id, algorithm), kind="recovery", block=False)
                last_recovery_at, last_recovery_step = clock(), train_step
```
`node.learner_main` passes `recovery_interval_sec=config.learner.recovery_interval_sec`; the SP4 queue path (`QueueOutputs`, tests) passes None. `NodeOutputs.send_checkpoint(kind="recovery")` sends `CheckpointHeader(kind="recovery", ...)` + the frame on the control connection (T2.2); a recovery point costs one `torch.save` of the trainer state per interval (300 s by default).

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_recovery_store.py tests/unit/test_sp5_upload_queue.py tests/unit/test_sp5_learner_recovery.py tests/unit/test_learner_v2.py -q`
Expected: all pass.

- [ ] **Step 5: The node: uploads and the restart policy (`src/colosseum/node/agent.py`)**

1. **Uploads.** Every `CheckpointHeader` + frame from a learner goes into one `UploadQueue` (kinds `final` / `recovery` / `periodic`; `periodic` is T2.2's kind of a periodic snapshot). Each main-loop pass starts `queue.take()` uploads in the transfer pool (`Hub/Upload` with `UploadHeader(kind=..., agent_id, generation, version, total_size, digest)`; `kind` on the wire: `checkpoint` for periodic, `final`, `recovery`); the pool posts `(ticket, reply | error)` into the node inbox → `queue.done(ticket, ok=reply.accepted, retry=error is not None)`; a refused upload is logged (WARNING once per (agent, reason)); `queue.replaced` growing gives one WARNING per agent (`checkpoint uploads of <agent> are slower than its checkpoint.interval; older periodic snapshots are skipped`) and an `Event(kind="warning", message=...)` to the hub once. Failed uploads wait while the session is down and start again after the next `Welcome`; at the stop deadline (T4.4) `drop_waiting()` (each dropped final: ERROR naming agent and version). If T2.5 / T2.4 already upload checkpoints another way, replace that path with this queue (one path; the old one must not survive next to it).
2. **Restart policy** in the handler of a child exit (`ChildSupervisor.exited()`), for exits the node did not ask for (not during a stop, not after a `Stop` / `stop_learner`):
   - a child that sent `ChildError` first (contract / player error) is not restarted (T2.5 / T4.4: the run stops);
   - `worker-<w>` / `evaluator-<i>`: if `supervisor.may_restart(name, now)` → start it again with incarnation + 1 (new connections; its envs start without a lineup and request one with the new incarnation, so the hub gives new epochs: CellTable) and send `Event(kind="proc_died", proc=name, exit_code=code, log_tail=log_tail(path), log_path=str(path))`; else `Event(kind="slot_disabled", proc=name, ...)` and a local WARNING; the slot stays off until the node process restarts;
   - `learner-<agent>`: never restarted by the node itself: `Event(kind="proc_died" if supervisor.may_restart(name, now) else "learner_failed", proc=name, generation=<its generation>, exit_code, log_tail, log_path)`; its local routes become "no learner" (`LearnerRoute(agent, None, via_node=False, generation)`) until a new assignment arrives; the restart window is counted per agent name, so restarts on this node accumulate across generations.
   - While the session is down, events are kept (at most `PENDING_EVENTS_MAX = 100`, oldest dropped with a WARNING) and sent after the next `Welcome`; a learner that died meanwhile is absent from the `Hello.learners`, so the hub re-places it on reconnect (the kept event then only adds its exit code and tail).
3. `Heartbeat.procs` carries `ProcState(name, state, restarts)` per child (T2.4); `restarts` counts this node's restarts of that name.

- [ ] **Step 6: The hub: recovery points, learner deaths, `failed` (`src/colosseum/hub/hub.py`, `server.py`, `core/run_dir.py`)**

1. `self._recovery = RecoveryStore(run_dir.root)` in `launch`; `RunDir.recovery_dir` = `root / "recovery"` (docstring line `recovery/<agent>/{recovery.frame, meta.json}  newest recovery point per agent (SP5)`).
2. **`_on_upload(kind="recovery")`**: the uploading node must be the agent's slot node (else refused `stale_generation`); `accepted, reason = self._recovery.accept(agent, spooled, generation=h.generation, version=h.version, env_steps=self._budget.value, current_generation=self._versions.generation(agent))`; reply; a refusal is counted in `dropped_uploads[reason]` with one WARNING per (node, reason). The hub never decodes the frame (header only).
3. **`_on_upload(kind="final")`**: as a periodic checkpoint (T4.1's fencing first), saved with `final=True`; `self._finals.add((agent, generation))` (T4.4's stop waits on it).
4. **Start source with the recovery point.** `_learner_start_source(agent)`: `point = self._recovery.get(agent)`; when `point` exists and `point.version >= hub-start version` → `("start:recovery", point.version, f"the recovery point v{point.version} (generation {point.generation})")`; else T4.1's hub-start source (spec 5.2: before the first recovery point a restart starts from `init` or the resume source). `_version_floor(agent)` adds `point.version + 1`. `_park(agent)` considers the point as a source (its weights are read with `decode_checkpoint`, the trainer-state leaf ignored). `Download(kind="start", ident="recovery")` streams `RecoveryStore.path(agent)`.
5. **`_on_event`**:
   - `proc_died` of `learner-<agent>` with `generation=g`: `slot = self._learners.died(agent, g)`; None → DEBUG (an old incarnation); else `learner_restarts += 1`, WARNING `Learner of <agent> (generation <g>) died on node <n> (<name>) with exit code <c>; restarting it there from <source> (see <log path>)` (T4.4 adds the hub-side tail path) and `self._start_learner(agent, slot)` (a new generation on the same slot, a new intake address on the node);
   - `learner_failed` of `learner-<agent>` with the current generation: `self._learners.fail(agent, f"{n} restarts within {window:g} s on node {node}")`; ERROR `Learner of <agent> failed (<reason>); training of <agent> stopped, the other agents go on; the run will exit with code 1`; `self._park(agent)`; `_announce_agents()`; `_place_learners()` (its slot is free); `system` value `failed_agents` = the failed agents;
   - `proc_died` of `worker-<w>` / `evaluator-<i>`: `worker_restarts += 1`; WARNING `<proc> of node <n> (<name>) died with exit code <c>; restarted (see <log path>)`;
   - `slot_disabled`: WARNING `<proc> of node <n> (<name>) is disabled after <k> restarts within <window:g> s`; for a worker `self._cells.drop_worker(n, w)` (its cells no longer count in the owner deficit).
6. **Every trainable agent failed** (and at least one exists): stop the run like a budget stop with exit code 1 and ERROR `every trainable agent failed; stopping`.
7. **Exit code.** In `_exit_code_after_shutdown` (T4.4 owns the function; this step adds one rule): a code of 0 becomes 1 when `self._learners.failed()` is not empty (ERROR listing them).
8. **`CellTable.drop_worker(node, worker)`** (`hub/state.py`): removes and returns the live cells of that worker (same as `drop_node`, filtered).

- [ ] **Step 7: Recovery / restart tests**

`tests/integration/test_sp5_node_restarts.py` (module docstring: "A node's restart policy (SP5 T4.2, spec 7.1): a killed learner is reported with its generation and log tail and never blocks its siblings; the hub's new assignment brings it back from the recovery point; the learner restart limit; workers restarted with new incarnations; a slot that keeps dying is disabled; a final snapshot behind periodic ones."). Setup: `node_kit.FakeHub` + `NodeAgent` in a thread with 1 worker and the learner of `a` (2 loaded processes); child pids from the `<proc> started (pid N)` line of their logs (`setup_process_logging`). Tests:
1. `test_a_killed_learner_is_reported_and_its_siblings_keep_running` — `start_learner(a, 1)`, wait for 2 announces; `os.kill(pid, SIGKILL)`; the FakeHub records `Event(kind="proc_died", proc="learner-a", generation=1, exit_code=-9)` with a non-empty `log_tail` that ends with the last line of `learner-a.log` and `log_path` = that file; the node keeps heartbeating and the worker's env steps keep growing (two `Heartbeat`s later `env_steps` is larger); no new announce arrives until the FakeHub sends `start_learner(a, 2, weights_blob="start:recovery", version_floor=...)` with the frame the node uploaded as `recovery` before the kill (`learner.recovery_interval_sec: 0.5`); then generation 2 announces versions ≥ the floor and its intake address is a new incarnation's (`-i1`).
2. `test_the_learner_restart_limit_reports_learner_failed` — `monkeypatch.setattr(colosseum.node.procs, "LEARNER_RESTARTS", (1, 1800.0))` (the node is in-process): first kill → `proc_died`; after the FakeHub's `start_learner(a, 2)` a second kill → `Event(kind="learner_failed", proc="learner-a", generation=2)`.
3. `test_a_killed_worker_restarts_with_a_new_incarnation` — kill -9 `worker-0`: `Event(proc_died, proc="worker-0", exit_code=-9)`; the next `LineupRequest` names the worker's cells with `incarnation == 1`; the learner process is the same pid and keeps announcing.
4. `test_a_worker_that_keeps_dying_disables_its_slot` — the FakeHub grants lineups whose opponent is a scripted `game_helpers.CrashBot` agent (`os._exit(3)` at its first act): 3 `proc_died` events, then `Event(kind="slot_disabled", proc="worker-0")`; no `worker-0` process after it; the node still heartbeats and its learner keeps running.
5. `test_a_final_snapshot_is_never_lost_behind_periodic_ones` — the FakeHub delays every `Upload` by 0.3 s; the learner has `checkpoint.interval: 1` (a periodic snapshot per train step); `stop_learner(a, 1, reason="stop_run")` after 20 steps → the FakeHub receives exactly one `final` upload of version ≥ 20 and fewer than 20 periodic ones; the node log has one WARNING `slower than its checkpoint.interval`.

`tests/integration/test_sp5_hub_failures.py` (module docstring: "The hub's failure handling with fake nodes (SP5 T4.2, spec 5.2, 7.1, 7.2): recovery points stored and fenced, learner deaths re-assigned from the recovery point or the start source, failed agents, machine loss, relocation, disabled slots, version floors."). Setup as `test_sp5_learner_placement.py` (`node_lost_after_sec: 0.5`, `learner_relocate_after_sec: 1.0`). Tests:
1. `test_a_learner_death_restarts_it_from_the_recovery_point` — `a` on `n1` (generation 1) announces 0..4 and uploads `recovery` (generation 1, version 4) → `UploadReply(accepted=True)`, `runs/<name>/recovery/a/recovery.frame` holds those bytes, `meta.json == {"generation": 1, "version": 4, "env_steps": <hub counter>}`; `Event(proc_died, proc="learner-a", generation=1)` → `n1` receives `start_learner(a, 2, device as before, weights_blob="start:recovery", version_floor=5)`; `n1.download(BlobRequest(kind="start", agent_id="a", ident="recovery"))` returns the uploaded bytes; `hub_state` counters `learner_restarts == 1`; the same event again (generation 1) changes nothing.
2. `test_a_restart_before_any_recovery_point_uses_the_start_source` — `a` with `init.from` and no recovery point: the event gives `start_learner(a, 2, weights_blob="init", version_floor=<max announced + 1>)`.
3. `test_a_recovery_point_of_a_stale_generation_is_refused` — after generation 2 is assigned, `upload(recovery, generation 1, version 9)` → `UploadReply(accepted=False, reason="stale_generation")` and the stored point is unchanged; the current generation from another node → refused; an older version of the current generation → `not_newer`; garbage bytes → `invalid`; `hub_state` counters `dropped_uploads` count each reason once.
4. `test_a_failed_learner_stops_its_agent_but_not_the_run` — two agents; `Event(learner_failed, proc="learner-a", generation=1)` → learners `a` `failed`, `system.failed_agents == ["a"]`, an ERROR `Learner of a failed`, `Announce` carries a parked `VersionInfo` of `a` (from its recovery point), new grants have `a`'s seats with `collect=False` and no cell owned by `a`, `b` keeps its learner; `hub.request_stop()` → `launch()` returns 1.
5. `test_every_failed_agent_stops_the_run_with_exit_1` — one agent, `learner_failed` → the run stops by itself, `launch()` returns 1, ERROR `every trainable agent failed`.
6. `test_a_restarted_learner_never_reuses_a_version` — `a` generation 1 announces up to 12, uploads `checkpoint` 20 and `recovery` 25 (both accepted; versions above the last announce are possible: a learner publishes every `weight_push_interval` steps); `proc_died` → `version_floor == 26`.
7. `test_machine_loss_drops_cells_and_parks_its_learner` — `a` on `n1` with cells of `n1` and `n2` granted; `n1.pause_heartbeats()` → after `node_lost_after_sec`: `hub_state` `nodes_lost == 1`, learners `a` `relocating`, the owner rotation's next grants to `n2` count only `n2`'s cells (the deficit of `a` and `b` over `n2`'s cells alone), an `Announce` with a parked `VersionInfo` of `a`; `a`'s seats in new grants have `collect=False`.
8. `test_relocation_after_the_wait_moves_the_learner` — continuing 7 with `a: cpu` and `n2` offering a CPU slot: one `learner_relocate_after_sec` after `n1`'s last heartbeat, `n2` receives `start_learner(a, 2, weights_blob="start:recovery")`; `n1.resume_heartbeats()` on a new session with `Hello.learners=[a gen 1]` → `stop_learner(a, 1, "superseded")`.
9. `test_a_node_back_before_the_wait_keeps_its_learner` — `n1` comes back (new session, `Hello.learners=[a gen 1]`) before `learner_relocate_after_sec`: no new generation, `a` running again, the parked weights withdrawn (`Announce` again names `a`'s accepted version).
10. `test_relocation_without_a_free_slot_waits_as_pending` — `n2` without CPU slots: after the wait `a` is `pending` with a reason; a third node with a CPU slot connects → `start_learner(a, 2)` there.
11. `test_strict_placement_waits_for_its_node` — `a.learner.placement: {node: n1, strict: true}`: after the wait `a` stays pending although `n2` has a slot; `n1` returning (without its learner) gets `start_learner(a, 2)`.
12. `test_a_disabled_slot_releases_its_cells` — `n1` with 2 workers × 2 envs granted cells; `Event(slot_disabled, proc="worker-1")` → `hub._cells.live()` has no `(n1, 1, *)` cell and a WARNING names the slot; `Event(proc_died, proc="worker-0")` → `worker_restarts == 1`.

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_recovery_store.py tests/unit/test_sp5_upload_queue.py tests/unit/test_sp5_learner_recovery.py tests/integration/test_sp5_node_restarts.py tests/integration/test_sp5_hub_failures.py -q`
Expected: all pass.

- [ ] **Step 8: Machine loss and relocation in the hub (`src/colosseum/hub/hub.py`)**

1. In T2.5's node-loss handling (the main-loop deadline over `NodeRegistry.lost(now, node_lost_after_sec)`), after the cells are dropped: `moved = self._learners.node_lost(entry.number, entry.last_heartbeat)`; per moved agent a WARNING `Learner of <agent>: node <n> (<name>) lost; waiting up to learner_relocate_after_sec (<s> s) for it` and `self._park(agent)`; `_announce_agents()`. Unfinished matches of the node are never reported (the node is gone; nothing to do).
2. A main-loop deadline (every pass; cheap): `due = self._learners.relocation_due(now, cfg.distributed.learner_relocate_after_sec)`; if any: INFO `relocating the learner of <agent>`, then `_place_learners()` (a new generation on a matching slot, from the recovery point; `strict` placement may keep it pending).
3. The node's return is T4.1's reconciliation (adopt within the wait, stop after it).

- [ ] **Step 9: Relocation over a real network cut (`tests/integration/test_sp5_relocation.py`)**

Module docstring: "Network cuts and relocation on localhost (SP5 T4.2, criterion 11 shape): a short cut changes nothing; a long one moves the learner to the other node with a new generation and stops the old one when the machine returns." Setup: an in-process hub (`cluster_kit.hub_in_thread`) with a toy config: trainable `a` with `learner.device: cpu`, `learner.placement.node: laptop`, `learner.recovery_interval_sec: 0.5`, `distributed.node_lost_after_sec: 1.0`, `learner_relocate_after_sec: 2.0`, `heartbeat_sec: 0.1`; two `NodeAgent`s in threads with their own identity roots: `pc` (1 worker, 1 CPU learner slot) connects directly, `laptop` (no workers, 1 CPU learner slot, `--no-data-port`: its chunks come over the hub relay) connects through `fault_proxy.TcpFaultProxy` (2 loaded processes: `pc`'s worker and the learner).
1. `test_a_cut_shorter_than_node_lost_changes_nothing` — wait for the learner running on `laptop`; `proxy.cut(0.5)`; afterwards: the same node number of `laptop`, generation still 1, the journal's training uids of `pc` keep their epochs' sequence (no new epoch for `pc`'s cells), `hub_state` `nodes_lost == 0`.
2. `test_a_long_cut_relocates_the_learner_and_stops_the_old_one` (`@pytest.mark.slow`) — `proxy.cut(5.0)`: `hub_state` learners `a` moves to `pc` with generation 2 and starts from `start:recovery`; its first accepted version is above every version generation 1 announced; when the cut ends `laptop` reconnects, receives `stop_learner(a, 1)` and its learner process exits; no journaled result has a seat of generation 1 above cap(1) (read the journal's seats and the hub's caps); `dropped_match_results` from `stale_generation` is counted, not lost silently.

- [ ] **Step 10: The two SP2 lifecycle tests whose subject spec 7.1 changes (`tests/integration/test_sp2_lifecycle.py`)**

- `test_killed_worker_gives_exit_1_and_points_to_its_log` becomes `test_a_killed_worker_is_restarted_and_its_log_named`: after `kill -9` of `worker-0` the run goes on: a new `worker-0` pid appears in its log, `logs/main.log` has a WARNING `worker-0 of node <n> (...) died with exit code -9; restarted (see <root>/logs/worker-0.log)`; then SIGINT → exit 130 with a final checkpoint (`final_checkpoints(root)`), and no descendant alive within 10 s.
- `test_learner_crash_in_train_step_gives_exit_1` keeps its name and exit code 1, with the new path: `game_helpers.CrashingAPPO` crashes every incarnation → 3 restarts, then `learner_failed`, then "every trainable agent failed" → exit 1; stderr has `Learner of agent_0 failed` and the learner log path; `learner-agent_0.log` has `injected train_step failure` (four times: one per incarnation).
List both changes in the task report (spec §8 changes the SP1 "a dead child fails the run" rule for workers and learners).

- [ ] **Step 11: Focused tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_recovery_store.py tests/unit/test_sp5_upload_queue.py tests/unit/test_sp5_learner_recovery.py tests/integration/test_sp5_node_restarts.py tests/integration/test_sp5_hub_failures.py tests/integration/test_sp5_relocation.py tests/integration/test_sp2_lifecycle.py -q -rw -m "not slow"`, then `-m slow` for `test_sp5_relocation.py`.
Expected: all pass.

- [ ] **Step 12: Full fast suite + ruff; slow suite**

Run the full fast suite and ruff, then the slow suite (alone, in the background). Expected: green, zero warnings; the slow suite's learning thresholds unchanged (a recovery point every 300 s does not occur in them).

- [ ] **Step 13: Commit and push**

```bash
git add src/colosseum/hub/recovery.py src/colosseum/node/uploads.py src/colosseum/learner/learner.py \
        src/colosseum/node/learner_main.py src/colosseum/node/agent.py src/colosseum/hub/hub.py \
        src/colosseum/hub/server.py src/colosseum/hub/state.py src/colosseum/core/run_dir.py \
        tests/unit/test_sp5_recovery_store.py tests/unit/test_sp5_upload_queue.py tests/unit/test_sp5_learner_recovery.py \
        tests/integration/test_sp5_node_restarts.py tests/integration/test_sp5_hub_failures.py \
        tests/integration/test_sp5_relocation.py tests/integration/test_sp2_lifecycle.py
git commit -m "feat: recovery points, learner and worker restarts, machine loss and relocation, final snapshot upload"
git push
```

---
### Task T4.3: In-place resume, journal schema 2 with the `node` column, league replay, SP4 run dirs in place

Spec 5.4 (all of it), 4.3 («Журнал: новая колонка `node` …, версия схемы в `meta`, миграция на месте при продолжении папки SP4 и при импорте»), 4.4 (counters restored from `hub_state.json`; WARNING when resuming at or past the budget), 5.1 (`league_state.json` holds the last accounted journal row); rulings R7 (a resumed hub's epochs start at the next block) and R8 (rows after the watermark are replayed into ELO and PFSP). `colosseum train --resume RUN` and `colosseum hub --resume RUN` continue the same run dir and run id; the run is refused while its process lives; without `-c` the saved config runs (plus `--set`), with `-c` the new one with a diff WARNING and strict checks of the remaining agents; the hub restores its counters, epochs, versions and generations, learners (adopted from reconnecting nodes or placed again from the recovery point / the newest snapshot with trainer state), the league (replayed from the journal) and the snapshot pool; an SP4 run dir is continued in place (its journal migrated to schema 2, its counters from `metrics.jsonl` and the journal, its learners from its snapshots). `training.resume_from` keeps its SP3 / SP4 child-run meaning.

**Files:**
- Create: `src/colosseum/hub/resume.py`
- Modify: `src/colosseum/arena/matchlog.py` (schema 2, migration, `node`, monotonic row ids, the reader of both versions, `results_after`, `folded_after`, `last_row`, `max_env_steps`)
- Modify: `src/colosseum/arena/retention.py` (only the newest final of the current run is protected)
- Modify: `src/colosseum/coordinator/coordinator.py` (`LeagueStateInfo`, `league_replay_window`, `loaded_league_state`, `league_state_journal_row`, `replay_results`; `save_league_state(..., journal_row=)` if T2.5 has not added it)
- Modify: `src/colosseum/core/run_dir.py` (docstring: `config.resolved.<k>.yaml`, `hub_state.json`, `recovery/`, `hub.json`)
- Modify: `src/colosseum/hub/hub.py` (start / resume functions only: `run_training(resume=)`, `run_hub(resume=)`, `Hub._resume_hub_state`, `_load_hub_state`, `_replay_journal`, `_resume_start_sources`; the resumed hub's port)
- Modify: `src/colosseum/cli.py` (`train --resume RUN` with `-c` optional; `hub --resume RUN`)
- Modify (if missing): `tests/fault_proxy.py` (`TcpFaultProxy.retarget(port)`)
- Test: `tests/unit/test_sp5_matchlog_v2.py`, `tests/unit/test_sp5_league_replay.py`, `tests/unit/test_sp5_resume_plan.py`, `tests/unit/test_sp5_retention_resume.py`, `tests/integration/test_sp5_resume_in_place.py`, `tests/integration/test_sp5_hub_restart_slow.py`

**Interfaces:**
- Consumes: `HubStateData`, `load_hub_state`, `HUB_STATE_FILE`, `BudgetCounter(base, sessions)`, `EpochAllocator(reserved_until, on_reserve)`, `VersionRegistry.from_state` (T2.1); `Hub`, `HubOptions(resume=...)`, `setup_run`, `validate_run_config`, `_record_run_state`, `read_hub_json` / `write_hub_json`, `MatchLogWriter` with commits and `league_state.json` `journal_row` (T2.5); `LearnerTable.load_resumed` (T4.1); `RecoveryStore` and the start-source rule (T4.2); `copy_sp4_run`, `SP4_RUN`, `sp4_manifest()` (T0.1; this task reads only `env_steps` and `final_version` from the manifest and everything else from the copied files); `resolve_resume`, `check_model_state`, `build_model`, `role_signature`.
- Produces (contract plus additions marked *):
  - `SCHEMA_VERSION = 2` and the `matches.node` column (contract); *`READABLE_SCHEMA_VERSIONS = (1, 2)`; *`MatchRow.node`; *`meta.next_match_id`; *`MatchLogReader.results_after(row, *, source="train", min_env_steps=None, batch=READ_BATCH)`, *`MatchLogReader.folded_after(row)`, *`last_row()`, *`max_env_steps(source="train")`;
  - `Coordinator.league_state_journal_row()` (contract); *`LeagueStateInfo`, *`league_replay_window(info)`, *`Coordinator.loaded_league_state`, *`Coordinator.replay_results(results) -> (replayed, skipped)`;
  - *`colosseum.hub.resume`: `ResumePlan`, `prepare_resume`, `commit_resume`, `check_not_running`, `check_resumed_agents`, `config_diff`, `flatten`, `next_backup_path`, `start_env_steps`, `resume_port`, `AGENTS_STATE_FILE`, `DIFF_LINES_MAX`;
  - `run_training(..., resume=RUN)`, `run_hub(..., resume=RUN)` (contract), CLI `train --resume RUN [-c CFG] [--set ...]`, `hub --resume RUN [-c CFG] [--port P] [--set ...]`;
  - *retention: one protected final per agent and run id (the newest).

- [ ] **Step 1: Write the failing library tests**

Create `tests/unit/test_sp5_matchlog_v2.py`:
```python
"""The journal's schema version 2 (SP5 T4.3; spec 4.3 and 5.4): the ``node`` column, the in-place migration of an
SP4 journal (also when a writer opens it again, and never touching a parent read by ``import_parent``), row ids
that are never reused (the league-state watermark stays exact after compaction), the reader of both versions, and
``results_after`` (the league replay of an in-place resume)."""
from __future__ import annotations

import dataclasses
import shutil
import sqlite3

import pytest

from colosseum.arena.matchlog import (
    MATCH_LOG_FILE,
    SCHEMA_VERSION,
    MatchLog,
    MatchLogError,
    MatchLogReader,
    RunInfo,
)
from game_helpers import copy_sp4_run, make_match_result, synthetic_journal_results

RUN = RunInfo(run_id="R", name="run", parent_run_id=None, game="ttt", config_hash="h", checkpoint_interval=10)
FREE = 10**12


def _sql(path, query, args=()):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(query, args).fetchall()
    finally:
        conn.close()


def _v1_journal(tmp_path):
    """An SP4 journal (schema version 1) written by SP4 code: the T0.1 run-dir fixture's."""
    path = copy_sp4_run(tmp_path) / MATCH_LOG_FILE
    assert _sql(path, "SELECT value FROM meta WHERE key = 'schema_version'") == [("1",)]
    return path


def test_new_journals_have_version_2_with_a_node_column(tmp_path):
    log = MatchLog(tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: FREE)
    results = synthetic_journal_results(3)
    results[1] = dataclasses.replace(results[1], node=4)
    log.append(results)
    log.close()
    path = tmp_path / MATCH_LOG_FILE
    assert SCHEMA_VERSION == 2
    assert _sql(path, "SELECT value FROM meta WHERE key = 'schema_version'") == [("2",)]
    assert _sql(path, "SELECT id, node FROM matches ORDER BY id") == [(1, None), (2, 4), (3, None)]
    with MatchLogReader(path) as reader:
        rows = reader.matches(limit=None)
        assert [r.node for r in reversed(rows)] == [None, 4, None]
        assert rows[1].to_dict()["node"] == 4 and rows[1].to_result()[0].node == 4


def test_an_sp4_journal_is_migrated_in_place_once(tmp_path, caplog):
    path = _v1_journal(tmp_path)
    before = _sql(path, "SELECT id, uid FROM matches ORDER BY id")
    top = before[-1][0]
    with caplog.at_level("INFO", logger="colosseum.arena.matchlog"):
        MatchLog(path, RUN, disk_free_fn=lambda _: FREE).close()
    assert "schema version 1 migrated to 2" in caplog.text
    assert _sql(path, "SELECT value FROM meta WHERE key = 'schema_version'") == [("2",)]
    assert _sql(path, "SELECT value FROM meta WHERE key = 'next_match_id'") == [(str(top + 1),)]
    assert _sql(path, "SELECT id, uid FROM matches ORDER BY id") == before
    assert _sql(path, "SELECT COUNT(*) FROM matches WHERE node IS NOT NULL") == [(0,)]
    caplog.clear()
    with caplog.at_level("INFO", logger="colosseum.arena.matchlog"):
        MatchLog(path, RUN, disk_free_fn=lambda _: FREE).close()          # a second open migrates nothing
    assert "migrated" not in caplog.text


def test_the_reader_reads_both_versions_and_refuses_others(tmp_path):
    path = _v1_journal(tmp_path)
    with MatchLogReader(path) as reader:                                  # read-only: no migration
        rows = reader.matches(limit=5)
        assert rows and all(r.node is None for r in rows)
    assert _sql(path, "SELECT value FROM meta WHERE key = 'schema_version'") == [("1",)]
    other = tmp_path / "other.db"
    shutil.copy(path, other)
    conn = sqlite3.connect(other)
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(MatchLogError, match="schema version 99"):
        MatchLogReader(other)
    with pytest.raises(MatchLogError, match="schema version 99"):
        MatchLog(other, RUN, disk_free_fn=lambda _: FREE)


def test_row_ids_are_never_reused_after_compaction_deleted_the_newest(tmp_path):
    """The R8 watermark (league_state.json journal_row) means 'every row up to here is accounted': an id must
    never come back, even when compaction folded the newest rows (matches of an evicted snapshot)."""
    log = MatchLog(tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: FREE)
    log.append(synthetic_journal_results(4))
    newest = make_match_result([(0, "main", "latest", "player", 31), (1, "main", "ckpt_v30")], uid="R/train/x",
                               env_steps=99)
    log.append([newest])
    assert log.last_row() == 5
    log.compact(window_bytes=10**9, evicted=["main@v30"])                 # folds the newest row into counts
    assert log.last_row() == 4
    log.append([make_match_result([(0, "main", "latest"), (1, "random", "fixed")], uid="R/train/y")])
    assert log.last_row() == 6                                             # not 5 again
    log.close()


def test_import_of_an_sp4_parent_leaves_the_parent_untouched(tmp_path):
    parent = _v1_journal(tmp_path)
    child = MatchLog(tmp_path / "child.db", dataclasses.replace(RUN, run_id="C", parent_run_id="P"),
                     disk_free_fn=lambda _: FREE)
    stats = child.import_parent(parent)
    child.append([make_match_result([(0, "agent_0", "latest"), (1, "agent_0", "ckpt_v20")], uid="C/train/z")])
    child.close()
    assert stats.folded > 0
    assert _sql(parent, "SELECT value FROM meta WHERE key = 'schema_version'") == [("1",)]
    assert _sql(tmp_path / "child.db", "SELECT value FROM meta WHERE key = 'schema_version'") == [("2",)]
    ids = [r[0] for r in _sql(tmp_path / "child.db", "SELECT id FROM matches ORDER BY id")]
    assert ids == sorted(set(ids))


def test_results_after_returns_the_unaccounted_rows_in_commit_order(tmp_path):
    log = MatchLog(tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: FREE)
    train = synthetic_journal_results(6)
    log.append(train[:3])
    log.append([make_match_result([(0, "main", "ckpt_v10"), (1, "random", "fixed")], uid="R/pulse/p0",
                                  source="pulse", block="b0")])
    log.append(train[3:], tournament=None)
    log.append([make_match_result([(0, "a#1", "latest"), (1, "b#2", "latest")], uid="T/x", source="tournament",
                                  block="t0")], tournament="t")
    log.close()
    with MatchLogReader(tmp_path / MATCH_LOG_FILE) as reader:
        rows = list(reader.results_after(2))
        assert [r for r, _ in rows] == [3, 5, 6, 7]                         # train rows only, ascending
        assert [res.uid for _, res in rows] == [r.uid for r in train[2:]]
        assert all(res.source == "train" for _, res in rows)
        assert [r for r, _ in reader.results_after(None, batch=2)] == [1, 2, 3, 5, 6, 7]
        steps = sorted(r.env_steps for r in train)
        assert [res.env_steps for _, res in reader.results_after(None, min_env_steps=steps[3])] == steps[4:]
        assert list(reader.results_after(7)) == []
        assert reader.max_env_steps() == steps[-1]
```

Create `tests/unit/test_sp5_league_replay.py`:
```python
"""The league replay of an in-place resume (SP5 T4.3, R8, spec 5.1 / 5.4): ``league_state.json``'s journal
watermark, the replay window for SP5 / SP4 / missing league states, and exactly-once accounting: a coordinator
that loads the state and replays the journal rows after the watermark equals one that accounted every result."""
from __future__ import annotations

import json

from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog, MatchLogReader, RunInfo
from colosseum.coordinator.coordinator import LEAGUE_STATE_FILE, LeagueStateInfo, league_replay_window
from game_helpers import make_coordinator, make_match_result, make_test_config, scripted_agent

RUN = RunInfo(run_id="R", name="run", parent_run_id=None, game="turns", config_hash="h", checkpoint_interval=10)


def _config():
    return make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent()})


def _results(n: int = 100):
    out = []
    for k in range(n):
        other = ("rnd", "fixed") if k % 3 else ("agent_0", "ckpt_v10")
        ranks = [1.0, 2.0] if k % 4 else ([2.0, 1.0] if k % 8 else [1.0, 1.0])
        out.append(make_match_result([(0, "agent_0", "latest", "player", k), (1, *other)], ranks,
                                     uid=f"R/train/n1/w0_e0_g1_ep{k}", env_steps=10 * (k + 1)))
    return out


def _journal(tmp_path, results):
    log = MatchLog(tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: 10**12)
    for k in range(0, len(results), 7):
        log.append(results[k:k + 7])
    log.close()
    return tmp_path / MATCH_LOG_FILE


def _replayed(tmp_path, config, state_path):
    fresh = make_coordinator(config, tmp_path / "fresh")
    if state_path is not None:
        assert fresh.load_league_state(state_path)
    after, min_steps = league_replay_window(fresh.loaded_league_state)
    with MatchLogReader(tmp_path / MATCH_LOG_FILE) as reader:
        replayed, skipped = fresh.replay_results(res for _, res in reader.results_after(after,
                                                                                        min_env_steps=min_steps))
    return fresh, replayed, skipped


def test_the_watermark_is_saved_and_loaded(tmp_path):
    coordinator = make_coordinator(_config(), tmp_path / "ckpt")
    path = tmp_path / LEAGUE_STATE_FILE
    coordinator.save_league_state(path, env_steps=5, journal_row=42)
    assert json.loads(path.read_text())["journal_row"] == 42
    fresh = make_coordinator(_config(), tmp_path / "ckpt2")
    assert fresh.loaded_league_state is None and fresh.league_state_journal_row() is None
    fresh.load_league_state(path)
    assert fresh.loaded_league_state == LeagueStateInfo(env_steps=5, journal_row=42, has_watermark=True)
    assert fresh.league_state_journal_row() == 42


def test_the_replay_window():
    assert league_replay_window(None) == (None, None)                                   # no state: every row
    assert league_replay_window(LeagueStateInfo(7, 42, True)) == (42, None)             # SP5: after the watermark
    assert league_replay_window(LeagueStateInfo(7, None, True)) == (None, None)         # SP5, nothing accounted
    assert league_replay_window(LeagueStateInfo(700, None, False)) == (None, 700)       # SP4: after its env steps


def test_replay_after_the_watermark_is_exactly_once(tmp_path):
    config, results = _config(), _results()
    _journal(tmp_path, results)
    full = make_coordinator(config, tmp_path / "full")
    for result in results[:60]:
        full.report_match_result(result)
    state = tmp_path / LEAGUE_STATE_FILE
    full.save_league_state(state, env_steps=600, journal_row=60)      # rows 1..60 accounted
    for result in results[60:]:                                        # committed, accounted, then the hub died
        full.report_match_result(result)
    fresh, replayed, skipped = _replayed(tmp_path, config, state)
    assert (replayed, skipped) == (40, 0)
    assert fresh.ratings_snapshot() == full.ratings_snapshot()


def test_an_sp4_league_state_replays_by_env_steps(tmp_path):
    config, results = _config(), _results()
    _journal(tmp_path, results)
    full = make_coordinator(config, tmp_path / "full")
    for result in results[:60]:
        full.report_match_result(result)
    state = tmp_path / LEAGUE_STATE_FILE
    full.save_league_state(state, env_steps=results[59].env_steps)
    data = json.loads(state.read_text())
    del data["journal_row"]                                            # as SP4 wrote it
    state.write_text(json.dumps(data))
    for result in results[60:]:
        full.report_match_result(result)
    fresh, replayed, _ = _replayed(tmp_path, config, state)
    assert replayed == 40 and fresh.ratings_snapshot() == full.ratings_snapshot()


def test_without_a_league_state_every_detailed_match_is_replayed(tmp_path):
    config, results = _config(), _results()
    _journal(tmp_path, results)
    full = make_coordinator(config, tmp_path / "full")
    for result in results:
        full.report_match_result(result)
    fresh, replayed, _ = _replayed(tmp_path, config, None)
    assert replayed == 100 and fresh.ratings_snapshot() == full.ratings_snapshot()


def test_folded_rows_after_the_watermark_are_counted(tmp_path):
    results = _results()
    log = MatchLog(tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: 10**12)
    log.append(results)
    log.compact(window_bytes=0, max_batch=10)                           # every training match folded
    log.close()
    with MatchLogReader(tmp_path / MATCH_LOG_FILE) as reader:
        assert list(reader.results_after(60)) == []
        assert reader.folded_after(60) == 40 and reader.folded_after(None) == 100


def test_a_result_the_game_cannot_rate_is_skipped(tmp_path, caplog):
    config = _config()
    fresh = make_coordinator(config, tmp_path / "c")
    bad = make_match_result([(0, "agent_0", "latest"), (1, "rnd", "fixed")], layout="gone", uid="x")
    with caplog.at_level("WARNING"):
        assert fresh.replay_results([bad, _results(1)[0]]) == (1, 1)
    assert "cannot be replayed" in caplog.text
```

Create `tests/unit/test_sp5_resume_plan.py`:
```python
"""In-place resume preparation (SP5 T4.3, spec 5.4): refusal while the run's process lives, the saved config or a
new one with a diff WARNING, ``--set`` in both cases, the run dir pinned, ``training.resume_from`` not applied
again, the start env-step count of an SP4 run dir, the at-budget WARNING, the numbered backup of the previous
resolved config."""
from __future__ import annotations

import json
import logging
import os

import pytest
import yaml

from colosseum.core.errors import ConfigError
from colosseum.core.run_state import write_run_state
from colosseum.hub.resume import (
    check_not_running,
    commit_resume,
    config_diff,
    flatten,
    next_backup_path,
    prepare_resume,
    resume_port,
    start_env_steps,
)
from game_helpers import copy_sp4_run


def _saved(root) -> dict:
    return yaml.safe_load((root / "config.resolved.yaml").read_text())


def test_refused_while_the_run_process_lives_and_allowed_after(tmp_path):
    root = copy_sp4_run(tmp_path)
    write_run_state(root, "running", os.getpid())                  # this test process: alive
    with pytest.raises(ConfigError, match=f"is running \\(pid {os.getpid()}\\)"):
        check_not_running(root)
    with pytest.raises(ConfigError, match="two hubs cannot share one run dir"):
        prepare_resume(root, None)
    write_run_state(root, "running", 2**22 + 12345)                # a dead pid: a killed run
    check_not_running(root)
    write_run_state(root, "finished", os.getpid(), exit_code=0)
    check_not_running(root)


def test_without_c_the_saved_config_with_set_and_the_dir_pinned(tmp_path, caplog):
    root = copy_sp4_run(tmp_path)
    moved = tmp_path / "elsewhere"
    root.rename(moved)                                              # a run dir copied to another machine / path
    with caplog.at_level(logging.WARNING):
        plan = prepare_resume(moved, None, {"training.total_timesteps": 10**7})
    assert plan.config.run.name == "elsewhere" and plan.config.run.dir == str(tmp_path)
    assert plan.config.training.total_timesteps == 10**7
    assert plan.diff == [] and not plan.from_file and not plan.sp5
    assert plan.run_id == json.loads((moved / "run.json").read_text())["run_id"]
    assert "differs" not in caplog.text and "names run dir" not in caplog.text


def test_with_c_the_new_config_and_a_diff_warning(tmp_path, caplog):
    root = copy_sp4_run(tmp_path)
    data = _saved(root)
    old_chunks = data["learner"]["batch_chunks"]
    data["learner"]["batch_chunks"] = old_chunks + 5
    data["run"]["name"] = "other"
    path = tmp_path / "new.yaml"
    path.write_text(yaml.safe_dump(data))
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.resume"):
        plan = prepare_resume(root, path, {"training.total_timesteps": 10**7})
    assert plan.from_file and plan.config.learner.batch_chunks == old_chunks + 5
    assert plan.config.run.name == root.name
    assert plan.diff == [f"learner.batch_chunks: {old_chunks!r} -> {old_chunks + 5!r}"]
    assert "differs from the run's saved config in 1 settings" in caplog.text
    assert "the resumed run stays in" in caplog.text


def test_resume_from_is_not_applied_again(tmp_path, caplog):
    root = copy_sp4_run(tmp_path)
    data = _saved(root)
    data["training"]["resume_from"] = str(tmp_path / "parent")
    (root / "config.resolved.yaml").write_text(yaml.safe_dump(data))
    with caplog.at_level(logging.INFO, logger="colosseum.hub.resume"):
        prepare_resume(root, None, {"training.total_timesteps": 10**7})
    assert "was imported when the run started" in caplog.text
    data["training"]["resume_from"] = str(tmp_path / "another")
    path = tmp_path / "new.yaml"
    path.write_text(yaml.safe_dump(data))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.resume"):
        prepare_resume(root, path, {"training.total_timesteps": 10**7})
    assert "is ignored: an in-place resume continues the run's own state" in caplog.text


def test_sp4_start_env_steps_and_the_budget_warning(tmp_path, caplog):
    root = copy_sp4_run(tmp_path)
    steps = start_env_steps(root)
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert steps == max(r["env_steps"] for r in records if "env_steps" in r) > 0
    with open(root / "metrics.jsonl", "a") as fh:
        fh.write('{"kind": "system", "env_st')                        # a torn last line of a killed run
    assert start_env_steps(root) == steps
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.resume"):
        plan = prepare_resume(root, None)
    assert plan.start_env_steps == steps
    assert "at or past training.total_timesteps" in caplog.text


def test_hub_state_counts_win_and_a_malformed_one_is_refused(tmp_path):
    root = copy_sp4_run(tmp_path)
    (root / "hub_state.json").write_text(json.dumps({"version": 1, "env_steps": 123}))
    plan = prepare_resume(root, None, {"training.total_timesteps": 10**7})
    assert plan.sp5 and plan.start_env_steps == 123
    (root / "hub_state.json").write_text("{")
    with pytest.raises(ConfigError, match="hub_state.json"):
        prepare_resume(root, None)


def test_not_a_run_dir(tmp_path):
    with pytest.raises(ConfigError, match="not a run dir"):
        prepare_resume(tmp_path / "missing", None)
    (tmp_path / "x" / "checkpoints").mkdir(parents=True)
    with pytest.raises(ConfigError, match="has no config.resolved.yaml"):
        prepare_resume(tmp_path / "x", None)


def test_commit_keeps_numbered_backups(tmp_path):
    root = copy_sp4_run(tmp_path)
    original = (root / "config.resolved.yaml").read_text()
    plan = prepare_resume(root, None, {"training.total_timesteps": 10**7})
    assert next_backup_path(root).name == "config.resolved.1.yaml"
    assert commit_resume(plan) == root / "config.resolved.1.yaml"
    assert (root / "config.resolved.1.yaml").read_text() == original
    assert _saved(root)["training"]["total_timesteps"] == 10**7
    plan = prepare_resume(root, None, {"training.total_timesteps": 2 * 10**7})
    assert commit_resume(plan).name == "config.resolved.2.yaml"


def test_flatten_and_diff_helpers(tmp_path):
    assert flatten({"a": {"b": 1, "c": {"d": [1]}}, "e": {}}) == {"a.b": 1, "a.c.d": [1], "e": {}}
    root = copy_sp4_run(tmp_path)
    a = prepare_resume(root, None, {"training.total_timesteps": 10**7}).config
    b = a.model_copy(update={"run": a.run.model_copy(update={"name": "zzz"})})
    assert config_diff(a, b) == []                                   # run.dir / run.name are pinned
    c = a.model_copy(update={"networks": a.networks.model_copy(update={"kwargs": {"hidden": 7}})})
    assert config_diff(a, c) == ["networks.kwargs.hidden: added (7)"]


def test_a_resumed_hub_listens_on_its_previous_port():
    assert resume_port(["10.0.0.5:47123", "192.168.1.2:47123"], None, 47100) == 47123
    assert resume_port(["10.0.0.5:47123"], 5000, 47100) == 5000                # --port wins
    assert resume_port([], None, 47100) == 47100                               # no hub.json: the config's
    assert resume_port(["a:1", "b:2"], None, 47100) == 47100                   # ambiguous: the config's
```
and append to it the strict-check test (it needs `setup_run`, which lives in `colosseum.hub.hub` after T2.5):
```python
def test_strict_checks_accept_the_saved_config_and_refuse_another_architecture(tmp_path):
    from colosseum.hub.hub import setup_run
    from colosseum.hub.resume import check_resumed_agents

    root = copy_sp4_run(tmp_path)
    plan = prepare_resume(root, None, {"training.total_timesteps": 10**7})
    setup = setup_run(plan.config, validate=False)
    checked = check_resumed_agents(root, setup.agent_configs, setup.role_specs)
    assert set(checked) == set(setup.agent_configs) and all("ckpt_v" in s for s in checked.values())
    data = _saved(root)
    data["networks"]["kwargs"] = {**(data["networks"].get("kwargs") or {}), "hidden": 7}
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(data))
    plan = prepare_resume(root, path, {"training.total_timesteps": 10**7})
    setup = setup_run(plan.config, validate=False)
    with pytest.raises(ConfigError, match="cannot continue with this config"):
        check_resumed_agents(root, setup.agent_configs, setup.role_specs)
```
(If the T0.1 fixture's model class takes no `hidden` keyword, change another architecture keyword of its `networks.kwargs`; the point is a weight-shape mismatch.)

Create `tests/unit/test_sp5_retention_resume.py`:
```python
"""Retention across in-place resumes (SP5 T4.3): the run id stays, so only the NEWEST final snapshot of the current
run is protected; earlier finals of the same run are ordinary snapshots (every stop / resume cycle would otherwise
add one more protected snapshot). Without a run id the SP3 rule (every final) stays."""
from __future__ import annotations

from colosseum.arena.retention import RetentionPolicy, SnapshotInfo, retain

RUN = "run-now"


def _policy(**kw) -> RetentionPolicy:
    base = dict(keep_last=2, keep_every_period=0, keep_top_k=0, top_k_min_games=0, max_snapshots=None,
                current_run_id=RUN)
    return RetentionPolicy(**{**base, **kw})


def _kept(decision) -> list[int]:
    return sorted(int(c.removeprefix("ckpt_v")) for c in decision.keep)


def test_only_the_newest_final_of_the_current_run_is_protected():
    entries = [SnapshotInfo("ckpt_v11", 11, True, RUN, False), SnapshotInfo("ckpt_v31", 31, True, RUN, False),
               SnapshotInfo("ckpt_v7", 7, True, "parent", False),
               *[SnapshotInfo(f"ckpt_v{v}", v, False, RUN, False) for v in (40, 50)]]
    decision = retain(entries, {}, _policy())
    assert _kept(decision) == [31, 40, 50]
    assert "final" in decision.reasons["ckpt_v31"] and decision.reasons["ckpt_v11"].startswith("evicted")
    assert _kept(retain(entries, {}, _policy(current_run_id=None))) == [7, 11, 31, 40, 50]


def test_a_resumed_run_keeps_its_previous_final_until_the_next_one():
    entries = [SnapshotInfo("ckpt_v31", 31, True, RUN, False),
               *[SnapshotInfo(f"ckpt_v{v}", v, False, RUN, False) for v in (40, 50, 60)]]
    assert _kept(retain(entries, {}, _policy())) == [31, 50, 60]
    entries.append(SnapshotInfo("ckpt_v65", 65, True, RUN, False))
    assert _kept(retain(entries, {}, _policy())) == [60, 65]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_matchlog_v2.py tests/unit/test_sp5_league_replay.py tests/unit/test_sp5_resume_plan.py tests/unit/test_sp5_retention_resume.py -q`
Expected: `assert SCHEMA_VERSION == 2` fails, `ImportError: cannot import name 'LeagueStateInfo'`, `ModuleNotFoundError: No module named 'colosseum.hub.resume'`, and `test_only_the_newest_final_of_the_current_run_is_protected` fails with `[11, 31, 40, 50]`.

- [ ] **Step 3: The journal (`src/colosseum/arena/matchlog.py`)**

Apply this change (made against `d61eb02`'s file; T2.5's additions to the module — `AppendStats.rows`, `MatchLogWriter.submit(..., token=)`, `take_commits` — are independent of it: where T2.5 records `matches.id` per result, it now records `next_id` instead of `cur.lastrowid`). Also update the module docstring: "schema version 2 (SP5): `matches.node` (the node that played the match; NULL for SP4 rows and local tournaments); a version-1 journal is migrated in place when a writer opens it and read as it is by `MatchLogReader`; row ids come from `meta.next_match_id` and are never reused, so `league_state.json`'s `journal_row` watermark stays exact after compaction deleted the newest rows".
```diff
--- a/src/colosseum/arena/matchlog.py
+++ b/src/colosseum/arena/matchlog.py
@@ -69,7 +69,10 @@
 logger = logging.getLogger(__name__)
 
 MATCH_LOG_FILE = "matches.db"
-SCHEMA_VERSION = 1
+SCHEMA_VERSION = 2
+# Schema versions this code opens: version 1 (SP4) is migrated in place by a writer (``MatchLog``) and read as is
+# by ``MatchLogReader`` (``node`` reads as None).
+READABLE_SCHEMA_VERSIONS = (1, 2)
 # The sources of a run's own ratings (live ratings, top-k, the report's ratings): tournaments rate only their own
 # matches (spec block 2). Defined here once (contract amendment A18).
 RUN_SOURCES = ("train", "pulse")
@@ -103,7 +106,7 @@
     "CREATE TABLE IF NOT EXISTS matches (id INTEGER PRIMARY KEY, uid TEXT NOT NULL UNIQUE, run INTEGER, "
     "source INTEGER NOT NULL, layout INTEGER NOT NULL, outcome_kind TEXT NOT NULL, length INTEGER NOT NULL, "
     "seed INTEGER, env_steps INTEGER, block TEXT, tournament INTEGER, worker INTEGER, ts REAL NOT NULL, "
-    "replay TEXT, size INTEGER NOT NULL)",
+    "replay TEXT, size INTEGER NOT NULL, node INTEGER)",
     "CREATE INDEX IF NOT EXISTS matches_by_source ON matches (source, id)",
     "CREATE INDEX IF NOT EXISTS matches_by_tournament ON matches (tournament, block) WHERE tournament IS NOT NULL",
     "CREATE TABLE IF NOT EXISTS seats (match INTEGER NOT NULL, seat INTEGER NOT NULL, team INTEGER NOT NULL, "
@@ -124,7 +127,9 @@
     "s2 = s2 + excluded.s2"
 )
 _MATCH_COLUMNS = ("uid, run, source, layout, outcome_kind, length, seed, env_steps, block, tournament, worker, "
-                  "ts, replay, size")
+                  "ts, replay, size, node")
+# The same columns of a version-1 journal (an SP4 parent read by ``import_parent``): no ``node``.
+_MATCH_COLUMNS_V1 = _MATCH_COLUMNS.replace(", node", ", NULL")
 _SEAT_COLUMNS = ("seat, team, role, agent, network, entity, version_start, version_end, rating_version, reward, "
                  "eliminated_step, category")
 
@@ -214,6 +219,7 @@
     seats: tuple[SeatRow, ...]
     teams: tuple[tuple[int, float, float], ...]     # (team, rank, score)
     replay: str | None = None                        # a replay link, when the env gave one (spec block 5)
+    node: int | None = None                          # SP5: the node that played it (None: SP4 rows, local tournaments)
 
     def to_dict(self) -> dict:
         """JSON-ready form (report, dashboard; contract amendment A7): the match's columns (no ``ts``: order is by
@@ -221,7 +227,7 @@
         return {"uid": self.uid, "run_id": self.run_id, "source": self.source, "layout": self.layout,
                 "outcome_kind": self.outcome_kind, "episode_length": self.length, "seed": self.seed,
                 "env_steps": self.env_steps, "block": self.block, "tournament": self.tournament,
-                "worker": self.worker, "replay": self.replay,
+                "worker": self.worker, "node": self.node, "replay": self.replay,
                 "seats": [{"seat": s.seat, "team": s.team, "role": s.role, "agent_id": s.agent_id,
                            "network_id": s.network_id, "entity": s.entity, "category": s.category,
                            "version_start": s.version_start, "version_end": s.version_end,
@@ -238,7 +244,7 @@
         result = MatchResult(match_id=self.uid, layout=self.layout, outcome_kind=self.outcome_kind, seats=seats,
                              teams=[TeamResult(team=t, rank=r, score=sc) for t, r, sc in self.teams],
                              episode_length=self.length, uid=self.uid, source=self.source, seed=self.seed,
-                             env_steps=self.env_steps, block=self.block)
+                             env_steps=self.env_steps, block=self.block, node=self.node)
         return result, [s.rating_version for s in self.seats]
 
 
@@ -330,6 +336,7 @@
         self._conn = conn
         self._ids: dict[tuple[str, str], int] = {}
         self._names: dict[int, str] = {}
+        self._schema = SCHEMA_VERSION         # the file's schema version (a reader of a version-1 file: 1)
 
     def _lookup(self, kind: str, name: str) -> int | None:
         key = (kind, name)
@@ -414,6 +421,19 @@
     def _generation(self) -> int:
         return int(self._meta("compact_generation", "0"))
 
+    def last_row(self) -> int | None:
+        """The highest row id of ``matches`` (None when empty)."""
+        found = self._conn.execute("SELECT MAX(id) FROM matches").fetchone()[0]
+        return None if found is None else int(found)
+
+    def max_env_steps(self, source: str = "train") -> int | None:
+        """The largest ``env_steps`` of the detailed matches of ``source`` (SP4 run dirs resumed in place)."""
+        sid = self._lookup("source", source)
+        if sid is None:
+            return None
+        found = self._conn.execute("SELECT MAX(env_steps) FROM matches WHERE source = ?", (sid,)).fetchone()[0]
+        return None if found is None else int(found)
+
     def _consistent(self, read: Callable[[], object], what: str):
         """Run ``read`` (several queries, each its own transaction) until no compaction committed during it
         (``compact_generation`` unchanged): a compaction moves matches from detail to counts, so a read across one
@@ -573,15 +593,39 @@
             if row is None:
                 conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)",
                                  [("schema_version", str(SCHEMA_VERSION)), ("detail_train_bytes", "0"),
-                                  ("detail_train_matches", "0"), ("compact_generation", "0")])
+                                  ("detail_train_matches", "0"), ("compact_generation", "0"),
+                                  ("next_match_id", "1")])
+            elif int(row[0]) == 1:
+                self._migrate_v1(conn)
             elif int(row[0]) != SCHEMA_VERSION:
                 raise MatchLogError(f"match log {self.path} has schema version {row[0]}; this Colosseum reads "
-                                    f"version {SCHEMA_VERSION} (move the file away to start a new journal)")
+                                    f"versions {READABLE_SCHEMA_VERSIONS} (move the file away to start a new journal)")
             conn.execute("COMMIT")
         except BaseException:
             _rollback(conn)
             raise
 
+    def _migrate_v1(self, conn: sqlite3.Connection) -> None:
+        """Version 1 (SP4) -> 2 in place, inside the caller's transaction: the ``node`` column (NULL for every SP4
+        row) and ``next_match_id`` (row ids are never reused, so ``league_state.json``'s journal watermark stays
+        exact after compaction deleted the newest rows; SP5 T4.3)."""
+        conn.execute("ALTER TABLE matches ADD COLUMN node INTEGER")
+        top = conn.execute("SELECT MAX(id) FROM matches").fetchone()[0]
+        conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),))
+        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('next_match_id', ?)", (str((top or 0) + 1),))
+        logger.info(f"Match log {self.path}: schema version 1 migrated to {SCHEMA_VERSION} in place (column node, "
+                    f"monotonic row ids)")
+
+    def _take_ids(self) -> int:
+        """The next row id of ``matches`` (inside a transaction): ``meta.next_match_id``, never below the highest
+        row id + 1."""
+        stored = int(self._meta("next_match_id", "1"))
+        top = self.last_row()
+        return max(stored, (top or 0) + 1)
+
+    def _store_next_id(self, next_id: int) -> None:
+        self._conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('next_match_id', ?)", (str(next_id),))
+
     def _register_run(self, run: RunInfo) -> int:
         def register() -> int:
             self._conn.execute(
@@ -677,6 +721,7 @@
 
         def body() -> None:
             tid = self._id("tournament", tournament) if tournament is not None else None
+            next_id = self._take_ids()
             train_bytes = train_matches = 0
             for result in results:
                 versions = self.rating_versions(result)
@@ -692,16 +737,18 @@
                     continue
                 size = estimated_size(result)
                 cur = self._conn.execute(
-                    f"INSERT OR IGNORE INTO matches ({_MATCH_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
-                    f"NULL, ?)",
-                    (result.uid, self._run_row, self._id("source", result.source), self._id("layout", result.layout),
-                     result.outcome_kind, int(result.episode_length), result.seed, result.env_steps, result.block,
-                     tid, _worker_of(result.match_id), time.time(), size))
+                    f"INSERT OR IGNORE INTO matches (id, {_MATCH_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
+                    f"?, ?, NULL, ?, ?)",
+                    (next_id, result.uid, self._run_row, self._id("source", result.source),
+                     self._id("layout", result.layout), result.outcome_kind, int(result.episode_length), result.seed,
+                     result.env_steps, result.block, tid, _worker_of(result.match_id), time.time(), size,
+                     result.node))
                 if cur.rowcount == 0:
                     stats.duplicates += 1
                     stats.accepted.append(False)
                     continue
-                match = int(cur.lastrowid)
+                match = next_id
+                next_id += 1
                 self._conn.executemany(
                     f"INSERT INTO seats (match, {_SEAT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     [(match, s.seat, s.team, self._id("role", s.role), self._id("agent", s.agent_id),
@@ -717,6 +764,7 @@
                 stats.accepted.append(True)
             if train_matches:
                 self._bump_detail(train_bytes, train_matches)
+            self._store_next_id(next_id)
 
         self._transaction(body)
         return stats
@@ -862,30 +910,34 @@
         where = "1" if train is None else "(source != ? OR tournament IS NOT NULL)"
         args: tuple = () if train is None else (train,)
         run_ids = {r[0]: r[1] for r in parent._conn.execute("SELECT id, run_id FROM runs")}
+        columns = _MATCH_COLUMNS if parent._schema >= 2 else _MATCH_COLUMNS_V1
+        next_id = self._take_ids()
         copied = last = 0
         while True:
             heads = parent._conn.execute(
-                f"SELECT id, {_MATCH_COLUMNS} FROM matches WHERE id > ? AND {where} ORDER BY id LIMIT ?",
+                f"SELECT id, {columns} FROM matches WHERE id > ? AND {where} ORDER BY id LIMIT ?",
                 (last, *args, READ_BATCH)).fetchall()
             if not heads:
+                self._store_next_id(next_id)
                 return copied
             last = heads[-1][0]
             for (pid, uid, run, source, layout, kind, length, seed, env_steps, block, tour, worker, ts, replay,
-                 size) in heads:
+                 size, node) in heads:
                 run_row = None
                 if run in run_ids:
                     row = self._conn.execute("SELECT id FROM runs WHERE run_id = ?", (run_ids[run],)).fetchone()
                     run_row = None if row is None else int(row[0])
                 tid = None if tour is None else self._id("tournament", parent._name(tour))
                 cur = self._conn.execute(
-                    f"INSERT OR IGNORE INTO matches ({_MATCH_COLUMNS}) "
-                    f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
-                    (uid, run_row, self._id("source", parent._name(source)),
+                    f"INSERT OR IGNORE INTO matches (id, {_MATCH_COLUMNS}) "
+                    f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
+                    (next_id, uid, run_row, self._id("source", parent._name(source)),
                      self._id("layout", parent._name(layout)), kind, length, seed, env_steps, block, tid, worker,
-                     ts, replay, size))
+                     ts, replay, size, node))
                 if cur.rowcount == 0:
                     continue
-                match = int(cur.lastrowid)
+                match = next_id
+                next_id += 1
                 seats = parent._conn.execute(f"SELECT {_SEAT_COLUMNS} FROM seats WHERE match = ?", (pid,)).fetchall()
                 self._conn.executemany(
                     f"INSERT INTO seats (match, {_SEAT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
@@ -937,9 +989,11 @@
         except sqlite3.DatabaseError as e:
             self._conn.close()
             raise MatchLogError(f"{self.path} is not a match log ({e})") from e
-        if version is None or int(version) != SCHEMA_VERSION:
+        if version is None or int(version) not in READABLE_SCHEMA_VERSIONS:
             self._conn.close()
-            raise MatchLogError(f"{self.path} is a match log of schema version {version}, not {SCHEMA_VERSION}")
+            raise MatchLogError(f"{self.path} is a match log of schema version {version}, not one of "
+                                f"{READABLE_SCHEMA_VERSIONS}")
+        self._schema = int(version)
 
     def __enter__(self) -> MatchLogReader:
         return self
@@ -1015,9 +1069,11 @@
         last: int | None = None
         page_size = READ_BATCH if limit is None else max(limit, 100)
         while limit is None or len(out) < limit:
+            node = "m.node" if self._schema >= 2 else "NULL"
             page = self._conn.execute(
                 f"SELECT m.id, m.uid, r.run_id, m.source, m.layout, m.outcome_kind, m.length, m.seed, m.env_steps, "
-                f"m.block, m.tournament, m.worker, m.ts, m.replay FROM matches m LEFT JOIN runs r ON r.id = m.run "
+                f"m.block, m.tournament, m.worker, m.ts, m.replay, {node} FROM matches m "
+                f"LEFT JOIN runs r ON r.id = m.run "
                 f"WHERE {where}{'' if last is None else ' AND m.id < ?'} ORDER BY m.id DESC LIMIT ?",
                 (*args, *(() if last is None else (last,)), page_size)).fetchall()
             if not page:
@@ -1032,6 +1088,40 @@
                     break
         return out
 
+    def results_after(self, row: int | None, *, source: str = "train", min_env_steps: int | None = None,
+                      batch: int = READ_BATCH) -> Iterator[tuple[int, MatchResult]]:
+        """``(row id, MatchResult)`` of every detailed match of ``source`` outside tournaments with a row id above
+        ``row`` (None: all) and, with ``min_env_steps``, ``env_steps`` above it, in row-id order (= commit order),
+        read in batches (SP5 T4.3: the league replay of an in-place resume). Matches already folded into
+        ``counts`` are not returned (``count_after`` tells how many rows are missing)."""
+        sid = self._lookup("source", source)
+        if sid is None:
+            return
+        last = -1 if row is None else int(row)
+        extra, args = ("", ()) if min_env_steps is None else (" AND env_steps > ?", (int(min_env_steps),))
+        node = "m.node" if self._schema >= 2 else "NULL"
+        while True:
+            heads = self._conn.execute(
+                f"SELECT m.id, m.uid, r.run_id, m.source, m.layout, m.outcome_kind, m.length, m.seed, m.env_steps, "
+                f"m.block, m.tournament, m.worker, m.ts, m.replay, {node} FROM matches m LEFT JOIN runs r "
+                f"ON r.id = m.run WHERE m.id > ? AND m.source = ? AND m.tournament IS NULL{extra} ORDER BY m.id "
+                f"LIMIT ?", (last, sid, *args, batch)).fetchall()
+            if not heads:
+                return
+            for head in heads:
+                yield int(head[0]), self._row(head).to_result()[0]
+            last = int(heads[-1][0])
+
+    def folded_after(self, row: int | None) -> int | None:
+        """How many journal rows above ``row`` (None: all) are no longer detailed (folded into ``counts`` by
+        compaction): they cannot be replayed. None for a version-1 journal (its row ids can be reused)."""
+        if self._schema < 2:
+            return None
+        assigned = int(self._meta("next_match_id", "1")) - 1 - (0 if row is None else int(row))
+        present = int(self._conn.execute("SELECT COUNT(*) FROM matches WHERE id > ?",
+                                         (-1 if row is None else int(row),)).fetchone()[0])
+        return max(0, assigned - present)
+
     def _row(self, row) -> MatchRow:
         mid = row[0]
         seats = tuple(SeatRow(
@@ -1044,7 +1134,7 @@
         return MatchRow(id=mid, uid=row[1], run_id=row[2], source=self._name(row[3]), layout=self._name(row[4]),
                         outcome_kind=row[5], length=int(row[6]), seed=row[7], env_steps=row[8], block=row[9],
                         tournament=self._name(row[10]), worker=row[11], ts=float(row[12]), seats=seats, teams=teams,
-                        replay=row[13])
+                        replay=row[13], node=row[14] if len(row) > 14 else None)
 
     def close(self) -> None:
         self._conn.close()
```

- [ ] **Step 4: The coordinator and retention (`src/colosseum/coordinator/coordinator.py`, `src/colosseum/arena/retention.py`)**

Coordinator (T2.5 adds `journal_row` to `save_league_state` and updates `test_sp4_league_state.py::test_the_coordinator_saves_and_loads_league_state`'s key set; if it already did, keep its version of that method and apply the rest):
```diff
--- a/src/colosseum/coordinator/coordinator.py
+++ b/src/colosseum/coordinator/coordinator.py
@@ -16,7 +16,8 @@
 import logging
 import random
 from collections import deque
-from collections.abc import Callable, Mapping, Sequence
+from collections.abc import Callable, Iterable, Mapping, Sequence
+from dataclasses import dataclass
 from pathlib import Path
 
 from colosseum.arena.retention import SnapshotScore, snapshot_scores
@@ -39,6 +40,27 @@
 LEAGUE_STATE_VERSION = 1
 
 
+@dataclass(frozen=True)
+class LeagueStateInfo:
+    """What a loaded ``league_state.json`` says about the journal (SP5 R8)."""
+
+    env_steps: int                  # the global env-step count it was written at
+    journal_row: int | None         # the highest journal row id accounted in it (None: none yet, or an SP4 file)
+    has_watermark: bool             # False: written before SP5 (no "journal_row" key)
+
+
+def league_replay_window(info: LeagueStateInfo | None) -> tuple[int | None, int | None]:
+    """``(after_row, min_env_steps)`` of the journal rows an in-place resume replays (SP5 T4.3): no league state ->
+    every detailed training match; an SP5 state -> the rows above its watermark; an SP4 state (no watermark: SP4
+    accounted at intake, before the journal) -> the matches stamped after the env-step count it was written at
+    (never one twice; a match stamped exactly at that count is not replayed)."""
+    if info is None:
+        return None, None
+    if info.has_watermark:
+        return info.journal_row, None
+    return None, info.env_steps
+
+
 class Coordinator:
     """Central coordinator of a single-machine training run."""
 
@@ -80,6 +102,7 @@
                                  for a, roles in self._agent_roles.items()}
         self._match_results: deque[MatchResult] = deque(maxlen=10000)
         self._refresh_round = 0
+        self._league_state: LeagueStateInfo | None = None
         validate_matchmaking(spec, self._player_roles, config)
         self._context = MatchmakerContext.from_config(
             config, spec, self._player_roles, rng=self._rng, snapshots_fn=self._checkpoint_ids,
@@ -240,17 +263,20 @@
             table["pfsp"] = pfsp.get(layout, {})
         return compact_ratings(snap) if compact else snap
 
-    def save_league_state(self, path: str | Path, *, env_steps: int) -> None:
+    def save_league_state(self, path: str | Path, *, env_steps: int, journal_row: int | None = None) -> None:
         """Write the ratings (``RatingBook``) and the PFSP statistics (with the forgotten snapshots) atomically to
-        ``path`` (spec block 7.4). A custom matchmaker's own state (``on_result``) is not saved."""
+        ``path`` (spec block 7.4). ``journal_row`` (SP5 R8): the highest journal row id accounted in them. A custom
+        matchmaker's own state (``on_result``) is not saved."""
         from colosseum.metrics.jsonl import write_json_atomic
 
         write_json_atomic(path, {"version": LEAGUE_STATE_VERSION, "env_steps": int(env_steps),
+                                 "journal_row": None if journal_row is None else int(journal_row),
                                  "ratings": self._ratings.state_dict(), "pfsp": self._pfsp.state_dict()})
 
     def load_league_state(self, path: str | Path) -> bool:
         """Restore a saved league state; False when ``path`` does not exist (an SP3 run dir, or a run that never saved
-        it). Entries of agents that are not in this config are dropped (INFO). An unreadable file is a ConfigError."""
+        it). Entries of agents that are not in this config are dropped (INFO). An unreadable file is a ConfigError.
+        What the file says about the journal is kept for the replay (``loaded_league_state``)."""
         import json
 
         path = Path(path)
@@ -263,11 +289,43 @@
             players = list(self._player_roles)
             self._ratings.load_state_dict(state["ratings"], agents=players)
             self._pfsp.load_state_dict(state["pfsp"], agents=players)
+            row = state.get("journal_row")
+            self._league_state = LeagueStateInfo(env_steps=int(state.get("env_steps") or 0),
+                                                 journal_row=None if row is None else int(row),
+                                                 has_watermark="journal_row" in state)
         except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as e:
             raise ConfigError(f"Cannot read the league state {path} ({type(e).__name__}: {e}); move it away to "
                               f"resume with empty ratings and PFSP statistics") from e
         return True
 
+    @property
+    def loaded_league_state(self) -> LeagueStateInfo | None:
+        """What ``load_league_state`` read about the journal (None: no file was loaded)."""
+        return self._league_state
+
+    def league_state_journal_row(self) -> int | None:
+        """The journal watermark of the loaded league state (None: none loaded, an SP4 file, or nothing accounted)."""
+        return None if self._league_state is None else self._league_state.journal_row
+
+    def replay_results(self, results: Iterable[MatchResult]) -> tuple[int, int]:
+        """In-place resume (SP5 T4.3, R8): account journaled results the saved league state does not hold yet, in
+        commit order, exactly as ``report_match_result``. A result the current game cannot rate (a layout it no
+        longer has, another outcome kind: the config changed with ``-c``) is skipped and counted. Returns
+        ``(replayed, skipped)``."""
+        replayed = skipped = 0
+        for result in results:
+            layouts = self._spec.layouts
+            if result.layout not in layouts or result.outcome_kind != self._spec.outcome_kind(result.layout):
+                if not skipped:
+                    logger.warning(f"Resume: journaled match {result.uid} (layout {result.layout!r}, "
+                                   f"{result.outcome_kind}) cannot be replayed into the ratings of this game; such "
+                                   f"matches are skipped")
+                skipped += 1
+                continue
+            self.report_match_result(result)
+            replayed += 1
+        return replayed, skipped
+
     def save_checkpoint_payload(self, payload: dict, meta_extra: dict | None = None) -> str:
         """Persist a learner checkpoint payload (see ``learner.make_checkpoint_payload``).
 
```

Retention (module docstring rule 1: "the newest final snapshot of the CURRENT run (an in-place resume keeps the run id, so earlier finals of the same run are ordinary: SP5)"):
```diff
--- a/src/colosseum/arena/retention.py
+++ b/src/colosseum/arena/retention.py
@@ -112,13 +112,17 @@
         raise ValueError(f"retain: duplicate checkpoint ids in {ids}")
     version = {e.checkpoint_id: e.version for e in ordered}
     newest = {e.checkpoint_id for e in ordered[-policy.keep_last:]}
+    # SP5 (in-place resume keeps the run id): only the newest final of the current run is protected, so every
+    # stop / resume cycle does not add another protected snapshot. Without a run id every final is (SP3 rule).
+    own_finals = [e.checkpoint_id for e in ordered if e.final and e.run_id == policy.current_run_id]
+    newest_final = own_finals[-1] if own_finals else None
 
     always: dict[str, list[str]] = {}
     for e in ordered:
         why = []
         if e.checkpoint_id in newest:
             why.append(KEEP_LAST)
-        if e.final and (policy.current_run_id is None or e.run_id == policy.current_run_id):
+        if e.final and (policy.current_run_id is None or e.checkpoint_id == newest_final):
             why.append(FINAL)
         if e.pinned:
             why.append(PINNED)
```

- [ ] **Step 5: Write `src/colosseum/hub/resume.py`**

```python
"""In-place resume of a run dir (SP5 spec block 5.4; task T4.3): ``colosseum hub --resume RUN`` and
``colosseum train --resume RUN`` continue the SAME dir and run id.

``prepare_resume`` runs before anything is written:
- the run dir must exist with ``config.resolved.yaml`` and ``checkpoints/``; it is refused (``ConfigError``) while
  ``run_state.json`` names a live process (two hubs on one run dir are impossible);
- the config: without ``-c`` the saved ``config.resolved.yaml`` (plus the agent set of ``agents_state.json`` once
  T8.2 writes it); with ``-c`` the new file, and a WARNING lists every setting that differs from the saved one;
  ``--set`` applies in both cases; ``run.dir`` / ``run.name`` always name the resumed dir (a ``-c`` file that names
  another one gets a WARNING);
- ``training.resume_from`` of the config is not applied again (the run imported its parent when it started; an
  in-place resume continues the run's own state): an INFO line, or a WARNING when ``-c`` names another source;
- the start env-step count: ``hub_state.json`` when present (an SP5 run dir), else (an SP4 / SP3 run dir) the
  largest count found in ``metrics.jsonl``, the journal, the snapshots' ``meta.json`` and ``ratings.json``;
  resuming at or past ``training.total_timesteps`` is a WARNING (spec 4.4).

``commit_resume`` then keeps the previous ``config.resolved.yaml`` as ``config.resolved.<k>.yaml`` (k = 1, 2, ...,
the next free number) and writes the effective one. Everything else (the journal migration, the league replay, the
hub counters, the snapshot pool, the learners' start points) is restored by ``colosseum.hub.hub.Hub``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.run_dir import RESOLVED_CONFIG_FILE, RunDir, read_run_id
from colosseum.core.run_state import process_alive, read_run_state
from colosseum.hub.state import HUB_STATE_FILE, load_hub_state

logger = logging.getLogger(__name__)

AGENTS_STATE_FILE = "agents_state.json"      # written by T8.2
_BACKUP_RE = re.compile(r"config\.resolved\.(\d+)\.yaml")
DIFF_LINES_MAX = 40
_PINNED_KEYS = ("run.dir", "run.name")


@dataclass(frozen=True)
class ResumePlan:
    run_dir: RunDir
    config: ColosseumConfig
    from_file: bool                  # -c given (False: the saved config)
    diff: list[str]                  # settings that differ from the saved config ("key: old -> new")
    run_id: str | None               # None for an SP3 run dir (no run.json yet; the hub creates one)
    sp5: bool                        # hub_state.json present
    start_env_steps: int


def check_not_running(root: str | Path) -> None:
    """ConfigError while ``run_state.json`` of ``root`` names a live process (spec 5.4)."""
    state = read_run_state(root)
    if state is not None and state.status == "running" and process_alive(state):
        raise ConfigError(f"Run {root} is running (pid {state.pid}); stop it first (Ctrl+C in its terminal) or "
                          f"wait until it exits: two hubs cannot share one run dir")


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    """Dotted keys of a nested dict (lists are leaves)."""
    if not isinstance(data, Mapping):
        return {prefix: data}
    out: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def config_diff(old: ColosseumConfig, new: ColosseumConfig) -> list[str]:
    """Settings that differ between two resolved configs, sorted by key: ``key: old -> new``, ``key: added``,
    ``key: removed`` (``run.dir`` / ``run.name`` are pinned to the run dir and left out)."""
    a = flatten(old.model_dump(mode="json", by_alias=True))
    b = flatten(new.model_dump(mode="json", by_alias=True))
    lines = []
    keys = set(a) | set(b)
    for key in sorted(keys):
        if key in _PINNED_KEYS:
            continue
        if (a.get(key, b.get(key)) == {} or b.get(key, a.get(key)) == {}) and any(
                k.startswith(f"{key}.") for k in keys):
            continue                                   # an empty section on one side, its keys on the other
        if key not in b:
            lines.append(f"{key}: removed (was {a[key]!r})")
        elif key not in a:
            lines.append(f"{key}: added ({b[key]!r})")
        elif a[key] != b[key]:
            lines.append(f"{key}: {a[key]!r} -> {b[key]!r}")
    return lines


def next_backup_path(root: str | Path) -> Path:
    """``config.resolved.<k>.yaml`` with the next free k (1 for the first resume)."""
    root = Path(root)
    taken = [int(m.group(1)) for p in root.glob("config.resolved.*.yaml") if (m := _BACKUP_RE.fullmatch(p.name))]
    return root / f"config.resolved.{max(taken, default=0) + 1}.yaml"


def _pin_run(config: ColosseumConfig, root: Path, *, warn: bool) -> ColosseumConfig:
    run = config.run
    if warn and ((run.name is not None and run.name != root.name)
                 or Path(run.dir).resolve() != root.parent.resolve()):
        logger.warning(f"Resume: the config names run dir {Path(run.dir) / str(run.name)}; the resumed run stays in "
                       f"{root}")
    return config.model_copy(update={"run": run.model_copy(update={"dir": str(root.parent), "name": root.name})})


def _apply_agents_state(config: ColosseumConfig, root: Path) -> ColosseumConfig:
    """The agent set of ``agents_state.json`` (T8.2 fills this in; until then the file does not exist)."""
    return config


def _load_saved(root: Path, overrides: Mapping[str, Any] | None) -> ColosseumConfig:
    path = root / RESOLVED_CONFIG_FILE
    if not path.is_file():
        raise ConfigError(f"{root} has no {RESOLVED_CONFIG_FILE}: it is not a run dir (resume needs the dir "
                          f"colosseum train / hub created, e.g. runs/<name>)")
    return load_config(path, dict(overrides) if overrides else None)


def start_env_steps(root: str | Path) -> int:
    """The env-step count an SP4 / SP3 run dir reached (no ``hub_state.json``): the largest of ``metrics.jsonl``
    (any record's ``env_steps``), the journal's training matches, the snapshots' ``meta.json`` and
    ``ratings.json`` (0 when none has one)."""
    root = Path(root)
    found = [0]
    metrics = root / "metrics.jsonl"
    if metrics.is_file():
        with metrics.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    value = json.loads(line).get("env_steps")
                except (ValueError, AttributeError):
                    continue                       # a torn last line of a killed run
                if isinstance(value, int):
                    found.append(value)
    journal = root / "matches.db"
    if journal.is_file():
        from colosseum.arena.matchlog import MatchLogReader

        with MatchLogReader(journal) as reader:
            found.append(reader.max_env_steps() or 0)
    for meta in (root / "checkpoints").glob("*/ckpt_v*/meta.json"):
        try:
            value = json.loads(meta.read_text()).get("env_steps")
        except (OSError, ValueError):
            continue
        if isinstance(value, int):
            found.append(value)
    try:
        value = json.loads((root / "ratings.json").read_text()).get("env_steps")
        if isinstance(value, int):
            found.append(value)
    except (OSError, ValueError):
        pass
    return max(found)


def _hub_state_env_steps(root: Path) -> int | None:
    """``hub_state.json``'s env-step count (None: no file, an SP4 / SP3 run dir); a malformed file is a
    ConfigError (the hub writes it atomically)."""
    path = root / HUB_STATE_FILE
    if not path.is_file():
        return None
    try:
        data = load_hub_state(path)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ConfigError(f"Cannot read {path} ({type(e).__name__}: {e}); the hub writes it atomically, so it was "
                          f"edited: restore it, or move it away to continue with the counters of metrics.jsonl "
                          f"and the journal") from e
    return None if data is None else int(data.env_steps)


def prepare_resume(run: str | Path, config_path: str | Path | None,
                   overrides: Mapping[str, Any] | None = None) -> ResumePlan:
    """Check and load an in-place resume (module docstring). Nothing is written."""
    root = Path(run)
    if not root.is_dir() or not (root / "checkpoints").is_dir():
        raise ConfigError(f"--resume {run}: not a run dir (expected a dir with checkpoints/ and "
                          f"{RESOLVED_CONFIG_FILE}, e.g. runs/<name>)")
    check_not_running(root)
    saved = _apply_agents_state(_load_saved(root, overrides), root)
    if config_path is None:
        config, diff = saved, []
    else:
        config = load_config(config_path, dict(overrides) if overrides else None)
        diff = config_diff(_pin_run(saved, root, warn=False), _pin_run(config, root, warn=False))
        if diff:
            shown = diff[:DIFF_LINES_MAX]
            more = f"\n  ... and {len(diff) - len(shown)} more" if len(diff) > len(shown) else ""
            logger.warning(f"Resume: {config_path} differs from the run's saved config in {len(diff)} settings:\n  "
                           + "\n  ".join(shown) + more)
    config = _pin_run(config, root, warn=config_path is not None)
    if config.training.resume_from:
        if config_path is not None and config.training.resume_from != saved.training.resume_from:
            logger.warning(f"Resume: training.resume_from={config.training.resume_from!r} is ignored: an in-place "
                           f"resume continues the run's own state (start a new run to resume from another one)")
        else:
            logger.info(f"Resume: training.resume_from={config.training.resume_from!r} was imported when the run "
                        f"started; the run continues from its own state")
    hub_steps = _hub_state_env_steps(root)
    steps = hub_steps if hub_steps is not None else start_env_steps(root)
    if steps >= config.training.total_timesteps:
        logger.warning(f"Resume: the run is at {steps} env steps, at or past training.total_timesteps "
                       f"({config.training.total_timesteps}); it stops at once unless you raise it "
                       f"(--set training.total_timesteps=N)")
    return ResumePlan(run_dir=RunDir(root, root.name), config=config, from_file=config_path is not None, diff=diff,
                      run_id=read_run_id(root), sp5=hub_steps is not None, start_env_steps=steps)


def commit_resume(plan: ResumePlan) -> Path | None:
    """Keep the previous ``config.resolved.yaml`` as ``config.resolved.<k>.yaml`` and write the effective one;
    returns the backup path (None when there was nothing to keep)."""
    root = plan.run_dir.root
    current = root / RESOLVED_CONFIG_FILE
    backup = None
    if current.is_file():
        backup = next_backup_path(root)
        shutil.copy2(current, backup)
    plan.run_dir.write_resolved_config(plan.config)
    return backup



def check_resumed_agents(root: str | Path, agent_configs: Mapping[str, ColosseumConfig],
                         role_specs: Mapping[str, Any]) -> dict[str, str]:
    """Strict checks of an in-place resume (spec 5.4), before anything is written: for every trainable agent of
    ``agent_configs`` (``setup_run``'s per-agent configs) that has snapshots in the run dir, the newest snapshot's
    role signature must be that of the agent's roles in this config and its weights must load into the agent's
    model (as a strict ``training.resume_from``); ConfigError naming the agent and the snapshot otherwise. An agent
    without snapshots (added with ``-c``) starts fresh. Returns ``{agent: checked snapshot}``."""
    from colosseum.coordinator.checkpoint_manager import check_model_state, resolve_resume
    from colosseum.core.registry import build_model
    from colosseum.core.roles import role_signature

    checked: dict[str, str] = {}
    for aid, acfg in agent_configs.items():
        if not (Path(root) / "checkpoints" / aid).is_dir():
            continue
        try:
            state = resolve_resume(str(root), aid, expected_signature=role_signature(role_specs[aid]))
            if state is None:
                continue
            check_model_state(build_model(acfg, role_specs[aid]), state["model_state"], state["source"])
        except ConfigError as e:
            raise ConfigError(f"--resume {root}: agent {aid!r} cannot continue with this config: {e}") from e
        checked[aid] = state["source"]
    return checked


def resume_port(previous_addresses: Sequence[str], port: int | None, config_port: int) -> int:
    """The port a resumed ``colosseum hub`` listens on: ``--port`` when given; else the one port of the previous
    incarnation's addresses (``hub.json``), so nodes reconnect to the address they know; else
    ``distributed.port``."""
    if port is not None:
        return int(port)
    ports = {address.rsplit(":", 1)[1] for address in previous_addresses if ":" in address}
    if len(ports) == 1:
        found = ports.pop()
        if found.isdigit():
            return int(found)
    return int(config_port)
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_matchlog_v2.py tests/unit/test_sp5_league_replay.py tests/unit/test_sp5_resume_plan.py tests/unit/test_sp5_retention_resume.py tests/unit/test_sp4_matchlog.py tests/unit/test_sp4_matchlog_writer.py tests/unit/test_sp4_retention.py tests/unit/test_sp4_league_state.py -q`
Expected: all pass.

- [ ] **Step 6: The hub's start and resume functions (`src/colosseum/hub/hub.py`)**

1. **`run_training(config_path, overrides=None, *, resume=None, listen=None, nodes_file=None)`** and **`run_hub(..., resume=None, port=None, ...)`** with `resume` set, before any process starts:
   1. `plan = prepare_resume(resume, config_path, overrides)` (refusals are `ConfigError`: exit 1 with the hint, nothing written);
   2. `validate_run_config(plan.config)`; `setup = setup_run(plan.config, validate=False)`; `check_resumed_agents(plan.run_dir.root, setup.agent_configs, setup.role_specs)` (refused with `-c` and another architecture or role signature: nothing written);
   3. `backup = commit_resume(plan)`; `setup_process_logging(plan.run_dir.logs, "main", console_level=logging.INFO)` (the log files are appended); INFO `Resume: continuing run <name> (run id <id>) in place at env step <N>; previous config kept as <backup>`; `print(f"Run directory: {root}")`; `_record_run_state(run_dir, "running")`;
   4. `Hub(plan.config, plan.run_dir, options=HubOptions(mode=..., resume=True, port=port_for_hub, ...), validated=True).launch()`; `run_state.json` at the end as for a fresh run.
   For `hub --resume` without `--port`: `port = resume_port(read_hub_json(root).addresses if it exists else (), None, cfg.distributed.port)` (nodes reconnect to the address they know; ruling P20). `train --resume` without `--listen` keeps the ephemeral loopback port (its local node is a fresh child).
2. **`Hub.launch` with `options.resume`** (replaces the fresh-run steps named in each item; the rest of `launch` is unchanged):
   - `self._run_id = run_dir.ensure_run_id()` keeps the id (an SP3 run dir gets one now: INFO);
   - **`_load_hub_state()`**: `data = load_hub_state(root / HUB_STATE_FILE)`; with data: `data.run_id` must equal `self._run_id` (else ConfigError `hub_state.json belongs to run <x>; this dir's run.json names <y>`); `self._budget = BudgetCounter(base=data.env_steps, sessions=data.sessions)` (nodes that kept their session add their increments since the last save; a node process that restarted starts a new session); `self._epochs = EpochAllocator(reserved_until=data.epoch_reserved_until, on_reserve=...)` so the first epoch is above every epoch the previous incarnation could have granted (R7); `self._versions = VersionRegistry.from_state(data.versions)` (generations and caps continue); counters (`worker_restarts`, `learner_restarts`, `lineup_fetch_failed`, `relay_bytes`, `dropped_chunks`, `dropped_match_results`, `dropped_uploads`, `dropped_announcements`, `lineup_redraw_failed`, `nodes_lost`) restored into the hub's counters and the metrics hub's cumulative bases, so every monotonic counter stays monotonic across the resume (T2.5's `MetricsHub` / `SystemStats` API for bases; add `restore_cumulative(counters)` if it has none); `self._learners.load_resumed(data.learners, now=time.monotonic())` → INFO per awaited agent `Resume: waiting for node <n> to bring back the learner of <agent> (generation <g>), up to learner_relocate_after_sec`. Without the file (an SP4 / SP3 run dir): `BudgetCounter(base=plan.start_env_steps)`, a fresh registry (floors come from the stored snapshots, T4.1), INFO `Resume: <root> has no hub_state.json (an SP4 run dir); counters continue from env step <N> (metrics.jsonl, the journal, the snapshots)`.
   - **`training.resume_from` is not applied** (`_resolve_resume` returns no states, `_import_snapshot_pool` and the journal's `import_parent` are skipped; `prepare_resume` logged why); `init` is still resolved (a start source of agents without snapshots).
   - **`_load_league_state(coordinator)`** reads the run's own `league_state.json` (not `training.resume_from`'s); a missing file: INFO.
   - **`_resume_start_sources()`** (T4.1's hook): per trainable agent the hub-start source is its newest stored snapshot that has `trainer_state.pt` (`CheckpointManager.list_checkpoints(agent)`, newest first) → `("start:<ckpt id>", version, f"snapshot <ckpt id>")`, its frame written lazily at the first assignment by `_write_start_frame(agent, ckpt)` (`load_model` numpy weights + the raw bytes of `trainer_state.pt` → `encode_checkpoint(..., kind="recovery", generation=0)` into `BlobStore.start_path(agent, ckpt id)`); none with trainer state → the newest snapshot weights only (trainer state `b""`) with the SP4 WARNING about a fresh optimizer; no snapshots → `init` / factory. T4.2's rule then prefers the recovery point when its version is not lower. INFO per agent `Resume [<agent>]: learners start from <description>`.
   - **The journal**: `_open_match_log` opens `MatchLog` (a version-1 file is migrated here: INFO from the library); then **`_replay_journal()`** before the gRPC server accepts sessions: `after, min_steps = league_replay_window(coordinator.loaded_league_state)`; `with MatchLogReader(path) as reader: replayed, skipped = coordinator.replay_results(r for _, r in reader.results_after(after, min_env_steps=min_steps)); folded = reader.folded_after(after)`; INFO `Resume: replayed <replayed> training matches of the journal into the ratings and PFSP statistics (<after row N | stamped after env step E | all rows>)`, plus `; <folded> were already folded into counts and are not replayed` when `folded`, plus the skipped count; the league state written at the next tick then carries the journal's last row as its watermark (T2.5: the highest accounted row).
   - `hub.json` and the running registry are rewritten for this incarnation (T2.5), with the port chosen above.
   - Nodes reconnect on their own (same identity, same hub address); a node whose `Hello.run_id` is this run keeps its cells' lineups until their TTL and its buffered results, which it resends (T2.4); the journal skips any it already had (uid), so nothing is accounted twice (R8).
3. **Agents added or removed by `-c`.** An added trainable agent has no snapshots: it starts fresh (`init` or factory) and its schedules count from env step 0 of the run until T8.2 adds start points (ruling P16); a removed agent's snapshots, journal rows and ratings stay; buffered results that name it are journaled (T2.1's intake accepts historical agents) and accounted (the ELO tables accept any id; PFSP skips non-trainable owners).

- [ ] **Step 7: The CLI (`src/colosseum/cli.py`)**

- `train`: `-c/--config` becomes optional (`required=False`); `--resume RUN` (`type=click.Path(exists=True, file_okay=False)`, help "Continue the run dir RUN in place (same dir and run id; refused while it runs). -c is optional: without it the run's saved config is used; --set applies either way."); without `--resume` a missing `-c` is a usage error (exit 2, `Missing option '-c' / '--config'.`); calls `run_training(config, overrides, resume=resume, listen=listen)`.
- `hub`: `--resume RUN` with the same help and the port rule of Step 6; `-c` optional with `--resume`.
- Exit codes as before (0 / 1 / 2 / 130 / 143).

- [ ] **Step 8: Integration tests (`tests/integration/test_sp5_resume_in_place.py`)**

Module docstring: "In-place resume through the CLI (SP5 T4.3, spec 5.4): the same dir and run id, counters, epochs, journal rows and snapshot versions continue; refused while the run lives; strict checks with -c; an SP4 run dir continued in place." Runs use `cli_runner` (`train_cmd` with `TINY`, a `resume` variant that adds `--resume ROOT` and drops `-c` unless given). Tests:
1. `test_train_resume_continues_the_run_in_place` — a TINY tic-tac-toe run (3 000 steps) then `train --resume ROOT --set training.total_timesteps=6000`: both exit 0; `run.json` unchanged; `config.resolved.1.yaml` equals the first resolved config and `config.resolved.yaml` has `total_timesteps: 6000`; `metrics.jsonl` has both incarnations' records and the first `system` record after the resume has `env_steps` ≥ the last one before it; `matches.db` has schema 2, its row ids strictly increase across both incarnations and the second incarnation's training uids have epochs (`_g<e>_`) above the first's maximum; `hub_state.json` `env_steps >= 6000` and `epoch_reserved_until` larger than after the first run; every snapshot saved by the second incarnation has a higher version than every one of the first; `logs/main.log` has two `main started (pid` lines; `run_state.json` `finished` with exit code 0; INFO `Resume: replayed 0 training matches` (the first run ended cleanly: its last tick wrote the watermark).
2. `test_resume_is_refused_while_the_run_is_alive` — `train` with a budget it never reaches, started in a session; once `run_state.json` says `running`, `train --resume ROOT` exits 1 with `is running (pid` in stderr and no `config.resolved.1.yaml`; then SIGINT to the first → 130; `train --resume ROOT --set training.total_timesteps=<its env steps + 1000>` then exits 0.
3. `test_an_sp4_run_dir_continues_in_place` — `copy_sp4_run(tmp_path)`; `train --resume COPY --set training.total_timesteps=<sp4_manifest()["env_steps"] + 1500>`: exit 0; `matches.db` schema 2, the SP4 rows have `node` NULL and every new row has the local node's number; `hub_state.json` exists; `logs/main.log` has `schema version 1 migrated to 2`, `has no hub_state.json (an SP4 run dir)` and `Resume: replayed`; `logs/learner-<agent>.log` has `Resumed from` naming the newest snapshot with `trainer_state.pt`; new snapshot versions are above `sp4_manifest()["final_version"]`; the SP4 `league_state.json` was loaded (INFO `league state ... restored`).
4. `test_a_changed_architecture_is_refused_before_anything_is_written` — a TINY run, then `train --resume ROOT -c changed.yaml` whose `networks.kwargs` changes a layer size: exit 1, stderr `cannot continue with this config`; `config.resolved.yaml` is byte-identical, no `config.resolved.1.yaml`, `run_state.json` still `finished`.
5. `test_with_c_the_diff_is_logged_and_an_agent_can_be_added` — a TINY run of an agent `main`; `train --resume ROOT -c more.yaml` where `more.yaml` = the saved config with `learner.batch_chunks` changed and a scripted anchor `rnd` (`colosseum.players.RandomBot`) with `matchmaking.opponents.anchors: 0.3`: exit 0; `logs/main.log` has `differs from the run's saved config in` and the `learner.batch_chunks` line; the journal has training matches with a seat of `rnd` after the resume.
6. Regression guards (unchanged files, still green): `tests/integration/test_sp4_sp3_resume.py`, `tests/integration/test_sp4_league_state_resume.py`, `tests/integration/test_sp3_sp2_resume.py` (`training.resume_from` = a child run with a new dir and run id).

- [ ] **Step 9: The hub restart on a live node (`tests/integration/test_sp5_hub_restart_slow.py`, `@pytest.mark.slow`)**

`test_kill_the_hub_and_resume_in_place_with_a_live_node` (criterion 11's hub row on localhost): `colosseum hub -c <toy config> --port 0` started with `cli_runner.run_in_session` (the hub identity under the test's XDG dirs; its token read in-process with `HubIdentity.load_or_create()`); `fault_proxy.TcpFaultProxy` targets the hub's port from `hub.json`; a `NodeAgent` in a thread connects through the proxy (1 worker + the learner: 2 loaded processes; `distributed.hub_lost_collect_sec` and `hub_lost_timeout_sec` well above the test's gap). Wait for ≥ 3 accepted versions and ≥ 200 journal rows; `kill -9` the hub's process group (the node lives); for 3 s the node keeps playing (its buffered unacked results grow; the learner keeps training: its log's `step=` lines grow) — then `hub --resume ROOT --port 0`, `proxy.retarget(<new port from hub.json>)`; the node reconnects with the same node id and number; the learner is adopted (`hub_state` learners: same generation, `running`), the versions the node announces after the resume are above every version accepted before the kill; at the budget the resumed hub exits 0; the node's result buffer is empty and nothing was dropped (`dropped_match_results_by_reason.node_buffer == 0`); no uid appears twice in the journal (UNIQUE) and the writer's `duplicates` equals the number of resent batches' results that were committed before the kill (≥ 0, logged); every training uid's epoch after the resume is above every epoch before it.

Add `TcpFaultProxy.retarget(port)` to `tests/fault_proxy.py` if T3.2 did not: new connections go to the new target port; open ones are closed.

- [ ] **Step 10: Focused tests, full fast suite + ruff**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_matchlog_v2.py tests/unit/test_sp5_league_replay.py tests/unit/test_sp5_resume_plan.py tests/unit/test_sp5_retention_resume.py tests/integration/test_sp5_resume_in_place.py tests/integration/test_sp4_sp3_resume.py tests/integration/test_sp4_league_state_resume.py tests/unit/test_sp4_matchlog.py tests/unit/test_sp4_report_data.py -q -rw`, then the slow test alone (`-m slow tests/integration/test_sp5_hub_restart_slow.py`), then the full fast suite and ruff. Expected: green, zero warnings. The report reads schema-1 and schema-2 journals (`MatchLogReader`); the dashboard and `colosseum report` of the T0.1 fixture are unchanged.

- [ ] **Step 11: Commit and push**

```bash
git add src/colosseum/hub/resume.py src/colosseum/arena/matchlog.py src/colosseum/arena/retention.py \
        src/colosseum/coordinator/coordinator.py src/colosseum/core/run_dir.py src/colosseum/hub/hub.py \
        src/colosseum/cli.py tests/fault_proxy.py tests/unit/test_sp5_matchlog_v2.py \
        tests/unit/test_sp5_league_replay.py tests/unit/test_sp5_resume_plan.py tests/unit/test_sp5_retention_resume.py \
        tests/integration/test_sp5_resume_in_place.py tests/integration/test_sp5_hub_restart_slow.py
git commit -m "feat: in-place resume of run dirs (also SP4), journal schema 2 with a node column, league replay"
git push
```

---
### Task T4.4: Hub away, the distributed stop, exit codes, bad data, log tails, portability

Spec 7.1 (rows «Хаб упал / перезапускается», «Узел прислал испорченные данные», «Нехватка `/dev/shm` или диска узла», the context of `EnvContractError` / `PlayerError`), 4.5 (`hub_lost_collect_sec`), 7.4 (stop), 7.5 (exit codes), 7.6 (monotonic clocks), 7.7 (logs and tails), 7.8 (portability). While the hub is away, nodes keep their lineups past the TTL, the direct data plane keeps learners training (versions through `GetLatest`, T3.1), results wait in the node buffer and are resent after `hub --resume`; a learner reachable only through the hub stops getting data after `hub_lost_collect_sec`; after `hub_lost_timeout_sec` a node stops its processes and keeps reconnecting every 30–60 s. A full shared-memory dir or cache disk is an ERROR with a hint and the node takes no new work until space returns. Every bad thing a node sends is dropped, counted by reason, with one WARNING per node and reason. The stop sends `stop_run`, waits for the finals and the result flush within `distributed.shutdown_grace_sec` (the local 10 s guarantee when every node is local), and nodes stay connected unless `--once`. The exit codes of `hub` / `train` and of the node agent follow spec 7.5. A dead remote process's last 64 KB of log land in `runs/<name>/logs/nodes/<node>/<proc>.log.tail`, and ERROR / WARNING lines name both paths. Parent death and process liveness get portable fallbacks.

**Split point:** dispatch (a) Steps 1–6 (hub away, the cache-full rule, their tests), (b) Steps 7–14 (stop and exit codes, bad data, log tails, portability and the clock audit, suites, commit).

**Files:**
- Modify: `src/colosseum/node/agent.py` (hub-away mode, the collect rule without the hub, the hub-lost timeout, cache-full handling, `stop_run`, signals and exit codes, `node_stopped` / `node_leaving` events, error contexts), `src/colosseum/node/session.py` (`away_for()`, the slow retry after the timeout), `src/colosseum/node/procs.py` (`watch_parent` re-export used by every child target), `src/colosseum/node/logs.py` (`tail_for_event`)
- Modify: `src/colosseum/hub/hub.py` (stop / exit-code functions only: `_shutdown`, `_stop_nodes`, `_exit_code_after_shutdown`, `_signal_exit_code`; `_write_log_tail`, `_count_drop` and their one-line hooks in the handlers)
- Modify: `src/colosseum/utils/process.py` (`PARENT_WATCH_SEC`, `start_parent_watchdog`, `watch_parent`; `init_child_process` uses it), `src/colosseum/core/run_state.py` (portable `process_start_ticks`)
- Modify: `tests/node_kit.py` (`run_node_main(options_json) -> int`: a node agent as a script, for the signal tests)
- Test: `tests/unit/test_sp5_portability.py`, `tests/unit/test_sp5_bad_data.py`, `tests/unit/test_sp5_log_tails.py`, `tests/integration/test_sp5_hub_away.py`, `tests/integration/test_sp5_stop_and_exit.py`

**Interfaces:**
- Consumes: `HubSession` (`connected`, reconnect backoff `RECONNECT_MIN_SEC` / `RECONNECT_MAX_SEC`), `HUB_LOST_RETRY_SEC = (30.0, 60.0)` (T2.3); `NodeAgent`, `ChildSupervisor.stop_all(grace)`, the results buffer, `node.logs.log_tail` / `node_log_dir`, `LOG_TAIL_BYTES` (T2.4); `LearnerRoute` and the relay / direct path flags (T3.2); `GetLatest` without the hub (T3.1); `CacheFullError` and the free-space check (T1.5); `UploadQueue.drop_waiting` and `self._finals` (T4.2); `LearnerTable` (T4.1); `LOCAL_SHUTDOWN_GRACE_SEC`, `Hub.request_stop` (T2.5).
- Produces (contract plus additions marked *):
  - `watch_parent()` in `colosseum.node.procs` (contract; implemented in `colosseum.utils.process` and re-exported), *`PARENT_WATCH_SEC = 1.0`, *`start_parent_watchdog(poll, *, getppid, on_death)`; `process_start_ticks` portable (Linux `/proc`, macOS `ps -o lstart=`);
  - *`HubSession.away_for() -> float` (0 while connected), *`HubSession.slow_retry(on: bool)`;
  - *Event kinds `node_stopped` (the node finished `stop_run`: children gone, results acked or given up) and `node_leaving` (the node stops on its own: Ctrl+C / SIGTERM / `--once` end); *`Command(kind="stop_run")` carries the grace in `reason` (`"grace=<sec>"`);
  - *hub counters `dropped_announcements`, `dropped_metrics`, `dropped_events` by reason (next to T2.x's `dropped_match_results_by_reason`, `dropped_chunks`, T4.2's `dropped_uploads`), all in `hub_state.json` counters;
  - `runs/<name>/logs/nodes/<node>/<proc>.log.tail` (contract);
  - *`node_kit.run_node_main(options_json)`.

- [ ] **Step 1: Write the failing hub-away tests (`tests/integration/test_sp5_hub_away.py`)**

Module docstring: "A node while its hub is away (SP5 T4.4, spec 4.5, 7.1): lineups outlive their TTL, results wait and arrive once, a relay-only learner stops getting data after hub_lost_collect_sec, processes stop after hub_lost_timeout_sec and come back with the hub, a full cache stops new work." Setup: `node_kit.FakeHub` behind `fault_proxy.TcpFaultProxy` (the node connects to the proxy), a toy config with `rollout.match_refresh_interval_sec: 0.5`, `distributed.heartbeat_sec: 0.1`, `hub_lost_collect_sec: 1.0`, `hub_lost_timeout_sec: 2.0`, `shutdown_grace_sec: 3.0`; a `NodeAgent` in a thread with 1 worker and (where the test says) the learner of `a`; `monkeypatch.setattr(colosseum.node.session, "HUB_LOST_RETRY_SEC", (0.2, 0.4))`. Tests:
1. `test_lineups_outlive_their_ttl_while_the_hub_is_away` — after the first grants, `proxy.cut(1.5)` (shorter than the timeout): during the cut the worker's env steps grow (the node's local counter, `NodeAgent.env_steps`), no `LineupRequest` reaches the FakeHub; after it, the node reconnects with the same session id and requests the expired cells in one `LineupRequest`.
2. `test_results_wait_in_the_buffer_and_arrive_once` — the FakeHub stops acking (records but never sends `Ack`), then `proxy.cut(1.0)`; after the reconnect it acks everything: every `ResultBatch` seq from 1 to the last arrived, the resent batches carry exactly the results of the unacked ones (same uids), each uid is acked once, the node's buffer is empty at the end.
3. `test_a_relay_only_learner_stops_collecting_after_hub_lost_collect_sec` — the node's worker collects for agent `b` whose learner (per the FakeHub's announcements) is reachable only through the relay (`VersionInfo.relay_only`, no data addresses); `proxy.cut(1.8)`: after `hub_lost_collect_sec` the node logs one WARNING `the hub is away 1 s: chunks for b (its learner is reachable only through the hub) are not collected`, the worker's chunks for `b` are counted as `no_learner` in its next `WorkerStats`; after the reconnect the route comes back (chunks counted as sent again).
4. `test_the_hub_lost_timeout_stops_the_processes_and_the_hub_brings_them_back` — learner `a` + 1 worker; `proxy.cut(4.0)`: after `hub_lost_timeout_sec` the node's children are gone (their pids dead) and the node logs `the hub is away 2 s: processes stopped; reconnecting every 0.2–0.4 s`; the node agent keeps running; after the cut, the `Welcome` (with `learners` = `a`) brings a worker and the learner back (new pids, the learner announces again).
5. `test_a_once_node_exits_1_after_the_hub_lost_timeout` — `NodeOptions(once=True)`, `proxy.cut(10.0)`: `NodeAgent.run()` returns 1 after `hub_lost_timeout_sec` (ruling P17), its children are gone.
6. `test_a_full_cache_stops_taking_work_until_space_returns` — monkeypatch T1.5's free-space probe in `colosseum.node.weights` to report 1 MiB free: the next fetch raises `CacheFullError`; the node logs an ERROR with T1.5's hint (`--shm-size` / `emptyDir: Memory`), sends `Event(kind="warning", error_kind="cache_full")` once, and sends no `LineupRequest` for three heartbeats while its cells keep playing; restore the probe → the next heartbeat's check clears the state (INFO `space is back`) and lineup requests resume.

- [ ] **Step 2: Run them to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp5_hub_away.py -q`
Expected: failures in 3–6 (no hub-lost collect rule, no timeout stop, no cache-full state); 1–2 may already pass on T2.4's buffer and lineup rules — they stay as the contract.

- [ ] **Step 3: The session client (`src/colosseum/node/session.py`)**

`HubSession` keeps `self._last_connected = time.monotonic()` (updated while the stream is up); `away_for()` = 0.0 while connected, else `time.monotonic() - self._last_connected`. `slow_retry(True)` switches the reconnect pause from the backoff `RECONNECT_MIN_SEC → RECONNECT_MAX_SEC` (doubling, jitter ≤ 20 %) to `random.uniform(*HUB_LOST_RETRY_SEC)`; `slow_retry(False)` and every successful connect switch back. The session id stays the same for the life of the node process (spec 4.1: a new id only at a new process).

- [ ] **Step 4: The node while the hub is away (`src/colosseum/node/agent.py`)**

Checked on every main-loop pass with `away = session.away_for()`:
1. **Lineups**: while not connected, no `LineupRequest` is sent; a cell whose TTL expired keeps its lineup and is marked "refresh due"; the first pass after the `Welcome` requests every due cell in one message (T2.4 rule; kept).
2. **Weights**: T3.1's `GetLatest` polling of the learner nodes' data ports while the hub is away (kept; the test of Step 9 of T4.3 and Step 1 here exercise it).
3. **`hub_lost_collect_sec`**: when `away >= distributed.hub_lost_collect_sec`, every agent whose current route goes through the hub relay (no reachable data address) gets `LearnerRoute(agent, None, via_node=False, generation)` on every local worker (their chunks for it are dropped and counted `no_learner`: results are still played and buffered) and one WARNING per agent; the routes are restored from the next `Welcome` / `Announce`.
4. **`hub_lost_timeout_sec`**: when `away >= distributed.hub_lost_timeout_sec`: `Stop(reason="hub_lost")` to every child (learners send no final: there is nobody to receive it; the hub restarts them from recovery points), `ChildSupervisor.stop_all(distributed.shutdown_grace_sec)`, cells and routes cleared, the results buffer kept, one WARNING `the hub is away <s> s: processes stopped; reconnecting every <a>–<b> s (Ctrl+C to stop)`, `session.slow_retry(True)`; with `--once` the agent then returns 1 from `run()`. On the next `Welcome` the node starts again exactly as at its first `Welcome` (workers, evaluators, the desired learners).
5. **Cache full**: a `CacheFullError` from any cache write of the node (a fetched version, a snapshot / fixed / start blob) → `self._cache_full = True`, ERROR with T1.5's hint once per episode of the condition, `Event(kind="warning", error_kind="cache_full", message=...)`; while set, no `LineupRequest`, no blob fetches and no `start_learner` is executed (the command is kept and run when space returns); every heartbeat re-checks the free space against `2 × <largest frame the node handled> + 64 MiB` and clears the flag (INFO). A learner's own publish that fails with `CacheFullError` is caught in `NodeOutputs.publish_weights` (ERROR once in the learner log, the publish skipped, training goes on) — the workers keep their older weights.

- [ ] **Step 5: Wire it and run the hub-away tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp5_hub_away.py -q`
Expected: all pass (about 20 s in all; test 4 is the longest).

- [ ] **Step 6: Commit the first half (split point a)**

```bash
git add src/colosseum/node/agent.py src/colosseum/node/session.py tests/integration/test_sp5_hub_away.py
git commit -m "feat: nodes keep working while the hub is away (lineups, buffered results, collect and timeout rules, full cache)"
git push
```

- [ ] **Step 7: Write the failing stop / exit / data / tail / portability tests**

`tests/unit/test_sp5_portability.py`:
```python
"""Portable process helpers (SP5 T4.4, spec 7.8): the parent-death watch (PR_SET_PDEATHSIG on Linux, a watchdog
thread elsewhere), process start times on Linux and macOS (``ps``), process liveness by pid + start time."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import pytest

from colosseum.core import run_state
from colosseum.core.run_state import RunState, process_alive, process_start_ticks
from colosseum.utils import process as proc


def test_watch_parent_uses_pdeathsig_on_linux_and_the_watchdog_elsewhere(monkeypatch):
    started = []
    monkeypatch.setattr(proc, "start_parent_watchdog", lambda *a, **k: started.append(1))
    monkeypatch.setattr(proc, "set_parent_death_signal", lambda sig: True)
    assert proc.watch_parent() == "pdeathsig" and started == []
    monkeypatch.setattr(proc, "set_parent_death_signal", lambda sig: False)    # macOS: no prctl
    assert proc.watch_parent() == "watchdog" and started == [1]


def test_the_watchdog_fires_once_the_parent_pid_changes():
    parent = [1234]
    died = threading.Event()
    thread = proc.start_parent_watchdog(0.01, getppid=lambda: parent[0], on_death=died.set)
    time.sleep(0.05)
    assert not died.is_set() and thread.is_alive()
    parent[0] = 1                                                           # re-parented: the parent is gone
    assert died.wait(2.0)
    thread.join(2.0)
    assert not thread.is_alive()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc")
def test_linux_start_ticks_identify_this_process():
    ticks = process_start_ticks(os.getpid())
    assert isinstance(ticks, int) and ticks == process_start_ticks(os.getpid())
    assert process_start_ticks(2**22 + 12345) is None


def test_macos_start_time_from_ps(monkeypatch):
    monkeypatch.setattr(run_state.sys, "platform", "darwin")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs["env"]["LC_ALL"]))
        pid = cmd[-1]
        out = "Sun Oct 18 02:15:25 2026\n" if pid == "42" else ""
        return subprocess.CompletedProcess(cmd, 0 if out else 1, stdout=out, stderr="")

    monkeypatch.setattr(run_state.subprocess, "run", fake_run)
    first = process_start_ticks(42)
    assert isinstance(first, int) and first == process_start_ticks(42)
    assert calls[0] == (["ps", "-o", "lstart=", "-p", "42"], "C")
    assert process_start_ticks(7) is None                                  # gone: ps prints nothing, exit 1
    monkeypatch.setattr(run_state.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("ps")))
    assert process_start_ticks(42) is None


def test_other_platforms_have_no_start_time(monkeypatch):
    monkeypatch.setattr(run_state.sys, "platform", "win32")
    assert process_start_ticks(os.getpid()) is None


def test_liveness_by_pid_and_start_time(monkeypatch):
    me = os.getpid()
    assert process_alive(RunState("running", me, proc_start=process_start_ticks(me)))
    assert not process_alive(RunState("running", 2**22 + 12345))
    if sys.platform.startswith("linux"):
        assert not process_alive(RunState("running", me, proc_start=process_start_ticks(me) + 1))   # pid reused
    monkeypatch.setattr(run_state, "process_start_ticks", lambda pid: None)   # no start time known: pid only
    assert process_alive(RunState("running", me, proc_start=5))
```
and append the clock audit (spec 7.6; it scans the SP5 packages; `import re` and `from pathlib import Path` go into the module's import block):
```python
SRC = Path(__file__).resolve().parents[2] / "src" / "colosseum"
_WALL_CLOCK = re.compile(r"\btime\.time\(\)|\bdatetime\.(?:datetime\.)?now\(")


def test_sp5_packages_measure_time_with_the_monotonic_clock():
    """Spec 7.6: durations, timeouts, leases and cadences use time.monotonic(); a wall-clock read in the SP5
    packages is allowed only as an informational timestamp marked ``# ts: informational``."""
    offenders = []
    for package in ("hub", "node", "net", "bootstrap"):
        for path in sorted((SRC / package).rglob("*.py")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if _WALL_CLOCK.search(line) and "ts: informational" not in line:
                    offenders.append(f"{path.relative_to(SRC)}:{number}: {line.strip()}")
    assert offenders == [], "use time.monotonic() for durations:\n" + "\n".join(offenders)
```
(Existing informational timestamps in those packages get the `# ts: informational` comment in this task.)

`tests/unit/test_sp5_log_tails.py` (module docstring: "Log tails of remote processes (SP5 T4.4, spec 7.7)"): an in-process hub + a `FakeNode`; tests:
1. `test_a_remote_proc_died_event_writes_its_tail` — `Event(kind="proc_died", proc="worker-0", exit_code=-9, log_tail=b"line 1\nline 2\n", log_path="/home/x/.cache/colosseum/logs/ab12cd34/<run>/worker-0.log")` from node 2 → `runs/<name>/logs/nodes/2/worker-0.log.tail` holds those bytes; the WARNING names both paths (`see <run tail path> (node log /home/x/...)`).
2. `test_a_tail_is_capped_and_overwritten` — a 200 KB `log_tail` is stored as its last `LOG_TAIL_BYTES` (64 KiB); a second event of the same proc overwrites the file.
3. `test_the_local_node_writes_no_tail_file` — the same event from the local node (its `log_path` is under `runs/<name>/logs/`) writes no `.log.tail` and the WARNING names `logs/worker-0.log`.
4. `test_a_hostile_proc_name_cannot_escape_the_logs_dir` — `proc="../../x"` → no file outside `logs/nodes/2/` (the name is made safe with `safe_path_component`), the event counted `dropped_events.invalid`.

`tests/unit/test_sp5_bad_data.py` (module docstring: "Bad data from a node (SP5 T4.4, spec 3.4 and 7.1): dropped, counted by reason, one WARNING per node and reason; the hub keeps working."): an in-process hub + `FakeNode`s `n1`, `n2`; each test sends the bad item three times from `n1` and once from `n2`, then one good item, and asserts the counter in `hub_state.json` counters, exactly two WARNING lines (one per node) for that reason, and that the good item is processed:
1. `test_invalid_results` — a result of an unknown layout, a seat of an unknown agent, a NaN reward → `dropped_match_results_by_reason.invalid` (T2.1's `ResultIntake.check`).
2. `test_stale_generation_results` — a seat above the cap of a superseded generation → `.stale_generation`.
3. `test_invalid_version_announcements` — an unknown agent, a negative version, a generation the hub never assigned → `dropped_announcements.invalid`.
4. `test_invalid_uploads` — a checkpoint upload whose bytes are not a frame, or whose header names another agent than `UploadHeader` → `UploadReply(accepted=False, reason="invalid")`, `dropped_uploads.invalid`, the spooled file deleted.
5. `test_invalid_metrics` — a `TrainMetrics` of an unknown agent, a `WorkerStats` with a negative counter → `dropped_metrics.invalid`; no record written.
6. `test_invalid_events` — an `Event` with an unknown kind or a `log_tail` above 1 MiB → `dropped_events.invalid` (a too-long tail is cut to `LOG_TAIL_BYTES` instead when its kind is valid: a valid event is never dropped for its tail).

`tests/integration/test_sp5_stop_and_exit.py` (module docstring: "The distributed stop and the exit codes (SP5 T4.4, spec 7.4, 7.5)."). Setups per test; ≤ 2 loaded processes each:
1. `test_budget_stop_collects_final_snapshots_and_the_node_stays` — an in-process hub (mode `hub`) + one `NodeAgent` in a thread (learner + 1 worker), small budget: the hub's `launch()` returns 0; `checkpoints/<agent>/` has a snapshot with `final: true` of the learner's last generation; the node agent is still running, its children are gone, it logged `run finished; waiting for the next run of this hub`; `node_stopped` reached the hub before it finished (main.log INFO `node <n> stopped`).
2. `test_a_once_node_exits_0_after_the_run` — the same with `NodeOptions(once=True)`: `NodeAgent.run()` returns 0 after the stop.
3. `test_the_local_grace_applies_without_remote_nodes` — `train` (only the local node): the hub's stop uses `LOCAL_SHUTDOWN_GRACE_SEC` (DEBUG line `stop grace 7 s (local nodes only)`); `train --listen` with one remote node in a thread → `stop grace <shutdown_grace_sec> s` (the remote final arrives).
4. `test_hub_exit_codes` — `colosseum hub` processes (`run_in_session`) with one `node_kit.run_node_main` process each: budget → 0; SIGINT on the hub → 130 and the node's children stop, the final snapshot saved; SIGTERM on the hub → 143; a failed agent (T4.2's `CrashingAPPO` agent next to a healthy one) → 1 at the end. (One test function per code if a combined one exceeds 20 s; the slower ones `@pytest.mark.slow`.)
5. `test_node_exit_codes` — a `node_kit.run_node_main` process: SIGINT → 130 after its children stopped (learners upload their finals when the hub is there: the hub saved a final); SIGTERM → 143; `nodes remove` (T6.3; until then the hub's `Command(kind="release")` sent by the test through the hub's admin hook) → 0; a protocol-major mismatch is T6.3's (not here).
6. `test_a_player_error_on_a_remote_node_stops_the_run_with_its_context` — an in-process hub + a `NodeAgent` in a thread whose worker plays a scripted `game_helpers.PlayerErrorBot` agent: the hub stops, `launch()` returns 1, its ERROR line contains `node <n> (<name>), worker 0, env `, `seat `, `episode step `, `layout `, `agent ` and both log paths (`runs/<name>/logs/nodes/<n>/worker-0.log.tail` and the node-local `worker-0.log`).
7. `test_node_leaving_relocates_at_once` — a node with a learner stops on its own (`request_stop` → `Event(kind="node_leaving")`): the hub marks it lost at once and relocates its learner without waiting `learner_relocate_after_sec` (ruling P24).

- [ ] **Step 8: Run them to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_portability.py tests/unit/test_sp5_log_tails.py tests/unit/test_sp5_bad_data.py tests/integration/test_sp5_stop_and_exit.py -q -m "not slow"`
Expected: `AttributeError: module 'colosseum.utils.process' has no attribute 'watch_parent'`, no `.log.tail` files, missing counters and WARNING lines, missing `node_stopped` / exit codes.

- [ ] **Step 9: Portability (`src/colosseum/utils/process.py`, `src/colosseum/core/run_state.py`, `src/colosseum/node/procs.py`)**

```diff
--- a/src/colosseum/utils/process.py
+++ b/src/colosseum/utils/process.py
@@ -39,6 +39,40 @@
         return False
 
 
+# How often the portable parent watch (no PR_SET_PDEATHSIG: macOS) checks the parent (SP5 spec 7.8).
+PARENT_WATCH_SEC = 1.0
+
+
+def start_parent_watchdog(poll: float = PARENT_WATCH_SEC, *, getppid: Callable[[], int] = os.getppid,
+                          on_death: Callable[[], None] | None = None) -> threading.Thread:
+    """A daemon thread that ends this process when its parent dies: every ``poll`` seconds it compares the parent
+    pid with the one at the start (an orphan is re-parented, e.g. to launchd), and then calls ``on_death`` (default:
+    ``os._exit(1)``: no cleanup can block it). The fallback of ``watch_parent`` where ``PR_SET_PDEATHSIG`` does not
+    exist."""
+    original = getppid()
+    die = on_death if on_death is not None else (lambda: os._exit(1))
+
+    def watch() -> None:
+        while True:
+            time.sleep(poll)
+            if getppid() != original:
+                die()
+                return
+
+    thread = threading.Thread(target=watch, name="parent-watch", daemon=True)
+    thread.start()
+    return thread
+
+
+def watch_parent(sig: int = signal.SIGTERM) -> str:
+    """Die with the parent (spec 7.8): ``PR_SET_PDEATHSIG`` on Linux (``"pdeathsig"``), else a watchdog thread
+    (``"watchdog"``, macOS). Children of nodes and of the hub call it first (``init_child_process``)."""
+    if set_parent_death_signal(sig):
+        return "pdeathsig"
+    start_parent_watchdog()
+    return "watchdog"
+
+
 def init_child_process() -> None:
     """Signal policy for every child: ignore Ctrl-C (the main process coordinates the stop)
     and die if the main process disappears (R3-17, R6-07).
@@ -52,7 +86,7 @@
     signal.signal(signal.SIGINT, signal.SIG_IGN)
     if hasattr(signal, "pthread_sigmask"):
         signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
-    set_parent_death_signal(signal.SIGTERM)
+    watch_parent(signal.SIGTERM)
     if not parent_alive():
         os._exit(1)
 
```
```diff
--- a/src/colosseum/core/run_state.py
+++ b/src/colosseum/core/run_state.py
@@ -16,6 +16,8 @@
 
 import json
 import os
+import subprocess
+import sys
 import time
 from dataclasses import dataclass, field
 from pathlib import Path
@@ -38,11 +40,34 @@
 
 
 def process_start_ticks(pid: int) -> int | None:
-    """Field 22 of ``/proc/<pid>/stat`` (start time in clock ticks after boot); None off Linux or if gone."""
+    """The start time of ``pid``, an integer compared only with itself on the same machine (a reused pid is not the
+    same process): on Linux field 22 of ``/proc/<pid>/stat`` (clock ticks after boot); on macOS
+    ``ps -o lstart= -p <pid>`` in seconds since the epoch (SP5 spec 7.8); None elsewhere or when the process is
+    gone."""
+    if sys.platform.startswith("linux"):
+        try:
+            text = Path(f"/proc/{int(pid)}/stat").read_text()
+            return int(text.rsplit(")", 1)[1].split()[19])
+        except (OSError, ValueError, IndexError):
+            return None
+    if sys.platform == "darwin":
+        return _ps_start_seconds(pid)
+    return None
+
+
+def _ps_start_seconds(pid: int) -> int | None:
+    """macOS: ``ps -o lstart=`` ("Mon Oct 18 12:00:00 2026", C locale) as local epoch seconds; None if gone."""
+    try:
+        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(int(pid))], capture_output=True, text=True,
+                             timeout=5.0, env={**os.environ, "LC_ALL": "C"})
+    except (OSError, subprocess.SubprocessError):
+        return None
+    text = " ".join(out.stdout.split())
+    if out.returncode != 0 or not text:
+        return None
     try:
-        text = Path(f"/proc/{int(pid)}/stat").read_text()
-        return int(text.rsplit(")", 1)[1].split()[19])
-    except (OSError, ValueError, IndexError):
+        return int(time.mktime(time.strptime(text, "%a %b %d %H:%M:%S %Y")))
+    except (ValueError, OverflowError):
         return None
 
 
```
`src/colosseum/node/procs.py`: `from colosseum.utils.process import watch_parent  # noqa: F401  (contract name)`; every child target of the node (`worker_target`, `learner_target`, `evaluator_target`) runs through `run_child` → `init_child_process` → `watch_parent`. Socket paths ≤ 104 characters on macOS and the shm dir are T2.2's / T1.5's (their tests stay).

- [ ] **Step 10: The stop (`src/colosseum/hub/hub.py`: `_shutdown`, `_stop_nodes`; `src/colosseum/node/agent.py`)**

Hub, on budget / `request_stop` / SIGINT / SIGTERM / "every agent failed":
1. `grace = LOCAL_SHUTDOWN_GRACE_SEC if every online node's data_path == "local" else cfg.distributed.shutdown_grace_sec`; `deadline = now + grace`; DEBUG `stop grace <g> s (<local nodes only | remote nodes>)`.
2. `_stop_nodes()`: `Command(kind="stop_run", reason=f"grace={grace:g}")` to every online node; the hub stops granting lineups and placing learners (`self._stopping = True`).
3. The main loop keeps handling events (results, uploads, metrics, events) until every learner that was starting or running at the stop delivered its final (`(agent, generation) in self._finals`) or is gone (its node lost, `node_stopped` received, or it died), and every online node sent `node_stopped` — or `deadline` passes (WARNING listing what did not arrive: `final snapshots missing: <agent> (node <n>)`; `nodes still stopping: <names>`).
4. Then T2.5's teardown (the journal writer's flush, live ratings, metrics, `league_state.json`, `hub_state.json`, pulse, the local node child within the remaining time, kill after); the local 10 s guarantee: with only local nodes everything is closed within `SHUTDOWN_GRACE_SEC + 3 s` as before.
Node, on `stop_run`: `Stop(reason="stop_run")` to every child (learners send their finals: `UploadQueue` with priority; workers flush their result batches); `deadline = now + grace` from the command; wait until the children exited (terminate at the deadline), the finals are uploaded or dropped at the deadline (`UploadQueue.drop_waiting`, ERROR per dropped final), and the results buffer is acked or the deadline passed; then `Event(kind="node_stopped")`; with `--once` close the session and return 0; else stay connected (INFO `run finished; waiting for the next run of this hub`) and take the next `Welcome` of a new run as a fresh start (persistent nodes, spec 2.1).
Node, on SIGINT / SIGTERM (its own signal handler, installed like `ProcessSupervisor.install_signal_handlers`): `Event(kind="node_leaving")` when connected; the same stop of its children with `shutdown_grace_sec`; return 128 + signum. Hub on `node_leaving`: the node is lost now (cells dropped) and its learners relocate without the wait (`lost_since = now - learner_relocate_after_sec`; ruling P24).

- [ ] **Step 11: Exit codes (`Hub._exit_code_after_shutdown`, `_signal_exit_code`; `NodeAgent.run`)**

`hub` / `train`: 0 budget reached or `request_stop`; 1 config error, contract / player error, a `failed` agent at the end (T4.2's rule), a fatal hub error; 2 bad arguments (click); 130 SIGINT; 143 SIGTERM — a signal that arrives while the run finishes still gives 128 + signum (SP2 rule kept). `run_state.json` is `finished` for 0 / 130 / 143, `failed` otherwise (T2.5). Node agent (`NodeAgent.run`, `join` later): 0 normal stop (`--once` after the run; `Command(kind="release")` = `nodes remove`); 1 the hub-lost timeout with `--once` (P17), access refused or a version mismatch it cannot update from (T6.3), a fatal error; 130 / 143 signals (after its children stopped).

- [ ] **Step 12: Bad data, tails and contexts (`_count_drop`, `_write_log_tail`; `node/logs.py`)**

1. **`_count_drop(node, reason_key, reason, what)`** in the hub: `counters[reason_key][reason] += 1` (kept in `hub_state.json`), and the first time per `(node, reason_key, reason)` a WARNING `node <n> (<name>): dropped <what> (<reason>); further ones are only counted`. Every intake path routes through it: results (`dropped_match_results_by_reason`), announcements (`dropped_announcements`), uploads (`dropped_uploads`), metrics (`dropped_metrics`), events (`dropped_events`), and T3.2's relay drops if they do not count already. No handler raises on node data: a handler exception from one message is caught at the dispatch, logged with its traceback once per (node, message kind), counted `dropped_events.handler_error`, and the loop goes on.
2. **`_write_log_tail(node, event)`** (called by `_on_event` for `proc_died`, `learner_failed`, `slot_disabled`, `child_error` of a remote node): `name = safe_path_component(event.proc, "proc")`; `path = run_dir.logs / "nodes" / str(node) / f"{name}.log.tail"`; write `event.log_tail[-LOG_TAIL_BYTES:]` atomically; returns the path, which the WARNING / ERROR of the event names together with `event.log_path`. Events of the local node (`data_path == "local"`, its logs already under `runs/<name>/logs/`) write no tail; the message names the local log.
3. **Node side**: `node.logs.tail_for_event(path) -> bytes` = `log_tail(path, LOG_TAIL_BYTES)` (empty bytes when the file is missing); every death / error event carries it and the absolute local path.
4. **Contexts**: a worker's `ChildError(kind="contract" | "player", message, context)` is forwarded as `Event(kind="child_error", error_kind=kind, message, context=f"node {n} ({name}), {context}", log_tail, log_path)`; the hub ERROR is `<message> (<context>); see <tail path> and <node log path>` and the run stops with exit 1 (T2.5's stop path).

- [ ] **Step 13: Focused tests, full fast suite + ruff**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_portability.py tests/unit/test_sp5_log_tails.py tests/unit/test_sp5_bad_data.py tests/integration/test_sp5_hub_away.py tests/integration/test_sp5_stop_and_exit.py tests/integration/test_sp2_lifecycle.py tests/unit/test_process_lifecycle.py tests/unit/test_sp4_run_state.py -q -rw -m "not slow"`, then the slow ones of `test_sp5_stop_and_exit.py`, then the full fast suite and ruff. Expected: green, zero warnings; `test_sp2_lifecycle.py::test_sigterm_stops_all_descendants_within_10s` and `test_killed_main_process_takes_all_descendants_with_it` unchanged and green (the local guarantee and parent death).

- [ ] **Step 14: Commit and push**

```bash
git add src/colosseum/node/agent.py src/colosseum/node/session.py src/colosseum/node/procs.py src/colosseum/node/logs.py \
        src/colosseum/hub/hub.py src/colosseum/utils/process.py src/colosseum/core/run_state.py tests/node_kit.py \
        tests/unit/test_sp5_portability.py tests/unit/test_sp5_log_tails.py tests/unit/test_sp5_bad_data.py \
        tests/integration/test_sp5_stop_and_exit.py
git commit -m "feat: distributed stop and exit codes, bad-data accounting, remote log tails, portable parent watch"
git push
```

---
## Contract notes

### Additions and deviations versus `00-overview.md`

1. **`colosseum.hub.placement` additions** (T4.1): `NodeCapacity.mps: bool = False` (a node's GPU slots are MPS, from `Hello.hardware.gpus[].kind`); `PlacementPlan.new(current)`; `parse_device`, `requested_slot`, `capacity_from_hello(node, name, tags, slots, gpu_kinds, *, unlimited)`; the pure learner table `LearnerTable` / `LearnerEntry` / `Reconciled` with statuses `pending` / `starting` / `running` / `relocating` / `failed` (`LEARNER_STATUSES`), `TABLE_STATE_VERSION = 1`, constants `CPU` / `CUDA` / `MPS`. `place(agents, nodes, *, current, relocate)` keeps the contract signature; `assigned` holds the kept and the new slots. `describe_plan` stays T7.2's.
2. **`LearnerTable.can_collect`** is true for `starting` and `running` (not only after the first announce): a starting learner's seats collect and it owns cells, because R11 holds such a lineup on the node until the learner's first weights are cached (no data is lost in the 30 s TTL after a learner starts).
3. **Parked weights** (T4.1): `VersionRegistry.park(info)` / `unpark(agent_id)` / `parked(agent_id)` (T2.1's class); `latest()` returns the parked info while parked; an accepted `announce` clears it; never saved in `hub_state.json`. A parked `VersionInfo` has `node = 0`, `generation = 0`, `relay_only = True`; `is_stale(agent, 0 | None, v)` is False. `BlobStore.parked_path(agent)` = `runs/<name>/blobs/latest-<agent>-parked.frame`; `Download(kind="latest")` serves it (with `since_version` → `not_modified`) before T3.1's relayed-latest path.
4. **Learner start points** (T4.1 – T4.3): `LearnerAssignment.weights_blob` ∈ {`""`, `init`, `start:resume`, `start:recovery`, `start:ckpt_v<N>`}; `trainer_state_blob` stays `""` (every `start:*` blob is one checkpoint frame from `encode_checkpoint(..., kind="recovery", generation=0)` for synthesized starts, or the stored recovery point); new `BlobRequest` kind **`start`** (ident `resume` / `recovery` / `ckpt_v<N>`); `BlobStore.start_path(agent, ident)` = `runs/<name>/blobs/start-<agent>-<ident>.frame`. The contract's `trainer_state` kind is not used by this part (T2.5 may keep it for something else; no learner start uses it).
5. **Proto** (T4.1): `Event.generation` (int64; learner events). Event kinds used: `proc_died`, `learner_failed`, `slot_disabled`, `child_error`, `warning` (contract) plus **`node_stopped`** and **`node_leaving`** (T4.4). `Command(kind="stop_learner")` uses `reason` ∈ {`superseded`, `stop_run`}; `Command(kind="stop_run")` carries `reason="grace=<sec>"`.
6. **`HubStateData.learners`** (T4.1): `LearnerTable.state(local_nodes=...)` (`{"version", "agents": {agent: {"status", "node", "device", "generation", "reason", "local"}}}`), loaded by T4.3 with `LearnerTable.load_resumed`.
7. **Hub counters** (in `hub_state.json` `counters`; T7.1 decides which become `system` keys): `lineup_redraw_failed` (T4.1), `dropped_uploads` `{stale_generation, invalid, not_newer}` (T4.1 / T4.2), `dropped_announcements`, `dropped_metrics`, `dropped_events` by reason (T4.4); `system` values `pending_agents` (contract) and **`failed_agents`** (T4.2).
8. **`colosseum.hub.hub.LINEUP_REDRAWS = 16`** (T4.1): lineups that name unplayable agents are drawn again (rejection sampling).
9. **`learner_process`** gains `send_final: Callable[[], bool] | None = None` (T4.1) and `clock: Callable[[], float] = time.monotonic` (T4.2); `ConnectionIntake.stop_reason` (the reason of the `Stop` it read). **`utils.seeding.learner_seed(seed, agent_index, generation=1)`** (T4.1).
10. **`colosseum.hub.recovery`** (T4.2, new module): `RecoveryStore`, `RecoveryPoint`, `read_frame_meta`, `RECOVERY_DIR` / `RECOVERY_FRAME` / `RECOVERY_META`. `RunDir.recovery_dir`.
11. **`colosseum.node.uploads`** (T4.2, new module): `UploadQueue`, `Upload`, `UPLOAD_CONCURRENCY = 2`; every checkpoint frame of a learner goes through it (one in flight per agent, finals never dropped). `colosseum.node.agent.PENDING_EVENTS_MAX = 100`.
12. **`CellTable.drop_worker(node, worker)`** (T4.2).
13. **`colosseum.hub.resume`** (T4.3, new module): `ResumePlan`, `prepare_resume`, `commit_resume`, `check_not_running`, `check_resumed_agents`, `config_diff`, `flatten`, `next_backup_path`, `start_env_steps`, `resume_port`, `AGENTS_STATE_FILE`, `DIFF_LINES_MAX`; a hook `_apply_agents_state` that T8.2 fills.
14. **Journal** (T4.3): `READABLE_SCHEMA_VERSIONS = (1, 2)` (the reader opens version 1 as it is; a writer migrates it); `meta.next_match_id` and explicit row ids (never reused); `MatchRow.node`; `MatchLogReader.results_after`, `folded_after`; `last_row`, `max_env_steps` on both reader and writer.
15. **Coordinator** (T4.3): `LeagueStateInfo`, `league_replay_window(info)`, `Coordinator.loaded_league_state`, `Coordinator.replay_results(results) -> (replayed, skipped)`, next to the contract's `league_state_journal_row()`. `load_league_state(path) -> bool` keeps its signature.
16. **Retention** (T4.3): only the newest final snapshot of the current run id is protected (deviation from SP4's "the final of the current run", needed because an in-place resume keeps the run id).
17. **`utils.process`** (T4.4): `PARENT_WATCH_SEC`, `start_parent_watchdog`, `watch_parent` (re-exported by `colosseum.node.procs`, the contract's location); `init_child_process` uses it. `core.run_state.process_start_ticks` reads `ps -o lstart=` on macOS.
18. **`HubSession.away_for()`, `HubSession.slow_retry(on)`** (T4.4).
19. **Test kits**: `fault_proxy.TcpFaultProxy.retarget(port)` (T4.3, if T3.2 lacks it); `node_kit.run_node_main(options_json)` (T4.4); the FakeNode / FakeHub / cluster_kit capabilities listed in the preamble.
20. **`hub --resume` without `--port`** listens on the port of the previous `hub.json` (`resume_port`), not `distributed.port`.

### Proposed rulings (what — why — cost if wrong)

- **P1 `learner.device: auto` takes CUDA slots only, never MPS** — matches `resolve_device("auto")` on one machine ("как `auto` локально"); MPS only by name — cost: a Mac-only cluster trains `auto` agents on the CPU unless `device: mps`.
- **P2 the local node (R18) spreads GPU learners over its GPUs, least used first** — one GPU: every learner on `cuda:0` as before; several GPUs: better use — cost: on a multi-GPU workstation the agents no longer all share `cuda:0` (a behaviour change only there).
- **P3 placement order: preferred nodes, then any node; ascending node number, then GPU index; agents in config order** — deterministic and testable — cost: no load-aware choice (spec: no rebalancing anyway).
- **P4 parked weights for agents without a learner; agents with no weights at all are excluded by redrawing lineups (16 draws, then no grant for that cell this pass, counted, WARNING)** — spec 2.6 «последними весами … иначе не выдаётся в составы» without changing the matchmaker API before T8.2; rejection sampling keeps the matchmaker's distribution conditioned on playable agents — cost: a mix that only names unplayable agents leaves cells idle (visible in `lineup_redraw_failed` and the WARNING).
- **P5 a starting learner collects and owns cells** (note 2) — its lineups wait for its weights on the node (R11), so nothing is collected with wrong weights — cost: cells of a learner that never comes up stay unplayed until their TTL.
- **P6 learner restarts are counted by the node but always re-assigned by the hub (new generation, start point from the hub); no learner restart while the hub is away** — a restart needs a new generation and a start point only the hub holds; uniform fencing — cost: a learner that crashes during a hub outage stays down until the hub returns.
- **P7 version floor = max(registry floor, newest stored snapshot + 1, recovery point + 1) on the hub, and the node's cached `latest` + 1** — snapshots and recovery points may carry versions no announcement ever named; without this a restarted learner could save a different `ckpt_v<N>` over an existing one — cost: gaps in version numbers (harmless).
- **P8 relocation waits `learner_relocate_after_sec` from the lost node's last heartbeat** — the same clock origin as `node_lost_after_sec` (≤ it by config validation) — cost: none.
- **P9 an in-place resume retries `failed` agents** — a resume is the owner's action, usually after a fix — cost: a still-broken agent spends its restart limit again.
- **P10 recovery points only from the current generation and not older than the stored one; the frame header wins over `meta.json`** — spec 7.2; one atomic `os.replace` per point — cost: none.
- **P11 a restart before the first recovery point starts from the hub-start source (`init`, the `training.resume_from` state, the in-place snapshot), not from a periodic snapshot of the run** — spec 5.2 literally — cost: up to `recovery_interval_sec` more training lost than with the newest snapshot.
- **P12 node checkpoint uploads: finals never dropped and first; one waiting recovery point and periodic snapshot per agent (a replaced periodic one is counted with a WARNING)** — closes the SP5 open item "final checkpoint lost behind a full queue" — cost: a hub slower than `checkpoint.interval` skips periodic snapshots (visible).
- **P13 journal row ids are never reused (`meta.next_match_id`)** — the R8 watermark must stay exact when compaction deleted the newest rows — cost: one meta read and write per append batch.
- **P14 an SP4 `league_state.json` (no watermark) replays the journal matches stamped after its `env_steps`** — SP4 accounted at intake; matches stamped after the save are exactly the unaccounted ones, up to matches stamped at the same count (not replayed) — cost: at most one metrics tick of SP4 matches missing from ELO / PFSP.
- **P15 only the newest final of the current run is protected** (note 16) — cost: none for runs without resume.
- **P16 an agent added by `--resume -c` counts its schedules from env step 0 until T8.2 adds start points** — T8.2 owns `agents_state.json` and start points; spec 6.3 is SP5b-movable — cost: a late agent's LR / λ / share schedules start mid-way.
- **P17 `--once` with the hub away past `hub_lost_timeout_sec` exits 1** — the run did not finish for this node — cost: none.
- **P18 the hub-lost collect rule is applied with `LearnerRoute(None)`: the worker's chunks for that agent are dropped and counted `no_learner`, matches are still played** — one existing mechanism, every loss counted — cost: CPU spent on chunks that are dropped for at most `hub_lost_timeout_sec − hub_lost_collect_sec`.
- **P19 a full cache stops new work (lineup requests, fetches, learner starts) and resumes when the free space is back, checked every heartbeat** — spec «узел не берёт работу» — cost: none.
- **P20 `hub --resume` without `--port` reuses the previous `hub.json` port** — nodes reconnect to the address they know although `--port` is not in the config — cost: a port taken meanwhile fails the start (ConfigError with the hint `--port`).
- **P21 learner seeds: generation 1 keeps SP4's `learner_seed(seed, index)`, later generations their own stream** — a restarted learner does not replay the same random stream; SP4 runs stay reproducible — cost: none.
- **P22 an in-place resume ignores `training.resume_from`** (INFO, WARNING when `-c` names another source) — the parent was imported at the first start; `training.resume_from` keeps its child-run meaning — cost: none.
- **P23 a stale-generation announcement also makes the hub stop that learner** — spec 7.2 «хаб велит его остановить», also mid-session — cost: none.
- **P24 a node that stops on its own (`node_leaving`) is lost at once and its learners relocate without the wait** — waiting 120 s for a deliberately stopped machine helps nothing — cost: a node restarted by hand within the wait gets its learner placed elsewhere.
- **P25 the SP1 rule "a dead child fails the run" is replaced for workers and learners by spec 7.1's restarts**: `test_sp2_lifecycle.py::test_killed_worker_gives_exit_1_and_points_to_its_log` becomes `test_a_killed_worker_is_restarted_and_its_log_named`, and `test_learner_crash_in_train_step_gives_exit_1` reaches exit 1 through the restart limit and "every trainable agent failed" — spec §8 / 7.1 — cost: a crash loop now takes 4 learner incarnations before the run stops.
- **P26 `train` without `--listen` stops with exit 1 when an agent cannot be placed on the machine** (e.g. `cuda` without CUDA) — no other node can ever join, so waiting would hang the run — cost: none.
- **P27 every trainable agent failed → the run stops with exit 1** — no agent can collect, so the budget would never be reached — cost: none.

### Assumptions about other parts

1. **T1.1**: a checkpoint frame's header is `kind="checkpoint"` with meta keys `kind` (`periodic` / `final` / `recovery`), `agent_id`, `generation`, `policy_version`, `final`; `decode_checkpoint` returns `make_checkpoint_payload`'s dict with `trainer_state_bytes` as `bytes`; `peek_meta` reads the header of a buffer that holds at least the header. `RecoveryStore` reads the 16 fixed bytes, then the header (`MAX_HEADER_BYTES` checked).
2. **T1.2**: `MatchResult.node` and `SeatResult.generation` exist; adding `Event.generation` is a compatible proto change (regenerated with `scripts/gen_proto.sh`).
3. **T1.4**: `LearnerConfig.placement` (`node`, `tags`, `strict`), `recovery_interval_sec`, `device` validation; `distributed.*` keys with the defaults this part names.
4. **T1.5**: `WeightCacheReader.latest_version(agent) -> (version, generation) | None`; `CacheFullError` with its hint; a free-space probe the tests can monkeypatch.
5. **T2.1**: `VersionRegistry()` with no arguments, `new_generation` returning 1 first, `announce` results as in the contract, `is_stale` false for generation 0 / None, `version_floor` = max accepted + 1; `BudgetCounter(base, sessions)`; `EpochAllocator(reserved_until, on_reserve)` starting above `reserved_until`; `CellTable` with `drop_node`; `HubStateData` serialisable with an added `learners` field; `ResultIntake.check` applying the cap rule.
6. **T2.2**: `ChunkIntake` / `LearnerOutputs` as in the contract (`get(timeout)` raising `queue.Empty`, `send_checkpoint(payload, *, kind, block)`); `learner_process(..., intake, outputs, version_floor, recovery_interval_sec)`; `ConnectionIntake` reads `Stop` (its `reason` is kept as `stop_reason`); each learner incarnation has its own intake address.
7. **T2.4 / T2.5**: `ChildSupervisor` with per-name incarnations, `may_restart`, `exited`, `stop_all`; the node's results buffer resends unacked batches after a reconnect; `Hello.learners` / `Welcome.learners`; the hub's handlers and main-loop deadlines named in the overview (R5); `_save_checkpoint` stamps the hub's env-step counter; `league_state.json` carries `journal_row` (T2.5 updates `test_sp4_league_state.py`'s key set); the local node's `data_path == "local"`; `hub.json` with `addresses`. Until T4.1, every learner is assigned with generation 1 on the local node; T4.1 replaces that assignment path.
8. **T3.1 / T3.2**: versions are broadcast in `Announce`; nodes fetch `latest` per `VersionInfo` (data port or relay) and fall back to the hub's `Download(kind="latest")`; `GetLatest` works without the hub; routes are derived from the agent's latest `VersionInfo` of a collecting generation; a seat's `collect` flag is set by the hub from "can collect"; the relay path is marked so the node knows which agents are relay-only.
9. **T5.1**: evaluators are `ChildSupervisor` children named `evaluator-<i>` and get T4.2's restart policy unchanged; block leases of a lost node are re-leased by T5.1 (not here).
10. **T7.1** reads the counters of note 7 from `hub_state.json` / the metrics hub and decides which become required `system` keys; **T7.2**'s `validate` prints the placement plan with `hub.placement`.
11. **T8.2** fills `colosseum.hub.resume._apply_agents_state` and agent start points; until then `agents_state.json` does not exist.
12. **T0.1**: `tests/game_helpers.copy_sp4_run(tmp_path)` copies an SP4 run dir with a schema-1 `matches.db`, `metrics.jsonl`, `league_state.json` (no `journal_row`), `run.json`, `run_state.json` and snapshots with and without `trainer_state.pt`; `sp4_manifest()` has `env_steps` and `final_version`.
