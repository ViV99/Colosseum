# SP5 Plan — Part 2: Hub and node (P2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** This part owns these parts of `docs/superpowers/specs/2026-10-17-sp5-distributed-design.md` (Russian; read the spec first, it is the binding authority) and of the controller's `plan-design.md`:
- **Block 1 (roles and topology):** the hub process (the evolved `Launcher`, R5) that owns the run dir and listens on one port; the node agent per machine whose children die with it; `colosseum train` = hub + local child node on a free loopback port (R12, R18), `runs/<name>/hub.json`, the machine's running-hubs registry; `colosseum hub -c cfg` and `train --listen [ADDR]` (T2.5). Learner placement on any node (T4.1), joining (T6.3), `hub --idle` (T5.2) are other parts.
- **Block 4.1 (session):** the gRPC session stream (`Hello` / `Welcome`, `Heartbeat`, lineups, results / `Ack`, metrics, versions, events, commands, announces), resume with resend of unacked batches, node loss after `node_lost_after_sec`, blobs in parts (T2.3, T2.4).
- **Block 4.2 (cells, epochs, lineups pull):** cells, epochs reserved in blocks (R7), TTL pull, lineup after cache (R11), the deficit rotation (it reproduces SP4's rotation on one node), no training cells while no agent can collect, hub-side matchmaker (T2.1, T2.4, T2.5). The evicted-snapshot hold is T3.1.
- **Block 4.3 (results):** node batches and its bounded buffer, ack after the journal (R8), globally unique uids with the epoch, intake checks, `env_steps` stamped by the hub (T2.1, T2.2, T2.4, T2.5). The journal `node` column is T4.3.
- **Block 4.4 (budget):** one budget counter per (node, session) with a base; LR progress from the hub (T2.1, T2.4, T2.5). Saving it for `--resume` is written here (`hub_state.json` on every metrics tick), reading it is T4.3.
- **Block 4.7 on one machine:** per-process connections without shared locks, one learner intake path (frame → decode → check → batch), end-to-end backpressure, at-most-once chunks with drop reasons, no collecting seats without a learner (T2.2, T2.4, T2.5). Cross-machine routing is T3.2, `max_policy_lag` T3.3.
- **Block 5.1:** SP3/SP4 logic in the hub unchanged (coordinator, retention, journal, live ratings, metrics, pulse child R6), `league_state.json` gains the journal watermark (T2.5).
- **The T2.5 switch** with criterion 2 (≥ 95 % of `main` `d61eb02` on one machine).
- Rulings in force here: R3–R13 (plan-design §4), R15–R18 (ledger).

**What the part delivers.** After T2.5:
- `colosseum train -c cfg` runs a `Hub` (the old `Launcher`'s logic, inputs from gRPC sessions) and one local node child that runs the workers and learners over Unix-socket connections and the shared-memory weight cache; `launcher.py` is gone and every SP1–SP4 test passes on the new path (ported where it poked `Launcher` internals);
- `colosseum hub -c cfg [--listen ADDR] [--port P] [--insecure]` serves nodes that connect in process (the `join` command is T6.3);
- training uids are `<run_id>/train/n<node>/w<W>_e<E>_g<epoch>_ep<K>`, episode seeds come from cell seeds (spec 7.6), every result is accounted exactly once after its journal commit and acked then (R8);
- the bench drives the `Hub`; criterion 2 is measured.

**Read `00-overview.md` first** (global constraints, file map, the binding interface contract). Names here are the contract's; additions and deviations are listed in `## Contract notes` at the end of this part.

**Conventions used in every task.**
- Run commands from the repository root as `env -u PYTHONPATH .venv/bin/python ...` / `env -u PYTHONPATH .venv/bin/ruff ...` (owner's machine: ROS sets `PYTHONPATH`); parallel worktrees: `PYTHONPATH=<worktree>/src` with the main checkout's venv (process README).
- "Full fast suite" = `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, zero warnings), then `env -u PYTHONPATH .venv/bin/ruff check .`.
- New test files start with `test_sp5_` and have unique basenames; support modules (`hub_kit`, `node_kit`) go into `known-first-party` of `pyproject.toml` in the task that creates them.
- Every file a test writes is under `tmp_path` (T1.3's autouse `_isolate_colosseum_dirs` redirects `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `COLOSSEUM_SHM_DIR`); at most 2 loaded CPU processes per fast test (hub and node processes are not counted), at most 3 per slow test; `OMP_NUM_THREADS=1`; CLI processes through `tests/cli_runner.py`; ports bound by the code under test on port 0.
- Durations and cadences use `time.monotonic()`; data is ordered by counters (epochs, seqs, versions, row ids), never by `ts`; no test asserts on wall-clock values or rates; waits poll a predicate with a bound (`_wait(pred, timeout)`).
- Messages go through `logging` (never `warnings.warn`), so `-rw` stays empty.
- Commits: conventional prefixes, no attribution lines; push after every task (`git push`, branch `sp5-distributed` or the worktree branch).

**Cross-task notes.**
- **Order:** T2.1 ‖ T2.2 (after group A) → T2.3 (after T1.2, T1.3, T2.1) → T2.4 (after T2.2, T2.3, T1.5) → T2.5 (alone). Until T2.5(b) the `Launcher` keeps working on the SP4 queue path (`QueueIntake` / `QueueOutputs`, T2.2), so every task ends green without the new path being used by `train`.
- **Part 01 names this part uses** (contract): `encode_frame` / `decode_frame` / `payload_byte_cap` / `FrameError` (T1.1); `encode_chunk` / `decode_chunk` / `ChunkMeta` / `ChunkError` / `ChunkValidator` / `WeightsMeta` / `encode_weights` / `decode_weights` / `encode_checkpoint` / `decode_checkpoint` (T1.1); `colosseum.net.proto.colosseum_v1_pb2(_grpc)`, `result_to_proto` / `results_from_batch` / `lineup_to_proto` / `lineup_from_proto` / `version_to_proto` / `version_from_proto`, `PROTOCOL_VERSION`, `Lineup.epoch`, `SeatResult.generation`, `MatchResult.node` (T1.2); `HubIdentity`, `NodeIdentity`, `NodeBook`, `Token`, `parse_token`, `AuthInterceptor`, `channel_options`, `server_options`, `hub_channel_credentials`, `fetch_hub_ca`, the `MD_*` metadata keys, `HUB_SERVER_NAME` (T1.3); `DistributedConfig` and the learner keys (T1.4); `WeightCacheWriter` / `WeightCacheReader` / `CacheKey` / `CachedWeights` / `cache_prefix` / `shm_dir` / `acquire_node_lock` / `cleanup_dead` / `CacheFullError` (T1.5). Where this part needs a little more of them it says so in `## Contract notes` (one proto field, one `NodeBook` method).
- **Proto additions made in this part** (T2.3 adds them to `colosseum_v1.proto` and regenerates with `scripts/gen_proto.sh`; new field numbers only, backward compatible): `Heartbeat.counters` (`map<string, int64>`: the node's cumulative counters, e.g. `lineup_fetch_failed`, `dropped_results_node_buffer`), `Event.kind` value `run_stopped` (a string value, no schema change). Nothing else in the proto changes.
- **What crosses which boundary:** node ↔ children: pickled `colosseum.net.local` dataclasses (R3) and frames; hub ↔ node: protobuf only; nothing of either contains a `torch.Tensor` (spawn kwargs included).
- **Generations:** every learner is generation 1 in this part (T2.5 calls `VersionRegistry.new_generation` once per agent); T4.1 makes generations real. The learner's generation rule for chunks (R16) and the hub's `stale_generation` result check already work with one generation.
- **One node with learners** until T3.1/T3.2: `hub` mode in T2.5 assigns every learner to the first node that connects without `--no-learners` (T4.1 replaces this by `placement.place`), and grants collecting seats only for agents whose learner runs on the requesting node (overview, "Node rules that cross tasks").
- **Pulse:** the SP4 pulse child stays a child of the hub (R6) until T5.1; its queues and `stop_event` (`mp.Event`) are the only `multiprocessing` primitives the hub keeps.
- **Shared-file edits after T2.5** follow the overview's rules (each later task edits only the handlers it names).

**Verified while writing this part** (scratch package under the session scratchpad, against the repo's `src` at `3022e6a`, Python 3.12.15, grpcio 1.84, with stand-ins for part 01's modules where noted):
- `colosseum.hub.state` (T2.1) exactly as printed below, with its 70 tests: all pass, ruff clean. The rotation test proves that the deficit rule with simultaneous expiry reproduces `trainable[(w * E + e + round) % n]` for 1–4 agents and five worker/env shapes; with staggered expiry (workers requesting one after another) it reproduces the start and keeps every agent's share within one cell afterwards, but not SP4's exact later sequence (the test asserts what holds; spec 4.2 claims only the simultaneous case). The result checks set `MatchResult.node` / `SeatResult.generation` as attributes, which works before and after T1.2 adds the fields.
- `colosseum.net.local` (T2.2) as printed, with the chunk codec replaced by a pickle stand-in for T1.1: listener / connect on Linux abstract sockets (wrong authkey refused, connect waits for a late listener, times out without one); `ConnectionIntake` reads every connection, stops reading while its deque is full (the senders block), and resumes; a sender `kill -9`ed in the middle of a 64 MiB frame closes only its own connection and counts one `disconnect` while another sender's frames keep arriving; a clean close loses nothing; an oversized frame is `invalid` and only its connection closes (the sender gets `BrokenPipeError`); misrouted, undecodable and stale-generation chunks are dropped by reason, older-generation chunks below the version floor are kept (R16); `LearnerEnvSteps`, `Stop` and the node's EOF reach the learner, `IntakeReport` reaches the node; a learner `kill -9`ed while a worker's sender is blocked in `send_bytes` makes the sender `broken` and every later frame is dropped and counted at once. Three repeated runs: 9 passed each, 8.3–8.5 s. `multiprocessing.connection.wait` on the listener's socket (`listener_socket`) is the way to stop an accept thread (closing does not wake `accept` on Linux).
- The `MatchLogWriter` token / commit change of T2.5 (R8) applied to a copy of `src`: the 15 SP4 writer tests pass (one line of `test_a_batch_still_in_flight_at_the_stop_deadline_is_counted_as_dropped` reads `writer._queue.qsize()`, which becomes `writer._queued`), `test_sp4_matchlog.py` and `test_sp4_journal_runs.py` pass (30 tests), and the 6 new token tests pass (whole-group commits in order with rows, a resend counted as duplicates, refusal without a drop count, failed commits on a write error and a dead writer, groups never split between appends, queued tokens counted as dropped at the stop).
- The session servicer pattern of T2.3 with a stand-alone generic gRPC handler: the server-side generator yields from a per-session queue while a reader thread consumes the request iterator and posts inbox events; a client cut makes the request iterator raise a bare `grpc.RpcError` **without `code()`** (catch the base class); `context.add_callback` fires on both a clean end and a cut.
- T2.2's edits of existing code applied to a copy of `src` (with stand-ins for T1.1 and T1.5): the new `learner_process` / `collect_batch` / `_publish_weights` / `set_policy_version`, `WeightPayload.generation`, `QueueIntake` / `QueueOutputs` in `_learner_main` and `run_distributed_learner`, and the test ports of T2.2 Step 8: 179 tests of the learner, launcher, pipeline, result-flow and pulse files pass; then **the whole fast suite on that copy, together with T2.5's writer change: 2166 passed** (the 2 failures were the scratch copy's missing `src/` layout in `test_infrastructure.py`). With the `MatchRunner` / `RolloutLoop` / seeding edits of T2.2 Steps 5–6 and T1.2's `Lineup.epoch` / `SeatResult.generation` / `MatchResult.node` added to the copy: `test_sp5_loop_epochs.py` (7), `test_sp5_learner_io.py` (3), `test_sp5_local_connections.py` (9), `test_sp5_worker_io.py` (9) and the SP2–SP4 loop contract files `test_rollout_loop_lineups.py`, `test_sp4_result_flow.py`, `test_sp2_eval_engine.py`: 57 passed.
- `grpc_util`'s blobs in parts (T2.3) against a compiled `BlobPart` with the contract's fields: 12 passed. The node's `ResultBuffer` / `CellTracker` (5), `ChildSupervisor` (4, spawned sleepers, `kill -9`, a SIGTERM-ignoring straggler killed within the bound), `hardware` (5) and `hub.registry` (2; it found that `getattr(path, "root", path)` resolves a `Path` to `/`) as printed.
- Not run here: the gRPC servicers against the real proto (T1.2 does not exist yet), the node agent with real children, and the hub. Their steps give behaviour, interfaces and test lists; the code shown for them (the session servicer, `_main_loop`, `_on_lineup_request`, `_on_result_batch` / `_drain_commits`) is a guide the implementer adapts.

**Traceability (spec item → task / test).** Every item of this part's blocks has a test; names are those of the steps below.

| Spec | Item | Task | Test(s) |
|---|---|---|---|
| 1 | hub owns the run dir and its state; `train` = hub + local child node; children die with the node | T2.5, T2.4 | `test_sp5_train_cli.py::test_train_runs_a_hub_and_a_local_node_to_the_budget`, `test_sp5_node_agent.py::test_children_die_with_their_node` |
| 1 | `train` without `--listen`: loopback ephemeral port, `hub.json`, running registry; `train --listen`; `hub` listens on all interfaces | T2.5 | `test_sp5_hub_registry.py::*`, `test_sp5_train_cli.py::test_train_listen_opens_a_tls_port_and_keeps_the_loopback`, `test_sp5_hub_mode.py::test_hub_trains_with_a_node_that_connects_in_process` |
| 1 | two `train` on one machine side by side | T2.5 | `test_sp5_two_trains_slow.py::test_two_trains_on_one_machine_run_side_by_side` (slow) |
| 4.1 | session messages, Hello / Welcome, resume, resend, dedup by (node, session, seq) | T2.3, T2.4 | `test_sp5_hub_server.py::test_a_cut_stream_resumes_and_resent_batches_are_accounted_once`, `test_sp5_node_agent.py::test_unacked_results_are_resent_after_a_reconnect_and_accounted_once` |
| 4.1 | node loss after `node_lost_after_sec`, early return keeps cells | T2.1, T2.3 | `test_sp5_hub_state.py::test_a_closed_stream_is_not_a_loss_only_silence_is`, `test_sp5_hub_server.py::test_a_silent_node_is_lost_and_comes_back_with_new_epochs` |
| 4.1 | blobs in parts | T2.3 | `test_sp5_blob_parts.py::*`, `test_sp5_hub_server.py::test_snapshots_are_converted_once_and_served_in_parts` |
| 4.2 | cells, epochs unique / monotonic, new per incarnation and per resume; R7 blocks | T2.1 | `test_sp5_hub_state.py::test_epochs_are_reserved_in_blocks_*`, `::test_a_cell_keeps_its_epoch_*`, `::test_an_epoch_reservation_survives_a_crash_of_the_hub` |
| 4.2 | TTL pull per cell, `match_refresh_interval_sec: 0` | T2.4, T2.5 | `test_sp5_node_books.py::test_cells_are_requested_once_until_answered_and_again_after_their_ttl`, `::test_whole_run_lineups_never_expire_*`, `test_sp5_hub_lineups.py::test_a_refresh_of_zero_grants_each_cell_once` |
| 4.2 | lineup after cache; failed fetch keeps the old lineup, `lineup_fetch_failed` | T2.4 | `test_sp5_node_agent.py::test_a_lineup_waits_for_its_snapshot_and_a_failed_fetch_keeps_the_old_one` |
| 4.2 | no latest fallback for a missing snapshot | T2.2, T2.5 | `test_sp5_loop_epochs.py::test_node_workers_never_replace_a_missing_snapshot` |
| 4.2 | deficit rotation, SP4 sequence on one node | T2.1, T2.5 | `test_sp5_hub_state.py::test_one_node_with_simultaneous_expiry_reproduces_the_sp4_rotation`, `test_sp5_hub_lineups.py::test_the_hub_rotates_owners_like_sp4_on_one_node` |
| 4.2 | no training cells while no agent can collect | T2.1, T2.5 | `test_sp5_hub_lineups.py::test_no_lineup_is_granted_while_no_agent_can_collect` |
| 4.2 | matchmaker / PFSP / `shuffle_seats` in the hub | T2.5 | `test_sp5_hub_lineups.py::test_lineups_come_from_the_coordinators_matchmaker` |
| 4.2 | a lost node's cells vanish; unfinished matches never reported | T2.3, T2.5 | `test_sp5_hub_server.py::test_a_silent_node_is_lost_and_comes_back_with_new_epochs` |
| 4.3 | node batches (256 / 0.25 s), bounded buffer, overflow counted + WARNING | T2.4 | `test_sp5_node_books.py::test_results_are_batched_*`, `::test_a_full_buffer_drops_the_new_batch_and_warns_once`, `test_sp5_node_agent.py::test_results_wait_in_the_buffer_while_the_hub_is_away` |
| 4.3 | ack after the journal commit (R8); accounting in commit order | T2.5 | `test_sp5_writer_commits.py::*`, `test_sp5_hub_results.py::*` |
| 4.3 | uid `<run>/train/n<node>/w_e_g_ep`, epoch-dependent seeds | T2.2 | `test_sp5_loop_epochs.py::*` |
| 4.3 | intake checks (`invalid`, `stale_generation`), one WARNING per node, never crashes | T2.1, T2.5 | `test_sp5_hub_state.py::test_every_malformed_result_is_invalid_and_nothing_raises`, `test_sp5_hub_results.py::test_invalid_results_are_dropped_counted_and_warned_once_per_node` |
| 4.3 | `env_steps` stamped by the hub at intake | T2.5 | `test_sp5_hub_results.py::test_results_are_stamped_with_the_hubs_env_steps_at_intake` |
| 4.4 | one budget, cumulative per (node, session), base + increments | T2.1, T2.5 | `test_sp5_hub_state.py::test_the_budget_counts_*`, `::test_a_resumed_budget_continues_*`, `test_sp5_hub_results.py::test_heartbeats_drive_the_budget_and_the_stop` |
| 4.4 | hub is the only stop authority; LR progress from the hub | T2.4, T2.5 | `test_sp5_node_agent.py::test_a_learner_trains_on_the_node_and_uploads_its_snapshots`, `test_sp5_hub_results.py::test_heartbeats_drive_the_budget_and_the_stop` |
| 4.4 | `hub_state.json` saved every metrics tick | T2.1, T2.5 | `test_sp5_hub_state.py::test_hub_state_round_trips_atomically`, `test_sp5_train_cli.py::test_train_runs_a_hub_and_a_local_node_to_the_budget` |
| 4.7 | per-process connections, no shared locks, `kill -9` blocks nobody | T2.2, T2.4 | `test_sp5_local_connections.py::test_a_sender_killed_mid_frame_closes_only_its_connection`, `::test_a_dead_learner_never_blocks_a_worker_send`, `test_sp5_node_agent.py::test_a_killed_learner_blocks_neither_the_worker_nor_the_node` |
| 4.7 | one intake path (frame → decode → check → batch) | T2.2 | `test_sp5_learner_io.py::test_the_learner_trains_from_frames_through_its_intake` |
| 4.7 | backpressure end to end | T2.2 | `test_sp5_local_connections.py::test_the_intake_reads_every_connection_and_applies_backpressure` |
| 4.7 | at most once; drops by reason | T2.2 | `test_sp5_local_connections.py::test_misrouted_garbage_and_stale_generation_chunks_are_dropped_by_reason`, `test_sp5_worker_io.py::test_chunks_without_a_learner_are_dropped_and_counted` |
| 4.7 | no collecting seats for agents without a learner | T2.1, T2.5 | `test_sp5_hub_state.py::test_restrict_collecting_*`, `test_sp5_hub_lineups.py::test_seats_of_agents_without_a_learner_do_not_collect` |
| 4.7 | speed on one machine (criterion 2) | T2.5 | the measurement in T2.5 step "Criterion 2" |
| 5.1 | SP3/SP4 logic unchanged in the hub | T2.5 | every ported SP2–SP4 test (T2.5(b) list), the slow suite |
| 5.1 | `league_state.json` journal watermark | T2.5 | `test_sp5_hub_results.py::test_the_league_state_records_the_last_accounted_journal_row` |
| 7.6 | cell seed `derive_seed(seed, epoch, node, worker, env)`, episode seed `SeedSequence([cell seed, K])` | T2.2 | `test_sp5_loop_epochs.py::test_cell_seeds_are_distinct_per_epoch_node_worker_env` |
| 7.7 (local) | local node logs in `runs/<name>/logs/` | T2.5 | `test_sp5_train_cli.py::test_train_runs_a_hub_and_a_local_node_to_the_budget` |

---


### Task T2.1: `colosseum.hub.state` — node registry, cells and epochs, deficit rotation, budget, versions and generations, result intake, `hub_state.json`

Spec 4.1 (node loss and return), 4.2 (cells, epochs, rotation by deficit, no training cells without a collecting agent), 4.3 (intake checks, dedup by (node, session, seq)), 4.4 (budget per (node, session) with a base), 4.5 / 7.2 (version registry, generations), 4.7 (seats of agents without a learner do not collect), 7.6 (monotonic hub clock); rulings R7 (epochs in blocks), R8 (ack bookkeeping), R9 (version floor), R16 (cap rule). A pure module: the hub's main loop (T2.5) owns every object here; the gRPC servicers (T2.3) and the minimal test loop (`hub_kit.MiniHubLoop`) use them too. It imports `colosseum.core.types` and `colosseum.arena.entities` only (no protobuf, no gRPC, no threads).

**Files:**
- Create: `src/colosseum/hub/__init__.py`, `src/colosseum/hub/state.py`
- Test: `tests/unit/test_sp5_hub_state.py`

**Interfaces:**
- Consumes: `Lineup`, `SeatAssignment`, `MatchResult` with T1.2's `MatchResult.node` and `SeatResult.generation`, `LATEST_NETWORK_ID`, `FIXED_NETWORK_ID` (`colosseum.core.types`); `snapshot_version` (`colosseum.arena.entities`); a `GameSpec` (duck-typed: `layouts`, `outcome_kind`).
- Produces (contract "colosseum.hub.state", plus additions marked *): `EPOCH_BLOCK`, `EVICTED_HOLD_FACTOR`, `HUB_STATE_FILE`, `HUB_STATE_VERSION`, *`DROP_REASONS`; *`NodeHello`; `NodeEntry`; `NodeRegistry` (`connect(..., *, data_path="relay")`, `heartbeat`, *`disconnect`, `lost`, *`remove`, `online`, `get`, *`by_id`, *`all`); `CellKey`; `EpochAllocator` (*`reserved_until`, *`block=` keyword); `OwnerRotation`; `CellEntry`; `CellTable` (*constructor `CellTable(epochs)`, `request`, *`grant`, *`owners`, `drop_node`, *`drop_cells`, *`get`, `live`); *`restrict_collecting`; `BudgetCounter` (*`add`); `VersionInfo` (*`to_dict` / `from_dict`); `VersionRegistry`; `ResultIntake` (`check(..., *, node=None, uid_prefix=None)`, *`accounts_at_intake`, *`count_drop`, *`dropped`, *`open`, `begin`, *`is_acked`, `committed`, `failed`, *`forget_node`); `HubStateData`, `save_hub_state`, `load_hub_state`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp5_hub_state.py`:
```python
"""Pure hub state (SP5 T2.1; spec 4.1-4.5, 4.7, 7.2, 7.6; rulings R7, R8, R16): node registry, epochs reserved in
blocks, cells, owner rotation by deficit (it reproduces SP4's rotation on one node), the budget per (node, session),
versions and generations with the cap rule, result checks and batch bookkeeping, hub_state.json."""
from __future__ import annotations

import dataclasses
import json
import math
import random

import pytest

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.state import (
    EPOCH_BLOCK,
    HUB_STATE_FILE,
    HUB_STATE_VERSION,
    BudgetCounter,
    CellKey,
    CellTable,
    EpochAllocator,
    HubStateData,
    NodeHello,
    NodeRegistry,
    OwnerRotation,
    ResultIntake,
    VersionInfo,
    VersionRegistry,
    load_hub_state,
    restrict_collecting,
    save_hub_state,
)
from game_helpers import TurnTakingGame, make_match_result

HELLO = NodeHello(hardware={"logical_cores": 4}, slots={"workers": 3}, tags=("gpu",), data_addresses=())


# ---- nodes ----------------------------------------------------------------------------------------------------


def test_a_node_is_new_then_resumed_with_its_session_then_returned_after_a_restart():
    reg = NodeRegistry()
    assert reg.connect("n1", 1, "pc", "s1", HELLO, now=0.0) == "new"
    assert reg.connect("n1", 1, "pc", "s1", HELLO, now=1.0) == "resumed"     # the stream was cut and came back
    assert reg.connect("n1", 1, "pc", "s2", HELLO, now=2.0) == "returned"    # the node process restarted
    entry = reg.by_id("n1")
    assert (entry.number, entry.name, entry.session_id, entry.state, entry.tags) == (1, "pc", "s2", "online", ("gpu",))
    assert reg.get(1) is entry and reg.get(2) is None and reg.online() == [entry]


def test_a_closed_stream_is_not_a_loss_only_silence_is():
    reg = NodeRegistry()
    reg.connect("n1", 1, "pc", "s1", HELLO, now=0.0)
    reg.disconnect("n1", "s1")
    assert reg.by_id("n1").state == "online" and reg.lost(now=5.0, after_sec=10.0) == []
    assert reg.connect("n1", 1, "pc", "s1", HELLO, now=6.0) == "returned"    # session id cleared by the close
    assert reg.heartbeat("n1", "s1", now=7.0)
    assert [e.node_id for e in reg.lost(now=17.5, after_sec=10.0)] == ["n1"]
    assert reg.lost(now=30.0, after_sec=10.0) == []                          # each loss once
    assert reg.connect("n1", 1, "pc", "s1", HELLO, now=31.0) == "returned"   # a lost node comes back as returned


def test_heartbeats_of_another_session_or_an_unknown_node_are_ignored():
    reg = NodeRegistry()
    reg.connect("n1", 1, "pc", "s1", HELLO, now=0.0)
    assert not reg.heartbeat("n1", "old", now=9.0) and not reg.heartbeat("nx", "s1", now=9.0)
    assert reg.by_id("n1").last_heartbeat == 0.0
    reg.remove("n1")
    assert reg.by_id("n1").state == "removed" and reg.online() == [] and not reg.heartbeat("n1", "s1", now=10.0)


# ---- epochs and cells -------------------------------------------------------------------------------------------


def test_epochs_are_reserved_in_blocks_before_they_are_handed_out():
    reserved: list[int] = []
    epochs = EpochAllocator(0, reserved.append)
    assert epochs.next() == 1 and reserved == [EPOCH_BLOCK]
    handed = [epochs.next() for _ in range(EPOCH_BLOCK - 1)]
    assert handed[-1] == EPOCH_BLOCK and reserved == [EPOCH_BLOCK]
    assert epochs.next() == EPOCH_BLOCK + 1 and reserved == [EPOCH_BLOCK, 2 * EPOCH_BLOCK]


def test_a_resumed_allocator_starts_after_the_saved_reservation():
    reserved: list[int] = []
    epochs = EpochAllocator(EPOCH_BLOCK, reserved.append)
    assert epochs.next() == EPOCH_BLOCK + 1 and reserved == [2 * EPOCH_BLOCK]


def test_a_failing_reservation_hands_out_no_epoch():
    calls = []

    def on_reserve(until):
        calls.append(until)
        if len(calls) == 1:
            raise OSError("disk full")

    epochs = EpochAllocator(0, on_reserve)
    with pytest.raises(OSError):
        epochs.next()
    assert epochs.reserved_until == 0
    assert epochs.next() == 1 and calls == [EPOCH_BLOCK, EPOCH_BLOCK]


def _table(reserved_until: int = 0) -> CellTable:
    return CellTable(EpochAllocator(reserved_until, lambda until: None))


def test_a_cell_keeps_its_epoch_within_an_incarnation_and_gets_a_new_one_otherwise():
    table, key = _table(), CellKey(1, 0, 2)
    (first,) = table.request(1, [(key, 0, 1)], now=0.0)
    (again,) = table.request(1, [(key, first.epoch, 1)], now=1.0)
    assert again.epoch == first.epoch
    (restarted,) = table.request(1, [(key, first.epoch, 2)], now=2.0)       # a new worker incarnation
    assert restarted.epoch > first.epoch and restarted.incarnation == 2
    (unknown,) = table.request(1, [(key, 999, 2)], now=3.0)                 # an epoch this table never granted
    assert unknown.epoch > restarted.epoch


def test_a_new_hub_process_never_keeps_an_epoch_of_the_old_one():
    (entry,) = _table(EPOCH_BLOCK).request(1, [(CellKey(1, 0, 0), 5, 1)], now=0.0)
    assert entry.epoch == EPOCH_BLOCK + 1


def test_cells_are_per_node_and_a_lost_node_takes_only_its_own():
    table = _table()
    table.request(1, [(CellKey(1, 0, e), 0, 1) for e in range(2)], now=0.0)
    table.request(2, [(CellKey(2, 0, 0), 0, 1)], now=0.0)
    with pytest.raises(ValueError):
        table.request(1, [(CellKey(2, 0, 1), 0, 1)], now=0.0)
    assert [e.key for e in table.drop_node(1)] == [CellKey(1, 0, 0), CellKey(1, 0, 1)]
    assert [e.key for e in table.live()] == [CellKey(2, 0, 0)]


def test_a_grant_moves_the_owner_to_previous_and_a_new_incarnation_keeps_it_as_previous():
    table, key = _table(), CellKey(1, 0, 0)
    table.request(1, [(key, 0, 1)], now=0.0)
    table.grant(key, "a", 30.0, now=0.0)
    entry = table.grant(key, "b", 30.0, now=30.0)
    assert (entry.owner, entry.previous, entry.ttl_sec, entry.granted_at) == ("b", "a", 30.0, 30.0)
    assert table.owners() == {key: "b"} and table.owners(exclude=[key]) == {}
    (fresh,) = table.request(1, [(key, entry.epoch, 2)], now=31.0)
    assert (fresh.owner, fresh.previous) == (None, "b")


# ---- owner rotation ---------------------------------------------------------------------------------------------


def _sp4_owner(agents, w, e, envs, rnd):
    return agents[(w * envs + e + rnd) % len(agents)]


@pytest.mark.parametrize("n_agents", [1, 2, 3, 4])
@pytest.mark.parametrize("workers,envs", [(1, 1), (1, 3), (2, 4), (3, 8), (2, 5)])
def test_one_node_with_simultaneous_expiry_reproduces_the_sp4_rotation(n_agents, workers, envs):
    agents = [f"a{i}" for i in range(n_agents)]
    rotation = OwnerRotation(agents)
    cells = [CellKey(1, w, e) for w in range(workers) for e in range(envs)]
    previous: dict[CellKey, str] = {}
    for rnd in range(6):
        owners = rotation.assign(cells, collecting=agents, owners={}, previous=previous)
        assert owners == {c: _sp4_owner(agents, c.worker, c.env, envs, rnd) for c in cells}, rnd
        previous = owners


@pytest.mark.parametrize("n_agents", [2, 3])
def test_workers_that_request_one_after_another_keep_the_shares_balanced(n_agents):
    """Staggered expiry: the start reproduces SP4 (workers start one after another); later rounds keep the shares
    within one cell of each other (the deficit rule prefers balance over SP4's exact sequence)."""
    agents = [f"a{i}" for i in range(n_agents)]
    rotation, workers, envs = OwnerRotation(agents), 3, 4
    current: dict[CellKey, str] = {}
    for rnd in range(5):
        for w in range(workers):                                   # each worker's cells expire on their own
            mine = [CellKey(1, w, e) for e in range(envs)]
            others = {k: v for k, v in current.items() if k.worker != w}
            got = rotation.assign(mine, collecting=agents, owners=others, previous=current)
            if rnd == 0:
                assert got == {c: _sp4_owner(agents, w, c.env, envs, 0) for c in mine}, w
            current.update(got)
            counts = [list(current.values()).count(a) for a in agents]
            if rnd > 0:
                assert max(counts) - min(counts) <= 1, (rnd, w, counts)


def test_shares_stay_balanced_while_nodes_join_and_leave():
    agents = ["a", "b", "c"]
    rotation, rng = OwnerRotation(agents), random.Random(0)
    current: dict[CellKey, str] = {}
    nodes = {1: 5}
    for step in range(40):
        if step % 7 == 3:
            nodes[len(nodes) + 1] = rng.randint(1, 9)             # a node joins
        if step % 11 == 10 and len(nodes) > 1:
            gone = rng.choice(sorted(nodes))
            del nodes[gone]                                        # a node is lost: its cells vanish
            current = {k: v for k, v in current.items() if k.node != gone}
        cells = [CellKey(n, 0, e) for n in sorted(nodes) for e in range(nodes[n])]
        current = rotation.assign(cells, collecting=agents, owners={}, previous=current)   # a full refresh
        counts = [list(current.values()).count(a) for a in agents]
        assert max(counts) - min(counts) <= 1, (step, counts)


def test_new_cells_go_to_the_agents_that_own_the_fewest():
    rotation = OwnerRotation(["a", "b", "c"])
    others = {CellKey(1, 0, e): o for e, o in enumerate("aaabbc")}
    got = rotation.assign([CellKey(2, 0, 0), CellKey(2, 0, 1), CellKey(2, 0, 2)], collecting=["a", "b", "c"],
                          owners=others, previous={})
    assert list(got.values()) == ["c", "b", "c"]


def test_only_collecting_agents_own_cells_and_without_any_there_is_no_owner():
    rotation = OwnerRotation(["a", "b", "c"])
    cells = [CellKey(1, 0, e) for e in range(4)]
    assert rotation.assign(cells, collecting=[], owners={}, previous={}) == {}
    got = rotation.assign(cells, collecting=["c", "a"], owners={CellKey(2, 0, 0): "b"}, previous={})
    assert list(got.values()) == ["a", "c", "a", "c"]                     # config order, b ignored


def test_restrict_collecting_turns_off_seats_of_agents_without_a_learner():
    lineup = Lineup("2p", [SeatAssignment("a", collect=True, source="owner"),
                           SeatAssignment("b", collect=True, source="rivals")], seed=None)
    assert restrict_collecting(lineup, ["a", "b"]) is lineup
    out = restrict_collecting(lineup, ["a"])
    assert [s.collect for s in out.seats] == [True, False] and [s.collect for s in lineup.seats] == [True, True]
    assert out.seats[1].source == "rivals" and out.layout == "2p"


# ---- budget -------------------------------------------------------------------------------------------------


def test_the_budget_counts_increments_per_session_and_never_twice():
    budget = BudgetCounter()
    assert budget.report("n1", "s1", 100) == 100 and budget.report("n1", "s1", 150) == 50
    assert budget.report("n1", "s1", 120) == 0                     # out of order / repeated: nothing
    assert budget.report("n1", "s2", 30) == 30                    # the node restarted: a new session from 0
    assert budget.report("n2", "s1", 7) == 7
    assert budget.value == 187
    assert budget.state() == {"env_steps": 187, "sessions": {"n1/s1": 150, "n1/s2": 30, "n2/s1": 7}}


def test_a_resumed_budget_continues_from_the_saved_sessions():
    saved = BudgetCounter()
    saved.report("n1", "s1", 150)
    budget = BudgetCounter(saved.state()["env_steps"], saved.state()["sessions"])
    assert budget.report("n1", "s1", 170) == 20 and budget.report("n1", "s9", 5) == 5 and budget.value == 175
    budget.add(25)
    assert budget.value == 200
    with pytest.raises(ValueError):
        BudgetCounter(-1)


# ---- versions and generations -------------------------------------------------------------------------------


def _info(version: int, generation: int, agent: str = "a") -> VersionInfo:
    return VersionInfo(agent_id=agent, generation=generation, version=version, size=10, digest="d" * 32, node=1)


def test_versions_grow_and_come_only_from_the_current_generation():
    reg = VersionRegistry()
    assert reg.generation("a") == 0 and reg.version_floor("a") == 0 and reg.latest("a") is None
    assert reg.announce(_info(0, 1)) == "invalid"                  # no generation assigned yet
    assert reg.new_generation("a") == 1
    assert reg.announce(_info(0, 1)) == "accepted" and reg.announce(_info(0, 1)) == "not_newer"
    assert reg.announce(_info(5, 1)) == "accepted" and reg.announce(_info(6, 2)) == "invalid"
    assert reg.new_generation("a") == 2
    assert reg.announce(_info(6, 1)) == "stale_generation"         # the old learner after a network split
    assert reg.announce(_info(3, 2)) == "not_newer"                # versions never go back (R9)
    assert reg.version_floor("a") == 6
    assert reg.announce(_info(6, 2)) == "accepted" and reg.latest("a") == _info(6, 2)


def test_a_superseded_generation_is_stale_only_above_its_cap():
    reg = VersionRegistry()
    reg.new_generation("a")
    for v in (0, 3, 5):
        reg.announce(_info(v, 1))
    reg.new_generation("a")
    assert not reg.is_stale("a", 1, 5) and not reg.is_stale("a", 1, 0)
    assert reg.is_stale("a", 1, 6) and reg.is_stale("a", 1, None)
    assert not reg.is_stale("a", 2, 99) and not reg.is_stale("a", None, 99) and not reg.is_stale("a", 0, 99)
    assert reg.is_stale("a", 3, 1) and reg.is_stale("other", 1, 0)
    reg.new_generation("a")                                        # generation 2 announced nothing: cap -1
    assert reg.is_stale("a", 2, 0)


def test_invalid_announcements_are_refused():
    reg = VersionRegistry()
    reg.new_generation("a")
    for bad in (dataclasses.replace(_info(1, 1), version=-1), dataclasses.replace(_info(1, 1), digest=""),
                dataclasses.replace(_info(1, 1), size=-5), dataclasses.replace(_info(1, 1), generation=0)):
        assert reg.announce(bad) == "invalid"


def test_the_version_registry_round_trips_through_its_state():
    reg = VersionRegistry()
    reg.new_generation("a")
    reg.announce(_info(4, 1))
    reg.new_generation("a")
    reg.announce(_info(7, 2))
    back = VersionRegistry.from_state(json.loads(json.dumps(reg.state())))
    assert back.state() == reg.state() and back.latest("a") == _info(7, 2)
    assert back.is_stale("a", 1, 5) and not back.is_stale("a", 1, 4) and back.version_floor("a") == 8


# ---- results ------------------------------------------------------------------------------------------------

SPEC = TurnTakingGame().spec
AGENTS = ("a", "random")
PREFIX = "R/train/n3/"


def _result(**changes):
    seats = [(0, "a", "latest", "player", 4, 5), (1, "random", "fixed", "player", None, None)]
    result = make_match_result(seats, uid=f"{PREFIX}w0_e1_g7_ep2", match_id="w0_e1_g7_ep2")
    result.node = changes.pop("node", 3)
    for seat in result.seats:
        seat.generation = 1 if seat.network_id == "latest" else None
    for key, value in changes.items():
        setattr(result, key, value)
    return result


def _versions() -> VersionRegistry:
    reg = VersionRegistry()
    reg.new_generation("a")
    return reg


def _check(result, versions=None):
    return ResultIntake(journal=True).check(result, spec=SPEC, agents=AGENTS, versions=versions or _versions(),
                                            node=3, uid_prefix=PREFIX)


def test_a_valid_result_passes_the_checks():
    assert _check(_result()) is None


def _seat(i, **changes):
    def mutate(result):
        for key, value in changes.items():
            setattr(result.seats[i], key, value)
        return result
    return mutate


@pytest.mark.parametrize("mutate", [
    lambda r: setattr(r, "layout", "9p") or r,
    lambda r: setattr(r, "outcome_kind", "rank") or r,
    lambda r: setattr(r, "seats", r.seats[:1]) or r,
    lambda r: setattr(r, "seats", None) or r,                       # raises inside: still only "invalid"
    lambda r: setattr(r, "source", "pulse") or r,
    lambda r: setattr(r, "node", 4) or r,
    lambda r: setattr(r, "uid", "R/train/n4/w0_e1_g7_ep2") or r,
    lambda r: setattr(r, "uid", PREFIX) or r,
    lambda r: setattr(r, "episode_length", -1) or r,
    lambda r: setattr(r.teams[0], "rank", math.nan) or r,
    lambda r: setattr(r.teams[1], "team", 5) or r,
    _seat(0, seat=1), _seat(0, role="hunter"), _seat(1, team=0), _seat(0, agent_id="ghost"),
    _seat(0, network_id="ckpt_vx"), _seat(0, reward=math.inf), _seat(0, version_end=-1),
])
def test_every_malformed_result_is_invalid_and_nothing_raises(mutate):
    assert _check(mutate(_result())) == "invalid"


def test_a_latest_seat_played_by_a_superseded_generation_above_its_cap_is_stale():
    versions = _versions()
    versions.announce(_info(5, 1))
    versions.new_generation("a")
    assert _check(_result(), versions) is None                     # version_end 5 <= cap(1) = 5
    stale = _result()
    stale.seats[0].version_end = 6
    assert _check(stale, versions) == "stale_generation"


def test_batches_are_new_once_and_acked_only_after_their_commit():
    intake = ResultIntake(journal=True)
    assert not intake.accounts_at_intake and ResultIntake(journal=False).accounts_at_intake
    assert intake.begin("n", "s", 1) and not intake.begin("n", "s", 1)      # in flight: a resend is skipped
    assert not intake.is_acked("n", "s", 1)
    intake.committed("n", "s", 1)
    assert intake.is_acked("n", "s", 1) and not intake.begin("n", "s", 1)
    assert intake.begin("n", "s", 2)
    intake.failed("n", "s", 2)                                    # the writer could not take it
    assert not intake.is_acked("n", "s", 2) and intake.begin("n", "s", 2)  # the node's resend is new again
    assert intake.begin("n", "other-session", 1)


def test_the_ack_watermark_advances_over_out_of_order_commits():
    intake = ResultIntake(journal=True)
    for seq in (1, 2, 3):
        intake.begin("n", "s", seq)
    intake.committed("n", "s", 3)
    intake.committed("n", "s", 1)
    assert intake.is_acked("n", "s", 3) and not intake.is_acked("n", "s", 2)
    intake.committed("n", "s", 2)
    seqs = intake._sessions[("n", "s")]
    assert (seqs.low, seqs.acked, seqs.in_flight) == (3, set(), set())


def test_open_takes_the_nodes_last_acked_seq():
    intake = ResultIntake(journal=True)
    intake.open("n", "s", 5)
    assert intake.is_acked("n", "s", 3) and not intake.begin("n", "s", 5) and intake.begin("n", "s", 6)


def test_drops_are_counted_by_reason_and_warned_once_per_node_and_reason():
    intake = ResultIntake(journal=True)
    assert intake.count_drop(3, "invalid", 2) and not intake.count_drop(3, "invalid")
    assert intake.count_drop(4, "invalid") and intake.count_drop(3, "stale_generation")
    assert intake.dropped == {"invalid": 4, "stale_generation": 1}


# ---- hub_state.json -----------------------------------------------------------------------------------------


def _data(**changes) -> HubStateData:
    base = dict(version=HUB_STATE_VERSION, run_id="R", env_steps=187, sessions={"n1/s1": 150},
                epoch_reserved_until=EPOCH_BLOCK, versions={}, counters={"worker_restarts": 0}, nodes_lost=1)
    return HubStateData(**{**base, **changes})


@pytest.mark.parametrize("fsync", [False, True])
def test_hub_state_round_trips_atomically(tmp_path, fsync):
    path = tmp_path / HUB_STATE_FILE
    save_hub_state(path, _data(), fsync=fsync)
    assert load_hub_state(path) == _data()
    assert sorted(p.name for p in tmp_path.iterdir()) == [HUB_STATE_FILE]


def test_a_missing_hub_state_is_none_and_a_malformed_one_a_value_error(tmp_path):
    path = tmp_path / HUB_STATE_FILE
    assert load_hub_state(path) is None
    path.write_text("{not json")
    with pytest.raises(ValueError, match="malformed"):
        load_hub_state(path)
    path.write_text(json.dumps({**dataclasses.asdict(_data()), "version": 99}))
    with pytest.raises(ValueError, match="version 99"):
        load_hub_state(path)


def test_an_epoch_reservation_survives_a_crash_of_the_hub(tmp_path):
    path = tmp_path / HUB_STATE_FILE
    reg = VersionRegistry()

    def reserve(until: int) -> None:
        save_hub_state(path, _data(epoch_reserved_until=until, versions=reg.state()), fsync=True)

    first = EpochAllocator(0, reserve)
    handed = [first.next() for _ in range(3)]                      # then the hub is killed: nothing else saved
    resumed = EpochAllocator(load_hub_state(path).epoch_reserved_until, reserve)
    assert resumed.next() > max(handed)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_hub_state.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.hub'`.

- [ ] **Step 3: Write `src/colosseum/hub/__init__.py` and `src/colosseum/hub/state.py`**

`src/colosseum/hub/__init__.py`:
```python
"""The hub process (SP5 spec block 1): pure state (``state``), gRPC servicers (``server``), the main loop (``hub``),
the machine's running-hubs registry (``registry``), the admin client (``admin``); later placement (T4.1) and
evaluation (T5.1)."""
```

`src/colosseum/hub/state.py` (complete; verified in the scratch package):
```python
"""Pure hub state (SP5 spec blocks 4.1-4.5, 4.7, 7.2, 7.6; plan rulings R7, R8, R16).

Everything here is owned by the hub's main loop (``colosseum.hub.hub.Hub``) and touched by no other thread; nothing
here does I/O except ``save_hub_state`` / ``load_hub_state`` and the ``on_reserve`` callback of ``EpochAllocator``.
Times are the hub's ``time.monotonic()`` values passed in by the caller (``now``); clocks of other machines never
appear here, only durations (``ttl_sec``).

- ``NodeRegistry``: the nodes the hub knows in this hub process: online / lost / removed, the session id of the live
  stream, the last heartbeat; ``connect`` tells a new node from a resumed stream and a returned node (spec 4.1).
- ``EpochAllocator`` (R7): epochs are reserved in blocks of ``EPOCH_BLOCK``; ``on_reserve(until)`` (the hub writes
  ``hub_state.json`` with fsync) runs before the first epoch of a new block is handed out, so a crash never reissues
  an epoch; a resumed hub starts after the saved reservation.
- ``CellTable``: cells (node, worker, env) with their epoch, the worker incarnation they belong to, their owner and
  previous owner, and when they were granted (spec 4.2). A request keeps a cell's epoch only for the same incarnation
  and an epoch this hub process granted; every other request gets a new epoch.
- ``OwnerRotation``: the deficit rule of spec 4.2. Every collecting agent's target share is equal, so the agent with
  the largest deficit is the one that owns the fewest live collecting cells; ties go to the first tied agent after the
  cell's previous owner in config order (no previous owner: the first tied agent). With one node whose cells all
  expire together this reproduces SP4's ``trainable[(g + round) % n]``.
- ``restrict_collecting``: seats of agents that cannot collect (no running learner) do not collect (spec 4.7).
- ``BudgetCounter``: the one budget (spec 4.4): a saved base plus the increments of each (node, session)'s cumulative
  env-step counter since its last saved value.
- ``VersionRegistry``: per agent the current learner generation, the highest accepted version, the version cap of
  every superseded generation and the latest announcement (spec 4.5, 7.2; R16: a superseded generation ``g`` is
  stale only above ``cap(g)``).
- ``ResultIntake``: checks of every match result a node sends (spec 4.3, 3.4: never raises) and the batch dedup / ack
  bookkeeping per (node, session, seq) (R8), with drop counters by reason and one WARNING per (node, reason).
- ``HubStateData`` / ``save_hub_state`` / ``load_hub_state``: ``runs/<name>/hub_state.json``.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from colosseum.arena.entities import snapshot_version
from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID, Lineup, MatchResult

__all__ = [
    "EPOCH_BLOCK", "EVICTED_HOLD_FACTOR", "HUB_STATE_FILE", "HUB_STATE_VERSION", "BudgetCounter", "CellEntry",
    "CellKey", "CellTable", "EpochAllocator", "HubStateData", "NodeEntry", "NodeHello", "NodeRegistry",
    "OwnerRotation", "ResultIntake", "VersionInfo", "VersionRegistry", "load_hub_state", "restrict_collecting",
    "save_hub_state",
]

EPOCH_BLOCK = 1024                   # R7: epochs reserved per hub_state.json write
EVICTED_HOLD_FACTOR = 2              # spec 4.2: an evicted snapshot stays servable <= 2 x match_refresh_interval_sec
HUB_STATE_FILE = "hub_state.json"
HUB_STATE_VERSION = 1

DROP_REASONS = ("invalid", "stale_generation")


# ---------------------------------------------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeHello:
    """What the hub keeps of a node's ``Hello`` (``hub.server.hello_from_proto``): no protobuf here."""

    hardware: dict = field(default_factory=dict)
    slots: dict = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    data_addresses: tuple[str, ...] = ()
    live_cells: tuple[tuple[int, int, int, int], ...] = ()   # (worker, env, epoch, incarnation)
    env_steps: int = 0
    last_acked_seq: int = 0


@dataclass
class NodeEntry:
    """One node as the hub sees it (spec 4.1). ``last_heartbeat`` is hub monotonic time."""

    node_id: str
    number: int
    name: str
    session_id: str | None
    state: Literal["online", "lost", "removed"]
    hardware: dict
    slots: dict
    tags: tuple[str, ...]
    data_addresses: tuple[str, ...]
    data_path: Literal["direct", "relay", "local"]
    last_heartbeat: float


class NodeRegistry:
    """Nodes of this hub process by node id (spec 4.1). A closed stream does not make a node lost: only silence for
    ``node_lost_after_sec`` does (``lost``), so a node that reconnects within it keeps its cells."""

    def __init__(self) -> None:
        self._nodes: dict[str, NodeEntry] = {}

    def connect(self, node_id: str, number: int, name: str, session_id: str, hello: NodeHello, now: float, *,
                data_path: Literal["direct", "relay", "local"] = "relay") -> Literal["new", "resumed", "returned"]:
        """A ``Hello``: ``new`` (first time in this hub process), ``resumed`` (the same session id of a node that is
        not lost: the stream was cut and came back; its cells stay), ``returned`` (a lost or removed node, or a new
        session id: the node process restarted; the caller drops the node's cells)."""
        entry = self._nodes.get(node_id)
        if entry is None:
            kind: Literal["new", "resumed", "returned"] = "new"
        elif entry.state == "online" and entry.session_id == session_id:
            kind = "resumed"
        else:
            kind = "returned"
        self._nodes[node_id] = NodeEntry(
            node_id=node_id, number=int(number), name=name, session_id=session_id, state="online",
            hardware=dict(hello.hardware), slots=dict(hello.slots), tags=tuple(hello.tags),
            data_addresses=tuple(hello.data_addresses), data_path=data_path, last_heartbeat=now)
        return kind

    def heartbeat(self, node_id: str, session_id: str, now: float) -> bool:
        """Record a heartbeat; False (ignored) for an unknown node, another session, or a node not online."""
        entry = self._nodes.get(node_id)
        if entry is None or entry.state != "online" or entry.session_id != session_id:
            return False
        entry.last_heartbeat = now
        return True

    def disconnect(self, node_id: str, session_id: str) -> None:
        """The node's stream closed; the node stays online until ``lost`` (it may come back with the session)."""
        entry = self._nodes.get(node_id)
        if entry is not None and entry.session_id == session_id and entry.state == "online":
            entry.session_id = None

    def lost(self, now: float, after_sec: float) -> list[NodeEntry]:
        """Mark and return the online nodes silent for more than ``after_sec`` (each once)."""
        out = []
        for entry in self._nodes.values():
            if entry.state == "online" and now - entry.last_heartbeat > after_sec:
                entry.state = "lost"
                entry.session_id = None
                out.append(entry)
        return out

    def remove(self, node_id: str) -> NodeEntry | None:
        entry = self._nodes.get(node_id)
        if entry is not None:
            entry.state = "removed"
            entry.session_id = None
        return entry

    def online(self) -> list[NodeEntry]:
        return sorted((e for e in self._nodes.values() if e.state == "online"), key=lambda e: e.number)

    def get(self, number: int) -> NodeEntry | None:
        return next((e for e in self._nodes.values() if e.number == number), None)

    def by_id(self, node_id: str) -> NodeEntry | None:
        return self._nodes.get(node_id)

    def all(self) -> list[NodeEntry]:
        return sorted(self._nodes.values(), key=lambda e: e.number)


# ---------------------------------------------------------------------------------------------------------------
# Cells, epochs, owners
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class CellKey:
    node: int
    worker: int
    env: int


class EpochAllocator:
    """Epoch numbers, unique and increasing within a run (R7). ``reserved_until``: the highest epoch an earlier hub
    process may have handed out (``hub_state.json``; 0 for a new run); this allocator starts at the next block, so the
    first epoch is ``reserved_until + 1``. ``on_reserve(until)`` must persist ``until`` durably (atomic write + fsync)
    before it returns; it runs before the first epoch above the current reservation is returned, and an exception
    from it propagates (no epoch is handed out unreserved)."""

    def __init__(self, reserved_until: int, on_reserve: Callable[[int], None], *, block: int = EPOCH_BLOCK) -> None:
        if reserved_until < 0 or block < 1:
            raise ValueError(f"EpochAllocator: reserved_until {reserved_until} and block {block}")
        self._reserved = int(reserved_until)
        self._next = int(reserved_until) + 1
        self._on_reserve = on_reserve
        self._block = int(block)

    @property
    def reserved_until(self) -> int:
        return self._reserved

    def next(self) -> int:
        epoch = self._next
        if epoch > self._reserved:
            until = self._reserved + self._block
            self._on_reserve(until)              # raises: nothing changes, the caller grants nothing
            self._reserved = until
        self._next = epoch + 1
        return epoch


class OwnerRotation:
    """Owner of every requested cell by the deficit rule (module docstring). ``agents``: the trainable agents in
    config order."""

    def __init__(self, agents: Sequence[str]) -> None:
        self._agents = list(agents)

    def assign(self, cells: Sequence[CellKey], *, collecting: Collection[str], owners: Mapping[CellKey, str],
               previous: Mapping[CellKey, str | None]) -> dict[CellKey, str]:
        """``cells``: the requested cells, in (node, worker, env) order; ``collecting``: agents that can collect now;
        ``owners``: the current owners of every other live collecting cell (other agents are ignored); ``previous``:
        the previous owner of a requested cell (absent or None: none). Returns an owner per requested cell; empty
        when no agent can collect."""
        order = [a for a in self._agents if a in set(collecting)]
        if not order:
            return {}
        requested = set(cells)
        owned = Counter(a for key, a in owners.items() if a in order and key not in requested)
        out: dict[CellKey, str] = {}
        for cell in cells:
            fewest = min(owned[a] for a in order)          # equal targets: the largest deficit owns the fewest
            tied = [a for a in order if owned[a] == fewest]
            prev = previous.get(cell)
            if prev in order:
                start = order.index(prev) + 1
                cyclic = order[start:] + order[:start]
                owner = next(a for a in cyclic if a in tied)
            else:
                owner = tied[0]
            out[cell] = owner
            owned[owner] += 1
        return out


@dataclass
class CellEntry:
    key: CellKey
    epoch: int
    incarnation: int
    owner: str | None = None
    previous: str | None = None
    granted_at: float | None = None      # hub monotonic time of the last grant
    ttl_sec: float = 0.0                 # of the last grant (0 = whole run)


class CellTable:
    """Live cells by key (spec 4.2); ``epochs`` gives new epochs."""

    def __init__(self, epochs: EpochAllocator) -> None:
        self._epochs = epochs
        self._cells: dict[CellKey, CellEntry] = {}
        self._granted: set[int] = set()        # epochs this hub process handed out

    def request(self, node: int, cells: Sequence[tuple[CellKey, int, int]], now: float) -> list[CellEntry]:
        """``cells``: ``(key, epoch, incarnation)`` the node asks a lineup for (epoch 0: none yet). Returns the
        entries in request order: the epoch stays for the same incarnation and an epoch this hub process granted to
        that cell; otherwise a new epoch. Every key must be of ``node`` (ValueError)."""
        out = []
        for key, epoch, incarnation in cells:
            if key.node != node:
                raise ValueError(f"cell {key} requested by node {node}")
            entry = self._cells.get(key)
            keep = (entry is not None and entry.incarnation == incarnation and entry.epoch == epoch
                    and epoch in self._granted)
            if not keep:
                new_epoch = self._epochs.next()
                self._granted.add(new_epoch)
                previous = None if entry is None else (entry.owner or entry.previous)
                entry = CellEntry(key=key, epoch=new_epoch, incarnation=int(incarnation), owner=None,
                                  previous=previous)
                self._cells[key] = entry
            out.append(entry)
        return out

    def grant(self, key: CellKey, owner: str, ttl_sec: float, now: float) -> CellEntry:
        """Record the lineup just granted: ``owner`` becomes the owner (the old one the previous owner)."""
        entry = self._cells[key]
        if entry.owner is not None:
            entry.previous = entry.owner
        entry.owner = owner
        entry.granted_at = now
        entry.ttl_sec = float(ttl_sec)
        return entry

    def owners(self, *, exclude: Collection[CellKey] = ()) -> dict[CellKey, str]:
        """Current owners of the live cells not in ``exclude``."""
        skip = set(exclude)
        return {k: e.owner for k, e in self._cells.items() if e.owner is not None and k not in skip}

    def drop_node(self, node: int) -> list[CellEntry]:
        dropped = [e for k, e in self._cells.items() if k.node == node]
        for entry in dropped:
            del self._cells[entry.key]
        return dropped

    def drop_cells(self, keys: Iterable[CellKey]) -> list[CellEntry]:
        out = [self._cells.pop(k) for k in list(keys) if k in self._cells]
        return out

    def get(self, key: CellKey) -> CellEntry | None:
        return self._cells.get(key)

    def live(self) -> list[CellEntry]:
        return [self._cells[k] for k in sorted(self._cells)]


def restrict_collecting(lineup: Lineup, collecting: Collection[str]) -> Lineup:
    """``lineup`` with ``collect=False`` for every seat whose agent cannot collect (spec 4.7); the lineup itself is
    not changed (a copy is returned when anything changes)."""
    can = set(collecting)
    if all(not s.collect or s.agent_id in can for s in lineup.seats):
        return lineup
    seats = [s if not s.collect or s.agent_id in can else replace(s, collect=False) for s in lineup.seats]
    return replace(lineup, seats=seats)


# ---------------------------------------------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------------------------------------------


class BudgetCounter:
    """The global env-step counter (spec 4.4): ``base`` plus every increment of a (node, session)'s cumulative
    counter since its last value. ``sessions`` (from ``hub_state.json``): the last value per
    ``"<node_id>/<session_id>"``, so a node of a saved session that reports again is counted from where the save
    left off."""

    def __init__(self, base: int = 0, sessions: Mapping[str, int] | None = None) -> None:
        if base < 0:
            raise ValueError(f"BudgetCounter: base {base} < 0")
        self._value = int(base)
        self._last: dict[str, int] = {str(k): int(v) for k, v in (sessions or {}).items()}

    def report(self, node_id: str, session_id: str, cumulative: int) -> int:
        """A node's cumulative env steps of its session; returns the increment applied (0 for a value that is not
        larger than the last one)."""
        key = f"{node_id}/{session_id}"
        cumulative = int(cumulative)
        last = self._last.get(key, 0)
        if cumulative <= last:
            return 0
        self._last[key] = cumulative
        self._value += cumulative - last
        return cumulative - last

    def add(self, n: int) -> None:
        """Steps that are not a node's (``training.resume_from``'s start point)."""
        if n < 0:
            raise ValueError(f"BudgetCounter.add({n})")
        self._value += int(n)

    @property
    def value(self) -> int:
        return self._value

    def state(self) -> dict:
        return {"env_steps": self._value, "sessions": dict(self._last)}


# ---------------------------------------------------------------------------------------------------------------
# Versions and generations
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionInfo:
    """One published version of an agent's latest weights (spec 4.5): where to get it and how to check it."""

    agent_id: str
    generation: int
    version: int
    size: int
    digest: str
    node: int
    addresses: tuple[str, ...] = ()
    relay_only: bool = False
    teacher_active: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["addresses"] = list(self.addresses)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> VersionInfo:
        return cls(agent_id=str(d["agent_id"]), generation=int(d["generation"]), version=int(d["version"]),
                   size=int(d["size"]), digest=str(d["digest"]), node=int(d["node"]),
                   addresses=tuple(d.get("addresses", ())), relay_only=bool(d.get("relay_only", False)),
                   teacher_active=bool(d.get("teacher_active", False)))


@dataclass
class _AgentVersions:
    generation: int = 0
    max_version: int = -1                       # highest accepted version over all generations
    gen_max: int = -1                           # highest accepted version of the current generation
    caps: dict[int, int] = field(default_factory=dict)   # superseded generation -> its highest accepted version
    latest: VersionInfo | None = None


class VersionRegistry:
    """Agent -> generation -> versions (spec 4.5, 7.2; R16)."""

    def __init__(self) -> None:
        self._agents: dict[str, _AgentVersions] = {}

    def _get(self, agent_id: str) -> _AgentVersions:
        return self._agents.setdefault(agent_id, _AgentVersions())

    def new_generation(self, agent_id: str) -> int:
        """The next learner generation of ``agent_id`` (first: 1); the current one's cap is recorded."""
        a = self._get(agent_id)
        if a.generation > 0:
            a.caps[a.generation] = a.gen_max
        a.generation += 1
        a.gen_max = -1
        return a.generation

    def generation(self, agent_id: str) -> int:
        a = self._agents.get(agent_id)
        return 0 if a is None else a.generation

    def announce(self, info: VersionInfo) -> Literal["accepted", "stale_generation", "not_newer", "invalid"]:
        """A learner's announcement: accepted only from the current generation and above every accepted version."""
        if (info.version < 0 or info.generation < 1 or info.size < 0 or not info.digest
                or not info.agent_id):
            return "invalid"
        a = self._agents.get(info.agent_id)
        if a is None or info.generation > a.generation:
            return "invalid"                         # the hub never assigned this generation
        if info.generation < a.generation:
            return "stale_generation"
        if info.version <= a.max_version:
            return "not_newer"
        a.max_version = a.gen_max = info.version
        a.latest = info
        return "accepted"

    def latest(self, agent_id: str) -> VersionInfo | None:
        a = self._agents.get(agent_id)
        return None if a is None else a.latest

    def version_floor(self, agent_id: str) -> int:
        """The first version a newly assigned learner may publish: max accepted + 1 (0 without any; R9)."""
        a = self._agents.get(agent_id)
        return 0 if a is None else a.max_version + 1

    def is_stale(self, agent_id: str, generation: int | None, version: int | None) -> bool:
        """Whether data played with (``generation``, ``version``) of ``agent_id``'s latest weights must be rejected
        (R16): unknown generation (None or 0: SP4 path) -> no; the current one -> no; a superseded ``g`` -> only above
        ``cap(g)`` (or without a version); a generation never assigned -> yes."""
        if generation is None or generation == 0:
            return False
        a = self._agents.get(agent_id)
        if a is None or generation > a.generation or generation < 0:
            return True
        if generation == a.generation:
            return False
        cap = a.caps.get(generation)
        return version is None or cap is None or version > cap

    def state(self) -> dict:
        return {aid: {"generation": a.generation, "max_version": a.max_version, "gen_max": a.gen_max,
                      "caps": {str(g): c for g, c in sorted(a.caps.items())},
                      "latest": None if a.latest is None else a.latest.to_dict()}
                for aid, a in sorted(self._agents.items())}

    @classmethod
    def from_state(cls, d: Mapping[str, Any]) -> VersionRegistry:
        reg = cls()
        for aid, s in d.items():
            reg._agents[str(aid)] = _AgentVersions(
                generation=int(s["generation"]), max_version=int(s["max_version"]),
                gen_max=int(s.get("gen_max", -1)), caps={int(g): int(c) for g, c in s.get("caps", {}).items()},
                latest=None if s.get("latest") is None else VersionInfo.from_dict(s["latest"]))
        return reg


# ---------------------------------------------------------------------------------------------------------------
# Result intake
# ---------------------------------------------------------------------------------------------------------------

_NETWORK_RE = re.compile(r"ckpt_v\d+")


@dataclass
class _SessionSeqs:
    low: int = 0                                # every seq <= low is acked
    acked: set[int] = field(default_factory=set)        # acked seqs above low
    in_flight: set[int] = field(default_factory=set)

    def mark_acked(self, seq: int) -> None:
        self.in_flight.discard(seq)
        if seq > self.low:
            self.acked.add(seq)
        while self.low + 1 in self.acked:
            self.low += 1
            self.acked.discard(self.low)


class ResultIntake:
    """Result checks and batch bookkeeping (spec 4.3, R8). ``journal``: whether accounting follows the journal's
    commits (``accounts_at_intake`` is its negation: without the journal the hub accounts a batch when it arrives and
    acks it at once)."""

    def __init__(self, *, journal: bool) -> None:
        self.journal = bool(journal)
        self._sessions: dict[tuple[str, str], _SessionSeqs] = {}
        self.dropped: Counter[str] = Counter()
        self._warned: set[tuple[int, str]] = set()

    @property
    def accounts_at_intake(self) -> bool:
        return not self.journal

    # ---- checks -------------------------------------------------------------------------------------------------

    def check(self, result: MatchResult, *, spec, agents: Collection[str], versions: VersionRegistry,
              node: int | None = None, uid_prefix: str | None = None) -> str | None:
        """None for a valid result, else the drop reason (``invalid`` / ``stale_generation``); never raises.
        ``agents``: every agent a result may name (current and historical); ``node`` / ``uid_prefix``: the sending
        node's number and ``<run_id>/train/n<node>/`` (a node cannot write another node's uids)."""
        try:
            return self._check(result, spec, agents, versions, node, uid_prefix)
        except Exception:  # noqa: BLE001 - untrusted input: anything unexpected is invalid
            return "invalid"

    @staticmethod
    def _check(result: MatchResult, spec, agents: Collection[str], versions: VersionRegistry, node: int | None,
               uid_prefix: str | None) -> str | None:
        if not isinstance(result, MatchResult) or result.source != "train":
            return "invalid"
        if result.layout not in spec.layouts or result.outcome_kind != spec.outcome_kind(result.layout):
            return "invalid"
        seat_specs = spec.layouts[result.layout]
        if len(result.seats) != len(seat_specs):
            return "invalid"
        if node is not None and result.node != node:
            return "invalid"
        if uid_prefix is not None and not (result.uid.startswith(uid_prefix) and len(result.uid) > len(uid_prefix)):
            return "invalid"
        if not isinstance(result.episode_length, int) or result.episode_length < 0:
            return "invalid"
        for i, (seat, seat_spec) in enumerate(zip(result.seats, seat_specs, strict=True)):
            if seat.seat != i or seat.role != seat_spec.role or seat.team != seat_spec.team:
                return "invalid"
            if seat.agent_id not in agents:
                return "invalid"
            net = seat.network_id
            if net not in (LATEST_NETWORK_ID, FIXED_NETWORK_ID) and snapshot_version(net) is None:
                return "invalid"
            if not math.isfinite(float(seat.reward)):
                return "invalid"
            for v in (seat.version_start, seat.version_end, seat.eliminated_step):
                if v is not None and (not isinstance(v, int) or v < 0):
                    return "invalid"
        teams = {s.team for s in seat_specs}
        if sorted(t.team for t in result.teams) != sorted(teams):
            return "invalid"
        if not all(math.isfinite(float(t.rank)) and math.isfinite(float(t.score)) for t in result.teams):
            return "invalid"
        for seat in result.seats:
            if seat.network_id == LATEST_NETWORK_ID and versions.is_stale(seat.agent_id, seat.generation,
                                                                          seat.version_end):
                return "stale_generation"
        return None

    def count_drop(self, node: int, reason: str, n: int = 1) -> bool:
        """Count ``n`` dropped results; True the first time for (node, reason): the caller logs one WARNING."""
        self.dropped[reason] += int(n)
        if (node, reason) in self._warned:
            return False
        self._warned.add((node, reason))
        return True

    # ---- batches ------------------------------------------------------------------------------------------------

    def open(self, node_id: str, session_id: str, last_acked: int) -> None:
        """A session's Hello: seqs up to ``last_acked`` are acked (the node says so; a hub restart forgot them)."""
        seqs = self._sessions.setdefault((node_id, session_id), _SessionSeqs())
        if last_acked > seqs.low:
            seqs.low = int(last_acked)
            seqs.acked = {s for s in seqs.acked if s > seqs.low}
            seqs.in_flight = {s for s in seqs.in_flight if s > seqs.low}
            while seqs.low + 1 in seqs.acked:
                seqs.low += 1
                seqs.acked.discard(seqs.low)

    def begin(self, node_id: str, session_id: str, seq: int) -> bool:
        """A batch arrives: True when it is new (now in flight); False when it is acked or in flight (skip it; an
        acked one is acked again: ``is_acked``)."""
        seqs = self._sessions.setdefault((node_id, session_id), _SessionSeqs())
        if seq <= seqs.low or seq in seqs.acked or seq in seqs.in_flight:
            return False
        seqs.in_flight.add(int(seq))
        return True

    def is_acked(self, node_id: str, session_id: str, seq: int) -> bool:
        seqs = self._sessions.get((node_id, session_id))
        return seqs is not None and (seq <= seqs.low or seq in seqs.acked)

    def committed(self, node_id: str, session_id: str, seq: int) -> None:
        """Every result of the batch is accounted: ack it."""
        self._sessions.setdefault((node_id, session_id), _SessionSeqs()).mark_acked(int(seq))

    def failed(self, node_id: str, session_id: str, seq: int) -> None:
        """Nothing of the batch was accounted: not acked, the node resends it."""
        seqs = self._sessions.get((node_id, session_id))
        if seqs is not None:
            seqs.in_flight.discard(int(seq))

    def forget_node(self, node_id: str) -> None:
        """The node is gone for good (``nodes remove``); a lost node keeps its bookkeeping (it may resend)."""
        for key in [k for k in self._sessions if k[0] == node_id]:
            del self._sessions[key]


# ---------------------------------------------------------------------------------------------------------------
# hub_state.json
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class HubStateData:
    """``runs/<name>/hub_state.json`` (overview, "hub.state")."""

    version: int
    run_id: str
    env_steps: int
    sessions: dict[str, int]
    epoch_reserved_until: int
    versions: dict
    counters: dict[str, Any]
    nodes_lost: int


def save_hub_state(path: str | Path, data: HubStateData, *, fsync: bool) -> None:
    """Write ``data`` atomically (temp file + ``os.replace``); with ``fsync`` the file and its directory are synced
    before this returns (epoch reservations, R7)."""
    path = Path(path)
    text = json.dumps(asdict(data), indent=2, sort_keys=True, allow_nan=False)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    if fsync:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def load_hub_state(path: str | Path) -> HubStateData | None:
    """The saved state, or None when the file does not exist; a malformed file or another version is a ValueError
    naming the file (the caller turns it into a ConfigError with a hint)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("version") != HUB_STATE_VERSION:
            raise ValueError(f"version {d.get('version')!r}, expected {HUB_STATE_VERSION}")
        return HubStateData(version=int(d["version"]), run_id=str(d["run_id"]), env_steps=int(d["env_steps"]),
                            sessions={str(k): int(v) for k, v in d["sessions"].items()},
                            epoch_reserved_until=int(d["epoch_reserved_until"]), versions=dict(d["versions"]),
                            counters=dict(d["counters"]), nodes_lost=int(d["nodes_lost"]))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"malformed {path}: {type(e).__name__}: {e}") from e
```

Notes for the implementer:
- `OwnerRotation` ignores the size of each agent's target share because every collecting agent's share is equal (spec 4.2 "целевая доля"); the largest deficit is therefore the fewest owned live collecting cells. If a later spec adds unequal shares, `assign` gets a `shares` argument; nothing else changes.
- `CellTable.request` keeps an epoch only when the hub process granted it (`_granted`); a resumed hub (T4.3) builds a new table, so every cell of every node gets a new epoch after a resume (spec 4.2) — the uids and seeds of a resumed run never repeat.
- `ResultIntake.check` runs on untrusted input and returns a reason instead of raising; it does not check that a snapshot seat names a stored snapshot (evicted snapshots' results stay valid, spec 4.3 "результаты убранных агентов принимаются" generalizes to snapshots).
- `save_hub_state(..., fsync=True)` is what `EpochAllocator.on_reserve` uses (R7); the metrics tick saves with `fsync=False`.

- [ ] **Step 4: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_hub_state.py -q`
Expected: `70 passed`.

- [ ] **Step 5: Full fast suite + ruff**

Run the full fast suite and `ruff check .`. Expected: green, zero warnings.

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/hub/__init__.py src/colosseum/hub/state.py tests/unit/test_sp5_hub_state.py
git commit -m "feat: hub state (node registry, cells and epochs in blocks, deficit rotation, budget, versions, result intake)"
git push
```

---


### Task T2.2: `colosseum.net.local` — process connections, worker and learner IO, epoch-aware ids and cell seeds, `version_floor`

Spec 4.7 (connections per process, no shared locks, one learner intake path, backpressure, at-most-once with reasons), 4.3 / 7.6 (the epoch in match ids and seeds), 4.5 (versions never go back: `version_floor`), 7.8 (socket paths); rulings R3 (pickled control messages on the same machine), R4 (the loops keep their structure), R10 (weights only through the cache), R11 (lineups name cache keys), R16 (the learner's generation rule), R17 (`Lineup.epoch` + `cell_seed_fn` + `uid_prefix`). After this task the node's children (T2.4) have their whole IO; the `Launcher` and the distributed learner keep running on SP4's queues through `QueueIntake` / `QueueOutputs`.

**Files:**
- Create: `src/colosseum/net/local.py`
- Modify: `src/colosseum/learner/learner.py` (`learner_process` on `ChunkIntake` / `LearnerOutputs`, `set_policy_version`, `_publish_weights` replaces `_push_weights`, `collect_batch` accepts decoded chunks)
- Modify: `src/colosseum/worker/match_runner.py` (`cell_seed_fn`; epoch in match ids and seeds), `src/colosseum/worker/rollout_loop.py` (`cell_seed_fn`, `uid_prefix`, `snapshot_fallback` pass-through)
- Modify: `src/colosseum/core/types.py` (`WeightPayload.generation`), `src/colosseum/utils/seeding.py` (`cell_seed`, `worker_seed`)
- Modify: `src/colosseum/launcher.py` (`_learner_main` builds `QueueIntake` / `QueueOutputs`), `src/colosseum/distributed.py` (`run_distributed_learner` likewise)
- Modify (port to the new `learner_process` signature): `tests/game_helpers.py` (`queue_learner_io`), `tests/unit/test_learner_v2.py`, `tests/integration/test_learner_v2_exit.py`, `tests/unit/test_sp2_ipc_latest.py`, `tests/unit/test_sp3_dagger_loss.py`
- Test: `tests/unit/test_sp5_local_connections.py`, `tests/unit/test_sp5_worker_io.py`, `tests/unit/test_sp5_learner_io.py`, `tests/contract/test_sp5_loop_epochs.py`

**Interfaces:**
- Consumes: `encode_chunk`, `decode_chunk`, `ChunkValidator`, `ChunkError`, `WeightsMeta`, `encode_weights`, `encode_checkpoint` (T1.1); `payload_byte_cap` (T1.1); `WeightCacheWriter.publish_latest`, `WeightCacheReader.read_latest` / `get`, `CacheKey`, `CacheFullError` (T1.5); `Lineup.epoch`, `MatchResult.node`, `SeatResult.generation` (T1.2); `ResultBatcher` (`worker.rollout_worker`), `LoopIO`, `RolloutLoop`, `MatchRunner`.
- Produces (contract "colosseum.net.local" and "Changes to existing loops", plus additions marked *): `address`, `listener`, `connect`, *`listener_socket`, *`RUNTIME_DIR_ENV`, *`MAX_SOCKET_PATH`, *`SEND_SLICE_SEC`, *`PROGRESS_INTERVAL_SEC`, *`INTAKE_REPORT_SEC`; the control messages `ChildHello`, `Stop`, `ChildError`, `CellLineup`, `WorkerLineups`, `WorkerEvict`, `LearnerRoute`, `WeightsReady`, `WorkerResults`, `WorkerProgress`, `LearnerEnvSteps`, `VersionPublished`, `CheckpointHeader`, `LearnerMetrics`, `IntakeReport`; *`ControlChannel`; *`FrameSender`; `WorkerIO` (constructor adds *`results_batch`, *`results_flush_sec`, *`clock`; methods `loop_io`, `ready`, `stopped`, *`pump`, *`take_lineups`, *`idle`, *`counters`, *`send_progress`, *`send_error`, *`close`, attribute *`stats_fn`, *`dropped`); `ChunkIntake`, `LearnerOutputs` (*`checkpoints` attribute, *`finish(stop_event)`); `ConnectionIntake` (`control` is a *`ControlChannel`; *`value`, *`close`); `NodeOutputs` (`control` is a *`ControlChannel`); `QueueIntake`, `QueueOutputs`; `learner_process(..., intake, outputs, ..., version_floor=0, recovery_interval_sec=None)`; *`set_policy_version`; `collect_batch` (items may be `TrajectoryChunk`); `WeightPayload.generation` (and `from_model(..., generation=0)`); `MatchRunner(..., cell_seed_fn=None)`, `RolloutLoop(..., cell_seed_fn=None, uid_prefix=None, snapshot_fallback=True)`; `cell_seed`, `worker_seed`; test helper *`game_helpers.queue_learner_io`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp5_local_connections.py` (verified in the scratch package with a pickle stand-in for T1.1's codec; here with the real codec and validator):
```python
"""Same-machine connections (SP5 T2.2; spec 4.7, 7.8; R3): listeners and connections with the node's authkey;
the learner's intake reads every connection, applies backpressure, and drops by reason (invalid, stale generation,
disconnect) without ever letting one sender's death block another; the control connection carries env steps, stop
and the intake report; a dead learner never blocks a worker's send."""
from __future__ import annotations

import math
import multiprocessing as mp
import multiprocessing.connection as mpc
import os
import queue
import secrets
import signal
import threading
import time
from collections import Counter

import pytest

from colosseum.core.types import TrajectoryChunk
from colosseum.net.chunks import ChunkValidator, encode_chunk
from colosseum.net.local import (
    ChildHello,
    ConnectionIntake,
    ControlChannel,
    FrameSender,
    IntakeReport,
    LearnerEnvSteps,
    Stop,
    address,
    connect,
    listener,
)
from game_helpers import chunk_v2_payload, learner_role

KEY = b"k" * 32


def _name(proc="x"):
    return address("abcdef12", secrets.token_hex(4), 1, proc)


def _chunk(version=0, agent="a"):
    return TrajectoryChunk.from_payload(chunk_v2_payload(version=version, agent_id=agent))


def _frame(version=0, agent="a", generation=0, nan=False):
    chunk = _chunk(version, agent)
    if nan:
        chunk.behavior_logp[0] = math.nan
    return encode_chunk(chunk, generation=generation, node=1, worker=0)


def _hello(conn):
    conn.send(ChildHello("worker", "w", 1, os.getpid(), "chunks"))
    return conn


def _wait(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class _Learner:
    """An in-process ConnectionIntake with its control pair."""

    def __init__(self, capacity=2, cap=1 << 26, generation=1, floor=0):
        self.name = _name("learner-a-i1")
        self.lst = listener(self.name, KEY)
        self.node_end, learner_end = mp.Pipe()
        self.control = ControlChannel(learner_end)
        validator = ChunkValidator(learner_role(), agent_id="a", teacher=False)
        self.intake = ConnectionIntake(self.lst, self.control, validator, agent_id="a", generation=generation,
                                       version_floor=floor, capacity=capacity, cap_bytes=cap)

    def close(self):
        self.intake.close()


@pytest.fixture
def learner():
    made = []

    def make(**kw):
        made.append(_Learner(**kw))
        return made[-1]
    yield make
    for lrn in made:
        lrn.close()


def test_a_wrong_authkey_is_refused_and_connect_times_out_without_a_listener():
    name = _name()
    lst = listener(name, KEY)
    errors = []

    def accept():
        try:
            lst.accept()
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    t = threading.Thread(target=accept, daemon=True)
    t.start()
    with pytest.raises(mpc.AuthenticationError):
        connect(name, b"z" * 32, timeout=2.0)
    t.join(5)
    assert errors and isinstance(errors[0], mpc.AuthenticationError)
    lst.close()
    start = time.monotonic()
    with pytest.raises(ConnectionError):
        connect(_name(), KEY, timeout=0.3)
    assert time.monotonic() - start < 2.0
    with pytest.raises(ValueError):
        listener("x" * 200, KEY)


def test_connect_waits_for_a_listener_that_comes_up_later():
    name = _name()
    box = {}

    def late():
        time.sleep(0.3)
        box["lst"] = lst = listener(name, KEY)
        box["conn"] = lst.accept()
    t = threading.Thread(target=late, daemon=True)
    t.start()
    conn = connect(name, KEY, timeout=5.0)
    conn.send("hi")
    t.join(5)
    assert box["conn"].recv() == "hi"
    box["lst"].close()


def test_the_intake_reads_every_connection_and_applies_backpressure(learner):
    lrn = learner(capacity=2)
    dropped = Counter()
    senders = [FrameSender(f"s{i}", lambda: _hello(connect(lrn.name, KEY, timeout=5)), dropped) for i in range(2)]
    for v in range(3):
        for s in senders:
            assert s.put(_frame(v), timeout=5.0)
    assert _wait(lambda: lrn.intake.depth == 2)
    time.sleep(0.3)
    assert lrn.intake.depth == 2                       # full: the reader stops reading the sockets
    got = [lrn.intake.get(timeout=5.0).policy_version for _ in range(6)]
    assert sorted(got) == [0, 0, 1, 1, 2, 2] and dropped == Counter()
    with pytest.raises(queue.Empty):
        lrn.intake.get(timeout=0.2)
    for s in senders:
        s.close()


def _big_sender(name, nbytes, ready):
    conn = _hello(connect(name, KEY, timeout=5))
    ready.set()
    conn.send_bytes(b"\0" * nbytes)                   # blocks: the learner does not read while full


def test_a_sender_killed_mid_frame_closes_only_its_connection(learner):
    lrn = learner(capacity=1)
    first = _hello(connect(lrn.name, KEY, timeout=5))
    first.send_bytes(_frame(0))
    assert _wait(lambda: lrn.intake.depth == 1)        # now full
    ready = mp.Event()
    proc = mp.Process(target=_big_sender, args=(lrn.name, 64 << 20, ready), daemon=True)
    proc.start()
    assert ready.wait(30)
    time.sleep(0.5)                                    # mid-send, blocked on the full socket
    os.kill(proc.pid, signal.SIGKILL)
    proc.join(10)
    assert lrn.intake.get(timeout=5.0).policy_version == 0      # frees the deque: the cut frame is read
    first.send_bytes(_frame(7))
    assert lrn.intake.get(timeout=10.0).policy_version == 7     # the other sender is unaffected
    assert _wait(lambda: lrn.intake.dropped.get("disconnect") == 1)


def test_a_clean_close_loses_nothing_and_an_oversized_frame_is_invalid(learner):
    lrn = learner(capacity=8, cap=200_000)
    conn = _hello(connect(lrn.name, KEY, timeout=5))
    conn.send_bytes(_frame(1))
    conn.close()
    assert lrn.intake.get(timeout=5.0).policy_version == 1
    big = _hello(connect(lrn.name, KEY, timeout=5))
    with pytest.raises(BrokenPipeError):               # the learner closes it after reading the length
        big.send_bytes(b"x" * 4_000_000)
    other = _hello(connect(lrn.name, KEY, timeout=5))
    other.send_bytes(_frame(2))
    assert lrn.intake.get(timeout=5.0).policy_version == 2
    assert _wait(lambda: lrn.intake.dropped == {"invalid": 1})


def test_misrouted_garbage_and_stale_generation_chunks_are_dropped_by_reason(learner):
    lrn = learner(capacity=16, generation=2, floor=10)
    conn = _hello(connect(lrn.name, KEY, timeout=5))
    for frame in (_frame(3, agent="b"), b"garbage", _frame(6, nan=True), _frame(12, generation=1),
                  _frame(4, generation=1), _frame(5, generation=3), _frame(11, generation=2), _frame(13, generation=0)):
        conn.send_bytes(frame)
    got = [lrn.intake.get(timeout=5.0).policy_version for _ in range(3)]
    assert got == [4, 11, 13]                          # gen 1 below the floor is kept (R16)
    assert lrn.intake.dropped == {"invalid": 3, "stale_generation": 2}


def test_the_control_connection_carries_env_steps_and_stop_and_reports_the_intake(learner):
    lrn = learner()
    lrn.node_end.send(LearnerEnvSteps(1234))
    assert _wait(lambda: lrn.intake.value == 1234)
    assert lrn.node_end.poll(3.0)
    report = lrn.node_end.recv()
    assert isinstance(report, IntakeReport) and report.depth == 0
    assert not lrn.intake.stop_flag.is_set()
    lrn.node_end.send(Stop("budget"))
    assert lrn.intake.stop_flag.wait(5)


def test_the_learner_stops_when_its_node_is_gone(learner):
    lrn = learner()
    lrn.node_end.close()
    assert lrn.intake.stop_flag.wait(5)


def _serve_then_die(name, ready):
    lst = listener(name, KEY)
    ready.set()
    conn = lst.accept()
    conn.recv()
    time.sleep(60)


def test_a_dead_learner_never_blocks_a_worker_send():
    name = _name("learner-a-i1")
    ready = mp.Event()
    proc = mp.Process(target=_serve_then_die, args=(name, ready), daemon=True)
    proc.start()
    assert ready.wait(30)
    dropped = Counter()
    sender = FrameSender("a", lambda: _hello(connect(name, KEY, timeout=5)), dropped)
    big = b"\0" * (8 << 20)
    assert sender.put(big, timeout=5.0)                # the thread blocks in send_bytes (nobody reads)
    os.kill(proc.pid, signal.SIGKILL)
    proc.join(10)
    assert _wait(lambda: sender.broken)
    start = time.monotonic()
    for _ in range(5):
        assert sender.put(b"y", timeout=0.5)            # consumed at once: dropped and counted
    assert time.monotonic() - start < 1.0
    assert dropped["disconnect"] == 6
    sender.close()
```

Create `tests/unit/test_sp5_worker_io.py` (verified in the scratch package; `FakeCache` stands for the T1.5 reader):
```python
"""WorkerIO (SP5 T2.2, spec 4.7, R10, R11): a node worker's LoopIO over its control / chunk connections and the
shared-memory cache: chunks to the local learner (or dropped and counted without one), lineups merged per env with
each snapshot loaded once, weights from the cache only when newer, results stamped with the node and the seat
generations and batched, cumulative progress, stop and first-weights signals over the control connection."""
from __future__ import annotations

import multiprocessing as mp
import secrets
import time
from dataclasses import dataclass

import numpy as np
import pytest

from colosseum.core.types import Lineup, SeatAssignment, TrajectoryChunk
from colosseum.net.chunks import ChunkValidator, WeightsMeta
from colosseum.net.local import (
    PROGRESS_INTERVAL_SEC,
    CellLineup,
    ConnectionIntake,
    ControlChannel,
    LearnerRoute,
    Stop,
    WeightsReady,
    WorkerEvict,
    WorkerIO,
    WorkerLineups,
    WorkerProgress,
    WorkerResults,
    address,
    listener,
)
from game_helpers import chunk_v2_payload, learner_role, make_match_result

KEY = b"k" * 32


@dataclass(frozen=True)
class Key:                                   # stands for node.weights.CacheKey in these tests
    kind: str
    agent_id: str = ""
    ident: str = ""


@dataclass
class Cached:
    state_dict: dict
    meta: WeightsMeta


class FakeCache:
    """A WeightCacheReader stand-in: ``latest[agent] = (version, generation)``; ``blobs[key] = state dict``."""

    def __init__(self):
        self.latest: dict[str, tuple[int, int]] = {}
        self.blobs: dict = {}
        self.reads: list = []

    def read_latest(self, agent_id, *, newer_than=-1):
        if agent_id not in self.latest or self.latest[agent_id][0] <= newer_than:
            return None
        version, generation = self.latest[agent_id]
        return Cached({"w": np.full(2, version, np.float32)},
                      WeightsMeta(agent_id=agent_id, kind="latest", generation=generation, policy_version=version))

    def get(self, key):
        self.reads.append(key)
        sd = self.blobs.get(key)
        return None if sd is None else Cached(sd, WeightsMeta(agent_id=key.agent_id, kind="snapshot", generation=0,
                                                              policy_version=0))


def _wait(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _io(cache=None, routes=None, **kw):
    node_end, worker_end = mp.Pipe()
    io = WorkerIO(node=3, worker=1, incarnation=2, control=worker_end, route=None, cache=cache or FakeCache(),
                  routes=routes or {}, authkey=KEY, **kw)
    return io, node_end


def _messages(node_end, kind, timeout=5.0):
    out = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        while node_end.poll(0.05):
            msg = node_end.recv()
            if isinstance(msg, kind):
                out.append(msg)
        if out:
            return out
    return out


def test_chunks_without_a_learner_are_dropped_and_counted():
    io, _ = _io(routes={"a": LearnerRoute("a", None, False, 0)})
    io.send_chunk(TrajectoryChunk.from_payload(chunk_v2_payload(agent_id="a")))
    io.send_chunk(TrajectoryChunk.from_payload(chunk_v2_payload(agent_id="b")))     # no route at all
    assert io.dropped["no_learner"] == 2
    assert io.counters()["dropped_chunks_no_learner"] == 2
    io.close()


def test_chunks_go_to_the_local_learner_as_frames_with_the_seat_generation():
    name = address("abcdef12", secrets.token_hex(4), 3, "learner-a-i1")
    lst = listener(name, KEY)
    node_end, learner_end = mp.Pipe()
    intake = ConnectionIntake(lst, ControlChannel(learner_end), ChunkValidator(learner_role(), agent_id="a",
                                                                               teacher=False),
                              agent_id="a", generation=1, version_floor=0, capacity=8, cap_bytes=1 << 24)
    cache = FakeCache()
    cache.latest["a"] = (4, 1)
    io, _ = _io(cache, routes={"a": LearnerRoute("a", name, False, 1)})
    assert io.poll_weights("a").generation == 1
    io.send_chunk(TrajectoryChunk.from_payload(chunk_v2_payload(version=4, agent_id="a")))
    chunk = intake.get(timeout=10.0)
    assert chunk.policy_version == 4 and intake.dropped == {}
    io.close()
    intake.close()


def test_weights_come_from_the_cache_only_when_newer():
    cache = FakeCache()
    io, _ = _io(cache)
    assert io.poll_weights("a") is None
    cache.latest["a"] = (3, 1)
    payload = io.poll_weights("a")
    assert (payload.policy_version, payload.generation) == (3, 1) and payload.state_dict["w"][0] == 3
    assert io.poll_weights("a") is None                                   # nothing newer
    cache.latest["a"] = (5, 2)
    assert io.poll_weights("a").policy_version == 5


def _lineup(net="ckpt_v5"):
    return Lineup("2p", [SeatAssignment("a", "latest", True, "owner"), SeatAssignment("a", net, False, "snapshots")])


def test_lineups_are_merged_per_env_and_each_snapshot_is_loaded_once():
    cache = FakeCache()
    snap = Key("snapshot", "a", "ckpt_v5")
    cache.blobs[snap] = {"w": np.zeros(2, np.float32)}
    io, node_end = _io(cache)
    node_end.send(WorkerLineups([CellLineup(0, _lineup(), {0: Key("latest", "a"), 1: snap}),
                                 CellLineup(2, _lineup(), {0: Key("latest", "a"), 1: snap})]))
    node_end.send(WorkerLineups([CellLineup(0, _lineup("ckpt_v5"), {0: Key("latest", "a"), 1: snap})]))
    assert _wait(lambda: (io.pump(), len(io._lineups) == 2)[1])
    cmd = io.poll_command()
    assert [lu is not None for lu in cmd.lineups] == [True, False, True]
    assert list(cmd.new_checkpoints) == ["a"] and list(cmd.new_checkpoints["a"]) == ["ckpt_v5"]
    assert cache.reads == [snap] and io.poll_command() is None
    node_end.send(WorkerEvict([snap]))
    node_end.send(WorkerLineups([CellLineup(1, _lineup(), {0: Key("latest", "a"), 1: snap})]))
    assert _wait(lambda: (io.pump(), bool(io._lineups))[1])
    cmd = io.poll_command()
    assert cmd.evict == {"a": ["ckpt_v5"]} and "ckpt_v5" in cmd.new_checkpoints["a"]   # reloaded after the evict


def test_a_snapshot_missing_from_the_cache_is_an_error():
    io, node_end = _io(FakeCache())
    node_end.send(WorkerLineups([CellLineup(0, _lineup(), {1: Key("snapshot", "a", "ckpt_v5")})]))
    assert _wait(lambda: (io.pump(), bool(io._lineups))[1])
    with pytest.raises(RuntimeError, match="not in the weight cache"):
        io.poll_command()


def test_results_are_stamped_with_the_node_and_seat_generations_and_batched():
    cache = FakeCache()
    cache.latest["a"] = (6, 2)
    io, node_end = _io(cache, results_batch=2, results_flush_sec=10.0)
    io.poll_weights("a")
    for k in range(2):
        io.report_result(make_match_result([(0, "a", "latest", "player", 6, 6), (1, "random", "fixed")],
                                           uid=f"R/train/n3/w1_e0_g4_ep{k}"))
    (batch,) = _messages(node_end, WorkerResults)
    assert len(batch.results) == 2
    for result in batch.results:
        assert result.node == 3
        assert result.seats[0].generation == 2 and result.seats[1].generation is None


def test_progress_is_cumulative_and_carries_the_loop_and_drop_counters():
    io, node_end = _io()
    io.stats_fn = lambda: {"episodes": 4, "dropped_results": 1}
    io.add_env_steps(10)
    time.sleep(PROGRESS_INTERVAL_SEC + 0.05)
    io.add_env_steps(5)
    (progress, *_) = _messages(node_end, WorkerProgress)
    assert progress.incarnation == 2 and progress.env_steps == 15
    assert progress.counters["episodes"] == 4 and progress.counters["dropped_chunks_no_learner"] == 0
    io.close()
    final = _messages(node_end, WorkerProgress)
    assert final[-1].env_steps == 15                                     # close sends the final counters


def test_stop_and_first_weights_come_over_the_control_connection():
    io, node_end = _io()
    assert not io.ready() and not io.stopped()
    node_end.send(WeightsReady(["a"]))
    assert _wait(io.ready)
    node_end.send(Stop("budget"))
    assert _wait(io.stopped)


def test_a_closed_node_connection_stops_the_worker():
    io, node_end = _io()
    node_end.close()
    assert _wait(io.stopped)
```

Create `tests/unit/test_sp5_learner_io.py` (verified in the scratch package with file-backed stand-ins for T1.5 and a pickle stand-in for T1.1):
```python
"""The learner on a node (SP5 T2.2; spec 4.6, 4.7, R4, R9, R10): ``learner_process`` trains from frames through a
``ConnectionIntake`` and publishes through ``NodeOutputs``: weights into the shared-memory cache with a
``VersionPublished``, train metrics, the final snapshot as a header and one frame on the control connection; the
version floor lifts the first published version; SP4's queue path (``QueueIntake`` / ``QueueOutputs``) is unchanged."""
from __future__ import annotations

import multiprocessing as mp
import secrets
import threading
import time
from collections import Counter

from colosseum.core.config import LearnerConfig
from colosseum.core.types import TrajectoryChunk
from colosseum.learner.learner import learner_process, set_policy_version
from colosseum.net.chunks import ChunkValidator, decode_checkpoint, encode_chunk
from colosseum.net.frame import payload_byte_cap
from colosseum.net.local import (
    CheckpointHeader,
    ChildHello,
    ConnectionIntake,
    ControlChannel,
    FrameSender,
    LearnerMetrics,
    NodeOutputs,
    Stop,
    VersionPublished,
    address,
    connect,
    listener,
)
from colosseum.node.weights import WeightCacheReader, WeightCacheWriter, cache_prefix, shm_dir
from game_helpers import chunk_v2_payload, learner_appo, learner_role

KEY = b"k" * 32


class NodeEnd:
    """The node's side of a learner's control connection: collects every message (and the frame after a header)."""

    def __init__(self, conn):
        self.conn = conn
        self.messages: list = []
        self.frames: list[bytes] = []
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        while True:
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                return
            self.messages.append(msg)
            if isinstance(msg, CheckpointHeader):
                self.frames.append(self.conn.recv_bytes())

    def of(self, kind):
        return [m for m in self.messages if isinstance(m, kind)]


def _wait(pred, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _start(tmp_path, *, version_floor=0, batches=3):
    run8 = secrets.token_hex(4)
    name = address("abcdef12", run8, 1, "learner-a-i1")
    node_conn, learner_conn = mp.Pipe()
    control = ControlChannel(learner_conn)
    intake = ConnectionIntake(listener(name, KEY), control, ChunkValidator(learner_role(), agent_id="a", teacher=False),
                              agent_id="a", generation=1, version_floor=version_floor, capacity=8,
                              cap_bytes=payload_byte_cap(64))
    prefix = cache_prefix("abcdef12" * 8, run8 * 4, 1)
    outputs = NodeOutputs(control, WeightCacheWriter(shm_dir(), prefix), agent_id="a", generation=1)
    node = NodeEnd(node_conn)
    algo = learner_appo()
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, intake=intake, outputs=outputs,
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"), stop_event=intake.stop_flag,
        progress_counter=intake, total_timesteps=1000, version_floor=version_floor))
    thread.start()
    sender = FrameSender("w0", lambda: _hello(connect(name, KEY, timeout=5)), Counter())
    for v in range(2 * batches):
        chunk = TrajectoryChunk.from_payload(chunk_v2_payload(version=version_floor + v // 2, agent_id="a"))
        assert sender.put(encode_chunk(chunk, generation=1, node=1, worker=0), timeout=30.0)
    return thread, intake, node, prefix, sender, algo


def _hello(conn):
    conn.send(ChildHello("worker", "worker-0", 1, 0, "chunks"))
    return conn


def test_the_learner_trains_from_frames_through_its_intake(tmp_path):
    thread, intake, node, prefix, sender, algo = _start(tmp_path)
    try:
        assert _wait(lambda: len(node.of(LearnerMetrics)) >= 3)
        published = node.of(VersionPublished)
        assert published[0].version == 0 and [p.version for p in published] == sorted(p.version for p in published)
        assert node.of(LearnerMetrics)[-1].metrics["train_step"] >= 3
        latest = WeightCacheReader(shm_dir(), prefix).read_latest("a")
        assert latest is not None and latest.meta.policy_version == published[-1].version
        assert latest.meta.generation == 1
    finally:
        node.conn.send(Stop("test"))
        thread.join(30)
        sender.close()
        intake.close()
    assert not thread.is_alive()
    (header,) = node.of(CheckpointHeader)
    assert (header.kind, header.agent_id, header.generation) == ("final", "a", 1)
    payload, meta = decode_checkpoint(node.frames[0], cap=payload_byte_cap(64))
    assert payload["agent_id"] == "a" and payload["final"] is True and payload["policy_version"] == algo.policy_version
    assert isinstance(payload["trainer_state_bytes"], bytes) and meta["kind"] == "final"


def test_the_version_floor_lifts_the_first_published_version(tmp_path):
    thread, intake, node, _, sender, algo = _start(tmp_path, version_floor=7, batches=1)
    try:
        assert _wait(lambda: node.of(LearnerMetrics))
        assert node.of(VersionPublished)[0].version == 7
        assert node.of(LearnerMetrics)[0].metrics["train_step"] == 8
    finally:
        node.conn.send(Stop("test"))
        thread.join(30)
        sender.close()
        intake.close()


def test_set_policy_version_keeps_the_rest_of_the_state():
    algo = learner_appo()
    before = {k: v for k, v in algo.state_dict().items() if k != "policy_version"}
    set_policy_version(algo, 12)
    after = algo.state_dict()
    assert algo.policy_version == 12 and after["consumed_samples"] == before["consumed_samples"]
```

Create `tests/contract/test_sp5_loop_epochs.py` (verified in the scratch package after the edits of steps 5–6):
```python
"""Epoch-aware match ids and cell seeds (SP5 T2.2; spec 4.3, 7.6; ruling R17): a lineup with an epoch gives match ids
``w<W>_e<E>_g<epoch>_ep<K>`` and episode seeds ``SeedSequence([cell seed, K])`` with the cell seed
``cell_seed(training.seed, epoch, node, worker, env)``; a node worker's uids carry ``n<node>/``; without an epoch and
without the new keywords the loop behaves exactly as in SP4; a node worker never replaces a missing snapshot."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from colosseum.core.types import SeatAssignment, WorkerCommand
from colosseum.utils.seeding import cell_seed, derive_seed, worker_seed
from game_harness import Collected, GameFactory, lineup, make_loop, run_steps
from game_helpers import Tick, TickGame, make_test_model

ROLE2 = TickGame([Tick(acting={0})], 2).spec.roles["player"]


def _episode(length: int) -> list[Tick]:
    return [Tick(acting={0, 1}, rewards={0: 1.0, 1: 1.0}) for _ in range(length)] + [
        Tick(over=True, rewards={0: 1.0, 1: -1.0})]


def _models():
    model = make_test_model(ROLE2)
    return {"a": lambda: model}


def _epoch(lu, epoch):
    return dataclasses.replace(lu, epoch=epoch)


def _seed(cell: int, k: int) -> int:
    return int(np.random.SeedSequence([cell, k]).generate_state(1)[0])


def test_without_an_epoch_ids_and_seeds_are_unchanged():
    loop, col = make_loop(GameFactory((_episode(2), 2)), _models(), [lineup("2p", "a", "a")], seed=5, run_id="R")
    run_steps(loop, 20)
    first = col.results[0]
    assert (first.match_id, first.uid) == ("w0_e0_ep0", "R/train/w0_e0_ep0")
    assert first.seed == int(np.random.SeedSequence([5, 0, 0]).generate_state(1)[0])


def test_an_epoch_goes_into_the_match_id_and_the_node_uid_prefix():
    loop, col = make_loop(GameFactory((_episode(2), 2)), _models(), [_epoch(lineup("2p", "a", "a"), 7)], seed=5,
                          run_id="R", uid_prefix="R/train/n3/", cell_seed_fn=lambda env, epoch: cell_seed(5, epoch, 3,
                                                                                                        0, env))
    run_steps(loop, 20)
    assert [r.match_id for r in col.results[:2]] == ["w0_e0_g7_ep0", "w0_e0_g7_ep1"]
    assert col.results[0].uid == "R/train/n3/w0_e0_g7_ep0"


def test_episode_seeds_follow_the_cell_seed_and_the_episode_index():
    loop, col = make_loop(GameFactory((_episode(2), 2), (_episode(2), 2)), _models(),
                          [_epoch(lineup("2p", "a", "a"), 7), _epoch(lineup("2p", "a", "a"), 7)], seed=5,
                          cell_seed_fn=lambda env, epoch: cell_seed(5, epoch, 3, 0, env))
    run_steps(loop, 30)
    for result in col.results:
        env = int(result.match_id.split("_e", 1)[1].split("_", 1)[0])
        k = int(result.match_id.rsplit("_ep", 1)[1])
        assert result.seed == _seed(cell_seed(5, 7, 3, 0, env), k)
    assert len({r.seed for r in col.results}) == len(col.results)


def test_a_new_epoch_changes_the_ids_and_seeds_of_the_same_env():
    col = Collected()
    col.commands.append(WorkerCommand(lineups=[_epoch(lineup("2p", "a", "a"), 8)]))
    loop, col = make_loop(GameFactory((_episode(2), 2)), _models(), [_epoch(lineup("2p", "a", "a"), 7)], seed=5,
                          collected=col, cell_seed_fn=lambda env, epoch: cell_seed(5, epoch, 3, 0, env))
    run_steps(loop, 20)
    first, second = col.results[0], col.results[1]
    assert first.match_id == "w0_e0_g7_ep0" and second.match_id == "w0_e0_g8_ep1"
    assert second.seed == _seed(cell_seed(5, 8, 3, 0, 0), 1)


def test_without_a_training_seed_episode_seeds_stay_none():
    loop, col = make_loop(GameFactory((_episode(2), 2)), _models(), [_epoch(lineup("2p", "a", "a"), 7)], seed=0,
                          cell_seed_fn=lambda env, epoch: cell_seed(None, epoch, 3, 0, env))
    run_steps(loop, 10)
    assert col.results and all(r.seed is None for r in col.results)


def test_cell_seeds_are_distinct_per_epoch_node_worker_env():
    seeds = {cell_seed(0, g, n, w, e) for g in range(1, 4) for n in range(3) for w in range(2) for e in range(3)}
    assert len(seeds) == 54 and all(0 <= s < 2**32 for s in seeds)
    assert cell_seed(0, 1, 2, 3, 4) == cell_seed(0, 1, 2, 3, 4) and cell_seed(1, 1, 2, 3, 4) != cell_seed(0, 1, 2, 3, 4)
    assert cell_seed(None, 1, 0, 0, 0) is None and worker_seed(None, 1, 0, 1) is None
    assert len({worker_seed(0, n, w, i) for n in range(2) for w in range(2) for i in range(1, 3)}) == 8
    assert cell_seed(0, 1, 0, 0, 0) != derive_seed(0, "cell", 1)


def test_node_workers_never_replace_a_missing_snapshot():
    snapshot_seat = lineup("2p", "a", SeatAssignment("a", "ckpt_v9", collect=False))
    with pytest.raises(ValueError, match="no model for agent 'a'"):
        make_loop(GameFactory((_episode(2), 2)), _models(), [snapshot_seat], snapshot_fallback=False)
    loop, col = make_loop(GameFactory((_episode(2), 2)), _models(), [snapshot_seat])      # SP4's in-process default
    assert [s.network_id for s in loop._runner.lineup(0).seats] == ["latest", "latest"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_local_connections.py tests/unit/test_sp5_worker_io.py tests/unit/test_sp5_learner_io.py tests/contract/test_sp5_loop_epochs.py -q`
Expected: collection errors (`No module named 'colosseum.net.local'`; `cannot import name 'cell_seed'`).

- [ ] **Step 3: Write `src/colosseum/net/local.py`**

Complete (verified in the scratch package as described in the preamble):
```python
"""Same-machine process connections (SP5 spec 4.7, 7.8; plan rulings R3, R4, R10, R11).

A node talks to each of its children (workers, learners, evaluators) over ``multiprocessing.connection`` Unix-socket
connections authenticated with the node's ``authkey``; no ``mp.Queue``, ``mp.Value`` or ``mp.Event`` is shared
between processes that restart, so a ``kill -9`` of any process can only break its own connections.

Addresses. ``address(ca8, run8, node, proc)`` names a listener: ``colosseum-{ca8}-{run8}-n{node}-{proc}`` with
``proc`` = ``node`` (the node's listener for every child) or ``learner-{agent}-i{incarnation}`` (one per learner
incarnation). Linux uses abstract sockets (``"\\0" + name``: no file, released with the process); elsewhere (macOS) a
socket file in ``$COLOSSEUM_RUNTIME_DIR`` or ``<tempdir>/cst-<uid>/`` whose path must be at most 104 characters
(ValueError otherwise).

Connections. Each child opens one ``control`` connection to the node; a worker also opens one ``route`` connection
to the node (frames for learners on other machines, T3.2) and one ``chunks`` connection per local learner it collects
for; the node opens one ``chunks`` connection to each local learner (frames from other machines, T3.2). The first
message on every connection is a ``ChildHello``. Control messages are pickled dataclasses (R3, the trust boundary of
today's queues); chunks and checkpoints travel as frames (``send_bytes``); weights only through the shared-memory
cache (R10). A ``ControlChannel`` serializes the sends of several threads of one process on one connection.

Worker side: ``WorkerIO`` is the ``LoopIO`` of ``RolloutLoop`` (R4). Each learner connection has a ``FrameSender``
thread with a one-frame slot, so ``send_chunk`` waits in short slices (it keeps flushing results and reading control
messages, and sees a stop) while a busy learner applies backpressure; a broken or missing learner drops the chunk and
counts it (``disconnect`` / ``no_learner``, spec 4.7: at most once).

Learner side: ``ConnectionIntake`` accepts the chunk connections, reads whole frames into a bounded deque and stops
reading while it is full (backpressure through the sockets); ``get`` decodes, validates (``ChunkValidator``) and
applies the generation rule (R16), counting every drop by reason. An EOF closes only that connection; a frame cut by
a dying sender closes it and counts ``disconnect``. ``NodeOutputs`` publishes weights into the shared-memory cache and
reports them, sends checkpoints and metrics to the node. ``QueueIntake`` / ``QueueOutputs`` keep SP4's queue path for
the launcher (until T2.5) and the distributed roles (until T7.2).
"""

from __future__ import annotations

import collections
import hashlib
import logging
import multiprocessing.connection as mpc
import os
import queue
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from colosseum.core.types import Lineup, MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand

if TYPE_CHECKING:
    from colosseum.net.chunks import ChunkValidator
    from colosseum.node.weights import CacheKey, WeightCacheReader, WeightCacheWriter

logger = logging.getLogger(__name__)

RUNTIME_DIR_ENV = "COLOSSEUM_RUNTIME_DIR"
MAX_SOCKET_PATH = 104            # macOS sun_path limit
MAX_ABSTRACT_NAME = 107          # Linux: 108 bytes of sun_path minus the leading NUL
CONNECT_RETRY_SEC = 0.05
SEND_SLICE_SEC = 0.5             # a blocked send_chunk wakes this often (results, control messages, stop)
PROGRESS_INTERVAL_SEC = 0.5      # WorkerProgress cadence (spec 4.4: cumulative counters)
INTAKE_REPORT_SEC = 1.0          # IntakeReport cadence
_GENERATIONS_KEPT = 256          # version -> generation entries a worker keeps per agent


# ---------------------------------------------------------------------------------------------------------------
# Addresses, listeners, connections
# ---------------------------------------------------------------------------------------------------------------


def address(ca8: str, run8: str, node: int, proc: str) -> str:
    """The name of a node-local listener (module docstring)."""
    return f"colosseum-{ca8}-{run8}-n{int(node)}-{proc}"


def _socket_address(name: str) -> str:
    if sys.platform.startswith("linux"):
        if len(name.encode()) > MAX_ABSTRACT_NAME:
            raise ValueError(f"socket name {name!r} is longer than {MAX_ABSTRACT_NAME} bytes")
        return "\0" + name
    root = Path(os.environ.get(RUNTIME_DIR_ENV) or Path(tempfile.gettempdir()) / f"cst-{os.getuid()}")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = str(root / name)
    if len(path) > MAX_SOCKET_PATH:
        raise ValueError(f"socket path {path!r} is longer than {MAX_SOCKET_PATH} characters; set "
                         f"{RUNTIME_DIR_ENV} to a shorter directory")
    return path


def listener(name: str, authkey: bytes) -> mpc.Listener:
    """A listener for ``name`` (a stale socket file of a dead process is replaced; abstract names cannot be stale)."""
    addr = _socket_address(name)
    if not addr.startswith("\0") and os.path.exists(addr):
        os.unlink(addr)
    return mpc.Listener(addr, family="AF_UNIX", backlog=64, authkey=authkey)


def connect(name: str, authkey: bytes, *, timeout: float) -> mpc.Connection:
    """Connect to ``name``, retrying while the listener does not exist yet; ``ConnectionError`` after ``timeout``."""
    addr = _socket_address(name)
    deadline = time.monotonic() + timeout
    while True:
        try:
            return mpc.Client(addr, family="AF_UNIX", authkey=authkey)
        except (FileNotFoundError, ConnectionRefusedError) as e:
            if time.monotonic() >= deadline:
                raise ConnectionError(f"no listener {name!r} after {timeout:.1f} s ({e})") from e
            time.sleep(CONNECT_RETRY_SEC)


def listener_socket(lst: mpc.Listener) -> Any:
    """The listening socket of ``lst`` (CPython's ``Listener._listener._socket``), for ``wait`` with a timeout, so an
    accept thread can be stopped (closing a socket does not wake a thread blocked in ``accept`` on Linux)."""
    return lst._listener._socket


# ---------------------------------------------------------------------------------------------------------------
# Control messages (pickled dataclasses, R3)
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class ChildHello:
    role: Literal["worker", "learner", "evaluator", "node"]
    name: str
    incarnation: int
    pid: int
    channel: Literal["control", "route", "chunks"]


@dataclass
class Stop:
    reason: str


@dataclass
class ChildError:
    """Sent by a child before it exits with an error (``contract``: EnvContractError, ``player``: PlayerError,
    ``crash``: any other exception); ``context`` is the error's location text."""

    kind: Literal["contract", "player", "crash"]
    message: str
    context: str


@dataclass
class CellLineup:
    """A lineup for one env of a worker; ``lineup.epoch`` is set; ``keys`` the cache key of every seat's weights
    (R11: all in the cache when this is sent)."""

    env: int
    lineup: Lineup
    keys: dict[int, CacheKey]


@dataclass
class WorkerLineups:
    cells: list[CellLineup]


@dataclass
class WorkerEvict:
    keys: list[CacheKey]


@dataclass
class LearnerRoute:
    """Where a worker sends ``agent_id``'s chunks: a local learner's listener ``address``, or the node's route
    connection (``via_node``); neither: the agent has no learner (chunks are dropped as ``no_learner``)."""

    agent_id: str
    address: str | None
    via_node: bool
    generation: int


@dataclass
class WeightsReady:
    """The first-weights wait of a worker ends (``agents``: those with weights in the cache)."""

    agents: list[str]


@dataclass
class WorkerResults:
    results: list[MatchResult]


@dataclass
class WorkerProgress:
    """A worker incarnation's cumulative counters (env steps and stats), every ``PROGRESS_INTERVAL_SEC``."""

    incarnation: int
    env_steps: int
    counters: dict[str, int]


@dataclass
class LearnerEnvSteps:
    env_steps: int


@dataclass
class VersionPublished:
    agent_id: str
    generation: int
    version: int
    segment: str
    size: int
    digest: str
    teacher_active: bool


@dataclass
class CheckpointHeader:
    """Followed on the same connection by one ``send_bytes(encode_checkpoint(...))`` of ``size`` bytes."""

    kind: Literal["periodic", "final", "recovery"]
    agent_id: str
    generation: int
    version: int
    size: int


@dataclass
class LearnerMetrics:
    metrics: dict


@dataclass
class IntakeReport:
    depth: int
    dropped: dict[str, int]


class ControlChannel:
    """The sending side of one connection, safe for several threads (one lock); ``closed`` after a failed send.
    Receiving stays with one thread of the process."""

    def __init__(self, conn: mpc.Connection) -> None:
        self.conn = conn
        self._lock = threading.Lock()
        self.closed = False

    def send(self, message: Any) -> bool:
        with self._lock:
            if self.closed:
                return False
            try:
                self.conn.send(message)
                return True
            except (OSError, EOFError, ValueError):
                self.closed = True
                return False

    def send_with_frame(self, message: Any, frame: bytes | memoryview) -> bool:
        """``message`` then ``frame`` as one unit (no other send in between)."""
        with self._lock:
            if self.closed:
                return False
            try:
                self.conn.send(message)
                self.conn.send_bytes(frame)
                return True
            except (OSError, EOFError, ValueError):
                self.closed = True
                return False

    def close(self) -> None:
        with self._lock:
            self.closed = True
            try:
                self.conn.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------------------------------------------


class FrameSender:
    """One connection's sender thread with a one-frame slot (module docstring). ``opener`` makes the connection in
    the thread (the hello is sent there too). ``put`` waits up to ``timeout`` for the slot; after a failure the sender
    is ``broken`` and every later or queued frame is counted in ``dropped`` (reason ``disconnect``)."""

    def __init__(self, name: str, opener: Callable[[], mpc.Connection], dropped: Counter) -> None:
        self.name = name
        self._opener = opener
        self._dropped = dropped
        self._slot: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self.broken = False
        self._closed = False
        self._conn: mpc.Connection | None = None
        self._thread = threading.Thread(target=self._run, name=f"frames-{name}", daemon=True)
        self._thread.start()

    def put(self, frame: bytes, timeout: float) -> bool:
        if self.broken or self._closed:
            self._count(1)
            return True                         # consumed (dropped and counted): the caller goes on
        try:
            self._slot.put(frame, timeout=timeout)
            return True
        except queue.Full:
            return False

    def _count(self, n: int) -> None:
        with self._lock:
            self._dropped["disconnect"] += n

    def _run(self) -> None:
        try:
            self._conn = self._opener()
        except Exception as e:  # noqa: BLE001 - the learner is gone or never came up
            logger.warning(f"frame sender {self.name}: cannot connect ({type(e).__name__}: {e}); its chunks are "
                           f"dropped (dropped_chunks.disconnect)")
            self._fail()
            return
        while not self._closed:
            try:
                frame = self._slot.get(timeout=SEND_SLICE_SEC)
            except queue.Empty:
                continue
            if frame is None:
                break
            try:
                self._conn.send_bytes(frame)
            except (OSError, EOFError, ValueError) as e:
                logger.warning(f"frame sender {self.name}: the receiver is gone ({type(e).__name__}: {e}); its "
                               f"chunks are dropped (dropped_chunks.disconnect)")
                self._count(1)
                self._fail()
                return
        self._close_conn()

    def _fail(self) -> None:
        self.broken = True
        n = 0
        while True:
            try:
                if self._slot.get_nowait() is not None:
                    n += 1
            except queue.Empty:
                break
        if n:
            self._count(n)
        self._close_conn()

    def _close_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass

    def close(self, timeout: float = 1.0) -> None:
        """Stop the thread; a frame still in the slot is counted as dropped."""
        self._closed = True
        try:
            pending = self._slot.get_nowait()
            if pending is not None:
                self._count(1)
        except queue.Empty:
            pass
        try:
            self._slot.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout)


class _ResultSink:
    """``ResultBatcher``'s queue over the control channel: a failed send is a drop (``queue.Full``)."""

    def __init__(self, control: ControlChannel) -> None:
        self._control = control

    def put_nowait(self, batch: list[MatchResult]) -> None:
        if not self._control.send(WorkerResults(batch)):
            raise queue.Full


class WorkerIO:
    """``LoopIO`` of a node's worker over its connections and the shared-memory cache (module docstring).

    ``pump`` reads every pending control message (lineups, evictions, routes, ``WeightsReady``, ``Stop``); the loop
    callbacks call it. ``stats_fn`` (set by the worker body once the loop exists) gives the loop's counters for
    ``WorkerProgress``; this object adds its chunk drops (``dropped_chunks_no_learner``, ``dropped_chunks_disconnect``)
    and the result batcher's drops (into ``dropped_results``)."""

    def __init__(self, *, node: int, worker: int, incarnation: int, control: mpc.Connection,
                 route: mpc.Connection | None, cache: WeightCacheReader, routes: Mapping[str, LearnerRoute],
                 authkey: bytes, results_batch: int = 256, results_flush_sec: float = 0.1,
                 clock: Callable[[], float] = time.monotonic) -> None:
        from colosseum.worker.rollout_worker import ResultBatcher

        self.node, self.worker, self.incarnation = int(node), int(worker), int(incarnation)
        self._control = ControlChannel(control)
        self._route_conn = route
        self._cache = cache
        self._authkey = authkey
        self._clock = clock
        self._routes: dict[str, LearnerRoute] = dict(routes)
        self._senders: dict[str, FrameSender] = {}       # by agent (local learner) or "<route>"
        self.dropped: Counter = Counter()                # chunk drops by reason
        self._batcher = ResultBatcher(_ResultSink(self._control), worker, max_results=results_batch,
                                      max_age_sec=results_flush_sec, clock=clock)
        self._lineups: dict[int, CellLineup] = {}        # pending, by env (a later one replaces an earlier one)
        self._evict: list[CacheKey] = []
        self._given: set[tuple[str, str]] = set()        # (agent, checkpoint id) handed to the loop
        self._loaded: dict[str, int] = {}                # agent -> version loaded in the loop
        self._generations: dict[str, collections.OrderedDict[int, int]] = {}
        self._ready = False
        self._stopped = False
        self._env_steps = 0
        self._last_progress = clock()
        self.stats_fn: Callable[[], dict[str, int]] | None = None

    # ---- control messages ----------------------------------------------------------------------------------

    def pump(self) -> None:
        conn = self._control.conn
        while not self._stopped:
            try:
                if not conn.poll(0):
                    return
                msg = conn.recv()
            except (OSError, EOFError):
                self._stopped = True            # the node is gone
                return
            if isinstance(msg, WorkerLineups):
                for cell in msg.cells:
                    self._lineups[cell.env] = cell
            elif isinstance(msg, WorkerEvict):
                self._evict.extend(msg.keys)
            elif isinstance(msg, LearnerRoute):
                self._set_route(msg)
            elif isinstance(msg, WeightsReady):
                self._ready = True
            elif isinstance(msg, Stop):
                self._stopped = True
            else:
                logger.warning(f"worker {self.worker}: unexpected control message {type(msg).__name__}")

    def _set_route(self, route: LearnerRoute) -> None:
        old = self._routes.get(route.agent_id)
        self._routes[route.agent_id] = route
        if old is not None and (old.address, old.via_node, old.generation) != (route.address, route.via_node,
                                                                              route.generation):
            sender = self._senders.pop(route.agent_id, None)
            if sender is not None:
                sender.close()

    def ready(self) -> bool:
        self.pump()
        return self._ready

    def stopped(self) -> bool:
        self.pump()
        return self._stopped

    def take_lineups(self) -> dict[int, CellLineup]:
        """The pending lineups by env (the worker body reads its first ones before the loop exists)."""
        self.pump()
        out, self._lineups = self._lineups, {}
        return out

    # ---- LoopIO --------------------------------------------------------------------------------------------

    def loop_io(self) -> Any:
        from colosseum.worker.rollout_loop import LoopIO

        return LoopIO(send_chunk=self.send_chunk, poll_weights=self.poll_weights, report_result=self.report_result,
                      poll_command=self.poll_command, add_env_steps=self.add_env_steps)

    def send_chunk(self, chunk: TrajectoryChunk) -> None:
        from colosseum.net.chunks import encode_chunk

        route = self._routes.get(chunk.agent_id)
        if route is None or (route.address is None and not route.via_node):
            self.dropped["no_learner"] += 1
            return
        sender = self._sender(route)
        frame = encode_chunk(chunk, generation=self._generation_of(chunk.agent_id, chunk.policy_version),
                             node=self.node, worker=self.worker, compress=False)   # lz4 for remote targets: T3.2
        while not sender.put(frame, timeout=SEND_SLICE_SEC):
            self.idle()
            if self._stopped:
                self.dropped["disconnect"] += 1
                return

    def _sender(self, route: LearnerRoute) -> FrameSender:
        key = route.agent_id if not route.via_node else "<route>"
        sender = self._senders.get(key)
        if sender is None:
            if route.via_node:
                conn = self._route_conn
                if conn is None:
                    raise RuntimeError(f"worker {self.worker}: no route connection for {route.agent_id!r}")
                opener: Callable[[], mpc.Connection] = lambda: conn  # noqa: E731
            else:
                name = route.address

                def opener() -> mpc.Connection:
                    c = connect(name, self._authkey, timeout=5.0)
                    c.send(ChildHello("worker", f"worker-{self.worker}", self.incarnation, os.getpid(), "chunks"))
                    return c
            sender = self._senders[key] = FrameSender(f"{key}@worker-{self.worker}", opener, self.dropped)
        return sender

    def poll_weights(self, agent_id: str) -> WeightPayload | None:
        cached = self._cache.read_latest(agent_id, newer_than=self._loaded.get(agent_id, -1))
        if cached is None:
            return None
        meta = cached.meta
        self._loaded[agent_id] = int(meta.policy_version)
        gens = self._generations.setdefault(agent_id, collections.OrderedDict())
        gens[int(meta.policy_version)] = int(meta.generation)
        while len(gens) > _GENERATIONS_KEPT:
            gens.popitem(last=False)
        return WeightPayload(agent_id=agent_id, policy_version=int(meta.policy_version),
                             state_dict=cached.state_dict, teacher_active=bool(meta.teacher_active),
                             generation=int(meta.generation))

    def _generation_of(self, agent_id: str, version: int | None) -> int:
        if version is None:
            return 0
        return self._generations.get(agent_id, {}).get(int(version), 0)

    def poll_command(self) -> WorkerCommand | None:
        self.pump()
        if not self._lineups and not self._evict:
            return None
        evict: dict[str, list[str]] = {}
        for key in self._evict:
            if key.kind == "snapshot":
                evict.setdefault(key.agent_id, []).append(key.ident)
                self._given.discard((key.agent_id, key.ident))
        self._evict = []
        new: dict[str, dict[str, dict]] = {}
        lineups: list[Lineup | None] = []
        for env, cell in sorted(self._lineups.items()):
            for key in cell.keys.values():
                if key.kind != "snapshot" or (key.agent_id, key.ident) in self._given:
                    continue
                cached = self._cache.get(key)
                if cached is None:   # the node sends a lineup only when its weights are cached (R11)
                    raise RuntimeError(f"worker {self.worker}: snapshot {key} of a granted lineup is not in the "
                                       f"weight cache")
                new.setdefault(key.agent_id, {})[key.ident] = cached.state_dict
                self._given.add((key.agent_id, key.ident))
            lineups.extend([None] * (env + 1 - len(lineups)))
            lineups[env] = cell.lineup
        self._lineups = {}
        return WorkerCommand(lineups=lineups, new_checkpoints=new, evict=evict)

    def report_result(self, result: MatchResult) -> None:
        result.node = self.node
        for seat in result.seats:
            if seat.network_id == "latest":
                seat.generation = self._generation_of(seat.agent_id, seat.version_end) or None
        self._batcher.add(result)

    def add_env_steps(self, n: int) -> None:
        self._env_steps += int(n)
        self.idle()

    # ---- housekeeping ----------------------------------------------------------------------------------------

    def idle(self) -> None:
        """Flush due results and progress, read control messages (every loop step and every blocked-send slice)."""
        self._batcher.flush_if_due()
        if self._clock() - self._last_progress >= PROGRESS_INTERVAL_SEC:
            self.send_progress()
        self.pump()

    def counters(self) -> dict[str, int]:
        stats = dict(self.stats_fn()) if self.stats_fn is not None else {}
        stats["dropped_results"] = int(stats.get("dropped_results", 0)) + self._batcher.dropped
        stats["dropped_chunks_no_learner"] = int(self.dropped["no_learner"])
        stats["dropped_chunks_disconnect"] = int(self.dropped["disconnect"])
        return {k: int(v) for k, v in stats.items()}

    def send_progress(self) -> None:
        self._last_progress = self._clock()
        self._control.send(WorkerProgress(self.incarnation, self._env_steps, self.counters()))

    def close(self) -> None:
        """Flush the pending results, send the final progress, stop the sender threads."""
        self._batcher.flush()
        self.send_progress()
        for sender in self._senders.values():
            sender.close()
        self._senders = {}

    def send_error(self, error: ChildError) -> None:
        self._control.send(error)


# ---------------------------------------------------------------------------------------------------------------
# Learner side
# ---------------------------------------------------------------------------------------------------------------


class ChunkIntake(Protocol):
    def get(self, timeout: float) -> TrajectoryChunk: ...       # queue.Empty on timeout

    @property
    def depth(self) -> int: ...

    @property
    def dropped(self) -> dict[str, int]: ...


class LearnerOutputs(Protocol):
    checkpoints: bool                      # whether checkpoints are wanted (SP4: a checkpoint queue was given)

    def publish_weights(self, payload: WeightPayload) -> None: ...

    def send_checkpoint(self, payload: dict, *, kind: Literal["periodic", "final", "recovery"], block: bool) -> bool:
        ...

    def send_metrics(self, metrics: dict) -> None: ...

    def finish(self, stop_event: Any) -> None: ...


class ConnectionIntake:
    """``ChunkIntake`` of a node's learner (module docstring). ``control``: the learner's control channel to its
    node (this object's reader thread is its only reader: ``LearnerEnvSteps`` -> ``value`` / ``env_steps``,
    ``Stop`` or EOF -> ``stop_flag``); ``capacity``: frames buffered (``learner.queue_size``); ``cap_bytes``: the
    largest frame accepted (``payload_byte_cap``). ``version_floor``: the first version of this generation; a chunk
    of an older generation is kept only below it (R16)."""

    def __init__(self, lst: mpc.Listener, control: ControlChannel, validator: ChunkValidator, *, agent_id: str,
                 generation: int, version_floor: int, capacity: int, cap_bytes: int) -> None:
        from colosseum.net.chunks import decode_chunk

        self._listener = lst
        self._control = control
        self._validator = validator
        self.agent_id = agent_id
        self.generation = int(generation)
        self.version_floor = int(version_floor)
        self._capacity = max(1, int(capacity))
        self._cap = int(cap_bytes)
        self._decode = decode_chunk
        self._frames: collections.deque[bytes] = collections.deque()
        self._cond = threading.Condition()
        self._dropped: Counter = Counter()
        self._new: list[mpc.Connection] = []
        self._conns: list[mpc.Connection] = []
        self._wake_r, self._wake_w = mpc.Pipe(duplex=False)
        self._closed = threading.Event()
        self.stop_flag = threading.Event()
        self.env_steps = 0
        self._accept_thread = threading.Thread(target=self._accept_loop, name=f"intake-accept-{agent_id}",
                                               daemon=True)
        self._read_thread = threading.Thread(target=self._read_loop, name=f"intake-read-{agent_id}", daemon=True)
        self._accept_thread.start()
        self._read_thread.start()

    # ---- ChunkIntake ---------------------------------------------------------------------------------------

    @property
    def value(self) -> int:
        """The hub's global env-step counter (``learner_process``'s ``progress_counter``)."""
        return self.env_steps

    @property
    def depth(self) -> int:
        with self._cond:
            return len(self._frames)

    @property
    def dropped(self) -> dict[str, int]:
        with self._cond:
            return dict(self._dropped)

    def get(self, timeout: float) -> TrajectoryChunk:
        deadline = time.monotonic() + timeout
        while True:
            with self._cond:
                while not self._frames:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise queue.Empty
                    self._cond.wait(left)
                frame = self._frames.popleft()
                was_full = len(self._frames) + 1 >= self._capacity
            if was_full:
                self._wake()
            chunk = self._accept(frame)
            if chunk is not None:
                return chunk

    def _accept(self, frame: bytes) -> TrajectoryChunk | None:
        from colosseum.net.chunks import ChunkError

        try:
            chunk, meta = self._decode(frame, cap=self._cap)
            if chunk.agent_id != self.agent_id or meta.agent_id != self.agent_id:
                self._drop("invalid")                 # a misrouted chunk
                return None
            gen = int(meta.generation)
            if gen != 0 and gen != self.generation and (gen > self.generation
                                                        or chunk.policy_version >= self.version_floor):
                self._drop("stale_generation")
                return None
            self._validator.validate(chunk)
        except ChunkError as e:
            self._drop(getattr(e, "reason", "invalid"))
            return None
        except Exception:  # noqa: BLE001 - untrusted bytes: any decode problem is an invalid frame
            self._drop("invalid")
            return None
        return chunk

    def _drop(self, reason: str, n: int = 1) -> None:
        with self._cond:
            first = self._dropped[reason] == 0
            self._dropped[reason] += n
        if first:
            logger.warning(f"learner {self.agent_id}: dropped a chunk ({reason}); further drops are only counted "
                           f"(dropped_chunks)")

    # ---- threads --------------------------------------------------------------------------------------------

    def _wake(self) -> None:
        try:
            self._wake_w.send_bytes(b"w")
        except OSError:
            pass

    def _accept_loop(self) -> None:
        sock = listener_socket(self._listener)
        while not self._closed.is_set():
            try:
                if not mpc.wait([sock], timeout=0.5):
                    continue
                conn = self._listener.accept()
            except OSError:
                if self._closed.is_set():
                    return
                continue
            try:
                hello = conn.recv() if conn.poll(5.0) else None
            except (OSError, EOFError):
                hello = None
            if not isinstance(hello, ChildHello) or hello.channel != "chunks":
                conn.close()
                continue
            with self._cond:
                self._new.append(conn)
            self._wake()

    def _read_loop(self) -> None:
        last_report = time.monotonic()
        control = self._control.conn
        while not self._closed.is_set():
            with self._cond:
                self._conns.extend(self._new)
                self._new = []
                full = len(self._frames) >= self._capacity
            waitables: list[Any] = [control, self._wake_r] + ([] if full else list(self._conns))
            try:
                ready = mpc.wait(waitables, timeout=0.5)
            except OSError:
                ready = []
            for obj in ready:
                if obj is self._wake_r:
                    self._drain_wake()
                elif obj is control:
                    self._read_control()
                else:
                    self._read_frame(obj)
            if time.monotonic() - last_report >= INTAKE_REPORT_SEC:
                last_report = time.monotonic()
                self._control.send(IntakeReport(self.depth, self.dropped))

    def _drain_wake(self) -> None:
        try:
            while self._wake_r.poll(0):
                self._wake_r.recv_bytes()
        except (OSError, EOFError):
            pass

    def _read_control(self) -> None:
        try:
            msg = self._control.conn.recv()
        except (OSError, EOFError):
            self.stop_flag.set()                  # the node is gone: stop (final snapshot is best effort)
            self._closed.set()
            return
        if isinstance(msg, LearnerEnvSteps):
            self.env_steps = int(msg.env_steps)
        elif isinstance(msg, Stop):
            self.stop_flag.set()
        else:
            logger.warning(f"learner {self.agent_id}: unexpected control message {type(msg).__name__}")

    def _read_frame(self, conn: mpc.Connection) -> None:
        try:
            frame = conn.recv_bytes(self._cap)
        except EOFError:                          # the sender closed between frames: nothing lost
            self._close_conn(conn)
            return
        except OSError as e:                      # cut mid-frame (killed sender) or larger than the cap
            self._close_conn(conn)
            self._drop("invalid" if "bad message length" in str(e) else "disconnect")
            return
        with self._cond:
            self._frames.append(frame)
            self._cond.notify()

    def _close_conn(self, conn: mpc.Connection) -> None:
        if conn in self._conns:
            self._conns.remove(conn)
        try:
            conn.close()
        except OSError:
            pass

    def close(self, timeout: float = 2.0) -> None:
        self._closed.set()
        self._wake()
        self._accept_thread.join(timeout)
        self._read_thread.join(timeout)
        for conn in self._conns + self._new:
            try:
                conn.close()
            except OSError:
                pass
        try:
            self._listener.close()
        except OSError:
            pass


class NodeOutputs:
    """``LearnerOutputs`` of a node's learner (module docstring): weights into the cache + ``VersionPublished``,
    checkpoints as ``CheckpointHeader`` + frame, metrics as ``LearnerMetrics``; all on ``control``."""

    checkpoints = True

    def __init__(self, control: ControlChannel, cache: WeightCacheWriter, *, agent_id: str, generation: int) -> None:
        self._control = control
        self._cache = cache
        self.agent_id = agent_id
        self.generation = int(generation)
        self._cache_error_logged = False

    def publish_weights(self, payload: WeightPayload) -> None:
        from colosseum.net.chunks import WeightsMeta, encode_weights
        from colosseum.node.weights import CacheFullError

        meta = WeightsMeta(agent_id=self.agent_id, kind="latest", generation=self.generation,
                           policy_version=int(payload.policy_version), teacher_active=bool(payload.teacher_active))
        frame = encode_weights(payload.state_dict, meta)
        try:
            segment = self._cache.publish_latest(self.agent_id, frame, version=meta.policy_version,
                                                 generation=self.generation, teacher_active=meta.teacher_active)
        except CacheFullError as e:
            logger.error(f"learner {self.agent_id}: v{meta.policy_version} not published: {e}",
                         exc_info=not self._cache_error_logged)
            self._cache_error_logged = True
            return
        digest = hashlib.blake2b(frame, digest_size=16).hexdigest()
        self._control.send(VersionPublished(self.agent_id, self.generation, meta.policy_version, segment,
                                            len(frame), digest, meta.teacher_active))

    def send_checkpoint(self, payload: dict, *, kind: Literal["periodic", "final", "recovery"], block: bool) -> bool:
        """Header + frame on the control connection (blocking while the node reads; ``block`` is accepted for the
        protocol: a socket send has no non-blocking form that keeps the frame whole). False: the node is gone."""
        from colosseum.net.chunks import encode_checkpoint

        frame = encode_checkpoint(payload, kind=kind, generation=self.generation)
        header = CheckpointHeader(kind, self.agent_id, self.generation, int(payload["policy_version"]), len(frame))
        return self._control.send_with_frame(header, frame)

    def send_metrics(self, metrics: dict) -> None:
        self._control.send(LearnerMetrics(dict(metrics)))

    def finish(self, stop_event: Any) -> None:
        """Nothing to release (the cache segments stay for the readers)."""


class QueueIntake:
    """``ChunkIntake`` over SP4's trajectory ``mp.Queue`` of payload dicts (decode + slot check, misrouted agent ->
    ValueError, as SP4's ``collect_batch``)."""

    def __init__(self, q: Any, *, agent_id: str | None = None) -> None:
        self._q = q
        self.agent_id = agent_id

    def get(self, timeout: float) -> TrajectoryChunk:
        from colosseum.core.types import validate_slot_structure

        item = self._q.get(timeout=timeout)
        if isinstance(item, TrajectoryChunk):
            return item
        if not isinstance(item, dict):
            raise TypeError(f"trajectory queue item must be a chunk payload dict, got {type(item).__name__}")
        chunk = TrajectoryChunk.from_payload(item)
        if self.agent_id is not None and chunk.agent_id != self.agent_id:
            raise ValueError(f"learner of agent {self.agent_id!r} received a chunk of agent {chunk.agent_id!r}; "
                             f"check the workers' learner addresses (run-workers -l AGENT=HOST:PORT)")
        validate_slot_structure(chunk)
        return chunk

    @property
    def depth(self) -> int:
        try:
            return int(self._q.qsize())
        except (NotImplementedError, AttributeError):
            return -1

    @property
    def dropped(self) -> dict[str, int]:
        return {}


@dataclass
class QueueOutputs:
    """``LearnerOutputs`` over SP4's queues: newest-wins weight mailboxes, the checkpoint queue (final: blocking put
    and flush), the metrics queue (never blocks); ``finish`` is SP4's ``_release_weight_queues``."""

    weight_queues: list = field(default_factory=list)
    checkpoint_queue: Any = None
    metrics_queue: Any = None
    trajectory_queue: Any = None
    weight_sync_interval: float = 5.0

    @property
    def checkpoints(self) -> bool:
        return self.checkpoint_queue is not None

    def publish_weights(self, payload: WeightPayload) -> None:
        from colosseum.core.ipc import put_latest

        for wq in self.weight_queues:
            if not put_latest(wq, payload):
                logger.debug(f"Learner [{payload.agent_id}]: weight mailbox busy, v{payload.policy_version} not "
                             f"delivered")

    def send_checkpoint(self, payload: dict, *, kind: Literal["periodic", "final", "recovery"], block: bool) -> bool:
        from colosseum.learner.learner import FINAL_CHECKPOINT_TIMEOUT_SEC, send_checkpoint
        from colosseum.utils.process import SHUTDOWN_GRACE_SEC, flush_queue

        if self.checkpoint_queue is None:
            return False
        sent = send_checkpoint(self.checkpoint_queue, payload, block=block, timeout=FINAL_CHECKPOINT_TIMEOUT_SEC)
        if kind == "final":
            flush_queue(self.checkpoint_queue, warn_after=SHUTDOWN_GRACE_SEC)
        return sent

    def send_metrics(self, metrics: dict) -> None:
        if self.metrics_queue is None:
            return
        try:
            self.metrics_queue.put_nowait(metrics)
        except queue.Full:
            pass

    def finish(self, stop_event: Any) -> None:
        from colosseum.learner.learner import _release_weight_queues, _weight_flush_timeout

        trajectory = self.trajectory_queue if self.trajectory_queue is not None else queue.Queue()
        _release_weight_queues(self.weight_queues, trajectory, stop_event,
                               timeout=_weight_flush_timeout(self.weight_sync_interval))
```

Notes for the implementer:
- `FrameSender.put` returns True when it *consumed* the frame: queued, or dropped and counted because the sender is broken; False only when the slot stayed busy for `timeout` (the caller then runs `WorkerIO.idle` and tries again). `WorkerIO.send_chunk` therefore never blocks for more than `SEND_SLICE_SEC` without flushing results, reporting progress and reading control messages, and a `Stop` ends a blocked send (the chunk is counted as `disconnect`).
- The route connection (`via_node`) is used by T3.2; until then no route is `via_node` and the worker body opens the route connection but sends nothing on it. lz4 for frames that leave the machine is T3.2's (overview deviation "lz4 only for frames that leave the machine").
- `ConnectionIntake` is the single reader of the learner's control connection; every send on it goes through the shared `ControlChannel` lock (the reader thread sends `IntakeReport`, the learner's main thread sends weights announcements, checkpoints and metrics).
- `ConnectionIntake` accepts only `ChildHello(channel="chunks")` connections; anything else is closed.
- `listener_socket` reads CPython's private `Listener._listener._socket`; `test_sp5_local_connections.py` covers it through the accept loop (a change in CPython fails those tests, not production silently).
- `QueueOutputs.finish` passes a dummy queue to SP4's `_release_weight_queues` when no trajectory queue was given.

- [ ] **Step 4: The learner (`src/colosseum/learner/learner.py`)**

Replace `learner_process` and `_push_weights`, add `set_policy_version`, and let `collect_batch` take decoded chunks. Imports: drop `multiprocessing as mp`, `SharedCounter`, `put_latest` and `flush_queue` (moved into `QueueOutputs`); add `TYPE_CHECKING` and, under it, `from colosseum.net.local import ChunkIntake, LearnerOutputs`. Update the module docstring's points 1, 3, 4 ("from its intake", "through its outputs: the node's shared-memory cache, SP4's mailboxes", "final snapshot through the outputs") and the progress paragraph (the counter is the hub's global env-step counter, through `ConnectionIntake.value`; `None` keeps SP2's distributed rule until T7.2).

The new functions (verified: with them, `queue_learner_io` and the ports below, `test_learner_v2.py`, `test_learner_v2_exit.py`, `test_sp2_ipc_latest.py`, `test_sp3_dagger_loss.py`, `test_sp2_learner_entry.py`, `test_sp3_init.py`, `test_sp3_learner_factory.py`, `test_sp2_distributed_roles.py`, `test_sp2_pipelines.py`, `test_sp2_budget_stop.py`, `test_sp2_launcher_lifecycle.py`, `test_sp2_launcher_checkpoints.py`, `test_sp4_first_weights.py`, `test_sp4_result_flow.py` and `test_sp4_pulse.py` pass: 179 tests):
```python
def set_policy_version(algorithm: BaseAlgorithm, version: int) -> None:
    """Set ``algorithm``'s policy version (R9: a new learner generation starts at the hub's version floor); the
    rest of its state is unchanged (a ``state_dict`` round trip, as ``apply_resume_state`` does without a trainer
    state)."""
    state = algorithm.state_dict()
    state["policy_version"] = int(version)
    algorithm.load_state_dict(state)


def learner_process(
    *,
    agent_id: str,
    algorithm_factory: Callable[[], BaseAlgorithm],
    intake: ChunkIntake,
    outputs: LearnerOutputs,
    config: LearnerConfig,
    stop_event: Any,
    checkpoint_interval: int = 0,
    resume_state: dict | None = None,
    progress_counter: Any | None = None,
    total_timesteps: int = 0,
    weight_sync_interval: float = 5.0,
    version_floor: int = 0,
    recovery_interval_sec: float | None = None,
) -> None:
    """Main learner process function (see module docstring).

    Args:
        agent_id: the trainable agent this learner owns
        algorithm_factory: factory to create the algorithm (includes network)
        intake: where decoded, checked chunks come from (``net.local.ConnectionIntake`` on a node,
            ``QueueIntake`` on SP4's queue path); ``collect_batch`` takes exactly ``batch_chunks`` of them
        outputs: where weights, checkpoints and metrics go (``NodeOutputs`` / ``QueueOutputs``)
        config: learner configuration
        stop_event: ``.is_set()`` / ``.wait(t)``: the run stops (``ConnectionIntake.stop_flag`` on a node)
        checkpoint_interval: send a checkpoint every N policy versions (0 = only the final one)
        resume_state: optional ``checkpoint_manager.resolve_resume`` result (numpy
            weights, trainer-state bytes or None, policy_version) to resume from
        progress_counter: ``.value`` = the global env-step counter (the hub's, through ``ConnectionIntake``;
            SP4's ``SharedCounter``); None: progress is ``consumed_samples / total_timesteps`` (SP2 distributed)
        total_timesteps: ``training.total_timesteps`` (0 = no budget: progress stays 0)
        weight_sync_interval: the workers' ``rollout.weight_sync_interval_sec`` (``QueueOutputs.finish``)
        version_floor: the first version this learner may publish (R9: the hub's max announced + 1); the
            policy version is raised to it before the first publish
        recovery_interval_sec: recovery points (T4.2); None: none
    """
    logger.info(f"Learner [{agent_id}]: starting on device={resolve_device(config.device)}")

    algorithm = algorithm_factory()

    # On resume both counters continue: train_step from the restored policy_version,
    # consumed_samples (the distributed budget) from the trainer state.
    consumed_samples = 0
    if resume_state is not None:
        apply_resume_state(algorithm, resume_state)
        consumed_samples = int(algorithm.state_dict().get("consumed_samples", 0))
    if version_floor > algorithm.policy_version:
        logger.info(f"Learner [{agent_id}]: policy version {algorithm.policy_version} raised to the hub's floor "
                    f"{version_floor}")
        set_policy_version(algorithm, version_floor)
    train_step = int(algorithm.policy_version)

    total_chunks_received = 0
    last_ckpt_version = -1

    # Off-policy: create replay buffer if algorithm requires it
    replay_buffer = algorithm.create_replay_buffer(config.queue_size * 4)

    try:
        _publish_weights(algorithm, agent_id, outputs)    # the first weights (first-weights wait, A-M3)

        while not stop_event.is_set():
            if (progress_counter is None and total_timesteps > 0
                    and consumed_samples >= total_timesteps):
                logger.info(f"Learner [{agent_id}]: consumed {consumed_samples} samples; budget reached")
                break

            # Block until exactly batch_chunks chunks arrived (None: stop requested).
            chunks = collect_batch(intake, config.batch_chunks, stop_event, agent_id=agent_id)
            if chunks is None:
                break
            total_chunks_received += len(chunks)
            consumed_samples += sum(c.num_acts for c in chunks)
            # Policy lag of the batch, measured before this train step bumps the version.
            lags = [algorithm.policy_version - c.policy_version for c in chunks]

            progress = _progress(progress_counter, consumed_samples, total_timesteps)
            algorithm.set_progress(progress)

            # Train step: off-policy adds to buffer, on-policy trains directly
            if replay_buffer is not None:
                for chunk in chunks:
                    replay_buffer.add(chunk)
                if len(replay_buffer) < config.batch_chunks:
                    continue
                metrics = algorithm.train_step(replay_buffer.sample(config.batch_chunks))
            else:
                metrics = algorithm.train_step(chunks)
            train_step += 1
            metrics["progress"] = float(progress)
            metrics["policy_lag_mean"] = float(np.mean(lags))
            metrics["policy_lag_max"] = float(np.max(lags))

            if train_step % config.weight_push_interval == 0:
                _publish_weights(algorithm, agent_id, outputs)

            if outputs.checkpoints and checkpoint_interval > 0:
                pv = algorithm.policy_version
                if pv > 0 and pv % checkpoint_interval == 0 and pv != last_ckpt_version:
                    if outputs.send_checkpoint(make_checkpoint_payload(agent_id, algorithm), kind="periodic",
                                               block=False):
                        last_ckpt_version = pv

            metrics["agent_id"] = agent_id
            metrics["train_step"] = train_step
            metrics["chunks_received"] = total_chunks_received
            metrics["consumed_samples"] = consumed_samples
            outputs.send_metrics(metrics)       # never blocks on a full sink

            if train_step % 10 == 0:
                logger.info(
                    f"Learner [{agent_id}]: step={train_step}, chunks={total_chunks_received}, "
                    f"progress={progress:.3f}, loss={metrics.get('total_loss', 0.0):.4f}"
                )
    finally:
        outputs.finish(stop_event)

    # Final snapshot on every stop (R3-07); the outputs deliver it (a node: header + frame on the control
    # connection; SP4: a blocking put, flushed).
    if outputs.checkpoints and outputs.send_checkpoint(make_checkpoint_payload(agent_id, algorithm, final=True),
                                                       kind="final", block=True):
        logger.info(f"Learner [{agent_id}]: sent final checkpoint v{algorithm.policy_version}")

    logger.info(f"Learner [{agent_id}]: finished. Total train_steps={train_step}")
```
and, replacing `collect_batch` and `_push_weights` at the end of the module:
```python
def collect_batch(
    q: Any,
    batch_size: int,
    stop_event: Any,
    poll_interval: float = 0.5,
    agent_id: str | None = None,
) -> list[TrajectoryChunk] | None:
    """Block until exactly ``batch_size`` chunk v2 payloads arrived; decode them.

    Waits in ``poll_interval`` slices and returns None as soon as ``stop_event``
    is set (a partial batch is dropped: no training on incomplete batches).
    Every decoded chunk's slot structure is checked (``validate_slot_structure``):
    a broken chunk raises ``ValueError`` naming its agent instead of silently
    corrupting the V-trace targets. With ``agent_id``, a chunk of another agent (a
    misrouted worker in distributed mode) raises ``ValueError``.
    """
    chunks: list[TrajectoryChunk] = []
    while len(chunks) < batch_size:
        if stop_event.is_set():
            return None
        try:
            payload = q.get(timeout=poll_interval)
        except Empty:
            continue
        if isinstance(payload, TrajectoryChunk):     # from an intake: decoded and checked already
            chunks.append(payload)
            continue
        if not isinstance(payload, dict):
            raise TypeError(
                f"trajectory queue item must be a chunk payload dict, got {type(payload).__name__}"
            )
        chunk = TrajectoryChunk.from_payload(payload)
        if agent_id is not None and chunk.agent_id != agent_id:
            raise ValueError(f"learner of agent {agent_id!r} received a chunk of agent {chunk.agent_id!r}; "
                             f"check the workers' learner addresses (run-workers -l AGENT=HOST:PORT)")
        validate_slot_structure(chunk)
        chunks.append(chunk)
    return chunks


def _publish_weights(algorithm: BaseAlgorithm, agent_id: str, outputs: LearnerOutputs) -> None:
    """Publish the current weights (numpy, with the scripted teacher's ``teacher_active``) through ``outputs``."""
    outputs.publish_weights(WeightPayload.from_model(
        agent_id, algorithm.policy_version, algorithm.model,
        teacher_active=bool(getattr(algorithm, "teacher_active", False))))
```

- [ ] **Step 5: Types and seeds**

`src/colosseum/core/types.py`, `WeightPayload`: add the field `generation: int = 0` after `teacher_active` (docstring: "``generation`` (SP5): the learner generation that published the weights (0: unknown / SP4 path)"), and `from_model(cls, agent_id, policy_version, model, teacher_active=False, generation=0)` passes it.

`src/colosseum/utils/seeding.py`, append (same hash scheme as `derive_seed`):
```python
def cell_seed(seed: int | None, epoch: int, node: int, worker: int, env: int) -> int | None:
    """Seed of the cell (node, worker, env) in ``epoch`` (SP5 spec 7.6): a 32-bit hash of all five numbers (None
    without a training seed); an episode's seed is ``SeedSequence([cell seed, episode index])`` (``MatchRunner``)."""
    if seed is None:
        return None
    digest = hashlib.sha256(f"colosseum:cell:{seed}:{epoch}:{node}:{worker}:{env}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def worker_seed(seed: int | None, node: int, worker: int, incarnation: int) -> int | None:
    """Process RNG seed of a node's worker incarnation (torch / numpy / random in ``RolloutLoop``); None without a
    training seed. Distinct per node, worker slot and incarnation (SP2's ``seed + worker_id * 1000`` collided across
    machines)."""
    if seed is None:
        return None
    digest = hashlib.sha256(f"colosseum:worker:{seed}:{node}:{worker}:{incarnation}".encode()).digest()
    return int.from_bytes(digest[:4], "little")
```

- [ ] **Step 6: The loops (`worker/match_runner.py`, `worker/rollout_loop.py`)**

`MatchRunner.__init__` gains the keyword `cell_seed_fn: Callable[[int, int], int | None] | None = None` (stored as `self._cell_seed_fn`). Two edits, both only for lineups whose `epoch` is not None (every other lineup behaves exactly as in SP4):
- `_reset_envs`, the seed of env `e`'s next episode:
```python
            lineup = self._envs[e].lineup
            k = self._envs[e].episode_index
            if lineup.seed is not None:
                seed = int(lineup.seed)
            elif lineup.epoch is not None and self._cell_seed_fn is not None:
                cell = self._cell_seed_fn(e, int(lineup.epoch))
                seed = None if cell is None else int(np.random.SeedSequence([int(cell), k]).generate_state(1)[0])
            else:
                seed = self._episode_seed(e, k)
            requests[e] = (seed, lineup.layout)
```
- `_match_result`: `match_id = f"{self._prefix}{e}_g{lineup.epoch}_ep{K}"` when `lineup.epoch is not None`, else SP4's `f"{self._prefix}{e}_ep{K}"` (`K = env_state.episode_index`; the lineup is the one that played the episode: the next lineup is applied after `on_episode_end`).
- Module docstring, "Match records": add one sentence: "A lineup with an ``epoch`` (SP5) makes the match id ``<prefix><env>_g<epoch>_ep<K>`` and, with ``cell_seed_fn``, the episode seed ``SeedSequence([cell_seed_fn(env, epoch), K])`` (None when it returns None)."

`RolloutLoop.__init__` gains `cell_seed_fn: Callable[[int, int], int | None] | None = None`, `uid_prefix: str | None = None` and `snapshot_fallback: bool = True`, and passes them to its `MatchRunner`: `cell_seed_fn=cell_seed_fn`, `uid_prefix=uid_prefix if uid_prefix is not None else (f"{run_id}/train/" if run_id else "")`, `snapshot_fallback=snapshot_fallback`. Docstring, "Match results": "the uid ``<run_id>/train/w<W>_e<E>_ep<K>`` (SP4), or ``<uid_prefix>w<W>_e<E>_g<epoch>_ep<K>`` for a node worker's epoch lineups (``uid_prefix = <run_id>/train/n<node>/``, SP5 spec 4.3)"; "Snapshot eviction": "a node worker passes ``snapshot_fallback=False``: its lineups name only cached snapshots (R11), and a missing one is an error, never latest + collect (spec 4.2)".

- [ ] **Step 7: The two `learner_process` call sites**

`src/colosseum/launcher.py`, `_learner_main` (the `Launcher` runs on it until T2.5):
```python
    from colosseum.net.local import QueueIntake, QueueOutputs

    learner_process(
        agent_id=agent_id,
        algorithm_factory=algorithm_factory,
        intake=QueueIntake(trajectory_queue, agent_id=agent_id),
        outputs=QueueOutputs(weight_queues=weight_queues, checkpoint_queue=checkpoint_queue,
                             metrics_queue=metrics_queue, trajectory_queue=trajectory_queue,
                             weight_sync_interval=config.rollout.weight_sync_interval_sec),
        config=config.learner,
        stop_event=stop_event,
        checkpoint_interval=checkpoint_interval,
        resume_state=resume_state,
        progress_counter=progress_counter,
        total_timesteps=total_timesteps,
        weight_sync_interval=config.rollout.weight_sync_interval_sec,
    )
```
`QueueIntake(..., agent_id=agent_id)` keeps SP4's misrouted-chunk `ValueError`. `src/colosseum/distributed.py`, `run_distributed_learner`: the same with `intake=QueueIntake(chunk_queue, agent_id=agent_id)`, `outputs=QueueOutputs(weight_queues=weight_sink, checkpoint_queue=checkpoint_queue, trajectory_queue=chunk_queue, weight_sync_interval=acfg.rollout.weight_sync_interval_sec)` (no metrics queue, `progress_counter=None` as today); import `QueueIntake, QueueOutputs` from `colosseum.net.local` next to the `learner_process` import (keep `resolve_device` imported from `colosseum.learner.learner`).

- [ ] **Step 8: Port the tests that call the old signature**

Append to `tests/game_helpers.py`:
```python
# ---------------------------------------------------------------------------
# SP5 (T2.2): learner_process on SP4's queues
# ---------------------------------------------------------------------------


def queue_learner_io(trajectory_queue, *, weight_queues=(), metrics_queue=None, checkpoint_queue=None,
                     agent_id: str | None = None, weight_sync_interval: float = 5.0) -> dict:
    """``intake`` / ``outputs`` keywords of ``learner_process`` over SP4-style queues (``QueueIntake`` /
    ``QueueOutputs``), for tests written against the SP4 signature."""
    from colosseum.net.local import QueueIntake, QueueOutputs

    return {"intake": QueueIntake(trajectory_queue, agent_id=agent_id),
            "outputs": QueueOutputs(weight_queues=list(weight_queues), checkpoint_queue=checkpoint_queue,
                                    metrics_queue=metrics_queue, trajectory_queue=trajectory_queue,
                                    weight_sync_interval=weight_sync_interval)}
```
Mechanical ports (assertions unchanged):
- `tests/unit/test_learner_v2.py`: import `queue_learner_io`; in `_run_learner` replace `trajectory_queue=traj` and the `weight_queues` default by `**queue_learner_io(traj, weight_queues=kwargs.pop("weight_queues", [NumpyOnlyQueue(maxsize=1)]), metrics_queue=kwargs.pop("metrics_queue", None), checkpoint_queue=kwargs.pop("checkpoint_queue", None))`; in the four direct `learner_process` calls replace `trajectory_queue=X, weight_queues=[Y]` (and `metrics_queue=` / `checkpoint_queue=` where given) by `**queue_learner_io(X, weight_queues=[Y], ...)`. The `collect_batch` tests stay as they are (dict items).
- `tests/integration/test_learner_v2_exit.py`: both `ctx.Process(target=learner_process, ...)` calls take `**queue_learner_io(traj, weight_queues=[weights])` (the dataclasses pickle with their `mp.Queue`s under spawn).
- `tests/unit/test_sp2_ipc_latest.py` and `tests/unit/test_sp3_dagger_loss.py`: `_push_weights(algo, "a", [q])` becomes `_publish_weights(algo, "a", QueueOutputs(weight_queues=[q]))` (import `_publish_weights` and `colosseum.net.local.QueueOutputs`).
- The monkeypatched fakes of `learner_process` in `test_sp2_learner_entry.py`, `test_sp3_init.py`, `test_sp3_learner_factory.py` and `test_sp2_distributed_roles.py` take `**kwargs` and need no change (verified).

- [ ] **Step 9: Run the new and touched tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_local_connections.py tests/unit/test_sp5_worker_io.py tests/unit/test_sp5_learner_io.py tests/contract/test_sp5_loop_epochs.py tests/unit/test_learner_v2.py tests/integration/test_learner_v2_exit.py tests/unit/test_sp2_ipc_latest.py tests/unit/test_sp3_dagger_loss.py tests/integration/test_sp2_pipelines.py tests/contract/test_sp4_result_flow.py -q`
Expected: all pass.

- [ ] **Step 10: Full fast suite + ruff; slow suite once**

Run the full fast suite and `ruff check .` (green, zero warnings). This task changes the learner and the worker loops: run the slow suite once, alone on a quiet machine, in the background (`env -u PYTHONPATH .venv/bin/python -m pytest -m slow -v`); every threshold must hold as before.

- [ ] **Step 11: Commit and push**

```bash
git add src/colosseum/net/local.py src/colosseum/learner/learner.py src/colosseum/worker/match_runner.py \
        src/colosseum/worker/rollout_loop.py src/colosseum/core/types.py src/colosseum/utils/seeding.py \
        src/colosseum/launcher.py src/colosseum/distributed.py tests/game_helpers.py tests/unit/test_learner_v2.py \
        tests/integration/test_learner_v2_exit.py tests/unit/test_sp2_ipc_latest.py tests/unit/test_sp3_dagger_loss.py \
        tests/unit/test_sp5_local_connections.py tests/unit/test_sp5_worker_io.py tests/unit/test_sp5_learner_io.py \
        tests/contract/test_sp5_loop_epochs.py
git commit -m "feat: same-machine process connections, worker and learner IO, epoch-aware match ids and cell seeds"
git push
```

---


### Task T2.3: Hub gRPC server, node session client, blobs in parts

Spec 4.1 (one session stream per node; `Hello` / `Welcome`; resume with resend; node loss after `node_lost_after_sec`; large data in parts by separate calls), 3.2 / 3.3 (TLS, `--insecure`, loopback: credentials from T1.3), 3.4 (hostile input: caps before allocation); R1 (application-level node authentication: T1.3's `AuthInterceptor`), R13 (hub logic tested with fake nodes against the real server). The servicers only parse, authenticate, convert and put inbox events; the main loop (here the test-support `MiniHubLoop`, from T2.5 `Hub`) owns all state. `RelaySend` / `RelayInbox` answer `UNIMPLEMENTED` until T3.2, `Bootstrap` is registered by T6.2.

**Files:**
- Create: `src/colosseum/hub/server.py`, `src/colosseum/node/session.py`, `tests/hub_kit.py`
- Modify: `src/colosseum/net/grpc_util.py` (blobs in parts), `src/colosseum/net/proto/colosseum_v1.proto` + regenerated `_pb2.py` / `_pb2_grpc.py` (`Heartbeat.counters`), `pyproject.toml` (`hub_kit` in `known-first-party`)
- Test: `tests/unit/test_sp5_blob_parts.py`, `tests/integration/test_sp5_hub_server.py`, `tests/integration/test_sp5_hub_session.py`

**Interfaces:**
- Consumes: `pb` (`colosseum.net.proto.colosseum_v1_pb2` / `_pb2_grpc`), `results_from_batch`, `PROTOCOL_VERSION` (T1.2); `HubIdentity` (`server_credentials`, `nodes`, `issue_node_certificate`, `token`), `NodeBook`, `AuthInterceptor`, `server_options`, `channel_options`, `hub_channel_credentials`, `MD_NODE_ID`, `MD_NODE_SECRET`, `HUB_SERVER_NAME` (T1.3); `colosseum.hub.state` (T2.1); `encode_weights`, `WeightsMeta` (T1.1); `read_weights_file` (`coordinator.checkpoint_manager`); `RunDir`.
- Produces (contract "colosseum.hub.server", "colosseum.node.session", `grpc_util` blob parts; additions marked *): `build_server(...)` (*`identity` required), *`LOOPBACK_KEY`, *`SERVER_THREADS`, *`OUTBOUND_QUEUE_SIZE`, *`UPLOAD_REPLY_TIMEOUT_SEC`; `SessionHandle`; `BlobStore` (*`blob_ident`, *`register_fixed(..., kind, agent_id)`); the inbox events `SessionOpened`, `SessionClosed`, `NodeMessageIn` (*`results`, *`error`), `RegisterIn`, `UploadIn`, `AdminIn`, *`StopRequested`; *`hello_from_proto`; `HubSession` (*`hello` callable called at each (re)connect, *`SessionLost` / *`SessionUp` inbox events, *`backoff_delays`); `grpc_util.PART_BYTES`, `send_parts`, `receive_parts` (returns `None` data for `not_modified`), *`BlobError`, *`blob_digest`, *`not_modified_part`; test support `hub_kit.FakeNode`, `hub_kit.MiniHubLoop`, `hub_kit.make_result`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp5_blob_parts.py` (verified in a scratch package against a protobuf `BlobPart` with the contract's fields: 12 passed):
```python
"""Blobs in parts (SP5 T2.3, spec 4.1 "Большие данные", 3.4): round trips of bytes and files, empty blobs, the
cap checked at the first part before anything is allocated, gaps / overlaps / short streams / digest mismatches
refused, a file target written atomically (nothing left on error), ``not_modified`` answers."""
from __future__ import annotations

import os

import pytest

from colosseum.net.grpc_util import PART_BYTES, BlobError, blob_digest, not_modified_part, receive_parts, send_parts


@pytest.mark.parametrize("size", [0, 1, PART_BYTES - 1, PART_BYTES, 3 * PART_BYTES + 5])
def test_bytes_round_trip_in_parts(size):
    data = os.urandom(size)
    parts = list(send_parts(data, meta_json='{"k": 1}'))
    assert len(parts) == max(1, -(-size // PART_BYTES)) and parts[-1].last and not any(p.last for p in parts[:-1])
    assert parts[0].digest == blob_digest(data) and all(not p.digest for p in parts[1:])
    assert receive_parts(parts, cap=size + 1) == (data, '{"k": 1}')


def test_a_file_round_trips_into_a_file(tmp_path):
    src = tmp_path / "src.frame"
    src.write_bytes(os.urandom(2 * PART_BYTES + 7))
    out, meta = receive_parts(send_parts(src, meta_json="m"), cap=10 * PART_BYTES, into=tmp_path / "dst.frame")
    assert out == tmp_path / "dst.frame" and out.read_bytes() == src.read_bytes() and meta == "m"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dst.frame", "src.frame"]


def test_the_cap_is_checked_at_the_first_part():
    parts = send_parts(os.urandom(3 * PART_BYTES))
    with pytest.raises(BlobError) as e:
        receive_parts(parts, cap=PART_BYTES)
    assert e.value.reason == "too_large"


@pytest.mark.parametrize("mutate,reason", [
    (lambda ps: [ps[0], ps[2]], "offset"),                        # a gap
    (lambda ps: [ps[0], ps[0]], "offset"),                        # a repeated part
    (lambda ps: ps[:2], "size"),                                  # cut short
    (lambda ps: [type(ps[0])(data=b"x" + ps[0].data[1:], offset=0, total_size=ps[0].total_size,
                              digest=ps[0].digest, last=False)] + ps[1:], "digest"),
])
def test_inconsistent_streams_are_refused_and_leave_no_file(tmp_path, mutate, reason):
    parts = list(send_parts(os.urandom(2 * PART_BYTES + 3)))
    with pytest.raises(BlobError) as e:
        receive_parts(mutate(parts), cap=10 * PART_BYTES, into=tmp_path / "x.frame")
    assert e.value.reason == reason and list(tmp_path.iterdir()) == []


def test_an_empty_stream_and_a_not_modified_answer():
    with pytest.raises(BlobError):
        receive_parts([], cap=10)
    assert receive_parts([not_modified_part('{"version": 4}')], cap=10) == (None, '{"version": 4}')
```

Create `tests/integration/test_sp5_hub_server.py` (real server on loopback port 0 with a real `HubIdentity` under `tmp_path`, `MiniHubLoop` as the main loop, `FakeNode` clients in the test process; `insecure=False` unless the test says otherwise). Each test, its setup and what it asserts:
- `test_registration_issues_an_identity_and_refuses_a_wrong_or_missing_token`: `FakeNode.register(token)` → `RegisterReply` with a 32-hex `node_id`, `number >= 1`, a PEM certificate whose CN is the node id (parsed with `cryptography`), a non-empty `node_secret`, the CA PEM; the hub's `NodeBook.verify(node_id, secret)` returns the record; a wrong token → `grpc.StatusCode.UNAUTHENTICATED`, no token → `UNAUTHENTICATED`; a banned machine id (`NodeBook.ban`) → `PERMISSION_DENIED`.
- `test_a_session_starts_with_hello_and_gets_a_welcome`: register, open a session, send `Hello(node_id, session_id="s1", protocol_major/minor = PROTOCOL_VERSION, live_cells=[], env_steps=0)`; `MiniHubLoop` sees `SessionOpened` (its `hello` is the parsed `pb.Hello`, `hello_from_proto` gives a `NodeHello`), answers `Welcome(node_number, run_id="R", config_json="{}")`; the node reads it as the first `HubMessage`.
- `test_a_session_without_hello_or_with_another_nodes_id_is_refused`: first message a `Heartbeat` → the stream ends with `INVALID_ARGUMENT`; a `Hello` whose `node_id` differs from the `x-colosseum-node` metadata → `PERMISSION_DENIED`; a `Hello` with another protocol major → `FAILED_PRECONDITION` with "protocol" in the details; nothing reaches the inbox in all three cases.
- `test_node_messages_reach_the_inbox_in_order_with_converted_results`: after `Welcome`, the node sends `Heartbeat(env_steps=10)`, `LineupRequest`, `ResultBatch(seq=1, results=[result_to_proto(make_result(...)) × 3])`, `MetricsBatch`, `VersionAnnounce`, `Event`; `MiniHubLoop.received` lists six `NodeMessageIn` with the oneof names in send order; the result batch's event carries `results` = three `MatchResult`s equal to the originals (dataclass equality) and `error is None`.
- `test_a_malformed_result_batch_reaches_the_inbox_as_an_error`: a `ResultBatch` whose result has a seat with an unknown oneof-free garbage field set so that `results_from_batch` raises (e.g. `outcome_kind="nonsense"` if T1.2's converter checks it, else monkeypatch `hub.server.results_from_batch` to raise ValueError) → `NodeMessageIn.results is None` and `error` names the exception; the session stays open (a later heartbeat arrives).
- `test_a_cut_stream_resumes_and_resent_batches_are_accounted_once`: `MiniHubLoop(hold_acks=True)`; the node sends batches 1–3 (2 results each), the test closes the node's channel (cut), `MiniHubLoop` sees `SessionClosed`; within `node_lost_after_sec` (2.0 here) a new channel opens a session with the same `session_id` and `last_acked_seq=0` → `NodeRegistry.connect` returns `"resumed"` and the node's cells (requested before the cut) keep their epochs; the node resends 1–3 and then sends 4; `MiniHubLoop.release_acks()`; the loop accounted exactly 8 results (each uid once) and acked seqs `[1, 2, 3, 4]`.
- `test_a_silent_node_is_lost_and_comes_back_with_new_epochs`: `MiniHubLoop(node_lost_after_sec=0.5)`; the node requests two cells (epochs e1, e2), then sends nothing; after the loop's loss check the node is `lost` and `CellTable.live()` has none of its cells; the node reconnects with a new `session_id` and the same `live_cells` → `"returned"`; its cell request gets epochs greater than e1, e2.
- `test_an_outbound_overflow_closes_the_session` (monkeypatch `hub.server.OUTBOUND_QUEUE_SIZE = 4`, a node that never reads its responses): `MiniHubLoop.flood(handle, 50)` → the handle closes with reason `"outbound overflow"`, `SessionClosed` arrives, the server keeps serving a new session.
- `test_snapshots_are_converted_once_and_served_in_parts`: a run dir with `checkpoints/a/ckpt_v5/model.pt` of a 3 MiB state dict (written with `torch.save` of tensors); two concurrent `Download(kind="snapshot", agent_id="a", ident="ckpt_v5")` calls → identical byte streams, `receive_parts` checks digest and size, `decode_weights` gives the state dict (equal arrays) with `WeightsMeta(kind="snapshot", policy_version=5)`; `runs/<name>/blobs/snapshot-a-ckpt_v5.frame` exists once (the conversion ran once: count via a monkeypatched `read_weights_file` wrapper); a missing snapshot → `NOT_FOUND`.
- `test_registered_fixed_blobs_are_served_by_ident`: `BlobStore.register_fixed(blob_ident("fixed", "bot", digest), frame, kind="fixed", agent_id="bot")` from the test (main-loop role); `Download(kind="fixed", agent_id="bot", ident=<ident>)` returns the frame byte for byte; an unknown ident → `NOT_FOUND`.
- `test_uploads_are_spooled_checked_and_handed_to_the_main_loop`: `Upload` of a 2.5 MiB frame (`UploadHeader(kind="checkpoint", agent_id="a", generation=1, version=20, total_size, digest, request_id="r1")` then data parts) → `MiniHubLoop` receives `UploadIn` whose `path` is under `runs/<name>/.incoming/` with the exact bytes, answers `accepted=True, checkpoint_id="ckpt_v20"` through the future; the client gets that reply. A wrong digest, a total above `payload_byte_cap(max_message_mb)`, or a stream without a header → `UploadReply(accepted=False, reason=...)` (reason `digest` / `too_large` / `header`), nothing reaches the inbox, `.incoming/` is empty afterwards.
- `test_relay_methods_are_not_implemented_yet`: `RelaySend` / `RelayInbox` → `UNIMPLEMENTED` (T3.2 replaces this test).

Create `tests/integration/test_sp5_hub_session.py` (the node-side client `HubSession` against the real server + `MiniHubLoop`):
- `test_the_session_connects_and_delivers_hub_messages_to_the_node_inbox`: `HubSession(address, credentials=hub_channel_credentials(ca_pem), node_id, node_secret, hello=..., inbox=list.append, max_message_mb=64).start()`; the node inbox gets `SessionUp` then the `Welcome` `HubMessage`; `connected` is True; `send(Heartbeat)` returns True and the loop receives it.
- `test_the_session_reconnects_with_backoff_and_the_same_session_id` (monkeypatch `RECONNECT_MIN_SEC = 0.05`, `RECONNECT_MAX_SEC = 0.2`): `MiniHubLoop.close_session(node_id, "test")` → the node inbox gets `SessionLost`; while disconnected `send` returns False; the session reconnects by itself, `hello()` was called again (counter 2), the hub sees the same `session_id`, `SessionUp` arrives again.
- `test_backoff_doubles_with_bounded_jitter`: `backoff_delays(random.Random(0))` → the first 8 delays: each within ±20 % of `min(0.5 × 2**k, 10.0)`, never above `10.0 × 1.2`.
- `test_close_stops_reconnecting`: `close()` → no reconnect within 1 s (the loop sees no new `SessionOpened`), the reader thread has ended.
- `test_a_tls_session_refuses_a_hub_with_another_ca`: the node pins another CA's PEM → the channel never becomes connected (`SessionLost` reasons name `UNAVAILABLE`), the hub's inbox stays empty.
- `test_an_insecure_hub_still_requires_the_node_secret`: `build_server(insecure=True, ...)`; a session with a wrong `node_secret` → `UNAUTHENTICATED`; with the right one it works.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_blob_parts.py tests/integration/test_sp5_hub_server.py tests/integration/test_sp5_hub_session.py -q`
Expected: import errors (`cannot import name 'send_parts'`, `No module named 'colosseum.hub.server'`).

- [ ] **Step 3: The proto field and blobs in parts**

`colosseum_v1.proto`: add `map<string, int64> counters = <next number>;` to `Heartbeat` (the node's cumulative counters: `lineup_fetch_failed`, `dropped_results_node_buffer`, ...); regenerate with `scripts/gen_proto.sh`; T1.2's round-trip test of `Heartbeat` (if any) gains the field.

Append to `src/colosseum/net/grpc_util.py` (imports: `hashlib`, `io`, `os`, `threading`, `Iterable`, `Iterator`, `Path`, `pb` — complete; verified as described):
```python
# ---- blobs in parts (T2.3; spec 4.1 "Большие данные") -----------------------------------------------------------

PART_BYTES = 1 << 20                     # blob part size
BLOB_DIGEST_SIZE = 16                    # blake2b digest bytes (the shm cache's digest, T1.5)


class BlobError(ValueError):
    """A blob stream that is not a whole, consistent blob (``reason``: ``too_large``, ``offset``, ``size``,
    ``digest``, ``empty``)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def blob_digest(data: bytes | memoryview) -> str:
    return hashlib.blake2b(data, digest_size=BLOB_DIGEST_SIZE).hexdigest()


def send_parts(data: bytes | Path, *, meta_json: str = "", digest: str | None = None) -> Iterator[pb.BlobPart]:
    """``data`` (bytes, or a file read part by part) as ``BlobPart``s of at most ``PART_BYTES``: the first part
    carries ``total_size``, ``digest`` (blake2b-128 of the whole blob; computed unless given) and ``meta_json``;
    the last has ``last``; an empty blob is one part."""
    if isinstance(data, Path):
        total = data.stat().st_size
        if digest is None:
            h = hashlib.blake2b(digest_size=BLOB_DIGEST_SIZE)
            with open(data, "rb") as f:
                for block in iter(lambda: f.read(PART_BYTES), b""):
                    h.update(block)
            digest = h.hexdigest()
        with open(data, "rb") as f:
            offset = 0
            while True:
                block = f.read(PART_BYTES)
                last = offset + len(block) >= total
                yield pb.BlobPart(data=block, offset=offset, total_size=total, last=last,
                                  digest=digest if offset == 0 else "", meta_json=meta_json if offset == 0 else "")
                offset += len(block)
                if last:
                    return
    view = memoryview(data)
    total = len(view)
    digest = blob_digest(view) if digest is None else digest
    offset = 0
    while True:
        block = bytes(view[offset:offset + PART_BYTES])
        last = offset + len(block) >= total
        yield pb.BlobPart(data=block, offset=offset, total_size=total, last=last,
                          digest=digest if offset == 0 else "", meta_json=meta_json if offset == 0 else "")
        offset += len(block)
        if last:
            return


def not_modified_part(meta_json: str = "") -> pb.BlobPart:
    """The single part of a ``GetLatest`` answer whose version is not newer (T3.1)."""
    return pb.BlobPart(not_modified=True, last=True, meta_json=meta_json)


def receive_parts(parts: Iterable[pb.BlobPart], *, cap: int, into: Path | None = None
                  ) -> tuple[bytes | Path | None, str]:
    """The blob of a part stream and its first part's ``meta_json``; ``None`` for a ``not_modified`` answer.
    Hostile input: ``total_size`` above ``cap`` is refused at the first part (nothing allocated), parts must be
    contiguous and end exactly at ``total_size``, the digest must match; ``BlobError`` otherwise. ``into``: write
    to that file (atomically: a temp file renamed at the end; removed on error) and return its path."""
    it = iter(parts)
    first = next(it, None)
    if first is None:
        raise BlobError("empty", "blob stream without parts")
    if first.not_modified:
        return None, first.meta_json
    total, digest, meta = int(first.total_size), first.digest, first.meta_json
    if total < 0 or total > cap:
        raise BlobError("too_large", f"blob of {total} bytes exceeds the cap of {cap} bytes")
    h = hashlib.blake2b(digest_size=BLOB_DIGEST_SIZE)
    tmp = None if into is None else into.with_name(f".{into.name}.part-{os.getpid()}-{threading.get_ident()}")
    sink = io.BytesIO() if into is None else open(tmp, "wb")
    try:
        offset, part, done = 0, first, False
        while True:
            if part.offset != offset or offset + len(part.data) > total:
                raise BlobError("offset", f"part at offset {part.offset} (expected {offset}) of a "
                                          f"{total}-byte blob")
            h.update(part.data)
            sink.write(part.data)
            offset += len(part.data)
            if part.last:
                done = True
                break
            part = next(it, None)
            if part is None:
                break
        if not done or offset != total:
            raise BlobError("size", f"blob ended at {offset} of {total} bytes")
        if h.hexdigest() != digest:
            raise BlobError("digest", "blob digest mismatch")
        if into is None:
            return sink.getvalue(), meta
        sink.close()
        os.replace(tmp, into)
        return into, meta
    except BaseException:
        if into is not None:
            sink.close()
            tmp.unlink(missing_ok=True)
        raise
```

- [ ] **Step 4: `src/colosseum/hub/server.py`**

Module docstring: the servicers of the hub port; they only parse, authenticate (the `AuthInterceptor` does it before a method runs), convert, read files and put events on the main loop's inbox; replies the main loop gives come through futures or a session's outbound queue; a servicer never touches hub state.

Constants: `SERVER_THREADS = 128` (every live session and relay stream holds one thread: 20 nodes × 2 plus transfers), `OUTBOUND_QUEUE_SIZE = 10_000`, `UPLOAD_REPLY_TIMEOUT_SEC = 60.0`, `REGISTER_REPLY_TIMEOUT_SEC = 30.0`, `LOOPBACK_KEY = "loopback"`.

Events (dataclasses, contract): `SessionOpened(handle, hello)`, `SessionClosed(node_id, session_id, reason)`, `NodeMessageIn(node_id, session_id, message, results=None, error=None)` (`results`: the converted `list[MatchResult]` of a `ResultBatch`; `error`: text when the conversion failed), `RegisterIn(request, peer, reply: Future)`, `UploadIn(header, path, node_id, reply: Future)`, `AdminIn(command, args, reply)`, `StopRequested()` (put by `Hub.request_stop`, T2.5).

`hello_from_proto(hello: pb.Hello) -> NodeHello` (`google.protobuf.json_format.MessageToDict` for `hardware` / `slots`, tuples for `tags` / `data_addresses`, `live_cells` as `(worker, env, epoch, incarnation)`).

`SessionHandle` and the `Session` method follow the pattern verified in the scratch prototype:
```python
_CLOSE = object()


class SessionHandle:
    """One live session stream (contract). ``send`` never blocks: a full outbound queue closes the session (the
    node reconnects and gets a full ``Welcome``)."""

    def __init__(self, node_id: str, session_id: str) -> None:
        self.node_id, self.session_id = node_id, session_id
        self._out: queue.Queue = queue.Queue(maxsize=OUTBOUND_QUEUE_SIZE)
        self.closed_reason: str | None = None

    def send(self, msg: pb.HubMessage) -> None:
        if self.closed_reason is not None:
            return
        try:
            self._out.put_nowait(msg)
        except queue.Full:
            self.close("outbound overflow")

    def close(self, reason: str) -> None:
        if self.closed_reason is None:
            self.closed_reason = reason
            try:
                self._out.put_nowait(_CLOSE)
            except queue.Full:                       # the generator sees closed_reason at its next poll
                pass

    def _next(self, timeout: float):
        if self.closed_reason is not None and self._out.empty():
            return _CLOSE
        try:
            return self._out.get(timeout=timeout)
        except queue.Empty:
            return None


class HubServicer(pb_grpc.HubServicer):
    def Session(self, request_iterator, context):
        node_id = metadata(context)[MD_NODE_ID]          # verified by the interceptor
        try:
            first = next(request_iterator)
        except (StopIteration, grpc.RpcError):
            return
        if first.WhichOneof("m") != "hello":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "the first session message must be Hello")
        hello = first.hello
        if hello.node_id != node_id:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "Hello names another node id than the credentials")
        if hello.protocol_major != PROTOCOL_VERSION[0]:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          f"protocol {hello.protocol_major}.{hello.protocol_minor} != hub {PROTOCOL_VERSION}")
        handle = SessionHandle(node_id, hello.session_id)
        self._inbox.put(SessionOpened(handle, hello))

        def reader() -> None:
            reason = "end of stream"
            try:
                for msg in request_iterator:
                    self._inbox.put(self._event(handle, msg))
            except grpc.RpcError:                         # a cut client: a bare RpcError without code() (verified)
                reason = "stream cut"
            except Exception as e:  # noqa: BLE001 - one bad stream never stops the hub
                reason = f"{type(e).__name__}: {e}"
            handle.close(reason)
            self._inbox.put(SessionClosed(handle.node_id, handle.session_id, reason))

        threading.Thread(target=reader, name=f"session-{node_id[:8]}", daemon=True).start()
        while context.is_active():
            msg = handle._next(0.5)
            if msg is _CLOSE:
                return
            if msg is not None:
                yield msg

    def _event(self, handle, msg) -> NodeMessageIn:
        if msg.WhichOneof("m") != "results":
            return NodeMessageIn(handle.node_id, handle.session_id, msg)
        try:
            return NodeMessageIn(handle.node_id, handle.session_id, msg, results=results_from_batch(msg.results))
        except Exception as e:  # noqa: BLE001 - untrusted input
            return NodeMessageIn(handle.node_id, handle.session_id, msg, error=f"{type(e).__name__}: {e}")
```
(`metadata(context)` = `dict(context.invocation_metadata())`; a second `Hello` inside the stream is an event like any other and the main loop ignores it with a WARNING.)

The other methods:
- `Register(request, context)`: put `RegisterIn(request, context.peer(), Future())`, wait `REGISTER_REPLY_TIMEOUT_SEC`; a `SecurityError` result → `context.abort(PERMISSION_DENIED, str(e))`; timeout → `UNAVAILABLE`.
- `Upload(request_iterator, context)`: the first part must carry `header` (else reply `accepted=False, reason="header"`); `total_size <= payload_byte_cap(max_message_mb)` (else `"too_large"`, nothing read further); data parts are written to `runs/<name>/.incoming/<request_id or uuid>.part` (directory created `0700`) with a running blake2b-128; size and digest checked at the end (`"size"` / `"digest"`, the file removed); then `UploadIn(header, path, node_id, Future())` and the future's `pb.UploadReply` is returned (timeout `UPLOAD_REPLY_TIMEOUT_SEC` → `accepted=False, reason="timeout"`, file removed). The main loop deletes the file after handling it.
- `Download(request, context)`: `path = blobs.frame_path(request.kind, request.agent_id, request.ident)`; `None` → `NOT_FOUND`; otherwise `yield from send_parts(path)`. Kinds served now: `snapshot`, `fixed`, `teacher`, `init`, `recovery`, `trainer_state` (raw bytes file, not a frame); `latest` → `NOT_FOUND` until T3.1.
- `RelaySend`, `RelayInbox`: `context.abort(UNIMPLEMENTED, "relay: T3.2")`.
- `Admin(request, context)`: `AdminIn(request.command, json.loads(request.args_json or "{}"), Future())`, wait 30 s, return the `pb.AdminReply`; `AdminStream`: `AdminIn(..., reply=queue.SimpleQueue())`, yield replies until one has `done`.

`BlobStore(run_dir)`:
- `frame_path(kind, agent_id, ident) -> Path | None`: `kind == "snapshot"`: `ident` must match `ckpt_v\d+` and `agent_id` be a safe path component (else None); the cached frame `blobs/snapshot-<agent>-<ident>.frame` if present, else convert `checkpoints/<agent>/<ident>/model.pt` (`read_weights_file`, a `ValueError` → None) into `encode_weights(state, WeightsMeta(agent_id, "snapshot", generation=0, policy_version=N, checkpoint_id=ident))`, written as a temp file + `os.replace`, under a per-key `threading.Lock` (two concurrent downloads convert once). Other kinds: `blobs/<kind>-<agent>-<ident>.frame` (`trainer_state`: `.bin`) when registered, else None.
- `register_fixed(blob_id, frame, *, kind, agent_id)` (main loop, at start): writes the file atomically. `blob_ident(kind, agent_id, data) -> str` = `blake2b(data, 16).hexdigest()[:16]`: the `ident` of a registered blob (unique per content; the file name carries kind and agent).
- Never deletes snapshot frames itself; T3.1 removes the frames of evicted snapshots after the hold.

`build_server(*, inbox, identity, insecure, listen, loopback_plain, blobs, max_message_mb, bootstrap=None) -> (server, ports)`: `grpc.server(ThreadPoolExecutor(SERVER_THREADS), interceptors=[AuthInterceptor(lambda: identity.token.secret, identity.nodes)], options=server_options(max_message_mb))`; registers `HubServicer` (and `bootstrap` when given, T6.2); for each address in `listen`: `add_insecure_port` if `insecure` else `add_secure_port(addr, identity.server_credentials())`; with `loopback_plain`: `add_insecure_port("127.0.0.1:0")` under `LOOPBACK_KEY`; a port 0 result of `add_*_port` (bind failure) → `OSError` naming the address. Returns the unstarted server and `{address or LOOPBACK_KEY: bound port}`. `identity` is required (the token is required in every mode).

- [ ] **Step 5: `src/colosseum/node/session.py`**

```python
RECONNECT_MIN_SEC, RECONNECT_MAX_SEC = 0.5, 10.0
HUB_LOST_RETRY_SEC = (30.0, 60.0)          # used by T4.4
_JITTER = 0.2


@dataclass
class SessionUp:
    """The stream is open and the Hello was sent (the next HubMessage is the Welcome)."""


@dataclass
class SessionLost:
    reason: str


def backoff_delays(rng: random.Random) -> Iterator[float]:
    """0.5, 1, 2, ... 10 s (spec 4.1), each with up to 20 % jitter either way."""
    delay = RECONNECT_MIN_SEC
    while True:
        yield delay * (1.0 + rng.uniform(-_JITTER, _JITTER))
        delay = min(RECONNECT_MAX_SEC, delay * 2.0)


class HubSession:
    """Session client (contract). One thread runs connect -> stream -> on loss wait the next backoff delay ->
    reconnect, until ``close``; the backoff restarts after a stream that stayed up for ``RECONNECT_MAX_SEC``.
    ``hello()`` is called at every (re)connect (it carries the node's live cells and last acked seq); every
    ``HubMessage`` goes to ``inbox`` as it is, plus ``SessionUp`` / ``SessionLost`` around each stream. ``send``
    puts a message on the current stream's outbound queue (False while disconnected: the caller keeps it,
    e.g. the result buffer resends after the next ``SessionUp``)."""

    def __init__(self, address, *, credentials, node_id, node_secret, hello, inbox, max_message_mb,
                 rng: random.Random | None = None) -> None: ...
    def start(self) -> None: ...
    def send(self, msg: pb.NodeMessage) -> bool: ...
    def close(self) -> None: ...        # ends the stream (outbound sentinel), stops the thread (join 5 s), closes the channel
    @property
    def connected(self) -> bool: ...    # a stream is up and its first response arrived
```
Implementation points: the request iterator is a generator over a `queue.Queue` that yields `NodeMessage(hello=hello())` first and returns on a sentinel; responses are read in the session thread (`for msg in call: inbox(msg)`); `grpc.RpcError` / end of stream → `inbox(SessionLost(reason))` with the status code when the client-side error has one; metadata `(MD_NODE_ID, node_id), (MD_NODE_SECRET, node_secret)` on the call; the channel is `grpc.secure_channel(address, credentials, options=channel_options(max_message_mb, server_name=HUB_SERVER_NAME))`, or `grpc.insecure_channel(address, options=channel_options(max_message_mb))` with `credentials=None` (insecure and loopback); a new channel per connect attempt (a broken channel is not reused).

- [ ] **Step 6: `tests/hub_kit.py`**

- `make_result(node=1, worker=0, env=0, epoch=1, k=0, *, run_id="R", agents=("a", "random"))` → a 1v1 `MatchResult` of the toy game `turns` (layout `2p`, role `player`) with uid `R/train/n<node>/w<W>_e<E>_g<epoch>_ep<K>`, `node` set, latest seat generation 1.
- `FakeNode(address, *, token, ca_pem=None, insecure=False)`: in-process client of the real server: `register(machine_id="m", name="fake") -> pb.RegisterReply`, `open(session_id, *, live_cells=(), last_acked=0)` (a session whose requests come from a queue; responses collected by a thread into `self.received`), `send(**oneof)`, `cut()` (closes the channel abruptly), `close()`, `wait_for(kind, n=1, timeout=10)`; `download(kind, agent, ident)` / `upload(header, data)` helpers over `send_parts` / `receive_parts`.
- `MiniHubLoop(server_inbox, *, identity, node_lost_after_sec=10.0, hold_acks=False, run_id="R")`: a thread over the inbox with `hub.state` objects: registers nodes (`NodeBook.register` + certificate), keeps `NodeRegistry`, `CellTable(EpochAllocator(0, lambda u: None))`, `ResultIntake(journal=False)`; answers `Hello` with `Welcome`, `LineupRequest` with grants of one fixed lineup per cell (owner `"a"`, epoch from the table, `ttl_sec` given at construction), `ResultBatch` by `begin` / accounting into `self.accounted` (uids) / ack (held while `hold_acks` until `release_acks()`), `Upload` by a configurable reply, heartbeats by `NodeRegistry.heartbeat`; checks loss every 0.1 s (`registry.lost` → `cells.drop_node`); records every event in `self.received`; `close_session(node_id, reason)`, `flood(handle, n)`, `stop()`. It is test support only: T2.5's `Hub` implements the same handlers on the real coordinator.
- Add `hub_kit` to `known-first-party` in `pyproject.toml`.

- [ ] **Step 7: Run the new tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_blob_parts.py tests/integration/test_sp5_hub_server.py tests/integration/test_sp5_hub_session.py -q`
Expected: all pass (no loaded CPU processes: every test runs in the test process).

- [ ] **Step 8: Full fast suite + ruff**

- [ ] **Step 9: Commit and push**

```bash
git add src/colosseum/hub/server.py src/colosseum/node/session.py src/colosseum/net/grpc_util.py \
        src/colosseum/net/proto/ tests/hub_kit.py tests/unit/test_sp5_blob_parts.py \
        tests/integration/test_sp5_hub_server.py tests/integration/test_sp5_hub_session.py pyproject.toml
git commit -m "feat: hub gRPC server (sessions, register, blobs in parts, uploads) and the node session client"
git push
```

---


### Task T2.4: `colosseum.node` — the node agent with a fake hub: children, results buffer, lineup pull, cache fetch, heartbeats

Spec 1 (one node per machine, its children die with it), 2.6 (hardware report and slots), 4.1 (the node side of the session), 4.2 (TTL pull per cell, lineup after cache, failed fetch keeps the old lineup), 4.3 (node batches, bounded buffer, resend after reconnect), 4.4 (cumulative env steps per session, LR progress forwarded to learners), 4.6 (fixed / teacher / init / snapshot blobs fetched by id into the shared-memory cache), 4.7 (connections per child, no shared locks; `kill -9` of any child blocks nobody), 7.1 (worker restarts with a new incarnation; contract and player errors stop the run), 7.7 (logs; the tails are T4.4); rulings R3, R10, R11, R12 (the same code runs `train`'s local node), R13 (node logic tested with a fake hub). Learner restarts, recovery points and relocation are T4.2; routing to other machines is T3.2; evaluators are T5.1.

**Split point:** dispatch (a) Steps 1–7 (pure helpers, children, hardware, logs, the worker body, the agent with workers, results and lineups; the tests of Step 1 except those marked (b)), (b) Steps 8–11 (learners on the node, the stop protocol, the kill tests (b)). Each dispatch ends with the full fast suite, a commit and a push.

**Files:**
- Create: `src/colosseum/node/agent.py`, `src/colosseum/node/procs.py`, `src/colosseum/node/worker_main.py`, `src/colosseum/node/learner_main.py`, `src/colosseum/node/hardware.py`, `src/colosseum/node/logs.py`, `tests/node_kit.py`
- Modify: `pyproject.toml` (`node_kit` in `known-first-party`); `src/colosseum/learner/factory.py` (`teacher_from_frozen`, `init_from_blob`: build a `TeacherSpec` / `InitState` from a blob's weights and the config, no file access)
- Test: `tests/unit/test_sp5_node_books.py`, `tests/unit/test_sp5_node_procs.py`, `tests/unit/test_sp5_node_hardware.py`, `tests/integration/test_sp5_node_agent.py`

**Interfaces:**
- Consumes: `HubSession`, `SessionUp`, `SessionLost` (T2.3); `send_parts`, `receive_parts` (T2.3); `WorkerIO`, `ConnectionIntake`, `NodeOutputs`, `ControlChannel`, the control messages, `listener`, `connect`, `address` (T2.2); `learner_process`, `cell_seed`, `worker_seed`, `RolloutLoop(cell_seed_fn, uid_prefix, snapshot_fallback)` (T2.2); `WeightCacheWriter` / `WeightCacheReader` / `CacheKey` / `cache_prefix` / `shm_dir` / `acquire_node_lock` / `cleanup_dead` / `CacheFullError` (T1.5); `decode_weights`, `WeightsMeta` (T1.1); `NodeIdentity`, `Token`, `hub_channel_credentials`, `fetch_hub_ca` (T1.3); `result_to_proto`, `lineup_from_proto`, `version_from_proto`, `PROTOCOL_VERSION` (T1.2); `DistributedConfig` (T1.4); `env_spec`, `resolve_agent_roles`, `agent_role_spec`, `build_model`, `build_algorithm`, `BotSpec`, `FrozenSpec`, `FixedPlayers`, `run_child`, `start_process`, `learner_seed`, `resolve_device`, `configure_torch_threads`, `resolve_learner_threads`.
- Produces (contract "colosseum.node", additions marked *): `NodeOptions` (*`credentials: tuple[str, int, str] | None`, *`shutdown_grace_sec: float | None`), `NodeAgent` (`run`, `request_stop`), `local_node_target`, `main`; *`ResultBuffer`, *`CellTracker`, *`LINEUP_RETRY_SEC`, *`METRICS_FLUSH_SEC`, *`LOCAL_CHILD_GRACE_SEC` (all in `node.agent`); `ChildSpec`, `ChildSupervisor` (*`get`, *`incarnation`, *`names`, *`sentinels`, *`restart`, *`forget`), `WORKER_RESTARTS`, `LEARNER_RESTARTS`; `Hardware`, `probe`, `default_slots`, `machine_id` (*`affinity_cores`, *`cgroup_quota_cores`, *`physical_cores_raw`, *`nvidia_gpus`, *`torch_versions`); `node_log_dir`, `log_tail`, `LOG_TAIL_BYTES`; `worker_target`, *`create_env` (moved from `launcher._create_env` in T2.5; here a copy that T2.5 makes the only one); `learner_target`; *`factory.teacher_from_frozen`, *`factory.init_from_blob`; test support `node_kit.FakeHub`, `node_kit.toy_node_options`.

**What the node does** (the behaviour every step implements; `NodeAgent.run` is a loop on the main thread):
1. **Identity.** `options.local`: `credentials` = `(node_id, number, node_secret)` given by the hub (no `Register`, R12). Otherwise `NodeIdentity.load_or_create(token.ca_fingerprint, options.identity_root)`; without a registration: fetch the CA (`fetch_hub_ca`, unless `insecure` / `loopback_plain`), `Register` with the token secret in metadata, `save_registration(reply)`. A refusal (`UNAUTHENTICATED` / `PERMISSION_DENIED`) is exit 1 with the hub's hint (T6.3 adds the version and platform checks).
2. **Session.** `HubSession` with `hello()` = `Hello(node_id, session_id (uuid4 hex per node process), PROTOCOL_VERSION, code_hash="" (T6.3), hardware (probe), slots (options or default_slots; local: workers = options.workers, evaluators = options.evaluators, cpu_learners = -1 meaning unlimited, no GPUs limit), data_addresses=[] (T3.2), run_id (once known), live_cells = CellTracker.live(), env_steps = the session's cumulative counter, last_acked_seq = ResultBuffer.last_acked, learners = running (agent, generation, version), name, tags)`.
3. **Welcome.** Parse `config_json` (`ColosseumConfig.model_validate_json`), `run_id`, the node number; compute `spec = env_spec(config)`, the trainable agents' roles (`resolve_agent_roles`), role specs, agent configs; the cache: `prefix = cache_prefix(token.ca_fingerprint, run_id, number)`, `cleanup_dead(shm_dir())`, `acquire_node_lock`, a `WeightCacheWriter` and a `WeightCacheReader`; fetch every `Welcome.fixed` blob (`Download` in the transfer pool) into the cache (`CacheKey("blob", agent_id, f"{kind}-{blob_id}")`); then start what is missing: `options.workers` workers (`worker-<i>`), the learners of `Welcome.learners` (b). A second `Welcome` after a reconnect reconciles (no new children for a resumed session; learners the hub no longer lists are stopped (b)).
4. **Children** run under `ChildSupervisor`; the node's listener (`address(ca8, run8, number, "node")`, authkey `os.urandom(32)`) accepts their connections in an accept thread that posts `(ChildHello, connection)` into the node inbox (wake pipe). The main loop waits on `multiprocessing.connection.wait(children's connections + process sentinels + wake, timeout=0.05)` and dispatches each message (below); `exited()` children: a worker with a `ChildError` already reported → nothing more (the hub stops the run); a worker without a report (signal, OOM) → `Event(kind="proc_died", proc, exit_code, log_path)` and, if `may_restart`, a new incarnation (its cells are dropped and re-added with the new incarnation: new epochs), else `Event(kind="slot_disabled")`; a learner (b, until T4.2) → `Event(kind="learner_failed")` (the hub stops the run with exit 1, as SP4 does for a dead child).
5. **Messages from children.** `WorkerResults` → `ResultBuffer.add` (send the cut batches); `WorkerProgress` → per (worker, incarnation) the last cumulative `env_steps` (the session's cumulative = their sum, it never decreases) and the counters (sent in `MetricsBatch.workers`); `ChildError` → `Event(kind="child_error", error_kind, message, context = "node <number> (<name>), " + the child's context, log_path)`; (b) `VersionPublished` → `VersionAnnounce` to the hub and the agent's latest key is now in the cache (pending grants may become deliverable); `CheckpointHeader` + frame → an `Upload` in the transfer pool (kind `checkpoint` / `final`), the reply logged (a refused periodic upload is a WARNING, a refused final an ERROR); `LearnerMetrics` → `MetricsBatch.train`; `IntakeReport` → `MetricsBatch.intake`.
6. **Lineups (R11).** Every pass: `CellTracker.due()` → one `LineupRequest` with all due cells (nothing while disconnected). A `LineupGrants` message: for each grant `CellTracker.granted(worker, env, epoch, ttl_sec, lineup_from_proto(grant.lineup), keys)` where `keys` = per seat: `latest` seat → `CacheKey("latest", agent)`, `ckpt_v<N>` seat → `CacheKey("snapshot", agent, "ckpt_v<N>")`, `fixed` seat of a frozen agent → its blob key, of a scripted agent → none. Then for each waiting grant: every key present (`latest`: `WeightCacheReader.latest_version(agent) is not None`; others `has`) → `deliver` and send `WorkerLineups` (one message per worker per pass) — preceded once per worker incarnation by `LearnerRoute` for every trainable agent and `WeightsReady(agents with a latest in the cache)`; a missing snapshot / blob → one `Download` per key in flight; its success → `WeightCacheWriter.put`, deliver the grants it completes; its failure (`NOT_FOUND`, digest, transport) → `failed(worker, env)` for every grant waiting for it and `lineup_fetch_failed += 1` per grant (sent in `Heartbeat.counters`), WARNING once per key; a missing `latest` is waited for (it appears when the learner publishes). Keys no cell references any more (`referenced()`) leave the cache except `latest` pointers and the `Welcome.fixed` blobs.
7. **Announces.** `Announce.env_steps` → `LearnerEnvSteps` to every learner (b); `Announce.evictions` → `WorkerEvict([snapshot key])` to every worker; the cache entry is dropped once `referenced()` no longer holds it.
8. **Heartbeat** every `distributed.heartbeat_sec` (from the Welcome's config): `env_steps` (the session's cumulative), `dropped_results` (= `ResultBuffer.dropped`), `procs` (`ProcState(name, "alive" | "dead" | "disabled", restarts)`), `load` (`os.getloadavg()[0]` where available, else 0), `cached_keys`, `counters` (`lineup_fetch_failed`, `dropped_results_node_buffer`, `worker_restarts`). `MetricsBatch` every `METRICS_FLUSH_SEC = 0.5`. Due result batches (`ResultBuffer.due()`) every pass. `Ack` → `ResultBuffer.ack`. `SessionUp` after a `SessionLost` → resend `ResultBuffer.unacked()` in seq order.
9. **Commands.** `start_learner` / `stop_learner` (b), `stop_run` (b: the stop protocol below), `release` (b: stop the run's children, forget the run, keep the session; `--once` exits 0, T6.3), `set_evaluators` / `update_code`: logged and ignored until T5.1 / T8.3.
10. **Stop** (b; `stop_run`, `request_stop`, the node's own SIGINT / SIGTERM for `join` T6.3): send `Stop` to every child; keep the main loop running (finals, results, progress) until every child exited or `grace` passed (`LOCAL_CHILD_GRACE_SEC = SHUTDOWN_GRACE_SEC - 2.0` for the local node; `distributed.shutdown_grace_sec - 5.0` otherwise); `stop_all(0)` kills the rest; flush the result buffer and wait for its acks and for the uploads in flight until the same deadline + 1 s; send `Event(kind="run_stopped")`; remove the node's cache files (every file of its prefix) and its lock; close the session; `run` returns 0 (the local node then exits; a `join` node waits for the next run in T6.3).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp5_node_books.py` (verified in the scratch package: 5 passed):
```python
"""The node's result buffer and cell tracker (SP5 T2.4; spec 4.2, 4.3, R11): batches by size and age, seqs, kept
until acked, overflow drops the new batch with one WARNING; cells requested once until answered, again after their TTL
or a failed fetch, grants wait for their weights, whole-run lineups never expire, dead incarnations lose their cells."""
from __future__ import annotations

import logging

from colosseum.node.agent import CellTracker, ResultBuffer


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_results_are_batched_by_size_and_age_numbered_and_kept_until_acked():
    clock = Clock()
    buf = ResultBuffer(batch=3, flush_sec=0.25, limit_bytes=1000, make=lambda seq, b: (seq, b),
                       size_of=lambda m: 10 * len(m[1]), clock=clock)
    assert buf.add([1, 2]) == [] and buf.due() == []
    assert buf.add([3, 4]) == [(1, [1, 2, 3])]
    clock.t = 0.3
    assert buf.due() == [(2, [4])]
    assert buf.unacked() == [(1, [1, 2, 3]), (2, [4])] and buf.kept_bytes == 40 and buf.last_acked == 0
    assert buf.ack([1, 99]) == 1 and buf.unacked() == [(2, [4])] and buf.last_acked == 1
    buf.ack([2])
    assert buf.unacked() == [] and buf.last_acked == 2 and buf.kept_bytes == 0


def test_a_full_buffer_drops_the_new_batch_and_warns_once(caplog):
    clock = Clock()
    buf = ResultBuffer(batch=2, flush_sec=1.0, limit_bytes=50, make=lambda seq, b: (seq, b),
                       size_of=lambda m: 20 * len(m[1]), clock=clock)
    with caplog.at_level(logging.WARNING):
        assert [s for s, _ in buf.add(range(6))] == [1]            # 40 kept, then two batches of 40 do not fit
        assert buf.add([7]) == [] and buf.flush() == []
    assert buf.dropped == 5 and buf.unacked() == [(1, [0, 1])]
    assert len([r for r in caplog.records if "result buffer full" in r.getMessage()]) == 1
    buf.ack([1])
    assert buf.add([8, 9]) == [(2, [8, 9])]                         # room again; numbering goes on


def test_cells_are_requested_once_until_answered_and_again_after_their_ttl():
    clock = Clock()
    cells = CellTracker(retry_sec=1.0, clock=clock)
    cells.add_worker(0, 2, incarnation=1)
    assert cells.due() == [(0, 0, 0, 1), (0, 1, 0, 1)] and cells.due() == []
    assert cells.granted(0, 0, epoch=7, ttl_sec=30.0, lineup="L0", keys={0: "k-latest"})
    assert cells.waiting() == [(0, 0, "L0", {0: "k-latest"})] and cells.referenced() == {"k-latest"}
    assert cells.deliver(0, 0) == ("L0", {0: "k-latest"}) and cells.waiting() == []
    clock.t = 1.5
    assert cells.due() == [(0, 1, 0, 1)]                            # no answer within retry_sec: asked again
    clock.t = 31.0
    assert (0, 0, 7, 1) in cells.due()                              # ttl expired: asked with its epoch
    assert cells.live() == [(0, 0, 7, 1)]


def test_a_failed_fetch_keeps_the_old_lineup_and_retries_later():
    clock = Clock()
    cells = CellTracker(retry_sec=2.0, clock=clock)
    cells.add_worker(0, 1, incarnation=1)
    cells.due()
    cells.granted(0, 0, epoch=3, ttl_sec=30.0, lineup="old", keys={0: "a"})
    cells.deliver(0, 0)
    clock.t = 31.0
    cells.due()
    cells.granted(0, 0, epoch=3, ttl_sec=30.0, lineup="new", keys={0: "snap"})
    assert cells.referenced() == {"a", "snap"}
    cells.failed(0, 0)
    assert cells.referenced() == {"a"} and cells.due() == []
    clock.t = 33.5
    assert cells.due() == [(0, 0, 3, 1)]


def test_whole_run_lineups_never_expire_and_dead_incarnations_lose_their_cells():
    clock = Clock()
    cells = CellTracker(retry_sec=1.0, clock=clock)
    cells.add_worker(1, 1, incarnation=1)
    cells.due()
    cells.granted(1, 0, epoch=4, ttl_sec=0.0, lineup="L", keys={})
    cells.deliver(1, 0)
    clock.t = 10**6
    assert cells.due() == []
    cells.drop_worker(1)
    assert not cells.granted(1, 0, epoch=5, ttl_sec=0.0, lineup="L", keys={})
    cells.add_worker(1, 1, incarnation=2)
    assert cells.due() == [(1, 0, 0, 2)]
```

Create `tests/unit/test_sp5_node_procs.py` (verified: 4 passed, about 5 s; one spawned sleeper at a time):
```python
"""Node children (SP5 T2.4; spec 1, 7.1): incarnations, exits reported once with their signal, restart limits per
role in a sliding window, a bounded stop that kills stragglers."""
from __future__ import annotations

import os
import signal
import time

from colosseum.node import procs
from colosseum.node.procs import ChildSpec, ChildSupervisor


def _sleeper(*, seconds: float, incarnation: int) -> None:
    from colosseum.utils.process import init_child_process
    init_child_process()
    time.sleep(seconds)


def _stubborn(*, incarnation: int) -> None:
    from colosseum.utils.process import init_child_process
    init_child_process()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)


def _wait(pred, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_incarnations_grow_and_exits_are_reported_once():
    sup = ChildSupervisor()
    assert sup.start(ChildSpec("worker-0", "worker", _sleeper, {"seconds": 0.0})) == 1
    assert _wait(lambda: sup.get("worker-0").exitcode is not None)
    assert sup.exited() == [("worker-0", 0)] and sup.exited() == []
    assert sup.restart("worker-0") == 2 and sup.incarnation("worker-0") == 2
    sup.stop_all(5.0)


def test_restart_limits_per_role_in_a_sliding_window(monkeypatch):
    monkeypatch.setattr(procs, "WORKER_RESTARTS", (2, 10.0))
    t = [0.0]
    sup = ChildSupervisor(clock=lambda: t[0])
    sup.start(ChildSpec("worker-0", "worker", _sleeper, {"seconds": 0.0}))
    sup.start(ChildSpec("learner-a", "learner", _sleeper, {"seconds": 0.0}))
    assert sup.may_restart("worker-0") and sup.may_restart("worker-0") and not sup.may_restart("worker-0")
    t[0] = 10.5
    assert sup.may_restart("worker-0")
    assert all(sup.may_restart("learner-a") for _ in range(3)) and not sup.may_restart("learner-a")
    assert not sup.may_restart("unknown")


def test_stop_all_kills_stragglers_within_its_bound():
    sup = ChildSupervisor()
    sup.start(ChildSpec("worker-0", "worker", _stubborn))
    assert _wait(lambda: sup.get("worker-0").is_alive())
    time.sleep(1.0)
    start = time.monotonic()
    assert sup.stop_all(0.5) == ["worker-0"]
    assert time.monotonic() - start < 4.0 and sup.get("worker-0").exitcode == -signal.SIGKILL


def test_a_killed_child_is_reported_with_its_signal():
    sup = ChildSupervisor()
    sup.start(ChildSpec("worker-0", "worker", _sleeper, {"seconds": 30.0}))
    assert _wait(lambda: sup.get("worker-0").pid is not None)
    os.kill(sup.get("worker-0").pid, signal.SIGKILL)
    assert _wait(lambda: sup.exited() == [("worker-0", -signal.SIGKILL)])
```

Create `tests/unit/test_sp5_node_hardware.py` (verified: 5 passed):
```python
"""Node hardware (SP5 T2.4; spec 2.6, 7.7): cgroup v2 / v1 quotas, physical cores, the probe honouring quota and
affinity, default slots, a stable machine id, bounded log tails."""
from __future__ import annotations

import math

from colosseum.node import hardware
from colosseum.node.hardware import Hardware, cgroup_quota_cores, default_slots, physical_cores_raw
from colosseum.node.logs import LOG_TAIL_BYTES, log_tail


def _hw(physical, gpus=()):
    return Hardware(physical * 2, physical, 1 << 30, tuple(gpus), "linux", "p", "3.12", "2.6.0", "cpu")


def test_cgroup_quotas_v2_and_v1(tmp_path):
    (tmp_path / "cpu.max").write_text("250000 100000\n")
    assert math.isclose(cgroup_quota_cores(tmp_path), 2.5)
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert cgroup_quota_cores(tmp_path) is None
    (tmp_path / "cpu.max").unlink()
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("150000")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000")
    assert math.isclose(cgroup_quota_cores(tmp_path), 1.5)
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1")
    assert cgroup_quota_cores(tmp_path) is None


def test_physical_cores_count_core_ids_per_socket(tmp_path):
    info = tmp_path / "cpuinfo"
    blocks = [f"processor: {i}\nphysical id: {i // 8}\ncore id: {(i % 8) // 2}\n" for i in range(16)]
    info.write_text("\n".join(blocks))
    assert physical_cores_raw(info) == 8


def test_probe_honours_the_quota_and_affinity(monkeypatch):
    monkeypatch.setattr(hardware, "affinity_cores", lambda: 16)
    monkeypatch.setattr(hardware, "cgroup_quota_cores", lambda: 2.5)
    monkeypatch.setattr(hardware, "physical_cores_raw", lambda: 8)
    monkeypatch.setattr(hardware, "nvidia_gpus", lambda: [{"index": 0, "name": "x", "memory_bytes": 1, "kind": "cuda"}])
    hw = hardware.probe()
    assert (hw.logical_cores, hw.physical_cores) == (3, 3) and hw.gpus[0]["kind"] == "cuda"


def test_default_slots_follow_the_spec():
    assert default_slots(_hw(1)) == {"workers": 1, "evaluators": 0, "cpu_learners": 0, "gpus": []}
    assert default_slots(_hw(3)) == {"workers": 2, "evaluators": 0, "cpu_learners": 0, "gpus": []}
    assert default_slots(_hw(8, [{"index": 0}, {"index": 1}])) == {"workers": 7, "evaluators": 1, "cpu_learners": 1,
                                                                     "gpus": [0, 1]}


def test_machine_id_is_stable_and_the_log_tail_is_bounded(tmp_path):
    assert hardware.machine_id() == hardware.machine_id() and hardware.machine_id()
    log = tmp_path / "worker-0.log"
    assert log_tail(log) == b""
    log.write_bytes(b"a" * 10 + b"b" * LOG_TAIL_BYTES)
    assert log_tail(log) == b"b" * LOG_TAIL_BYTES and log_tail(log, 4) == b"bbbb"
```

Create `tests/integration/test_sp5_node_agent.py`: a `NodeAgent` running in a thread of the test process against `node_kit.FakeHub` (real gRPC server built with `hub.server.build_server(insecure=True)` and a real `HubIdentity` under `tmp_path`; a scripted inbox handler), on the toy game `turns` with `make_test_config("turns", agents={"a": {}, "random": scripted_agent(), "bot2": scripted_agent()})`, `rollout.envs_per_worker=2`, `chunk_length=8`, at most one worker and one learner (≤ 2 loaded CPU processes). Each test, its setup and assertions:
- `test_the_node_registers_with_the_token_and_keeps_its_identity`: a non-local node registers (FakeHub answers `Register`); `NodeIdentity.load_or_create(fp, identity_root)` afterwards has the node id and number; a second `NodeAgent` with the same identity root sends `Hello` with the same node id and does not register again; a wrong token → `run()` returns 1 and the log names the refusal.
- `test_workers_start_on_welcome_and_ask_lineups_for_every_env`: Welcome with no learners and no fixed blobs; FakeHub records one `LineupRequest` whose cells are `(worker 0, env 0/1, epoch 0, incarnation 1)`; it grants `random` vs `bot2` (fixed seats, `collect=False`) with epochs 5, 6 and `ttl_sec=1.0`; results arrive in `ResultBatch`es with uids `R/train/n<number>/w0_e<E>_g<5|6>_ep<K>`, `node == number`; after the TTL a second request carries epochs 5 and 6 (FakeHub grants the same epochs).
- `test_results_are_batched_and_acked` (`distributed.results_batch=4`): every `ResultBatch` has at most 4 results, seqs `1, 2, 3, ...`; acked batches are not resent.
- `test_unacked_results_are_resent_after_a_reconnect_and_accounted_once`: FakeHub holds acks, then closes the session stream; the node reconnects (backoff monkeypatched to 0.05 s) with the same `session_id` and `last_acked_seq`; it resends every unacked batch in seq order; FakeHub dedups by seq (`ResultIntake`) and its accounted uids are unique and complete (every episode the worker reported, counted by the worker's `episodes` counter in `WorkerStats`).
- `test_results_wait_in_the_buffer_while_the_hub_is_away` (`distributed.node_result_buffer_mb` tiny, e.g. 0.002): FakeHub stops answering (server stopped), the node keeps playing; `ResultBuffer.dropped > 0`, one WARNING "node result buffer full"; when FakeHub comes back on the same port, the kept batches arrive and the next `Heartbeat` has `dropped_results == counters["dropped_results_node_buffer"] == ResultBuffer.dropped`.
- `test_heartbeats_carry_cumulative_env_steps_across_a_worker_restart`: kill -9 the worker (`os.kill(pid, SIGKILL)`); FakeHub receives `Event(kind="proc_died", proc="worker-0")`; a new `LineupRequest` with incarnation 2 and epoch 0; the heartbeats' `env_steps` never decrease and keep growing after the restart; `counters["worker_restarts"] == 1`.
- `test_a_worker_beyond_its_restart_limit_disables_its_slot` (monkeypatch `node.procs.WORKER_RESTARTS = (1, 600.0)`; kill twice): `Event(kind="slot_disabled", proc="worker-0")`, no third start, `ProcState("worker-0", "disabled", 1)`.
- `test_a_contract_error_is_reported_with_its_context_and_not_restarted`: the toy env class `game_helpers.ContractBreakingGame` (add to `game_helpers`: steps with an acting seat outside the layout after 3 steps, raising `EnvContractError`) → `Event(kind="child_error", error_kind="contract")` whose `context` starts with `node <number> (`, names `worker 0, env`, `episode step` and `layout 2p`, and whose `log_path` ends with `worker-0.log`; no restart.
- `test_a_lineup_waits_for_its_snapshot_and_a_failed_fetch_keeps_the_old_one`: a fixed-seat lineup plays first; then FakeHub grants a lineup with seat `a@ckpt_v5` and serves `Download(snapshot, a, ckpt_v5)` only after it releases a gate (a `threading.Event`): results keep the old lineup until the gate opens, then results with `ckpt_v5` (network id in the seat) appear; next a grant naming `ckpt_v9`, which FakeHub answers `NOT_FOUND`: the old lineup keeps playing, `Heartbeat.counters["lineup_fetch_failed"] == 1`, the cell is requested again after `LINEUP_RETRY_SEC`.
- `test_evictions_reach_the_workers_and_leave_the_cache`: after `ckpt_v5` played, `Announce(evictions=[Eviction("a", "ckpt_v5")])` plus a new grant without it → the worker log has "unloaded evicted snapshot ckpt_v5"; the shm file `{prefix}-a-ckpt_v5` is gone.
- `test_frozen_players_come_from_the_hubs_blobs`: config with a frozen agent (`frozen_agent(write_ckpt_dir(...))`); FakeHub registers its blob and lists it in `Welcome.fixed`; the node fetches it before starting the worker (FakeHub sees the `Download` first); a lineup seating the frozen agent plays (results name it with network `fixed`).
- (b) `test_a_learner_trains_on_the_node_and_uploads_its_snapshots`: Welcome with `LearnerAssignment("a", generation=1, version_floor=0, device="cpu", env_steps=0)`; FakeHub grants `a` vs `a` (latest, owner collecting) only after it got the first `VersionAnnounce("a", generation=1, version=0)`; the learner trains (`MetricsBatch.train` with `train_step >= 3`), `VersionAnnounce` versions grow, `Upload(kind="checkpoint")` arrives at `checkpoint.interval`; FakeHub's `Announce(env_steps=N)` reaches the learner (`train` metrics' `progress` > 0); `Command(stop_run)` → `Upload(kind="final")`, `Event(kind="run_stopped")`, `run()` returns 0, no child alive, no shm file of the node's prefix left.
- (b) `test_a_killed_learner_blocks_neither_the_worker_nor_the_node`: during the previous setup, kill -9 the learner; results keep arriving from the worker (its chunks are dropped as `disconnect` / `no_learner` and counted in `WorkerStats.counters`), heartbeats continue, `Event(kind="learner_failed", proc="learner-a")` arrives.
- (b) `test_children_die_with_their_node`: run the node as a spawned process (`local_node_target`-like helper in `node_kit`), then kill -9 it; its worker and learner pids disappear within 5 s (`PR_SET_PDEATHSIG`).
- (b) `test_stop_finishes_within_the_local_grace`: `Command(stop_run)` with a learner busy training → `run()` returns within `LOCAL_CHILD_GRACE_SEC + 2.0` s (measured with `time.monotonic()`, the only duration asserted here) and the final upload arrived.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_node_books.py tests/unit/test_sp5_node_procs.py tests/unit/test_sp5_node_hardware.py tests/integration/test_sp5_node_agent.py -q`
Expected: import errors (`No module named 'colosseum.node.agent'`, `'colosseum.node.procs'`, ...).

- [ ] **Step 3: `src/colosseum/node/procs.py`, `hardware.py`, `logs.py`**

`src/colosseum/node/procs.py` (complete; verified):
```python
"""Child processes of a node (SP5 spec 1, 7.1): spawn, named incarnations, restart limits, bounded stop.

Every child is started with ``utils.process.start_process`` (SIGINT blocked until its ``run_child`` body ignores it)
and runs ``init_child_process`` (``PR_SET_PDEATHSIG``: it dies with the node; T4.4 adds the macOS watchdog). The
supervisor never shares a lock, queue or event with a child: it only starts, polls (``exitcode``), terminates and
kills them; everything else goes over the children's own connections (``net.local``)."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from colosseum.utils.process import start_process

logger = logging.getLogger(__name__)

WORKER_RESTARTS = (3, 600.0)        # max restarts per window (s): workers and evaluators (spec 7.1)
LEARNER_RESTARTS = (3, 1800.0)      # learners (T4.2 uses it)


@dataclass(frozen=True)
class ChildSpec:
    name: str                                       # "worker-0", "learner-main", "evaluator-0"
    role: Literal["worker", "learner", "evaluator"]
    target: Callable
    kwargs: dict = field(default_factory=dict)
    daemon: bool = True


@dataclass
class _Child:
    spec: ChildSpec
    proc: Any
    incarnation: int
    reported: bool = False                          # exit returned by exited() already


class ChildSupervisor:
    """Children by name. ``start`` gives a new incarnation (1, 2, ...) per name and passes it to the target as the
    ``incarnation`` keyword; ``exited`` returns the children that ended since its last call; ``may_restart`` applies
    the restart limit of the child's role in a sliding window (``clock``: monotonic); ``stop_all(grace)`` waits up to
    ``grace`` for every child to exit (the caller sent ``Stop`` first), then terminates and kills the rest (about
    1 s each) and returns their names."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, ctx: Any = None) -> None:
        self._clock = clock
        self._ctx = ctx if ctx is not None else mp.get_context("spawn")
        self._children: dict[str, _Child] = {}
        self._incarnations: dict[str, int] = {}
        self._restarts: dict[str, deque[float]] = {}

    def start(self, spec: ChildSpec) -> int:
        incarnation = self._incarnations.get(spec.name, 0) + 1
        self._incarnations[spec.name] = incarnation
        proc = self._ctx.Process(target=spec.target, name=spec.name, daemon=spec.daemon,
                                 kwargs={**spec.kwargs, "incarnation": incarnation})
        start_process(proc)
        self._children[spec.name] = _Child(spec, proc, incarnation)
        return incarnation

    def get(self, name: str) -> Any | None:
        child = self._children.get(name)
        return None if child is None else child.proc

    def incarnation(self, name: str) -> int:
        return self._incarnations.get(name, 0)

    def names(self) -> list[str]:
        return list(self._children)

    def sentinels(self) -> dict[int, str]:
        """Process sentinels of the live children (for ``multiprocessing.connection.wait``)."""
        return {c.proc.sentinel: n for n, c in self._children.items() if c.proc.exitcode is None}

    def exited(self) -> list[tuple[str, int | None]]:
        out = []
        for name, child in self._children.items():
            if not child.reported and child.proc.exitcode is not None:
                child.reported = True
                out.append((name, child.proc.exitcode))
        return out

    def may_restart(self, name: str, now: float | None = None) -> bool:
        """Whether ``name`` may start once more: fewer than the role's limit of restarts in the window (and the
        restart is then recorded)."""
        child = self._children.get(name)
        if child is None:
            return False
        limit, window = LEARNER_RESTARTS if child.spec.role == "learner" else WORKER_RESTARTS
        now = self._clock() if now is None else now
        times = self._restarts.setdefault(name, deque())
        while times and now - times[0] > window:
            times.popleft()
        if len(times) >= limit:
            return False
        times.append(now)
        return True

    def restart(self, name: str) -> int:
        return self.start(self._children[name].spec)

    def forget(self, name: str) -> None:
        self._children.pop(name, None)

    def stop_all(self, grace: float) -> list[str]:
        deadline = self._clock() + max(0.0, grace)
        while self._clock() < deadline and any(c.proc.is_alive() for c in self._children.values()):
            time.sleep(0.05)
        remaining = [n for n, c in self._children.items() if c.proc.is_alive()]
        for name in remaining:
            self._children[name].proc.terminate()
        self._join(remaining, 1.0)
        for name in [n for n in remaining if self._children[n].proc.is_alive()]:
            self._children[name].proc.kill()
        self._join(remaining, 1.0)
        if remaining:
            logger.warning(f"node: terminated children that did not stop within {grace:.1f} s: {remaining}")
        return remaining

    def _join(self, names: list[str], timeout: float) -> None:
        end = self._clock() + timeout
        for name in names:
            self._children[name].proc.join(max(0.0, end - self._clock()))
```
(`watch_parent` is added by T4.4.)

`src/colosseum/node/hardware.py` (complete; verified with the fakes of the tests):
```python
"""What a node reports about its machine (SP5 spec 2.6) and the default slots.

Cores honour the process's CPU affinity and a cgroup v2 / v1 CPU quota (containers); physical cores come from
``/proc/cpuinfo`` (pairs of ``physical id`` / ``core id``), else ``sysctl hw.physicalcpu`` (macOS), else the logical
count; both are capped by the quota. GPUs come from ``nvidia-smi`` (also ``/usr/lib/wsl/lib/nvidia-smi`` on WSL2),
without initializing CUDA in the node process; MPS from ``torch.backends.mps``. Every probe is a module-level
function, so tests replace them."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_NVIDIA_SMI = ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi")
_CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclass(frozen=True)
class Hardware:
    logical_cores: int
    physical_cores: int
    memory_bytes: int
    gpus: tuple[dict, ...]          # {"index", "name", "memory_bytes", "kind": "cuda" | "mps"}
    os: str
    platform: str
    python: str
    torch_version: str
    torch_build: str


def affinity_cores() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def cgroup_quota_cores(root: Path = _CGROUP_ROOT) -> float | None:
    """The CPU quota in cores (cgroup v2 ``cpu.max``, else v1 ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``); None
    without a quota."""
    try:
        text = (root / "cpu.max").read_text().split()
        if text and text[0] != "max":
            return int(text[0]) / int(text[1])
        if text:
            return None
    except (OSError, ValueError, IndexError):
        pass
    try:
        quota = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
        return quota / period if quota > 0 and period > 0 else None
    except (OSError, ValueError):
        return None


def physical_cores_raw(cpuinfo: Path = Path("/proc/cpuinfo")) -> int | None:
    try:
        pairs, phys = set(), None
        for line in cpuinfo.read_text().splitlines():
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "physical id":
                phys = value.strip()
            elif key == "core id":
                pairs.add((phys, value.strip()))
        if pairs:
            return len(pairs)
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True,
                                      timeout=5).stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


def memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


def nvidia_gpus() -> list[dict]:
    for exe in _NVIDIA_SMI:
        path = shutil.which(exe) or (exe if os.path.isfile(exe) else None)
        if path is None:
            continue
        try:
            out = subprocess.run([path, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        gpus = []
        for line in out.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[0].isdigit():
                gpus.append({"index": int(parts[0]), "name": parts[1], "memory_bytes": int(float(parts[2])) << 20,
                             "kind": "cuda"})
        return gpus
    return []


def mps_available() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001 - no torch / no MPS
        return False


def torch_versions() -> tuple[str, str]:
    """``(version, build)``: ``2.6.0+cu124`` -> ``("2.6.0", "cu124")``; no suffix: ``"cpu"`` (or ``"mps"`` on macOS)."""
    try:
        import torch
        version = str(torch.__version__)
    except Exception:  # noqa: BLE001
        return "", ""
    base, _, build = version.partition("+")
    return base, build or ("mps" if sys.platform == "darwin" else "cpu")


def probe() -> Hardware:
    logical = affinity_cores()
    quota = cgroup_quota_cores()
    if quota is not None:
        logical = max(1, min(logical, math.ceil(quota)))
    physical = physical_cores_raw() or logical
    physical = max(1, min(physical, logical))
    gpus = nvidia_gpus()
    if not gpus and sys.platform == "darwin" and mps_available():
        gpus = [{"index": 0, "name": "mps", "memory_bytes": 0, "kind": "mps"}]
    version, build = torch_versions()
    return Hardware(logical_cores=logical, physical_cores=physical, memory_bytes=memory_bytes(), gpus=tuple(gpus),
                    os=sys.platform, platform=platform.platform(), python=platform.python_version(),
                    torch_version=version, torch_build=build)


def default_slots(hw: Hardware) -> dict:
    """Spec 2.6: workers = physical cores - 1 (at least 1); one evaluator and one CPU learner on >= 4 physical
    cores; one learner slot per GPU."""
    big = hw.physical_cores >= 4
    return {"workers": max(1, hw.physical_cores - 1), "evaluators": 1 if big else 0,
            "cpu_learners": 1 if big else 0, "gpus": [int(g["index"]) for g in hw.gpus]}


def machine_id() -> str:
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            text = Path(path).read_text().strip()
            if text:
                return text
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], capture_output=True, text=True,
                                 timeout=5).stdout
            for line in out.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split("=")[-1].strip().strip('"')
        except (OSError, subprocess.SubprocessError):
            pass
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:32]
```

`src/colosseum/node/logs.py` (complete; `LOG_TAIL_BYTES` is the overview's constant):
```python
"""Node logs (SP5 spec 7.7): a remote node's processes log under ``<cache_home>/colosseum/logs/<ca8>/<run_id>/``;
``log_tail`` is what a node sends the hub when a process dies (T4.4 stores it as ``<proc>.log.tail``)."""

from __future__ import annotations

from pathlib import Path

LOG_TAIL_BYTES = 64 * 1024


def node_log_dir(ca8: str, run_id: str) -> Path:
    from colosseum.net.security import cache_home

    path = cache_home() / "colosseum" / "logs" / ca8 / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_tail(path: Path, limit: int = LOG_TAIL_BYTES) -> bytes:
    """The last ``limit`` bytes of ``path`` (b"" when it does not exist)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read()
    except OSError:
        return b""
```

- [ ] **Step 4: The pure helpers of `src/colosseum/node/agent.py`**

At the top of `node/agent.py` (after the module docstring, which states the loop of "What the node does"), the two books (complete; verified):
```python
LINEUP_RETRY_SEC = 1.0          # a requested cell without an answer, or after a failed fetch, is asked again
METRICS_FLUSH_SEC = 0.5         # MetricsBatch cadence
LOCAL_CHILD_GRACE_SEC = SHUTDOWN_GRACE_SEC - 2.0   # children of train's local node: before the hub's 7 s


class ResultBuffer:
    """The node's unacked result batches (spec 4.3). Results are gathered into batches of at most ``batch`` results or
    every ``flush_sec`` (``due``), numbered per session from 1; ``make(seq, results)`` builds the message once (a
    ``ResultBatch``) and ``size_of(message)`` its encoded size. A batch is kept until its ack, also while the hub is
    away; ``limit_bytes`` bounds the kept bytes: a new batch that does not fit is dropped whole and counted in
    ``dropped`` (one WARNING; overview deviation: the newest is dropped, the oldest keep being resent).
    ``unacked()`` lists the kept messages in seq order (resent after every reconnect)."""

    def __init__(self, *, batch: int, flush_sec: float, limit_bytes: int, make: Callable[[int, list], Any],
                 size_of: Callable[[Any], int], clock: Callable[[], float]) -> None:
        self._batch, self._flush_sec, self._limit = int(batch), float(flush_sec), int(limit_bytes)
        self._make, self._size_of, self._clock = make, size_of, clock
        self._pending: list = []
        self._since = clock()
        self._next_seq = 1
        self._kept: dict[int, tuple[Any, int]] = {}
        self._bytes = 0
        self.dropped = 0
        self._warned = False

    def add(self, results: Iterable) -> list[Any]:
        """Gather ``results``; returns the messages cut now (to send)."""
        out = []
        for r in results:
            if not self._pending:
                self._since = self._clock()
            self._pending.append(r)
            if len(self._pending) >= self._batch:
                out.extend(self._cut())
        return out

    def due(self) -> list[Any]:
        """The pending results as one message once ``flush_sec`` passed since the first of them."""
        if self._pending and self._clock() - self._since >= self._flush_sec:
            return self._cut()
        return []

    def flush(self) -> list[Any]:
        return self._cut() if self._pending else []

    def _cut(self) -> list[Any]:
        batch, self._pending = self._pending, []
        message = self._make(self._next_seq, batch)
        size = int(self._size_of(message))
        if self._bytes + size > self._limit:
            self.dropped += len(batch)
            if not self._warned:
                self._warned = True
                logger.warning(f"node result buffer full ({self._bytes / 2**20:.1f} MB unacked); dropped {len(batch)} "
                               f"match results (dropped_match_results.node_buffer; further drops are only counted)")
            return []
        self._kept[self._next_seq] = (message, size)
        self._next_seq += 1
        self._bytes += size
        return [message]

    def ack(self, seqs: Iterable[int]) -> int:
        n = 0
        for seq in seqs:
            kept = self._kept.pop(int(seq), None)
            if kept is not None:
                self._bytes -= kept[1]
                n += 1
        return n

    def unacked(self) -> list[Any]:
        return [self._kept[seq][0] for seq in sorted(self._kept)]

    @property
    def last_acked(self) -> int:
        """Every seq below the smallest kept one is acked (``Hello.last_acked_seq``)."""
        return (min(self._kept) if self._kept else self._next_seq) - 1

    @property
    def kept_bytes(self) -> int:
        return self._bytes


@dataclass
class _Cell:
    worker: int
    env: int
    incarnation: int
    epoch: int = 0
    deadline: float | None = None       # node monotonic; None: no lineup yet / due now; inf: whole run
    waiting: Any = None                 # a grant waiting for its weights: (lineup, keys)
    keys: dict = field(default_factory=dict)   # seat -> key of the lineup the worker has or has been sent
    requested: bool = False             # in a LineupRequest without an answer yet
    retry_at: float = 0.0


class CellTracker:
    """The node's cells (spec 4.2, R11): which need a lineup (new incarnation, TTL expired, refused grant), grants
    that wait for their weights, and the cache keys every live or waiting lineup references (``referenced``: a key
    that is not referenced may leave the cache). ``retry_sec``: how long a requested cell without an answer, or one
    whose fetch failed, waits before it is requested again."""

    def __init__(self, *, retry_sec: float, clock: Callable[[], float]) -> None:
        self._cells: dict[tuple[int, int], _Cell] = {}
        self._retry = float(retry_sec)
        self._clock = clock

    def add_worker(self, worker: int, envs: int, incarnation: int) -> None:
        for e in range(envs):
            self._cells[(worker, e)] = _Cell(worker, e, incarnation)

    def drop_worker(self, worker: int) -> None:
        for key in [k for k in self._cells if k[0] == worker]:
            del self._cells[key]

    def due(self) -> list[tuple[int, int, int, int]]:
        """Cells to request now, as ``(worker, env, epoch, incarnation)``; marks them requested."""
        now = self._clock()
        out = []
        for c in self._cells.values():
            if c.waiting is not None or now < c.retry_at:
                continue
            if c.deadline is None or now >= c.deadline:
                c.requested = True
                c.retry_at = now + self._retry
                out.append((c.worker, c.env, c.epoch, c.incarnation))
        return sorted(out)

    def granted(self, worker: int, env: int, epoch: int, ttl_sec: float, lineup: Any, keys: Mapping) -> bool:
        """A grant: kept as waiting until ``ready``; False for a cell this node no longer has (a dead incarnation)."""
        c = self._cells.get((worker, env))
        if c is None:
            return False
        now = self._clock()
        c.requested = False
        c.epoch = int(epoch)
        c.deadline = float("inf") if ttl_sec <= 0 else now + float(ttl_sec)
        c.waiting = (lineup, dict(keys))
        return True

    def waiting(self) -> list[tuple[int, int, Any, dict]]:
        return [(c.worker, c.env, c.waiting[0], c.waiting[1]) for c in self._cells.values() if c.waiting is not None]

    def deliver(self, worker: int, env: int) -> tuple[Any, dict]:
        """The waiting grant's weights are cached: it goes to the worker now."""
        c = self._cells[(worker, env)]
        lineup, keys = c.waiting
        c.waiting = None
        c.keys = keys
        return lineup, keys

    def failed(self, worker: int, env: int) -> None:
        """Fetching the waiting grant's weights failed: drop it, keep the old lineup, ask again after ``retry_sec``."""
        c = self._cells.get((worker, env))
        if c is None:
            return
        c.waiting = None
        c.deadline = None
        c.retry_at = self._clock() + self._retry

    def referenced(self) -> set:
        keys = set()
        for c in self._cells.values():
            keys.update(c.keys.values())
            if c.waiting is not None:
                keys.update(c.waiting[1].values())
        return keys

    def live(self) -> list[tuple[int, int, int, int]]:
        """``(worker, env, epoch, incarnation)`` of the cells with an epoch (``Hello.live_cells``)."""
        return sorted((c.worker, c.env, c.epoch, c.incarnation) for c in self._cells.values() if c.epoch)

    def epoch(self, worker: int, env: int) -> int:
        c = self._cells.get((worker, env))
        return 0 if c is None else c.epoch
```

- [ ] **Step 5: The worker body (`src/colosseum/node/worker_main.py`)**

```python
def create_env(env_class_path: str, kwargs: dict):
    """Create an env instance inside a worker process (was ``launcher._create_env``)."""


def worker_target(*, name: str, log_dir: str, incarnation: int, **kwargs) -> None:
    """Process entry point: ``run_child(name, log_dir, _worker_main, incarnation=incarnation, **kwargs)``."""


def _worker_main(*, incarnation: int, node: int, worker: int, node_address: str, authkey: bytes,
                 config: ColosseumConfig, agent_ids: list[str], agent_roles: dict[str, list[str]],
                 agent_configs: dict[str, ColosseumConfig], role_specs: dict[str, RoleSpec],
                 bots: dict[str, BotSpec], frozen_keys: dict[str, CacheKey], fixed_roles: dict[str, tuple[str, ...]],
                 teachers: dict[str, BotSpec], cache_root: str, cache_prefix: str, run_id: str) -> None: ...
```
Body: `configure_torch_threads(config.rollout.torch_threads)`; `control = connect(node_address, authkey, timeout=10)`, send `ChildHello("worker", name, incarnation, os.getpid(), "control")`; `route` likewise with channel `"route"`; `cache = WeightCacheReader(Path(cache_root), cache_prefix)`; `io = WorkerIO(node=node, worker=worker, incarnation=incarnation, control=control, route=route, cache=cache, routes={}, authkey=authkey, results_batch=RESULTS_BATCH_MAX, results_flush_sec=RESULTS_BATCH_SEC)` (the worker → node batching of SP4's FX1; the node re-batches by `distributed.results_batch`); `fixed = FixedPlayers(bots=bots, frozen={a: FrozenSpec(agent_id=a, roles=meta.roles, networks=meta.architecture, model_state=cached.state_dict, source=f"hub blob {key.ident}") for a, key in frozen_keys.items()}, roles=fixed_roles)` (`cached = cache.get(key)`; a missing one is a `RuntimeError`: the node fetched every fixed blob before it started the worker); wait for the first lineups of every env (`io.take_lineups()` accumulated per env, polling every 0.05 s, ending early when `io.stopped()`: then return); `checkpoint_state_dicts_by_agent` from the snapshot keys of those lineups (read from the cache); `seed = config.training.seed`; build `RolloutLoop(worker_id=worker, env_fn=partial(create_env, config.env.env_class, config.env.kwargs), num_envs=config.rollout.envs_per_worker, chunk_length=config.rollout.chunk_length, agent_ids=agent_ids, agent_roles=agent_roles, model_factories={a: partial(build_model, agent_configs[a], role_specs[a]) ...}, io=io.loop_io(), lineups=[cells[e].lineup for e in range(num_envs)], weight_sync_interval=config.rollout.weight_sync_interval_sec, checkpoint_state_dicts_by_agent=..., seed=worker_seed(seed, node, worker, incarnation), vec_env_kind=config.rollout.vec_env, subproc_workers=config.rollout.subproc_workers, max_idle_steps=config.env.max_idle_steps, fixed_players=fixed, teachers=teachers, run_id=run_id, wait_for_first_weights=lambda: io.stopped() or io.ready(), cell_seed_fn=lambda env, epoch: cell_seed(seed, epoch, node, worker, env), uid_prefix=f"{run_id}/train/n{node}/", snapshot_fallback=False)`; `io.stats_fn = lambda: loop.stats`; `loop.run(should_stop=lambda: (io.idle(), io.stopped())[1])`; `finally: io.close(); loop.close()`. Errors: `EnvContractError` → `io.send_error(ChildError("contract", str(e), _context(e)))` then re-raise; `PlayerError` → `"player"`; any other `Exception` → `"crash"` (the run stops: a reported Python exception in user code is not restarted; spec 7.1 restarts only deaths without a report). `_context(e)` is the error text's location prefix (`"worker W, env E, seat P, episode step K, layout L"` as SP2 / SP3 format it; the node prepends `node N (name), `).

- [ ] **Step 6: The agent (`src/colosseum/node/agent.py`)**

`NodeOptions` as in the contract plus `credentials: tuple[str, int, str] | None = None` (the local node's `(node_id, number, node_secret)`) and `shutdown_grace_sec: float | None = None` (None: the rule of item 10). `NodeAgent(options)`: `run() -> int` is the loop of "What the node does" items 1–9 (item 10 and the learner parts in (b)); `request_stop()` is thread-safe (an inbox event). Threads: the session reader (`HubSession`), the listener's accept thread, a transfer pool (`concurrent.futures.ThreadPoolExecutor(max_workers=4)`) for downloads and uploads; all post results into the node inbox (`queue.SimpleQueue` + a `multiprocessing.Pipe` as wake handle in the `wait` list). The main loop owns the children, the cells, the result buffer, the cache index and the counters; it never blocks on the network (every RPC runs in the pool or the session thread) and never on a child (it reads only connections that `wait` reported ready; sends to children are small control messages).

`main(argv=None) -> int` parses `NodeOptions` from the command line for `python -m colosseum.node.agent` (T6.2's loader execs it; T6.3 defines the flags; here only `--hub ADDR --token-file F [--insecure]` for tests).

`local_node_target(*, options: NodeOptions, log_dir: str) -> None`: `run_child("node", log_dir, lambda: sys.exit(NodeAgent(options).run()))` (the hub's spawn child, R12: spawn carries the hub's `sys.path` and cwd; `init_child_process` makes it ignore SIGINT and die with the hub).

Learner spawn kwargs (b) are built here: `learner_target(name=f"learner-{agent}", log_dir, node, agent_id, generation, listener_name=address(ca8, run8, number, f"learner-{agent}-i{incarnation}") (the incarnation is the supervisor's next one), node_address, authkey, config=agent_config, role_spec, spec, seed=learner_seed(training.seed, agent_index), teacher, init_state, resume_state, version_floor, env_steps, num_learners (learners on this node), cache_root, cache_prefix, checkpoint_interval=checkpoint.interval, total_timesteps, cap_bytes=payload_byte_cap(distributed.max_message_mb))`. The blobs of an assignment are fetched first (in the pool) and decoded in the node (`decode_weights(..., copy=True)`): `weights_blob` + `trainer_state_blob` → `resume_state = {"model_state", "trainer_state" (bytes), "policy_version" (the frame's meta), "env_steps", "source": f"hub {kind} blob"}`; a `teacher` blob (from `Welcome.fixed`, `agent_id` = the student) → `teacher_from_frozen(config, agent_id, frozen)`; a scripted teacher → `TeacherSpec(kind="scripted", bot=BotSpec(...))` from the config (`resolve_teacher` needs no file for it); an `init` blob → `init_from_blob(config, agent_id, state, source)`.

- [ ] **Step 7: `tests/node_kit.py` and the (a) tests**

- `FakeHub(tmp_path, config, *, ttl_sec=1.0)`: `build_server(inbox, identity=HubIdentity.load_or_create(tmp_path / "hub-id"), insecure=True, listen=["127.0.0.1:0"], loopback_plain=False, blobs=BlobStore(run_dir), max_message_mb=64)`; a handler thread over the inbox with scripted behaviour (`welcome(**fields)`, `grant(cell -> lineup)`, `hold_acks`, `serve_blob(kind, agent, ident, data, gate=None)`, `not_found(kind, agent, ident)`, `send_command(...)`, `announce(...)`, `close_session()`, `stop_server()` / `restart_server()` on the same port); records every message (`messages(kind)`), uses `hub.state.ResultIntake(journal=False)` for dedup and `CellTable` for epochs.
- `toy_node_options(tmp_path, hub, **kw) -> NodeOptions`: `hub_address=hub.address`, `token=hub.identity.token`, `insecure=True`, `workers=1`, `identity_root=tmp_path / "node-id"`, `log_dir=tmp_path / "node-logs"`.
- `run_node_in_thread(options) -> (agent, thread)`; `wait_for(pred, timeout)`.
- Add `node_kit` to `known-first-party`.
- `learner/factory.py`: `teacher_from_frozen(config, agent_id, frozen: FrozenSpec) -> TeacherSpec` (the `kickstart` settings of the agent's effective config, `kind="neural"`, `source=frozen.source`); `init_from_blob(config, agent_id, model_state, source) -> InitState` (`strict` from the agent's `init.strict`, `report=[]`: the hub logged the report). Both read no file.

Run the (a) tests: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_node_books.py tests/unit/test_sp5_node_procs.py tests/unit/test_sp5_node_hardware.py tests/integration/test_sp5_node_agent.py -q -k "not learner and not children_die and not stop_finishes"` — all pass. Then the full fast suite + ruff, commit `feat: node agent (children, result buffer, lineup pull after cache, heartbeats) with a fake hub`, push. **(End of dispatch (a).)**

- [ ] **Step 8: Learners on the node (`src/colosseum/node/learner_main.py`)** (b)

```python
def learner_target(*, name: str, log_dir: str, incarnation: int, **kwargs) -> None:
    """``run_child(name, log_dir, _learner_main, incarnation=incarnation, **kwargs)``."""
```
`_learner_main`: `apply_global_seed(seed)`; `device = resolve_device(config.learner.device)`; `configure_torch_threads(resolve_learner_threads(config.learner.torch_threads, device, config.rollout.num_workers, config.rollout.torch_threads, num_learners))`; `lst = listener(listener_name, authkey)` first, then `control = connect(node_address, authkey, timeout=10)` + `ChildHello("learner", name, incarnation, pid, "control")` (the node sends workers the route only after this hello, so the listener exists when they connect); `channel = ControlChannel(control)`; `teacher_chunks = teacher is not None and teacher.kind == "scripted"`; `intake = ConnectionIntake(lst, channel, ChunkValidator(role_spec, agent_id=agent_id, teacher=teacher_chunks), agent_id=agent_id, generation=generation, version_floor=version_floor, capacity=config.learner.queue_size, cap_bytes=cap_bytes)`; `intake.env_steps = env_steps`; `outputs = NodeOutputs(channel, WeightCacheWriter(Path(cache_root), cache_prefix), agent_id=agent_id, generation=generation)`; the `algorithm_factory` of SP4's `_learner_main` (`build_algorithm(config, role_spec, spec, device=device, teacher=teacher)`, then `apply_init` with `init_state`); `learner_process(agent_id=..., algorithm_factory=..., intake=intake, outputs=outputs, config=config.learner, stop_event=intake.stop_flag, checkpoint_interval=checkpoint_interval, resume_state=resume_state, progress_counter=intake, total_timesteps=total_timesteps, weight_sync_interval=config.rollout.weight_sync_interval_sec, version_floor=version_floor)`; `finally: intake.close()`; errors → `ChildError` on the channel as in the worker body.

- [ ] **Step 9: Learner handling and the stop protocol in the agent** (b)

Implement items 3 (learners of the Welcome), 5 (learner messages), 7 (`LearnerEnvSteps` forwarding), 9 (`start_learner` / `stop_learner` / `stop_run` / `release`) and 10 of "What the node does". Routes: after a learner's control hello, send `LearnerRoute(agent, address=listener_name, via_node=False, generation)` to every worker (and to every worker that starts later); a learner that stops or dies → `LearnerRoute(agent, None, False, 0)`.

- [ ] **Step 10: Run the (b) tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp5_node_agent.py -q`
Expected: all pass.

- [ ] **Step 11: Full fast suite + ruff; commit and push** (b)

```bash
git add src/colosseum/node/ src/colosseum/learner/factory.py tests/node_kit.py tests/game_helpers.py \
        tests/unit/test_sp5_node_books.py tests/unit/test_sp5_node_procs.py tests/unit/test_sp5_node_hardware.py \
        tests/integration/test_sp5_node_agent.py pyproject.toml
git commit -m "feat: node agent learners, uploads and the stop protocol"
git push
```

---


### Task T2.5: The switch — `train` = `Hub` + local node; `colosseum hub`; R8 accounting; tests ported; criterion 2

Spec 1 (the hub owns the run; `train` = hub + local child node on a free loopback port; `hub.json` and the running registry; `train --listen`; `hub` on all interfaces), 4.2 (lineups answered by the hub's matchmaker with epochs, the deficit rotation, no training cells without a collecting agent, seats of agents without a learner do not collect), 4.3 (intake checks, `env_steps` stamped at intake, ack after the journal commit), 4.4 (budget from heartbeats, the hub alone stops), 5.1 (SP3/SP4 logic unchanged; the journal watermark in `league_state.json`), 7.5 (exit codes of `hub` / `train`), criterion 2; rulings R5 (the hub is the evolved `Launcher`), R6 (the pulse child stays), R8, R12, R18. Runs alone.

**Split point:** dispatch (a) Steps 1–9: the hub and its wiring (`hub.py`, `registry.py`, `admin.py`, the writer change, the coordinator and metrics edits, the CLI, the bench), with `launcher.py` still present and its tests still passing on it; (b) Steps 10–13: every test that imports `colosseum.launcher` ported to the hub (list in Step 10), `launcher.py` deleted, the slow and GPU suites, the criterion-2 measurement. Each dispatch ends green, committed and pushed.

**Files:**
- Create: `src/colosseum/hub/hub.py`, `src/colosseum/hub/registry.py`, `src/colosseum/hub/admin.py`
- Modify: `src/colosseum/arena/matchlog.py` (`AppendStats.rows`, `Commit`, `MatchLogWriter.submit(..., token=)`, `take_commits`, `dead`), `src/colosseum/coordinator/coordinator.py` (`lineup_for`, `save_league_state(..., journal_row=None)`), `src/colosseum/metrics/aggregator.py` (worker stats keyed by (node, worker, incarnation)), `src/colosseum/net/security.py` (`NodeBook.add_ephemeral`), `src/colosseum/cli.py` (`train --listen`, the `hub` group), `scripts/bench_throughput.py` (`run_one` on `Hub`, `_make_config` `distributed.heartbeat_sec: 0.25`), `tests/cli_runner.py` (`hub_cmd`, `start_hub`), `src/colosseum/distributed.py` (`_create_env` import from `node.worker_main`)
- Delete (b): `src/colosseum/launcher.py`
- Modify (b): the ported tests of Step 10
- Test: `tests/unit/test_sp5_writer_commits.py`, `tests/unit/test_sp5_hub_registry.py`, `tests/unit/test_sp5_hub_lineups.py`, `tests/unit/test_sp5_hub_results.py`, `tests/unit/test_sp5_hub_admin.py`, `tests/integration/test_sp5_train_cli.py`, `tests/integration/test_sp5_hub_mode.py`, `tests/integration/test_sp5_two_trains_slow.py` (slow)

**Interfaces:**
- Consumes: every T2.1–T2.4 name; SP4's `Coordinator`, `CheckpointManager`, `MatchLog` / `MatchLogWriter`, `LiveRatings`, `MetricsHub`, `MetricsWriter`, `WandBLogger`, `PulseFeeder` / `pulse_target`, `ProcessSupervisor`, `setup_process_logging`, `RunDir`, `write_run_state`; `HubIdentity` (T1.3).
- Produces (contract "colosseum.hub.hub", "colosseum.hub.registry", "colosseum.hub.admin", "colosseum.arena.matchlog changes", "Metrics"; additions marked *): `LocalNodeOptions`, `HubOptions`, `Hub` (`env_steps_done`, `request_stop`, `launch`, the R5 method names), `run_training(config_path, overrides=None, *, listen=None)`, `run_hub(config_path, overrides=None, *, listen=None, port=None, insecure=False)` (*`resume`, `idle`, `nodes_file`, `dashboard` are added by T4.3, T5.2, T8.1, T7.1), `RunSetup`, `setup_run`, `validate_run_config`, `warn_static_ownership_skew` (moved), `INBOX_POLL_SEC`, `LOCAL_SHUTDOWN_GRACE_SEC`; `HUB_JSON`, `HubInfo` (*`loopback_tls`, *`to_json`, *`from_dict`), `write_hub_json`, `read_hub_json`, `register_running`, `unregister_running`, `running_hubs`, *`this_process_info`, *`alive`; `HubTarget`, `resolve_hub(..., insecure=False)`, `call`, `stream`, `ADMIN_COMMANDS` (+ *`"hub.status"`); `AppendStats.rows`, `Commit`, `MatchLogWriter.submit(..., token=None)`, `take_commits`, *`MatchLogWriter.dead`; `Coordinator.lineup_for`, `save_league_state(path, *, env_steps, journal_row=None)`; *`NodeBook.add_ephemeral(name, machine_id) -> (record, secret)` (number 0, kept in memory, never written to `nodes.json`); test support `cli_runner.hub_cmd`, `cli_runner.start_hub`.

**The hub's structure** (`colosseum.hub.hub.Hub`, R5). `launcher.py`'s code moves here; what stays word for word: `_build_coordinator` (its `env_steps` lambda reads `self._budget.value`), `_open_metrics`, `_open_match_log` (plus the token rule of Step 3), `_open_live_ratings`, `_wire_retention`, `_reapply_retention`, `_live_snapshot`, `_stored_snapshot_versions`, `_finish_live_ratings`, `_on_snapshot_evicted`, `_match_log_stats`, `_stop_match_log`, `_finish_metrics` (queue depths from the learners' `IntakeStats`), `_resolve_resume` (the env-step start goes into `BudgetCounter.add`), `_resolve_init`, `_load_league_state`, `_save_league_state` (with `journal_row`), `_import_snapshot_pool`, `_save_checkpoint`, `_pump_pulse`, `_write_pulse_state`, `_finish_pulse`, `_exit_code_after_shutdown`, `_signal_exit_code`, and the pulse queue handling (`_drain` / `QueueReader` / `release_command_queues` for the pulse queues only). What goes: the trajectory, weight, checkpoint, metrics, results and command queues, `SharedCounter`, `_start_children`'s workers and learners, `_refresh_worker_matches`, `_resolve_lineups`, `_drain_results` / `_drain_metrics` / `_drain_all_checkpoints` (replaced by handlers), `_flatten_result_batches`. State the main loop owns (all built in `launch`): `self._registry = NodeRegistry()`, `self._epochs = EpochAllocator(0, self._reserve_epochs)`, `self._cells = CellTable(self._epochs)`, `self._rotation = OwnerRotation(trainable)`, `self._versions = VersionRegistry()`, `self._intake = ResultIntake(journal=match_log.enabled)`, `self._budget = BudgetCounter()`, `self._sessions: dict[str, SessionHandle]` (by node id), `self._learner_node: dict[str, int]` (agent → node number), `self._learner_alive: dict[str, bool]`, `self._queue_depths: dict[str, int]`, `self._journal_row: int | None`, `self._admin_handlers: dict[str, Callable[[dict], dict]]` (T2.5: `hub.status`), `self._inbox = queue.SimpleQueue()`.

- [ ] **Step 1: Write the failing tests (part (a))**

Create `tests/unit/test_sp5_writer_commits.py` (verified against a copy of `src` with the Step 3 change: 6 passed; with it the 15 SP4 writer tests and the 30 tests of `test_sp4_matchlog.py` + `test_sp4_journal_runs.py` pass, after the one-line edit of Step 3):
```python
"""The journal writer's tokens and commits (SP5 T2.5, ruling R8): a tokened submit is written whole in one transaction
and committed in order with its rows; a resend after a hub restart is all duplicates; a tokened submit that does not
fit is refused whole and not counted; a write error or a dead writer gives failed commits and drops only untokened
results; groups never split between appends; tokens left at the stop are counted as dropped."""
from __future__ import annotations

import sqlite3
import time
from functools import partial

from colosseum.arena.matchlog import MATCH_LOG_FILE, WRITE_BATCH, AppendStats, MatchLog, MatchLogWriter, RunInfo
from game_helpers import synthetic_journal_results

RUN = RunInfo(run_id="R", name="run", parent_run_id=None, game="ttt", config_hash="h", checkpoint_interval=10)


def _wait(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class CommitFakeLog:
    def __init__(self, fail_appends=0):
        self.batches = []
        self.fail_appends = fail_appends

    def append(self, results):
        if self.fail_appends:
            self.fail_appends -= 1
            raise sqlite3.OperationalError("disk I/O error")
        self.batches.append(list(results))
        return AppendStats(detailed=len(results), accepted=[True] * len(results), rows=[None] * len(results))

    def rating_versions(self, result):
        return [None] * len(result.seats)

    def compact(self, **_):
        from colosseum.arena.matchlog import CompactStats
        return CompactStats()

    def sizes(self):
        from colosseum.arena.matchlog import LogSizes
        return LogSizes(0, 0, 0, 0, 0, 0, 10**12, False)

    def close(self):
        pass


def _commits(writer, n):
    got = []
    assert _wait(lambda: (got.extend(writer.take_commits()) or True) and len(got) >= n)
    return got


def test_tokened_submits_are_committed_whole_in_order_with_their_rows(tmp_path):
    factory = partial(MatchLog, tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: 10**12)
    writer = MatchLogWriter(factory, compact_interval_sec=3600.0, window_bytes=10**9)
    writer.start()
    results = synthetic_journal_results(9)
    groups = [results[:3], results[3:5], results[5:]]
    for seq, group in enumerate(groups, 1):
        assert writer.submit(group, token=("n", "s", seq)) == 0
    commits = _commits(writer, 3)
    assert [c.token for c in commits] == [("n", "s", 1), ("n", "s", 2), ("n", "s", 3)]
    assert [len(c.accepted) for c in commits] == [3, 2, 4] and all(not c.failed and c.duplicates == 0 for c in commits)
    rows = [c.last_row for c in commits]
    assert rows == sorted(rows) and rows[-1] == 9
    assert writer.submit(groups[0], token=("n", "s2", 1)) == 0               # a resend after a hub restart
    (again,) = _commits(writer, 1)
    assert (again.accepted, again.duplicates, again.last_row, again.failed) == ([], 3, None, False)
    stats = writer.stop(5.0)
    assert (stats.written, stats.duplicates, stats.dropped) == (9, 3, 0)


def test_a_tokened_submit_that_does_not_fit_is_refused_whole_and_not_counted():
    writer = MatchLogWriter(CommitFakeLog, queue_size=5, compact_interval_sec=3600.0, window_bytes=1)
    assert writer.submit(synthetic_journal_results(4), token="a") == 0
    assert writer.submit(synthetic_journal_results(2, first=4), token="b") == 2
    assert writer.stats.dropped == 0 and writer.stats.pending == 4
    assert writer.submit(synthetic_journal_results(3, first=6)) == 2           # untokened: SP4 rule
    assert writer.stats.dropped == 2
    writer.start()
    assert [c.token for c in _commits(writer, 1)] == ["a"]
    writer.stop(5.0)


def test_a_write_error_fails_the_tokened_groups_and_drops_only_the_untokened():
    log = CommitFakeLog(fail_appends=1)
    writer = MatchLogWriter(lambda: log, compact_interval_sec=3600.0, window_bytes=1)
    writer.submit(synthetic_journal_results(3), token="t")
    writer.submit(synthetic_journal_results(2, first=3))
    writer.start()
    (commit,) = _commits(writer, 1)
    assert commit.failed and commit.token == "t" and commit.accepted == []
    stats = writer.stop(5.0)
    assert (stats.dropped, stats.write_errors, stats.written) == (2, 1, 0)


def test_a_dead_writer_fails_queued_tokens_and_refuses_new_ones():
    def broken():
        raise sqlite3.OperationalError("unable to open database file")

    writer = MatchLogWriter(broken, compact_interval_sec=3600.0, window_bytes=1)
    writer.submit(synthetic_journal_results(3), token="t")
    writer.start()
    (commit,) = _commits(writer, 1)
    assert commit.failed and writer.dead and writer.stats.dropped == 0
    assert writer.submit(synthetic_journal_results(2), token="u") == 2 and writer.stats.dropped == 0
    writer.stop(5.0)


def test_a_tokened_group_is_never_split_between_two_appends():
    log = CommitFakeLog()
    writer = MatchLogWriter(lambda: log, compact_interval_sec=3600.0, window_bytes=1)
    results = synthetic_journal_results(4 * 1500)
    for k in range(4):
        writer.submit(results[k * 1500:(k + 1) * 1500], token=k)
    writer.start()
    _commits(writer, 4)
    writer.stop(5.0)
    for k in range(4):
        uids = {r.uid for r in results[k * 1500:(k + 1) * 1500]}
        assert sum(1 for b in log.batches if uids & {r.uid for r in b}) == 1
    assert max(len(b) for b in log.batches) <= 1500 + WRITE_BATCH


def test_queued_tokened_results_left_at_the_stop_are_counted_as_dropped():
    class Slow(CommitFakeLog):
        def append(self, results):
            time.sleep(1.5)
            return super().append(results)

    writer = MatchLogWriter(Slow, compact_interval_sec=3600.0, window_bytes=1)
    for k in range(3):
        writer.submit(synthetic_journal_results(WRITE_BATCH, first=k * WRITE_BATCH), token=k)
    writer.start()
    time.sleep(0.3)
    stats = writer.stop(timeout=1.0)
    assert stats.written + stats.dropped == 3 * WRITE_BATCH and stats.dropped >= 2 * WRITE_BATCH
```

Create `tests/unit/test_sp5_hub_registry.py` (verified: 2 passed):
```python
"""hub.json and the running-hubs registry (SP5 T2.5; spec 1, 3.3)."""
from __future__ import annotations

import json
import multiprocessing as mp

from colosseum.hub.registry import (
    HUB_JSON,
    HubInfo,
    read_hub_json,
    register_running,
    running_hubs,
    this_process_info,
    unregister_running,
    write_hub_json,
)


def _info(**changes):
    base = dict(run_id="R", run_dir="/runs/x", addresses=("0.0.0.0:47100",), loopback="127.0.0.1:41234",
                tls=False, ca_fingerprint="f" * 64, identity_dir="/id", mode="train")
    return this_process_info(**{**base, **changes})


def test_hub_json_round_trips_and_a_malformed_one_reads_as_none(tmp_path):
    info = _info()
    write_hub_json(tmp_path, info)
    assert read_hub_json(tmp_path) == info and not info.loopback_tls
    assert _info(mode="hub", tls=True).loopback_tls
    (tmp_path / HUB_JSON).write_text("{")
    assert read_hub_json(tmp_path) is None and read_hub_json(tmp_path / "missing") is None


def _noop():
    pass


def test_the_running_registry_lists_live_hubs_and_removes_stale_entries(tmp_path):
    register_running(tmp_path, _info())
    proc = mp.get_context("spawn").Process(target=_noop)
    proc.start()
    proc.join()
    dead = HubInfo(**{**json.loads(_info().to_json()), "addresses": (), "pid": proc.pid, "proc_start": None})
    register_running(tmp_path, dead)
    (tmp_path / "running" / "999999999.json").write_text("not json")
    assert [h.pid for h in running_hubs(tmp_path)] == [_info().pid]
    assert sorted(p.name for p in (tmp_path / "running").iterdir()) == [f"{_info().pid}.json"]
    unregister_running(tmp_path, _info().pid)
    assert running_hubs(tmp_path) == []
```

Create `tests/unit/test_sp5_hub_lineups.py` (a `Hub` built with `HubOptions(local_node=None)` and the setup steps of `launch` up to the coordinator, through a helper `hub_kit.lineup_hub(config, tmp_path)` that returns the hub with its state objects and a recording `SessionHandle` stand-in; `_on_lineup_request` called directly):
- `test_the_hub_rotates_owners_like_sp4_on_one_node`: config `turns` with agents `a, b, c` (`matchmaking.opponents: {latest: 1.0}` as in `test_owner_rotates_with_global_env_index_and_round`), all three marked collecting (an accepted `VersionInfo` of generation 1 each), one node with 2 workers × 4 envs requesting all 8 cells at once, three rounds; each grant's collecting owner (the only collecting agent of the lineup) equals `trainable[(w * 4 + e + round) % 3]`.
- `test_lineups_come_from_the_coordinators_matchmaker`: with `opponents: {latest: 0.5, anchors: 0.5}` and a scripted anchor, every granted lineup passes `check_lineup` and both categories appear over 200 cells (the matchmaker ran in the hub); with `matchmaking.shuffle_seats: true` both seat orders appear.
- `test_every_grant_carries_its_epoch_and_the_refresh_ttl`: grants have `epoch` = the cell's (`CellTable`), `lineup.epoch` equal to it, `ttl_sec == rollout.match_refresh_interval_sec`.
- `test_a_refresh_of_zero_grants_each_cell_once`: `match_refresh_interval_sec: 0` → `ttl_sec == 0` (the node never asks again).
- `test_no_lineup_is_granted_while_no_agent_can_collect`: no accepted version yet → the reply has no grants and the cells keep their epochs for the next request; after the first accepted version the same request is answered.
- `test_seats_of_agents_without_a_learner_do_not_collect`: agents `a, b`, only `a` collecting, `opponents: {rivals: 1.0}` → `b`'s seats have `collect=False`, `a` owns every cell.
- `test_until_t32_a_node_collects_only_for_its_own_learners`: two nodes, `a`'s learner on node 1; node 2's request → no grants (and a DEBUG line); node 1's → granted.

Create `tests/unit/test_sp5_hub_results.py` (the same helper; a real `MatchLogWriter` under `tmp_path` or the journal off; a recording coordinator wrapper counts `report_match_result` calls per uid; `NodeMessageIn` events built from `hub_kit.make_result`):
- `test_results_are_accounted_once_after_their_commit_and_acked_then`: batch seq 1 with 3 results → nothing accounted before `_drain_commits`; after the writer commits: 3 accounted in order, `Ack(seqs=[1])` sent on the handle, `league_state.json` written at the next tick has `journal_row == 3`.
- `test_a_resent_batch_is_skipped_while_in_flight_and_reacked_after`: seq 1 twice before the commit → submitted once; seq 1 again after the ack → no submit, `Ack([1])` again.
- `test_a_resend_after_a_hub_restart_is_accounted_once`: a second `Hub` on the same journal file (resume not needed: a fresh `ResultIntake`) gets seq 1 again → the writer's commit has `accepted == []`, `duplicates == 3`, nothing accounted, acked.
- `test_a_batch_the_writer_cannot_take_is_neither_accounted_nor_acked`: writer `queue_size` 2 (monkeypatched `WRITER_QUEUE_SIZE`), a batch of 3 → no ack, `intake.failed`; the same seq resent after room exists → accounted once, acked.
- `test_a_dead_writer_switches_to_accounting_at_intake`: a writer whose factory raises → the first batch's failed commit; its resend is accounted at intake, acked at once, and `dropped_match_results_by_reason["writer"]` (a hub counter, exposed in `system` by T7.1; here read from `hub._drops`) counts it; one ERROR line.
- `test_without_the_journal_results_are_accounted_at_intake`: `match_log.enabled: false` → accounted and acked in the same pass, dedup by seq only.
- `test_invalid_results_are_dropped_counted_and_warned_once_per_node`: a batch with one result of an unknown layout and one of another node's uid → the valid rest is submitted; `hub._drops == {"invalid": 2}`; one WARNING per node and reason over two such batches.
- `test_results_are_stamped_with_the_hubs_env_steps_at_intake`: `budget.report(...)` to 1234 before the batch → every journaled result has `env_steps == 1234`; a result arriving with `env_steps` set keeps it (SP4 ruling 18).
- `test_heartbeats_drive_the_budget_and_the_stop`: heartbeats of two sessions (cumulative 600 and 500) with `training.total_timesteps: 1000` → `env_steps_done == 1100` and `_main_loop` returns 0 with "Training budget reached: 1100 env steps (budget 1000)" in the log.
- `test_the_league_state_records_the_last_accounted_journal_row`: after two commits (rows 1–3, 4–5) `league_state.json["journal_row"] == 5`; without the journal it is null.

Create `tests/unit/test_sp5_hub_admin.py`:
- `test_resolve_hub_prefers_the_flag_then_the_run_then_the_single_running_hub` (fake identity and registry dirs under `tmp_path`, `fetch_hub_ca` monkeypatched): `--hub ADDR` with a token → that address, TLS, the CA from `fetch_hub_ca`; `--run RUN` → `hub.json`'s `loopback` (plain for `mode == "train"`), the local identity's token; neither with one running hub → that one; none → `SecurityError` "no running hub"; two → `SecurityError` listing both run dirs.
- `test_admin_status_answers_on_a_running_hub`: an in-process `Hub` in `hub` mode on port 0 (`insecure=True`, no nodes, `training.total_timesteps` large, stopped by `request_stop` at the end) → `call(target, "hub.status")` returns `{"code": 0, "data": {"run_id": ..., "env_steps": 0, "nodes": []}}`; an unknown command → code 2 with "unknown command".

Create `tests/integration/test_sp5_train_cli.py` (`cli_runner.run_train` on `configs/examples/tic_tac_toe.yaml` with `TINY`):
- `test_train_runs_a_hub_and_a_local_node_to_the_budget`: exit 0; `logs/` has `main.log`, `node.log`, `worker-0.log`, `learner-agent_0.log`; `hub.json` has `mode == "train"`, `tls is False`, `loopback` `127.0.0.1:<port>` with a non-zero port and no `addresses`; `hub_state.json` has `env_steps >= 3000`, one session, `epoch_reserved_until == 1024`; `matches.db` uids all match `^<run_id>/train/n0/w0_e[0-7]_g\d+_ep\d+$` and are unique; checkpoints carry `env_steps`; the running registry under the redirected data home is empty after the exit; `run_state.json` is `finished` with exit code 0.
- `test_train_listen_opens_a_tls_port_and_keeps_the_loopback`: `train --listen 127.0.0.1 --set distributed.port=0` → exit 0; `hub.json.addresses == ["127.0.0.1:<p>"]` with `tls is True` and a separate plain `loopback`; the local node used the loopback (its log names `127.0.0.1:<loopback port>`).
- `test_sigint_and_sigterm_stop_train_with_final_checkpoints` (port of SP2's CLI signal tests through `cli_runner.run_in_session` with the startup marker): SIGINT after the first checkpoint → exit 130, a final snapshot (`meta.json` `final: true`), no process of the session alive after 10 s; SIGTERM → 143 likewise.
- `test_a_contract_error_on_the_node_stops_train_with_exit_1_and_its_context`: the `ContractBreakingGame` config → exit 1; stderr / `main.log` has one ERROR naming `node 0 (`, `worker 0`, `env`, `episode step`, `layout` and the path of `worker-0.log`.
- `test_a_dead_local_node_fails_the_run`: `COLOSSEUM_TEST_KILL_NODE_AFTER` hook (the local node `os._exit(9)`s after N passes; read only when set) → exit 1 naming `node.log`.

Create `tests/integration/test_sp5_hub_mode.py`:
- `test_hub_trains_with_a_node_that_connects_in_process`: `cli_runner.start_hub(config, tmp_path, port=0, insecure=True)` (a `colosseum hub` process; reads `hub.json` for its address); a `NodeAgent` in a thread of the test (`NodeOptions(hub_address, token from the hub identity under the redirected data home, insecure=True, workers=1, identity_root=tmp_path / "node")`): it registers, gets the learner of `agent_0` (the first node with learner capacity, T2.5's rule) and one worker; the hub process exits 0 at the budget; the node's `run()` returns 0 after `stop_run`; `hub.json.mode == "hub"`; uids carry `n1/`.

Create `tests/integration/test_sp5_two_trains_slow.py` (`@pytest.mark.slow`):
- `test_two_trains_on_one_machine_run_side_by_side`: two `colosseum train` on `tic_tac_toe.yaml` with `rollout.num_workers=1`, `rollout.envs_per_worker=2`, started together with no hub identity yet (both create or read it), each to `training.total_timesteps=2000`: both exit 0, different loopback ports, different run ids, both journals complete, no shm file of either left. (Four lightly loaded processes: proposed ruling 3 of the contract notes.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_writer_commits.py tests/unit/test_sp5_hub_registry.py tests/unit/test_sp5_hub_lineups.py tests/unit/test_sp5_hub_results.py tests/unit/test_sp5_hub_admin.py tests/integration/test_sp5_train_cli.py tests/integration/test_sp5_hub_mode.py -q`
Expected: import errors (`cannot import name 'Commit'`, `No module named 'colosseum.hub.hub'`, ...).

- [ ] **Step 3: The journal writer (`src/colosseum/arena/matchlog.py`)**

`AppendStats` gains `rows: list[int | None] = field(default_factory=list)` (per result: its `matches.id`, None for a duplicate or a result folded into counts); `MatchLog.append` appends `None` at the two `stats.accepted.append(...)` sites of the counts-only and duplicate paths and `match` after `stats.accepted.append(True)` of a detailed one. Add `from typing import Any`. Replace `class MatchLogWriter` (keep `WriterStats`, the constants and everything above it) by (complete; verified as described in the preamble):
```python
@dataclass(frozen=True)
class Commit:
    """What the writer did with one tokened ``submit`` (SP5 R8): the results journaled as new rows (in commit order;
    ``accepted``), how many were duplicates (uid already in the journal), the highest ``matches.id`` of this commit
    (None: nothing detailed), or ``failed`` (a write error, or the writer died before writing it: nothing of the
    token is in the journal by this writer's doing; the caller does not account or ack it, the sender resends)."""

    token: Any
    accepted: list[MatchResult]
    duplicates: int
    last_row: int | None
    failed: bool = False


@dataclass
class _Group:
    """One ``submit``'s results: written in one transaction (a group never spans two batches)."""

    results: list[MatchResult]
    token: Any = None                 # None: an untokened submit (SP4 behaviour: no Commit)


class MatchLogWriter:
    """Journal writer thread (module docstring). ``log_factory()`` opens the thread's own ``MatchLog`` (called in
    the thread: a ``MatchLog`` connection belongs to the thread that opened it); ``table`` (optional) gets every
    journaled result; ``clock`` measures the compaction cadence and the stop deadline (monotonic).

    Thread rules: the thread touches nothing of the caller but its queue, its stats, its in-flight count, the evicted
    entities and the commits (all under its lock) and ``table`` (internally locked). ``submit``, ``take_commits``,
    ``note_evicted`` and ``stop`` are called from one thread (the main loop).

    Submits are groups: a tokened ``submit`` is written in one transaction (with the batch it is part of; a batch
    takes whole groups up to ``WRITE_BATCH`` results, or one larger group); an untokened one is cut into groups of
    ``WRITE_BATCH``. ``queue_size`` bounds the queued results.

    - Without a token (SP4): what does not fit is dropped and counted (``dropped``, one WARNING); no ``Commit``.
    - With a token (SP5, R8): all or nothing: a submit that does not fit, or reaches a stopped or dead writer, is
      refused whole (the return value is its size) and NOT counted as dropped (the caller resends it later or
      accounts it otherwise); after the thread wrote its group, ``take_commits`` returns a ``Commit`` for the token.
      A write error (or the thread's death) gives a failed ``Commit`` (``write_errors`` counts the error; the results
      are not counted as dropped: the sender resends them).

    Accounting: a batch the thread took from the queue is "in flight" (``pending``) until its append returns.
    When ``stop`` gives up waiting for the thread, the in-flight and queued results are counted as dropped and the
    writer is abandoned: whatever the thread does afterwards no longer changes the stats or the commits. The thread's
    append of the in-flight batch may still commit, so the journal can then hold up to ``WRITE_BATCH`` results that
    are also counted as dropped (an over-count, never a loss). When the journal cannot be opened or the thread dies,
    the writer is ``dead``: queued untokened results are counted as dropped, queued tokened groups get failed
    commits, later submits are refused."""

    def __init__(self, log_factory: Callable[[], MatchLog], *, queue_size: int = WRITER_QUEUE_SIZE,
                 compact_interval_sec: float, window_bytes: int, table: ComparisonTable | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._factory = log_factory
        self._groups: queue.Queue[_Group] = queue.Queue()
        self._capacity = int(queue_size)
        self._queued = 0                 # results in queued groups (under the lock)
        self._interval = float(compact_interval_sec)
        self._window = int(window_bytes)
        self._untouched_limit = max(int(window_bytes), UNTOUCHED_WARNING_FLOOR)
        self._table = table
        self._clock = clock
        self._lock = threading.Lock()
        self._stats = WriterStats()
        self._in_flight = 0
        self._commits: list[Commit] = []
        self._abandoned = False      # stop() gave up waiting: the thread's late updates are ignored
        self._activity = "starting"  # what the thread is doing (set by the thread; named when stop() gives up)
        self._dead = False           # the thread cannot write (the journal is unusable or the thread died)
        self._evicted: set[EntityKey] = set()
        self._stop = threading.Event()
        self._deadline = math.inf
        self._thread: threading.Thread | None = None
        self._warned_drop = self._warned_error = self._warned_untouched = False

    @property
    def stats(self) -> WriterStats:
        with self._lock:
            return self._snapshot_locked()

    @property
    def dead(self) -> bool:
        """The journal is unusable (it could not be opened, or the thread died): submits are refused."""
        return self._dead

    def _snapshot_locked(self) -> WriterStats:
        return replace(self._stats, pending=self._queued + self._in_flight)

    def _bump_locked(self, **delta: int) -> None:
        self._stats = replace(self._stats, **{k: getattr(self._stats, k) + v for k, v in delta.items()})

    def _drain_locked(self) -> list[_Group]:
        """Remove every queued group."""
        out = []
        while True:
            try:
                out.append(self._groups.get_nowait())
            except queue.Empty:
                break
        self._queued = 0
        return out

    def _settle_locked(self, groups: Sequence[_Group], *, failed_commits: bool) -> int:
        """Results of ``groups`` that will never be written: untokened ones are counted as dropped (returned);
        tokened ones get failed commits when ``failed_commits`` (else they are counted as dropped too, at stop)."""
        dropped = 0
        for group in groups:
            if group.token is not None and failed_commits:
                self._commits.append(Commit(group.token, [], 0, None, failed=True))
            else:
                dropped += len(group.results)
        if dropped:
            self._bump_locked(dropped=dropped)
        return dropped

    def _thread_update(self, *, values: Mapping | None = None, **delta: int) -> bool:
        """A stats update by the thread; ignored (False) once the writer is abandoned."""
        with self._lock:
            if self._abandoned:
                return False
            self._bump_locked(**delta)
            if values:
                self._stats = replace(self._stats, **values)
            return True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="match-log-writer", daemon=True)
        self._thread.start()

    def submit(self, results: Sequence[MatchResult], *, token: Any = None) -> int:
        """Queue ``results`` without blocking (class docstring); returns how many were not taken."""
        results = list(results)
        if not results:
            return 0
        with self._lock:
            if self._stop.is_set() or self._dead:
                if token is not None:
                    return len(results)
                reason = "is stopped" if self._stop.is_set() else "is not running (the journal is unusable)"
                taken = 0
            elif token is not None:
                if self._queued + len(results) > self._capacity:
                    return len(results)
                taken = len(results)
            else:
                reason = f"queue is full ({self._capacity} results)"
                taken = max(0, min(len(results), self._capacity - self._queued))
            if token is not None:
                self._groups.put_nowait(_Group(results, token))
            else:                                    # no atomicity needed: batches as in SP4
                for start in range(0, taken, WRITE_BATCH):
                    self._groups.put_nowait(_Group(results[start:min(taken, start + WRITE_BATCH)]))
            self._queued += taken
            dropped = len(results) - taken
            if dropped:
                self._bump_locked(dropped=dropped)
        if dropped and not self._warned_drop:
            self._warned_drop = True
            logger.warning(f"Match log writer {reason}; dropped {dropped} match results (counted in "
                           f"dropped_match_results; further drops are only counted)")
        return dropped

    def take_commits(self) -> list[Commit]:
        """The commits of tokened submits since the last call, in commit order (main loop)."""
        with self._lock:
            out, self._commits = self._commits, []
        return out

    def note_evicted(self, entities: Collection[EntityKey]) -> None:
        """Entities of evicted snapshots: their training matches are folded at the next compaction pass."""
        with self._lock:
            self._evicted.update(entities)

    def stop(self, timeout: float) -> WriterStats:
        """Stop compacting, write what is queued, close the journal; within ``timeout`` seconds. What could not be
        written in time (queued, or in flight when the thread does not stop in time) is counted as dropped (one
        WARNING; tokened groups too: the run is over); the returned stats are final. A thread that does not stop in
        time is abandoned; the WARNING names what it was doing. Its in-flight batch may still be committed (an
        over-count of at most ``WRITE_BATCH`` in ``dropped``); stuck in compaction or another journal operation, it
        has no results in flight. Idempotent; harmless without ``start``."""
        timeout = max(0.0, float(timeout))
        self._deadline = self._clock() + max(0.0, timeout - _STOP_MARGIN_SEC)
        self._stop.set()
        alive = False
        if self._thread is not None:
            self._thread.join(timeout)
            alive = self._thread.is_alive()
        with self._lock:
            in_flight = 0
            if alive:
                self._abandoned = True
                in_flight, self._in_flight = self._in_flight, 0
                if in_flight:
                    self._bump_locked(dropped=in_flight)
            left = self._settle_locked(self._drain_locked(), failed_commits=False)
            final = self._snapshot_locked()
        if alive:
            if in_flight:
                logger.warning(f"Match log writer did not stop within {timeout:.1f} s (it was writing a batch of "
                               f"{in_flight} results); they are counted as dropped (dropped_match_results), but the "
                               f"thread may still commit them: an over-count of at most {WRITE_BATCH} results; the "
                               f"journal stays open until the process exits")
            else:
                logger.warning(f"Match log writer did not stop within {timeout:.1f} s (it was {self._activity}; no "
                               f"results in flight, e.g. waiting for the journal's write lock); the journal stays "
                               f"open until the process exits")
        if left:
            logger.warning(f"Match log writer stopped with {left} results not written (counted in "
                           f"dropped_match_results)")
        return final

    # ---- the thread --------------------------------------------------------------------------------------------

    def _take(self, timeout: float) -> list[_Group]:
        """Whole groups up to ``WRITE_BATCH`` results (at least one group), now in flight; waits up to ``timeout``
        for the first."""
        groups: list[_Group] = []
        if timeout > 0:
            try:
                groups.append(self._groups.get(timeout=timeout))
            except queue.Empty:
                return []
        with self._lock:
            if self._abandoned:
                if groups:
                    self._bump_locked(dropped=len(groups[0].results))
                return []
            if not groups:
                try:
                    groups.append(self._groups.get_nowait())
                except queue.Empty:
                    return []
            total = len(groups[0].results)
            while total < WRITE_BATCH:
                try:
                    group = self._groups.get_nowait()
                except queue.Empty:
                    break
                groups.append(group)
                total += len(group.results)
            self._queued -= total
            self._in_flight = total
        return groups

    def _error(self, what: str, error: Exception) -> None:
        """One ERROR per failure; the traceback only with the first."""
        logger.error(f"Match log: {what} ({type(error).__name__}: {error}); training goes on",
                     exc_info=not self._warned_error)
        self._warned_error = True

    def _write(self, log: MatchLog, groups: list[_Group]) -> None:
        batch = [r for g in groups for r in g.results]
        self._activity = f"writing {len(batch)} results"
        try:
            stats = log.append(batch)
        except Exception as e:  # noqa: BLE001 - a journal problem must never stop training
            with self._lock:
                if self._abandoned:
                    return
                self._in_flight = 0
                self._bump_locked(write_errors=1)
                lost = self._settle_locked(groups, failed_commits=True)
            what = (f"writing {len(batch)} results failed; {lost} are counted in dropped_match_results"
                    + (", the others go back to their senders" if lost < len(batch) else ""))
            self._error(what, e)
            return
        commits = []
        offset = 0
        for group in groups:
            n = len(group.results)
            if group.token is not None:
                flags = stats.accepted[offset:offset + n]
                rows = [r for r in stats.rows[offset:offset + n] if r is not None]
                commits.append(Commit(group.token, [r for r, ok in zip(group.results, flags, strict=True) if ok],
                                      flags.count(False), max(rows) if rows else None))
            offset += n
        with self._lock:
            if self._abandoned:                     # counted as dropped by stop()
                return
            self._in_flight = 0
            self._bump_locked(written=stats.detailed + stats.counted, duplicates=stats.duplicates)
            self._stats = replace(self._stats, degraded=stats.degraded)
            self._commits.extend(commits)
        if self._table is None:
            return
        self._activity = "adding journaled results to the live comparison table"
        try:
            for result, accepted in zip(batch, stats.accepted, strict=True):
                if accepted:
                    self._table.add_result(result, log.rating_versions(result))
        except Exception as e:  # noqa: BLE001 - journaled already; only the live ratings miss them
            if self._thread_update(write_errors=1):
                self._error(f"adding {len(batch)} journaled results to the live comparison table failed", e)

    def _compact(self, log: MatchLog) -> None:
        self._activity = "compacting the journal"
        with self._lock:
            evicted, self._evicted = self._evicted, set()
        try:
            result = log.compact(window_bytes=self._window, evicted=evicted, should_stop=self._stop.is_set)
            sizes = log.sizes()
        except Exception as e:  # noqa: BLE001 - a journal problem must never stop training
            with self._lock:
                self._evicted |= evicted
            if self._thread_update(write_errors=1):
                self._error("compaction failed", e)
            return
        if result.interrupted:
            with self._lock:
                self._evicted |= evicted          # the evicted entities not folded yet stay pending
        untouched = sizes.file_bytes - sizes.detail_bytes
        if untouched > self._untouched_limit and not self._warned_untouched:
            self._warned_untouched = True
            logger.warning(f"Match log: the parts compaction does not touch (counts, tournament and pulse matches) "
                           f"take {untouched / 2**20:.0f} MB, more than the detail window; the journal grows past "
                           f"match_log.detail_window_mb")
        self._thread_update(compactions=1, folded=result.folded, evicted_folded=result.evicted_folded,
                            values={"sizes": sizes})

    def _refresh_sizes(self, log: MatchLog) -> None:
        self._activity = "reading the journal sizes"
        try:
            sizes = log.sizes()
        except Exception as e:  # noqa: BLE001 - only the reported sizes are stale
            self._error("reading the journal sizes failed", e)
            return
        self._thread_update(values={"sizes": sizes})

    def _die(self, in_flight: Sequence[_Group] = ()) -> None:
        """The thread cannot write any more: untokened in-flight and queued results are dropped, tokened groups get
        failed commits (their senders resend them to the journal-off path); later submits are refused."""
        with self._lock:
            self._dead = True
            if self._abandoned:
                return
            self._in_flight = 0
            self._bump_locked(write_errors=1)
            self._settle_locked([*in_flight, *self._drain_locked()], failed_commits=True)

    def _loop(self, log: MatchLog) -> None:
        self._refresh_sizes(log)
        last_compact = self._clock()
        while not self._stop.is_set():
            self._activity = "waiting for results"
            groups = self._take(0.2)
            if groups:
                self._current = groups
                self._write(log, groups)
                self._current = []
            if not self._stop.is_set() and self._clock() - last_compact >= self._interval:
                self._compact(log)
                last_compact = self._clock()
        while self._clock() < self._deadline:          # stop(): flush the queue, no more compaction
            groups = self._take(0.0)
            if not groups:
                break
            self._current = groups
            self._write(log, groups)
            self._current = []
        self._refresh_sizes(log)

    def _run(self) -> None:
        self._current: list[_Group] = []
        self._activity = "opening the journal"
        try:
            log = self._factory()
        except Exception as e:  # noqa: BLE001 - reported; results are only counted from now on
            logger.error(f"Match log: opening the journal failed ({type(e).__name__}: {e}); match results are "
                         f"counted as dropped, training goes on", exc_info=True)
            self._die()
            return
        try:
            self._loop(log)
        except Exception as e:  # noqa: BLE001 - reported; results are only counted from now on
            logger.error(f"Match log: the match-log writer thread died ({type(e).__name__}: {e}); match results "
                         f"are counted as dropped from now on, training goes on", exc_info=True)
            self._die(self._current)
        finally:
            self._activity = "closing the journal"
            log.close()
```
Module docstring, the writer paragraph: add "SP5 (R8): a tokened submit is all or nothing and is committed as a whole; the hub accounts and acks a node's batch only from its ``Commit``". In `tests/unit/test_sp4_matchlog_writer.py::test_a_batch_still_in_flight_at_the_stop_deadline_is_counted_as_dropped` replace `writer._queue.qsize() == 2 * WRITE_BATCH` by `writer._queued == 2 * WRITE_BATCH` (the queue holds groups now; the assertion's meaning is unchanged).

- [ ] **Step 4: Coordinator, metrics, security**

- `Coordinator.lineup_for(owner: str) -> Lineup`: `lineup = self._matchmaker.lineup_for(owner)`; `check_lineup(self._context, lineup, type(self._matchmaker).__name__)`; return it. `generate_lineups` stays (tests and T0.1's baselines) and calls `lineup_for` per env.
- `Coordinator.save_league_state(path, *, env_steps, journal_row=None)`: the file gains `"journal_row": journal_row` (`int | None`); `load_league_state` ignores it (T4.3 reads it through `league_state_journal_row()`).
- `metrics/aggregator.py`, `SystemStats`: worker stats are keyed by `(stats.get("node"), int(stats["worker_id"]), int(stats.get("incarnation", 0)))` everywhere `int(stats["worker_id"])` was the key (`_workers`, `_cumulative`); docstring: "keyed by (node, worker, incarnation): counters of every incarnation of every worker of every node add up, so the run-level counters stay monotonic across joins, restarts and resumes (SP5 spec 8)". The SP4 path (no `node`, no `incarnation`) keys `(None, w, 0)` and behaves as before.
- `net/security.py`, `NodeBook.add_ephemeral(name: str, machine_id: str) -> tuple[NodeRecord, str]`: a record with number 0 and a fresh secret, kept in memory only (never written to `nodes.json`, removed when the `NodeBook` is dropped); `verify` accepts it; `is_banned` never matches it. This is how `train`'s local node authenticates on the loopback port (R12: no `Register`, no persistent entry for a node that lives as long as its hub).

- [ ] **Step 5: `src/colosseum/hub/registry.py`**

Complete (verified; note `_root`: a `Path` has a `root` attribute, so `getattr(path, "root", path)` would write to `/`):
```python
"""Where a hub can be found (SP5 spec 1, 3.3): ``runs/<name>/hub.json`` (written at start, kept after the stop) and
the machine's running-hubs registry ``<hub identity>/running/<pid>.json`` (removed at the stop; entries of dead pids
are removed by ``running_hubs``). ``addresses`` are the hub port's listen addresses (``host:port``, TLS unless
``tls`` is False); ``loopback`` is the address a client on this machine uses: ``train``'s plain loopback port, or
``127.0.0.1:<port>`` of a ``hub`` listening on all interfaces (TLS like ``addresses``), None when the hub listens only
on another address."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from colosseum.core.run_state import RunState, process_alive, process_start_ticks
from colosseum.utils.fs import write_text_atomic

HUB_JSON = "hub.json"
RUNNING_DIR = "running"


@dataclass(frozen=True)
class HubInfo:
    run_id: str
    run_dir: str
    pid: int
    proc_start: int | None
    addresses: tuple[str, ...]
    loopback: str | None
    tls: bool
    ca_fingerprint: str
    identity_dir: str
    mode: str                      # "train" | "hub" | "idle"

    @property
    def loopback_tls(self) -> bool:
        """``train``'s loopback port is plain; a ``hub``'s loopback address is its TLS port (unless insecure)."""
        return self.tls and self.mode != "train"

    def to_json(self) -> str:
        d = asdict(self)
        d["addresses"] = list(self.addresses)
        return json.dumps(d, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> HubInfo:
        return cls(run_id=str(d["run_id"]), run_dir=str(d["run_dir"]), pid=int(d["pid"]),
                   proc_start=None if d.get("proc_start") is None else int(d["proc_start"]),
                   addresses=tuple(d.get("addresses", ())), loopback=d.get("loopback"), tls=bool(d["tls"]),
                   ca_fingerprint=str(d["ca_fingerprint"]), identity_dir=str(d["identity_dir"]), mode=str(d["mode"]))


def _read(path: Path) -> HubInfo | None:
    try:
        return HubInfo.from_dict(json.loads(path.read_text()))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _root(run_dir) -> Path:
    """A ``RunDir``'s root, or a path (``Path.root`` is "/": never ``getattr(path, "root")``)."""
    return Path(run_dir) if isinstance(run_dir, (str, os.PathLike)) else Path(run_dir.root)


def write_hub_json(run_dir, info: HubInfo) -> Path:
    return write_text_atomic(_root(run_dir) / HUB_JSON, info.to_json())


def read_hub_json(run_dir) -> HubInfo | None:
    """The run's ``hub.json`` (None when missing or malformed)."""
    return _read(_root(run_dir) / HUB_JSON)


def alive(info: HubInfo) -> bool:
    return process_alive(RunState(status="running", pid=info.pid, proc_start=info.proc_start))


def register_running(identity_dir: Path, info: HubInfo) -> Path:
    d = Path(identity_dir) / RUNNING_DIR
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return write_text_atomic(d / f"{info.pid}.json", info.to_json())


def unregister_running(identity_dir: Path, pid: int) -> None:
    (Path(identity_dir) / RUNNING_DIR / f"{int(pid)}.json").unlink(missing_ok=True)


def running_hubs(identity_dir: Path) -> list[HubInfo]:
    """The live hubs of this machine (by pid and process start); stale and malformed entries are removed."""
    d = Path(identity_dir) / RUNNING_DIR
    out = []
    for path in sorted(d.glob("*.json")) if d.is_dir() else []:
        info = _read(path)
        if info is None or not alive(info):
            path.unlink(missing_ok=True)
            continue
        out.append(info)
    return out


def this_process_info(**fields) -> HubInfo:
    """A ``HubInfo`` of the calling process (pid and start ticks filled in)."""
    pid = os.getpid()
    return HubInfo(pid=pid, proc_start=process_start_ticks(pid), **fields)
```

- [ ] **Step 6: `src/colosseum/hub/hub.py`**

Module docstring: the hub process (spec block 1, R5): the old `Launcher` with its inputs from node sessions; the main loop of "Threads in the hub" (overview). Code moved from `launcher.py` keeps its docstrings; every reference to queues in them is updated.

Constants: `INBOX_POLL_SEC = 0.05`, `LOCAL_SHUTDOWN_GRACE_SEC = SHUTDOWN_GRACE_SEC`, `ANNOUNCE_SEC` = `distributed.heartbeat_sec` (read from the config: the cadence of `Announce(env_steps, versions, evictions)`).

`HubOptions`, `LocalNodeOptions` as in the contract (`LocalNodeOptions.from_config(config, run_dir)`: `workers=rollout.num_workers`, `evaluators=0` (the SP4 pulse child does the pulse until T5.1; the contract's "1 with evaluation.enabled" applies from T5.1), `log_dir=run_dir.logs`, `data_port=False`).

`Hub.launch()` (the order matters: everything that can fail with a `ConfigError` runs before any child or port exists, as in SP4):
```python
    def launch(self) -> int:
        cfg = self._config
        self._run_id = self._run_dir.ensure_run_id()
        ...                                                    # the Launcher's INFO lines, setup_run, skew warning
        coordinator = self._build_coordinator(setup)
        resume_states = self._resolve_resume(setup.agent_configs, setup.role_specs)
        self._load_league_state(coordinator)
        self._import_snapshot_pool(coordinator)
        self._teachers = {aid: resolve_teacher(cfg, aid, setup.spec) for aid in trainable}
        self._init_states = self._resolve_init(setup.spec, resume_states)
        self._checkpoint_meta = {...}                          # as the Launcher
        self._identity = HubIdentity.load_or_create(self._options.identity_dir)
        self._blobs = BlobStore(self._run_dir)
        self._register_blobs(setup, resume_states)             # Welcome.fixed and the learners' resume blobs
        self._pulse_setup()                                    # R6: plan, feeder, the two mp.Queues
        code = 1
        try:
            self._supervisor.install_signal_handlers()        # ProcessSupervisor(mp.Event): the local node + pulse
            self._open_metrics(resume_states)
            self._open_match_log(coordinator)
            self._open_live_ratings()
            self._wire_retention()
            self._start_server()                               # build_server + start; hub.json; running registry
            started = False
            try:
                self._start_local_node()                       # R12 (train only)
                self._start_pulse()                            # R6
                started = True
                logger.info("Training started. Press Ctrl+C to stop.")
                code = self._main_loop()
            except BaseException as error:
                if not started:
                    logger.error("Starting the local node or the pulse failed; stopping what started")
                self._shutdown_after_error(error)
                raise
            killed = self._shutdown()
            code = self._exit_code_after_shutdown(code, killed)
        finally:
            try:
                self._finish_metrics()
            finally:
                try:
                    self._stop_server()                        # server.stop(1.0); unregister_running; hub.json kept
                finally:
                    try:
                        self._supervisor.restore_signal_handlers()
                    finally:
                        self._release_pulse_queues()
        return code
```

`_register_blobs`: for every frozen agent of `setup.fixed.frozen`: `frame = encode_weights(frozen.model_state, WeightsMeta(agent_id, "fixed", 0, 0, architecture=frozen.networks, roles=frozen.roles))`, `ident = blob_ident("fixed", agent_id, frame)`, `self._blobs.register_fixed(ident, frame, kind="fixed", agent_id=agent_id)`, `self._fixed_blobs.append(pb.FixedBlob(agent_id=agent_id, blob_id=ident, kind="fixed"))`; a neural kickstart teacher of agent X: kind `"teacher"`, `agent_id = X` (the student), architecture / roles of the teacher's `FrozenSpec`; an `init` state of X: kind `"init"`, `policy_version 0`; a resume state of X: kind `"recovery"` frame of its model state with `policy_version` = the resumed version, and its trainer-state bytes (when present) as kind `"trainer_state"` (`.bin`), both referenced by X's `LearnerAssignment.weights_blob` / `trainer_state_blob`.

`_start_server`: `listen` and modes as in the overview ("Ports"): `train` without `--listen` → `build_server(listen=[], loopback_plain=True)`; `train --listen ADDR` → `listen=[f"{ADDR}:{distributed.port}"]` + `loopback_plain=True`; `hub` → `listen=[f"{options.listen or '[::]'}:{options.port if options.port is not None else distributed.port}"]`, `loopback_plain=False`; `insecure` from the options / `distributed.insecure`. Then `server.start()`, `write_hub_json`, `register_running(identity.path, info)`, one INFO line per address (`Hub listening on 127.0.0.1:41234 (loopback, plain)`); T6.3 prints the join commands.

`_start_local_node` (train): `record, secret = self._identity.nodes.add_ephemeral("local", machine_id())`; `NodeOptions(hub_address=f"127.0.0.1:{ports[LOOPBACK_KEY]}", token=self._identity.token, loopback_plain=True, workers=..., evaluators=0, local=True, credentials=(record.node_id, 0, secret), log_dir=self._run_dir.logs)`; `mp.Process(target=local_node_target, name="node", kwargs={"options": options, "log_dir": str(self._run_dir.logs)}, daemon=False)` (the node starts its own children), `start_process`, `self._supervisor.add("node", proc)`.

`_main_loop`:
```python
    def _main_loop(self) -> int:
        total = self._config.training.total_timesteps
        while True:
            self._drain_inbox(INBOX_POLL_SEC)      # handlers: sessions, messages, uploads, register, admin, stop
            self._drain_commits()                  # R8: commits -> coordinator, metrics hub, acks
            self._reapply_retention()
            self._pump_pulse()
            now = time.monotonic()
            for entry in self._registry.lost(now, self._dist.node_lost_after_sec):
                self._on_node_lost(entry)          # cells dropped (CellTable.drop_node), WARNING, nodes_lost += 1
            if now >= self._next_announce:
                self._broadcast_announce()         # env_steps, new versions, evictions (coordinator.take_evictions)
                self._next_announce = now + self._dist.heartbeat_sec
            env_steps = self._budget.value
            if self._hub.maybe_tick(env_steps=env_steps, ratings=self._coordinator.ratings_snapshot,
                                    queue_depths=dict(self._queue_depths), match_log=self._match_log_stats(),
                                    bt=self._live_snapshot()):
                self._save_league_state()
                self._save_hub_state(fsync=False)
            if self._fatal is not None:            # a child_error / learner_failed event, a dead local node
                logger.error(self._fatal)
                return 1
            if self._supervisor.received_signal is not None:
                return self._signal_exit_code()
            if self._stop_requested.is_set():
                logger.info(f"Stop requested after {env_steps} env steps (budget {total})")
                return 0
            if env_steps >= total:
                logger.info(f"Training budget reached: {env_steps} env steps (budget {total})")
                return 0
            failure = self._supervisor.first_failure()          # the local node or the pulse child died
            if failure is not None:
                logger.error(failure.message())
                self._reported_failures.add(failure.name)
                return 1
```

Handlers (each catches and logs its own exceptions with one ERROR per kind; nothing a node sends stops the hub):
- `_on_session_opened(ev)`: `node_id` → its record (`NodeBook`, the ephemeral local one included) → `number`, `name`; `kind = self._registry.connect(node_id, number, name, session_id, hello_from_proto(hello), now, data_path="local" if number == 0 else "relay")`; `"returned"` → `self._cells.drop_node(number)`; `self._intake.open(node_id, session_id, hello.last_acked_seq)`; `self._sessions[node_id] = handle`; learners: the local node gets every trainable agent with `generation = self._versions.new_generation(agent)` on its first session (later sessions of the same node keep them: `Hello.learners`); in `hub` mode the first node whose slots allow learners (`not no_learners`, `cpu_learners != 0 or gpus`) gets them all (T4.1 replaces this); `handle.send(HubMessage(welcome=Welcome(node_number, run_id, config_json=self._config.model_dump_json(by_alias=True), agents=[AgentState(aid, kind, "active", 0) ...], versions=[version_to_proto(v) ...], evictions=[], learners=[LearnerAssignment(agent, generation, version_floor=self._versions.version_floor(agent), device=learner.device, weights_blob, trainer_state_blob, env_steps=self._budget.value) for the agents placed on this node], evaluators=0, env_steps=self._budget.value, fixed=self._fixed_blobs, collect_allowed=True)))`.
- `_on_session_closed(ev)`: `self._registry.disconnect(...)`, drop the handle (the node stays online until `node_lost_after_sec`).
- `_on_message(ev)` dispatches on the oneof: `heartbeat` → `self._registry.heartbeat(...)` (False: ignored), `self._budget.report(node_id, session_id, hb.env_steps)`, the node's `counters` kept for `system` (T7.1); `lineup_request` → `_on_lineup_request`; `results` → `_on_result_batch`; `metrics` → `_on_metrics_batch` (`train` → `self._hub.on_train_metrics({**values, "agent_id", "train_step"})`, `workers` → `self._hub.on_worker_stats({"node": number, "worker_id": w, "incarnation": i, **counters})`, `intake` → `self._queue_depths[agent] = depth`); `version` → `_on_version` (`self._versions.announce(version_from_proto(info))`: `accepted` → the agent can collect; anything else → a WARNING once per (node, reason)); `event` → `_on_event` (`child_error` → `self._fatal = f"{error_kind} error on {context} (log: {log_path})"`; `learner_failed` → `self._fatal` likewise (until T4.2); `proc_died` → WARNING + `worker_restarts += 1`; `slot_disabled` → WARNING; `run_stopped` → noted for the shutdown; `warning` → WARNING with the node's name); `hello` inside a stream → WARNING, ignored.
- `_on_lineup_request(node_id, session_id, req)`:
```python
    def _on_lineup_request(self, node_id: str, session_id: str, req) -> None:
        entry = self._registry.by_id(node_id)
        handle = self._sessions.get(node_id)
        if entry is None or handle is None or entry.session_id != session_id:
            return
        now = time.monotonic()
        requested = [(CellKey(entry.number, c.worker, c.env), int(c.epoch), int(c.incarnation)) for c in req.cells]
        cells = self._cells.request(entry.number, requested, now)
        collecting = self._collecting_on(entry.number)       # accepted latest, learner alive (T2.5: on this node)
        if not collecting:
            logger.debug(f"node {entry.number}: no agent can collect; {len(cells)} cells wait")
            return
        keys = [c.key for c in cells]
        owners = self._rotation.assign(keys, collecting=collecting, owners=self._cells.owners(exclude=keys),
                                       previous={c.key: c.previous for c in cells})
        ttl = float(self._config.rollout.match_refresh_interval_sec)
        grants = []
        for cell in cells:
            lineup = restrict_collecting(self._coordinator.lineup_for(owners[cell.key]), collecting)
            lineup = dataclasses.replace(lineup, epoch=cell.epoch)
            self._cells.grant(cell.key, owners[cell.key], ttl, now)
            grants.append(pb.Grant(worker=cell.key.worker, env=cell.key.env, epoch=cell.epoch, ttl_sec=ttl,
                                   lineup=lineup_to_proto(lineup)))
        handle.send(pb.HubMessage(grants=pb.LineupGrants(grants=grants)))
```
- `_on_result_batch(ev)` and `_drain_commits()` (R8):
```python
    def _on_result_batch(self, ev: NodeMessageIn) -> None:
        entry = self._registry.by_id(ev.node_id)
        seq = int(ev.message.results.seq)
        if entry is None:
            return
        if not self._intake.begin(ev.node_id, ev.session_id, seq):
            if self._intake.is_acked(ev.node_id, ev.session_id, seq):
                self._ack(ev.node_id, seq)                      # the node missed the ack
            return
        if ev.results is None:                                  # undecodable: drop it whole, ack (no poison resend)
            self._drop(entry.number, "invalid", len(ev.message.results.results), ev.error)
            self._settled(ev.node_id, ev.session_id, seq)
            return
        prefix = f"{self._run_id}/train/n{entry.number}/"
        valid = []
        for result in ev.results:
            reason = self._intake.check(result, spec=self._setup.spec, agents=self._known_agents,
                                        versions=self._versions, node=entry.number, uid_prefix=prefix)
            if reason is not None:
                self._drop(entry.number, reason, 1)
                continue
            if result.env_steps is None:
                result.env_steps = self._budget.value           # stamped at intake (spec 4.3)
            valid.append(result)
        self._results_received += len(valid)
        if not valid:
            self._settled(ev.node_id, ev.session_id, seq)
            return
        writer = self._match_log_writer
        if writer is None or writer.dead:
            if writer is not None:                              # journal unusable: accounted, counted as not journaled
                self._drop_writer(len(valid))
            self._account(valid)
            self._settled(ev.node_id, ev.session_id, seq)
            return
        if writer.submit(valid, token=(ev.node_id, ev.session_id, seq)):
            self._intake.failed(ev.node_id, ev.session_id, seq)  # no room: the node resends it

    def _drain_commits(self) -> None:
        if self._match_log_writer is None:
            return
        for commit in self._match_log_writer.take_commits():
            node_id, session_id, seq = commit.token
            if commit.failed:
                self._intake.failed(node_id, session_id, seq)    # resent; a dead writer then takes the intake path
                continue
            self._account(commit.accepted)                       # duplicates (uid already journaled) are not
            if commit.last_row is not None:
                self._journal_row = max(self._journal_row or 0, commit.last_row)
            self._settled(node_id, session_id, seq)

    def _account(self, results) -> None:
        for result in results:
            self._coordinator.report_match_result(result)
            self._hub.on_match_result(result)

    def _settled(self, node_id: str, session_id: str, seq: int) -> None:
        self._intake.committed(node_id, session_id, seq)
        self._ack(node_id, seq)
```
(`_drop(node, reason, n, detail=None)`: `self._drops[reason] += n`; `ResultIntake.count_drop` → one WARNING per (node, reason) naming the node and the first detail; `_drop_writer(n)` counts `self._drops["writer"]` with one ERROR; `_ack(node_id, seq)` sends `Ack(seqs=[seq])` on the node's live handle, if any; `_known_agents` = every agent id of the config.)
- `_on_upload(ev)`: kinds `checkpoint` / `final`: read the spooled frame, `decode_checkpoint(..., cap=payload_byte_cap(max_message_mb))`, check `payload["agent_id"] == header.agent_id`, the generation against `self._versions.generation(agent)` (T2.5: must equal it, else `accepted=False, reason="stale_generation"`), then `checkpoint_id = self._save_checkpoint(payload)` (the Launcher's method: `env_steps` stamped from the budget, R9; the pulse feeder gets the snapshot) and reply `accepted=True, checkpoint_id`; a decode / save error → `accepted=False, reason=...` with one ERROR. The spooled file is removed in every case.
- `_on_register(ev)`: `NodeBook.register(machine_id, name, tags, node_id or None)` unless banned (`SecurityError` → the future's exception), `identity.issue_node_certificate(node_id, public_key_pem)`, reply `RegisterReply(...)` (T6.3 adds the version / platform checks before it).
- `_on_admin(ev)`: `self._admin_handlers.get(command)` → `AdminReply(code=0, data_json=json.dumps(handler(args)))`; unknown → `code=2, text="unknown command ..."`; a handler exception → `code=1` with its text. `hub.status` → `{"run_id", "mode", "env_steps", "nodes": [{"number", "name", "state"} ...]}`.
- `_reserve_epochs(until)` (R7): `self._save_hub_state(fsync=True, epoch_reserved_until=until)`; `_save_hub_state` writes `HubStateData(version=HUB_STATE_VERSION, run_id, env_steps=budget.value, sessions=budget.state()["sessions"], epoch_reserved_until=self._epochs.reserved_until (or the new until), versions=self._versions.state(), counters={"worker_restarts", "learner_restarts", "lineup_fetch_failed", "relay_bytes": 0, "dropped_chunks": {}, "dropped_match_results": dict(self._drops)}, nodes_lost)`; a failing tick write is one ERROR (never a crash), a failing reservation propagates (no epoch is granted unreserved; the handler logs it and grants nothing).
- `_broadcast_announce()`: `Announce(versions=<accepted since the last announce>, evictions=<coordinator.take_evictions() as Eviction>, env_steps=budget.value, collect_allowed=True)` to every live handle.
- `request_stop()`: sets `self._stop_requested` and puts `StopRequested()` on the inbox (thread-safe).

`_shutdown()`: `deadline = now + (LOCAL_SHUTDOWN_GRACE_SEC if train else distributed.shutdown_grace_sec)`; set the pulse `stop_event`; send `Command(kind="stop_run")` to every live session; keep draining the inbox and the commits (finals arrive as uploads, last results as batches) until every node that had a session sent `run_stopped` (train: the local node process exited) or the deadline; then `self._supervisor.kill_remaining()` (the local node and the pulse child), the pulse queues' `_finish_reads` as in SP4; returns the killed names. `_shutdown_after_error` as in SP4.

`run_training(config_path, overrides=None, *, listen=None) -> int`: SP4's body (`spawn`, logging, `load_config`, `validate_run_config` before the run dir, the resume source check, `apply_global_seed`, `RunDir.create`, resolved config, `run_state.json`) with `Hub(config, run_dir, options=HubOptions(mode="train", listen=listen, local_node=LocalNodeOptions.from_config(config, run_dir)), validated=True).launch()`. `run_hub(config_path, overrides=None, *, listen=None, port=None, insecure=False) -> int`: the same with `HubOptions(mode="hub", listen=listen, port=port, insecure=insecure, local_node=None)`.

Moved unchanged: `RunSetup`, `setup_run`, `validate_run_config`, `warn_static_ownership_skew`; `_create_env` becomes `colosseum.node.worker_main.create_env` (T2.4's copy) and `distributed.py` imports it from there.

- [ ] **Step 7: `src/colosseum/hub/admin.py`**

`HubTarget(address, tls, ca_pem, token)` (frozen dataclass); `ADMIN_COMMANDS` (contract) + `"hub.status"`; `resolve_hub(*, hub=None, run=None, token=None, token_file=None, insecure=False) -> HubTarget` exactly as the contract's comment and the test of Step 1 (`--hub ADDR`: the token from `resolve_token` or, when none is given and the machine's hub identity has the address's fingerprint, the local token; TLS unless `insecure`, `ca_pem = fetch_hub_ca(host, port, fingerprint=token.ca_fingerprint, timeout=connect_timeout)`; `--run RUN`: `read_hub_json` (missing: `SecurityError` "RUN has no hub.json: it never ran a hub"), the address = `loopback` or `addresses[0]`, TLS = `loopback_tls` / `tls`, the identity of `identity_dir` (`HubIdentity.load_or_create(Path(info.identity_dir))` — it exists) for token and CA; neither: `running_hubs(hub_identity_dir())`). `call(target, command, args=None, *, timeout=30.0) -> dict` (`{"code", "text", "data"}`), `stream(target, command, args=None) -> Iterator[dict]`: a channel per call (`grpc.secure_channel` with `hub_channel_credentials(ca_pem)` and the server name override, or insecure), metadata `(MD_TOKEN, token.secret)`.

- [ ] **Step 8: The CLI, `cli_runner`, the bench**

- `cli.py`, `train`: `--listen` with an optional value (`@click.option("--listen", "listen", is_flag=False, flag_value="0.0.0.0", default=None, metavar="[ADDR]", help="Also accept other machines' nodes on ADDR:distributed.port (TLS); without ADDR all interfaces")`); calls `colosseum.hub.hub.run_training(config, overrides=..., listen=listen)`. Help text: "Train the config's agents: a hub and a local node on this machine (other machines join with --listen)."
- `cli.py`, `hub`: `@main.group("hub", invoke_without_command=True)` with `--config/-c` (required when no subcommand), `--listen ADDR`, `--port P` (`IntRange(0, 65535)`), `--insecure`, `--set`; without a subcommand → `limit_main_blas_threads()`, `_config_errors()`, `sys.exit(run_hub(...))`; a missing `-c` → `click.UsageError` (exit 2). Help: "Run a hub that machines join (colosseum join, T6.3); exit 0 at the budget, 1 on errors, 130 / 143 on signals."
- `tests/cli_runner.py`: `hub_cmd(config, run_parent, name, overrides=None, *, port=0, insecure=True, listen="127.0.0.1")` and `start_hub(...) -> (Popen, run_root)` (a `run_in_session`-style process that the caller waits for; `wait_for_hub_json(run_root, timeout)` returns the `HubInfo`).
- `scripts/bench_throughput.py`: `run_one` builds `Hub(config, run_dir, options=HubOptions(mode="train", local_node=LocalNodeOptions.from_config(config, run_dir)))`, samples `hub.env_steps_done`, stops with `hub.request_stop()` after the window; when the measured tree has no `colosseum.hub.hub` (`--tree` of `main`), it falls back to `colosseum.launcher.Launcher` with SP4's sampling and `_stop_event.set`. `_make_config` gains `"distributed": {"heartbeat_sec": 0.25}` (the counter quantization stays below 1 % of the 60 s window); `tests/unit/test_bench_throughput.py` follows.

- [ ] **Step 9: Run the (a) tests; full fast suite + ruff; commit and push**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_writer_commits.py tests/unit/test_sp5_hub_registry.py tests/unit/test_sp5_hub_lineups.py tests/unit/test_sp5_hub_results.py tests/unit/test_sp5_hub_admin.py tests/integration/test_sp5_train_cli.py tests/integration/test_sp5_hub_mode.py tests/unit/test_sp4_matchlog_writer.py tests/unit/test_bench_throughput.py -q`; then the full fast suite (the `Launcher` tests still pass: `launcher.py` is untouched except the writer it shares) and ruff.
```bash
git add src/colosseum/hub/ src/colosseum/arena/matchlog.py src/colosseum/coordinator/coordinator.py \
        src/colosseum/metrics/aggregator.py src/colosseum/net/security.py src/colosseum/cli.py src/colosseum/distributed.py \
        scripts/bench_throughput.py tests/cli_runner.py tests/hub_kit.py tests/unit/test_sp5_*.py \
        tests/integration/test_sp5_train_cli.py tests/integration/test_sp5_hub_mode.py \
        tests/integration/test_sp5_two_trains_slow.py tests/unit/test_sp4_matchlog_writer.py tests/unit/test_bench_throughput.py
git commit -m "feat: the hub (evolved launcher) with a local node; colosseum train and colosseum hub on it; R8 accounting"
git push
```
**(End of dispatch (a).)**

- [ ] **Step 10: Port the tests that use `colosseum.launcher`** (b)

Rules (overview, "T2.5 switch rules"): same assertions on the `Hub` equivalents; inbox events instead of queue items; a test is deleted only when the spec removes its subject; the task report lists every ported and deleted test. Helpers added to `tests/hub_kit.py` for the ports: `make_hub(config, run_dir, **options) -> Hub` (train options), `hub_with_coordinator(config, tmp_path)` (a `Hub` whose `_coordinator`, `_budget`, `_hub` (metrics) and `_setup` are set as `launch` sets them, without starting anything), `result_event(hub, results, *, seq, node_id="local")` (a `NodeMessageIn` for `_on_result_batch`), `upload_event(hub, payload, kind)` (a spooled `UploadIn`).

| File | What changes |
|---|---|
| `tests/unit/test_sp2_launcher_lifecycle.py` | `Launcher(...).launch()` → `make_hub(...).launch()`; `_stop_event.set` → `request_stop`; the failed-start, signal-race, failing-teardown, nonzero-child and metrics-closing tests keep their assertions (`_supervisor.names == ["node"]`, plus `"pulse"` where enabled); **deleted**: `test_launch_closes_and_releases_its_queues`, `test_launch_leaves_no_queue_feeder_thread_running` (their subject, the run's `mp.Queue`s, is gone; replaced by `test_sp5_train_cli.py`'s "no process alive" checks), `test_drain_skips_checkpoint_queue_of_a_learner_killed_mid_message`, `test_slow_read_from_an_exited_learner_is_kept_until_the_deadline` (checkpoints arrive by upload; the killed-mid-frame case is `test_sp5_local_connections.py`) |
| `tests/unit/test_sp2_launcher_checkpoints.py` | resume tests on `hub._resolve_resume` / `env_steps_done` (the budget base); `_save_checkpoint` tests unchanged on the hub; `test_shutdown_saves_checkpoints_still_queued` → finals arrive as uploads during `_shutdown` (`upload_event`); **deleted**: `test_missing_checkpoint_falls_back_to_latest_and_collects`, `test_resolve_lineups_two_agents`, `test_refresh_marks_checkpoints_sent_only_after_successful_put` (spec 4.2 removes the fallback and the per-worker deltas), `test_drain_saves_remaining_payloads_after_a_failed_save` → ported to two uploads, the first failing to save (the second still saved, the first replied `accepted=False`) |
| `tests/contract/test_sp4_result_flow.py` | `test_the_launcher_flattens_batches_...` → `test_the_hub_accounts_each_result_once_after_its_commit` (results through `_on_result_batch` + `_drain_commits`: each to the coordinator, the metrics hub and the journal once); `test_a_results_queue_item_that_is_not_a_batch_...` → an undecodable batch is counted `invalid` and acked (no crash); the worker-side tests stay on `rollout_worker_process` (it lives until T7.2) |
| `tests/contract/test_sp3_snapshot_eviction.py` | `test_the_launcher_sends_evictions_to_every_worker_...` → the hub's `Announce.evictions` reach every session (`_broadcast_announce`), the node's `WorkerEvict` is `test_sp5_node_agent.py`'s |
| `tests/contract/test_sp3_fixed_players_wiring.py` | `_resolve_lineups` keeps fixed seats → `_on_lineup_request` keeps fixed seats and sources; `setup_run` from `colosseum.hub.hub` |
| `tests/contract/test_sp4_first_weights.py` | `test_the_launcher_worker_waits_for_the_first_weights` → the node worker body passes `wait_for_first_weights` (monkeypatched `RolloutLoop` in `node.worker_main`); the distributed one stays until T7.2 |
| `tests/integration/test_sp2_budget_stop.py`, `test_sp2_pipelines.py`, `test_sp2_metrics_outputs.py`, `test_sp3_fixed_players_worker.py` | `Launcher(config, run).launch()` → `make_hub(config, run).launch()` (same assertions; logs now `node.log` + children) |
| `tests/integration/test_sp4_result_burst.py` | the queue-burst test stays on `ResultBatcher` (worker side); the launcher half → `_on_result_batch` with 2 000 results in 8 batches: none dropped |
| `tests/unit/test_sp2_entry_points.py`, `test_sp4_run_state.py` | monkeypatch `colosseum.hub.hub.Hub` instead of `launcher.Launcher`; `run_training` from `colosseum.hub.hub` |
| `tests/unit/test_sp2_learner_entry.py`, `test_sp3_learner_factory.py`, `test_sp3_init.py` | `_learner_main` / `_learner_target` → `colosseum.node.learner_main._learner_main` (its kwargs: Step 8 of T2.4); "the launcher passes X to learners / workers" → the node's spawn kwargs carry X (the node built in a test with `node_kit`'s fakes, `ChildSupervisor.start` monkeypatched to record specs) |
| `tests/unit/test_sp3_coordinator_league.py`, `test_sp3_snapshot_retention.py`, `test_sp3_sp2_compat.py`, `test_sp4_live_ratings.py`, `test_sp4_pulse.py`, `test_sp4_snapshot_store.py`, `test_wandb_logger.py`, `test_sp2_run_dir.py` | `Launcher` → `hub_with_coordinator` / `Hub`; the same method names (R5); `_env_step_counter.add(n)` → `_budget.add(n)`; `_pulse_requests` / `_pulse_results` unchanged (R6) |
| `tests/unit/test_ipc_queue_helpers.py` | `test_helpers_moved_out_of_the_launcher` → asserts `colosseum.launcher` does not exist and the helpers are in `core.ipc` |
| T0.1's baselines (`tests/contract/test_sp5_*` of part 01) | those that drive `Launcher` internals are rewritten against the hub with the same assertions (rotation through `_on_lineup_request`, result flow through `_on_result_batch`) |

Then delete `src/colosseum/launcher.py`; `grep -rn "colosseum.launcher\|from colosseum import launcher" src tests scripts` finds only the bench's documented `--tree` fallback.

- [ ] **Step 11: Full fast suite + ruff; slow suite; GPU suite** (b)

The full fast suite (zero warnings) and ruff; then, alone on a quiet machine, in the background: `env -u PYTHONPATH .venv/bin/python -m pytest -m slow -v` (every SP2–SP4 threshold holds on `train` = hub + local node; the new two-trains test passes) and `env -u PYTHONPATH .venv/bin/python -m pytest -m gpu -q -W error` (CUDA learners on the local node).

- [ ] **Step 12: Criterion 2** (b; the controller runs it on a quiet machine)

Thresholds fixed before measuring (spec §3 criterion 2, SP4's rule 3.6): `train` on this branch ≥ 95 % of `main` `d61eb02` at 1, 2 and 4 workers, env steps per second by counters, two runs per variant, alternating. Commands (a `main` worktree at `d61eb02` in `../main`):
```bash
env -u PYTHONPATH PYTHONPATH=../main/src .venv/bin/python scripts/bench_throughput.py --tree ../main --workers 1 2 4 --json docs/benchmarks/sp5-c2-main-1.json
env -u PYTHONPATH .venv/bin/python scripts/bench_throughput.py --workers 1 2 4 --json docs/benchmarks/sp5-c2-branch-1.json
env -u PYTHONPATH PYTHONPATH=../main/src .venv/bin/python scripts/bench_throughput.py --tree ../main --workers 1 2 4 --json docs/benchmarks/sp5-c2-main-2.json
env -u PYTHONPATH .venv/bin/python scripts/bench_throughput.py --workers 1 2 4 --json docs/benchmarks/sp5-c2-branch-2.json
```
Report per worker count the four rates, the ratio of the means, and the bench's bound classification. If a ratio is below 0.95 the task reports a profile of the hub and the local node (`py-spy dump` of both under load, or `cProfile` of the hub main loop for 30 s) and the controller decides; nothing is tuned silently.

- [ ] **Step 13: Commit and push** (b)

```bash
git add -A src/colosseum tests scripts docs/benchmarks
git commit -m "refactor: launcher.py removed; every SP1-SP4 test runs on the hub and its local node"
git push
```

---


## Contract notes

### Additions and deviations versus `00-overview.md`

1. **`hub.state` additions** (T2.1): `NodeHello` (the hub keeps no protobuf in pure state; `hub.server.hello_from_proto` converts), `NodeRegistry.connect(..., *, data_path="relay")`, `disconnect`, `remove`, `by_id`, `all`; `EpochAllocator(..., *, block=EPOCH_BLOCK)` and `reserved_until`; `CellTable(epochs)` constructor plus `grant`, `owners(exclude=)`, `drop_cells`, `get`; `restrict_collecting(lineup, collecting)`; `BudgetCounter.add` (the `training.resume_from` start point); `VersionInfo.to_dict` / `from_dict`; `ResultIntake.check(..., *, node=None, uid_prefix=None)` (a node cannot write another node's uids), `accounts_at_intake`, `count_drop` / `dropped` (one WARNING per (node, reason)), `open(node_id, session_id, last_acked)` (uses `Hello.last_acked_seq`), `is_acked` (re-ack of a resent, acked batch), `forget_node`; `DROP_REASONS`.
2. **`OwnerRotation` reproduces SP4 only with simultaneous expiry** (spec 4.2's claim). With workers whose cells expire one after another the start is SP4's and later rounds keep every agent's share within one cell, but the exact SP4 sequence is not kept (the deficit rule prefers balance); T2.5's rotation test uses simultaneous requests.
3. **`colosseum.net.local` additions** (T2.2): `ControlChannel` (several threads of one process send on one connection: the learner's intake reports and its outputs) — `ConnectionIntake` and `NodeOutputs` take it instead of a raw `Connection`; `FrameSender` (one sender thread per learner connection, so a blocked chunk send keeps flushing results and sees a stop); `listener_socket`; `WorkerIO` gains `pump`, `take_lineups`, `idle`, `counters`, `send_progress`, `send_error`, `close`, `stats_fn`, `dropped`, and the keywords `results_batch`, `results_flush_sec`, `clock`; `LearnerOutputs` gains `checkpoints` (SP4's "no checkpoint queue, no checkpoint") and `finish(stop_event)` (SP4's `_release_weight_queues` for `QueueOutputs`); `ConnectionIntake.value` / `close`; constants `RUNTIME_DIR_ENV`, `MAX_SOCKET_PATH`, `SEND_SLICE_SEC`, `PROGRESS_INTERVAL_SEC`, `INTAKE_REPORT_SEC`.
4. **`RolloutLoop(snapshot_fallback=True)`** (T2.2): node workers pass False (spec 4.2: no latest fallback; a missing snapshot of a granted lineup is an error, R11 makes it impossible). With the default, in-process loops keep SP4's behaviour.
5. **`learner.learner.set_policy_version`** and **`_publish_weights(algorithm, agent_id, outputs)`** replace `_push_weights` (T2.2). The final snapshot is sent only when `outputs.checkpoints` (SP4: only with a checkpoint queue).
6. **`grpc_util` blob additions** (T2.3): `BlobError`, `blob_digest` (blake2b-128, the shm cache's digest), `not_modified_part`; `receive_parts` returns `None` data for a `not_modified` answer (T3.1's `GetLatest`); `send_parts(..., digest=None)`.
7. **`hub.server` additions** (T2.3): `identity` is required in `build_server` (the token is required in every mode); `LOOPBACK_KEY` names the plain loopback port in the returned port map; `SERVER_THREADS`, `OUTBOUND_QUEUE_SIZE`, `UPLOAD_REPLY_TIMEOUT_SEC`, `REGISTER_REPLY_TIMEOUT_SEC`; `NodeMessageIn.results` / `error`; `StopRequested`; `hello_from_proto`; `BlobStore.register_fixed(blob_id, frame, *, kind, agent_id)` and `blob_ident(kind, agent_id, data)`; blob files `blobs/<kind>-<agent>-<ident>.frame` (`trainer_state`: `.bin`). The cache key of a fixed / teacher / init blob on a node is `CacheKey("blob", agent_id, f"{kind}-{blob_id}")`.
8. **Proto** (T2.3): `Heartbeat.counters` (`map<string, int64>`, the node's cumulative counters: `lineup_fetch_failed`, `dropped_results_node_buffer`, `worker_restarts`); the `Event.kind` value `run_stopped` (the node finished `stop_run`: the hub's shutdown waits for it). No other proto change.
9. **`node.session` additions** (T2.3): the `SessionUp` / `SessionLost` inbox events, `backoff_delays(rng)`, `hello` is a callable called at every (re)connect.
10. **`node.agent` additions** (T2.4): `NodeOptions.credentials` (the local node's `(node_id, number, secret)`, given by the hub: no `Register` for it) and `shutdown_grace_sec`; `ResultBuffer`, `CellTracker`, `LINEUP_RETRY_SEC`, `METRICS_FLUSH_SEC`, `LOCAL_CHILD_GRACE_SEC`; `procs.ChildSupervisor` gains `get`, `incarnation`, `names`, `sentinels`, `restart`, `forget` and passes `incarnation` to the target; `hardware` exposes its probes (`affinity_cores`, `cgroup_quota_cores`, `physical_cores_raw`, `nvidia_gpus`, `torch_versions`); `learner.factory.teacher_from_frozen` / `init_from_blob`.
11. **`train`'s local node is node number 0** and authenticates through `NodeBook.add_ephemeral(name, machine_id)` (T2.5, an addition to T1.3's `NodeBook`): an in-memory record, never in `nodes.json`, so `train` runs leave no node entries behind and its uids are `<run_id>/train/n0/...`. Contract text "numbers ≥ 1 never reused" holds for registered nodes.
12. **`hub.hub` signatures in T2.5**: `run_training(config_path, overrides=None, *, listen=None)` and `run_hub(config_path, overrides=None, *, listen=None, port=None, insecure=False)`; the other contract keywords (`resume`, `idle`, `nodes_file`, `dashboard`) are added by the tasks that implement them (T4.3, T5.2, T8.1, T7.1).
13. **`LocalNodeOptions.evaluators` is 0 until T5.1** (the SP4 pulse child does the pulse, R6); the contract's "1 with `evaluation.enabled`" starts with T5.1.
14. **`hub` mode before T4.1** assigns every learner to the first node that connects with learner capacity, and grants a node collecting lineups only for agents whose learner runs on it (overview: "until T3.2"); a node without learners gets no lineups until T3.1 / T3.2.
15. **`hub.registry` additions**: `HubInfo.loopback_tls` (plain for `train`, TLS like the addresses for `hub`), `to_json` / `from_dict`, `this_process_info`, `alive`; `RUNNING_DIR`.
16. **`hub.admin`**: `resolve_hub(..., insecure=False)` (an insecure hub needs it on the client); the `hub.status` admin command (end-to-end check of the admin path; later commands are the contract's).
17. **`MatchLogWriter.dead`** (T2.5): the hub's switch to the journal-off rule needs it; a tokened `submit` is never counted in `WriterStats.dropped` when refused (the node resends), only when its group is still queued at the stop. Untokened submits keep SP4's behaviour (cut into `WRITE_BATCH` groups). `tests/unit/test_sp4_matchlog_writer.py` changes one internal read (`_queue.qsize()` → `_queued`).
18. **Child crashes with a report are fatal** (T2.4 / T2.5): a worker that sends a `ChildError` (contract, player or any Python exception) stops the run with exit 1 (SP4's "dead child stops the run"); only deaths without a report (signal, OOM) restart (spec 7.1). Until T4.2 a learner death of any kind stops the run with exit 1.
19. **Worker → node results** keep SP4's FX1 batching (`RESULTS_BATCH_MAX` / `RESULTS_BATCH_SEC`); the node batches again by `distributed.results_batch` / `results_flush_sec` before the hub.

### Proposed rulings (what — why — cost if wrong)

1. **Reported Python exceptions of workers are fatal, not restarted** — spec 7.1 restarts a worker that "died without a report"; an exception in env or bot code is a reported bug that would repeat (three restarts, then a disabled slot, would hide it for minutes) — cost: a transient Python error in user code (e.g. a flaky file read in a bot) stops the run instead of being retried.
2. **The deficit rotation's exact SP4 sequence is guaranteed only for simultaneous expiry** (note 2) — staggered expiry gives balanced shares instead; the spec claims only the simultaneous case — cost: a test or a user expecting SP4's exact owner order after staggered worker starts sees another (equally balanced) order.
3. **The two-`train` check is one slow test with four lightly loaded processes** (two workers with two envs each, two learners) — criterion 6 needs two runs at once, which cannot be done with three processes — cost: the slow test exceeds the "≤ 3 loaded processes" convention by one; if it flakes on a busy machine it is run alone.
4. **Checkpoint sends of a node's learner always block until the node reads** (`NodeOutputs.send_checkpoint` ignores `block`) — a Unix-socket send has no non-blocking form that keeps a frame whole, and the node reads continuously — cost: a node stuck elsewhere stalls its learner's periodic checkpoints (not its training data path, which is separate).
5. **The local node's learner grace is `SHUTDOWN_GRACE_SEC - 2.0`** (5 s, like SP4's final-checkpoint put timeout) and remote nodes' `distributed.shutdown_grace_sec - 5.0` — finals must reach the hub before it stops waiting — cost: a learner whose final snapshot takes longer to build than that is killed without one (as in SP4).
6. **An undecodable result batch is acked and its results counted `invalid`** — not acking would make the node resend a poison batch forever — cost: a batch broken by a hub-side converter bug is lost (counted).
7. **Results that fail the intake checks are dropped, the rest of their batch is journaled** — one bad result must not lose its neighbours — cost: none known.
8. **Node cache files are removed when the node's run ends** (the node deletes its prefix) — otherwise `/dev/shm` fills over runs; `cleanup_dead` covers crashed nodes — cost: a reader still mapping a segment keeps it (POSIX), so nothing breaks.

### Assumptions about other parts

1. **T1.1**: `ChunkError` has a `reason` attribute (`invalid` for validator failures); `decode_chunk` raises `FrameError` / `ChunkError` (both `ValueError`) on hostile input and never anything that is not an `Exception`; `ChunkValidator(role_spec, *, agent_id, teacher)` checks finite `behavior_logp`; `encode_checkpoint` / `decode_checkpoint` round-trip `make_checkpoint_payload()` dicts with `meta["kind"]`.
2. **T1.2**: `Lineup.epoch`, `SeatResult.generation`, `MatchResult.node` default to None; `results_from_batch` converts every field this part reads; `scripts/gen_proto.sh` regenerates after T2.3 adds `Heartbeat.counters`.
3. **T1.3**: `HubIdentity.load_or_create` is safe when two processes create the identity at the same time (the two-train test starts both without one); `NodeBook` can take `add_ephemeral` (T2.5 adds it if T1.3 has not); `AuthInterceptor` verifies node id + secret for `Session`, `Upload`, `Download`, and the token for `Register` / `Admin*`.
4. **T1.4**: `DistributedConfig` exists with the keys used here (`heartbeat_sec`, `node_lost_after_sec`, `results_batch`, `results_flush_sec`, `node_result_buffer_mb`, `max_message_mb`, `shutdown_grace_sec`, `port`, `insecure`).
5. **T1.5**: `WeightCacheReader.read_latest(agent, newer_than=)` returns `CachedWeights` whose `meta` is the frame's `WeightsMeta` (with `generation`, `policy_version`, `teacher_active`, `architecture`, `roles`), `latest_version` reads only the pointer, `put(key, frame)` / `has(key)` / `get(key)` work for `snapshot` and `blob` keys; `shm_dir()` honours `COLOSSEUM_SHM_DIR` (the test fixture).
6. **T0.1**: its result-flow and rotation baselines that use `Launcher` internals are listed by file in its part; T2.5(b) ports them with the same assertions (Step 10's last row).
7. **T3.1 / T3.2**: replace "collecting only for agents with a learner on the node" and the `hub`-mode learner rule, add the evicted-snapshot hold, `GetLatest` (it uses `receive_parts`' `None` for `not_modified`), relay servicers (replacing `test_relay_methods_are_not_implemented_yet`), the route connection's lz4 and `NodeData`.
8. **T4.1 / T4.2**: generations become real (one per assignment), learner restarts replace "a learner death stops the run", `LEARNER_RESTARTS` is used; `Event(kind="learner_failed")` keeps its name.
9. **T4.3** reads `hub_state.json` and `league_state.json["journal_row"]` as written here; **T4.4** extends `_shutdown` (the `run_stopped` wait), node-side hub-away buffering (the `ResultBuffer` already keeps batches while disconnected) and `watch_parent`.
10. **T7.1** turns the hub's `_drops`, the nodes' `Heartbeat.counters` and `IntakeStats.dropped` into the `system` keys; `SystemStats` is already keyed by (node, worker, incarnation).
