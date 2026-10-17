# SP5 Plan — Part 5: Evaluation on nodes (evaluators, block leases, pulse on blocks, hub-run tournaments, `hub --idle`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** This part owns spec block **5.3** of `docs/superpowers/specs/2026-10-17-sp5-distributed-design.md` (Russian; the binding authority — read it first) and controller ruling **R6** (the SP4 pulse child that the hub spawns since T2.5 goes away in T5.1):
- **T5.1** — evaluator processes on nodes (`node.evaluator_main`), the hub's block queue with leases on the hub clock (`distributed.block_lease_sec`, re-lease after expiry or node loss, a block's result whole and once, the first complete one wins, later ones counted), the lease protocol on the session (`BlockLeaseRequest` / `BlockLease` / `BlockResult` / `BlockAck`, `Hello.leases`, `cancel_leases`, `set_evaluators`), the node's evaluation bookkeeping (`node.leases`), evaluation blobs (`BlobStore` kind `eval`), and the pulse on blocks (`HubPulse`: SP4's `PulseFeeder` choice and backlog, stage-P blocks, journal only, the final snapshot of a run never evaluated); the hub's pulse child, its queues and `pulse_target` / `pulse_main` are deleted.
- **T5.2** — hub-run tournaments (`colosseum tournament --hub ADDR|RUN_DIR [--token T] [--token-file F] [--cancel NAME]`): the hub runs SP4's schedule / halving / confirmation / reports in its own thread per tournament (`HubTournament`, `TournamentService`), blocks go to the evaluators ahead of the pulse, the client follows the progress, Ctrl+C detaches (130), `--cancel` cancels, exit codes 0 / 130 / 1 / 2; tournaments continue after `hub --resume`; candidates from other run dirs are read on the hub machine (refused when another live hub owns that run dir); `colosseum hub --idle -c cfg` (a hub without trainable agents, nodes only evaluate).

Not in this part: evaluator *slots* and the `--evaluators` flag of `join` (T2.4 `default_slots`, T6.3), the restart limits of evaluators (T4.2 `ChildSupervisor`, `WORKER_RESTARTS`), the `Download` RPC and the node's blob fetch machinery (T2.3, T3.1), `hub --resume` itself (T4.3), the «Nodes» section (T7.1), docs (T7.5: `DISTRIBUTED_GUIDE` «Турнир на кластере и `hub --idle`», ARENA_GUIDE «`tournament --hub`, pulse на оценщиках»).

**What the part delivers.** After T5.2:
- a snapshot saved during training is evaluated by whichever nodes run evaluators (one or many machines, in parallel), on the same stage-P seeds as in SP4; its matches reach the journal only, all at once when its last block is in; a slow pulse still finishes snapshots (the backlog drops only snapshots not started); no evaluator → the pulse waits (INFO);
- a lost evaluator, a lost machine, a cut session or a restarted hub never loses an evaluation block silently: leases expire after `block_lease_sec` (600 s) or come back at once on node loss / session resume, each block is replayed at most `MAX_BLOCK_ATTEMPTS` (3) times, duplicates and late results are counted and dropped;
- `colosseum tournament --hub runs/x ...` runs a whole SP4 tournament on a running hub (`train`, `hub` or `hub --idle`), surviving client disconnects and hub restarts; the local `colosseum tournament` is unchanged;
- `colosseum hub --idle -c cfg.yaml` starts a hub whose nodes only evaluate.

**Read `00-overview.md` first** (global constraints, file map, the binding interface contract). Names here are the contract's; additions and deviations are listed in `## Contract notes` at the end.

**Conventions used in every task.**
- Every command from the repository root as `env -u PYTHONPATH .venv/bin/python ...` / `env -u PYTHONPATH .venv/bin/ruff ...`. "Full fast suite" = `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, **zero warnings**) followed by `env -u PYTHONPATH .venv/bin/ruff check .`.
- New test files start with `test_sp5_`; basenames and support-module names are unique across `tests/`; no `__init__.py`. The new support module `tests/eval_kit.py` (T5.1) goes into `[tool.ruff.lint.isort] known-first-party` of `pyproject.toml` in the same commit.
- Every file a test writes is under `tmp_path`; the autouse `_isolate_colosseum_dirs` (T1.3) already redirects `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `COLOSSEUM_SHM_DIR` (so the shm weight cache of every test lives in `tmp_path`). At most **2 loaded CPU processes per fast test** (an evaluator counts; hub and node processes do not); processes through `tests/cli_runner.py` (`run_in_session` for `hub`, `tournament --hub`).
- Hub logic is tested on `colosseum.hub.evaluation` objects with a fake `send` (R13); the lease protocol over the wire with the real gRPC server and `hub_kit.FakeNode` (T2.3); node logic with `node_kit.FakeHub` (T2.4); real processes only where the process boundary is the subject (evaluator kill -9, `train` with pulse, `tournament --hub` CLI).
- Every lease deadline, progress cadence and timeout uses `time.monotonic()` (the hub's clock for leases; nothing crosses the wire as a deadline: `BlockLease.lease_sec` is a duration, informational for the node); data are ordered by counters and uids; no test asserts on wall-clock values.
- Messages go through `logging` (never `warnings.warn`). Commits: conventional prefixes, no attribution lines; push after every task (branch `sp5-distributed`).

**Cross-task notes.**
- **State of the tree when T5.1 starts** (after T4.3 ‖ T4.4): `train` = `Hub` + local node child (R12); `Hub` (R5) keeps `Launcher`'s method names; the hub still spawns the SP4 pulse child (`arena.pulse.pulse_target`, R6) fed by `_pump_pulse` through an `mp.Queue` pair; `BlobStore` serves snapshots / fixed / teacher / init / recovery frames (T2.3, T3.1); the node fetches blobs into the shm cache with its transfer pool (T3.1); `ChildSupervisor` restarts workers and evaluators (`WORKER_RESTARTS`, T4.2); the journal has the `node` column (T4.3); `hub --resume` exists (T4.3). Edit points below are named by function / class (not lines); adapt them to the code as it then is.
- **Shared files.** T5.1 edits `hub/hub.py` only in the evaluation wiring it names (construction, `_save_checkpoint`, `_pump_pulse`, `_finish_pulse`, `_finish_metrics`, `_shutdown`'s start, the session / node-loss / node-message handlers' evaluation calls, the `Welcome` evaluator count, the metrics tick's `evaluation` value); `hub/server.py` only in `BlobStore` and the `Download` kind dispatch; `node/agent.py` only in a new "evaluation" section (evaluator children, lease messages, `set_evaluators`, `cancel_leases`, the reconnect hook). T5.2 edits `hub/hub.py` only in `_on_admin`'s `tournament.*` commands, two inbox handlers, the start (`TournamentService`, resume) and the stop (`stop` / `join`), plus the `mode == "idle"` branches it names; `cli.py` only in `tournament_cmd` and the `hub` group's `--idle`. T6.2 may run in parallel and touches only `BootstrapServicer` in `hub/server.py` (no overlap).
- **Proto.** Both tasks add fields to `src/colosseum/net/proto/colosseum_v1.proto` (next free field numbers; never renumber) and regenerate with `scripts/gen_proto.sh` (T1.2); the generated files are committed. `PROTOCOL_VERSION` stays `(1, 0)`: every addition is a new optional field or message, and nothing on the branch talks to an older node.
- **Pulse rules kept (SP4, spec 5.3):** domain P seeds (`schedule.stage_seed`, `training.seed` or 0), block ids `pulse-<run id[:12]>/P/<layout>/<agent>@v<N>/<composition>/<k>` and uids `<block id>/r<rotation>`, `evaluation.every` / `max_backlog` / `blocks` / `num_envs` / `deterministic` / `layouts` / `field` / `agents`, results to the journal only (contract amendment A20 of SP4: never the coordinator, ELO, PFSP, matchmaker or the episode statistics), `env_steps` = the snapshot's, the final snapshot of a run is offered but never evaluated (SP4 T6.1 ruling), `pulse_state.json` keys and its "not resumed" ruling, the `system` keys `pulse_skipped_snapshots` / `pulse_evaluated_snapshots` / `pulse_pending_snapshots`, the `Match results: ... (<w> from workers, <p> from the pulse)` summary line.
- **Tournament rules kept (SP4):** `run_tournament` and its modules are reused, not forked: the hub path enters through one new keyword (`hooks`) and changes nothing for the local path (every SP4 tournament test stays green unchanged); whole blocks journaled one transaction each, resume by `--name` with the settings check, snapshot links, `--pin`, reports.

**Verified while writing this part.** The library code of both tasks was run in a scratch package (`/tmp/claude-1000/-home-viv-dev-repos-Colosseum/b41cd787-f9ea-471b-9664-8578a7f7b4b1/scratchpad/part05/`) against the real repository code (`env -u PYTHONPATH .venv/bin/python`, `PYTHONPATH=<scratch>:src:tests`), with stand-ins only for contract items of earlier SP5 tasks: a protobuf subset of the contract's messages plus this part's additions (compiled with `grpc_tools.protoc`, grpcio 1.84 / protobuf 7.36), minimal `lineup_to_proto` / `result_to_proto` / `result_from_proto` converters, `CacheKey` / `WeightsMeta` / `CachedWeights` and a dict-backed cache reader (read-only numpy views, as `WeightCacheReader` gives). 59 scratch tests passed — this part's own test code of `test_sp5_block_queue.py`, `test_sp5_eval_messages.py`, `test_sp5_evaluation_service.py`, `test_sp5_node_leases.py`, the six thread-based tests of `test_sp5_evaluator.py` (through `tests/eval_kit.py` with the cache stand-in), the end-to-end pulse tests of `test_sp5_pulse_blocks.py`, `test_sp5_tournament_hooks.py` (the thread filter of `tournament.log` checked to fail without it) and `test_sp5_hub_tournaments.py`, with imports pointed at the scratch modules, plus earlier exploratory tests: `BlockQueue` (priority and FIFO, lease grouping by owner and match target, round-robin dispatch, first-complete-wins after an expiry, late / cancelled / unknown, attempts and failure, node loss without an attempt, `keep_only`, cancel and resubmission), `check_block_results`, the reworked `PulseFeeder` (the SP4 feeder tests' assertions), `HubPulse` on the `turns` toy game (links, identical stage-P seeds for two snapshots, uids, players with proto round trips, completion with `env_steps`, cleanup), `HubEvaluation` (join with unknown held leases, lease dispatch, acks, journal-only delivery with `node` stamped, late duplicate, fatal pulse error → `fail_run`, invalid results replayed with one WARNING), `NodeLeases`, the evaluator loop in a thread over a `multiprocessing.Pipe` with real `play_lineups` (whole blocks whose results pass `check_block_results`, missing weights → `EvalFailed`, cancel mid-play, a `PulseFailBot` `PlayerError` typed as `player` while the loop keeps running), and three end-to-end hub-run tournaments (`TournamentService` + `HubEvaluation` + an in-process fake node with one evaluator thread) on `write_fake_run` snapshots: `select` with `--halving-blocks 2 --confirm-blocks 6 --finalists 2` picks `main@v30` with one fetch per snapshot; cancel after 5 blocks, then `resume` plays exactly the 60 − held missing blocks (the rest from the journal); a run dir owned by another live pid is refused with the hint. The SP4 `run.py` hook edits of T5.2 were applied to a copy of the module and exercised by those tournaments. Not verified in scratch (they need T1.x–T4.x code that does not exist yet): the gRPC server and `FakeNode` / `FakeHub`, the real `net.local` connections and shm cache, `NodeAgent`, `Hub`, the CLI.

**Traceability (spec 5.3 and R6 → task / test).**

| Spec item | Task | Tests |
|---|---|---|
| Evaluator = node process taking blocks (seed + composition + rotations); engine `play_lineups` / SP4 schedule | T5.1 | `test_sp5_evaluator.py` (all), `test_sp5_node_evaluation.py::test_a_lease_is_fetched_once_played_on_an_evaluator_and_acked` |
| Lease on the hub clock, `block_lease_sec` 600, re-lease after expiry or node loss | T5.1 | `test_sp5_block_queue.py::test_an_expired_lease_requeues_and_its_late_result_still_wins`, `::test_attempts_node_loss_and_failure`, `test_sp5_hub_evaluation.py::test_node_loss_re_leases_the_blocks_to_another_node` |
| A block's result only whole and only once; first complete wins; late counted and dropped | T5.1 | `test_sp5_block_queue.py::test_check_block_results_accepts_exactly_the_block`, `::test_an_expired_lease_requeues_and_its_late_result_still_wins`, `test_sp5_pulse_blocks.py::test_hub_evaluation_leases_acks_and_journals_pulse_results_only` |
| Pulse: `PulseFeeder` cuts every `every`-th snapshot into blocks, played in parallel on machines | T5.1 | `test_sp5_pulse_blocks.py::test_hub_pulse_blocks_have_the_sp4_seeds_uids_and_players`, `test_sp5_block_queue.py::test_priority_fifo_grouping_and_round_robin` |
| SP4 pulse rules (domain P, `max_backlog`, journal only, final snapshot never evaluated) | T5.1 | `test_sp4_pulse.py` (ported), `test_sp4_pulse_runs.py` (ported) |
| A snapshot dropped by `max_backlog` withdraws its leased blocks (counted) | T5.1 | proposed ruling 1 (only unstarted snapshots are dropped); the withdrawal path: `test_sp5_pulse_blocks.py::test_a_failed_or_stopped_snapshot_withdraws_its_blocks` |
| No evaluators → the pulse waits, INFO | T5.1 | `test_sp5_evaluation_service.py::test_blocks_wait_with_one_info_while_no_node_runs_an_evaluator` |
| A snapshot's evaluation ends when all its blocks are played | T5.1 | `test_sp5_pulse_blocks.py::test_hub_pulse_completes_a_snapshot_with_its_last_block` |
| `tournament --hub` submits; the hub runs schedule / halving / confirmation / reports in `tournaments/<name>/` of its run dir | T5.2 | `test_sp5_hub_tournaments.py::test_select_on_a_run_runs_on_the_hub_evaluators`, `test_sp5_tournament_hub_cli.py::test_tournament_on_an_idle_hub_runs_on_two_nodes` |
| Continues after `hub --resume` | T5.2 | `test_sp5_hub_tournaments.py::test_a_running_tournament_resumes_from_the_journal`, `test_sp5_tournament_hub_cli.py::test_a_killed_hub_resumes_the_tournament` (slow) |
| Tournament blocks ahead of the pulse | T5.1 / T5.2 | `test_sp5_block_queue.py::test_priority_fifo_grouping_and_round_robin`, `test_sp5_hub_tournaments.py::test_tournament_blocks_go_before_pulse_blocks` |
| Client progress; Ctrl+C detaches; `--cancel NAME` | T5.2 | `test_sp5_hub_tournaments.py::test_progress_replay_and_detached_subscribers`, `::test_cancel_withdraws_the_leases_and_keeps_the_journal`, `test_sp5_tournament_hub_cli.py::test_detach_then_cancel` |
| Client exit codes 0 / 130 / 1 (and 2 for arguments) | T5.2 | `test_sp5_tournament_hub_cli.py::test_detach_then_cancel`, `::test_usage_errors_exit_with_2`, `::test_tournament_on_an_idle_hub_runs_on_two_nodes`, `test_sp5_hub_tournaments.py::test_a_failing_block_fails_the_tournament_with_its_error` |
| Candidates of other run dirs read on the hub machine; weights to evaluators by the protocol | T5.2 | `test_sp5_hub_tournaments.py::test_select_on_a_run_runs_on_the_hub_evaluators` (one fetch per player), `test_sp5_hub_evaluation.py::test_eval_blobs_are_served_by_download_and_removed_on_unregister` |
| Without `--hub` the tournament is local, as in SP4 | T5.2 | every SP4 tournament test unchanged; `test_sp5_tournament_hooks.py::test_without_hooks_nothing_changes` |
| `hub --idle`: a hub without trainable agents, nodes only evaluate, a new run dir | T5.2 | `test_sp5_tournament_hub_cli.py::test_tournament_on_an_idle_hub_runs_on_two_nodes`, `test_sp5_hub_idle.py` |
| Reports of tournaments on other runs go to `runs/x/tournaments/` only when no running hub owns `runs/x`, else an error with the hint | T5.2 | `test_sp5_hub_tournaments.py::test_a_run_owned_by_another_live_hub_is_refused` |
| Block seeds from the SP4 domains, independent of the machine (spec 7.6) | T5.1 | `test_sp5_pulse_blocks.py::test_hub_pulse_blocks_have_the_sp4_seeds_uids_and_players`, `test_sp5_tournament_hub_cli.py::test_tournament_on_an_idle_hub_runs_on_two_nodes` (two nodes, one seed set) |
| Evaluator restarts ≤ 3 / 10 min (spec 7.1; limits from T4.2) | T5.1 | `test_sp5_node_evaluation.py::test_an_evaluator_killed_mid_lease_is_restarted_and_its_blocks_reported_as_crash` |
| One fetch per machine of evaluation weights (spec 4.6) | T5.1 | `test_sp5_node_evaluation.py::test_a_lease_is_fetched_once_played_on_an_evaluator_and_acked`, `::test_unused_evaluation_blobs_are_dropped_from_the_cache` |
| R6: the pulse child removed | T5.1 | `test_sp5_pulse_blocks.py::test_the_pulse_child_is_gone` |

---

### Task T5.1: Evaluators, block leases, pulse on blocks; the pulse child removed

Spec 5.3 (evaluators, leases, pulse), 7.1 (evaluator restarts, bad data), 7.6 (block seeds), 4.6 (one fetch per machine), ruling R6. An **evaluator** is a node child process (`evaluator-<i>`, spawn, restarted like a worker by `ChildSupervisor` within `WORKER_RESTARTS`) that plays **leases**: a lease is consecutive whole blocks of one owner (a pulse snapshot or a tournament) with at least `TASK_MATCHES × num_envs` matches (SP4 `make_tasks`), played with SP4's `play_lineups` on a `CachePlayerPool` (neural players built from the machine's shm weight cache, scripted players as `ScriptedPlayer`s). The hub keeps every block in a `BlockQueue` (pure; hub monotonic clock): nodes ask for slots (`BlockLeaseRequest.free_slots`, an increment), the hub grants leases round robin over the asking nodes (`BlockLease`), a node reports one `BlockResult` per block and keeps it until the hub's `BlockAck` (resent after a reconnect; `Hello.leases` names the leases it holds), the hub accepts a block's results only whole (`check_block_results`) and only once (the first complete result of any lease that held it; later ones `late`), re-leases a block after its lease expires (`distributed.block_lease_sec`, 600 s), at once after node loss or when a resumed session no longer holds it, and fails it after `MAX_BLOCK_ATTEMPTS` (3) leases that ended without a result. The hub tells each node how many evaluators to run (`Welcome.evaluators`, `Command(set_evaluators)`): its evaluator slots while any evaluation work can exist (pulse on, or a tournament active), else 0 — so a run without pulse starts no evaluator process. The **pulse** moves onto blocks (`HubPulse`): SP4's `PulseFeeder` (every `every`-th snapshot, the final one offered, backlog of `max_backlog` waiting snapshots, the oldest waiting dropped), SP4's stage-P blocks (`pulse_blocks`), results to the journal only and all at once when the snapshot's last block is in; the snapshot's weights are hard-linked under `runs/<name>/eval/pulse/` and served to nodes as an evaluation blob. The SP4 pulse child, its queues and `pulse_target` / `pulse_main` are deleted (R6).

**Split point:** dispatch **(a) Steps 1–8** (protocol fields, messages, the pure `BlockQueue` / `check_block_results` / `HubEvaluation`, `node.leases`, the evaluator process and `tests/eval_kit.py`; the hub still runs the SP4 pulse child, the tree stays green), commit and push; then **(b) Steps 9–16** (the `PulseFeeder` rework and `HubPulse`, `BlobStore` evaluation blobs, the node agent's evaluation section, the hub wiring, the pulse child removed, the SP4 pulse tests ported, the integration tests, the slow suite), commit and push.

**Files:**
- Create: `src/colosseum/hub/evaluation.py`, `src/colosseum/node/leases.py`, `src/colosseum/node/evaluator_main.py`, `tests/eval_kit.py`
- Modify (a): `src/colosseum/net/proto/colosseum_v1.proto` + regenerated `colosseum_v1_pb2.py` / `colosseum_v1_pb2_grpc.py`; `src/colosseum/net/messages.py` (`PlayerRef`, `BlockTask`, converters, priorities); `src/colosseum/net/local.py` (`EvalTask`, `EvalCancel`, `EvalDone`, `EvalFailed`); `src/colosseum/arena/tournament/engine.py` (`_Placement` → `MatchPlacement` with `where(..., block_ids)`, `_error_kind` → `error_kind`); `tests/game_helpers.py` (`PulseFailBot` also fails in evaluator processes); `pyproject.toml` (`eval_kit` in `known-first-party`)
- Modify (b): `src/colosseum/arena/pulse.py` (the child removed, `PulseFeeder` / `PulseRequest` reworked); `src/colosseum/core/config.py` (two `EvaluationConfig` descriptions); `src/colosseum/hub/server.py` (`BlobStore.register_weights` / `unregister`, `Download` kind `eval`); `src/colosseum/node/agent.py` (evaluation section); `src/colosseum/hub/hub.py` (evaluation wiring; the pulse child and its queues removed); `src/colosseum/hub/hub.py::run_training` (the local node's evaluator slot); `tests/game_helpers.py` (`PulseFailBot`: evaluator processes only)
- Test (a): `tests/unit/test_sp5_eval_messages.py`, `tests/unit/test_sp5_block_queue.py`, `tests/unit/test_sp5_evaluation_service.py`, `tests/unit/test_sp5_node_leases.py`, `tests/integration/test_sp5_evaluator.py`
- Test (b): `tests/unit/test_sp5_pulse_blocks.py`, `tests/integration/test_sp5_node_evaluation.py`, `tests/integration/test_sp5_hub_evaluation.py`; ported: `tests/unit/test_sp4_pulse.py`, `tests/integration/test_sp4_pulse_runs.py`

**Interfaces:**
- Consumes: `pb` (`colosseum.net.proto.colosseum_v1_pb2`), `lineup_to_proto` / `lineup_from_proto` / `result_to_proto` / `result_from_proto` (T1.2); `CacheKey`, `WeightCacheWriter` / `WeightCacheReader`, `shm_dir`, `cache_prefix` (T1.5); `encode_weights` / `decode_weights`, `WeightsMeta` (T1.1); `listener` / `connect` / `address`, `ChildHello`, `Stop` (T2.2); `HubSession`, `SessionHandle`, `BlobStore`, `NodeMessageIn` (T2.3); `NodeAgent`, `ChildSupervisor`, `ChildSpec`, `NodeOptions`, `node_kit.FakeHub` (T2.4); `Hub` and its R5 handlers, `LocalNodeOptions`, `run_training` (T2.5); the node's blob fetch into the cache (T3.1); `hub_kit.FakeNode` (T2.3); `WORKER_RESTARTS` and evaluator restarts (T4.2); the hub's run-stopping error path for a contract / player error (T2.5 / T4.4; called `Hub._fail_run(message)` below); SP4 `pulse_plan`, `pulse_blocks`, `config_player`, `TournamentPlayer`, `check_blocks`, `play_lineups`, `build_frozen_model`, `make_bot`, `ScriptedPlayer`, `link_or_copy`, `read_weights_file`.
- Produces (contract, plus additions marked *):
  - `colosseum.hub.evaluation`: `PlayerRef` and `BlockTask` (re-exported from `net.messages`*), `Lease`, `BlockQueue` (+ `want`* / `unwant`* / `dispatch`* / `release`* / `keep_only`* / `forget`* / `take_failed`* / `task`* / `leases_of`* / `queued`* / `live`* / `stats`*), *`check_block_results`, *`MAX_BLOCK_ATTEMPTS = 3`, *`PULSE_PREFIX = "pulse:"`, *`TOURNAMENT_PREFIX = "tournament:"`, *`RETRY_ERRORS`, *`FATAL_ERRORS`, *`pulse_owner`, *`player_ref`, *`EvalBlobs`, *`JoinReply`, *`HubEvaluation`, `HubPulse` (b);
  - *`colosseum.net.messages`: `PlayerRef`, `BlockTask`, `PULSE_PRIORITY = 1`, `TOURNAMENT_PRIORITY = 0`, `player_ref_to_proto`, `player_ref_from_proto`, `block_task_to_proto`, `block_task_from_proto`, `lease_to_proto`;
  - *`colosseum.net.local`: `EvalTask`, `EvalCancel`, `EvalDone`, `EvalFailed`;
  - *`colosseum.node.leases`: `NodeLease`, `BlockOutcome`, `NodeLeases`, `EVAL_BLOBS_KEPT = 8`;
  - `colosseum.node.evaluator_main`: `evaluator_target(**kwargs)`, *`evaluator_main`, *`evaluator_loop`, *`CachePlayerPool`, *`MissingWeights`, *`play_task`, *`CONNECT_TIMEOUT_SEC = 30.0`, *`POLL_SEC = 0.2`;
  - *`colosseum.arena.tournament.engine`: `MatchPlacement` (was `_Placement`), `error_kind` (was `_error_kind`);
  - *proto: `Hello.leases`, `BlockLease.num_envs`, `BlockResult.error_kind`, message `BlockAck {lease_id, block_ids}` in `HubMessage.block_ack`, `Command.lease_ids` (kind `cancel_leases`), `BlobRequest.kind = "eval"`;
  - *`BlobStore.register_weights(blob_id, weights_file, *, agent_id, networks, roles)`, `BlobStore.unregister(blob_id)` (b);
  - *run-dir paths `eval/pulse/<agent>/<checkpoint id>/` (b); *`system` key `evaluation` (b).

- [ ] **Step 1 (a): Write the failing tests of the messages, the block queue and the evaluation service**

Create `tests/unit/test_sp5_eval_messages.py`:
```python
"""Evaluation messages (SP5 T5.1, spec 5.3): PlayerRef / BlockTask round trips through protobuf, their checks,
and the protocol fields this task adds."""
from __future__ import annotations

import pytest

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.net.messages import (
    BlockTask,
    PlayerRef,
    block_task_from_proto,
    lease_to_proto,
    player_ref_from_proto,
    player_ref_to_proto,
)
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node.weights import CacheKey
from colosseum.players.registry import BotSpec

NETWORKS = {"model_class": "game_helpers.GameTestModel", "kwargs": {"core": "none", "hidden": 16}}
NEURAL = PlayerRef("main@v3", "main", "ckpt_v3", ("player",), "neural", blob=CacheKey("blob", "main", "pulse-main-ckpt_v3"),
                   networks=NETWORKS)
BOT = PlayerRef("ones", "ones", "fixed", ("player",), "scripted", bot=BotSpec("game_helpers.ConstantBot", {"value": 1}))


def _lineups(block: str) -> tuple[Lineup, ...]:
    return tuple(Lineup("2p", [SeatAssignment("main", "ckpt_v3", False), SeatAssignment("ones", "fixed", False)],
                        seed=11, block=block, uid=f"{block}/r{r}") for r in range(2))


def test_player_refs_round_trip_through_protobuf():
    for ref in (NEURAL, BOT):
        assert player_ref_from_proto(player_ref_to_proto(ref)) == ref
    p = player_ref_to_proto(NEURAL)
    assert p.blob_id == "pulse-main-ckpt_v3" and p.kind == "neural" and not p.bot_class


def test_player_refs_check_their_kind():
    with pytest.raises(ValueError, match="blob"):
        PlayerRef("x", "x", "fixed", ("player",), "neural")
    with pytest.raises(ValueError, match="bot"):
        PlayerRef("x", "x", "fixed", ("player",), "scripted")
    with pytest.raises(ValueError, match="kind"):
        PlayerRef("x", "x", "fixed", ("player",), "human", bot=BOT.bot)


def test_non_json_bot_kwargs_are_refused_with_the_player_name():
    ref = PlayerRef("odd", "odd", "fixed", ("player",), "scripted", bot=BotSpec("game_helpers.ConstantBot", {"v": {1, 2}}))
    with pytest.raises(ValueError, match="bot kwargs of odd"):
        player_ref_to_proto(ref)


def test_block_tasks_and_leases_round_trip():
    tasks = [BlockTask(f"t/A/2p/main@v3/ones/{k}", "tournament", "tournament:t", _lineups(f"t/A/2p/main@v3/ones/{k}"),
                       (NEURAL, BOT), 0, True, 3) for k in range(2)]
    lease = lease_to_proto("L1", 600.0, tasks)
    assert (lease.lease_id, lease.lease_sec, lease.num_envs, len(lease.blocks)) == ("L1", 600.0, 3, 2)
    back = [block_task_from_proto(b, num_envs=lease.num_envs) for b in lease.blocks]
    for got, want in zip(back, tasks):
        assert (got.block_id, got.source, got.deterministic, got.num_envs) == (want.block_id, "tournament", True, 3)
        assert (got.owner, got.priority) == ("", 0)                         # hub-side fields stay on the hub
        assert got.lineups == want.lineups and got.players == want.players
        assert got.matches == 2


def test_new_session_fields_exist():
    hello = pb.Hello(leases=["L1", "L2"])
    assert list(hello.leases) == ["L1", "L2"]
    assert pb.BlockLease(num_envs=4).num_envs == 4
    assert pb.BlockResult(error_kind="abandoned").error_kind == "abandoned"
    msg = pb.HubMessage(block_ack=pb.BlockAck(lease_id="L1", block_ids=["b0"]))
    assert msg.WhichOneof("m") == "block_ack"
    assert list(pb.Command(kind="cancel_leases", lease_ids=["L1"]).lease_ids) == ["L1"]
```

Create `tests/unit/test_sp5_block_queue.py`:
```python
"""The hub's block queue (SP5 T5.1, spec 5.3): priority and FIFO, leases of one owner with a match target, round
robin over nodes, a block's result whole and once (first complete wins, late / cancelled / unknown counted),
expiry, attempts, node loss, session resume, cancel. Pure: the clock is a number."""
from __future__ import annotations

import pytest

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.evaluation import MAX_BLOCK_ATTEMPTS, BlockQueue, check_block_results
from colosseum.net.messages import BlockTask, PlayerRef
from colosseum.players.registry import BotSpec
from game_helpers import make_match_result

BOT = PlayerRef("rnd", "rnd", "fixed", ("player",), "scripted", bot=BotSpec("colosseum.players.RandomBot"))


def _task(block_id, owner="tournament:t", priority=0, matches=2, num_envs=1):
    lineups = tuple(Lineup("2p", [SeatAssignment("a", "ckpt_v1", False), SeatAssignment("rnd", "fixed", False)],
                           seed=7, block=block_id, uid=f"{block_id}/r{r}") for r in range(matches))
    source = "pulse" if owner.startswith("pulse:") else "tournament"
    return BlockTask(block_id, source, owner, lineups, (BOT,), priority, False, num_envs)


def _ids():
    counter = iter(range(10_000))
    return lambda: f"L{next(counter)}"


def _results(task):
    return [make_match_result([(0, lu.seats[0].agent_id, lu.seats[0].network_id),
                               (1, lu.seats[1].agent_id, lu.seats[1].network_id)],
                              uid=lu.uid, block=task.block_id, seed=lu.seed, source=task.source)
            for lu in task.lineups]


def test_priority_fifo_grouping_and_round_robin():
    q = BlockQueue(task_matches=2, new_id=_ids())          # target 2 x num_envs (1) = 2 matches per lease
    q.submit([_task("p0", "pulse:a:c1", 1), _task("p1", "pulse:a:c1", 1)])
    q.submit([_task("t0"), _task("t1"), _task("t2")])
    (lease, tasks), = q.lease(1, 1, 0.0, 10.0)
    assert [t.block_id for t in tasks] == ["t0"] and lease.deadline == 10.0 and lease.node == 1
    q.want(1, 2)
    q.want(2, 2)
    got = q.dispatch(0.0, 10.0)
    assert [(n, [t.block_id for t in ts]) for n, _l, ts in got] == [(1, ["t1"]), (2, ["t2"]), (1, ["p0"]), (2, ["p1"])]
    assert q.waiting() == {}
    big = BlockQueue(task_matches=4, new_id=_ids())
    big.submit([_task(f"b{i}", num_envs=2) for i in range(5)])        # target 8 matches: 4 blocks of 2
    (_l1, first), (_l2, rest) = big.lease(3, 2, 0.0, 1.0)
    assert [t.block_id for t in first] == ["b0", "b1", "b2", "b3"] and [t.block_id for t in rest] == ["b4"]


def test_a_lease_never_mixes_owners():
    q = BlockQueue(task_matches=10, new_id=_ids())
    q.submit([_task("x0", "tournament:x"), _task("y0", "tournament:y"), _task("x1", "tournament:x")])
    assert [[t.block_id for t in ts] for _l, ts in q.lease(1, 3, 0.0, 1.0)] == [["x0"], ["y0"], ["x1"]]


def test_dispatch_keeps_the_wanted_slots_when_the_queue_runs_dry():
    q = BlockQueue(task_matches=1, new_id=_ids())
    q.want(4, 3)
    assert q.dispatch(0.0, 1.0) == [] and q.waiting() == {4: 3}
    q.submit([_task("t0")])
    (node, _lease, tasks), = q.dispatch(0.0, 1.0)
    assert node == 4 and [t.block_id for t in tasks] == ["t0"] and q.waiting() == {4: 2}
    q.unwant(4)
    assert q.waiting() == {}


def test_an_expired_lease_requeues_and_its_late_result_still_wins():
    q = BlockQueue(task_matches=1, new_id=_ids())
    t = _task("t0")
    q.submit([t])
    (lease, _), = q.lease(1, 1, 0.0, 5.0)
    assert q.expire(4.9) == [] and q.expire(5.0) == ["t0"]                 # back in the queue, one attempt used
    (lease2, _), = q.lease(2, 1, 6.0, 5.0)
    assert q.complete(lease.lease_id, "t0", _results(t), 7.0) == "accepted"   # the expired lease came back first
    assert q.complete(lease2.lease_id, "t0", _results(t), 8.0) == "late"
    assert q.complete("nope", "t0", [], 8.0) == "unknown"                   # never issued for this block
    assert q.complete("nope", "zz", [], 8.0) == "unknown"
    assert q.leases_of(2) == [] and q.task("t0") is None
    q.forget("tournament:t")
    assert q.complete(lease2.lease_id, "t0", [], 9.0) == "late"             # remembered by id after forget


def test_attempts_node_loss_and_failure():
    q = BlockQueue(task_matches=1, new_id=_ids())
    q.submit([_task("t0")])
    for _ in range(MAX_BLOCK_ATTEMPTS - 1):
        (lease, _), = q.lease(1, 1, 0.0, 1.0)
        assert q.release(lease.lease_id, "t0", count=True, reason="crash") == "requeued"
    (lease, _), = q.lease(1, 1, 0.0, 1.0)
    assert q.drop_node(1) == ["t0"]                                          # node loss: no attempt used
    (lease, _), = q.lease(1, 1, 0.0, 1.0)
    assert q.release(lease.lease_id, "t0", reason="crash") == "failed"
    assert q.take_failed() == [("t0", "tournament:t", "crash")] and q.take_failed() == []
    assert q.task("t0") is None and q.complete(lease.lease_id, "t0", [], 0.0) == "late"
    assert q.release("other", "t0") == "ignored"


def test_keep_only_cancel_and_resubmit():
    q = BlockQueue(task_matches=1, new_id=_ids())
    q.submit([_task("t0"), _task("t1"), _task("t2"), _task("p0", "pulse:a:c", 1)])
    (a, _), (_b, _) = q.lease(1, 2, 0.0, 9.0)
    (c, _), = q.lease(2, 1, 0.0, 9.0)
    assert q.keep_only(1, [a.lease_id]) == ["t1"] and q.leases_of(1) == [a.lease_id]
    blocks, leases = q.cancel("tournament:t")
    assert blocks == ["t0", "t1", "t2"] and leases == {1: [a.lease_id], 2: [c.lease_id]}
    assert q.complete(a.lease_id, "t0", [], 0.0) == "cancelled"
    assert q.queued() == 1 and q.queued("pulse:") == 1 and q.queued("tournament:") == 0 and q.live("pulse:a:c") == 1
    q.submit([_task("t0")])                                                 # a cancelled block may be queued again
    with pytest.raises(ValueError, match="already"):
        q.submit([_task("t0")])
    with pytest.raises(ValueError, match="no lineups"):
        q.submit([BlockTask("e", "tournament", "tournament:t", (), (BOT,), 0, False, 1)])


def test_stats_count_every_outcome():
    q = BlockQueue(task_matches=1, new_id=_ids())
    q.submit([_task("t0"), _task("t1")])
    (lease, _), (lease1, _) = q.lease(1, 2, 0.0, 1.0)
    q.complete(lease.lease_id, "t0", [], 0.0)
    q.complete(lease.lease_id, "t0", [], 0.0)
    q.expire(5.0)
    stats = q.stats()            # current states (queued, leased_now, leases_now) and cumulative counters
    assert (stats["submitted"], stats["leases"], stats["leased"], stats["completed"], stats["late"],
            stats["expired"], stats["requeued"], stats["queued"], stats["leased_now"],
            stats["leases_now"]) == (2, 2, 2, 1, 1, 1, 1, 1, 0, 0)


def test_check_block_results_accepts_exactly_the_block():
    t = _task("t0")
    assert check_block_results(t, _results(t)) is None
    assert "results for" in check_block_results(t, _results(t)[:1])
    repeated = _results(t)
    repeated[1].uid = repeated[0].uid
    assert "repeats" in check_block_results(t, repeated)
    seed = _results(t)
    seed[0].seed = 8
    assert "seed" in check_block_results(t, seed)
    seat = _results(t)
    seat[0].seats[1].agent_id = "x"
    assert "seat 1" in check_block_results(t, seat)
    seats = _results(t)
    seats[0].seats.pop()
    assert "seats" in check_block_results(t, seats)
    nan = _results(t)
    nan[0].teams[0].score = float("nan")
    assert "non-finite" in check_block_results(t, nan)
    kind = _results(t)
    kind[0].outcome_kind = "chess"
    assert "outcome kind" in check_block_results(t, kind)
```
Create `tests/unit/test_sp5_evaluation_service.py`:
```python
"""The hub's evaluation service (SP5 T5.1, spec 5.3) with owner sinks (the pulse comes in (b)): demand and
set_evaluators, joins with held leases, leases to asking nodes, acks, delivery once with the node stamped,
invalid results, typed errors, crashes, attempts, node loss, the waiting INFO, stop. No gRPC: ``send`` records."""
from __future__ import annotations

import logging

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.evaluation import MAX_BLOCK_ATTEMPTS, HubEvaluation
from colosseum.net.messages import BlockTask, PlayerRef, block_task_from_proto, result_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.players.registry import BotSpec
from game_helpers import make_match_result

BOT = PlayerRef("rnd", "rnd", "fixed", ("player",), "scripted", bot=BotSpec("colosseum.players.RandomBot"))
OWNER = "tournament:t"


def _tasks(n, owner=OWNER):
    out = []
    for k in range(n):
        block = f"t/A/2p/a/rnd/{k}"
        lineups = tuple(Lineup("2p", [SeatAssignment("a", "ckpt_v1", False), SeatAssignment("rnd", "fixed", False)],
                               seed=100 + k, block=block, uid=f"{block}/r{r}") for r in range(2))
        out.append(BlockTask(block, "tournament", owner, lineups, (BOT,), 0, False, 1))
    return out


class Recorder:
    def __init__(self):
        self.sent: list[tuple[int, pb.HubMessage]] = []
        self.sink: list[tuple[str, str, object]] = []

    def send(self, node, msg):
        self.sent.append((node, msg))

    def of(self, kind):
        return [(n, getattr(m, kind)) for n, m in self.sent if m.WhichOneof("m") == kind]


def _service(rec, **kw):
    return HubEvaluation(lease_sec=600, send=rec.send, clock=kw.pop("clock", lambda: 0.0), **kw)


def _result_msg(lease_id, spec, *, drop=0):
    task = block_task_from_proto(spec, num_envs=1)
    results = [make_match_result([(0, "a", "ckpt_v1"), (1, "rnd", "fixed")], uid=lu.uid, block=task.block_id,
                                 seed=lu.seed, source="tournament") for lu in task.lineups]
    return pb.BlockResult(lease_id=lease_id, block_id=task.block_id,
                          results=[result_to_proto(r) for r in results[:len(results) - drop]])


def test_join_returns_the_evaluator_count_from_the_demand_and_cancels_unknown_leases():
    rec = Recorder()
    ev = _service(rec)
    assert ev.node_joined(1, slots=2, held_leases=[], fresh=True).evaluators == 0      # no work can exist yet
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    assert [c.evaluators for _n, c in rec.of("command")] == [2]                        # set_evaluators 2
    join = ev.node_joined(2, slots=1, held_leases=["old-1", "old-2"], fresh=False)
    assert join.evaluators == 1
    (msg,) = join.after_welcome
    assert msg.command.kind == "cancel_leases" and list(msg.command.lease_ids) == ["old-1", "old-2"]
    ev.remove_owner(OWNER)
    assert sorted((n, c.evaluators) for n, c in rec.of("command") if c.kind == "set_evaluators") == \
        [(1, 0), (1, 2), (2, 0)]


def test_leases_are_granted_to_asking_nodes_and_results_reach_the_owner_once_with_the_node():
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.submit(_tasks(3))
    assert rec.of("lease") == []                                           # nobody asked
    ev.on_lease_request(1, 1)
    ((node, lease),) = rec.of("lease")
    assert node == 1 and lease.num_envs == 1 and len(lease.blocks) == 2      # TASK_MATCHES x 1 env = 4 matches
    for spec in lease.blocks:
        ev.on_block_result(1, _result_msg(lease.lease_id, spec))
    ev.on_block_result(1, _result_msg(lease.lease_id, lease.blocks[0]))     # a duplicate (a resend after a cut)
    assert [(k, key) for k, key, _v in rec.sink] == [("block", lease.blocks[0].block_id),
                                                     ("block", lease.blocks[1].block_id)]
    assert {r.node for _k, _b, results in rec.sink for r in results} == {1}
    assert len(rec.of("block_ack")) == 3 and ev.queue.stats()["late"] == 1


def test_invalid_results_are_replayed_and_warned_once_per_node(caplog):
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.submit(_tasks(2))
    ev.on_lease_request(1, 1)
    ((_n, lease),) = rec.of("lease")
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.evaluation"):
        ev.on_block_result(1, _result_msg(lease.lease_id, lease.blocks[0], drop=1))
        ev.on_block_result(1, _result_msg(lease.lease_id, lease.blocks[1], drop=1))
    assert ev.invalid_results == 2 and ev.queue.queued() == 2 and rec.sink == []
    assert len([r for r in caplog.records if "invalid results" in r.getMessage()]) == 1


def test_a_typed_block_error_fails_its_owner_and_a_crash_is_replayed():
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.submit(_tasks(4))
    ev.on_lease_request(1, 1)
    ((_n, lease),) = rec.of("lease")
    first, second = (s.block_id for s in lease.blocks)
    ev.on_block_result(1, pb.BlockResult(lease_id=lease.lease_id, block_id=first, failed=True, error_kind="crash"))
    assert ev.queue.task(first) is not None and ev.queue.queued() == 3          # back in the queue
    ev.on_block_result(1, pb.BlockResult(lease_id=lease.lease_id, block_id=second, failed=True,
                                         error_kind="player", error="bot raised on purpose"))
    ((kind, key, text),) = rec.sink
    assert (kind, key) == ("failed", "player") and "bot raised on purpose" in text and "node 1" in text
    assert ev.queue.queued() == 0 and ev.queue.live(OWNER) == 0                # the owner's blocks are gone


def test_a_block_that_fails_its_attempts_fails_its_owner():
    rec = Recorder()
    now = [0.0]
    ev = _service(rec, clock=lambda: now[0])
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.submit(_tasks(1))
    for attempt in range(MAX_BLOCK_ATTEMPTS):
        ev.on_lease_request(1, 1)
        now[0] += 601.0                                                    # the lease expires (600 s)
        ev.tick()
    ((kind, key, text),) = rec.sink
    assert (kind, key) == ("failed", "attempts") and f"failed {MAX_BLOCK_ATTEMPTS} times" in text
    assert "lease expired" in text and ev.queue.stats()["expired"] == MAX_BLOCK_ATTEMPTS


def test_node_loss_and_a_fresh_session_release_the_leases_without_an_attempt():
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.node_joined(2, slots=1, held_leases=[], fresh=True)
    ev.submit(_tasks(2))
    ev.on_lease_request(1, 1)
    ((_n, lease),) = rec.of("lease")
    ev.on_lease_request(2, 1)                                              # node 2 waits: nothing queued
    ev.node_lost(1)                                                        # node 1's blocks go to node 2 at once
    assert [n for n, _l in rec.of("lease")] == [1, 2] and ev.queue.stats()["requeued"] == 2
    ev.node_joined(2, slots=1, held_leases=[], fresh=True)                 # node 2's process restarted
    assert ev.queue.queued() == 2 and ev.queue.stats()["failed"] == 0


def test_a_resumed_session_keeps_its_held_leases_only():
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=2, held_leases=[], fresh=True)
    ev.submit(_tasks(4))
    ev.on_lease_request(1, 2)
    (_n, a), (_n2, b) = rec.of("lease")
    ev.session_closed(1)
    join = ev.node_joined(1, slots=2, held_leases=[a.lease_id], fresh=False)
    assert join.after_welcome == [] and ev.queue.leases_of(1) == [a.lease_id] and ev.queue.queued() == 2


def test_blocks_wait_with_one_info_while_no_node_runs_an_evaluator(caplog):
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.submit(_tasks(2))
    with caplog.at_level(logging.INFO, logger="colosseum.hub.evaluation"):
        ev.tick()
        ev.tick()
    assert len([r for r in caplog.records if "no connected node runs an evaluator" in r.getMessage()]) == 1
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.tick()
    ev.node_lost(1)
    with caplog.at_level(logging.INFO, logger="colosseum.hub.evaluation"):
        ev.tick()
    assert len([r for r in caplog.records if "no connected node runs an evaluator" in r.getMessage()]) == 2


def test_stop_ends_the_dispatch():
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.stop()
    ev.submit(_tasks(1))
    ev.on_lease_request(1, 1)
    ev.tick()
    assert rec.of("lease") == [] and ev.stats()["queued"] == 1
```

- [ ] **Step 2 (a): Run them: RED**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_eval_messages.py tests/unit/test_sp5_block_queue.py tests/unit/test_sp5_evaluation_service.py -q`
Expected: collection errors — `ImportError: cannot import name 'BlockTask' from 'colosseum.net.messages'` and `ModuleNotFoundError: No module named 'colosseum.hub.evaluation'`.

- [ ] **Step 3 (a): Protocol fields, messages, local messages**

Add to `src/colosseum/net/proto/colosseum_v1.proto` (next free field numbers of each message; never renumber existing fields):
```proto
message Hello { ...; repeated string leases = <next>; }          // evaluation leases the node holds (T5.1)
message BlockLease { ...; int32 num_envs = <next>; }               // envs an evaluator plays the lease with
message BlockResult { ...; string error_kind = <next>; }
  // "" with results; failed: abandoned | crash | missing_weights (re-leased) | contract | player | config (fatal)
message BlockAck { string lease_id = 1; repeated string block_ids = 2; }
message HubMessage { oneof m { ...; BlockAck block_ack = <next>; } }
message Command { ...; repeated string lease_ids = <next>; }      // kind "cancel_leases"
// BlobRequest.kind gains the value "eval" (no schema change)
```
Regenerate with `scripts/gen_proto.sh` and commit the regenerated `colosseum_v1_pb2.py` / `colosseum_v1_pb2_grpc.py` (ruff excludes them, T1.2).

Append to `src/colosseum/net/messages.py` (module docstring: one paragraph "Evaluation (T5.1)"; the imports go to the top of the module):
```python
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from colosseum.core.types import Lineup
from colosseum.node.weights import CacheKey
from colosseum.players.registry import BotSpec

TOURNAMENT_PRIORITY = 0      # spec 5.3: tournament blocks go before pulse blocks
PULSE_PRIORITY = 1


@dataclass(frozen=True)
class PlayerRef:
    """One player of an evaluation block as an evaluator builds it (spec 5.3): a neural player's weights are the
    machine-cache blob ``blob`` (kind ``blob``, fetched from the hub with ``BlobRequest(kind="eval")``) built with
    ``networks`` (a ``NetworkConfig`` dump) for ``roles``; a scripted player is ``bot``. ``(agent_id, network_id)``
    is how the block's seats name it; ``name`` is the entity name used in messages."""

    name: str
    agent_id: str
    network_id: str
    roles: tuple[str, ...]
    kind: Literal["neural", "scripted"]
    blob: CacheKey | None = None
    networks: dict | None = None
    bot: BotSpec | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("neural", "scripted"):
            raise ValueError(f"player {self.name!r}: kind must be neural or scripted, got {self.kind!r}")
        if self.kind == "neural" and (self.blob is None or self.blob.kind != "blob" or not self.blob.ident):
            raise ValueError(f"player {self.name!r}: a neural player needs a blob cache key")
        if self.kind == "scripted" and self.bot is None:
            raise ValueError(f"player {self.name!r}: a scripted player needs a bot")


@dataclass(frozen=True)
class BlockTask:
    """One evaluation block (an SP4 schedule block: one seed, one composition, every rotation) as the hub queues
    it: ``lineups`` carry the block's seed, id and uids; ``players`` every player its seats name; ``owner`` the pulse
    snapshot (``pulse:<agent>:<checkpoint id>``) or tournament (``tournament:<name>``) it belongs to; ``priority``
    ``TOURNAMENT_PRIORITY`` runs before ``PULSE_PRIORITY``; ``num_envs`` the envs an evaluator plays its lease
    with. On a node (``block_task_from_proto``) ``owner`` is "" and ``priority`` 0."""

    block_id: str
    source: Literal["pulse", "tournament"]
    owner: str
    lineups: tuple[Lineup, ...]
    players: tuple[PlayerRef, ...]
    priority: int
    deterministic: bool
    num_envs: int = 8

    @property
    def matches(self) -> int:
        return len(self.lineups)


def _json(value: Any, what: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what} is not JSON ({e}); hub-run evaluation sends it to nodes as JSON") from e


def player_ref_to_proto(ref: PlayerRef) -> pb.PlayerRefProto:
    return pb.PlayerRefProto(
        name=ref.name, agent_id=ref.agent_id, network_id=ref.network_id, roles=list(ref.roles), kind=ref.kind,
        blob_id="" if ref.blob is None else ref.blob.ident,
        networks_json="" if ref.networks is None else _json(ref.networks, f"the networks of {ref.name}"),
        bot_class="" if ref.bot is None else ref.bot.class_path,
        bot_kwargs_json="" if ref.bot is None else _json(dict(ref.bot.kwargs), f"the bot kwargs of {ref.name}"))


def player_ref_from_proto(p: pb.PlayerRefProto) -> PlayerRef:
    return PlayerRef(
        name=p.name, agent_id=p.agent_id, network_id=p.network_id, roles=tuple(p.roles), kind=p.kind,
        blob=CacheKey("blob", p.agent_id, p.blob_id) if p.blob_id else None,
        networks=json.loads(p.networks_json) if p.networks_json else None,
        bot=BotSpec(p.bot_class, json.loads(p.bot_kwargs_json) if p.bot_kwargs_json else {}) if p.bot_class else None)


def block_task_to_proto(task: BlockTask) -> pb.BlockSpec:
    return pb.BlockSpec(block_id=task.block_id, source=task.source,
                        lineups=[lineup_to_proto(lu) for lu in task.lineups],
                        players=[player_ref_to_proto(p) for p in task.players], deterministic=task.deterministic)


def block_task_from_proto(p: pb.BlockSpec, *, num_envs: int) -> BlockTask:
    return BlockTask(block_id=p.block_id, source=p.source, owner="",
                     lineups=tuple(lineup_from_proto(x) for x in p.lineups),
                     players=tuple(player_ref_from_proto(x) for x in p.players), priority=0,
                     deterministic=p.deterministic, num_envs=int(num_envs))


def lease_to_proto(lease_id: str, lease_sec: float, tasks: Sequence[BlockTask]) -> pb.BlockLease:
    return pb.BlockLease(lease_id=lease_id, lease_sec=float(lease_sec), blocks=[block_task_to_proto(t) for t in tasks],
                         num_envs=tasks[0].num_envs if tasks else 0)
```
(`net.messages` importing `node.weights.CacheKey` follows `net.local`, which already does; `node.weights` is a pure library.)

Append to `src/colosseum/net/local.py` (the control messages section; module docstring: "node <-> evaluator" bullets):
```python
from colosseum.net.messages import BlockTask


# node -> evaluator (T5.1)
@dataclass
class EvalTask:
    """Play these whole blocks (every cache key of their players is in the machine cache) and report them."""

    lease_id: str
    source: str
    blocks: list[BlockTask]
    num_envs: int
    deterministic: bool


@dataclass
class EvalCancel:
    """The hub withdrew the lease: stop playing it (the evaluator answers ``EvalDone(cancelled=True)``)."""

    lease_id: str


# evaluator -> node (T5.1)
@dataclass
class EvalDone:
    """The lease's results by block id (all its blocks), or ``cancelled`` (nothing to report)."""

    lease_id: str
    results: dict[str, list[MatchResult]] = field(default_factory=dict)
    cancelled: bool = False


@dataclass
class EvalFailed:
    """The lease failed with a typed error; the evaluator keeps running (a crash is the process exit instead)."""

    lease_id: str
    kind: Literal["contract", "player", "config", "missing_weights"]
    message: str
    where: str
```

- [ ] **Step 4 (a): `colosseum.hub.evaluation` — the block queue, the result check and the evaluation service**

Create `src/colosseum/hub/evaluation.py` (T5.1 (a) content; (b) appends `HubPulse`, T5.2 the tournaments):
```python
"""Evaluation on nodes (SP5 spec 5.3): block leases, the evaluation service of the hub, the pulse on blocks (T5.1),
hub-run tournaments (T5.2).

- ``BlockQueue`` (pure, the hub's monotonic clock passed in): every evaluation block from submission to its
  result (class docstring).
- ``check_block_results``: a block counts only whole.
- ``HubEvaluation`` (main loop only; no gRPC, no threads): the queue, the owners of blocks (the pulse, hub-run
  tournaments), how many evaluators each node runs, and the lease protocol on the sessions:
  ``BlockLeaseRequest(free_slots)`` (an increment: slots the node has not asked for yet) -> ``BlockLease`` (one
  lease per grant, round robin over the asking nodes); ``BlockResult`` per block -> ``BlockAck`` always (the node
  forgets it; a result is never resent once acked); ``Hello.leases`` on a resumed session: leases the node no
  longer holds go back to the queue at once, leases the hub does not know are cancelled (``cancel_leases``); a new
  node process (fresh session) or node loss: its leases go back without using an attempt; ``set_evaluators``: a
  node runs its evaluator slots while any evaluation work can exist (pulse on, or an owner registered), else 0.
- Results of a pulse block go to ``HubPulse``; when its snapshot is complete, to the journal writer only (SP4
  amendment A20). Results of other owners go to their sink (T5.2). Every result is stamped with the node number
  of the evaluator that played it. A block that fails with a typed error (``FATAL_ERRORS``) fails its owner (the
  pulse: the run stops with exit 1, as an SP4 pulse crash did; a tournament: it fails); a block that fails
  ``MAX_BLOCK_ATTEMPTS`` times fails its owner too (the pulse: the snapshot is skipped with a WARNING).
"""

from __future__ import annotations

import contextlib
import heapq
import itertools
import logging
import math
import os
import shutil
import time
import uuid
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from colosseum.arena.tournament.engine import TASK_MATCHES
from colosseum.arena.tournament.players import TournamentPlayer
from colosseum.core.types import MatchResult
from colosseum.net.messages import (
    PULSE_PRIORITY,
    TOURNAMENT_PRIORITY,
    BlockTask,
    PlayerRef,
    lease_to_proto,
    result_from_proto,
)
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node.weights import CacheKey

__all__ = ["FATAL_ERRORS", "MAX_BLOCK_ATTEMPTS", "PULSE_PREFIX", "PULSE_PRIORITY", "RETRY_ERRORS", "TOURNAMENT_PREFIX",
           "TOURNAMENT_PRIORITY", "BlockQueue", "BlockTask", "EvalBlobs", "HubEvaluation", "JoinReply", "Lease",
           "PlayerRef", "check_block_results", "player_ref", "pulse_owner"]

logger = logging.getLogger(__name__)

MAX_BLOCK_ATTEMPTS = 3          # leases of one block that may end without a result (expiry, crash, bad data)
FINISHED_MEMORY = 100_000       # finished block ids remembered to tell "late" / "cancelled" from "unknown"
COUNTER_KEYS = ("submitted", "leases", "leased", "completed", "late", "cancelled", "unknown", "expired", "requeued",
                "failed", "cancelled_blocks")   # BlockQueue.stats(): every key always present (stable system record)
PULSE_PREFIX = "pulse:"
TOURNAMENT_PREFIX = "tournament:"
RETRY_ERRORS = ("abandoned", "crash", "missing_weights")     # BlockResult.error_kind values that re-lease the block
FATAL_ERRORS = ("contract", "player", "config")               # the code is wrong: replaying cannot help


def pulse_owner(agent_id: str, checkpoint_id: str) -> str:
    return f"{PULSE_PREFIX}{agent_id}:{checkpoint_id}"


@dataclass(frozen=True)
class Lease:
    lease_id: str
    node: int
    block_ids: tuple[str, ...]
    deadline: float                 # hub monotonic clock


@dataclass
class _Entry:
    task: BlockTask
    seq: int
    state: Literal["queued", "leased", "done"] = "queued"
    lease: str | None = None
    attempts: int = 0
    issued: set[str] = field(default_factory=set)


@dataclass
class _LeaseState:
    lease: Lease
    owner: str
    remaining: set[str]


class BlockQueue:
    """Evaluation blocks waiting for, or leased to, node evaluators (spec 5.3). Pure.

    - ``submit``: blocks queue by ``(priority, submission order)``: tournaments (0) before pulse (1), FIFO within.
    - ``lease(node, slots)``: up to ``slots`` leases; a lease is consecutive queued blocks of ONE owner with at
      least ``task_matches * num_envs`` matches (SP4 ``make_tasks``), so an evaluator's envs are rarely half empty.
    - ``want`` / ``dispatch``: nodes ask for slots (an increment); ``dispatch`` grants one lease per waiting node in
      turn (round robin by node number, continuing after the last node served), so blocks spread over machines.
    - ``complete``: the first complete result of any lease that held the block wins ("accepted"; also after that
      lease expired), later ones are "late"; a cancelled block's are "cancelled"; a block (or lease) never issued
      "unknown". Every outcome is counted.
    - ``release`` / ``expire`` / ``drop_node`` / ``keep_only``: a block whose lease ended without a result goes back
      to its place in the queue; ``expire`` and counted releases (an evaluator crash, missing weights, invalid
      results) use one of ``max_attempts`` attempts, after which the block fails (``take_failed``); node loss and a
      node's lost leases do not count (the network, not the block).
    - ``cancel(owner)``: the owner's queued and leased blocks are gone; returns the block ids and the leases per
      node to tell the nodes about.
    """

    def __init__(self, *, task_matches: int = TASK_MATCHES, max_attempts: int = MAX_BLOCK_ATTEMPTS,
                 new_id: Callable[[], str] | None = None) -> None:
        self._task_matches = int(task_matches)
        self._max_attempts = int(max_attempts)
        self._new_id = new_id or (lambda: uuid.uuid4().hex)
        self._entries: dict[str, _Entry] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._leases: dict[str, _LeaseState] = {}
        self._finished: OrderedDict[str, str] = OrderedDict()
        self._failed: list[tuple[str, str, str]] = []
        self._waiting: dict[int, int] = {}
        self._cursor = -1
        self._seq = itertools.count()
        self.counters: Counter[str] = Counter()

    # --- submission ---
    def submit(self, tasks: Sequence[BlockTask]) -> None:
        tasks = list(tasks)
        seen: set[str] = set()
        for task in tasks:
            if task.block_id in self._entries or task.block_id in seen:
                raise ValueError(f"block {task.block_id} is already queued or leased")
            if not task.lineups:
                raise ValueError(f"block {task.block_id} has no lineups")
            seen.add(task.block_id)
        for task in tasks:
            self._finished.pop(task.block_id, None)          # a cancelled block queued again (a resumed tournament)
            entry = _Entry(task, next(self._seq))
            self._entries[task.block_id] = entry
            heapq.heappush(self._heap, (task.priority, entry.seq, task.block_id))
        self.counters["submitted"] += len(tasks)

    def _peek(self) -> _Entry | None:
        while self._heap:
            _priority, seq, block_id = self._heap[0]
            entry = self._entries.get(block_id)
            if entry is not None and entry.state == "queued" and entry.seq == seq:
                return entry
            heapq.heappop(self._heap)                          # stale: leased, done, cancelled or queued again
        return None

    # --- leases ---
    def lease(self, node: int, slots: int, now: float, lease_sec: float) -> list[tuple[Lease, list[BlockTask]]]:
        out: list[tuple[Lease, list[BlockTask]]] = []
        for _ in range(max(0, int(slots))):
            head = self._peek()
            if head is None:
                break
            owner = head.task.owner
            target = max(1, self._task_matches * int(head.task.num_envs))
            taken: list[_Entry] = []
            matches = 0
            while matches < target:
                entry = self._peek()
                if entry is None or entry.task.owner != owner:
                    break
                heapq.heappop(self._heap)
                taken.append(entry)
                matches += entry.task.matches
            lease = Lease(self._new_id(), int(node), tuple(e.task.block_id for e in taken),
                          float(now) + float(lease_sec))
            for entry in taken:
                entry.state, entry.lease = "leased", lease.lease_id
                entry.issued.add(lease.lease_id)
            self._leases[lease.lease_id] = _LeaseState(lease, owner, set(lease.block_ids))
            self.counters["leases"] += 1
            self.counters["leased"] += len(taken)
            out.append((lease, [e.task for e in taken]))
        return out

    def want(self, node: int, slots: int) -> None:
        if int(slots) > 0:
            self._waiting[int(node)] = self._waiting.get(int(node), 0) + int(slots)

    def unwant(self, node: int) -> None:
        self._waiting.pop(int(node), None)

    def waiting(self) -> dict[int, int]:
        return dict(self._waiting)

    def dispatch(self, now: float, lease_sec: float) -> list[tuple[int, Lease, list[BlockTask]]]:
        out: list[tuple[int, Lease, list[BlockTask]]] = []
        while self._waiting and self._peek() is not None:
            nodes = sorted(self._waiting)
            node = next((n for n in nodes if n > self._cursor), nodes[0])
            self._cursor = node
            (lease, tasks), = self.lease(node, 1, now, lease_sec)
            out.append((node, lease, tasks))
            self._waiting[node] -= 1
            if self._waiting[node] <= 0:
                del self._waiting[node]
        return out

    # --- outcomes ---
    def task(self, block_id: str) -> BlockTask | None:
        """The block's task while it is queued or leased (None: done, cancelled, failed or unknown)."""
        entry = self._entries.get(block_id)
        return entry.task if entry is not None and entry.state != "done" else None

    def complete(self, lease_id: str, block_id: str, results: Sequence[MatchResult],
                 now: float) -> Literal["accepted", "late", "cancelled", "unknown"]:
        entry = self._entries.get(block_id)
        if entry is None:
            outcome = self._finished.get(block_id, "unknown")
        elif lease_id not in entry.issued:
            outcome = "unknown"
        elif entry.state == "done":
            outcome = "late"
        else:
            if entry.state == "leased":
                self._drop_from_lease(entry.lease, block_id)
            self._drop_from_lease(lease_id, block_id)
            entry.state, entry.lease = "done", None
            outcome = "accepted"
        self.counters["completed" if outcome == "accepted" else outcome] += 1
        return outcome

    def _drop_from_lease(self, lease_id: str | None, block_id: str) -> None:
        state = self._leases.get(lease_id) if lease_id is not None else None
        if state is not None:
            state.remaining.discard(block_id)
            if not state.remaining:
                del self._leases[lease_id]

    def release(self, lease_id: str, block_id: str, *, count: bool = True,
                reason: str = "released") -> Literal["requeued", "failed", "ignored"]:
        entry = self._entries.get(block_id)
        if entry is None or entry.state != "leased" or entry.lease != lease_id:
            return "ignored"
        self._drop_from_lease(lease_id, block_id)
        return self._requeue(entry, count=count, reason=reason)

    def _requeue(self, entry: _Entry, *, count: bool, reason: str) -> Literal["requeued", "failed"]:
        if count:
            entry.attempts += 1
        if entry.attempts >= self._max_attempts:
            del self._entries[entry.task.block_id]
            self._remember(entry.task.block_id, "late")
            self._failed.append((entry.task.block_id, entry.task.owner, reason))
            self.counters["failed"] += 1
            return "failed"
        entry.state, entry.lease = "queued", None
        heapq.heappush(self._heap, (entry.task.priority, entry.seq, entry.task.block_id))
        self.counters["requeued"] += 1
        return "requeued"

    def _release_leases(self, lease_ids: Sequence[str], *, count: bool, reason: str) -> list[str]:
        out: list[str] = []
        for lease_id in lease_ids:
            state = self._leases.pop(lease_id)
            for block_id in sorted(state.remaining):
                entry = self._entries.get(block_id)
                if entry is not None and entry.state == "leased" and entry.lease == lease_id and \
                        self._requeue(entry, count=count, reason=reason) == "requeued":
                    out.append(block_id)
        return out

    def expire(self, now: float) -> list[str]:
        """Leases past their deadline: their blocks go back to the queue (one attempt each); returns their ids."""
        due = [lid for lid, s in self._leases.items() if s.lease.deadline <= now]
        self.counters["expired"] += len(due)
        return self._release_leases(due, count=True, reason="its lease expired")

    def drop_node(self, node: int) -> list[str]:
        self.unwant(node)
        return self._release_leases([lid for lid, s in self._leases.items() if s.lease.node == node], count=False,
                                    reason="node lost")

    def keep_only(self, node: int, lease_ids: Sequence[str]) -> list[str]:
        """A node's session resumed: the leases it no longer holds (``Hello.leases``) go back to the queue."""
        keep = set(lease_ids)
        return self._release_leases([lid for lid, s in self._leases.items()
                                     if s.lease.node == node and lid not in keep],
                                    count=False, reason="the node no longer holds the lease")

    def cancel(self, owner: str) -> tuple[list[str], dict[int, list[str]]]:
        blocks = sorted(bid for bid, e in self._entries.items() if e.task.owner == owner and e.state != "done")
        leases: dict[int, list[str]] = {}
        for lease_id, state in list(self._leases.items()):
            if state.owner == owner:
                del self._leases[lease_id]
                leases.setdefault(state.lease.node, []).append(lease_id)
        for block_id in blocks:
            del self._entries[block_id]
            self._remember(block_id, "cancelled")
        self.forget(owner)
        self.counters["cancelled_blocks"] += len(blocks)
        return blocks, leases

    def forget(self, owner: str) -> None:
        """Done blocks of a finished owner: remembered by id only (their later results are "late")."""
        for block_id in [b for b, e in self._entries.items() if e.task.owner == owner and e.state == "done"]:
            del self._entries[block_id]
            self._remember(block_id, "late")

    def _remember(self, block_id: str, what: str) -> None:
        self._finished[block_id] = what
        self._finished.move_to_end(block_id)
        while len(self._finished) > FINISHED_MEMORY:
            self._finished.popitem(last=False)

    def take_failed(self) -> list[tuple[str, str, str]]:
        """``(block id, owner, reason)`` of blocks that used up their attempts since the last call."""
        out, self._failed = self._failed, []
        return out

    # --- views ---
    def leases_of(self, node: int) -> list[str]:
        return sorted(lid for lid, s in self._leases.items() if s.lease.node == node)

    def queued(self, owner_prefix: str = "") -> int:
        return sum(1 for e in self._entries.values() if e.state == "queued" and e.task.owner.startswith(owner_prefix))

    def live(self, owner: str) -> int:
        return sum(1 for e in self._entries.values() if e.task.owner == owner and e.state != "done")

    def stats(self) -> dict[str, int]:
        states = Counter(e.state for e in self._entries.values())
        return {"queued": states["queued"], "leased_now": states["leased"], "leases_now": len(self._leases),
                **{k: int(self.counters[k]) for k in COUNTER_KEYS}}


def check_block_results(task: BlockTask, results: Sequence[MatchResult]) -> str | None:
    """None when ``results`` are exactly the block's matches: one per lineup (by uid, each once), with the block's
    id, the lineup's layout and seed, the block's source, one seat result per seat naming the lineup's player, a
    known outcome kind and finite numbers. Else a short reason (the block is played again)."""
    lineups = {lu.uid: lu for lu in task.lineups}
    if len(results) != len(lineups):
        return f"{len(results)} results for {len(lineups)} matches"
    seen: set[str] = set()
    for r in results:
        lineup = lineups.get(r.uid)
        if lineup is None or r.uid in seen:
            return f"match uid {r.uid!r} is not one of the block's or repeats"
        seen.add(r.uid)
        if r.block != task.block_id or r.layout != lineup.layout or r.seed != lineup.seed or r.source != task.source:
            return f"match {r.uid}: block, layout, seed or source differ from the lineup"
        if r.outcome_kind not in ("score", "wdl", "rank") or r.episode_length < 0 or not r.teams:
            return f"match {r.uid}: outcome kind, length or teams are malformed"
        if sorted(s.seat for s in r.seats) != list(range(len(lineup.seats))):
            return f"match {r.uid}: seats {sorted(s.seat for s in r.seats)} for {len(lineup.seats)} seats"
        for s in r.seats:
            seat = lineup.seats[s.seat]
            if (s.agent_id, s.network_id) != (seat.agent_id, seat.network_id):
                return f"match {r.uid}: seat {s.seat} is {s.agent_id}/{s.network_id}, not the lineup's player"
        numbers = [s.reward for s in r.seats] + [x for t in r.teams for x in (t.rank, t.score)]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in numbers):
            return f"match {r.uid}: non-finite rewards, ranks or scores"
    return None


class EvalBlobs(Protocol):
    """``BlobStore``'s evaluation blobs (T5.1 (b)): served by ``Download(kind="eval", ident=blob_id)``."""

    def register_weights(self, blob_id: str, weights_file: Path, *, agent_id: str, networks: Mapping[str, Any],
                         roles: Sequence[str]) -> None: ...

    def unregister(self, blob_id: str) -> None: ...


def player_ref(player: TournamentPlayer, blob_id: str | None) -> PlayerRef:
    """The evaluator's view of a tournament / pulse player (a neural one needs its registered ``blob_id``)."""
    if player.source.kind == "bot":
        return PlayerRef(player.name, player.agent_id, player.network_id, tuple(player.roles), "scripted",
                         bot=player.source.bot)
    if not blob_id:
        raise ValueError(f"player {player.name!r}: a neural player needs a blob id")
    return PlayerRef(player.name, player.agent_id, player.network_id, tuple(player.roles), "neural",
                     blob=CacheKey("blob", player.agent_id, blob_id), networks=dict(player.source.networks or {}))


@dataclass(frozen=True)
class JoinReply:
    evaluators: int                               # Welcome.evaluators
    after_welcome: list                           # HubMessages to send right after the Welcome (cancel_leases)


def cancel_leases_message(lease_ids: Sequence[str]) -> pb.HubMessage:
    return pb.HubMessage(command=pb.Command(kind="cancel_leases", lease_ids=list(lease_ids)))


class HubEvaluation:
    """The hub's evaluation service (module docstring). ``send(node, HubMessage)`` puts a message on that node's
    session (``SessionHandle.send``; a message to a node without a session is dropped: its leases are reconciled
    by ``Hello.leases`` when it returns). ``journal(results)`` is the journal writer's ``submit`` (pulse results);
    ``fail_run(text)`` stops the run with exit 1 (a typed error in a pulse block)."""

    def __init__(self, *, lease_sec: float, send: Callable[[int, pb.HubMessage], None], pulse: Any = None,
                 journal: Callable[[list[MatchResult]], int] | None = None,
                 fail_run: Callable[[str], None] | None = None, clock: Callable[[], float] = time.monotonic,
                 queue: BlockQueue | None = None) -> None:
        self.queue = queue if queue is not None else BlockQueue()
        self.pulse = pulse                       # HubPulse (T5.1 (b)) or None
        self._lease_sec = float(lease_sec)
        self._send, self._journal, self._fail_run, self._clock = send, journal, fail_run, clock
        self._slots: dict[int, int] = {}
        self._running: dict[int, int] = {}
        self._sinks: dict[str, Callable[[str, str, Any], None]] = {}
        self._warned_invalid: set[int] = set()
        self._waiting_logged = False
        self._dispatching = True
        self.invalid_results = 0
        self.pulse_results = 0                    # pulse matches given to the journal writer

    # --- demand: how many evaluators each node runs ---
    @property
    def active(self) -> bool:
        return (self.pulse is not None and self.pulse.accepting) or bool(self._sinks)

    def evaluators_for(self, node: int) -> int:
        return self._slots.get(node, 0) if self.active else 0

    def _update_demand(self) -> None:
        for node in sorted(self._slots):
            wanted = self.evaluators_for(node)
            if self._running.get(node) != wanted:
                self._running[node] = wanted
                self._send(node, pb.HubMessage(command=pb.Command(kind="set_evaluators", evaluators=wanted)))

    # --- node lifecycle ---
    def node_joined(self, node: int, *, slots: int, held_leases: Sequence[str], fresh: bool) -> JoinReply:
        self._slots[node] = max(0, int(slots))
        self.queue.unwant(node)
        if fresh:
            self.queue.drop_node(node)                  # a new node process: its old evaluators are gone
        else:
            self.queue.keep_only(node, held_leases)
        unknown = sorted(set(held_leases) - set(self.queue.leases_of(node)))   # an earlier hub's, or cancelled
        self._running[node] = wanted = self.evaluators_for(node)
        self._dispatch()
        return JoinReply(wanted, [cancel_leases_message(unknown)] if unknown else [])

    def session_closed(self, node: int) -> None:
        self.queue.unwant(node)                          # its leases stay until it returns or is lost

    def node_lost(self, node: int) -> None:
        self.queue.drop_node(node)
        self._slots.pop(node, None)
        self._running.pop(node, None)
        self._dispatch()

    # --- messages from nodes ---
    def on_lease_request(self, node: int, free_slots: int) -> None:
        if node in self._slots and self._dispatching:
            self.queue.want(node, free_slots)
            self._dispatch()

    def on_block_result(self, node: int, msg: pb.BlockResult) -> None:
        self._send(node, pb.HubMessage(block_ack=pb.BlockAck(lease_id=msg.lease_id, block_ids=[msg.block_id])))
        task = self.queue.task(msg.block_id)
        if msg.failed:
            if task is not None:
                self._on_failed(node, msg, task)
            return
        try:
            results = [result_from_proto(p) for p in msg.results]
        except Exception as e:  # noqa: BLE001 - node data is untrusted (spec 7.1): counted, never fatal
            results, reason = [], f"malformed results ({type(e).__name__})"
        else:
            reason = check_block_results(task, results) if task is not None else None
        if task is not None and reason is not None:
            self.invalid_results += 1
            if node not in self._warned_invalid:
                self._warned_invalid.add(node)
                logger.warning(f"evaluation: node {node} sent invalid results for block {msg.block_id} ({reason}); "
                               f"the block is played again (further invalid results of this node are only counted)")
            self.queue.release(msg.lease_id, msg.block_id, count=True, reason=f"invalid results ({reason})")
            self._after_change()
            return
        if self.queue.complete(msg.lease_id, msg.block_id, results, self._clock()) != "accepted":
            return
        for result in results:
            result.node = node
        self._deliver(task, results)
        self._after_change()

    def _on_failed(self, node: int, msg: pb.BlockResult, task: BlockTask) -> None:
        if msg.error_kind in FATAL_ERRORS:
            self._owner_failed(task.owner, msg.error_kind,
                               f"{task.source} block {task.block_id} failed on node {node}: {msg.error}")
        else:
            self.queue.release(msg.lease_id, msg.block_id, count=msg.error_kind != "abandoned",
                               reason=f"{msg.error_kind or 'failed'} on node {node}")
        self._after_change()

    def _owner_failed(self, owner: str, kind: str, text: str) -> None:
        self.cancel(owner)
        if owner.startswith(PULSE_PREFIX):
            if self.pulse is not None:
                self.pulse.on_failed(owner, text)
            if kind in FATAL_ERRORS and self._fail_run is not None:
                self._fail_run(text)                     # SP4: a pulse error stops the run (exit 1)
        else:
            sink = self._sinks.get(owner)
            if sink is not None:
                sink("failed", kind, text)

    def _deliver(self, task: BlockTask, results: list[MatchResult]) -> None:
        if task.owner.startswith(PULSE_PREFIX):
            played = self.pulse.on_block(task.block_id, results) if self.pulse is not None else None
            if played is not None:
                self.queue.forget(task.owner)
                self.pulse_results += len(played)
                if self._journal is not None:
                    self._journal(played)                # A20: the journal only; the writer counts its drops
        else:
            sink = self._sinks.get(task.owner)
            if sink is not None:
                sink("block", task.block_id, results)

    # --- owners other than the pulse (T5.2) ---
    def add_owner(self, owner: str, sink: Callable[[str, str, Any], None]) -> None:
        self._sinks[owner] = sink
        self._update_demand()

    def submit(self, tasks: Sequence[BlockTask]) -> None:
        self.queue.submit(tasks)
        self._dispatch()

    def remove_owner(self, owner: str) -> None:
        self.cancel(owner)
        self._sinks.pop(owner, None)
        self._update_demand()

    def cancel(self, owner: str) -> list[str]:
        blocks, leases = self.queue.cancel(owner)
        for node, lease_ids in sorted(leases.items()):
            self._send(node, cancel_leases_message(lease_ids))
        return blocks

    # --- main loop ---
    def tick(self) -> None:
        """Every main-loop pass: lease expiry, failed blocks, the next pulse snapshot, dispatch, the waiting INFO."""
        self.queue.expire(self._clock())
        self._handle_failed()
        if self.pulse is not None and self._dispatching and self.queue.queued(PULSE_PREFIX) == 0:
            tasks = self.pulse.start_next()      # the next snapshot once the last one's blocks are all leased
            if tasks:
                self.queue.submit(tasks)
        self._dispatch()
        self._log_waiting()

    def _after_change(self) -> None:
        self._handle_failed()
        self._dispatch()

    def _handle_failed(self) -> None:
        for block_id, owner, reason in self.queue.take_failed():
            self._owner_failed(owner, "attempts", f"block {block_id} failed {MAX_BLOCK_ATTEMPTS} times (last: {reason})")

    def _dispatch(self) -> None:
        if not self._dispatching:
            return
        for node, lease, tasks in self.queue.dispatch(self._clock(), self._lease_sec):
            self._send(node, pb.HubMessage(lease=lease_to_proto(lease.lease_id, self._lease_sec, tasks)))

    def _log_waiting(self) -> None:
        waiting = self.queue.queued() > 0 and sum(self._running.values()) == 0
        if waiting and not self._waiting_logged:
            logger.info(f"evaluation: {self.queue.queued()} blocks wait: no connected node runs an evaluator (nodes "
                        f"with evaluator slots take them: colosseum join --evaluators N)")
        self._waiting_logged = waiting

    def stop(self) -> None:
        """The run stops: no new lease, no new pulse snapshot (leased blocks may still report until ``finish``)."""
        self._dispatching = False
        if self.pulse is not None:
            self.pulse.stop()

    def finish(self) -> None:
        """Before the journal writer stops: snapshots still in flight stay unevaluated; their blocks are cancelled."""
        if self.pulse is not None:
            for owner in self.pulse.finish():
                self.cancel(owner)

    def stats(self) -> dict[str, Any]:
        """The ``system`` record's ``evaluation`` value."""
        return {**self.queue.stats(), "invalid_results": self.invalid_results,
                "evaluators": sum(self._running.values()), "pulse_results": self.pulse_results}
```
(`contextlib`, `os`, `shutil` and the `TOURNAMENT_PRIORITY` re-export are used by (b) and T5.2; ruff's unused-import check runs on the finished module — drop what (a) does not use and let (b) add it back.)

- [ ] **Step 5 (a): GREEN for the messages, the queue and the service**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_eval_messages.py tests/unit/test_sp5_block_queue.py tests/unit/test_sp5_evaluation_service.py -q`
Expected: all pass.

- [ ] **Step 6 (a): Write the failing tests of the node's leases and the evaluator; the evaluation test kit**

Create `tests/unit/test_sp5_node_leases.py`:
```python
"""The node's evaluation bookkeeping (SP5 T5.1): slot requests as increments, leases waiting for their weights,
assignments to idle evaluators, outcomes kept until acked and resent after a reconnect, dead and stopped
evaluators, failed fetches, cancels, the evaluation blobs kept in the machine cache."""
from __future__ import annotations

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.net.messages import BlockTask, PlayerRef
from colosseum.node.leases import EVAL_BLOBS_KEPT, NodeLease, NodeLeases
from colosseum.node.weights import CacheKey

KEY = CacheKey("blob", "main", "pulse-main-ckpt_v3")


def _lease(lease_id, key=KEY, blocks=("b0",)):
    ref = PlayerRef(key.ident, key.agent_id, "ckpt_v3", ("player",), "neural", blob=key, networks={})
    tasks = [BlockTask(b, "pulse", "", (Lineup("2p", [SeatAssignment("main", "ckpt_v3")], seed=1, block=b,
                                               uid=f"{b}/r0"),), (ref,), 0, False, 2) for b in blocks]
    return NodeLease(lease_id, "pulse", tasks, 2, False)


def _two_idle():
    nl = NodeLeases()
    nl.evaluator_up(0)
    nl.evaluator_up(1)
    return nl


def test_request_asks_for_idle_evaluators_once():
    nl = _two_idle()
    assert nl.request(connected=False) == 0
    assert nl.request(connected=True) == 2 and nl.request(connected=True) == 0
    nl.on_lease(_lease("L1"), have=lambda k: True)
    assert nl.request(connected=True) == 0                     # 2 idle - 1 waiting lease - 1 still asked


def test_a_lease_waits_for_its_weights_then_an_idle_evaluator():
    nl = _two_idle()
    assert nl.on_lease(_lease("L1"), have=lambda k: False) == [KEY]
    assert nl.assignments() == []
    assert nl.on_fetched(KEY, True) == []
    ((index, lease),) = nl.assignments()
    assert (index, lease.lease_id, lease.state) == (0, "L1", "playing") and nl.idle() == [1]


def test_outcomes_stay_unacked_until_the_hub_acks_them():
    nl = _two_idle()
    nl.on_lease(_lease("L1", blocks=("b0", "b1")), have=lambda k: True)
    nl.assignments()
    outcomes = nl.on_done(0, "L1", {"b0": ["r0"], "b1": ["r1"]}, cancelled=False)
    assert [(o.block_id, o.results, o.failed) for o in outcomes] == [("b0", ["r0"], False), ("b1", ["r1"], False)]
    held, resend = nl.on_reconnect()
    assert held == ["L1"] and len(resend) == 2
    nl.on_ack("L1", ["b0", "b1"])
    assert nl.on_reconnect() == ([], [])


def test_done_without_a_block_reports_a_crash_for_it():
    nl = _two_idle()
    nl.on_lease(_lease("L1", blocks=("b0", "b1")), have=lambda k: True)
    nl.assignments()
    outcomes = nl.on_done(0, "L1", {"b0": ["r0"]}, cancelled=False)
    assert [(o.block_id, o.error_kind) for o in outcomes] == [("b0", ""), ("b1", "crash")]


def test_a_dead_or_stopped_evaluator_releases_its_lease():
    nl = _two_idle()
    nl.on_lease(_lease("L1"), have=lambda k: True)
    nl.on_lease(_lease("L2"), have=lambda k: True)
    nl.assignments()
    assert [(o.block_id, o.error_kind) for o in nl.evaluator_down(0)] == [("b0", "crash")]
    assert [(o.lease_id, o.error_kind) for o in nl.evaluator_down(1, error_kind="abandoned")] == [("L2", "abandoned")]
    assert nl.up() == [] and nl.evaluator_down(1) == []


def test_a_failed_fetch_reports_missing_weights():
    nl = _two_idle()
    nl.on_lease(_lease("L1"), have=lambda k: False)
    ((outcome,),) = [nl.on_fetched(KEY, False)]
    assert outcome.failed and outcome.error_kind == "missing_weights" and KEY.ident in outcome.error
    assert nl.assignments() == []


def test_cancel_frees_the_evaluator_only_when_it_answers():
    nl = _two_idle()
    nl.on_lease(_lease("L1"), have=lambda k: True)
    ((index, _lease_),) = nl.assignments()
    assert nl.on_cancel(["L1", "unknown"]) == [(index, "L1")] and nl.idle() == [1]
    assert nl.on_done(index, "L1", {}, cancelled=True) == [] and nl.idle() == [0, 1]
    assert nl.on_reconnect() == ([], [])


def test_unused_evaluation_blobs_beyond_the_newest_eight_are_dropped():
    nl = _two_idle()
    keys = [CacheKey("blob", "p", f"b{n}") for n in range(11)]
    for n, key in enumerate(keys):
        nl.on_lease(_lease(f"X{n}", key=key), have=lambda k: True)
        nl.on_cancel([f"X{n}"])
    nl.on_lease(_lease("live", key=keys[0]), have=lambda k: True)      # b0 is in use again (a live lease)
    drop = nl.drop_candidates()                                          # 10 unused: the oldest 10 - 8 go
    assert drop == keys[1:1 + 10 - EVAL_BLOBS_KEPT] and keys[0] not in drop
    assert nl.drop_candidates() == []
```

Create `tests/eval_kit.py` (test support; add `"eval_kit"` to `known-first-party`):
```python
"""Evaluation test kit (SP5 T5.1): toy-game evaluation blocks as ``BlockTask``s, the machine weight cache (real
``WeightCacheWriter`` / ``WeightCacheReader`` under the test's ``COLOSSEUM_SHM_DIR``) filled with their players'
weights, a ``FakeBlobs`` registry standing in for ``BlobStore``'s evaluation blobs, and ``EvalNode``: an in-process
node that leases blocks from a ``HubEvaluation`` and plays them on an evaluator thread (``evaluator_loop`` over a
``multiprocessing.Pipe``; no process). ``drive`` is the test's main loop."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from colosseum.arena.tournament.players import PlayerSource, TournamentPlayer, config_player
from colosseum.arena.tournament.schedule import gauntlet_blocks
from colosseum.coordinator.checkpoint_manager import read_weights_file
from colosseum.core.registry import env_spec, make_env
from colosseum.hub.evaluation import HubEvaluation, player_ref
from colosseum.net.chunks import WeightsMeta, encode_weights
from colosseum.net.local import EvalCancel, EvalDone, EvalTask, Stop
from colosseum.net.messages import BlockTask, block_task_from_proto, result_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node.evaluator_main import evaluator_loop
from colosseum.node.leases import NodeLease, NodeLeases
from colosseum.node.weights import CacheKey, WeightCacheReader, WeightCacheWriter, cache_prefix, shm_dir
from colosseum.players.registry import load_fixed_players
from game_helpers import agent_role_of, write_ckpt_dir

CA_FINGERPRINT = "ab" * 32


def snapshot_player(config, root: Path, agent: str, version: int, *, seed: int = 0) -> TournamentPlayer:
    """A stored snapshot ``root/ckpt_v<version>`` (``write_ckpt_dir``) as a tournament player."""
    path = write_ckpt_dir(root, config, agent, policy_version=version, seed=seed)
    roles, _role = agent_role_of(config, agent)
    networks = config.get_agent_config(agent).networks.model_dump(mode="json", by_alias=True)
    name = f"{agent}@v{version}"
    return TournamentPlayer(name=name, display=name, kind="snapshot", agent_id=agent, network_id=f"ckpt_v{version}",
                            roles=tuple(roles), lineage=agent, version=version,
                            source=PlayerSource("model", origin=str(path), weights=str(path / "model.pt"),
                                                networks=networks))


def fixed_players(config, names: Sequence[str]) -> dict[str, TournamentPlayer]:
    spec = env_spec(config)
    fixed = load_fixed_players(config, spec, list(names))
    return {n: config_player(config, spec, n, fixed) for n in names}


def gauntlet_tasks(config, candidate: TournamentPlayer, field: Mapping[str, TournamentPlayer], *, blocks: int,
                   owner: str = "tournament:t", priority: int = 0, num_envs: int = 2, seed: int = 0,
                   layout: str = "2p", blobs: Any = None) -> list[BlockTask]:
    """Stage-A gauntlet blocks of ``candidate`` against ``field`` as tasks of ``owner``; neural players get the blob
    id ``<owner name>-<agent>-<network>`` (registered with ``blobs`` when given)."""
    prefix = owner.split(":", 1)[1]
    spec = env_spec(config)
    players = {candidate.name: candidate, **field}
    sched = gauntlet_blocks(spec, layout, [candidate.name], list(field), players, blocks=blocks, seed=seed,
                            stage="A", prefix=prefix)
    refs = {}
    for p in players.values():
        blob_id = f"{prefix}-{p.agent_id}-{p.network_id}" if p.source.kind == "model" else None
        if blob_id is not None and blobs is not None:
            blobs.register_weights(blob_id, Path(p.source.weights), agent_id=p.agent_id,
                                   networks=dict(p.source.networks or {}), roles=p.roles)
        refs[(p.agent_id, p.network_id)] = player_ref(p, blob_id)
    source = "pulse" if owner.startswith("pulse:") else "tournament"
    return [BlockTask(b.id, source, owner, tuple(b.lineups),
                      tuple(refs[k] for k in sorted({(s.agent_id, s.network_id) for lu in b.lineups for s in lu.seats})),
                      priority, False, num_envs) for b in sched]


class FakeBlobs:
    """``EvalBlobs`` stand-in: ``live[blob_id] = (weights file, agent id, networks, roles)``; ``log`` of calls."""

    def __init__(self) -> None:
        self.live: dict[str, tuple[Path, str, dict, tuple[str, ...]]] = {}
        self.log: list[tuple[str, str]] = []

    def register_weights(self, blob_id, weights_file, *, agent_id, networks, roles) -> None:
        assert Path(weights_file).is_file(), weights_file
        self.live[blob_id] = (Path(weights_file), agent_id, dict(networks), tuple(roles))
        self.log.append(("register", blob_id))

    def unregister(self, blob_id) -> None:
        self.live.pop(blob_id, None)
        self.log.append(("unregister", blob_id))


class EvalCache:
    """The machine's weight cache of one fake node (prefix of ``node`` and ``run_id``)."""

    def __init__(self, node: int = 1, run_id: str = "eval-kit-run") -> None:
        self.root, self.prefix = shm_dir(), cache_prefix(CA_FINGERPRINT, run_id, node)
        self.writer = WeightCacheWriter(self.root, self.prefix)
        self.reader = WeightCacheReader(self.root, self.prefix)

    def put(self, key: CacheKey, weights_file: Path, networks: Mapping, roles: Sequence[str]) -> None:
        meta = WeightsMeta(agent_id=key.agent_id, kind="fixed", generation=0, policy_version=0,
                           architecture=dict(networks), roles=tuple(roles))
        self.writer.put(key, encode_weights(read_weights_file(weights_file), meta))

    def has(self, key: CacheKey) -> bool:
        return self.reader.has(key)


class EvalNode:
    """An in-process node: ``NodeLeases`` + one evaluator thread; ``inbox`` gets the hub's messages (``route``)."""

    def __init__(self, config, blobs: FakeBlobs, evaluation: HubEvaluation, *, number: int = 1,
                 slots: int = 1) -> None:
        self.number, self.blobs, self.ev = number, blobs, evaluation
        self.cache = EvalCache(number)
        self.leases = NodeLeases()
        self.inbox: list[pb.HubMessage] = []
        self.fetches = 0
        self.results_sent = 0
        self._conn, child = mp.Pipe()
        self._thread = threading.Thread(target=evaluator_loop, args=(child,), daemon=True, kwargs=dict(
            spec=env_spec(config), reader=self.cache.reader, env_fn=lambda: make_env(config),
            max_idle_steps=config.env.max_idle_steps, node=number, index=0))
        self._thread.start()
        self.leases.evaluator_up(0)
        self.slots = slots

    def join(self, *, fresh: bool = True) -> int:
        held, resend = self.leases.on_reconnect()
        reply = self.ev.node_joined(self.number, slots=self.slots, held_leases=held, fresh=fresh)
        self.inbox.extend(reply.after_welcome)
        for outcome in resend:
            self._report(outcome)
        return reply.evaluators

    def pump(self) -> None:
        free = self.leases.request(connected=True)
        if free:
            self.ev.on_lease_request(self.number, free)
        while self.inbox:
            msg = self.inbox.pop(0)
            kind = msg.WhichOneof("m")
            if kind == "lease":
                tasks = [block_task_from_proto(b, num_envs=msg.lease.num_envs) for b in msg.lease.blocks]
                lease = NodeLease(msg.lease.lease_id, tasks[0].source, tasks, msg.lease.num_envs,
                                  tasks[0].deterministic)
                for key in self.leases.on_lease(lease, have=self.cache.has):
                    entry = self.blobs.live.get(key.ident)
                    if entry is not None:
                        self.cache.put(key, entry[0], entry[2], entry[3])
                        self.fetches += 1
                    for outcome in self.leases.on_fetched(key, entry is not None):
                        self._report(outcome)
            elif kind == "block_ack":
                self.leases.on_ack(msg.block_ack.lease_id, list(msg.block_ack.block_ids))
            elif kind == "command" and msg.command.kind == "cancel_leases":
                for _index, lease_id in self.leases.on_cancel(list(msg.command.lease_ids)):
                    self._conn.send(EvalCancel(lease_id))
        for _index, lease in self.leases.assignments():
            self._conn.send(EvalTask(lease.lease_id, lease.source, lease.blocks, lease.num_envs, lease.deterministic))
        while self._conn.poll(0):
            m = self._conn.recv()
            outcomes = (self.leases.on_done(0, m.lease_id, m.results, m.cancelled) if isinstance(m, EvalDone)
                        else self.leases.on_failed(0, m.lease_id, m.kind, f"{m.message} ({m.where})"))
            for outcome in outcomes:
                self._report(outcome)

    def _report(self, outcome) -> None:
        msg = pb.BlockResult(lease_id=outcome.lease_id, block_id=outcome.block_id, failed=outcome.failed,
                             error_kind=outcome.error_kind, error=outcome.error)
        if outcome.results is not None:
            msg.results.extend(result_to_proto(r) for r in outcome.results)
        self.results_sent += 1
        self.ev.on_block_result(self.number, msg)

    def close(self) -> None:
        self._conn.send(Stop("test end"))
        self._thread.join(10)


def route(nodes: Mapping[int, EvalNode]) -> Callable[[int, pb.HubMessage], None]:
    """A ``HubEvaluation.send`` that puts messages in the nodes' inboxes (fill ``nodes`` after building the
    evaluation)."""
    return lambda number, msg: nodes[number].inbox.append(msg) if number in nodes else None


def drive(evaluation: HubEvaluation, nodes: Sequence[EvalNode], until: Callable[[], bool], *,
          inbox: Any = None, on_event: Callable[[Any], None] | None = None, timeout: float = 120.0) -> None:
    """The test's main loop: inbox events, ``evaluation.tick()``, every node's pump, until ``until()``."""
    deadline = time.monotonic() + timeout
    while not until():
        if time.monotonic() > deadline:
            raise AssertionError(f"drive: not done after {timeout} s ({evaluation.stats()})")
        while inbox is not None and not inbox.empty():
            on_event(inbox.get_nowait())
        evaluation.tick()
        for node in nodes:
            node.pump()
        time.sleep(0.005)
```

Create `tests/integration/test_sp5_evaluator.py`:
```python
"""The evaluator (SP5 T5.1): its loop in a thread over a Pipe (cheap) and one real process (connection to the
node listener, the shm cache, kill -9 without blocking the node side)."""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time

import pytest

from colosseum.core.registry import env_spec, make_env
from colosseum.hub.evaluation import check_block_results
from colosseum.net.local import ChildHello, EvalCancel, EvalDone, EvalFailed, EvalTask, Stop, listener, address
from colosseum.node.evaluator_main import CachePlayerPool, evaluator_loop, evaluator_target
from colosseum.utils.logging import ENV_PROCESS_NAME
from eval_kit import EvalCache, FakeBlobs, fixed_players, gauntlet_tasks, snapshot_player
from game_helpers import make_test_config, scripted_agent

AGENTS = {"main": {}, "rnd": scripted_agent(), "ones": scripted_agent("game_helpers.ConstantBot", value=1)}


def _cfg(agents=AGENTS):
    return make_test_config("turns", agents=agents)


def _filled(cfg, tasks, blobs, node=1):
    cache = EvalCache(node)
    for task in tasks:
        for p in task.players:
            if p.blob is not None and not cache.has(p.blob):
                path, _agent, networks, roles = blobs.live[p.blob.ident]
                cache.put(p.blob, path, networks, roles)
    return cache


def _loop(cfg, cache):
    parent, child = mp.Pipe()
    thread = threading.Thread(target=evaluator_loop, args=(child,), daemon=True, kwargs=dict(
        spec=env_spec(cfg), reader=cache.reader, env_fn=lambda: make_env(cfg),
        max_idle_steps=cfg.env.max_idle_steps, node=3, index=0))
    thread.start()
    return parent, thread


def _tasks(cfg, tmp_path, *, version=3, blocks=2, owner="tournament:t", field=("rnd", "ones"), blobs=None):
    cand = snapshot_player(cfg, tmp_path / f"snap{version}", "main", version, seed=version)
    return gauntlet_tasks(cfg, cand, fixed_players(cfg, field), blocks=blocks, owner=owner, blobs=blobs)


def test_the_evaluator_plays_whole_blocks_from_the_cache(tmp_path):
    cfg, blobs = _cfg(), FakeBlobs()
    tasks = _tasks(cfg, tmp_path, blobs=blobs)
    conn, thread = _loop(cfg, _filled(cfg, tasks, blobs))
    conn.send(EvalTask("L1", "tournament", tasks, 2, False))
    assert conn.poll(60)
    done = conn.recv()
    assert isinstance(done, EvalDone) and not done.cancelled and set(done.results) == {t.block_id for t in tasks}
    for task in tasks:
        assert check_block_results(task, done.results[task.block_id]) is None
    conn.send(Stop("end"))
    thread.join(10)
    assert not thread.is_alive()


def test_missing_weights_fail_the_lease_and_the_loop_goes_on(tmp_path):
    cfg, blobs = _cfg(), FakeBlobs()
    tasks = _tasks(cfg, tmp_path, blobs=blobs)
    other = _tasks(cfg, tmp_path, version=4, owner="tournament:u", blobs=blobs)    # not in the cache
    conn, thread = _loop(cfg, _filled(cfg, tasks, blobs))
    conn.send(EvalTask("L2", "tournament", other, 2, False))
    assert conn.poll(30)
    failed = conn.recv()
    assert isinstance(failed, EvalFailed) and failed.kind == "missing_weights" and "u-main-ckpt_v4" in failed.message
    conn.send(EvalTask("L3", "tournament", tasks, 2, False))
    assert conn.poll(60) and isinstance(conn.recv(), EvalDone)
    conn.send(Stop("end"))
    thread.join(10)


def test_a_cancel_stops_the_lease_mid_play(tmp_path):
    cfg, blobs = _cfg(), FakeBlobs()
    tasks = _tasks(cfg, tmp_path, blocks=40, blobs=blobs)
    conn, thread = _loop(cfg, _filled(cfg, tasks, blobs))
    conn.send(EvalTask("L1", "tournament", tasks, 2, False))
    conn.send(EvalCancel("L1"))
    assert conn.poll(30)
    answer = conn.recv()
    assert isinstance(answer, EvalDone) and answer.cancelled and answer.lease_id == "L1" and answer.results == {}
    conn.send(Stop("end"))
    thread.join(10)


def test_a_player_error_is_typed_and_the_loop_survives(tmp_path, monkeypatch):
    agents = {"main": {}, "bad": scripted_agent("game_helpers.PulseFailBot")}
    cfg, blobs = _cfg(agents), FakeBlobs()
    tasks = _tasks(cfg, tmp_path, blocks=1, field=("bad",), blobs=blobs)
    monkeypatch.setenv(ENV_PROCESS_NAME, "evaluator-0")         # PulseFailBot fails in evaluator processes
    conn, thread = _loop(cfg, _filled(cfg, tasks, blobs))
    conn.send(EvalTask("L1", "tournament", tasks, 1, False))
    assert conn.poll(60)
    failed = conn.recv()
    assert isinstance(failed, EvalFailed) and failed.kind == "player"
    assert "pulse failure on purpose" in failed.message and "node 3, evaluator 0" in failed.message
    assert failed.where.startswith(("match ", "block "))
    conn.send(Stop("end"))
    thread.join(10)
    assert not thread.is_alive()


def test_a_closed_connection_ends_the_loop(tmp_path):
    cfg, blobs = _cfg(), FakeBlobs()
    conn, thread = _loop(cfg, EvalCache())
    conn.close()
    thread.join(10)
    assert not thread.is_alive()


def test_the_pool_keys_models_by_blob(tmp_path):
    cfg, blobs = _cfg(), FakeBlobs()
    first = _tasks(cfg, tmp_path / "a", owner="tournament:a", blobs=blobs)
    second = _tasks(cfg, tmp_path / "b", owner="tournament:b", blobs=blobs)    # the same seat names, other weights
    cache = _filled(cfg, first + second, blobs)
    pool = CachePlayerPool(env_spec(cfg), cache.reader)
    pool.reserve(first[0].players)
    one = pool.get("main", "ckpt_v3")
    pool.reserve(second[0].players)
    two = pool.get("main", "ckpt_v3")
    assert one is not two and pool.loads == 2 and pool.get("nobody", "fixed") is None
    pool.reserve(first[0].players)
    assert pool.get("main", "ckpt_v3") is one and pool.loads == 2               # kept (LRU)


def test_the_evaluator_process_connects_plays_and_dies_without_blocking_the_node(tmp_path):
    """One real process: ChildHello on the node listener, a lease played from the shm cache, then kill -9 in the
    middle of a long lease: the node side sees EOF at once (no shared lock, spec 4.7 / 7.1)."""
    cfg, blobs = _cfg(), FakeBlobs()
    tasks = _tasks(cfg, tmp_path, blobs=blobs)
    long = _tasks(cfg, tmp_path, version=5, blocks=400, owner="tournament:long", blobs=blobs)
    cache = _filled(cfg, tasks + long, blobs)
    authkey = os.urandom(32)
    name = address("ab" * 4, "evalkit1", 1, "node")
    lst = listener(name, authkey)
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=evaluator_target, daemon=True, kwargs=dict(
        name="evaluator-0", index=0, incarnation=0, node=1, node_address=name, authkey=authkey,
        cache_root=str(cache.root), cache_prefix=cache.prefix, config=cfg, log_dir=str(tmp_path / "logs")))
    proc.start()
    try:
        conn = lst.accept()
        hello = conn.recv()
        assert isinstance(hello, ChildHello) and (hello.role, hello.channel, hello.name) == \
            ("evaluator", "control", "evaluator-0")
        conn.send(EvalTask("L1", "tournament", tasks, 2, False))
        assert conn.poll(120) and isinstance(conn.recv(), EvalDone)
        conn.send(EvalTask("L2", "tournament", long, 2, False))
        time.sleep(1.0)
        os.kill(proc.pid, signal.SIGKILL)
        start = time.monotonic()
        with pytest.raises((EOFError, ConnectionResetError, OSError)):
            while time.monotonic() - start < 10:
                if conn.poll(0.5):
                    conn.recv()
        assert time.monotonic() - start < 10
        proc.join(10)
        assert proc.exitcode == -signal.SIGKILL
        assert "evaluator-0" in (tmp_path / "logs" / "evaluator-0.log").read_text()
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        lst.close()
```
(Adapt the `address(...)` / `listener(...)` arguments to T2.2's final signatures; the test needs one node listener name and an authkey.)

- [ ] **Step 7 (a): Run them: RED**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_node_leases.py tests/integration/test_sp5_evaluator.py -q`
Expected: collection errors — `ModuleNotFoundError: No module named 'colosseum.node.leases'` / `'colosseum.node.evaluator_main'` (`eval_kit` imports both).

- [ ] **Step 8 (a): The engine rename, `node.leases`, `node.evaluator_main`; GREEN; full suite; commit**

`src/colosseum/arena/tournament/engine.py`: rename `_Placement` → `MatchPlacement` and `_error_kind` → `error_kind` (public: the evaluator uses them); `MatchPlacement.where(message, context, block_ids)` takes the block **ids** (`ids = list(block_ids)` instead of `[b.id for b in blocks]`), and `_worker_main` passes `[b.id for b in blocks]`. No behaviour change (`test_sp4_tournament_engine.py` stays green).

`tests/game_helpers.py`: `PulseFailBot.act` raises when `os.environ.get(ENV_PROCESS_NAME, "")` is `"pulse"` **or starts with `"evaluator"`** (docstring: "the pulse process (SP4) or an evaluator process (SP5)"); (b) drops the `"pulse"` case. `pyproject.toml`: `"eval_kit"` in `known-first-party`.

Create `src/colosseum/node/leases.py`:
```python
"""Evaluator leases on a node (SP5 spec 5.3; T5.1): the node agent's evaluation bookkeeping, pure (the agent turns
its answers into messages and child commands).

- Evaluators are indexes ``0 .. n-1``; ``evaluator_up`` / ``evaluator_down`` follow the children.
- ``request``: how many more slots to ask the hub for (``BlockLeaseRequest.free_slots``, an increment): idle
  evaluators minus leases waiting for one minus slots already asked; ``on_reconnect`` forgets the asked slots (a
  new session: the hub forgot them too).
- A lease (``on_lease``) waits for the weights of its players (``fetching``; the agent fetches the keys it returns,
  one fetch per key and machine), then for an idle evaluator (``ready`` -> ``assignments``: ``playing``).
- Every block outcome (``BlockOutcome``: results, or a failure kind) stays unacked until the hub's ``BlockAck``
  and is resent after a reconnect; ``on_reconnect`` names the leases held (``Hello.leases``).
- A dead evaluator's lease reports ``crash`` per block, a stopped one's ``abandoned``, a failed fetch
  ``missing_weights``; ``on_cancel`` drops a lease without reporting it (the evaluator stays busy until it answers).
- ``drop_candidates``: evaluation blobs no lease uses, beyond the newest ``EVAL_BLOBS_KEPT``, to remove from the
  machine cache.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from colosseum.core.types import MatchResult
from colosseum.net.messages import BlockTask, PlayerRef
from colosseum.node.weights import CacheKey

EVAL_BLOBS_KEPT = 8        # unused evaluation blobs kept in the machine cache (newest first)


@dataclass
class NodeLease:
    """One lease as the node holds it: ``fetching`` its weights, ``ready`` for an evaluator, ``playing``."""

    lease_id: str
    source: str
    blocks: list[BlockTask]
    num_envs: int
    deterministic: bool
    state: Literal["fetching", "ready", "playing"] = "fetching"
    evaluator: int | None = None
    missing: set[CacheKey] = field(default_factory=set)

    @property
    def players(self) -> list[PlayerRef]:
        out: dict[tuple[str, str, str], PlayerRef] = {}
        for task in self.blocks:
            for p in task.players:
                out.setdefault((p.agent_id, p.network_id, p.blob.ident if p.blob else ""), p)
        return list(out.values())

    @property
    def keys(self) -> set[CacheKey]:
        return {p.blob for p in self.players if p.blob is not None}


@dataclass(frozen=True)
class BlockOutcome:
    """What the node reports for one block (``BlockResult``): results, or a failure with its kind."""

    lease_id: str
    block_id: str
    results: list[MatchResult] | None = None
    error_kind: str = ""          # abandoned | crash | missing_weights | contract | player | config
    error: str = ""

    @property
    def failed(self) -> bool:
        return self.results is None


class NodeLeases:
    """Module docstring."""

    def __init__(self) -> None:
        self._evaluators: dict[int, str | None] = {}     # up evaluators: index -> lease id playing (None: idle)
        self._leases: dict[str, NodeLease] = {}
        self._unacked: dict[tuple[str, str], BlockOutcome] = {}
        self._asked = 0
        self._eval_keys: OrderedDict[CacheKey, None] = OrderedDict()

    # --- evaluators ---
    def evaluator_up(self, index: int) -> None:
        self._evaluators[index] = None

    def evaluator_down(self, index: int, *, error_kind: str = "crash", error: str = "") -> list[BlockOutcome]:
        """An evaluator exited (``crash``) or is being stopped (``abandoned``): its lease's blocks are released."""
        lease_id = self._evaluators.pop(index, None)
        return [] if lease_id is None else self._abandon(lease_id, error_kind, error)

    def idle(self) -> list[int]:
        return sorted(i for i, lease in self._evaluators.items() if lease is None)

    def up(self) -> list[int]:
        return sorted(self._evaluators)

    # --- leases ---
    def on_lease(self, lease: NodeLease, have: Callable[[CacheKey], bool]) -> list[CacheKey]:
        """A lease from the hub; returns the cache keys to fetch (the agent dedups fetches already running)."""
        self._asked = max(0, self._asked - 1)
        lease.missing = {k for k in lease.keys if not have(k)}
        lease.state = "fetching" if lease.missing else "ready"
        self._leases[lease.lease_id] = lease
        for key in sorted(lease.keys, key=lambda k: (k.agent_id, k.ident)):
            self._eval_keys[key] = None
            self._eval_keys.move_to_end(key)
        return sorted(lease.missing, key=lambda k: (k.agent_id, k.ident))

    def on_fetched(self, key: CacheKey, ok: bool) -> list[BlockOutcome]:
        out: list[BlockOutcome] = []
        for lease in list(self._leases.values()):
            if lease.state != "fetching" or key not in lease.missing:
                continue
            if ok:
                lease.missing.discard(key)
                if not lease.missing:
                    lease.state = "ready"
            else:
                out += self._abandon(lease.lease_id, "missing_weights",
                                     f"the weights {key.ident} could not be fetched from the hub")
        return out

    def assignments(self) -> list[tuple[int, NodeLease]]:
        """Ready leases handed to idle evaluators (lowest index first, leases in arrival order)."""
        out: list[tuple[int, NodeLease]] = []
        idle = self.idle()
        for lease in self._leases.values():
            if not idle:
                break
            if lease.state == "ready":
                index = idle.pop(0)
                lease.state, lease.evaluator = "playing", index
                self._evaluators[index] = lease.lease_id
                out.append((index, lease))
        return out

    def on_done(self, index: int, lease_id: str, by_block: Mapping[str, Sequence[MatchResult]],
                cancelled: bool) -> list[BlockOutcome]:
        if self._evaluators.get(index) == lease_id:
            self._evaluators[index] = None
        lease = self._leases.pop(lease_id, None)
        if lease is None or cancelled:
            return []
        out = []
        for task in lease.blocks:
            got = by_block.get(task.block_id)
            outcome = (BlockOutcome(lease_id, task.block_id, list(got)) if got is not None else
                       BlockOutcome(lease_id, task.block_id, None, "crash", "the evaluator returned no results"))
            self._unacked[(lease_id, task.block_id)] = outcome
            out.append(outcome)
        return out

    def on_failed(self, index: int, lease_id: str, kind: str, message: str) -> list[BlockOutcome]:
        if self._evaluators.get(index) == lease_id:
            self._evaluators[index] = None
        return self._abandon(lease_id, kind, message)

    def _abandon(self, lease_id: str, kind: str, error: str) -> list[BlockOutcome]:
        lease = self._leases.pop(lease_id, None)
        if lease is None:
            return []
        if lease.evaluator is not None and self._evaluators.get(lease.evaluator) == lease_id:
            self._evaluators[lease.evaluator] = None
        out = [BlockOutcome(lease_id, task.block_id, None, kind, error) for task in lease.blocks]
        for outcome in out:
            self._unacked[(lease_id, outcome.block_id)] = outcome
        return out

    def on_ack(self, lease_id: str, block_ids: Sequence[str]) -> None:
        for block_id in block_ids:
            self._unacked.pop((lease_id, block_id), None)

    def on_cancel(self, lease_ids: Sequence[str]) -> list[tuple[int, str]]:
        """The hub withdrew these leases: ``(evaluator, lease id)`` to send ``EvalCancel`` to; nothing of them is
        reported."""
        out: list[tuple[int, str]] = []
        for lease_id in lease_ids:
            lease = self._leases.pop(lease_id, None)
            if lease is not None and lease.state == "playing" and lease.evaluator is not None:
                out.append((lease.evaluator, lease_id))
            for key in [k for k in self._unacked if k[0] == lease_id]:
                del self._unacked[key]
        return out

    # --- the session ---
    def request(self, *, connected: bool) -> int:
        if not connected:
            return 0
        waiting = sum(1 for lease in self._leases.values() if lease.state in ("fetching", "ready"))
        free = len(self.idle()) - waiting - self._asked
        if free <= 0:
            return 0
        self._asked += free
        return free

    def on_reconnect(self) -> tuple[list[str], list[BlockOutcome]]:
        """A new session: the leases to name in ``Hello.leases`` and the outcomes to send again."""
        self._asked = 0
        held = sorted({*self._leases, *(lease_id for lease_id, _ in self._unacked)})
        return held, list(self._unacked.values())

    # --- the machine cache ---
    def keys_in_use(self) -> set[CacheKey]:
        return {k for lease in self._leases.values() for k in lease.keys}

    def drop_candidates(self) -> list[CacheKey]:
        """Evaluation blobs to remove from the cache: unused by every lease, beyond the newest ``EVAL_BLOBS_KEPT``."""
        used = self.keys_in_use()
        unused = [k for k in self._eval_keys if k not in used]
        drop = unused[:max(0, len(unused) - EVAL_BLOBS_KEPT)]
        for key in drop:
            del self._eval_keys[key]
        return drop
```

Create `src/colosseum/node/evaluator_main.py`:
```python
"""Evaluator process of a node (SP5 spec 5.3; T5.1).

``evaluator_target`` (spawn target, ``ChildSpec(role="evaluator")``, name ``evaluator-<i>``): ``run_child`` (log
``<log dir>/evaluator-<i>.log``, the child signal policy, the parent-death watch of T4.4 as every node child), then
``evaluator_main``: one control connection to the node (``ChildHello(role="evaluator", channel="control")``), a
read-only ``WeightCacheReader`` of the machine's cache, and ``evaluator_loop``:
- ``EvalTask``: play the lease's lineups (all its whole blocks) with SP4's ``play_lineups`` on a ``CachePlayerPool``
  (``source`` = the lease's: ``pulse`` / ``tournament``; ``match_id_prefix`` the same; the torch sampling seeded
  with the lease's first lineup seed; ``num_envs`` and ``deterministic`` from the lease; the error context
  ``<source>, node N, evaluator E, env ...``); report ``EvalDone`` with the results grouped by block (each block's
  sorted by uid), never a part of a lease;
- a stop (``Stop``, or the node's connection closed) and an ``EvalCancel`` of the playing lease are checked before
  every env step: a stop ends the process without a report (the node reports the lease ``abandoned``), a cancel
  answers ``EvalDone(cancelled=True)``;
- a ``ConfigError`` / ``EnvContractError`` / ``PlayerError`` of a lease is ``EvalFailed(kind, message, where)``
  (``where``: the match uid and block, ``MatchPlacement``), weights missing from the cache ``EvalFailed("missing_
  weights")``; the evaluator keeps running. Any other exception crashes the process (``run_child`` logs the
  traceback; the node restarts it within ``WORKER_RESTARTS`` and reports the lease ``crash``).
"""

from __future__ import annotations

import functools
import json
import logging
import os
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from colosseum.arena.tournament.engine import MODEL_CACHE_SIZE, MatchPlacement, error_kind
from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError, EnvContractError, PlayerError
from colosseum.core.types import MatchResult
from colosseum.envs.game import GameSpec
from colosseum.net.local import ChildHello, EvalCancel, EvalDone, EvalFailed, EvalTask, Stop, connect
from colosseum.net.messages import PlayerRef
from colosseum.players.registry import FrozenSpec, build_frozen_model, make_bot
from colosseum.utils.process import run_child
from colosseum.worker.match_runner import ScriptedPlayer

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SEC = 30.0
POLL_SEC = 0.2
_KINDS = {"EnvContractError": "contract", "PlayerError": "player", "ConfigError": "config"}


class MissingWeights(RuntimeError):
    """A player's weights are not in the machine cache (the node hands a lease over only when they are)."""


class CachePlayerPool:
    """``PlayerPool`` of one lease's players over the machine's weight cache: neural players are built from the
    cache's read-only views on first use (``build_frozen_model`` copies them) and kept in an LRU of ``capacity``
    models keyed by blob (two tournaments' players with equal seat names never share a model; a lease's players
    are reserved); scripted players are ``ScriptedPlayer`` factories."""

    def __init__(self, spec: GameSpec, reader: Any, capacity: int = MODEL_CACHE_SIZE) -> None:
        self._spec, self._reader, self._capacity = spec, reader, int(capacity)
        self._current: dict[tuple[str, str], PlayerRef] = {}
        self._models: OrderedDict[Any, Any] = OrderedDict()
        self._bots: dict[tuple[str, str, str], ScriptedPlayer] = {}
        self._reserved: set[Any] = set()
        self.loads = 0

    def reserve(self, players: Sequence[PlayerRef]) -> None:
        self._current = {(p.agent_id, p.network_id): p for p in players}
        self._reserved = {p.blob for p in players if p.kind == "neural"}
        for player in players:
            if player.kind == "neural":
                self._model(player)
        self._trim()

    def get(self, agent_id: str, network_id: str) -> Any:
        player = self._current.get((agent_id, network_id))
        if player is None:
            return None
        if player.kind == "scripted":
            key = (player.agent_id, player.bot.class_path, json.dumps(player.bot.kwargs, sort_keys=True))
            if key not in self._bots:
                self._bots[key] = ScriptedPlayer(functools.partial(make_bot, player.bot, self._spec))
            return self._bots[key]
        return self._model(player)

    def _model(self, player: PlayerRef) -> Any:
        model = self._models.get(player.blob)
        if model is not None:
            self._models.move_to_end(player.blob)
            return model
        cached = self._reader.get(player.blob)
        if cached is None:
            raise MissingWeights(f"the weights of {player.name} (blob {player.blob.ident}) are not in this machine's "
                                 f"weight cache")
        networks = player.networks or cached.meta.architecture or {}
        frozen = FrozenSpec(agent_id=player.agent_id, roles=tuple(player.roles), networks=dict(networks),
                            model_state=cached.state_dict, source=f"{player.name} (blob {player.blob.ident})",
                            networks_source=player.name)
        model = build_frozen_model(None, frozen, self._spec)
        self.loads += 1
        self._models[player.blob] = model
        self._trim()
        return model

    def _trim(self) -> None:
        limit = max(self._capacity, len(self._reserved))
        for key in list(self._models):
            if len(self._models) <= limit:
                break
            if key not in self._reserved:
                del self._models[key]


class _Control:
    """The control connection; messages read while a lease plays wait in ``pending``."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self.pending: deque[Any] = deque()
        self.stop = False

    def _read(self, timeout: float) -> None:
        try:
            while self._conn.poll(timeout):
                message = self._conn.recv()
                if isinstance(message, Stop):
                    self.stop = True
                self.pending.append(message)
                timeout = 0
        except (EOFError, OSError):
            self.stop = True                       # the node is gone (the parent-death watch fires as well)

    def next(self, timeout: float) -> Any:
        if not self.pending:
            self._read(timeout)
        return self.pending.popleft() if self.pending else None

    def cancelled(self, lease_id: str) -> bool:
        """Read what arrived (no wait): a stop, or an ``EvalCancel`` of ``lease_id``."""
        self._read(0)
        return self.stop or any(isinstance(m, EvalCancel) and m.lease_id == lease_id for m in self.pending)

    def send(self, message: Any) -> None:
        try:
            self._conn.send(message)
        except (EOFError, OSError):
            self.stop = True


def play_task(task: EvalTask, *, pool: CachePlayerPool, control: _Control, env_fn: Callable[[], Any],
              max_idle_steps: int, node: int, index: int) -> None:
    """Play one lease and report it on ``control`` (module docstring)."""
    from colosseum.eval import play_lineups

    lineups = [lu for block in task.blocks for lu in block.lineups]
    context = f"{task.source}, node {node}, evaluator {index}, "
    placement = MatchPlacement(lineups, task.num_envs)
    players = {(p.agent_id, p.network_id, p.blob.ident if p.blob else ""): p
               for block in task.blocks for p in block.players}
    try:
        pool.reserve(list(players.values()))
        played = play_lineups(env_fn=env_fn, pool=pool, lineups=lineups, num_envs=task.num_envs,
                              seed=lineups[0].seed, deterministic=task.deterministic, max_idle_steps=max_idle_steps,
                              observer=placement, source=task.source, context=context, match_id_prefix=task.source,
                              should_stop=lambda: control.cancelled(task.lease_id))
    except MissingWeights as e:
        control.send(EvalFailed(task.lease_id, "missing_weights", str(e), f"lease {task.lease_id}"))
        return
    except (ConfigError, EnvContractError, PlayerError) as e:
        control.send(EvalFailed(task.lease_id, _KINDS[error_kind(e)], str(e),
                                placement.where(str(e), context, [b.block_id for b in task.blocks])))
        return
    if len(played) < len(lineups):                   # stopped or cancelled before the end
        if not control.stop:
            control.send(EvalDone(task.lease_id, cancelled=True))
        return
    by_block: dict[str, list[MatchResult]] = {}
    for result in played:
        by_block.setdefault(result.block, []).append(result)
    control.send(EvalDone(task.lease_id, {b: sorted(r, key=lambda x: x.uid) for b, r in by_block.items()}))


def evaluator_loop(conn: Any, *, spec: GameSpec, reader: Any, env_fn: Callable[[], Any], max_idle_steps: int,
                   node: int, index: int) -> None:
    """The evaluator's main loop over its control connection; returns on ``Stop`` or a closed connection."""
    control = _Control(conn)
    pool = CachePlayerPool(spec, reader)
    while not control.stop:
        message = control.next(POLL_SEC)
        if message is None or isinstance(message, (Stop, EvalCancel)):
            continue                                  # a cancel of a lease that no longer plays: nothing to do
        if isinstance(message, EvalTask):
            play_task(message, pool=pool, control=control, env_fn=env_fn, max_idle_steps=max_idle_steps,
                      node=node, index=index)
            control.pending = deque(m for m in control.pending
                                    if not (isinstance(m, EvalCancel) and m.lease_id == message.lease_id))
        else:
            logger.warning(f"evaluator {index}: unexpected message {type(message).__name__} ignored")


def evaluator_main(*, name: str, index: int, incarnation: int, node: int, node_address: str, authkey: bytes,
                   cache_root: str, cache_prefix: str, config: ColosseumConfig) -> None:
    from colosseum.core.registry import env_spec, make_env
    from colosseum.core.threads import configure_torch_threads
    from colosseum.node.weights import WeightCacheReader

    configure_torch_threads(1)
    spec = env_spec(config)
    conn = connect(node_address, authkey, timeout=CONNECT_TIMEOUT_SEC)
    conn.send(ChildHello(role="evaluator", name=name, incarnation=incarnation, pid=os.getpid(), channel="control"))
    logger.info(f"{name}: node {node}, cache {cache_prefix}")
    evaluator_loop(conn, spec=spec, reader=WeightCacheReader(Path(cache_root), cache_prefix),
                   env_fn=functools.partial(make_env, config), max_idle_steps=config.env.max_idle_steps, node=node,
                   index=index)


def evaluator_target(*, log_dir: str | None = None, **kwargs: Any) -> None:
    """Spawn target of a node's evaluator child (module docstring)."""
    run_child(kwargs["name"], log_dir, evaluator_main, **kwargs)
```
(If T2.4's `worker_target` also calls `watch_parent()` (T4.4) or passes the bundle's `sys.path` / cwd in its kwargs, `evaluator_target` does the same, in the same order: one helper for both if T2.4 has one.)

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_node_leases.py tests/integration/test_sp5_evaluator.py tests/integration/test_sp4_tournament_engine.py -q`
Expected: all pass (`test_sp5_evaluator.py`: 7 tests; one evaluator process at a time).

Then the full fast suite + ruff (zero warnings). Commit: `feat: evaluation blocks, leases and the evaluator process (libraries)`; push.

- [ ] **Step 9 (b): Write the failing tests of the pulse on blocks**

Create `tests/unit/test_sp5_pulse_blocks.py`:
```python
"""Pulse on evaluator blocks (SP5 T5.1, spec 5.3, R6): HubPulse (links, SP4 stage-P blocks, completion, cleanup),
HubEvaluation with the pulse (pipelining, journal only, a typed error stops the run, failed and stopped snapshots),
the pulse played end to end on two in-process nodes, and the pulse child gone."""
from __future__ import annotations

import logging

from colosseum.arena import pulse as pulse_module
from colosseum.arena.pulse import pulse_plan
from colosseum.arena.tournament.players import config_player
from colosseum.core.registry import env_spec
from colosseum.hub.evaluation import HubEvaluation, HubPulse, check_block_results, pulse_owner
from colosseum.net.messages import player_ref_from_proto, player_ref_to_proto, result_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.players.registry import load_fixed_players, resolve_player_roles
from eval_kit import EvalNode, FakeBlobs, drive, route
from game_helpers import make_match_result, make_test_config, scripted_agent, write_ckpt_dir

AGENTS = {"main": {}, "rnd": scripted_agent(), "ones": scripted_agent("game_helpers.ConstantBot", value=1)}
RUN_ID = "0123456789abcdef"


def _cfg(**evaluation):
    return make_test_config("turns", agents=AGENTS, evaluation={"enabled": True, **evaluation})


def _pulse(tmp_path, cfg, blobs):
    spec = env_spec(cfg)
    plan = pulse_plan(cfg, spec, resolve_player_roles(cfg, spec))
    fixed = load_fixed_players(cfg, spec, plan.field)
    players = {n: config_player(cfg, spec, n, fixed) for n in plan.field}
    return HubPulse(cfg, spec, plan, run_id=RUN_ID, field_players=players, eval_dir=tmp_path / "eval", blobs=blobs)


def _offer(pulse, cfg, tmp_path, version, *, final=False):
    write_ckpt_dir(tmp_path / "ckpts", cfg, "main", policy_version=version, seed=version)
    return pulse.offer("main", f"ckpt_v{version}", version, 10 * version, tmp_path / "ckpts" / f"ckpt_v{version}",
                       final=final)


def _fake_results(task):
    return [make_match_result([(0, lu.seats[0].agent_id, lu.seats[0].network_id),
                               (1, lu.seats[1].agent_id, lu.seats[1].network_id)],
                              uid=lu.uid, block=task.block_id, seed=lu.seed, source=task.source)
            for lu in task.lineups]


def test_hub_pulse_links_registers_and_cleans_up(tmp_path):
    cfg, blobs = _cfg(blocks=1, max_backlog=1), FakeBlobs()
    pulse = _pulse(tmp_path, cfg, blobs)
    assert _offer(pulse, cfg, tmp_path, 3)
    link = tmp_path / "eval" / "pulse" / "main" / "ckpt_v3"
    assert (link / "model.pt").is_file() and (link / "meta.json").is_file()
    assert _offer(pulse, cfg, tmp_path, 6)                       # backlog 1: v3 (waiting) is dropped
    assert not link.exists() and pulse.feeder.state()["skipped"] == {"main": [3]}
    assert not pulse.offer("other", "ckpt_v1", 1, 1, tmp_path / "nowhere")
    tasks = pulse.start_next()
    assert tasks and ("register", "pulse-main-ckpt_v6") in blobs.log
    pulse.finish()
    assert ("unregister", "pulse-main-ckpt_v6") in blobs.log and not (tmp_path / "eval" / "pulse").exists()


def test_hub_pulse_blocks_have_the_sp4_seeds_uids_and_players(tmp_path):
    cfg, blobs = _cfg(blocks=3, num_envs=2), FakeBlobs()
    pulse = _pulse(tmp_path, cfg, blobs)
    for version in (5, 10):
        _offer(pulse, cfg, tmp_path, version)
    first, second = pulse.start_next(), pulse.start_next()
    assert len(first) == 2 * 3 and {t.owner for t in first} == {pulse_owner("main", "ckpt_v5")}
    seeds = [sorted({lu.seed for t in ts for lu in t.lineups}) for ts in (first, second)]
    assert seeds[0] == seeds[1] and len(seeds[0]) == 3                     # stage P: the same seeds for both
    uids = {lu.uid for t in first for lu in t.lineups}
    assert len(uids) == 2 * 3 * 2 and all(u.startswith("pulse-0123456789ab/P/2p/main@v5/") for u in uids)
    assert {(t.priority, t.num_envs, t.source) for t in first} == {(1, 2, "pulse")}
    for task in first:
        kinds = {(p.agent_id, p.kind) for p in task.players}
        assert ("main", "neural") in kinds and len(task.players) == 2
        assert [player_ref_from_proto(player_ref_to_proto(p)) for p in task.players] == list(task.players)


def test_hub_pulse_completes_a_snapshot_with_its_last_block(tmp_path):
    cfg, blobs = _cfg(blocks=3, num_envs=2), FakeBlobs()
    pulse = _pulse(tmp_path, cfg, blobs)
    _offer(pulse, cfg, tmp_path, 5)
    tasks = pulse.start_next()
    out = None
    for i, task in enumerate(tasks):
        results = _fake_results(task)
        assert check_block_results(task, results) is None
        out = pulse.on_block(task.block_id, results)
        assert (out is None) == (i < len(tasks) - 1)
    assert len(out) == 12 and [r.uid for r in out] == sorted(r.uid for r in out) and {r.env_steps for r in out} == {50}
    assert pulse.feeder.state()["evaluated"] == {"main": [5]} and pulse.owners() == []
    assert pulse.on_block(tasks[0].block_id, _fake_results(tasks[0])) is None    # a block of a finished snapshot


def test_hub_evaluation_leases_acks_and_journals_pulse_results_only(tmp_path):
    cfg, blobs = _cfg(blocks=2, num_envs=2), FakeBlobs()
    pulse = _pulse(tmp_path, cfg, blobs)
    sent, journal = [], []
    ev = HubEvaluation(lease_sec=600, send=lambda n, m: sent.append((n, m)), pulse=pulse,
                       journal=lambda rs: journal.append(rs) or 0)
    _offer(pulse, cfg, tmp_path, 3)
    assert ev.node_joined(1, slots=2, held_leases=[], fresh=True).evaluators == 2   # pulse on: evaluators run
    ev.tick()
    assert ev.queue.queued("pulse:") == 4
    ev.on_lease_request(1, 2)
    (lease,) = [m.lease for _n, m in sent if m.WhichOneof("m") == "lease"]          # 4 blocks x 2 = 8 matches
    assert len(lease.blocks) == 4 and lease.num_envs == 2 and ev.queue.waiting() == {1: 1}
    from colosseum.net.messages import block_task_from_proto
    for spec_ in lease.blocks:
        task = block_task_from_proto(spec_, num_envs=lease.num_envs)
        ev.on_block_result(1, pb.BlockResult(lease_id=lease.lease_id, block_id=task.block_id,
                                             results=[result_to_proto(r) for r in _fake_results(task)]))
    assert len(journal) == 1 and len(journal[0]) == 8 and {r.node for r in journal[0]} == {1}
    assert ev.pulse_results == 8 and len([1 for _n, m in sent if m.WhichOneof("m") == "block_ack"]) == 4
    ev.on_block_result(1, pb.BlockResult(lease_id=lease.lease_id, block_id=lease.blocks[0].block_id))
    assert len(journal) == 1 and ev.queue.stats()["late"] == 1                     # acked, counted, dropped


def test_a_typed_pulse_error_stops_the_run(tmp_path):
    cfg, blobs = _cfg(blocks=1, num_envs=1), FakeBlobs()
    pulse = _pulse(tmp_path, cfg, blobs)
    sent, fatal = [], []
    ev = HubEvaluation(lease_sec=600, send=lambda n, m: sent.append((n, m)), pulse=pulse, fail_run=fatal.append)
    _offer(pulse, cfg, tmp_path, 3)
    ev.node_joined(4, slots=1, held_leases=[], fresh=True)
    ev.tick()
    ev.on_lease_request(4, 1)
    (lease,) = [m.lease for _n, m in sent if m.WhichOneof("m") == "lease"]
    ev.on_block_result(4, pb.BlockResult(lease_id=lease.lease_id, block_id=lease.blocks[0].block_id, failed=True,
                                         error_kind="player", error="pulse failure on purpose (match x)"))
    (text,) = fatal
    assert "pulse block" in text and "node 4" in text and "pulse failure on purpose" in text
    assert pulse.feeder.state()["skipped"] == {"main": [3]} and ev.queue.queued("pulse:") == 0


def test_the_next_snapshot_starts_once_the_last_ones_blocks_are_all_leased(tmp_path):
    cfg, blobs = _cfg(blocks=2, num_envs=1), FakeBlobs()
    pulse = _pulse(tmp_path, cfg, blobs)
    sent = []
    ev = HubEvaluation(lease_sec=600, send=lambda n, m: sent.append((n, m)), pulse=pulse)
    for version in (3, 6, 9):
        _offer(pulse, cfg, tmp_path, version)
    ev.tick()
    assert ev.queue.queued("pulse:") == 4 and pulse.feeder.pending == 3 and len(pulse.owners()) == 1
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.on_lease_request(1, 1)                                  # 4 matches: 2 of the 4 blocks
    ev.tick()
    assert len(pulse.owners()) == 1
    ev.on_lease_request(1, 1)                                  # the rest of the first snapshot is leased
    ev.tick()
    assert pulse.owners() == [pulse_owner("main", "ckpt_v3"), pulse_owner("main", "ckpt_v6")]


def test_a_failed_or_stopped_snapshot_withdraws_its_blocks(tmp_path, caplog):
    cfg, blobs = _cfg(blocks=1, num_envs=1), FakeBlobs()
    pulse = _pulse(tmp_path, cfg, blobs)
    sent, now = [], [0.0]
    ev = HubEvaluation(lease_sec=600, send=lambda n, m: sent.append((n, m)), pulse=pulse, clock=lambda: now[0])
    _offer(pulse, cfg, tmp_path, 3)
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.tick()
    with caplog.at_level(logging.WARNING):
        for _ in range(3):                                     # three expired leases: the block fails
            ev.on_lease_request(1, 1)
            now[0] += 601
            ev.tick()
    assert pulse.feeder.state()["skipped"] == {"main": [3]} and ev.queue.live(pulse_owner("main", "ckpt_v3")) == 0
    assert any("could not be evaluated" in r.getMessage() for r in caplog.records)
    assert not (tmp_path / "eval" / "pulse" / "main" / "ckpt_v3").exists()
    _offer(pulse, cfg, tmp_path, 6)
    ev.tick()
    ev.on_lease_request(1, 1)
    last = [m.lease for _n, m in sent if m.WhichOneof("m") == "lease"][-1]
    ev.stop()
    with caplog.at_level(logging.INFO):
        ev.finish()
    cancels = [m.command for _n, m in sent if m.WhichOneof("m") == "command" and m.command.kind == "cancel_leases"]
    assert cancels and last.lease_id in cancels[-1].lease_ids
    assert any("was not evaluated (the run stopped first" in r.getMessage() for r in caplog.records)
    assert not (tmp_path / "eval" / "pulse").exists() and pulse.feeder.state()["unevaluated"] == {"main": [6]}


def test_the_final_snapshot_offered_after_the_stop_is_never_evaluated(tmp_path):
    cfg, blobs = _cfg(blocks=1), FakeBlobs()
    pulse = _pulse(tmp_path, cfg, blobs)
    ev = HubEvaluation(lease_sec=600, send=lambda n, m: None, pulse=pulse)
    ev.stop()
    assert _offer(pulse, cfg, tmp_path, 9, final=True)        # offered (recorded), not linked, never started
    ev.tick()
    assert pulse.owners() == [] and not (tmp_path / "eval" / "pulse" / "main" / "ckpt_v9").exists()
    assert pulse.feeder.unevaluated_final() == [("main", "ckpt_v9")]


def test_the_pulse_plays_on_evaluators_end_to_end(tmp_path):
    cfg, blobs = _cfg(blocks=2, num_envs=2), FakeBlobs()
    pulse = _pulse(tmp_path, cfg, blobs)
    journal, nodes = [], {}
    ev = HubEvaluation(lease_sec=600, send=route(nodes), pulse=pulse, journal=lambda rs: journal.append(rs) or 0)
    nodes[1] = EvalNode(cfg, blobs, ev, number=1)
    nodes[2] = EvalNode(cfg, blobs, ev, number=2)
    try:
        assert [n.join() for n in nodes.values()] == [1, 1]
        for version in (3, 6):
            assert _offer(pulse, cfg, tmp_path, version)
        drive(ev, list(nodes.values()), lambda: len(journal) == 2)
    finally:
        for n in nodes.values():
            n.close()
    assert pulse.feeder.state()["evaluated"] == {"main": [3, 6]} and ev.pulse_results == 16
    seeds = [sorted({r.seed for r in rs}) for rs in journal]
    assert seeds[0] == seeds[1] and len(seeds[0]) == 2
    assert [{r.env_steps for r in rs} for rs in journal] == [{30}, {60}]
    assert {r.source for rs in journal for r in rs} == {"pulse"} and {r.node for rs in journal for r in rs} <= {1, 2}
    assert sum(n.fetches for n in nodes.values()) <= 4 and blobs.live == {}        # one fetch per machine and snapshot


def test_the_pulse_child_is_gone():
    import inspect

    from colosseum.hub import hub as hub_module

    for name in ("pulse_target", "pulse_main", "PulseResults", "_PulsePool"):
        assert not hasattr(pulse_module, name), name
    source = inspect.getsource(hub_module)
    assert "pulse_target" not in source and "_pulse_requests" not in source and "_pulse_results" not in source
```

Port `tests/unit/test_sp4_pulse.py` (same file, same test names where the subject stays; the task report lists every change):
- imports: `PulseFeeder, PulseRequest, pulse_plan` only (no `PulseResults`, `pulse_target`);
- `_request(...)` builds `PulseRequest(agent, ckpt, version, env_steps, weights="", final=final)`;
- the feeder tests use the new API: `offer(agent, ckpt, version, env_steps, final=...)` (no state dict), `start_next()` instead of `pump(q)` (`feeder.start_next().version == 8` for the backlog test), `done(agent, ckpt)` instead of `done(PulseResults(...))`; every assertion on `state()`, `skipped`, `unevaluated_final()` and the INFO texts is kept;
- `test_the_feeder_keeps_no_weights_of_a_request_in_flight` becomes `test_the_feeder_holds_paths_not_weights`: `PulseRequest` has no `state_dict` field (`dataclasses.fields`), an in-flight request's `weights` is the linked path string;
- `test_the_pulse_process_plays_every_snapshot_on_the_same_seeds`: deleted (its subject, the pulse process, is gone; `test_sp5_pulse_blocks.py::test_hub_pulse_blocks_have_the_sp4_seeds_uids_and_players` and `::test_the_pulse_plays_on_evaluators_end_to_end` keep its assertions);
- `test_the_launcher_offers_saved_snapshots_and_journals_pulse_results_only` (ported to `Hub` by T2.5) becomes `test_the_hub_offers_saved_snapshots_and_journals_pulse_results_only`: a `Hub` built as T2.5's port builds it (no launch), `hub._evaluation` replaced by a `HubEvaluation` with a recording `send`, `journal=writer.submit` of a recorder that allows only `submit`, the coordinator a recorder that allows nothing; `hub._save_checkpoint(payload of v7)` → `runs/<name>/eval/pulse/main/ckpt_v7/model.pt` exists; `hub._pump_pulse()` → 4 queued pulse blocks; `hub._evaluation.node_joined(1, slots=1, ...)`, `on_lease_request(1, 1)` and fake `BlockResult`s of every leased block (as in `test_hub_evaluation_leases_acks_and_journals_pulse_results_only`; repeat until all are journaled) → `writer.calls == [("submit", (sorted results,))]`, the system values `pulse_skipped_snapshots` 0 / `pulse_evaluated_snapshots` 1 / `pulse_pending_snapshots` 0 and `evaluation` (a dict with `completed == 4`), `pulse_state.json` `evaluated == {"main": [7]}`;
- `_run_pulse_in_process` becomes `_play_pulse(cfg, tmp_path, versions)`: a `HubPulse` + `HubEvaluation` + one `eval_kit.EvalNode`, `drive` until every offered snapshot is evaluated; returns the journaled lists. `test_the_pulse_plays_score_layouts_for_two_agents`, `test_unscheduled_compositions_are_logged_once_per_agent` (the drop INFO now comes from `colosseum.hub.evaluation`) and `test_one_team_games_need_no_pulse_field_and_two_team_games_still_do` keep their assertions on uids, sources, seats, seeds and log lines on it;
- `test_a_stop_during_play_ends_the_pulse_without_sending_the_snapshot`: on `_play_pulse` with `blocks=50`: after the first lease is sent to the evaluator, `ev.stop()` and `EvalNode.close()` (a `Stop` reaches the evaluator mid-play) → nothing journaled, `pulse.finish()` lists the snapshot as unevaluated;
- `test_a_failing_pulse_state_write_is_logged_and_training_goes_on` and `test_the_texts_about_the_final_snapshot_say_a_local_run_never_evaluates_it`: on the `Hub` with a `HubPulse` (`hub._evaluation.pulse.feeder.offer(...)`, `hub._pump_pulse()`, `hub._finish_pulse()`); assertions unchanged;
- `test_put_until_stop_still_sends_after_the_stop_when_there_is_room`, `test_play_lineups_ends_early_when_asked_to_stop`, the config / validate / role-coverage tests and `test_system_values_reach_every_later_system_record`: unchanged.

Create `tests/integration/test_sp5_node_evaluation.py` (a `NodeAgent` with `node_kit.FakeHub`, T2.4; the node runs in a thread of the test process, its evaluator is the only loaded process; blocks are built with `eval_kit.gauntlet_tasks` on the `turns` toy config and their weights served by the fake hub's `Download(kind="eval")` as `encode_weights` frames (`WeightsMeta(kind="fixed", architecture=..., roles=...)`); add to `node_kit.FakeHub` what these tests need if T2.4 / T3.1 did not: `serve_blob(kind, agent_id, ident, frame)`, a download counter per ident, `refuse_downloads()`, `drop_session()`, `wait_for(kind, predicate, timeout)` over the node messages it received):
- `test_a_lease_is_fetched_once_played_on_an_evaluator_and_acked`: Welcome `evaluators=1` → the node starts `evaluator-0` (its `ProcState` in the next `Heartbeat`) and sends `BlockLeaseRequest(free_slots=1)`; the fake hub answers one `BlockLease` of two blocks (`lease_to_proto`); the node downloads the blob once (counter 1), sends two `BlockResult`s whose results pass `check_block_results` against the tasks; after the fake hub's `BlockAck` the node asks again (`free_slots=1`); a second lease naming the same blob downloads nothing (counter still 1).
- `test_unacked_block_results_are_resent_after_a_reconnect_and_hello_names_the_leases`: the fake hub never acks and drops the session after the `BlockResult`s; the node reconnects (backoff from T2.3) with `Hello.leases == [lease id]` and sends the same two `BlockResult`s again (same `lease_id` / `block_id`, equal results); after an ack, a third session's `Hello.leases` is empty.
- `test_set_evaluators_starts_and_stops_evaluators_and_releases_their_leases`: Welcome `evaluators=0` → no evaluator process; `Command(set_evaluators, evaluators=1)` → `evaluator-0` up and a lease request; a 400-block lease → while it plays, `Command(set_evaluators, 0)` → one `BlockResult(failed=True, error_kind="abandoned")` per block, the evaluator process ends within the local stop bound (≤ `SHUTDOWN_GRACE_SEC` + 1 s, monotonic).
- `test_cancel_leases_stops_playing_and_reports_nothing`: a 400-block lease, then `Command(cancel_leases, lease_ids=[it])` → no `BlockResult` of it ever, and a new `BlockLeaseRequest` (the evaluator is idle again) within 10 s.
- `test_an_evaluator_killed_mid_lease_is_restarted_and_its_blocks_reported_as_crash`: a 400-block lease; `kill -9` of the evaluator's pid (from its `ProcState` / log) → `BlockResult(failed=True, error_kind="crash")` per block, an `Event(kind="proc_died", proc="evaluator-0")` with a log tail, the evaluator restarted (incarnation 1 in the next heartbeat) and a new lease request; heartbeats never stopped meanwhile (the node's main loop did not block: counters by heartbeat count, not time).
- `test_a_failed_blob_fetch_releases_the_lease_as_missing_weights`: `refuse_downloads()` → `BlockResult(failed=True, error_kind="missing_weights")` per block; the evaluator never got the task (its log has no "lease" line).
- `test_a_player_error_is_reported_with_its_kind_and_the_evaluator_keeps_running`: a field of `game_helpers.PulseFailBot` → `BlockResult(failed=True, error_kind="player")` whose `error` holds "pulse failure on purpose", "node <n>, evaluator 0" and the evaluator's log path; a second lease (other field) is played by the same evaluator pid.
- `test_unused_evaluation_blobs_are_dropped_from_the_cache`: ten one-block leases, each with its own blob → after the last, the node's shm dir (`shm_dir()` under `tmp_path`) holds the segments of at most `EVAL_BLOBS_KEPT` evaluation blobs, the newest ones.

Create `tests/integration/test_sp5_hub_evaluation.py` (the real hub gRPC server of T2.3 — `build_server` on loopback, insecure, with a `BlobStore` on a `make_test_run_dir` run dir — plus `hub_kit.FakeNode` clients; the test thread is the main loop: it drains the inbox and calls a `HubEvaluation` built with `send = lambda n, m: handles[n].send(m)`, `handles` filled from `SessionOpened` events; no evaluator process):
- `test_leases_and_results_travel_over_the_session`: a FakeNode with `Slots(evaluators=1)` opens a session → the loop's `node_joined` → `Welcome.evaluators == 1` (the test sends the Welcome as T2.5's handler would); 2 tournament tasks submitted with a sink; the FakeNode sends `BlockLeaseRequest(1)` → receives a `BlockLease` with both blocks, sends both `BlockResult`s → receives two `BlockAck`s; the sink got both blocks with `node` = the FakeNode's number.
- `test_eval_blobs_are_served_by_download_and_removed_on_unregister`: `BlobStore.register_weights("t-ab-main", <write_ckpt_dir model.pt>, agent_id="main", networks=..., roles=...)` → the FakeNode's `Download(BlobRequest(kind="eval", agent_id="main", ident="t-ab-main"))` returns a frame that `decode_weights` reads into the checkpoint's state dict (`np.testing.assert_array_equal` per key) with `meta.architecture == networks` and `meta.roles`; the converted frame is a file under `runs/<name>/blobs/`; after `unregister` the frame file is gone and a download is `NOT_FOUND`; an unregistered ident or a path-like ident (`../x`) is `NOT_FOUND` / `INVALID_ARGUMENT`.
- `test_node_loss_re_leases_the_blocks_to_another_node`: two FakeNodes; node A takes a lease and goes silent; after the loop calls `node_lost(A)` (the test calls it, standing in for T2.5's loss check), node B (already asking) receives a lease with the same block ids.
- `test_a_resumed_session_with_hello_leases_keeps_them`: node A takes a lease, its stream is cut and it reconnects with the same session and `Hello.leases=[lease id]` → no `cancel_leases`, the lease is still A's (`queue.leases_of(A)`); reconnecting with `Hello.leases=["bogus"]` → a `Command(cancel_leases, ["bogus"])` right after the `Welcome` and A's real lease back in the queue.

Port `tests/integration/test_sp4_pulse_runs.py` (same file and names; `train` with pulse = one rollout worker + one evaluator = two loaded processes):
- `test_pulse_evaluates_snapshots_into_the_journal_on_the_same_seeds`: unchanged assertions on `pulse_state.json`, the journal (per version one seed set, `env_steps` = the snapshot's `meta.json`) and the `Match results: ... (<w> from workers, <p> from the pulse) ...` line; `logs/pulse.log` → `logs/evaluator-0.log` (it exists and ends with the `run_child` "evaluator-0 finished" line: the evaluator stopped by itself); every pulse result's journal `node` is the local node's number; the last `system` record has `pulse_skipped_snapshots` and an `evaluation` dict with `completed` ≥ the number of pulse blocks played.
- `test_a_pulse_crash_stops_the_run_with_a_pointer_to_its_log`: exit code 1; `logs/main.log` holds "pulse failure on purpose", "evaluator 0" and the path of `logs/evaluator-0.log` (the error comes through `fail_run`); `PulseFailBot` fails in evaluator processes (T5.1 (b) removes its `"pulse"` case).
- `test_the_pulse_dies_with_the_run`: the evaluator's pid from `logs/evaluator-0.log` ("started (pid N)"); `kill -9` of the `train` process → the local node dies with it (T4.4) and the evaluator with the node: gone within 10 s.

- [ ] **Step 10 (b): Run them: RED**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_pulse_blocks.py tests/unit/test_sp4_pulse.py tests/integration/test_sp5_node_evaluation.py tests/integration/test_sp5_hub_evaluation.py tests/integration/test_sp4_pulse_runs.py -q`
Expected: `ImportError: cannot import name 'HubPulse' from 'colosseum.hub.evaluation'`; the ported SP4 tests fail on the new feeder API (`TypeError: PulseFeeder.offer() takes 5 positional arguments`) and `logs/evaluator-0.log` missing; the node tests time out waiting for an evaluator process (the node does not start evaluators yet); `BlobStore` has no `register_weights`.

- [ ] **Step 11 (b): `arena/pulse.py` without the child; `HubPulse`**

`src/colosseum/arena/pulse.py`:
- **delete** `PulseResults`, `_Ticket`, `_PulsePool`, `_quiet`, `pulse_target`, `pulse_main` and the imports only they used (`contextlib`, `functools`, `queue`, `numpy`, `check_blocks`, `make_tasks`, `put_until_stop`, `PlayerSource`, `TournamentPlayer`, `config_player`, `FixedPlayers`, `FrozenSpec`, `build_frozen_model`, `make_bot`, `run_child`, `ScriptedPlayer`, `FIXED_NETWORK_ID` if unused);
- **keep** `PULSE_STATE_FILE`, `PulseAgent`, `PulsePlan`, `_Seat`, `_plan_seats`, `pulse_blocks`, `pulse_plan`, `describe_pulse` unchanged;
- **replace** `PulseRequest` and `PulseFeeder` with the code below; rewrite the module docstring's two halves: "Hub (`PulseFeeder`, main loop): ... (the SP4 text about the choice, the backlog and `state()`)" and "Evaluation: `colosseum.hub.evaluation.HubPulse` turns each started request into stage-P blocks for the node evaluators (SP5 T5.1); results go to the journal only, all at once per snapshot". The `describe_pulse` line and the `every` description keep "a local run stops before it is played: evaluate it with colosseum tournament" (SP4 B-M4 test).

```python
@dataclass
class PulseRequest:
    """One offered snapshot (hub main loop only; no weights: ``weights`` is the path of its linked ``model.pt``)."""

    agent_id: str
    checkpoint_id: str
    version: int
    env_steps: int
    weights: str = ""
    final: bool = False


class PulseFeeder:
    """Which saved snapshots the pulse evaluates, in which order (module docstring). Main loop only.

    ``offer`` queues every ``every``-th saved snapshot of an evaluated agent (and the final one) in a backlog of
    ``max_backlog`` waiting requests, dropping the oldest waiting one when it is full (INFO; ``take_dropped`` hands
    the dropped requests to the caller, which removes their links); ``start_next`` moves the oldest waiting request
    in flight (its blocks are then queued for the evaluators); ``done`` / ``fail`` end an in-flight request
    (evaluated / skipped). In-flight requests are never dropped by the backlog (a slow pulse still finishes
    snapshots; SP4 kept its one in-flight request the same way)."""

    def __init__(self, agents: Collection[str], every: int, max_backlog: int) -> None:
        self._agents = set(agents)
        self._every = int(every)
        self._max_backlog = int(max_backlog)
        self._saved: dict[str, int] = defaultdict(int)
        self._waiting: deque[PulseRequest] = deque()
        self._in_flight: dict[tuple[str, str], PulseRequest] = {}
        self._offered: dict[str, list[int]] = defaultdict(list)
        self._evaluated: dict[str, list[int]] = defaultdict(list)
        self._skipped: dict[str, list[int]] = defaultdict(list)
        self._final: dict[str, str] = {}
        self._skipped_final: set[tuple[str, str]] = set()     # final snapshots the backlog pushed out
        self._failed_final: set[tuple[str, str]] = set()      # final snapshots whose blocks failed
        self._dropped: list[PulseRequest] = []
        self.skipped = 0
        self.evaluated = 0
        self.changed = False

    def offer(self, agent_id: str, checkpoint_id: str, version: int, env_steps: int, *, final: bool = False,
              weights: str = "") -> bool:
        """A saved snapshot; True when it was queued for evaluation."""
        if agent_id not in self._agents:
            return False
        self._saved[agent_id] += 1
        if not final and self._saved[agent_id] % self._every != 0:
            return False
        if final:
            self._final[agent_id] = checkpoint_id
        if len(self._waiting) >= self._max_backlog:
            old = self._waiting.popleft()
            self._skip(old, f"backlog of {self._max_backlog} full; newer snapshots go first", backlog=True)
            self._dropped.append(old)
        self._waiting.append(PulseRequest(agent_id, checkpoint_id, int(version), int(env_steps), str(weights),
                                          bool(final)))
        self._offered[agent_id].append(int(version))
        self.changed = True
        return True

    def _skip(self, request: PulseRequest, why: str, *, backlog: bool) -> None:
        self._skipped[request.agent_id].append(request.version)
        self.skipped += 1
        if request.final:
            (self._skipped_final if backlog else self._failed_final).add((request.agent_id, request.checkpoint_id))
        what = "the final snapshot" if request.final else "snapshot"
        logger.info(f"pulse: {what} {request.checkpoint_id} of {request.agent_id} skipped ({why})")
        self.changed = True

    def take_dropped(self) -> list[PulseRequest]:
        """Waiting requests the backlog dropped since the last call."""
        out, self._dropped = self._dropped, []
        return out

    def start_next(self) -> PulseRequest | None:
        """The oldest waiting request, now in flight (None: nothing waits)."""
        if not self._waiting:
            return None
        request = self._waiting.popleft()
        self._in_flight[(request.agent_id, request.checkpoint_id)] = request
        return request

    def done(self, agent_id: str, checkpoint_id: str) -> PulseRequest | None:
        request = self._in_flight.pop((agent_id, checkpoint_id), None)
        if request is not None:
            self._evaluated[agent_id].append(request.version)
            self.evaluated += 1
            self.changed = True
        return request

    def fail(self, agent_id: str, checkpoint_id: str, reason: str) -> PulseRequest | None:
        """An in-flight request whose blocks could not be played (WARNING; counted as skipped)."""
        request = self._in_flight.pop((agent_id, checkpoint_id), None)
        if request is not None:
            logger.warning(f"pulse: snapshot {checkpoint_id} of {agent_id} could not be evaluated ({reason})")
            self._skip(request, reason, backlog=False)
        return request

    def in_flight(self) -> list[PulseRequest]:
        return list(self._in_flight.values())

    @property
    def pending(self) -> int:
        return len(self._waiting) + len(self._in_flight)

    def _unevaluated(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = defaultdict(list)
        for request in [*self._in_flight.values(), *self._waiting]:
            out[request.agent_id].append(request.version)
        return {a: sorted(v) for a, v in out.items()}

    def skipped_by_backlog(self, agent_id: str, checkpoint_id: str) -> bool:
        """Whether that final snapshot left the backlog unevaluated (a newer request pushed it out)."""
        return (agent_id, checkpoint_id) in self._skipped_final

    def unevaluated_final(self) -> list[tuple[str, str]]:
        """``(agent, checkpoint id)`` of final snapshots never evaluated: waiting, in flight, skipped or failed."""
        left = {(r.agent_id, r.checkpoint_id) for r in [*self._in_flight.values(), *self._waiting]}
        gone = left | self._skipped_final | self._failed_final
        return [(a, c) for a, c in self._final.items() if (a, c) in gone]

    def state(self) -> dict[str, Any]:
        unevaluated = self._unevaluated()
        agents = sorted(self._agents)
        return {"offered": {a: list(self._offered.get(a, [])) for a in agents if a in self._offered},
                "evaluated": {a: sorted(self._evaluated.get(a, [])) for a in agents if a in self._offered},
                "skipped": {a: sorted(self._skipped.get(a, [])) for a in agents if a in self._offered},
                "unevaluated": {a: unevaluated.get(a, []) for a in agents if a in self._offered}}
```
(`_finish_pulse`'s "the backlog skipped it" / "the run stopped first" texts stay right: a final snapshot whose blocks failed is not `skipped_by_backlog`, so it reads "the run stopped first" only if it is still in flight; a failed one already had its WARNING.)

`src/colosseum/core/config.py` (`EvaluationConfig`, descriptions only): `enabled` → "Evaluate new snapshots on node evaluators (needs match_log.enabled)."; `num_envs` → "Envs an evaluator plays a pulse lease with.". No schema change (`scripts/bench_throughput.py::_make_config` unaffected).

`tests/game_helpers.py`: `PulseFailBot` fails only in evaluator processes (`os.environ.get(ENV_PROCESS_NAME, "").startswith("evaluator")`; docstring updated).

Append to `src/colosseum/hub/evaluation.py` (and to its `__all__`: `HubPulse`; module docstring: the pulse bullet):
```python
from colosseum.arena.pulse import PulseFeeder, PulsePlan, PulseRequest, pulse_blocks
from colosseum.arena.tournament.engine import check_blocks
from colosseum.arena.tournament.players import PlayerSource
from colosseum.arena.tournament.schedule import DroppedComposition
from colosseum.coordinator.checkpoint_manager import link_or_copy
from colosseum.core.config import ColosseumConfig
from colosseum.envs.game import GameSpec

_SCHEDULE_LOGGER = "colosseum.arena.tournament.schedule"
_MODEL_FILES = ("model.pt", "meta.json")


@contextlib.contextmanager
def _quiet(name: str):
    """The logger ``name`` passes WARNING and above only, inside the block (main loop only)."""
    target = logging.getLogger(name)
    level = target.level
    target.setLevel(logging.WARNING)
    try:
        yield
    finally:
        target.setLevel(level)


@dataclass
class _SnapshotEval:
    request: PulseRequest
    blob_id: str
    pending: set[str]
    total: int
    results: dict[str, list[MatchResult]] = field(default_factory=dict)


class HubPulse:
    """Pulse on evaluator blocks (spec 5.3; replaces the SP4 pulse child, R6). Main loop only.

    ``offer`` (after every checkpoint save) records the snapshot in the ``PulseFeeder`` and, when it is queued and
    the run is not stopping, hard-links its ``model.pt`` / ``meta.json`` into
    ``runs/<name>/eval/pulse/<agent>/<checkpoint id>/`` (retention may evict the snapshot meanwhile; SP4 kept its
    weights in memory instead). ``start_next`` turns the oldest waiting snapshot into blocks: SP4's stage-P
    gauntlet (``pulse_blocks``: the same seeds for every snapshot; uids
    ``pulse-<run id[:12]>/P/<layout>/<agent>@v<N>/<composition>/<k>/r<rotation>``) as ``BlockTask``s of owner
    ``pulse:<agent>:<checkpoint id>``, ``PULSE_PRIORITY``, ``evaluation.num_envs`` envs, ``evaluation.deterministic``,
    its weights registered as the eval blob ``pulse-<agent>-<checkpoint id>``. ``on_block`` collects a block's results
    (``env_steps`` = the snapshot's); when every block is in, the snapshot is evaluated and its results (sorted by
    uid) are returned for the journal, as SP4's one message per snapshot. ``on_failed`` skips a snapshot whose block
    failed. ``stop``: no snapshot starts any more; ``finish`` (before the journal writer stops): snapshots still in
    flight stay unevaluated (INFO), every link and blob is removed. Frozen field players are registered once as
    ``field-<name>``; scripted ones travel as bots. Old links are removed at start (``pulse_state.json`` is not
    resumed, SP4 ruling)."""

    def __init__(self, config: ColosseumConfig, spec: GameSpec, plan: PulsePlan, *, run_id: str,
                 field_players: Mapping[str, TournamentPlayer], eval_dir: Path, blobs: EvalBlobs) -> None:
        ev = config.evaluation
        self.feeder = PulseFeeder(plan.agents, ev.every, ev.max_backlog)
        self._config, self._spec, self._plan, self._blobs = config, spec, plan, blobs
        self._seed = config.training.seed if config.training.seed is not None else 0
        self._prefix = f"pulse-{run_id[:12]}"
        self._dir = Path(eval_dir) / "pulse"
        shutil.rmtree(self._dir, ignore_errors=True)
        self._field = {name: field_players[name] for name in plan.field}
        self._field_refs: dict[str, PlayerRef] = {}
        self._field_blobs: list[str] = []
        for name, player in self._field.items():
            blob_id = None
            if player.source.kind == "model":
                blob_id = f"field-{name}"
                blobs.register_weights(blob_id, Path(player.source.weights), agent_id=player.agent_id,
                                       networks=dict(player.source.networks or {}), roles=player.roles)
                self._field_blobs.append(blob_id)
            self._field_refs[name] = player_ref(player, blob_id)
        self._evals: dict[str, _SnapshotEval] = {}
        self._owner_of: dict[str, str] = {}
        self._reported: set[str] = set()
        self.accepting = True

    def offer(self, agent_id: str, checkpoint_id: str, version: int, env_steps: int, snapshot_dir: Path, *,
              final: bool = False) -> bool:
        if agent_id not in self._plan.agents:
            return False
        target = self._dir / agent_id / checkpoint_id
        queued = self.feeder.offer(agent_id, checkpoint_id, version, env_steps, final=final,
                                   weights=str(target / "model.pt"))
        for old in self.feeder.take_dropped():
            shutil.rmtree(self._dir / old.agent_id / old.checkpoint_id, ignore_errors=True)
        if queued and self.accepting:
            _link_snapshot(Path(snapshot_dir), target)
        return queued

    def start_next(self) -> list[BlockTask]:
        if not self.accepting:
            return []
        request = self.feeder.start_next()
        if request is None:
            return []
        agent = self._plan.agents[request.agent_id]
        owner = pulse_owner(request.agent_id, request.checkpoint_id)
        blob_id = f"pulse-{request.agent_id}-{request.checkpoint_id}"
        name = f"{request.agent_id}@v{request.version}"
        origin = f"pulse snapshot {request.checkpoint_id} of {request.agent_id}"
        candidate = TournamentPlayer(name=name, display=name, kind="snapshot", agent_id=request.agent_id,
                                     network_id=request.checkpoint_id, roles=agent.roles, lineage=request.agent_id,
                                     source=PlayerSource("model", origin=origin, weights=request.weights,
                                                         networks=dict(agent.networks)), version=request.version)
        dropped: list[DroppedComposition] = []
        with _quiet(_SCHEDULE_LOGGER):     # the same drops for every snapshot: logged once per agent below
            blocks = pulse_blocks(self._spec, agent.layouts, name, list(self._field), {name: candidate, **self._field},
                                  blocks=self._config.evaluation.blocks, seed=self._seed, prefix=self._prefix,
                                  dropped=dropped)
        if request.agent_id not in self._reported:
            self._reported.add(request.agent_id)
            for d in dropped:
                logger.info(f"pulse: agent {request.agent_id!r}, layout {d.layout!r}: composition "
                            f"{'+'.join(d.composition) or '(homogeneous team)'} is not played: {d.reason}")
        check_blocks(blocks, players=[candidate, *self._field.values()])
        self._blobs.register_weights(blob_id, Path(request.weights), agent_id=request.agent_id,
                                     networks=dict(agent.networks), roles=agent.roles)
        ref = player_ref(candidate, blob_id)
        ev = self._config.evaluation
        tasks = [BlockTask(block_id=b.id, source="pulse", owner=owner, lineups=tuple(b.lineups),
                           players=(ref, *(self._field_refs[f] for f in b.composition)), priority=PULSE_PRIORITY,
                           deterministic=ev.deterministic, num_envs=ev.num_envs) for b in blocks]
        self._evals[owner] = _SnapshotEval(request, blob_id, {t.block_id for t in tasks}, len(tasks))
        for task in tasks:
            self._owner_of[task.block_id] = owner
        return tasks

    def on_block(self, block_id: str, results: Sequence[MatchResult]) -> list[MatchResult] | None:
        owner = self._owner_of.get(block_id)
        state = self._evals.get(owner) if owner is not None else None
        if state is None or block_id not in state.pending:
            return None
        for result in results:
            result.env_steps = state.request.env_steps
        state.results[block_id] = list(results)
        state.pending.discard(block_id)
        if state.pending:
            return None
        self._close(owner)
        self.feeder.done(state.request.agent_id, state.request.checkpoint_id)
        played = sorted((r for rs in state.results.values() for r in rs), key=lambda r: r.uid)
        logger.info(f"pulse: {state.request.agent_id}@v{state.request.version} evaluated ({len(played)} matches)")
        return played

    def on_failed(self, owner: str, reason: str) -> None:
        state = self._evals.get(owner)
        if state is not None:
            self._close(owner)
            self.feeder.fail(state.request.agent_id, state.request.checkpoint_id, reason)

    def _close(self, owner: str) -> None:
        state = self._evals.pop(owner)
        for block_id in [b for b, o in self._owner_of.items() if o == owner]:
            del self._owner_of[block_id]
        self._blobs.unregister(state.blob_id)
        shutil.rmtree(self._dir / state.request.agent_id / state.request.checkpoint_id, ignore_errors=True)

    def owners(self) -> list[str]:
        return list(self._evals)

    def stop(self) -> None:
        self.accepting = False

    def finish(self) -> list[str]:
        """At the end: snapshots in flight stay unevaluated (INFO); every link and blob is removed. Returns their
        owners (the caller cancels their blocks)."""
        self.accepting = False
        owners = list(self._evals)
        for owner in owners:
            state = self._evals[owner]
            logger.info(f"pulse: snapshot {state.request.checkpoint_id} of {state.request.agent_id} was not evaluated "
                        f"(the run stopped first; {state.total - len(state.pending)} of {state.total} blocks played)")
            self._close(owner)
        for blob_id in self._field_blobs:
            self._blobs.unregister(blob_id)
        self._field_blobs = []
        shutil.rmtree(self._dir, ignore_errors=True)
        return owners


def _link_snapshot(src: Path, dst: Path) -> None:
    """``model.pt`` + ``meta.json`` of ``src`` hard-linked (else copied) into ``dst`` (a tmp dir renamed)."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".tmp-{dst.name}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        for name in _MODEL_FILES:
            link_or_copy(src / name, tmp / name)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
```
`finish` keeps the in-flight requests in the feeder (they stay "unevaluated" in `pulse_state.json`, as SP4's request being played at the stop).

- [ ] **Step 12 (b): `BlobStore` evaluation blobs and `Download(kind="eval")`**

`src/colosseum/hub/server.py`, class `BlobStore` only (thread-safe: one `threading.Lock` over its registration dict; the main loop registers, servicer threads read):
- `register_weights(blob_id, weights_file, *, agent_id, networks, roles) -> None`: `blob_id` must match `^[A-Za-z0-9_][A-Za-z0-9_.-]*$` (ValueError otherwise; it becomes a file name); records `(Path(weights_file), agent_id, dict(networks), tuple(roles))`; registering an id again with the same file is a no-op, with another file replaces it and removes its old converted frame;
- `unregister(blob_id) -> None`: drops the record and deletes `runs/<name>/blobs/eval-<agent>-<blob_id>.frame` if present (`missing_ok`); unknown ids are ignored;
- `frame_path("eval", agent_id, ident)`: `None` for an unregistered ident (→ the servicer's `NOT_FOUND`, as for an unknown snapshot) or an agent id that differs from the registration; otherwise the converted frame, made on first use by the servicer thread: `read_weights_file(weights_file)` → `encode_weights(state, WeightsMeta(agent_id, kind="fixed", generation=0, policy_version=0, architecture=networks, roles=roles))` → written to a temp file in `blobs/` and renamed into place **under the lock after re-checking that the id is still registered** (a conversion racing `unregister` deletes its temp file instead: no frame survives its blob); a weights file that cannot be read → `None` with one ERROR naming the blob and the file (the node reports `missing_weights`; the block is replayed and fails its owner after `MAX_BLOCK_ATTEMPTS`).
- The `Download` servicer accepts `BlobRequest.kind == "eval"` and routes it to `frame_path` like the other kinds (the T2.3 dispatch table gains one entry; the node's T3.1 fetch helper needs no change beyond accepting the kind).

- [ ] **Step 13 (b): The node agent's evaluation section**

`src/colosseum/node/agent.py` (a new section; one `NodeLeases` instance owned by the main loop):
- **Slots and demand.** `Hello.slots.evaluators` = the node's evaluator slots (T2.4 `default_slots` / `NodeOptions.evaluators`; the local node of `train`: 1, Step 14). The number to run is `Welcome.evaluators`, later `Command(kind="set_evaluators", evaluators=n)`: start `evaluator-<i>` for every `i < n` not running (`ChildSpec(name=f"evaluator-{i}", role="evaluator", target=evaluator_target, kwargs=...)` with the same connection / cache / config / log kwargs a worker gets, plus `index=i`, `node_address` = the node's listener name, `cache_root` / `cache_prefix` of the node's cache); stop every `i >= n`: `leases.evaluator_down(i, error_kind="abandoned")` at once (report the outcomes), then `Stop` on its control connection (`ChildSupervisor` ends it within its bound). `n` never exceeds the slots.
- **Child events.** A `ChildHello(role="evaluator")` on a control connection → `leases.evaluator_up(i)`; `EvalDone` → `leases.on_done(i, ...)`; `EvalFailed` → `leases.on_failed(i, lease_id, kind, f"{message} ({where}; log {evaluator log path})")`; an evaluator's exit (crash, kill, OOM) → `leases.evaluator_down(i, error_kind="crash", error=<exit code and log path>)` before T4.2's restart logic restarts it (`WORKER_RESTARTS`; a disabled slot stays down: its index is not restarted, `Event(slot_disabled)` as for workers). A typed evaluator error is **not** a `ChildError` of the run: it reaches the hub only as a failed `BlockResult`.
- **Hub messages.** `HubMessage.lease` → `NodeLease(lease_id, source, [block_task_from_proto(b, num_envs=lease.num_envs) ...], lease.num_envs, deterministic)` → `leases.on_lease(lease, have=cache_reader.has)` → for each returned key not already being fetched: the T3.1 transfer-pool fetch of `BlobRequest(kind="eval", agent_id=key.agent_id, ident=key.ident)` into `WeightCacheWriter.put(key, frame)` (frames are checked with `decode_weights(..., cap=payload_byte_cap(max_message_mb))` before the write, as T3.1 does for snapshots); its outcome posts `leases.on_fetched(key, ok)`. `block_ack` → `leases.on_ack`. `Command(cancel_leases)` → for each `(i, lease_id)` of `leases.on_cancel(...)`: `EvalCancel(lease_id)` on evaluator `i`'s control connection.
- **Every main-loop pass** (after the messages): `leases.assignments()` → `EvalTask(lease_id, source, blocks, num_envs, deterministic)` on that evaluator's connection; every `BlockOutcome` the calls above returned → `NodeMessage(block_result=pb.BlockResult(lease_id, block_id, results=[result_to_proto(r) ...], failed, error_kind, error))` on the session (when not connected they stay unacked and go after the reconnect); `free = leases.request(connected=session.connected)` → `NodeMessage(lease_request=pb.BlockLeaseRequest(free_slots=free))` when `free > 0`; `leases.drop_candidates()` → `WeightCacheWriter.drop(key)` for each (only evaluation blobs: `NodeLeases` tracks only keys of leases).
- **Reconnect** (the `Hello` builder of T2.3/T2.4): `held, resend = leases.on_reconnect()` → `Hello.leases = held`; after the `Welcome`, the `resend` outcomes go before any new lease request.
- **Stop** (`Command(stop_run)` / node stop, T4.4): evaluators are stopped with the other children; their leases are not reported (the hub is stopping; a resumed hub knows no lease and cancels the ones named in `Hello.leases`).

- [ ] **Step 14 (b): The hub wiring; the pulse child removed**

`src/colosseum/hub/hub.py` (`Hub`; only these points):
- **Construction** (where T2.5 builds the pulse feeder and queues): build `self._evaluation = HubEvaluation(lease_sec=config.distributed.block_lease_sec, send=self._send_to_node, pulse=pulse, journal=self._submit_pulse_results, fail_run=self._fail_run)`, with `pulse = HubPulse(config, spec, pulse_plan(config, spec, setup.player_roles), run_id=self._run_id, field_players={n: config_player(config, spec, n, setup.fixed) for n in plan.field}, eval_dir=self._run_dir.root / "eval", blobs=self._blobs)` when `config.evaluation.enabled` (else `None`). `_send_to_node(number, msg)` = the online session's `SessionHandle.send(msg)`; nothing for a node without a session. `_submit_pulse_results(results)` = `self._pulse_results_received += len(results)` then `self._match_log_writer.submit(results)` (no token: SP4 semantics — a drop is counted by the writer; never the coordinator or `MetricsHub`, amendment A20).
- **Removed** (R6): the pulse `mp.Process` (`pulse_target`), `self._pulse_requests`, `self._pulse_results`, their entries in the queue release / `_main_produced_queues` / supervisor lists and the `_drain(..., "pulse-results", ...)` call; `ipc.SharedCounter` / `QueueReader` imports the hub no longer needs (T7.2 removes what the old roles still use).
- **`_save_checkpoint(payload)`**: after `save_checkpoint_payload` returns the checkpoint id, `if self._evaluation.pulse is not None: self._evaluation.pulse.offer(aid, checkpoint_id, version, env_steps, self._run_dir.checkpoints / aid / checkpoint_id, final=bool(payload.get("final")))` (the just-saved snapshot always survives its own save: put-then-evict, R-C1).
- **`_pump_pulse()`** (every main-loop pass): `self._evaluation.tick()`; the system values `pulse_skipped_snapshots` / `pulse_evaluated_snapshots` / `pulse_pending_snapshots` from `pulse.feeder` (when the pulse is on) and `evaluation` = `self._evaluation.stats()` (always); `_write_pulse_state()` when `feeder.changed` (unchanged).
- **Sessions** (`_on_session_opened`): after the node registry's `connect(...)` → `join = self._evaluation.node_joined(number, slots=hello.slots.evaluators, held_leases=list(hello.leases), fresh=<connect returned "new">)`; `Welcome.evaluators = join.evaluators`; send the Welcome, then `join.after_welcome`. `_on_session_closed` → `session_closed(number)`; the node-loss check (`NodeRegistry.lost`) → `node_lost(number)` for each lost node; a removed node (`nodes remove`, T6.3) → `node_lost` as well.
- **Node messages** (`_on_node_message` dispatch by `WhichOneof`): `lease_request` → `on_lease_request(number, msg.lease_request.free_slots)`; `block_result` → `on_block_result(number, msg.block_result)`.
- **Stop**: when the stop begins (`_shutdown`'s first step) → `self._evaluation.stop()`; `_finish_metrics`: replace the pulse drain before the writer stops with `self._evaluation.finish()` and keep `_finish_pulse()` after it (the INFO per unevaluated final snapshot; `pulse_state.json`).
- **`run_training` (R12)**: `LocalNodeOptions.evaluators = 1` always (the slot); the hub runs it only while evaluation work can exist (proposed ruling 2), so a run without pulse still starts no evaluator process.

- [ ] **Step 15 (b): GREEN; full suite; slow suite**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_pulse_blocks.py tests/unit/test_sp4_pulse.py tests/integration/test_sp5_node_evaluation.py tests/integration/test_sp5_hub_evaluation.py tests/integration/test_sp4_pulse_runs.py -q`
Expected: all pass.

Then `grep -rn "pulse_target\|pulse_main\|PulseResults\|pulse\.log" src tests scripts` → no hits outside docs (docs are T7.5's). Full fast suite + ruff (zero warnings). The pulse path of training changed, so run the slow suite once, alone, in the background: `env -u PYTHONPATH .venv/bin/python -m pytest -m slow -v` (in particular `tests/learning/test_sp4_pipeline_slow.py`: pulse ρ ≥ 0.85 with the pulse on an evaluator; its thresholds are unchanged).

- [ ] **Step 16 (b): Commit, push, report**

Commit: `feat: pulse on node evaluators; the hub's pulse child removed (R6)`; push. The task report lists every SP4 pulse test ported (and how) or deleted (`test_the_pulse_process_plays_every_snapshot_on_the_same_seeds`, with the SP5 tests that carry its assertions), and the slow-suite numbers of the pipeline test.


### Task T5.2: Hub-run tournaments, `tournament --hub`, `hub --idle`

Spec 5.3 (the tournament on a running hub, `hub --idle`), 3.3 (admin target resolution), 7.5 (client exit codes), 1 (`hub --idle`). `colosseum tournament --hub ADDR|RUN_DIR ...` sends the tournament's description (`TournamentOptions` as JSON, the `-c` config's text) to a running hub (`train`, `hub` or `hub --idle`) over `AdminStream("tournament.submit")`. The hub runs SP4's `run_tournament` in a thread of its own (`HubTournament`) whose `play` hook queues the blocks with the hub's evaluators (`TournamentBlocks` inbox event → `HubEvaluation.submit`, priority 0: before the pulse) and waits for their results (put by the main loop into the tournament's queue); everything else — schedule, halving, confirmation, whole-block journaling in one transaction each, resume by `--name`, snapshot links, BT, reports, `--pin` — is SP4 code. The tournament's root is its first `--run` (refused when another live process owns that run dir, spec 5.3) or the hub's own run dir; candidate paths are read on the hub machine; neural players reach the evaluators as evaluation blobs (`t-<hash8>-<hash12>`). The client prints the progress; Ctrl+C detaches it (130; the tournament goes on); `--cancel NAME` cancels; 0 finished (also `confirmed: false`), 1 cancelled / failed / the hub stopped, 2 bad arguments. `hub_tournaments.json` in the hub's run dir records every hub-run tournament; after `hub --resume` (or `train --resume`) the ones still `running` continue (SP4 resume: whole blocks reused from the journal). `colosseum hub --idle -c cfg` starts a hub without training: no learners, no workers on the nodes (`Welcome.no_workers`), no pulse, no budget (it runs until a signal), evaluators on demand.

**Split point:** dispatch **(a) Steps 1–6** (the `run_tournament` hooks, `HubTournament` / `TournamentService` and their in-process tests), commit and push; then **(b) Steps 7–12** (the hub wiring of the admin commands and events, resume and stop, `hub --idle`, `Welcome.no_workers`, the CLI client, the integration tests), commit and push.

**Files:**
- Modify (a): `src/colosseum/arena/tournament/run.py` (`TournamentHooks`, `run_tournament(..., hooks=None)`, `BlockRunner(play=, progress=)`, `_journal_run(..., home=)`, `_file_logging(..., this_thread=)`); `src/colosseum/hub/evaluation.py` (T5.2 part)
- Modify (b): `src/colosseum/net/proto/colosseum_v1.proto` + generated (`Welcome.no_workers`); `src/colosseum/hub/hub.py` (`_on_admin` `tournament.*`, `_on_tournament_blocks`, `_on_tournament_finished`, the start, the stop, `mode == "idle"`); `src/colosseum/node/agent.py` (`Welcome.no_workers`); `src/colosseum/cli.py` (`tournament_cmd`: `--hub`, `--token`, `--token-file`, `--cancel`, `--num-workers` default; the `hub` group's `--idle`); `tests/cli_runner.py` (`tournament_hub_cmd`)
- Test (a): `tests/unit/test_sp5_tournament_hooks.py`, `tests/unit/test_sp5_hub_tournaments.py`
- Test (b): `tests/integration/test_sp5_hub_idle.py`, `tests/integration/test_sp5_tournament_hub_cli.py` (one slow test)

**Interfaces:**
- Consumes: T5.1's `HubEvaluation` (`add_owner`, `submit`, `cancel`, `remove_owner`, `stats`), `player_ref`, `EvalBlobs`, `BlockTask`, `TOURNAMENT_PRIORITY`, `eval_kit`; SP4 `run_tournament`, `TournamentOptions`, `RunSelection`, `check_blocks`, `TournamentError`, `read_run_state` / `process_alive`; T2.5 `colosseum.hub.admin.resolve_hub` / `call` / `stream`, `AdminIn`, `HubOptions.mode`, `run_hub(..., idle=)`, `read_hub_json`; T4.3 `hub --resume`; T3.1 `cluster_kit` (a hub process and `NodeAgent` threads).
- Produces (contract `HubTournament`, plus additions marked *): *`TournamentHooks(play, home, progress)`, *`run_tournament(options, *, hooks=None)`; in `colosseum.hub.evaluation`: `HubTournament`, *`TournamentService`, *`TournamentProgress`, *`Reply`, *`reply_to_proto`, *`ProtoReplies`, *`TournamentBlocks`, *`TournamentFinished`, *`TournamentCancelled`, *`TournamentInterrupted`, *`options_to_dict` / *`options_from_dict`, *`check_root`, *`TOURNAMENTS_FILE = "hub_tournaments.json"`, *`PROGRESS_SEC = 2.0`, *`SUBSCRIBER_BACKLOG = 256`, *`LAST_LINES = 20`, *`JOIN_SEC = 2.0`; *proto `Welcome.no_workers`; admin commands `tournament.submit` (stream), `tournament.status`, `tournament.cancel` (contract `ADMIN_COMMANDS`); CLI `tournament --hub/--token/--token-file/--cancel`, `hub --idle`; *run-dir files `hub_tournaments.json`, `tournaments/<name>/submitted-config.yaml`.

- [ ] **Step 1 (a): Write the failing tests**

Create `tests/unit/test_sp5_tournament_hooks.py`:
```python
"""``run_tournament(..., hooks=...)`` (SP5 T5.2): a hub's play function replaces the engine, the tournament lives in
the hub's run dir without ``--run`` (journal under the hub run's id), progress lines, a thread-only
``tournament.log``; without hooks nothing changes."""
from __future__ import annotations

import logging
import threading

import colosseum.arena.tournament.run as run_module
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
from colosseum.arena.tournament.engine import LazyPlayerPool
from colosseum.arena.tournament.run import TournamentHooks, TournamentOptions, run_tournament
from colosseum.core.registry import make_env
from colosseum.core.run_dir import RunDir
from colosseum.eval import play_lineups
from game_helpers import scripted_agent, write_fake_run

AGENTS = {"main": {}, "rnd": scripted_agent()}


def _local_play(calls):
    def play(config, spec, players, blocks, on_block):
        calls.append(len(blocks))
        pool = LazyPlayerPool(players.values(), spec)
        for block in blocks:
            pool.reserve({(s.agent_id, s.network_id) for lu in block.lineups for s in lu.seats})
            results = play_lineups(env_fn=lambda: make_env(config), pool=pool, lineups=list(block.lineups),
                                   num_envs=2, seed=block.seed, source="tournament", match_id_prefix="tournament")
            on_block(block.id, sorted(results, key=lambda r: r.uid))
    return play


def test_hooks_play_the_blocks_in_the_hub_run_dir(tmp_path):
    root, _cfg = write_fake_run(tmp_path / "hubrun", "turns", agents=AGENTS, versions=(10,))
    run_id = RunDir.open(root).ensure_run_id()
    calls, lines = [], []
    options = TournamentOptions(mode="gauntlet", candidates=(("c", str(root / "checkpoints" / "main" / "ckpt_v10")),),
                                field=("rnd",), blocks=2, num_envs=2, name="h",
                                config_path=str(root / "config.resolved.yaml"))
    outcome = run_tournament(options, hooks=TournamentHooks(play=_local_play(calls), home=root,
                                                            progress=lines.append))
    assert calls == [2] and outcome.played_blocks == 2 and outcome.directory == root / "tournaments" / "h"
    assert (root / "tournaments" / "h" / "report.json").is_file() and lines and "2 blocks to play" in lines[0]
    reader = MatchLog.open_readonly(root / MATCH_LOG_FILE)
    try:
        runs = {r.run_id: r for r in reader.runs()}
        results = reader.match_results(source="tournament", tournament="h")
    finally:
        reader.close()
    assert list(runs) == [run_id] and len(results) == 4               # the hub run's id, no tournament "run"
    again = run_tournament(options, hooks=TournamentHooks(play=_local_play(calls), home=root))
    assert again.played_blocks == 0 and again.reused_blocks == 2 and calls == [2]   # resume from the journal


def test_the_tournament_log_holds_only_its_thread(tmp_path):
    root, _cfg = write_fake_run(tmp_path / "hubrun", "turns", agents=AGENTS, versions=(10,))
    options = TournamentOptions(mode="gauntlet", runs=(run_module.RunSelection(path=root),), field=("rnd",), blocks=1,
                                num_envs=2, name="t")
    stop = threading.Event()

    def noise():
        while not stop.is_set():
            logging.getLogger("colosseum.hub.hub").info("noise from the main loop")
            stop.wait(0.001)

    thread = threading.Thread(target=noise)
    thread.start()
    try:
        run_tournament(options, hooks=TournamentHooks(play=_local_play([]), home=root))
    finally:
        stop.set()
        thread.join()
    log = (root / "tournaments" / "t" / "logs" / "tournament.log").read_text()
    assert "Tournament t" in log and "noise" not in log


def test_without_hooks_nothing_changes(tmp_path, monkeypatch):
    root, _cfg = write_fake_run(tmp_path / "run", "turns", agents=AGENTS, versions=(10,))
    seen = []

    def fake_play_blocks(config, spec, players, blocks, *, on_block, **kwargs):
        seen.append(kwargs["num_workers"])
        _local_play([])(config, spec, players, blocks, on_block)

    monkeypatch.setattr(run_module, "play_blocks", fake_play_blocks)
    options = TournamentOptions(mode="gauntlet", runs=(run_module.RunSelection(path=root),), field=("rnd",), blocks=1,
                                num_envs=2, num_workers=2, name="l")
    assert run_tournament(options).played_blocks == 1 and seen == [2]
```

Create `tests/unit/test_sp5_hub_tournaments.py` (no process: the evaluators are `eval_kit.EvalNode` threads):
```python
"""Hub-run tournaments (SP5 T5.2, spec 5.3): TournamentService + HubEvaluation + in-process eval_kit nodes (one
evaluator thread each, no process): select on a run, cancel, resume from the journal, ownership of run dirs,
duplicates, status and follow, progress for clients, a failing block, the hub stopping, priority over the pulse."""
from __future__ import annotations

import json
import queue
from pathlib import Path

import pytest

from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
from colosseum.arena.tournament.players import RunSelection
from colosseum.arena.tournament.run import TournamentOptions
from colosseum.core.run_state import write_run_state
from colosseum.hub.evaluation import (
    SUBSCRIBER_BACKLOG,
    HubEvaluation,
    Reply,
    TournamentBlocks,
    TournamentFinished,
    TournamentProgress,
    TournamentService,
    options_from_dict,
    options_to_dict,
)
from eval_kit import EvalNode, FakeBlobs, drive, route
from game_helpers import scripted_agent, write_fake_run

AGENTS = {"main": {}, "rnd": scripted_agent(), "ones": scripted_agent("game_helpers.ConstantBot", value=1),
          "bad": scripted_agent("game_helpers.PulseFailBot")}
STRONG, UNIFORM, WEAK = [20.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-20.0, 20.0, 0.0]


class Hubby:
    """The hub's pieces a tournament needs, driven by the test (``drive``)."""

    def __init__(self, tmp_path, *, evaluators=1, versions=(10, 20, 30)):
        self.root, self.cfg = write_fake_run(tmp_path / "run", "turns", agents=AGENTS, versions=versions,
                                             biases={10: WEAK, 20: UNIFORM, 30: STRONG})
        self.blobs, self.inbox, self.nodes = FakeBlobs(), queue.SimpleQueue(), {}
        self.evaluation = HubEvaluation(lease_sec=600, send=route(self.nodes))
        for number in range(1, evaluators + 1):
            self.add_node(number)
        self.service = TournamentService(home=self.root, config_path=self.root / "config.resolved.yaml",
                                         evaluation=self.evaluation, blobs=self.blobs, post=self.inbox.put,
                                         evaluators=lambda: self.evaluation.stats()["evaluators"])

    def add_node(self, number):
        self.nodes[number] = EvalNode(self.cfg, self.blobs, self.evaluation, number=number)
        self.nodes[number].join()

    def on_event(self, event):
        if isinstance(event, TournamentBlocks):
            self.service.on_blocks(event)
        elif isinstance(event, TournamentFinished):
            self.service.on_finished(event)

    def run(self, until, timeout=120.0):
        drive(self.evaluation, list(self.nodes.values()), until, inbox=self.inbox, on_event=self.on_event,
              timeout=timeout)

    def submit(self, **kw):
        replies = queue.SimpleQueue()
        options = TournamentOptions(runs=(RunSelection(path=self.root),), num_envs=2, **kw)
        self.service.submit({"options": options_to_dict(options)}, replies)
        return replies

    def registry(self):
        return json.loads((self.root / "hub_tournaments.json").read_text())["tournaments"]

    def journal(self, name):
        reader = MatchLog.open_readonly(self.root / MATCH_LOG_FILE)
        try:
            return reader.blocks(name), reader.match_results(source="tournament", tournament=name)
        finally:
            reader.close()

    def close(self):
        for node in self.nodes.values():
            node.close()


@pytest.fixture
def hub(tmp_path):
    h = Hubby(tmp_path)
    yield h
    h.close()


def _drain(replies, into):
    while True:
        try:
            into.append(replies.get_nowait())
        except queue.Empty:
            return


def _done(replies, got, service):
    def check():
        _drain(replies, got)
        return any(r.done for r in got) and not service.active
    return check


def test_select_on_a_run_runs_on_the_hub_evaluators(hub):
    replies, got = hub.submit(mode="select", field=("rnd",), halving_blocks=2, confirm_blocks=6, finalists=2, seed=3,
                              name="sel"), []
    assert hub.evaluation.active                                       # demand: the node's evaluator runs
    hub.run(_done(replies, got, hub.service))
    assert got[0].data == {"name": "sel"} and "submitted" in got[0].text
    assert any("blocks to play" in r.text for r in got)
    final = got[-1]
    assert final.done and final.code == 0 and "Blocks:" in final.text and "Written:" in final.text, final.text
    report = json.loads((hub.root / "tournaments" / "sel" / "report.json").read_text())
    (selection,) = report["selection"].values()
    assert selection["winner"] == "main@v30"
    blocks, results = hub.journal("sel")
    assert blocks and len({r.uid for r in results}) == len(results)
    assert hub.registry()["sel"]["status"] == "finished" and not hub.evaluation.active and hub.blobs.live == {}
    assert hub.nodes[1].fetches == 3                                   # one fetch per snapshot and machine
    log = (hub.root / "tournaments" / "sel" / "logs" / "tournament.log").read_text()
    assert "Tournament sel" in log


def test_cancel_withdraws_the_leases_and_keeps_the_journal(hub):
    replies, got = hub.submit(mode="gauntlet", field=("rnd",), blocks=20, name="g"), []
    hub.run(lambda: hub.evaluation.queue.stats()["completed"] >= 5)
    assert hub.service.cancel("g").code == 0 and hub.service.cancel("nope").code == 1
    hub.run(_done(replies, got, hub.service))
    assert got[-1].code == 1 and "cancelled" in got[-1].text
    assert hub.registry()["g"]["status"] == "cancelled" and hub.evaluation.queue.live("tournament:g") == 0
    held = len(hub.journal("g")[0])
    assert 5 <= held < 60


def test_a_running_tournament_resumes_from_the_journal(hub):
    replies, got = hub.submit(mode="gauntlet", field=("rnd",), blocks=20, name="g"), []
    hub.run(lambda: hub.evaluation.queue.stats()["completed"] >= 5)
    hub.service.cancel("g")
    hub.run(_done(replies, got, hub.service))
    held = len(hub.journal("g")[0])
    data = json.loads((hub.root / "hub_tournaments.json").read_text())
    data["tournaments"]["g"]["status"] = "running"                     # as a killed hub leaves it
    (hub.root / "hub_tournaments.json").write_text(json.dumps(data))
    before = hub.evaluation.queue.stats()["completed"]
    assert hub.service.resume() == ["g"]
    hub.run(lambda: not hub.service.active)
    assert hub.registry()["g"]["status"] == "finished"
    assert hub.evaluation.queue.stats()["completed"] - before == 60 - held      # the rest only
    assert len(hub.journal("g")[0]) == 60


def test_a_run_owned_by_another_live_hub_is_refused(hub, tmp_path):
    import os

    other, _cfg = write_fake_run(tmp_path / "other", "turns", agents=AGENTS, versions=(10,))
    write_run_state(other, "running", os.getppid())                  # a live process that is not this hub
    replies = queue.SimpleQueue()
    options = TournamentOptions(mode="gauntlet", runs=(RunSelection(path=other),), field=("rnd",), blocks=1, name="x")
    hub.service.submit({"options": options_to_dict(options)}, replies)
    reply = replies.get_nowait()
    assert reply.done and reply.code == 1 and "submit the tournament to that hub" in reply.text
    assert hub.service.active == [] and not (other / "tournaments" / "x").exists()


def test_a_second_tournament_with_the_name_of_a_running_one_is_refused(hub):
    hub.submit(mode="gauntlet", field=("rnd",), blocks=20, name="g")
    second = hub.submit(mode="gauntlet", field=("rnd",), blocks=1, name="g")
    reply = second.get_nowait()
    assert reply.done and reply.code == 1 and "is running on this hub" in reply.text
    hub.service.cancel("g")
    hub.run(lambda: not hub.service.active)


def test_status_and_follow(hub):
    replies, got = hub.submit(mode="gauntlet", field=("rnd",), blocks=2, name="g"), []
    late = queue.SimpleQueue()
    hub.service.follow("g", late)
    hub.run(_done(replies, got, hub.service))
    follow = []
    _drain(late, follow)
    assert follow[0].text == got[0].text and follow[-1].done and follow[-1].code == 0   # replayed, then live
    status = hub.service.status({"name": "g"})
    assert status.code == 0 and status.data["g"]["status"] == "finished" and status.data["g"]["active"] is False
    assert hub.service.status({"name": "nope"}).code == 1
    gone = queue.SimpleQueue()
    hub.service.follow("g", gone)
    assert gone.get_nowait().code == 1


def test_progress_replay_and_detached_subscribers():
    progress = TournamentProgress()
    early, stalled = queue.SimpleQueue(), queue.SimpleQueue()
    progress.subscribe(early)
    progress.subscribe(stalled)
    for k in range(SUBSCRIBER_BACKLOG + 5):
        progress.line(f"line {k}")
        if not early.empty():
            early.get_nowait()                                  # an attached client reads; ``stalled`` never does
    progress.publish(Reply(0, "end", done=True))
    assert stalled.qsize() == SUBSCRIBER_BACKLOG                # dropped once its backlog was full
    late = queue.SimpleQueue()
    progress.subscribe(late)
    replay = []
    _drain(late, replay)
    assert replay[-1].done and len(replay) == 20 + 1            # the last LAST_LINES lines, then the final reply


def test_a_failing_block_fails_the_tournament_with_its_error(hub, monkeypatch):
    import colosseum.eval
    from colosseum.core.errors import PlayerError

    def failing(**_kwargs):                                   # the evaluator's play: a bot error in a match
        raise PlayerError("tournament, node 1, evaluator 0, env 0, seat 1: bot raised on purpose")

    monkeypatch.setattr(colosseum.eval, "play_lineups", failing)
    replies, got = hub.submit(mode="gauntlet", field=("rnd",), blocks=2, name="b"), []
    hub.run(_done(replies, got, hub.service))
    final = got[-1]
    assert final.done and final.code == 1 and final.text.startswith("Tournament error")
    assert "bot raised on purpose" in final.text and "failed on node 1" in final.text
    entry = hub.registry()["b"]
    assert entry["status"] == "failed" and entry["error"].startswith("PlayerError")
    assert hub.evaluation.queue.live("tournament:b") == 0


def test_the_hub_stopping_interrupts_and_keeps_it_resumable(hub):
    replies, got = hub.submit(mode="gauntlet", field=("rnd",), blocks=20, name="g"), []
    hub.run(lambda: hub.evaluation.queue.stats()["completed"] >= 2)
    hub.service.stop()
    hub.run(_done(replies, got, hub.service))
    assert got[-1].code == 1 and "hub --resume" in got[-1].text and hub.registry()["g"]["status"] == "running"
    refused = queue.SimpleQueue()
    hub.service.submit({"options": options_to_dict(TournamentOptions(runs=(RunSelection(path=hub.root),),
                                                                     name="h"))}, refused)
    assert "stopping" in refused.get_nowait().text


def test_tournament_blocks_go_before_pulse_blocks(tmp_path):
    from colosseum.core.types import Lineup, SeatAssignment
    from colosseum.net.messages import BlockTask, PlayerRef
    from colosseum.players.registry import BotSpec

    hub = Hubby(tmp_path, evaluators=0)
    try:
        bot = PlayerRef("rnd", "rnd", "fixed", ("player",), "scripted", bot=BotSpec("colosseum.players.RandomBot"))
        block = "p/P/2p/x/rnd/0"
        pulse_task = BlockTask(block, "pulse", "pulse:x:ckpt_v1", (Lineup(
            "2p", [SeatAssignment("rnd", "fixed", False), SeatAssignment("rnd", "fixed", False)], seed=1, block=block,
            uid=f"{block}/r0"),), (bot,), 1, False, 1)
        hub.evaluation.queue.submit([pulse_task])                       # queued first
        hub.submit(mode="gauntlet", field=("rnd",), blocks=1, name="g")
        hub.run(lambda: hub.evaluation.queue.queued("tournament:") > 0)  # no node yet: both wait
        hub.add_node(1)
        hub.run(lambda: hub.evaluation.queue.stats()["leases"] >= 1)
        assert hub.evaluation.queue.queued("pulse:") == 1 and hub.evaluation.queue.queued("tournament:") == 0
        hub.service.cancel("g")
        hub.run(lambda: not hub.service.active)
    finally:
        hub.close()


def test_options_round_trip_through_json():
    options = TournamentOptions(mode="select", candidates=(("x", "/abs/x.pt"), ("rnd", None)),
                                runs=(RunSelection(path=Path("/abs/run"), alias="a", agent="main", snapshots="top:3"),),
                                field=("rnd",), layouts=("2p",), seed=4, deterministic=True, num_envs=3, name="n",
                                pin=True, overrides={"env.kwargs": {}})
    back = options_from_dict(json.loads(json.dumps(options_to_dict(options))))
    assert back == options
    with pytest.raises(Exception, match="unknown fields"):
        options_from_dict({"num_workers": 2})
```

- [ ] **Step 2 (a): Run them: RED**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_tournament_hooks.py tests/unit/test_sp5_hub_tournaments.py -q`
Expected: `ImportError: cannot import name 'TournamentHooks' from 'colosseum.arena.tournament.run'` and `cannot import name 'TournamentService' from 'colosseum.hub.evaluation'`.

- [ ] **Step 3 (a): The `run_tournament` hooks**

`src/colosseum/arena/tournament/run.py` (the local path is unchanged; every SP4 tournament test stays green):
```python
import threading
from collections.abc import Callable

PlayFn = Callable[[ColosseumConfig, Any, Mapping[str, TournamentPlayer], Sequence[Block],
                   Callable[[str, list[MatchResult]], None]], None]


@dataclass(frozen=True)
class TournamentHooks:
    """How a hub runs a tournament (SP5 T5.2). ``play(config, spec, players, blocks, on_block)`` replaces
    ``engine.play_blocks`` (the hub's evaluators play the blocks; the same contract: ``on_block`` once per finished
    block, with all its results); ``home`` is the hub's run dir: the tournament's root without ``--run`` (journal
    ``home/matches.db`` under the hub run's id, reports in ``home/tournaments/<name>/``); ``progress`` gets one line
    per batch of blocks for the client. ``tournament.log`` then holds only the calling thread's records (the hub
    process logs everything else)."""

    play: PlayFn
    home: Path | None = None
    progress: Callable[[str], None] | None = None
```
- `BlockRunner.__init__(..., play: PlayFn | None = None, progress: Callable[[str], None] | None = None)`; in `__call__`, when `todo` is not empty: `progress(f"{len(todo)} blocks to play ({self.played} played and {self.reused} taken from the journal so far)")` if given, then `self._play(self._config, self._spec, self._players, todo, self._store)` when a play hook is given, else the module-level `play_blocks(...)` exactly as now (looked up at call time: SP4 tests monkeypatch `run_module.play_blocks`).
- `_journal_run(options, config, db, tournament_id, first_config, home=None)`: without `--run` and with `home`, the hub's run plays the role of the first run (`first = Path(home)`, `first_config = config`: `RunDir.open(home).ensure_run_id()`, its known `runs` row reused); otherwise unchanged.
- `_file_logging(path, *, this_thread=False)`: with `this_thread`, `ident = threading.get_ident()` and `handler.addFilter(lambda record: record.thread == ident)`.
- `run_tournament(options, *, hooks=None)`: the root is `options.runs[0].path`, else `options.output`, else `hooks.home` (else the SP4 ConfigError); `_file_logging(..., this_thread=hooks is not None)`; `_run(..., hooks)` passes `home=hooks.home` to `_journal_run` when there is no `--run` and no `--output`, and `play=hooks.play`, `progress=hooks.progress` to `BlockRunner`.

- [ ] **Step 4 (a): Hub-run tournaments in `colosseum.hub.evaluation`**

Append to `src/colosseum/hub/evaluation.py` (module docstring: a "Hub-run tournaments (T5.2)" bullet summarizing `TournamentService`; `__all__` gains the produced names):
```python
# --- T5.2: hub-run tournaments (imports go to the top of the module) ---
import dataclasses
import hashlib
import json
import queue
import threading
from collections import deque
from datetime import datetime

from colosseum.arena.tournament.engine import TournamentError, check_blocks
from colosseum.arena.tournament.players import RunSelection
from colosseum.arena.tournament.run import TournamentHooks, TournamentOptions, run_tournament
from colosseum.core.config import check_path_component
from colosseum.core.errors import ConfigError, EnvContractError, PlayerError
from colosseum.core.run_state import process_alive, read_run_state
from colosseum.utils.fs import write_text_atomic

TOURNAMENTS_FILE = "hub_tournaments.json"
TOURNAMENTS_VERSION = 1
PROGRESS_SEC = 2.0              # at most one "k of n blocks" line per tournament per this many seconds
SUBSCRIBER_BACKLOG = 256        # replies a client may leave unread before it counts as detached
LAST_LINES = 20                 # lines replayed to a client that subscribes late
JOIN_SEC = 2.0                  # how long the hub's stop waits for a tournament thread
_OPTION_FIELDS = ("mode", "candidates", "runs", "field", "layouts", "blocks", "halving_blocks", "confirm_blocks",
                  "finalists", "seed", "deterministic", "num_envs", "name", "pin", "config_path", "overrides")


class TournamentCancelled(Exception):
    """``tournament --cancel NAME``."""


class TournamentInterrupted(Exception):
    """The hub stops: the tournament continues after ``hub --resume``."""


@dataclass
class Reply:
    """One reply to a tournament client (``pb.AdminReply``: code, text, data_json, done)."""

    code: int
    text: str
    data: dict | None = None
    done: bool = False


def options_to_dict(options: TournamentOptions) -> dict[str, Any]:
    d = {k: getattr(options, k) for k in _OPTION_FIELDS}
    d["candidates"] = [list(c) for c in options.candidates]
    d["runs"] = [{"path": str(r.path), "alias": r.alias, "agent": r.agent, "snapshots": r.snapshots} for r in options.runs]
    d["field"] = None if options.field is None else list(options.field)
    d["layouts"] = list(options.layouts)
    d["overrides"] = dict(options.overrides or {})
    return d


def options_from_dict(d: Mapping[str, Any]) -> TournamentOptions:
    unknown = sorted(set(d) - set(_OPTION_FIELDS))
    if unknown:
        raise ConfigError(f"tournament request: unknown fields {unknown}")
    kw = dict(d)
    kw["candidates"] = tuple((str(n), None if p is None else str(p)) for n, p in d.get("candidates", ()))
    kw["runs"] = tuple(RunSelection(path=Path(r["path"]), alias=r.get("alias"), agent=r.get("agent"),
                                    snapshots=r.get("snapshots") or "all") for r in d.get("runs", ()))
    kw["field"] = None if d.get("field") is None else tuple(d["field"])
    kw["layouts"] = tuple(d.get("layouts", ()))
    return TournamentOptions(**kw)


class TournamentProgress:
    """The replies of one hub-run tournament for its clients (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Any] = []
        self._lines: deque[Reply] = deque(maxlen=LAST_LINES)
        self._final: Reply | None = None

    def subscribe(self, q: Any) -> None:
        with self._lock:
            for reply in self._lines:
                q.put(reply)
            if self._final is not None:
                q.put(self._final)
            else:
                self._subscribers.append(q)

    def publish(self, reply: Reply) -> None:
        with self._lock:
            if reply.done:
                self._final = reply
            else:
                self._lines.append(reply)
            keep = []
            for q in self._subscribers:
                if q.qsize() >= SUBSCRIBER_BACKLOG:
                    continue                           # nobody reads it any more: a detached client
                q.put(reply)
                if not reply.done:
                    keep.append(q)
            self._subscribers = keep

    def line(self, text: str) -> None:
        self.publish(Reply(0, text))


@dataclass
class TournamentBlocks:
    """thread -> main loop (inbox event): queue these blocks; register the players' weights first."""

    name: str
    tasks: list[BlockTask]
    blobs: dict[str, TournamentPlayer]     # blob id -> neural player


@dataclass
class TournamentFinished:
    """thread -> main loop (inbox event): the tournament's thread ended."""

    name: str
    status: str                             # finished | cancelled | failed | running (interrupted: resumable)
    error: str | None = None


_TYPED = {"contract": EnvContractError, "player": PlayerError, "config": ConfigError}


class HubTournament:
    """One tournament run by the hub in its own thread (spec 5.3; T5.2): ``run_tournament`` with hooks whose
    ``play`` queues the blocks with the hub's evaluators (``TournamentBlocks``) and waits for their results on
    ``results`` (put by the main loop: ``("block", id, results)``, ``("failed", kind, text)``,
    ``("cancelled", "", None)``). Its thread talks to the main loop only through ``post`` (inbox events) and
    ``results``."""

    def __init__(self, *, options: TournamentOptions, home: Path, post: Callable[[object], None],
                 stopping: threading.Event, progress: TournamentProgress | None = None) -> None:
        self.options = options
        self.name = options.name
        self.owner = f"{TOURNAMENT_PREFIX}{self.name}"
        self.home = Path(home)
        self.progress = progress or TournamentProgress()
        self.results: queue.SimpleQueue = queue.SimpleQueue()
        self.blob_ids: set[str] = set()
        self._post, self._stopping = post, stopping
        self._thread = threading.Thread(target=self._main, name=f"tournament-{self.name}", daemon=True)
        root = Path(options.runs[0].path) if options.runs else self.home
        self._blob_prefix = "t-" + hashlib.sha256(f"{root.resolve()}/{self.name}".encode()).hexdigest()[:8]

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def cancel(self) -> None:
        self.results.put(("cancelled", "", None))

    def _blob_id(self, player: TournamentPlayer) -> str:
        return f"{self._blob_prefix}-{hashlib.sha256(player.name.encode()).hexdigest()[:12]}"

    def _play(self, config, spec, players: Mapping[str, TournamentPlayer], blocks, on_block) -> None:
        check_blocks(blocks, players=list(players.values()))
        by_seat = {(p.agent_id, p.network_id): p for p in players.values()}
        refs, blobs = {}, {}
        for p in players.values():
            blob_id = self._blob_id(p) if p.source.kind == "model" else None
            if blob_id is not None:
                blobs[blob_id] = p
            refs[p.name] = player_ref(p, blob_id)
        tasks = []
        for b in blocks:
            seats = sorted({(s.agent_id, s.network_id) for lu in b.lineups for s in lu.seats})
            tasks.append(BlockTask(block_id=b.id, source="tournament", owner=self.owner, lineups=tuple(b.lineups),
                                   players=tuple(refs[by_seat[k].name] for k in seats), priority=TOURNAMENT_PRIORITY,
                                   deterministic=self.options.deterministic, num_envs=self.options.num_envs))
        self._post(TournamentBlocks(self.name, tasks, blobs))
        pending = {b.id for b in blocks}
        total, last = len(pending), time.monotonic()
        while pending:
            if self._stopping.is_set():
                raise TournamentInterrupted()
            try:
                kind, key, value = self.results.get(timeout=0.5)
            except queue.Empty:
                continue
            if kind == "block":
                if key in pending:
                    pending.discard(key)
                    on_block(key, sorted(value, key=lambda r: r.uid))
                if pending and time.monotonic() - last >= PROGRESS_SEC:
                    last = time.monotonic()
                    self.progress.line(f"{total - len(pending)} of {total} blocks played")
            elif kind == "failed":
                raise _TYPED.get(key, TournamentError)(value)
            elif kind == "cancelled":
                raise TournamentCancelled()

    def _main(self) -> None:
        status, error = "failed", None
        try:
            outcome = run_tournament(self.options, hooks=TournamentHooks(play=self._play, home=self.home,
                                                                         progress=self.progress.line))
        except TournamentCancelled:
            status, error = "cancelled", "cancelled"
            self.progress.publish(Reply(1, f"tournament {self.name} cancelled", done=True))
        except TournamentInterrupted:
            status, error = "running", "the hub stopped"
            self.progress.publish(Reply(1, f"the hub stopped: tournament {self.name} continues after hub --resume",
                                        done=True))
        except (ConfigError, EnvContractError, PlayerError, TournamentError) as e:
            error = f"{type(e).__name__}: {e}"
            logger.error(f"tournament {self.name} failed: {error}")
            self.progress.publish(Reply(1, f"Tournament error: {e}", done=True))
        except Exception as e:  # noqa: BLE001 - a tournament must never take the hub down
            error = f"{type(e).__name__}: {e}"
            logger.exception(f"tournament {self.name} crashed")
            self.progress.publish(Reply(1, f"Tournament error: {error}", done=True))
        else:
            status = "finished"
            lines = ["", outcome.report.text(),
                     f"Blocks: {outcome.played_blocks} played, {outcome.reused_blocks} from the journal",
                     *(f"Written: {p}" for p in outcome.written), *(f"Pinned {w}" for w in outcome.pinned),
                     *(f"Pin error: {e}" for e in outcome.pin_errors)]
            self.progress.publish(Reply(1 if outcome.pin_errors else 0, "\n".join(lines), done=True,
                                        data={"directory": str(outcome.directory), "pinned": outcome.pinned,
                                              "pin_errors": outcome.pin_errors}))
        finally:
            self._post(TournamentFinished(self.name, status, error))


def check_root(path: Path) -> None:
    """A tournament of this hub may not write into a run dir that a live process owns (spec 5.3)."""
    state = read_run_state(path)
    if state is not None and state.status == "running" and process_alive(state) and state.pid != os.getpid():
        raise ConfigError(f"--run {path}: the run belongs to a running hub (pid {state.pid}); submit the tournament "
                          f"to that hub: colosseum tournament --hub {path} ...")


class TournamentService:
    """The hub's tournaments (main loop only): submit / status / cancel from admin requests, the inbox events of
    their threads, ``hub_tournaments.json`` (resume after ``hub --resume``) and the stop."""

    def __init__(self, *, home: Path, config_path: Path, evaluation, blobs, post: Callable[[object], None],
                 evaluators: Callable[[], int]) -> None:
        self._home, self._config_path = Path(home), Path(config_path)
        self._evaluation, self._blobs, self._post, self._evaluators = evaluation, blobs, post, evaluators
        self._active: dict[str, HubTournament] = {}
        self._stopping = threading.Event()
        self._path = self._home / TOURNAMENTS_FILE

    # --- the registry ---
    def _read(self) -> dict:
        try:
            data = json.loads(self._path.read_text())
        except FileNotFoundError:
            return {"version": TOURNAMENTS_VERSION, "tournaments": {}}
        if data.get("version") != TOURNAMENTS_VERSION:
            raise ConfigError(f"{self._path}: version {data.get('version')!r}, expected {TOURNAMENTS_VERSION}")
        return data

    def _record(self, name: str, **fields: Any) -> None:
        data = self._read()
        data["tournaments"].setdefault(name, {}).update(fields)
        write_text_atomic(self._path, json.dumps(data, indent=2, sort_keys=True) + "\n")

    # --- admin ---
    def submit(self, args: Mapping[str, Any], subscriber: Any) -> None:
        try:
            options = self._prepare(args)
        except ConfigError as e:
            subscriber.put(Reply(1, f"Tournament error: {e}", done=True))
            return
        self._start(options, subscriber)

    def _prepare(self, args: Mapping[str, Any]) -> TournamentOptions:
        if self._stopping.is_set():
            raise ConfigError("the hub is stopping")
        options = options_from_dict(args.get("options", {}))
        name = options.name or f"{options.mode}-{datetime.now():%Y%m%d-%H%M%S}"
        check_path_component(name, "tournament name")
        if name in self._active:
            raise ConfigError(f"tournament {name!r} is running on this hub; follow it or cancel it with --cancel {name}")
        root = Path(options.runs[0].path) if options.runs else self._home
        if options.runs and root.resolve() != self._home.resolve():
            check_root(root)
        config_path = options.config_path
        text = args.get("config_yaml")
        if text is not None:
            tdir = root / "tournaments" / name
            tdir.mkdir(parents=True, exist_ok=True)
            config_path = str(write_text_atomic(tdir / "submitted-config.yaml", str(text)))
        elif config_path is None and not options.runs:
            config_path = str(self._config_path)
        return dataclasses.replace(options, name=name, config_path=config_path, num_workers=1, output=None)

    def _start(self, options: TournamentOptions, subscriber: Any | None) -> HubTournament:
        t = HubTournament(options=options, home=self._home, post=self._post, stopping=self._stopping)
        if subscriber is not None:
            t.progress.subscribe(subscriber)
        self._active[t.name] = t
        self._record(t.name, root=str(Path(options.runs[0].path) if options.runs else self._home), status="running",
                     options=options_to_dict(options), error=None)
        self._evaluation.add_owner(t.owner, lambda kind, key, value, q=t.results: q.put((kind, key, value)))
        t.progress.publish(Reply(0, f"tournament {t.name} ({options.mode}) submitted to the hub", data={"name": t.name}))
        if self._evaluators() == 0:
            t.progress.line("no connected node runs an evaluator yet: the blocks wait (nodes with evaluator slots "
                            "take them: colosseum join --evaluators N)")
        t.start()
        return t

    def status(self, args: Mapping[str, Any]) -> Reply:
        data = self._read()["tournaments"]
        name = args.get("name")
        rows = {n: {k: v for k, v in d.items() if k != "options"} for n, d in data.items() if name in (None, n)}
        for n, row in rows.items():
            row["active"] = n in self._active
        return Reply(0 if rows or name is None else 1, json.dumps(rows, indent=2, sort_keys=True), data=rows,
                     done=True)

    def follow(self, name: str, subscriber: Any) -> None:
        t = self._active.get(name)
        if t is None:
            subscriber.put(Reply(1, f"no tournament {name!r} runs on this hub", done=True))
        else:
            t.progress.subscribe(subscriber)

    def cancel(self, name: str) -> Reply:
        t = self._active.get(name)
        if t is None:
            return Reply(1, f"no tournament {name!r} runs on this hub", done=True)
        self._evaluation.cancel(t.owner)
        t.cancel()
        return Reply(0, f"tournament {name} cancelled", done=True)

    # --- inbox events of the threads ---
    def on_blocks(self, event: TournamentBlocks) -> None:
        t = self._active.get(event.name)
        if t is None:
            return
        for blob_id, p in event.blobs.items():
            if blob_id not in t.blob_ids:
                self._blobs.register_weights(blob_id, Path(p.source.weights), agent_id=p.agent_id,
                                             networks=dict(p.source.networks or {}), roles=p.roles)
                t.blob_ids.add(blob_id)
        self._evaluation.submit(event.tasks)

    def on_finished(self, event: TournamentFinished) -> None:
        t = self._active.pop(event.name, None)
        if t is None:
            return
        self._evaluation.remove_owner(t.owner)
        for blob_id in t.blob_ids:
            self._blobs.unregister(blob_id)
        self._record(event.name, status=event.status, error=event.error)

    # --- lifecycle ---
    def resume(self) -> list[str]:
        """After ``hub --resume``: every tournament the registry lists as running continues (SP4 resume by name:
        the journal's whole blocks are reused)."""
        out = []
        for name, entry in sorted(self._read()["tournaments"].items()):
            if entry.get("status") == "running" and name not in self._active:
                logger.info(f"tournament {name} continues (the hub was resumed)")
                self._start(options_from_dict(entry["options"]), None)
                out.append(name)
        return out

    def stop(self) -> None:
        self._stopping.set()

    def join(self, timeout: float = JOIN_SEC) -> None:
        deadline = time.monotonic() + timeout
        for t in list(self._active.values()):
            t.join(max(0.0, deadline - time.monotonic()))

    @property
    def active(self) -> list[str]:
        return sorted(self._active)


def reply_to_proto(reply: Reply) -> pb.AdminReply:
    return pb.AdminReply(code=int(reply.code), text=reply.text, data_json=json.dumps(reply.data or {}),
                         done=bool(reply.done))


class ProtoReplies:
    """A subscriber over the admin servicer's reply queue: ``Reply`` -> ``pb.AdminReply`` (``Hub._on_admin``)."""

    def __init__(self, q: Any) -> None:
        self._q = q

    def put(self, reply: Reply) -> None:
        self._q.put(reply_to_proto(reply))

    def qsize(self) -> int:
        return self._q.qsize()
```
`TournamentService` is main-loop only; `HubTournament._play` / `_main` run in the tournament's thread and touch the main loop only through `post` (inbox events) and the tournament's `results` queue (the main loop's sink puts there). `TournamentProgress` is the one object both sides use; it is locked.

- [ ] **Step 5 (a): GREEN**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_tournament_hooks.py tests/unit/test_sp5_hub_tournaments.py tests/integration/test_sp4_tournament_cli.py tests/unit/test_sp4_tournament_modes.py -q`
Expected: all pass (the SP4 tournament tests unchanged).

- [ ] **Step 6 (a): Full suite; commit**

Full fast suite + ruff (zero warnings). Commit: `feat: hub-run tournaments (run_tournament hooks, TournamentService)`; push.

- [ ] **Step 7 (b): Write the failing integration tests**

Processes: the hub through `cli_runner` (`start_hub` / `run_in_session`, T2.5; not loaded), nodes as `NodeAgent` threads in the test process (`cluster_kit`, T3.1; their identity roots under `tmp_path`; `workers=0`, `evaluators=1`, `cpu_learners=0`), so the loaded processes are the evaluators only (≤ 2); the client `colosseum tournament --hub ...` through `run_in_session` / a session `Popen` (not loaded). Waits poll files and counters (`wait_for`), never wall-clock rates. Add to `tests/cli_runner.py`: `tournament_hub_cmd(hub: Path | str, *args) -> list[str]` (`[sys.executable, "-m", "colosseum", "tournament", "--hub", str(hub), *map(str, args)]`).

Create `tests/integration/test_sp5_hub_idle.py`:
- `test_an_idle_hub_makes_a_run_dir_and_trains_nothing`: a `turns` config with a trainable `main`, a scripted `rnd` and `evaluation.enabled: true`; `colosseum hub --idle -c cfg --set run.dir=<tmp> --set run.name=idle` → `hub.json` appears with `mode == "idle"`; the run dir has `config.resolved.yaml`, `run.json`, `run_state.json` (`running`); `logs/main.log` has "hub --idle: agents ['main'] are not trained" and "pulse is off in hub --idle"; a node thread joins: its `Welcome` has `no_workers` true and `evaluators == 0` (no tournament: no demand), the node starts no worker and no evaluator process (its heartbeat `procs` is empty), the hub received no `LineupRequest` (its `system` record: no cells); SIGINT to the hub → exit 130, `run_state.json` `stopped`, `checkpoints/` empty or absent.
- `test_an_idle_hub_checks_the_config_in_the_play_scope`: the trainable agent has `init.from: missing.pt` → `hub --idle` starts (and stops on SIGINT with 130), while `colosseum hub -c` on the same config exits 1 with the `init` ConfigError.

Create `tests/integration/test_sp5_tournament_hub_cli.py`:
- `test_tournament_on_an_idle_hub_runs_on_two_nodes`: `write_fake_run(tmp/x)` (versions 10 / 20 / 30 biased WEAK / UNIFORM / STRONG, as `test_sp4_tournament_cli.py`); an idle hub on `-c tmp/x/config.resolved.yaml` in `tmp/runs/idle`; two node threads (evaluators 1 each); `colosseum tournament --hub tmp/runs/idle --run tmp/x --field rnd --mode select --halving-blocks 2 --confirm-blocks 6 --finalists 2 --num-envs 2 --name sel` → exit 0; stdout has "submitted to the hub", "blocks to play", the report text, "Blocks:" and "Written:"; `tmp/x/tournaments/sel/report.json` selects `main@v30`; `tmp/x/matches.db` holds the tournament's matches once each (unique uids) with `node` from both node numbers (the round robin gave both leases); `tmp/runs/idle/hub_tournaments.json` has `sel` `finished`; the stage A / B seeds equal those of the same tournament run locally (`colosseum tournament --run <copy of x> ... --name sel`: the same block seeds, schedule domains A / B, spec 7.6).
- `test_detach_then_cancel`: the same setup with one node; a session `Popen` of `tournament --hub ... --mode gauntlet --field rnd --blocks 200 --name g`; once its stdout shows "submitted to the hub", SIGINT → exit 130 and stderr "Detached: the tournament continues on the hub (cancel it with --cancel g)"; the journal's blocks of `g` keep growing (wait for one more); `tournament --hub tmp/runs/idle --cancel g` → exit 0, stdout "tournament g cancelled"; `hub_tournaments.json` `g` → `cancelled` (wait ≤ 30 s); a second `--cancel g` → exit 1 ("no tournament 'g' runs on this hub").
- `test_usage_errors_exit_with_2` (parametrized, no hub needed): `--cancel g` without `--hub`; `--hub X --output o`; `--hub X --num-workers 2`; each exit 2 with the message naming the flag. And `--hub <a run dir without hub.json>` → exit 1 with the T2.5 `resolve_hub` hint; `--hub 127.0.0.1:<a closed port>` → exit 1 "cannot reach the hub".
- `test_a_killed_hub_resumes_the_tournament` (**slow**): idle hub + one node; a detached 60-block gauntlet; after ≥ 3 blocks are journaled, `kill -9` of the hub; `colosseum hub --resume tmp/runs/idle` → the node reconnects (T4.3 / T4.4), the tournament continues by itself ("tournament g continues (the hub was resumed)" in `main.log`), `hub_tournaments.json` `g` → `finished`, the journal holds all 60 blocks once each.

- [ ] **Step 8 (b): Run them: RED**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp5_hub_idle.py tests/integration/test_sp5_tournament_hub_cli.py -q -m "not slow"`
Expected: `colosseum hub --idle` and `tournament --hub` are unknown options (exit 2: "No such option"); the parametrized usage tests fail on their messages.

- [ ] **Step 9 (b): `Welcome.no_workers`; the node without workers**

Proto: `message Welcome { ...; bool no_workers = <next>; }` (true: the hub has no training work; the node runs no workers); regenerate. `src/colosseum/node/agent.py`, where T2.4 starts workers on the `Welcome`: with `no_workers`, start none (and stop running ones at once with `Stop`: a node that served a training hub before keeps nothing running); the worker slots stay in `Hello.slots` (the hub may change its mind after a resume with another mode).

- [ ] **Step 10 (b): The hub wiring and `hub --idle`**

`src/colosseum/hub/hub.py` (`Hub`; only these points):
- **Start** (after `self._evaluation` and `self._blobs` exist): `self._tournaments = TournamentService(home=self._run_dir.root, config_path=self._run_dir.resolved_config_path, evaluation=self._evaluation, blobs=self._blobs, post=self._inbox.put, evaluators=lambda: self._evaluation.stats()["evaluators"])`; with `options.resume` (T4.3's resume path, after the server is up): `self._tournaments.resume()`.
- **Inbox dispatch**: `TournamentBlocks` → `_on_tournament_blocks(event)` = `self._tournaments.on_blocks(event)`; `TournamentFinished` → `_on_tournament_finished(event)` = `self._tournaments.on_finished(event)`.
- **`_on_admin`**: `tournament.submit` (an `AdminStream`: `reply` is the servicer's queue) → `self._tournaments.submit(args, ProtoReplies(reply))`; `tournament.status` → `reply_to_proto(self._tournaments.status(args))` (unary), or with `args["follow"]` on a stream `self._tournaments.follow(args["name"], ProtoReplies(reply))`; `tournament.cancel` → `reply_to_proto(self._tournaments.cancel(args["name"]))`. A malformed request (missing keys, `ConfigError` from `options_from_dict`) answers `code 1` with the message; the main loop never raises on admin input.
- **Stop**: the stop's first step also calls `self._tournaments.stop()`; before the journal writer stops, `self._tournaments.join(JOIN_SEC)` (the threads close their own `MatchLog` connections; a thread still validating after `JOIN_SEC` is a daemon and dies with the process; its tournament stays `running` and resumable).
- **`HubOptions.mode == "idle"`** (`run_hub(..., idle=True)`):
  - validation in the "play" scope (`validate_config(config, scope="play", players=config.fixed_agent_ids())`), so a training config whose `init` sources do not exist yet still serves tournaments; one INFO "hub --idle: agents [...] are not trained" when the config has trainable agents, one INFO "pulse is off in hub --idle" when `evaluation.enabled`;
  - no learner is assigned (no placement), no pulse (`HubEvaluation(pulse=None)`), no checkpoint is saved, the coordinator is not asked for lineups (`Coordinator.generate_lineups` / `lineup_for` are never called without trainable agents, spec 4.2): `Welcome.no_workers = True`, `collect_allowed = False`, a `LineupRequest` gets no grants;
  - no budget: the main loop runs until SIGINT (130), SIGTERM (143) or `request_stop()` (0); `run_state.json`, `hub.json` (`mode: "idle"`), metrics ticks (`system`) and the journal as for any hub; the run dir is new (`runs/<name>/`) or resumed in place with `--resume` (its tournaments continue).
- The evaluator demand needs nothing more: `TournamentService` registers each tournament as an owner (`add_owner`), so `set_evaluators` goes out to every node while a tournament runs and back to 0 after.

- [ ] **Step 11 (b): The CLI**

`src/colosseum/cli.py`:
- the `hub` group (T2.5): `--idle` flag ("A hub without training: its nodes only evaluate (colosseum tournament --hub)"), passed to `run_hub(..., idle=True)`.
- `tournament_cmd`: new options `--hub` ("ADDR (host:port) or the RUN_DIR of a running hub (its hub.json): run the tournament on that hub; its node evaluators play the blocks and paths are read on the hub machine"), `--token`, `--token-file` (admin target and token, spec 3.3; also `COLOSSEUM_TOKEN`; on the hub machine the local identity is enough), `--cancel NAME` ("With --hub: cancel the hub's tournament NAME"); `--num-workers` default `None` (local: 1). Usage errors (exit 2): `--cancel` without `--hub`; `--hub` with `--output` ("the tournament lives in the hub's run dir, or in its first --run"), with `--num-workers` ("the hub's evaluators play the blocks; set them with colosseum join --evaluators"); the SP4 checks (`--pin` needs `--mode select` and `--run`) apply to both paths; without `--hub` and without `--run`, `-c` and `--output` stay required (SP4).
- with `--hub`, after the SP4 option parsing builds `TournamentOptions` (`num_workers=1`, `output=None`):
```python
def _tournament_on_hub(options, *, hub: str, token: str | None, token_file: str | None, config: str | None,
                       cancel: str | None) -> None:
    """``tournament --hub`` (spec 5.3): submit and follow, or cancel; exits 0 / 1 / 130."""
    from pathlib import Path

    import grpc

    from colosseum.hub.admin import call, resolve_hub, stream
    from colosseum.hub.evaluation import options_to_dict
    from colosseum.net.security import SecurityError

    as_dir = Path(hub).is_dir()
    try:
        target = resolve_hub(hub=None if as_dir else hub, run=hub if as_dir else None, token=token,
                             token_file=token_file)
    except SecurityError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    name = cancel or options.name
    try:
        if cancel is not None:
            reply = call(target, "tournament.cancel", {"name": cancel})
            click.echo(reply["text"], err=reply["code"] != 0)
            sys.exit(reply["code"])
        args = {"options": options_to_dict(_local_paths(options))}
        if config is not None:
            args["config_yaml"] = Path(config).read_text()
        for reply in stream(target, "tournament.submit", args):
            name = (reply.get("data") or {}).get("name", name)
            if reply["text"]:
                click.echo(reply["text"], err=reply["code"] != 0 and reply["done"])
            if reply["done"]:
                sys.exit(reply["code"])
    except KeyboardInterrupt:
        click.echo(f"Detached: the tournament continues on the hub (cancel it with --cancel {name})", err=True)
        sys.exit(130)
    except grpc.RpcError as e:
        click.echo(f"Error: cannot reach the hub at {target.address} ({e.code().name}: {e.details()})", err=True)
        sys.exit(1)
    click.echo("Error: the hub closed the stream before the tournament ended (see its log)", err=True)
    sys.exit(1)


def _local_paths(options):
    """Paths that exist on this machine made absolute (a hub on this machine reads them from any cwd); the others
    are sent as given (read on the hub machine)."""
    import dataclasses
    from pathlib import Path

    def fix(p):
        return str(Path(p).resolve()) if p is not None and Path(p).exists() else p

    return dataclasses.replace(
        options, candidates=tuple((n, fix(p)) for n, p in options.candidates),
        runs=tuple(dataclasses.replace(r, path=Path(fix(r.path))) for r in options.runs), config_path=None)
```
  The admin helpers' replies are dicts `{"code", "text", "data", "done"}` (T2.5 `call` / `stream`). The help text of `tournament` gains the `--hub` exit codes: 0 finished (also `confirmed: false`), 1 cancelled, failed, refused or the hub stopped, 2 bad arguments, 130 detached.

- [ ] **Step 12 (b): GREEN; full suite; the slow test; commit**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp5_hub_idle.py tests/integration/test_sp5_tournament_hub_cli.py tests/integration/test_sp4_tournament_cli.py -q -m "not slow"`
Expected: all pass. Then the full fast suite + ruff (zero warnings), and once, alone: `env -u PYTHONPATH .venv/bin/python -m pytest -m slow tests/integration/test_sp5_tournament_hub_cli.py -v`. Commit: `feat: tournament --hub, hub --idle`; push.

---

## Contract notes

### Additions and deviations versus `00-overview.md`

1. **`PlayerRef` and `BlockTask` live in `colosseum.net.messages`** (re-exported by `colosseum.hub.evaluation`, where the contract names them): the node and the evaluator need them, and the converters may live only in `net.messages`. `BlockTask` fields are tuples (`lineups`, `players`) plus `num_envs` (the envs a lease is played with: `evaluation.num_envs` for the pulse, `--num-envs` for a tournament); `PULSE_PRIORITY = 1`, `TOURNAMENT_PRIORITY = 0`; converters `player_ref_to_proto` / `player_ref_from_proto` / `block_task_to_proto` / `block_task_from_proto` / `lease_to_proto`.
2. **Proto additions** (T5.1): `Hello.leases` (repeated string), `BlockLease.num_envs`, `BlockResult.error_kind` (`""` | `abandoned` | `crash` | `missing_weights` | `contract` | `player` | `config`), message `BlockAck {lease_id, block_ids}` as `HubMessage.block_ack`, `Command.lease_ids` with the new kind `cancel_leases`, the `BlobRequest.kind` value `eval`; (T5.2) `Welcome.no_workers`. `PROTOCOL_VERSION` stays `(1, 0)`.
3. **Lease protocol:** `BlockLeaseRequest.free_slots` is an **increment** (slots not asked for yet); the hub answers asynchronously (one `BlockLease` per grant, round robin over waiting nodes; no empty replies). One lease = consecutive whole blocks of one owner with ≥ `TASK_MATCHES × num_envs` matches. Every `BlockResult` is acked (`BlockAck`) whatever its outcome; the node resends unacked ones after a reconnect.
4. **`BlockQueue` API beyond the contract:** `want` / `unwant` / `waiting` / `dispatch`, `release(lease_id, block_id, *, count, reason)`, `keep_only(node, lease_ids)`, `forget(owner)`, `take_failed()`, `task(block_id)`, `leases_of(node)`, `queued(prefix)`, `live(owner)`, `stats()`; `complete` returns `"accepted" | "late" | "cancelled" | "unknown"` (the contract's three plus `cancelled`); `cancel(owner)` returns `(block ids, {node: [lease ids]})` (the hub must tell the nodes); `MAX_BLOCK_ATTEMPTS = 3`.
5. **`HubEvaluation`** (new, `hub.evaluation`): the main loop's evaluation service (`node_joined` → `JoinReply(evaluators, after_welcome)`, `session_closed`, `node_lost`, `on_lease_request`, `on_block_result`, `add_owner` / `submit` / `remove_owner` / `cancel`, `tick`, `stop`, `finish`, `stats`); `check_block_results`; `player_ref`; `EvalBlobs` protocol.
6. **`HubPulse` replaces `PulseFeeder.pump`:** `PulseFeeder` keeps `offer` (without weights: `offer(agent, ckpt, version, env_steps, *, final, weights="")`), the backlog, `state()`, `unevaluated_final()`, `skipped_by_backlog()`, and gains `start_next`, `done(agent, ckpt)`, `fail(agent, ckpt, reason)`, `take_dropped`, `in_flight`; `PulseRequest` has `weights: str` instead of `state_dict`. `PulseResults`, `pulse_target`, `pulse_main` are deleted (R6).
7. **Evaluation blobs:** `BlobStore.register_weights(blob_id, weights_file, *, agent_id, networks, roles)` / `unregister(blob_id)`, served by `Download(kind="eval", agent_id, ident=blob_id)` as `encode_weights` frames with `WeightsMeta(kind="fixed", architecture=networks, roles=roles)` (no new `WeightsMeta` kind), converted lazily in servicer threads under the registration lock. Blob ids: `pulse-<agent>-<ckpt id>`, `field-<name>`, `t-<hash8>-<hash12>`. Node cache keys `CacheKey("blob", agent_id, blob_id)`.
8. **Node side:** `colosseum.node.leases` (`NodeLease`, `BlockOutcome`, `NodeLeases`, `EVAL_BLOBS_KEPT = 8`) — a new module, not in the overview's file map; local messages `EvalTask`, `EvalCancel`, `EvalDone`, `EvalFailed` in `net.local`; `evaluator_main` exposes `evaluator_loop`, `CachePlayerPool`, `MissingWeights`, `play_task`, `CONNECT_TIMEOUT_SEC`, `POLL_SEC`. A typed evaluator error is never a `ChildError` (it travels as a failed `BlockResult`).
9. **Engine renames:** `arena.tournament.engine._Placement` → `MatchPlacement` (`where(message, context, block_ids)`), `_error_kind` → `error_kind`.
10. **Run-dir paths:** `eval/pulse/<agent>/<checkpoint id>/` (hard links of snapshots waiting for or under pulse evaluation; cleared at start and finish), `hub_tournaments.json` (`{"version": 1, "tournaments": {name: {"root", "status": running|finished|cancelled|failed, "options", "error"}}}`), `tournaments/<name>/submitted-config.yaml` (the `-c` text a client sent). Logs: `evaluator-<i>.log` replaces `pulse.log` (T7.5 updates the docs).
11. **`system` record:** a new key `evaluation` (`HubEvaluation.stats()`: `queued`, `leased_now`, `leases_now`, the cumulative `submitted`, `leases`, `leased`, `completed`, `late`, `cancelled`, `unknown`, `expired`, `requeued`, `failed`, `cancelled_blocks`, plus `invalid_results`, `evaluators`, `pulse_results`). T7.1 decides whether it joins `REQUIRED_KEYS["system"]` (it is always set from T5.1 on).
12. **Hub-run tournaments (T5.2):** `TournamentHooks` + `run_tournament(options, *, hooks=None)`; `TournamentService`, `HubTournament`, `TournamentProgress`, `Reply` / `reply_to_proto` / `ProtoReplies`, inbox events `TournamentBlocks` / `TournamentFinished`, `options_to_dict` / `options_from_dict`, `check_root`, constants `PROGRESS_SEC`, `SUBSCRIBER_BACKLOG`, `LAST_LINES`, `JOIN_SEC`. Admin: `tournament.submit` (stream; first reply's `data` is `{"name"}`), `tournament.status` (`name` optional, `follow` on a stream), `tournament.cancel`. CLI: `tournament --hub ADDR|RUN_DIR [--token T] [--token-file F] [--cancel NAME]` (`--token-file` is added for symmetry with the other admin commands); `--num-workers` and `--output` are usage errors with `--hub`.
13. **`LocalNodeOptions.evaluators` is a slot, always 1** (R18 said "1 with `evaluation.enabled`, else 0"); the hub runs it on demand — see proposed ruling 2.
14. **The hub's run-stopping error path** is named `Hub._fail_run(message)` here (a typed error in a pulse block stops the run with exit 1, as the SP4 pulse crash did); bind it to whatever T2.5 / T4.4 call the path that a worker's `ChildError(kind="contract"|"player")` takes.

### Proposed rulings (what — why — cost if wrong)

1. **The backlog drops only snapshots whose evaluation has not started** (SP4's "oldest waiting request"); a started snapshot is finished (or fails after `MAX_BLOCK_ATTEMPTS`), and the next one starts as soon as the last one's blocks are all leased (pipelining over many evaluators) — why: spec 5.3 says a backlog-dropped snapshot withdraws its leased blocks, but dropping started snapshots starves a slow pulse (with evaluation slower than `max_backlog` snapshot intervals no snapshot would ever finish); SP4 likewise never dropped its in-flight request; the withdrawal path (cancel + `cancel_leases`, results counted as `cancelled`) is still used for failed snapshots and at the stop — cost if wrong: at most (evaluator leases in flight) snapshots are evaluated that a stricter reading would have dropped for newer ones.
2. **Evaluators run on demand:** every node (also `train`'s local node, whose slot is now always 1) runs its evaluator slots only while the pulse is on or a hub-run tournament is active (`set_evaluators`), else 0 — why: no idle evaluator processes (memory, criterion 2's process set unchanged for runs without pulse), while `tournament --hub` against a plain `train` still finds an evaluator — cost if wrong: a tournament's first lease waits for an evaluator to spawn (seconds).
3. **Pulse results reach the journal per snapshot, all at once**, and a snapshot is "evaluated" only when all its blocks are in; blocks of a snapshot still in flight at the stop are discarded (the snapshot stays unevaluated) — why: SP4 sent one message per snapshot and its pulse bootstrap and report assume whole snapshots — cost if wrong: at most one partly played snapshot per in-flight lease is lost at the stop.
4. **A typed error (`contract` / `player` / `config`) in a pulse block stops the run with exit 1; in a tournament block it fails that tournament only** — why: SP4 (a pulse crash stopped the run; a tournament worker's error ended the tournament) and spec 7.1 (contract / player errors stop the run) — cost if wrong: a broken field bot stops a long training run (as in SP4).
5. **Lease state is not persisted:** after a hub restart, the nodes' held leases are unknown to the new hub (cancelled at once via `cancel_leases`), their unacked block results are acked and dropped as `unknown`; pulse snapshots restart nowhere (`pulse_state.json` is not resumed, SP4) and tournaments replay the blocks missing from the journal — why: the journal is the record of finished blocks; leases are minutes of work — cost if wrong: up to one lease of work per evaluator is replayed after a hub restart.
6. **`MAX_BLOCK_ATTEMPTS = 3`:** expiries, evaluator crashes, missing weights and invalid results count; node loss, a node restart and an evaluator stopped by `set_evaluators` do not — why: a block that kills its evaluator or never finishes in `block_lease_sec` must not loop forever; the network is not the block's fault — cost if wrong: a flaky evaluator machine can fail a block that would pass elsewhere (the owner sees the reason).
7. **A run dir is "owned by another running hub" when its `run_state.json` says `running` with a live pid other than this hub's** (the T4.3 liveness rule) — why: `run_state.json` is the existing ownership record (also for SP4 dirs); `hub.json` alone survives a crash — cost if wrong: none known.
8. **Hub-run tournaments with the name of a running one are refused; names are unique per hub** (any root) — why: the client cancels and follows by name — cost if wrong: two runs' tournaments cannot share a name on one hub at the same time.
9. **Client paths:** a `-a` / `--run` path that exists on the client machine is sent absolute; others as given (read on the hub machine); `-c` is sent as text — why: the same command works on the hub machine from any cwd and from another machine — cost if wrong: a relative path that exists on both machines but means different things is read as the client's absolute path.
10. **`hub --idle` has no budget and exits 130 / 143 on signals** (0 only via `request_stop`) — why: spec 7.5 codes; an idle hub has nothing to finish — cost if wrong: scripts that wait for exit 0 must stop it otherwise.

### Assumptions about other parts

1. **T1.2** provides `lineup_to_proto` / `lineup_from_proto` (with `seed`, `block`, `uid`) and `result_to_proto` / `result_from_proto` (with `node`) in `net.messages`, and `scripts/gen_proto.sh`; `MatchResult.node` exists.
2. **T1.5** `WeightCacheWriter.put(key, frame)` / `drop(key)`, `WeightCacheReader.get(key) -> CachedWeights | None` / `has(key)` (read-only views, `meta` from the frame), `cache_prefix`, `shm_dir()` honouring `COLOSSEUM_SHM_DIR`; **T1.1** `encode_weights` / `decode_weights` and `WeightsMeta(kind="fixed", architecture, roles)`.
3. **T2.2** `net.local`: `listener(name, authkey)`, `connect(name, authkey, *, timeout)`, `address(...)`, `ChildHello`, `Stop`, pickled dataclass messages on `multiprocessing.connection`.
4. **T2.3** `BlobStore` serves `Download` from `frame_path(kind, agent, ident)` (adding a kind is one dispatch entry), `SessionHandle.send`, `AdminIn.reply` is a `Future` (unary) or the servicer's queue of `pb.AdminReply` for `AdminStream` (the servicer streams until `done`); `hub_kit.FakeNode` can send any `NodeMessage` and read `HubMessage`s and call `Download`.
5. **T2.4** `NodeAgent` has a main loop over child connections where an "evaluation" section fits; `ChildSupervisor` / `ChildSpec(role="evaluator")`; workers are started on the `Welcome`; `node_kit.FakeHub` can script `Welcome` / `Command` / `BlockLease` / `BlockAck`, serve blobs and drop a session (Step 9 of T5.1 lists what to add if missing).
6. **T2.5** `Hub` keeps the R5 handlers named in Step 14 / Step 10, `_pump_pulse` / `_finish_pulse` / `_write_pulse_state` / `_save_checkpoint` with SP4 semantics, `HubOptions.mode` (`"idle"` exists), `run_hub(..., idle=)`, `admin.resolve_hub` / `call` / `stream` returning dicts `{code, text, data, done}`, `cli_runner.start_hub`.
7. **T3.1** the node's transfer pool fetches any `BlobRequest` kind into the cache and posts the outcome to the node inbox; `cluster_kit` starts a hub process and `NodeAgent` threads with their own identity roots and slot options.
8. **T4.2** evaluators restart within `WORKER_RESTARTS` and a disabled slot is reported (`Event(slot_disabled)`); **T4.3** `hub --resume` / `train --resume` call the Hub's start with `options.resume` (T5.2 adds `TournamentService.resume()` there) and the journal has the `node` column; **T4.4** children die with their node and the node with `train`'s hub; the hub's stop sends `stop_run`.
9. **T7.1** shows the `system.evaluation` counters (or not) and decides on `REQUIRED_KEYS`; **T7.4** covers pulse on a remote evaluator in the localhost distributed pipeline (slow) and the two-node pulse in the acceptance; **T7.5** documents `tournament --hub`, `hub --idle`, pulse on evaluators, `evaluator-<i>.log` and proposed rulings 1–3 (ARENA_GUIDE, DISTRIBUTED_GUIDE).
